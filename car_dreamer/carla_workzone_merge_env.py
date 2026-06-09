import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos


class CarlaWorkzoneMergeEnv(CarlaWptEnv):
    """
    Workzone merge task.

    Ego starts in lane -1 (leftmost, x=5.24) behind a randomly spawned parked
    Cybertruck. The agent must stay in lane until close to the Cybertruck,
    then merge right into lane -2 (x=8.74) to avoid it. Episode ends
    immediately once the agent has passed the Cybertruck in lane -2.
    No return to original lane required.

    Road geometry (Town04, y-axis travel, yaw=-90.3):
        lane -1: x=5.24  (leftmost / fast lane — ego + Cybertruck)
        lane -2: x=8.74  (merge target)
        lane -3: x=12.24
        lane -4: x=15.74

    Cybertruck spawns randomly between y=50 and y=80.
    Ego always starts at y=130 — 50-80 units behind truck.
    Success triggers when ego is in lane -2 and past Cybertruck.

    **Provided Tasks**: ``carla_workzone_merge``

    Reward components:
    * ``stay_same_lane``: Flat penalty for staying in lane -1 within merge window.
    * ``early_lane_change``: Flat penalty for merging too early.
    * ``r_merge``: One-time reward for passing Cybertruck in lane -2.
    * ``steer_smooth``: Penalty for rapid steering changes.
    * ``collision``: Penalty for collision.

    Terminal conditions:
    * ``success``: In lane -2 and past the Cybertruck.
    * ``is_collision``: Collided.
    * ``time_exceeded``: Episode time limit reached.
    * ``out_of_lane``: Went beyond lane boundaries.
    """

    # Confirmed lane centers from CARLA query at y=65
    LANE1_X = 5.24   # lane -1 center — leftmost fast lane
    LANE2_X = 8.74   # lane -2 center — merge target

    def on_reset(self) -> None:
        # Randomly spawn Cybertruck in lane -1 at random y position
        nonego_y = np.random.uniform(
            self._config.nonego_spawn_y_range[0],
            self._config.nonego_spawn_y_range[1],
        )
        self.nonego_spawn_point = [self.LANE1_X, nonego_y, 0.1, 0.0, -90.0, 0.0]

        nonego_transform = carla.Transform(
            carla.Location(*self.nonego_spawn_point[:3]),
            carla.Rotation(*self.nonego_spawn_point[-3:]),
        )

        # Spawn Cybertruck as static parked obstacle
        bp = self._world.get_blueprint("vehicle.tesla.cybertruck")
        self.nonego = self._world.try_spawn_actor(transform=nonego_transform, blueprint=bp)
        if self.nonego is None:
            raise RuntimeError("Cybertruck spawn failed")
        self.nonego.set_simulate_physics(False)
        print(f"Cybertruck spawned at x={self.LANE1_X:.2f}, y={nonego_y:.2f}")

        # Ego spawns at y=130 in lane -1 — always behind Cybertruck
        self.ego_src = self._config.lane_start_points[0]
        ego_transform = carla.Transform(
            carla.Location(x=self.LANE1_X, y=self.ego_src[1], z=self.ego_src[2]),
            carla.Rotation(yaw=-90),
        )
        self.ego = self._world.spawn_actor(transform=ego_transform)

        # Path planning — straight along lane -1 toward destination
        ego_dest = self._config.lane_end_points
        dest_location = carla.Location(x=self.LANE1_X, y=ego_dest[0][1], z=ego_dest[0][2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        self._merged = False

        # Smoothness tracking
        self.prev_steer = 0.0

        # Set spectator
        spectator = self._world._world.get_spectator()
        ego_transform.location.z += 150
        ego_transform.rotation.pitch = -70
        spectator.set_transform(ego_transform)

    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        self.ego.apply_control(control)
        # No control on nonego — static parked Cybertruck

    # =========================================================
    # Reward
    # =========================================================
    def reward(self):
        total_reward, info = super().reward()

        # Remove base out of lane penalty — we handle boundaries ourselves
        total_reward -= info["r_out_of_lane"]
        del info["r_out_of_lane"]

        reward_scales = self._config.reward.scales
        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_y = self.nonego.get_transform().location.y

        # Lane membership using confirmed lane centers
        in_lane1 = abs(ego_x - self.LANE1_X) < 1.5
        in_lane2 = abs(ego_x - self.LANE2_X) < 1.5

        # Distance to Cybertruck: positive = not reached yet, negative = past it
        dist_to_truck = ego_y - nonego_y

        early_dist = self._config.reward.early_lane_change_dist

        # -----------------------------------------------
        # Stay same lane — flat penalty in merge window
        # -----------------------------------------------
        p_stay_same_lane = 0.0
        if 0 < dist_to_truck < early_dist and in_lane1:
            p_stay_same_lane = -reward_scales["stay_same_lane"]

        # -----------------------------------------------
        # Early lane change penalty
        # -----------------------------------------------
        p_early_lane_change = 0.0
        if dist_to_truck > early_dist and in_lane2:
            p_early_lane_change = -reward_scales["early_lane_change"]

        # -----------------------------------------------
        # Merge reward — one-time for passing Cybertruck in lane -2
        # -----------------------------------------------
        r_merge = 0.0
        if in_lane2 and ego_y < nonego_y and not self._merged:
            r_merge = reward_scales["r_merge"]
            self._merged = True

        # -----------------------------------------------
        # Steering smoothness penalty
        # -----------------------------------------------
        current_steer = self.ego.get_control().steer
        p_steer_smooth = -abs(current_steer - self.prev_steer) * reward_scales["steer_smooth"]
        self.prev_steer = current_steer

        total_reward += p_stay_same_lane + p_early_lane_change + r_merge + p_steer_smooth

        info.update(
            {
                "dist_to_truck": dist_to_truck,
                "in_lane1": in_lane1,
                "in_lane2": in_lane2,
                "merged": self._merged,
                "p_stay_same_lane": p_stay_same_lane,
                "p_early_lane_change": p_early_lane_change,
                "r_merge": r_merge,
                "p_steer_smooth": p_steer_smooth,
            }
        )

        return total_reward, info

    # =========================================================
    # Terminal conditions
    # =========================================================
    def get_terminal_conditions(self):
        ego_x = self.ego.get_location().x
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()

        # Out of lane boundary check
        info["out_of_lane"] = (
            self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres
            or ego_x < terminal_config.left_lane_boundry
            or ego_x > terminal_config.right_lane_boundry
        )

        # Success: in lane -2 and past Cybertruck — no return needed
        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_y = self.nonego.get_transform().location.y
        in_lane2 = abs(ego_x - self.LANE2_X) < 1.5
        info["success"] = in_lane2 and ego_y < nonego_y and self._merged

        return info
