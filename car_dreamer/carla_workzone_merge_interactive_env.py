import math
import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos


class CarlaWorkzoneMergeInteractiveEnv(CarlaWptEnv):
    """
    Workzone merge task with interactive lane-2 traffic.

    Scene (Town04, y-axis travel, yaw=-90):
        Lane -1 (x=5.24): ego vehicle + parked Cybertruck (closure)
        Lane -2 (x=8.74): moving background vehicle

    Episode:
        - Cybertruck spawns randomly in lane -1 (y=50-80)
        - Ego spawns at y=130, drives toward Cybertruck
        - Background vehicle spawns in lane -2 (y=105-140)
          * Behind ego (y>130): faster speed (5-8 m/s)
          * Ahead of ego (y<130): slower speed (2-4 m/s)
        - Ego must merge right into lane -2 to avoid Cybertruck

    Success: ego in lane -2, past Cybertruck
    Failure: collision with Cybertruck or bg vehicle, out of lane, timeout

    Infrastructure sensor features logged per step (observable from fixed camera):
        ego_x, ego_y          — position
        vx, vy                — velocity components
        speed_norm            — total speed
        lateral_change_rate   — d(ego_x)/dt — key merge commitment signal
        dist_to_closure       — distance to Cybertruck
        lateral_offset_lane1  — offset from lane -1 center
        lateral_offset_lane2  — offset from lane -2 center
        gap_size              — space in lane -2
        gap_closing_rate      — how fast gap is closing
        bg_x, bg_y            — background vehicle position
        bg_speed              — background vehicle speed
    """

    LANE1_X = 5.24  # lane -1 center (confirmed from CARLA query)
    LANE2_X = 8.74  # lane -2 center

    def on_reset(self) -> None:
        # --- Cybertruck (static closure in lane -1) ---
        nonego_y = np.random.uniform(
            self._config.nonego_spawn_y_range[0],
            self._config.nonego_spawn_y_range[1],
        )
        self.nonego = self._world.try_spawn_actor(
            transform=carla.Transform(
                carla.Location(x=self.LANE1_X, y=nonego_y, z=0.1),
                carla.Rotation(yaw=-90),
            ),
            blueprint=self._world.get_blueprint("vehicle.tesla.cybertruck"),
        )
        if self.nonego is None:
            raise RuntimeError("Cybertruck spawn failed")
        self.nonego.set_simulate_physics(False)
        self._nonego_y = nonego_y
        self._nonego_location = (self.LANE1_X, nonego_y)

        # --- Background vehicle (moving in lane -2) ---
        bg_y = float(np.random.uniform(105.0, 140.0))
        self._bg_speed = float(np.random.uniform(5.0, 8.0) if bg_y > 130.0
                               else np.random.uniform(2.0, 4.0))
        self.bg_vehicle = self._world.spawn_actor(
            transform=carla.Transform(
                carla.Location(x=self.LANE2_X, y=bg_y, z=0.1),
                carla.Rotation(yaw=-90),
            )
        )

        # --- Ego vehicle (lane -1, y=130) ---
        ego_src = self._config.lane_start_points[0]
        self.ego = self._world.spawn_actor(
            transform=carla.Transform(
                carla.Location(x=self.LANE1_X, y=ego_src[1], z=ego_src[2]),
                carla.Rotation(yaw=-90),
            )
        )

        # --- Path planner ---
        ego_dest = self._config.lane_end_points
        self.ego_planner = FixedEndingPlanner(
            self.ego,
            carla.Location(x=self.LANE1_X, y=ego_dest[0][1], z=ego_dest[0][2]),
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        # --- Episode state ---
        self._merged = False
        self._prev_ego_x = self.LANE1_X
        self._prev_gap_size = abs(bg_y - nonego_y)
        self.prev_steer = 0.0

        # --- Spectator: closure camera POV ---
        self._world._world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=self.LANE1_X, y=nonego_y + 3.0, z=5.0),
                carla.Rotation(pitch=-10, yaw=90),
            )
        )

    def get_state(self):
        """Expose closure position for closure camera handler."""
        state = super().get_state()
        state["nonego_location"] = self._nonego_location
        return state

    def get_ego_planner(self):
        return self.ego_planner

    def apply_control(self, action) -> None:
        self.ego.apply_control(self.get_vehicle_control(action))
        self.bg_vehicle.apply_control(self._get_bg_control())

    def _get_bg_control(self):
        """Constant speed straight-line control for bg vehicle."""
        speed = math.sqrt(self.bg_vehicle.get_velocity().x ** 2 +
                          self.bg_vehicle.get_velocity().y ** 2)
        throttle = 0.6 if speed < self._bg_speed else 0.0
        return carla.VehicleControl(throttle=float(throttle), steer=0.0, brake=0.0)

    def _hit_cybertruck(self):
        return (self.is_collision() and
                self.ego.get_location().distance(self.nonego.get_location()) < 6.0)

    def _hit_bg_vehicle(self):
        return (self.is_collision() and
                self.ego.get_location().distance(self.bg_vehicle.get_location()) < 6.0)

    # =========================================================
    # Reward
    # =========================================================
    def reward(self):
        total_reward, info = super().reward()

        # Remove base penalties we handle ourselves
        total_reward -= info.pop("r_out_of_lane", 0)
        if info.get("r_collision", 0) != 0:
            total_reward -= info["r_collision"]
            info["r_collision"] = 0.0

        scales = self._config.reward.scales
        ego_x, ego_y = get_vehicle_pos(self.ego)
        nonego_y = self._nonego_y
        bg_x, bg_y = get_vehicle_pos(self.bg_vehicle)

        # Lane membership
        in_lane1 = abs(ego_x - self.LANE1_X) < 1.5
        in_lane2 = abs(ego_x - self.LANE2_X) < 1.5

        # Key distances
        dist_to_closure = ego_y - nonego_y
        early_dist = self._config.reward.early_lane_change_dist

        # ---- Reward components ----
        # Penalize staying in lane -1 within merge window
        p_stay = -scales["stay_same_lane"] if (0 < dist_to_closure < early_dist and in_lane1) else 0.0

        # Penalize merging too early
        p_early = -scales["early_lane_change"] if (dist_to_closure > early_dist and in_lane2) else 0.0

        # One-time merge reward
        r_merge = 0.0
        if in_lane2 and ego_y < nonego_y and not self._merged:
            r_merge = scales["r_merge"]
            self._merged = True

        # Collision penalties
        r_col_truck = -scales["collision"] if self._hit_cybertruck() else 0.0
        r_col_bg    = -scales["collision"] if self._hit_bg_vehicle() else 0.0

        # Steering smoothness
        steer = self.ego.get_control().steer
        p_smooth = -abs(steer - self.prev_steer) * scales["steer_smooth"]
        self.prev_steer = steer

        total_reward += p_stay + p_early + r_merge + r_col_truck + r_col_bg + p_smooth

        # ---- Infrastructure sensor features ----
        dt = self._config.world.fixed_delta_seconds
        lateral_change_rate = (ego_x - self._prev_ego_x) / dt
        self._prev_ego_x = ego_x

        gap_size = max(0.0, abs(bg_y - nonego_y) - 5.0)  # subtract vehicle length
        gap_closing_rate = (gap_size - self._prev_gap_size) / dt
        self._prev_gap_size = gap_size

        info.update({
            # Infrastructure-observable features
            "ego_x":               ego_x,
            "ego_y":               ego_y,
            "vx":                  float(self.ego.get_velocity().x),
            "vy":                  float(self.ego.get_velocity().y),
            "lateral_change_rate": lateral_change_rate,
            "dist_to_closure":     dist_to_closure,
            "lateral_offset_lane1": abs(ego_x - self.LANE1_X),
            "lateral_offset_lane2": abs(ego_x - self.LANE2_X),
            "gap_size":            gap_size,
            "gap_closing_rate":    gap_closing_rate,
            "bg_x":                bg_x,
            "bg_y":                bg_y,
            "bg_speed":            self._bg_speed,
            # Episode state
            "in_lane1":            in_lane1,
            "in_lane2":            in_lane2,
            "merged":              self._merged,
            "dist_to_bg":          abs(ego_y - bg_y),
        })

        return total_reward, info

    # =========================================================
    # Terminal conditions
    # =========================================================
    def get_terminal_conditions(self):
        ego_x, ego_y = get_vehicle_pos(self.ego)
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()

        info["out_of_lane"] = (
            self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres
            or ego_x < terminal_config.left_lane_boundry
            or ego_x > terminal_config.right_lane_boundry
        )
        info["success"] = (
            abs(ego_x - self.LANE2_X) < 1.5
            and ego_y < self._nonego_y
            and self._merged
        )
        info["is_collision"] = self._hit_cybertruck() or self._hit_bg_vehicle()

        return info
