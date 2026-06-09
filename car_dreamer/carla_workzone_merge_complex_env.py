import math
import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos


class CarlaWorkzoneMergeComplexEnv(CarlaWptEnv):
    """
    Workzone merge task — complex scene with follower + two bg vehicles.

    Scene (Town04, y-axis travel, yaw=-90, decreasing y = forward):

        Lane -1 (x=5.24):
            y=050-080 : Cybertruck (static closure)
            y=130     : ego (fixed spawn)
            y=148-165 : follower (pressure from behind)

        Lane -2 (x=8.74):
            y=100-120 : bg1 (lead vehicle, ahead of ego, just cruises)
            y=145-160 : bg2 (following vehicle, behind ego, yields for bg1 + ego)
                        once bg2 passes closure it cruises freely — no braking

    Ego must merge into the gap between bg1 (ahead) and bg2 (behind) in lane -2.

    Difficulty varies naturally by speed:
        bg2_speed > bg1_speed  → gap closing  → hard merge
        bg2_speed < bg1_speed  → gap opening  → easy merge
        bg2_speed = bg1_speed  → gap stable   → medium merge

    Success: ego in lane -2, past Cybertruck
    Failure: collision with any vehicle, out of lane, timeout

    16 Infrastructure-observable features logged per step:
        ego_x, ego_y, vy, speed_norm, lateral_change_rate,
        dist_to_closure, gap_size, gap_closing_rate,
        bg_y, bg_speed, gap12_size, gap12_closing_rate,
        bg2_y, bg2_speed, follower_dist, follower_speed

    Collision type features logged per step (for analysis):
        collision_closure, collision_traffic, collision_follower
    """

    LANE1_X = 5.24
    LANE2_X = 8.74

    def on_reset(self) -> None:

        # --- Cybertruck (static closure in lane -1) ---
        nonego_y = float(np.random.uniform(
            self._config.nonego_spawn_y_range[0],
            self._config.nonego_spawn_y_range[1],
        ))
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

        # --- bg1 (lane -2, AHEAD of ego — lead vehicle, just cruises) ---
        # y=100-120: always ahead of ego (y=130)
        bg1_y = float(np.random.uniform(100.0, 120.0))
        self._bg_speed = float(np.random.uniform(3.0, 7.0))
        self.bg_vehicle = self._world.spawn_actor(
            transform=carla.Transform(
                carla.Location(x=self.LANE2_X, y=bg1_y, z=0.1),
                carla.Rotation(yaw=-90),
            )
        )

        # --- bg2 (lane -2, BEHIND ego — following vehicle) ---
        # y=145-160: behind ego (y=130), creates closing pressure
        # yields for bg1 + ego in merge zone
        # cruises freely once past closure
        bg2_y = float(np.random.uniform(145.0, 160.0))
        self._bg2_speed = float(np.random.uniform(4.0, 8.0))
        self.bg2_vehicle = self._world.spawn_actor(
            transform=carla.Transform(
                carla.Location(x=self.LANE2_X, y=bg2_y, z=0.1),
                carla.Rotation(yaw=-90),
            )
        )

        # --- Follower (lane -1, behind ego, brake pressure) ---
        # y=148-165: behind ego (y=130)
        follower_y = float(np.random.uniform(148.0, 165.0))
        self._follower_target_speed = float(np.random.uniform(3.0, 6.0))
        self.follower = self._world.spawn_actor(
            transform=carla.Transform(
                carla.Location(x=self.LANE1_X, y=follower_y, z=0.1),
                carla.Rotation(yaw=-90),
            )
        )

        # --- Ego vehicle (lane -1, y=130, fixed) ---
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

        # gap between bg1 (ahead) and closure
        self._prev_gap_size = max(0.0, bg1_y - nonego_y - 5.0)

        # gap between bg2 (behind) and bg1 (ahead) — the actual merge gap
        # bg2_y > bg1_y → gap12 = bg2_y - bg1_y - 5.0
        self._prev_gap12_size = max(0.0, bg2_y - bg1_y - 5.0)

        self.prev_steer = 0.0

        # --- Spectator: closure camera POV ---
        self._world._world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=self.LANE1_X, y=nonego_y + 3.0, z=5.0),
                carla.Rotation(pitch=-10, yaw=90),
            )
        )

    # =========================================================
    # State / planner accessors
    # =========================================================

    def get_state(self):
        state = super().get_state()
        state["nonego_location"] = self._nonego_location
        return state

    def get_ego_planner(self):
        return self.ego_planner

    # =========================================================
    # Vehicle controls
    # =========================================================

    def apply_control(self, action) -> None:
        self.ego.apply_control(self.get_vehicle_control(action))
        self.bg_vehicle.apply_control(self._get_bg1_control())
        self.bg2_vehicle.apply_control(self._get_bg2_control())
        self.follower.apply_control(self._get_follower_control())

    def _get_bg1_control(self):
        """
        bg1 — lead vehicle in lane -2, ahead of ego.
        Dumb actor: cruises at constant target speed.
        No awareness of other vehicles.
        """
        vel   = self.bg_vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
        throttle = 0.6 if speed < self._bg_speed else 0.0
        return carla.VehicleControl(
            throttle=float(throttle), steer=0.0, brake=0.0
        )

    def _get_bg2_control(self):
        """
        bg2 — following vehicle in lane -2, behind ego.

        PAST closure (bg2_y < nonego_y):
            Cruise freely — no braking, drives away cleanly.
            Prevents bg2 from stopping near closure and causing
            false collision when ego merges.

        BEFORE closure (bg2_y >= nonego_y):
            Aware of two things:
            1. bg1 ahead: brakes if closing too fast
            2. ego merging nearby: yields to give ego space

            dist > 25m to bg1  : accelerate
            15-25m to bg1      : cruise
            8-15m to bg1       : gentle brake
            < 8m to bg1        : emergency brake
            ego merging nearby : yield
        """
        bg1_y = self.bg_vehicle.get_location().y
        bg2_y = self.bg2_vehicle.get_location().y
        ego_x = self.ego.get_location().x
        ego_y = self.ego.get_location().y

        bg2_vel   = self.bg2_vehicle.get_velocity()
        bg2_speed = math.sqrt(bg2_vel.x ** 2 + bg2_vel.y ** 2)

        # Past closure — cruise freely, no braking
        if bg2_y < self._nonego_y:
            throttle = 0.6 if bg2_speed < self._bg2_speed else 0.0
            return carla.VehicleControl(
                throttle=float(throttle), steer=0.0, brake=0.0
            )

        # bg2 behind bg1 → dist positive = safe gap
        dist_to_bg1 = bg2_y - bg1_y

        # ego is merging if moved laterally toward lane -2
        ego_merging = ego_x > (self.LANE1_X + 1.0)
        ego_nearby  = abs(ego_y - bg2_y) < 12.0

        if dist_to_bg1 < 8.0:
            # emergency brake — about to hit bg1
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        elif dist_to_bg1 < 15.0:
            # gentle brake — closing on bg1
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
        elif ego_merging and ego_nearby:
            # ego merging right next to bg2 — yield
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.4)
        elif dist_to_bg1 < 25.0:
            # cruise — comfortable following distance
            throttle = 0.4 if bg2_speed < self._bg2_speed else 0.0
            return carla.VehicleControl(
                throttle=float(throttle), steer=0.0, brake=0.0
            )
        else:
            # accelerate — too far behind bg1
            throttle = 0.6 if bg2_speed < self._bg2_speed else 0.0
            return carla.VehicleControl(
                throttle=float(throttle), steer=0.0, brake=0.0
            )

    def _get_follower_control(self):
        """
        Follower — lane -1, behind ego.
        Applies increasing brake pressure as it closes on ego.

        dist > 25m : accelerate
        12-25m     : cruise
        8-12m      : gentle brake
        < 8m       : emergency brake
        """
        ego_y     = self.ego.get_location().y
        fol_y     = self.follower.get_location().y
        fol_vel   = self.follower.get_velocity()
        fol_speed = math.sqrt(fol_vel.x ** 2 + fol_vel.y ** 2)

        dist = fol_y - ego_y  # positive = follower behind ego

        if dist < 8.0:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        elif dist < 12.0:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.3)
        elif dist < 25.0:
            throttle = 0.4 if fol_speed < self._follower_target_speed else 0.0
            return carla.VehicleControl(
                throttle=float(throttle), steer=0.0, brake=0.0
            )
        else:
            throttle = 0.6 if fol_speed < self._follower_target_speed else 0.0
            return carla.VehicleControl(
                throttle=float(throttle), steer=0.0, brake=0.0
            )

    # =========================================================
    # Collision helpers
    # =========================================================

    def _hit_cybertruck(self):
        return (self.is_collision() and
                self.ego.get_location().distance(
                    self.nonego.get_location()) < 6.0)

    def _hit_bg_vehicle(self):
        return (self.is_collision() and
                self.ego.get_location().distance(
                    self.bg_vehicle.get_location()) < 6.0)

    def _hit_bg2_vehicle(self):
        """
        Only count bg2 collision if bg2 is still in the merge zone.
        Once bg2 passes closure it is irrelevant — ignore collisions.
        This prevents false positives from bg2 stopping near closure.
        """
        bg2_y = self.bg2_vehicle.get_location().y
        if bg2_y < self._nonego_y:
            return False
        return (self.is_collision() and
                self.ego.get_location().distance(
                    self.bg2_vehicle.get_location()) < 6.0)

    def _hit_follower(self):
        return (self.is_collision() and
                self.ego.get_location().distance(
                    self.follower.get_location()) < 6.0)

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

        scales   = self._config.reward.scales
        ego_x,  ego_y  = get_vehicle_pos(self.ego)
        bg_x,   bg_y   = get_vehicle_pos(self.bg_vehicle)    # bg1 ahead
        bg2_x,  bg2_y  = get_vehicle_pos(self.bg2_vehicle)   # bg2 behind
        fol_x,  fol_y  = get_vehicle_pos(self.follower)
        nonego_y = self._nonego_y

        ego_vel    = self.ego.get_velocity()
        vy         = float(ego_vel.y)
        speed_norm = math.sqrt(ego_vel.x ** 2 + vy ** 2)

        fol_vel        = self.follower.get_velocity()
        follower_speed = math.sqrt(fol_vel.x ** 2 + fol_vel.y ** 2)

        in_lane1 = abs(ego_x - self.LANE1_X) < 1.5
        in_lane2 = abs(ego_x - self.LANE2_X) < 1.5

        dist_to_closure = ego_y - nonego_y
        early_dist = self._config.reward.early_lane_change_dist

        # ---- Reward components ----
        p_stay = (
            -scales["stay_same_lane"]
            if (0 < dist_to_closure < early_dist and in_lane1) else 0.0
        )
        p_early = (
            -scales["early_lane_change"]
            if (dist_to_closure > early_dist and in_lane2) else 0.0
        )

        r_merge = 0.0
        if in_lane2 and ego_y < nonego_y and not self._merged:
            r_merge = scales["r_merge"]
            self._merged = True

        # compute collision flags once — reused for reward + info
        hit_truck = self._hit_cybertruck()
        hit_bg    = self._hit_bg_vehicle()
        hit_bg2   = self._hit_bg2_vehicle()
        hit_fol   = self._hit_follower()

        r_col_truck = -scales["collision"] if hit_truck else 0.0
        r_col_bg    = -scales["collision"] if hit_bg    else 0.0
        r_col_bg2   = -scales["collision"] if hit_bg2   else 0.0
        r_col_fol   = -scales["collision"] if hit_fol   else 0.0

        steer    = self.ego.get_control().steer
        p_smooth = -abs(steer - self.prev_steer) * scales["steer_smooth"]
        self.prev_steer = steer

        total_reward += (
            p_stay + p_early + r_merge
            + r_col_truck + r_col_bg + r_col_bg2 + r_col_fol
            + p_smooth
        )

        # ---- Infrastructure sensor features ----
        dt = self._config.world.fixed_delta_seconds

        lateral_change_rate = (ego_x - self._prev_ego_x) / dt
        self._prev_ego_x = ego_x

        # gap between bg1 (ahead) and closure
        gap_size = max(0.0, bg_y - nonego_y - 5.0)
        gap_closing_rate = (gap_size - self._prev_gap_size) / dt
        self._prev_gap_size = gap_size

        # gap between bg2 (behind) and bg1 (ahead) — actual merge gap
        # bg2_y > bg1_y → gap12 = bg2_y - bg1_y - 5.0
        gap12_size = max(0.0, bg2_y - bg_y - 5.0)
        gap12_closing_rate = (gap12_size - self._prev_gap12_size) / dt
        self._prev_gap12_size = gap12_size

        follower_dist = fol_y - ego_y  # positive = follower behind ego

        info.update({
            # 16 infrastructure-observable features
            "ego_x":               ego_x,
            "ego_y":               ego_y,
            "vy":                  vy,
            "speed_norm":          speed_norm,
            "lateral_change_rate": lateral_change_rate,
            "dist_to_closure":     dist_to_closure,
            "gap_size":            gap_size,
            "gap_closing_rate":    gap_closing_rate,
            "bg_y":                bg_y,
            "bg_speed":            self._bg_speed,
            "gap12_size":          gap12_size,
            "gap12_closing_rate":  gap12_closing_rate,
            "bg2_y":               bg2_y,
            "bg2_speed":           self._bg2_speed,
            "follower_dist":       follower_dist,
            "follower_speed":      follower_speed,
            # Collision type breakdown (analysis only, not terminal)
            "collision_closure":   hit_truck,
            "collision_traffic":   hit_bg or hit_bg2,
            "collision_follower":  hit_fol,
            # Episode state
            "in_lane1":            in_lane1,
            "in_lane2":            in_lane2,
            "merged":              self._merged,
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
        info["is_collision"] = (
            self._hit_cybertruck()  or
            self._hit_bg_vehicle()  or
            self._hit_bg2_vehicle() or  # false positives removed via _hit_bg2_vehicle()
            self._hit_follower()
        )

        return info
