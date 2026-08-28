"""
car_dreamer/carla_workzone_env.py

Work zone merge environment, rebuilt for the infrastructure-forecaster
project. Inherits CarlaBaseEnv directly -- NOT CarlaWptEnv.

Why not CarlaWptEnv (see PROJECT_MEMO.md 3.6):
  - it is a waypoint-FOLLOWING base class, and this task requires
    vehicles to LEAVE the planned path
  - its r_speed penalises lateral velocity
  - out_of_lane is measured against the closed lane's own waypoints
  - destination_reached is an independent terminal condition

What this env does NOT have, deliberately:
  - no ego. Every vehicle is a Driver instance with sampled parameters.
    Whichever ones spawn in the closed lane must merge; the rest drive.
  - no reward shaping. reward() returns 0.0 and exists only as the
    per-step logging channel (reward_info is merged into info).
  - no scripted cooperation. Nothing yields to anyone.

Two WorldManager behaviours this env is written around:
  1. reset() runs on_reset() while the world is in ASYNCHRONOUS mode, so
     the server free-runs during spawning. A sleep settles the vehicles;
     calling world.step() here would fire on_step() before episode state
     exists.
  2. step() calls world.tick() and then on_step(), and CarlaBaseEnv.step()
     separately increments self._time_step. This env keeps its own
     self._n_ticks to avoid double counting.
"""

import time
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np

from .carla_base_env import CarlaBaseEnv

# The driver model and taxonomy live outside the car_dreamer package.
from workzone.driver.driver import Driver, MergeState, sample_driver_params
from workzone.taxonomy.events import classify_vehicle, update_stop_tracking


class CarlaWorkzoneEnv(CarlaBaseEnv):
    """
    Work zone lane closure with a heterogeneous population of drivers.

    Scene (Town04, travel toward decreasing y):

        approach_start_y ... taper_start_y ... taper_end_y ... buffer_end_y
              230                 0              -165            -215
                                                                  ^
                                                          work vehicle

        lane -1 (closed) : vehicles here must merge before taper_end
        lane -2 (open)   : vehicles here just drive

    Longitudinal position is reported as `s`, distance along travel with
    taper_start at 0, so nothing downstream reads a raw world coordinate.
    """

    # ==========================================================
    # Geometry
    # ==========================================================

    def progress(self, loc: carla.Location) -> float:
        """Distance along the corridor, increasing forward."""
        return -(loc.y - self._taper_start_y)

    def _load_geometry(self) -> None:
        c = self._config
        self.CLOSED_LANE = int(c.closed_lane_id)
        self.OPEN_LANE = int(c.open_lane_id)
        self.LANE_REF_X = float(c.lane_ref_x)

        self._taper_start_y = float(c.taper_start_y)
        self.APPROACH_S = -(float(c.approach_start_y) - self._taper_start_y)
        self.TAPER_START_S = 0.0
        self.TAPER_END_S = -(float(c.taper_end_y) - self._taper_start_y)
        self.BUFFER_END_S = -(float(c.buffer_end_y) - self._taper_start_y)

        self.POSTED_SPEED = float(c.posted_speed)
        self.STEP_CAP = int(c.step_cap)
        self.DT = float(c.world.fixed_delta_seconds)

    def _closure_info(self) -> Dict:
        return {
            "closed_lane_id": self.CLOSED_LANE,
            "open_lane_id": self.OPEN_LANE,
            "taper_start_s": self.TAPER_START_S,
            "taper_end_s": self.TAPER_END_S,
            "progress_fn": self.progress,
        }

    def _waypoint_in_lane(self, y: float, lane_id: int
                          ) -> Optional[carla.Waypoint]:
        amap = self._world.carla_map
        ref = amap.get_waypoint(carla.Location(x=self.LANE_REF_X, y=y, z=0.3),
                                project_to_road=True,
                                lane_type=carla.LaneType.Driving)
        if ref.lane_id == lane_id:
            return ref
        for nb in (ref.get_left_lane(), ref.get_right_lane()):
            if nb is not None and nb.lane_id == lane_id:
                return nb
        return None

    @staticmethod
    def _transform_at(wp: carla.Waypoint) -> carla.Transform:
        return carla.Transform(
            carla.Location(wp.transform.location.x,
                           wp.transform.location.y,
                           wp.transform.location.z + 0.3),   # 0.3, verified
            wp.transform.rotation)

    # ==========================================================
    # Reset
    # ==========================================================

    def on_reset(self) -> None:
        """
        NOTE: WorldManager runs this in ASYNCHRONOUS mode. Do not call
        self._world.step() here -- it would fire on_step() before episode
        state exists. The server free-runs, so a sleep settles physics.
        """
        self._load_geometry()

        self._episode_index = getattr(self, "_episode_index", -1) + 1
        seed = int(getattr(self._config, "episode_seed", 0)) + self._episode_index
        self._rng = np.random.default_rng(seed)
        self._episode_seed = seed

        # episode state BEFORE spawning, so nothing reads it half-built
        self._n_ticks = 0
        self._resolved: Dict[int, str] = {}
        self._events: List[Tuple[int, int, str]] = []
        self.vehicles: List[carla.Actor] = []
        self.drivers: List[Driver] = []

        self._spawn_work_vehicle()
        self._spawn_traffic()

    def _spawn_work_vehicle(self) -> None:
        """Physical hazard at the end of the buffer. Static, no physics."""
        y = self._taper_start_y - self.BUFFER_END_S
        wp = self._waypoint_in_lane(y, self.CLOSED_LANE)
        if wp is None:
            raise RuntimeError(f"no closed lane at work-vehicle y={y:.1f}")

        self.work_vehicle = self._world.spawn_actor(
            transform=self._transform_at(wp),
            blueprint=self._world.get_blueprint("vehicle.tesla.cybertruck"))
        self.work_vehicle.set_simulate_physics(False)

    def _spawn_traffic(self) -> None:
        """
        Lay vehicles down in disjoint longitudinal slots, allocated
        SEPARATELY per lane.

        Sharing one slot pool across both lanes packed the open lane at
        ~29 m spacing -- 1.5 s gaps at 20 m/s, against critical gaps of
        2.5-6 s. No gap was ever acceptable and every closed-lane driver
        stopped at the taper. Per-lane allocation keeps open-lane spacing
        at a full headway apart.
        """
        c = self._config
        n = int(c.n_vehicles)
        closed_frac = float(c.closed_lane_fraction)
        headway = float(c.min_spawn_headway)

        span = abs(self.APPROACH_S)              # 230 m of approach
        slots = max(int(span / headway), 2)

        n_closed = int(round(n * closed_frac))
        lane_plan = [(self.CLOSED_LANE, n_closed),
                     (self.OPEN_LANE, n - n_closed)]

        pending: List[Tuple[carla.Actor, object]] = []

        for lane, count in lane_plan:
            if count <= 0:
                continue
            chosen = self._rng.choice(slots, size=min(count, slots),
                                      replace=False)
            for slot in chosen:
                s = self.APPROACH_S + (slot + self._rng.uniform(0.15, 0.85)) * headway
                y = self._taper_start_y - s

                wp = self._waypoint_in_lane(y, lane)
                if wp is None:
                    print(f"[spawn] lane {lane} slot {slot}: "
                          f"no lane at y={y:.1f}")
                    continue

                v = self._world.try_spawn_actor(
                    transform=self._transform_at(wp),
                    blueprint=self._world.get_blueprint("vehicle.tesla.model3"))
                if v is None:
                    print(f"[spawn] lane {lane} slot {slot}: "
                          f"spawn failed at y={y:.1f}")
                    continue

                self.vehicles.append(v)
                pending.append(
                    (v, sample_driver_params(self._rng, self.POSTED_SPEED)))

        n_c = sum(1 for v in self.vehicles
                  if self._world.carla_map.get_waypoint(
                      v.get_location(), project_to_road=True,
                      lane_type=carla.LaneType.Driving).lane_id == self.CLOSED_LANE)
        print(f"[spawn] {len(self.vehicles)}/{n} placed "
              f"({n_c} closed, {len(self.vehicles) - n_c} open), "
              f"slots={slots}, headway={headway:.0f}m")

        if not self.vehicles:
            raise RuntimeError("no vehicles spawned")

        # The world is free-running here. Let physics settle before reading
        # positions -- CARLA does not apply the spawn transform immediately,
        # and a stale get_location() assigns every driver to the wrong lane.
        time.sleep(float(getattr(c, "settle_seconds", 0.6)))

        amap = self._world.carla_map
        closure = self._closure_info()
        for v, params in pending:
            self.drivers.append(Driver(v, params, amap, closure))

        # the observer needs an actor to attach to; this confers no
        # behavioural privilege -- it is a Driver like all the others
        self.ego = self.vehicles[0]

    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego

    # ==========================================================
    # Per-step
    # ==========================================================

    def apply_control(self, action) -> None:
        """`action` is ignored. Every vehicle produces its own control."""
        for d in self.drivers:
            d.vehicle.apply_control(d.step(self._neighbours(d), self.DT))

    def on_step(self) -> None:
        """Called by WorldManager.step() after the tick."""
        self._n_ticks += 1

        for d in self.drivers:
            update_stop_tracking(d, self.DT)
            if d.vehicle.id in self._resolved:
                continue
            outcome = classify_vehicle(d, self.TAPER_START_S, self.TAPER_END_S)
            if outcome is not None:
                self._resolved[d.vehicle.id] = outcome.value
                self._events.append((self._n_ticks, d.vehicle.id, outcome.value))

        # Despawn vehicles past the work zone -- they are outside the
        # sensed region, and Town04's ramp junctions confuse the
        # lane-follower down there. Never despawn the ego: the observer
        # holds a reference to it.
        still_here = []
        for d in self.drivers:
            if (d.vehicle.id != self.ego.id
                    and d.progress() > self.BUFFER_END_S + 30):
                try:
                    d.vehicle.destroy()
                except Exception:
                    pass
            else:
                still_here.append(d)
        self.drivers = still_here

    def _neighbours(self, subject: Driver) -> Dict:
        """
        What one driver can observe: the vehicle ahead in its own lane, and
        the lead/lag pair in the lane it is targeting. Purely relative --
        no vehicle is told that any other is special.
        """
        s_self = subject.progress()
        own_lane = subject.current_waypoint().lane_id
        tgt_lane = subject.target_lane_id
        half_self = subject.vehicle.bounding_box.extent.x

        own_lead = target_lead = target_lag = None

        for other in self.drivers:
            if other is subject:
                continue
            s_o = other.progress()
            lane_o = other.current_waypoint().lane_id
            d = s_o - s_self
            clear = half_self + other.vehicle.bounding_box.extent.x
            info = {"gap": max(abs(d) - clear, 0.0), "speed": other.speed()}

            if lane_o == own_lane and d > 0:
                if own_lead is None or d < own_lead["_d"]:
                    own_lead = {**info, "_d": d}
            if lane_o == tgt_lane:
                if d > 0 and (target_lead is None or d < target_lead["_d"]):
                    target_lead = {**info, "_d": d}
                if d < 0 and (target_lag is None or -d < target_lag["_d"]):
                    target_lag = {**info, "_d": -d}

        return {"own_lead": own_lead,
                "target_lead": target_lead,
                "target_lag": target_lag}

    # ==========================================================
    # Logging channel
    # ==========================================================

    def reward(self) -> Tuple[float, Dict]:
        """
        No RL here. Returns 0.0. This method exists because CarlaBaseEnv
        merges reward_info into info every step, which makes it the natural
        place to emit per-vehicle track data.

        Tracks are emitted at OBJECT level (position, velocity, heading,
        extent) rather than as rendered BEV frames. Sensing degradation is
        injected here, before rasterisation, which is where a real roadside
        sensor's error actually lives -- and it keeps the dataset in
        single-digit GB instead of over a terabyte.
        """
        tracks = []
        for d in self.drivers:
            tf = d.vehicle.get_transform()
            vel = d.vehicle.get_velocity()
            ext = d.vehicle.bounding_box.extent
            tracks.append({
                "id": d.vehicle.id,
                "s": round(d.progress(), 3),
                "lane": d.current_waypoint().lane_id,
                "origin_lane": d.origin_lane_id,
                "x": round(tf.location.x, 3),
                "y": round(tf.location.y, 3),
                "yaw": round(tf.rotation.yaw, 2),
                "vx": round(vel.x, 3),
                "vy": round(vel.y, 3),
                "speed": round(d.speed(), 3),
                "half_len": round(ext.x, 3),
                "half_wid": round(ext.y, 3),
                "state": d.state.value,
                "n_aborts": d.n_aborts,
            })

        n_closed = sum(1 for d in self.drivers
                       if d.origin_lane_id == self.CLOSED_LANE)

        info = {
            "tracks": tracks,
            "n_ticks": self._n_ticks,
            "episode_seed": self._episode_seed,
            "n_vehicles": len(self.drivers),
            "n_closed_lane": n_closed,
            "n_resolved": len(self._resolved),
            "outcomes": dict(self._resolved),
        }
        return 0.0, info

    # ==========================================================
    # Terminal conditions
    # ==========================================================

    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        ONLY genuine episode endings. CarlaBaseEnv._is_terminal ends the
        episode on ANY True key, so secondary flags must never appear here.
        """
        closed = [d for d in self.drivers
                  if d.origin_lane_id == self.CLOSED_LANE]

        all_resolved = bool(closed) and all(
            d.vehicle.id in self._resolved for d in closed)

        cleared = bool(self.drivers) and all(
            d.progress() >= self.BUFFER_END_S for d in self.drivers)

        return {
            "all_resolved": all_resolved,
            "cleared_corridor": cleared,
            "time_exceeded": self._n_ticks >= self.STEP_CAP,
        }

    def get_state(self) -> Dict:
        return {
            "timesteps": self._n_ticks,
            "taper_start_s": self.TAPER_START_S,
            "taper_end_s": self.TAPER_END_S,
        }
