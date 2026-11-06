"""
car_dreamer/carla_workzone_env.py

Work zone merge environment, rebuilt for the infrastructure-forecaster
project. Inherits CarlaBaseEnv directly -- NOT CarlaWptEnv.

Why not CarlaWptEnv
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
               30                 0               -45             -55
        s =   -30                  0               +45             +55
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
        self.INFLOW_RATE = float(c.inflow_rate)
        self.MIN_INFLOW_HEADWAY = float(c.min_inflow_headway)
        self.INFLOW_START_S = float(c.inflow_start_s)
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
        # sensors from the previous episode are attached to actors the
        # WorldManager is about to destroy
        self._destroy_collision_sensors()

        self._load_geometry()

        self._episode_index = getattr(self, "_episode_index", -1) + 1
        seed = int(getattr(self._config, "episode_seed", 0)) + self._episode_index
        self._rng = np.random.default_rng(seed)
        self._episode_seed = seed

        # episode state BEFORE spawning, so nothing reads it half-built
        self._n_ticks = 0
        self._resolved: Dict[int, str] = {}
        self._collisions: Dict[int, list] = {}
        self._blocked: Dict[int, int] = {}
        self._impact_s: Dict[int, float] = {}
        self._resolved_s: Dict[int, float] = {}
        self._final_s: Dict[int, float] = {}
        self._events: List[Tuple[int, int, str]] = []
        self.vehicles: List[carla.Actor] = []
        self.drivers: List[Driver] = []
        self._own_ids: set = set()
        self._sensor_by_vehicle: Dict[int, carla.Actor] = {}
        self._pending_inflow: List[Tuple[carla.Actor, object]] = []
        self._next_inflow_t = 0.0
        self._n_inflow = 0

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

    def _draw_headway(self) -> float:
        """
        Shifted-exponential time headway, in seconds.

        Plain exponential puts mass at zero headway, which is physically
        impossible for 4.8 m vehicles. Shifting by a floor keeps the
        Poisson character while respecting car-following.
        """
        mean = 3600.0 / self.INFLOW_RATE
        excess = max(mean - self.MIN_INFLOW_HEADWAY, 0.1)
        return self.MIN_INFLOW_HEADWAY + float(self._rng.exponential(excess))

    def _place_vehicle(self, lane_id: int, s: float, speed: float
                       ) -> Optional[carla.Actor]:
        """
        Put one vehicle in `lane_id` at corridor position `s`, already
        moving at `speed`. Returns None if the slot is occupied or the
        lane does not exist there.

        Inflow vehicles get their velocity at spawn. Starting them at rest
        would make every one accelerate through the approach, so the open
        lane would be a rolling queue rather than free-flowing traffic and
        the gaps a merger sees would be wrong. The vehicles seeded at reset
        are placed at rest instead; see _spawn_traffic.
        """
        y = self._taper_start_y - s
        wp = self._waypoint_in_lane(y, lane_id)
        if wp is None:
            return None

        v = self._world.try_spawn_actor(
            transform=self._transform_at(wp),
            blueprint=self._world.get_blueprint("vehicle.tesla.model3"))
        if v is None:
            return None

        if speed > 0.0:
            fwd = wp.transform.get_forward_vector()
            v.set_target_velocity(
                carla.Vector3D(fwd.x * speed, fwd.y * speed, 0.0))

        self.vehicles.append(v)
        self._attach_collision_sensor(v)
        return v

    def _spawn_traffic(self) -> None:
        """
        Closed-lane vehicles in disjoint slots, plus a seeded open-lane
        stream that _maybe_spawn_inflow then keeps fed.

        The open lane used to be a fixed set placed at t=0. That made
        every episode "merge problem for ~15 s, then empty road":
        occupancy near the taper peaked around t=7-10 s and reached zero
        by t=22, so the first car in the queue met traffic and the last
        met clear road. Queue position alone predicted the outcome at
        AUC ~0.81 -- a spawn artifact, not merge behaviour.
        """
        c = self._config
        n_closed = int(c.n_closed)
        headway = float(c.min_spawn_headway)

        span = abs(self.APPROACH_S)
        slots = max(int(span / headway), 2)

        pending: List[Tuple[carla.Actor, object]] = []
        placed_closed = 0
        placed_open = 0

        # closed lane: discrete slots, at rest
        chosen = self._rng.choice(slots, size=min(n_closed, slots),
                                  replace=False)
        for slot in chosen:
            s = self.APPROACH_S + (slot + self._rng.uniform(0.15, 0.85)) * headway
            v = self._place_vehicle(self.CLOSED_LANE, s, 0.0)
            if v is None:
                print(f"[spawn] closed slot {slot}: failed at s={s:.1f}")
                continue
            pending.append(
                (v, sample_driver_params(self._rng, self.POSTED_SPEED)))
            placed_closed += 1

        # open lane: seed the stream so it is already established at t=0
        s = self.INFLOW_START_S
        while s < self.TAPER_END_S:
            # At rest, NOT at speed. reset() runs in ASYNCHRONOUS mode, so
            # the server free-runs while the rest of the spawn loop executes
            # and a vehicle given velocity here travels ~200 m before the
            # episode starts -- past the despawn line, so it vanishes on the
            # first tick. Inflow vehicles spawn during synchronous steps and
            # do get their velocity.
            v = self._place_vehicle(self.OPEN_LANE, s, 0.0)
            if v is not None:
                pending.append(
                    (v, sample_driver_params(self._rng, self.POSTED_SPEED)))
                placed_open += 1
            s += self._draw_headway() * self.POSTED_SPEED

        print(f"[spawn] {placed_closed}/{n_closed} closed, "
              f"{placed_open} open seeded, "
              f"inflow={self.INFLOW_RATE:.0f} veh/h")

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

        # The observer needs an actor to attach to. Prefer a closed-lane
        # vehicle so the recorded view shows a driver that has to merge;
        # this confers no behavioural privilege.
        self._next_inflow_t = self._draw_headway()

        merging = [d for d in self.drivers
                   if d.origin_lane_id == self.CLOSED_LANE]
        self.ego = (merging[0] if merging else self.drivers[0]).vehicle

    def _maybe_spawn_inflow(self) -> None:
        """
        Feed the open lane continuously from the upstream boundary.

        Vehicles are promoted to Drivers one tick after placement. CARLA
        does not apply the spawn transform until the next tick, so a
        Driver built immediately would read a stale location and pick the
        wrong origin lane.
        """
        for v, params in self._pending_inflow:
            self.drivers.append(Driver(v, params, self._world.carla_map,
                                       self._closure_info()))
        self._pending_inflow = []

        t = self._n_ticks * self.DT
        if t < self._next_inflow_t:
            return

        v = self._place_vehicle(self.OPEN_LANE, self.INFLOW_START_S,
                                self.POSTED_SPEED)
        if v is None:
            # boundary occupied; retry at the floor rather than burning a
            # whole headway, which would thin the stream at high rates
            self._next_inflow_t = t + self.MIN_INFLOW_HEADWAY
            return

        self._pending_inflow.append(
            (v, sample_driver_params(self._rng, self.POSTED_SPEED)))
        self._n_inflow += 1
        self._next_inflow_t = t + self._draw_headway()

    def _attach_collision_sensor(self, v: carla.Actor) -> None:
        """
        One collision sensor per vehicle. CarlaBaseEnv.is_collision() reads
        a single global flag that cannot say WHICH vehicles collided;
        per-vehicle sensors give exact attribution and the partner id.

        `other_is_vehicle` resolves against self._own_ids at event time,
        not at attach time, because inflow vehicles appear later.
        """
        self._own_ids.add(v.id)
        bp = self._world.get_blueprint("sensor.other.collision")
        sensor = self._world.spawn_unmanaged_actor(
            carla.Transform(), bp, attach_to=v)

        def on_hit(event, vid=v.id):
            other = event.other_actor
            other_id = other.id if other is not None else -1
            imp = event.normal_impulse
            self._collisions.setdefault(vid, []).append({
                "tick": self._n_ticks,
                "other_id": other_id,
                "other_is_vehicle": other_id in self._own_ids,
                "impulse": round(
                    (imp.x ** 2 + imp.y ** 2 + imp.z ** 2) ** 0.5, 2),
            })

        sensor.listen(on_hit)
        self._collision_sensors.append(sensor)
        self._sensor_by_vehicle[v.id] = sensor

    def _destroy_collision_sensors(self) -> None:
        self._sensor_by_vehicle = {}
        for s in getattr(self, "_collision_sensors", []):
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self._collision_sensors = []

    def _blocked_by(self, subject: Driver) -> int:
        """
        Id of a stopped vehicle within 12 m ahead in the same lane, or -1.

        A follower halted by the queue never made a merge decision, so its
        stop says nothing about gap acceptance. Recording the blocker keeps
        the two cases separable.
        """
        s_me = subject.progress()
        lane = subject.current_waypoint().lane_id
        best, best_d = -1, 1e9
        for other in self.drivers:
            if other is subject or other.speed() >= 0.5:
                continue
            if other.current_waypoint().lane_id != lane:
                continue
            d = other.progress() - s_me
            if 0 < d < 12.0 and d < best_d:
                best, best_d = other.vehicle.id, d
        return best

    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego

    # ==========================================================
    # Per-step
    # ==========================================================

    def apply_control(self, action) -> None:
        """`action` is ignored. Every vehicle produces its own control."""
        alive = []
        for d in self.drivers:
            try:
                d.vehicle.apply_control(d.step(self._neighbours(d), self.DT))
                alive.append(d)
            except RuntimeError:
                pass          # actor gone; drop it from the population
        self.drivers = alive


    def on_step(self) -> None:
        """Called by WorldManager.step() after the tick."""
        self._n_ticks += 1
        self._maybe_spawn_inflow()

        for d in self.drivers:
            update_stop_tracking(d, self.DT)

            # Collision is checked BEFORE the resolved-skip, because a
            # vehicle that stopped and then crashed would otherwise keep
            # its stop label -- that mislabelled most collisions.
            #
            # It counts as a merge failure only for a closed-lane vehicle
            # that is at or past the taper start. A rear-end back in the
            # approach is ordinary queue traffic, not a merge decision, so
            # it is recorded as an event without an outcome. The impact
            # position is stored either way.
            if (self._collisions.get(d.vehicle.id)
                    and self._resolved.get(d.vehicle.id) != "collision"):
                if d.vehicle.id not in self._impact_s:
                    self._impact_s[d.vehicle.id] = round(d.progress(), 2)
                in_zone = (d.origin_lane_id == self.CLOSED_LANE
                           and self._impact_s[d.vehicle.id] >= self.TAPER_START_S)
                if in_zone:
                    self._resolved[d.vehicle.id] = "collision"
                    self._events.append(
                        (self._n_ticks, d.vehicle.id, "collision"))
                    continue

            if d.vehicle.id in self._resolved:
                continue

            outcome = classify_vehicle(d, self.TAPER_START_S, self.TAPER_END_S)
            if outcome is not None:
                self._resolved[d.vehicle.id] = outcome.value
                self._resolved_s[d.vehicle.id] = round(d.progress(), 2)
                if outcome.value == "stop_in_lane":
                    self._blocked[d.vehicle.id] = self._blocked_by(d)
                self._events.append((self._n_ticks, d.vehicle.id, outcome.value))

        # Despawn vehicles past the work zone. Compare the actor OBJECT,
        # not the id -- CARLA recycles ids across episodes, so an id check
        # can destroy the ego by mistake.
        # last known position for every live vehicle, refreshed each tick so
        # it survives despawn. Lets intrusion be an analysis-time query
        # instead of something baked into the outcome label.
        for d in self.drivers:
            self._final_s[d.vehicle.id] = round(d.progress(), 2)

        keep = []
        for d in self.drivers:
            past = d.progress() > self.BUFFER_END_S + 30
            if past and d.vehicle is not self.ego:
                # Destroy the sensor BEFORE its parent. An orphaned
                # collision sensor keeps streaming to a dead actor
                # ("Invalid session: no stream available"), and with
                # continuous inflow these accumulate every episode until
                # the server segfaults.
                s_ = self._sensor_by_vehicle.pop(d.vehicle.id, None)
                if s_ is not None:
                    try:
                        s_.stop()
                        s_.destroy()
                    except Exception:
                        pass
                    if s_ in self._collision_sensors:
                        self._collision_sensors.remove(s_)
                try:
                    d.vehicle.destroy()
                except Exception:
                    pass
            else:
                keep.append(d)
        self.drivers = keep

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

        # The work vehicle lives in self.work_vehicle, not self.drivers, so
        # the loop above never returned it and no driver's IDM ever braked
        # for the closure. 90% of all collisions trace back to this. It is a
        # physical obstacle in the closed lane and must be a lead vehicle.
        if own_lane == self.CLOSED_LANE:
            d_wv = self.progress(self.work_vehicle.get_location()) - s_self
            if d_wv > 0:
                clear = half_self + self.work_vehicle.bounding_box.extent.x
                if own_lead is None or d_wv < own_lead["_d"]:
                    own_lead = {"gap": max(d_wv - clear, 0.0),
                                "speed": 0.0, "_d": d_wv}

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
            "n_inflow": self._n_inflow,
            "n_resolved": len(self._resolved),
            "outcomes": dict(self._resolved),
            "collisions": {k: list(v) for k, v in self._collisions.items()},
            "blocked_by": dict(self._blocked),
            "impact_s": dict(self._impact_s),
            "resolved_s": dict(self._resolved_s),
            "final_s": dict(self._final_s),
            "n_collisions": sum(len(v) for v in self._collisions.values()),
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

        # A vehicle part-way across should finish. Without this an episode
        # can end while a merge is in progress, labelling that vehicle by
        # whatever state it happened to be in.
        mid_merge = any(d.state == MergeState.COMMITTED
                        and d.vehicle.id not in self._resolved
                        for d in self.drivers)

        all_resolved = (bool(closed)
                        and all(d.vehicle.id in self._resolved for d in closed)
                        and not mid_merge)

        cleared = (bool(self.drivers)
                   and all(d.progress() >= self.BUFFER_END_S for d in self.drivers)
                   and not mid_merge)

        return {
            "all_resolved": all_resolved,
            "cleared_corridor": cleared,
            "time_exceeded": self._n_ticks >= self.STEP_CAP,
        }

    def get_state(self) -> Dict:
        wv = self.work_vehicle.get_location()
        return {
            "timesteps": self._n_ticks,
            "taper_start_s": self.TAPER_START_S,
            "taper_end_s": self.TAPER_END_S,
            # ClosureCameraHandler reads this to position itself. Without
            # it the camera stays at its spawn transform (world origin).
            "nonego_location": (wv.x, wv.y),
        }

