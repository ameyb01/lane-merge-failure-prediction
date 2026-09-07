"""
workzone/driver/driver.py

A single driver model. Every vehicle in the scene gets one -- there is no
ego privilege and no actor knows it is special. Whether a vehicle behaves
like a lead vehicle, a follower, or a merger is a consequence of its
sampled parameters and its lane, not of a hand-written script.

Longitudinal : IDM (Treiber, Hennecke, Helbing 2000) + closed-loop speed
               tracking. Open-loop throttle = accel/3, inherited from
               CarDreamer's get_vehicle_control, does not track a target
               speed and plateaus near 11.5 m/s against drag.
Lateral      : pure pursuit (validated at 19.2 m/s over 446 m in Town04)
Merge        : gap acceptance against a per-driver critical gap
Abort        : mechanistic -- triggered when the accepted gap deteriorates.
               NOT a sampled probability. An injected abort rate would mean
               predicting a parameter we injected ourselves.

Every constant marked [ASSUMED] needs a citation or a sensitivity sweep.
See PROJECT_MEMO.md section 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import carla
import numpy as np


# ============================================================
# Parameters
# ============================================================

@dataclass(frozen=True)
class DriverParams:
    """Per-driver behavioural parameters, sampled once at spawn."""

    # --- merge decision ---
    critical_gap: float          # s   -- min acceptable lead AND lag time gap
    merge_init_time: float       # s   -- time-to-taper-end at which seeking starts
    abort_margin: float          # -   -- abort if gap falls below this * critical_gap

    # --- IDM longitudinal ---
    desired_speed: float         # m/s
    time_headway: float          # s
    min_gap: float               # m
    max_accel: float             # m/s^2
    comfort_decel: float         # m/s^2

    # --- vehicle / controller ---
    wheelbase: float = 2.9        # m  -- Tesla Model 3; set per blueprint
    lookahead: float = 10.0       # m  -- pure pursuit
    kp: float = 0.35              # speed-tracking proportional gain
    ki: float = 0.05              # speed-tracking integral gain
    merge_duration: float = 3.0   # s  -- [ASSUMED] time to complete the crossing
    creep_speed: float = 2.0      # m/s [ASSUMED] merge-from-queue speed

def sample_driver_params(rng: np.random.Generator,
                         posted_speed: float = 20.12) -> DriverParams:
    """
    Draw one driver.

    RANGES BELOW ARE PLACEHOLDERS. Each needs a source before any result
    is reported. Candidates: freeway work zone gap-acceptance studies for
    critical_gap and merge_init_time; Treiber et al. for the IDM block.
    """
    return DriverParams(
        # [ASSUMED] freeway merge critical gaps are usually quoted in the
        # 2.5-6 s range; this spread is a guess pending calibration
        critical_gap=float(rng.uniform(1.0, 3.0)),

        # [ASSUMED] time-to-taper-end at which a driver starts looking
        merge_init_time=float(rng.uniform(3.0, 7.0)),

        # [ASSUMED] fraction of critical_gap below which a committed
        # driver pulls back
        abort_margin=float(rng.uniform(0.7, 0.95)),

        # posted speed +/- 15%, a common desired-speed spread
        desired_speed=float(posted_speed * rng.uniform(0.85, 1.15)),

        # IDM ranges, roughly Treiber et al. -- verify before citing
        time_headway=float(rng.uniform(1.0, 2.0)),
        min_gap=float(rng.uniform(2.0, 3.0)),
        max_accel=float(rng.uniform(0.8, 2.0)),
        comfort_decel=float(rng.uniform(1.5, 2.5)),
    )


class MergeState(Enum):
    APPROACHING = "approaching"   # not yet looking for a gap
    SEEKING     = "seeking"       # looking, no gap accepted
    COMMITTED   = "committed"     # gap accepted, moving across
    MERGED      = "merged"        # fully in the target lane
    NOT_MERGING = "not_merging"   # already in the open lane; nothing to do


# ============================================================
# Geometry helpers
# ============================================================

def lane_neighbour(wp: carla.Waypoint, target_lane_id: int
                   ) -> Optional[carla.Waypoint]:
    """
    Adjacent waypoint with the given lane_id, without assuming which side
    it is on. In Town04 the open lane -2 is to the RIGHT of the closed
    lane -1, and get_left_lane() returns oncoming lane 1 -- but that is
    verified rather than assumed here.
    """
    for nb in (wp.get_left_lane(), wp.get_right_lane()):
        if nb is not None and nb.lane_id == target_lane_id:
            return nb
    return None


def advance(wp: carla.Waypoint, distance: float,
            lane_id: int) -> Optional[carla.Waypoint]:
    """
    wp.next() branches at junctions. Return None rather than guessing --
    Town04's highway is flagged as a junction wherever a ramp joins, and
    falling back to opts[0] can steer a vehicle onto the ramp.
    """
    opts = wp.next(distance)
    if not opts:
        return None
    same = [w for w in opts if w.lane_id == lane_id]
    return same[0] if same else None


def signed_lateral_offset(loc: carla.Location, wp: carla.Waypoint) -> float:
    """Lateral offset from a lane centreline. Sign is consistent, not absolute."""
    fwd = wp.transform.get_forward_vector()
    dx = loc.x - wp.transform.location.x
    dy = loc.y - wp.transform.location.y
    return dx * (-fwd.y) + dy * fwd.x


# ============================================================
# Driver
# ============================================================

class Driver:
    """
    One driver. Call step() once per simulation tick.

    `closure_info` carries the work zone geometry:
        closed_lane_id, open_lane_id, taper_start_s, taper_end_s, progress_fn
    where `s` is longitudinal distance measured along travel. Nothing here
    reads a raw world coordinate, so the model survives a geometry or map
    change.
    """

    def __init__(self, vehicle: carla.Actor, params: DriverParams,
                 carla_map: carla.Map, closure_info: Dict):
        self.vehicle = vehicle
        self.p = params
        self.map = carla_map
        self.closure = closure_info

        wp = self.map.get_waypoint(vehicle.get_location(), project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        self.origin_lane_id = wp.lane_id

        if wp.lane_id == closure_info["closed_lane_id"]:
            self.state = MergeState.APPROACHING
            self.target_lane_id = closure_info["open_lane_id"]
        else:
            self.state = MergeState.NOT_MERGING
            self.target_lane_id = wp.lane_id

        self.n_aborts = 0
        self.merge_init_progress: Optional[float] = None
        self.merge_complete_progress: Optional[float] = None

        # longitudinal controller state
        self.v_cmd = 0.0
        self._ierr = 0.0

        self.last_debug: Dict = {}

    # ---------- observations ----------

    def speed(self) -> float:
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2)

    def current_waypoint(self) -> carla.Waypoint:
        return self.map.get_waypoint(self.vehicle.get_location(),
                                     project_to_road=True,
                                     lane_type=carla.LaneType.Driving)

    def progress(self) -> float:
        """Distance along the corridor, increasing in the direction of travel."""
        return self.closure["progress_fn"](self.vehicle.get_location())

    def distance_to_taper_end(self) -> float:
        return self.closure["taper_end_s"] - self.progress()

    def time_to_taper_end(self) -> float:
        """Seconds until the closed lane runs out, at CURRENT speed."""
        v = max(self.speed(), 0.5)
        return self.distance_to_taper_end() / v

    def time_to_taper_end_free(self) -> float:
        """
        Horizon at DESIRED speed. Used for the seeking trigger.

        Braking must not delay the decision to start looking: if it does,
        slowing down and deciding late reinforce each other -- the driver
        brakes for the closure, current-speed ttt rises, seeking is
        postponed, and it brakes harder. That deadlock produced 17/20
        STOP_IN_LANE outcomes before this was split out.
        """
        return self.distance_to_taper_end() / self.p.desired_speed

    # ---------- longitudinal: IDM ----------

    def idm_accel(self, lead: Optional[Dict]) -> float:
        v = self.speed()
        v0, T = self.p.desired_speed, self.p.time_headway
        s0, a, b = self.p.min_gap, self.p.max_accel, self.p.comfort_decel

        free = 1.0 - (v / v0) ** 4

        if lead is None:
            return a * free

        s = max(lead["gap"], 0.1)
        dv = v - lead["speed"]
        s_star = s0 + max(0.0, v * T + (v * dv) / (2.0 * math.sqrt(a * b)))
        return a * (free - (s_star / s) ** 2)

    def closure_as_lead(self) -> Optional[Dict]:
        """
        The end of the closed lane behaves like a stationary vehicle. A
        driver who has not merged must decelerate for it, which is what
        produces STOP_IN_LANE and forced merges without scripting either.

        Released once COMMITTED: a driver mid-crossing needs forward speed
        to complete the lateral move, and the room check in update_state
        already guarantees it committed with enough space.
        """
        if self.state in (MergeState.COMMITTED, MergeState.MERGED,
                          MergeState.NOT_MERGING):
            return None
        return {"gap": max(self.distance_to_taper_end(), 0.1), "speed": 0.0}

    @staticmethod
    def most_constraining(*leads) -> Optional[Dict]:
        """Of several possible constraints, IDM should respond to the closest."""
        real = [l for l in leads if l is not None]
        return min(real, key=lambda l: l["gap"]) if real else None

    def longitudinal_control(self, accel: float, dt: float,
                             lead: Optional[Dict] = None):
        """
        Integrate the IDM acceleration into a speed command, then close the
        loop on it with a PI controller. Returns (throttle, brake).
        """
        self.v_cmd = float(np.clip(self.v_cmd + accel * dt,
                                   0.0, self.p.desired_speed * 1.2))

        # Creep so a committed driver can still steer across -- pure pursuit
        # produces almost no steering below ~1 m/s. But never creep into
        # something: an unconditional floor drove committed vehicles at
        # 2.1 m/s into stationary cars, and suppressing it on any braking
        # at all left them deadlocked mid-merge. Suppress only when a
        # vehicle is genuinely close ahead.
        if self.state == MergeState.COMMITTED:
            blocked = lead is not None and lead["gap"] < 6.0
            if not blocked:
                self.v_cmd = max(self.v_cmd, self.p.creep_speed)

        err = self.v_cmd - self.speed()
        self._ierr = float(np.clip(self._ierr + err * dt, -10.0, 10.0))
        u = self.p.kp * err + self.p.ki * self._ierr
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, 1.0))

    def pure_pursuit_steer(self, target_lane_id: int) -> float:
        t = self.vehicle.get_transform()
        wp = self.current_waypoint()

        aim = wp
        if wp.lane_id != target_lane_id:
            nb = lane_neighbour(wp, target_lane_id)
            if nb is not None:
                aim = nb

        tgt = advance(aim, self.p.lookahead, aim.lane_id)
        if tgt is None:
            return 0.0

        yaw = math.radians(t.rotation.yaw)
        dx = tgt.transform.location.x - t.location.x
        dy = tgt.transform.location.y - t.location.y
        fx = dx * math.cos(yaw) + dy * math.sin(yaw)
        fy = -dx * math.sin(yaw) + dy * math.cos(yaw)

        steer = math.atan2(2.0 * self.p.wheelbase * fy, max(fx, 1e-3) ** 2)
        return float(np.clip(steer, -1.0, 1.0))

    # ---------- merge decision ----------

    def evaluate_gap(self, neighbours: Dict) -> Dict:
        """
        Gap acceptance against a per-driver critical gap.

        lead_time : how long until I close on the vehicle ahead in the
                    target lane, at the rate I am closing on it
        lag_time  : how long until the vehicle behind in the target lane
                    reaches me, at the rate it is closing on me

        Both use CLOSING speed, not absolute speed. Dividing by absolute
        speed reports a vehicle 39 m back travelling at nearly my own speed
        as 2.8 s away when the true figure is 50 s, which made drivers abort
        merges they had already almost completed.

        Both must exceed critical_gap. Time-based, so it stays meaningful
        across speeds.
        """
        lead = neighbours.get("target_lead")
        lag = neighbours.get("target_lag")

        v = max(self.speed(), 0.5)

        if lead:
            lead_time = lead["gap"] / max(v - lead["speed"], 0.1)
        else:
            lead_time = float("inf")

        if lag:
            lag_time = lag["gap"] / max(lag["speed"], 0.5)
        else:
            lag_time = float("inf")

        binding = min(lead_time, lag_time)
        return {
            "lead_time": lead_time,
            "lag_time": lag_time,
            "binding": binding,
            "acceptable": binding >= self.p.critical_gap,
        }

    def update_state(self, neighbours: Dict) -> Dict:
        if self.state in (MergeState.NOT_MERGING, MergeState.MERGED):
            return {}

        gap = self.evaluate_gap(neighbours)

        if self.state == MergeState.APPROACHING:
            # free-speed horizon, so braking cannot postpone the decision
            if self.time_to_taper_end_free() <= self.p.merge_init_time:
                self.state = MergeState.SEEKING

        if self.state == MergeState.SEEKING:
            # A gap is only usable if there is still room to complete the
            # lateral move before the closed lane runs out. Without this,
            # drivers commit at the last metre and get cut off mid-crossing.
            room = self.distance_to_taper_end()
            need = min(max(self.speed(), 1.0) * self.p.merge_duration, 15.0)
            if gap["acceptable"] and room >= need:
                self.state = MergeState.COMMITTED
                self.merge_init_progress = self.progress()

        elif self.state == MergeState.COMMITTED:
            # mechanistic abort: the accepted gap deteriorated
            if gap["binding"] < self.p.critical_gap * self.p.abort_margin:
                self.state = MergeState.SEEKING
                self.n_aborts += 1
                self.merge_init_progress = None
            elif self.fully_in_lane(self.target_lane_id):
                self.state = MergeState.MERGED
                self.merge_complete_progress = self.progress()

        return gap

    def fully_in_lane(self, lane_id: int) -> bool:
        """
        Merge is complete once the vehicle is in the target lane and the
        bulk of its footprint is inside it. A strict full-footprint test
        never passes for a vehicle creeping across from a queue, which is
        exactly the case we most need to classify correctly.
        """
        wp = self.current_waypoint()
        if wp.lane_id != lane_id:
            return False
        half_w = self.vehicle.bounding_box.extent.y
        offset = abs(signed_lateral_offset(self.vehicle.get_location(), wp))
        return offset <= wp.lane_width / 2.0 - 0.3 * half_w
    # ---------- main ----------

    def step(self, neighbours: Dict, dt: float) -> carla.VehicleControl:
        gap = self.update_state(neighbours)

        if self.state in (MergeState.COMMITTED, MergeState.MERGED):
            steer_lane = self.target_lane_id
            lead = self.most_constraining(neighbours.get("target_lead"),
                                          neighbours.get("own_lead"))
        else:
            steer_lane = self.origin_lane_id
            lead = self.most_constraining(neighbours.get("own_lead"),
                                          self.closure_as_lead())

        accel = self.idm_accel(lead)
        throttle, brake = self.longitudinal_control(accel, dt, lead)
        steer = self.pure_pursuit_steer(steer_lane)

        self.last_debug = {
            "state": self.state.value,
            "v": round(self.speed(), 2),
            "v_cmd": round(self.v_cmd, 2),
            "v0": round(self.p.desired_speed, 2),
            "ttt": round(self.time_to_taper_end(), 2),
            "ttt_free": round(self.time_to_taper_end_free(), 2),
            "thr": round(throttle, 2),
            "brk": round(brake, 2),
            "n_aborts": self.n_aborts,
            **{k: (round(v, 2) if isinstance(v, float) and math.isfinite(v)
                   else v) for k, v in gap.items()},
        }

        # NOTE: CarDreamer's get_vehicle_control negates steer. Kept
        # positive here; flip if the vehicle turns the wrong way.
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
