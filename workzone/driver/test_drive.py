"""
workzone/driver/test_drive.py

Step 1 harness. No CarDreamer, no env class, no RL. Plain CARLA.

Three Driver instances of the same class with different sampled
parameters: one in the closed lane that must merge, two in the open lane
that do not know it exists. If one class reproduces "merger", "lead" and
"lag" through parameters alone, the abstraction holds.

Run:
    python workzone/driver/test_drive.py --episodes 1 --verbose
    python workzone/driver/test_drive.py --episodes 20
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import carla
import numpy as np

from driver import Driver, MergeState, sample_driver_params

# ---- geometry, from workzone/configs/geometry.yaml ----
CLOSED_LANE_ID = -1
OPEN_LANE_ID   = -2      # verified: right of lane -1. lane 1 is oncoming.
LANE_X         = 5.24

TAPER_START_Y  =    0.0
TAPER_END_Y    = -165.0

POSTED_SPEED = 20.12     # m/s, 45 mph
DT           = 0.1
STEP_CAP     = 800
SETTLE_TICKS = 10        # let physics settle before control or collisions

TAPER_START_S = -TAPER_START_Y   #   0.0
TAPER_END_S   = -TAPER_END_Y     # 165.0

# Disjoint spawn bands. Travel is toward -y, so LOWER y is further ahead.
SPAWN_BANDS = {
    "open_lead": (150.0, 175.0),   # ahead of the merger, open lane
    "merger":    (195.0, 210.0),   # must leave the closed lane
    "open_lag":  (225.0, 240.0),   # behind the merger, open lane
}


def progress(loc: carla.Location) -> float:
    """Distance along the corridor, increasing forward. taper_start is 0."""
    return -loc.y


# ============================================================
# Scene
# ============================================================

def spawn_vehicle(world, amap, y: float, lane_id: int,
                  model: str = "vehicle.tesla.model3") -> Optional[carla.Actor]:
    """Explicit blueprint. Never leave this to the vehicle.audi* default."""
    ref = amap.get_waypoint(carla.Location(x=LANE_X, y=y, z=0.3),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    wp = ref
    if ref.lane_id != lane_id:
        for nb in (ref.get_left_lane(), ref.get_right_lane()):
            if nb is not None and nb.lane_id == lane_id:
                wp = nb
                break
        else:
            print(f"  ! no lane {lane_id} at y={y:.1f}")
            return None

    tf = carla.Transform(
        carla.Location(wp.transform.location.x,
                       wp.transform.location.y,
                       wp.transform.location.z + 0.3),
        wp.transform.rotation)
    bp = world.get_blueprint_library().filter(model)[0]
    return world.try_spawn_actor(bp, tf)


def build_neighbours(subject: Driver, drivers: List[Driver]) -> Dict:
    """
    What one driver can see: the vehicle ahead in its own lane, and the
    lead/lag pair in the target lane. Purely relative -- no vehicle is
    told which one is "the ego".
    """
    s_self = subject.progress()
    own_lane = subject.current_waypoint().lane_id
    tgt_lane = subject.target_lane_id
    half_self = subject.vehicle.bounding_box.extent.x

    own_lead = target_lead = target_lag = None

    for other in drivers:
        if other is subject:
            continue
        s_o = other.progress()
        lane_o = other.current_waypoint().lane_id
        d = s_o - s_self                       # > 0 means ahead
        clearance = half_self + other.vehicle.bounding_box.extent.x
        info = {"gap": max(abs(d) - clearance, 0.0), "speed": other.speed()}

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


def classify(d: Driver, collided: bool) -> str:
    """Provisional. Full taxonomy lands in workzone/taxonomy/ at Step 3."""
    if collided:
        return "COLLISION"
    if d.state == MergeState.MERGED:
        s = d.merge_complete_progress or 0.0
        final_quarter = TAPER_END_S - 0.25 * (TAPER_END_S - TAPER_START_S)
        return "FORCED_MERGE" if s >= final_quarter else "SUCCESS"
    if d.speed() < 0.5:
        return "STOP_IN_LANE"
    return "NO_MERGE"


# ============================================================
# Episode
# ============================================================

def run_episode(world, amap, rng, verbose=False) -> Dict:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

    closure = {
        "closed_lane_id": CLOSED_LANE_ID,
        "open_lane_id": OPEN_LANE_ID,
        "taper_start_s": TAPER_START_S,
        "taper_end_s": TAPER_END_S,
        "progress_fn": progress,
    }

    plan = [("open_lead", OPEN_LANE_ID),
            ("merger",    CLOSED_LANE_ID),
            ("open_lag",  OPEN_LANE_ID)]

    actors, drivers, names, params, col = [], [], [], [], None
    try:
        for name, lane in plan:
            lo, hi = SPAWN_BANDS[name]
            y = float(rng.uniform(lo, hi))
            v = spawn_vehicle(world, amap, y, lane)
            if v is None:
                raise RuntimeError(f"spawn failed: {name} at y={y:.1f}")
            actors.append(v)
            names.append(name)
            params.append(sample_driver_params(rng, POSTED_SPEED))

        # Settle physics BEFORE constructing drivers. CARLA has not applied
        # the spawn transform until the world ticks, so get_location() is
        # stale and every driver would be assigned to the wrong lane.
        for _ in range(SETTLE_TICKS):
            for v in actors:
                v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            world.tick()

        for v, p in zip(actors, params):
            drivers.append(Driver(v, p, amap, closure))

        # fail loudly rather than producing plausible nonsense
        for n, d, (_, want) in zip(names, drivers, plan):
            if d.origin_lane_id != want:
                raise RuntimeError(
                    f"{n}: origin lane {d.origin_lane_id}, expected {want}")

        merger = drivers[names.index("merger")]

        actor_ids = [a.id for a in actors]
        collided, hit_name = False, None

        def on_collision(event):
            nonlocal collided, hit_name
            other = event.other_actor
            if other is not None and other.id in actor_ids:
                collided = True
                hit_name = names[actor_ids.index(other.id)]

        col_bp = world.get_blueprint_library().find("sensor.other.collision")
        col = world.spawn_actor(col_bp, carla.Transform(),
                                attach_to=merger.vehicle)
        col.listen(on_collision)

        i = 0
        t_start = time.time()
        stopped = 0
        for i in range(STEP_CAP):
            for d in drivers:
                d.vehicle.apply_control(d.step(build_neighbours(d, drivers), DT))
            world.tick()

            if verbose and i % 10 == 0:
                print(f"  t={i*DT:5.1f}s  {merger.last_debug}", flush=True)
                for n, d in zip(names, drivers):
                    loc = d.vehicle.get_location()
                    print(f"      {n:10s} y={loc.y:8.2f} x={loc.x:6.2f} "
                          f"lane={d.current_waypoint().lane_id:3d} "
                          f"v={d.speed():5.2f} v0={d.p.desired_speed:5.2f} "
                          f"s={d.progress():7.1f} {d.state.value}", flush=True)

            if collided:
                break
            # outcome is decided once the merger passes the taper end
            if merger.progress() >= TAPER_END_S:
                break
            if merger.speed() < 0.5:
                stopped += 1
                if stopped > 20:        # 2 s stationary
                    break
            else:
                stopped = 0

        return {
            "outcome": classify(merger, collided),
            "hit": hit_name,
            "ticks": i,
            "n_aborts": merger.n_aborts,
            "critical_gap": round(merger.p.critical_gap, 2),
            "merge_init_time": round(merger.p.merge_init_time, 2),
            "desired_speed": round(merger.p.desired_speed, 2),
            "merge_s": (round(merger.merge_complete_progress, 1)
                        if merger.merge_complete_progress else None),
            "final_s": round(merger.progress(), 1),
            "final_speed": round(merger.speed(), 2),
            "final_state": merger.state.value,
        }
    finally:
        if col is not None:
            try:
                col.stop(); col.destroy()
            except Exception:
                pass
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        settings.synchronous_mode = False
        world.apply_settings(settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if "Town04" not in world.get_map().name:
        print("Loading Town04 ...")
        world = client.load_world("Town04")
    amap = world.get_map()

    results = []
    for ep in range(args.episodes):
        rng = np.random.default_rng(args.seed + ep)     # seeded, reproducible
        print(f"\n--- episode {ep} (seed {args.seed + ep}) ---")
        r = run_episode(world, amap, rng, args.verbose)
        results.append(r)
        print(f"  {r}")

    print("\n=== summary ===")
    for outcome in ["SUCCESS", "FORCED_MERGE", "NO_MERGE",
                    "STOP_IN_LANE", "COLLISION"]:
        n = sum(1 for r in results if r["outcome"] == outcome)
        if n:
            print(f"  {outcome:14s} {n}/{len(results)}")
    aborts = [r["n_aborts"] for r in results]
    speeds = [r["final_speed"] for r in results]
    print(f"  aborts: total={sum(aborts)} max={max(aborts) if aborts else 0}")
    print(f"  final speed: mean={np.mean(speeds):.2f} "
          f"min={min(speeds):.2f} max={max(speeds):.2f}")


if __name__ == "__main__":
    main()
