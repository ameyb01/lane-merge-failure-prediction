"""
workzone/driver/debug_spawn.py

No control, no driving. Spawn the three vehicles, print exactly where
they landed and what the map says about them, then tick a few times with
zero throttle to confirm nothing explodes on its own.

Run:  python workzone/driver/debug_spawn.py
"""

import carla
import numpy as np

CLOSED_LANE_ID = -1
OPEN_LANE_ID   = -2
LANE_X         = 5.24
TAPER_END_Y    = -165.0
TAPER_END_S    = -TAPER_END_Y


def progress(loc):
    return -loc.y


def spawn_vehicle(world, amap, y, lane_id, model="vehicle.tesla.model3"):
    ref = amap.get_waypoint(carla.Location(x=LANE_X, y=y, z=0.3),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
    print(f"    ref waypoint: lane={ref.lane_id} "
          f"loc=({ref.transform.location.x:.2f}, {ref.transform.location.y:.2f}) "
          f"yaw={ref.transform.rotation.yaw:.1f} junction={ref.is_junction}")

    wp = ref
    if ref.lane_id != lane_id:
        left, right = ref.get_left_lane(), ref.get_right_lane()
        print(f"    left={left.lane_id if left else None} "
              f"right={right.lane_id if right else None}")
        for nb in (left, right):
            if nb is not None and nb.lane_id == lane_id:
                wp = nb
                break
        else:
            print(f"    !! no lane {lane_id} here")
            return None

    tf = carla.Transform(
        carla.Location(wp.transform.location.x,
                       wp.transform.location.y,
                       wp.transform.location.z + 0.3),
        wp.transform.rotation)
    bp = world.get_blueprint_library().filter(model)[0]
    v = world.try_spawn_actor(bp, tf)
    if v is None:
        print("    !! try_spawn_actor returned None")
    return v


def main():
    client = carla.Client("localhost", 2000); client.set_timeout(60.0)
    world = client.get_world()
    if "Town04" not in world.get_map().name:
        world = client.load_world("Town04")
    amap = world.get_map()

    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.1
    world.apply_settings(s)

    rng = np.random.default_rng(0)
    spawns = [
        (float(rng.uniform(200, 230)), CLOSED_LANE_ID, "merger"),
        (float(rng.uniform(150, 180)), OPEN_LANE_ID,   "open_lead"),
        (float(rng.uniform(215, 245)), OPEN_LANE_ID,   "open_lag"),
    ]

    actors, names = [], []
    try:
        for y, lane, name in spawns:
            print(f"\n[{name}] requested y={y:.1f} lane={lane}")
            v = spawn_vehicle(world, amap, y, lane)
            if v is None:
                continue
            actors.append(v); names.append(name)
            ext = v.bounding_box.extent
            print(f"    spawned id={v.id} "
                  f"half_len={ext.x:.2f} half_wid={ext.y:.2f}")

        world.tick()

        print("\n--- after 1 tick, zero control ---")
        for n, v in zip(names, actors):
            loc = v.get_location()
            wp = amap.get_waypoint(loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
            print(f"  {n:10s} y={loc.y:8.2f} x={loc.x:6.2f} z={loc.z:5.2f} "
                  f"lane={wp.lane_id:3d} progress={progress(loc):8.2f} "
                  f"remaining_to_taper_end={TAPER_END_S - progress(loc):8.2f}")

        print("\n--- pairwise longitudinal separation (m) ---")
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                d = progress(actors[j].get_location()) - \
                    progress(actors[i].get_location())
                print(f"  {names[i]:10s} -> {names[j]:10s}  d={d:+8.2f}")

        print("\n--- 20 ticks, zero control ---")
        for k in range(20):
            for v in actors:
                v.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
            world.tick()
        for n, v in zip(names, actors):
            loc = v.get_location()
            vel = v.get_velocity()
            print(f"  {n:10s} y={loc.y:8.2f} z={loc.z:5.2f} "
                  f"speed={(vel.x**2+vel.y**2)**0.5:5.2f}")

    finally:
        for v in actors:
            try:
                v.destroy()
            except Exception:
                pass
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == "__main__":
    main()
