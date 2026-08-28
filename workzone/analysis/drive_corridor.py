import carla, math

client = carla.Client("localhost", 2000); client.set_timeout(30.0)
world = client.get_world(); amap = world.get_map()
s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.1
world.apply_settings(s)

LANE, TARGET_V, LOOKAHEAD = -1, 20.0, 10.0

def next_wp(wp, d):
    opts = wp.next(d)
    same = [w for w in opts if w.lane_id == LANE]
    return (same or opts)[0] if opts else None

bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
tf = amap.get_waypoint(carla.Location(x=5.55, y=230.0, z=0.3),
                       project_to_road=True).transform
tf.location.z += 0.5
car = world.spawn_actor(bp, tf)

try:
    bad, i = [], 0
    for i in range(600):
        t   = car.get_transform()
        wp  = amap.get_waypoint(t.location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
        tgt = next_wp(wp, LOOKAHEAD)
        if tgt is None:
            bad.append((round(t.location.y,1), "no-next")); break

        yaw = math.radians(t.rotation.yaw)
        dx  = tgt.transform.location.x - t.location.x
        dy  = tgt.transform.location.y - t.location.y
        fy  = -dx*math.sin(yaw) + dy*math.cos(yaw)
        fx  =  dx*math.cos(yaw) + dy*math.sin(yaw)
        steer = max(-1.0, min(1.0, math.atan2(2*2.9*fy, max(fx,1e-3)**2)))

        v = car.get_velocity().length()
        car.apply_control(carla.VehicleControl(
            throttle=0.7 if v < TARGET_V else 0.0,
            brake=0.3 if v > TARGET_V+3 else 0.0,
            steer=steer))
        world.tick()

        loc = car.get_location()
        w2  = amap.get_waypoint(loc, project_to_road=False)
        if w2 is None or w2.lane_id != LANE:
            bad.append((round(loc.y,1), None if w2 is None else w2.lane_id))
        if loc.y < -215: break

    print("final y:", round(car.get_location().y,1), "ticks:", i,
          "speed:", round(car.get_velocity().length(),1))
    print("off-lane:", len(bad), bad[:5])
finally:
    car.destroy()
    s.synchronous_mode = False; world.apply_settings(s)
