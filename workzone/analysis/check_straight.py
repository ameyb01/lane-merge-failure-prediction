import carla

client = carla.Client("localhost", 2000); client.set_timeout(20.0)
world = client.get_world()
amap  = world.get_map()

# start at the ego spawn, lane 1
wp = amap.get_waypoint(carla.Location(x=5.24, y=130.0, z=0.1),
                       project_to_road=True)
print("start:", wp.transform.location, "yaw", round(wp.transform.rotation.yaw, 2))

def walk(wp, forward, step=5.0, max_dev=3.0, limit=400):
    yaw0, dist, cur = wp.transform.rotation.yaw, 0.0, wp
    for _ in range(limit):
        nxt = cur.next(step) if forward else cur.previous(step)
        if not nxt: break
        cur = nxt[0]
        dev = abs((cur.transform.rotation.yaw - yaw0 + 180) % 360 - 180)
        if dev > max_dev: break
        dist += step
    return dist, cur.transform.location

f_d, f_loc = walk(wp, True)
b_d, b_loc = walk(wp, False)
print(f"straight ahead : {f_d:.0f} m  -> {f_loc}")
print(f"straight behind: {b_d:.0f} m  -> {b_loc}")
print(f"total usable   : {f_d + b_d:.0f} m")
