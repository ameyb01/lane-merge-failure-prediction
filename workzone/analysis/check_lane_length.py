import carla

client = carla.Client("localhost", 2000); client.set_timeout(20.0)
amap = client.get_world().get_map()

wp = amap.get_waypoint(carla.Location(x=5.24, y=130.0, z=0.1), project_to_road=True)
print("road_id", wp.road_id, "lane_id", wp.lane_id, "s", round(wp.s, 1))

def walk(wp, forward, step=5.0, limit=600):
    dist, cur, yaw0 = 0.0, wp, wp.transform.rotation.yaw
    max_dev = 0.0
    for _ in range(limit):
        nxt = cur.next(step) if forward else cur.previous(step)
        if not nxt or len(nxt) > 1:      # lane ends or junction branches
            break
        cur = nxt[0]
        if cur.is_junction:
            break
        dev = abs((cur.transform.rotation.yaw - yaw0 + 180) % 360 - 180)
        max_dev = max(max_dev, dev)
        dist += step
    return dist, max_dev, cur

f_d, f_dev, f_wp = walk(wp, True)
b_d, b_dev, b_wp = walk(wp, False)
print(f"ahead : {f_d:.0f} m, max heading dev {f_dev:.1f} deg")
print(f"behind: {b_d:.0f} m, max heading dev {b_dev:.1f} deg")
print(f"total : {f_d + b_d:.0f} m")

# does the adjacent lane exist along the whole stretch?
for name, w in [("ahead end", f_wp), ("behind end", b_wp)]:
    r = w.get_right_lane(); l = w.get_left_lane()
    print(name, "right:", r.lane_id if r else None, "left:", l.lane_id if l else None)
