import carla
client = carla.Client("localhost", 2000); client.set_timeout(30.0)
amap = client.get_world().get_map()

rows = []
y = 235.0
while y >= -230.0:
    wp = amap.get_waypoint(carla.Location(x=5.24, y=y, z=0.1),
                           project_to_road=True, lane_type=carla.LaneType.Driving)
    same = []
    for nb in (wp.get_left_lane(), wp.get_right_lane()):
        # same travel direction = same sign of lane_id
        if nb and nb.lane_type == carla.LaneType.Driving and \
           (nb.lane_id * wp.lane_id) > 0:
            same.append(nb.lane_id)
    rows.append((y, wp.lane_id, wp.is_junction, same, wp.transform.location.x))
    y -= 5.0

# longest run: not a junction, and at least one same-direction neighbour
best = cur = None
for r in rows:
    ok = (not r[2]) and len(r[3]) > 0
    if ok:
        cur = (r[0], r[0]) if cur is None else (cur[0], r[0])
        if best is None or (cur[0]-cur[1]) > (best[0]-best[1]):
            best = cur
    else:
        cur = None

for r in rows:
    flag = "" if (not r[2] and r[3]) else "   <-- unusable"
    print(f"y={r[0]:7.1f} lane={r[1]:3d} junc={str(r[2]):5s} "
          f"same_dir={r[3]} x={r[4]:6.2f}{flag}")

print(f"\nlongest clean run: y {best[0]:.0f} -> {best[1]:.0f} "
      f"= {best[0]-best[1]:.0f} m")
