import carla
client = carla.Client("localhost", 2000); client.set_timeout(30.0)
amap = client.get_world().get_map()

for y in [230, 150, 75, 0, -80, -165, -215]:
    wp = amap.get_waypoint(carla.Location(x=5.24, y=float(y), z=0.1),
                           project_to_road=True)
    adj = wp.get_left_lane() or wp.get_right_lane()
    print(f"y={y:7.1f}  lane_id={wp.lane_id:3d}  x={wp.transform.location.x:6.2f}  "
          f"adj={adj.lane_id if adj else None}  junction={wp.is_junction}")
