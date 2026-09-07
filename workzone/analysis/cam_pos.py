import carla, car_dreamer

env, cfg = car_dreamer.create_task("carla_workzone", [])
env.reset()
env.step(env.action_space.sample())

wv = env.unwrapped.work_vehicle
t = wv.get_transform()
bb = wv.bounding_box
print(f"truck  x={t.location.x:6.2f} y={t.location.y:7.2f} z={t.location.z:5.2f} "
      f"yaw={t.rotation.yaw:7.2f}  half-extent z={bb.extent.z:.2f}")

for a in env.unwrapped._world.carla_world.get_actors():
    if a.type_id.startswith("sensor.camera.rgb"):
        ct = a.get_transform()
        print(f"camera x={ct.location.x:6.2f} y={ct.location.y:7.2f} z={ct.location.z:5.2f} "
              f"pitch={ct.rotation.pitch:6.1f} yaw={ct.rotation.yaw:7.2f}")
