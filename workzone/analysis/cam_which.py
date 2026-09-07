import cv2, car_dreamer
env, _ = car_dreamer.create_task("carla_workzone", [])
obs = env.reset()
for i in range(60):
    obs, _, done, info = env.step(env.action_space.sample())
    if done: break
for k in ("closure_camera", "camera"):
    if k in obs:
        cv2.imwrite(f"which_{k}.png", obs[k][:, :, ::-1])
        print(k, obs[k].shape)
