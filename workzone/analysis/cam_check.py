import numpy as np, cv2, car_dreamer

env, cfg = car_dreamer.create_task("carla_workzone", [])
obs = env.reset()
print("obs keys:", list(obs.keys()))
if "closure_camera" in obs:
    print("closure_camera shape:", obs["closure_camera"].shape,
          "mean:", obs["closure_camera"].mean().round(1))

for i in range(120):
    obs, _, done, info = env.step(env.action_space.sample())
    if i in (0, 40, 80, 119):
        img = obs["closure_camera"]
        cv2.imwrite(f"cam_{i:03d}.png", img[:, :, ::-1])
        rows = sorted(info["tracks"], key=lambda r: -r["s"])
        print(f"tick {i}: " + " | ".join(
            f"{r['id']} lane{r['lane']} s={r['s']:.0f}" for r in rows[:4]))
    if done:
        break
print("wrote cam_000.png cam_040.png cam_080.png cam_119.png")
