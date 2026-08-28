import numpy as np

d = "data/workzone_complex_merged"
f = np.load(f"{d}/features.npy")
L = np.load(f"{d}/lengths.npy")
m = np.arange(f.shape[1])[None, :] < L[:, None]

ego_y, vy, spd = f[:, :, 1], f[:, :, 2], f[:, :, 3]

for name, arr in [("vy", vy), ("speed_norm", spd), ("ego_y", ego_y)]:
    print(f"{name:12s} p5/50/95: {np.percentile(arr[m], [5,50,95]).round(2)}")

dy = np.abs(np.diff(ego_y, axis=1))
v  = np.abs(vy[:, :-1])
m2 = m[:, :-1] & m[:, 1:] & (v > 1.0)
print("implied dt per step:", round(float(np.median(dy[m2] / v[m2])), 4))
print("episode length p50/p95:", np.percentile(L, [50, 95]))
