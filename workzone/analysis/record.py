"""
workzone/analysis/record.py

Record episodes as a side-by-side video: the fixed roadside camera at the
work zone on the left, a merging driver's own view on the right.

Frames come straight out of the observation dict -- CarlaBaseEnv.step()
returns every enabled camera as a numpy array each tick. Requires `camera`
in observation.enabled, and its own CARLA instance.

    python workzone/analysis/record.py --episodes 4 --out videos
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

import car_dreamer

PANELS = ["closure_camera", "camera"]
LABELS = {"closure_camera": "roadside sensor at closure",
          "camera": "merging driver view"}

STATE_COLOR = {"approaching": (150, 150, 150), "seeking": (64, 176, 224),
               "committed": (48, 112, 224), "merged": (96, 176, 64),
               "not_merging": (165, 110, 58)}


def panel(img, label, size):
    if img is None or getattr(img, "size", 0) == 0:
        img = np.zeros((size, size, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = cv2.resize(img[:, :, ::-1], (size, size),
                     interpolation=cv2.INTER_AREA)
    bar = np.full((28, size, 3), 24, dtype=np.uint8)
    cv2.putText(bar, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def overlay(frame, info, t):
    h, w = frame.shape[:2]
    strip = np.full((88, w, 3), 18, dtype=np.uint8)
    cv2.putText(strip, f"t = {t:5.1f} s", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    tracks = sorted(info.get("tracks", []), key=lambda r: -r["s"])
    for i, tr in enumerate(tracks[:6]):
        col = STATE_COLOR.get(tr["state"], (150, 150, 150))
        x = 8 + (i % 3) * (w // 3)
        y = 42 + (i // 3) * 20
        txt = (f"{tr['id']}  lane {tr['lane']:>2}  "
               f"s={tr['s']:7.1f}  v={tr['speed']:5.2f}  {tr['state']}")
        cv2.putText(strip, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    col, 1, cv2.LINE_AA)
    return np.vstack([frame, strip])


def write_video(frame_dir, out_path, fps):
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-framerate", str(fps),
               "-i", os.path.join(frame_dir, "f_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
               out_path]
        if subprocess.run(cmd).returncode == 0:
            return True
        print("  ffmpeg failed, falling back to cv2")
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not frames:
        return False
    first = cv2.imread(os.path.join(frame_dir, frames[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(os.path.join(frame_dir, f)))
    vw.release()
    return True


def record_episode(env, out_path, size, fps, dt=0.1):
    obs = env.reset()
    missing = [k for k in PANELS if k not in obs]
    if missing:
        raise SystemExit(
            f"observation keys {missing} not enabled.\n"
            f"Add to tasks.yaml:\n"
            f"    observation.enabled: [collision, closure_camera, camera]")

    tmp = tempfile.mkdtemp(prefix="wzrec_")
    n, done, info = 0, False, {}
    try:
        while not done:
            obs, _, done, info = env.step(env.action_space.sample())
            row = np.hstack([panel(obs.get(k), LABELS[k], size) for k in PANELS])
            cv2.imwrite(os.path.join(tmp, f"f_{n:05d}.png"),
                        overlay(row, info, n * dt))
            n += 1
        if not write_video(tmp, out_path, fps):
            print("  no frames written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ticks": n, "outcomes": info.get("outcomes", {}),
            "seed": info.get("episode_seed", -1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="videos")
    ap.add_argument("--task", default="carla_workzone")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env, _ = car_dreamer.create_task(args.task, [])

    for ep in range(args.episodes):
        stem = os.path.join(args.out, f"ep_{ep:02d}")
        meta = record_episode(env, stem + ".mp4", args.size, args.fps)
        outs = sorted(set(meta["outcomes"].values()))
        tagged = f"{stem}_{'_'.join(outs)}.mp4" if outs else stem + ".mp4"
        if outs and os.path.exists(stem + ".mp4"):
            os.replace(stem + ".mp4", tagged)
        kb = os.path.getsize(tagged) // 1024 if os.path.exists(tagged) else 0
        print(f"[ep {ep}] seed={meta['seed']} ticks={meta['ticks']} "
              f"{meta['outcomes']}  -> {tagged} ({kb} KB)")

    print(f"\nwrote to {args.out}/")


if __name__ == "__main__":
    main()
