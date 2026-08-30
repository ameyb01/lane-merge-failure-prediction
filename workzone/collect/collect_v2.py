"""
workzone/collect/collect_v2.py

Run N episodes of the workzone environment and write them to disk.

Storage is OBJECT-LEVEL tracks, not rendered BEV frames. Rationale
(PROJECT_MEMO.md section 8):

  - 128x128x3 uint8 at ~400 ticks x 5000 episodes is over 1 TB. Object
    state -- position, velocity, heading, extent per actor per tick -- is
    a few hundred bytes per tick, so the same dataset is single-digit GB.
  - Sensing degradation is injected at track level, BEFORE rasterisation,
    which is where a real roadside sensor's error actually lives. Baking
    BEV frames would make the Phase D study impossible without
    re-rendering everything.

One .npz per episode plus a manifest.jsonl, so collection is resumable
and a corrupt episode costs one file rather than the run.

Usage:
    python workzone/collect/collect_v2.py --episodes 100 --out data/wz_v2
    python workzone/collect/collect_v2.py --episodes 500 --out data/wz_v2 \
        --n-vehicles 10 --closed-fraction 0.4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

import car_dreamer


# per-track fields, in stored column order
TRACK_FIELDS = ["id", "s", "lane", "origin_lane", "x", "y", "yaw",
                "vx", "vy", "speed", "half_len", "half_wid", "n_aborts"]

# state strings are stored separately as an integer code
STATE_CODES = {"approaching": 0, "seeking": 1, "committed": 2,
               "merged": 3, "not_merging": 4}


def episode_to_arrays(frames: List[List[Dict]]) -> Dict[str, np.ndarray]:
    """
    Convert a list of per-tick track lists into padded arrays.

    Vehicles can leave the corridor mid-episode, so the per-tick vehicle
    count is not constant. Arrays are (T, V, F) with a validity mask,
    where V is the number of distinct vehicle ids seen in the episode.
    """
    ids = sorted({tr["id"] for frame in frames for tr in frame})
    idx = {vid: i for i, vid in enumerate(ids)}

    T, V, F = len(frames), len(ids), len(TRACK_FIELDS)
    tracks = np.zeros((T, V, F), dtype=np.float32)
    states = np.full((T, V), -1, dtype=np.int8)
    valid = np.zeros((T, V), dtype=bool)

    for t, frame in enumerate(frames):
        for tr in frame:
            v = idx[tr["id"]]
            tracks[t, v] = [tr[k] for k in TRACK_FIELDS]
            states[t, v] = STATE_CODES.get(tr["state"], -1)
            valid[t, v] = True

    return {"tracks": tracks, "states": states, "valid": valid,
            "vehicle_ids": np.array(ids, dtype=np.int64)}


def run_episode(env, ep_index: int) -> Dict:
    """One episode. Returns arrays plus metadata, or raises."""
    env.unwrapped._episode_index = ep_index - 1
    env.reset()
    frames: List[List[Dict]] = []
    t0 = time.time()

    done = False
    info: Dict = {}
    while not done:
        _, _, done, info = env.step(env.action_space.sample())
        frames.append(info["tracks"])

    data = episode_to_arrays(frames)
    meta = {
        "episode": ep_index,
        "seed": int(info.get("episode_seed", -1)),
        "ticks": len(frames),
        "n_vehicles": int(info.get("n_vehicles", 0)),
        "n_closed_lane": int(info.get("n_closed_lane", 0)),
        "outcomes": {str(k): v for k, v in info.get("outcomes", {}).items()},
        "wall_seconds": round(time.time() - t0, 1),
    }
    return {"data": data, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", type=str, default="data/wz_v2")
    ap.add_argument("--task", type=str, default="carla_workzone")
    ap.add_argument("--n-vehicles", type=int, default=None)
    ap.add_argument("--closed-fraction", type=float, default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")

    # resume: skip episodes already on disk
    done_ids = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["episode"])
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} episodes already collected")

    overrides = []
    if args.n_vehicles is not None:
        overrides.append(f"--env.n_vehicles={args.n_vehicles}")
    if args.closed_fraction is not None:
        overrides.append(f"--env.closed_lane_fraction={args.closed_fraction}")

    env, cfg = car_dreamer.create_task(args.task, overrides)

    counts: Dict[str, int] = {}
    failures = 0

    for ep in range(args.episodes):
        if ep in done_ids:
            continue
        try:
            result = run_episode(env, ep + args.seed_offset)
        except Exception as exc:                     # keep going; log it
            failures += 1
            print(f"[ep {ep}] FAILED: {type(exc).__name__}: {exc}")
            continue

        np.savez_compressed(os.path.join(args.out, f"ep_{ep:05d}.npz"),
                            **result["data"])
        with open(manifest_path, "a") as f:
            f.write(json.dumps(result["meta"]) + "\n")

        for outcome in result["meta"]["outcomes"].values():
            counts[outcome] = counts.get(outcome, 0) + 1

        m = result["meta"]
        print(f"[ep {ep:4d}] seed={m['seed']} ticks={m['ticks']:4d} "
              f"closed={m['n_closed_lane']} "
              f"{m['outcomes']} ({m['wall_seconds']}s)")

    print("\n=== collection summary ===")
    total = sum(counts.values())
    for outcome, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {outcome:14s} {n:5d}  ({n/max(total,1)*100:5.1f}%)")
    print(f"  total merge events: {total}")
    print(f"  failed episodes:    {failures}")
    print(f"  written to:         {args.out}")


if __name__ == "__main__":
    main()
