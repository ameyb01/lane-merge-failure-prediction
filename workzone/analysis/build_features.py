#!/usr/bin/env python3
"""
workzone/analysis/build_features.py

Build the Phase C supervised dataset from collected episodes.

One row per (closed-lane vehicle, observation tick). Features are limited
to what a roadside sensor could measure: positions, velocities and extents
of the vehicles in the corridor.

EXCLUDED, because they are driver-internal and would leak the label:
  - `states` (approaching / seeking / committed / merged)
  - `n_aborts`, a count of internal state transitions

SAMPLING. Every SAMPLE_EVERY ticks, from the vehicle's first tick until
the tick its outcome is assigned, and no later. Rows taken after
resolution are not predictions: the merge has already happened or the
stop has already been recorded, so the model would be reading the answer
off the scene. The previous version sampled three fixed positions, two of
which sat inside the merge zone, so a share of rows described completed
events rather than pending ones.

Each row carries `t_to_resolve`, the seconds between the observation and
the moment the outcome was assigned. That is the horizon axis: bin on it
to get accuracy as a function of available warning time, rather than
reporting three fixed points.

`s_at_obs` is kept as well, so the old fixed-position horizons can still
be recovered for a comparison table.

Usage:
    python workzone/analysis/build_features.py \
        --data data/wzf_light data/wzf_medium data/wzf_heavy \
        --out  data/features
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# track array layout (collect_v2.TRACK_FIELDS)
F_ID, F_S, F_LANE, F_ORIGIN, F_X, F_Y, F_YAW = 0, 1, 2, 3, 4, 5, 6
F_VX, F_VY, F_SPEED, F_HALFLEN, F_HALFWID, F_ABORTS = 7, 8, 9, 10, 11, 12

CLOSED_LANE = -1
OPEN_LANE = -2
TAPER_START_S = 0.0
TAPER_END_S = 45.0
POSTED_SPEED = 9.0
DT = 0.1

SAMPLE_EVERY = 10          # ticks; 1.0 s at 10 Hz
CORRIDOR_M = 165.0         # inflow boundary to despawn. NOT a taper length;

FEATURE_NAMES = [
    "s",                    # own position along the corridor
    "speed",
    "dist_to_end",          # metres of closed lane left
    "lat_offset",           # lateral offset from the vehicle's own start
    "own_lead_gap",
    "own_lead_speed",
    "own_lead_dv",          # closing speed on that vehicle
    "tgt_lead_gap",
    "tgt_lead_speed",
    "tgt_lead_dv",
    "tgt_lag_gap",
    "tgt_lag_speed",
    "tgt_lag_dv",
    "slot",                 # lead gap + lag gap + own length
    "n_open_ahead",
    "n_open_behind",
    "n_closed_ahead",
    "open_mean_speed",
    "open_density",         # open-lane vehicles per 100 m
]

BIG = 999.0                # stand-in for "no such neighbour"


def neighbours_at(frame: np.ndarray, mask: np.ndarray, i: int) -> Dict:
    """Lead/lag in the subject's own lane and in the open lane."""
    subj = frame[i]
    s_me, lane_me, half_me = subj[F_S], subj[F_LANE], subj[F_HALFLEN]

    own_lead = tgt_lead = tgt_lag = None
    d_own = d_lead = d_lag = 1e9

    for j in np.where(mask)[0]:
        if j == i:
            continue
        o = frame[j]
        d = o[F_S] - s_me
        if o[F_LANE] == lane_me and 0 < d < d_own:
            own_lead, d_own = o, d
        if o[F_LANE] == OPEN_LANE:
            if 0 < d < d_lead:
                tgt_lead, d_lead = o, d
            if d < 0 and -d < d_lag:
                tgt_lag, d_lag = o, -d
    return {"own_lead": own_lead, "tgt_lead": tgt_lead, "tgt_lag": tgt_lag,
            "d_own": d_own, "d_lead": d_lead, "d_lag": d_lag}


def extract_row(frame: np.ndarray, mask: np.ndarray, i: int,
                x_ref: float) -> np.ndarray:
    subj = frame[i]
    s_me = float(subj[F_S])
    v_me = float(subj[F_SPEED])
    half_me = float(subj[F_HALFLEN])

    nb = neighbours_at(frame, mask, i)

    def pack(o, d, ahead: bool):
        if o is None:
            return BIG, 0.0, 0.0
        gap = max(d - (half_me + float(o[F_HALFLEN])), 0.0)
        sp = float(o[F_SPEED])
        dv = (v_me - sp) if ahead else (sp - v_me)
        return gap, sp, dv

    own_gap, own_sp, own_dv = pack(nb["own_lead"], nb["d_own"], True)
    ld_gap, ld_sp, ld_dv = pack(nb["tgt_lead"], nb["d_lead"], True)
    lg_gap, lg_sp, lg_dv = pack(nb["tgt_lag"], nb["d_lag"], False)

    slot = (ld_gap + lg_gap + 2.0 * half_me
            if ld_gap < BIG and lg_gap < BIG else BIG)

    open_mask = mask & (frame[:, F_LANE] == OPEN_LANE)
    closed_mask = mask & (frame[:, F_LANE] == CLOSED_LANE)
    s_all = frame[:, F_S]

    n_open_ahead = int(np.sum(open_mask & (s_all > s_me) & (s_all <= TAPER_END_S)))
    n_open_behind = int(np.sum(open_mask & (s_all < s_me) & (s_all >= s_me - 60.0)))
    n_closed_ahead = int(np.sum(closed_mask & (s_all > s_me) & (s_all <= TAPER_END_S)))

    open_sp = frame[open_mask, F_SPEED]
    open_mean = float(open_sp.mean()) if open_sp.size else 0.0
    open_density = float(np.sum(open_mask)) / CORRIDOR_M * 100.0

    return np.array([
        s_me, v_me, TAPER_END_S - s_me, float(subj[F_X]) - x_ref,
        own_gap, own_sp, own_dv,
        ld_gap, ld_sp, ld_dv,
        lg_gap, lg_sp, lg_dv,
        slot,
        n_open_ahead, n_open_behind, n_closed_ahead,
        open_mean, open_density,
    ], dtype=np.float32)


def resolution_tick(tracks: np.ndarray, valid: np.ndarray, v: int,
                    live: np.ndarray, meta: Dict, vid: int) -> Optional[int]:
    """
    The tick at which this vehicle's outcome was assigned.

    The manifest stores WHERE it resolved, not when, so the tick is
    recovered by finding where the vehicle first reached that position.
    Falls back to the last live tick when the position is missing, which
    only happens for collisions resolved through the override path.
    """
    rs = meta.get("resolved_s", {}).get(str(vid))
    if rs is None:
        rs = meta.get("impact_s", {}).get(str(vid))
    if rs is None:
        return int(live[-1])

    s_series = tracks[:, v, F_S]
    for t in live:
        if s_series[t] >= float(rs) - 0.05:
            return int(t)
    return int(live[-1])


def process_episode(npz_path: str, meta: Dict, density: int) -> List[Tuple]:
    d = np.load(npz_path, allow_pickle=True)
    tracks, valid, ids = d["tracks"], d["valid"], d["vehicle_ids"]
    T, V, _ = tracks.shape

    outcomes = meta["outcomes"]
    ep = meta["episode"]
    rows: List[Tuple] = []

    for v in range(V):
        vid = int(ids[v])
        out = outcomes.get(str(vid))
        if out is None:
            continue                       # open-lane vehicle, no label

        live = np.where(valid[:, v])[0]
        if live.size == 0:
            continue
        t0 = int(live[0])
        if tracks[t0, v, F_ORIGIN] != CLOSED_LANE:
            continue

        x_ref = float(tracks[t0, v, F_X])
        t_res = resolution_tick(tracks, valid, v, live, meta, vid)

        for t in live:
            t = int(t)
            if t > t_res:
                break                      # after resolution: not a prediction
            if (t - t0) % SAMPLE_EVERY != 0:
                continue
            feats = extract_row(tracks[t], valid[t], v, x_ref)
            rows.append((feats, out, ep, vid, density,
                         (t_res - t) * DT,              # t_to_resolve
                         float(tracks[t, v, F_S])))     # s_at_obs

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="data/features")
    args = ap.parse_args()

    rate_of = {"light": 400, "medium": 800, "heavy": 1200}

    X, y_out, episode, vehicle, density, t_res, s_obs = [], [], [], [], [], [], []

    for root in args.data:
        tag = next((k for k in rate_of if k in root), None)
        if tag is None:
            raise SystemExit(f"cannot infer density from path: {root}")

        with open(os.path.join(root, "manifest.jsonl")) as f:
            metas = [json.loads(line) for line in f]

        for meta in metas:
            p = os.path.join(root, f"ep_{meta['episode']:05d}.npz")
            if not os.path.exists(p):
                continue
            for feats, out, ep, vid, dens, tr, so in process_episode(
                    p, meta, rate_of[tag]):
                X.append(feats)
                y_out.append(out)
                episode.append(ep)
                vehicle.append(vid)
                density.append(dens)
                t_res.append(tr)
                s_obs.append(so)

        print(f"[{tag:6s}] {len(X)} rows so far")

    X = np.stack(X)
    y_out = np.array(y_out)
    episode = np.array(episode, dtype=np.int32)
    vehicle = np.array(vehicle, dtype=np.int64)
    density = np.array(density, dtype=np.int32)
    t_res = np.array(t_res, dtype=np.float32)
    s_obs = np.array(s_obs, dtype=np.float32)

    # Binary target: did the vehicle get across under its own power?
    # forced_merge counts as a merge; it completed, just late.
    y = np.isin(y_out, ["success", "forced_merge"]).astype(np.int8)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "features.npz")
    np.savez_compressed(
        path, X=X, y=y, outcome=y_out, episode=episode, vehicle=vehicle,
        density=density, t_to_resolve=t_res, s_at_obs=s_obs,
        feature_names=np.array(FEATURE_NAMES))

    print(f"\nwrote {path}")
    print(f"  X {X.shape}   positives {int(y.sum())}/{len(y)} "
          f"({100 * y.mean():.1f}%)")

    # a vehicle is unique only within its density and episode
    key = (density.astype(np.int64) * 10_000_000
           + episode.astype(np.int64) * 1000 + vehicle % 1000)
    _, counts = np.unique(key, return_counts=True)
    print(f"  vehicles {len(counts)}   rows per vehicle: "
          f"median {np.median(counts):.0f}  max {counts.max()}")

    print("\n  horizon bins (seconds before the outcome was assigned):")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8),
                   (8, 12), (12, 999)]:
        m = (t_res >= lo) & (t_res < hi)
        if m.sum():
            print(f"    {lo:3.0f}-{hi:<3.0f}s  n={int(m.sum()):7d}  "
                  f"merge rate {100 * y[m].mean():5.1f}%")


if __name__ == "__main__":
    main()
