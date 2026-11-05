#!/usr/bin/env python3
"""
Operating point per observation window at a matched false-alarm rate.

The threshold is set separately in each window so that the false-alarm
rate is the same everywhere, which makes recall comparable across
windows. Reuses the folds, dedupe and scoring in calibrate.py.

    python workzone/analysis/operating_table.py
    python workzone/analysis/operating_table.py --fa 0.05
"""
import argparse, sys
import numpy as np
from sklearn.metrics import brier_score_loss

sys.path.insert(0, "workzone/analysis")
import calibrate as C

BINS = [(-30, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 30), (30, 45)]

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="data/features/features.npz")
ap.add_argument("--fa", type=float, default=0.10, help="target false-alarm rate")
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = np.load(a.features, allow_pickle=True)
X, y = d["X"], d["y"].astype(int)
s_obs = d["s_at_obs"]
ep, veh, dens, tres = d["episode"], d["vehicle"], d["density"], d["t_to_resolve"]

print(f"target false-alarm rate {100*a.fa:.0f}%   "
      f"mlp, {a.splits}-fold grouped CV, seed {a.seed}, out-of-fold\n")
print(f"{'window':>12} {'n':>7} {'thresh':>7} {'recall':>8} {'prec':>7} "
      f"{'false al':>9} {'warn/100':>9} {'lead':>7} {'ECE':>7} {'skill':>7}")

for lo, hi in BINS:
    m = np.where((s_obs >= lo) & (s_obs < hi))[0]
    keep = m[C.dedupe_one_per_vehicle(veh[m], ep[m], dens[m])]
    Xb, yb = X[keep], y[keep]
    gb = dens[keep].astype(np.int64) * 100000 + ep[keep].astype(np.int64)

    p = C.out_of_fold_scores(Xb, yb, gb, "mlp", a.splits, a.seed)

    # false alarms are merging vehicles warned about, so the threshold is
    # the target quantile of the predictions among vehicles that merged
    t = float(np.quantile(p[yb == 1], a.fa))
    o = C.operating_points(yb, p, [t])[0]

    _, ece = C.reliability(yb, p)
    b = brier_score_loss(yb, p)
    bb = brier_score_loss(yb, np.full_like(p, yb.mean()))

    print(f"{lo:+4d}..{hi:+4d} {len(keep):7d} {t:7.3f} "
          f"{100*o['recall']:7.1f}% {100*o['precision']:6.1f}% "
          f"{100*o['fp_rate']:8.1f}% {o['warned_per_100']:9.1f} "
          f"{np.median(tres[keep]):6.1f}s {ece:7.3f} {1-b/bb:7.3f}")
