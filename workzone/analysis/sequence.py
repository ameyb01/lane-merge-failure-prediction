#!/usr/bin/env python3
"""
workzone/analysis/sequence.py

Does watching a vehicle for a few seconds beat looking at it once?

The snapshot model sees one instant. A deployed sensor sees a stream, so
the obvious question is whether the history carries information the
current frame does not: a gap that has been closing for three seconds
looks the same in a snapshot as one that just opened.

WHAT THIS IS. The window of past observations is stacked into one feature
vector and fed to the same MLP used for the snapshot result. With a 1 s
sampling interval, a 5 s window is 5 x 19 = 95 features. This is not a
recurrent network. It has access to exactly the same history a recurrent
model would, so it answers whether the history helps; it does not answer
whether a recurrent architecture would exploit it more efficiently. That
is left to future work, and this file is the baseline it would have to beat.

A second variant, --summary, replaces the stack with hand-built temporal
features: the current value, the change over the window, and a linear
slope per feature. Fewer inputs, more interpretable, and it isolates
whether what matters is the trend rather than the raw history.

WINDOWS ARE BUILT FROM features.npz, no re-extraction. Rows for one
vehicle are already a time series sampled every SAMPLE_EVERY ticks, so
the window ending at row k is rows k-W+1 .. k for that same vehicle.

SHORT WINDOWS ARE LEFT-PADDED by repeating the earliest available
observation, because a vehicle that spawned close to the taper has little
history. `n_real` records how many observations were genuine, and is
reported so the padding is not hidden.

Evaluation matches evaluate.py exactly: the same position bin, the same
one-row-per-vehicle dedupe, the same grouped folds over episodes.

Usage:
    python workzone/analysis/sequence.py
    python workzone/analysis/sequence.py --summary
    python workzone/analysis/sequence.py --bin -10 0
"""

import argparse
import time
import warnings
from typing import Dict, Iterator, List, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HEADLINE = (0, 10)
WINDOWS = [1, 3, 5, 10]          # observations; 1 s apart, so also seconds

MLP_CFG = dict(hidden_layer_sizes=(32,), alpha=1e-4, activation="relu",
               solver="adam", learning_rate_init=1e-3, batch_size=256,
               max_iter=300, early_stopping=False, n_iter_no_change=15,
               tol=1e-5)


def grouped_folds(groups: np.ndarray, n_splits: int = 3,
                  seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    uniq, counts = uniq[order], counts[order]
    order = np.argsort(-counts, kind="stable")
    uniq, counts = uniq[order], counts[order]

    fold_of: Dict[int, int] = {}
    load = np.zeros(n_splits, dtype=np.int64)
    for g, c in zip(uniq, counts):
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += c

    assign = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        yield np.where(assign != f)[0], np.where(assign == f)[0]


def vehicle_key(vehicle, episode, density) -> np.ndarray:
    """A vehicle id repeats across episodes, so the key needs all three."""
    return (density.astype(np.int64) * 10_000_000
            + episode.astype(np.int64) * 1000 + vehicle % 1000)


def build_windows(X: np.ndarray, key: np.ndarray, t_res: np.ndarray,
                  W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every row, the stack of the W observations ending at it.

    Returns (N, W, F) and the count of genuine observations per row.
    Rows are ordered within each vehicle by decreasing time-to-resolve,
    which is chronological order.
    """
    N, F = X.shape
    out = np.zeros((N, W, F), dtype=np.float32)
    n_real = np.zeros(N, dtype=np.int16)

    order = np.lexsort((-t_res, key))
    k_sorted = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k_sorted)) + 1, len(order)]

    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]                 # this vehicle's rows, in time order
        seq = X[idx]
        for j in range(len(idx)):
            lo = max(0, j - W + 1)
            chunk = seq[lo:j + 1]
            pad = W - len(chunk)
            if pad:
                chunk = np.vstack([np.repeat(chunk[:1], pad, axis=0), chunk])
            out[idx[j]] = chunk
            n_real[idx[j]] = min(j + 1, W)
    return out, n_real


def summarise(win: np.ndarray) -> np.ndarray:
    """
    Current value, change across the window, and least-squares slope,
    per feature. 3F inputs regardless of window length.
    """
    W = win.shape[1]
    cur = win[:, -1, :]
    delta = win[:, -1, :] - win[:, 0, :]
    if W > 1:
        t = np.arange(W, dtype=np.float32)
        t = (t - t.mean())
        denom = float((t ** 2).sum())
        slope = np.einsum("nwf,w->nf", win, t) / denom
    else:
        slope = np.zeros_like(cur)
    return np.concatenate([cur, delta, slope], axis=1).astype(np.float32)


def cv_auc(X, y, g, clf_factory, seeds: int, splits: int):
    aucs = []
    for seed in range(seeds):
        for tr, te in grouped_folds(g, splits, seed):
            if len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = clf_factory(seed)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    if not aucs:
        return float("nan"), 0.0
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/features.npz")
    ap.add_argument("--bin", nargs=2, type=float, default=list(HEADLINE),
                    metavar=("LO", "HI"))
    ap.add_argument("--summary", action="store_true",
                    help="temporal summaries instead of the raw stack")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    s_obs, t_res = d["s_at_obs"], d["t_to_resolve"]
    episode, vehicle, density = d["episode"], d["vehicle"], d["density"]

    key = vehicle_key(vehicle, episode, density)

    lo, hi = args.bin
    in_bin = (s_obs >= lo) & (s_obs < hi)

    # one row per vehicle in the bin, the first it contributed
    idx_bin = np.where(in_bin)[0]
    _, first = np.unique(key[idx_bin], return_index=True)
    keep = np.sort(idx_bin[first])

    yb = y[keep]
    gb = density[keep].astype(np.int64) * 100000 + episode[keep].astype(np.int64)

    print(f"position bin s = {lo:+.0f} to {hi:+.0f}")
    print(f"  {len(keep)} vehicles, {100 * yb.mean():.1f}% merged, "
          f"{len(np.unique(gb))} episodes")
    print(f"  mode: {'temporal summaries' if args.summary else 'stacked window'}")
    print(f"  protocol: {args.splits}-fold grouped CV x {args.seeds} seed(s)\n")

    rows = []
    for W in WINDOWS:
        win, n_real = build_windows(X, key, t_res, W)
        wb, nb = win[keep], n_real[keep]

        feats = summarise(wb) if args.summary else wb.reshape(len(wb), -1)

        lin = cv_auc(feats, yb, gb,
                     lambda s: LogisticRegression(max_iter=3000),
                     args.seeds, args.splits)

        if args.skip_mlp:
            mlp = (float("nan"), 0.0)
            dt = 0.0
        else:
            t0 = time.time()
            mlp = cv_auc(feats, yb, gb,
                         lambda s: MLPClassifier(random_state=s, **MLP_CFG),
                         args.seeds, args.splits)
            dt = time.time() - t0

        full = float((nb >= W).mean())
        rows.append((W, feats.shape[1], full, lin, mlp))
        print(f"  W={W:2d}  {feats.shape[1]:4d} features  "
              f"full windows {100 * full:5.1f}%  "
              f"linear {lin[0]:.3f}  MLP {mlp[0]:.3f}  ({dt:.0f}s)",
              flush=True)

    print(f"\n{'window':>8} {'secs':>5} {'inputs':>7} {'full':>7} "
          f"{'linear':>14} {'MLP':>14}")
    for W, nf, full, lin, mlp in rows:
        print(f"{W:8d} {W:5d} {nf:7d} {100 * full:6.1f}% "
              f"{lin[0]:8.3f}±{lin[1]:.3f} {mlp[0]:8.3f}±{mlp[1]:.3f}")

    base = rows[0][4][0]
    best = max(r[4][0] for r in rows)
    print(f"\nsnapshot (W=1) MLP {base:.3f}   best windowed {best:.3f}   "
          f"gain {best - base:+.3f}")
    print("\n'full' is the share of vehicles that had the whole window "
          "available;\nthe rest were left-padded with their earliest "
          "observation, so a low\nvalue means the long-window numbers rest "
          "partly on repeated frames.")


if __name__ == "__main__":
    main()
