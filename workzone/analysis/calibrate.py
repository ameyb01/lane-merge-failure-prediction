#!/usr/bin/env python3
"""
workzone/analysis/calibrate.py

Turn the ranking result into an operational one.

AUC says the scores are in the right ORDER. It says nothing about whether a
score of 0.7 means a 70% chance of merging, and nothing about what you
would actually get if you deployed a threshold. This script answers both.

  1. CALIBRATION. Group the held-out predictions into equal-count bins and
     compare the mean predicted probability against the observed merge rate
     in each bin. A calibrated model sits on the diagonal. Reported as a
     reliability table plus Brier score and expected calibration error.

  2. OPERATING POINTS. Sweep the warning threshold and report, at each one,
     how many real failures are caught and how many false alarms it costs.
     This is the table an engineer reads.

Predictions are OUT OF FOLD: every row is scored by a model that never saw
its episode. Calibration measured on training predictions is meaningless.

Alarm cost is reported per 100 vehicles screened rather than per hour,
because vehicles-per-hour in the closed lane depends on assumptions that
belong in the report, not in this script.

Usage:
    python workzone/analysis/calibrate.py
    python workzone/analysis/calibrate.py --bin -10 0      # a different position bin
    python workzone/analysis/calibrate.py --model logistic
"""

import argparse
import warnings
from typing import Dict, Iterator, List, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HEADLINE = (0, 10)

MLP_CFG = dict(hidden_layer_sizes=(32,), alpha=1e-4, activation="relu",
               solver="adam", learning_rate_init=1e-3, batch_size=256,
               max_iter=300, early_stopping=False, n_iter_no_change=15,
               tol=1e-5)


def grouped_folds(groups: np.ndarray, n_splits: int = 5,
                  seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Same episode-level splitting as evaluate.py."""
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


def dedupe_one_per_vehicle(vehicle: np.ndarray, episode: np.ndarray,
                           density: np.ndarray) -> np.ndarray:
    """
    Keep the first row each vehicle contributes. A vehicle is unique only
    within its density and episode, so the key combines all three.

    Without this, a vehicle that crawls through the bin emits several
    near-identical rows and is weighted more than one that passes quickly.
    Slow vehicles are disproportionately the ones that fail, so the
    duplication inflates the apparent signal.
    """
    key = (density.astype(np.int64) * 10_000_000
           + episode.astype(np.int64) * 1000 + vehicle % 1000)
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


def out_of_fold_scores(X: np.ndarray, y: np.ndarray, g: np.ndarray,
                       model: str, splits: int, seed: int) -> np.ndarray:
    """One held-out probability per row."""
    p = np.full(len(y), np.nan)
    for tr, te in grouped_folds(g, splits, seed):
        sc = StandardScaler().fit(X[tr])
        if model == "mlp":
            clf = MLPClassifier(random_state=seed, **MLP_CFG)
        else:
            clf = LogisticRegression(max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    if np.isnan(p).any():
        raise RuntimeError("some rows never landed in a test fold")
    return p


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """
    Equal-count bins. Quantile edges rather than fixed width, because
    predictions cluster and fixed-width bins come out nearly empty at the
    extremes.
    """
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if not m.any():
            continue
        pred, obs, n = p[m].mean(), y[m].mean(), int(m.sum())
        rows.append((pred, obs, n))
        ece += (n / len(y)) * abs(pred - obs)
    return rows, ece


def operating_points(y: np.ndarray, p: np.ndarray,
                     thresholds: List[float]):
    """
    A warning fires when P(merge) < threshold, i.e. the model thinks this
    vehicle is in trouble.

    POSITIVE CLASS FOR THIS TABLE IS FAILURE, not merge, because the thing
    being detected is the failure. y == 0 is a failure.
    """
    fail = (y == 0)
    n_fail, n_ok = int(fail.sum()), int((~fail).sum())
    out = []
    for t in thresholds:
        warn = p < t
        tp = int((warn & fail).sum())
        fp = int((warn & ~fail).sum())
        recall = tp / n_fail if n_fail else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        fp_rate = fp / n_ok if n_ok else float("nan")
        per_100 = 100.0 * warn.mean()
        out.append(dict(t=t, recall=recall, precision=precision,
                        fp_rate=fp_rate, warned_per_100=per_100,
                        tp=tp, fp=fp))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/features.npz")
    ap.add_argument("--bin", nargs=2, type=float, default=list(HEADLINE),
                    metavar=("LO", "HI"))
    ap.add_argument("--model", choices=["mlp", "logistic"], default="mlp")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None,
                    help="write a reliability plot to this path")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    s_obs = d["s_at_obs"]
    episode, vehicle, density = d["episode"], d["vehicle"], d["density"]

    lo, hi = args.bin
    m = np.where((s_obs >= lo) & (s_obs < hi))[0]
    keep = m[dedupe_one_per_vehicle(vehicle[m], episode[m], density[m])]

    Xb, yb = X[keep], y[keep]
    gb = density[keep].astype(np.int64) * 100000 + episode[keep].astype(np.int64)

    print(f"position bin s = {lo:+.0f} to {hi:+.0f}")
    print(f"  {len(keep)} vehicles, {100 * yb.mean():.1f}% merged, "
          f"{len(np.unique(gb))} episodes")
    print(f"  model {args.model}, {args.splits}-fold grouped CV, "
          f"out-of-fold predictions\n")

    p = out_of_fold_scores(Xb, yb, gb, args.model, args.splits, args.seed)

    auc = roc_auc_score(yb, p)
    brier = brier_score_loss(yb, p)
    base = yb.mean()
    brier_base = brier_score_loss(yb, np.full_like(p, base))

    print(f"AUC            {auc:.3f}")
    print(f"Brier          {brier:.4f}   "
          f"(always predicting the base rate: {brier_base:.4f})")
    print(f"skill score    {1 - brier / brier_base:.3f}   "
          f"(1.0 perfect, 0.0 no better than the base rate)\n")

    rows, ece = reliability(yb, p)
    print("RELIABILITY  (equal-count bins; calibrated means predicted "
          "about equals observed)")
    print(f"  {'predicted':>10} {'observed':>10} {'n':>7} {'gap':>8}")
    for pred, obs, n in rows:
        print(f"  {pred:10.3f} {obs:10.3f} {n:7d} {obs - pred:+8.3f}")
    print(f"\n  expected calibration error {ece:.4f}")
    print("  (mean gap between predicted and observed, weighted by bin size)\n")

    ths = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    ops = operating_points(yb, p, ths)
    print("OPERATING POINTS  (warn when P(merge) < threshold; "
          "the event being detected is FAILURE)")
    print(f"  {'thresh':>7} {'recall':>8} {'precision':>10} "
          f"{'false alarm':>12} {'warned':>8}")
    print(f"  {'':>7} {'caught':>8} {'of warned':>10} "
          f"{'rate':>12} {'per 100':>8}")
    for o in ops:
        print(f"  {o['t']:7.2f} {100 * o['recall']:7.1f}% "
              f"{100 * o['precision']:9.1f}% {100 * o['fp_rate']:11.1f}% "
              f"{o['warned_per_100']:7.1f}")

    print("\n  recall      of the vehicles that failed, how many were warned about")
    print("  precision   of the vehicles warned about, how many really failed")
    print("  false alarm of the vehicles that merged fine, how many were warned about")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pred = [r[0] for r in rows]
            obs = [r[1] for r in rows]
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1,
                    label="perfect calibration")
            ax.plot(pred, obs, "o-", lw=1.5, label=args.model)
            ax.set_xlabel("predicted P(merge)")
            ax.set_ylabel("observed merge rate")
            ax.set_title(f"Reliability, s = {lo:+.0f} to {hi:+.0f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=160)
            print(f"\nwrote {args.plot}")
        except ImportError:
            print("\nmatplotlib not available, skipped the plot")


if __name__ == "__main__":
    main()
