#!/usr/bin/env python3
"""
workzone/analysis/evaluate.py

Prediction accuracy as a function of where along the corridor the
observation was taken.

POSITION, NOT TIME-TO-RESOLUTION, IS THE HORIZON AXIS. How long a vehicle
takes to resolve is itself a strong signal of its outcome: one that merges
cleanly resolves in seconds and never produces a long-horizon row, while
one that stops in the queue produces rows at 15, 20, 25 s. Binning on
time-to-resolution therefore selects for failures at exactly the horizons
of interest -- the 12 s bin is 95% failures. Binning on position asks a
well-posed question instead: among vehicles still unresolved at position
s, how well can the outcome be predicted?

THE POPULATION STILL CHANGES WITH POSITION, in two ways that belong in the
reporting rather than being corrected away:
  - upstream bins contain only vehicles that spawned that far back
  - downstream bins contain only vehicles that have not yet resolved, and
    successes leave the sample by merging, so the base rate falls sharply
Both are real conditionals a deployed sensor faces. Base rate is printed
next to every AUC so each number can be read against its population.

HEADLINE BIN is s = 0, the start of the merge zone: a natural operational
reference point with a roughly balanced base rate and a large sample.

Models, all under identical grouped CV:
  - single trivial features, as a leak check
  - logistic regression on all 19
  - MLP on all 19, with an inner grouped split for model selection

Usage:
    python workzone/analysis/evaluate.py               # as reported
    python workzone/analysis/evaluate.py --skip-mlp    # quick, no MLP
"""

import argparse
import sys
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

# Bin edges along the corridor. The merge zone is 0 to 45.
BINS = [(-30, -20), (-20, -10), (-10, 0), (0, 10),
        (10, 20), (20, 30), (30, 45)]
HEADLINE = (0, 10)

# Kept narrow deliberately. At three seeds and five splits the inner
# selection picks 64 units in about half the fits and 32 in a third,
# and a fixed 32-unit layer matches the selection to within 0.005 in
# every window.
GRID: List[Dict] = [
    {"hidden_layer_sizes": (32,), "alpha": 1e-4},
    {"hidden_layer_sizes": (64,), "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-2},
]

COMMON = dict(activation="relu", solver="adam", learning_rate_init=1e-3,
              batch_size=256, max_iter=300, early_stopping=False,
              n_iter_no_change=15, tol=1e-5)


def grouped_folds(groups: np.ndarray, n_splits: int = 5,
                  seed: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Folds formed over EPISODES. Vehicles in one episode share a scene: if A
    blocks B their outcomes are not independent, and each vehicle also
    contributes many rows. A random split would put the same scene on both
    sides and inflate the score.
    """
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


def check_no_leak(groups, train, test) -> None:
    overlap = np.intersect1d(np.unique(groups[train]), np.unique(groups[test]))
    if overlap.size:
        raise AssertionError(f"group leak: {overlap.size} episodes on both sides")


def score(sc, clf, X, y) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, clf.predict_proba(sc.transform(X))[:, 1])


def cv_logistic(X, y, g, seeds: int, splits: int) -> Tuple[float, float]:
    aucs = []
    for seed in range(seeds):
        for tr, te in grouped_folds(g, splits, seed):
            check_no_leak(g, tr, te)
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            a = score(sc, clf, X[te], y[te])
            if not np.isnan(a):
                aucs.append(a)
    return (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (float("nan"), 0.0)


def cv_mlp(X, y, g, seeds: int, splits: int) -> Tuple[float, float, Dict]:
    aucs, picks = [], {}
    for seed in range(seeds):
        for tr, te in grouped_folds(g, splits, seed):
            check_no_leak(g, tr, te)
            # model selection on a further grouped split of the TRAIN side
            inner = list(grouped_folds(g[tr], 4, seed))
            itr, iva = inner[0]
            best, best_auc = GRID[0], -1.0
            for cfg in GRID:
                sc = StandardScaler().fit(X[tr][itr])
                m = MLPClassifier(random_state=seed, **cfg, **COMMON)
                m.fit(sc.transform(X[tr][itr]), y[tr][itr])
                a = score(sc, m, X[tr][iva], y[tr][iva])
                if not np.isnan(a) and a > best_auc:
                    best, best_auc = cfg, a
            sc = StandardScaler().fit(X[tr])
            m = MLPClassifier(random_state=seed, **best, **COMMON)
            m.fit(sc.transform(X[tr]), y[tr])
            a = score(sc, m, X[te], y[te])
            if not np.isnan(a):
                aucs.append(a)
            k = f"{best['hidden_layer_sizes']} a={best['alpha']:g}"
            picks[k] = picks.get(k, 0) + 1
    return (float(np.mean(aucs)), float(np.std(aucs)), picks) if aucs \
        else (float("nan"), 0.0, picks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/features.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--per-density", action="store_true")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    s_obs, episode, density = d["s_at_obs"], d["episode"], d["density"]
    names = [str(v) for v in d["feature_names"]]
    groups = density.astype(np.int64) * 100000 + episode.astype(np.int64)

    # One row per vehicle per bin. A stopped vehicle sits at the same
    # position for 20 s and emits 20 near-identical rows, all labelled
    # failure, which inflates the failure class downstream on top of the
    # genuine selection effect. Keeping the first row per vehicle in each
    # bin makes the bins comparable.
    veh_key = (density.astype(np.int64) * 10 ** 7
               + episode.astype(np.int64) * 1000 + d["vehicle"] % 1000)
    keep = np.zeros(len(y), dtype=bool)
    for lo, hi in BINS:
        idx = np.where((s_obs >= lo) & (s_obs < hi))[0]
        _, first = np.unique(veh_key[idx], return_index=True)
        keep[idx[first]] = True
    X, y, s_obs, episode, density, groups = (
        X[keep], y[keep], s_obs[keep], episode[keep],
        density[keep], groups[keep])

    print(f"rows {len(y)} (deduped to one per vehicle per bin)   "
          f"positives {100 * y.mean():.1f}%   "
          f"episodes {len(np.unique(groups))}")
    print(f"protocol: {args.splits}-fold grouped CV x {args.seeds} seed(s)\n")

    trivial = {
        "position only": ["s"],
        "speed only": ["speed"],
        "queue position": ["n_closed_ahead"],
        "open-lane density": ["open_density"],
        "lag gap only": ["tgt_lag_gap"],
    }

    hdr = f"{'':22s}" + "".join(f"{f'{lo:+d}..{hi:+d}':>14s}" for lo, hi in BINS)
    print(hdr)
    print(f"{'n':22s}" + "".join(
        f"{int(((s_obs >= lo) & (s_obs < hi)).sum()):>14d}" for lo, hi in BINS))
    print(f"{'base rate (merged)':22s}" + "".join(
        f"{100 * y[(s_obs >= lo) & (s_obs < hi)].mean():>13.1f}%"
        for lo, hi in BINS))
    print("-" * len(hdr))

    for label, feats in trivial.items():
        cols = [names.index(f) for f in feats]
        cells = []
        for lo, hi in BINS:
            m = (s_obs >= lo) & (s_obs < hi)
            mu, sd = cv_logistic(X[m][:, cols], y[m], groups[m],
                                 args.seeds, args.splits)
            cells.append(f"{mu:.3f}")
        print(f"{label:22s}" + "".join(f"{c:>14s}" for c in cells))

    print("-" * len(hdr))
    cells = []
    for lo, hi in BINS:
        m = (s_obs >= lo) & (s_obs < hi)
        mu, sd = cv_logistic(X[m], y[m], groups[m], args.seeds, args.splits)
        cells.append(f"{mu:.3f}±{sd:.3f}")
    print(f"{'all 19 (linear)':22s}" + "".join(f"{c:>14s}" for c in cells))

    if not args.skip_mlp:
        cells, all_picks = [], {}
        for lo, hi in BINS:
            m = (s_obs >= lo) & (s_obs < hi)
            t0 = time.time()
            mu, sd, picks = cv_mlp(X[m], y[m], groups[m],
                                   args.seeds, args.splits)
            cells.append(f"{mu:.3f}±{sd:.3f}")
            for k, v in picks.items():
                all_picks[k] = all_picks.get(k, 0) + v
            print(f"  [{lo:+d}..{hi:+d}] MLP {time.time() - t0:.0f}s",
                  flush=True)
        print(f"{'all 19 (MLP)':22s}" + "".join(f"{c:>14s}" for c in cells))
        print("\nselected configs: " + ", ".join(
            f"{k} x{v}" for k, v in sorted(all_picks.items(),
                                           key=lambda kv: -kv[1])))

    lo, hi = HEADLINE
    m = (s_obs >= lo) & (s_obs < hi)
    print(f"\nHEADLINE, s = {lo} to {hi} (start of the merge zone): "
          f"n={int(m.sum())}, base rate {100 * y[m].mean():.1f}% merged")

    if args.per_density:
        print("\nper density, all 19 (linear):")
        for rate in [400, 800, 1200]:
            cells = []
            for lo, hi in BINS:
                m = (s_obs >= lo) & (s_obs < hi) & (density == rate)
                mu, _ = cv_logistic(X[m], y[m], groups[m], 1, args.splits)
                cells.append(f"{mu:.3f}")
            print(f"  {rate:4d} veh/h" + "".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
