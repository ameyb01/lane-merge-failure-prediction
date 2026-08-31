"""
workzone/analysis/dump_episode.py

Print everything that happened in one episode, tick by tick, and
reconstruct what each merging driver actually saw when it made its
decisions.

The gap columns are recomputed here the same way Driver.evaluate_gap does
it, so the printed lead_t / lag_t are the numbers the driver was comparing
against its critical gap -- not a proxy. The "closing" columns show what
those times would be if computed from relative speed instead, which is the
correct physical quantity.

    python workzone/analysis/dump_episode.py --data data/wz_d4 --episode 72
    python workzone/analysis/dump_episode.py --data data/wz_d4 --episode 72 --every 1
    python workzone/analysis/dump_episode.py --data data/wz_d4 --episode 72 --focus 653
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

F = {n: i for i, n in enumerate(
    ["id", "s", "lane", "origin_lane", "x", "y", "yaw",
     "vx", "vy", "speed", "half_len", "half_wid", "n_aborts"])}

ST = {0: "approach", 1: "seeking", 2: "committd", 3: "merged",
      4: "notmerge", -1: "gone"}

TAPER_START_S, TAPER_END_S, BUFFER_END_S = 0.0, 165.0, 215.0


def neighbours(tr, va, t, me, ids, target_lane):
    """
    Reproduce Driver._neighbours for one vehicle at one tick: nearest
    vehicle ahead in its own lane, and the lead/lag pair in the lane it
    is targeting. Distances are bumper-to-bumper.
    """
    s_me = tr[t, me, F["s"]]
    lane_me = int(tr[t, me, F["lane"]])
    half_me = tr[t, me, F["half_len"]]

    own_lead = tgt_lead = tgt_lag = None
    for j in range(tr.shape[1]):
        if j == me or not va[t, j]:
            continue
        s_o = tr[t, j, F["s"]]
        lane_o = int(tr[t, j, F["lane"]])
        d = s_o - s_me
        clear = half_me + tr[t, j, F["half_len"]]
        info = {"id": ids[j], "d": d,
                "gap": max(abs(d) - clear, 0.0),
                "speed": tr[t, j, F["speed"]]}

        if lane_o == lane_me and d > 0:
            if own_lead is None or d < own_lead["d"]:
                own_lead = info
        if lane_o == target_lane:
            if d > 0 and (tgt_lead is None or d < tgt_lead["d"]):
                tgt_lead = info
            if d < 0 and (tgt_lag is None or -d < -tgt_lag["d"]):
                tgt_lag = info
    return own_lead, tgt_lead, tgt_lag


def fmt(x, w=6, p=1):
    return f"{x:{w}.{p}f}" if x is not None and np.isfinite(x) else " " * (w - 3) + "inf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--every", type=int, default=5,
                    help="print every Nth tick (default 5 = 0.5 s)")
    ap.add_argument("--focus", type=int, default=None,
                    help="only show gap detail for this vehicle id")
    args = ap.parse_args()

    path = os.path.join(args.data, f"ep_{args.episode:05d}.npz")
    d = np.load(path)
    tr, st, va = d["tracks"], d["states"], d["valid"]
    T, V = st.shape

    ids = []
    for v in range(V):
        col = tr[va[:, v], v, F["id"]]
        ids.append(int(col[0]) if len(col) else -1)

    meta = {}
    mpath = os.path.join(args.data, "manifest.jsonl")
    if os.path.exists(mpath):
        for line in open(mpath):
            m = json.loads(line)
            if m.get("episode") == args.episode:
                meta = m
                break

    print(f"=== episode {args.episode}  seed={meta.get('seed','?')}  "
          f"{T} ticks  vehicles={ids}")
    print(f"    outcomes: {meta.get('outcomes', {})}")
    print(f"    geometry: taper_start=0  taper_end={TAPER_END_S}  "
          f"buffer_end={BUFFER_END_S}\n")

    # ---- per-vehicle summary -------------------------------------
    print("--- per-vehicle summary ---")
    for v in range(V):
        m = va[:, v]
        if not m.any():
            continue
        first, last = np.argmax(m), T - 1 - np.argmax(m[::-1])
        print(f"  id={ids[v]}  origin_lane={int(tr[first,v,F['origin_lane']]):3d}  "
              f"spawn_s={tr[first,v,F['s']]:7.1f}  final_s={tr[last,v,F['s']]:7.1f}  "
              f"max_v={tr[m,v,F['speed']].max():5.2f}  "
              f"final_v={tr[last,v,F['speed']]:5.2f}  "
              f"aborts={int(tr[last,v,F['n_aborts']])}  "
              f"final_state={ST.get(int(st[last,v]),'?')}")
    print()

    # ---- state transitions ---------------------------------------
    print("--- state transitions ---")
    prev = {}
    for t in range(T):
        for v in range(V):
            if not va[t, v]:
                continue
            s_now = int(st[t, v])
            if prev.get(v) != s_now:
                print(f"  t={t*0.1:6.1f}s  id={ids[v]}  "
                      f"{ST.get(prev.get(v),'-'):>8s} -> {ST.get(s_now,'?'):<8s}  "
                      f"s={tr[t,v,F['s']]:7.1f}  v={tr[t,v,F['speed']]:5.2f}  "
                      f"lane={int(tr[t,v,F['lane']]):3d}  "
                      f"aborts={int(tr[t,v,F['n_aborts']])}")
                prev[v] = s_now
    print()

    # ---- full timeline -------------------------------------------
    print("--- timeline (all vehicles) ---")
    print("    t     " + "".join(
        f"| {ids[v]}: lane    s      v   state    " for v in range(V)))
    for t in range(0, T, args.every):
        row = f"  {t*0.1:6.1f}  "
        for v in range(V):
            if not va[t, v]:
                row += "|   --                              "
                continue
            row += (f"| {int(tr[t,v,F['lane']]):4d} {tr[t,v,F['s']]:7.1f} "
                    f"{tr[t,v,F['speed']]:5.2f} {ST.get(int(st[t,v]),'?'):<8s} ")
        print(row)
    print()

    # ---- gap detail for merging vehicles -------------------------
    print("--- gap evaluation as the driver computed it ---")
    print("    lead_t / lag_t are gap / ABSOLUTE speed, which is what")
    print("    evaluate_gap uses. lead_c / lag_c are gap / CLOSING speed,")
    print("    the physically correct version. Compare them.\n")

    for v in range(V):
        if not va[:, v].any():
            continue
        first = np.argmax(va[:, v])
        origin = int(tr[first, v, F["origin_lane"]])
        if origin != -1:
            continue                      # only vehicles that must merge
        if args.focus is not None and ids[v] != args.focus:
            continue

        print(f"  vehicle {ids[v]}  (origin lane {origin})")
        print("      t     state     s      v   | tgt_lead        gap  lead_t lead_c"
              " | tgt_lag         gap   lag_t  lag_c | own_lead   gap")
        for t in range(0, T, args.every):
            if not va[t, v]:
                continue
            own, lead, lag = neighbours(tr, va, t, v, ids, target_lane=-2)
            v_me = max(tr[t, v, F["speed"]], 0.5)

            lead_t = lead["gap"] / v_me if lead else float("inf")
            lead_c = (lead["gap"] / max(v_me - lead["speed"], 0.1)
                      if lead else float("inf"))
            lag_t = lag["gap"] / max(lag["speed"], 0.5) if lag else float("inf")
            lag_c = (lag["gap"] / max(lag["speed"] - v_me, 0.1)
                     if lag else float("inf"))

            print(f"   {t*0.1:6.1f}  {ST.get(int(st[t,v]),'?'):<8s}"
                  f"{tr[t,v,F['s']]:7.1f} {tr[t,v,F['speed']]:5.2f} |"
                  f" {str(lead['id']) if lead else '   -':>6s}"
                  f" {fmt(lead['gap'] if lead else None, 10)}"
                  f" {fmt(lead_t)} {fmt(lead_c)} |"
                  f" {str(lag['id']) if lag else '   -':>6s}"
                  f" {fmt(lag['gap'] if lag else None, 10)}"
                  f" {fmt(lag_t)} {fmt(lag_c)} |"
                  f" {str(own['id']) if own else '   -':>6s}"
                  f" {fmt(own['gap'] if own else None, 8)}")
        print()


if __name__ == "__main__":
    main()
