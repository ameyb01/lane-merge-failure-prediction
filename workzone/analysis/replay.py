"""
workzone/analysis/replay.py

Turn a collected episode into a standalone HTML animation.

Reads the object-level tracks written by collect_v2.py and emits a single
self-contained .html file -- no CARLA, no display, no matplotlib. Works
over SSH while collection is running.

    # see what is in a dataset, and pick an episode by outcome
    python workzone/analysis/replay.py --data data/wz_d4 --list
    python workzone/analysis/replay.py --data data/wz_d4 --list stop_in_lane

    # render one
    python workzone/analysis/replay.py --data data/wz_d4 --episode 12

Then serve it and tunnel:
    python -m http.server 8800 --directory data/wz_d4/replays
    ssh -L 8800:localhost:8800 amey@eilab-server1
    open http://localhost:8800
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# column order written by collect_v2.py
F = {n: i for i, n in enumerate(
    ["id", "s", "lane", "origin_lane", "x", "y", "yaw",
     "vx", "vy", "speed", "half_len", "half_wid", "n_aborts"])}

STATE_NAMES = {0: "approaching", 1: "seeking", 2: "committed",
               3: "merged", 4: "not_merging", -1: "gone"}

# geometry, from workzone/configs/geometry.yaml
APPROACH_S, TAPER_START_S, TAPER_END_S, BUFFER_END_S = -230.0, 0.0, 165.0, 215.0


def load_episode(data_dir: str, ep: int):
    path = os.path.join(data_dir, f"ep_{ep:05d}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    d = np.load(path)
    tracks, states, valid = d["tracks"], d["states"], d["valid"]

    T, V = states.shape
    frames = []
    for t in range(T):
        row = []
        for v in range(V):
            if not valid[t, v]:
                continue
            tr = tracks[t, v]
            row.append({
                "id": int(tr[F["id"]]),
                "s": round(float(tr[F["s"]]), 2),
                "lane": int(tr[F["lane"]]),
                "orig": int(tr[F["origin_lane"]]),
                "v": round(float(tr[F["speed"]]), 2),
                "hl": round(float(tr[F["half_len"]]), 2),
                "hw": round(float(tr[F["half_wid"]]), 2),
                "st": int(states[t, v]),
                "ab": int(tr[F["n_aborts"]]),
            })
        frames.append(row)
    return frames


def episode_meta(data_dir: str, ep: int):
    mpath = os.path.join(data_dir, "manifest.jsonl")
    if not os.path.exists(mpath):
        return {}
    for line in open(mpath):
        m = json.loads(line)
        if m.get("episode") == ep:
            return m
    return {}


def list_episodes(data_dir: str, want: str | None):
    mpath = os.path.join(data_dir, "manifest.jsonl")
    if not os.path.exists(mpath):
        raise SystemExit(f"no manifest in {data_dir}")
    shown = 0
    for line in open(mpath):
        m = json.loads(line)
        outs = list(m.get("outcomes", {}).values())
        if want and want not in outs:
            continue
        print(f"  ep {m['episode']:5d}  ticks={m['ticks']:4d}  "
              f"veh={m['n_vehicles']}  closed={m['n_closed_lane']}  "
              f"{sorted(outs)}")
        shown += 1
        if shown >= 40:
            print("  ... (truncated)")
            break
    if shown == 0:
        print(f"  no episodes with outcome '{want}'")


HTML = """<!doctype html>
<meta charset="utf-8">
<title>episode __EP__ &mdash; __DIR__</title>
<style>
  body {{ background:#111; color:#ddd; font:13px/1.5 ui-monospace,Menlo,monospace;
         margin:0; padding:18px; }}
  canvas {{ background:#1a1a1a; border-radius:6px; width:100%; }}
  .bar {{ display:flex; gap:12px; align-items:center; margin:12px 0; }}
  button {{ background:#2a2a2a; color:#ddd; border:1px solid #444;
            border-radius:4px; padding:5px 14px; font:inherit; cursor:pointer; }}
  button:hover {{ background:#333; }}
  input[type=range] {{ flex:1; }}
  .key {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; color:#999; }}
  .key i {{ display:inline-block; width:11px; height:11px; border-radius:2px;
            margin-right:5px; vertical-align:-1px; }}
  table {{ border-collapse:collapse; margin-top:14px; }}
  td,th {{ padding:2px 12px 2px 0; text-align:left; font-weight:normal; }}
  th {{ color:#777; }}
</style>

<h3 style="margin:0 0 4px">episode __EP__ &mdash; __DIR__</h3>
<div style="color:#888">__META__</div>

<canvas id="c" width="1400" height="230"></canvas>

<div class="bar">
  <button id="play">play</button>
  <input type="range" id="scrub" min="0" max="__TMAX__" value="0">
  <span id="clock" style="min-width:130px"></span>
</div>

<div class="key">
  <span><i style="background:#666"></i>approaching</span>
  <span><i style="background:#e0b040"></i>seeking</span>
  <span><i style="background:#e07030"></i>committed</span>
  <span><i style="background:#40b060"></i>merged</span>
  <span><i style="background:#3a6ea5"></i>not merging</span>
</div>

<table id="tbl"></table>

<script>
const FRAMES = __FRAMES__;
const GEO = __GEO__;
const COLOR = {{0:"#666", 1:"#e0b040", 2:"#e07030", 3:"#40b060", 4:"#3a6ea5"}};
const NAME  = {{0:"approaching",1:"seeking",2:"committed",3:"merged",4:"not_merging"}};

const c = document.getElementById("c"), g = c.getContext("2d");
const scrub = document.getElementById("scrub");
const clock = document.getElementById("clock");
const tbl = document.getElementById("tbl");
const playBtn = document.getElementById("play");

const PAD = 40;
const S0 = GEO.approach - 20, S1 = GEO.buffer_end + 30;
const px = s => PAD + (s - S0) / (S1 - S0) * (c.width - 2*PAD);
// lane -1 (closed) on top, lane -2 (open) below
const py = lane => lane === -1 ? 78 : 138;

function draw(t) {{
  g.clearRect(0, 0, c.width, c.height);

  // lanes
  for (const lane of [-1, -2]) {{
    g.fillStyle = lane === -1 ? "#242424" : "#202020";
    g.fillRect(PAD, py(lane) - 22, c.width - 2*PAD, 44);
  }}

  // the closed lane ends at the taper
  g.fillStyle = "rgba(200,60,60,0.14)";
  g.fillRect(px(GEO.taper_start), py(-1) - 22,
             px(GEO.taper_end) - px(GEO.taper_start), 44);

  // markers
  g.font = "11px ui-monospace,monospace";
  for (const [s, label] of [[GEO.taper_start, "taper start"],
                            [GEO.taper_end, "taper end"],
                            [GEO.buffer_end, "work vehicle"]]) {{
    g.strokeStyle = "#555"; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(px(s), 34); g.lineTo(px(s), 190); g.stroke();
    g.setLineDash([]);
    g.fillStyle = "#888"; g.fillText(label, px(s) + 4, 28);
  }}

  // work vehicle
  g.fillStyle = "#8a3a3a";
  g.fillRect(px(GEO.buffer_end) - 12, py(-1) - 10, 24, 20);

  // lane labels
  g.fillStyle = "#666";
  g.fillText("lane -1  (closed)", 4, py(-1) + 4);
  g.fillText("lane -2  (open)", 4, py(-2) + 4);

  // vehicles
  const rows = FRAMES[t] || [];
  for (const v of rows) {{
    const w = Math.max(8, (v.hl * 2) / (S1 - S0) * (c.width - 2*PAD));
    const x = px(v.s), y = py(v.lane);
    g.fillStyle = COLOR[v.st] || "#555";
    g.fillRect(x - w/2, y - 9, w, 18);
    g.fillStyle = "#111";
    g.font = "10px ui-monospace,monospace";
    g.fillText(String(v.id).slice(-3), x - w/2 + 3, y + 4);
  }}

  clock.textContent = `t=${{(t * 0.1).toFixed(1)}}s   tick ${{t}}/${{FRAMES.length-1}}`;

  let html = "<tr><th>id</th><th>lane</th><th>s</th><th>speed</th>"
           + "<th>state</th><th>aborts</th></tr>";
  for (const v of rows) {{
    html += `<tr><td>${{v.id}}</td><td>${{v.lane}}</td>`
          + `<td>${{v.s.toFixed(1)}}</td><td>${{v.v.toFixed(2)}}</td>`
          + `<td style="color:${{COLOR[v.st]}}">${{NAME[v.st] || "-"}}</td>`
          + `<td>${{v.ab}}</td></tr>`;
  }}
  tbl.innerHTML = html;
}}

let t = 0, timer = null;
scrub.oninput = () => {{ t = +scrub.value; draw(t); }};
playBtn.onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; playBtn.textContent = "play"; return; }}
  playBtn.textContent = "pause";
  timer = setInterval(() => {{
    t = (t + 1) % FRAMES.length;
    scrub.value = t; draw(t);
  }}, 50);   // 2x real time
}};
draw(0);
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir, e.g. data/wz_d4")
    ap.add_argument("--episode", type=int)
    ap.add_argument("--list", nargs="?", const="", default=None,
                    metavar="OUTCOME",
                    help="list episodes, optionally filtered by outcome")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.list is not None:
        list_episodes(args.data, args.list or None)
        return

    if args.episode is None:
        raise SystemExit("give --episode N (or --list to browse)")

    frames = load_episode(args.data, args.episode)
    meta = episode_meta(args.data, args.episode)

    meta_txt = (f"{meta.get('ticks','?')} ticks &middot; "
                f"{meta.get('n_vehicles','?')} vehicles &middot; "
                f"{meta.get('n_closed_lane','?')} in closed lane &middot; "
                f"seed {meta.get('seed','?')} &middot; "
                + ", ".join(f"{k}: {v}"
                            for k, v in meta.get("outcomes", {}).items()))

    geo = {"approach": APPROACH_S, "taper_start": TAPER_START_S,
           "taper_end": TAPER_END_S, "buffer_end": BUFFER_END_S}

    html = (HTML
            .replace("__FRAMES__", json.dumps(frames))
            .replace("__GEO__", json.dumps(geo))
            .replace("__TMAX__", str(len(frames) - 1))
            .replace("__EP__", str(args.episode))
            .replace("__DIR__", args.data)
            .replace("__META__", meta_txt))

    out = args.out or os.path.join(args.data, "replays",
                                   f"ep_{args.episode:05d}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)

    print(f"wrote {out}  ({len(frames)} ticks, {os.path.getsize(out)//1024} KB)")
    print(f"\nserve it:\n  python -m http.server 8800 --directory "
          f"{os.path.dirname(out)}")
    print(f"tunnel from your laptop:\n  ssh -L 8800:localhost:8800 "
          f"$USER@eilab-server1\nthen open http://localhost:8800")


if __name__ == "__main__":
    main()
