# Predicting merge failure at work-zone lane closures from a fixed roadside sensor

A fork of [CarDreamer](https://github.com/ucd-dare/CarDreamer). The work for
this report is in `workzone/` and in `car_dreamer/carla_workzone_env.py`.
Everything else is upstream, and upstream's README is kept here as
[README_CarDreamer.md](README_CarDreamer.md), which is where the installation
instructions are.

    git diff upstream/master --stat    # scope of the changes

## What it does

A two-lane corridor in CARLA with a lane closure. There is no ego vehicle:
every vehicle runs its own driver model with independently sampled
parameters, so merging behaviour comes out of the interaction between
drivers rather than from a script, and nothing yields to anyone. Models then
predict which closed-lane vehicles will fail to merge, using only quantities
a fixed roadside sensor could measure.

## The files that matter

| File | What it is |
|---|---|
| `car_dreamer/carla_workzone_env.py` | The environment. Corridor, closure, episode loop, outcome assignment. Inherits `CarlaBaseEnv` directly, since the task needs vehicles to leave the planned path. |
| `car_dreamer/configs/tasks.yaml` | Corridor geometry. Authoritative for taper and buffer positions. |
| `workzone/driver/driver.py` | Driver model: IDM longitudinal, pure pursuit lateral, gap acceptance, merge state machine. Aborts are mechanistic, never a sampled probability. |
| `workzone/taxonomy/events.py` | Outcome definitions. Thresholds are fractions of taper length, not metres. |
| `workzone/collect/collect_v2.py` | Episode collection. One `.npz` of object-level tracks per episode plus a `manifest.jsonl`. |
| `workzone/analysis/build_features.py` | Builds the supervised dataset. 19 features per observation, sampled every second, stopping at resolution. |
| `workzone/analysis/evaluate.py` | AUC by observation window for all seven models. |
| `workzone/analysis/calibrate.py` | Reliability, Brier, skill score and a threshold sweep for one window. |
| `workzone/analysis/operating_table.py` | Operating point per window at a matched false-alarm rate. |
| `workzone/analysis/sequence.py` | One frame versus stacked frames. |

Everything else under `workzone/analysis/` is a one-off diagnostic from
development.

## Reproducing the results

`data/features/features.npz` and the three episode manifests are committed,
so the analysis runs without CARLA and without re-collecting.

    python workzone/analysis/evaluate.py          # AUC by observation window
    python workzone/analysis/operating_table.py   # operating points per window
    python workzone/analysis/calibrate.py --bin 0 10
    python workzone/analysis/sequence.py          # frame stacking

The outcome distribution comes straight from the manifests, one JSON object
per episode with an `outcomes` map per vehicle.

To rebuild from scratch, with a CARLA server running:

    python workzone/collect/collect_v2.py --episodes 1700 --out data/wzf_light  --inflow-rate 400
    python workzone/collect/collect_v2.py --episodes 1700 --out data/wzf_medium --inflow-rate 800
    python workzone/collect/collect_v2.py --episodes 1700 --out data/wzf_heavy  --inflow-rate 1200
    python workzone/analysis/build_features.py \
        --data data/wzf_light data/wzf_medium data/wzf_heavy \
        --out  data/features

## The dataset

`wzf_light`, `wzf_medium` and `wzf_heavy` are the runs the report uses:
400, 800 and 1200 veh/h, 1,700 episodes each, 14,994 labelled closed-lane
vehicles, 191,141 feature rows. Raw episode files are not committed; the
manifests and the feature matrix are.

Tracks are stored at object level, as positions, velocities and extents,
rather than as rendered frames. That is where a real roadside sensor's error
lives, and it keeps the dataset in single-digit gigabytes instead of over a
terabyte.

## Caveats

- `CORRIDOR_M = 165.0` in `build_features.py` is the inflow-boundary-to-despawn
  length, not a taper length. It is correct.
- `test_drive.py`, `debug_spawn.py`, `replay.py` and `dump_episode.py` carry an
  old 165 m taper constant. They are diagnostics and do not describe the
  reported corridor. `car_dreamer/configs/tasks.yaml` is authoritative.
- `carla_workzone_merge_env.py`, `carla_workzone_merge_complex_env.py` and
  `carla_workzone_merge_interactive_env.py` are earlier versions, still
  registered by the package. The environment used for this report is
  `carla_workzone_env.py`.
