#!/usr/bin/env bash
# Collect at several traffic densities unattended. Edits n_vehicles in
# tasks.yaml between runs, then calls run_batches.sh, which handles CARLA
# crashes and resumes from each manifest.
#
# Usage: bash workzone/collect/run_sweep.sh 1700

set -u

PER_DENSITY="${1:-1700}"
DENSITIES="${DENSITIES:-4 6 10}"
REPO="$HOME/research2/CarDreamer"
YAML="car_dreamer/configs/tasks.yaml"

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p data
SWEEP_LOG="data/sweep.log"

set_n_vehicles() {
    python - "$1" <<'PYEOF'
import re, sys
n = int(sys.argv[1])
p = "car_dreamer/configs/tasks.yaml"
s = open(p).read()
new, count = re.subn(r"(carla_workzone:.*?)    n_vehicles: \d+",
                     lambda m: m.group(1) + f"    n_vehicles: {n}",
                     s, count=1, flags=re.S)
assert count == 1, "did not find n_vehicles in the carla_workzone block"
open(p, "w").write(new)
PYEOF
}

echo "=== sweep start $(date) : ${PER_DENSITY} eps at [$DENSITIES] ===" \
    | tee -a "$SWEEP_LOG"

for N in $DENSITIES; do
    if ! set_n_vehicles "$N"; then
        echo "[$(date +%H:%M:%S)] FAILED to set n_vehicles=$N, skipping" \
            | tee -a "$SWEEP_LOG"
        continue
    fi

    got=$(grep -A1 "posted_speed: 20.12" "$YAML" | grep n_vehicles | tr -dc '0-9')
    if [ "$got" != "$N" ]; then
        echo "[$(date +%H:%M:%S)] config says n_vehicles=$got, wanted $N -- skipping" \
            | tee -a "$SWEEP_LOG"
        continue
    fi

    OUT="data/wz_d$N"
    echo "[$(date +%H:%M:%S)] density $N -> $OUT" | tee -a "$SWEEP_LOG"
    bash workzone/collect/run_batches.sh "$PER_DENSITY" "$OUT"

    have=$( [ -f "$OUT/manifest.jsonl" ] && wc -l < "$OUT/manifest.jsonl" || echo 0 )
    echo "[$(date +%H:%M:%S)] density $N done: $have / $PER_DENSITY" \
        | tee -a "$SWEEP_LOG"
done

pkill -u "$USER" -f CarlaUE4 >/dev/null 2>&1

echo "=== sweep finished $(date) ===" | tee -a "$SWEEP_LOG"
for N in $DENSITIES; do
    OUT="data/wz_d$N"
    have=$( [ -f "$OUT/manifest.jsonl" ] && wc -l < "$OUT/manifest.jsonl" || echo 0 )
    echo "  n=$N : $have episodes" | tee -a "$SWEEP_LOG"
done
