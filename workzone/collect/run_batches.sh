#!/usr/bin/env bash
# CARLA reliably dies after a few hundred episodes. Restart it between
# batches and let collect_v2.py resume from manifest.jsonl.
#
# Usage: bash workzone/collect/run_batches.sh 5000 data/wz_5000

set -u

TARGET="${1:-200}"
OUT="${2:-data/wz_200}"
CARLA="${CARLA:-$HOME/software/CarlaUE4.sh}"
REPO="$HOME/research2/CarDreamer"

cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }
mkdir -p "$OUT"
LOG="$OUT/collect.log"

start_carla() {
    pkill -u "$USER" -f CarlaUE4 >/dev/null 2>&1
    sleep 5
    "$CARLA" -RenderOffScreen -carla-port=2000 -quality-level=Low \
        >>"$OUT/carla.log" 2>&1 &
    for _ in $(seq 1 60); do
        sleep 2
        nc -z localhost 2000 && { sleep 15; return 0; }
    done
    return 1
}

done_count() {
    [ -f "$OUT/manifest.jsonl" ] && wc -l < "$OUT/manifest.jsonl" || echo 0
}

echo "=== batch collection: target=$TARGET out=$OUT ===" | tee -a "$LOG"

stalls=0
while :; do
    have=$(done_count)
    echo "[$(date +%H:%M:%S)] have $have / $TARGET" | tee -a "$LOG"
    [ "$have" -ge "$TARGET" ] && break

    if ! start_carla; then
        echo "[$(date +%H:%M:%S)] CARLA failed to start, retrying" | tee -a "$LOG"
        sleep 20
        continue
    fi

    python workzone/collect/collect_v2.py \
        --episodes "$TARGET" --out "$OUT" >>"$LOG" 2>&1

    after=$(done_count)
    if [ "$after" -le "$have" ]; then
        stalls=$((stalls + 1))
        echo "[$(date +%H:%M:%S)] no progress (stall $stalls/3)" | tee -a "$LOG"
        [ "$stalls" -ge 3 ] && {
            echo "giving up: three batches with no progress" | tee -a "$LOG"
            break
        }
    else
        stalls=0
    fi
done

pkill -u "$USER" -f CarlaUE4 >/dev/null 2>&1
echo "=== done: $(done_count) episodes in $OUT ===" | tee -a "$LOG"
