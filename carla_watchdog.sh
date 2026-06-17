#!/usr/bin/env bash
# Per-GPU progress watchdog for persistent CARLA collection.
#
# Why this exists:
#   CARLA 0.9.10 servers can occasionally hang (for example during world
#   reloads on large maps) while the Leaderboard's own per-route timeout has
#   not yet fired. When that happens the worker stays blocked inside
#   `manage_continuous.py run` and no sensor data is written, so the GPU sits
#   idle until walltime. There was previously no automatic recovery for this.
#
# What it does:
#   Monitors whether new sensor data is still being written for the job that is
#   currently running on THIS GPU. If nothing new appears for WATCHDOG_STALL
#   seconds, it kills the stuck Leaderboard client and CARLA server for this
#   GPU's port and writes the restart flag that persistent_carla_worker.sh
#   already consumes, which brings up a fresh server and moves on.
#
# Why "data growth" instead of log timestamps:
#   Long routes on big maps can be quiet in the logs for minutes while still
#   progressing, so a log-mtime check produces false positives and kills
#   healthy jobs. New frames on disk are a reliable signal that collection is
#   actually advancing.
set -uo pipefail

GPU_ID=${GPU_ID:?GPU_ID must be set (0..GPUS_PER_NODE-1)}
NODE_NAME=${SLURMD_NODENAME:-$(hostname)}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
STATE_DIR=${STATE_DIR:-$PROJECT_ROOT/collection_state}
DATASET_DIR=${DATASET_DIR:-$PROJECT_ROOT/dataset}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs}
QUEUE_FILE="$STATE_DIR/job_queue.json"

PORT_SPACING=${PORT_SPACING:-100}
BASE_RPC_PORT=${BASE_RPC_PORT:-2000}
RPC_PORT=$(( BASE_RPC_PORT + PORT_SPACING * GPU_ID ))

# Tunables (env-overridable)
WATCHDOG_INTERVAL=${WATCHDOG_INTERVAL:-120}   # seconds between checks
WATCHDOG_STALL=${WATCHDOG_STALL:-900}         # no new data for this long => stuck
WATCHDOG_COOLDOWN=${WATCHDOG_COOLDOWN:-90}    # pause after a restart

mkdir -p "$STATE_DIR/restart" "$LOG_DIR"
log="$LOG_DIR/watchdog_${NODE_NAME}_gpu${GPU_ID}.log"

# Print the SAVE_PATH of the job currently running on THIS node+GPU, or nothing.
# Path layout mirrors manage_continuous.py:
#   <DATASET>/<agent>/weather_<w>/map_<NN>/<route_stem>
_current_save_path() {
  python3 - "$QUEUE_FILE" "$NODE_NAME" "$GPU_ID" "$DATASET_DIR" <<'PY'
import json, sys, os
qf, node, gpu, ds = sys.argv[1:]
try:
    q = json.load(open(qf))
except Exception:
    sys.exit(0)
for j in q.get('jobs', []):
    if j.get('status') != 'running':
        continue
    if str(j.get('node')) != node or str(j.get('gpu')) != str(gpu):
        continue
    agent = str(j.get('agent', ''))
    route = str(j.get('route', ''))
    stem = route[:-4] if route.endswith('.xml') else route
    try:    wlabel = "weather_%d" % int(j.get('weather', 0))
    except Exception: wlabel = "weather_%s" % j.get('weather', 0)
    try:    mlabel = "map_%02d" % int(j.get('town'))
    except Exception: mlabel = "map_%s" % (j.get('town') or 'unknown')
    print(os.path.join(ds, agent, wlabel, mlabel, stem))
    break
PY
}

# Count files under a directory (0 if missing).
_count() { find "$1" -type f 2>/dev/null | wc -l | tr -d ' '; }

echo "[watchdog] start node=$NODE_NAME gpu=$GPU_ID rpc=$RPC_PORT interval=${WATCHDOG_INTERVAL}s stall=${WATCHDOG_STALL}s" | tee -a "$log"

cur=""; last_count=0; last_change=$(date +%s)
while true; do
  sleep "$WATCHDOG_INTERVAL"
  sp="$(_current_save_path)"
  now=$(date +%s)

  # No job running on this GPU -> nothing to watch.
  if [[ -z "$sp" ]]; then cur=""; last_change=$now; continue; fi

  # New job (different save path) -> reset progress tracking.
  if [[ "$sp" != "$cur" ]]; then
    cur="$sp"; last_count=$(_count "$sp"); last_change=$now; continue
  fi

  cnt=$(_count "$sp")
  if (( cnt > last_count )); then
    last_count=$cnt; last_change=$now; continue
  fi

  # No new data since last_change.
  age=$(( now - last_change ))
  if (( age > WATCHDOG_STALL )); then
    echo "[watchdog] $(date -u +%FT%TZ) STALL: no new data for ${age}s on gpu${GPU_ID} (job dir: $sp) -> restarting CARLA" | tee -a "$log"
    # Kill the stuck Leaderboard client and CARLA server for THIS GPU's port so
    # the worker's `manage_continuous.py run` returns and re-ensures CARLA.
    pkill -9 -f "leaderboard_evaluator.*--port[ =]${RPC_PORT}($|[^0-9])" 2>/dev/null || true
    pkill -9 -f "carla-rpc-port=${RPC_PORT}($|[^0-9])" 2>/dev/null || true
    touch "$STATE_DIR/restart/${NODE_NAME}_gpu${GPU_ID}.restart"
    last_change=$(date +%s)
    sleep "$WATCHDOG_COOLDOWN"
  fi
done
