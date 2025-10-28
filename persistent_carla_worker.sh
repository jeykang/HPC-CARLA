#!/usr/bin/env bash
# One persistent worker per GPU: ensures a CARLA server, then loops running jobs.
# Writes Leaderboard results to the SAME route directory used by consolidated_agent.py.
set -euo pipefail

# ----- Required/typical env -----
GPU_ID=${GPU_ID:?GPU_ID must be set (0..GPUS_PER_NODE-1)}
NODE_NAME=${SLURMD_NODENAME:-$(hostname)}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
STATE_DIR=${STATE_DIR:-$PROJECT_ROOT/collection_state}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs}
DATASET_DIR=${DATASET_DIR:-$PROJECT_ROOT/dataset}   # <- agent also defaults to this

PORT_SPACING=${PORT_SPACING:-100}
BASE_RPC_PORT=${BASE_RPC_PORT:-2000}
TM_OFFSET=${TM_OFFSET:-5000}
RPC_PORT=$((BASE_RPC_PORT + GPU_ID * PORT_SPACING))
TM_PORT=$((RPC_PORT + TM_OFFSET))

mkdir -p "$LOG_DIR" "$STATE_DIR/health" "$STATE_DIR/restart"

# ----- Ensure CARLA server for this GPU -----
python3 "$PROJECT_ROOT/carla_server_manager.py" ensure \
  --gpu "$GPU_ID" \
  --base-rpc-port "$BASE_RPC_PORT" \
  --port-spacing "$PORT_SPACING" \
  --tm-offset "$TM_OFFSET" || true

HB="$STATE_DIR/health/${NODE_NAME}_gpu${GPU_ID}.json"
log="$LOG_DIR/persistent_worker_gpu${GPU_ID}.log"
touch "$log"

# Initial heartbeat
python3 - "$HB" "$NODE_NAME" "$GPU_ID" "$RPC_PORT" "$TM_PORT" <<'PY'
import json, sys, os, datetime
p, node, gpu, rpc, tm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.makedirs(os.path.dirname(p), exist_ok=True)
d={"node":node,"gpu_id":gpu,"status":"idle","rpc_port":rpc,"tm_port":tm,
   "message":"worker started","last_heartbeat":datetime.datetime.utcnow().isoformat()+"Z"}
open(p,"w").write(json.dumps(d, indent=2))
PY

echo "[worker] node=$NODE_NAME gpu=$GPU_ID rpc=$RPC_PORT tm=$TM_PORT" | tee -a "$log"

# ----- Common CARLA/Leaderboard env -----
export LOCAL_GPUS=${LOCAL_GPUS:-${GPUS_PER_NODE:-8}}
export GPU_ID
export CARLA_HOST=127.0.0.1
export CARLA_PORT=$RPC_PORT
export TM_PORT=$TM_PORT

# === NEW: derive SAVE_PATH exactly like consolidated_agent.py and point CHECKPOINT_ENDPOINT there ===
# Inputs this uses (if provided by your scheduler / job runner):
#   AGENT_NAME, WEATHER_INDEX, TOWN_NUM, ROUTE_NAME, ROUTES_FILE, DATASET_DIR
# Fallbacks mirror consolidated_agent.py behavior.
AGENT_NAME=${AGENT_NAME:-$(basename "${TEAM_CONFIG:-agent}")}   # loose fallback
WEATHER_INDEX=${WEATHER_INDEX:-0}

# Weather label (no padding necessary)
WEATHER_LABEL="weather_${WEATHER_INDEX}"

# If TOWN_NUM not provided, try to infer from ROUTES_FILE name (e.g., ".../routes_town01_long.xml")
if [[ -z "${TOWN_NUM:-}" && -n "${ROUTES_FILE:-}" ]]; then
  # extract digits following 'town'
  TOWN_NUM=$(echo "$ROUTES_FILE" | sed -n 's/.*town\([0-9]\+\).*/\1/p' || true)
fi
TOWN_NUM=${TOWN_NUM:-unknown}

# Map label with zero-padding when numeric (map_01, map_10, etc.)
if [[ "$TOWN_NUM" =~ ^[0-9]+$ ]]; then
  printf -v MAP_LABEL "map_%02d" "$TOWN_NUM"
else
  MAP_LABEL="map_${TOWN_NUM}"
fi

# Route name: prefer explicit ROUTE_NAME; otherwise use the stem of ROUTES_FILE
if [[ -n "${ROUTE_NAME:-}" ]]; then
  ROUTE_LABEL="$ROUTE_NAME"
elif [[ -n "${ROUTES_FILE:-}" ]]; then
  base="$(basename "$ROUTES_FILE")"
  ROUTE_LABEL="${base%.*}"
else
  ROUTE_LABEL="route_unknown"
fi

# Final path matches consolidated_agent.py:
#   {DATASET_DIR}/{AGENT_NAME}/{weather_x}/{map_xx}/{route_label}
SAVE_PATH_DEFAULT="${DATASET_DIR}/${AGENT_NAME}/${WEATHER_LABEL}/${MAP_LABEL}/${ROUTE_LABEL}"
export SAVE_PATH="${SAVE_PATH:-$SAVE_PATH_DEFAULT}"
mkdir -p "$SAVE_PATH"

# Leaderboard writes metrics here (one JSON per route)
export CHECKPOINT_ENDPOINT="${SAVE_PATH}/results.json"
# (Optional) CARLA logs
export RECORD_PATH="${RECORD_PATH:-$STATE_DIR/carla_logs/${NODE_NAME}/gpu${GPU_ID}}"

echo "[worker] SAVE_PATH=$SAVE_PATH" | tee -a "$log"
echo "[worker] CHECKPOINT_ENDPOINT=$CHECKPOINT_ENDPOINT" | tee -a "$log"
# ================================================================================================

# ----- Main job loop -----
while true; do
  # On-demand server restart
  if [[ -f "$STATE_DIR/restart/${NODE_NAME}_gpu${GPU_ID}.restart" ]]; then
    echo "[worker] restart flag detected; re-ensuring CARLA..." | tee -a "$log"
    rm -f "$STATE_DIR/restart/${NODE_NAME}_gpu${GPU_ID}.restart" || true
    python3 "$PROJECT_ROOT/carla_server_manager.py" ensure \
      --gpu "$GPU_ID" \
      --base-rpc-port "$BASE_RPC_PORT" \
      --port-spacing "$PORT_SPACING" \
      --tm-offset "$TM_OFFSET" | tee -a "$log" || true
  fi

  set +e
  python3 "$PROJECT_ROOT/manage_continuous.py" run \
    --host 127.0.0.1 \
    --port "$RPC_PORT" \
    --trafficManagerPort "$TM_PORT" >>"$log" 2>&1
  rc=$?
  set -e

  case "$rc" in
    0)
      echo "[worker] job completed ok" | tee -a "$log"
      ;;
    2)
      # no pending jobs — stay alive
      python3 - "$HB" <<'PY'
import json, sys, datetime
p=sys.argv[1]
try: d=json.load(open(p))
except: d={}
d['status']='idle'; d['message']='no jobs pending'
d['last_heartbeat']=datetime.datetime.utcnow().isoformat()+'Z'
open(p,'w').write(json.dumps(d, indent=2))
PY
      sleep 15
      ;;
    130) echo "[worker] interrupted"; exit 130 ;;
    *)  echo "[worker] job failed (rc=$rc); continuing..." | tee -a "$log" ;;
  esac
done
