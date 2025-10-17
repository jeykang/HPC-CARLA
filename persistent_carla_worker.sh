#!/usr/bin/env bash
# One worker per GPU: ensures a CARLA server, then loops running jobs.
set -euo pipefail

GPU_ID=${GPU_ID:?GPU_ID must be set (0..GPUS_PER_NODE-1)}
NODE_NAME=${SLURMD_NODENAME:-$(hostname)}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
STATE_DIR=${STATE_DIR:-$PROJECT_ROOT/collection_state}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs}

PORT_SPACING=${PORT_SPACING:-100}
BASE_RPC_PORT=${BASE_RPC_PORT:-2000}
TM_OFFSET=${TM_OFFSET:-5000}
RPC_PORT=$((BASE_RPC_PORT + GPU_ID * PORT_SPACING))
TM_PORT=$((RPC_PORT + TM_OFFSET))

mkdir -p "$LOG_DIR" "$STATE_DIR/health" "$STATE_DIR/restart"

# Bring up (or re-confirm) the CARLA server for this GPU.
python3 "$PROJECT_ROOT/carla_server_manager.py" ensure \
  --gpu "$GPU_ID" \
  --base-rpc-port "$BASE_RPC_PORT" \
  --port-spacing "$PORT_SPACING" \
  --tm-offset "$TM_OFFSET" || true

HB="$STATE_DIR/health/${NODE_NAME}_gpu${GPU_ID}.json"
log="$LOG_DIR/persistent_worker_gpu${GPU_ID}.log"
touch "$log"

# Initial idle heartbeat so the dashboard shows the port mapping before first job.
python3 - "$HB" "$NODE_NAME" "$GPU_ID" "$RPC_PORT" "$TM_PORT" <<'PY'
import json, sys, os, datetime
p, node, gpu, rpc, tm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.makedirs(os.path.dirname(p), exist_ok=True)
d={"node":node,"gpu_id":gpu,"status":"idle","rpc_port":rpc,"tm_port":tm,
   "message":"worker started","last_heartbeat":datetime.datetime.utcnow().isoformat()+"Z"}
open(p,"w").write(json.dumps(d, indent=2))
PY

echo "[worker] node=$NODE_NAME gpu=$GPU_ID rpc=$RPC_PORT tm=$TM_PORT" | tee -a "$log"

# Make sure the evaluator knows which GPU/ports to use.
export LOCAL_GPUS=${LOCAL_GPUS:-${GPUS_PER_NODE:-8}}
export GPU_ID
export CARLA_HOST=127.0.0.1
export CARLA_PORT=$RPC_PORT
export TM_PORT=$TM_PORT

# Main job loop.
while true; do
  # Optional: allow out-of-band restart of just this GPU's server
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
    0)  echo "[worker] job completed ok"       | tee -a "$log" ;;
    1)  echo "[worker] job failed (will loop)" | tee -a "$log" ;;
    2)  # No pending jobs right now—stay alive but write a fresh idle heartbeat.
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
    *)  echo "[worker] unexpected rc=$rc (will loop)" | tee -a "$log" ;;
  esac
done
