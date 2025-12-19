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

mkdir -p "$STATE_DIR/health" "$STATE_DIR/restart" "$LOG_DIR"
log="$LOG_DIR/worker_${NODE_NAME}_gpu${GPU_ID}.log"

# Derive per-GPU RPC/TM ports
RPC_PORT=$(( BASE_RPC_PORT + PORT_SPACING * GPU_ID ))
TM_PORT=$(( TM_OFFSET + PORT_SPACING * GPU_ID ))

# Emit initial heartbeat
python3 - <<'PY' "$STATE_DIR/health" "$NODE_NAME" "$GPU_ID" "$RPC_PORT" "$TM_PORT"
import os, sys, json, datetime, pathlib
health_dir, node, gpu, rpc, tm = sys.argv[1:]
p=pathlib.Path(health_dir)/f"{node}_gpu{gpu}.json"
p.parent.mkdir(parents=True, exist_ok=True)
d={"node":node,"gpu_id":int(gpu),"status":"idle","rpc_port":int(rpc),"tm_port":int(tm),
   "message":"worker started","last_heartbeat":datetime.datetime.utcnow().isoformat()+"Z"}
p.write_text(json.dumps(d, indent=2))
PY

echo "[worker] node=$NODE_NAME gpu=$GPU_ID rpc=$RPC_PORT tm=$TM_PORT" | tee -a "$log"

# ----- Common CARLA/Leaderboard env (per-worker, not per-job) -----
export LOCAL_GPUS=${LOCAL_GPUS:-${GPUS_PER_NODE:-8}}
export GPU_ID
export CARLA_HOST=127.0.0.1
export CARLA_PORT=$RPC_PORT
export TM_PORT=$TM_PORT

# (Removed SAVE_PATH/CHECKPOINT_ENDPOINT precomputation; per-job paths are set by manage_continuous.py)
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
    0)  echo "[worker] job completed ok" | tee -a "$log" ;;
    2)  echo "[worker] no pending jobs; sleeping and updating heartbeat" | tee -a "$log"
        python3 - <<'PY' "$STATE_DIR/health" "$NODE_NAME" "$GPU_ID"
import json, sys, datetime, pathlib
health_dir, node, gpu = sys.argv[1:]
p=pathlib.Path(health_dir)/f"{node}_gpu{gpu}.json"
try: d=json.loads(p.read_text())
except Exception: d={}
d['status']='idle'; d['message']='no jobs pending'
d['last_heartbeat']=datetime.datetime.utcnow().isoformat()+'Z'
p.write_text(json.dumps(d, indent=2))
PY
        sleep 15
        ;;
    130) echo "[worker] interrupted"; exit 130 ;;
    *)  echo "[worker] job failed (rc=$rc); continuing..." | tee -a "$log" ;;
  esac
done
