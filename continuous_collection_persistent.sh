#!/usr/bin/env bash
# Coordinator for persistent mode on a single node.
# Spawns one persistent_carla_worker.sh per local GPU and exits when the queue drains.
set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT}"
STATE_DIR=${STATE_DIR:-$PROJECT_ROOT/collection_state}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs}
mkdir -p "$STATE_DIR/health" "$STATE_DIR/restart" "$LOG_DIR"

PORT_SPACING=${PORT_SPACING:-100}
TM_OFFSET=${TM_OFFSET:-5000}
GPUS_PER_NODE=${GPUS_PER_NODE:-${LOCAL_GPUS:-8}}
NODE_NAME=${SLURMD_NODENAME:-$(hostname)}
NODE_ID=${SLURM_NODEID:-0}

# Stable run identifier shared across all subprocesses (workers/server manager/metrics)
HPC_CARLA_RUN_ID=${HPC_CARLA_RUN_ID:-${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}}
export HPC_CARLA_RUN_ID

# Derive per-node port base so nodes never collide.
BASE_RPC_PORT=${BASE_RPC_PORT:-$((2000 + NODE_ID * 1000))}

echo "[coordinator] node=$NODE_NAME id=$NODE_ID gpus=$GPUS_PER_NODE base_rpc=$BASE_RPC_PORT"
[[ -n "${SLURM_JOB_ID:-}" ]] && echo "$SLURM_JOB_ID" > "$STATE_DIR/current_slurm_job.txt" || true

# Record coordinator lifecycle for wall-time scaling plots.
RUN_DIR="$STATE_DIR/runs/$HPC_CARLA_RUN_ID"
mkdir -p "$RUN_DIR"
node_run_file="$RUN_DIR/node_${NODE_NAME}.json"
python3 - <<'PY' "$node_run_file" "$NODE_NAME" "$NODE_ID" "$GPUS_PER_NODE" "$BASE_RPC_PORT" "$HPC_CARLA_RUN_ID"
import json, sys, datetime
path, node, node_id, gpus, base_rpc, run_id = sys.argv[1:]
payload={
  "run_id": run_id,
  "node": node,
  "node_id": int(node_id),
  "gpus_per_node": int(gpus),
  "base_rpc_port": int(base_rpc),
  "coordinator_start_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(path, 'w') as f:
  json.dump(payload, f, indent=2)
PY

# (Best-effort) start/ensure a pool of CARLA servers for all local GPUs so ports are ready.
python3 "$PROJECT_ROOT/carla_server_manager.py" start \
  --gpus auto \
  --base-rpc-port "$BASE_RPC_PORT" \
  --port-spacing "$PORT_SPACING" \
  --tm-offset "$TM_OFFSET" | tee -a "$LOG_DIR/carla_pool_${NODE_NAME}.log" || true

# 1) make sure scripts are executable
chmod +x "${PROJECT_ROOT}/launch_metrics_daemon.sh" || true

# 2) start metrics daemon per node
bash "${PROJECT_ROOT}/launch_metrics_daemon.sh"

# Spawn workers.
pids=()
for gpu in $(seq 0 $((GPUS_PER_NODE - 1))); do
  (
    export GPU_ID=$gpu
    export BASE_RPC_PORT
    export PORT_SPACING
    export TM_OFFSET
    exec bash "$PROJECT_ROOT/persistent_carla_worker.sh"
  ) &
  pids+=($!)
done

# Supervisor: exit once the global queue is empty AND nothing is running.
while true; do
  status=$(python3 "$PROJECT_ROOT/manage_continuous.py" status --json 2>/dev/null || true)
  pending=$(python3 - <<'PY' "$status"
import json, sys
raw=sys.argv[1]
try:
  d=json.loads(raw)
  print(int(d.get('jobs',{}).get('pending',0)))
except Exception:
  print('NA')
PY
)
  running=$(python3 - <<'PY' "$status"
import json, sys
raw=sys.argv[1]
try:
  d=json.loads(raw)
  print(int(d.get('jobs',{}).get('running',0)))
except Exception:
  print('NA')
PY
)
  echo "[coordinator] pending=${pending} running=${running}" | tee -a "$LOG_DIR/coordinator_${NODE_NAME}.log"
  if [[ "$pending" == "0" && "$running" == "0" ]]; then
    echo "[coordinator] queue drained; stopping workers." | tee -a "$LOG_DIR/coordinator_${NODE_NAME}.log"
    break
  fi
  sleep 30
done

# Record coordinator end timestamp.
python3 - <<'PY' "$node_run_file"
import json, sys, datetime
path=sys.argv[1]
try:
  d=json.load(open(path))
except Exception:
  d={}
d['coordinator_end_ts']=datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(path,'w') as f:
  json.dump(d,f,indent=2)
PY

# Graceful shutdown.
for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
wait || true