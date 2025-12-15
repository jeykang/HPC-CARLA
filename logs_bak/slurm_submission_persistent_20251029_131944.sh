#!/bin/bash
# Dynamically generated SLURM submission script
# Generated at: 2025-10-29T13:19:44.657804
# Mode: persistent

#SBATCH --job-name=carla_collection_persistent
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod10
#SBATCH --gpus-per-node=8
#SBATCH --time=336:00:00
#SBATCH --exclusive
#SBATCH --output=/scratch/autodr_test/HPC-CARLA-persistent/logs/collection_persistent_%A_%N.out
#SBATCH --error=/scratch/autodr_test/HPC-CARLA-persistent/logs/collection_persistent_%A_%N.err

# Environment setup
export PROJECT_ROOT=/scratch/autodr_test/HPC-CARLA-persistent
export STATE_DIR=/scratch/autodr_test/HPC-CARLA-persistent/collection_state
export LOG_DIR=/scratch/autodr_test/HPC-CARLA-persistent/logs
export DATASET_DIR=/scratch/autodr_test/HPC-CARLA-persistent/dataset

# Multi-node GPU configuration
export GPUS_PER_NODE=8
export NUM_NODES=1
export NUM_GPUS=8  # Total across all nodes

# Detect if running under SLURM and adjust if needed
if [ -n "$SLURM_JOB_ID" ]; then
    # Get actual values from SLURM environment
    if [ -n "$SLURM_NNODES" ]; then
        export NUM_NODES=$SLURM_NNODES
    fi
    if [ -n "$SLURM_GPUS_PER_NODE" ]; then
        # Parse SLURM GPU string (e.g., "gpu:8" or just "8")
        export GPUS_PER_NODE=$(echo $SLURM_GPUS_PER_NODE | grep -oE "[0-9]+$")
    fi
    # Recalculate total GPUs based on actual SLURM allocation
    export NUM_GPUS=$((GPUS_PER_NODE * NUM_NODES))
    echo "SLURM allocated: $NUM_NODES nodes × $GPUS_PER_NODE GPUs/node = $NUM_GPUS total GPUs"
fi

export BASE_RPC_PORT=${BASE_RPC_PORT:-2000}
export BASE_TM_PORT=${BASE_TM_PORT:-8000}


# Log node information
echo "=========================================="
echo "SLURM JOB INFORMATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Node name: $SLURMD_NODENAME"
echo "Node ID: ${SLURM_NODEID:-0}"
echo "Number of nodes: $NUM_NODES"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $NUM_GPUS"
echo "=========================================="
echo ""

# For multi-node coordination
export NODE_ID=${SLURM_NODEID:-0}
export NODE_NAME=${SLURMD_NODENAME:-$(hostname)}
export IS_MASTER=$([[ ${NODE_ID} -eq 0 ]] && echo "true" || echo "false")

# Adjust port ranges per node to avoid conflicts
# Each node gets a different port range
export BASE_RPC_PORT=$((2000 + NODE_ID * 1000))
export BASE_TM_PORT=$((8000 + NODE_ID * 1000))
echo "This node will use RPC ports ${BASE_RPC_PORT}-$((BASE_RPC_PORT + GPUS_PER_NODE * 10))"

# Multi-node persistent CARLA server management
if [ "$IS_MASTER" == "true" ]; then
    echo "Master node: Will coordinate multi-node setup"
else
    echo "Worker node $NODE_ID: Waiting for master initialization"
    sleep 10
fi

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

# Derive per-node port base so nodes never collide.
BASE_RPC_PORT=${BASE_RPC_PORT:-$((2000 + NODE_ID * 1000))}

echo "[coordinator] node=$NODE_NAME id=$NODE_ID gpus=$GPUS_PER_NODE base_rpc=$BASE_RPC_PORT"
[[ -n "${SLURM_JOB_ID:-}" ]] && echo "$SLURM_JOB_ID" > "$STATE_DIR/current_slurm_job.txt" || true

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
  status=$(python3 - <<'PY'
import json, os
p=os.path.join(os.environ.get('STATE_DIR','collection_state'),'job_queue.json')
try:
  q=json.load(open(p))
  pending=sum(1 for j in q['jobs'] if j['status']=='pending')
  running=sum(1 for j in q['jobs'] if j['status'] in ('running','assigned'))
  print(f"{pending},{running}")
except Exception:
  print("NA,NA")
PY
)
  IFS=, read -r pending running <<<"$status"
  echo "[coordinator] pending=${pending} running=${running}" | tee -a "$LOG_DIR/coordinator_${NODE_NAME}.log"
  if [[ "$pending" == "0" && "$running" == "0" ]]; then
    echo "[coordinator] queue drained; stopping workers." | tee -a "$LOG_DIR/coordinator_${NODE_NAME}.log"
    break
  fi
  sleep 30
done

# Graceful shutdown.
for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
wait || true
