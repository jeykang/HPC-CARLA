#!/bin/bash
# Dynamically generated SLURM submission script
# Generated at: 2025-12-15T09:22:30.398174
# Mode: persistent

#SBATCH --job-name=carla_collection_persistent
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-pr-a-pod09
#SBATCH --gpus-per-node=8
#SBATCH --time=168:00:00
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

: "${PROJECT_ROOT:=$(pwd)}"
: "${BASE_RPC_PORT:=${BASE_RPC_PORT:-$((2000 + ${SLURM_NODEID:-0} * 1000))}}"
: "${PORT_SPACING:=${PORT_SPACING:-100}}"
: "${TM_OFFSET:=${TM_OFFSET:-5000}}"
: "${CARLA_SIF:=carla_official.sif}"

# Optional virtualization:
# - Set VIRTUAL_GPUS (or NUM_GPUS) to spawn that many independent CARLA servers/clients.
# - Set PHYSICAL_CUDA_DEVICE to pin all servers to one physical CUDA device.
: "${VIRTUAL_GPUS:=${VIRTUAL_GPUS:-${NUM_GPUS:-}}}"
: "${PHYSICAL_CUDA_DEVICE:=${PHYSICAL_CUDA_DEVICE:-}}"

cd "${PROJECT_ROOT}"

# Discover GPUs on this node (honors CUDA_VISIBLE_DEVICES if set)
discover_gpus() {
  # Virtual mode: slots are 0..VIRTUAL_GPUS-1
  if [[ -n "${VIRTUAL_GPUS:-}" ]]; then
    seq 0 $((VIRTUAL_GPUS-1))
    return 0
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a map <<< "${CUDA_VISIBLE_DEVICES}"
    echo "${!map[@]}"
  elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    seq 0 $((SLURM_GPUS_ON_NODE-1))
  else
    nvidia-smi --list-gpus | nl -v0 -w1 -s: | awk -F: '{print $1}'
  fi
}

start_persistent_servers() {
  echo "[node $(hostname)] Starting persistent CARLA servers..."
  local -a GPUS=($(discover_gpus))
  local GPUS_ARG
  GPUS_ARG=$(IFS=, ; echo "${GPUS[*]}")

  # If PHYSICAL_CUDA_DEVICE is set, the manager will pin all slots to that device.
  if [[ -n "${PHYSICAL_CUDA_DEVICE:-}" ]]; then
    export PHYSICAL_CUDA_DEVICE
  fi

  python3 -u carla_server_manager.py start \
    --gpus "${GPUS_ARG}" \
    --base-rpc-port "${BASE_RPC_PORT}" \
    --port-spacing "${PORT_SPACING}" \
    --tm-offset "${TM_OFFSET}"
  echo "[node $(hostname)] Servers are up."
}

wait_for_port() {
  local port="$1"
  local deadline=$((SECONDS+120))
  until timeout 0.5 bash -lc "echo > /dev/tcp/127.0.0.1/${port}" 2>/dev/null; do
    if (( SECONDS > deadline )); then
      echo "[wait] Timed out waiting for port ${port}"
      return 1
    fi
    sleep 0.5
  done
  return 0
}

run_worker() {
  local GPU_ID="$1"
  local RPC_PORT=$((BASE_RPC_PORT + GPU_ID * PORT_SPACING))
  local TM_PORT=$((RPC_PORT + TM_OFFSET))

  echo "[GPU ${GPU_ID}] client-only worker starting (rpc=${RPC_PORT}, tm=${TM_PORT})"

  # Ensure server is listening
  wait_for_port "${RPC_PORT}"

  # Export to make generate_single_job.sh client-only and to unify ports.
  export CLIENT_ONLY=1
  export PERSISTENT=1
  export GPU_ID
  export CARLA_HOST=127.0.0.1
  export CARLA_PORT="${RPC_PORT}"
  export TM_PORT="${TM_PORT}"
  export BASE_RPC_PORT PORT_SPACING TM_OFFSET

  # Loop until the queue is empty.
  while true; do
    python3 -u manage_continuous.py run --host "${CARLA_HOST}" --port "${CARLA_PORT}" --trafficManagerPort "${TM_PORT}" || rc=$?
    rc=${rc:-0}
    if [[ "${rc}" -eq 2 ]]; then
      echo "[GPU ${GPU_ID}] No pending jobs; worker exiting."
      break
    fi
    # Any other exit code indicates the job failed (already recorded in DB); continue.
    unset rc
  done
}

main() {
  trap 'python3 -u carla_server_manager.py stop || true' EXIT

  start_persistent_servers

  local -a GPUS=($(discover_gpus))
  echo "Detected GPUs: ${GPUS[*]}"

  # One worker per GPU
  for gid in "${GPUS[@]}"; do
    ( run_worker "${gid}" ) &
    # Small stagger reduces I/O bursts
    sleep 1
  done

  wait
  echo "[node $(hostname)] All workers finished."
}

main "$@"
