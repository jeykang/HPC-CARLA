#!/usr/bin/env bash
# Persistent-mode coordinator for a single SLURM node.
# - Starts a CARLA server per GPU via carla_server_manager.py
# - Spawns a client-only worker per GPU that runs evaluation jobs
#   against the already-running server.
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
