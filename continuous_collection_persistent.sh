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

cd "${PROJECT_ROOT}"

# Discover GPUs on this node (honors CUDA_VISIBLE_DEVICES if set)
discover_gpus() {
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
  python3 -u carla_server_manager.py start \
    --gpus auto \
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

  # Call the existing single-job launcher; it MUST NOT launch CARLA in persistent mode.
  # (We ship a fixed version of generate_single_job.sh that respects CLIENT_ONLY/PERSISTENT.)
  bash ./generate_single_job.sh
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
