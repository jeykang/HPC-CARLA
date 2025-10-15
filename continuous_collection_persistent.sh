#!/bin/bash
# Persistent collection coordinator (per node)
set -euo pipefail

# --- Paths & env
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export STATE_DIR="${STATE_DIR:-$PROJECT_ROOT/collection_state}"
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$STATE_DIR/health" "$LOG_DIR"

NODE_NAME="${SLURMD_NODENAME:-$(hostname)}"
NODE_ID="${SLURM_NODEID:-0}"

# One thousand ports per node to avoid conflicts across nodes
export BASE_RPC_PORT="${BASE_RPC_PORT:-$((2000 + NODE_ID * 1000))}"
export PORT_SPACING="${PORT_SPACING:-100}"
export TM_OFFSET="${TM_OFFSET:-5000}"

# How many GPUs on THIS node
discover_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    IFS=',' read -ra map <<< "$CUDA_VISIBLE_DEVICES"
    for i in "${!map[@]}"; do echo "$i"; done
  else
    n="${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-8}}"
    for ((i=0;i<n;i++)); do echo "$i"; done
  fi
}

derive_rpc() { echo $(( BASE_RPC_PORT + $1 * PORT_SPACING )); }
derive_tm () { echo $(( $(derive_rpc "$1") + TM_OFFSET )); }

wait_for_port() {
  local port="$1" timeout="$2" t0 t1
  t0=$(date +%s); t1=$(( t0 + timeout ))
  while [[ $(date +%s) -lt $t1 ]]; do
    if (echo > "/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  return 1
}

start_heartbeat() {
  local gid="$1" rpc tm
  rpc="$(derive_rpc "$gid")"; tm="$(derive_tm "$gid")"
  HEALTH_NODE_DIR="$STATE_DIR/health/$NODE_NAME"
  mkdir -p "$HEALTH_NODE_DIR"
  # Detach: one heartbeat per GPU
  GPU_ID="$gid" BASE_RPC_PORT="$BASE_RPC_PORT" PORT_SPACING="$PORT_SPACING" TM_OFFSET="$TM_OFFSET" \
    NODE_NAME="$NODE_NAME" LOG_DIR="$LOG_DIR" STATE_DIR="$STATE_DIR" \
    bash "$PROJECT_ROOT/gpu_healthbeat_daemon.sh" \
      >> "$LOG_DIR/heartbeat_${NODE_NAME}_gpu${gid}.log" 2>&1 &
}

ensure_server() {
  local gid="$1" rpc
  rpc="$(derive_rpc "$gid")"
  if ! (echo > "/dev/tcp/127.0.0.1/$rpc") >/dev/null 2>&1; then
    echo "[coordinator:$NODE_NAME] gpu${gid}: ensuring server on $rpc"
    python3 "$PROJECT_ROOT/carla_server_manager.py" ensure \
      --gpu "$gid" --base-rpc-port "$BASE_RPC_PORT" --port-spacing "$PORT_SPACING" --tm-offset "$TM_OFFSET" || true
  fi
  wait_for_port "$rpc" 90 || { echo "[coordinator] gpu${gid}: server not listening on $rpc"; }
}

run_worker() {
  local gid="$1" rpc tm
  rpc="$(derive_rpc "$gid")"; tm="$(derive_tm "$gid")"
  # IMPORTANT: leave the actual job logic to persistent_carla_worker.sh (as before)
  # We only prep env, ensure server, and then exec the existing worker.
  LOG_FILE="$LOG_DIR/worker_${NODE_NAME}_gpu${gid}.log"
  echo "[worker $NODE_NAME/gpu${gid}] launching persistent_carla_worker.sh (RPC=$rpc, TM=$tm)" | tee -a "$LOG_FILE"

  # Export what the worker/generator expects
  export GPU_ID="$gid"
  export CARLA_HOST="127.0.0.1"
  export CARLA_PORT="$rpc"
  export TM_PORT="$tm"
  export CLIENT_ONLY=1
  export NODE_NAME

  # Make sure the CARLA server is actually there:
  ensure_server "$gid"

  # Now hand off to your existing worker script
  bash "$PROJECT_ROOT/persistent_carla_worker.sh" >> "$LOG_FILE" 2>&1 &
}

main() {
  echo "[coordinator:$NODE_NAME] base_rpc=$BASE_RPC_PORT spacing=$PORT_SPACING tm_off=$TM_OFFSET"
  mapfile -t GPUS < <(discover_gpus)

  # Start per-GPU heartbeat + ensure servers + run workers
  for gid in "${GPUS[@]}"; do
    start_heartbeat "$gid"
    ensure_server "$gid"
    run_worker "$gid"
  done

  # Keep the coordinator alive so sbatch doesn’t kill children
  wait
}

main "$@"
