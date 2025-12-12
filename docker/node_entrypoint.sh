#!/usr/bin/env bash
set -euo pipefail

cd /workspace

: "${PROJECT_ROOT:=/workspace}"
: "${STATE_DIR:=${PROJECT_ROOT}/collection_state}"
: "${LOG_DIR:=${PROJECT_ROOT}/logs}"
: "${DATASET_DIR:=${PROJECT_ROOT}/dataset}"

: "${VIRTUAL_GPUS:=4}"
: "${PHYSICAL_CUDA_DEVICE:=0}"

: "${BASE_RPC_PORT:=2000}"
: "${PORT_SPACING:=100}"
: "${TM_OFFSET:=5000}"

: "${IDLE_SLEEP_S:=10}"
: "${EXIT_ON_IDLE:=0}"
: "${AUTO_RESET:=0}"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$DATASET_DIR"

# Ensure CARLA python egg is on PYTHONPATH if present.
# Carla images contain a single egg under ${CARLA_ROOT}/PythonAPI/carla/dist/
if [[ -n "${CARLA_ROOT:-}" ]] && compgen -G "${CARLA_ROOT}/PythonAPI/carla/dist/carla-*-py*-linux-x86_64.egg" > /dev/null; then
  export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist/$(basename ${CARLA_ROOT}/PythonAPI/carla/dist/carla-*-py*-linux-x86_64.egg):${PYTHONPATH:-}"
fi

export PROJECT_ROOT STATE_DIR LOG_DIR DATASET_DIR
export BASE_RPC_PORT PORT_SPACING TM_OFFSET

GPU_LIST=""
for ((i=0; i< VIRTUAL_GPUS; i++)); do GPU_LIST+="${i},"; done
GPU_LIST="${GPU_LIST%,}"

echo "[docker-node] physical CUDA device inside container: ${PHYSICAL_CUDA_DEVICE}"
echo "[docker-node] virtual GPU slots: ${GPU_LIST}"
echo "[docker-node] ports: base=${BASE_RPC_PORT} spacing=${PORT_SPACING} tm_offset=${TM_OFFSET}"

if [[ "$AUTO_RESET" = "1" ]]; then
  if [[ ! -f "${STATE_DIR}/collection.db" ]]; then
    echo "[docker-node] AUTO_RESET=1 and no collection.db found; running reset"
    python3 -u manage_continuous.py reset
  else
    echo "[docker-node] AUTO_RESET=1 but collection.db exists; leaving as-is"
  fi
fi

# Start CARLA servers (one per virtual slot) inside this container.
# All slots share the same physical GPU device.
export CARLA_LAUNCH_BACKEND="${CARLA_LAUNCH_BACKEND:-native}"
python3 -u carla_server_manager.py start \
  --gpus "${GPU_LIST}" \
  --base-rpc-port "${BASE_RPC_PORT}" \
  --port-spacing "${PORT_SPACING}" \
  --tm-offset "${TM_OFFSET}"

worker_loop() {
  local slot="$1"
  local rpc=$((BASE_RPC_PORT + slot * PORT_SPACING))
  local tm=$((rpc + TM_OFFSET))

  export GPU_ID="${slot}"
  export CLIENT_ONLY=1
  export PERSISTENT=1

  echo "[worker slot=${slot}] starting against 127.0.0.1:${rpc} (tm=${tm})"

  while true; do
    set +e
    python3 -u manage_continuous.py run --host 127.0.0.1 --port "${rpc}" --trafficManagerPort "${tm}"
    rc=$?
    set -e

    if [[ "$rc" = "2" ]]; then
      # idle
      if [[ "$EXIT_ON_IDLE" = "1" ]]; then
        echo "[worker slot=${slot}] idle; exiting"
        return 0
      fi
      sleep "${IDLE_SLEEP_S}"
      continue
    fi

    # For transient failures, don’t spin.
    sleep 1
  done
}

# Spawn one worker loop per virtual slot
for ((i=0; i< VIRTUAL_GPUS; i++)); do
  worker_loop "$i" &
done

# Graceful shutdown
trap 'echo "[docker-node] stopping"; python3 -u carla_server_manager.py stop || true' EXIT

wait
