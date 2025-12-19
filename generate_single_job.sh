#!/usr/bin/env bash
# Launches ONE evaluation job on the CURRENT GPU.
# In persistent mode (CLIENT_ONLY=1 or PERSISTENT=1), this script WILL NOT start CARLA.
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"
: "${CARLA_SIF:=carla_official.sif}"
: "${UE4_QUALITY:=Epic}"

: "${GPU_ID:=${GPU_ID:-0}}"
: "${BASE_RPC_PORT:=${BASE_RPC_PORT:-2000}}"
: "${PORT_SPACING:=${PORT_SPACING:-100}}"
: "${TM_OFFSET:=${TM_OFFSET:-5000}}"

# Derive ports (persistent scheme by default)
RPC_PORT_DEFAULT=$((BASE_RPC_PORT + GPU_ID * PORT_SPACING))
TM_PORT_DEFAULT=$((RPC_PORT_DEFAULT + TM_OFFSET))

# Allow explicit override from env
: "${CARLA_HOST:=127.0.0.1}"
: "${CARLA_PORT:=${CARLA_PORT:-$RPC_PORT_DEFAULT}}"
: "${TM_PORT:=${TM_PORT:-$TM_PORT_DEFAULT}}"

# Detect persistent (client-only) mode
CLIENT_ONLY="${CLIENT_ONLY:-0}"
if [[ "${PERSISTENT:-0}" = "1" ]]; then CLIENT_ONLY=1; fi

echo "[GPU ${GPU_ID}] Job starting (client-only=${CLIENT_ONLY}; host=${CARLA_HOST}; rpc=${CARLA_PORT}; tm=${TM_PORT})"

launch_carla() {
  local rpc="$1"
  echo "[GPU ${GPU_ID}] Launching CARLA server at :${rpc}"

  # Nuke stale listeners if any (rare on non-persistent path)
  bash -lc "fuser -k -TERM ${rpc}/tcp || true"
  bash -lc "lsof -ti tcp:${rpc} | xargs -r kill -TERM || true"

  NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
  singularity exec --nv -B "${PROJECT_ROOT}:/workspace" "${CARLA_SIF}" \
    bash -lc 'ulimit -c 0 ;
              DISABLE_PYTHON=1 SDL_VIDEODRIVER=offscreen
              "${CARLA_ROOT:-/home/carla}"/CarlaUE4.sh -opengl -RenderOffScreen -nosound \
                -quality-level='"${UE4_QUALITY}"' \
                -carla-rpc-port='"${rpc}"' -carla-streaming-port=0 -world-port='"${rpc}"' -server' \
    >/dev/null 2>&1 &
  echo $! > "carla_${GPU_ID}.pid"
}

wait_ready() {
  local port="$1"
  local deadline=$((SECONDS+120))
  until timeout 0.5 bash -lc "echo > /dev/tcp/127.0.0.1/${port}" 2>/dev/null; do
    if (( SECONDS > deadline )); then
      echo "[GPU ${GPU_ID}] FATAL: CARLA process failed to start."
      return 1
    fi
    sleep 0.5
  done
  return 0
}

cleanup_carla() {
  local pidfile="carla_${GPU_ID}.pid"
  if [[ -f "${pidfile}" ]]; then
    local pid
    pid="$(cat "${pidfile}")" || true
    rm -f "${pidfile}" || true
    if [[ -n "${pid}" ]]; then
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
}

run_evaluation() {
  # IMPORTANT:
  # We rely on your existing Python-side evaluator to read CARLA_HOST/CARLA_PORT/TM_PORT
  # OR accept --host/--port/--trafficManagerPort. Most CARLA leaderboard wrappers do.
  # If your wrapper requires explicit flags, append them to EVAL_EXTRA_ARGS below.
  local EVAL_EXTRA_ARGS="--host ${CARLA_HOST} --port ${CARLA_PORT} --trafficManagerPort ${TM_PORT}"

  # Hand off to your project’s orchestrator (replace with your exact command if different).
  # The commonly-used entry is a repo-specific wrapper; keeping it generic:
  python3 -u manage_continuous.py run ${EVAL_EXTRA_ARGS}
}

trap '[[ "${CLIENT_ONLY}" = "1" ]] || cleanup_carla' EXIT

if [[ "${CLIENT_ONLY}" = "1" ]]; then
  echo "[GPU ${GPU_ID}] Client-only mode: using existing persistent server."
  # Sanity check that the port is alive
  if ! timeout 0.5 bash -lc "echo > /dev/tcp/${CARLA_HOST}/${CARLA_PORT}" 2>/dev/null; then
    echo "[GPU ${GPU_ID}] ERROR: No CARLA listening at ${CARLA_HOST}:${CARLA_PORT}."
    exit 1
  fi
else
  launch_carla "${CARLA_PORT}"
  wait_ready "${CARLA_PORT}"
fi

# Run one evaluation job
run_evaluation
echo "[GPU ${GPU_ID}] Job finished."
