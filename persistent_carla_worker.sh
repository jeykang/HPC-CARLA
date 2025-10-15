#!/usr/bin/env bash
# Historical worker used in “persistent” runs.
# Now forced to be CLIENT-ONLY to avoid double-spawning CARLA.
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"
: "${BASE_RPC_PORT:=${BASE_RPC_PORT:-2000}}"
: "${PORT_SPACING:=${PORT_SPACING:-100}}"
: "${TM_OFFSET:=${TM_OFFSET:-5000}}"

: "${GPU_ID:=${GPU_ID:-0}}"
RPC_PORT=$((BASE_RPC_PORT + GPU_ID * PORT_SPACING))
TM_PORT=$((RPC_PORT + TM_OFFSET))

export CLIENT_ONLY=1
export PERSISTENT=1
export GPU_ID
export CARLA_HOST=127.0.0.1
export CARLA_PORT="${RPC_PORT}"
export TM_PORT="${TM_PORT}"

# ensure dirs
mkdir -p "${STATE_DIR:-$PROJECT_ROOT/collection_state}" "${LOG_DIR:-$PROJECT_ROOT/logs}"

# start healthbeat (background)
export PROJECT_ROOT STATE_DIR LOG_DIR GPU_ID BASE_RPC_PORT PORT_SPACING TM_OFFSET
export HEARTBEAT_SECS="${HEARTBEAT_SECS:-10}"
export STATUS="${STATUS:-idle}" MESSAGE="${MESSAGE:-}" JOBS_COMPLETED="${JOBS_COMPLETED:-0}"
"${PROJECT_ROOT}/gpu_healthbeat_daemon.sh" >"${LOG_DIR}/healthbeat_gpu${GPU_ID}.log" 2>&1 &
echo "[worker] healthbeat started (pid=$!) for GPU ${GPU_ID}"


echo "[GPU ${GPU_ID}] persistent_carla_worker: client-only against ${CARLA_HOST}:${CARLA_PORT}"
bash ./generate_single_job.sh
