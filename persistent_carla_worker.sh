#!/usr/bin/env bash
# Historical worker used in “persistent” runs.
# Runs CLIENT-ONLY (servers are managed separately).
set -euo pipefail

: "${PROJECT_ROOT:=$(pwd)}"
: "${STATE_DIR:=${PROJECT_ROOT}/collection_state}"
: "${LOG_DIR:=${PROJECT_ROOT}/logs}"
: "${DATASET_DIR:=${PROJECT_ROOT}/dataset}"

: "${BASE_RPC_PORT:=${BASE_RPC_PORT:-2000}}"
: "${PORT_SPACING:=${PORT_SPACING:-100}}"
: "${TM_OFFSET:=${TM_OFFSET:-5000}}"

: "${GPU_ID:=${GPU_ID:-0}}"
# ensure GPU_ID is set for manage_continuous.py
export GPU_ID="${GPU_ID:-${SLURM_LOCALID:-${CUDA_VISIBLE_DEVICES%%,*}}}"


NODE="$(hostname -s)"
RPC_PORT=$((BASE_RPC_PORT + GPU_ID * PORT_SPACING))
TM_PORT=$((RPC_PORT + TM_OFFSET))

export CLIENT_ONLY=1
export PERSISTENT=1

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${DATASET_DIR}"

# Ensure JOB_FILE is set (used by heartbeat to show busy vs idle)
: "${JOB_FILE:=${STATE_DIR}/${NODE}_gpu${GPU_ID}.job}"

# If the caller didn't already start the heartbeat, start it here (idempotent).
if ! pgrep -f "gpu_healthbeat_daemon.sh" >/dev/null 2>&1; then
  export GPU_ID RPC_PORT TM_PORT JOB_FILE STATE_DIR
  nohup bash "${PROJECT_ROOT}/gpu_healthbeat_daemon.sh" \
    > "${LOG_DIR}/healthbeat_${NODE}_gpu${GPU_ID}.log" 2>&1 &
  echo "[worker ${NODE}/gpu${GPU_ID}] heartbeat started pid=$!"
fi

# Typical loop is delegated to your existing single-job runner.
# It will call `python3 manage_continuous.py run ...` internally.
# Keep as-is to minimize disruption.
if [[ -x "${PROJECT_ROOT}/generate_single_job.sh" ]]; then
  echo "[worker ${NODE}/gpu${GPU_ID}] launching generate_single_job.sh (RPC=${RPC_PORT}, TM=${TM_PORT})"
  exec bash "${PROJECT_ROOT}/generate_single_job.sh"
else
  echo "[worker ${NODE}/gpu${GPU_ID}] FATAL: generate_single_job.sh not found or not executable." >&2
  exit 1
fi
