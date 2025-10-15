#!/usr/bin/env bash
# Lightweight per-GPU heartbeat writer for persistent workers.
# Writes JSON to $STATE_DIR/health/${NODE_NAME}_gpu${GPU_ID}.json every 5s.

set -euo pipefail

: "${STATE_DIR:?STATE_DIR must be set}"
HEALTH_DIR="$STATE_DIR/health"
mkdir -p "$HEALTH_DIR"

NODE_NAME="${SLURMD_NODENAME:-$(hostname)}"
GPU_ID="${GPU_ID:-0}"
RPC_PORT="${RPC_PORT:-$((2000 + GPU_ID*10))}"
TM_PORT="${TM_PORT:-$((8000 + GPU_ID*10))}"

# A file that the worker can touch with current job id, e.g. echo 123 > $STATE_DIR/gpu${GPU_ID}.job
JOB_FILE="${JOB_FILE:-$STATE_DIR/gpu${GPU_ID}.job}"

# Function to infer status: 'busy' if JOB_FILE non-empty, else 'idle'
infer_status() {
  if [[ -s "$JOB_FILE" ]]; then
    echo "busy"
  else
    echo "idle"
  fi
}

while true; do
  NOW="$(date -Is --utc)"
  STATUS="$(infer_status)"
  CUR_JOB=""
  if [[ -s "$JOB_FILE" ]]; then
    CUR_JOB="$(cat "$JOB_FILE" | tr -d '\n' || true)"
  fi

  cat > "${HEALTH_DIR}/${NODE_NAME}_gpu${GPU_ID}.json" <<EOF
{
  "node": "${NODE_NAME}",
  "gpu_id": ${GPU_ID},
  "status": "${STATUS}",
  "rpc_port": ${RPC_PORT},
  "tm_port": ${TM_PORT},
  "current_job": "${CUR_JOB}",
  "jobs_completed": 0,
  "last_heartbeat": "${NOW}",
  "message": ""
}
EOF

  sleep 5
done
