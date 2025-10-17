#!/usr/bin/env bash
set -euo pipefail
NODE_NAME="${NODE_NAME:-${SLURMD_NODENAME:-$(hostname)}}"
GPU_ID="${GPU_ID:-0}"
STATE_DIR="${STATE_DIR:-$PWD/collection_state}"
RPC="${CARLA_PORT:-2000}"
TM="${TM_PORT:-5000}"
INTERVAL="${INTERVAL:-20}"

mkdir -p "${STATE_DIR}/health"
while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "${STATE_DIR}/health/${NODE_NAME}_gpu${GPU_ID}.json" <<EOF
{"node":"${NODE_NAME}","gpu_id":${GPU_ID},"status":"idle","message":"hb","rpc_port":${RPC},"tm_port":${TM},"last_heartbeat":"${ts}"}
EOF
  sleep "${INTERVAL}"
done
