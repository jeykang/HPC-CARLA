#!/usr/bin/env bash
# launch_metrics_daemon.sh
set -euo pipefail
: "${PROJECT_ROOT:=$(pwd)}"
: "${STATE_DIR:=${PROJECT_ROOT}/collection_state}"

mkdir -p "${STATE_DIR}/metrics"
echo "[metrics] starting on node ${SLURMD_NODENAME:-$(hostname)}; state=${STATE_DIR}"
# Prefer python3 from host; fall back to env
nohup python3 "${PROJECT_ROOT}/metrics_daemon.py" \
  --state-dir "${STATE_DIR}" --interval "${METRICS_INTERVAL:-2}" \
  >/dev/null 2>&1 &
echo $! > "${STATE_DIR}/metrics_daemon.pid"
