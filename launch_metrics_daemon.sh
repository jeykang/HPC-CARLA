#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
STATE_DIR=${STATE_DIR:-$PROJECT_ROOT/collection_state}
LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/logs}
NODE_NAME=${SLURMD_NODENAME:-$(hostname)}

METRICS_DIR="$STATE_DIR/metrics/node/$NODE_NAME"
PID_FILE="$METRICS_DIR/metrics_daemon.pid"
OUT_LOG="$LOG_DIR/metrics_daemon_${NODE_NAME}.log"

mkdir -p "$METRICS_DIR/last" "$LOG_DIR"

# If already running, exit cleanly.
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[metrics] daemon already running (pid=$pid)" >>"$OUT_LOG"
    exit 0
  fi
fi

# Launch a single daemon per node.
(
  exec python3 "$PROJECT_ROOT/tools/metrics_daemon.py" \
    --project-root "$PROJECT_ROOT" \
    --interval-sec "${METRICS_INTERVAL_SEC:-2}" \
    --rotate-gpu-mib "${METRICS_ROTATE_GPU_MIB:-256}" \
    --rotate-system-mib "${METRICS_ROTATE_SYSTEM_MIB:-64}" \
    >>"$OUT_LOG" 2>&1
) &

pid=$!
echo "$pid" >"$PID_FILE"
# Backwards compat: some tooling expects this global pid file.
echo "$pid" >"$STATE_DIR/metrics_daemon.pid" || true

echo "[metrics] started daemon pid=$pid" >>"$OUT_LOG"
