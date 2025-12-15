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

echo "[GPU ${GPU_ID}] persistent_carla_worker: client-only against ${CARLA_HOST}:${CARLA_PORT}"

while true; do
	python3 -u manage_continuous.py run --host "${CARLA_HOST}" --port "${CARLA_PORT}" --trafficManagerPort "${TM_PORT}" || rc=$?
	rc=${rc:-0}
	if [[ "${rc}" -eq 2 ]]; then
		echo "[GPU ${GPU_ID}] No pending jobs; worker exiting."
		break
	fi
	unset rc
done
