#!/bin/bash
# ================================================================
#  Single job launcher that connects to persistent CARLA servers
#  FIXED: Proper variable substitution in heredoc
# ================================================================

set -e

# Set all defaults based on original pattern
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
AGENT_TYPE=${AGENT_TYPE:-interfuser}
WEATHER_IDX=${WEATHER_START:-0}
ROUTE_FILE=${ROUTE_FILE:-routes_town01_short.xml}
GPU_ID=${GPU_ID:-0}

# Derived paths
LOG_DIR="${PROJECT_ROOT}/logs"
DATASET_DIR="${PROJECT_ROOT}/dataset"
CARLA_SIF="${PROJECT_ROOT}/carla_official.sif"
WORKSPACE_DIR="/workspace"
AGENT_DIR_CONTAINER="${WORKSPACE_DIR}/leaderboard/team_code"

# Create directories if needed
mkdir -p "$LOG_DIR" "$DATASET_DIR"

# Port configuration - assumes servers are already running
# Using 100-port spacing for safety
RPC_PORT=$((2000 + GPU_ID * 100))
TM_PORT=$((8000 + GPU_ID * 100))

echo "[GPU $GPU_ID] Job configuration:"
echo "  Agent: $AGENT_TYPE"
echo "  Weather: $WEATHER_IDX"
echo "  Route: $ROUTE_FILE"
echo "  Ports: RPC=$RPC_PORT, TM=$TM_PORT"

# Extract town number from route file
TOWN_NUM=$(echo "$ROUTE_FILE" | sed -n 's/.*routes_town\([0-9]\+\).*/\1/p')
if [ ${#TOWN_NUM} -eq 1 ]; then
    TOWN_NUM="0${TOWN_NUM}"
fi

if [ -z "$TOWN_NUM" ]; then
    echo "[GPU $GPU_ID] ERROR: Could not extract town number from route file: $ROUTE_FILE" >&2
    exit 1
fi

# Agent configuration
CONSOLIDATED_AGENT_PATH="${AGENT_DIR_CONTAINER}/consolidated_agent.py"
AGENT_YAML_CONFIG_CONTAINER="${AGENT_DIR_CONTAINER}/configs/${AGENT_TYPE}.yaml"

# Validate required files exist on host
ROUTES_HOST="${PROJECT_ROOT}/leaderboard/data/training_routes/${ROUTE_FILE}"
SCENARIOS_HOST="${PROJECT_ROOT}/leaderboard/data/scenarios/town${TOWN_NUM}_all_scenarios.json"
AGENT_CONFIG_HOST="${PROJECT_ROOT}/leaderboard/team_code/configs/${AGENT_TYPE}.yaml"

echo "[GPU $GPU_ID] Validating required files..."
if [ ! -f "$ROUTES_HOST" ]; then
    echo "[GPU $GPU_ID] ERROR: Route file not found: $ROUTES_HOST" >&2
    exit 1
fi

if [ ! -f "$SCENARIOS_HOST" ]; then
    echo "[GPU $GPU_ID] ERROR: Scenario file not found: $SCENARIOS_HOST" >&2
    exit 1
fi

if [ ! -f "$AGENT_CONFIG_HOST" ]; then
    echo "[GPU $GPU_ID] ERROR: Agent config not found: $AGENT_CONFIG_HOST" >&2
    exit 1
fi

# Check if CARLA server is running on expected port
echo "[GPU $GPU_ID] Checking CARLA server status..."
if ! nc -z localhost $RPC_PORT 2>/dev/null; then
    echo "[GPU $GPU_ID] WARNING: CARLA server not responding on port $RPC_PORT" >&2
    echo "[GPU $GPU_ID] Attempting to start CARLA server..." >&2
    
    # Try to start the server using the manager
    python3 -c "
import sys
import os
sys.path.insert(0, '$PROJECT_ROOT')
from carla_server_manager import CarlaServer

server = CarlaServer($GPU_ID, $RPC_PORT, 
                    carla_sif='$CARLA_SIF',
                    project_root='$PROJECT_ROOT')
if server.start():
    print('[GPU $GPU_ID] Successfully started CARLA server')
    sys.exit(0)
else:
    print('[GPU $GPU_ID] Failed to start CARLA server')
    sys.exit(1)
" || {
    echo "[GPU $GPU_ID] ERROR: Could not start CARLA server" >&2
    exit 1
}
    # Wait a bit for server to stabilize
    sleep 10
fi

echo "[GPU $GPU_ID] CARLA server is ready on port $RPC_PORT"

# Run the evaluation job (no CARLA startup needed!)
# Note: Using heredoc WITHOUT quotes to allow variable substitution
singularity exec --nv \
  --bind "${PROJECT_ROOT}:${WORKSPACE_DIR}" \
  "$CARLA_SIF" bash << EVALUATION_SCRIPT
set -ex

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# PYTHONPATH setup
export CARLA_ROOT=/home/carla
export PYTHONPATH=\$CARLA_ROOT/PythonAPI/carla/dist/*:\$CARLA_ROOT/PythonAPI/carla:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}/scenario_runner:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}/leaderboard:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}:\$PYTHONPATH
export SCENARIO_RUNNER_ROOT=${WORKSPACE_DIR}/scenario_runner

# Prepare save path
ROUTE_NAME_NO_EXT="\${ROUTE_FILE%.xml}"
ROUTE_NAME_NO_EXT=\$(basename "\$ROUTE_NAME_NO_EXT")
SAVE_PATH="${WORKSPACE_DIR}/dataset/agent-${AGENT_TYPE}/weather-${WEATHER_IDX}/\${ROUTE_NAME_NO_EXT}"
mkdir -p "\$SAVE_PATH"

# Set environment variables
export ROUTES=${WORKSPACE_DIR}/leaderboard/data/training_routes/${ROUTE_FILE}
export SCENARIOS=${WORKSPACE_DIR}/leaderboard/data/scenarios/town${TOWN_NUM}_all_scenarios.json
export TEAM_AGENT="${CONSOLIDATED_AGENT_PATH}"
export TEAM_CONFIG="${AGENT_YAML_CONFIG_CONTAINER}"
export CHECKPOINT_ENDPOINT="\$SAVE_PATH/results.json"
export SAVE_PATH
export CONSOLIDATED_AGENT="true"

# Verify files exist inside container
if [ ! -f "\$ROUTES" ]; then 
    echo "[GPU ${GPU_ID}] FATAL: Route file not found in container: \$ROUTES" >&2
    exit 1
fi
if [ ! -f "\$SCENARIOS" ]; then 
    echo "[GPU ${GPU_ID}] FATAL: Scenario file not found in container: \$SCENARIOS" >&2
    exit 1
fi
if [ ! -f "\$TEAM_AGENT" ]; then 
    echo "[GPU ${GPU_ID}] FATAL: Consolidated agent not found: \$TEAM_AGENT" >&2
    exit 1
fi
if [ ! -f "\$TEAM_CONFIG" ]; then 
    echo "[GPU ${GPU_ID}] FATAL: Agent config not found: \$TEAM_CONFIG" >&2
    exit 1
fi

echo "[GPU ${GPU_ID}] Starting evaluation (connecting to existing CARLA server)..."
echo "  Agent: ${AGENT_TYPE}"
echo "  Weather: ${WEATHER_IDX}"
echo "  Route: ${ROUTE_FILE}"
echo "  Town: ${TOWN_NUM}"
echo "  Server port: ${RPC_PORT}"
echo "  Save path: \$SAVE_PATH"

# Change to workspace for relative imports
cd ${WORKSPACE_DIR}

# Run leaderboard evaluation (connects to existing CARLA server)
python ${WORKSPACE_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py \\
    --agent=\$TEAM_AGENT \\
    --agent-config=\$TEAM_CONFIG \\
    --routes=\$ROUTES \\
    --scenarios=\$SCENARIOS \\
    --checkpoint=\$CHECKPOINT_ENDPOINT \\
    --port=${RPC_PORT} \\
    --trafficManagerPort=${TM_PORT} \\
    --host=localhost \\
    --timeout=60 \\
    --debug=1 \\
    --record="\$SAVE_PATH" || EXIT_CODE=\$?

if [ \${EXIT_CODE:-0} -ne 0 ]; then
    echo "[GPU ${GPU_ID}] Evaluation failed with exit code: \${EXIT_CODE}" >&2
    
    # Check if it's a server connectivity issue
    if ! nc -z localhost ${RPC_PORT} 2>/dev/null; then
        echo "[GPU ${GPU_ID}] CARLA server appears to have crashed" >&2
        exit 99  # Special exit code for server crash
    fi
fi

echo "[GPU ${GPU_ID}] Evaluation complete"
exit \${EXIT_CODE:-0}
EVALUATION_SCRIPT

EXIT_CODE=$?

# Special handling for server crashes
if [ $EXIT_CODE -eq 99 ]; then
    echo "[GPU $GPU_ID] Detected server crash, will be restarted by manager"
fi

exit $EXIT_CODE