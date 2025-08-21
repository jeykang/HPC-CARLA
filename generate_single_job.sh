#!/bin/bash
# ================================================================
#  Single job launcher for continuous collection system
#  Runs one specific agent/weather/route combination on one GPU
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

# Port configuration
RPC_PORT=$((2000 + GPU_ID * 10))
TM_PORT=$((8000 + GPU_ID * 10))

# Extract town number from route file more reliably
# Handle both routes_town01_short.xml and routes_town1_short.xml patterns
TOWN_NUM=$(echo "$ROUTE_FILE" | sed -n 's/.*routes_town\([0-9]\+\).*/\1/p')

# Ensure town number is zero-padded to 2 digits
if [ ${#TOWN_NUM} -eq 1 ]; then
    TOWN_NUM="0${TOWN_NUM}"
fi

echo "[GPU $GPU_ID] Route file: $ROUTE_FILE -> Town: $TOWN_NUM"

# Validate that we extracted a town number
if [ -z "$TOWN_NUM" ]; then
    echo "[GPU $GPU_ID] ERROR: Could not extract town number from route file: $ROUTE_FILE" >&2
    exit 1
fi

# Agent configuration
CONSOLIDATED_AGENT_PATH="${AGENT_DIR_CONTAINER}/consolidated_agent.py"
AGENT_YAML_CONFIG_CONTAINER="${AGENT_DIR_CONTAINER}/configs/${AGENT_TYPE}.yaml"

# Validate required files exist on host before starting
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
    echo "[GPU $GPU_ID] This likely means route $ROUTE_FILE doesn't match town${TOWN_NUM}" >&2
    exit 1
fi

if [ ! -f "$AGENT_CONFIG_HOST" ]; then
    echo "[GPU $GPU_ID] ERROR: Agent config not found: $AGENT_CONFIG_HOST" >&2
    exit 1
fi

if [ ! -f "$CARLA_SIF" ]; then
    echo "[GPU $GPU_ID] ERROR: CARLA singularity image not found: $CARLA_SIF" >&2
    exit 1
fi

echo "[GPU $GPU_ID] All required files validated successfully"

# Run the job
singularity exec --nv \
  --bind "${PROJECT_ROOT}:${WORKSPACE_DIR}" \
  "$CARLA_SIF" bash << CARLA_SCRIPT
set -ex

export CUDA_VISIBLE_DEVICES=$GPU_ID
export CARLA_ROOT=/home/carla

# PYTHONPATH setup
export PYTHONPATH=\$CARLA_ROOT/PythonAPI/carla/dist/*:\$CARLA_ROOT/PythonAPI/carla:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}/scenario_runner:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}/leaderboard:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}:\$PYTHONPATH
export SCENARIO_RUNNER_ROOT=${WORKSPACE_DIR}/scenario_runner

export XDG_RUNTIME_DIR=/tmp/runtime-\$USER-gpu${GPU_ID}-\$\$
mkdir -p \$XDG_RUNTIME_DIR

cd /home/carla

export SDL_AUDIODRIVER=dsp
export SDL_VIDEODRIVER=offscreen

# Launch CARLA server
echo "[GPU $GPU_ID] Starting CARLA server on port $RPC_PORT..."
./CarlaUE4.sh -carla-rpc-port=$RPC_PORT -carla-streaming-port=0 -nosound -quality-level=Epic &
CARLA_PID=\$!
echo "[GPU $GPU_ID] CARLA launched with PID \$CARLA_PID"

# Wait for CARLA to initialize
echo "[GPU $GPU_ID] Waiting 60s for CARLA to initialize..."
sleep 60

if ! ps -p \$CARLA_PID >/dev/null; then
    echo "[GPU $GPU_ID] FATAL: CARLA process failed to start." >&2
    exit 1
fi

# Prepare save path
ROUTE_NAME_NO_EXT="\${ROUTE_FILE%.xml}"
SAVE_PATH="${WORKSPACE_DIR}/dataset/agent-${AGENT_TYPE}/weather-${WEATHER_IDX}/\${ROUTE_NAME_NO_EXT}"
mkdir -p "\$SAVE_PATH"

# Set environment variables
export ROUTES=${WORKSPACE_DIR}/leaderboard/data/training_routes/$ROUTE_FILE
export SCENARIOS=${WORKSPACE_DIR}/leaderboard/data/scenarios/town${TOWN_NUM}_all_scenarios.json
export TEAM_AGENT="$CONSOLIDATED_AGENT_PATH"
export TEAM_CONFIG="$AGENT_YAML_CONFIG_CONTAINER"
export CHECKPOINT_ENDPOINT="\$SAVE_PATH/results.json"
export SAVE_PATH
export CONSOLIDATED_AGENT="true"

# Double-check files exist inside container
if [ ! -f "\$ROUTES" ]; then 
    echo "[GPU $GPU_ID] FATAL: Route file not found in container: \$ROUTES" >&2
    kill \$CARLA_PID 2>/dev/null || true
    exit 1
fi
if [ ! -f "\$SCENARIOS" ]; then 
    echo "[GPU $GPU_ID] FATAL: Scenario file not found in container: \$SCENARIOS" >&2
    echo "[GPU $GPU_ID] Expected town${TOWN_NUM}_all_scenarios.json for route $ROUTE_FILE" >&2
    kill \$CARLA_PID 2>/dev/null || true
    exit 1
fi
if [ ! -f "\$TEAM_AGENT" ]; then 
    echo "[GPU $GPU_ID] FATAL: Consolidated agent not found: \$TEAM_AGENT" >&2
    kill \$CARLA_PID 2>/dev/null || true
    exit 1
fi
if [ ! -f "\$TEAM_CONFIG" ]; then 
    echo "[GPU $GPU_ID] FATAL: Agent config not found: \$TEAM_CONFIG" >&2
    kill \$CARLA_PID 2>/dev/null || true
    exit 1
fi

echo "[GPU $GPU_ID] Starting evaluation:"
echo "  Agent: ${AGENT_TYPE}"
echo "  Weather: ${WEATHER_IDX}"
echo "  Route: ${ROUTE_FILE}"
echo "  Town: ${TOWN_NUM}"
echo "  Scenarios: town${TOWN_NUM}_all_scenarios.json"
echo "  Save path: \$SAVE_PATH"

# Run leaderboard evaluation
python ${WORKSPACE_DIR}/leaderboard/leaderboard/leaderboard_evaluator.py \
    --agent=\$TEAM_AGENT \
    --agent-config=\$TEAM_CONFIG \
    --routes=\$ROUTES \
    --scenarios=\$SCENARIOS \
    --checkpoint=\$CHECKPOINT_ENDPOINT \
    --port=$RPC_PORT \
    --trafficManagerPort=$TM_PORT \
    --host=localhost \
    --timeout=60 \
    --record="\$SAVE_PATH" || EXIT_CODE=\$?

# Check exit code
if [ \${EXIT_CODE:-0} -ne 0 ]; then
    echo "[GPU $GPU_ID] Evaluation failed with exit code: \${EXIT_CODE}" >&2
fi

# Clean up CARLA
echo "[GPU $GPU_ID] Terminating CARLA..."
kill \$CARLA_PID 2>/dev/null || true
wait \$CARLA_PID 2>/dev/null || true

# Clean up runtime directory
rm -rf \$XDG_RUNTIME_DIR

exit \${EXIT_CODE:-0}
CARLA_SCRIPT