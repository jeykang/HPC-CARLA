#!/bin/bash
# ================================================================
#  Modular distributed data collection using consolidated agent
# ================================================================

#SBATCH --job-name=modular_agent_run
#SBATCH --nodelist=hpc-pr-a-pod08
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/modular_run_%A_%a.out
#SBATCH --error=logs/modular_run_%A_%a.err

# Path Configuration
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

LOG_DIR="${PROJECT_ROOT}/logs"
CARLA_SIF="${PROJECT_ROOT}/carla_official.sif"
AGENT_DIR_HOST="${PROJECT_ROOT}/leaderboard/team_code"
DATASET_DIR="${PROJECT_ROOT}/dataset"

WORKSPACE_DIR="/workspace"
AGENT_DIR_CONTAINER="${WORKSPACE_DIR}/leaderboard/team_code"

echo "=========================================="
echo "Path Configuration:"
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  LOG_DIR:      ${LOG_DIR}"
echo "  CARLA_SIF:    ${CARLA_SIF}"
echo "  AGENT_DIR:    ${AGENT_DIR_HOST}"
echo "  DATASET_DIR:  ${DATASET_DIR}"
echo "=========================================="

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$DATASET_DIR"

# Validation
if [ ! -f "$CARLA_SIF" ]; then
    echo "FATAL: Singularity image not found at $CARLA_SIF" >&2
    echo "       Please ensure carla_official.sif is in the project root directory" >&2
    echo "       Or set PROJECT_ROOT environment variable to the correct location" >&2
    exit 1
fi

if [ ! -d "$AGENT_DIR_HOST" ]; then
    echo "FATAL: Agent directory not found at $AGENT_DIR_HOST" >&2
    echo "       Please ensure leaderboard/team_code exists in the project root" >&2
    exit 1
fi

# Agent Configuration
export AGENT_TYPE=${AGENT_TYPE:-interfuser}

CONSOLIDATED_AGENT_PATH="${AGENT_DIR_CONTAINER}/consolidated_agent.py"
AGENT_YAML_CONFIG_HOST="${AGENT_DIR_HOST}/configs/${AGENT_TYPE}.yaml"
AGENT_YAML_CONFIG_CONTAINER="${AGENT_DIR_CONTAINER}/configs/${AGENT_TYPE}.yaml"

if [ ! -f "$AGENT_YAML_CONFIG_HOST" ]; then
    echo "FATAL: Agent YAML config not found for AGENT_TYPE='${AGENT_TYPE}'." >&2
    echo "       Expected it at: ${AGENT_YAML_CONFIG_HOST}" >&2
    if [ -d "${AGENT_DIR_HOST}/configs" ]; then
        ls -la "${AGENT_DIR_HOST}/configs/"*.yaml 2>/dev/null | sed 's/^/         /'
    fi
    exit 1
fi

export TEAM_AGENT="$CONSOLIDATED_AGENT_PATH"
export TEAM_CONFIG="$AGENT_YAML_CONFIG_CONTAINER"
export CONSOLIDATED_AGENT="true"

echo "--- LAUNCH CONFIGURATION ---"
echo "AGENT_TYPE:          $AGENT_TYPE"
echo "CONSOLIDATED_AGENT:  $TEAM_AGENT"
echo "AGENT_CONFIG_YAML:   $TEAM_CONFIG"
echo "Using consolidated agent wrapper for automatic data saving"
echo "------------------------------------"

if [ ! -f "${AGENT_DIR_HOST}/consolidated_agent.py" ]; then
    echo "FATAL: Consolidated agent not found at ${AGENT_DIR_HOST}/consolidated_agent.py" >&2
    echo "Please ensure the enhanced consolidated agent is installed." >&2
    exit 1
fi

# General Configuration
WEATHER_NAMES=(
  ClearNoon CloudyNoon WetNoon WetCloudyNoon MidRainyNoon HardRainNoon SoftRainNoon
  ClearSunset CloudySunset WetSunset WetCloudySunset MidRainySunset HardRainSunset SoftRainSunset
)

BASE_RPC_PORT=2000
BASE_TM_PORT=8000

ROUTE_FILES=(
    "routes_town01_short.xml" "routes_town01_long.xml" 
    "routes_town02_short.xml" "routes_town02_long.xml"
    "routes_town03_short.xml" "routes_town03_long.xml" 
    "routes_town04_short.xml" "routes_town04_long.xml"
)

TOWN_NUMBERS=( "01" "01" "02" "02" "03" "03" "04" "04" )

WEATHER_START=${WEATHER_START:-1}
WEATHER_END=${WEATHER_END:-8}

# Main execution
echo "Starting distributed run using CONSOLIDATED AGENT for: $AGENT_TYPE"
echo "Data will be automatically saved by the consolidated agent wrapper"
echo "Dataset will be saved to: ${DATASET_DIR}/agent-${AGENT_TYPE}/"

# Generate launcher scripts for each GPU
for GPU_ID in {0..7}; do
    WEATHER_IDX=$((WEATHER_START + GPU_ID))
    if [ $WEATHER_IDX -gt $WEATHER_END ] || [ $WEATHER_IDX -ge ${#WEATHER_NAMES[@]} ]; then 
        continue
    fi

    ROUTE_IDX=$GPU_ID
    RPC_PORT=$((BASE_RPC_PORT + GPU_ID * 10))
    TM_PORT=$((BASE_TM_PORT + GPU_ID * 10))
    ROUTE_FILE=${ROUTE_FILES[$ROUTE_IDX]}
    TOWN_NUM=${TOWN_NUMBERS[$ROUTE_IDX]}

    cat > "$LOG_DIR/launch_gpu${GPU_ID}.sh" << 'LAUNCHER'
#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=__GPU_ID__

singularity exec --nv \
  --bind __PROJECT_ROOT__:__WORKSPACE_DIR__ \
  "__CARLA_SIF__" bash << 'CARLA_SCRIPT'
set -ex

export CARLA_ROOT=/home/carla

# PYTHONPATH setup
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/*:$CARLA_ROOT/PythonAPI/carla:$PYTHONPATH
export PYTHONPATH=__WORKSPACE_DIR__/scenario_runner:$PYTHONPATH
export PYTHONPATH=__WORKSPACE_DIR__/leaderboard:$PYTHONPATH
export PYTHONPATH=__WORKSPACE_DIR__:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=__WORKSPACE_DIR__/scenario_runner

export XDG_RUNTIME_DIR=/tmp/runtime-$USER-gpu__GPU_ID__
mkdir -p $XDG_RUNTIME_DIR

cd /home/carla

export SDL_AUDIODRIVER=dsp
export SDL_VIDEODRIVER=offscreen

# Launch CARLA server
./CarlaUE4.sh -carla-rpc-port=__RPC_PORT__ -carla-streaming-port=0 -nosound -quality-level=Epic &
CARLA_PID=$!
echo "[GPU __GPU_ID__] CARLA launched with PID $CARLA_PID. Waiting 60s for initialization..."
sleep 60

if ! ps -p $CARLA_PID >/dev/null; then
    echo "[GPU __GPU_ID__] FATAL: CARLA process failed to start." >&2
    exit 1
fi

# Prepare save paths
ROUTE_FILE_NAME="__ROUTE_FILE__"
ROUTE_NAME_NO_EXT="${ROUTE_FILE_NAME%.xml}"
SAVE_PATH="__WORKSPACE_DIR__/dataset/agent-__AGENT_TYPE__/weather-__WEATHER_IDX__/${ROUTE_NAME_NO_EXT}"
mkdir -p "$SAVE_PATH"

# Set environment variables
export ROUTES=__WORKSPACE_DIR__/leaderboard/data/training_routes/__ROUTE_FILE__
export SCENARIOS=__WORKSPACE_DIR__/leaderboard/data/scenarios/town__TOWN_NUM___all_scenarios.json
export TEAM_AGENT="__TEAM_AGENT__"
export TEAM_CONFIG="__TEAM_CONFIG__"
export CHECKPOINT_ENDPOINT="$SAVE_PATH/results.json"
export SAVE_PATH
export CONSOLIDATED_AGENT="true"

# Verify required files
if [ ! -f "$ROUTES" ]; then 
    echo "[GPU __GPU_ID__] FATAL: Route file not found: $ROUTES" >&2
    exit 1
fi
if [ ! -f "$SCENARIOS" ]; then 
    echo "[GPU __GPU_ID__] FATAL: Scenario file not found: $SCENARIOS" >&2
    exit 1
fi
if [ ! -f "$TEAM_AGENT" ]; then 
    echo "[GPU __GPU_ID__] FATAL: Consolidated agent not found: $TEAM_AGENT" >&2
    exit 1
fi
if [ ! -f "$TEAM_CONFIG" ]; then 
    echo "[GPU __GPU_ID__] FATAL: Agent config not found: $TEAM_CONFIG" >&2
    exit 1
fi

echo "[GPU __GPU_ID__] Starting leaderboard evaluation with consolidated agent..."
echo "  Agent wrapper: $TEAM_AGENT"
echo "  Agent config: $TEAM_CONFIG"
echo "  Save path: $SAVE_PATH"
echo "  Route: $ROUTES"
echo "  Scenarios: $SCENARIOS"

# Run leaderboard evaluation
python __WORKSPACE_DIR__/leaderboard/leaderboard/leaderboard_evaluator.py \
    --agent=$TEAM_AGENT \
    --agent-config=$TEAM_CONFIG \
    --routes=$ROUTES \
    --scenarios=$SCENARIOS \
    --checkpoint=$CHECKPOINT_ENDPOINT \
    --port=__RPC_PORT__ \
    --trafficManagerPort=__TM_PORT__ \
    --host=localhost \
    --timeout=60 \
    --debug=1 \
    --record="$SAVE_PATH"

echo "[GPU __GPU_ID__] Evaluation completed. Terminating CARLA..."
kill $CARLA_PID || true
wait $CARLA_PID 2>/dev/null || true
echo "[GPU __GPU_ID__] CARLA terminated."
CARLA_SCRIPT
LAUNCHER

    # Replace placeholders
    sed -i "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s|__WORKSPACE_DIR__|$WORKSPACE_DIR|g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s|__CARLA_SIF__|$CARLA_SIF|g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s/__GPU_ID__/$GPU_ID/g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s/__WEATHER_IDX__/$WEATHER_IDX/g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s/__ROUTE_FILE__/$ROUTE_FILE/g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s/__TOWN_NUM__/$TOWN_NUM/g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s/__RPC_PORT__/$RPC_PORT/g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s/__TM_PORT__/$TM_PORT/g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s|__AGENT_TYPE__|$AGENT_TYPE|g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s|__TEAM_AGENT__|$TEAM_AGENT|g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    sed -i "s|__TEAM_CONFIG__|$TEAM_CONFIG|g" "$LOG_DIR/launch_gpu${GPU_ID}.sh"
    
    chmod +x "$LOG_DIR/launch_gpu${GPU_ID}.sh"
done

# Launch all instances
echo -e "\n--- LAUNCHING ALL INSTANCES WITH CONSOLIDATED AGENT ---"
declare -a PIDS
for GPU_ID in {0..7}; do
    WEATHER_IDX=$((WEATHER_START + GPU_ID))
    if [ $WEATHER_IDX -gt $WEATHER_END ] || [ $WEATHER_IDX -ge ${#WEATHER_NAMES[@]} ]; then 
        continue
    fi
    
    "$LOG_DIR/launch_gpu${GPU_ID}.sh" \
        > "$LOG_DIR/gpu${GPU_ID}_w${WEATHER_IDX}.out" \
        2> "$LOG_DIR/gpu${GPU_ID}_w${WEATHER_IDX}.err" &
    PID=$!
    PIDS+=($PID)
    echo "  Launched GPU $GPU_ID (Weather $WEATHER_IDX) with PID: $PID"
    echo "  Using consolidated agent to wrap: $AGENT_TYPE"
    sleep 5
done

# Wait for completion
echo -e "\n--- WAITING FOR COMPLETION ---"
FAILED_COUNT=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU_ID=$i
    
    if wait $PID; then
        echo "SUCCESS: GPU $GPU_ID (PID $PID) completed."
    else
        EXIT_CODE=$?
        echo "FAILURE: GPU $GPU_ID (PID $PID) failed with exit code $EXIT_CODE."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        
        ERR_LOG="$LOG_DIR/gpu${GPU_ID}_w$((WEATHER_START + GPU_ID)).err"
        if [ -f "$ERR_LOG" ]; then
            echo "--- Last 10 lines from error log ($ERR_LOG): ---"
            tail -n 10 "$ERR_LOG" | sed 's/^/    /'
            echo "----------------------------------------------------"
        fi
    fi
done

# Summary
echo -e "\n=========================================="
echo "All instances completed."
if [ $FAILED_COUNT -eq 0 ]; then
    echo "Status: SUCCESS - All GPUs completed successfully"
else
    echo "Status: PARTIAL - $FAILED_COUNT GPU(s) failed"
fi
echo "Data has been saved to: ${DATASET_DIR}/agent-${AGENT_TYPE}/"
echo "Logs are available in: ${LOG_DIR}/"
echo "==========================================="

exit $FAILED_COUNT
