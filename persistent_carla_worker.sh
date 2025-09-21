#!/bin/bash
# ================================================================
#  Persistent CARLA Worker - Maintains one CARLA instance per GPU
#  Processes multiple jobs without restarting CARLA
#  Self-contained for SLURM cluster execution
# ================================================================

set -e

# Parameters
GPU_ID=${1:-0}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
WORKSPACE_DIR="/workspace"
CARLA_SIF="${PROJECT_ROOT}/carla_official.sif"
LOG_DIR="${PROJECT_ROOT}/logs"
STATE_DIR="${PROJECT_ROOT}/collection_state"

# Port configuration (unique per GPU)
RPC_PORT=$((2000 + GPU_ID * 10))
TM_PORT=$((8000 + GPU_ID * 10))

# Create state directories
HEALTH_DIR="${STATE_DIR}/health"
mkdir -p "$HEALTH_DIR" "$LOG_DIR"

# Health status file (readable from login node)
HEALTH_FILE="${HEALTH_DIR}/gpu${GPU_ID}.json"

# Logging
WORKER_LOG="${LOG_DIR}/persistent_worker_gpu${GPU_ID}.log"
exec 1> >(tee -a "$WORKER_LOG")
exec 2>&1

echo "[GPU $GPU_ID] Starting persistent CARLA worker at $(date)"
echo "[GPU $GPU_ID] Node: $(hostname)"
echo "[GPU $GPU_ID] Ports: RPC=$RPC_PORT, TM=$TM_PORT"
echo "[GPU $GPU_ID] CARLA image: $CARLA_SIF"
echo "[GPU $GPU_ID] Workspace: $WORKSPACE_DIR"

# Verify CARLA singularity image exists
if [ ! -f "$CARLA_SIF" ]; then
    echo "[GPU $GPU_ID] ERROR: CARLA singularity image not found at $CARLA_SIF"
    write_health_status "error" "CARLA image not found" null
    exit 1
fi

# Write initial health status
write_health_status() {
    local status=$1
    local message=$2
    local carla_pid=${3:-null}
    
    cat > "$HEALTH_FILE" << EOF
{
    "gpu_id": $GPU_ID,
    "node": "$(hostname)",
    "status": "$status",
    "message": "$message",
    "carla_pid": $carla_pid,
    "worker_pid": $,
    "rpc_port": $RPC_PORT,
    "tm_port": $TM_PORT,
    "timestamp": "$(date -Iseconds)",
    "timestamp_unix": $(date +%s)
}
EOF
}

# Cleanup function
cleanup() {
    echo "[GPU $GPU_ID] Cleaning up..."
    
    # Kill CARLA if we know its PID
    if [ -n "$CARLA_PID" ]; then
        if kill -0 "$CARLA_PID" 2>/dev/null; then
            echo "[GPU $GPU_ID] Terminating CARLA (PID: $CARLA_PID)..."
            kill "$CARLA_PID" 2>/dev/null || true
            sleep 5
            # Force kill if still running
            if kill -0 "$CARLA_PID" 2>/dev/null; then
                echo "[GPU $GPU_ID] Force killing CARLA..."
                kill -9 "$CARLA_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Kill any process on our RPC port (cleanup zombies)
    echo "[GPU $GPU_ID] Cleaning up port $RPC_PORT..."
    fuser -k ${RPC_PORT}/tcp 2>/dev/null || true
    
    # Also check for any lingering singularity processes for this GPU
    pkill -f "singularity.*gpu${GPU_ID}" 2>/dev/null || true
    
    # Mark as stopped in health file
    write_health_status "stopped" "Worker terminated" null
    
    echo "[GPU $GPU_ID] Cleanup complete"
}

# Set up trap for cleanup on exit
trap cleanup EXIT INT TERM

# Initialize health status
write_health_status "starting" "Worker initializing" null

# Function to launch CARLA
launch_carla() {
    echo "[GPU $GPU_ID] Launching CARLA server..."
    
    # Kill any existing process on this port using fuser (standard tool)
    fuser -k ${RPC_PORT}/tcp 2>/dev/null || true
    sleep 2
    
    # Launch CARLA in background using singularity
    singularity exec --nv \
        --bind "${PROJECT_ROOT}:${WORKSPACE_DIR}" \
        "$CARLA_SIF" bash -c "
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        export XDG_RUNTIME_DIR=/tmp/runtime-\$USER-gpu${GPU_ID}-$
        mkdir -p \$XDG_RUNTIME_DIR
        cd /home/carla
        export SDL_AUDIODRIVER=dsp
        export SDL_VIDEODRIVER=offscreen
        ./CarlaUE4.sh -carla-rpc-port=$RPC_PORT -carla-streaming-port=0 -nosound -quality-level=Epic
    " &
    
    CARLA_PID=$!
    
    # Wait for CARLA to initialize
    echo "[GPU $GPU_ID] Waiting for CARLA to initialize (PID: $CARLA_PID)..."
    write_health_status "starting_carla" "Waiting for CARLA to initialize" $CARLA_PID
    
    local wait_time=0
    local max_wait=120
    
    while [ $wait_time -lt $max_wait ]; do
        if ! kill -0 "$CARLA_PID" 2>/dev/null; then
            echo "[GPU $GPU_ID] ERROR: CARLA process died during initialization"
            write_health_status "error" "CARLA died during initialization" null
            return 1
        fi
        
        # Check if port is listening using netstat (more standard than lsof)
        if netstat -tln 2>/dev/null | grep -q ":$RPC_PORT "; then
            echo "[GPU $GPU_ID] CARLA is ready on port $RPC_PORT"
            write_health_status "ready" "CARLA running and ready" $CARLA_PID
            return 0
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
        echo "[GPU $GPU_ID] Still waiting... ($wait_time/$max_wait seconds)"
    done
    
    echo "[GPU $GPU_ID] ERROR: CARLA failed to start within $max_wait seconds"
    write_health_status "error" "CARLA failed to start" null
    return 1
}

# Function to check if CARLA is healthy (simplified without Python carla module)
check_carla_health() {
    # Check if process exists
    if [ -z "$CARLA_PID" ] || ! kill -0 "$CARLA_PID" 2>/dev/null; then
        echo "[GPU $GPU_ID] CARLA process is dead"
        write_health_status "unhealthy" "CARLA process dead" null
        return 1
    fi
    
    # Check if port is still listening
    if ! netstat -tln 2>/dev/null | grep -q ":$RPC_PORT "; then
        echo "[GPU $GPU_ID] CARLA port $RPC_PORT is not listening"
        write_health_status "unhealthy" "Port not listening" $CARLA_PID
        return 1
    fi
    
    # Simple TCP connection test using bash
    timeout 2 bash -c "echo > /dev/tcp/localhost/$RPC_PORT" 2>/dev/null
    if [ $? -eq 0 ]; then
        write_health_status "healthy" "CARLA responding" $CARLA_PID
        return 0
    else
        echo "[GPU $GPU_ID] CARLA not responding to connection test"
        write_health_status "unhealthy" "Not responding" $CARLA_PID
        return 1
    fi
}

# Function to reset CARLA world (simplified without Python carla module)
reset_carla_world() {
    echo "[GPU $GPU_ID] Resetting CARLA world..."
    
    # Since we can't use the carla Python module, we rely on the leaderboard
    # to clean up after itself. We just wait a bit for any cleanup to happen.
    # The leaderboard evaluator should destroy actors when it finishes.
    
    # Give CARLA time to clean up after the previous run
    sleep 5
    
    # Update health status
    write_health_status "world_reset" "Ready for next job" $CARLA_PID
    
    echo "[GPU $GPU_ID] World reset complete (cleanup pause)"
}

# Function to run a single evaluation job
run_evaluation_job() {

    local JOB_ID=$1
    local AGENT_TYPE=$2
    local WEATHER_IDX=$3
    local ROUTE_FILE=$4

    ROUTE_NAME_NO_EXT="${ROUTE_FILE%.xml}"
    ROUTE_NAME_NO_EXT=$(basename "$ROUTE_NAME_NO_EXT")
    SAVE_PATH="${WORKSPACE_DIR}/dataset/agent-${AGENT_TYPE}/weather-${WEATHER_IDX}/${ROUTE_NAME_NO_EXT}"
    export SAVE_PATH
    mkdir -p "$SAVE_PATH"

    # Construct a separate recorder file name
    RECORD_PATH="${SAVE_PATH}/${ROUTE_NAME_NO_EXT}_${WEATHER_IDX}_${JOB_ID}.log"

    export RECORD_PATH

    
    echo "[GPU $GPU_ID] Running job $JOB_ID: agent=$AGENT_TYPE weather=$WEATHER_IDX route=$ROUTE_FILE"
    
    # Extract town number from route file
    local TOWN_NUM=$(echo "$ROUTE_FILE" | sed -n 's/.*routes_town\([0-9]\+\).*/\1/p')
    if [ ${#TOWN_NUM} -eq 1 ]; then
        TOWN_NUM="0${TOWN_NUM}"
    fi
    
    # Validate files
    local ROUTES_HOST="${PROJECT_ROOT}/leaderboard/data/training_routes/${ROUTE_FILE}"
    local SCENARIOS_HOST="${PROJECT_ROOT}/leaderboard/data/scenarios/town${TOWN_NUM}_all_scenarios.json"
    local AGENT_CONFIG_HOST="${PROJECT_ROOT}/leaderboard/team_code/configs/${AGENT_TYPE}.yaml"
    
    if [ ! -f "$ROUTES_HOST" ] || [ ! -f "$SCENARIOS_HOST" ] || [ ! -f "$AGENT_CONFIG_HOST" ]; then
        echo "[GPU $GPU_ID] ERROR: Required files not found for job $JOB_ID"
        return 1
    fi
    
    # Run evaluation in singularity (without launching CARLA - it's already running)
    singularity exec --nv \
        --bind "${PROJECT_ROOT}:${WORKSPACE_DIR}" \
        "$CARLA_SIF" bash << EVAL_SCRIPT
set -e

export CUDA_VISIBLE_DEVICES=$GPU_ID
export CARLA_ROOT=/home/carla

# PYTHONPATH setup
export PYTHONPATH=\$CARLA_ROOT/PythonAPI/carla/dist/carla-*-py*.egg:\$CARLA_ROOT/PythonAPI/carla:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}/scenario_runner:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}/leaderboard:\$PYTHONPATH
export PYTHONPATH=${WORKSPACE_DIR}:\$PYTHONPATH
export SCENARIO_RUNNER_ROOT=${WORKSPACE_DIR}/scenario_runner

# Use SAVE_PATH and RECORD_PATH computed and exported by the outer script.
export SAVE_PATH
export RECORD_PATH

# Set environment variables
export ROUTES=${WORKSPACE_DIR}/leaderboard/data/training_routes/$ROUTE_FILE
export SCENARIOS=${WORKSPACE_DIR}/leaderboard/data/scenarios/town${TOWN_NUM}_all_scenarios.json
export TEAM_AGENT="${WORKSPACE_DIR}/leaderboard/team_code/consolidated_agent.py"
export TEAM_CONFIG="${WORKSPACE_DIR}/leaderboard/team_code/configs/${AGENT_TYPE}.yaml"
export CHECKPOINT_ENDPOINT="\$SAVE_PATH/results.json"
export SAVE_PATH
export CONSOLIDATED_AGENT="true"

echo "[GPU $GPU_ID] Starting evaluation for job $JOB_ID..."
echo "  Save path: \$SAVE_PATH"
echo "  Record path: \$RECORD_PATH"

# Change to workspace directory for relative imports
cd ${WORKSPACE_DIR}

# Run leaderboard evaluation (CARLA is already running on port $RPC_PORT)
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
    --debug=0 \
    --record="\$RECORD_PATH"

EVAL_SCRIPT
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[GPU $GPU_ID] Job $JOB_ID completed successfully"
    else
        echo "[GPU $GPU_ID] Job $JOB_ID failed with exit code $EXIT_CODE"
    fi
    
    return $EXIT_CODE
}

# Source the job management functions
# These should be provided via environment when launched from continuous_collection_persistent.sh
if [ -z "$(type -t get_next_job)" ]; then
    echo "[GPU $GPU_ID] Warning: Job management functions not found in environment"
    echo "[GPU $GPU_ID] Attempting to source from continuous_collection.sh"
    
    if [ -f "${PROJECT_ROOT}/continuous_collection.sh" ]; then
        source "${PROJECT_ROOT}/continuous_collection.sh" 2>/dev/null || true
    elif [ -f "${PROJECT_ROOT}/continuous_collection_persistent.sh" ]; then
        # Try to extract just the functions we need
        eval "$(grep -A 50 '^get_next_job()' ${PROJECT_ROOT}/continuous_collection_persistent.sh)"
        eval "$(grep -A 50 '^mark_job_complete()' ${PROJECT_ROOT}/continuous_collection_persistent.sh)"
    else
        echo "[GPU $GPU_ID] FATAL: Cannot find job management functions"
        exit 1
    fi
fi

# Main worker loop
echo "[GPU $GPU_ID] Starting main worker loop"

# Global CARLA PID variable
CARLA_PID=""

# Start CARLA once
if ! launch_carla; then
    echo "[GPU $GPU_ID] Failed to launch CARLA, exiting"
    exit 1
fi

# Track consecutive failures
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=3

# Process jobs
while true; do
    # Update health status
    write_health_status "waiting_for_job" "Ready for next job" $CARLA_PID
    
    # Get next job (using existing function from continuous_collection.sh)
    JOB_INFO=$(get_next_job $GPU_ID 2>/dev/null || echo "NO_MORE_JOBS")
    
    if [ "$JOB_INFO" == "NO_MORE_JOBS" ]; then
        echo "[GPU $GPU_ID] No more jobs available"
        write_health_status "completed" "No more jobs" $CARLA_PID
        break
    fi
    
    # Parse job info
    IFS='|' read -r JOB_ID AGENT WEATHER ROUTE <<< "$JOB_INFO"
    
    write_health_status "running_job" "Processing job $JOB_ID: $AGENT/$ROUTE" $CARLA_PID
    
    # Check CARLA health before running job
    if ! check_carla_health; then
        echo "[GPU $GPU_ID] CARLA is unhealthy, restarting..."
        
        # Kill existing CARLA
        if [ -n "$CARLA_PID" ]; then
            kill "$CARLA_PID" 2>/dev/null || true
            sleep 5
            kill -9 "$CARLA_PID" 2>/dev/null || true
        fi
        
        # Clean up port
        fuser -k ${RPC_PORT}/tcp 2>/dev/null || true
        sleep 5
        
        # Relaunch CARLA
        if ! launch_carla; then
            echo "[GPU $GPU_ID] Failed to restart CARLA, marking job as failed"
            mark_job_complete $JOB_ID $GPU_ID 0 "failed" 2>/dev/null
            write_health_status "error" "Cannot restart CARLA" null
            exit 1
        fi
    fi
    
    # Reset world between jobs
    reset_carla_world
    
    # Run the evaluation
    START_TIME=$(date +%s)
    
    if run_evaluation_job $JOB_ID $AGENT $WEATHER $ROUTE; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        mark_job_complete $JOB_ID $GPU_ID $DURATION "completed" 2>/dev/null
        CONSECUTIVE_FAILURES=0
        write_health_status "job_completed" "Job $JOB_ID completed successfully" $CARLA_PID
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        mark_job_complete $JOB_ID $GPU_ID $DURATION "failed" 2>/dev/null
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        write_health_status "job_failed" "Job $JOB_ID failed" $CARLA_PID
        
        # If too many consecutive failures, restart CARLA
        if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
            echo "[GPU $GPU_ID] Too many consecutive failures, restarting CARLA..."
            
            # Kill and restart CARLA
            if [ -n "$CARLA_PID" ]; then
                kill "$CARLA_PID" 2>/dev/null || true
                sleep 5
                kill -9 "$CARLA_PID" 2>/dev/null || true
            fi
            
            fuser -k ${RPC_PORT}/tcp 2>/dev/null || true
            sleep 10
            
            if ! launch_carla; then
                echo "[GPU $GPU_ID] Failed to restart CARLA after failures, exiting"
                write_health_status "error" "Cannot recover from failures" null
                exit 1
            fi
            
            CONSECUTIVE_FAILURES=0
        fi
    fi
    
    # Brief pause between jobs
    sleep 5
done

echo "[GPU $GPU_ID] Worker finished at $(date)"
write_health_status "finished" "All jobs completed" $CARLA_PID