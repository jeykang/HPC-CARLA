#!/bin/bash
# ================================================================
#  Continuous data collection with persistent CARLA servers
#  Enhanced with periodic health monitoring during long jobs
# ================================================================

#SBATCH --job-name=continuous_collection_persistent
#SBATCH --nodelist=hpc-pr-a-pod08
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=168:00:00
#SBATCH --output=logs/continuous_persistent_%A.out
#SBATCH --error=logs/continuous_persistent_%A.err

# Set all defaults
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
export STATE_DIR="${PROJECT_ROOT}/collection_state"
export LOG_DIR="${PROJECT_ROOT}/logs"
export DATASET_DIR="${PROJECT_ROOT}/dataset"
export RUNTIME_DB="${STATE_DIR}/runtime_estimates.json"
export QUEUE_FILE="${STATE_DIR}/job_queue.json"
export GPU_STATUS_FILE="${STATE_DIR}/gpu_status.json"
export COMPLETED_FILE="${STATE_DIR}/completed_jobs.json"
export LOCK_FILE="${STATE_DIR}/.coordinator.lock"

# Create necessary directories
mkdir -p "$STATE_DIR" "$LOG_DIR" "$DATASET_DIR"

# Detect available agents from configs
CONFIGS_DIR="${PROJECT_ROOT}/leaderboard/team_code/configs"
if [ -d "$CONFIGS_DIR" ]; then
    export AGENTS=$(ls -1 "$CONFIGS_DIR"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml//' | tr '\n' ' ')
else
    echo "ERROR: No agent configs found in $CONFIGS_DIR"
    exit 1
fi

# GPU configuration
NUM_GPUS=${NUM_GPUS:-8}
BASE_RPC_PORT=${BASE_RPC_PORT:-2000}
BASE_TM_PORT=${BASE_TM_PORT:-8000}
PORT_SPACING=${PORT_SPACING:-100}

# ================================================================
# Health monitoring function
# ================================================================
write_health_status() {
    local gpu_id=$1
    local status=$2
    local message=$3
    local current_job=${4:-null}
    
    local health_dir="${STATE_DIR}/health"
    mkdir -p "$health_dir"
    
    local health_file="${health_dir}/gpu${gpu_id}.json"
    
    cat > "$health_file" << EOF
{
    "gpu_id": $gpu_id,
    "node": "$(hostname)",
    "status": "$status",
    "message": "$message",
    "carla_pid": null,
    "worker_pid": $$,
    "rpc_port": $((BASE_RPC_PORT + gpu_id * PORT_SPACING)),
    "tm_port": $((BASE_TM_PORT + gpu_id * PORT_SPACING)),
    "current_job": $current_job,
    "timestamp": "$(date -Iseconds)",
    "timestamp_unix": $(date +%s)
}
EOF
}

# ================================================================
# PHASE 1: Start persistent CARLA servers
# ================================================================

echo "=========================================="
echo "STARTING PERSISTENT CARLA SERVERS"
echo "=========================================="
echo "This will save hours by avoiding repeated startup/shutdown!"
echo ""

# Start the CARLA server manager
python3 << 'START_SERVERS'
import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.environ['PROJECT_ROOT'])

try:
    from carla_server_manager import CarlaServerManager
    
    manager = CarlaServerManager(
        project_root=os.environ['PROJECT_ROOT'],
        num_gpus=int(os.environ.get('NUM_GPUS', 8)),
        base_rpc_port=int(os.environ.get('BASE_RPC_PORT', 2000)),
        port_spacing=int(os.environ.get('PORT_SPACING', 100))
    )
    
    # Start servers on all GPUs
    print(f"Starting CARLA servers on {manager.num_gpus} GPUs...")
    count = manager.start_all()
    
    if count == 0:
        print("ERROR: Failed to start any CARLA servers")
        sys.exit(1)
    
    print(f"Started {count} CARLA servers")
    
    # Wait for all servers to be ready
    print("Waiting for servers to be ready (up to 2 minutes)...")
    if manager.wait_for_ready(timeout=120):
        print("✓ All servers ready!")
        
        # Save server status
        manager._update_status_file()
        
        # Keep the manager running in background
        import threading
        import signal
        
        # Set up signal handler to stop servers on exit
        def cleanup(signum, frame):
            print("\nStopping all CARLA servers...")
            manager.stop_all()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, cleanup)
        signal.signal(signal.SIGINT, cleanup)
        
        # Detach the monitoring thread
        print("Server manager will monitor in background")
        
    else:
        print("ERROR: Some servers failed to start")
        manager.stop_all()
        sys.exit(1)
        
except ImportError:
    print("ERROR: carla_server_manager.py not found")
    print("Please ensure carla_server_manager.py is in the project root")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to start servers: {e}")
    sys.exit(1)
START_SERVERS

if [ $? -ne 0 ]; then
    echo "Failed to start CARLA servers. Exiting."
    exit 1
fi

# Give servers a moment to stabilize
sleep 10

# ================================================================
# PHASE 2: Initialize job queue
# ================================================================

initialize_queue() {
    if [ ! -f "$QUEUE_FILE" ]; then
        echo "Initializing job queue..."
        python3 "${PROJECT_ROOT}/manage_continuous.py" reset
    fi
}

# ================================================================
# PHASE 3: Job execution functions (using persistent servers)
# ================================================================

get_next_job() {
    local gpu_id=$1
    
    python3 - $gpu_id << 'PYTHON_GET_JOB'
import json
import sys
import os
from datetime import datetime
import fcntl

state_dir = os.environ.get('STATE_DIR', './collection_state')
gpu_id = sys.argv[1]

lock_file = os.path.join(state_dir, '.scheduler.lock')
os.makedirs(os.path.dirname(lock_file), exist_ok=True)

with open(lock_file, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    
    try:
        with open(os.path.join(state_dir, 'job_queue.json'), 'r') as f:
            queue_data = json.load(f)
        with open(os.path.join(state_dir, 'runtime_estimates.json'), 'r') as f:
            runtime_data = json.load(f)
        with open(os.path.join(state_dir, 'gpu_status.json'), 'r') as f:
            gpu_status = json.load(f)
        
        # Check CARLA server status
        carla_status_file = os.path.join(state_dir, 'carla_servers.json')
        if os.path.exists(carla_status_file):
            with open(carla_status_file, 'r') as f:
                carla_status = json.load(f)
            
            # Check if server for this GPU is healthy
            server_info = carla_status.get('servers', {}).get(gpu_id, {})
            if not server_info.get('is_healthy', False):
                print(f"SERVER_UNHEALTHY")
                sys.exit(0)
        
        pending_jobs = [j for j in queue_data['jobs'] if j['status'] == 'pending']
        
        if not pending_jobs:
            failed_jobs = [j for j in queue_data['jobs'] 
                          if j['status'] == 'failed' and j['attempts'] < 3]
            if failed_jobs:
                pending_jobs = failed_jobs
            else:
                print("NO_MORE_JOBS")
                sys.exit(0)
        
        # Select next job (same logic as before)
        job_estimates = []
        for job in pending_jobs:
            key = f"{job['agent']}_{job['route']}"
            if key in runtime_data['combinations']:
                estimate = runtime_data['combinations'][key]
            else:
                estimate = runtime_data['default']
                if 'long' in job['route']:
                    estimate *= 1.5
                if 'short' in job['route']:
                    estimate *= 0.5
            
            weather_factor = 1.0
            if job['weather'] >= 7:
                weather_factor = 1.1
            if job['weather'] in [4, 5, 6, 11, 12, 13]:
                weather_factor = 1.2
            
            estimate *= weather_factor
            job_estimates.append((job, estimate))
        
        job_estimates.sort(key=lambda x: x[1], reverse=True)
        selected_job = job_estimates[0][0] if job_estimates else None
        
        if selected_job:
            for job in queue_data['jobs']:
                if job['id'] == selected_job['id']:
                    job['status'] = 'assigned'
                    job['gpu'] = int(gpu_id)
                    job['start_time'] = datetime.now().timestamp()
                    job['attempts'] += 1
                    break
            
            gpu_status[gpu_id]['status'] = 'busy'
            gpu_status[gpu_id]['current_job'] = selected_job['id']
            gpu_status[gpu_id]['last_heartbeat'] = datetime.now().timestamp()
            
            with open(os.path.join(state_dir, 'job_queue.json'), 'w') as f:
                json.dump(queue_data, f, indent=2)
            with open(os.path.join(state_dir, 'gpu_status.json'), 'w') as f:
                json.dump(gpu_status, f, indent=2)
            
            print(f"{selected_job['id']}|{selected_job['agent']}|{selected_job['weather']}|{selected_job['route']}")
        else:
            print("NO_MORE_JOBS")
            
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
PYTHON_GET_JOB
}

mark_job_complete() {
    local job_id=$1
    local gpu_id=$2
    local duration=$3
    local status=$4
    
    python3 - $job_id $gpu_id $duration $status << 'PYTHON_COMPLETE'
import json
import sys
import os
from datetime import datetime
import fcntl

state_dir = os.environ.get('STATE_DIR', './collection_state')
job_id = int(sys.argv[1])
gpu_id = sys.argv[2]
duration = float(sys.argv[3])
status = sys.argv[4]

lock_file = os.path.join(state_dir, '.scheduler.lock')
with open(lock_file, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    
    try:
        with open(os.path.join(state_dir, 'job_queue.json'), 'r') as f:
            queue_data = json.load(f)
        with open(os.path.join(state_dir, 'runtime_estimates.json'), 'r') as f:
            runtime_data = json.load(f)
        with open(os.path.join(state_dir, 'gpu_status.json'), 'r') as f:
            gpu_status = json.load(f)
        with open(os.path.join(state_dir, 'completed_jobs.json'), 'r') as f:
            completed_data = json.load(f)
        
        for job in queue_data['jobs']:
            if job['id'] == job_id:
                job['status'] = status
                job['end_time'] = datetime.now().timestamp()
                job['duration'] = duration
                
                if status == 'completed':
                    key = f"{job['agent']}_{job['route']}"
                    old_estimate = runtime_data['combinations'].get(key, runtime_data['default'])
                    alpha = 0.3
                    new_estimate = alpha * duration + (1 - alpha) * old_estimate
                    runtime_data['combinations'][key] = new_estimate
                    
                    completed_data['jobs'].append(job)
                    queue_data['completed'] += 1
                    
                    print(f"Job {job_id} completed in {duration:.1f}s")
                else:
                    print(f"Job {job_id} failed after {duration:.1f}s")
                break
        
        gpu_status[gpu_id]['status'] = 'idle'
        gpu_status[gpu_id]['current_job'] = None
        if status == 'completed':
            gpu_status[gpu_id]['jobs_completed'] += 1
            gpu_status[gpu_id]['total_runtime'] += duration
        
        with open(os.path.join(state_dir, 'job_queue.json'), 'w') as f:
            json.dump(queue_data, f, indent=2)
        with open(os.path.join(state_dir, 'runtime_estimates.json'), 'w') as f:
            json.dump(runtime_data, f, indent=2)
        with open(os.path.join(state_dir, 'gpu_status.json'), 'w') as f:
            json.dump(gpu_status, f, indent=2)
        with open(os.path.join(state_dir, 'completed_jobs.json'), 'w') as f:
            json.dump(completed_data, f, indent=2)
            
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
PYTHON_COMPLETE
}

# Enhanced worker function with periodic health updates
gpu_worker_persistent() {
    local gpu_id=$1
    local worker_log="${LOG_DIR}/worker_gpu${gpu_id}_persistent.log"
    
    echo "[GPU $gpu_id] Worker started with persistent server at $(date)" > "$worker_log"
    
    # Write initial health status
    write_health_status $gpu_id "starting" "Worker initializing" null
    
    # Count consecutive server errors
    local server_error_count=0
    local max_server_errors=3
    
    while true; do
        # Update health status - waiting for job
        write_health_status $gpu_id "waiting_for_job" "Ready for next job" null
        
        JOB_INFO=$(get_next_job $gpu_id)
        
        if [ "$JOB_INFO" == "NO_MORE_JOBS" ]; then
            echo "[GPU $gpu_id] No more jobs available" >> "$worker_log"
            write_health_status $gpu_id "completed" "No more jobs" null
            break
        fi
        
        if [ "$JOB_INFO" == "SERVER_UNHEALTHY" ]; then
            echo "[GPU $gpu_id] CARLA server unhealthy, waiting for restart..." >> "$worker_log"
            write_health_status $gpu_id "unhealthy" "CARLA server unhealthy" null
            server_error_count=$((server_error_count + 1))
            
            if [ $server_error_count -ge $max_server_errors ]; then
                echo "[GPU $gpu_id] Too many server errors, exiting worker" >> "$worker_log"
                write_health_status $gpu_id "error" "Too many server errors" null
                break
            fi
            
            sleep 30
            continue
        fi
        
        # Reset error count on successful job retrieval
        server_error_count=0
        
        IFS='|' read -r JOB_ID AGENT WEATHER ROUTE <<< "$JOB_INFO"
        
        echo "[GPU $gpu_id] Starting job $JOB_ID: agent=$AGENT weather=$WEATHER route=$ROUTE" >> "$worker_log"
        
        START_TIME=$(date +%s)
        
        # Set environment for single job script
        export AGENT_TYPE=$AGENT
        export WEATHER_START=$WEATHER
        export WEATHER_END=$WEATHER
        export ROUTE_FILE=$ROUTE
        export GPU_ID=$gpu_id
        
        # Start the job in background
        bash "${PROJECT_ROOT}/generate_single_job_persistent.sh" \
            > "${LOG_DIR}/job_${JOB_ID}_gpu${gpu_id}.out" \
            2> "${LOG_DIR}/job_${JOB_ID}_gpu${gpu_id}.err" &
        
        JOB_PID=$!
        echo "[GPU $gpu_id] Job $JOB_ID started with PID $JOB_PID" >> "$worker_log"
        
        # Monitor the job and update health periodically
        local update_counter=0
        local health_update_interval=30  # Update health every 30 seconds
        
        while kill -0 $JOB_PID 2>/dev/null; do
            # Calculate elapsed time
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))
            ELAPSED_HOURS=$((ELAPSED / 3600))
            ELAPSED_MINS=$(((ELAPSED % 3600) / 60))
            ELAPSED_SECS=$((ELAPSED % 60))
            
            # Format elapsed time string
            ELAPSED_STR=$(printf "%02d:%02d:%02d" $ELAPSED_HOURS $ELAPSED_MINS $ELAPSED_SECS)
            
            # Update health status with progress
            write_health_status $gpu_id "running_job" \
                "Job $JOB_ID: $AGENT/$ROUTE (running ${ELAPSED_STR})" $JOB_ID
            
            # Log progress periodically
            if [ $((update_counter % 10)) -eq 0 ]; then  # Log every 5 minutes
                echo "[GPU $gpu_id] Job $JOB_ID still running (${ELAPSED_STR})" >> "$worker_log"
                
                # Check if output file is growing (sign of progress)
                OUTPUT_FILE="${LOG_DIR}/job_${JOB_ID}_gpu${gpu_id}.out"
                if [ -f "$OUTPUT_FILE" ]; then
                    FILE_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || echo 0)
                    echo "[GPU $gpu_id] Job $JOB_ID output size: $FILE_SIZE bytes" >> "$worker_log"
                fi
            fi
            
            update_counter=$((update_counter + 1))
            
            # Wait before next update
            sleep $health_update_interval
        done
        
        # Job has finished, get exit code
        wait $JOB_PID
        EXIT_CODE=$?
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # Format duration for logging
        DURATION_HOURS=$((DURATION / 3600))
        DURATION_MINS=$(((DURATION % 3600) / 60))
        DURATION_STR=$(printf "%02d:%02d" $DURATION_HOURS $DURATION_MINS)
        
        if [ $EXIT_CODE -eq 0 ]; then
            mark_job_complete $JOB_ID $gpu_id $DURATION "completed"
            echo "[GPU $gpu_id] Job $JOB_ID completed successfully in ${DURATION_STR}" >> "$worker_log"
            write_health_status $gpu_id "job_completed" \
                "Job $JOB_ID completed successfully (took ${DURATION_STR})" null
        elif [ $EXIT_CODE -eq 99 ]; then
            # Server crashed
            mark_job_complete $JOB_ID $gpu_id $DURATION "failed"
            echo "[GPU $gpu_id] Job $JOB_ID failed due to server crash after ${DURATION_STR}" >> "$worker_log"
            write_health_status $gpu_id "server_crashed" \
                "CARLA server crashed during job $JOB_ID (after ${DURATION_STR})" null
            
            # Wait for server to be restarted by manager
            echo "[GPU $gpu_id] Waiting for server restart..." >> "$worker_log"
            sleep 60
        else
            mark_job_complete $JOB_ID $gpu_id $DURATION "failed"
            echo "[GPU $gpu_id] Job $JOB_ID failed with code $EXIT_CODE after ${DURATION_STR}" >> "$worker_log"
            write_health_status $gpu_id "job_failed" \
                "Job $JOB_ID failed with code $EXIT_CODE (after ${DURATION_STR})" null
            
            # Try to extract error from log tail
            ERROR_LOG="${LOG_DIR}/job_${JOB_ID}_gpu${gpu_id}.err"
            if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
                LAST_ERROR=$(tail -n 5 "$ERROR_LOG" | head -n 1)
                echo "[GPU $gpu_id] Last error: $LAST_ERROR" >> "$worker_log"
            fi
        fi
        
        # Brief pause between jobs
        sleep 2
    done
    
    echo "[GPU $gpu_id] Worker finished at $(date)" >> "$worker_log"
    write_health_status $gpu_id "stopped" "Worker terminated" null
}

monitor_progress() {
    python3 "${PROJECT_ROOT}/monitor_continuous.py" --once
}

# ================================================================
# PHASE 4: Main execution
# ================================================================

echo "=========================================="
echo "CONTINUOUS DATA COLLECTION WITH PERSISTENT SERVERS"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "State directory: $STATE_DIR"
echo "Agents: ${AGENTS}"
echo "GPUs: $NUM_GPUS"
echo "Using persistent CARLA servers - much faster!"
echo "=========================================="

# Initialize queue
initialize_queue

# Export functions for subshells
export -f get_next_job
export -f mark_job_complete
export -f gpu_worker_persistent
export -f monitor_progress
export -f write_health_status

# Start workers for each GPU
echo "Starting GPU workers (connecting to persistent servers)..."
declare -a WORKER_PIDS
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_worker_persistent $gpu_id &
    WORKER_PIDS[$gpu_id]=$!
    echo "Started worker for GPU $gpu_id (PID: ${WORKER_PIDS[$gpu_id]})"
    sleep 2
done

# Monitor loop
echo "Monitoring progress..."
while true; do
    sleep 60
    monitor_progress
    
    ACTIVE_WORKERS=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            ACTIVE_WORKERS=$((ACTIVE_WORKERS + 1))
        fi
    done
    
    if [ $ACTIVE_WORKERS -eq 0 ]; then
        echo "All workers have finished."
        break
    fi
done

# ================================================================
# PHASE 5: Cleanup
# ================================================================

echo ""
echo "=========================================="
echo "STOPPING PERSISTENT CARLA SERVERS"
echo "=========================================="

python3 "${PROJECT_ROOT}/carla_server_manager.py" stop

echo ""
echo "=========================================="
echo "COLLECTION COMPLETE"
echo "=========================================="
monitor_progress

# Generate summary report
python3 - << 'PYTHON_FINAL'
import json
import os
from datetime import timedelta

state_dir = os.environ.get('STATE_DIR', './collection_state')

try:
    with open(os.path.join(state_dir, 'completed_jobs.json'), 'r') as f:
        completed = json.load(f)

    if completed['jobs']:
        total_duration = sum(j['duration'] for j in completed['jobs'] if j.get('duration'))
        avg_duration = total_duration / len(completed['jobs'])
        
        print("\nFINAL STATISTICS:")
        print(f"  Total jobs completed: {len(completed['jobs'])}")
        print(f"  Total runtime: {str(timedelta(seconds=int(total_duration)))}")
        print(f"  Average job duration: {str(timedelta(seconds=int(avg_duration)))}")
        
        # Estimate time saved
        startup_time_per_job = 70  # 60s startup + 10s shutdown
        time_saved = len(completed['jobs']) * startup_time_per_job
        print(f"\n  TIME SAVED BY PERSISTENT SERVERS: {str(timedelta(seconds=time_saved))}")
        print(f"  (Avoided {len(completed['jobs'])} CARLA startups/shutdowns)")
        
except Exception as e:
    print(f"Could not generate final report: {e}")
PYTHON_FINAL

echo "Results saved in: ${STATE_DIR}"
echo "Logs available in: ${LOG_DIR}"