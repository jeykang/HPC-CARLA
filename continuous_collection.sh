#!/bin/bash
# ================================================================
#  Continuous data collection with intelligent GPU scheduling
#  Fixed version with proper defaults and function exports
# ================================================================

#SBATCH --job-name=continuous_collection
#SBATCH --nodelist=hpc-pr-a-pod10,hpc-pr-a-pod11
#SBATCH --gres=gpu:16
#SBATCH --nodes=2
#SBATCH --time=168:00:00
#SBATCH --output=logs/continuous_%A.out
#SBATCH --error=logs/continuous_%A.err

# Set all defaults based on original pattern
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
    echo "Using default agent list"
    export AGENTS="interfuser"
fi

# Weather and route configuration
export WEATHER_INDICES="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
export ROUTE_FILES="routes_town01_short.xml routes_town01_long.xml routes_town02_short.xml routes_town02_long.xml routes_town03_short.xml routes_town03_long.xml routes_town04_short.xml routes_town04_long.xml routes_town05_short.xml routes_town05_long.xml routes_town06_short.xml routes_town06_long.xml routes_town07_short.xml routes_town07_long.xml"

# GPU configuration
NUM_GPUS=${NUM_GPUS:-8}
BASE_RPC_PORT=${BASE_RPC_PORT:-2000}
BASE_TM_PORT=${BASE_TM_PORT:-8000}

# Node info
NODE_NAME=${SLURMD_NODENAME:-$(hostname)}

# Initialize job queue if not exists
initialize_queue() {
    if [ ! -f "$QUEUE_FILE" ]; then
        echo "Initializing job queue..."
        python3 - << 'PYTHON_INIT'
import json
import sys
import os

state_dir = os.environ.get('STATE_DIR', './collection_state')
agents = os.environ.get('AGENTS', 'interfuser').split()
weather_indices = list(range(15))
route_files = os.environ.get('ROUTE_FILES', 'routes_town01_short.xml').split()

os.makedirs(state_dir, exist_ok=True)

jobs = []
job_id = 0
for agent in agents:
    for weather in weather_indices:
        for route in route_files:
            jobs.append({
                'id': job_id,
                'agent': agent,
                'weather': weather,
                'route': route,
                'status': 'pending',
                'attempts': 0,
                'gpu': None,
                'start_time': None,
                'end_time': None,
                'duration': None
            })
            job_id += 1

queue_file = os.path.join(state_dir, 'job_queue.json')
with open(queue_file, 'w') as f:
    json.dump({'jobs': jobs, 'total': len(jobs), 'completed': 0}, f, indent=2)

print(f"Initialized queue with {len(jobs)} jobs")

gpu_status = {}
for i in range(int(os.environ.get('NUM_GPUS', 8))):
    gpu_status[str(i)] = {
        'status': 'idle',
        'current_job': None,
        'jobs_completed': 0,
        'total_runtime': 0,
        'last_heartbeat': None
    }

gpu_status_file = os.path.join(state_dir, 'gpu_status.json')
with open(gpu_status_file, 'w') as f:
    json.dump(gpu_status, f, indent=2)

runtime_estimates = {
    'default': 3600,
    'combinations': {}
}

for agent in agents:
    for route in route_files:
        key = f"{agent}_{route}"
        if 'short' in route:
            runtime_estimates['combinations'][key] = 1800
        else:
            runtime_estimates['combinations'][key] = 5400

runtime_file = os.path.join(state_dir, 'runtime_estimates.json')
with open(runtime_file, 'w') as f:
    json.dump(runtime_estimates, f, indent=2)

completed_file = os.path.join(state_dir, 'completed_jobs.json')
with open(completed_file, 'w') as f:
    json.dump({'jobs': []}, f, indent=2)
PYTHON_INIT
    fi
}

# Get next job with intelligent scheduling
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
        
        pending_jobs = [j for j in queue_data['jobs'] if j['status'] == 'pending']
        
        if not pending_jobs:
            failed_jobs = [j for j in queue_data['jobs'] 
                          if j['status'] == 'failed' and j['attempts'] < 3]
            if failed_jobs:
                pending_jobs = failed_jobs
            else:
                print("NO_MORE_JOBS")
                sys.exit(0)
        
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

# Mark job as complete and update runtime estimates
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
                    
                    print(f"Job {job_id} completed in {duration:.1f}s (estimate was {old_estimate:.1f}s)")
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

# Worker function for each GPU
gpu_worker() {
    local gpu_id=$1
    local gpu_log="${LOG_DIR}/node_${NODE_NAME}_gpu${gpu_id}_consolidated.log"

    
    # Initialize GPU log file with header
    {
        echo "=========================================="
        echo "GPU $gpu_id Worker Started: $(date)"
        echo "Node: $(hostname)"
        echo "=========================================="
        echo ""
    } > "$gpu_log"
    
    while true; do
        JOB_INFO=$(get_next_job $gpu_id)
        
        if [ "$JOB_INFO" == "NO_MORE_JOBS" ]; then
            echo "[GPU $gpu_id] No more jobs available at $(date)" >> "$gpu_log"
            break
        fi
        
        IFS='|' read -r JOB_ID AGENT WEATHER ROUTE <<< "$JOB_INFO"
        
        # Add job header to consolidated log
        {
            echo ""
            echo "=========================================="
            echo "[GPU $gpu_id] Starting Job #$JOB_ID"
            echo "Time: $(date)"
            echo "Agent: $AGENT"
            echo "Weather: $WEATHER"
            echo "Route: $ROUTE"
            echo "=========================================="
        } >> "$gpu_log"
        
        START_TIME=$(date +%s)
        
        # Set environment for single job script
        export AGENT_TYPE=$AGENT
        export WEATHER_START=$WEATHER
        export WEATHER_END=$WEATHER
        export ROUTE_FILE=$ROUTE
        export GPU_ID=$gpu_id
        
        # Check if single job script exists
        if [ ! -f "${PROJECT_ROOT}/generate_single_job.sh" ]; then
            echo "[GPU $gpu_id] ERROR: generate_single_job.sh not found" >> "$gpu_log"
            mark_job_complete $JOB_ID $gpu_id 0 "failed"
            continue
        fi
        
        # Run the job and append output to consolidated log
        # Add a subshell to capture both stdout and stderr together
        {
            bash "${PROJECT_ROOT}/generate_single_job.sh" 2>&1
            echo "Exit code: $?"
        } >> "$gpu_log"
        
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # Add job footer to consolidated log
        {
            echo "----------------------------------------"
            echo "[GPU $gpu_id] Job #$JOB_ID completed"
            echo "Duration: ${DURATION} seconds"
            echo "Status: $([ $EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
            echo "Time: $(date)"
            echo "=========================================="
            echo ""
        } >> "$gpu_log"
        
        if [ $EXIT_CODE -eq 0 ]; then
            mark_job_complete $JOB_ID $gpu_id $DURATION "completed"
        else
            mark_job_complete $JOB_ID $gpu_id $DURATION "failed"
        fi
        
        sleep 5
    done
    
    # Add worker completion footer
    {
        echo ""
        echo "=========================================="
        echo "GPU $gpu_id Worker Finished: $(date)"
        echo "=========================================="
    } >> "$gpu_log"
}

# Status monitor
monitor_progress() {
    python3 - << 'PYTHON_MONITOR'
import json
import os
from datetime import datetime, timedelta

state_dir = os.environ.get('STATE_DIR', './collection_state')

try:
    with open(os.path.join(state_dir, 'job_queue.json'), 'r') as f:
        queue_data = json.load(f)
    with open(os.path.join(state_dir, 'gpu_status.json'), 'r') as f:
        gpu_status = json.load(f)
    with open(os.path.join(state_dir, 'runtime_estimates.json'), 'r') as f:
        runtime_data = json.load(f)
    
    total = queue_data['total']
    completed = queue_data['completed']
    pending = sum(1 for j in queue_data['jobs'] if j['status'] == 'pending')
    running = sum(1 for j in queue_data['jobs'] if j['status'] in ['assigned', 'running'])
    failed = sum(1 for j in queue_data['jobs'] if j['status'] == 'failed')
    
    print("\n" + "="*60)
    print(f"CONTINUOUS COLLECTION STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Total Jobs:     {total}")
    print(f"Completed:      {completed} ({100*completed/total:.1f}%)")
    print(f"Running:        {running}")
    print(f"Pending:        {pending}")
    print(f"Failed:         {failed}")
    print("-"*60)
    print("GPU Status:")
    
    for gpu_id in range(int(os.environ.get('NUM_GPUS', 8))):
        gpu_info = gpu_status.get(str(gpu_id), {})
        status = gpu_info.get('status', 'unknown')
        jobs_done = gpu_info.get('jobs_completed', 0)
        
        if status == 'busy' and gpu_info.get('current_job') is not None:
            current_job = next((j for j in queue_data['jobs'] if j['id'] == gpu_info['current_job']), None)
            if current_job:
                job_desc = f"{current_job['agent']}/{current_job['route']}"
                if current_job.get('start_time'):
                    elapsed = datetime.now().timestamp() - current_job['start_time']
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    status = f"busy ({job_desc}, {elapsed_str})"
        
        print(f"  GPU {gpu_id}: {status} | Completed: {jobs_done}")
    
    if completed > 0 and pending > 0:
        completed_jobs = [j for j in queue_data['jobs'] if j['status'] == 'completed' and j.get('duration')]
        if completed_jobs:
            avg_duration = sum(j['duration'] for j in completed_jobs) / len(completed_jobs)
            estimated_remaining = (pending * avg_duration) / int(os.environ.get('NUM_GPUS', 8))
            print("-"*60)
            print(f"Estimated time remaining: {str(timedelta(seconds=int(estimated_remaining)))}")
    
    print("="*60)
    
except Exception as e:
    print(f"Error reading status: {e}")
PYTHON_MONITOR
}

# Main execution
echo "=========================================="
echo "CONTINUOUS DATA COLLECTION SYSTEM"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "State directory: $STATE_DIR"
echo "Agents: ${AGENTS}"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Initialize queue
initialize_queue

# Export functions for subshells
export -f get_next_job
export -f mark_job_complete
export -f gpu_worker
export -f monitor_progress

# Start workers for each GPU
echo "Starting GPU workers..."
declare -a WORKER_PIDS
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_worker $gpu_id &
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

# Final status
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
        
        sorted_jobs = sorted(completed['jobs'], key=lambda x: x.get('duration', 0))
        shortest = sorted_jobs[0]
        longest = sorted_jobs[-1]
        
        print(f"  Shortest job: {shortest['agent']}/{shortest['route']} ({shortest['duration']:.1f}s)")
        print(f"  Longest job: {longest['agent']}/{longest['route']} ({longest['duration']:.1f}s)")
except Exception as e:
    print(f"Could not generate final report: {e}")
PYTHON_FINAL

echo "Results saved in: ${STATE_DIR}"
echo "Logs available in: ${LOG_DIR}"