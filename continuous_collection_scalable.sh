#!/bin/bash
# ================================================================
#  Scalable continuous data collection with automatic GPU/node detection
#  Supports multi-node execution with dynamic GPU allocation
# ================================================================

#SBATCH --job-name=continuous_collection_scalable
#SBATCH --gpus-per-node=8
#SBATCH --nodes=2-4
#SBATCH --ntasks-per-node=1
#SBATCH --time=168:00:00
#SBATCH --output=logs/continuous_scalable_%A_%N.out
#SBATCH --error=logs/continuous_scalable_%A_%N.err

# Configuration
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
STATE_DIR="${PROJECT_ROOT}/collection_state"
LOG_DIR="${PROJECT_ROOT}/logs"

# Create necessary directories
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Detect SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM job $SLURM_JOB_ID"
    
    # Auto-detect number of GPUs per node
    if [ -n "$SLURM_GPUS_PER_NODE" ]; then
        # Parse SLURM GPU string (e.g., "gpu:8" or "gpu:v100:4")
        GPUS_PER_NODE=$(echo $SLURM_GPUS_PER_NODE | grep -oE '[0-9]+$')
    elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Count CUDA devices
        GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    else
        # Default fallback
        GPUS_PER_NODE=8
    fi
    
    # Get node information
    NODE_ID=$SLURM_NODEID
    NODE_NAME=$SLURMD_NODENAME
    NUM_NODES=$SLURM_NNODES
    TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))
    
    # For multi-node coordination
    IS_MASTER=$([[ $SLURM_NODEID -eq 0 ]] && echo "true" || echo "false")
    
else
    # Local execution (no SLURM)
    echo "Running locally (no SLURM detected)"
    GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 8)}
    NODE_ID=0
    NODE_NAME=$(hostname)
    NUM_NODES=1
    TOTAL_GPUS=$GPUS_PER_NODE
    IS_MASTER="true"
fi

echo "=========================================="
echo "SCALABLE CONTINUOUS COLLECTION SYSTEM"
echo "=========================================="
echo "Node: $NODE_NAME (ID: $NODE_ID)"
echo "GPUs on this node: $GPUS_PER_NODE"
echo "Total nodes: $NUM_NODES"
echo "Total GPUs: $TOTAL_GPUS"
echo "Is master: $IS_MASTER"
echo "=========================================="

# Calculate port ranges to avoid conflicts
# Each node gets a port range based on its ID
BASE_RPC_PORT=$((2000 + NODE_ID * 100))
BASE_TM_PORT=$((8000 + NODE_ID * 100))

echo "Port range for this node: RPC ${BASE_RPC_PORT}-$((BASE_RPC_PORT + GPUS_PER_NODE * 10))"

# Initialize shared state (master node only)
initialize_scalable_queue() {
    if [ "$IS_MASTER" != "true" ]; then
        return
    fi
    
    echo "Master node: Initializing job queue for $TOTAL_GPUS GPUs across $NUM_NODES nodes..."
    
    python3 - << PYTHON_SCRIPT
import json
import sys
import os

state_dir = os.environ['STATE_DIR']
total_gpus = int(os.environ['TOTAL_GPUS'])
num_nodes = int(os.environ['NUM_NODES'])
gpus_per_node = int(os.environ['GPUS_PER_NODE'])

# Get agents from configs
configs_dir = os.path.join(os.environ['PROJECT_ROOT'], 'leaderboard/team_code/configs')
import glob
agent_configs = glob.glob(os.path.join(configs_dir, '*.yaml'))
agents = [os.path.basename(c).replace('.yaml', '') for c in agent_configs]

if not agents:
    print("WARNING: No agent configs found")
    agents = ['sample_agent']

weather_indices = list(range(15))
route_files = [
    "routes_town01_short.xml", "routes_town01_long.xml",
    "routes_town02_short.xml", "routes_town02_long.xml",
    "routes_town03_short.xml", "routes_town03_long.xml",
    "routes_town04_short.xml", "routes_town04_long.xml",
    "routes_town05_short.xml", "routes_town05_long.xml",
    "routes_town06_short.xml", "routes_town06_long.xml",
    "routes_town07_short.xml", "routes_town07_long.xml"
]

# Generate all combinations
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
                'node': None,  # Added node tracking
                'start_time': None,
                'end_time': None,
                'duration': None
            })
            job_id += 1

# Save queue
queue_file = os.path.join(state_dir, 'job_queue.json')
with open(queue_file, 'w') as f:
    json.dump({'jobs': jobs, 'total': len(jobs), 'completed': 0}, f, indent=2)

print(f"Initialized queue with {len(jobs)} jobs for {total_gpus} GPUs across {num_nodes} nodes")

# Initialize GPU status for all nodes
gpu_status = {}
for node in range(num_nodes):
    for gpu in range(gpus_per_node):
        global_gpu_id = node * gpus_per_node + gpu
        gpu_status[f"{node}_{gpu}"] = {
            'global_id': global_gpu_id,
            'node_id': node,
            'local_gpu_id': gpu,
            'status': 'idle',
            'current_job': None,
            'jobs_completed': 0,
            'total_runtime': 0,
            'last_heartbeat': None
        }

gpu_status_file = os.path.join(state_dir, 'gpu_status.json')
with open(gpu_status_file, 'w') as f:
    json.dump(gpu_status, f, indent=2)

print(f"Initialized {len(gpu_status)} GPU workers")

# Initialize runtime estimates
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

# Initialize completed jobs file
completed_file = os.path.join(state_dir, 'completed_jobs.json')
with open(completed_file, 'w') as f:
    json.dump({'jobs': []}, f, indent=2)

# Node coordination file
node_status = {
    'nodes': {
        str(i): {
            'status': 'starting',
            'hostname': None,
            'gpus': gpus_per_node,
            'last_heartbeat': None
        } for i in range(num_nodes)
    }
}

node_status_file = os.path.join(state_dir, 'node_status.json')
with open(node_status_file, 'w') as f:
    json.dump(node_status, f, indent=2)

PYTHON_SCRIPT
}

# Get next job for this node's GPU
get_next_job_scalable() {
    local node_id=$1
    local local_gpu_id=$2
    
    python3 - << PYTHON_SCRIPT
import json
import sys
import os
import fcntl
from datetime import datetime

state_dir = os.environ['STATE_DIR']
node_id = int(sys.argv[1])
local_gpu_id = int(sys.argv[2])
gpu_key = f"{node_id}_{local_gpu_id}"

lock_file = os.path.join(state_dir, '.scheduler.lock')
os.makedirs(os.path.dirname(lock_file), exist_ok=True)

with open(lock_file, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    
    try:
        # Load current state
        with open(os.path.join(state_dir, 'job_queue.json'), 'r') as f:
            queue_data = json.load(f)
        with open(os.path.join(state_dir, 'runtime_estimates.json'), 'r') as f:
            runtime_data = json.load(f)
        with open(os.path.join(state_dir, 'gpu_status.json'), 'r') as f:
            gpu_status = json.load(f)
        
        # Find pending jobs
        pending_jobs = [j for j in queue_data['jobs'] if j['status'] == 'pending']
        
        if not pending_jobs:
            failed_jobs = [j for j in queue_data['jobs'] 
                          if j['status'] == 'failed' and j['attempts'] < 3]
            if failed_jobs:
                pending_jobs = failed_jobs
            else:
                print("NO_MORE_JOBS")
                sys.exit(0)
        
        # Estimate runtime for each job
        job_estimates = []
        for job in pending_jobs:
            key = f"{job['agent']}_{job['route']}"
            estimate = runtime_data['combinations'].get(key, runtime_data['default'])
            
            # Weather complexity factor
            weather_factor = 1.0
            if job['weather'] >= 7:
                weather_factor = 1.1
            if job['weather'] in [4, 5, 6, 11, 12, 13]:
                weather_factor = 1.2
            
            estimate *= weather_factor
            job_estimates.append((job, estimate))
        
        # Sort jobs by estimated runtime (longest first)
        job_estimates.sort(key=lambda x: x[1], reverse=True)
        
        # Select job for this GPU
        selected_job = job_estimates[0][0] if job_estimates else None
        
        if selected_job:
            # Update job status
            for job in queue_data['jobs']:
                if job['id'] == selected_job['id']:
                    job['status'] = 'assigned'
                    job['gpu'] = local_gpu_id
                    job['node'] = node_id
                    job['start_time'] = datetime.now().timestamp()
                    job['attempts'] += 1
                    break
            
            # Update GPU status
            gpu_status[gpu_key]['status'] = 'busy'
            gpu_status[gpu_key]['current_job'] = selected_job['id']
            gpu_status[gpu_key]['last_heartbeat'] = datetime.now().timestamp()
            
            # Save updated state
            with open(os.path.join(state_dir, 'job_queue.json'), 'w') as f:
                json.dump(queue_data, f, indent=2)
            with open(os.path.join(state_dir, 'gpu_status.json'), 'w') as f:
                json.dump(gpu_status, f, indent=2)
            
            print(f"{selected_job['id']}|{selected_job['agent']}|{selected_job['weather']}|{selected_job['route']}")
        else:
            print("NO_MORE_JOBS")
            
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)

PYTHON_SCRIPT $1 $2
}

# Mark job complete (node-aware)
mark_job_complete_scalable() {
    local job_id=$1
    local node_id=$2
    local local_gpu_id=$3
    local duration=$4
    local status=$5
    
    python3 - << PYTHON_SCRIPT
import json
import sys
import os
import fcntl
from datetime import datetime

state_dir = os.environ['STATE_DIR']
job_id = int(sys.argv[1])
node_id = int(sys.argv[2])
local_gpu_id = int(sys.argv[3])
duration = float(sys.argv[4])
status = sys.argv[5]
gpu_key = f"{node_id}_{local_gpu_id}"

lock_file = os.path.join(state_dir, '.scheduler.lock')
with open(lock_file, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    
    try:
        # Load files
        with open(os.path.join(state_dir, 'job_queue.json'), 'r') as f:
            queue_data = json.load(f)
        with open(os.path.join(state_dir, 'runtime_estimates.json'), 'r') as f:
            runtime_data = json.load(f)
        with open(os.path.join(state_dir, 'gpu_status.json'), 'r') as f:
            gpu_status = json.load(f)
        with open(os.path.join(state_dir, 'completed_jobs.json'), 'r') as f:
            completed_data = json.load(f)
        
        # Update job status
        for job in queue_data['jobs']:
            if job['id'] == job_id:
                job['status'] = status
                job['end_time'] = datetime.now().timestamp()
                job['duration'] = duration
                job['completed_by_node'] = node_id
                job['completed_by_gpu'] = local_gpu_id
                
                if status == 'completed':
                    # Update runtime estimates
                    key = f"{job['agent']}_{job['route']}"
                    old_estimate = runtime_data['combinations'].get(key, runtime_data['default'])
                    alpha = 0.3
                    new_estimate = alpha * duration + (1 - alpha) * old_estimate
                    runtime_data['combinations'][key] = new_estimate
                    
                    completed_data['jobs'].append(job)
                    queue_data['completed'] += 1
                    
                    print(f"Node {node_id} GPU {local_gpu_id}: Job {job_id} completed in {duration:.1f}s")
                break
        
        # Update GPU status
        gpu_status[gpu_key]['status'] = 'idle'
        gpu_status[gpu_key]['current_job'] = None
        if status == 'completed':
            gpu_status[gpu_key]['jobs_completed'] += 1
            gpu_status[gpu_key]['total_runtime'] += duration
        
        # Save state
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

PYTHON_SCRIPT $1 $2 $3 $4 $5
}

# GPU worker for scalable system
gpu_worker_scalable() {
    local local_gpu_id=$1
    local worker_log="${LOG_DIR}/worker_node${NODE_ID}_gpu${local_gpu_id}.log"
    
    echo "[Node $NODE_ID GPU $local_gpu_id] Worker started at $(date)" > "$worker_log"
    
    while true; do
        # Get next job
        JOB_INFO=$(get_next_job_scalable $NODE_ID $local_gpu_id)
        
        if [ "$JOB_INFO" == "NO_MORE_JOBS" ]; then
            echo "[Node $NODE_ID GPU $local_gpu_id] No more jobs available" >> "$worker_log"
            break
        fi
        
        # Parse job info
        IFS='|' read -r JOB_ID AGENT WEATHER ROUTE <<< "$JOB_INFO"
        
        echo "[Node $NODE_ID GPU $local_gpu_id] Starting job $JOB_ID: agent=$AGENT weather=$WEATHER route=$ROUTE" >> "$worker_log"
        
        # Calculate ports for this GPU on this node
        RPC_PORT=$((BASE_RPC_PORT + local_gpu_id * 10))
        TM_PORT=$((BASE_TM_PORT + local_gpu_id * 10))
        
        # Run the job
        START_TIME=$(date +%s)
        
        export AGENT_TYPE=$AGENT
        export WEATHER_START=$WEATHER
        export WEATHER_END=$WEATHER
        export ROUTE_FILE=$ROUTE
        export GPU_ID=$local_gpu_id
        export RPC_PORT=$RPC_PORT
        export TM_PORT=$TM_PORT
        
        # Run single job
        bash "${PROJECT_ROOT}/generate_single_job.sh" \
            > "${LOG_DIR}/job_${JOB_ID}_node${NODE_ID}_gpu${local_gpu_id}.out" \
            2> "${LOG_DIR}/job_${JOB_ID}_node${NODE_ID}_gpu${local_gpu_id}.err"
        
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ $EXIT_CODE -eq 0 ]; then
            mark_job_complete_scalable $JOB_ID $NODE_ID $local_gpu_id $DURATION "completed"
            echo "[Node $NODE_ID GPU $local_gpu_id] Job $JOB_ID completed in ${DURATION}s" >> "$worker_log"
        else
            mark_job_complete_scalable $JOB_ID $NODE_ID $local_gpu_id $DURATION "failed"
            echo "[Node $NODE_ID GPU $local_gpu_id] Job $JOB_ID failed with code $EXIT_CODE" >> "$worker_log"
        fi
        
        sleep 5
    done
    
    echo "[Node $NODE_ID GPU $local_gpu_id] Worker finished at $(date)" >> "$worker_log"
}

# Node heartbeat (for monitoring)
node_heartbeat() {
    while true; do
        python3 - << PYTHON_SCRIPT
import json
import os
from datetime import datetime

state_dir = os.environ['STATE_DIR']
node_id = os.environ['NODE_ID']
node_name = os.environ['NODE_NAME']

try:
    node_status_file = os.path.join(state_dir, 'node_status.json')
    if os.path.exists(node_status_file):
        with open(node_status_file, 'r') as f:
            node_status = json.load(f)
    else:
        node_status = {'nodes': {}}
    
    node_status['nodes'][node_id] = {
        'status': 'active',
        'hostname': node_name,
        'gpus': int(os.environ['GPUS_PER_NODE']),
        'last_heartbeat': datetime.now().timestamp()
    }
    
    with open(node_status_file, 'w') as f:
        json.dump(node_status, f, indent=2)
        
except Exception as e:
    print(f"Heartbeat error: {e}")

PYTHON_SCRIPT
        sleep 30
    done
}

# Wait for state initialization (for worker nodes)
wait_for_initialization() {
    echo "Waiting for master node to initialize state..."
    while [ ! -f "$STATE_DIR/job_queue.json" ]; do
        sleep 5
    done
    echo "State initialized, starting workers..."
}

# Main execution
if [ "$IS_MASTER" == "true" ]; then
    # Master node: Initialize and coordinate
    initialize_scalable_queue
    
    # Start heartbeat
    node_heartbeat &
    HEARTBEAT_PID=$!
fi

# All nodes: Wait for initialization then start workers
if [ "$IS_MASTER" != "true" ]; then
    wait_for_initialization
    node_heartbeat &
    HEARTBEAT_PID=$!
fi

# Start workers for each GPU on this node
echo "Starting $GPUS_PER_NODE GPU workers on node $NODE_ID..."
declare -a WORKER_PIDS

for gpu_id in $(seq 0 $((GPUS_PER_NODE - 1))); do
    gpu_worker_scalable $gpu_id &
    WORKER_PIDS[$gpu_id]=$!
    echo "Started worker for GPU $gpu_id (PID: ${WORKER_PIDS[$gpu_id]})"
    sleep 2
done

# Monitor workers
echo "Node $NODE_ID: Monitoring workers..."
while true; do
    sleep 60
    
    ACTIVE_WORKERS=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            ACTIVE_WORKERS=$((ACTIVE_WORKERS + 1))
        fi
    done
    
    echo "Node $NODE_ID: $ACTIVE_WORKERS/$GPUS_PER_NODE workers active"
    
    if [ $ACTIVE_WORKERS -eq 0 ]; then
        echo "Node $NODE_ID: All workers finished"
        break
    fi
done

# Cleanup
kill $HEARTBEAT_PID 2>/dev/null || true

# Master node: Final report
if [ "$IS_MASTER" == "true" ]; then
    echo ""
    echo "=========================================="
    echo "COLLECTION COMPLETE - FINAL REPORT"
    echo "=========================================="
    
    python3 - << PYTHON_SCRIPT
import json
import os
from datetime import timedelta

state_dir = os.environ['STATE_DIR']

with open(os.path.join(state_dir, 'completed_jobs.json'), 'r') as f:
    completed = json.load(f)

with open(os.path.join(state_dir, 'gpu_status.json'), 'r') as f:
    gpu_status = json.load(f)

if completed['jobs']:
    total_jobs = len(completed['jobs'])
    total_duration = sum(j['duration'] for j in completed['jobs'] if j.get('duration'))
    avg_duration = total_duration / total_jobs
    
    print(f"Total jobs completed: {total_jobs}")
    print(f"Total runtime: {timedelta(seconds=int(total_duration))}")
    print(f"Average job duration: {timedelta(seconds=int(avg_duration)))}")
    
    # Per-node statistics
    node_stats = {}
    for job in completed['jobs']:
        node = job.get('completed_by_node', 'unknown')
        if node not in node_stats:
            node_stats[node] = {'count': 0, 'time': 0}
        node_stats[node]['count'] += 1
        node_stats[node]['time'] += job.get('duration', 0)
    
    print("\nPer-node performance:")
    for node, stats in sorted(node_stats.items()):
        if stats['count'] > 0:
            avg_time = stats['time'] / stats['count']
            print(f"  Node {node}: {stats['count']} jobs, avg {timedelta(seconds=int(avg_time))}")
    
    # GPU utilization
    total_gpu_time = sum(g['total_runtime'] for g in gpu_status.values())
    if total_gpu_time > 0:
        efficiency = (total_jobs * avg_duration) / total_gpu_time * 100
        print(f"\nOverall efficiency: {efficiency:.1f}% GPU utilization")

PYTHON_SCRIPT
fi

echo "Node $NODE_ID finished"