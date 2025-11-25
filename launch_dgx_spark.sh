#!/bin/bash
set -euo pipefail

# CONFIGURATION
# ---------------------------------------------------------
export PROJECT_ROOT="$(pwd)"
export CARLA_SIF="${PROJECT_ROOT}/carla_official.sif"
export FORCE_LOGICAL_GPUS=4   # How many parallel sims to run
export NUM_GPUS=${FORCE_LOGICAL_GPUS} 
export PHYSICAL_GPU_ID=0      # The actual GPU ID of the GB10

# MPS CONFIGURATION (The Fix)
# ---------------------------------------------------------
# We create a local directory for MPS pipes so we don't conflict 
# with system permissions or other users in /tmp
export MPS_CACHE_DIR="${PROJECT_ROOT}/.mps"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_CACHE_DIR}/pipe"
export CUDA_MPS_LOG_DIRECTORY="${MPS_CACHE_DIR}/log"

# Ensure these directories exist before starting the daemon
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
mkdir -p "${CUDA_MPS_LOG_DIRECTORY}"
# ---------------------------------------------------------

echo "[DGX Spark] Setting up Single-Device Parallelization..."
echo "[DGX Spark] MPS Directory: ${MPS_CACHE_DIR}"

# 1. Start NVIDIA MPS (Multi-Process Service)
echo "[DGX Spark] Starting CUDA MPS..."
nvidia-cuda-mps-control -d

# 2. Start the Continuous CLI in Local + Persistent Mode
echo "[DGX Spark] Launching Manager..."
# Note: We use --local to skip SLURM and --persistent for the optimized server logic
python3 "${PROJECT_ROOT}/continuous_cli.py" --persistent start --local \
    --slurm-gpus ${FORCE_LOGICAL_GPUS} \
    --agents interfuser \
    --routes routes_town01_short.xml

# 3. Cleanup Trap
cleanup() {
    echo ""
    echo "[DGX Spark] Shutting down..."
    
    # Stop MPS Daemon
    echo quit | nvidia-cuda-mps-control
    
    # Clean up the pipe directories to prevent stale locks next time
    rm -rf "${MPS_CACHE_DIR}"
    
    echo "[DGX Spark] Done."
}
trap cleanup EXIT

# 4. Tail the logs
echo "[DGX Spark] Collection running. Tailing logs (Ctrl+C to stop)..."
# Wait a moment for logs to appear
sleep 2
tail -f "${PROJECT_ROOT}/logs/"* 2>/dev/null