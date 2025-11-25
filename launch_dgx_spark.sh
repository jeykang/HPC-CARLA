#!/bin/bash
set -euo pipefail

# CONFIGURATION
# ---------------------------------------------------------
export PROJECT_ROOT="$(pwd)"
export CARLA_SIF="${PROJECT_ROOT}/carla_official.sif"
export FORCE_LOGICAL_GPUS=4   # How many parallel sims to run
export NUM_GPUS=${FORCE_LOGICAL_GPUS} # Tell Python manager to create 4 slots
export PHYSICAL_GPU_ID=0      # The actual GPU ID of the GB10
# ---------------------------------------------------------

echo "[DGX Spark] Setting up Single-Device Parallelization..."

# 1. Start NVIDIA MPS (Multi-Process Service)
# This is CRITICAL. It allows 4 processes to share the GPU context 
# without serializing their work.
echo "[DGX Spark] Starting CUDA MPS..."
nvidia-cuda-mps-control -d

# 2. Start the Continuous CLI in Local + Persistent Mode
# We pass --persistent to use your optimized "Server Manager" logic.
# We pass --local to skip SLURM submission.
echo "[DGX Spark] Launching Manager..."
python3 "${PROJECT_ROOT}/continuous_cli.py" --persistent start --local \
    --slurm-gpus ${FORCE_LOGICAL_GPUS} \
    --agents interfuser \
    --routes routes_town01_short.xml

# 3. Cleanup Trap
cleanup() {
    echo "[DGX Spark] Stopping MPS..."
    echo quit | nvidia-cuda-mps-control
    echo "[DGX Spark] Done."
}
trap cleanup EXIT

# 4. Tail the logs so the script doesn't exit immediately
echo "[DGX Spark] Collection running. Tailing logs (Ctrl+C to stop)..."
touch "${PROJECT_ROOT}/logs/continuous_*.log" # Ensure glob doesn't fail
tail -f "${PROJECT_ROOT}/logs/"* 2>/dev/null