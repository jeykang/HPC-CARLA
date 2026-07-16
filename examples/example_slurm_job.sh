#!/usr/bin/env bash
# =============================================================================
# HPC-CARLA — example SLURM job (single node). Read top-to-bottom; it's a tutorial.
# =============================================================================
#
# WHAT THIS DOES
#   Acquires one GPU node, then runs the project's persistent-CARLA collection on
#   it: one CARLA server + one agent worker per GPU, pulling jobs (agent × route ×
#   weather) off a shared queue until the queue drains. Results land under
#   dataset/<agent>/weather_N/map_NN/<route>/ (results.json + manifest.json).
#
# HOW THE PIECES FIT
#   continuous_cli.py   -> the single entry point (queue + launch + monitor)
#   manage_continuous.py-> picks the next job, launches the leaderboard evaluator
#   carla_server_manager-> one persistent CARLA per GPU (started by the coordinator)
#   the agent pipeline  -> defined per agent in leaderboard/team_code/configs/*.yaml
#
# TWO WAYS TO LAUNCH
#   (A) RECOMMENDED for real/multi-node runs — from the LOGIN node, let the CLI
#       generate and submit the SLURM job for you:
#           python3 continuous_cli.py reset --smoke
#           python3 continuous_cli.py --persistent start --slurm \
#               --slurm-nodes 2 --slurm-gpus 8 --slurm-time 96:00:00 \
#               --slurm-nodelist hpc-pr-a-pod09,hpc-pr-a-pod17
#       (do NOT sbatch that — `start --slurm` submits the job itself.)
#
#   (B) THIS SCRIPT — a hand-written, single-node job you submit directly with
#       `sbatch`, so you can see the SLURM mechanics and customise them. It uses
#       `start --local`, which runs the coordinator *inside* this allocation.
#
# QUICK START
#   cd /path/to/HPC-CARLA-persistent          # the repo root
#   python3 continuous_cli.py reset --smoke   # build a tiny ~1h validation queue
#   sbatch examples/example_slurm_job.sh      # submit this job
#   python3 continuous_cli.py monitor         # watch progress (from the login node)
# =============================================================================

# --- SLURM directives --------------------------------------------------------
# NOTE: this cluster wants --gres=gpu:N (NOT --gpus=N).
#SBATCH --job-name=carla-intro
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                 # GPUs on the node (== one CARLA+agent each)
#SBATCH --time=08:00:00              # walltime; raise for the full sweep
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
# #SBATCH --nodelist=hpc-pr-a-pod09  # uncomment to pin a specific node
# #SBATCH --partition=<your-partition>

set -euo pipefail

# --- Environment -------------------------------------------------------------
# PROJECT_ROOT = the repo. SLURM_SUBMIT_DIR is where you ran `sbatch` from;
# set it explicitly if you submit from elsewhere.
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT"

# The CARLA Singularity image (built separately; ~6 GB). Override with --sif or
# the CARLA_SIF env var if it lives elsewhere.
export CARLA_SIF="${CARLA_SIF:-$PROJECT_ROOT/carla_official.sif}"

# How many GPUs this node exposes (match --gres above).
export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"

# --- Optional runtime knobs (sensible defaults; tune as needed) --------------
export JOB_TIMEOUT_SEC="${JOB_TIMEOUT_SEC:-14400}"   # hard per-job wall-clock cap (4h)
export RUN_SEED="${RUN_SEED:-2000}"                  # fixed seed -> reproducible scenarios
export AGENT_GPU_OFFSET="${AGENT_GPU_OFFSET:-0}"     # 0 = co-locate agent with its CARLA GPU
export DEAD_SERVER_BACKOFF_SEC="${DEAD_SERVER_BACKOFF_SEC:-20}"

echo "[job] node=$(hostname) gpus=$GPUS_PER_NODE sif=$CARLA_SIF seed=$RUN_SEED timeout=${JOB_TIMEOUT_SEC}s"

# --- Make sure there is a queue to work on -----------------------------------
# Build one if it's missing. `reset --smoke` = a tiny validation queue (a handful
# of single-route jobs, ~1h). For a real run use a full `reset` (optionally
# filtered: --agents tcp lav  --weather 0 1 2 ...). You can also run reset from
# the login node before submitting.
if [ ! -f collection_state/job_queue.json ]; then
  echo "[job] no queue found; building a smoke queue"
  python3 continuous_cli.py reset --smoke
fi

# --- Launch -----------------------------------------------------------------
# `--persistent start --local` runs the per-node coordinator in THIS allocation:
#   sets up the container binds + EVAL_CMD_TEMPLATE, starts one CARLA per GPU,
#   spawns one agent worker per GPU, and supervises until the queue drains.
# (Use `start --slurm` instead — from the login node, NOT here — for multi-node.)
python3 continuous_cli.py --persistent start --local

echo "[job] queue drained or walltime reached. Inspect results with:"
echo "      python3 tools/classify_outcomes.py        # infra-vs-agent outcome summary"
echo "      python3 tools/verification_report.py       # per-agent comparison report"
