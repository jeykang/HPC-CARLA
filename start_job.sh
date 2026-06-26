#!/usr/bin/env bash
# HPC-CARLA job launcher (SLURM + Singularity/Apptainer) — configurable.
#
# Sets up container binds + EVAL_CMD_TEMPLATE, optionally resets the queue, then
# submits the persistent collection job. Everything is overridable via flags or
# env vars; nothing is reset unless you ask for it.
#
# Usage:
#   ./start_job.sh [options]
#
# Queue options (a reset is performed ONLY if one of these is given):
#   --reset                 Reset the full queue before starting
#   --smoke                 Reset to the tiny validation queue (12 single-route jobs)
#   --agents "a b"          Reset, limited to these agents          (implies --reset)
#   --weather "0 1 2"       Reset, limited to these weather indices (implies --reset)
#   --routes "f1 f2"        Reset, limited to these route files     (implies --reset)
#   --limit N               Cap the reset queue at N jobs           (implies --reset)
#   (no queue option) -> keep the existing queue and just (re)start workers
#
# SLURM options (defaults shown):
#   --nodelist LIST         (hpc-pr-a-pod09,hpc-pr-a-pod17)
#   --nodes N               (2)
#   --gpus N                (8)         GPUs per node
#   --time HH:MM:SS         (336:00:00)
#   --partition NAME        (unset)
#   --slurm-extra "..."     extra args passed through to continuous_cli start
#
# Runtime knobs (forwarded to the workers on every node):
#   --sif PATH              CARLA_SIF                  (PROJECT_ROOT/carla_official.sif)
#   --job-timeout SEC       JOB_TIMEOUT_SEC           (per-job wall-clock cap; default 14400)
#   --agent-gpu-offset N    AGENT_GPU_OFFSET          (0 = co-locate agent with its CARLA GPU)
#   --agent-gpu-pin N       AGENT_GPU_PIN             (force all agents onto GPU N; benchmark)
#   --dead-server-backoff S DEAD_SERVER_BACKOFF_SEC   (sleep after skipping a dead server)
#
# Other:
#   --dry-run               Print the reset/start commands without executing
#   -h | --help             Show this help and exit
#
# Examples:
#   ./start_job.sh --smoke                      # quick validation run
#   ./start_job.sh --reset --nodes 2 --gpus 8   # full sweep, fresh queue
#   ./start_job.sh                              # resume existing queue
#   ./start_job.sh --agent-gpu-pin 0 --smoke    # benchmark: dogpile config
set -euo pipefail

# --- Defaults ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
export PROJECT_ROOT
export CARLA_SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"

DO_RESET=0
RESET_SMOKE=0
RESET_AGENTS=""
RESET_WEATHER=""
RESET_ROUTES=""
RESET_LIMIT=""

SLURM_NODELIST="${SLURM_NODELIST_DEFAULT:-hpc-pr-a-pod09,hpc-pr-a-pod17}"
SLURM_NODES="${SLURM_NODES_DEFAULT:-2}"
SLURM_GPUS="${SLURM_GPUS_DEFAULT:-8}"
SLURM_TIME="${SLURM_TIME_DEFAULT:-336:00:00}"
SLURM_PARTITION=""
SLURM_EXTRA=""

DRY_RUN=0

usage() { sed -n '2,49p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reset)                DO_RESET=1; shift ;;
    --smoke)                DO_RESET=1; RESET_SMOKE=1; shift ;;
    --agents)               DO_RESET=1; RESET_AGENTS="$2"; shift 2 ;;
    --weather)              DO_RESET=1; RESET_WEATHER="$2"; shift 2 ;;
    --routes)               DO_RESET=1; RESET_ROUTES="$2"; shift 2 ;;
    --limit)                DO_RESET=1; RESET_LIMIT="$2"; shift 2 ;;
    --nodelist)             SLURM_NODELIST="$2"; shift 2 ;;
    --nodes)                SLURM_NODES="$2"; shift 2 ;;
    --gpus)                 SLURM_GPUS="$2"; shift 2 ;;
    --time)                 SLURM_TIME="$2"; shift 2 ;;
    --partition)            SLURM_PARTITION="$2"; shift 2 ;;
    --slurm-extra)          SLURM_EXTRA="$2"; shift 2 ;;
    --sif)                  export CARLA_SIF="$2"; shift 2 ;;
    --job-timeout)          export JOB_TIMEOUT_SEC="$2"; shift 2 ;;
    --agent-gpu-offset)     export AGENT_GPU_OFFSET="$2"; shift 2 ;;
    --agent-gpu-pin)        export AGENT_GPU_PIN="$2"; shift 2 ;;
    --dead-server-backoff)  export DEAD_SERVER_BACKOFF_SEC="$2"; shift 2 ;;
    --dry-run)              DRY_RUN=1; shift ;;
    -h|--help)              usage; exit 0 ;;
    *) echo "[start_job][ERROR] Unknown option: $1" >&2; echo "Try --help" >&2; exit 2 ;;
  esac
done

echo "[start_job] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[start_job] CARLA_SIF=${CARLA_SIF}"

if [[ ! -f "${CARLA_SIF}" ]]; then
  echo "[start_job][FATAL] Missing image: ${CARLA_SIF}" >&2
  echo "           Build the SIF first on a machine that allows it." >&2
  exit 1
fi

# --- Container binds ---------------------------------------------------------
# Workspace bind (project tree -> /workspace) + the libnvidia-gpucomp workaround
# (driver 575.x split GPU compute into a lib the cluster's --nv doesn't auto-bind;
# the SIF carries a 0-byte placeholder so the bind destination always exists).
BIND_SPECS=( "${PROJECT_ROOT}:/workspace" )
NVIDIA_GPUCOMP_HOST="/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.575.57.08"
BIND_SPECS+=( "${NVIDIA_GPUCOMP_HOST}:${NVIDIA_GPUCOMP_HOST}" )

_bind_join() {
  local var="$1"; local cur="${!var:-}"
  for spec in "${BIND_SPECS[@]}"; do
    case ",${cur}," in
      *",${spec},"*) : ;;
      *) cur="${cur:+${cur},}${spec}" ;;
    esac
  done
  export "${var}=${cur}"
}
_bind_join SINGULARITY_BINDPATH
_bind_join APPTAINER_BINDPATH

# Pass through useful context (NOTE: do not override container PYTHONPATH).
export SINGULARITYENV_PROJECT_ROOT="${PROJECT_ROOT}"
export APPTAINERENV_PROJECT_ROOT="${PROJECT_ROOT}"
export SINGULARITYENV_CARLA_SIF="${CARLA_SIF}"
export APPTAINERENV_CARLA_SIF="${CARLA_SIF}"

echo "[start_job] SINGULARITY_BINDPATH=${SINGULARITY_BINDPATH}"
echo "[start_job] knobs: JOB_TIMEOUT_SEC=${JOB_TIMEOUT_SEC:-default(14400)} AGENT_GPU_OFFSET=${AGENT_GPU_OFFSET:-0} AGENT_GPU_PIN=${AGENT_GPU_PIN:-<colocate>} DEAD_SERVER_BACKOFF_SEC=${DEAD_SERVER_BACKOFF_SEC:-default(20)}"

# --- Command template for the evaluator -------------------------------------
EVAL_CMD_TEMPLATE="$(cat <<'EOF'
singularity exec --nv --pwd /workspace \
  -B /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.575.57.08:/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.575.57.08 \
  "${{CARLA_SIF}}" bash -lc '
  set -euo pipefail
  export PYTHONPATH="/workspace:/workspace/leaderboard:/workspace/scenario_runner:${{PYTHONPATH:-}}"
  export ROUTES="{ROUTES_FILE}"
  export SCENARIOS="{SCENARIOS_FILE}"
  python3 -m leaderboard.leaderboard_evaluator \
    --routes "{ROUTES_FILE}" \
    --scenarios "{SCENARIOS_FILE}" \
    --agent "{AGENT_CODE}" \
    --agent-config "{AGENT_CFG}" \
    --host "{HOST}" --port "{PORT}" --trafficManagerPort "{TM_PORT}"
'
EOF
)"
export EVAL_CMD_TEMPLATE

# --- Build the reset + start commands ---------------------------------------
CLI="${PROJECT_ROOT}/continuous_cli.py"

RESET_CMD=( python3 "${CLI}" reset )
[[ -n "${RESET_AGENTS}" ]]  && RESET_CMD+=( --agents  ${RESET_AGENTS} )
[[ -n "${RESET_WEATHER}" ]] && RESET_CMD+=( --weather ${RESET_WEATHER} )
[[ -n "${RESET_ROUTES}" ]]  && RESET_CMD+=( --routes  ${RESET_ROUTES} )
[[ -n "${RESET_LIMIT}" ]]   && RESET_CMD+=( --limit   "${RESET_LIMIT}" )
[[ "${RESET_SMOKE}" -eq 1 ]] && RESET_CMD+=( --smoke )

START_CMD=( python3 "${CLI}" --persistent start --slurm
            --slurm-nodelist="${SLURM_NODELIST}"
            --slurm-nodes="${SLURM_NODES}"
            --slurm-gpus="${SLURM_GPUS}"
            --slurm-time="${SLURM_TIME}" )
[[ -n "${SLURM_PARTITION}" ]] && START_CMD+=( --slurm-partition="${SLURM_PARTITION}" )
# shellcheck disable=SC2206
[[ -n "${SLURM_EXTRA}" ]] && START_CMD+=( ${SLURM_EXTRA} )

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# --- Execute -----------------------------------------------------------------
if [[ "${DO_RESET}" -eq 1 ]]; then
  echo "[start_job] Resetting queue: ${RESET_CMD[*]}"
  run "${RESET_CMD[@]}"
else
  echo "[start_job] Keeping existing queue (no reset requested; pass --reset/--smoke to change)."
fi

echo "[start_job] Starting: ${START_CMD[*]}"
run "${START_CMD[@]}"
