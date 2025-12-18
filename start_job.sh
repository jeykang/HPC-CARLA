#!/usr/bin/env bash
# HPC-CARLA job launcher (SLURM + Singularity/Apptainer)
# - Binds the repo into the container at /workspace
# - Ensures Python can import leaderboard + scenario_runner
# - Provides a correct EVAL_CMD_TEMPLATE for manage_continuous.py
set -euo pipefail

# --- Resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
export PROJECT_ROOT

# Singularity/Apptainer image
export CARLA_SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"

echo "[start_job] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[start_job] CARLA_SIF=${CARLA_SIF}"

if [[ ! -f "${CARLA_SIF}" ]]; then
  echo "[start_job][FATAL] Missing image: ${CARLA_SIF}"
  echo "           If you only have carla_official.def, build the SIF first on a machine that allows it."
  exit 1
fi

# --- Container binds ---------------------------------------------------------
BIND_SPEC="${PROJECT_ROOT}:/workspace"

# Singularity env
if [[ -z "${SINGULARITY_BINDPATH:-}" ]]; then
  export SINGULARITY_BINDPATH="${BIND_SPEC}"
else
  case ",${SINGULARITY_BINDPATH}," in
    *",${BIND_SPEC},"*) : ;;
    *) export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH},${BIND_SPEC}";;
  esac
fi

# Apptainer env (some clusters alias Singularity → Apptainer)
if [[ -z "${APPTAINER_BINDPATH:-}" ]]; then
  export APPTAINER_BINDPATH="${BIND_SPEC}"
else
  case ",${APPTAINER_BINDPATH}," in
    *",${BIND_SPEC},"*) : ;;
    *) export APPTAINER_BINDPATH="${APPTAINER_BINDPATH},${BIND_SPEC}";;
  esac
fi

# NOTE: Do NOT override PYTHONPATH from the host. The container’s %environment
# already includes the CARLA egg + /workspace paths. Overriding here would drop the egg.

# Also pass through useful context
export SINGULARITYENV_PROJECT_ROOT="${PROJECT_ROOT}"
export APPTAINERENV_PROJECT_ROOT="${PROJECT_ROOT}"
export SINGULARITYENV_CARLA_SIF="${CARLA_SIF}"
export APPTAINERENV_CARLA_SIF="${CARLA_SIF}"

echo "[start_job] SINGULARITY_BINDPATH=${SINGULARITY_BINDPATH}"
echo "[start_job] APPTAINER_BINDPATH=${APPTAINER_BINDPATH}"

# --- Command template for the evaluator -------------------------------------
# Escape ${CARLA_SIF} so Python .format() doesn’t treat it as a placeholder.
EVAL_CMD_TEMPLATE="$(cat <<'EOF'
singularity exec --nv --pwd /workspace "${{CARLA_SIF}}" bash -lc '
  set -euo pipefail
  export PYTHONPATH="/workspace:/workspace/leaderboard:/workspace/scenario_runner:${{PYTHONPATH:-}}"
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

# (Optional) direct entrypoint override — unused by your manager, but harmless:
export EVAL_ENTRYPOINT=""

# --- Quick smoke check (non-fatal) ------------------------------------------
#echo "[start_job] Smoke check: list /workspace inside container"
#singularity exec --nv --pwd /workspace "${CARLA_SIF}" bash -lc 'ls -la . || true'

#echo "[start_job] Smoke check: python can import leaderboard.leaderboard_evaluator"
# Preserve container PYTHONPATH (which includes the CARLA egg) and just prepend repo paths.
#singularity exec --nv --pwd /workspace "${CARLA_SIF}" bash -lc \
#  'PYTHONPATH="/workspace:/workspace/leaderboard:/workspace/scenario_runner:$PYTHONPATH" \
#   python3 -c "import leaderboard.leaderboard_evaluator as L; print(getattr(L, \"__file__\", \"ok\"))" || true'

# --- Kick off orchestration --------------------------------------------------
# Reset any stale state (non-fatal if empty)
python3 "${PROJECT_ROOT}/continuous_cli.py" reset || true

# Adjust these SLURM flags to your cluster defaults if needed
python3 "${PROJECT_ROOT}/continuous_cli.py" --persistent start --slurm \
  --slurm-nodelist="hpc-pr-a-pod09" \
  --slurm-gpus=8 --slurm-nodes=1 --slurm-time=336:00:00
