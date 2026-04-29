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
# Workspace bind (project tree -> /workspace inside container).
BIND_SPECS=( "${PROJECT_ROOT}:/workspace" )

# Workaround for missing libnvidia-gpucomp.so inside the container.
# Driver 575.x split GPU compute code into a new userspace lib that this
# cluster's Singularity --nv (legacy nvliblist.conf) doesn't auto-bind.
# UE4 4.24's RHI dlopens libGLX_nvidia.so.0 / libnvidia-glcore.so.575.57.08,
# both of which depend on libnvidia-gpucomp.so.575.57.08 — without it, UE4
# dies before the RPC server starts. We bind it from the host into
# /.singularity.d/libs/ where it joins the rest of the --nv-bound NVIDIA libs
# (that dir is already on LD_LIBRARY_PATH).
NVIDIA_GPUCOMP_HOST="/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.575.57.08"
if [[ -e "${NVIDIA_GPUCOMP_HOST}" ]]; then
  # Bind onto a placeholder pre-created in the SIF at the SAME path. The path
  # is on the loader's default search and on the container's LD_LIBRARY_PATH,
  # so no LD_LIBRARY_PATH gymnastics needed. Requires the SIF to have been
  # built from the latest .def (which contains a `touch` of this path).
  BIND_SPECS+=( "${NVIDIA_GPUCOMP_HOST}:${NVIDIA_GPUCOMP_HOST}" )
else
  echo "[start_job][WARN] ${NVIDIA_GPUCOMP_HOST} not found on host — UE4 RHI init may still fail."
fi

_bind_join() {
  # Append each entry in BIND_SPECS to the named env var (Singularity or Apptainer
  # bindpath), avoiding duplicates. POSIX-safe: comma-separated.
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
  --slurm-nodelist="hpc-pr-a-pod09,hpc-pr-a-pod17" \
  --slurm-gpus=8 --slurm-nodes=2 --slurm-time=336:00:00