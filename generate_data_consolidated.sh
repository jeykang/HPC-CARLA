#!/bin/bash
# Single-node, multi-GPU CARLA data collection via ConsolidatedAgent (Singularity + SLURM)

#SBATCH --job-name=carla_collect
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---------- Config (override via env) ----------
AGENT_TYPE="${AGENT_TYPE:-interfuser}"

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
AGENT_DIR_HOST="${AGENT_DIR_HOST:-${PROJECT_ROOT}/leaderboard/team_code}"
CARLA_SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

# Evaluation span
WEATHER_START="${WEATHER_START:-1}"
WEATHER_END="${WEATHER_END:-8}"

# GPU and port settings
GPUS="${GPUS:-8}"
BASE_RPC_PORT="${BASE_RPC_PORT:-2000}"
BASE_TM_PORT="${BASE_TM_PORT:-8000}"

# Agent + config paths (host/container)
AGENT_YAML_CONFIG_HOST="${AGENT_YAML_CONFIG_HOST:-${AGENT_DIR_HOST}/configs/${AGENT_TYPE}.yaml}"
AGENT_YAML_CONFIG_CONTAINER="${AGENT_YAML_CONFIG_CONTAINER:-/workspace/leaderboard/team_code/configs/${AGENT_TYPE}.yaml}"
CONSOLIDATED_AGENT_PATH="${CONSOLIDATED_AGENT_PATH:-/workspace/leaderboard/team_code/consolidated_agent.py}"

TEAM_AGENT="${TEAM_AGENT:-$CONSOLIDATED_AGENT_PATH}"
TEAM_CONFIG="${TEAM_CONFIG:-$AGENT_YAML_CONFIG_CONTAINER}"

# ---------- Sanity checks ----------
[ -f "$CARLA_SIF" ] || { echo "FATAL: CARLA Singularity image not found: $CARLA_SIF" >&2; exit 1; }
[ -f "${AGENT_DIR_HOST}/consolidated_agent.py" ] || { echo "FATAL: consolidated_agent.py not found in $AGENT_DIR_HOST" >&2; exit 1; }
[ -f "$AGENT_YAML_CONFIG_HOST" ] || { echo "FATAL: Agent YAML config not found: $AGENT_YAML_CONFIG_HOST" >&2; exit 1; }

mkdir -p "$LOG_DIR"

echo "--- LAUNCH CONFIGURATION ---"
echo "PROJECT_ROOT:           $PROJECT_ROOT"
echo "CARLA_SIF:              $CARLA_SIF"
echo "AGENT_TYPE:             $AGENT_TYPE"
echo "TEAM_AGENT (container): $TEAM_AGENT"
echo "TEAM_CONFIG (container):$TEAM_CONFIG"
echo "LOG_DIR:                $LOG_DIR"
echo "WEATHER:                $WEATHER_START..$WEATHER_END"
echo "GPUS:                   $GPUS"
echo "------------------------------------"

# Weather and routes
WEATHER_NAMES=(
  ClearNoon CloudyNoon WetNoon WetCloudyNoon MidRainyNoon HardRainNoon SoftRainNoon
  ClearSunset CloudySunset WetSunset WetCloudySunset MidRainySunset HardRainSunset SoftRainSunset
)
ROUTE_FILES=(
  "routes_town01_short.xml" "routes_town01_long.xml" "routes_town02_short.xml" "routes_town02_long.xml"
  "routes_town03_short.xml" "routes_town03_long.xml" "routes_town04_short.xml" "routes_town04_long.xml"
)
TOWN_NUMBERS=( "01" "01" "02" "02" "03" "03" "04" "04" )

# ---------- Build per-GPU launchers ----------
for (( GPU_ID=0; GPU_ID<GPUS; GPU_ID++ )); do
    WEATHER_IDX=$((WEATHER_START + GPU_ID))
    if (( WEATHER_IDX > WEATHER_END || WEATHER_IDX >= ${#WEATHER_NAMES[@]} )); then
        continue
    fi

    ROUTE_IDX=$(( GPU_ID % ${#ROUTE_FILES[@]} ))
    RPC_PORT=$((BASE_RPC_PORT + GPU_ID * 10))
    TM_PORT=$((BASE_TM_PORT + GPU_ID * 10))
    ROUTE_FILE="${ROUTE_FILES[$ROUTE_IDX]}"
    TOWN_NUM="${TOWN_NUMBERS[$ROUTE_IDX]}"

    LAUNCH_SH="$LOG_DIR/launch_gpu${GPU_ID}.sh"
    cat >"$LAUNCH_SH" <<EOF
#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID}

singularity exec --nv --bind "${PROJECT_ROOT}":/workspace "${CARLA_SIF}" bash -s <<'CARLA_SCRIPT'
set -euo pipefail

export CARLA_ROOT=\${CARLA_ROOT:-/home/carla}
export PYTHONPATH="\$(echo \${CARLA_ROOT}/PythonAPI/carla/dist/carla-*-py*-linux-x86_64.egg):\${PYTHONPATH}"
export PYTHONPATH="\${CARLA_ROOT}/PythonAPI/carla:\${PYTHONPATH}"
export PYTHONPATH="/workspace/scenario_runner:/workspace/leaderboard:/workspace:\${PYTHONPATH}"
export SCENARIO_RUNNER_ROOT="/workspace/scenario_runner"

export XDG_RUNTIME_DIR="/tmp/runtime-\$USER-gpu${GPU_ID}"
mkdir -p "\$XDG_RUNTIME_DIR"

# Launch CARLA (offscreen)
"\${CARLA_ROOT}/CarlaUE4.sh" -carla-rpc-port=${RPC_PORT} -carla-streaming-port=0 -nosound -quality-level=Epic &
CARLA_PID=\$!
echo "[GPU ${GPU_ID}] CARLA PID \$CARLA_PID; waiting 60s..."
sleep 60
if ! ps -p \$CARLA_PID >/dev/null; then
    echo "[GPU ${GPU_ID}] FATAL: CARLA process failed to start." >&2
    exit 1
fi

ROUTE_FILE_NAME="${ROUTE_FILE}"
ROUTE_NAME_NO_EXT="\${ROUTE_FILE_NAME%.xml}"
SAVE_PATH="/workspace/dataset/agent-${AGENT_TYPE}/weather-${WEATHER_IDX}/\${ROUTE_NAME_NO_EXT}"
mkdir -p "\$SAVE_PATH"

export ROUTES="/workspace/leaderboard/data/training_routes/${ROUTE_FILE}"
export SCENARIOS="/workspace/leaderboard/data/scenarios/town${TOWN_NUM}_all_scenarios.json"
export TEAM_AGENT="${TEAM_AGENT}"
export TEAM_CONFIG="${TEAM_CONFIG}"
export CHECKPOINT_ENDPOINT="\$SAVE_PATH/results.json"
export SAVE_PATH
export CONSOLIDATED_AGENT="true"

[ -f "\$ROUTES" ] || { echo "[GPU ${GPU_ID}] FATAL: Route file not found: \$ROUTES" >&2; exit 1; }
[ -f "\$SCENARIOS" ] || { echo "[GPU ${GPU_ID}] FATAL: Scenario file not found: \$SCENARIOS" >&2; exit 1; }
[ -f "\$TEAM_AGENT" ] || { echo "[GPU ${GPU_ID}] FATAL: consolidated agent not found: \$TEAM_AGENT" >&2; exit 1; }
[ -f "\$TEAM_CONFIG" ] || { echo "[GPU ${GPU_ID}] FATAL: agent config not found: \$TEAM_CONFIG" >&2; exit 1; }

echo "[GPU ${GPU_ID}] Starting leaderboard evaluation..."
echo "  Agent wrapper: \$TEAM_AGENT"
echo "  Agent config:  \$TEAM_CONFIG"
echo "  Save path:     \$SAVE_PATH"

python /workspace/leaderboard/leaderboard/leaderboard_evaluator.py \
    --agent="\$TEAM_AGENT" \
    --agent-config="\$TEAM_CONFIG" \
    --routes="\$ROUTES" \
    --scenarios="\$SCENARIOS" \
    --checkpoint="\$CHECKPOINT_ENDPOINT" \
    --port=${RPC_PORT} \
    --trafficManagerPort=${TM_PORT} \
    --host=localhost \
    --timeout=60 --debug=1 --record="\$SAVE_PATH"

echo "[GPU ${GPU_ID}] Evaluation completed."
kill \$CARLA_PID || true
CARLA_SCRIPT
EOF
    chmod +x "$LAUNCH_SH"
done

# ---------- Launch and wait ----------
echo -e "\n--- LAUNCHING ALL INSTANCES ---"
PIDS=()
for (( GPU_ID=0; GPU_ID<GPUS; GPU_ID++ )); do
    WEATHER_IDX=$((WEATHER_START + GPU_ID))
    if (( WEATHER_IDX > WEATHER_END || WEATHER_IDX >= ${#WEATHER_NAMES[@]} )); then
        continue
    fi
    OUT="$LOG_DIR/gpu${GPU_ID}_w${WEATHER_IDX}.out"
    ERR="$LOG_DIR/gpu${GPU_ID}_w${WEATHER_IDX}.err"
    "$LOG_DIR/launch_gpu${GPU_ID}.sh" >"$OUT" 2>"$ERR" &
    PID=$!
    PIDS+=($PID)
    echo "  Launched GPU $GPU_ID (Weather $WEATHER_IDX) PID $PID"
    sleep 5
done

echo -e "\n--- WAITING FOR COMPLETION ---"
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "SUCCESS: Worker $i (PID $PID) completed."
    else
        EXIT_CODE=$?
        echo "FAILURE: Worker $i (PID $PID) exit code $EXIT_CODE."
        ERR_LOG="$LOG_DIR/gpu${i}_w$((WEATHER_START + i)).err"
        if [ -f "$ERR_LOG" ]; then
            echo "--- Last 20 lines ($ERR_LOG) ---"
            tail -n 20 "$ERR_LOG" | sed 's/^/    /'
        fi
    fi
done

echo -e "\nAll instances completed."
echo "Data saved under: $PROJECT_ROOT/dataset/agent-$AGENT_TYPE/"
