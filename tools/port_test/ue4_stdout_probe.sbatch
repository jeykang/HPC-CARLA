#!/bin/bash
# Goal: surface UE4's actual error. Two angles:
#   (1) compare HOST kernel driver version vs the lib versions Singularity
#       bound (we saw both 515.86.01 and 575.57.08 in /.singularity.d/libs).
#       Mismatch is a known cause of "GPU init dies at t=5s with no log".
#   (2) force UE4 to flush its log to stdout (-log -stdout -FullStdOutLogOutput
#       -ForceLogFlush -unattended), and *also* bind ONLY Saved/Logs over a
#       writable tmpdir (not all of /home/carla) so UE4's own log file is
#       captured as backup.
#
# Usage: sbatch -w hpc-pr-a-pod09 ue4_stdout_probe.sbatch

#SBATCH --job-name=ue4_stdout
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:08:00
#SBATCH --output=ue4_stdout_probe_%A_%N.out
#SBATCH --error=ue4_stdout_probe_%A_%N.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/autodr_test/HPC-CARLA-persistent}"
SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"
NODE="${SLURMD_NODENAME:-$(hostname)}"
SAVEDIR="$(mktemp -d /tmp/ue4_saved.XXXXXX)"
mkdir -p "${SAVEDIR}/Logs" "${SAVEDIR}/Config"

echo "=========================================="
echo "Host driver vs container-lib version + UE4 stdout capture"
echo "Node : ${NODE}    SIF : ${SIF}    Saved bind : ${SAVEDIR}"
echo "Date : $(date -Is)"
echo "=========================================="

echo
echo "############### Driver version comparison ###############"
echo "--- HOST /proc/driver/nvidia/version ---"
cat /proc/driver/nvidia/version 2>&1 || echo "(host /proc/driver/nvidia/version unreadable from this node)"
echo
echo "--- HOST nvidia-smi (driver line) ---"
nvidia-smi 2>&1 | head -5 || echo "(no host nvidia-smi)"
echo
echo "--- Container-bound NVIDIA lib versions ---"
ls /.singularity.d/libs/ 2>/dev/null | grep -oE '\.so(\.[0-9.]+)+' | sort -u
echo

# The point of this run:
#  Bind ONLY Saved/ as writable so UE4 can write its log file there. The rest
#  of /home/carla (including CarlaUE4.sh and the read-only assets) stays as-is.
#  And force the engine to dump logs to stdout too.
run_carla() {
    local label="$1"; shift
    local rhi="$1"
    local rpc=$(( 56000 + RANDOM % 1000 ))
    echo
    echo "############### CARLA [${label}/${rhi}] rpc=${rpc} ###############"
    local out; out="$(mktemp /tmp/ue4_${label}_${rhi}_XXXX.out)"

    # Bind a writable dir over CARLA's Saved/ ONLY. Path inside container is
    # the standard UE4 location. CarlaUE4.sh, binaries, content remain SIF-native.
    (
        singularity run --nv \
            -B "${SAVEDIR}:/home/carla/CarlaUE4/Saved" \
            "${SIF}" \
            -RenderOffScreen "-${rhi}" \
            -log -stdout -FullStdOutLogOutput -ForceLogFlush -unattended -nosplash \
            -carla-rpc-port="${rpc}"
    ) >"${out}" 2>&1 &
    local pid=$!
    local up=0 i
    for i in $(seq 1 18); do
        sleep 5
        if ss -ltn 2>/dev/null | grep -q ":${rpc}\b"; then up=1; break; fi
        if ! kill -0 "${pid}" 2>/dev/null; then echo "  (process died at t=$((i*5))s)"; break; fi
        echo "  t=$((i*5))s : not listening"
    done
    if [[ "${up}" == "1" ]]; then
        echo "RESULT[${label}/${rhi}]: rpc_port_up=YES at t<=$((i*5))s"
    else
        echo "RESULT[${label}/${rhi}]: rpc_port_up=NO"
    fi
    echo "--- container stdout/stderr (last 80) ---"
    tail -n 80 "${out}" 2>/dev/null
    echo "--- Saved/Logs/ (UE4's own log dir) ---"
    ls -la "${SAVEDIR}/Logs" 2>/dev/null
    for f in "${SAVEDIR}/Logs"/*.log; do
        [[ -e "$f" ]] || continue
        echo "----- ${f} (last 80 lines) -----"
        tail -n 80 "$f"
    done
    # Drain Saved between runs so each one is isolated
    rm -f "${SAVEDIR}"/Logs/*.log "${SAVEDIR}"/Crashes/*.log 2>/dev/null

    kill -TERM "${pid}" 2>/dev/null || true; sleep 2
    kill -KILL "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
}

# Definitive dlopen test — fixed quoting (the previous probe's heredoc was eaten).
echo
echo "############### dlopen test (fixed) ###############"
singularity exec --nv "${SIF}" python3 -c '
import ctypes, os
for name in ["libGLX_nvidia.so.0","libEGL_nvidia.so.0","libnvidia-ml.so.1",
             "libnvidia-glcore.so.575.57.08","libnvidia-glcore.so.515.86.01",
             "libcuda.so.1","libvulkan.so.1","libGL.so.1"]:
    try:
        ctypes.CDLL(name)
        print("  dlopen OK :", name)
    except OSError as e:
        print("  dlopen FAIL:", name, "->", e)
'

run_carla "saved_bind" "opengl"
run_carla "saved_bind" "vulkan"
run_carla "saved_bind" "nullrhi"

rm -rf "${SAVEDIR}"
echo
echo "############### Done $(date -Is) ###############"
