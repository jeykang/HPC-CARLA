#!/bin/bash
# Sharper diagnostic. Two hypotheses to discriminate:
#
#   H1: NVIDIA graphics libs are *not* visible inside container
#       (ldconfig view was a script bug; check the real bind path).
#   H2: /home/carla is read-only inside SIF; UE4 dies trying to write its
#       Saved/Logs/Saved/Config. Mount a writable HOME and retry.
#
# Also: capture UE4's *own* log file (Saved/Logs/CarlaUE4.log) which is where
# UE4 writes real error messages — Singularity stdout shows nothing useful.
#
# Usage: sbatch -w hpc-pr-a-pod09 lib_and_log_probe.sbatch

#SBATCH --job-name=lib_log_probe
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:08:00
#SBATCH --output=lib_and_log_probe_%A_%N.out
#SBATCH --error=lib_and_log_probe_%A_%N.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/autodr_test/HPC-CARLA-persistent}"
SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"
NODE="${SLURMD_NODENAME:-$(hostname)}"
TMPHOME="$(mktemp -d /tmp/carla_home.XXXXXX)"

echo "=========================================="
echo "Lib visibility + UE4 log capture probe"
echo "Node : ${NODE}    SIF : ${SIF}    TMPHOME : ${TMPHOME}"
echo "Date : $(date -Is)"
echo "=========================================="

echo
echo "############### H1: where Singularity ACTUALLY puts NVIDIA libs ###############"
singularity exec --nv "${SIF}" bash -lc '
    echo "--- LD_LIBRARY_PATH ---"; echo "${LD_LIBRARY_PATH:-(unset)}"
    echo
    echo "--- /.singularity.d/libs/ (where --nv binds host NVIDIA libs) ---"
    ls -la /.singularity.d/libs/ 2>&1 | head -50
    echo
    echo "--- find /usr -name libGLX_nvidia* / libEGL_nvidia* / libnvidia-* (capped) ---"
    find /usr /.singularity.d -maxdepth 6 \( -name "libGLX_nvidia*" -o -name "libEGL_nvidia*" -o -name "libnvidia-*" -o -name "libcuda*" \) 2>/dev/null | head -40
    echo
    echo "--- dlopen test (definitive — does the dynamic loader find them?) ---"
    python3 - <<PY
import ctypes, ctypes.util, os
for name in ["libGLX_nvidia.so.0","libEGL_nvidia.so.0","libnvidia-ml.so.1","libnvidia-glcore.so","libcuda.so.1","libvulkan.so.1","libGL.so.1"]:
    try:
        h = ctypes.CDLL(name, mode=ctypes.RTLD_LAZY|os.RTLD_GLOBAL if hasattr(os,"RTLD_GLOBAL") else ctypes.RTLD_LAZY)
        print(f"  dlopen OK : {name}")
    except OSError as e:
        print(f"  dlopen FAIL: {name}  -> {e}")
PY
    echo
    echo "--- Vulkan ICD JSON content ---"
    cat /etc/vulkan/icd.d/nvidia_icd.json 2>&1 | head -30
'

run_carla() {
    local label="$1"; shift
    local extra_singularity="$1"; shift
    local extra_carla="$1"
    local rpc=$(( 55000 + RANDOM % 1000 ))
    echo
    echo "############### CARLA [${label}] rpc=${rpc} ###############"
    echo "  singularity flags : ${extra_singularity}"
    echo "  carla flags       : ${extra_carla}"
    local carla_log; carla_log="$(mktemp /tmp/carla_${label}_XXXX.stdouterr)"

    # Pre-create UE4 saved dirs in the writable HOME so any -B /home/carla path works.
    mkdir -p "${TMPHOME}/CarlaUE4/Saved/Logs" "${TMPHOME}/CarlaUE4/Saved/Config"

    (
        # shellcheck disable=SC2086
        singularity run --nv ${extra_singularity} "${SIF}" \
            ${extra_carla} -RenderOffScreen -carla-rpc-port="${rpc}"
    ) >"${carla_log}" 2>&1 &
    local pid=$!
    local up=0 i
    for i in $(seq 1 18); do
        sleep 5
        if ss -ltn 2>/dev/null | grep -q ":${rpc}\b"; then up=1; break; fi
        if ! kill -0 "${pid}" 2>/dev/null; then echo "  (process died at t=$((i*5))s)"; break; fi
        echo "  t=$((i*5))s : not listening"
    done
    if [[ "${up}" == "1" ]]; then
        echo "RESULT[${label}]: rpc_port_up=YES at t<=$((i*5))s"
    else
        echo "RESULT[${label}]: rpc_port_up=NO"
    fi

    echo "--- container stdout/stderr (last 30) ---"
    tail -n 30 "${carla_log}" 2>/dev/null

    # The interesting log: UE4's own log file. Try common paths.
    echo "--- UE4 Saved/Logs (looking for CarlaUE4.log) ---"
    for d in "${TMPHOME}/CarlaUE4/Saved/Logs"; do
        ls -la "$d" 2>/dev/null
        for f in "$d"/*.log; do
            [[ -e "$f" ]] || continue
            echo "----- ${f} (last 60 lines) -----"
            tail -n 60 "$f"
        done
    done

    kill -TERM "${pid}" 2>/dev/null || true; sleep 2
    kill -KILL "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
}

# A: baseline (matches project today)
run_carla "baseline_opengl"        ""                                       "-opengl"
# B: writable HOME (H2 test). CARLA writes to /home/carla/CarlaUE4/Saved/...
run_carla "writable_home_opengl"   "-B ${TMPHOME}:/home/carla"             "-opengl"
# C: writable HOME + vulkan
run_carla "writable_home_vulkan"   "-B ${TMPHOME}:/home/carla"             "-vulkan"
# D: writable HOME + writable tmpfs (in case other paths need write too)
run_carla "writable_tmpfs_opengl"  "--writable-tmpfs -B ${TMPHOME}:/home/carla" "-opengl"

# Cleanup home dir
rm -rf "${TMPHOME}"
echo
echo "############### Done $(date -Is) ###############"
