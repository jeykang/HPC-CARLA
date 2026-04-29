#!/bin/bash
# Hypothesis test: legacy --nv on this cluster doesn't bind NVIDIA graphics
# driver libs (libGLX_nvidia, libEGL_nvidia, libnvidia-glcore...) into the
# container, even though /dev/nvidia* + libcuda are fine. Newer --nvccli
# (NVIDIA Container Toolkit) typically does. Confirm by re-running the same
# probes with --nvccli and checking whether the missing libs appear and
# whether OpenGL CARLA can now bind its RPC port.
#
# Usage: sbatch -w hpc-pr-a-pod09 nvccli_probe.sbatch

#SBATCH --job-name=nvccli_probe
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:08:00
#SBATCH --output=nvccli_probe_%A_%N.out
#SBATCH --error=nvccli_probe_%A_%N.err

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/autodr_test/HPC-CARLA-persistent}"
SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"
NODE="${SLURMD_NODENAME:-$(hostname)}"

echo "=========================================="
echo "Singularity --nvccli vs --nv comparison"
echo "Node : ${NODE}    SIF : ${SIF}"
echo "Date : $(date -Is)"
echo "=========================================="

dump_libs() {
    local label="$1"; shift   # "nv" or "nvccli"
    local nvflag="$1"; shift   # "--nv" or "--nvccli"
    echo
    echo "############### libs visible with ${nvflag} (${label}) ###############"
    if ! singularity exec ${nvflag} "${SIF}" true 2>/tmp/nvflag_err.$$; then
        echo "FAILED to invoke singularity with ${nvflag}:"
        cat /tmp/nvflag_err.$$; rm -f /tmp/nvflag_err.$$
        return 1
    fi
    rm -f /tmp/nvflag_err.$$
    singularity exec ${nvflag} "${SIF}" bash -lc '
        echo "--- /dev/nvidia* (full) ---"
        ls -la /dev/nvidia* 2>&1
        echo
        echo "--- key NVIDIA driver libs ---"
        for n in libGLX_nvidia libEGL_nvidia libvulkan_nvidia libnvidia-glcore libnvidia-eglcore libnvidia-rtcore libnvidia-ml libcuda; do
            echo "[$n]"
            ldconfig -p | grep -i "$n" | head -3 || echo "  (none)"
        done
        echo
        echo "--- vulkan ICD search path ---"
        ls -la /etc/vulkan/icd.d 2>&1 | head
        ls -la /usr/share/vulkan/icd.d 2>&1 | head
    '
}

run_carla() {
    local label="$1"; shift   # "nv" or "nvccli"
    local nvflag="$1"; shift   # "--nv" or "--nvccli"
    local rhi="$1"            # "opengl" or "vulkan"
    local rpc=$(( 54000 + RANDOM % 1000 ))
    echo
    echo "############### CARLA [${label}/${rhi}] rpc=${rpc} ###############"
    local carla_log; carla_log="$(mktemp /tmp/carla_${label}_${rhi}_XXXX.log)"
    (
        singularity run ${nvflag} "${SIF}" \
            -log -RenderOffScreen "-${rhi}" -carla-rpc-port="${rpc}"
    ) >"${carla_log}" 2>&1 &
    local pid=$!
    local up=0
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
    echo "--- last 40 lines ---"
    tail -n 40 "${carla_log}" 2>/dev/null
    kill -TERM "${pid}" 2>/dev/null || true; sleep 2
    kill -KILL "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
}

dump_libs "nv-baseline" "--nv"
dump_libs "nvccli"      "--nvccli"

run_carla "nv-baseline" "--nv"     "opengl"
run_carla "nvccli"      "--nvccli" "opengl"
run_carla "nvccli"      "--nvccli" "vulkan"

echo
echo "############### Done $(date -Is) ###############"
