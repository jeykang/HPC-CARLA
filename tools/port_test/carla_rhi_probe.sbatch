#!/bin/bash
# Second-stage diagnostic: TCP is fine (port_test passed), so isolate the
# CARLA/UE4 init path. We:
#   (1) enumerate the GPU userspace stack as exposed by nvidia-container-cli
#       (/dev/nvidia*, libGLX_nvidia, libEGL_nvidia, libvulkan, vulkaninfo, eglinfo)
#   (2) run CARLA in foreground with verbose UE4 logging redirected to stdout,
#       and concurrently poll the RPC port from the host every 5s for 90s.
#       Tries OpenGL first, then Vulkan, then no-renderer (-nullrhi) to see
#       which init paths actually progress.
#
# Usage:  sbatch -w hpc-pr-a-pod09 carla_rhi_probe.sbatch
# Output: carla_rhi_probe_<jobid>_<node>.{out,err}

#SBATCH --job-name=carla_rhi
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=carla_rhi_probe_%A_%N.out
#SBATCH --error=carla_rhi_probe_%A_%N.err

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/autodr_test/HPC-CARLA-persistent}"
SIF="${CARLA_SIF:-${PROJECT_ROOT}/carla_official.sif}"
NODE="${SLURMD_NODENAME:-$(hostname)}"
PORT_BASE=$(( 53000 + RANDOM % 1000 ))   # avoid CARLA prod range

echo "=========================================="
echo "CARLA / UE4 RHI init diagnostic"
echo "Node : ${NODE}    SIF : ${SIF}    base port : ${PORT_BASE}"
echo "Date : $(date -Is)"
echo "=========================================="

probe_stack() {
    echo
    echo "############### GPU userspace stack ###############"
    singularity exec --nv "${SIF}" bash -lc '
        echo "--- /dev/nvidia* ---"
        ls -la /dev/nvidia* 2>&1 | head
        echo
        echo "--- /dev/dri ---"
        ls -la /dev/dri 2>&1
        echo
        echo "--- nvidia-container-cli would expose these (libs visible inside): ---"
        for n in libGLX_nvidia libEGL_nvidia libvulkan libGL.so libEGL.so libGLX.so libnvidia-glcore libnvidia-eglcore libnvidia-rtcore; do
            echo "[$n]"
            ldconfig -p | grep -i "$n" || echo "  (none)"
        done
        echo
        echo "--- vulkaninfo ---"
        if command -v vulkaninfo >/dev/null 2>&1; then
            vulkaninfo --summary 2>&1 | head -40 || vulkaninfo 2>&1 | head -40
        else
            echo "(vulkaninfo not present in container)"
        fi
        echo
        echo "--- eglinfo / glxinfo ---"
        for tool in eglinfo glxinfo; do
            if command -v "$tool" >/dev/null 2>&1; then
                echo "[$tool]"; "$tool" 2>&1 | head -20
            else
                echo "($tool not present)"
            fi
        done
    '
}

run_carla_with_rhi() {
    local label="$1"; shift
    local extra_flags="$*"
    local rpc=$(( PORT_BASE++ ))
    echo
    echo "############### CARLA run [${label}] rpc=${rpc} flags='${extra_flags}' ###############"
    # Start CARLA in background with UE4 verbose logging.
    # Use `singularity run` (matches project) but force -log so UE4 actually emits init lines.
    local carla_log
    carla_log="$(mktemp /tmp/carla_${label}_XXXX.log)"
    (
        singularity run --nv "${SIF}" \
            -log -RenderOffScreen \
            -carla-rpc-port="${rpc}" \
            ${extra_flags}
    ) >"${carla_log}" 2>&1 &
    local pid=$!
    echo "carla pid=${pid} log=${carla_log}"

    # Poll for port up to 90s.
    local up=0
    for i in $(seq 1 18); do
        sleep 5
        if ss -ltn 2>/dev/null | grep -q ":${rpc}\b"; then
            up=1; break
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "  (process died at t=$((i*5))s)"; break
        fi
        echo "  t=$((i*5))s : not listening yet"
    done

    if [[ "${up}" == "1" ]]; then
        echo "RESULT[${label}]: rpc_port_up=YES at t<=$((i*5))s"
    else
        echo "RESULT[${label}]: rpc_port_up=NO"
    fi

    echo "--- last 60 lines of CARLA log ---"
    tail -n 60 "${carla_log}" 2>/dev/null
    echo "--- end CARLA log (full saved at ${carla_log}) ---"

    # Tear down
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

probe_stack

# Three init paths: OpenGL (project default), Vulkan, NullRHI (no renderer).
# NullRHI bypasses GPU entirely — if even THIS fails to bind RPC, the issue is
# below UE4's renderer choice. If NullRHI works but OpenGL/Vulkan don't, the
# problem is purely in the GPU init path.
run_carla_with_rhi "opengl"  "-opengl"
run_carla_with_rhi "vulkan"  "-vulkan"
run_carla_with_rhi "nullrhi" "-nullrhi"

echo
echo "############### Done $(date -Is) ###############"
