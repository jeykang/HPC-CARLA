#!/usr/bin/env bash
# Decisive NVIDIA GL/EGL driver-lib probe for the CARLA Signal-11 investigation.
#
# The earlier `ldconfig -p` probe was misleading: --nv injects driver libs into
# /.singularity.d/libs (on LD_LIBRARY_PATH), which is NOT in the ld.so cache, so
# ldconfig reports them MISSING whether or not they are present. This checks the
# real locations, and whether CARLA is silently on Mesa software GL (which would
# explain both the slowdown and the crashes).
#
# Run on a compute node:
#   srun --nodes=1 --nodelist=hpc-pr-a-pod20 --gres=gpu:1 --time=00:05:00 bash tools/gl_probe.sh
set +e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${CARLA_SIF:-$HERE/carla_official.sif}"

echo "==================== HOST ($(hostname)) ===================="
echo "-- NVIDIA GL/EGL userspace libs present on the host (bind candidates) --"
for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
  ls -1 "$d" 2>/dev/null \
    | grep -iE "libEGL_nvidia|libGLX_nvidia|libnvidia-glcore|libnvidia-eglcore|libnvidia-glsi|libGLESv2_nvidia|libnvidia-glvkspirv" \
    | sed "s|^|  $d/|"
done
echo "-- nvidia-smi --"
nvidia-smi -L 2>&1 | head -3

echo ""
echo "==================== INSIDE CONTAINER (singularity exec --nv) ===================="
singularity exec --nv "$SIF" bash -lc '
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "-- GL/EGL libs --nv injected into /.singularity.d/libs --"
  ls -1 /.singularity.d/libs 2>/dev/null | grep -iE "EGL|GLX|glcore|eglcore|nvidia|GL" || echo "  (nothing GL-related injected)"
  echo "-- authoritative presence check (find across lib dirs) --"
  for l in libEGL_nvidia.so libGLX_nvidia.so libnvidia-glcore.so libnvidia-eglcore.so libEGL.so.1 libGL.so.1 libOpenGL.so.0; do
    hit=$(find /usr/lib /lib /.singularity.d -name "${l}*" 2>/dev/null | head -1)
    printf "  %-22s %s\n" "$l" "${hit:-NOT-FOUND}"
  done
  echo "-- Mesa software-GL fallback present? (llvmpipe/swrast => slow-but-working render) --"
  find /usr/lib /lib -iname "*swrast*" -o -iname "*llvmpipe*" 2>/dev/null | head -3 || true
  echo "-- what libGL.so.1 actually resolves to (Mesa vs NVIDIA) --"
  find /usr/lib /lib /.singularity.d -name "libGL.so.1*" 2>/dev/null | head -3
'
