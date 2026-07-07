#!/usr/bin/env bash
# Focused follow-up: is there an NVIDIA userspace/kernel version MISMATCH?
# The host has both 515.86.01 and 575.57.08 GL libs; the kernel driver is one of
# them. GL context creation segfaults if the loaded libEGL_nvidia/libGLX_nvidia
# entry point doesn't match the running kernel module.
set +e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${CARLA_SIF:-$HERE/carla_official.sif}"

echo "== running KERNEL driver version =="
cat /proc/driver/nvidia/version 2>/dev/null
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1

echo ""
echo "== HOST GL entry-point SONAME -> which version? =="
for l in libEGL_nvidia.so.0 libGLX_nvidia.so.0 libGLESv2_nvidia.so.2; do
  f="/usr/lib/x86_64-linux-gnu/$l"
  printf "  %-24s -> %s\n" "$l" "$(readlink -f "$f" 2>/dev/null || echo '(plain file / none)')"
done

echo ""
echo "== IN-CONTAINER (--nv): what libEGL/libGLX entry points resolve to =="
singularity exec --nv "$SIF" bash -lc '
  for l in libEGL_nvidia.so.0 libGLX_nvidia.so.0 libGLESv2_nvidia.so.2; do
    for d in /.singularity.d/libs /usr/lib/x86_64-linux-gnu; do
      f="$d/$l"
      if [ -e "$f" ]; then
        tgt=$(readlink "$f" 2>/dev/null); [ -z "$tgt" ] && tgt="(plain file, size $(stat -c%s "$f" 2>/dev/null))"
        printf "  %-42s -> %s\n" "$f" "$tgt"
      fi
    done
  done
  echo "-- both core versions really injected? --"
  ls -1 /.singularity.d/libs | grep -E "glcore|eglcore" | sort
'
