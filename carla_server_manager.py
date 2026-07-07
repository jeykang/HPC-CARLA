#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA Server Manager (persistent pool)

Commands:
  start   -- launch a per-GPU server pool
  stop    -- stop anything we launched (best-effort)
  health  -- quick status of known servers (by port reachability)
  ensure  -- idempotently ensure a server exists for ONE gpu (used by workers)
"""

import os, sys, json, time, socket, signal, subprocess, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
STATE_DIR    = Path(os.environ.get("STATE_DIR", PROJECT_ROOT / "collection_state"))
LOG_DIR      = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BASE      = int(os.environ.get("BASE_RPC_PORT", 2000))
DEFAULT_SPACING   = int(os.environ.get("PORT_SPACING", 100))
DEFAULT_TM_OFFSET = int(os.environ.get("TM_OFFSET", 5000))

# Use container’s default CARLA_ROOT (/home/carla per your .def) via %environment/%runscript.
SIF_PATH   = str(os.environ.get("CARLA_SIF", PROJECT_ROOT / "carla_official.sif"))
NODE_NAME  = os.environ.get("SLURMD_NODENAME", os.uname().nodename)

STATE_FILE = STATE_DIR / f"carla_servers_{NODE_NAME}.json"

def _read_state() -> Dict:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"node": NODE_NAME, "servers": {}}

def _write_state(state: Dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)

def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False

def wait_for_port(host: str, port: int, deadline: float) -> bool:
    while time.time() < deadline:
        if is_port_open(host, port):
            return True
        time.sleep(0.2)
    return False

def _proc_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False

def _pids_on_port(rpc: int) -> List[int]:
    """PIDs of container/CARLA processes launched for this RPC port, matched on
    the exact -carla-rpc-port=<rpc> argv token (verified via /proc) so a sibling
    GPU's server on another port is never touched."""
    token = f"-carla-rpc-port={rpc}".encode()
    try:
        out = subprocess.check_output(["pgrep", "-f", f"carla-rpc-port={rpc}"],
                                      text=True, stderr=subprocess.DEVNULL)
        cand = [int(x) for x in out.split()]
    except Exception:
        cand = []
    pids, me = [], os.getpid()
    for pid in cand:
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if token in f.read().split(b"\0"):
                    pids.append(pid)
        except Exception:
            pass
    return pids

def _kill_gpu_server(gpu_id: int, rpc: int, grace: float = 6.0) -> None:
    """Kill any server bound to this GPU's RPC port and wait for the port to
    free. A segfaulted CARLA can leave a child holding GPU memory or the socket;
    relaunching over it makes the new server's GL init crash (the unrecoverable
    crash-loop). Matches only this port, so sibling GPUs are untouched."""
    if not _pids_on_port(rpc) and not is_port_open("127.0.0.1", rpc):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = _pids_on_port(rpc)
        if not pids:
            break
        for pid in pids:
            try:
                os.killpg(os.getpgid(pid), sig)   # server has its own group
            except Exception:
                try: os.kill(pid, sig)
                except Exception: pass
        deadline = time.time() + grace
        while time.time() < deadline and _pids_on_port(rpc):
            time.sleep(0.3)
    t = time.time() + 5
    while time.time() < t and is_port_open("127.0.0.1", rpc):
        time.sleep(0.3)

def _wait_healthy(rpc: int, pid: Optional[int], timeout: int) -> bool:
    """Healthy = the RPC port opens AND the process is still alive a few seconds
    later. Catches the 'segfault right after binding the port' case that plain
    wait_for_port() would wrongly report as up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pid is not None and not _proc_alive(pid):
            return False
        if is_port_open("127.0.0.1", rpc):
            break
        time.sleep(0.5)
    else:
        return False
    stable_until = time.time() + 3
    while time.time() < stable_until:
        if pid is not None and not _proc_alive(pid):
            return False
        time.sleep(0.5)
    return is_port_open("127.0.0.1", rpc)

# --- GPU parking: after repeated boot failures, stop assigning jobs to a GPU ---
PARK_DIR = STATE_DIR / "parked"
def _park_path(gpu_id: int) -> Path:
    return PARK_DIR / f"{NODE_NAME}_gpu{gpu_id}.parked"
def _park_gpu(gpu_id: int, reason: str) -> None:
    PARK_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _park_path(gpu_id).write_text(json.dumps({"reason": reason, "at": ts}))
def _clear_park(gpu_id: int) -> None:
    try: _park_path(gpu_id).unlink()
    except Exception: pass

def discover_gpus() -> List[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        mapping = [x for x in cvd.split(",") if x.strip() != ""]
        return list(range(len(mapping)))
    n = int(os.environ.get("SLURM_GPUS_ON_NODE", os.environ.get("GPUS_PER_NODE", "8")))
    return list(range(n))

def _derive_ports(gpu_id: int, base: int, spacing: int, tm_off: int) -> Tuple[int,int]:
    rpc = base + gpu_id*spacing
    tm  = rpc + tm_off
    return rpc, tm

def _gpu_home(gpu_id: int) -> Path:
    """Isolated, writable HOME for one GPU's CARLA/UE4 instance.

    Singularity bind-mounts the host $HOME into every container by default, so
    all per-GPU servers would otherwise share ~/.config/Epic, ~/.cache (the GL
    shader cache) and lock files -- concurrent writes there cause correlated
    boot lockups on a multi-GPU node. Give each GPU its own directory instead."""
    return STATE_DIR / "ue4_home" / f"gpu{gpu_id}"

def _container_env_for_gpu(gpu_id: int) -> Dict[str, str]:
    """
    Pass GPU selection and headless SDL into the container.
    NOTE: We do NOT set CARLA_ROOT here — container's %environment handles it.
    """
    env = os.environ.copy()
    env.update({
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES": str(gpu_id),
        "SINGULARITYENV_NVIDIA_VISIBLE_DEVICES": str(gpu_id),
        "SINGULARITYENV_SDL_VIDEODRIVER": "offscreen",
        "SINGULARITYENV_SDL_AUDIODRIVER": "dummy",
        "SINGULARITYENV_DISABLE_PYTHON": "1",   # CARLA binary only
        # GLVND EGL vendor registration: the ubuntu:20.04 SIF base doesn't
        # include the NVIDIA EGL vendor file (that came from the CUDA Docker
        # base in the old carlasim image). Without it libEGL.so.1 (GLVND)
        # finds no EGL implementation → UE4 OpenGL RHI crashes at context
        # creation. Point directly at the file in /workspace (always bound).
        "SINGULARITYENV___EGL_VENDOR_LIBRARY_FILENAMES": "/workspace/nvidia_egl_vendor.json",
        # Vulkan ICD: same problem — nvidia_icd.json was baked into the old
        # carlasim image but absent from the new ubuntu base.
        "SINGULARITYENV_VK_ICD_FILENAMES": "/workspace/nvidia_icd.json",
        # Mirror for Apptainer
        "APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES": "/workspace/nvidia_egl_vendor.json",
        "APPTAINERENV_VK_ICD_FILENAMES": "/workspace/nvidia_icd.json",
        # Isolated per-GPU HOME/tmp/shader-cache (bound at /carla_home in
        # _build_run_args) so 8 UE4 instances don't collide on shared
        # ~/.config/Epic, ~/.cache and lock files -> correlated boot lockups.
        "SINGULARITYENV_HOME": "/carla_home",
        "SINGULARITYENV_TMPDIR": "/carla_home/tmp",
        "SINGULARITYENV_XDG_CACHE_HOME": "/carla_home/.cache",
        "APPTAINERENV_HOME": "/carla_home",
        "APPTAINERENV_TMPDIR": "/carla_home/tmp",
        "APPTAINERENV_XDG_CACHE_HOME": "/carla_home/.cache",
    })
    # Prevent core dumps inside container
    env["SINGULARITYENV_ULIMIT_CORE"] = "0"
    return env

def _find_gpucomp() -> Optional[str]:
    """Locate libnvidia-gpucomp.so.<driver> for the current node's NVIDIA driver.

    The filename embeds the driver version and the directory varies by node type
    (e.g. 575.57.08 under /usr/lib/x86_64-linux-gnu on L40S nodes, 580.82.07 under
    /cm/local/apps/cuda/libs/current/lib64 on H100 nodes), so we search the loader
    cache and known locations rather than hardcode a single path. Returns the
    absolute path, or None if not found.
    """
    import glob
    import subprocess
    # 1) ldconfig cache - most reliable; respects the node's configured lib paths.
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if "libnvidia-gpucomp.so" in line and "=>" in line:
                path = line.split("=>")[-1].strip()
                if os.path.exists(path):
                    return path
    except Exception:
        pass
    # 2) Fallback: glob known locations (L40S, H100/Bright-cluster, RHEL).
    for pattern in (
        "/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.*",
        "/cm/local/apps/cuda/libs/current/lib64/libnvidia-gpucomp.so.*",
        "/usr/lib64/libnvidia-gpucomp.so.*",
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def _build_run_args(gpu_id: int, rpc: int, tm: int) -> List[str]:
    """
    Use 'singularity run' so the container's %runscript invokes ${CARLA_ROOT}/CarlaUE4.sh.
    All UE4/CARLA flags are passed as args to the runscript.
    """
    args = [
        "singularity", "run", "--nv",
        "-B", f"{str(PROJECT_ROOT)}:/workspace",  # project tree at /workspace for sidecars
        "-B", f"{_gpu_home(gpu_id)}:/carla_home", # isolated per-GPU HOME (shader cache, locks)
    ]
    # The NVIDIA driver requires libnvidia-gpucomp.so but the cluster's --nv
    # (legacy nvliblist.conf) doesn't auto-bind it. Without this, UE4's GL/Vulkan
    # RHI dlopens libGLX_nvidia which then can't resolve gpucomp, and CARLA
    # dies before binding the RPC port. The filename embeds the driver version
    # and the directory varies by node type (L40S 575 vs H100 580), so we detect
    # it at runtime via _find_gpucomp() instead of hardcoding - this keeps the
    # pipeline portable across heterogeneous GPU partitions.
    gpucomp = _find_gpucomp()
    if gpucomp:
        args += ["-B", f"{gpucomp}:{gpucomp}"]
    # Additional maps (Town06, Town07, Town10HD) are not in the base SIF image.
    # A merged Maps directory (base towns + additional towns) lives at
    # PROJECT_ROOT/carla_maps/ and is bind-mounted over the SIF's Maps dir.
    EXTRA_MAPS = PROJECT_ROOT / "carla_maps"
    if EXTRA_MAPS.is_dir():
        args += ["-B", f"{EXTRA_MAPS}:/home/carla/CarlaUE4/Content/Carla/Maps"]
    # Renderer and quality are env-configurable so we can A/B them without code
    # edits while chasing the uniform Signal-11 segfaults. Defaults preserve the
    # historical launch (-opengl, Epic). Set CARLA_RENDER_FLAG="-vulkan" or ""
    # (empty => UE4's Vulkan default) and CARLA_QUALITY=Low to experiment.
    render_flag = os.environ.get("CARLA_RENDER_FLAG", "-opengl")
    quality     = os.environ.get("CARLA_QUALITY", "Epic")
    carla_args = [
        SIF_PATH,
        render_flag,
        "-RenderOffScreen",
        f"-quality-level={quality}",
    ]
    # Force UE4 to stream its FULL log (LogRHI/LogInit + the crash callstack) to
    # stdout, which we capture in carla_<node>_gpu<id>.log. Without this, UE4
    # logs only to the container's ephemeral Saved/Logs/CarlaUE4.log, which is
    # lost when the process segfaults -- so our capture showed only the crash
    # handler with zero fault context, making the crashes undiagnosable. Gated by
    # an env flag so it can be silenced once the segfaults are understood.
    if os.environ.get("CARLA_UE4_STDOUT", "1") == "1":
        carla_args += ["-stdout", "-FullStdOutLogOutput"]
    carla_args += [
        f"-carla-rpc-port={rpc}",
        f"-trafficManagerPort={tm}",
        "-carla-server",
    ]
    # Drop an empty render_flag token (CARLA_RENDER_FLAG="" => Vulkan default).
    args += [a for a in carla_args if a]
    return args

def start_one(gpu_id: int, rpc: int, tm: int) -> Optional[int]:
    log_path = LOG_DIR / f"carla_{NODE_NAME}_gpu{gpu_id}.log"
    # Clean slate: kill any lingering/segfaulted server still bound to this RPC
    # port before relaunching, so the new process doesn't inherit a fouled GPU
    # context or a held socket (the unrecoverable crash-loop we diagnosed).
    _kill_gpu_server(gpu_id, rpc)
    # Isolated, writable per-GPU HOME (shader cache / config / tmp) must exist.
    home = _gpu_home(gpu_id)
    for sub in ("tmp", ".cache", ".config"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "ab", buffering=0) as logf:
            proc = subprocess.Popen(
                _build_run_args(gpu_id, rpc, tm),
                stdout=logf, stderr=logf,
                env=_container_env_for_gpu(gpu_id),
                start_new_session=True,   # own process group -> clean, isolated kills
            )
        # Persist state
        state = _read_state()
        servers = state.setdefault("servers", {})
        servers[str(gpu_id)] = {
            "gpu": gpu_id, "rpc_port": rpc, "tm_port": tm,
            "pid": proc.pid, "node": NODE_NAME, "log": str(log_path),
        }
        _write_state(state)
        return proc.pid
    except Exception as e:
        print(f"[server_manager] failed to start gpu{gpu_id}: {e}", file=sys.stderr)
        return None

def start_pool(gpus: List[int], base: int, spacing: int, tm_off: int) -> Dict[str, Dict]:
    started = {}
    for gid in gpus:
        rpc, tm = _derive_ports(gid, base, spacing, tm_off)
        if is_port_open("127.0.0.1", rpc):
            started[str(gid)] = {"rpc_port": rpc, "tm_port": tm, "pid": None, "already_running": True}
            continue
        pid = start_one(gid, rpc, tm)
        ok = _wait_healthy(rpc, pid, 120)
        started[str(gid)] = {"rpc_port": rpc, "tm_port": tm, "pid": pid, "listening": ok}
    return started

def stop_pool() -> None:
    state = _read_state()
    for rec in (state.get("servers") or {}).values():
        pid = rec.get("pid")
        try:
            if pid:
                os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    # State file left in place for health checks

def health() -> int:
    state = _read_state()
    servers = state.get("servers") or {}
    for k, rec in sorted(servers.items(), key=lambda kv: int(kv[0])):
        rpc = rec.get("rpc_port")
        ok  = is_port_open("127.0.0.1", rpc)
        print(f"gpu{k}: rpc={rpc} tm={rec.get('tm_port')} pid={rec.get('pid')} {'OK' if ok else 'DOWN'}")
    return 0

def ensure(gpu_id: int, base: int, spacing: int, tm_off: int) -> int:
    rpc, tm = _derive_ports(gpu_id, base, spacing, tm_off)
    # Trust an existing listener -- never kill a server that's happily serving.
    if is_port_open("127.0.0.1", rpc):
        _clear_park(gpu_id)
        return 0
    attempts = int(os.environ.get("CARLA_BOOT_ATTEMPTS", "3"))
    timeout  = int(os.environ.get("CARLA_BOOT_TIMEOUT_SEC", "120"))
    for attempt in range(1, attempts + 1):
        _kill_gpu_server(gpu_id, rpc)   # also done inside start_one; cheap + explicit
        print(f"[server_manager] gpu{gpu_id}: launch attempt {attempt}/{attempts} on rpc {rpc}")
        pid = start_one(gpu_id, rpc, tm)
        if pid and _wait_healthy(rpc, pid, timeout):
            print(f"[server_manager] gpu{gpu_id}: healthy on rpc {rpc}")
            _clear_park(gpu_id)
            return 0
        print(f"[server_manager] gpu{gpu_id}: boot attempt {attempt} failed", file=sys.stderr)
        _kill_gpu_server(gpu_id, rpc)   # tidy the failed attempt before retrying
    # Every attempt segfaulted on boot -> park the GPU so workers stop throwing
    # jobs at it (which would just fast-fail and burn the queue). The worker
    # periodically re-runs ensure(); a later success clears the park.
    _park_gpu(gpu_id, f"CARLA failed to boot {attempts}x on rpc {rpc}")
    print(f"[server_manager] gpu{gpu_id}: PARKED after {attempts} failed boots", file=sys.stderr)
    return 3

def parse_args():
    p = argparse.ArgumentParser("carla_server_manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="start persistent server pool")
    sp.add_argument("--gpus", type=str, default="auto", help='Comma list like "0,1,2" or "auto" for 0..(N-1)')
    sp.add_argument("--base-rpc-port", type=int, default=DEFAULT_BASE)
    sp.add_argument("--port-spacing", type=int, default=DEFAULT_SPACING)
    sp.add_argument("--tm-offset", type=int, default=DEFAULT_TM_OFFSET)

    sub.add_parser("stop", help="stop persistent server pool")
    sub.add_parser("health", help="check pool health")

    se = sub.add_parser("ensure", help="ensure a server exists for ONE gpu")
    se.add_argument("--gpu", type=int, required=True)
    se.add_argument("--base-rpc-port", type=int, default=DEFAULT_BASE)
    se.add_argument("--port-spacing", type=int, default=DEFAULT_SPACING)
    se.add_argument("--tm-offset", type=int, default=DEFAULT_TM_OFFSET)

    return p.parse_args()

def main():
    args = parse_args()
    if args.cmd == "start":
        gpus = discover_gpus() if args.gpus == "auto" else [int(x) for x in args.gpus.split(",")]
        info = start_pool(gpus, args.base_rpc_port, args.port_spacing, args.tm_offset)
        print(json.dumps({"started": info}, indent=2))
        return 0
    if args.cmd == "stop":
        stop_pool(); print("STOPPED"); return 0
    if args.cmd == "health":
        return health()
    if args.cmd == "ensure":
        return ensure(args.gpu, args.base_rpc_port, args.port_spacing, args.tm_offset)
    return 0

if __name__ == "__main__":
    sys.exit(main())