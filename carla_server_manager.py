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

import os, sys, json, time, socket, signal, subprocess, argparse, shlex
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

# CARLA streaming port handling:
# - Default 0 matches the historical single-job launcher in this repo and avoids collisions with
#   unknown services on fixed ports.
# - Set CARLA_STREAMING_PORT_MODE=offset to use rpc+CARLA_STREAMING_OFFSET instead.
DEFAULT_STREAMING_OFFSET = int(os.environ.get("CARLA_STREAMING_OFFSET", 10))
DEFAULT_STREAMING_PORT_MODE = os.environ.get("CARLA_STREAMING_PORT_MODE", "zero").strip().lower()

# Startup can be slow on cold caches; allow override via env.
DEFAULT_START_TIMEOUT = float(os.environ.get("CARLA_START_TIMEOUT", "300"))

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
    """
    Best-effort atomic write.

    IMPORTANT: This file can be written concurrently (coordinator + many workers).
    Use a PID-scoped temp path to avoid clobbering a shared ".tmp".
    """
    try:
        tmp = STATE_FILE.with_suffix(f".{os.getpid()}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        # Never fail server startup because bookkeeping couldn't persist.
        print(f"[server_manager] warning: failed to persist state: {e}", file=sys.stderr)

def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False

def _pid_is_alive(pid: Optional[int]) -> bool:
    try:
        if not pid:
            return False
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False

def wait_for_port(host: str, port: int, deadline: float, pid: Optional[int] = None) -> bool:
    while time.time() < deadline:
        if pid is not None and not _pid_is_alive(pid):
            return False
        if is_port_open(host, port):
            return True
        time.sleep(0.2)
    return False

def discover_gpus() -> List[int]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        mapping = [x for x in cvd.split(",") if x.strip() != ""]
        return list(range(len(mapping)))
    n = int(os.environ.get("SLURM_GPUS_ON_NODE", os.environ.get("GPUS_PER_NODE", "8")))
    return list(range(n))

def _derive_ports(gpu_id: int, base: int, spacing: int, tm_off: int) -> Tuple[int, int, int]:
    rpc = base + gpu_id*spacing
    tm  = rpc + tm_off
    if DEFAULT_STREAMING_PORT_MODE == "offset":
        streaming = rpc + min(DEFAULT_STREAMING_OFFSET, max(1, spacing - 1))
    else:
        # Default to "zero": let CARLA choose, and avoid fixed-port conflicts.
        streaming = 0
    return rpc, tm, streaming

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
    })
    # Prevent core dumps inside container
    env["SINGULARITYENV_ULIMIT_CORE"] = "0"
    return env

def _build_run_args(rpc: int, tm: int, streaming: int) -> List[str]:
    """
    Use 'singularity run' so the container's %runscript invokes ${CARLA_ROOT}/CarlaUE4.sh.
    All UE4/CARLA flags are passed as args to the runscript.
    """
    bind_spec = f"{str(PROJECT_ROOT)}:/workspace"
    # Avoid duplicate binds if the submission script already set SINGULARITY_BINDPATH/APPTAINER_BINDPATH.
    bindpath = os.environ.get("SINGULARITY_BINDPATH") or os.environ.get("APPTAINER_BINDPATH") or ""
    add_bind = f",{bind_spec}," not in f",{bindpath},"

    args: List[str] = ["singularity", "run", "--nv"]
    if add_bind:
        args.extend(["-B", bind_spec])  # mount project at /workspace for Python sidecars

    args.extend(
        [
            SIF_PATH,
            "-opengl",
            # Historical runs used this spelling; keep it for compatibility.
            "-RenderOffscreen",
            "-quality-level=Epic",
            f"-carla-rpc-port={rpc}",
            # Ensure the server uses the expected port; some CARLA builds rely on world-port.
            f"-world-port={rpc}",
            f"-carla-streaming-port={streaming}",
            "-carla-server",
        ]
    )
    return args

def _tail_file(path: Path, max_lines: int = 80, max_bytes: int = 64 * 1024) -> str:
    """Return a best-effort tail of a potentially large log file."""
    try:
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
            except Exception:
                pass
            data = f.read().decode("utf-8", errors="ignore")
        return "\n".join(data.splitlines()[-max_lines:])
    except Exception:
        return ""

def _socket_diag(port: int) -> str:
    """Best-effort socket diagnostic via ss/netstat."""
    cmds = [
        ["ss", "-ltnp"],
        ["netstat", "-ltnp"],
    ]
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                continue
            out = p.stdout or ""
            matches = [ln for ln in out.splitlines() if f":{int(port)}" in ln]
            if matches:
                return "\n".join(matches[-20:])
        except Exception:
            continue
    return ""

def start_one(gpu_id: int, rpc: int, tm: int, streaming: int) -> Optional[int]:
    log_path = LOG_DIR / f"carla_{NODE_NAME}_gpu{gpu_id}.log"
    start_ts = float(time.time())
    try:
        # Prefix the log with the exact command line for post-mortem debugging.
        cmd = _build_run_args(rpc, tm, streaming)
        with open(log_path, "ab", buffering=0) as logf:
            header = (
                f"[server_manager] launch gpu={gpu_id} rpc={rpc} tm={tm} streaming={streaming}\n"
                f"[server_manager] cmd: {shlex.join(cmd)}\n"
            )
            logf.write(header.encode("utf-8", errors="ignore"))
        with open(log_path, "ab", buffering=0) as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf, stderr=logf,
                env=_container_env_for_gpu(gpu_id),
            )
        # Persist state
        state = _read_state()
        servers = state.setdefault("servers", {})
        servers[str(gpu_id)] = {
            "gpu": gpu_id, "rpc_port": rpc, "tm_port": tm, "streaming_port": streaming,
            "pid": proc.pid, "node": NODE_NAME, "log": str(log_path),
            "start_ts_unix": start_ts,
        }
        _write_state(state)
        return proc.pid
    except Exception as e:
        print(f"[server_manager] failed to start gpu{gpu_id}: {e}", file=sys.stderr)
        return None

def start_pool(gpus: List[int], base: int, spacing: int, tm_off: int) -> Dict[str, Dict]:
    started = {}
    for gid in gpus:
        rpc, tm, streaming = _derive_ports(gid, base, spacing, tm_off)
        if is_port_open("127.0.0.1", rpc):
            started[str(gid)] = {
                "rpc_port": rpc,
                "tm_port": tm,
                "streaming_port": streaming,
                "pid": None,
                "already_running": True,
                "ready_seconds": 0.0,
            }
            continue
        pid = start_one(gid, rpc, tm, streaming)
        t0 = time.time()
        ok = wait_for_port("127.0.0.1", rpc, time.time() + DEFAULT_START_TIMEOUT, pid=pid)
        started[str(gid)] = {
            "rpc_port": rpc,
            "tm_port": tm,
            "streaming_port": streaming,
            "pid": pid,
            "listening": ok,
            "ready_seconds": float(time.time() - t0),
        }
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
    rpc, tm, streaming = _derive_ports(gpu_id, base, spacing, tm_off)
    if is_port_open("127.0.0.1", rpc):
        return 0

    # If we have a recorded PID and it is still alive, don't spawn a duplicate.
    try:
        st = _read_state()
        rec = (st.get("servers") or {}).get(str(gpu_id)) or {}
        existing_pid = rec.get("pid")
        if _pid_is_alive(existing_pid):
            ok = wait_for_port("127.0.0.1", rpc, time.time() + DEFAULT_START_TIMEOUT, pid=existing_pid)
            if ok:
                return 0
            print(
                f"[server_manager] gpu{gpu_id}: pid {existing_pid} alive but still no listener on {rpc}",
                file=sys.stderr,
            )
    except Exception:
        pass

    print(f"[server_manager] gpu{gpu_id}: no listener on {rpc}, launching …", file=sys.stderr, flush=True)
    pid = start_one(gpu_id, rpc, tm, streaming)
    ok = wait_for_port("127.0.0.1", rpc, time.time() + DEFAULT_START_TIMEOUT, pid=pid)
    if ok:
        return 0

    # Diagnostics for debugging via shared logs.
    print(f"[server_manager] gpu{gpu_id}: still no listener on {rpc}", file=sys.stderr, flush=True)
    if pid:
        print(
            f"[server_manager] gpu{gpu_id}: spawned pid={pid} alive={_pid_is_alive(pid)}",
            file=sys.stderr,
            flush=True,
        )

    sock = _socket_diag(rpc)
    if sock:
        print(f"[server_manager] gpu{gpu_id}: socket diag for :{rpc}\n{sock}", file=sys.stderr)

    try:
        st = _read_state()
        rec = (st.get("servers") or {}).get(str(gpu_id)) or {}
        log_path = rec.get("log")
        if not log_path:
            log_path = str(LOG_DIR / f"carla_{NODE_NAME}_gpu{gpu_id}.log")
        tail = _tail_file(Path(log_path))
        if tail:
            print(f"[server_manager] gpu{gpu_id}: tail of {log_path}\n{tail}", file=sys.stderr)
    except Exception:
        pass

    return 2

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
