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

def _build_run_args(rpc: int, tm: int) -> List[str]:
    """
    Use 'singularity run' so the container's %runscript invokes ${CARLA_ROOT}/CarlaUE4.sh.
    All UE4/CARLA flags are passed as args to the runscript.
    """
    args = [
        "singularity", "run", "--nv",
        "-B", f"{str(PROJECT_ROOT)}:/workspace",  # mount project at /workspace for Python sidecars
        SIF_PATH,
        "-opengl",
        "-RenderOffscreen",
        "-quality-level=Epic",
        f"-carla-rpc-port={rpc}",
        f"-trafficManagerPort={tm}",
        "-carla-server",
    ]
    # If your image expects '-nosound' (your %runscript already adds it), no need to pass it again.
    return args

def start_one(gpu_id: int, rpc: int, tm: int) -> Optional[int]:
    log_path = LOG_DIR / f"carla_{NODE_NAME}_gpu{gpu_id}.log"
    try:
        with open(log_path, "ab", buffering=0) as logf:
            proc = subprocess.Popen(
                _build_run_args(rpc, tm),
                stdout=logf, stderr=logf,
                env=_container_env_for_gpu(gpu_id),
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
        ok = wait_for_port("127.0.0.1", rpc, time.time() + 120)
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
    if is_port_open("127.0.0.1", rpc):
        return 0
    print(f"[server_manager] gpu{gpu_id}: no listener on {rpc}, launching …")
    pid = start_one(gpu_id, rpc, tm)
    ok = wait_for_port("127.0.0.1", rpc, time.time() + 120)
    if not ok:
        print(f"[server_manager] gpu{gpu_id}: still no listener on {rpc}", file=sys.stderr)
        return 2
    return 0

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
