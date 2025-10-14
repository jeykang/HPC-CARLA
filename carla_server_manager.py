#!/usr/bin/env python3
"""
CARLA persistent server pool manager.

- Starts exactly one CARLA server per visible GPU.
- Uses a per-node BASE_RPC_PORT and a fixed PORT_SPACING so ports never collide
  across nodes or GPUs:
    RPC_PORT(gpu) = BASE_RPC_PORT + gpu * PORT_SPACING
    TM_PORT(gpu)  = RPC_PORT + TM_OFFSET   (default TM_OFFSET = 5000)

- Writes a JSON state file with PIDs and ports.
- Provides a simple health check and a graceful stop.
"""

import os
import sys
import json
import time
import signal
import socket
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

DEFAULT_BASE = int(os.environ.get("BASE_RPC_PORT", "2000"))
DEFAULT_SPACING = int(os.environ.get("PORT_SPACING", "100"))
DEFAULT_TM_OFFSET = int(os.environ.get("TM_OFFSET", "5000"))
DEFAULT_UE4_QUALITY = os.environ.get("UE4_QUALITY", "Epic")

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
STATE_DIR = PROJECT_ROOT / "collection_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / f"carla_servers_{socket.gethostname()}.json"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SINGULARITY_IMAGE = os.environ.get("CARLA_SIF", "carla_official.sif")
# Use container-native path by default; expand inside container shell.
UE4_LAUNCH = os.environ.get("UE4_LAUNCH", "${CARLA_ROOT}/CarlaUE4.sh")


def tcp_listening(host: str, port: int, timeout: float = 0.25) -> bool:
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


def kill_port(port: int):
    """Brutally clear any lingering listeners on a port (container or host)."""
    cmds = [
        ["bash", "-lc", f"fuser -k -TERM {port}/tcp || true"],
        ["bash", "-lc", f"lsof -ti tcp:{port} | xargs -r kill -TERM || true"],
    ]
    for c in cmds:
        try:
            subprocess.run(c, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            pass


def launch_server(gpu_id: int, rpc_port: int, tm_port: int) -> subprocess.Popen:
    # Extra paranoia: free ports in case a zombie is around
    kill_port(rpc_port)
    kill_port(tm_port)

    # Per-GPU log so we can actually debug UE4 failures
    log_path = LOG_DIR / f"carla_gpu{gpu_id}.log"
    log_f = open(log_path, "ab", buffering=0)

    env = os.environ.copy()
    # Singularity honors NVIDIA_VISIBLE_DEVICES; set CUDA_VISIBLE_DEVICES too for safety
    env["NVIDIA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CARLA_SERVER"] = "1"

    # Only warn about missing host-side script if user explicitly points into /workspace
    if UE4_LAUNCH.startswith("/workspace/"):
        host_ue4 = PROJECT_ROOT / UE4_LAUNCH[len("/workspace/"):]
        if not host_ue4.exists():
            log_f.write(f"[manager] WARNING: UE4 launch script not found at {host_ue4}\n".encode())

    # Headless launch. Add SDL_VIDEODRIVER=offscreen to avoid X dependencies.
    cmd = [
        "singularity", "exec", "--nv",
        "-B", f"{PROJECT_ROOT}:/workspace",
        SINGULARITY_IMAGE,
        "bash", "-lc",
        (
            "ulimit -c 0 ; "
            "DISABLE_PYTHON=1 "
            "SDL_VIDEODRIVER=offscreen "
            f"{UE4_LAUNCH} "
            f"-opengl -RenderOffScreen -nosound -quality-level={DEFAULT_UE4_QUALITY} "
            f"-carla-rpc-port={rpc_port} -carla-streaming-port=0 -world-port={rpc_port} -server"
        ),
    ]

    # Log the exact command and environment hints
    log_f.write(
        f"[manager] launching gpu={gpu_id} rpc={rpc_port} tm={tm_port} "
        f"sif={SINGULARITY_IMAGE} ue4={UE4_LAUNCH}\n".encode()
    )

    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
    return proc


def wait_ready(proc: subprocess.Popen, port: int, wait_s: float = 120.0) -> bool:
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            # UE4 exited early; no point waiting further
            return False
        if tcp_listening("127.0.0.1", port):
            return True
        time.sleep(0.5)
    return False


def start_pool(gpus: List[int], base: int, spacing: int, tm_offset: int) -> Dict[int, Dict]:
    info: Dict[int, Dict] = {}
    for gpu in gpus:
        rpc = base + gpu * spacing
        tm = rpc + tm_offset
        proc = launch_server(gpu, rpc, tm)
        ok = wait_ready(proc, rpc, 120.0)
        if not ok:
            # Surface a helpful error message with the last lines of the per-GPU log.
            try:
                with open(LOG_DIR / f"carla_gpu{gpu}.log", "rb") as lf:
                    tail = b"".join(lf.readlines()[-50:])
                sys.stderr.write(
                    f"\n[manager] GPU {gpu} failed to become ready on :{rpc}.\n"
                    f"--- last log lines ---\n{tail.decode(errors='ignore')}\n"
                    f"-----------------------\n"
                )
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            raise RuntimeError(f"CARLA on GPU {gpu} failed to become ready (rpc={rpc}).")

        info[gpu] = {
            "gpu": gpu,
            "rpc_port": rpc,
            "tm_port": tm,
            "pid": proc.pid,
            "started_at": int(time.time()),
        }

    STATE_FILE.write_text(json.dumps({"node": socket.gethostname(), "servers": info}, indent=2))
    return info


def stop_pool(state_path: Path = STATE_FILE):
    if not state_path.exists():
        return
    data = json.loads(state_path.read_text())
    for _gpu, rec in data.get("servers", {}).items():
        pid = rec.get("pid")
        if not pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    try:
        state_path.unlink()
    except Exception:
        pass


def health() -> int:
    # Simple liveness check: sockets open for every recorded server
    if not STATE_FILE.exists():
        print(json.dumps({"status": "absent"}))
        return 1
    data = json.loads(STATE_FILE.read_text())
    bad = []
    for rec in data.get("servers", {}).values():
        port = rec.get("rpc_port")
        if not tcp_listening("127.0.0.1", int(port)):
            bad.append(rec)
    status = {"status": "ok" if not bad else "degraded", "bad": bad}
    print(json.dumps(status, indent=2))
    return 0 if not bad else 2


def parse_args():
    p = argparse.ArgumentParser("carla_server_manager")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="start persistent server pool")
    sp.add_argument("--gpus", type=str, default="auto",
                    help='Comma list like "0,1,2" or "auto" for 0..(N-1)')
    sp.add_argument("--base-rpc-port", type=int, default=DEFAULT_BASE)
    sp.add_argument("--port-spacing", type=int, default=DEFAULT_SPACING)
    sp.add_argument("--tm-offset", type=int, default=DEFAULT_TM_OFFSET)

    sub.add_parser("stop", help="stop persistent server pool")
    sub.add_parser("health", help="check pool health")

    return p.parse_args()


def discover_gpus() -> List[int]:
    # Respect CUDA_VISIBLE_DEVICES if present
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        mapping = [x for x in cvd.split(",") if x.strip() != ""]
        return list(range(len(mapping)))
    # Fallback: assume 0..N-1 where N comes from SLURM or nvidia-smi
    n = int(os.environ.get("SLURM_GPUS_ON_NODE", os.environ.get("NGPUS", "1")))
    return list(range(n))


def main():
    args = parse_args()
    if args.cmd == "start":
        gpus = discover_gpus() if args.gpus == "auto" else [int(x) for x in args.gpus.split(",")]
        info = start_pool(gpus, args.base_rpc_port, args.port_spacing, args.tm_offset)
        print(json.dumps({"started": info}, indent=2))
        return 0
    if args.cmd == "stop":
        stop_pool()
        print("STOPPED")
        return 0
    if args.cmd == "health":
        return health()
    return 0


if __name__ == "__main__":
    sys.exit(main())
