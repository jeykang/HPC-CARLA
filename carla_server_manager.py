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

# UE4/CARLA launch tuning (overridable without code changes for HPC debugging).
DEFAULT_QUALITY_LEVEL = os.environ.get("CARLA_QUALITY_LEVEL", "Epic")
DEFAULT_RHI_FLAG = os.environ.get("CARLA_RHI_FLAG", "-opengl")
# Accept either spelling; CARLA/UE4 flags are case-sensitive across builds.
# Historically this repo used "-RenderOffscreen" (lowercase 's') in the persistent server manager.
DEFAULT_RENDER_FLAG = os.environ.get("CARLA_RENDER_FLAG", "-RenderOffscreen")
DEFAULT_SERVER_FLAG = os.environ.get("CARLA_SERVER_FLAG", "-carla-server")
DEFAULT_EXTRA_UE4_ARGS = shlex.split(os.environ.get("CARLA_EXTRA_UE4_ARGS", ""))
DEFAULT_STDOUT_LOG = os.environ.get("CARLA_STDOUT_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}
DEFAULT_HOME_MODE = os.environ.get("CARLA_HOME_MODE", "gpu").strip().lower()  # "gpu" or "default"
DEFAULT_WRITABLE_TMPFS = os.environ.get("CARLA_WRITABLE_TMPFS", "0").strip().lower() in {"1", "true", "yes", "on"}

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

    # Map a logical gpu_id (0..N-1) to the scheduler-provided CUDA_VISIBLE_DEVICES entry if present.
    # This matters because OpenGL/driver selection is often controlled via NVIDIA_VISIBLE_DEVICES at
    # container launch time (apptainer/singularity --nv hook), not only via in-container env vars.
    gpu_selector = str(gpu_id)
    try:
        cvd = env.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            mapping = [x.strip() for x in cvd.split(",") if x.strip() != ""]
            if 0 <= int(gpu_id) < len(mapping):
                gpu_selector = mapping[int(gpu_id)]
    except Exception:
        gpu_selector = str(gpu_id)

    env.update({
        # Host-level vars: consumed by apptainer/singularity's --nv GPU injection logic on some clusters.
        "CUDA_VISIBLE_DEVICES": gpu_selector,
        "NVIDIA_VISIBLE_DEVICES": gpu_selector,

        # In-container vars: ensure CARLA/UE4 sees the intended device list too.
        "SINGULARITYENV_CUDA_VISIBLE_DEVICES": gpu_selector,
        "SINGULARITYENV_NVIDIA_VISIBLE_DEVICES": gpu_selector,
        "APPTAINERENV_CUDA_VISIBLE_DEVICES": gpu_selector,
        "APPTAINERENV_NVIDIA_VISIBLE_DEVICES": gpu_selector,
        "SINGULARITYENV_SDL_VIDEODRIVER": "offscreen",
        "SINGULARITYENV_SDL_AUDIODRIVER": "dummy",
        "APPTAINERENV_SDL_VIDEODRIVER": "offscreen",
        "APPTAINERENV_SDL_AUDIODRIVER": "dummy",
        "SINGULARITYENV_DISABLE_PYTHON": "1",   # CARLA binary only
        "APPTAINERENV_DISABLE_PYTHON": "1",
    })
    # Prevent core dumps inside container
    env["SINGULARITYENV_ULIMIT_CORE"] = "0"
    return env

def _bindpath_maps_container(bindpath: str, container_path: str) -> bool:
    """
    Best-effort check whether a bind spec already targets container_path.
    Handles common formats like: host:container or host:container:opts
    """
    try:
        for ent in (bindpath or "").split(","):
            ent = ent.strip()
            if not ent:
                continue
            parts = ent.split(":")
            if len(parts) >= 2 and parts[1] == container_path:
                return True
    except Exception:
        pass
    return False

def _gpu_saved_dir(gpu_id: int) -> Path:
    return STATE_DIR / "ue4_saved" / NODE_NAME / f"gpu{int(gpu_id)}"

def _gpu_home_dir(gpu_id: int) -> Path:
    return STATE_DIR / "carla_home" / NODE_NAME / f"gpu{int(gpu_id)}" / "home"

def _gpu_config_dir(gpu_id: int) -> Path:
    # NOTE: UE4 frequently writes logs/config under $HOME/.config; we keep HOME per-GPU by default.
    return _gpu_home_dir(gpu_id) / ".config"

def _build_run_args(gpu_id: int, rpc: int, tm: int, streaming: int) -> List[str]:
    """
    Use 'singularity run' so the container's %runscript invokes ${CARLA_ROOT}/CarlaUE4.sh.
    All UE4/CARLA flags are passed as args to the runscript.
    """
    bind_spec = f"{str(PROJECT_ROOT)}:/workspace"
    # Avoid duplicate binds if the submission script already set SINGULARITY_BINDPATH/APPTAINER_BINDPATH.
    bindpath = os.environ.get("SINGULARITY_BINDPATH") or os.environ.get("APPTAINER_BINDPATH") or ""
    add_bind = f",{bind_spec}," not in f",{bindpath},"

    args: List[str] = ["singularity", "run", "--nv"]
    if DEFAULT_WRITABLE_TMPFS:
        args.append("--writable-tmpfs")

    # IMPORTANT: Singularity/Apptainer defaults to binding the *host user* $HOME into the container.
    # UE4 commonly writes logs/config under $HOME/.config and caches under $HOME/.cache. If $HOME is on
    # a slow/quota-limited filesystem, CARLA can hang early with no visible logs. We default to a
    # per-GPU home rooted under STATE_DIR to avoid cross-GPU contention and capture logs deterministically.
    container_home = "/tmp/carla_home"
    if DEFAULT_HOME_MODE == "gpu":
        home_host = _gpu_home_dir(gpu_id)
        try:
            home_host.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        args.extend(["--home", f"{str(home_host)}:{container_home}"])

    if add_bind:
        args.extend(["-B", bind_spec])  # mount project at /workspace for Python sidecars

    # Persist UE4 Saved/Logs per GPU so we can see why CARLA isn't listening.
    # IMPORTANT: do not bind over /home/carla itself (that is CARLA_ROOT in this image).
    saved_host = _gpu_saved_dir(gpu_id)
    cfg_host = _gpu_config_dir(gpu_id)
    try:
        saved_host.mkdir(parents=True, exist_ok=True)
        cfg_host.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Avoid duplicate bind warnings/errors if user provided these via SINGULARITY_BINDPATH.
    if not _bindpath_maps_container(bindpath, "/home/carla/CarlaUE4/Saved"):
        args.extend(["-B", f"{str(saved_host)}:/home/carla/CarlaUE4/Saved"])
    if not _bindpath_maps_container(bindpath, "/home/carla/.config"):
        args.extend(["-B", f"{str(cfg_host)}:/home/carla/.config"])

    ue4_stdout_args: List[str] = []
    if DEFAULT_STDOUT_LOG:
        # Some UE4/CARLA builds log mostly to Saved/Logs; these flags push logs to stdout too.
        ue4_stdout_args = ["-stdout", "-FullStdOutLogOutput", "-unattended", "-NoSplash"]

    args.extend(
        [
            SIF_PATH,
            DEFAULT_RHI_FLAG,
            DEFAULT_RENDER_FLAG,
            *ue4_stdout_args,
            f"-quality-level={DEFAULT_QUALITY_LEVEL}",
            f"-carla-rpc-port={rpc}",
            # Ensure the server uses the expected port; some CARLA builds rely on world-port.
            f"-world-port={rpc}",
            # IMPORTANT for multi-instance setups: avoid collisions on the Traffic Manager port.
            f"-trafficManagerPort={tm}",
            f"-carla-streaming-port={streaming}",
            *DEFAULT_EXTRA_UE4_ARGS,
            DEFAULT_SERVER_FLAG,
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

def _ps_diag(pid: Optional[int]) -> str:
    """Best-effort process diagnostic for a PID."""
    if not pid:
        return ""
    pid_s = str(int(pid))
    cmds = [
        ["ps", "-o", "pid=,ppid=,stat=,etime=,args=", "-p", pid_s],
        ["ps", "-o", "pid=,ppid=,stat=,etime=,cmd=", "-p", pid_s],
        ["ps", "-fp", pid_s],
    ]
    for cmd in cmds:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
            out = (p.stdout or "").strip()
            if out:
                return out
            err = (p.stderr or "").strip()
            if err:
                return err
        except Exception:
            continue
    return ""

def _proc_cmdline(pid: Optional[int]) -> str:
    try:
        if not pid:
            return ""
        p = Path("/proc") / str(int(pid)) / "cmdline"
        data = p.read_bytes()
        s = data.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        return s
    except Exception:
        return ""

def _dir_diag(path: Path, max_entries: int = 60) -> str:
    """
    Best-effort directory diagnostic: list a few entries and count visible files.
    """
    try:
        if not path.exists():
            return f"{path} (missing)"
        if path.is_file():
            return f"{path} (file)"
        entries = []
        for i, p in enumerate(sorted(path.iterdir(), key=lambda x: x.name)):
            if i >= max_entries:
                entries.append("…")
                break
            suffix = "/" if p.is_dir() else ""
            entries.append(p.name + suffix)
        return f"{path} entries={len(list(path.iterdir()))} sample={entries}"
    except Exception as e:
        return f"{path} (diag error: {e})"

def _latest_file_in_dir(directory: Path, patterns: List[str]) -> Optional[Path]:
    try:
        if not directory.exists():
            return None
        files: List[Path] = []
        for pat in patterns:
            files.extend(directory.glob(pat))
        files = [p for p in files if p.is_file()]
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None

def _ue4_log_tail(gpu_id: int) -> Tuple[str, str]:
    """
    Return (path, tail) of the newest UE4 log persisted via our bind mounts.

    NOTE: Depending on the build, UE4 logs may appear in either:
      - <project>/Saved/Logs (we bind to /home/carla/CarlaUE4/Saved)
      - ~/.config/Epic/.../Saved/Logs (we bind to /home/carla/.config)
    """
    try:
        candidates: List[Path] = []

        # Project Saved logs
        candidates.append(_gpu_saved_dir(gpu_id) / "Logs")

        # User config logs (common for UE4 in containers / headless setups)
        cfg = _gpu_config_dir(gpu_id)
        candidates.append(cfg / "Epic" / "CarlaUE4" / "Saved" / "Logs")
        candidates.append(cfg / "Epic" / "UnrealEngine" / "4.24" / "Saved" / "Logs")

        # If UE4 ignored our config bind and wrote to the host user's $HOME, check there too.
        host_home = Path.home()
        candidates.append(host_home / ".config" / "Epic" / "CarlaUE4" / "Saved" / "Logs")
        candidates.append(host_home / ".config" / "Epic" / "UnrealEngine" / "4.24" / "Saved" / "Logs")

        newest: Optional[Path] = None
        for d in candidates:
            f = _latest_file_in_dir(d, ["*.log", "*.txt"])
            if f and (newest is None or f.stat().st_mtime > newest.stat().st_mtime):
                newest = f

        if not newest:
            return "", ""
        return str(newest), _tail_file(newest)
    except Exception:
        return "", ""

def start_one(gpu_id: int, rpc: int, tm: int, streaming: int) -> Optional[int]:
    log_path = LOG_DIR / f"carla_{NODE_NAME}_gpu{gpu_id}.log"
    start_ts = float(time.time())
    try:
        # Prefix the log with the exact command line for post-mortem debugging.
        cmd = _build_run_args(gpu_id, rpc, tm, streaming)
        env = _container_env_for_gpu(gpu_id)
        with open(log_path, "ab", buffering=0) as logf:
            header = (
                f"[server_manager] launch gpu={gpu_id} rpc={rpc} tm={tm} streaming={streaming}\n"
                f"[server_manager] ue4_saved_host: {_gpu_saved_dir(gpu_id)}\n"
                f"[server_manager] ue4_home_host: {_gpu_home_dir(gpu_id)} (mode={DEFAULT_HOME_MODE})\n"
                f"[server_manager] ue4_config_host: {_gpu_config_dir(gpu_id)}\n"
                f"[server_manager] env CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')} NVIDIA_VISIBLE_DEVICES={env.get('NVIDIA_VISIBLE_DEVICES','')}\n"
                f"[server_manager] cmd: {shlex.join(cmd)}\n"
            )
            logf.write(header.encode("utf-8", errors="ignore"))
        with open(log_path, "ab", buffering=0) as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf, stderr=logf,
                env=env,
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
        ps_line = _ps_diag(pid)
        if ps_line:
            print(f"[server_manager] gpu{gpu_id}: ps: {ps_line}", file=sys.stderr, flush=True)
        cmdline = _proc_cmdline(pid)
        if cmdline:
            print(f"[server_manager] gpu{gpu_id}: /proc cmdline: {cmdline}", file=sys.stderr, flush=True)

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

    # If UE4 wrote logs under Saved/Logs (persisted via bind mount), include those too.
    try:
        ue4_path, ue4_tail = _ue4_log_tail(gpu_id)
        if ue4_tail:
            print(f"[server_manager] gpu{gpu_id}: tail of UE4 log {ue4_path}\n{ue4_tail}", file=sys.stderr)
        elif ue4_path:
            print(f"[server_manager] gpu{gpu_id}: UE4 log exists but is empty: {ue4_path}", file=sys.stderr)
        else:
            saved_logs = _gpu_saved_dir(gpu_id) / "Logs"
            cfg_logs = _gpu_config_dir(gpu_id) / "Epic" / "CarlaUE4" / "Saved" / "Logs"
            ue_logs = _gpu_config_dir(gpu_id) / "Epic" / "UnrealEngine" / "4.24" / "Saved" / "Logs"
            print(
                f"[server_manager] gpu{gpu_id}: no UE4 log found under {saved_logs} or {cfg_logs} or {ue_logs} or {Path.home() / '.config'}",
                file=sys.stderr,
            )
    except Exception:
        pass

    # Dump a quick view of the expected writable dirs (helps distinguish "bind unused" vs "no writes").
    try:
        saved = _gpu_saved_dir(gpu_id)
        home = _gpu_home_dir(gpu_id)
        cfg = _gpu_config_dir(gpu_id)
        print(f"[server_manager] gpu{gpu_id}: dirs:", file=sys.stderr)
        print(f"[server_manager] gpu{gpu_id}:   {_dir_diag(saved)}", file=sys.stderr)
        print(f"[server_manager] gpu{gpu_id}:   {_dir_diag(saved / 'Logs')}", file=sys.stderr)
        print(f"[server_manager] gpu{gpu_id}:   {_dir_diag(home)}", file=sys.stderr)
        print(f"[server_manager] gpu{gpu_id}:   {_dir_diag(cfg)}", file=sys.stderr)
        print(f"[server_manager] gpu{gpu_id}:   host_home={Path.home()}", file=sys.stderr)
        print(f"[server_manager] gpu{gpu_id}:   {_dir_diag(Path.home() / '.config' / 'Epic')}", file=sys.stderr)
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
