#!/usr/bin/env python3
"""
CARLA Health Manager for Persistent Instance Collection
Monitors persistent CARLA workers via shared filesystem.
Supports both legacy per-GPU health files (collection_state/health/gpu*.json)
and the v2 namespaced gpu_status.json (collection_state/gpu_status.json).
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# ------------------------------
# Utilities
# ------------------------------

def _fmt_age(seconds: float) -> str:
    if seconds is None or seconds == float('inf'):
        return "--"
    try:
        seconds = int(seconds)
    except Exception:
        return "--"
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class CarlaHealthManager:
    """
    Monitors CARLA health through shared filesystem state files.
    Designed to work from a login node without direct cluster access.

    Expected locations (relative to PROJECT_ROOT or CWD):
      - collection_state/job_queue.json
      - collection_state/gpu_status.json              (v2 namespaced schema)
      - collection_state/health/gpu<N>.json           (legacy, flat)
      - collection_state/health/<node>/gpu<N>.json    (preferred, per-node)
    """

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.environ.get("PROJECT_ROOT", os.getcwd()))
        # Allow explicit STATE_DIR override
        self.state_dir = Path(os.environ.get("STATE_DIR", self.project_root / "collection_state"))
        self.health_dir = self.state_dir / "health"
        self.logs_dir = Path(os.environ.get("LOG_DIR", self.project_root / "logs"))

        # Configuration (match persistent scheme)
        self.num_gpus = int(os.environ.get("NUM_GPUS", os.environ.get("GPUS_PER_NODE", 8)))
        self.base_rpc_port = int(os.environ.get("BASE_RPC_PORT", 2000))
        self.port_spacing = int(os.environ.get("PORT_SPACING", 100))
        self.tm_offset = int(os.environ.get("TM_OFFSET", 5000))
        self.stale_threshold = int(os.environ.get("STALE_SECS", 90))  # seconds

        # Create dirs if needed (non-fatal)
        try:
            self.health_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # ------------------------------
    # Data access
    # ------------------------------

    def _derive_ports(self, gpu_id: int) -> Tuple[int, int]:
        rpc = self.base_rpc_port + gpu_id * self.port_spacing
        tm = rpc + self.tm_offset
        return rpc, tm

    def _read_json(self, p: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def get_gpu_health(self, gpu_id: int, node: Optional[str] = None, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Read legacy/health-file status for a specific GPU.
        Returns a normalized dict the dashboard can render.
        """
        if path is not None:
            health_file = Path(path)
        elif node:
            health_file = self.health_dir / node / f"gpu{gpu_id}.json"
        else:
            health_file = self.health_dir / f"gpu{gpu_id}.json"

        rpc, tm = self._derive_ports(gpu_id)
        default_status: Dict[str, Any] = {
            "gpu_id": gpu_id,
            "node": "unknown",
            "status": "unknown",
            "message": "No health data available",
            "carla_pid": None,
            "worker_pid": None,
            "rpc_port": rpc,
            "tm_port": tm,
            "timestamp": None,
            "timestamp_unix": 0,
            "is_stale": True,
            "age_seconds": float("inf"),
            "jobs_completed": 0,
        }

        if not health_file.exists():
            return default_status

        data = self._read_json(health_file)
        if not isinstance(data, dict):
            return default_status

        now = time.time()
        ts_unix = float(data.get("timestamp_unix") or 0)
        age_sec = now - ts_unix if ts_unix else float("inf")
        is_stale = age_sec > self.stale_threshold

        # Normalize
        out = default_status.copy()
        out.update(data)
        out["age_seconds"] = age_sec
        out["is_stale"] = is_stale
        if is_stale:
            out["status"] = "stale"
            out["message"] = f"No update for {int(age_sec)}s"
        return out

    def _fallback_gpu_status_from_v2(self) -> List[Dict[str, Any]]:
        """
        Fallback: read collection_state/gpu_status.json (v2, namespaced) and
        synthesize per-node/per-gpu rows when health/*.json files are missing.
        """
        statuses: List[Dict[str, Any]] = []
        status_file = self.state_dir / "gpu_status.json"
        data = self._read_json(status_file)
        if not isinstance(data, dict):
            return statuses
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return statuses

        now = time.time()
        for node_name, gpu_map in sorted(nodes.items()):
            if not isinstance(gpu_map, dict):
                continue
            for gpu_key, ginfo in sorted(gpu_map.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 9999):
                try:
                    gid = int(gpu_key)
                except Exception:
                    continue
                hb = None
                ts_text = None
                ginfo = ginfo or {}
                if "last_heartbeat" in ginfo:
                    hb = ginfo["last_heartbeat"]
                age = float("inf")
                if isinstance(hb, (int, float)):
                    age = max(0.0, now - float(hb))
                    ts_text = datetime.fromtimestamp(float(hb)).strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(hb, str):
                    # try ISO-ish
                    s = hb.replace("T", " ").split(".")[0]
                    try:
                        dt = datetime.fromisoformat(s)
                        age = max(0.0, now - dt.timestamp())
                        ts_text = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        ts_text = hb
                        age = float("inf")

                rpc, tm = self._derive_ports(gid)
                msg = ""
                cj = ginfo.get("current_job")
                if isinstance(cj, dict):
                    agent = cj.get("agent") or cj.get("agent_name")
                    town = cj.get("town") or cj.get("map") or cj.get("town_num")
                    route = cj.get("route") or cj.get("route_id") or cj.get("route_name")
                    weather = cj.get("weather") or cj.get("weather_idx") or cj.get("weather_index")
                    parts = []
                    if agent: parts.append(str(agent))
                    if town: parts.append(f"T{town}")
                    if route: parts.append(str(route))
                    if weather is not None: parts.append(f"W{weather}")
                    if parts:
                        msg = " / ".join(parts)

                statuses.append({
                    "gpu_id": gid,
                    "node": str(node_name),
                    "status": ginfo.get("status", "unknown"),
                    "message": msg or ginfo.get("message", ""),
                    "carla_pid": None,
                    "worker_pid": None,
                    "rpc_port": rpc,
                    "tm_port": tm,
                    "timestamp": ts_text,
                    "timestamp_unix": 0,
                    "is_stale": age > self.stale_threshold,
                    "age_seconds": age,
                    "jobs_completed": ginfo.get("jobs_completed", 0),
                })
        return statuses

    def get_all_gpu_status(self) -> List[Dict[str, Any]]:
        """
        Collect per-node/per-gpu statuses from preferred health/*.json files.
        If none are present, fall back to gpu_status.json (v2 namespaced).
        """
        statuses: List[Dict[str, Any]] = []

        # Preferred: per-node health files
        if self.health_dir.exists():
            for node_dir in sorted(self.health_dir.glob("*")):
                if node_dir.is_dir():
                    for f in sorted(node_dir.glob("gpu*.json")):
                        try:
                            gid = int(f.stem.replace("gpu", ""))
                        except Exception:
                            continue
                        statuses.append(self.get_gpu_health(gid, path=f))
            # Legacy flat
            for f in sorted(self.health_dir.glob("gpu*.json")):
                try:
                    gid = int(f.stem.replace("gpu", ""))
                except Exception:
                    continue
                statuses.append(self.get_gpu_health(gid, path=f))

        if not statuses:
            statuses = self._fallback_gpu_status_from_v2()

        # Deduplicate by (node, gpu_id)
        best: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for s in statuses:
            key = (str(s.get("node", "unknown")), int(s.get("gpu_id", -1)))
            # Prefer entries with explicit node names and non-stale status
            prev = best.get(key)
            if prev is None or (prev.get("node", "unknown") == "unknown" and s.get("node", "unknown") != "unknown") or (prev.get("is_stale", True) and not s.get("is_stale", False)):
                best[key] = s

        ordered = [best[k] for k in sorted(best.keys(), key=lambda t: (t[0], int(t[1])))]
        return ordered

    def get_collection_status(self) -> Dict[str, int]:
        qf = self.state_dir / "job_queue.json"
        data = self._read_json(qf) or {}
        jobs = data.get("jobs") or []
        return {
            "total": int(data.get("total", len(jobs))),
            "completed": int(data.get("completed", 0)),
            "pending": sum(1 for j in jobs if j.get("status") == "pending"),
            "running": sum(1 for j in jobs if j.get("status") in ("assigned", "running")),
            "failed": sum(1 for j in jobs if j.get("status") == "failed"),
        }

    # ------------------------------
    # Actions / UI
    # ------------------------------

    def print_status(self) -> None:
        # Clear screen if possible
        try:
            os.system("clear")
        except Exception:
            pass

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 100)
        print(f"CARLA HEALTH MONITOR  |  {now}")
        print("=" * 100)

        cs = self.get_collection_status()
        print(f"Tasks: total={cs['total']}  completed={cs['completed']}  running={cs['running']}  "
              f"pending={cs['pending']}  failed={cs['failed']}")
        print("-" * 100)
        print(f"{'GPU':<4} {'Node':<18} {'Status':<16} {'Jobs':<6} {'Age':<8} {'RPC':<6} {'TM':<6} {'Message'}")
        print("-" * 100)

        any_rows = False
        for s in self.get_all_gpu_status():
            any_rows = True
            gpu = s.get("gpu_id", "-")
            node = (s.get("node") or "unknown")[:18]
            status = s.get("status") or "unknown"
            jobs_done = s.get("jobs_completed", 0)
            age = _fmt_age(s.get("age_seconds"))
            rpc = s.get("rpc_port", "")
            tm = s.get("tm_port", "")
            msg = s.get("message") or ""
            # decorate stale
            if s.get("is_stale"):
                status = f"⚠ {status}"
            elif status in ("ready", "healthy", "running_job", "waiting_for_job", "idle"):
                status = f"✓ {status}"
            print(f"{gpu:<4} {node:<18} {status:<16} {jobs_done:<6} {age:<8} {rpc:<6} {tm:<6} {msg}")

        if not any_rows:
            print("No GPU health data found.\n"
                  f"- Expected per-node files under: {self.health_dir}/<node>/gpu<N>.json\n"
                  f"- Or a v2 status file: {self.state_dir/'gpu_status.json'}")
        print()

    def monitor(self, interval: int = 30, auto_restart: bool = False) -> None:
        print(f"Monitoring CARLA instances every {interval}s. Press Ctrl+C to stop.")
        try:
            while True:
                self.print_status()
                # Simple auto-restart heuristic (noop by default)
                if auto_restart:
                    for s in self.get_all_gpu_status():
                        if s.get("is_stale") and s.get("status") in ("busy", "running_job"):
                            self._restart_gpu_worker_safe(s.get("gpu_id"), s.get("node"))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")

    # ------------------------------
    # Optional: restart / log helpers
    # ------------------------------

    def _restart_gpu_worker_safe(self, gpu_id: int, node: Optional[str]) -> None:
        """
        Submit a SLURM job to restart a specific *worker* on the node.
        This implementation emits an sbatch that simply exports GPU_ID and calls persistent_carla_worker.sh.
        Adjust to your environment as needed.
        """
        if gpu_id is None:
            return
        script = self.project_root / "restart_gpu_worker.sh"
        script.write_text(f"""#!/bin/bash
#SBATCH -J restart_wkr_{gpu_id}
#SBATCH -o {self.logs_dir}/restart_gpu_{gpu_id}.out
#SBATCH -e {self.logs_dir}/restart_gpu_{gpu_id}.err
{f"#SBATCH -w {node}" if node else ""}

export PROJECT_ROOT="{self.project_root}"
export STATE_DIR="{self.state_dir}"
export LOG_DIR="{self.logs_dir}"
export GPU_ID="{gpu_id}"
export BASE_RPC_PORT="{self.base_rpc_port}"
export PORT_SPACING="{self.port_spacing}"
export TM_OFFSET="{self.tm_offset}"

echo "[restart] restarting worker on GPU {gpu_id} (node={node}) at $(date)"
if [[ -x "./persistent_carla_worker.sh" ]]; then
  ./persistent_carla_worker.sh --gpu "$GPU_ID"
else
  echo "persistent_carla_worker.sh not found; please restart manually."
fi
""")
        try:
            script.chmod(0o755)
            subprocess.run(["sbatch", str(script)], check=False)
        except Exception:
            pass

    def show_log(self, gpu_id: int, lines: int = 50) -> None:
        """
        Tail the worker log if present.
        """
        log_file = self.logs_dir / f"worker_gpu{gpu_id}.log"
        if not log_file.exists():
            print(f"No log found at {log_file}")
            return
        try:
            subprocess.run(["tail", "-n", str(lines), str(log_file)], check=False)
        except Exception as e:
            print(f"Error tailing log: {e}")

# ------------------------------
# CLI
# ------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA Health Monitor (persistent mode)")
    sub = parser.add_subparsers(dest="command", required=False)

    # status (default)
    sub.add_parser("status", help="Show one-time status")

    # monitor
    mon = sub.add_parser("monitor", help="Continuously monitor")
    mon.add_argument("--interval", type=int, default=30)
    mon.add_argument("--auto-restart", action="store_true")

    # log
    lg = sub.add_parser("log", help="Show worker log")
    lg.add_argument("gpu_id", type=int)
    lg.add_argument("--lines", type=int, default=50)

    # restart
    rs = sub.add_parser("restart", help="Submit SLURM job to restart worker")
    rs.add_argument("gpu_id", type=int, nargs="?")
    rs.add_argument("--node", type=str, default=None)

    # cleanup (remove stale health files)
    sub.add_parser("cleanup", help="Remove stale per-GPU health files")

    args = parser.parse_args()
    mgr = CarlaHealthManager()

    if args.command in (None, "status"):
        mgr.print_status()
        return 0
    if args.command == "monitor":
        mgr.monitor(interval=args.interval, auto_restart=args.auto_restart)
        return 0
    if args.command == "log":
        mgr.show_log(args.gpu_id, args.lines)
        return 0
    if args.command == "restart":
        mgr._restart_gpu_worker_safe(args.gpu_id, args.node)
        return 0
    if args.command == "cleanup":
        count = 0
        if mgr.health_dir.exists():
            for p in mgr.health_dir.rglob("gpu*.json"):
                try:
                    p.unlink()
                    count += 1
                except Exception:
                    pass
        print(f"Removed {count} health files.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
