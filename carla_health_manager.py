#!/usr/bin/env python3
"""
CARLA Health Manager (persistent)
Renders a per-node / per-GPU dashboard for the persistent workers.

Key change vs prior version:
- Always MERGE per-GPU health files (collection_state/health/...) with
  v2 namespaced collection_state/gpu_status.json, then de-duplicate.
  This guarantees you see all GPUs even if only some nodes emit health files.

Works with:
  - collection_state/job_queue.json               (for header counts)
  - collection_state/health/<node>/gpu<N>.json    (preferred legacy)
  - collection_state/health/gpu<N>.json           (flat legacy)
  - collection_state/gpu_status.json              (v2 namespaced)
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


def _fmt_age(seconds: float) -> str:
    if seconds is None or seconds == float("inf"):
        return "--"
    try:
        s = int(seconds)
    except Exception:
        return "--"
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class CarlaHealthManager:
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.environ.get("PROJECT_ROOT", os.getcwd()))
        self.state_dir = Path(os.environ.get("STATE_DIR", self.project_root / "collection_state"))
        self.health_dir = self.state_dir / "health"
        self.logs_dir = Path(os.environ.get("LOG_DIR", self.project_root / "logs"))

        self.num_gpus = int(os.environ.get("NUM_GPUS", os.environ.get("GPUS_PER_NODE", 8)))
        self.base_rpc_port = int(os.environ.get("BASE_RPC_PORT", 2000))
        self.port_spacing = int(os.environ.get("PORT_SPACING", 100))
        self.tm_offset = int(os.environ.get("TM_OFFSET", 5000))
        self.stale_threshold = int(os.environ.get("STALE_SECS", 90))

        for d in (self.health_dir, self.logs_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # ---------- IO helpers ----------

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

    # ---------- Health (legacy files) ----------

    def _read_health_file(self, path: Path) -> Dict[str, Any]:
        rpc, tm = self._derive_ports(self._gpu_id_from_path(path))
        out = {
            "gpu_id": self._gpu_id_from_path(path),
            "node": path.parent.name if path.parent != self.health_dir else "unknown",
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
        data = self._read_json(path)
        if not isinstance(data, dict):
            return out

        now = time.time()
        ts_unix = float(data.get("timestamp_unix") or 0)
        age = now - ts_unix if ts_unix else float("inf")
        is_stale = age > self.stale_threshold

        out.update(data)
        out["age_seconds"] = age
        out["is_stale"] = is_stale
        if is_stale:
            # keep original status text but flag as stale
            out["message"] = out.get("message") or f"No update for {int(age)}s"
        return out

    @staticmethod
    def _gpu_id_from_path(path: Path) -> int:
        try:
            return int(path.stem.replace("gpu", ""))
        except Exception:
            return -1

    def _collect_from_health_dir(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self.health_dir.exists():
            return rows

        # Per-node dirs
        for node_dir in sorted(self.health_dir.glob("*")):
            if node_dir.is_dir():
                for f in sorted(node_dir.glob("gpu*.json")):
                    rows.append(self._read_health_file(f))

        # Flat legacy
        for f in sorted(self.health_dir.glob("gpu*.json")):
            rows.append(self._read_health_file(f))

        return rows

    # ---------- Health (v2 gpu_status.json) ----------

    def _collect_from_v2_status(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        status_file = self.state_dir / "gpu_status.json"
        data = self._read_json(status_file)
        if not isinstance(data, dict):
            return rows
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return rows

        now = time.time()
        for node_name, gpu_map in sorted(nodes.items()):
            if not isinstance(gpu_map, dict):
                continue
            for gpu_key, info in sorted(
                gpu_map.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 9999
            ):
                try:
                    gid = int(gpu_key)
                except Exception:
                    continue

                # Heartbeat -> age
                age = float("inf")
                ts_txt = None
                hb = (info or {}).get("last_heartbeat")
                if isinstance(hb, (int, float)):
                    age = max(0.0, now - float(hb))
                    ts_txt = datetime.fromtimestamp(float(hb)).strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(hb, str):
                    try:
                        s = hb.replace("T", " ").split(".")[0]
                        dt = datetime.fromisoformat(s)
                        age = max(0.0, now - dt.timestamp())
                        ts_txt = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        ts_txt = hb

                rpc, tm = self._derive_ports(gid)

                # Nicely formatted message: agent / T<town> / route / W<weather>
                msg = ""
                cj = (info or {}).get("current_job") or {}
                parts = []
                agent = cj.get("agent") or cj.get("agent_name")
                town = cj.get("town") or cj.get("map") or cj.get("town_num")
                route = cj.get("route") or cj.get("route_id") or cj.get("route_name")
                weather = cj.get("weather") or cj.get("weather_idx") or cj.get("weather_index")
                if agent:
                    parts.append(str(agent))
                if town:
                    parts.append(f"T{town}")
                if route:
                    parts.append(str(route))
                if weather is not None and weather != "":
                    parts.append(f"W{weather}")
                if parts:
                    msg = " / ".join(parts)

                rows.append(
                    {
                        "gpu_id": gid,
                        "node": str(node_name),
                        "status": (info or {}).get("status", "unknown"),
                        "message": msg or (info or {}).get("message", ""),
                        "carla_pid": None,
                        "worker_pid": None,
                        "rpc_port": rpc,
                        "tm_port": tm,
                        "timestamp": ts_txt,
                        "timestamp_unix": 0,
                        "is_stale": age > self.stale_threshold,
                        "age_seconds": age,
                        "jobs_completed": (info or {}).get("jobs_completed", 0),
                    }
                )
        return rows

    # ---------- Merge & summarize ----------

    @staticmethod
    def _prefers(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """
        Return True if 'a' is preferable to 'b' for the same (node, gpu_id).
        Preference rules:
          1) non-stale beats stale
          2) named node beats 'unknown'
          3) has timestamp beats none
        """
        if a.get("is_stale", True) != b.get("is_stale", True):
            return not a.get("is_stale", True)
        if (a.get("node") or "unknown") != (b.get("node") or "unknown"):
            return (a.get("node") or "unknown") != "unknown"
        a_ts = a.get("timestamp_unix") or 0
        b_ts = b.get("timestamp_unix") or 0
        return a_ts >= b_ts

    def get_all_gpu_status(self) -> List[Dict[str, Any]]:
        # Collect from BOTH sources
        from_health = self._collect_from_health_dir()
        from_v2 = self._collect_from_v2_status()

        best: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for row in from_health + from_v2:
            node = str(row.get("node") or "unknown")
            try:
                gid = int(row.get("gpu_id", -1))
            except Exception:
                gid = -1
            key = (node, gid)
            if key not in best or self._prefers(row, best[key]):
                best[key] = row

        # Sort by node, then GPU
        ordered_keys = sorted(best.keys(), key=lambda t: (t[0], int(t[1])))
        return [best[k] for k in ordered_keys]

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

    # ---------- UI ----------

    def print_status(self) -> None:
        try:
            os.system("clear")
        except Exception:
            pass

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 100)
        print(f"CARLA HEALTH MONITOR  |  {now}")
        print("=" * 100)

        cs = self.get_collection_status()
        print(
            f"Tasks: total={cs['total']}  completed={cs['completed']}  "
            f"running={cs['running']}  pending={cs['pending']}  failed={cs['failed']}"
        )
        print("-" * 100)
        print(f"{'GPU':<4} {'Node':<18} {'Status':<16} {'Jobs':<6} {'Age':<8} {'RPC':<6} {'TM':<6} {'Message'}")
        print("-" * 100)

        rows = self.get_all_gpu_status()
        if not rows:
            print(
                "No GPU health data found.\n"
                f"- Expected per-node files under: {self.health_dir}/<node>/gpu<N>.json\n"
                f"- Or a v2 status file: {self.state_dir / 'gpu_status.json'}"
            )
            print()
            return

        for s in rows:
            gpu = s.get("gpu_id", "-")
            node = (s.get("node") or "unknown")[:18]
            status = s.get("status") or "unknown"
            jobs_done = s.get("jobs_completed", 0)
            age = _fmt_age(s.get("age_seconds"))
            rpc = s.get("rpc_port", "")
            tm = s.get("tm_port", "")
            msg = s.get("message") or ""

            if s.get("is_stale"):
                status = f"⚠ {status}"
            elif status in ("ready", "healthy", "running_job", "waiting_for_job", "idle", "busy"):
                status = f"✓ {status}"

            print(f"{gpu:<4} {node:<18} {status:<16} {jobs_done:<6} {age:<8} {rpc:<6} {tm:<6} {msg}")
        print()

    # ---------- Optional helpers ----------

    def _restart_gpu_worker_safe(self, gpu_id: int, node: Optional[str]) -> None:
        script = self.project_root / "restart_gpu_worker.sh"
        script.write_text(
            f"""#!/bin/bash
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
"""
        )
        try:
            script.chmod(0o755)
            subprocess.run(["sbatch", str(script)], check=False)
        except Exception:
            pass

    def show_log(self, gpu_id: int, lines: int = 50) -> None:
        log_file = self.logs_dir / f"worker_gpu{gpu_id}.log"
        if not log_file.exists():
            print(f"No log found at {log_file}")
            return
        try:
            subprocess.run(["tail", "-n", str(lines), str(log_file)], check=False)
        except Exception as e:
            print(f"Error tailing log: {e}")

    def monitor(self, interval: int = 30, auto_restart: bool = False) -> None:
        print(f"Monitoring CARLA instances every {interval}s. Press Ctrl+C to stop.")
        try:
            while True:
                self.print_status()
                if auto_restart:
                    for s in self.get_all_gpu_status():
                        if s.get("is_stale") and s.get("status") in ("busy", "running_job"):
                            self._restart_gpu_worker_safe(s.get("gpu_id"), s.get("node"))
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="CARLA Health Monitor (persistent)")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("status", help="Show one-time status")
    mon = sub.add_parser("monitor", help="Continuously monitor")
    mon.add_argument("--interval", type=int, default=30)
    mon.add_argument("--auto-restart", action="store_true")

    lg = sub.add_parser("log", help="Show worker log")
    lg.add_argument("gpu_id", type=int)
    lg.add_argument("--lines", type=int, default=50)

    rs = sub.add_parser("restart", help="Submit SLURM job to restart worker")
    rs.add_argument("gpu_id", type=int, nargs="?")
    rs.add_argument("--node", type=str, default=None)

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
