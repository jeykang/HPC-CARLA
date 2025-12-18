#!/usr/bin/env python3
"""Lightweight node-level metrics daemon.

Writes JSONL streams under:
  collection_state/metrics/node/<node>/{system.jsonl,gpu.jsonl}

Also writes last snapshots under:
  collection_state/metrics/node/<node>/last/{system.json,gpu.json}

This is intentionally dependency-free (stdlib + calling `nvidia-smi`).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _read_meminfo() -> Tuple[int, int]:
    """Return (ram_total_mib, ram_used_mib)."""
    mem_total_kib = None
    mem_available_kib = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kib = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
        if mem_total_kib is None:
            return 0, 0
        total_mib = int(mem_total_kib / 1024)
        if mem_available_kib is None:
            return total_mib, 0
        used_mib = int((mem_total_kib - mem_available_kib) / 1024)
        return total_mib, used_mib
    except Exception:
        return 0, 0


class _CpuSampler:
    def __init__(self) -> None:
        self._last: Optional[Tuple[int, int]] = None

    def sample_pct(self) -> Optional[float]:
        """Return CPU utilization percent since last sample."""
        try:
            parts = Path("/proc/stat").read_text().splitlines()[0].split()
            # cpu user nice system idle iowait irq softirq steal guest guest_nice
            vals = [int(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            if self._last is None:
                self._last = (idle, total)
                return None
            last_idle, last_total = self._last
            self._last = (idle, total)
            d_idle = idle - last_idle
            d_total = total - last_total
            if d_total <= 0:
                return None
            return max(0.0, min(100.0, 100.0 * (1.0 - (d_idle / d_total))))
        except Exception:
            return None


def _run_cmd(cmd: List[str], timeout: float = 2.0) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def _parse_csv_lines(raw: str) -> List[List[str]]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return [[x.strip() for x in ln.split(",")] for ln in lines]


def _gpu_query() -> List[Dict[str, Any]]:
    """Query per-GPU utilization and metadata via nvidia-smi."""
    fields = [
        "index",
        "uuid",
        "name",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.used",
        "clocks.sm",
        "clocks.mem",
        "pstate",
        "power.draw",
        "power.limit",
        "fan.speed",
    ]
    rc, out, _err = _run_cmd(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout=3.0,
    )
    if rc != 0:
        return []

    rows = _parse_csv_lines(out)

    # Processes
    proc_rc, proc_out, _proc_err = _run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=3.0,
    )
    procs_by_uuid: Dict[str, List[Dict[str, Any]]] = {}
    if proc_rc == 0:
        for r in _parse_csv_lines(proc_out):
            if len(r) < 4:
                continue
            uuid, pid, name, used = r[0], r[1], r[2], r[3]
            try:
                pid_i = int(pid)
            except Exception:
                continue
            try:
                used_i = int(float(used))
            except Exception:
                used_i = None
            procs_by_uuid.setdefault(uuid, []).append(
                {
                    "gpu_uuid": uuid,
                    "pid": pid_i,
                    "name": name,
                    "used_mem_MiB": used_i,
                }
            )

    out_gpus: List[Dict[str, Any]] = []
    for r in rows:
        if len(r) != len(fields):
            continue
        d = dict(zip(fields, r))

        def _f(x: str) -> Optional[float]:
            try:
                return float(x)
            except Exception:
                return None

        def _i(x: str) -> Optional[int]:
            try:
                return int(float(x))
            except Exception:
                return None

        uuid = d.get("uuid")
        out_gpus.append(
            {
                "index": _i(d.get("index", "")),
                "uuid": uuid,
                "name": d.get("name"),
                "tempC": _f(d.get("temperature.gpu", "")),
                "util_sm": _f(d.get("utilization.gpu", "")),
                "util_mem": _f(d.get("utilization.memory", "")),
                "mem_total_MiB": _i(d.get("memory.total", "")),
                "mem_used_MiB": _i(d.get("memory.used", "")),
                "clock_sm_MHz": _i(d.get("clocks.sm", "")),
                "clock_mem_MHz": _i(d.get("clocks.mem", "")),
                "pstate": d.get("pstate"),
                "power_W": _f(d.get("power.draw", "")),
                "power_cap_W": _f(d.get("power.limit", "")),
                "fan_pct": _f(d.get("fan.speed", "")),
                "processes": procs_by_uuid.get(uuid or "", []),
            }
        )

    return out_gpus


def _nvidia_versions() -> Dict[str, Optional[str]]:
    """Return {'driver_version': ..., 'cuda_version': ...} if available."""
    rc, out, _err = _run_cmd(["nvidia-smi"], timeout=3.0)
    if rc != 0:
        return {"driver_version": None, "cuda_version": None}
    driver = None
    cuda = None
    for line in out.splitlines():
        if "Driver Version" in line and driver is None:
            # Example: | NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2 |
            if "Driver Version:" in line:
                try:
                    driver = line.split("Driver Version:", 1)[1].split()[0]
                except Exception:
                    pass
            if "CUDA Version:" in line:
                try:
                    cuda = line.split("CUDA Version:", 1)[1].split()[0]
                except Exception:
                    pass
    return {"driver_version": driver, "cuda_version": cuda}


def _db_gpu_job_ids(db_path: Path, node: str) -> Dict[int, Optional[int]]:
    """Map gpu_id -> current_job_id from gpu_status table (best-effort)."""
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT gpu_id, current_job_id FROM gpu_status WHERE node=?", (node,)
        )
        out: Dict[int, Optional[int]] = {}
        for row in cur.fetchall():
            try:
                gid = int(row["gpu_id"])
            except Exception:
                continue
            cjid = row["current_job_id"]
            try:
                out[gid] = int(cjid) if cjid is not None else None
            except Exception:
                out[gid] = None
        conn.close()
        return out
    except Exception:
        return {}


def _rotate_if_needed(path: Path, rotate_mib: int) -> None:
    try:
        if rotate_mib <= 0 or not path.exists():
            return
        size_mib = path.stat().st_size / (1024 * 1024)
        if size_mib < rotate_mib:
            return
        ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rotated = path.with_name(path.name.replace(".jsonl", f".{ts}.jsonl"))
        path.rename(rotated)
    except Exception:
        return


def _write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--interval-sec", type=float, default=2.0)
    ap.add_argument("--rotate-gpu-mib", type=int, default=256)
    ap.add_argument("--rotate-system-mib", type=int, default=64)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    state_dir = Path(os.environ.get("STATE_DIR", project_root / "collection_state"))
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    run_id = os.environ.get("HPC_CARLA_RUN_ID") or os.environ.get("SLURM_JOB_ID")

    metrics_dir = state_dir / "metrics" / "node" / node
    system_path = metrics_dir / "system.jsonl"
    gpu_path = metrics_dir / "gpu.jsonl"
    last_dir = metrics_dir / "last"

    db_path = state_dir / "collection.db"

    cpu = _CpuSampler()
    versions = _nvidia_versions()

    print(f"[metrics] starting daemon node={node} interval={args.interval_sec}s run_id={run_id}")

    while True:
        ts = _utc_now_iso()
        try:
            _rotate_if_needed(system_path, args.rotate_system_mib)
            _rotate_if_needed(gpu_path, args.rotate_gpu_mib)

            # System
            ram_total, ram_used = _read_meminfo()
            cpu_pct = cpu.sample_pct()
            try:
                load_1 = os.getloadavg()[0]
            except Exception:
                load_1 = None
            sys_obj: Dict[str, Any] = {
                "ts": ts,
                "cpu_pct": cpu_pct,
                "load_1": load_1,
                "ram_total_MiB": ram_total,
                "ram_used_MiB": ram_used,
                "node": node,
                "run_id": run_id,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                **versions,
            }
            _write_jsonl(system_path, sys_obj)
            _write_json(last_dir / "system.json", sys_obj)

            # GPU
            job_ids = _db_gpu_job_ids(db_path, node)
            gpus = _gpu_query()
            gpu_events: List[Dict[str, Any]] = []
            for g in gpus:
                gid = g.get("index")
                rec = {
                    "ts": ts,
                    **g,
                    "node": node,
                    "run_id": run_id,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "job_id": job_ids.get(int(gid)) if gid is not None else None,
                }
                gpu_events.append(rec)
                _write_jsonl(gpu_path, rec)
            _write_json(last_dir / "gpu.json", gpu_events)

        except KeyboardInterrupt:
            print("[metrics] stopping")
            return 0
        except Exception as e:
            # Never die; just log and keep going.
            print(f"[metrics] error: {e}")

        time.sleep(max(0.2, float(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
