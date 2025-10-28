#!/usr/bin/env python3
# hpc_metrics.py
import os, sys, json, time, subprocess, socket
from pathlib import Path
from datetime import datetime, timezone

# Optional deps
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

class NVMLUnavailable(Exception): ...

def iso_now():
    return datetime.now(timezone.utc).isoformat()

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

def safe_int(x, default=None):
    try: return int(float(x))
    except Exception: return default

def node_name():
    return os.environ.get('SLURMD_NODENAME') or os.uname().nodename

class GPUSampler:
    """
    Collect GPU metrics via nvidia-smi (no hard NVML dependency).
    Returns a list of dicts keyed by gpu index.
    """
    Q_GPU = [
        "index","uuid","name","temperature.gpu","utilization.gpu","utilization.memory",
        "memory.total","memory.used","clocks.sm","clocks.mem","pstate","power.draw","power.limit","fan.speed"
    ]
    Q_PROC = ["gpu_uuid","pid","process_name","used_memory"]

    def __init__(self):
        # Build uuid->index map
        self.uuid_to_index = {}
        g = run(["nvidia-smi","--query-gpu="+",".join(self.Q_GPU),
                 "--format=csv,noheader,nounits"])
        if g.returncode != 0:
            raise NVMLUnavailable(g.stderr.strip() or "nvidia-smi not available")
        for line in g.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx = safe_int(parts[0]); uuid = parts[1]
            self.uuid_to_index[uuid] = idx

    def sample(self):
        out = run(["nvidia-smi","--query-gpu="+",".join(self.Q_GPU),
                   "--format=csv,noheader,nounits"])
        if out.returncode != 0:
            raise NVMLUnavailable(out.stderr.strip())
        gpus = []
        now = iso_now()
        for line in out.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            rec = {
                "ts": now,
                "index": safe_int(p[0]), "uuid": p[1], "name": p[2],
                "tempC": safe_float(p[3]), "util_sm": safe_float(p[4]),
                "util_mem": safe_float(p[5]),
                "mem_total_MiB": safe_int(p[6]), "mem_used_MiB": safe_int(p[7]),
                "clock_sm_MHz": safe_int(p[8]), "clock_mem_MHz": safe_int(p[9]),
                "pstate": p[10], "power_W": safe_float(p[11]),
                "power_cap_W": safe_float(p[12]), "fan_pct": safe_float(p[13]),
            }
            gpus.append(rec)
        # processes
        pro = run(["nvidia-smi","--query-compute-apps="+",".join(self.Q_PROC),
                   "--format=csv,noheader,nounits"])
        proc = []
        if pro.returncode == 0 and pro.stdout.strip():
            for line in pro.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                proc.append({
                    "gpu_uuid": p[0], "pid": safe_int(p[1]), "name": p[2],
                    "used_mem_MiB": safe_int(p[3]),
                })
        # attach procs to each gpu
        procs_by_uuid = {}
        for pr in proc:
            procs_by_uuid.setdefault(pr["gpu_uuid"], []).append(pr)
        for rec in gpus:
            rec["processes"] = procs_by_uuid.get(rec["uuid"], [])
        return gpus

class SystemSampler:
    def sample(self):
        now = iso_now()
        data = {"ts": now}
        if psutil:
            try:
                data.update({
                    "cpu_pct": psutil.cpu_percent(interval=None),
                    "load_1": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
                    "ram_total_MiB": int(psutil.virtual_memory().total/1048576),
                    "ram_used_MiB": int(psutil.virtual_memory().used/1048576),
                    "swap_total_MiB": int(psutil.swap_memory().total/1048576),
                    "swap_used_MiB": int(psutil.swap_memory().used/1048576),
                    "net_bytes_sent": int(sum(i.bytes_sent for i in psutil.net_io_counters(pernic=True).values())),
                    "net_bytes_recv": int(sum(i.bytes_recv for i in psutil.net_io_counters(pernic=True).values())),
                    "disk_read_bytes": int(sum(i.read_bytes for i in psutil.disk_io_counters(perdisk=True).values())),
                    "disk_write_bytes": int(sum(i.write_bytes for i in psutil.disk_io_counters(perdisk=True).values())),
                })
                return data
            except Exception:
                pass
        # Minimal fallback via /proc
        try:
            with open("/proc/meminfo") as f:
                mi = {k.strip(":"): int(v.split()[0]) for k,v in
                      [ln.split(None,1) for ln in f if ":" in ln]}
            mem_total = mi.get("MemTotal")
            mem_avail = mi.get("MemAvailable")
        except Exception:
            mem_total = mem_avail = None
        data.update({
            "cpu_pct": None,
            "load_1": None,
            "ram_total_MiB": int(mem_total/1024) if mem_total else None,
            "ram_used_MiB": int((mem_total-mem_avail)/1024) if mem_total and mem_avail else None,
        })
        return data

class MetricsWriter:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.metrics_dir = self.state_dir/"metrics"/"node"/node_name()
        (self.metrics_dir/"last").mkdir(parents=True, exist_ok=True)
        (self.metrics_dir).mkdir(parents=True, exist_ok=True)
        self.health_dir = self.state_dir/"health"

    def _active_job_by_gpu(self):
        # read all healthbeats and build (gpu->current_job) map
        mapping = {}
        if self.health_dir.exists():
            for p in self.health_dir.glob("*.json"):
                try:
                    d = json.load(open(p))
                    gid = d.get("gpu_id")
                    if gid is not None:
                        mapping[int(gid)] = d.get("current_job")
                except Exception:
                    pass
        return mapping

    def write_gpu_samples(self, samples):
        job_map = self._active_job_by_gpu()
        ts = iso_now()
        # append JSONL
        with open(self.metrics_dir/"gpu.jsonl","a") as f:
            for s in samples:
                rec = dict(s)
                rec["node"] = node_name()
                rec["job_id"] = job_map.get(rec["index"])
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
                # last/ cache per gpu
                last = dict(rec)
                json.dump(last, open(self.metrics_dir/"last"/f"gpu{rec['index']}.json","w"))

    def write_system_sample(self, sample):
        rec = dict(sample)
        rec["node"] = node_name()
        with open(self.metrics_dir/"system.jsonl","a") as f:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
