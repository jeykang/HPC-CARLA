#!/usr/bin/env python3
"""
Persistent CARLA health and status monitor (queue-aware).

- Aggregates per-GPU heartbeat JSON files under $STATE_DIR/health
- Cross-references job_queue.json to mark GPUs BUSY when jobs are running
- Derives node list from current SLURM job (state/current_slurm_job.txt), if present
- Prints a clean table with Node, GPU, Status, Jobs, Age, RPC/TM, Message
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', os.getcwd()))
STATE_DIR = Path(os.environ.get('STATE_DIR', PROJECT_ROOT / 'collection_state'))
HEALTH_DIR = STATE_DIR / 'health'
RESTART_DIR = STATE_DIR / 'restart'
QUEUE_FILE = STATE_DIR / 'job_queue.json'

def _now_utc():
    return datetime.now(timezone.utc)

def _iso():
    return _now_utc().isoformat()

def _read_json(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def _age_sec(iso_ts: str) -> float:
    try:
        dt = datetime.fromisoformat((iso_ts or '').replace('Z', '+00:00'))
        return max(0.0, (_now_utc() - dt).total_seconds())
    except Exception:
        return float('inf')

def _get_current_job_id() -> str:
    jf = STATE_DIR / 'current_slurm_job.txt'
    return jf.read_text().strip() if jf.exists() else ""

def _get_nodes_from_job(job_id: str):
    """Return list of nodes from scontrol (best effort)."""
    if not job_id:
        return []
    try:
        out = subprocess.check_output(
            ['bash','-lc',
             f"scontrol show hostnames $(scontrol show job {job_id} | "
             "awk -F= '/NodeList/ {print $2}' | tr -d '\n' )"],
            text=True, stderr=subprocess.DEVNULL
        )
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        return []

def _scan_beats(stale_after=30):
    beats = []
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(HEALTH_DIR.glob('*.json')):
        d = _read_json(p)
        if not isinstance(d, dict):
            continue
        d.setdefault('file', str(p))
        # Try to infer gpu_id from filename if missing
        if 'gpu_id' not in d:
            name = p.stem  # e.g., slurm-nodeA_gpu3
            try:
                if 'gpu' in name:
                    d['gpu_id'] = int(name.split('gpu')[-1])
            except Exception:
                pass
        hb = d.get('last_heartbeat') or d.get('timestamp')
        d['age_sec'] = _age_sec(hb) if hb else float('inf')
        status = (d.get('status') or 'unknown').lower()
        if d['age_sec'] > stale_after and status not in ('down', 'stale', 'busy'):
            d['status'] = 'stale'
        beats.append(d)
    return beats

def _index_running_jobs():
    """Map (node, gpu_id) -> running job minimal info."""
    idx = {}
    q = _read_json(QUEUE_FILE) or {}
    for j in (q.get('jobs') or []):
        if j.get('status') == 'running':
            node = j.get('node') or ''
            gpu  = j.get('gpu')
            if gpu is None:
                continue
            idx[(node, int(gpu))] = {
                "id": j.get('id'),
                "agent": j.get('agent'),
                "route": j.get('route'),
                "weather": j.get('weather'),
                "start_time": j.get('start_time'),
            }
    return idx

def _fmt_age(age):
    if age == float('inf'):
        return '—'
    return f"{int(age):>4}s"

def _print_table(beats, nodes_hint=None):
    nodes_hint = set(nodes_hint or [])
    print("="*110)
    print(f"CARLA HEALTH @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*110)
    print(f"{'Node':<22} {'GPU':>3}  {'Status':<8} {'Jobs':>5}  {'Age':>6}  {'RPC':>5}  {'TM':>5}  Message")
    print("-"*110)

    run_idx = _index_running_jobs()
    seen_nodes = set()
    busy = idle = stale = 0

    for b in sorted(beats, key=lambda x: (x.get('node','zzz'), int(x.get('gpu_id', -1)))):
        node = b.get('node','?')
        seen_nodes.add(node)
        gpu = b.get('gpu_id', '?')

        # Overlay queue truth: running job => busy (and show job metadata)
        run = run_idx.get((node, int(gpu))) if isinstance(gpu, int) or (isinstance(gpu, str) and gpu.isdigit()) else None
        if run:
            b['status'] = 'busy'
            b['message'] = f"job {run['id']} {run['agent']}/{run['route']} w{run['weather']}"

        st = (b.get('status') or 'unknown').lower()
        if st == 'busy': busy += 1
        elif st == 'idle': idle += 1
        elif st == 'stale': stale += 1

        jobs = b.get('jobs_completed') or b.get('jobs') or '-'
        age = _fmt_age(b.get('age_sec', float('inf')))
        rpc = b.get('rpc_port') or '-'
        tm  = b.get('tm_port') or '-'
        msg = b.get('message') or ''
        print(f"{node:<22} {str(gpu):>3}  {st:<8} {str(jobs):>5}  {age:>6}  {str(rpc):>5}  {str(tm):>5}  {msg}")

    # Nodes with no beats (allocated but quiet)
    for n in sorted(nodes_hint - seen_nodes):
        print(f"{n:<22} {'—':>3}  {'unknown':<8} {'—':>5}  {'—':>6}  {'—':>5}  {'—':>5}  (no heartbeat)")

    print("-"*110)
    print(f"Summary: busy={busy} idle={idle} stale={stale} total={len(beats)}")
    print("="*110)

def cmd_status(args):
    job_id = _get_current_job_id()
    nodes = _get_nodes_from_job(job_id)
    beats = _scan_beats(stale_after=args.stale_after)
    _print_table(beats, nodes_hint=nodes)

def cmd_monitor(args):
    try:
        while True:
            os.system('clear')
            cmd_status(args)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

def cmd_restart(args):
    RESTART_DIR.mkdir(parents=True, exist_ok=True)
    gpu_id = int(args.gpu_id)
    node = args.node or os.environ.get('SLURMD_NODENAME') or os.uname().nodename
    flag = RESTART_DIR / f"{node}_gpu{gpu_id}.restart"
    flag.write_text(_iso())
    print(f"Requested restart for {node} GPU {gpu_id}: {flag}")

def cmd_cleanup(args):
    beats = _scan_beats(stale_after=args.stale_after)
    removed = 0
    for b in beats:
        if b.get('status') in ('stale', 'down') and b.get('file'):
            try:
                Path(b['file']).unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    print(f"Removed {removed} stale heartbeat files.")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_status = sub.add_parser('status', help='print one-time health table')
    p_status.add_argument('--stale-after', type=int, default=30)
    p_status.set_defaults(func=cmd_status)

    p_monitor = sub.add_parser('monitor', help='continuous monitor (clears screen)')
    p_monitor.add_argument('--interval', type=int, default=30)
    p_monitor.add_argument('--stale-after', type=int, default=30)
    p_monitor.set_defaults(func=cmd_monitor)

    p_restart = sub.add_parser('restart', help='request restart for a GPU')
    p_restart.add_argument('gpu_id', type=int)
    p_restart.add_argument('--node', type=str, default=None)
    p_restart.set_defaults(func=cmd_restart)

    p_cleanup = sub.add_parser('cleanup', help='remove stale heartbeat files')
    p_cleanup.add_argument('--stale-after', type=int, default=300)
    p_cleanup.set_defaults(func=cmd_cleanup)

    if len(sys.argv) == 1:
        args = parser.parse_args(['status'])
    else:
        args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
