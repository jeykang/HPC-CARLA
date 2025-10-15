#!/usr/bin/env python3
"""
Persistent CARLA health and status monitor.

- Aggregates live per-GPU heartbeat JSON files under $STATE_DIR/health
- Derives node list from current SLURM job (state/current_slurm_job.txt), if present
- Prints a clean table with Node, GPU, Status, Jobs, Age, RPC/TM, Message
- Supports:
    python carla_health_manager.py monitor --interval 30
    python carla_health_manager.py status
    python carla_health_manager.py restart <gpu_id>   # writes a restart request flag
    python carla_health_manager.py cleanup            # removes stale health files
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

STATE_DIR = Path(os.environ.get('STATE_DIR', Path(os.environ.get('PROJECT_ROOT', os.getcwd())) / 'collection_state'))
HEALTH_DIR = STATE_DIR / 'health'
RESTART_DIR = STATE_DIR / 'restart'

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
        dt = datetime.fromisoformat(iso_ts.replace('Z', '+00:00'))
        return max(0.0, (_now_utc() - dt).total_seconds())
    except Exception:
        return float('inf')

def _get_current_job_id() -> str:
    jf = STATE_DIR / 'current_slurm_job.txt'
    if jf.exists():
        return jf.read_text().strip()
    return ""

def _get_nodes_from_job(job_id: str):
    """Return list of nodes from scontrol (best effort)."""
    if not job_id:
        return []
    try:
        out = subprocess.check_output(['bash','-lc', f"scontrol show hostnames $(scontrol show job {job_id} | awk -F= '/NodeList/ {{print $2}}' | tr -d '\n' )"], text=True, stderr=subprocess.DEVNULL)
        nodes = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return nodes
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
        hb = d.get('last_heartbeat') or d.get('timestamp')
        age = _age_sec(hb) if hb else float('inf')
        d['age_sec'] = age
        status = (d.get('status') or 'unknown').lower()
        if age > stale_after and status not in ('down', 'stale'):
            d['status'] = 'stale'
        beats.append(d)
    return beats

def _fmt_age(age):
    if age == float('inf'):
        return '—'
    return f"{int(age):>4}s"

def _print_table(beats, nodes_hint=None):
    nodes_hint = set(nodes_hint or [])
    print("="*100)
    print(f"CARLA HEALTH @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    print(f"{'Node':<22} {'GPU':>3}  {'Status':<8} {'Jobs':>5}  {'Age':>6}  {'RPC':>5}  {'TM':>5}  Message")
    print("-"*100)

    seen_nodes = set()
    busy = idle = stale = 0
    for b in sorted(beats, key=lambda x: (x.get('node','zzz'), int(x.get('gpu_id', -1)))):
        node = b.get('node','?')
        seen_nodes.add(node)
        gpu = b.get('gpu_id', '?')
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

    # Show nodes with no beats yet (allocated but quiet)
    for n in sorted(nodes_hint - seen_nodes):
        print(f"{n:<22} {'—':>3}  {'unknown':<8} {'—':>5}  {'—':>6}  {'—':>5}  {'—':>5}  (no heartbeat)")

    print("-"*100)
    print(f"Summary: busy={busy} idle={idle} stale={stale} total={len(beats)}")
    print("="*100)

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
    """Signal a GPU worker to restart by dropping a request flag file."""
    RESTART_DIR.mkdir(parents=True, exist_ok=True)
    gpu_id = int(args.gpu_id)
    node = args.node or os.environ.get('SLURMD_NODENAME') or os.uname().nodename
    flag = RESTART_DIR / f"{node}_gpu{gpu_id}.restart"
    flag.write_text(_iso())
    print(f"Requested restart for {node} GPU {gpu_id}: {flag}")

def cmd_cleanup(args):
    """Remove stale heartbeat files (older than --stale-after)."""
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
        # Default to status if no args (matches continuous_cli behavior that calls with subcmds)
        args = parser.parse_args(['status'])
    else:
        args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
