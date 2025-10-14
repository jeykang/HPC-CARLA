#!/usr/bin/env python3
"""
CARLA Health Manager for Persistent Instance Collection
Monitors persistent CARLA instances through shared filesystem
Works from login node without direct cluster access
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class CarlaHealthManager:
    """
    Monitors CARLA health through shared filesystem state files.
    Designed to work from login node without direct cluster access.
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.environ.get('PROJECT_ROOT', os.getcwd()))
        self.state_dir = self.project_root / 'collection_state'
        self.health_dir = self.state_dir / 'health'
        self.log_dir = self.project_root / 'logs'

        # Configuration (match persistent scheme)
        self.num_gpus = int(os.environ.get('NUM_GPUS', 8))
        self.base_rpc_port = int(os.environ.get('BASE_RPC_PORT', 2000))
        self.port_spacing = int(os.environ.get('PORT_SPACING', 100))
        self.tm_offset = int(os.environ.get('TM_OFFSET', 5000))

        # Create directories
        self.health_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Health check thresholds
        self.stale_threshold = 120  # seconds before considering health data stale

    def _derive_ports(self, gpu_id: int) -> Tuple[int, int]:
        rpc = self.base_rpc_port + gpu_id * self.port_spacing
        tm = rpc + self.tm_offset
        return rpc, tm

    def get_gpu_health(self, gpu_id: int, node: str = None, path: Path = None) -> Dict:
        """Read health status for a specific GPU from file."""
        if path is not None:
            health_file = Path(path)
        elif node:
            health_file = self.health_dir / node / f'gpu{gpu_id}.json'
        else:
            health_file = self.health_dir / f'gpu{gpu_id}.json'

        rpc, tm = self._derive_ports(gpu_id)

        default_status = {
            'gpu_id': gpu_id,
            'status': 'unknown',
            'message': 'No health data available',
            'node': 'unknown',
            'carla_pid': None,
            'worker_pid': None,
            'rpc_port': rpc,
            'tm_port': tm,
            'timestamp': None,
            'timestamp_unix': 0,
            'is_stale': True,
            'age_seconds': float('inf')
        }

        if not health_file.exists():
            return default_status

        try:
            with open(health_file, 'r') as f:
                health_data = json.load(f)

            current_time = time.time()
            data_time = health_data.get('timestamp_unix', 0)
            age_seconds = current_time - data_time

            health_data['age_seconds'] = age_seconds
            health_data['is_stale'] = age_seconds > self.stale_threshold

            if health_data['is_stale']:
                health_data['status'] = 'stale'
                health_data['message'] = f"No update for {int(age_seconds)}s"

            return health_data

        except (json.JSONDecodeError, IOError) as e:
            default_status['message'] = f"Error reading health file: {e}"
            return default_status

    def get_all_gpu_status(self) -> List[Dict]:
        # Discover both flat and node-subdir formats
        statuses: List[Dict] = []
        # Node subdirs
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
        # Deduplicate by gpu_id preferring node-specific entries
        best: Dict[int, Dict] = {}
        for s in statuses:
            gid = s['gpu_id']
            if gid not in best or s.get('node', 'unknown') != 'unknown':
                best[gid] = s
        ordered = [best[k] for k in sorted(best.keys())]
        return ordered

    def get_collection_status(self) -> Dict:
        queue_file = self.state_dir / 'job_queue.json'
        try:
            with open(queue_file, 'r') as f:
                q = json.load(f)
            return {
                'total': q.get('total', 0),
                'completed': q.get('completed', 0),
                'pending': sum(1 for j in q['jobs'] if j['status'] == 'pending'),
                'running': sum(1 for j in q['jobs'] if j['status'] in ['assigned', 'running']),
                'failed': sum(1 for j in q['jobs'] if j['status'] == 'failed')
            }
        except Exception:
            return {'total': 0, 'completed': 0, 'pending': 0, 'running': 0, 'failed': 0}

    def restart_gpu_worker(self, gpu_id: int, node: Optional[str]=None) -> bool:
        """
        Submit a SLURM job to restart a specific GPU *worker* (client-only).
        IMPORTANT: Do NOT kill the server port here in persistent mode.
        """
        restart_script = self.project_root / 'restart_gpu_worker.sh'

        if node is None:
            h = self.get_gpu_health(gpu_id)
            node = h.get('node') if isinstance(h, dict) else None
        node = node or os.environ.get('SLURMD_NODENAME', '')

        rpc, _tm = self._derive_ports(gpu_id)

        # Build optional SBATCH line without putting a backslash in an f-string expression
        nodelist_line = f"\n#SBATCH --nodelist={node}" if node else ""

        script_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=restart_gpu{gpu_id}\n"
            f"#SBATCH --gres=gpu:1\n"
            f"#SBATCH --time=168:00:00\n"
            f"#SBATCH --output=logs/restart_gpu{gpu_id}_%j.out"
            f"{nodelist_line}\n\n"
            f"set -euo pipefail\n"
            f"export PROJECT_ROOT={self.project_root}\n"
            f"export GPU_ID={gpu_id}\n"
            f"export BASE_RPC_PORT={self.base_rpc_port}\n"
            f"export PORT_SPACING={self.port_spacing}\n"
            f"export TM_OFFSET={self.tm_offset}\n\n"
            f"echo \"[restart] restarting worker on GPU {gpu_id}; expecting server at 127.0.0.1:{rpc}\"\n"
            f"# Client-only worker (must not spawn/stop CARLA)\n"
            f"{self.project_root}/persistent_carla_worker.sh $GPU_ID\n"
        )

        try:
            with open(restart_script, 'w') as f:
                f.write(script_content)
            os.chmod(restart_script, 0o755)
        except Exception as e:
            print(f"Error writing restart script: {e}")
            return False

        try:
            result = subprocess.run(['sbatch', str(restart_script)],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"Submitted restart job for GPU {gpu_id}: Job ID {job_id}")
                return True
            else:
                print(f"Failed to submit restart job: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error submitting restart job: {e}")
            return False

    def monitor(self, interval: int = 30, auto_restart: bool = False) -> None:
        print(f"Monitoring CARLA instances every {interval} seconds.")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                self.print_status()
                if auto_restart:
                    unhealthy = []
                    for status in self.get_all_gpu_status():
                        gid = status['gpu_id']
                        if status['is_stale'] and status.get('gpu_status') == 'busy':
                            unhealthy.append((gid, "Worker not responding"))
                        elif status['status'] in ['error', 'unhealthy']:
                            unhealthy.append((gid, status['message']))
                    if unhealthy:
                        print("\n⚠️  Unhealthy instances detected:")
                        for gid, reason in unhealthy:
                            print(f"  GPU {gid}: {reason}")
                            self.restart_gpu_worker(gid)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

    def print_status(self) -> None:
        try:
            os.system('clear')
        except Exception:
            pass
        print("="*90)
        print(f"CARLA HEALTH MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*90)

        collection = self.get_collection_status()
        print(f"Collection: {collection['completed']}/{collection['total']} completed | "
              f"{collection['running']} running | {collection['pending']} pending | "
              f"{collection['failed']} failed")
        print("-"*90)

        print(f"{'GPU':<4} {'Node':<15} {'Status':<15} {'Jobs':<6} {'Age':<8} {'Message':<30}")
        print("-"*90)

        for status in self.get_all_gpu_status():
            gid = status['gpu_id']
            node = status.get('node', 'unknown')[:15]
            status_text = status['status']
            if status['is_stale']:
                status_display = f"⚠ {status_text}"
            elif status_text in ['ready', 'healthy', 'running_job', 'waiting_for_job']:
                status_display = f"✓ {status_text}"
            else:
                status_display = f"✗ {status_text}"
            age = int(status.get('age_seconds', 0))
            msg = (status.get('message') or '')[:30]
            print(f"{gid:<4} {node:<15} {status_display:<15} {status.get('jobs', 0):<6} {age:<8} {msg:<30}")

def main():
    parser = argparse.ArgumentParser(
        description='CARLA Health Manager - Monitor persistent instances via shared filesystem'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    subparsers.add_parser('status', help='Show current status of all CARLA instances')

    mon = subparsers.add_parser('monitor', help='Continuously monitor instances')
    mon.add_argument('--interval', type=int, default=30)
    mon.add_argument('--auto-restart', action='store_true')

    logp = subparsers.add_parser('log', help='Show worker log')
    logp.add_argument('gpu_id', type=int)
    logp.add_argument('--lines', type=int, default=50)

    rp = subparsers.add_parser('restart', help='Submit SLURM job to restart GPU worker')
    rp.add_argument('--node', help='Target node name (optional)')
    rp.add_argument('gpu_id', nargs='?', type=int)

    subparsers.add_parser('cleanup', help='Clean up health files')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    manager = CarlaHealthManager()

    if args.command == 'status':
        manager.print_status()
    elif args.command == 'monitor':
        manager.monitor(args.interval, args.auto_restart)
    elif args.command == 'log':
        gid = args.gpu_id
        lines = args.lines
        log = manager.log_dir / f'persistent_worker_gpu{gid}.log'
        if log.exists():
            print(log.read_text()[-(lines*200):])
        else:
            print(f"No worker log found for GPU {gid}")
    elif args.command == 'restart':
        if args.gpu_id is not None:
            manager.restart_gpu_worker(args.gpu_id, getattr(args, 'node', None))
        else:
            print("Restarting all GPUs.")
            for gid in range(manager.num_gpus):
                manager.restart_gpu_worker(gid)
                time.sleep(2)
    elif args.command == 'cleanup':
        # simple cleanup: remove stale health files
        for f in manager.health_dir.rglob("gpu*.json"):
            try:
                f.unlink()
            except Exception:
                pass

if __name__ == '__main__':
    main()
