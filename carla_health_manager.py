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
from datetime import datetime, timedelta
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
        
        # Configuration
        self.num_gpus = int(os.environ.get('NUM_GPUS', 8))
        self.base_rpc_port = int(os.environ.get('BASE_RPC_PORT', 2000))
        self.base_tm_port = int(os.environ.get('BASE_TM_PORT', 8000))
        
        # Create directories
        self.health_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Health check thresholds
        self.stale_threshold = 120  # seconds before considering health data stale
    
    def get_gpu_health(self, gpu_id: int) -> Dict:
        """Read health status for a specific GPU from file"""
        health_file = self.health_dir / f'gpu{gpu_id}.json'
        
        default_status = {
            'gpu_id': gpu_id,
            'status': 'unknown',
            'message': 'No health data available',
            'node': 'unknown',
            'carla_pid': None,
            'worker_pid': None,
            'rpc_port': self.base_rpc_port + gpu_id * 10,
            'tm_port': self.base_tm_port + gpu_id * 10,
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
            
            # Calculate age of health data
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
    
    def get_gpu_job_status(self, gpu_id: int) -> Dict:
        """Get current job status for a GPU from the queue file"""
        gpu_status_file = self.state_dir / 'gpu_status.json'
        
        if not gpu_status_file.exists():
            return {'current_job': None, 'jobs_completed': 0}
        
        try:
            with open(gpu_status_file, 'r') as f:
                gpu_data = json.load(f)
                
            gpu_info = gpu_data.get(str(gpu_id), {})
            return {
                'current_job': gpu_info.get('current_job'),
                'jobs_completed': gpu_info.get('jobs_completed', 0),
                'gpu_status': gpu_info.get('status', 'unknown')
            }
        except (json.JSONDecodeError, IOError):
            return {'current_job': None, 'jobs_completed': 0}
    
    def get_all_gpu_status(self) -> List[Dict]:
        """Get combined health and job status for all GPUs"""
        all_status = []
        
        for gpu_id in range(self.num_gpus):
            health = self.get_gpu_health(gpu_id)
            job_status = self.get_gpu_job_status(gpu_id)
            
            # Combine health and job status
            combined = {**health, **job_status}
            all_status.append(combined)
        
        return all_status
    
    def get_collection_status(self) -> Dict:
        """Get overall collection status from queue file"""
        queue_file = self.state_dir / 'job_queue.json'
        
        if not queue_file.exists():
            return {
                'total': 0,
                'completed': 0,
                'pending': 0,
                'running': 0,
                'failed': 0
            }
        
        try:
            with open(queue_file, 'r') as f:
                queue_data = json.load(f)
            
            return {
                'total': queue_data.get('total', 0),
                'completed': queue_data.get('completed', 0),
                'pending': sum(1 for j in queue_data['jobs'] if j['status'] == 'pending'),
                'running': sum(1 for j in queue_data['jobs'] if j['status'] in ['assigned', 'running']),
                'failed': sum(1 for j in queue_data['jobs'] if j['status'] == 'failed')
            }
        except (json.JSONDecodeError, IOError):
            return {
                'total': 0,
                'completed': 0,
                'pending': 0,
                'running': 0,
                'failed': 0
            }
    
    def restart_gpu_worker(self, gpu_id: int) -> bool:
        """
        Submit a SLURM job to restart a specific GPU worker.
        This is the only way to restart from login node.
        """
        restart_script = self.project_root / 'restart_gpu_worker.sh'
        
        # Create restart script if it doesn't exist
        if not restart_script.exists():
            script_content = f"""#!/bin/bash
#SBATCH --job-name=restart_gpu{gpu_id}
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/restart_gpu{gpu_id}_%j.out

export PROJECT_ROOT={self.project_root}
export GPU_ID={gpu_id}

# Source environment
source {self.project_root}/env_setup.sh 2>/dev/null || true

# Kill any existing CARLA on this GPU's port
fuser -k {self.base_rpc_port + gpu_id * 10}/tcp 2>/dev/null || true
sleep 5

# Start persistent worker
{self.project_root}/persistent_carla_worker.sh $GPU_ID
"""
            with open(restart_script, 'w') as f:
                f.write(script_content)
            os.chmod(restart_script, 0o755)
        
        # Submit restart job
        try:
            result = subprocess.run(
                ['sbatch', str(restart_script)],
                capture_output=True,
                text=True
            )
            
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
        """Continuously monitor CARLA instances through health files"""
        print(f"Monitoring CARLA instances every {interval} seconds...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.print_status()
                
                if auto_restart:
                    # Check for unhealthy instances
                    unhealthy = []
                    all_status = self.get_all_gpu_status()
                    
                    for status in all_status:
                        gpu_id = status['gpu_id']
                        
                        # Check various failure conditions
                        if status['is_stale'] and status['gpu_status'] == 'busy':
                            unhealthy.append((gpu_id, "Worker not responding"))
                        elif status['status'] in ['error', 'unhealthy']:
                            unhealthy.append((gpu_id, status['message']))
                    
                    if unhealthy:
                        print("\n⚠️  Unhealthy instances detected:")
                        for gpu_id, reason in unhealthy:
                            print(f"  GPU {gpu_id}: {reason}")
                            if auto_restart:
                                print(f"  Attempting to restart GPU {gpu_id}...")
                                self.restart_gpu_worker(gpu_id)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
    
    def print_status(self) -> None:
        """Print status table for all GPUs"""
        os.system('clear')  # Clear screen for better display
        
        print("="*90)
        print(f"CARLA HEALTH MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*90)
        
        # Get collection status
        collection = self.get_collection_status()
        print(f"Collection: {collection['completed']}/{collection['total']} completed | "
              f"{collection['running']} running | {collection['pending']} pending | "
              f"{collection['failed']} failed")
        print("-"*90)
        
        # Headers
        print(f"{'GPU':<4} {'Node':<15} {'Status':<15} {'Jobs':<6} {'Age':<8} {'Message':<30}")
        print("-"*90)
        
        all_status = self.get_all_gpu_status()
        
        for status in all_status:
            gpu_id = status['gpu_id']
            node = status.get('node', 'unknown')[:15]
            
            # Status coloring (terminal colors)
            status_text = status['status']
            if status['is_stale']:
                status_display = f"⚠ {status_text}"
            elif status_text in ['ready', 'healthy', 'running_job', 'waiting_for_job']:
                status_display = f"✓ {status_text}"
            elif status_text in ['error', 'unhealthy', 'stopped']:
                status_display = f"✗ {status_text}"
            else:
                status_display = f"? {status_text}"
            
            # Age display
            age = status.get('age_seconds', float('inf'))
            if age == float('inf'):
                age_str = "No data"
            elif age < 60:
                age_str = f"{int(age)}s"
            elif age < 3600:
                age_str = f"{int(age/60)}m"
            else:
                age_str = f"{int(age/3600)}h"
            
            # Message truncation
            message = status.get('message', '')[:30]
            
            # Jobs completed
            jobs = status.get('jobs_completed', 0)
            
            print(f"{gpu_id:<4} {node:<15} {status_display:<15} {jobs:<6} {age_str:<8} {message:<30}")
        
        print("="*90)
        
        # Show any recent worker logs with errors
        self.check_recent_errors()
    
    def check_recent_errors(self) -> None:
        """Check for recent errors in worker logs"""
        error_count = 0
        
        for gpu_id in range(self.num_gpus):
            log_file = self.log_dir / f'persistent_worker_gpu{gpu_id}.log'
            
            if log_file.exists():
                try:
                    # Get last 100 lines of log
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]
                    
                    # Look for recent errors
                    for line in lines:
                        if 'ERROR' in line or 'FATAL' in line:
                            error_count += 1
                            if error_count == 1:
                                print("\nRecent Errors:")
                            print(f"  GPU {gpu_id}: {line.strip()[:80]}")
                            
                except IOError:
                    pass
    
    def show_worker_log(self, gpu_id: int, lines: int = 50) -> None:
        """Display recent lines from a worker log"""
        log_file = self.log_dir / f'persistent_worker_gpu{gpu_id}.log'
        
        if not log_file.exists():
            print(f"No log file found for GPU {gpu_id}")
            return
        
        print(f"\n=== Last {lines} lines from GPU {gpu_id} worker log ===\n")
        
        try:
            with open(log_file, 'r') as f:
                log_lines = f.readlines()[-lines:]
                for line in log_lines:
                    print(line.rstrip())
        except IOError as e:
            print(f"Error reading log file: {e}")
    
    def cleanup_health_files(self) -> None:
        """Remove all health status files"""
        for health_file in self.health_dir.glob('gpu*.json'):
            health_file.unlink()
        print(f"Cleaned up health files in {self.health_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='CARLA Health Manager - Monitor persistent instances via shared filesystem'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    subparsers.add_parser('status', help='Show current status of all CARLA instances')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Continuously monitor instances')
    monitor_parser.add_argument('--interval', type=int, default=30,
                               help='Update interval in seconds (default: 30)')
    monitor_parser.add_argument('--auto-restart', action='store_true',
                               help='Automatically submit restart jobs for unhealthy instances')
    
    # Log command
    log_parser = subparsers.add_parser('log', help='Show worker log')
    log_parser.add_argument('gpu_id', type=int, help='GPU ID')
    log_parser.add_argument('--lines', type=int, default=50,
                           help='Number of lines to show (default: 50)')
    
    # Restart command (submits SLURM job)
    restart_parser = subparsers.add_parser('restart', 
                                          help='Submit SLURM job to restart GPU worker')
    restart_parser.add_argument('gpu_id', nargs='?', type=int,
                               help='GPU ID to restart (all if not specified)')
    
    # Cleanup command
    subparsers.add_parser('cleanup', help='Clean up health files')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create manager
    manager = CarlaHealthManager()
    
    # Execute command
    if args.command == 'status':
        manager.print_status()
    
    elif args.command == 'monitor':
        manager.monitor(args.interval, args.auto_restart)
    
    elif args.command == 'log':
        manager.show_worker_log(args.gpu_id, args.lines)
    
    elif args.command == 'restart':
        if args.gpu_id is not None:
            manager.restart_gpu_worker(args.gpu_id)
        else:
            print("Restarting all GPUs...")
            for gpu_id in range(manager.num_gpus):
                manager.restart_gpu_worker(gpu_id)
                time.sleep(2)  # Stagger submissions
    
    elif args.command == 'cleanup':
        manager.cleanup_health_files()


if __name__ == '__main__':
    main()