#!/usr/bin/env python3
"""
Real-time monitoring dashboard for continuous data collection
Fixed to show actual GPU count and node information
"""

import json
import os
import sys
import time
import curses
from datetime import datetime, timedelta
from pathlib import Path
import argparse

class ContinuousMonitor:
    def __init__(self, state_dir=None):
        if state_dir is None:
            project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
            state_dir = os.path.join(project_root, 'collection_state')
        
        self.state_dir = Path(state_dir)
        self.queue_file = self.state_dir / 'job_queue.json'
        self.gpu_status_file = self.state_dir / 'gpu_status.json'
        self.runtime_file = self.state_dir / 'runtime_estimates.json'
        self.completed_file = self.state_dir / 'completed_jobs.json'
        self.node_status_file = self.state_dir / 'node_status.json'
        self.carla_servers_file = self.state_dir / 'carla_servers.json'
        
    def load_state(self):
        """Load current state from files"""
        try:
            with open(self.queue_file, 'r') as f:
                self.queue_data = json.load(f)
            with open(self.gpu_status_file, 'r') as f:
                self.gpu_status = json.load(f)
            with open(self.runtime_file, 'r') as f:
                self.runtime_data = json.load(f)
            with open(self.completed_file, 'r') as f:
                self.completed_data = json.load(f)
            
            # Try to load node status if available (for multi-node setups)
            self.node_status = {}
            if self.node_status_file.exists():
                with open(self.node_status_file, 'r') as f:
                    node_data = json.load(f)
                    self.node_status = node_data.get('nodes', {})
            
            # Try to load CARLA server status if available (for persistent mode)
            self.carla_servers = {}
            if self.carla_servers_file.exists():
                with open(self.carla_servers_file, 'r') as f:
                    carla_data = json.load(f)
                    self.carla_servers = carla_data.get('servers', {})
            
            return True
        except FileNotFoundError as e:
            return False
        except json.JSONDecodeError:
            return False
    
    def get_actual_gpu_count(self):
        """Detect the actual number of GPUs from the status file"""
        # Count GPUs in gpu_status
        gpu_count = len(self.gpu_status)
        
        # If we have CARLA servers info, use that as it's more accurate
        if self.carla_servers:
            gpu_count = max(gpu_count, len(self.carla_servers))
        
        # Check environment variable
        env_gpu_count = os.environ.get('NUM_GPUS')
        if env_gpu_count:
            try:
                gpu_count = int(env_gpu_count)
            except ValueError:
                pass
        
        # Ensure at least 1 GPU
        return max(1, gpu_count)
    
    def get_node_for_gpu(self, gpu_id):
        """Get node identifier for a GPU"""
        gpu_str = str(gpu_id)
        
        # Method 1: Check if GPU ID contains node info (e.g., "0_1" for node 0, gpu 1)
        if '_' in gpu_str:
            parts = gpu_str.split('_')
            if len(parts) == 2:
                return f"Node {parts[0]}"
        
        # Method 2: Check current job for node assignment
        gpu_info = self.gpu_status.get(gpu_str, {})
        current_job_id = gpu_info.get('current_job')
        if current_job_id is not None:
            # Find job in queue
            for job in self.queue_data.get('jobs', []):
                if job['id'] == current_job_id:
                    if 'node' in job:
                        return f"Node {job['node']}"
                    if 'completed_by_node' in job:
                        return f"Node {job['completed_by_node']}"
        
        # Method 3: Check completed jobs for this GPU
        for job in self.completed_data.get('jobs', []):
            if job.get('gpu') == gpu_id:
                if 'node' in job:
                    return f"Node {job['node']}"
                if 'completed_by_node' in job:
                    return f"Node {job['completed_by_node']}"
        
        # Method 4: Check node status file
        for node_id, node_info in self.node_status.items():
            if node_info.get('gpus'):
                # Assume GPUs are numbered sequentially per node
                gpus_per_node = node_info['gpus']
                node_num = int(node_id) if node_id.isdigit() else 0
                gpu_start = node_num * gpus_per_node
                gpu_end = gpu_start + gpus_per_node
                if gpu_id >= gpu_start and gpu_id < gpu_end:
                    hostname = node_info.get('hostname', f'node{node_id}')
                    return hostname
        
        # Method 5: Try to determine from SLURM environment
        slurm_nodename = os.environ.get('SLURMD_NODENAME')
        if slurm_nodename:
            return slurm_nodename
        
        # Default: Check if we're in single-node or multi-node mode
        num_nodes = len(self.node_status) if self.node_status else 1
        if num_nodes > 1:
            # Multi-node: try to calculate which node
            gpus_per_node = self.get_actual_gpu_count() // num_nodes
            if gpus_per_node > 0:
                node_id = gpu_id // gpus_per_node
                return f"Node {node_id}"
        
        # Single node or unknown
        import socket
        return socket.gethostname().split('.')[0]  # Short hostname
    
    def get_statistics(self):
        """Calculate current statistics"""
        stats = {
            'total': self.queue_data['total'],
            'completed': self.queue_data['completed'],
            'pending': sum(1 for j in self.queue_data['jobs'] if j['status'] == 'pending'),
            'running': sum(1 for j in self.queue_data['jobs'] if j['status'] in ['assigned', 'running']),
            'failed': sum(1 for j in self.queue_data['jobs'] if j['status'] == 'failed'),
            'retry': sum(1 for j in self.queue_data['jobs'] if j['status'] == 'failed' and j['attempts'] < 3)
        }
        
        # Calculate completion percentage
        stats['completion_pct'] = (stats['completed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Calculate average runtime
        completed_jobs = self.completed_data.get('jobs', [])
        if completed_jobs:
            durations = [j['duration'] for j in completed_jobs if j.get('duration')]
            stats['avg_duration'] = sum(durations) / len(durations) if durations else 0
        else:
            stats['avg_duration'] = 0
        
        # Estimate time remaining
        if stats['completed'] > 0 and stats['pending'] > 0:
            active_gpus = sum(1 for g in self.gpu_status.values() if g['status'] != 'offline')
            if active_gpus > 0:
                stats['eta'] = (stats['pending'] * stats['avg_duration']) / active_gpus
            else:
                stats['eta'] = float('inf')
        else:
            stats['eta'] = 0
        
        # Add GPU count
        stats['gpu_count'] = self.get_actual_gpu_count()
        
        return stats
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        gpu_info = []
        current_time = datetime.now().timestamp()
        
        # Get actual GPU IDs from status file
        gpu_ids = []
        
        # Collect all GPU IDs from gpu_status
        for gpu_str in self.gpu_status.keys():
            # Handle both simple IDs (0, 1, 2) and node-based IDs (0_0, 0_1, 1_0)
            try:
                if '_' in gpu_str:
                    # Node-based ID
                    gpu_ids.append(gpu_str)
                else:
                    # Simple ID
                    gpu_ids.append(int(gpu_str))
            except (ValueError, TypeError):
                gpu_ids.append(gpu_str)
        
        # Sort IDs (handle mixed types)
        gpu_ids.sort(key=lambda x: (str(x) if isinstance(x, str) else f"{x:03d}"))
        
        # If no GPUs in status, use the detected count
        if not gpu_ids:
            gpu_ids = list(range(self.get_actual_gpu_count()))
        
        for gpu_id in gpu_ids:
            gpu_str = str(gpu_id)
            
            if gpu_str not in self.gpu_status:
                # GPU not in status file, show as offline
                info = {
                    'id': gpu_id,
                    'node': self.get_node_for_gpu(gpu_id),
                    'status': 'offline',
                    'jobs_completed': 0,
                    'current_job': None,
                    'progress': ''
                }
            else:
                gpu_data = self.gpu_status[gpu_str]
                info = {
                    'id': gpu_id,
                    'node': self.get_node_for_gpu(gpu_id),
                    'status': gpu_data['status'],
                    'jobs_completed': gpu_data['jobs_completed'],
                    'current_job': None,
                    'progress': ''
                }
                
                if gpu_data['status'] == 'busy' and gpu_data['current_job'] is not None:
                    # Find current job details
                    current_job = next((j for j in self.queue_data['jobs'] 
                                      if j['id'] == gpu_data['current_job']), None)
                    if current_job:
                        info['current_job'] = current_job
                        
                        # Calculate progress
                        if current_job.get('start_time'):
                            elapsed = current_time - current_job['start_time']
                            key = f"{current_job['agent']}_{current_job['route']}"
                            expected = self.runtime_data['combinations'].get(key, 
                                                                            self.runtime_data['default'])
                            progress_pct = min(100, (elapsed / expected) * 100)
                            info['progress'] = f"{progress_pct:.0f}%"
                            info['elapsed'] = elapsed
                            info['expected'] = expected
            
            gpu_info.append(info)
        
        return gpu_info
    
    def get_recent_jobs(self, limit=5):
        """Get recently completed jobs"""
        completed = self.completed_data.get('jobs', [])
        return completed[-limit:] if completed else []
    
    def get_job_distribution(self):
        """Get distribution of jobs by agent and status"""
        distribution = {}
        
        for job in self.queue_data['jobs']:
            agent = job['agent']
            if agent not in distribution:
                distribution[agent] = {
                    'completed': 0, 'pending': 0, 'running': 0, 'failed': 0
                }
            
            status = job['status']
            if status in ['assigned', 'running']:
                distribution[agent]['running'] += 1
            elif status == 'completed':
                distribution[agent]['completed'] += 1
            elif status == 'failed':
                distribution[agent]['failed'] += 1
            elif status == 'pending':
                distribution[agent]['pending'] += 1
        
        return distribution
    
    def render_dashboard(self, stdscr):
        """Render the monitoring dashboard using curses"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)    # Non-blocking input
        
        # Color pairs
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
        
        while True:
            # Check for quit command
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                break
            
            # Load current state
            if not self.load_state():
                stdscr.clear()
                stdscr.addstr(0, 0, "Error: Cannot read state files. Is collection running?", 
                            curses.color_pair(3))
                stdscr.refresh()
                time.sleep(2)
                continue
            
            # Get dimensions
            height, width = stdscr.getmaxyx()
            
            # Clear screen
            stdscr.clear()
            
            # Header
            header = "CONTINUOUS DATA COLLECTION MONITOR"
            stdscr.addstr(0, (width - len(header)) // 2, header, 
                        curses.A_BOLD | curses.color_pair(4))
            stdscr.addstr(1, 0, "=" * min(width - 1, 80))
            
            # Statistics
            stats = self.get_statistics()
            row = 3
            
            # Progress bar
            progress_width = min(50, width - 20)
            filled = int(progress_width * stats['completion_pct'] / 100)
            progress_bar = f"[{'#' * filled}{'-' * (progress_width - filled)}]"
            stdscr.addstr(row, 0, f"Progress: {progress_bar} {stats['completion_pct']:.1f}%")
            row += 2
            
            # Job counts
            stdscr.addstr(row, 0, f"Total Jobs: {stats['total']}", curses.color_pair(5))
            stdscr.addstr(row, 20, f"Completed: {stats['completed']}", curses.color_pair(1))
            stdscr.addstr(row, 40, f"Running: {stats['running']}", curses.color_pair(2))
            row += 1
            stdscr.addstr(row, 20, f"Pending: {stats['pending']}", curses.color_pair(5))
            stdscr.addstr(row, 40, f"Failed: {stats['failed']}", curses.color_pair(3))
            row += 2
            
            # Time estimates
            if stats['avg_duration'] > 0:
                avg_time = str(timedelta(seconds=int(stats['avg_duration'])))
                stdscr.addstr(row, 0, f"Avg Duration: {avg_time}")
                if stats['eta'] > 0 and stats['eta'] != float('inf'):
                    eta_time = str(timedelta(seconds=int(stats['eta'])))
                    stdscr.addstr(row, 25, f"ETA: {eta_time}")
            row += 2
            
            # GPU Status with node info
            num_gpus = stats['gpu_count']
            stdscr.addstr(row, 0, f"GPU STATUS ({num_gpus} GPUs):", curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, "-" * min(width - 1, 80))
            row += 1
            
            gpu_info = self.get_gpu_info()
            for gpu in gpu_info:
                node_str = gpu.get('node', 'Unknown')
                
                if gpu['status'] == 'idle':
                    status_str = f"{node_str} GPU {gpu['id']}: IDLE"
                    color = curses.color_pair(5)
                elif gpu['status'] == 'busy':
                    if gpu['current_job']:
                        job = gpu['current_job']
                        # Truncate route name if needed
                        route_display = job['route'][:15] if len(job['route']) > 15 else job['route']
                        status_str = f"{node_str} GPU {gpu['id']}: {job['agent']}/{route_display} {gpu['progress']}"
                    else:
                        status_str = f"{node_str} GPU {gpu['id']}: BUSY"
                    color = curses.color_pair(2)
                elif gpu['status'] == 'offline':
                    status_str = f"{node_str} GPU {gpu['id']}: OFFLINE"
                    color = curses.color_pair(3)
                else:
                    status_str = f"{node_str} GPU {gpu['id']}: {gpu['status'].upper()}"
                    color = curses.color_pair(5)
                
                # Ensure string fits in terminal width
                max_status_len = width - 15
                if len(status_str) > max_status_len:
                    status_str = status_str[:max_status_len-3] + "..."
                
                stdscr.addstr(row, 0, status_str, color)
                stdscr.addstr(row, min(width-15, 60), f"Done: {gpu['jobs_completed']}")
                row += 1
                
                if row >= height - 5:
                    break  # Prevent overflow
            row += 1
            
            # Job Distribution by Agent (if space allows)
            if row < height - 5:
                stdscr.addstr(row, 0, "AGENT DISTRIBUTION:", curses.A_BOLD)
                row += 1
                distribution = self.get_job_distribution()
                for agent, counts in list(distribution.items())[:3]:  # Show top 3 agents
                    if row >= height - 2:
                        break
                    total_agent = sum(counts.values())
                    done_pct = (counts['completed'] / total_agent * 100) if total_agent > 0 else 0
                    status_str = f"{agent:12} [{counts['completed']:3}/{total_agent:3}] {done_pct:5.1f}%"
                    stdscr.addstr(row, 0, status_str[:width-1])
                    row += 1
            
            # Footer
            footer = "Press 'q' to quit | Updates every 2 seconds"
            if height > row + 2:
                stdscr.addstr(height - 1, 0, footer, curses.color_pair(4))
            
            # Refresh
            stdscr.refresh()
            time.sleep(2)
    
    def print_summary(self):
        """Print a text summary (non-interactive)"""
        if not self.load_state():
            print("Error: Cannot read state files. Is collection running?")
            return
        
        stats = self.get_statistics()
        gpu_info = self.get_gpu_info()
        
        print("\n" + "="*60)
        print(f"CONTINUOUS COLLECTION STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Progress bar
        progress_width = 40
        filled = int(progress_width * stats['completion_pct'] / 100)
        progress_bar = f"[{'#' * filled}{'-' * (progress_width - filled)}]"
        print(f"Progress: {progress_bar} {stats['completion_pct']:.1f}%")
        print()
        
        print(f"Total Jobs:     {stats['total']}")
        print(f"Completed:      {stats['completed']}")
        print(f"Running:        {stats['running']}")
        print(f"Pending:        {stats['pending']}")
        print(f"Failed:         {stats['failed']}")
        
        if stats['avg_duration'] > 0:
            avg_time = str(timedelta(seconds=int(stats['avg_duration'])))
            print(f"Avg Duration:   {avg_time}")
            if stats['eta'] > 0 and stats['eta'] != float('inf'):
                eta_time = str(timedelta(seconds=int(stats['eta'])))
                print(f"ETA:            {eta_time}")
        
        print("-"*60)
        print(f"GPU Status ({stats['gpu_count']} GPUs in use):")
        for gpu in gpu_info:
            node_str = gpu.get('node', 'Unknown')
            if gpu['status'] == 'busy' and gpu['current_job']:
                job = gpu['current_job']
                print(f"  {node_str} GPU {gpu['id']}: {job['agent']}/{job['route']} ({gpu['progress']}) | Completed: {gpu['jobs_completed']}")
            elif gpu['status'] == 'offline':
                print(f"  {node_str} GPU {gpu['id']}: OFFLINE | Completed: {gpu['jobs_completed']}")
            else:
                print(f"  {node_str} GPU {gpu['id']}: {gpu['status'].upper()} | Completed: {gpu['jobs_completed']}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Monitor continuous data collection')
    
    # Set default based on PROJECT_ROOT environment variable or current directory
    default_state_dir = os.path.join(
        os.environ.get('PROJECT_ROOT', os.getcwd()),
        'collection_state'
    )
    
    parser.add_argument('--state-dir', default=default_state_dir,
                       help='Path to state directory')
    parser.add_argument('--once', action='store_true',
                       help='Print summary once and exit')
    args = parser.parse_args()
    
    monitor = ContinuousMonitor(args.state_dir)
    
    if args.once:
        monitor.print_summary()
    else:
        try:
            curses.wrapper(monitor.render_dashboard)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\nError: {e}")
            print("\nFalling back to text mode...")
            while True:
                try:
                    monitor.print_summary()
                    time.sleep(5)
                except KeyboardInterrupt:
                    break


if __name__ == '__main__':
    main()