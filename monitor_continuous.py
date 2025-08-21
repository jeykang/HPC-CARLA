#!/usr/bin/env python3
"""
Real-time monitoring dashboard for continuous data collection
Provides live updates on job progress, GPU utilization, and estimates
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
            return True
        except FileNotFoundError as e:
            return False
        except json.JSONDecodeError:
            return False
    
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
        
        return stats
    
    def get_gpu_info(self):
        """Get detailed GPU information"""
        gpu_info = []
        current_time = datetime.now().timestamp()
        
        for gpu_id in range(8):
            gpu_str = str(gpu_id)
            if gpu_str not in self.gpu_status:
                continue
                
            gpu_data = self.gpu_status[gpu_str]
            info = {
                'id': gpu_id,
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
            
            # GPU Status
            stdscr.addstr(row, 0, "GPU STATUS:", curses.A_BOLD)
            row += 1
            stdscr.addstr(row, 0, "-" * min(width - 1, 80))
            row += 1
            
            gpu_info = self.get_gpu_info()
            for gpu in gpu_info:
                if gpu['status'] == 'idle':
                    status_str = f"GPU {gpu['id']}: IDLE"
                    color = curses.color_pair(5)
                elif gpu['status'] == 'busy':
                    if gpu['current_job']:
                        job = gpu['current_job']
                        status_str = f"GPU {gpu['id']}: {job['agent']}/{job['route'][:15]} {gpu['progress']}"
                    else:
                        status_str = f"GPU {gpu['id']}: BUSY"
                    color = curses.color_pair(2)
                else:
                    status_str = f"GPU {gpu['id']}: OFFLINE"
                    color = curses.color_pair(3)
                
                stdscr.addstr(row, 0, status_str[:width-15], color)
                stdscr.addstr(row, min(width-15, 50), f"Done: {gpu['jobs_completed']}")
                row += 1
            row += 1
            
            # Job Distribution by Agent
            if row < height - 5:
                stdscr.addstr(row, 0, "AGENT DISTRIBUTION:", curses.A_BOLD)
                row += 1
                distribution = self.get_job_distribution()
                for agent, counts in distribution.items():
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
        print("GPU Status:")
        for gpu in gpu_info:
            if gpu['status'] == 'busy' and gpu['current_job']:
                job = gpu['current_job']
                print(f"  GPU {gpu['id']}: {job['agent']}/{job['route']} ({gpu['progress']}) | Completed: {gpu['jobs_completed']}")
            else:
                print(f"  GPU {gpu['id']}: {gpu['status'].upper()} | Completed: {gpu['jobs_completed']}")
        
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