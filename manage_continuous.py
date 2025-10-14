#!/usr/bin/env python3
"""
Management utility for continuous data collection system
Provides control over job queue, retries, and collection parameters
"""

import json
import os
import sys
import subprocess
import time
from typing import Optional
import argparse
import fcntl
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

class ContinuousManager:
    def __init__(self, state_dir: str = None):
        if state_dir is None:
            project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
            state_dir = os.path.join(project_root, 'collection_state')
        
        self.project_root = Path(os.environ.get('PROJECT_ROOT', os.getcwd()))
        self.state_dir = Path(state_dir)
        self.queue_file = self.state_dir / 'job_queue.json'
        self.gpu_status_file = self.state_dir / 'gpu_status.json'
        self.runtime_file = self.state_dir / 'runtime_estimates.json'
        self.completed_file = self.state_dir / 'completed_jobs.json'
        self.lock_file = self.state_dir / '.scheduler.lock'

        self.node_name = os.environ.get('SLURMD_NODENAME') or os.uname().nodename
        # Prefer per-node GPU count; fall back to total or 8
        self.local_gpus = int(os.environ.get('LOCAL_GPUS')
                              or os.environ.get('GPUS_PER_NODE')
                              or os.environ.get('NUM_GPUS', 8))
        
        # Paths for discovering available routes and scenarios
        self.routes_dir = self.project_root / 'leaderboard/data/training_routes'
        self.scenarios_dir = self.project_root / 'leaderboard/data/scenarios'
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _load_gpu_status(self) -> Dict[str, Any]:
        """Load gpu_status.json, supporting both legacy (flat) and v2 (namespaced) formats."""
        if self.gpu_status_file.exists():
            try:
                with open(self.gpu_status_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        # Default to v2 namespaced schema
        return {"schema": 2, "nodes": {}}

    def _ensure_node_gpu_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure current node has entries 0..local_gpus-1 without clobbering other nodes."""
        if "nodes" not in data or not isinstance(data["nodes"], dict):
            # Upgrade legacy flat -> v2
            legacy = data if isinstance(data, dict) else {}
            data = {"schema": 2, "nodes": {self.node_name: {}}}
            for k, v in legacy.items():
                if str(k).isdigit():
                    data["nodes"][self.node_name][str(k)] = v
        nodes = data.setdefault("nodes", {})
        node_map = nodes.setdefault(self.node_name, {})
        for i in range(self.local_gpus):
            key = str(i)
            if key not in node_map:
                node_map[key] = {
                    "status": "idle",
                    "current_job": None,
                    "jobs_completed": 0,
                    "total_runtime": 0,
                    "last_heartbeat": None
                }
        return data
    
    def _with_lock(self, func):
        """Execute function with file lock"""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return func()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    
    def _discover_routes_and_scenarios(self) -> Dict[str, List[str]]:
        """
        Discover available route files and their corresponding scenarios.
        Returns a mapping of town -> list of route files for that town.
        """
        town_routes = {}
        
        if not self.routes_dir.exists():
            print(f"Warning: Routes directory not found at {self.routes_dir}")
            return {}
        
        # Find all route XML files
        route_files = list(self.routes_dir.glob('*.xml'))
        
        for route_file in route_files:
            # Extract town number from filename (e.g., routes_town01_short.xml -> 01)
            match = re.search(r'routes_town(\d+)_', route_file.name)
            if match:
                town_num = match.group(1)
                
                # Check if corresponding scenario file exists
                scenario_file = self.scenarios_dir / f'town{town_num}_all_scenarios.json'
                if scenario_file.exists():
                    if town_num not in town_routes:
                        town_routes[town_num] = []
                    town_routes[town_num].append(route_file.name)
                else:
                    print(f"Warning: No scenario file found for {route_file.name} "
                          f"(expected {scenario_file})")
        
        # Sort routes within each town
        for town in town_routes:
            town_routes[town].sort()
        
        return town_routes
    
    def _get_valid_combinations(self, agents_list: List[str] = None, 
                               weather_list: List[int] = None,
                               routes_list: List[str] = None) -> List[Dict]:
        """
        Generate only valid combinations of agent/weather/route.
        Ensures routes are matched with their correct town scenarios.
        """
        # Discover available routes grouped by town
        town_routes = self._discover_routes_and_scenarios()
        
        if not town_routes:
            print("ERROR: No valid route/scenario combinations found!")
            print(f"  Checked routes dir: {self.routes_dir}")
            print(f"  Checked scenarios dir: {self.scenarios_dir}")
            return []
        
        # Default agents
        if agents_list is None:
            configs_dir = self.project_root / 'leaderboard/team_code/configs'
            agents_list = []
            if configs_dir.exists():
                agents_list = [f.stem for f in configs_dir.glob('*.yaml')]
            if not agents_list:
                print("Warning: No agent configs found, using default")
                agents_list = ['interfuser']
        
        # Default weather
        if weather_list is None:
            weather_list = list(range(15))  # 0-14
        
        # Filter routes if specific ones requested
        if routes_list is not None:
            # Filter town_routes to only include requested routes
            filtered_town_routes = {}
            for town, routes in town_routes.items():
                filtered = [r for r in routes if r in routes_list]
                if filtered:
                    filtered_town_routes[town] = filtered
            town_routes = filtered_town_routes
        
        # Generate combinations
        combinations = []
        
        print(f"\nDiscovered routes by town:")
        for town, routes in sorted(town_routes.items()):
            print(f"  Town {town}: {len(routes)} routes")
            for route in routes:
                print(f"    - {route}")
        
        print(f"\nGenerating combinations:")
        print(f"  Agents: {agents_list}")
        print(f"  Weather conditions: {len(weather_list)} (indices: {min(weather_list)}-{max(weather_list)})")
        print(f"  Towns with routes: {sorted(town_routes.keys())}")
        
        for agent in agents_list:
            for weather_idx in weather_list:
                for town, routes in sorted(town_routes.items()):
                    for route in routes:
                        combinations.append({
                            'agent': agent,
                            'weather': weather_idx,
                            'route': route,
                            'town': town,  # Store town for reference
                        })
        
        print(f"\nTotal valid combinations: {len(combinations)}")
        return combinations
    
    def reset_queue(self, agents: List[str] = None, weather: List[int] = None, 
                   routes: List[str] = None):
        """Reset the job queue with specified combinations"""
        def _reset(agents_list, weather_list, routes_list):
            # Get valid combinations
            combinations = self._get_valid_combinations(agents_list, weather_list, routes_list)
            
            if not combinations:
                print("ERROR: No valid combinations could be generated!")
                return 0
            
            # Generate jobs from combinations
            jobs = []
            for job_id, combo in enumerate(combinations):
                jobs.append({
                    'id': job_id,
                    'agent': combo['agent'],
                    'weather': combo['weather'],
                    'route': combo['route'],
                    'town': combo['town'],  # Store town for debugging
                    'status': 'pending',
                    'attempts': 0,
                    'gpu': None,
                    'start_time': None,
                    'end_time': None,
                    'duration': None
                })
            
            # Save new queue
            queue_data = {'jobs': jobs, 'total': len(jobs), 'completed': 0}
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
            
            # Initialize/upgrade GPU status without clobbering other nodes
            gpu_status = self._load_gpu_status()
            gpu_status = self._ensure_node_gpu_status(gpu_status)
            with open(self.gpu_status_file, 'w') as f:
                json.dump(gpu_status, f, indent=2)
            
            # Reset completed jobs
            with open(self.completed_file, 'w') as f:
                json.dump({'jobs': []}, f, indent=2)
            
            # Initialize runtime estimates with better defaults
            runtime_estimates = {
                'default': 3600,
                'combinations': {}
            }
            
            # Create initial estimates based on route length
            for job in jobs:
                key = f"{job['agent']}_{job['route']}"
                if key not in runtime_estimates['combinations']:
                    # Shorter runtime for 'short' routes
                    if 'short' in job['route']:
                        runtime_estimates['combinations'][key] = 1800  # 30 min
                    elif 'long' in job['route']:
                        runtime_estimates['combinations'][key] = 5400  # 90 min
                    else:
                        runtime_estimates['combinations'][key] = 3600  # 60 min default
            
            with open(self.runtime_file, 'w') as f:
                json.dump(runtime_estimates, f, indent=2)
            
            print(f"\nQueue reset with {len(jobs)} valid jobs")
            
            # Print summary by town
            jobs_by_town = {}
            for job in jobs:
                town = job.get('town', 'unknown')
                if town not in jobs_by_town:
                    jobs_by_town[town] = 0
                jobs_by_town[town] += 1
            
            print("\nJobs per town:")
            for town in sorted(jobs_by_town.keys()):
                print(f"  Town {town}: {jobs_by_town[town]} jobs")
            
            return len(jobs)
        
        return self._with_lock(lambda: _reset(agents, weather, routes))
    
    def retry_failed(self, max_attempts: int = 3):
        """Reset failed jobs for retry"""
        def _retry():
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            
            retry_count = 0
            for job in queue_data['jobs']:
                if job['status'] == 'failed' and job['attempts'] < max_attempts:
                    job['status'] = 'pending'
                    retry_count += 1
            
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
            
            print(f"Reset {retry_count} failed jobs for retry")
            return retry_count
        
        return self._with_lock(_retry)
    
    def add_jobs(self, agent: str, weather: List[int] = None, routes: List[str] = None):
        """Add new jobs to the queue"""
        def _add(agent_name, weather_list, routes_list):
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            
            # Get next job ID
            max_id = max((j['id'] for j in queue_data['jobs']), default=-1)
            job_id = max_id + 1
            
            # Get valid combinations for the new jobs
            combinations = self._get_valid_combinations([agent_name], weather_list, routes_list)
            
            if not combinations:
                print("ERROR: No valid combinations could be generated!")
                return 0
            
            # Add new jobs
            new_jobs = []
            for combo in combinations:
                new_jobs.append({
                    'id': job_id,
                    'agent': combo['agent'],
                    'weather': combo['weather'],
                    'route': combo['route'],
                    'town': combo.get('town', 'unknown'),
                    'status': 'pending',
                    'attempts': 0,
                    'gpu': None,
                    'start_time': None,
                    'end_time': None,
                    'duration': None
                })
                job_id += 1
            
            queue_data['jobs'].extend(new_jobs)
            queue_data['total'] = len(queue_data['jobs'])
            
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
            
            # Update runtime estimates for new combinations
            with open(self.runtime_file, 'r') as f:
                runtime_data = json.load(f)
            
            for job in new_jobs:
                key = f"{job['agent']}_{job['route']}"
                if key not in runtime_data['combinations']:
                    if 'short' in job['route']:
                        runtime_data['combinations'][key] = 1800
                    elif 'long' in job['route']:
                        runtime_data['combinations'][key] = 5400
                    else:
                        runtime_data['combinations'][key] = 3600
            
            with open(self.runtime_file, 'w') as f:
                json.dump(runtime_data, f, indent=2)
            
            print(f"Added {len(new_jobs)} new valid jobs for agent '{agent_name}'")
            return len(new_jobs)
        
        return self._with_lock(lambda: _add(agent, weather, routes))
    
    def cancel_pending(self, agent: str = None):
        """Cancel pending jobs"""
        def _cancel(agent_name):
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            
            cancel_count = 0
            for job in queue_data['jobs']:
                if job['status'] == 'pending':
                    if agent_name is None or job['agent'] == agent_name:
                        job['status'] = 'cancelled'
                        cancel_count += 1
            
            with open(self.queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
            
            if agent_name:
                print(f"Cancelled {cancel_count} pending jobs for agent '{agent_name}'")
            else:
                print(f"Cancelled all {cancel_count} pending jobs")
            return cancel_count
        
        return self._with_lock(lambda: _cancel(agent))
    
    def get_status(self):
        """Get current collection status"""
        try:
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            with open(self.gpu_status_file, 'r') as f:
                gpu_status = json.load(f)
            with open(self.runtime_file, 'r') as f:
                runtime_data = json.load(f)
            
            # Calculate statistics
            # Support flat (legacy) or namespaced (v2) gpu_status
            def _iter_gpu_states(d):
                if isinstance(d, dict) and "nodes" in d and isinstance(d["nodes"], dict):
                    for _node, gmap in d["nodes"].items():
                        for _gid, g in gmap.items():
                            yield g
                elif isinstance(d, dict):
                    for _gid, g in d.items():
                        if isinstance(g, dict):
                            yield g

            status = {
                'total': queue_data['total'],
                'completed': queue_data['completed'],
                'pending': sum(1 for j in queue_data['jobs'] if j['status'] == 'pending'),
                'running': sum(1 for j in queue_data['jobs'] if j['status'] in ['assigned', 'running']),
                'failed': sum(1 for j in queue_data['jobs'] if j['status'] == 'failed'),
                'cancelled': sum(1 for j in queue_data['jobs'] if j['status'] == 'cancelled'),
                'gpus_active': sum(1 for g in _iter_gpu_states(gpu_status) if g.get('status') == 'busy'),
                'gpus_idle': sum(1 for g in _iter_gpu_states(gpu_status) if g.get('status') == 'idle')

            }
            
            return status
        except FileNotFoundError:
            return None
    
    def export_results(self, output_file: str = 'collection_results.json'):
        """Export completed job results"""
        try:
            with open(self.completed_file, 'r') as f:
                completed_data = json.load(f)
            
            # Organize results
            results = {
                'summary': {
                    'total_completed': len(completed_data['jobs']),
                    'export_time': datetime.now().isoformat()
                },
                'jobs': completed_data['jobs'],
                'statistics': {}
            }
            
            # Calculate statistics
            if completed_data['jobs']:
                durations = [j['duration'] for j in completed_data['jobs'] if j.get('duration')]
                if durations:
                    results['statistics'] = {
                        'total_runtime': sum(durations),
                        'average_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations)
                    }
                
                # Group by agent
                by_agent = {}
                for job in completed_data['jobs']:
                    agent = job['agent']
                    if agent not in by_agent:
                        by_agent[agent] = []
                    by_agent[agent].append(job['duration'] if job.get('duration') else 0)
                
                results['by_agent'] = {
                    agent: {
                        'count': len(times),
                        'total_time': sum(times),
                        'avg_time': sum(times) / len(times) if times else 0
                    }
                    for agent, times in by_agent.items()
                }
                
                # Group by town to analyze failure patterns
                by_town = {}
                for job in completed_data['jobs']:
                    town = job.get('town', 'unknown')
                    if town not in by_town:
                        by_town[town] = {'completed': 0, 'total_time': 0}
                    by_town[town]['completed'] += 1
                    by_town[town]['total_time'] += job.get('duration', 0)
                
                results['by_town'] = by_town
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Exported results to {output_file}")
            return output_file
        
        except FileNotFoundError:
            print("No completed jobs found")
            return None
    
    def optimize_runtime_estimates(self):
        """Optimize runtime estimates based on completed jobs"""
        def _optimize():
            with open(self.completed_file, 'r') as f:
                completed_data = json.load(f)
            with open(self.runtime_file, 'r') as f:
                runtime_data = json.load(f)
            
            if not completed_data['jobs']:
                print("No completed jobs to analyze")
                return 0
            
            # Group completed jobs by agent-route combination
            combinations = {}
            for job in completed_data['jobs']:
                if job.get('duration'):
                    key = f"{job['agent']}_{job['route']}"
                    if key not in combinations:
                        combinations[key] = []
                    combinations[key].append(job['duration'])
            
            # Update estimates using median of actual runtimes
            updates = 0
            for key, durations in combinations.items():
                if len(durations) >= 2:  # Need at least 2 samples
                    durations.sort()
                    median = durations[len(durations)//2]
                    old_estimate = runtime_data['combinations'].get(key, runtime_data['default'])
                    
                    # Weighted update
                    new_estimate = 0.7 * median + 0.3 * old_estimate
                    runtime_data['combinations'][key] = new_estimate
                    updates += 1
                    
                    print(f"Updated {key}: {old_estimate:.0f}s -> {new_estimate:.0f}s")
            
            with open(self.runtime_file, 'w') as f:
                json.dump(runtime_data, f, indent=2)
            
            print(f"Optimized {updates} runtime estimates")
            return updates
        
        return self._with_lock(_optimize)
    
    def show_runtime_analysis(self):
        """Show detailed runtime analysis"""
        try:
            with open(self.completed_file, 'r') as f:
                completed_data = json.load(f)
            with open(self.runtime_file, 'r') as f:
                runtime_data = json.load(f)
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            
            if not completed_data['jobs']:
                print("No completed jobs to analyze")
                return
            
            print("\nRUNTIME ANALYSIS")
            print("="*60)
            
            # Analyze failure patterns
            failed_jobs = [j for j in queue_data['jobs'] if j['status'] == 'failed']
            if failed_jobs:
                print("\nFAILURE ANALYSIS:")
                print(f"Total failed jobs: {len(failed_jobs)}")
                
                # Group failures by town
                failures_by_town = {}
                for job in failed_jobs:
                    town = job.get('town', 'unknown')
                    if town not in failures_by_town:
                        failures_by_town[town] = []
                    failures_by_town[town].append(job)
                
                print("\nFailures by town:")
                for town, jobs in sorted(failures_by_town.items()):
                    print(f"  Town {town}: {len(jobs)} failures")
                    # Show unique routes that failed
                    failed_routes = set(j['route'] for j in jobs)
                    for route in sorted(failed_routes):
                        count = sum(1 for j in jobs if j['route'] == route)
                        print(f"    - {route}: {count} failures")
            
            # Group by agent
            by_agent = {}
            for job in completed_data['jobs']:
                if job.get('duration'):
                    agent = job['agent']
                    if agent not in by_agent:
                        by_agent[agent] = {'durations': [], 'routes': {}}
                    by_agent[agent]['durations'].append(job['duration'])
                    
                    route = job['route']
                    if route not in by_agent[agent]['routes']:
                        by_agent[agent]['routes'][route] = []
                    by_agent[agent]['routes'][route].append(job['duration'])
            
            # Print analysis for each agent
            for agent, data in sorted(by_agent.items()):
                durations = data['durations']
                avg_time = sum(durations) / len(durations)
                min_time = min(durations)
                max_time = max(durations)
                
                print(f"\n{agent}:")
                print(f"  Jobs completed: {len(durations)}")
                print(f"  Average time: {str(timedelta(seconds=int(avg_time)))}")
                print(f"  Min/Max: {str(timedelta(seconds=int(min_time)))} - "
                      f"{str(timedelta(seconds=int(max_time)))}")
                
                # Route breakdown
                print("  By route:")
                for route, route_times in sorted(data['routes'].items()):
                    route_avg = sum(route_times) / len(route_times)
                    estimate_key = f"{agent}_{route}"
                    estimate = runtime_data['combinations'].get(estimate_key, 
                                                               runtime_data['default'])
                    accuracy = (1 - abs(route_avg - estimate) / route_avg) * 100
                    
                    print(f"    {route[:30]:30} Avg: {route_avg:6.0f}s "
                          f"Est: {estimate:6.0f}s ({accuracy:.0f}% accurate)")
            
            print("="*60)
            
        except FileNotFoundError:
            print("State files not found")
    
    def check_setup(self):
        """Check if the environment is properly set up"""
        print("\nCHECKING SETUP")
        print("="*60)
        
        # Check routes directory
        if self.routes_dir.exists():
            route_files = list(self.routes_dir.glob('*.xml'))
            print(f"✓ Routes directory exists: {self.routes_dir}")
            print(f"  Found {len(route_files)} route files")
        else:
            print(f"✗ Routes directory not found: {self.routes_dir}")
        
        # Check scenarios directory
        if self.scenarios_dir.exists():
            scenario_files = list(self.scenarios_dir.glob('*.json'))
            print(f"✓ Scenarios directory exists: {self.scenarios_dir}")
            print(f"  Found {len(scenario_files)} scenario files")
        else:
            print(f"✗ Scenarios directory not found: {self.scenarios_dir}")
        
        # Check for matching route/scenario pairs
        town_routes = self._discover_routes_and_scenarios()
        if town_routes:
            print(f"\n✓ Found {sum(len(r) for r in town_routes.values())} valid route/scenario combinations")
            for town, routes in sorted(town_routes.items()):
                print(f"  Town {town}: {len(routes)} routes")
        else:
            print("\n✗ No valid route/scenario combinations found!")
        
        # Check agent configs
        configs_dir = self.project_root / 'leaderboard/team_code/configs'
        if configs_dir.exists():
            agent_configs = list(configs_dir.glob('*.yaml'))
            print(f"\n✓ Agent configs directory exists: {configs_dir}")
            print(f"  Found {len(agent_configs)} agent configurations:")
            for config in agent_configs:
                print(f"    - {config.stem}")
        else:
            print(f"\n✗ Agent configs directory not found: {configs_dir}")
        
        print("="*60)

    def run_next_job(self, host: str = "127.0.0.1", port: int = 2000, tm_port: int = 5000, extra_args: list = None) -> int:
        """
        Pop the next pending job from the queue and execute it via the CARLA leaderboard evaluator.
        Returns process return code (0 on success), or 2 if no pending jobs.
        """
        extra_args = extra_args or []

        def _reserve_next():
            with open(self.queue_file, 'r') as f:
                q = json.load(f)
            for job in q['jobs']:
                if job.get('status') == 'pending':
                    job['status'] = 'running'
                    job['attempts'] = int(job.get('attempts', 0)) + 1
                    job['gpu'] = int(os.environ.get('GPU_ID', -1)) if os.environ.get('GPU_ID') else None
                    job['start_time'] = datetime.utcnow().isoformat() + 'Z'
                    with open(self.queue_file, 'w') as f:
                        json.dump(q, f, indent=2)
                    return job
            return None

        job = self._with_lock(_reserve_next)
        if not job:
            print("No pending jobs remain. Nothing to run.")
            return 2

        # Resolve paths
        agent_name = job['agent']
        route_name = job['route']
        town = job.get('town', 'unknown')
        weather_idx = job.get('weather', 0)

        agent_cfg = self.project_root / 'leaderboard' / 'team_code' / 'configs' / f'{agent_name}.yaml'
        agent_code = self.project_root / 'leaderboard' / 'team_code' / 'consolidated_agent.py'
        routes_file = self.routes_dir / route_name

        m = re.search(r'(?:town)?(\d+)', str(town)) or re.search(r'routes_town(\d+)_', route_name)
        town_num = m.group(1) if m else None
        if town_num:
            scenarios_file = self.scenarios_dir / f'town{town_num}_all_scenarios.json'
        else:
            candidates = sorted(self.scenarios_dir.glob('town*_all_scenarios.json'))
            scenarios_file = candidates[0] if candidates else None

        missing = []
        if not agent_cfg.exists(): missing.append(str(agent_cfg))
        if not agent_code.exists(): missing.append(str(agent_code))
        if not routes_file.exists(): missing.append(str(routes_file))
        if not (scenarios_file and scenarios_file.exists()): missing.append(str(scenarios_file) if scenarios_file else "scenarios_file")

        if missing:
            print("FATAL: Cannot run job due to missing files:")
            for mpath in missing:
                print(" -", mpath)
            def _fail():
                with open(self.queue_file, 'r') as f:
                    q = json.load(f)
                for j in q['jobs']:
                    if j['id'] == job['id']:
                        j['status'] = 'failed'
                        j['end_time'] = datetime.utcnow().isoformat() + 'Z'
                        j['duration'] = 0
                        break
                with open(self.queue_file, 'w') as f:
                    json.dump(q, f, indent=2)
            self._with_lock(_fail)
            return 1

        # Build command (env overrides supported)
        eval_entry = os.environ.get('EVAL_ENTRYPOINT', '')
        eval_cmd_template = os.environ.get('EVAL_CMD_TEMPLATE', '').strip()

        host = str(host); port = str(port); tm_port = str(tm_port)
        env = os.environ.copy()
        env.update({
            'AGENT_CFG': str(agent_cfg),
            'ROUTES_FILE': str(routes_file),
            'SCENARIOS_FILE': str(scenarios_file),
            'WEATHER_INDEX': str(weather_idx),
            'CARLA_HOST': host,
            'CARLA_PORT': port,
            'TM_PORT': tm_port,
        })

        from pathlib import Path
        route_stem = Path(routes_file).stem  # e.g., "routes_town04_tiny"
        env.update({
            'AGENT_NAME': str(agent_name),
            'ROUTE_NAME': str(route_stem),
            'TOWN_NUM': str(town_num) if town_num is not None else '',
            # make sure DATASET_DIR is visible inside the process even if MC is run standalone
            'DATASET_DIR': env.get('DATASET_DIR', str(self.project_root / 'dataset')),
        })

        if eval_cmd_template:
            fmt = eval_cmd_template.format(
                AGENT_CFG=str(agent_cfg),
                AGENT_CODE=str(agent_code),
                ROUTES_FILE=str(routes_file),
                SCENARIOS_FILE=str(scenarios_file),
                HOST=host, PORT=port, TM_PORT=tm_port, WEATHER=weather_idx
            )
            cmd = ['bash','-lc', fmt]
        else:
            sif = os.environ.get('CARLA_SIF', 'carla_official.sif')
            cmd = [
                'singularity','exec','--nv', sif, 'python3','-m','leaderboard.leaderboard_evaluator',
                '--routes', str(routes_file),
                '--scenarios', str(scenarios_file),
                '--agent', str(agent_code),
                '--agent-config', str(agent_cfg),
                '--host', host, '--port', port, '--trafficManagerPort', tm_port
            ]
            if extra_args:
                cmd.extend(extra_args)

        print(f"[RUN] Job {job['id']} agent={agent_name} route={route_name} weather={weather_idx}")
        start_ts = time.time()
        try:
            rc = subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            rc = 130
        end_ts = time.time()

        def _finish():
            duration = int(end_ts - start_ts)
            with open(self.queue_file, 'r') as f:
                q = json.load(f)
            for j in q['jobs']:
                if j['id'] == job['id']:
                    j['status'] = 'completed' if rc == 0 else 'failed'
                    j['end_time'] = datetime.utcnow().isoformat() + 'Z'
                    j['duration'] = duration
                    break
            with open(self.queue_file, 'w') as f:
                json.dump(q, f, indent=2)

            if self.completed_file.exists():
                with open(self.completed_file, 'r') as f:
                    comp = json.load(f)
            else:
                comp = {'jobs': []}
            entry = dict(job)
            entry['status'] = 'completed' if rc == 0 else 'failed'
            entry['end_time'] = datetime.utcnow().isoformat() + 'Z'
            entry['duration'] = int(end_ts - start_ts)
            comp['jobs'].append(entry)
            with open(self.completed_file, 'w') as f:
                json.dump(comp, f, indent=2)

        self._with_lock(_finish)
        return rc



def main():
    parser = argparse.ArgumentParser(description='Manage continuous data collection')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Check command
    subparsers.add_parser('check', help='Check setup and available combinations')
    
    # Status command
    subparsers.add_parser('status', help='Show current status')

    # Run command (execute next pending job)
    run_parser = subparsers.add_parser('run', help='Run the next pending job (one evaluation)')
    run_parser.add_argument('--host', default=os.environ.get('CARLA_HOST', '127.0.0.1'))
    run_parser.add_argument('--port', type=int, default=int(os.environ.get('CARLA_PORT', '2000')))
    run_parser.add_argument('--trafficManagerPort', type=int, default=int(os.environ.get('TM_PORT', os.environ.get('TRAFFIC_MANAGER_PORT', '5000'))))
    run_parser.add_argument('extra', nargs=argparse.REMAINDER, help='Additional args passed to evaluator')

    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset job queue')
    reset_parser.add_argument('--agents', nargs='+', help='Agents to include')
    reset_parser.add_argument('--weather', nargs='+', type=int, help='Weather indices')
    reset_parser.add_argument('--routes', nargs='+', help='Route files')
    
    # Retry command
    retry_parser = subparsers.add_parser('retry', help='Retry failed jobs')
    retry_parser.add_argument('--max-attempts', type=int, default=3, help='Maximum attempts')
    
    # Add jobs command
    add_parser = subparsers.add_parser('add', help='Add new jobs')
    add_parser.add_argument('agent', help='Agent name')
    add_parser.add_argument('--weather', nargs='+', type=int, help='Weather indices')
    add_parser.add_argument('--routes', nargs='+', help='Route files')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel pending jobs')
    cancel_parser.add_argument('--agent', help='Cancel only for specific agent')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results')
    export_parser.add_argument('--output', default='collection_results.json', 
                              help='Output file')
    
    # Optimize command
    subparsers.add_parser('optimize', help='Optimize runtime estimates')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Show runtime analysis')
    
    # Parse arguments
    default_state_dir = os.path.join(
        os.environ.get('PROJECT_ROOT', os.getcwd()),
        'collection_state'
    )
    parser.add_argument('--state-dir', default=default_state_dir,
                       help='Path to state directory')
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    manager = ContinuousManager(args.state_dir)
    
    if args.command == 'check':
        manager.check_setup()
    
    elif args.command == 'status':
        status = manager.get_status()
        if status:
            print("\nCOLLECTION STATUS")
            print("="*40)
            for key, value in status.items():
                print(f"{key:15}: {value}")
            print("="*40)
        else:
            print("No active collection found")
    
    elif args.command == 'reset':
        manager.reset_queue(args.agents, args.weather, args.routes)
    
    elif args.command == 'retry':
        manager.retry_failed(args.max_attempts)
    
    elif args.command == 'add':
        manager.add_jobs(args.agent, args.weather, args.routes)
    
    elif args.command == 'cancel':
        manager.cancel_pending(args.agent)
    
    elif args.command == 'export':
        manager.export_results(args.output)
    
    elif args.command == 'optimize':
        manager.optimize_runtime_estimates()
    
    elif args.command == 'analyze':
        manager.show_runtime_analysis()

    elif args.command == 'run':
        extra = args.extra if hasattr(args, 'extra') else []
        if extra and extra[0] == '--':  # argparse quirk when using REMAINDER
            extra = extra[1:]
        rc = manager.run_next_job(host=args.host, port=args.port,
                                tm_port=args.trafficManagerPort, extra_args=extra)
        sys.exit(rc)



if __name__ == '__main__':
    main()