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
import signal
import socket
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

    def _init_gpu_status_skeleton(self) -> Dict[str, Any]:
        """
        Return an EMPTY gpu status map.
        We purposely do NOT pre-populate the login node with fake GPUs.
        Real rows appear via healthbeats from workers.
        """
        return {"schema": 2, "nodes": {}}

    def _with_lock(self, func):
        """Execute function with file lock"""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return func()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    
    def _discover_routes_and_scenarios(self, include_smoke: bool = False) -> Dict[str, List[str]]:
        """
        Discover available route files and their corresponding scenarios.
        Returns a mapping of town -> list of route files for that town.

        `*_smoke.xml` files (generated for `reset --smoke`) are excluded from
        normal discovery so they never pollute a full sweep; pass
        include_smoke=True to surface them.
        """
        town_routes = {}

        if not self.routes_dir.exists():
            print(f"Warning: Routes directory not found at {self.routes_dir}")
            return {}

        # Find all route XML files
        route_files = list(self.routes_dir.glob('*.xml'))

        for route_file in route_files:
            if route_file.name.endswith('_smoke.xml') and not include_smoke:
                continue
            # Extract town number from filename (e.g., routes_town01_short.xml -> 01)
            match = re.search(r'routes_town(\d+)_', route_file.name, flags=re.IGNORECASE)
            if match:
                town_num = match.group(1)
                
                # Check if corresponding scenario file exists
                scenario_file = self.scenarios_dir / f'town{town_num}_all_scenarios.json'
                if scenario_file.exists():
                    town_routes.setdefault(town_num, []).append(route_file.name)
                else:
                    print(f"Warning: No scenario file found for {route_file.name} "
                          f"(expected {scenario_file})")
        
        # Sort routes within each town
        for town in town_routes:
            town_routes[town].sort()
        
        return town_routes
    
    def _get_valid_combinations(self, agents_list: List[str] = None,
                               weather_list: List[int] = None,
                               routes_list: List[str] = None,
                               include_smoke: bool = False) -> List[Dict]:
        """
        Generate only valid combinations of agent/weather/route.
        Ensures routes are matched with their correct town scenarios.
        """
        # Discover available routes grouped by town
        town_routes = self._discover_routes_and_scenarios(include_smoke=include_smoke)
        
        if not town_routes:
            print("ERROR: No valid route/scenario combinations found!")
            print(f"  Checked routes dir: {self.routes_dir}")
            print(f"  Checked scenarios dir: {self.scenarios_dir}")
            return []
        
        # Default agents
        if agents_list is None:
            configs_dir = self.project_root / 'leaderboard/team_code/configs'
            agents_list = [f.stem for f in configs_dir.glob('*.yaml')] if configs_dir.exists() else []
            if not agents_list:
                print("Warning: No agent configs found, using default")
                agents_list = ['interfuser']
        
        # Default weather: 0-20 covers all CARLA presets including night conditions.
        # (0=ClearNoon … 13=SoftRainSunset; 14=ClearNight … 20=HardRainNight)
        if weather_list is None:
            weather_list = list(range(21))  # 0-20
        
        # Filter routes if specific ones requested
        if routes_list is not None:
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
                            'town': town,
                            'status': 'pending',
                            'attempts': 0,
                        })
        
        print(f"\nTotal valid combinations: {len(combinations)}")
        return combinations

    def _ensure_smoke_routes(self, towns=('01', '03')) -> List[str]:
        """Generate minimal single-route files for a fast smoke test.

        For each town, extracts the FIRST <route> from routes_town{NN}_tiny.xml
        into routes_town{NN}_smoke.xml (idempotent). Returns the smoke filenames
        that were successfully created. One short route per town is enough to
        validate the pipeline end-to-end without running the hundreds of routes
        a full *_tiny.xml file contains.
        """
        import xml.etree.ElementTree as ET
        created = []
        for nn in towns:
            src = self.routes_dir / f'routes_town{nn}_tiny.xml'
            if not src.exists():
                print(f"[smoke] skip town{nn}: {src.name} not found")
                continue
            try:
                root = ET.parse(str(src)).getroot()
                first = root.find('route')
                if first is None:
                    print(f"[smoke] skip town{nn}: no <route> in {src.name}")
                    continue
                new_root = ET.Element(root.tag, root.attrib)
                new_root.append(first)
                out = self.routes_dir / f'routes_town{nn}_smoke.xml'
                ET.ElementTree(new_root).write(str(out), encoding='utf-8', xml_declaration=True)
                created.append(out.name)
            except Exception as e:
                print(f"[smoke] failed to build town{nn} smoke route: {e}")
        return created

    def _derive_gpu_id(self, fallback_port: int) -> int:
        env_gpu = os.environ.get('GPU_ID')
        if env_gpu is not None and str(env_gpu).isdigit():
            return int(env_gpu)
        base = int(os.environ.get('BASE_RPC_PORT', '2000'))
        spacing = int(os.environ.get('PORT_SPACING', '100'))
        try:
            return max(0, (int(fallback_port) - base) // spacing)
        except Exception:
            return 0

    def _health_path(self, gpu_id: int) -> Path:
        return self.state_dir / 'health' / f"{self.node_name}_gpu{gpu_id}.json"

    def _write_health(self, gpu_id: int, status: str, message: str = "",
                      rpc_port: int = None, tm_port: int = None, current_job: int = None):
        p = self._health_path(gpu_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _safe_read_json(p) if p.exists() else {}
        data.update({
            "node": self.node_name,
            "gpu_id": int(gpu_id),
            "status": status,
            "message": message,
            "rpc_port": rpc_port,
            "tm_port": tm_port,
            "current_job": current_job,
            "last_heartbeat": datetime.utcnow().isoformat() + 'Z'
        })
        with open(p, 'w') as f:
            json.dump(data, f, indent=2)

    def _bump_gpu_stats(self, gpu_id: int, duration_s: int, completed_ok: bool):
        st = self._load_gpu_status()
        nodes = st.setdefault("nodes", {})
        nd = nodes.setdefault(self.node_name, {})
        gi = nd.setdefault(str(int(gpu_id)), {"jobs_completed": 0, "total_runtime": 0})
        if completed_ok:
            gi["jobs_completed"] = int(gi.get("jobs_completed", 0)) + 1
        gi["total_runtime"] = int(gi.get("total_runtime", 0)) + max(0, int(duration_s))
        with open(self.gpu_status_file, 'w') as f:
            json.dump(st, f, indent=2)

        
    def reset_queue(self, agents: List[str] = None, weather: List[int] = None,
                   routes: List[str] = None, smoke: bool = False, limit: int = None):
        """Reset the job queue with specified combinations.

        smoke=True builds a tiny validation queue: tiny routes only + weather 0
        (unless routes/weather are given explicitly), so the pipeline can be
        validated end-to-end in ~1h instead of the full multi-day sweep.
        limit=N caps the queue at N jobs, interleaved across agents so a small
        cap still exercises every agent.
        """
        def _reset(agents_list, weather_list, routes_list):
            # Smoke preset: smallest representative slice that still touches
            # every agent. Only fills in defaults the caller didn't specify.
            # NB: *_tiny.xml files contain hundreds of short routes each, so we
            # generate single-route *_smoke.xml files instead of using them.
            if smoke:
                if routes_list is None:
                    routes_list = self._ensure_smoke_routes()
                    if not routes_list:
                        print("ERROR: could not generate any smoke routes")
                        return 0
                    print(f"[smoke] using single-route files: {routes_list}")
                if weather_list is None:
                    weather_list = [0]
                    print("[smoke] using weather [0]")

            # Get valid combinations
            combinations = self._get_valid_combinations(agents_list, weather_list, routes_list,
                                                        include_smoke=smoke)
            if not combinations:
                print("ERROR: No valid combinations could be generated!")
                return 0

            # Cap total jobs, interleaving by agent so a small limit stays balanced.
            if limit and limit > 0 and len(combinations) > limit:
                from collections import defaultdict, deque
                by_agent = defaultdict(deque)
                for c in combinations:
                    by_agent[c['agent']].append(c)
                queues = list(by_agent.values())
                interleaved = []
                while queues and len(interleaved) < limit:
                    for q in [q for q in queues if q]:
                        interleaved.append(q.popleft())
                        if len(interleaved) >= limit:
                            break
                    queues = [q for q in queues if q]
                print(f"[limit] capping {len(combinations)} -> {len(interleaved)} jobs")
                combinations = interleaved
            
            # Generate jobs
            jobs = []
            for job_id, combo in enumerate(combinations):
                jobs.append({
                    'id': job_id,
                    'agent': combo['agent'],
                    'weather': combo['weather'],
                    'route': combo['route'],
                    'town': combo['town'],
                    'status': 'pending',
                    'attempts': 0,
                    'gpu': None,
                    'start_time': None,
                    'end_time': None,
                    'duration': None
                })
            
            with open(self.queue_file, 'w') as f:
                json.dump({'jobs': jobs, 'total': len(jobs), 'completed': 0}, f, indent=2)
            
            # IMPORTANT: write an EMPTY gpu status skeleton (no fake login-node rows)
            with open(self.gpu_status_file, 'w') as f:
                json.dump(self._init_gpu_status_skeleton(), f, indent=2)
            
            # Reset completed jobs
            with open(self.completed_file, 'w') as f:
                json.dump({'jobs': []}, f, indent=2)
            
            # Initial runtime estimates (heuristic by route length)
            runtime_estimates = {'default': 3600, 'combinations': {}}
            for job in jobs:
                key = f"{job['agent']}_{job['route']}"
                if key not in runtime_estimates['combinations']:
                    if 'short' in job['route']:
                        runtime_estimates['combinations'][key] = 1800
                    elif 'long' in job['route']:
                        runtime_estimates['combinations'][key] = 5400
                    else:
                        runtime_estimates['combinations'][key] = 3600
            with open(self.runtime_file, 'w') as f:
                json.dump(runtime_estimates, f, indent=2)
            
            print(f"\nQueue reset with {len(jobs)} valid jobs")
            jobs_by_town = {}
            for job in jobs:
                jobs_by_town[job['town']] = jobs_by_town.get(job['town'], 0) + 1
            print("\nJobs per town:")
            for town in sorted(jobs_by_town.keys()):
                print(f"  Town {town}: {jobs_by_town[town]} jobs")
            return len(jobs)
        
        return self._with_lock(lambda: _reset(agents, weather, routes))
    
    def reclaim_stale(self, stale_hours: float = 6.0):
        """Recover jobs stuck in 'running' state whose workers are no longer alive.

        A job is considered stale if it has been running longer than `stale_hours`
        with no active health beacon.  For each stale job:
          - If a run_summary.json exists at the expected save_path, the worker
            finished successfully but _finish() never ran; mark completed and
            write a best-effort entry to completed_jobs.json.
          - Otherwise, reset to pending so the job gets retried.
        """
        import shutil as _shutil

        def _label_weather(idx):
            try: return f"weather_{int(idx)}"
            except: return f"weather_{idx}"

        def _label_map(tn):
            try: return f"map_{int(tn):02d}"
            except: return f"map_{tn or 'unknown'}"

        def _reclaim():
            with open(self.queue_file, 'r') as f:
                q = json.load(f)

            dataset_dir = os.environ.get('DATASET_DIR',
                str(self.project_root / 'dataset'))
            now = datetime.utcnow()
            reclaimed_complete = 0
            reclaimed_pending  = 0

            for job in q['jobs']:
                if job.get('status') != 'running':
                    continue
                start = job.get('start_time')
                if start:
                    try:
                        elapsed = (now - datetime.fromisoformat(start.rstrip('Z'))).total_seconds() / 3600.0
                        if elapsed < stale_hours:
                            continue  # still young enough to be legitimately running
                    except Exception:
                        pass  # unparseable timestamp — treat as stale

                route_stem  = Path(job['route']).stem
                weather_lbl = _label_weather(job.get('weather', 0))
                map_lbl     = _label_map(job.get('town', ''))
                save_path   = os.path.join(dataset_dir, job['agent'], weather_lbl, map_lbl, route_stem)
                summary_path = Path(save_path) / 'run_summary.json'

                if summary_path.exists():
                    # Worker finished — mark completed and record in completed_jobs.json
                    job['status']   = 'completed'
                    job['end_time'] = job.get('end_time') or now.isoformat() + 'Z'
                    try:
                        with open(summary_path) as sf:
                            sm = json.load(sf)
                        job['global_steps'] = sm.get('global_steps')
                    except Exception:
                        pass
                    # Try to read leaderboard scores if results.json was written
                    results_path = Path(save_path) / 'results.json'
                    try:
                        with open(results_path) as rf:
                            lb = json.load(rf)
                        records = lb.get('_checkpoint', {}).get('records', [])
                        if records:
                            composed  = [r['scores']['score_composed'] for r in records if 'scores' in r]
                            route_pct = [r['scores']['score_route']    for r in records if 'scores' in r]
                            job['score_composed'] = round(sum(composed) / len(composed), 4) if composed else None
                            job['score_route']    = round(sum(route_pct) / len(route_pct), 4) if route_pct else None
                            job['score_n_routes'] = len(records)
                    except Exception:
                        pass
                    reclaimed_complete += 1
                else:
                    job['status']   = 'pending'
                    job['end_time'] = None
                    reclaimed_pending += 1

            with open(self.queue_file, 'w') as f:
                json.dump(q, f, indent=2)

            # Append newly-completed entries to completed_jobs.json
            if reclaimed_complete:
                if self.completed_file.exists():
                    with open(self.completed_file) as f:
                        comp = json.load(f)
                else:
                    comp = {'jobs': []}
                existing_ids = {j['id'] for j in comp['jobs']}
                for job in q['jobs']:
                    if job.get('status') == 'completed' and job['id'] not in existing_ids:
                        comp['jobs'].append(dict(job))
                with open(self.completed_file, 'w') as f:
                    json.dump(comp, f, indent=2)

            print(f"Reclaimed {reclaimed_complete} completed + {reclaimed_pending} pending (was stale-running)")
            return reclaimed_complete, reclaimed_pending

        return self._with_lock(_reclaim)

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
            max_id = max((j['id'] for j in queue_data['jobs']), default=-1)
            job_id = max_id + 1
            
            combinations = self._get_valid_combinations([agent_name], weather_list, routes_list)
            if not combinations:
                print("ERROR: No valid combinations could be generated!")
                return 0
            
            new_jobs = []
            for combo in combinations:
                new_jobs.append({
                    'id': job_id,
                    'agent': combo['agent'],
                    'weather': combo['weather'],
                    'route': combo['route'],
                    'town': combo['town'],
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
                if job['status'] == 'pending' and (agent_name is None or job['agent'] == agent_name):
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
    
    def _scan_health(self) -> list:
        """Return list of live healthbeats found under state_dir/health."""
        health_dir = self.state_dir / 'health'
        entries = []
        if health_dir.exists():
            for p in sorted(health_dir.glob('*.json')):
                try:
                    with open(p) as f:
                        d = json.load(f)
                    entries.append(d)
                except Exception:
                    pass
        return entries

    def get_status(self):
        """Get current collection status"""
        try:
            with open(self.queue_file, 'r') as f:
                queue_data = json.load(f)
            # Try to load gpu_status (may be empty by design)
            if self.gpu_status_file.exists():
                with open(self.gpu_status_file, 'r') as f:
                    gpu_status = json.load(f)
            else:
                gpu_status = {"schema": 2, "nodes": {}}
            with open(self.runtime_file, 'r') as f:
                runtime_data = json.load(f)
            
            # Healthbeats override counters for active/idle
            beats = self._scan_health()
            beats_now = 0
            beats_busy = 0
            beats_idle = 0
            now = time.time()
            for b in beats:
                beats_now += 1
                status = (b.get('status') or '').lower()
                if status == 'busy':
                    beats_busy += 1
                elif status == 'idle':
                    beats_idle += 1
                else:
                    # stale/unknown don't count as active or idle
                    pass

            status = {
                'total': queue_data['total'],
                'completed': queue_data['completed'],
                'pending': sum(1 for j in queue_data['jobs'] if j['status'] == 'pending'),
                'running': sum(1 for j in queue_data['jobs'] if j['status'] in ['assigned', 'running']),
                'failed': sum(1 for j in queue_data['jobs'] if j['status'] == 'failed'),
                'cancelled': sum(1 for j in queue_data['jobs'] if j['status'] == 'cancelled'),
                'gpus_active': beats_busy,
                'gpus_idle': beats_idle
            }
            return status
        except FileNotFoundError:
            return None
    
    def export_results(self, output_file: str = 'collection_results.json'):
        """Export completed job results"""
        try:
            with open(self.completed_file, 'r') as f:
                completed_data = json.load(f)
            
            results = {
                'summary': {
                    'total_completed': len(completed_data['jobs']),
                    'export_time': datetime.utcnow().isoformat() + 'Z'
                },
                'jobs': completed_data['jobs'],
                'statistics': {}
            }
            
            if completed_data['jobs']:
                durations = [j['duration'] for j in completed_data['jobs'] if j.get('duration')]
                if durations:
                    results['statistics'] = {
                        'total_runtime': sum(durations),
                        'average_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations)
                    }
                
                by_agent = {}
                for job in completed_data['jobs']:
                    agent = job['agent']
                    by_agent.setdefault(agent, []).append(job.get('duration', 0))
                
                results['by_agent'] = {
                    agent: {
                        'count': len(times),
                        'total_time': sum(times),
                        'avg_time': sum(times) / len(times) if times else 0
                    }
                    for agent, times in by_agent.items()
                }
                
                by_town = {}
                for job in completed_data['jobs']:
                    town = job.get('town', 'unknown')
                    by_town.setdefault(town, {'completed': 0, 'total_time': 0})
                    by_town[town]['completed'] += 1
                    by_town[town]['total_time'] += job.get('duration', 0)
                
                results['by_town'] = by_town
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Exported results to {output_file}")
            return output_file
        
        except FileNotFoundError:
            print("No completed jobs found")
            return None
    
    def prune_redundant(self, dry_run: bool = False) -> int:
        """Mark pending easy-weather jobs redundant when a harder variant already completed.

        Logic: for each (agent, route) pair, compute the difficulty score of every
        completed job.  Any *pending* job whose difficulty score is strictly lower than
        the maximum completed difficulty for that pair is considered redundant — if the
        model handled the hardest condition we've seen, the easier variant adds little
        new signal — and is marked 'skipped'.

        Returns the number of jobs pruned.  Pass dry_run=True to report without writing.
        """
        import math as _math
        import xml.etree.ElementTree as _ET

        # ── Difficulty helpers (duplicated locally so this method is self-contained) ─
        _WEATHER_DIFF = [
            0.0, 0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 1.5, 2.0,
            3.5, 4.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.5, 5.0, 5.5,
        ]

        def _geo(route_name, _cache={}):
            if route_name in _cache:
                return _cache[route_name]
            try:
                root = _ET.parse(str(self.routes_dir / route_name)).getroot()
                scores = []
                for el in root.findall('route'):
                    wps = el.findall('waypoint')
                    if len(wps) < 2:
                        scores.append(0.0); continue
                    xs   = [float(w.get('x', 0)) for w in wps]
                    ys   = [float(w.get('y', 0)) for w in wps]
                    yaws = [float(w.get('yaw', 0)) % 360 for w in wps]
                    path = sum(_math.hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) for i in range(len(xs)-1))
                    deltas = []
                    for i in range(1, len(yaws)):
                        d = abs(yaws[i] - yaws[i-1])
                        if d > 180: d = 360 - d
                        deltas.append(d)
                    scores.append(sum(1 for d in deltas if d > 45) * 2.0
                                  + path / 500.0 + sum(deltas) / 180.0)
                v = sum(scores) / len(scores) if scores else 0.0
            except Exception:
                v = 0.0
            _cache[route_name] = v
            return v

        def _difficulty(agent, route, weather):
            w = _WEATHER_DIFF[weather] if weather < len(_WEATHER_DIFF) else 2.5
            return _geo(route) + w

        def _do_prune():
            try:
                with open(self.completed_file) as f:
                    comp = json.load(f)
                completed_jobs = comp.get('jobs', [])
            except Exception:
                completed_jobs = []

            with open(self.queue_file) as f:
                q = json.load(f)

            # Max difficulty successfully completed per (agent, route)
            best: dict = {}
            for j in completed_jobs:
                if j.get('status') == 'completed':
                    key = (j['agent'], j['route'])
                    d = _difficulty(j['agent'], j['route'], int(j.get('weather', 0)))
                    best[key] = max(best.get(key, 0.0), d)

            pruned = 0
            for j in q['jobs']:
                if j.get('status') != 'pending':
                    continue
                key = (j['agent'], j['route'])
                if key not in best:
                    continue
                d = _difficulty(j['agent'], j['route'], int(j.get('weather', 0)))
                if d < best[key]:
                    if not dry_run:
                        j['status'] = 'skipped'
                    pruned += 1
                    print(f"  {'[dry] ' if dry_run else ''}skip job {j['id']}"
                          f" {j['agent']}/{j['route']} w{j['weather']}"
                          f" (difficulty {d:.2f} < best {best[key]:.2f})")

            if not dry_run and pruned:
                with open(self.queue_file, 'w') as f:
                    json.dump(q, f, indent=2)

            return pruned

        n = self._with_lock(_do_prune)
        label = "would prune" if dry_run else "pruned"
        print(f"Redundant pruning: {label} {n} pending jobs.")
        return n

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
            
            combinations = {}
            for job in completed_data['jobs']:
                if job.get('duration'):
                    key = f"{job['agent']}_{job['route']}"
                    combinations.setdefault(key, []).append(job['duration'])
            
            updates = 0
            for key, durations in combinations.items():
                if len(durations) >= 2:
                    durations.sort()
                    median = durations[len(durations)//2]
                    old_estimate = runtime_data['combinations'].get(key, runtime_data['default'])
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
            
            failed_jobs = [j for j in queue_data['jobs'] if j['status'] == 'failed']
            if failed_jobs:
                print("\nFAILURE ANALYSIS:")
                print(f"Total failed jobs: {len(failed_jobs)}")
                failures_by_town = {}
                for job in failed_jobs:
                    failures_by_town.setdefault(job.get('town', 'unknown'), []).append(job)
                print("\nFailures by town:")
                for town, jobs in sorted(failures_by_town.items()):
                    print(f"  Town {town}: {len(jobs)} failures")
                    failed_routes = set(j['route'] for j in jobs)
                    for route in sorted(failed_routes):
                        count = sum(1 for j in jobs if j['route'] == route)
                        print(f"    - {route}: {count} failures")
            
            by_agent = {}
            for job in completed_data['jobs']:
                if job.get('duration'):
                    a = job['agent']
                    by_agent.setdefault(a, {'durations': [], 'routes': {}})
                    by_agent[a]['durations'].append(job['duration'])
                    r = job['route']
                    by_agent[a]['routes'].setdefault(r, []).append(job['duration'])
            
            for agent, data in sorted(by_agent.items()):
                durations = data['durations']
                avg_time = sum(durations) / len(durations)
                min_time = min(durations); max_time = max(durations)
                print(f"\n{agent}:")
                print(f"  Jobs completed: {len(durations)}")
                print(f"  Average time: {str(timedelta(seconds=int(avg_time)))}")
                print(f"  Min/Max: {str(timedelta(seconds=int(min_time)))} - {str(timedelta(seconds=int(max_time)))}")
                print("  By route:")
                for route, route_times in sorted(data['routes'].items()):
                    route_avg = sum(route_times) / len(route_times)
                    estimate_key = f"{agent}_{route}"
                    estimate = runtime_data['combinations'].get(estimate_key, runtime_data['default'])
                    accuracy = (1 - abs(route_avg - estimate) / max(route_avg, 1e-6)) * 100
                    print(f"    {route[:30]:30} Avg: {route_avg:6.0f}s Est: {estimate:6.0f}s ({accuracy:.0f}% accurate)")
            print("="*60)
        except FileNotFoundError:
            print("State files not found")
    
    def check_setup(self):
        """Check if the environment is properly set up"""
        print("\nCHECKING SETUP")
        print("="*60)
        if self.routes_dir.exists():
            route_files = list(self.routes_dir.glob('*.xml'))
            print(f"✓ Routes directory exists: {self.routes_dir}")
            print(f"  Found {len(route_files)} route files")
        else:
            print(f"✗ Routes directory not found: {self.routes_dir}")
        
        if self.scenarios_dir.exists():
            scenario_files = list(self.scenarios_dir.glob('*.json'))
            print(f"✓ Scenarios directory exists: {self.scenarios_dir}")
            print(f"  Found {len(scenario_files)} scenario files")
        else:
            print(f"✗ Scenarios directory not found: {self.scenarios_dir}")
        
        town_routes = self._discover_routes_and_scenarios()
        if town_routes:
            print(f"\n✓ Found {sum(len(r) for r in town_routes.values())} valid route/scenario combinations")
            for town, routes in sorted(town_routes.items()):
                print(f"  Town {town}: {len(routes)} routes")
        else:
            print("\n✗ No valid route/scenario combinations found!")
        
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

        def _estimate_sec(job):
            # prefer longest first (your heuristic), but also prefer fewer attempts
            try:
                with open(self.runtime_file, 'r') as f:
                    rt = json.load(f)
            except Exception:
                rt = {"default": 3600, "combinations": {}}
            key = f"{job['agent']}_{job['route']}"
            est = rt.get("combinations", {}).get(key, rt.get("default", 3600))
            return int(est)

        def _route_difficulty(route_name, _cache={}):
            """Parse route XML and return mean per-route geometric difficulty score.

            Score = sharp_turns*2 + path_length_m/500 + total_heading_change_deg/180
              - sharp_turns: waypoint-to-waypoint heading jumps > 45° (intersections)
              - path_length: total Euclidean distance along waypoints
              - total_heading_change: accumulated absolute heading change (curves/loops)
            Result is memoised — each XML file is parsed at most once per process.
            """
            import xml.etree.ElementTree as _ET
            import math as _math
            if route_name in _cache:
                return _cache[route_name]
            xml_path = self.routes_dir / route_name
            try:
                root = _ET.parse(str(xml_path)).getroot()
                route_scores = []
                for route_el in root.findall('route'):
                    wps = route_el.findall('waypoint')
                    if len(wps) < 2:
                        route_scores.append(0.0)
                        continue
                    xs   = [float(w.get('x',   0.0)) for w in wps]
                    ys   = [float(w.get('y',   0.0)) for w in wps]
                    yaws = [float(w.get('yaw', 0.0)) % 360 for w in wps]

                    path_len = sum(
                        _math.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i])
                        for i in range(len(xs) - 1)
                    )
                    heading_deltas = []
                    for i in range(1, len(yaws)):
                        d = abs(yaws[i] - yaws[i - 1])
                        if d > 180:
                            d = 360 - d
                        heading_deltas.append(d)
                    total_heading = sum(heading_deltas)
                    sharp_turns   = sum(1 for d in heading_deltas if d > 45)

                    route_scores.append(sharp_turns * 2.0 + path_len / 500.0 + total_heading / 180.0)

                score = sum(route_scores) / len(route_scores) if route_scores else 0.0
            except Exception:
                score = 0.0
            _cache[route_name] = score
            return score

        # Weather difficulty table indexed by position in _WEATHER_IDS from consolidated_agent.py:
        # [ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset,
        #  MidRainyNoon, MidRainSunset, WetCloudyNoon, WetCloudySunset,
        #  HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset,
        #  ClearNight, CloudyNight, WetNight, WetCloudyNight,
        #  SoftRainNight, MidRainyNight, HardRainNight]
        _WEATHER_DIFF = [
            0.0,  # 0  ClearNoon        — baseline, full visibility
            0.5,  # 1  ClearSunset      — low sun, mild glare
            0.5,  # 2  CloudyNoon
            1.0,  # 3  CloudySunset
            1.5,  # 4  WetNoon          — slippery road surface
            2.0,  # 5  WetSunset
            2.5,  # 6  MidRainyNoon     — rain + reduced visibility
            3.0,  # 7  MidRainSunset
            1.5,  # 8  WetCloudyNoon
            2.0,  # 9  WetCloudySunset
            3.5,  # 10 HardRainNoon     — heavy rain, low visibility
            4.0,  # 11 HardRainSunset
            2.0,  # 12 SoftRainNoon
            2.5,  # 13 SoftRainSunset
            3.0,  # 14 ClearNight       — darkness, perception-heavy
            3.5,  # 15 CloudyNight
            4.0,  # 16 WetNight
            4.5,  # 17 WetCloudyNight
            4.5,  # 18 SoftRainNight
            5.0,  # 19 MidRainyNight
            5.5,  # 20 HardRainNight    — worst case
        ]

        def _scenario_difficulty(route_name, _cache={}, _town_cache={}):
            """Count adversarial trigger locations within 25 m of the route.

            All scenario types (Scenario1..Scenario10) share identical spawn
            positions in each town JSON — confirmed by inspection. We therefore:
              1. Extract unique event positions from any one type.
              2. For each route, count how many of those positions are within
                 RADIUS metres of at least one waypoint (inner loop short-circuits).
              3. Deduplicate hits at 20 m grid resolution to avoid over-counting
                 the multiple offset variations clustered around each road point.
              4. Scale by mean scenario-type difficulty weight × SCALE.

            TYPE_WEIGHT reflects hazard severity:
              Scenario1  — slow leading vehicle (low challenge)
              Scenario3  — cut-in vehicle
              Scenario4  — stationary obstacle
              Scenario7  — pedestrian at marked crossing
              Scenario8  — jaywalking pedestrian
              Scenario9  — sudden appearance (pedestrian/cyclist from occluded area)
              Scenario10 — slow vehicle + secondary hazard
            """
            import re as _re, xml.etree.ElementTree as _ET, math as _math

            if route_name in _cache:
                return _cache[route_name]

            m = _re.search(r'town(\d+)', route_name, _re.IGNORECASE)
            if not m:
                _cache[route_name] = 0.0
                return 0.0

            town_tag = f"town{m.group(1).zfill(2)}"

            if town_tag not in _town_cache:
                TYPE_WEIGHT = {
                    'Scenario1':  1.0,
                    'Scenario3':  3.0,
                    'Scenario4':  2.0,
                    'Scenario7':  2.5,
                    'Scenario8':  3.5,
                    'Scenario9':  4.5,
                    'Scenario10': 2.0,
                }
                try:
                    spath = self.scenarios_dir / f"{town_tag}_all_scenarios.json"
                    with open(str(spath)) as _f:
                        sdata = json.load(_f)
                    town_key = list(sdata['available_scenarios'][0].keys())[0]
                    slist = sdata['available_scenarios'][0][town_key]

                    # Unique positions (all types share the same set — verified)
                    evt_pos = list({
                        (round(ev['transform']['x'], 1), round(ev['transform']['y'], 1))
                        for ev in slist[0]['available_event_configurations']
                    })

                    ws = [TYPE_WEIGHT.get(s['scenario_type'], 2.0) for s in slist]
                    mean_w = sum(ws) / len(ws) if ws else 2.0
                    _town_cache[town_tag] = (evt_pos, mean_w)
                except Exception:
                    _town_cache[town_tag] = ([], 2.0)

            evt_pos, mean_w = _town_cache[town_tag]
            if not evt_pos:
                _cache[route_name] = 0.0
                return 0.0

            RADIUS = 25.0   # metres — distance from waypoint to trigger location
            SCALE  = 0.25   # per hit-cell contribution (calibrated to geometry score scale)

            try:
                root = _ET.parse(str(self.routes_dir / route_name)).getroot()
                route_scores = []
                for route_el in root.findall('route'):
                    wps = [
                        (float(w.get('x', 0.0)), float(w.get('y', 0.0)))
                        for w in route_el.findall('waypoint')
                    ]
                    hit_cells = set()
                    for ex, ey in evt_pos:
                        for wx, wy in wps:
                            if _math.hypot(wx - ex, wy - ey) <= RADIUS:
                                hit_cells.add((int(ex // 20), int(ey // 20)))
                                break   # one waypoint match is enough per event
                    route_scores.append(len(hit_cells) * mean_w * SCALE)
                score = sum(route_scores) / len(route_scores) if route_scores else 0.0
            except Exception:
                score = 0.0

            _cache[route_name] = score
            return score

        def _job_difficulty(job):
            route        = job.get('route', '')
            route_score  = _route_difficulty(route) + _scenario_difficulty(route)
            weather_idx  = int(job.get('weather', 0))
            weather_score = _WEATHER_DIFF[weather_idx] if weather_idx < len(_WEATHER_DIFF) else 2.5
            return route_score + weather_score

        def _reserve_next():
            with open(self.queue_file, 'r') as f:
                q = json.load(f)

            # choose among PENDING: fewest attempts first, then agent priority (lav first for
            # debugging), then hardest scenario first (route geometry + weather), then longest
            # estimated time as tiebreak.
            _agent_priority = {"interfuser": 0, "tcp": 1, "lav": 2}
            pending = [j for j in q['jobs'] if j.get('status') == 'pending']
            if not pending:
                return None

            pending.sort(key=lambda j: (
                j.get('attempts', 0),
                _agent_priority.get(j.get('agent', ''), 99),
                -_job_difficulty(j),
                -_estimate_sec(j),
            ))

            job = pending[0]
            job['status'] = 'running'
            job['attempts'] = int(job.get('attempts', 0)) + 1

            # stamp node+gpu so the monitor can correlate
            job['node'] = self.node_name
            # ensure gpu id exists even if GPU_ID wasn't exported
            job['gpu'] = int(os.environ.get('GPU_ID')) if os.environ.get('GPU_ID') is not None else self._derive_gpu_id(port)

            job['start_time'] = datetime.utcnow().isoformat() + 'Z'
            with open(self.queue_file, 'w') as f:
                json.dump(q, f, indent=2)
            return job


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

        m = re.search(r'(?:town)?(\d+)', str(town)) or re.search(r'routes_town(\d+)_', route_name, flags=re.IGNORECASE)
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
        eval_cmd_template = os.environ.get('EVAL_CMD_TEMPLATE', '').strip()
        host = str(host); port = str(port); tm_port = str(tm_port)
        env = os.environ.copy()

        # Core job context always available here:
        route_stem = Path(routes_file).stem  # e.g., "routes_town04_tiny"

        # Fill base env first
        env.update({
            'AGENT_CFG':       str(agent_cfg),
            'ROUTES_FILE':     str(routes_file),
            'SCENARIOS_FILE':  str(scenarios_file),
            'WEATHER_INDEX':   str(weather_idx),
            'CARLA_HOST':      host,
            'CARLA_PORT':      port,
            'TM_PORT':         tm_port,

            'AGENT_NAME':      str(agent_name),
            'ROUTE_NAME':      str(route_stem),
            'TOWN_NUM':        str(town_num) if town_num is not None else '',
            'DATASET_DIR':     env.get('DATASET_DIR', str(self.project_root / 'dataset')),
        })

        # Derive normalized labels exactly like consolidated_agent
        def _label_weather(wi):
            try:  return f"weather_{int(wi)}"
            except: return f"weather_{str(wi)}"

        def _label_map(tn):
            try:
                return f"map_{int(tn):02d}"
            except:
                return f"map_{tn or 'unknown'}"

        weather_label = _label_weather(env['WEATHER_INDEX'])
        map_label     = _label_map(env.get('TOWN_NUM', ''))

        # Construct the per-job SAVE_PATH and CHECKPOINT_ENDPOINT (where LB writes results.json)
        save_path = os.path.join(env['DATASET_DIR'], env['AGENT_NAME'], weather_label, map_label, env['ROUTE_NAME'])
        env.update({
            'SAVE_PATH':            save_path,
            'CHECKPOINT_ENDPOINT':  os.path.join(save_path, 'results.json'),
        })

        # On retry (attempts > 1), wipe the previous run's artefacts so sensor
        # frames from the failed attempt don't intermingle with the new run.
        if job.get('attempts', 1) > 1 and Path(save_path).exists():
            import shutil
            shutil.rmtree(save_path)
        
        if 'WEATHER_PRESET' in os.environ and os.environ['WEATHER_PRESET'].strip():
            env['WEATHER_PRESET'] = os.environ['WEATHER_PRESET'].strip()

        # --- Pin the agent's PyTorch to a specific GPU ---
        # Without this, CUDA_VISIBLE_DEVICES was never set for the agent, so the
        # model's hardcoded device="cuda" resolved to cuda:0 (physical GPU 0) in
        # every worker — all 8 workers' inference dogpiled GPU 0 while the CARLA
        # servers were correctly spread across GPUs. Co-locate each agent on its
        # own CARLA server's GPU. AGENT_GPU_OFFSET (default 0 = co-locate) shifts
        # the agent onto a neighbour GPU for split experiments; with all GPUs
        # visible inside the container, CUDA_VISIBLE_DEVICES selects the physical
        # one as torch cuda:0.
        carla_gpu = job.get('gpu')
        if carla_gpu is None:
            carla_gpu = self._derive_gpu_id(port)
        try:
            local_gpus = max(1, int(self.local_gpus))
            pin = os.environ.get('AGENT_GPU_PIN')
            if pin is not None and pin.strip() != '':
                # Benchmark/override: force ALL agents onto one fixed GPU.
                # AGENT_GPU_PIN=0 reproduces the pre-fix dogpile (every worker's
                # inference on GPU 0) for A/B comparison against co-location.
                agent_gpu = int(pin) % local_gpus
            else:
                offset = int(os.environ.get('AGENT_GPU_OFFSET', '0'))
                agent_gpu = (int(carla_gpu) + offset) % local_gpus
        except Exception:
            agent_gpu = int(carla_gpu)
        env['CUDA_VISIBLE_DEVICES'] = str(agent_gpu)

        # Hard forward into container env
        def _mirror(keys):
            for k in keys:
                if k in env and env[k] is not None:
                    env['SINGULARITYENV_' + k] = str(env[k])
                    env['APPTAINERENV_' + k]   = str(env[k])

        _mirror([
            'AGENT_CFG','ROUTES_FILE','SCENARIOS_FILE',
            'AGENT_NAME','ROUTE_NAME','TOWN_NUM','DATASET_DIR',
            'WEATHER_INDEX','WEATHER_PRESET',
            'CARLA_HOST','CARLA_PORT','TM_PORT',
            'SAVE_PATH','CHECKPOINT_ENDPOINT',
            'CUDA_VISIBLE_DEVICES'
        ])


        # Proceed to build the command as before (EVAL_CMD_TEMPLATE or fallback)
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
                '--checkpoint', os.path.join(save_path, 'results.json'),
                '--host', host, '--port', port, '--trafficManagerPort', tm_port
            ]


        # Emit header in the same shape continuous_collection.sh writes, so
        # `manage_continuous.py logs --job N --gpu G` greps work in persistent mode too.
        print("=" * 42)
        print(f"[GPU {job.get('gpu', '?')}] Starting Job #{job['id']}")
        print(f"Agent: {agent_name}")
        print(f"Route: {route_name}")
        print(f"Weather: {weather_idx}")
        print("=" * 42)
        print(f"[RUN] Job {job['id']} agent={agent_name} route={route_name} weather={weather_idx}")
        gpu_id_for_display = job['gpu'] if job.get('gpu') is not None else self._derive_gpu_id(port)
        self._write_health(
            gpu_id=gpu_id_for_display,
            status="busy",
            message=f"running job {job['id']} {agent_name}/{route_stem} w{weather_idx}",
            rpc_port=int(port),
            tm_port=int(tm_port),
            current_job=job['id']
        )

        # --- Fast-fail if the CARLA server on this port isn't reachable ---
        # On a degraded node the evaluator otherwise blocks the full client
        # timeout (~600s) at get_trafficmanager() before dying. A sub-second
        # socket probe turns that 10-minute waste into an instant skip. The job
        # is returned to 'pending' (attempt refunded) so it retries once a
        # healthy server is back; a short backoff avoids busy-spin on a dead node.
        def _port_open(h, p, t=1.0):
            try:
                with socket.create_connection((h, int(p)), timeout=t):
                    return True
            except Exception:
                return False

        if not _port_open(host, port):
            print(f"[run] CARLA RPC {host}:{port} unreachable; skipping job {job['id']} (server down)")
            def _requeue():
                with open(self.queue_file, 'r') as f:
                    q = json.load(f)
                for j in q['jobs']:
                    if j['id'] == job['id']:
                        j['status'] = 'pending'
                        j['attempts'] = max(0, int(j.get('attempts', 1)) - 1)
                        j['start_time'] = None
                        break
                with open(self.queue_file, 'w') as f:
                    json.dump(q, f, indent=2)
            self._with_lock(_requeue)
            self._write_health(gpu_id=gpu_id_for_display, status="idle",
                               message=f"server down on port {port}; job {job['id']} requeued",
                               rpc_port=int(port), tm_port=int(tm_port), current_job=None)
            time.sleep(float(os.environ.get('DEAD_SERVER_BACKOFF_SEC', '20')))
            return 3

        # --- Launch with a hard wall-clock cap ---
        # The leaderboard's route timeout is measured in *game* time; on a node
        # where the sim runs ~8-12x slower than real-time (observed median ratio
        # ~0.12) a single route can burn many real hours, and a 20-route file
        # (e.g. routes_town04_long) can run for days. JOB_TIMEOUT_SEC bounds the
        # real wall-clock per job regardless of game-time progress.
        job_timeout = int(os.environ.get('JOB_TIMEOUT_SEC', str(4 * 3600)))
        start_ts = time.time()
        rc = None
        proc = None
        try:
            # start_new_session => own process group, so we can kill the whole
            # singularity->python subtree on timeout (the persistent CARLA
            # server runs in a separate group and is left untouched).
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=job_timeout)
            except subprocess.TimeoutExpired:
                print(f"[run] job {job['id']} exceeded JOB_TIMEOUT_SEC={job_timeout}s; terminating")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                try:
                    rc = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    try:
                        rc = proc.wait(timeout=30)
                    except Exception:
                        rc = 124
                if rc is None:
                    rc = 124  # conventional 'timed out' exit code
        except KeyboardInterrupt:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
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

            # Parse leaderboard scores from results.json if available.
            results_path = Path(save_path) / 'results.json'
            try:
                with open(results_path) as _rf:
                    lb = json.load(_rf)
                records = lb.get('_checkpoint', {}).get('records', [])
                if records:
                    composed = [r['scores']['score_composed'] for r in records if 'scores' in r]
                    route_pct = [r['scores']['score_route']    for r in records if 'scores' in r]
                    entry['score_composed']   = round(sum(composed)  / len(composed),  4) if composed  else None
                    entry['score_route']      = round(sum(route_pct) / len(route_pct), 4) if route_pct else None
                    entry['score_n_routes']   = len(records)
                    entry['route_statuses']   = [r.get('status', '') for r in records]
            except Exception:
                pass  # results.json absent or malformed; scores omitted

            comp['jobs'].append(entry)
            with open(self.completed_file, 'w') as f:
                json.dump(comp, f, indent=2)

            gpu_id_for_display = job.get('gpu')
            if gpu_id_for_display is None:
                gpu_id_for_display = self._derive_gpu_id(port)

            self._bump_gpu_stats(gpu_id_for_display, duration, rc == 0)
            self._write_health(
                gpu_id=gpu_id_for_display,
                status="idle",
                message=f"last job {job['id']} rc={rc} dur={duration}s",
                rpc_port=port,
                tm_port=tm_port,
                current_job=None
            )

        self._with_lock(_finish)
        return rc

def _safe_read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def main():
    parser = argparse.ArgumentParser(description='Manage continuous data collection')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    subparsers.add_parser('check', help='Check setup and available combinations')
    subparsers.add_parser('status', help='Show current status')

    run_parser = subparsers.add_parser('run', help='Run the next pending job (one evaluation)')
    run_parser.add_argument('--host', default=os.environ.get('CARLA_HOST', '127.0.0.1'))
    run_parser.add_argument('--port', type=int, default=int(os.environ.get('CARLA_PORT', '2000')))
    run_parser.add_argument('--trafficManagerPort', type=int, default=int(os.environ.get('TM_PORT', os.environ.get('TRAFFIC_MANAGER_PORT', '5000'))))
    run_parser.add_argument('extra', nargs=argparse.REMAINDER, help='Additional args passed to evaluator')

    reset_parser = subparsers.add_parser('reset', help='Reset job queue')
    reset_parser.add_argument('--agents', nargs='+', help='Agents to include')
    reset_parser.add_argument('--weather', nargs='+', type=int, help='Weather indices')
    reset_parser.add_argument('--routes', nargs='+', help='Route files')
    reset_parser.add_argument('--smoke', action='store_true',
                              help='Tiny validation queue (tiny routes + weather 0) for ~1h end-to-end checks')
    reset_parser.add_argument('--limit', type=int, default=None,
                              help='Cap total jobs, interleaved across agents')
    
    retry_parser = subparsers.add_parser('retry', help='Retry failed jobs')
    retry_parser.add_argument('--max-attempts', type=int, default=3, help='Maximum attempts')
    
    add_parser = subparsers.add_parser('add', help='Add new jobs')
    add_parser.add_argument('agent', help='Agent name')
    add_parser.add_argument('--weather', nargs='+', type=int, help='Weather indices')
    add_parser.add_argument('--routes', nargs='+', help='Route files')
    
    cancel_parser = subparsers.add_parser('cancel', help='Cancel pending jobs')
    cancel_parser.add_argument('--agent', help='Cancel only for specific agent')
    
    export_parser = subparsers.add_parser('export', help='Export results')
    export_parser.add_argument('--output', default='collection_results.json', 
                              help='Output file')
    
    subparsers.add_parser('optimize', help='Optimize runtime estimates')
    subparsers.add_parser('analyze', help='Show runtime analysis')

    prune_parser = subparsers.add_parser('prune', help='Skip pending jobs made redundant by harder completed jobs')
    prune_parser.add_argument('--dry-run', action='store_true', help='Report without modifying the queue')

    reclaim_parser = subparsers.add_parser('reclaim', help='Recover jobs stuck in running state after worker crash')
    reclaim_parser.add_argument('--stale-hours', type=float, default=6.0,
                                help='Hours without health beacon before a running job is considered stale (default: 6)')
    
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
        manager.reset_queue(args.agents, args.weather, args.routes,
                            smoke=getattr(args, 'smoke', False),
                            limit=getattr(args, 'limit', None))
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
    elif args.command == 'prune':
        manager.prune_redundant(dry_run=getattr(args, 'dry_run', False))
    elif args.command == 'reclaim':
        manager.reclaim_stale(stale_hours=args.stale_hours)
    elif args.command == 'run':
        extra = args.extra if hasattr(args, 'extra') else []
        if extra and extra[0] == '--':  # argparse quirk when using REMAINDER
            extra = extra[1:]
        rc = manager.run_next_job(host=args.host, port=args.port,
                                tm_port=args.trafficManagerPort, extra_args=extra)
        sys.exit(rc)

if __name__ == '__main__':
    main()