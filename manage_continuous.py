#!/usr/bin/env python3
"""
SQLite-Optimized Management utility for continuous data collection.
Replaces JSON flat-files with a high-concurrency database.
"""

import sqlite3
import os
import sys
import subprocess
import time
import argparse
import re
import json # Only for exporting results
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class ContinuousManager:
    def __init__(self, state_dir: str = None):
        if state_dir is None:
            project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
            state_dir = os.path.join(project_root, 'collection_state')
        
        self.project_root = Path(os.environ.get('PROJECT_ROOT', os.getcwd()))
        self.state_dir = Path(state_dir)
        self.db_path = self.state_dir / 'collection.db'
        
        # Paths for discovery
        self.routes_dir = self.project_root / 'leaderboard/data/training_routes'
        self.scenarios_dir = self.project_root / 'leaderboard/data/scenarios'
        
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        """Get a database connection with Row factory enabled."""
        # Python 3.6 sqlite3.connect does not reliably accept PathLike; use str.
        conn = sqlite3.connect(str(self.db_path), timeout=60.0) # High timeout for safety
        conn.row_factory = sqlite3.Row
        # WAL mode is CRITICAL for concurrency (allows readers while writing)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        return conn

    def _init_db(self):
        """Initialize the schema if it doesn't exist."""
        with self._get_conn() as conn:
            # Job Queue Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    weather INTEGER NOT NULL,
                    route TEXT NOT NULL,
                    town TEXT,
                    status TEXT DEFAULT 'pending', -- pending, running, completed, failed, cancelled
                    attempts INTEGER DEFAULT 0,
                    gpu_id INTEGER,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration REAL
                )
            """)
            # Index for fast FIFO popping
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status_id ON jobs(status, id)")

            # Runtime Estimates Table (Key-Value)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runtime_estimates (
                    key TEXT PRIMARY KEY,
                    estimate REAL
                )
            """)

            # GPU Status Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpu_status (
                    node TEXT,
                    gpu_id INTEGER,
                    status TEXT,
                    current_job_id INTEGER,
                    jobs_completed INTEGER DEFAULT 0,
                    total_runtime REAL DEFAULT 0,
                    last_heartbeat REAL,
                    PRIMARY KEY (node, gpu_id)
                )
            """)
    
    # --- Discovery Logic (Unchanged) ---
    def _discover_routes_and_scenarios(self) -> Dict[str, List[str]]:
        town_routes = {}
        if not self.routes_dir.exists(): return {}
        
        for route_file in self.routes_dir.glob('*.xml'):
            match = re.search(r'routes_town(\d+)_', route_file.name)
            if match:
                town_num = match.group(1)
                scenario_file = self.scenarios_dir / f'town{town_num}_all_scenarios.json'
                if scenario_file.exists():
                    town_routes.setdefault(town_num, []).append(route_file.name)

        # Optional: restrict to a known set of towns.
        # This is useful in Docker images that ship only a subset of maps.
        allowed_raw = os.environ.get("CARLA_ALLOWED_TOWNS")
        if allowed_raw:
            allowed: set = set()
            for token in re.split(r"[\s,]+", allowed_raw.strip()):
                if not token:
                    continue
                m = re.search(r"(?:town)?0*(\d+)", token, flags=re.IGNORECASE)
                if m:
                    allowed.add(m.group(1).zfill(2))
            if allowed:
                town_routes = {k: v for k, v in town_routes.items() if k.zfill(2) in allowed}
        
        for town in town_routes: town_routes[town].sort()
        return town_routes

    def _get_valid_combinations(self, agents_list: List[str] = None, 
                               weather_list: List[int] = None,
                               routes_list: List[str] = None) -> List[Dict]:
        town_routes = self._discover_routes_and_scenarios()
        if not town_routes: return []

        if agents_list is None:
            configs_dir = self.project_root / 'leaderboard/team_code/configs'
            agents_list = [f.stem for f in configs_dir.glob('*.yaml')] if configs_dir.exists() else ['interfuser']
        
        if weather_list is None: weather_list = list(range(15))

        if routes_list:
            filtered = {}
            for town, routes in town_routes.items():
                valid = [r for r in routes if r in routes_list]
                if valid: filtered[town] = valid
            town_routes = filtered

        combinations = []
        for agent in agents_list:
            for weather_idx in weather_list:
                for town, routes in town_routes.items():
                    for route in routes:
                        combinations.append({
                            'agent': agent, 'weather': weather_idx,
                            'route': route, 'town': town
                        })
        return combinations

    # --- Job Management ---
    def reset_queue(self, agents: List[str] = None, weather: List[int] = None, routes: List[str] = None):
        """Wipe DB and repopulate."""
        combinations = self._get_valid_combinations(agents, weather, routes)
        if not combinations:
            print("No valid combinations found.")
            return

        with self._get_conn() as conn:
            # Transactional Reset
            conn.execute("DELETE FROM jobs")
            conn.execute("DELETE FROM runtime_estimates")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='jobs'") # Reset ID counter
            
            # Bulk Insert
            conn.executemany(
                "INSERT INTO jobs (agent, weather, route, town) VALUES (:agent, :weather, :route, :town)",
                combinations
            )
            
            # Initialize Estimates
            estimates = []
            for combo in combinations:
                key = f"{combo['agent']}_{combo['route']}"
                est = 3600
                if 'short' in combo['route']: est = 1800
                elif 'long' in combo['route']: est = 5400
                estimates.append((key, est))
            
            conn.executemany("INSERT OR IGNORE INTO runtime_estimates (key, estimate) VALUES (?, ?)", estimates)
            
        print(f"Queue reset with {len(combinations)} jobs.")

    def add_jobs(self, agent: str, weather: List[int] = None, routes: List[str] = None):
        combinations = self._get_valid_combinations([agent], weather, routes)
        if not combinations: return

        with self._get_conn() as conn:
            conn.executemany(
                "INSERT INTO jobs (agent, weather, route, town) VALUES (:agent, :weather, :route, :town)",
                combinations
            )

            # Ensure there is a runtime estimate for each (agent, route) pair.
            # Scheduling uses these estimates to run shorter jobs first.
            estimates = []
            for combo in combinations:
                key = f"{combo['agent']}_{combo['route']}"
                est = 3600
                if 'short' in combo['route']:
                    est = 1800
                elif 'long' in combo['route']:
                    est = 5400
                estimates.append((key, est))
            conn.executemany(
                "INSERT OR IGNORE INTO runtime_estimates (key, estimate) VALUES (?, ?)",
                estimates,
            )
        print(f"Added {len(combinations)} jobs for {agent}.")

    def retry_failed(self, max_attempts: int = 3):
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status='pending' WHERE status='failed' AND attempts < ?", 
                (max_attempts,)
            )
            print(f"Reset {cur.rowcount} failed jobs for retry.")

    def cancel_pending(self, agent: str = None):
        with self._get_conn() as conn:
            if agent:
                cur = conn.execute("UPDATE jobs SET status='cancelled' WHERE status='pending' AND agent=?", (agent,))
            else:
                cur = conn.execute("UPDATE jobs SET status='cancelled' WHERE status='pending'")
            print(f"Cancelled {cur.rowcount} jobs.")

    # --- Execution Logic (The Optimized Hot Path) ---
    def run_next_job(self, host: str, port: int, tm_port: int, extra_args: list = None) -> int:
        """
        Atomically pop the next job from DB, run it, and update status.
        Uses explicit transactions to prevent race conditions between workers.
        """
        gpu_id = int(os.environ.get('GPU_ID', -1))
        node_name = os.environ.get('SLURMD_NODENAME', 'unknown')
        
        job_data = None
        
        # 1. ATOMIC POP
        with self._get_conn() as conn:
            # BEGIN IMMEDIATE prevents other writers from starting a transaction
            # ensuring we don't grab the same job as another worker.
            conn.execute("BEGIN IMMEDIATE")

            # Intelligent scheduling: run the shortest estimated jobs first.
            # Estimates are tracked in runtime_estimates keyed by "<agent>_<route>".
            cursor = conn.execute(
                """
                SELECT j.id, j.agent, j.route, j.town, j.weather, j.attempts
                FROM jobs AS j
                LEFT JOIN runtime_estimates AS r
                  ON r.key = (j.agent || '_' || j.route)
                WHERE j.status='pending'
                ORDER BY COALESCE(r.estimate, 999999.0) ASC, j.id ASC
                LIMIT 1
                """
            )
            job_data = cursor.fetchone()
            
            if job_data:
                # Mark as running immediately
                conn.execute("""
                    UPDATE jobs 
                    SET status='running', start_time=?, attempts=attempts+1, gpu_id=? 
                    WHERE id=?
                """, (datetime.utcnow().isoformat(), gpu_id, job_data['id']))
                
                # Update GPU Status
                cur2 = conn.execute("""
                    UPDATE gpu_status
                    SET status='busy', current_job_id=?, last_heartbeat=?
                    WHERE node=? AND gpu_id=?
                """, (job_data['id'], time.time(), node_name, gpu_id))

                # SQLite < 3.24 doesn't support ON CONFLICT DO UPDATE; do update-then-insert.
                # If no row existed, insert it.
                if getattr(cur2, "rowcount", 0) == 0:
                    conn.execute(
                        "INSERT INTO gpu_status (node, gpu_id, status, current_job_id, last_heartbeat) VALUES (?, ?, 'busy', ?, ?)",
                        (node_name, gpu_id, job_data['id'], time.time()),
                    )
            
            # Commit happens automatically on exit of context manager
        
        if not job_data:
            print("No pending jobs found.")
            return 2 # Exit code 2 indicates idle

        # 2. PREPARE EXECUTION
        agent_cfg = self.project_root / 'leaderboard/team_code/configs' / f"{job_data['agent']}.yaml"
        agent_code = self.project_root / 'leaderboard/team_code/consolidated_agent.py'
        routes_file = self.routes_dir / job_data['route']
        
        # Find Scenario File
        m = re.search(r'(?:town)?(\d+)', str(job_data['town'])) or re.search(r'routes_town(\d+)_', job_data['route'])
        town_num = m.group(1) if m else "01" # Fallback
        scenarios_file = self.scenarios_dir / f'town{town_num}_all_scenarios.json'
        
        # 3. RUN EVALUATOR
        print(f"[RUN] Job {job_data['id']} ({job_data['agent']}/{job_data['route']}) on GPU {gpu_id}")
        
        # Execution backend:
        # - If EVAL_CMD_TEMPLATE is set (HPC/Singularity path), format and run it via bash -lc
        # - Otherwise, run evaluator directly in the current Python environment (Docker/local path)
        eval_template = os.environ.get('EVAL_CMD_TEMPLATE')

        if eval_template:
            rendered = eval_template.format(
                ROUTES_FILE=str(routes_file),
                SCENARIOS_FILE=str(scenarios_file),
                AGENT_CODE=str(agent_code),
                AGENT_CFG=str(agent_cfg),
                HOST=str(host),
                PORT=str(port),
                TM_PORT=str(tm_port),
            )
            cmd = ['bash', '-lc', rendered]
            if extra_args:
                # Extra args in template mode must already be represented in the template
                # (kept for API compatibility; no-op here)
                pass
        else:
            cmd = [
                'python3', '-m', 'leaderboard.leaderboard_evaluator',
                '--routes', str(routes_file),
                '--scenarios', str(scenarios_file),
                '--agent', str(agent_code),
                '--agent-config', str(agent_cfg),
                '--host', str(host),
                '--port', str(port),
                '--trafficManagerPort', str(tm_port)
            ]
            if extra_args:
                cmd.extend(extra_args)
        
        env = os.environ.copy()
        env['WEATHER_INDEX'] = str(job_data['weather'])
        # Used by ConsolidatedAgent for dataset path + metadata.
        env.setdefault('WEATHER', str(job_data['weather']))
        env.setdefault('WEATHERS', str(job_data['weather']))
        env.setdefault('ROUTES', str(routes_file))

        # Prefer writing all collected outputs under the dedicated dataset dir.
        # docker-compose sets DATASET_DIR=/workspace/dataset.
        if 'DATASET_DIR' in env and env['DATASET_DIR']:
            env.setdefault('HPC_CARLA_DATASET_ROOT', env['DATASET_DIR'])

        start_ts = time.time()
        try:
            rc = subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            rc = 130
        duration = time.time() - start_ts

        final_status = 'completed' if rc == 0 else 'failed'

        # 4. UPDATE RESULT
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs 
                SET status=?, end_time=?, duration=? 
                WHERE id=?
            """, (final_status, datetime.utcnow().isoformat(), duration, job_data['id']))
            
            # Update GPU Status to Idle
            conn.execute("""
                UPDATE gpu_status 
                SET status='idle', current_job_id=NULL, 
                    jobs_completed=jobs_completed+1, total_runtime=total_runtime+?
                WHERE node=? AND gpu_id=?
            """, (duration, node_name, gpu_id))
            
            # Update Runtime Estimate (Weighted Average)
            if rc == 0:
                key = f"{job_data['agent']}_{job_data['route']}"
                row = conn.execute("SELECT estimate FROM runtime_estimates WHERE key=?", (key,)).fetchone()
                if row and row[0] is not None:
                    new_est = (float(row[0]) * 0.7) + (duration * 0.3)
                    conn.execute("UPDATE runtime_estimates SET estimate=? WHERE key=?", (new_est, key))
                else:
                    conn.execute("INSERT INTO runtime_estimates (key, estimate) VALUES (?, ?)", (key, duration))

        return rc

    # --- Reporting ---
    def get_status(self):
        with self._get_conn() as conn:
            # Aggregate Job Stats
            job_stats = conn.execute("""
                SELECT status, COUNT(*) as count FROM jobs GROUP BY status
            """).fetchall()
            stats = {row['status']: row['count'] for row in job_stats}
            
            total = sum(stats.values())
            print("\nCOLLECTION STATUS (SQLite)")
            print("="*40)
            print(f"Total:      {total}")
            print(f"Completed:  {stats.get('completed', 0)}")
            print(f"Pending:    {stats.get('pending', 0)}")
            print(f"Running:    {stats.get('running', 0)}")
            print(f"Failed:     {stats.get('failed', 0)}")
            print("-" * 40)
            
            # GPU Status
            gpus = conn.execute("SELECT * FROM gpu_status ORDER BY gpu_id").fetchall()
            print("Active GPUs:")
            for gpu in gpus:
                job_str = f"Job #{gpu['current_job_id']}" if gpu['current_job_id'] else ""
                print(f"  GPU {gpu['gpu_id']}: {gpu['status'].upper():<6} | {gpu['jobs_completed']} done | {job_str}")

    def export_results(self, output_file: str):
        with self._get_conn() as conn:
            # Fetch completed jobs as dicts
            rows = conn.execute("SELECT * FROM jobs WHERE status='completed'").fetchall()
            jobs = [dict(row) for row in rows]
            
            data = {
                "summary": {"total_completed": len(jobs), "exported_at": datetime.now().isoformat()},
                "jobs": jobs
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Exported {len(jobs)} jobs to {output_file}")

    def analyze(self):
        with self._get_conn() as conn:
            print("\nRUNTIME ANALYSIS")
            print("="*60)
            
            # Failed Routes
            failures = conn.execute("""
                SELECT route, COUNT(*) as cnt FROM jobs WHERE status='failed' GROUP BY route ORDER BY cnt DESC LIMIT 5
            """).fetchall()
            if failures:
                print("Top Failed Routes:")
                for r in failures: print(f"  {r['route']}: {r['cnt']} failures")
            
            # Avg Runtime per Agent
            avgs = conn.execute("""
                SELECT agent, AVG(duration) as avg_dur, COUNT(*) as cnt 
                FROM jobs WHERE status='completed' GROUP BY agent
            """).fetchall()
            print("\nPerformance by Agent:")
            for r in avgs:
                print(f"  {r['agent']}: {int(r['avg_dur'])}s avg over {r['cnt']} runs")

# --- Main CLI Entrypoint (Compatible with existing args) ---
def main():
    parser = argparse.ArgumentParser(description='SQLite Manager for Continuous Collection')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Run
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--host', default='127.0.0.1')
    run_parser.add_argument('--port', type=int, default=2000)
    run_parser.add_argument('--trafficManagerPort', type=int, default=5000)
    run_parser.add_argument('extra', nargs=argparse.REMAINDER)
    
    # Management
    reset = subparsers.add_parser('reset')
    reset.add_argument('--agents', nargs='+')
    reset.add_argument('--weather', nargs='+', type=int)
    reset.add_argument('--routes', nargs='+')
    
    subparsers.add_parser('status')
    
    add = subparsers.add_parser('add')
    add.add_argument('agent')
    add.add_argument('--weather', nargs='+', type=int)
    add.add_argument('--routes', nargs='+')
    
    subparsers.add_parser('retry')
    subparsers.add_parser('analyze')
    
    exp = subparsers.add_parser('export')
    exp.add_argument('output', nargs='?', default='results.json')

    # State Dir (Backwards compat)
    parser.add_argument('--state-dir', default=None)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    manager = ContinuousManager(args.state_dir)

    if args.command == 'run':
        extra = args.extra if hasattr(args, 'extra') else []
        if extra and extra[0] == '--': extra = extra[1:]
        rc = manager.run_next_job(args.host, args.port, args.trafficManagerPort, extra)
        sys.exit(rc)
    elif args.command == 'reset':
        manager.reset_queue(args.agents, args.weather, args.routes)
    elif args.command == 'status':
        manager.get_status()
    elif args.command == 'add':
        manager.add_jobs(args.agent, args.weather, args.routes)
    elif args.command == 'retry':
        manager.retry_failed()
    elif args.command == 'export':
        manager.export_results(args.output)
    elif args.command == 'analyze':
        manager.analyze()

if __name__ == '__main__':
    main()