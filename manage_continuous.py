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
from typing import List, Dict, Any, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _get_run_id() -> str:
    return (
        os.environ.get("HPC_CARLA_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
        or "local"
    )


def _append_run_event(state_dir: Path, event: Dict[str, Any]) -> None:
    """Append a single JSONL event under collection_state/runs/<run_id>/events.jsonl."""
    try:
        run_id = _get_run_id()
        run_dir = state_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / "events.jsonl"
        event = dict(event)
        event.setdefault("ts", _utc_now_iso())
        event.setdefault("run_id", run_id)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break collection.
        return


def _write_health_file(state_dir: Path, node: str, gpu_id: int, payload: Dict[str, Any]) -> None:
    """Write a per-GPU health JSON file in the format expected by carla_health_manager.py."""
    try:
        health_dir = state_dir / "health" / node
        health_dir.mkdir(parents=True, exist_ok=True)
        p = health_dir / f"gpu{int(gpu_id)}.json"
        now_unix = time.time()
        d = dict(payload)
        d.setdefault("node", node)
        d.setdefault("gpu_id", int(gpu_id))
        d["timestamp"] = datetime.utcnow().isoformat() + "Z"
        d["timestamp_unix"] = now_unix
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        return

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

            # Parsed Leaderboard results per job (summary)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_results (
                    job_id INTEGER PRIMARY KEY,
                    checkpoint_path TEXT,
                    progress_current INTEGER,
                    progress_total INTEGER,
                    global_status TEXT,
                    score_route REAL,
                    score_penalty REAL,
                    score_composed REAL,
                    global_infractions_json TEXT,
                    global_meta_json TEXT,
                    parsed_at TIMESTAMP
                )
                """
            )

            # Parsed Leaderboard records per route (one row per RouteScenario_*)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_route_results (
                    job_id INTEGER NOT NULL,
                    record_index INTEGER NOT NULL,
                    route_id TEXT,
                    status TEXT,
                    score_route REAL,
                    score_penalty REAL,
                    score_composed REAL,
                    infractions_json TEXT,
                    meta_json TEXT,
                    PRIMARY KEY (job_id, record_index)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_route_results_job_id ON job_route_results(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_route_results_route_id ON job_route_results(route_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_route_results_status ON job_route_results(status)")


    # --- Results parsing / logging ---

    @staticmethod
    def _safe_filename(s: str) -> str:
        s = str(s)
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or "item"

    def _make_checkpoint_path(self, job: sqlite3.Row) -> Path:
        ckpt_dir = self.state_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        route_stem = Path(str(job["route"])).stem
        fname = "job_{id}_{agent}_w{weather}_town{town}_{route}.json".format(
            id=int(job["id"]),
            agent=self._safe_filename(job["agent"]),
            weather=int(job["weather"]),
            town=self._safe_filename(str(job["town"] or "")),
            route=self._safe_filename(route_stem),
        )
        return ckpt_dir / fname

    @staticmethod
    def _json_dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, sort_keys=False)

    @staticmethod
    def _summarize_infractions(infractions: Any) -> Dict[str, Any]:
        # Records store infractions as lists of strings; global_record stores floats.
        if not isinstance(infractions, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in infractions.items():
            if isinstance(v, list):
                out[k] = len(v)
            else:
                out[k] = v
        return out

    def _parse_and_store_checkpoint(self, job_row: sqlite3.Row, checkpoint_path: Path) -> Optional[str]:
        if not checkpoint_path.exists():
            return None

        try:
            payload = json.loads(checkpoint_path.read_text())
        except Exception:
            return None

        ckpt = payload.get("_checkpoint") if isinstance(payload, dict) else None
        if not isinstance(ckpt, dict):
            return None

        progress = ckpt.get("progress") or [None, None]
        progress_current = None
        progress_total = None
        try:
            progress_current = int(progress[0])
            progress_total = int(progress[1])
        except Exception:
            progress_current, progress_total = None, None

        global_record = ckpt.get("global_record") or {}
        if not isinstance(global_record, dict):
            global_record = {}

        scores = global_record.get("scores") or {}
        infractions = global_record.get("infractions") or {}
        meta = global_record.get("meta") or {}
        status = global_record.get("status")

        def fget(d, k):
            try:
                return float(d.get(k))
            except Exception:
                return None

        score_route = fget(scores, "score_route")
        score_penalty = fget(scores, "score_penalty")
        score_composed = fget(scores, "score_composed")

        parsed_at = datetime.utcnow().isoformat()

        # Store summary + per-route records
        records = ckpt.get("records") or []
        if not isinstance(records, list):
            records = []

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO job_results (
                    job_id, checkpoint_path, progress_current, progress_total,
                    global_status, score_route, score_penalty, score_composed,
                    global_infractions_json, global_meta_json, parsed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(job_row["id"]),
                    str(checkpoint_path),
                    progress_current,
                    progress_total,
                    str(status) if status is not None else None,
                    score_route,
                    score_penalty,
                    score_composed,
                    self._json_dumps(infractions if isinstance(infractions, dict) else {}),
                    self._json_dumps(meta if isinstance(meta, dict) else {}),
                    parsed_at,
                ),
            )

            # Replace route records for this job_id
            conn.execute("DELETE FROM job_route_results WHERE job_id=?", (int(job_row["id"]),))

            to_insert = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                rid = rec.get("route_id")
                rstatus = rec.get("status")
                rscores = rec.get("scores") or {}
                rinfractions = rec.get("infractions") or {}
                rmeta = rec.get("meta") or {}
                idx = rec.get("index")
                try:
                    idx_i = int(idx)
                except Exception:
                    continue

                to_insert.append(
                    (
                        int(job_row["id"]),
                        idx_i,
                        str(rid) if rid is not None else None,
                        str(rstatus) if rstatus is not None else None,
                        fget(rscores, "score_route"),
                        fget(rscores, "score_penalty"),
                        fget(rscores, "score_composed"),
                        self._json_dumps(rinfractions if isinstance(rinfractions, dict) else {}),
                        self._json_dumps(rmeta if isinstance(rmeta, dict) else {}),
                    )
                )
            if to_insert:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO job_route_results (
                        job_id, record_index, route_id, status,
                        score_route, score_penalty, score_composed,
                        infractions_json, meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    to_insert,
                )

        return str(status) if status is not None else None
    
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
            _write_health_file(
                self.state_dir,
                node_name,
                gpu_id,
                {
                    "status": "idle",
                    "message": "no jobs pending",
                },
            )
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

        # Deterministic dataset directory (used for coverage/mosaic figures)
        route_stem = Path(str(routes_file)).stem
        run_tag = f"job{int(job_data['id'])}_gpu{gpu_id}"
        dataset_root = (
            os.environ.get('HPC_CARLA_DATASET_ROOT')
            or os.environ.get('DATASET_DIR')
            or (str(self.project_root / 'dataset') if (self.project_root / 'dataset').is_dir() else None)
        )
        dataset_dir = None
        if dataset_root:
            dataset_dir = str(Path(dataset_root) / str(job_data['agent']) / str(job_data['weather']) / f"{route_stem}_{run_tag}")

        # Emit health + run event (start)
        _write_health_file(
            self.state_dir,
            node_name,
            gpu_id,
            {
                "status": "running_job",
                "message": f"running job {int(job_data['id'])}",
                "current_job_id": int(job_data['id']),
                "rpc_port": int(port),
                "tm_port": int(tm_port),
            },
        )
        _append_run_event(
            self.state_dir,
            {
                "event": "job_start",
                "node": node_name,
                "gpu_id": gpu_id,
                "job_id": int(job_data['id']),
                "agent": job_data['agent'],
                "route": job_data['route'],
                "town": job_data['town'],
                "weather": int(job_data['weather']),
                "rpc_port": int(port),
                "tm_port": int(tm_port),
                "checkpoint_path": str(self._make_checkpoint_path(job_data)),
                "dataset_dir": dataset_dir,
            },
        )

        checkpoint_path = self._make_checkpoint_path(job_data)
        # Ensure directory exists (it should, but be safe).
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
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
                CHECKPOINT=str(checkpoint_path),
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
                '--trafficManagerPort', str(tm_port),
                '--checkpoint', str(checkpoint_path),
            ]
            if extra_args:
                cmd.extend(extra_args)
        
        env = os.environ.copy()
        env['WEATHER_INDEX'] = str(job_data['weather'])
        # Used by ConsolidatedAgent for dataset path + metadata.
        env.setdefault('WEATHER', str(job_data['weather']))
        env.setdefault('WEATHERS', str(job_data['weather']))
        env.setdefault('ROUTES', str(routes_file))
        env.setdefault('CHECKPOINT_PATH', str(checkpoint_path))

        # Make dataset path deterministic and job-traceable for paper plots.
        env.setdefault('HPC_CARLA_RUN_ID', _get_run_id())
        env.setdefault('HPC_CARLA_JOB_ID', str(int(job_data['id'])))
        env.setdefault('HPC_CARLA_RUN_TAG', run_tag)
        env.setdefault('HPC_CARLA_AGENT_NAME', str(job_data['agent']))

        # Prefer writing all collected outputs under the dedicated dataset dir.
        # docker-compose sets DATASET_DIR=/workspace/dataset.
        if 'DATASET_DIR' in env and env['DATASET_DIR']:
            env.setdefault('HPC_CARLA_DATASET_ROOT', env['DATASET_DIR'])

        # Ensure critical run/job metadata makes it into Singularity/Apptainer containers.
        # Singularity reliably forwards variables prefixed with SINGULARITYENV_.
        # Apptainer uses APPTAINERENV_.
        passthrough_keys = [
            'HPC_CARLA_RUN_ID',
            'HPC_CARLA_JOB_ID',
            'HPC_CARLA_RUN_TAG',
            'HPC_CARLA_AGENT_NAME',
        ]
        for k in passthrough_keys:
            v = env.get(k)
            if v is None:
                continue
            env.setdefault(f'SINGULARITYENV_{k}', str(v))
            env.setdefault(f'APPTAINERENV_{k}', str(v))

        # Make dataset root inside container deterministic and container-visible.
        # The repo is always bound at /workspace in start_job.sh.
        env.setdefault('SINGULARITYENV_HPC_CARLA_DATASET_ROOT', '/workspace/dataset')
        env.setdefault('APPTAINERENV_HPC_CARLA_DATASET_ROOT', '/workspace/dataset')
        env.setdefault('SINGULARITYENV_DATASET_DIR', '/workspace/dataset')
        env.setdefault('APPTAINERENV_DATASET_DIR', '/workspace/dataset')

        start_ts = time.time()
        try:
            evaluator_rc = subprocess.call(cmd, env=env)
        except KeyboardInterrupt:
            evaluator_rc = 130
        duration = time.time() - start_ts

        parsed_checkpoint_status: Optional[str] = None
        try:
            parsed_checkpoint_status = self._parse_and_store_checkpoint(job_data, checkpoint_path)
        except Exception:
            parsed_checkpoint_status = None

        def _checkpoint_success(status: Optional[str]) -> bool:
            if status is None:
                return False
            s = str(status).strip().lower()
            return s in {"completed", "success", "succeeded", "finished", "passed"}

        if evaluator_rc != 0:
            final_status = "failed"
        else:
            # Some evaluator paths print exceptions but still exit 0; treat missing/failed checkpoints as failure.
            if not checkpoint_path.exists():
                final_status = "failed"
            elif not _checkpoint_success(parsed_checkpoint_status):
                final_status = "failed"
            else:
                final_status = "completed"

        # Return code for the worker loop: 0 only for success, 1 for failure, 2 for idle.
        manager_rc = 0 if final_status == "completed" else (130 if evaluator_rc == 130 else 1)

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
            if manager_rc == 0:
                key = f"{job_data['agent']}_{job_data['route']}"
                row = conn.execute("SELECT estimate FROM runtime_estimates WHERE key=?", (key,)).fetchone()
                if row and row[0] is not None:
                    new_est = (float(row[0]) * 0.7) + (duration * 0.3)
                    conn.execute("UPDATE runtime_estimates SET estimate=? WHERE key=?", (new_est, key))
                else:
                    conn.execute("INSERT INTO runtime_estimates (key, estimate) VALUES (?, ?)", (key, duration))

        # Best-effort attach per-run dataset summary.
        run_summary = None
        if dataset_dir:
            try:
                p = Path(dataset_dir) / 'run_summary.json'
                if p.exists():
                    run_summary = json.loads(p.read_text())
            except Exception:
                run_summary = None

        # Emit run event (end) + update health.
        _append_run_event(
            self.state_dir,
            {
                "event": "job_end",
                "node": node_name,
                "gpu_id": gpu_id,
                "job_id": int(job_data['id']),
                "agent": job_data['agent'],
                "route": job_data['route'],
                "town": job_data['town'],
                "weather": int(job_data['weather']),
                "rpc_port": int(port),
                "tm_port": int(tm_port),
                "rc": int(manager_rc),
                "evaluator_rc": int(evaluator_rc),
                "duration_sec": float(duration),
                "final_status": final_status,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_status": parsed_checkpoint_status,
                "dataset_dir": dataset_dir,
                "run_summary": run_summary,
            },
        )
        _write_health_file(
            self.state_dir,
            node_name,
            gpu_id,
            {
                "status": "idle" if manager_rc == 0 else "error",
                "message": "job completed" if manager_rc == 0 else f"job failed rc={manager_rc}",
                "current_job_id": None,
                "rpc_port": int(port),
                "tm_port": int(tm_port),
            },
        )

        return manager_rc

    # --- Reporting ---
    def get_status(self):
        d = self.get_status_dict()
        stats = d.get("jobs", {})
        print("\nCOLLECTION STATUS (SQLite)")
        print("="*40)
        print(f"Total:      {stats.get('total', 0)}")
        print(f"Completed:  {stats.get('completed', 0)}")
        print(f"Pending:    {stats.get('pending', 0)}")
        print(f"Running:    {stats.get('running', 0)}")
        print(f"Failed:     {stats.get('failed', 0)}")
        print("-" * 40)

        print("Active GPUs:")
        for gpu in d.get("gpus", []):
            job_str = f"Job #{gpu.get('current_job_id')}" if gpu.get('current_job_id') else ""
            print(
                f"  GPU {gpu.get('gpu_id')}: {str(gpu.get('status','')).upper():<6} | "
                f"{gpu.get('jobs_completed', 0)} done | {job_str}"
            )

    def get_status_dict(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            job_stats = conn.execute(
                "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
            ).fetchall()
            stats = {row['status']: int(row['count']) for row in job_stats}
            total = int(sum(stats.values()))
            stats_out = {
                "total": total,
                "completed": int(stats.get('completed', 0)),
                "pending": int(stats.get('pending', 0)),
                "running": int(stats.get('running', 0)),
                "failed": int(stats.get('failed', 0)),
                "cancelled": int(stats.get('cancelled', 0)),
            }
            gpus = conn.execute(
                "SELECT node, gpu_id, status, current_job_id, jobs_completed, total_runtime, last_heartbeat FROM gpu_status ORDER BY node, gpu_id"
            ).fetchall()
            gpus_out = []
            for r in gpus:
                gpus_out.append({k: r[k] for k in r.keys()})
            return {
                "ts": _utc_now_iso(),
                "run_id": _get_run_id(),
                "jobs": stats_out,
                "gpus": gpus_out,
            }

    def export_results(self, output_file: str):
        with self._get_conn() as conn:
            job_rows = conn.execute("SELECT * FROM jobs ORDER BY id ASC").fetchall()
            jobs = [dict(r) for r in job_rows]

            # Attach parsed leaderboard summary when available
            results_rows = conn.execute("SELECT * FROM job_results").fetchall()
            results_by_job = {int(r["job_id"]): dict(r) for r in results_rows}

            # Route-level summaries (small) for export
            route_rows = conn.execute(
                """
                SELECT job_id, record_index, route_id, status,
                       score_route, score_penalty, score_composed,
                       infractions_json, meta_json
                FROM job_route_results
                ORDER BY job_id ASC, record_index ASC
                """
            ).fetchall()
            route_by_job: Dict[int, List[Dict[str, Any]]] = {}
            for r in route_rows:
                job_id = int(r["job_id"])
                infra = {}
                meta = {}
                try:
                    infra = json.loads(r["infractions_json"] or "{}")
                except Exception:
                    infra = {}
                try:
                    meta = json.loads(r["meta_json"] or "{}")
                except Exception:
                    meta = {}

                route_by_job.setdefault(job_id, []).append(
                    {
                        "record_index": int(r["record_index"]),
                        "route_id": r["route_id"],
                        "status": r["status"],
                        "scores": {
                            "score_route": r["score_route"],
                            "score_penalty": r["score_penalty"],
                            "score_composed": r["score_composed"],
                        },
                        "infractions": self._summarize_infractions(infra),
                        "meta": meta,
                    }
                )

            runs: List[Dict[str, Any]] = []
            index = {
                "by_agent": {},
                "by_weather": {},
                "by_town": {},
                "by_route": {},
                "by_agent_weather_town_route": {},
            }

            status_counts: Dict[str, int] = {}
            for job in jobs:
                job_id = int(job.get("id"))
                agent = str(job.get("agent"))
                weather = str(job.get("weather"))
                town = str(job.get("town") or "")
                route = str(job.get("route"))
                status = str(job.get("status"))
                status_counts[status] = status_counts.get(status, 0) + 1

                def add_idx(bucket: Dict[str, List[int]], key: str):
                    bucket.setdefault(key, []).append(job_id)

                add_idx(index["by_agent"], agent)
                add_idx(index["by_weather"], weather)
                add_idx(index["by_town"], town)
                add_idx(index["by_route"], route)
                combo_key = f"{agent}|{weather}|{town}|{route}"
                # In case of retries/duplicates, store a list.
                prev = index["by_agent_weather_town_route"].get(combo_key)
                if prev is None:
                    index["by_agent_weather_town_route"][combo_key] = [job_id]
                else:
                    prev.append(job_id)

                jr = results_by_job.get(job_id)
                leaderboard = None
                if jr:
                    try:
                        leaderboard = {
                            "checkpoint_path": jr.get("checkpoint_path"),
                            "progress": [jr.get("progress_current"), jr.get("progress_total")],
                            "global_status": jr.get("global_status"),
                            "scores": {
                                "score_route": jr.get("score_route"),
                                "score_penalty": jr.get("score_penalty"),
                                "score_composed": jr.get("score_composed"),
                            },
                            "global_infractions": json.loads(jr.get("global_infractions_json") or "{}"),
                            "global_meta": json.loads(jr.get("global_meta_json") or "{}"),
                            "parsed_at": jr.get("parsed_at"),
                        }
                    except Exception:
                        leaderboard = {
                            "checkpoint_path": jr.get("checkpoint_path"),
                            "progress": [jr.get("progress_current"), jr.get("progress_total")],
                            "global_status": jr.get("global_status"),
                        }

                runs.append(
                    {
                        "job": job,
                        "leaderboard": leaderboard,
                        "route_records": route_by_job.get(job_id, []),
                    }
                )

            data = {
                "schema_version": 2,
                "exported_at": datetime.now().isoformat(),
                "summary": {
                    "total_jobs": len(jobs),
                    "status_counts": status_counts,
                },
                "index": index,
                "runs": runs,

                # Backwards-ish compatibility with older exports.
                "jobs": jobs,
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
    # Optional JSON output for scripting (coordinator, plotting)
    # Keep as a flag on the status command to preserve backwards compatibility.
    status_parser = subparsers.choices.get('status')
    if status_parser is not None:
        status_parser.add_argument('--json', action='store_true', help='Print machine-readable JSON')
    
    add = subparsers.add_parser('add')
    add.add_argument('agent')
    add.add_argument('--weather', nargs='+', type=int)
    add.add_argument('--routes', nargs='+')
    
    subparsers.add_parser('retry')
    subparsers.add_parser('analyze')
    
    exp = subparsers.add_parser('export')
    exp.add_argument('output', nargs='?', default='results.json')
    exp.add_argument('--output', dest='output_flag', default=None)

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
        if getattr(args, 'json', False):
            print(json.dumps(manager.get_status_dict(), ensure_ascii=False))
        else:
            manager.get_status()
    elif args.command == 'add':
        manager.add_jobs(args.agent, args.weather, args.routes)
    elif args.command == 'retry':
        manager.retry_failed()
    elif args.command == 'export':
        out = args.output_flag or args.output
        manager.export_results(out)
    elif args.command == 'analyze':
        manager.analyze()

if __name__ == '__main__':
    main()
