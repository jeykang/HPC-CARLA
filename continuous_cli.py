#!/usr/bin/env python3
"""
Unified command-line interface for continuous data collection system
Supports both original (restart-per-job) and persistent CARLA modes
Includes dynamic SLURM configuration
"""

import os
import sys
import json
import subprocess
import signal
import time
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

class SLURMConfig:
    """SLURM job configuration"""
    def __init__(self):
        self.job_name = None
        self.nodes = None
        self.nodelist = None
        self.exclude = None
        self.gpus_per_node = None
        self.gres = None
        self.time = None
        self.partition = None
        self.qos = None
        self.account = None
        self.mail_type = None
        self.mail_user = None
        self.mem = None
        self.cpus_per_task = None
        self.exclusive = True
        self.output = None
        self.error = None
        self.array = None
        self.dependency = None
        self.extra_directives = []
    
    def to_directives(self):
        """Convert configuration to SBATCH directives"""
        directives = []
        
        if self.job_name:
            directives.append(f"#SBATCH --job-name={self.job_name}")
        if self.nodes:
            directives.append(f"#SBATCH --nodes={self.nodes}")
        if self.nodelist:
            directives.append(f"#SBATCH --nodelist={self.nodelist}")
        if self.exclude:
            directives.append(f"#SBATCH --exclude={self.exclude}")
        if self.gpus_per_node:
            directives.append(f"#SBATCH --gpus-per-node={self.gpus_per_node}")
        elif self.gres:
            directives.append(f"#SBATCH --gres={self.gres}")
        if self.time:
            directives.append(f"#SBATCH --time={self.time}")
        if self.partition:
            directives.append(f"#SBATCH --partition={self.partition}")
        if self.qos:
            directives.append(f"#SBATCH --qos={self.qos}")
        if self.account:
            directives.append(f"#SBATCH --account={self.account}")
        if self.mail_type:
            directives.append(f"#SBATCH --mail-type={self.mail_type}")
        if self.mail_user:
            directives.append(f"#SBATCH --mail-user={self.mail_user}")
        if self.mem:
            directives.append(f"#SBATCH --mem={self.mem}")
        if self.cpus_per_task:
            directives.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.exclusive:
            directives.append("#SBATCH --exclusive")
        if self.output:
            directives.append(f"#SBATCH --output={self.output}")
        if self.error:
            directives.append(f"#SBATCH --error={self.error}")
        if self.array:
            directives.append(f"#SBATCH --array={self.array}")
        if self.dependency:
            directives.append(f"#SBATCH --dependency={self.dependency}")
        
        # Add any extra custom directives
        for directive in self.extra_directives:
            if not directive.startswith("#SBATCH"):
                directive = f"#SBATCH {directive}"
            directives.append(directive)
        
        return directives
    
    @classmethod
    def from_args(cls, args):
        """Create SLURMConfig from argparse arguments"""
        config = cls()
        
        # Map CLI arguments to config attributes
        if hasattr(args, 'slurm_nodes') and args.slurm_nodes:
            config.nodes = args.slurm_nodes
        if hasattr(args, 'slurm_nodelist') and args.slurm_nodelist:
            config.nodelist = args.slurm_nodelist
        if hasattr(args, 'slurm_exclude') and args.slurm_exclude:
            config.exclude = args.slurm_exclude
        if hasattr(args, 'slurm_gpus') and args.slurm_gpus:
            config.gpus_per_node = args.slurm_gpus
        if hasattr(args, 'slurm_time') and args.slurm_time:
            config.time = args.slurm_time
        if hasattr(args, 'slurm_partition') and args.slurm_partition:
            config.partition = args.slurm_partition
        if hasattr(args, 'slurm_qos') and args.slurm_qos:
            config.qos = args.slurm_qos
        if hasattr(args, 'slurm_account') and args.slurm_account:
            config.account = args.slurm_account
        if hasattr(args, 'slurm_mem') and args.slurm_mem:
            config.mem = args.slurm_mem
        if hasattr(args, 'slurm_cpus') and args.slurm_cpus:
            config.cpus_per_task = args.slurm_cpus
        if hasattr(args, 'slurm_exclusive') and not args.slurm_exclusive:
            config.exclusive = False
        if hasattr(args, 'slurm_mail_type') and args.slurm_mail_type:
            config.mail_type = args.slurm_mail_type
        if hasattr(args, 'slurm_mail_user') and args.slurm_mail_user:
            config.mail_user = args.slurm_mail_user
        if hasattr(args, 'slurm_extra') and args.slurm_extra:
            config.extra_directives = args.slurm_extra
        
        return config


class ContinuousCLI:
    def __init__(self, project_root: str = None, persistent_mode: bool = False):
        self.project_root = Path(project_root or os.environ.get('PROJECT_ROOT', os.getcwd()))
        self.state_dir = self.project_root / 'collection_state'
        self.log_dir = self.project_root / 'logs'
        self.dataset_dir = self.project_root / 'dataset'
        self.persistent_mode = persistent_mode
        
        # Set environment variables for child processes
        os.environ['PROJECT_ROOT'] = str(self.project_root)
        os.environ['STATE_DIR'] = str(self.state_dir)
        os.environ['LOG_DIR'] = str(self.log_dir)
        os.environ['DATASET_DIR'] = str(self.dataset_dir)
        
        # Script paths - choose based on mode
        if self.persistent_mode:
            self.coordinator_script = self.project_root / 'continuous_collection_persistent.sh'
            self.monitor_script = self.project_root / 'carla_health_manager.py'
            self.worker_script = self.project_root / 'persistent_carla_worker.sh'
        else:
            self.coordinator_script = self.project_root / 'continuous_collection.sh'
            # Original mode has no dedicated monitor script; monitor() loops `status` instead.
            self.monitor_script = None
            self.worker_script = None
        
        self.manager_script = self.project_root / 'manage_continuous.py'
        
        # Create state directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.persistent_mode:
            (self.state_dir / 'health').mkdir(parents=True, exist_ok=True)
    
    def check_prerequisites(self) -> bool:
        """Check if required files exist for the selected mode"""
        required_files = [
            'carla_official.sif',
            'leaderboard/team_code/consolidated_agent.py',
            self.manager_script
        ]
        
        if self.persistent_mode:
            required_files.extend([
                'continuous_collection_persistent.sh',
                'persistent_carla_worker.sh',
                'carla_health_manager.py'
            ])
        else:
            required_files.extend([
                'continuous_collection.sh',
                'generate_single_job.sh'
            ])
        
        missing = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing.append(file)
        
        if missing:
            print(f"❌ Missing required files for {'persistent' if self.persistent_mode else 'original'} mode:")
            for file in missing:
                print(f"  - {file}")
            return False
        
        print(f"✓ All prerequisites found for {'persistent' if self.persistent_mode else 'original'} mode")
        return True
    
    def generate_slurm_script(self, slurm_config: SLURMConfig, 
                        agents: list = None, weather: list = None, 
                        routes: list = None) -> Path:
        """Generate a dynamic SLURM submission script."""
        
        # Set defaults based on mode
        mode_str = "persistent" if self.persistent_mode else "original"
        
        if not slurm_config.job_name:
            slurm_config.job_name = f"carla_collection_{mode_str}"
        
        if not slurm_config.output:
            slurm_config.output = f"{self.log_dir}/collection_{mode_str}_%A_%N.out"
        
        if not slurm_config.error:
            slurm_config.error = f"{self.log_dir}/collection_{mode_str}_%A_%N.err"
        
        # Default time if not specified
        if not slurm_config.time:
            slurm_config.time = "72:00:00"
        
        # Default GPUs if not specified
        if not slurm_config.gpus_per_node and not slurm_config.gres:
            slurm_config.gres = "gpu:8"
        
        # Calculate GPU counts for environment setup
        if slurm_config.gpus_per_node:
            gpus_per_node = slurm_config.gpus_per_node
        elif slurm_config.gres:
            # Extract number from gres (e.g., "gpu:8" -> 8)
            import re
            match = re.search(r'gpu:(\d+)', slurm_config.gres)
            gpus_per_node = int(match.group(1)) if match else 8
        else:
            gpus_per_node = 8
        
        # Calculate number of nodes and total GPUs
        num_nodes = 1
        if slurm_config.nodes:
            # Parse nodes specification (could be "2", "2-4", etc.)
            nodes_str = str(slurm_config.nodes)
            if '-' in nodes_str:
                # Range specified, use minimum for calculation
                num_nodes = int(nodes_str.split('-')[0])
            else:
                num_nodes = int(nodes_str)
        elif slurm_config.nodelist:
            # Count nodes in nodelist
            import re
            # Parse nodelist like "node[01-04]" or "node01,node02"
            if ',' in slurm_config.nodelist:
                num_nodes = len(slurm_config.nodelist.split(','))
            elif '[' in slurm_config.nodelist and ']' in slurm_config.nodelist:
                # Extract range
                match = re.search(r'\[(\d+)-(\d+)\]', slurm_config.nodelist)
                if match:
                    start, end = int(match.group(1)), int(match.group(2))
                    num_nodes = end - start + 1
                else:
                    # Single node in brackets
                    num_nodes = 1
            else:
                # Single node
                num_nodes = 1
        
        # Calculate total GPUs
        total_gpus = gpus_per_node * num_nodes
        
        print(f"Configuring for {num_nodes} nodes with {gpus_per_node} GPUs per node = {total_gpus} total GPUs")
        
        # Read the base coordinator script
        with open(self.coordinator_script, 'r') as f:
            base_script = f.read()
        
        # Extract the main logic (skip the SBATCH directives)
        script_lines = base_script.split('\n')
        main_logic_start = 0
        for i, line in enumerate(script_lines):
            if not line.startswith('#SBATCH') and not line.startswith('#!/bin/bash'):
                if line.strip() and not line.startswith('#'):
                    main_logic_start = i
                    break
        
        main_logic = '\n'.join(script_lines[main_logic_start:])
        
        # Build the new script
        script_parts = [
            "#!/bin/bash",
            "# Dynamically generated SLURM submission script",
            f"# Generated at: {datetime.now().isoformat()}",
            f"# Mode: {mode_str}",
            ""
        ]
        
        # Add SLURM directives
        script_parts.extend(slurm_config.to_directives())
        script_parts.append("")
        
        # Add environment setup with proper multi-node support
        script_parts.extend([
            "# Environment setup",
            f"export PROJECT_ROOT={self.project_root}",
            f"export STATE_DIR={self.state_dir}",
            f"export LOG_DIR={self.log_dir}",
            f"export DATASET_DIR={self.dataset_dir}",
            "",
            "# Multi-node GPU configuration",
            f"export GPUS_PER_NODE={gpus_per_node}",
            f"export NUM_NODES={num_nodes}",
            f"export NUM_GPUS={total_gpus}  # Total across all nodes",
            "",
            "# Detect if running under SLURM and adjust if needed",
            'if [ -n "$SLURM_JOB_ID" ]; then',
            '    # Get actual values from SLURM environment',
            '    if [ -n "$SLURM_NNODES" ]; then',
            '        export NUM_NODES=$SLURM_NNODES',
            '    fi',
            '    if [ -n "$SLURM_GPUS_PER_NODE" ]; then',
            '        # Parse SLURM GPU string (e.g., "gpu:8" or just "8")',
            '        export GPUS_PER_NODE=$(echo $SLURM_GPUS_PER_NODE | grep -oE "[0-9]+$")',
            '    fi',
            '    # Recalculate total GPUs based on actual SLURM allocation',
            '    export NUM_GPUS=$((GPUS_PER_NODE * NUM_NODES))',
            '    echo "SLURM allocated: $NUM_NODES nodes × $GPUS_PER_NODE GPUs/node = $NUM_GPUS total GPUs"',
            'fi',
            "",
            "export BASE_RPC_PORT=${BASE_RPC_PORT:-2000}",
            "export BASE_TM_PORT=${BASE_TM_PORT:-8000}",
            ""
        ])

        # Forward optional runtime knobs from the submit environment so every
        # node's workers see them. sbatch --export=ALL already propagates the
        # submit env, but baking the values in is robust to --export changes and
        # documents what the run was configured with. (Port bases are computed
        # per-node below, so they are intentionally not forwarded here.)
        _passthrough = [
            'JOB_TIMEOUT_SEC', 'AGENT_GPU_PIN', 'AGENT_GPU_OFFSET',
            'DEAD_SERVER_BACKOFF_SEC', 'CARLA_SIF', 'RUN_SEED',
            # CARLA server render/quality/logging knobs (segfault debugging).
            'CARLA_RENDER_FLAG', 'CARLA_QUALITY', 'CARLA_UE4_STDOUT',
            # Server boot-hardening: retry/park + parked-worker backoff + stagger.
            'CARLA_BOOT_ATTEMPTS', 'CARLA_BOOT_TIMEOUT_SEC', 'PARK_RETRY_SEC',
            'SERVER_STAGGER_SEC',
        ]
        _fwd = [f'export {k}="{os.environ[k]}"' for k in _passthrough if os.environ.get(k)]
        if _fwd:
            script_parts.append("# Forwarded runtime knobs (from launcher env)")
            script_parts.extend(_fwd)
            script_parts.append("")

        # Add agent/weather/route configuration if specified
        if agents:
            script_parts.append(f"export AGENTS=\"{' '.join(agents)}\"")
        if weather:
            script_parts.append(f"export WEATHER_INDICES=\"{' '.join(map(str, weather))}\"")
        if routes:
            script_parts.append(f"export ROUTE_FILES=\"{' '.join(routes)}\"")
        
        script_parts.append("")
        
        # Add node information logging
        script_parts.extend([
            "# Log node information",
            'echo "=========================================="',
            'echo "SLURM JOB INFORMATION"',
            'echo "=========================================="',
            'echo "Job ID: $SLURM_JOB_ID"',
            'echo "Node list: $SLURM_JOB_NODELIST"',
            'echo "Node name: $SLURMD_NODENAME"',
            'echo "Node ID: ${SLURM_NODEID:-0}"',
            'echo "Number of nodes: $NUM_NODES"',
            'echo "Partition: $SLURM_JOB_PARTITION"',
            'echo "GPUs per node: $GPUS_PER_NODE"',
            'echo "Total GPUs: $NUM_GPUS"',
            'echo "=========================================="',
            'echo ""',
            "",
            '# For multi-node coordination',
            'export NODE_ID=${SLURM_NODEID:-0}',
            'export NODE_NAME=${SLURMD_NODENAME:-$(hostname)}',
            'export IS_MASTER=$([[ ${NODE_ID} -eq 0 ]] && echo "true" || echo "false")',
            "",
            '# Adjust port ranges per node to avoid conflicts',
            '# Each node gets a different port range',
            'export BASE_RPC_PORT=$((2000 + NODE_ID * 1000))',
            'export BASE_TM_PORT=$((8000 + NODE_ID * 1000))',
            'echo "This node will use RPC ports ${BASE_RPC_PORT}-$((BASE_RPC_PORT + GPUS_PER_NODE * 10))"',
            "",
        ])
        
        # For persistent mode, need to handle multi-node CARLA servers
        if self.persistent_mode:
            script_parts.extend([
                '# Multi-node persistent CARLA server management',
                'if [ "$IS_MASTER" == "true" ]; then',
                '    echo "Master node: Will coordinate multi-node setup"',
                'else',
                '    echo "Worker node $NODE_ID: Waiting for master initialization"',
                '    sleep 10',
                'fi',
                "",
            ])
        
        # Add the main logic
        # If multi-node, launch one coordinator per node via srun; else inline
        if int(num_nodes) > 1:
            script_parts.extend([
                f'export COORDINATOR_SCRIPT="{self.coordinator_script}"',
                'echo "Launching per-node coordinators with srun..."',
                'srun --nodes=$NUM_NODES --ntasks-per-node=1 bash -c \'',
                '  set -e',
                '  export NODE_ID=${SLURM_NODEID:-0}',
                '  export NODE_NAME=${SLURMD_NODENAME:-$(hostname)}',
                '  export GPU_ID_OFFSET=$(( ${SLURM_NODEID:-0} * ${GPUS_PER_NODE} ))',
                '  export TOTAL_GPUS=${NUM_GPUS}',               # keep global for scheduler files                
                '  export NUM_GPUS=${GPUS_PER_NODE}',
                '  export LOCAL_GPUS=${GPUS_PER_NODE}',        # local count on this node
                '  export BASE_RPC_PORT=$((2000 + ${SLURM_NODEID:-0} * 1000))',
                '  export BASE_TM_PORT=$((8000 + ${SLURM_NODEID:-0} * 1000))',
                '  bash "$COORDINATOR_SCRIPT"',
                '\'',
                'wait',
                ''
            ])
        else:
            script_parts.append(main_logic)
        
        # Create temporary script file
        script_content = '\n'.join(script_parts)
        
        # Save to a persistent location for debugging/review
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        script_path = self.log_dir / f"slurm_submission_{mode_str}_{timestamp}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        return script_path
    
    def _setup_eval_env(self):
        """Set up the container env the evaluator needs (formerly start_job.sh).

        Exports the Singularity/Apptainer binds (project tree -> /workspace plus
        the libnvidia-gpucomp workaround), context vars, and EVAL_CMD_TEMPLATE.
        These propagate to the SLURM job via sbatch --export=ALL and on to the
        per-GPU workers, so `manage_continuous.py run` launches the leaderboard
        evaluator inside the container with /workspace bound and the GL stack
        usable. Idempotent; safe to call repeatedly.
        """
        sif = os.environ.get('CARLA_SIF', str(self.project_root / 'carla_official.sif'))
        os.environ['CARLA_SIF'] = sif
        if not Path(sif).is_file():
            print(f"[start][WARN] CARLA image not found: {sif}")

        # Purge stale Python bytecode so the container recompiles agent code from
        # the bind-mounted source this run. The container's Python did not
        # reliably invalidate __pycache__ across the bind mount, so edits to
        # leaderboard/team_code could silently have no effect (old .pyc ran).
        # With PYTHONDONTWRITEBYTECODE=1 in EVAL_CMD_TEMPLATE, no stale .pyc
        # re-accumulates after this one-time purge.
        import shutil
        _purged = 0
        for base in ('leaderboard', 'scenario_runner'):
            for pc in (self.project_root / base).rglob('__pycache__'):
                shutil.rmtree(pc, ignore_errors=True); _purged += 1
        if _purged:
            print(f"[start] purged {_purged} stale __pycache__ dir(s)")

        gpucomp = "/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.575.57.08"
        binds = [f"{self.project_root}:/workspace", f"{gpucomp}:{gpucomp}"]
        for var in ('SINGULARITY_BINDPATH', 'APPTAINER_BINDPATH'):
            cur = [b for b in os.environ.get(var, '').split(',') if b]
            for b in binds:
                if b not in cur:
                    cur.append(b)
            os.environ[var] = ','.join(cur)

        for prefix in ('SINGULARITYENV_', 'APPTAINERENV_'):
            os.environ[prefix + 'PROJECT_ROOT'] = str(self.project_root)
            os.environ[prefix + 'CARLA_SIF'] = sif

        # The {{...}} are escaped braces (literal shell ${...}); {NAME} are
        # str.format placeholders filled per-job by manage_continuous.run_next_job
        # (which supplies CHECKPOINT so each job writes its own results.json).
        os.environ['EVAL_CMD_TEMPLATE'] = (
            'singularity exec --nv --pwd /workspace \\\n'
            f'  -B {gpucomp}:{gpucomp} \\\n'
            '  "${{CARLA_SIF}}" bash -lc \'\n'
            '  set -euo pipefail\n'
            '  export PYTHONPATH="/workspace:/workspace/leaderboard:/workspace/scenario_runner:${{PYTHONPATH:-}}"\n'
            '  export PYTHONFAULTHANDLER=1   # dump a Python stack on fatal signals (SIGSEGV) to the worker log\n'
            '  export PYTHONDONTWRITEBYTECODE=1  # never write .pyc: bound /workspace source edits must take effect (stale __pycache__ silently ran old agent code)\n'
            '  python3 -m leaderboard.leaderboard_evaluator \\\n'
            '    --routes "{ROUTES_FILE}" \\\n'
            '    --scenarios "{SCENARIOS_FILE}" \\\n'
            '    --agent "{AGENT_CODE}" \\\n'
            '    --agent-config "{AGENT_CFG}" \\\n'
            '    --checkpoint "{CHECKPOINT}" \\\n'
            '    --trafficManagerSeed "{TM_SEED}" --carlaProviderSeed "{PROVIDER_SEED}" \\\n'
            '    --host "{HOST}" --port "{PORT}" --trafficManagerPort "{TM_PORT}"\n'
            "'"
        )

    def start(self, use_slurm: bool = None, agents: list = None,
             weather: list = None, routes: list = None,
             slurm_config: SLURMConfig = None):
        """Start continuous collection"""
        # Set up the container/eval environment (replaces start_job.sh).
        self._setup_eval_env()

        # Check prerequisites
        if not self.check_prerequisites():
            print("\nPlease ensure all required files are in place.")
            print("For persistent mode, you need the persistent CARLA scripts.")
            print("For original mode, you need the standard collection scripts.")
            return False
        
        # Initialize queue if needed
        if not (self.state_dir / 'job_queue.json').exists():
            print("Initializing job queue...")
            self.reset(agents, weather, routes)
        
        # Auto-detect SLURM if not specified
        if use_slurm is None:
            use_slurm = subprocess.run(['which', 'sbatch'], 
                                     capture_output=True).returncode == 0
        
        mode_str = "persistent CARLA" if self.persistent_mode else "original"
        
        if use_slurm:
            print(f"Starting {mode_str} collection with SLURM...")
            
            # Use dynamic script if SLURM config provided
            if slurm_config and any([
                slurm_config.nodes, slurm_config.nodelist, 
                slurm_config.gpus_per_node, slurm_config.time,
                slurm_config.partition, slurm_config.extra_directives
            ]):
                print("Generating dynamic SLURM script with custom configuration...")
                script_to_submit = self.generate_slurm_script(slurm_config, agents, weather, routes)
                print(f"Generated script: {script_to_submit}")
                
                # Show configuration
                print("\nSLURM Configuration:")
                for directive in slurm_config.to_directives():
                    print(f"  {directive}")
                print()
            else:
                # Use existing script
                script_to_submit = self.coordinator_script
                
                # Make scripts executable for persistent mode
                if self.persistent_mode:
                    for script in [self.coordinator_script, self.worker_script]:
                        if script and script.exists():
                            script.chmod(0o755)
            
            # Submit the job
            result = subprocess.run(['sbatch', str(script_to_submit)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"✓ Job submitted: {job_id}")
                print(f"✓ Mode: {mode_str}")
                print(f"Monitor with: python continuous_cli.py monitor{' --persistent' if self.persistent_mode else ''}")
                
                # Save job ID for tracking
                job_file = self.state_dir / 'current_slurm_job.txt'
                job_file.write_text(job_id)
                
                return True
            else:
                print(f"✗ Failed to submit job: {result.stderr}")
                return False
        else:
            print(f"Starting {mode_str} collection locally...")
            log_file = self.log_dir / f'continuous_{datetime.now():%Y%m%d_%H%M%S}.log'
            
            # Set environment for number of GPUs
            if slurm_config and slurm_config.gpus_per_node:
                os.environ['NUM_GPUS'] = str(slurm_config.gpus_per_node)
            
            with open(log_file, 'w') as log:
                process = subprocess.Popen(['bash', str(self.coordinator_script)],
                                         stdout=log, stderr=subprocess.STDOUT)
                print(f"✓ Started with PID: {process.pid}")
                print(f"✓ Mode: {mode_str}")
                print(f"✓ Log file: {log_file}")
                print(f"Monitor with: python continuous_cli.py monitor{' --persistent' if self.persistent_mode else ''}")
                
                # Save PID for later
                pid_file = self.state_dir / 'coordinator.pid'
                pid_file.write_text(str(process.pid))
        
        return True
    
    def stop(self, force: bool = False):
        """Stop continuous collection"""
        pid_file = self.state_dir / 'coordinator.pid'
        job_file = self.state_dir / 'current_slurm_job.txt'
        
        # Try local PID file first
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text())
                if force:
                    os.kill(pid, signal.SIGKILL)
                    print(f"✓ Force stopped process {pid}")
                else:
                    os.kill(pid, signal.SIGTERM)
                    print(f"✓ Sent stop signal to process {pid}")
                    print("Jobs will complete gracefully...")
                pid_file.unlink()
            except ProcessLookupError:
                print("Process not running")
                pid_file.unlink()
            except Exception as e:
                print(f"Error stopping process: {e}")
                return False
        
        # Try saved SLURM job ID
        if job_file.exists():
            try:
                job_id = job_file.read_text().strip()
                subprocess.run(['scancel', job_id])
                print(f"✓ Cancelled SLURM job {job_id}")
                job_file.unlink()
            except Exception as e:
                print(f"Error cancelling SLURM job: {e}")
        
        # Try to find and stop SLURM job by name
        job_name = 'carla_collection_persistent' if self.persistent_mode else 'carla_collection_original'
        result = subprocess.run(['squeue', '-n', job_name, '-h', '-o', '%A'],
                              capture_output=True, text=True)
        if result.stdout.strip():
            job_ids = result.stdout.strip().split('\n')
            for job_id in job_ids:
                subprocess.run(['scancel', job_id])
                print(f"✓ Cancelled SLURM job {job_id}")
        
        # For persistent mode, clean up CARLA instances
        if self.persistent_mode and force:
            print("Cleaning up persistent CARLA instances...")
            self.cleanup_carla()
        
        return True
    
    def cleanup_carla(self):
        """Clean up persistent CARLA instances (persistent mode only)"""
        if not self.persistent_mode:
            print("Cleanup is only for persistent mode")
            return
        
        # Try to use health manager if available
        health_manager = self.project_root / 'carla_health_manager.py'
        if health_manager.exists():
            subprocess.run(['python3', str(health_manager), 'cleanup'])
        
        # Clean up health files
        health_dir = self.state_dir / 'health'
        if health_dir.exists():
            for health_file in health_dir.glob('*.json'):
                health_file.unlink()
            print("✓ Cleaned up health files")
    
    def monitor(self, once: bool = False):
        """Launch monitoring dashboard"""
        if self.persistent_mode:
            # Use health manager for persistent mode
            cmd = ['python3', str(self.monitor_script)]
            if once:
                cmd.append('status')
            else:
                cmd.extend(['monitor', '--interval', '30'])
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
            return

        # Original mode: loop manage_continuous.py status (no dedicated monitor binary).
        try:
            if once:
                subprocess.run(['python3', str(self.manager_script), 'status'])
            else:
                while True:
                    os.system('clear')
                    subprocess.run(['python3', str(self.manager_script), 'status'])
                    time.sleep(30)
        except KeyboardInterrupt:
            pass
    
    def status(self):
        """Show current status"""
        subprocess.run(['python3', str(self.manager_script), 'status'])
        
        # Show current SLURM job if exists
        job_file = self.state_dir / 'current_slurm_job.txt'
        if job_file.exists():
            job_id = job_file.read_text().strip()
            print(f"\nCurrent SLURM job: {job_id}")
            
            # Get job details including node info
            result = subprocess.run(['scontrol', 'show', 'job', job_id],
                                capture_output=True, text=True)
            if result.returncode == 0:
                # Parse and show key details
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['JobState=', 'RunTime=', 'TimeLimit=', 'NodeList=']):
                        print(f"  {line.strip()}")
        
        # For persistent mode, also show health status with nodes
        if self.persistent_mode:
            print("\n" + "="*40)
            print("CARLA INSTANCE HEALTH:")
            print("="*40)
            health_manager = self.project_root / 'carla_health_manager.py'
            if health_manager.exists():
                subprocess.run(['python3', str(health_manager), 'status'])
    
    def reset(self, agents: list = None, weather: list = None, routes: list = None,
              smoke: bool = False, limit: int = None):
        """Reset job queue"""
        cmd = ['python3', str(self.manager_script), 'reset']
        if agents:
            cmd.extend(['--agents'] + agents)
        if weather:
            cmd.extend(['--weather'] + [str(w) for w in weather])
        if routes:
            cmd.extend(['--routes'] + routes)
        if smoke:
            cmd.append('--smoke')
        if limit:
            cmd.extend(['--limit', str(limit)])

        subprocess.run(cmd)

    def reclaim(self, stale_hours: float = 6.0):
        """Recover jobs stuck in 'running' after a worker crash."""
        subprocess.run(['python3', str(self.manager_script), 'reclaim',
                        '--stale-hours', str(stale_hours)])

    def prune(self, dry_run: bool = False):
        """Drop pending jobs made redundant by harder completed ones."""
        cmd = ['python3', str(self.manager_script), 'prune']
        if dry_run:
            cmd.append('--dry-run')
        subprocess.run(cmd)

    def validate_config(self, config: str = None, all_: bool = False):
        """Validate/introspect an agent pipeline YAML (delegates to tools/validate_config.py)."""
        tool = self.project_root / 'tools' / 'validate_config.py'
        cmd = ['python3', str(tool)]
        if all_ or not config:
            cmd.append('--all')
        else:
            cmd.append(config)
        return subprocess.run(cmd).returncode

    def new_agent(self, name: str, force: bool = False):
        """Scaffold a starter pipeline config at configs/<name>.yaml.

        Reuses an existing config's sensors (so they're valid) and a trivial
        FixedControl pipeline, giving a runnable skeleton to extend. Validates
        the result before writing.
        """
        import yaml
        cfg_dir = self.project_root / 'leaderboard' / 'team_code' / 'configs'
        out = cfg_dir / f'{name}.yaml'
        if out.exists() and not force:
            print(f"✗ {out} already exists (use --force to overwrite)")
            return 1
        template = cfg_dir / 'tcp.yaml'
        sensors = []
        if template.exists():
            base = yaml.safe_load(open(template))
            for s in base.get('sensors', []):
                if s.get('id') in ('rgb', 'speed', 'imu', 'gps'):
                    sensors.append(s)
        scaffold = {
            'sensors': sensors or [{'type': 'sensor.speedometer', 'id': 'speed'}],
            'pipeline': [
                {'module': 'team_code.pipeline_modules', 'class': 'ExtractSpeed',
                 'args': {'sensor_id': 'speed', 'out_key': 'speed'}},
                {'module': 'team_code.pipeline_engine', 'class': 'FixedControl',
                 'args': {'throttle': 0.3, 'steer': 0.0, 'brake': 0.0}},
            ],
        }
        with open(out, 'w') as f:
            f.write(f"# Starter pipeline for agent '{name}'.\n"
                    f"# Extend the pipeline with stages from team_code.pipeline_modules;\n"
                    f"# see leaderboard/team_code/PIPELINE_MODULES.md and the tcp/lav/interfuser configs.\n"
                    f"# Validate with: python3 continuous_cli.py validate-config {out}\n\n")
            yaml.safe_dump(scaffold, f, sort_keys=False)
        print(f"✓ wrote {out}")
        return self.validate_config(str(out))
    
    def retry(self, max_attempts: int = 3):
        """Retry failed jobs"""
        subprocess.run(['python3', str(self.manager_script), 'retry',
                       '--max-attempts', str(max_attempts)])
    
    def add_jobs(self, agent: str, weather: list = None, routes: list = None):
        """Add new jobs"""
        cmd = ['python3', str(self.manager_script), 'add', agent]
        if weather:
            cmd.extend(['--weather'] + [str(w) for w in weather])
        if routes:
            cmd.extend(['--routes'] + routes)
        
        subprocess.run(cmd)
    
    def cancel(self, agent: str = None):
        """Cancel pending jobs"""
        cmd = ['python3', str(self.manager_script), 'cancel']
        if agent:
            cmd.extend(['--agent', agent])
        
        subprocess.run(cmd)
    
    def export(self, output: str = 'collection_results.json'):
        """Export results"""
        subprocess.run(['python3', str(self.manager_script), 'export',
                       '--output', output])
    
    def optimize(self):
        """Optimize runtime estimates"""
        subprocess.run(['python3', str(self.manager_script), 'optimize'])
    
    def analyze(self):
        """Show runtime analysis"""
        subprocess.run(['python3', str(self.manager_script), 'analyze'])
    
    def health(self, gpu_id: Optional[int] = None):
        """Show CARLA health status (persistent mode only)"""
        if not self.persistent_mode:
            print("Health monitoring is only available in persistent mode")
            print("Use --persistent flag to enable persistent CARLA mode")
            return
        
        health_manager = self.project_root / 'carla_health_manager.py'
        if not health_manager.exists():
            print("Health manager not found. Is persistent mode properly installed?")
            return
        
        if gpu_id is not None:
            subprocess.run(['python3', str(health_manager), 'log', str(gpu_id)])
        else:
            subprocess.run(['python3', str(health_manager), 'status'])
    
    def restart_gpu(self, gpu_id: int):
        """Restart a specific GPU worker (persistent mode only)"""
        if not self.persistent_mode:
            print("GPU restart is only available in persistent mode")
            return
        
        health_manager = self.project_root / 'carla_health_manager.py'
        if health_manager.exists():
            subprocess.run(['python3', str(health_manager), 'restart', str(gpu_id)])
        else:
            print("Health manager not found")
    
    def logs(self, tail: bool = False, job_id: Optional[int] = None, 
        gpu_id: Optional[int] = None):
        """View logs"""
        # Resolve a GPU's log file across naming schemes:
        #   continuous_collection.sh   -> node_<NODE>_gpu<id>_consolidated.log
        #   persistent_carla_worker.sh -> worker_<NODE>_gpu<id>.log
        def _find_gpu_log(gid):
            for pattern in (f'*gpu{gid}_consolidated.log', f'worker_*_gpu{gid}.log', f'*gpu{gid}*.log'):
                matches = sorted(self.log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    return matches[0]
            return None

        if job_id is not None:
            # For specific job, grep from consolidated log
            if gpu_id is not None:
                log = _find_gpu_log(gpu_id)
                if log is not None:
                    # Search for specific job in the consolidated log
                    import subprocess
                    result = subprocess.run(
                        ['grep', '-A', '50', f'Job #{job_id}', str(log)],
                        capture_output=True, text=True
                    )
                    if result.stdout:
                        print(f"\n=== Job {job_id} from GPU {gpu_id} log ({log.name}) ===")
                        print(result.stdout)
                    else:
                        print(f"Job {job_id} not found in GPU {gpu_id} log ({log.name})")
                else:
                    print(f"No consolidated log found for GPU {gpu_id}")
            else:
                print("Please specify GPU ID to search for job")

        elif gpu_id is not None:
            log = _find_gpu_log(gpu_id)
            if log is not None:
                if tail:
                    subprocess.run(['tail', '-f', str(log)])
                else:
                    # Show last portion of log
                    subprocess.run(['tail', '-n', '100', str(log)])
            else:
                print(f"No log found for GPU {gpu_id}")
        
        else:
            # Show summary of all GPU logs (covers both naming schemes)
            gpu_logs = sorted(self.log_dir.glob('*gpu*.log'))
            
            if gpu_logs:
                print("Available GPU logs:")
                for log in gpu_logs:
                    size = log.stat().st_size / (1024 * 1024)  # Size in MB
                    print(f"  {log.name}: {size:.1f} MB")
                print("\nUse --gpu <id> to view a specific GPU log")
            else:
                print("No GPU logs found")
    
    def summary(self):
        """Generate and display summary report"""
        try:
            # Load state files
            with open(self.state_dir / 'job_queue.json', 'r') as f:
                queue_data = json.load(f)
            with open(self.state_dir / 'completed_jobs.json', 'r') as f:
                completed_data = json.load(f)
            with open(self.state_dir / 'gpu_status.json', 'r') as f:
                gpu_status = json.load(f)
            
            print("\n" + "="*60)
            print(f"CONTINUOUS COLLECTION SUMMARY ({'Persistent' if self.persistent_mode else 'Original'} Mode)")
            print("="*60)

            # Overall statistics — derive from job statuses (the stored
            # 'total'/'completed' counters are never updated after reset).
            jobs = queue_data.get('jobs', [])
            total = len(jobs)
            def _n(*st):
                return sum(1 for j in jobs if j.get('status') in st)
            completed = _n('completed')
            failed = _n('failed')
            pending = _n('pending')
            running = _n('assigned', 'running')

            print(f"\nOverall Statistics:")
            print(f"  Total jobs: {total}")
            if total > 0:
                print(f"  Completed: {completed} ({100*completed/total:.1f}%)")
            else:
                print("  Completed: 0")
            print(f"  Failed:    {failed}")
            print(f"  Running:   {running}")
            print(f"  Pending:   {pending}")

            # Time statistics
            avg_time = None
            durations = [j['duration'] for j in completed_data.get('jobs', []) if j.get('duration')]
            if durations:
                total_time = sum(durations)
                avg_time = total_time / len(durations)
                print(f"\nTime Statistics:")
                print(f"  Total job runtime: {timedelta(seconds=int(total_time))}")
                print(f"  Average duration:  {timedelta(seconds=int(avg_time))}")
                print(f"  Min/Max:           {timedelta(seconds=int(min(durations)))} - "
                      f"{timedelta(seconds=int(max(durations)))}")

            # GPU statistics — gpu_status.json is the v2 namespaced format
            # {"nodes": {node: {gpu: {jobs_completed, total_runtime}}}};
            # fall back to a legacy flat {gpu: {...}} map.
            nodes = gpu_status.get('nodes') if isinstance(gpu_status, dict) else None
            if not isinstance(nodes, dict) or not nodes:
                flat = {k: v for k, v in (gpu_status or {}).items() if isinstance(v, dict) and 'jobs_completed' in v}
                nodes = {'(local)': flat} if flat else {}
            total_gpu_time = 0
            rows = []
            for node in sorted(nodes):
                for gpu_id in sorted(nodes[node], key=lambda x: int(x) if str(x).isdigit() else 0):
                    gi = nodes[node][gpu_id]
                    jobs_done = gi.get('jobs_completed', 0)
                    runtime = gi.get('total_runtime', 0)
                    total_gpu_time += runtime
                    if jobs_done > 0:
                        rows.append((node, gpu_id, jobs_done, runtime / jobs_done))
            if rows:
                print(f"\nGPU Performance:")
                for node, gpu_id, jobs_done, avg in rows:
                    print(f"  {node} GPU {gpu_id}: {jobs_done} jobs, avg {timedelta(seconds=int(avg))}")

            # Efficiency: useful GPU-time (sum of completed-job durations) over
            # total GPU busy-time. Only meaningful with both numbers present.
            if total_gpu_time > 0 and durations:
                efficiency = sum(durations) / total_gpu_time * 100
                print(f"\nEfficiency: {efficiency:.1f}% (completed-job time / total GPU busy-time)")

            print("="*60)
            
        except FileNotFoundError:
            print("No collection data found. Start a collection first.")
        except Exception as e:
            print(f"Error generating summary: {e}")
    
    def check_mode(self):
        """Check and display current mode settings"""
        print(f"\nCurrent Configuration:")
        print(f"  Mode: {'Persistent CARLA' if self.persistent_mode else 'Original (Restart per job)'}")
        print(f"  Project root: {self.project_root}")
        print(f"  Coordinator script: {self.coordinator_script.name}")
        
        if self.persistent_mode:
            print(f"\nPersistent Mode Features:")
            print(f"  ✓ One CARLA instance per GPU")
            print(f"  ✓ No restarts between jobs")
            print(f"  ✓ Health monitoring via shared files (heartbeats)")
            print(f"  ✓ Recovery via `reclaim` (stuck jobs) and `restart-gpu` (manual)")
        else:
            print(f"\nOriginal Mode Features:")
            print(f"  • CARLA restarts for each job")
            print(f"  • Simple architecture")
            print(f"  • No health monitoring needed")


def add_slurm_arguments(parser):
    """Add SLURM configuration arguments to a parser"""
    slurm_group = parser.add_argument_group('SLURM configuration')
    
    slurm_group.add_argument('--slurm-nodes', type=str,
                            help='Number of nodes (e.g., "2" or "2-4")')
    slurm_group.add_argument('--slurm-nodelist', type=str,
                            help='Specific nodes (e.g., "node[01-04]")')
    slurm_group.add_argument('--slurm-exclude', type=str,
                            help='Nodes to exclude')
    slurm_group.add_argument('--slurm-gpus', type=int,
                            help='GPUs per node (default: 8)')
    slurm_group.add_argument('--slurm-time', type=str,
                            help='Time limit (e.g., "72:00:00")')
    slurm_group.add_argument('--slurm-partition', type=str,
                            help='Partition/queue name')
    slurm_group.add_argument('--slurm-qos', type=str,
                            help='Quality of service')
    slurm_group.add_argument('--slurm-account', type=str,
                            help='Account for billing')
    slurm_group.add_argument('--slurm-mem', type=str,
                            help='Memory per node (e.g., "128G")')
    slurm_group.add_argument('--slurm-cpus', type=int,
                            help='CPUs per task')
    slurm_group.add_argument('--slurm-exclusive', action='store_false',
                            help='Disable exclusive node allocation')
    slurm_group.add_argument('--slurm-mail-type', type=str,
                            help='Mail notifications (e.g., "BEGIN,END,FAIL")')
    slurm_group.add_argument('--slurm-mail-user', type=str,
                            help='Email address for notifications')
    slurm_group.add_argument('--slurm-extra', nargs='+',
                            help='Extra SLURM directives')


def main():
    parser = argparse.ArgumentParser(
        description='Unified CLI for continuous data collection with dynamic SLURM support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with custom SLURM configuration
  python continuous_cli.py --persistent start --slurm \\
      --slurm-nodes 2 --slurm-gpus 4 --slurm-time 48:00:00
  
  # Use specific nodes
  python continuous_cli.py start --slurm \\
      --slurm-nodelist "node[01-02]" --slurm-partition gpu_large
  
  # Start with email notifications
  python continuous_cli.py --persistent start --slurm \\
      --slurm-mail-type "BEGIN,END,FAIL" --slurm-mail-user me@example.com
  
  # Custom memory and CPU allocation
  python continuous_cli.py start --slurm \\
      --slurm-mem 256G --slurm-cpus 32
  
  # Add extra SLURM directives
  python continuous_cli.py start --slurm \\
      --slurm-extra "--constraint=v100" "--gres-flags=enforce-binding"
        """
    )
    
    # Global options
    default_project_root = os.environ.get('PROJECT_ROOT', os.getcwd())
    parser.add_argument('--project-root', default=default_project_root,
                       help='Project root directory')
    parser.add_argument('--persistent', action='store_true',
                       help='Use persistent CARLA mode (one instance per GPU)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command with SLURM options
    start_parser = subparsers.add_parser('start', help='Start collection')
    start_parser.add_argument('--slurm', action='store_true', help='Use SLURM')
    start_parser.add_argument('--local', action='store_true', help='Run locally')
    start_parser.add_argument('--agents', nargs='+', help='Agents to include')
    start_parser.add_argument('--weather', nargs='+', type=int, help='Weather indices')
    start_parser.add_argument('--routes', nargs='+', help='Route files')
    # Queue control at start time (a reset runs only if one of these is given).
    start_parser.add_argument('--reset', action='store_true', help='Reset the queue before starting')
    start_parser.add_argument('--smoke', action='store_true', help='Reset to the tiny validation queue (implies --reset)')
    start_parser.add_argument('--limit', type=int, default=None, help='Cap the queue at N jobs on reset (implies --reset)')
    # Runtime knobs (forwarded to the workers on every node).
    start_parser.add_argument('--sif', help='Path to CARLA Singularity image (sets CARLA_SIF)')
    start_parser.add_argument('--job-timeout', type=int, help='Per-job wall-clock cap, seconds (JOB_TIMEOUT_SEC)')
    start_parser.add_argument('--agent-gpu-offset', type=int, help='Offset agent GPU from its CARLA GPU (AGENT_GPU_OFFSET; 0=co-locate)')
    start_parser.add_argument('--agent-gpu-pin', type=int, help='Force all agents onto one GPU (AGENT_GPU_PIN; benchmark)')
    start_parser.add_argument('--dead-server-backoff', type=int, help='Sleep seconds after skipping a dead server (DEAD_SERVER_BACKOFF_SEC)')
    start_parser.add_argument('--seed', type=int, help='Fixed RNG seed for scenario spawns + traffic manager (RUN_SEED; reproducible eval)')

    # Add SLURM configuration options to start command
    add_slurm_arguments(start_parser)
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop collection')
    stop_parser.add_argument('--force', action='store_true', help='Force stop')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor progress')
    monitor_parser.add_argument('--once', action='store_true', help='Show once and exit')
    
    # Status command
    subparsers.add_parser('status', help='Show status')
    
    # Check-mode command
    subparsers.add_parser('check-mode', help='Show current mode configuration')
    
    # Health command (persistent mode)
    health_parser = subparsers.add_parser('health', help='Show CARLA health (persistent mode)')
    health_parser.add_argument('gpu_id', nargs='?', type=int, help='GPU ID for detailed info')
    
    # Restart-gpu command (persistent mode)
    restart_parser = subparsers.add_parser('restart-gpu', help='Restart GPU worker (persistent mode)')
    restart_parser.add_argument('gpu_id', type=int, help='GPU ID to restart')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset queue')
    reset_parser.add_argument('--agents', nargs='+', help='Agents')
    reset_parser.add_argument('--weather', nargs='+', type=int, help='Weather')
    reset_parser.add_argument('--routes', nargs='+', help='Routes')
    reset_parser.add_argument('--smoke', action='store_true',
                              help='Tiny validation queue (single-route files + weather 0)')
    reset_parser.add_argument('--limit', type=int, default=None,
                              help='Cap total jobs, interleaved across agents')

    # Reclaim command (recover stuck 'running' jobs after a worker crash)
    reclaim_parser = subparsers.add_parser('reclaim', help='Recover jobs stuck running after a worker crash')
    reclaim_parser.add_argument('--stale-hours', type=float, default=6.0,
                                help='Hours before a running job is considered stale (default 6)')

    # Prune command (drop pending jobs made redundant by harder completed ones)
    prune_parser = subparsers.add_parser('prune', help='Drop pending jobs made redundant by harder completed ones')
    prune_parser.add_argument('--dry-run', action='store_true', help='Report without modifying the queue')

    # Validate-config command (offline pipeline validation + introspection)
    vc_parser = subparsers.add_parser('validate-config', help='Validate/introspect an agent pipeline YAML (no CARLA)')
    vc_parser.add_argument('config', nargs='?', help='Path to an agent config (default: all in configs/)')
    vc_parser.add_argument('--all', action='store_true', help='Validate every config')

    # New-agent command (scaffold a starter pipeline config)
    na_parser = subparsers.add_parser('new-agent', help='Scaffold a starter agent pipeline config')
    na_parser.add_argument('name', help='Agent name (creates configs/<name>.yaml)')
    na_parser.add_argument('--force', action='store_true', help='Overwrite if the config exists')
    
    # Retry command
    retry_parser = subparsers.add_parser('retry', help='Retry failed')
    retry_parser.add_argument('--max-attempts', type=int, default=3)
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add jobs')
    add_parser.add_argument('agent', help='Agent name')
    add_parser.add_argument('--weather', nargs='+', type=int)
    add_parser.add_argument('--routes', nargs='+')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel jobs')
    cancel_parser.add_argument('--agent', help='Specific agent')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export results')
    # Accept both `export results.json` (positional) and `export --output results.json`
    # (matches manage_continuous.py and the common convention).
    export_parser.add_argument('output', nargs='?', default='results.json')
    export_parser.add_argument('--output', '-o', dest='output_flag', default=None,
                               help='Output file (overrides the positional argument)')
    
    # Optimize command
    subparsers.add_parser('optimize', help='Optimize estimates')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Analyze runtimes')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='View logs')
    logs_parser.add_argument('--tail', '-f', action='store_true', help='Follow log')
    logs_parser.add_argument('--job', type=int, help='Job ID')
    logs_parser.add_argument('--gpu', type=int, help='GPU ID')
    
    # Summary command
    subparsers.add_parser('summary', help='Show summary')
    
    # Cleanup command (persistent mode)
    subparsers.add_parser('cleanup', help='Clean up CARLA instances (persistent mode)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI instance with mode setting
    cli = ContinuousCLI(args.project_root, args.persistent)
    
    # Execute commands
    if args.command == 'start':
        use_slurm = args.slurm if args.slurm or args.local else None
        if args.local:
            use_slurm = False

        # Apply runtime knobs / SIF to the env so they reach the workers.
        if getattr(args, 'sif', None):
            os.environ['CARLA_SIF'] = args.sif
        for attr, env_name in (('job_timeout', 'JOB_TIMEOUT_SEC'),
                               ('agent_gpu_offset', 'AGENT_GPU_OFFSET'),
                               ('agent_gpu_pin', 'AGENT_GPU_PIN'),
                               ('dead_server_backoff', 'DEAD_SERVER_BACKOFF_SEC'),
                               ('seed', 'RUN_SEED')):
            v = getattr(args, attr, None)
            if v is not None:
                os.environ[env_name] = str(v)

        # Reset the queue only if explicitly requested.
        if getattr(args, 'reset', False) or getattr(args, 'smoke', False) or getattr(args, 'limit', None):
            cli.reset(args.agents, args.weather, args.routes,
                      smoke=getattr(args, 'smoke', False), limit=getattr(args, 'limit', None))

        # Create SLURM config from arguments
        slurm_config = SLURMConfig.from_args(args) if use_slurm else None

        cli.start(use_slurm, args.agents, args.weather, args.routes, slurm_config)
    
    elif args.command == 'stop':
        cli.stop(args.force)
    
    elif args.command == 'monitor':
        cli.monitor(args.once)
    
    elif args.command == 'status':
        cli.status()
    
    elif args.command == 'check-mode':
        cli.check_mode()
    
    elif args.command == 'health':
        cli.health(args.gpu_id if 'gpu_id' in args else None)
    
    elif args.command == 'restart-gpu':
        cli.restart_gpu(args.gpu_id)
    
    elif args.command == 'reset':
        cli.reset(args.agents, args.weather, args.routes,
                  smoke=getattr(args, 'smoke', False),
                  limit=getattr(args, 'limit', None))

    elif args.command == 'reclaim':
        cli.reclaim(args.stale_hours)

    elif args.command == 'prune':
        cli.prune(getattr(args, 'dry_run', False))

    elif args.command == 'validate-config':
        sys.exit(cli.validate_config(getattr(args, 'config', None), getattr(args, 'all', False)))

    elif args.command == 'new-agent':
        sys.exit(cli.new_agent(args.name, getattr(args, 'force', False)))
    
    elif args.command == 'retry':
        cli.retry(args.max_attempts)
    
    elif args.command == 'add':
        cli.add_jobs(args.agent, args.weather, args.routes)
    
    elif args.command == 'cancel':
        cli.cancel(args.agent)
    
    elif args.command == 'export':
        cli.export(getattr(args, 'output_flag', None) or args.output)
    
    elif args.command == 'optimize':
        cli.optimize()
    
    elif args.command == 'analyze':
        cli.analyze()
    
    elif args.command == 'logs':
        cli.logs(args.tail, args.job, args.gpu)
    
    elif args.command == 'summary':
        cli.summary()
    
    elif args.command == 'cleanup':
        if not cli.persistent_mode:
            print("Cleanup is only for persistent mode. Use --persistent flag.")
        else:
            cli.cleanup_carla()


if __name__ == '__main__':
    main()