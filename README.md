Version of the entire system with:

- Persistent CARLA instances per-GPU
- Automatic, continuous scheduling of all possible agent-route-scenario combinations
- Per-instance healthchecks

Example usage:

`python3 continuous_cli.py reset` to regenerate agent-route-scenario combos

`python3 continuous_cli.py --persistent start --slurm --slurm-gpus 8 --slurm-time 96:00:00 --slurm-nodes 1 --slurm-nodelist hpc-pr-a-pod09` where everything after start defines various SLURM options

`python3 continuous_cli.py --persistent monitor` to use healthchecks to monitor status of each instance

Metrics outputs (for paper figures):

- Per-job summary: `collection_state/completed_jobs.json` (includes `duration_s`, `start_ts_unix`, `end_ts_unix`, `rc`, and best-effort `score` parsed from `results.json`)
- Job lifecycle events: `collection_state/metrics/events.jsonl` (`job_start`/`job_end`)
- Persistent CARLA startup timing: `collection_state/metrics/servers/<node>/carla_pool.jsonl` and `collection_state/metrics/servers/<node>/carla_pool_last.json`
- Node/GPU utilization + hardware snapshot:
  - `collection_state/metrics/node/<node>/gpu.jsonl`
  - `collection_state/metrics/node/<node>/system.jsonl`
  - `collection_state/metrics/node/<node>/static.json`

Figure generation helper:

`python3 genfig.py --state-root collection_state --dataset-root dataset --outdir paper_figures`
