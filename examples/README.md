# Examples — getting started on the cluster

New here? This is the shortest path to a working run.

> **Running on a _different_ cluster (e.g. L40S, for the A100-vs-L40S comparison)?**
> Use the portable one-file launcher **[`run_cluster.sh`](run_cluster.sh)** with the
> **[cross-cluster setup guide](CLUSTER_SETUP.md)** — edit one config block, then
> `setup` → `smoke` → `run` → `export`. The rest of this page assumes our A100 nodes.

## The model in one paragraph
You don't write a job per agent/route. You build a **queue** of jobs
(agent × route × weather) once, then launch **persistent workers** — one CARLA
server + one agent per GPU — that drain the queue. `continuous_cli.py` is the
single entry point for building the queue, launching, and monitoring.

## Fastest path (recommended)
From the **login node** — the CLI generates and submits the SLURM job for you:
```bash
python3 continuous_cli.py reset --smoke            # tiny ~1h validation queue
python3 continuous_cli.py --persistent start --slurm \
    --slurm-nodes 1 --slurm-gpus 8 --slurm-time 08:00:00 \
    --slurm-nodelist hpc-pr-a-pod09
python3 continuous_cli.py monitor                  # watch progress
```
Drop `--smoke` (use a plain `reset`) for the full agent × route × weather sweep,
and raise `--slurm-nodes/--slurm-time` accordingly.

## Direct sbatch (to learn / customise)
[`example_slurm_job.sh`](example_slurm_job.sh) is a heavily-commented, single-node
job you submit yourself:
```bash
python3 continuous_cli.py reset --smoke
sbatch examples/example_slurm_job.sh
```
It does the same thing the CLI would generate, but you can read every line.

## After a run — read the results
```bash
python3 tools/classify_outcomes.py      # outcomes split into agent-result vs infra-failure
python3 tools/verification_report.py    # per-agent comparison (pass-rate, score, by town/weather)
```

## Useful knobs (env vars or CLI flags on `start`)
| knob | flag | meaning |
|---|---|---|
| `JOB_TIMEOUT_SEC` | `--job-timeout` | hard per-job wall-clock cap (default 4h) |
| `RUN_SEED` | `--seed` | fixed seed → identical scenarios across agents (reproducible) |
| `AGENT_GPU_OFFSET` | `--agent-gpu-offset` | 0 = co-locate agent with its CARLA GPU |
| `CARLA_SIF` | `--sif` | path to the CARLA Singularity image |

## Developing a new agent (no cluster needed)
```bash
python3 continuous_cli.py new-agent myagent                       # scaffold a config
python3 continuous_cli.py validate-config leaderboard/team_code/configs/myagent.yaml
python3 tests/test_pipeline_engine.py && python3 tests/test_config_schema.py
```
See [`../leaderboard/team_code/PIPELINE_MODULES.md`](../leaderboard/team_code/PIPELINE_MODULES.md)
for the module catalogue and the context-key contract.
