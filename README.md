# HPC-CARLA Persistent

Scalable autonomous-driving data collection on an HPC cluster. Runs persistent CARLA servers
per GPU and continuously schedules agent–route–weather combinations as SLURM jobs, collecting
sensor data and leaderboard metrics across all agents.

---

## Overview

Three imitation-learning agents from the CARLA Leaderboard are reimplemented as modular
inference pipelines and run across every combination of:

- **Agents**: TCP, LAV, InterFuser
- **Routes**: all training routes (town01–07, town10; long/short/tiny variants)
- **Weather**: 15 CARLA weather presets (ClearNoon → HardRainNight)

Each agent has been ported from its original monolithic class into a YAML-driven pipeline of
composable `pipeline_modules.py` stages. This decoupling makes sensor wiring, model parameters,
and control logic independently configurable without touching agent code.

---

## Repository Structure

```
.
├── manage_continuous.py          # Job queue, scheduling, worker orchestration
├── continuous_cli.py             # CLI front-end (reset / start / monitor)
├── leaderboard/
│   ├── team_code/
│   │   ├── consolidated_agent.py # Universal agent wrapper (pipeline + legacy modes)
│   │   ├── pipeline_engine.py    # Runs the pipeline list each tick
│   │   ├── pipeline_modules.py   # All reusable pipeline stage classes (~2500 lines)
│   │   ├── configs/
│   │   │   ├── tcp.yaml          # TCP pipeline config
│   │   │   ├── lav.yaml          # LAV pipeline config
│   │   │   └── interfuser.yaml   # InterFuser pipeline config
│   │   ├── tcp/                  # TCP model code + checkpoint
│   │   ├── lav/                  # LAV model code + checkpoints
│   │   └── interfuser/           # InterFuser model code + checkpoints
│   └── data/
│       ├── training_routes/      # Route XML files
│       └── scenarios/            # Scenario JSON files (adversarial event configs)
├── collection_state/             # Live job queue, runtime estimates, metrics
├── dataset/                      # Collected sensor data (per agent/weather/map/route)
└── logs/                         # Per-GPU worker logs (worker_<node>_gpu<N>.log)
```

---

## Agent Implementation Status

### TCP — Trajectory-guided Control with PID
**Status: confirmed working.**

Implements the dual-path control strategy from [TCP (Chen et al., 2022)](https://github.com/OpenDriveLab/TCP):
a Beta-distribution policy head and a PID waypoint tracker are blended based on whether the
vehicle is turning.

**Key implementation notes:**
- `mu_branches` and `sigma_branches` from the model are Softplus outputs — they are the α and β
  Beta distribution parameters directly, not mean/sigma. `TCPBetaControl` implements
  `_get_action_beta` (mode formula) correctly.
- `TCPPIDControl` includes the three-angle outlier-rejection logic from `TCP.control_pid()`:
  the GPS-derived angle overrides the waypoint angle when waypoints are noisy on straight roads.
- Blend weights: straight = 30% Beta + 70% PID; turning = 70% Beta + 30% PID.

Config: `leaderboard/team_code/configs/tcp.yaml`

---

### LAV — Learning from All Vehicles
**Status: implementation complete; end-to-end confirmation in progress.**

Implements the full LAV perception and planning pipeline from
[LAV (Chen et al., 2022)](https://github.com/dotchen/LAV). The pipeline is:

```
RGB cameras → ERFNet segmentation
LiDAR → ego-body filter → point painting → temporal stacking (3 frames)
                                         ↓
                             PointPillarNet BEV detection
                                         ↓
                          UniPlanner trajectory (GRU, 6 commands)
                                         ↓
                         collision check → PID control → brake override
```

**Implementation bugs fixed during development:**

| Bug | Symptom | Fix |
|-----|---------|-----|
| `torch_scatter` missing from container | crash at import | replaced `scatter_max`/`scatter_mean` with pure PyTorch |
| `num_input: 11` in YAML | `state_dict` size mismatch | set to 16 (`raw(11) + decorate(5)`) |
| `ExtractLidarXYZ` dropped intensity column | `(N,15)×(16,64)` matmul crash | added `num_cols=4` to keep XYZI |
| `PointPillarNet.nx/ny` stored as float | `torch.zeros` TypeError | cast to `int` in `__init__` |
| `BEVHeatmapNMS` height filter operator precedence | silent false-positive vehicle detections | matched reference `and`/`or` precedence |

Config: `leaderboard/team_code/configs/lav.yaml`

---

### InterFuser — Interpretable Multi-sensor Fusion Transformer
**Status: implemented; not yet systematically validated.**

Implements the transformer-based multi-sensor fusion model from
[InterFuser (Shao et al., 2023)](https://github.com/opendilab/InterFuser). Takes multi-camera
RGB, LiDAR histogram, route target point, and speed as input; outputs waypoints, junction flag,
traffic light state, stop sign detection, and traffic object metadata.

The `InterfuserControllerModule` wraps the original safety-aware PID controller with traffic light
and stop-sign override logic.

Config: `leaderboard/team_code/configs/interfuser.yaml`

---

## Infrastructure

### Job Scheduling

`manage_continuous.py` maintains a JSON job queue and dispatches jobs across available GPU slots.
Jobs are sorted by:

1. **Fewest attempts first** — failed jobs are retried but not prioritised over fresh ones.
2. **Agent priority** — currently LAV first (debugging), then TCP, then InterFuser.
3. **Difficulty score** — harder scenarios run before easier ones (see below).
4. **Estimated runtime** — longer routes run first within the same difficulty tier.

### Difficulty-aware Scheduling

Each job is scored before scheduling:

- **Route geometry**: `sharp_turns×2 + path_length/500 + total_heading_change/180`
  — counts 45°+ waypoint-to-waypoint heading jumps as intersection proxies, rewards path length
  and overall road complexity.
- **Scenario density**: counts unique adversarial scenario trigger locations (vehicles, pedestrians,
  cyclists) within 25 m of the route, weighted by mean scenario-type difficulty. All scenario
  types share identical spawn locations per town; the weight reflects type severity
  (Scenario9 "sudden appearance" = 4.5 vs Scenario1 "slow vehicle" = 1.0).
- **Weather difficulty**: additive offset 0.0 (ClearNoon) to 5.5 (HardRainNight) based on the
  `_WEATHER_IDS` table in `consolidated_agent.py`.

This ensures that the most failure-revealing combinations run early, giving useful signal before
the cluster exhausts easier jobs.

### Persistent CARLA Servers

Each GPU runs a persistent CARLA server managed by `carla_server_manager.py`. Workers connect
to existing servers rather than starting new ones per job, eliminating the ~60 s startup overhead
per job. Server health is monitored separately by `carla_health_manager.py`.

### Data Collection

`ConsolidatedAgent` can operate in collection mode (`COLLECT_DATA=1`) alongside inference. When
enabled, it saves raw sensor data per-frame before running the agent, structured as:

```
dataset/<agent>/weather_<N>/map_<NN>/<route_name>/<sensor_id>/<frame>.npy
```

Run summaries (`run_summary.json`) record frames-saved-per-sensor and `global_steps` for each
completed run. A `global_steps: 0` result with exactly 1 frame per sensor indicates a pipeline
crash on the first inference step.

### Metrics

- Per-job summary: `collection_state/completed_jobs.json`
- Job lifecycle events: `collection_state/metrics/events.jsonl`
- CARLA server timing: `collection_state/metrics/servers/<node>/carla_pool.jsonl`
- GPU/system utilisation: `collection_state/metrics/node/<node>/gpu.jsonl`

Figure generation: `python3 genfig.py --state-root collection_state --dataset-root dataset --outdir paper_figures`

---

## Usage

### Initial setup

```bash
# Generate job queue for all agent/route/weather combinations
python3 continuous_cli.py reset

# Optional: target specific agents, routes, or weather indices
python3 continuous_cli.py reset --agents tcp lav --weather 0 1 2 5 10 14
```

### Starting workers (SLURM)

```bash
python3 continuous_cli.py --persistent start \
  --slurm \
  --slurm-gpus 8 \
  --slurm-time 96:00:00 \
  --slurm-nodes 2 \
  --slurm-nodelist hpc-pr-a-pod09,hpc-pr-a-pod17
```

### Monitoring

```bash
python3 continuous_cli.py --persistent monitor
```

### Adding jobs without resetting the queue

```bash
python3 continuous_cli.py add --agent tcp --weather 14 15 16
```

---

## What Remains

| Item | Status | Notes |
|------|--------|-------|
| Confirm LAV end-to-end | Done | Confirmed working — runs show motion without crashes |
| Validate InterFuser end-to-end | In progress | Set to priority 0; jobs now running |
| Redundant scenario pruning | Done | `python3 manage_continuous.py prune [--dry-run]` — skips pending easy-weather jobs when a harder same-route run already completed |
| Expand weather to night presets (indices 14–20) | Done | Default weather range changed to 0–20; next `reset` will include night conditions |
| Leaderboard score reporting | Done | `_finish()` now parses `results.json` and records `score_composed`, `score_route`, `score_n_routes`, `route_statuses` into `completed_jobs.json` |
| Data quality audit | Pending | Verify collected frames are free from silent pipeline bugs across all three agents |

---

## Module Reference

See [`leaderboard/team_code/PIPELINE_MODULES.md`](leaderboard/team_code/PIPELINE_MODULES.md) for
a full reference of all pipeline stage classes, their arguments, and the context keys they
read/write.
