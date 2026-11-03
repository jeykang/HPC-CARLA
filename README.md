# HPC-CARLA Persistent

Distributed verification of autonomous-driving agents on a **SLURM + Singularity** cluster. A
persistent CARLA server per GPU continuously drains a queue of **agent × route × weather** jobs,
collecting per-route CARLA-Leaderboard driving scores (and, optionally, raw sensor data) across a
roster of driving agents.

**Paper:** *HPC-CARLA: Closed-Loop Evaluation of Autonomous Driving Agents as a Resilient,
Schedulable HPC Workload* — accepted at WORKS 2026 (SC26 Workshops). J. Kang, S. Jung, K. Kim,
C. Gardner, J.-S. Yeom, T.-H. Ahn, J.-W. Kim.

The current focus is a **cross-hardware comparison (A100 ↔ L40S).** The A100 is a compute GPU with
a weak real-time rasteriser, so CARLA renders slowly and segfaults intermittently at GL-context
creation; the pipeline is hardened to *recover* from that rather than crash. A portable launcher
under [`examples/`](examples/) lets a contributor reproduce the same run on RTX-class hardware to
quantify the difference.

> **Deep reference:** [`PAPER_REFERENCE.md`](PAPER_REFERENCE.md) is the living technical reference —
> architecture, every agent, the difficulty model, and cluster reliability. **This README is the
> entry point; that file is the detail.**

---

## Agent roster

Every agent is a CARLA-Leaderboard method re-implemented as a **modular inference pipeline** — a
YAML list of composable [`pipeline_modules.py`](leaderboard/team_code/pipeline_modules.py) stages
driven by [`consolidated_agent.py`](leaderboard/team_code/consolidated_agent.py), not a monolithic
class. Sensors, model, and control are independently configurable without touching agent code, and a
generic `TorchModelRunner` loads any checkpointed model from `{module, class_name, args}`.

| Agent | Method | Sensors | Status |
|-------|--------|---------|--------|
| **TCP** | Trajectory-guided Control + PID (NeurIPS'22) | 1 camera | productive |
| **InterFuser** | Interpretable multi-sensor fusion transformer (CoRL'22) | 3 cameras + LiDAR | productive |
| **CILRS** | Conditional Imitation Learning + ResNet (ICCV'19) | 1 camera | productive ([integration audit, §5.4](PAPER_REFERENCE.md)) |
| **NEAT** | Neural Attention Fields (ICCV'21) | 3 cameras | productive |
| **Roach** | RL-Coached imitation (ICCV'21) | 1 camera | productive |
| **LAV** | Learning from All Vehicles (CVPR'22) | 4 cameras + LiDAR | **server-blocked on A100** — crashes CARLA at `load_world`; runs on L40S ([L40S run](#l40s-cross-hardware-run)) |

Four of the five productive agents are camera-only; **only InterFuser (and LAV) use LiDAR.** That
modality split drives the difficulty analysis below. Agent weights are gitignored and fetched on
demand by each agent's `fetch_weights.sh`.

---

## Repository layout

```
.
├── continuous_cli.py                 # single entry point: reset / start / monitor / summary / export
├── manage_continuous.py              # job queue + scheduler (difficulty + illumination coverage)
├── carla_server_manager.py           # per-GPU persistent CARLA server + segfault resilience
├── carla_health_manager.py           # out-of-band server health monitoring
├── continuous_collection_persistent.sh   # SLURM coordinator (one per allocation)
├── persistent_carla_worker.sh        # per-GPU worker (server + agent)
├── leaderboard/team_code/
│   ├── consolidated_agent.py         # universal agent: runs a YAML pipeline each tick
│   ├── pipeline_modules.py           # reusable pipeline stage classes (~2500 lines)
│   ├── configs/{tcp,interfuser,cilrs,neat,roach,lav}.yaml
│   └── {tcp,interfuser,cilrs,neat,roach,lav}/   # per-agent model code + fetch_weights.sh
├── tools/                            # analysis (see below): harvest_results, difficulty_validation,
│                                     #   sensitivity_matrix, weather_axes, verification_report, ...
├── examples/                         # portable cross-cluster launcher + CLUSTER_SETUP.md
├── collection_state/                 # live job queue, completed_jobs.json, metrics
├── dataset/                          # per-route results.json (+ optional per-frame sensor data)
├── PAPER_REFERENCE.md                # the deep technical reference
└── genfig.py                         # figure generation
```

---

## How it works

### Persistent servers + segfault resilience
Each GPU runs one long-lived CARLA server that workers connect to, eliminating the ~60 s per-job
startup. On the A100, GL-context creation segfaults intermittently (a host driver/rasteriser issue,
not a code bug); `carla_server_manager.py` makes this **recoverable**: it kills the stale server,
health-checks the reboot (port open *and* process still alive), retries, and **parks** a GPU that
cannot boot (so it stops burning the queue) while periodically re-attempting. Per-GPU `HOME`
isolation prevents shader-cache contention. See [`PAPER_REFERENCE.md` §9](PAPER_REFERENCE.md).

### How route evaluations are saved
A job runs one agent through one route file in one weather preset. The routes run in sequence.
For example, the eight `_tiny` files contain 117–442 routes each. The evaluator checkpoints each
route's outcome and scores in `results.json`. If the job stops partway through the file, whether
because of a time limit or an error, the records already written can still be harvested.

[`tools/harvest_results.py`](tools/harvest_results.py) collects these records in
`per_route_results.csv`, one row per saved route evaluation. A row can describe a completed drive,
a driving failure, or a simulator crash. The queue records the outcome of the enclosing job separately.

### Scheduling — job-first, hardest-first, coverage-aware
`manage_continuous.py` sorts pending jobs by
`(attempts, coverage_deficit, coverage_count, −difficulty, agent, −est)`:

- **Job-first / agent-interleaved.** Difficulty is *agent-independent* (`route + scenario +
  weather`), so all agents' jobs for a `(route, weather)` tie and the `agent` field interleaves them
  — the queue drains as `[all agents × condition 1], [all agents × condition 2], …`, so an early
  cutoff leaves **every agent with balanced coverage on the same conditions**. (This replaced an old
  hard-coded agent priority that drained one agent first and starved the others.)
- **Hardest-first.** Within a tier the hardest `(route+scenario+weather)` runs first, so `prune` can
  drop the easier same-route variants as redundant.
- **Illumination-stratified coverage.** Pure hardest-first collapses the completed sample onto the
  darkest+rainiest presets (night). `COVERAGE_QUOTA` (env, default 3) guarantees that many finished
  jobs per `(agent, illumination-bin ∈ {noon, sunset, night})` before reverting to hardest-first;
  `COVERAGE_QUOTA=0` restores the original sort exactly.

### Difficulty and per-model sensitivity
Each job gets a scalar difficulty (route geometry + scenario density + weather) used for scheduling
and pruning. But a *single scalar* washes out against performance because different architectures
fail on different axes. So difficulty is also analysed as a **vector**:

- [`tools/weather_axes.py`](tools/weather_axes.py) decomposes the 0–20 weather ordinal into physical
  axes (`illum_dark / precip / road_water / cloud / fog`).
- [`tools/sensitivity_matrix.py`](tools/sensitivity_matrix.py) fits a **per-agent noisy-OR** failure
  model `P(fail)=1−exp(−Σ λⱼ·xⱼ)`, reporting each axis's hazard weight per agent with CIs — and
  flagging axes the current sample can't identify. Validated on synthetic ground truth
  ([`tools/noisy_or_sanity.py`](tools/noisy_or_sanity.py)): it recovers, e.g., that camera-only
  agents are illumination-sensitive while LiDAR agents are not.
- [`tools/difficulty_validation.py`](tools/difficulty_validation.py) correlates difficulty against
  per-route driving score.

---

## Usage

From the **login node** — the CLI generates and submits the SLURM job (do not `sbatch` by hand):

```bash
# 1. Build the queue (agent × route × weather). --smoke = tiny ~1h validation queue.
python3 continuous_cli.py reset            # or: reset --smoke

# 2. Launch persistent workers on SLURM (one CARLA server + one agent per GPU).
python3 continuous_cli.py --persistent start --slurm \
    --slurm-nodes 1 --slurm-gpus 8 --slurm-time 48:00:00 \
    --slurm-nodelist hpc-pr-a-pod17

# 3. Watch it (the run keeps going if you stop watching).
python3 continuous_cli.py --persistent monitor
```

The run is **resumable** (rerun `start` to reuse completed work) and self-balances across agents.

**After / during a run — read the results:**
```bash
python3 tools/harvest_results.py        # per-route scores (the real metric) + illumination coverage
python3 tools/verification_report.py    # per-agent pass-rate / score, by town/weather
python3 tools/classify_outcomes.py      # agent-result vs infra-failure taxonomy
python3 tools/sensitivity_matrix.py     # per-axis × per-agent difficulty sensitivity
```

**Running on a different cluster (L40S)?** See [`examples/CLUSTER_SETUP.md`](examples/CLUSTER_SETUP.md)
and the one-file launcher [`examples/run_cluster.sh`](examples/run_cluster.sh) — edit one config
block, then `setup → smoke → run → export`.

---

## Final campaign (paper numbers)

The GIST campaign ran on two nodes with eight A100 GPUs each. All 2,835 evaluation jobs
(5 agents × 27 route files × 21 weather presets) reached a terminal state in 108.6 hours.
The queue marked 511 jobs `completed` and 2,324 `failed`. **Of the 13,059 saved route evaluations,
12,316 (94%) came from jobs marked `failed`.** Each job can contribute several records, including records of unsuccessful drives.

Each agent had a saved record in all 168 town × weather combinations and a non-crash record in
126. In the 42 combinations from Town06 and Town10, all 525 records were simulator crashes.

| Agent | route-evals | mean driving score | mean route completion |
|-------|------------:|-------------------:|----------------------:|
| TCP | 2,515 | 66.8 | 80.8 |
| Roach | 3,304 | 61.8 | 87.5 |
| InterFuser | 1,238 | 56.6 | 71.8 |
| NEAT | 1,860 | 55.1 | 79.4 |
| CILRS | 4,142 | 24.6 | 54.1 |

Both means include the 105 Town06 and Town10 crash records per agent. Each of those records
contributes zero to both columns. Every agent was assigned the same 567 jobs, but the jobs
produced different sets of episode records, so the means cover different episodes.

CILRS had the lowest mean driving score. We checked channel order, normalization, command mapping,
brake gating, and checkpoint loading and found no integration error that explained its low score.
See the [CILRS audit in §5.4](PAPER_REFERENCE.md) for the implementation checks.

Full figures and tables: [`METRICS_REFRESH_RESULTS_2026-07-21.md`](METRICS_REFRESH_RESULTS_2026-07-21.md).

## L40S cross-hardware run

TCP, NEAT, and LAV also ran on SLU's Libra cluster. For TCP and NEAT, the two sites used the same
container, policy configuration, route files, weather presets, and random seed (`RUN_SEED=2000`).
Both the GPU and the driver differed between GIST and SLU, so their individual effects cannot be
separated. The SLU launch scripts set a one-hour limit per job for TCP and NEAT. On GIST, 359 A100
jobs recorded evaluator durations longer than 61 minutes.

| Agent | route-evals (L40S) |
|-------|-------------------:|
| TCP | 5,930 |
| NEAT | 3,027 |
| LAV | 1,255 |

The table compares driving scores for episodes matched on agent, route file, route ID, weather,
and seed. Both full matched sets include Town01–Town07 and Town10.

| Population | episodes | mean change (L40S − A100) | identical scores |
|------------|---------:|--------------------------:|-----------------:|
| TCP, all matched | 2,505 | +15.9 | 59% |
| TCP, Towns 01–05 † | 2,014 | −0.05 | 73% |
| NEAT, all matched | 1,516 | −0.7 | 70% |

† The TCP comparison restricted to Towns 01–05 was added after the camera-ready. TCP's overall
score shift was concentrated in Town07 (a suspected scoring anomaly) and Town06/Town10, where
all A100 records were simulator crashes.

None of TCP's 104 matched A100 simulator crashes recurred on L40S. For NEAT, all 105 recurred.
Those 105 pairs scored zero on both stacks and count toward NEAT's 70% score agreement.

In the four-route illumination comparison, NEAT's mean L40S driving scores at noon, sunset,
and night were 75.9, 69.2, and 71.7, respectively. Its noon-to-night change was −4.2 points on
L40S and −27.0 on A100. LAV's L40S means were 32.3, 33.7, and 29.5 (−2.8 points). A separate
model of episode failure fitted to the full campaigns gave NEAT a darkness weight of 0.58 on
A100 and 0.50 on L40S.

Watchdog stall restarts on L40S: LAV 388, TCP 1, NEAT 0.

---

## Documentation map

- [`PAPER_REFERENCE.md`](PAPER_REFERENCE.md) — full architecture, per-agent detail, difficulty model, reliability.
- [`examples/CLUSTER_SETUP.md`](examples/CLUSTER_SETUP.md) — reproduce on your cluster (A100 ↔ L40S).
- [`leaderboard/team_code/PIPELINE_MODULES.md`](leaderboard/team_code/PIPELINE_MODULES.md) — pipeline stage catalogue + context-key contract (to add/modify an agent).
