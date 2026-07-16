# HPC-CARLA Persistent

Distributed verification of autonomous-driving agents on a **SLURM + Singularity** cluster. A
persistent CARLA server per GPU continuously drains a queue of **agent × route × weather** jobs,
collecting per-route CARLA-Leaderboard driving scores (and, optionally, raw sensor data) across a
roster of driving agents.

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
| **CILRS** | Conditional Imitation Learning + ResNet (ICCV'19) | 1 camera | productive — a deliberately weak baseline ([§5.4](PAPER_REFERENCE.md)) |
| **NEAT** | Neural Attention Fields (ICCV'21) | 3 cameras | productive |
| **Roach** | RL-Coached imitation (ICCV'21) | 1 camera | productive |
| **LAV** | Learning from All Vehicles (CVPR'22) | 4 cameras + LiDAR | **server-blocked on A100** — crashes CARLA at `load_world`; may run on L40S |

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

### The metric is *per route*, not per file
Each route *file* is a **suite** of many short routes (a `_tiny` file holds 300–450), run in
sequence and **checkpointed per route** into `results.json`. On unstable hardware a server often
crashes mid-suite, so whole *files* rarely finish — but every route completed beforehand is saved,
even in jobs the queue marks `failed`. [`tools/harvest_results.py`](tools/harvest_results.py)
recovers those per-route scores from every job's checkpoint; on the current run **~89% of harvested
route-evals come from `failed` jobs.** The reportable unit is the **route-eval**, and
`per_route_results.csv` is the primary artifact.

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

## Current status (snapshot, 2026-07-07 — live; use `harvest_results.py` for current numbers)

Active run on A100 (`pod17`): **1,680-job** queue (5 productive agents × short/tiny routes × 21
weathers), draining continuously. Per-route harvest so far:

| Agent | route-evals | mean driving score |
|-------|------------:|-------------------:|
| roach | 78 | 86.2 |
| tcp | 48 | 83.8 |
| neat | 37 | 81.2 |
| interfuser | 16 | 67.3 |
| cilrs | 94 | 32.4 |

CILRS is low **by design** — audited end-to-end and confirmed genuine weak-baseline behaviour, not
an integration bug ([`PAPER_REFERENCE.md` §5.4](PAPER_REFERENCE.md)). Illumination coverage is being
backfilled by the coverage-aware scheduler so the per-model sensitivity axis becomes identifiable.

---

## Documentation map

- [`PAPER_REFERENCE.md`](PAPER_REFERENCE.md) — full architecture, per-agent detail, difficulty model, reliability.
- [`examples/CLUSTER_SETUP.md`](examples/CLUSTER_SETUP.md) — reproduce on your cluster (A100 ↔ L40S).
- [`leaderboard/team_code/PIPELINE_MODULES.md`](leaderboard/team_code/PIPELINE_MODULES.md) — pipeline stage catalogue + context-key contract (to add/modify an agent).
