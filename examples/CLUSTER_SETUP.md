# Running on your cluster — the A100 ↔ L40S comparison

This guide gets a contributor from a fresh clone to a comparable collection run on
**different GPU hardware**. The motivating case is **L40S vs A100**: the A100 is a
compute GPU with a weak real-time rasteriser, which makes CARLA render slowly and
segfault intermittently (see [`PAPER_REFERENCE.md`](../PAPER_REFERENCE.md) §9). The
L40S has an RTX-class rasteriser, so CARLA should render **faster and more stably**.
Running the *same* agents / routes / weathers on both lets us quantify that.

Everything below is driven by one script: [`run_cluster.sh`](run_cluster.sh).

---

## 1. What you need

- The repo, on a shared filesystem your compute nodes can read.
- **SLURM** + **Singularity/Apptainer** on the cluster.
- The **CARLA image** `carla_official.sif` (~6 GB). Either copy ours, or build it:
  ```bash
  singularity build carla_official.sif carla_official_u2004.def
  ```
- **Agent weights** — fetched automatically from public sources (S3 / W&B) by each
  agent's `fetch_weights.sh`; `setup` (below) does this for you.

You do **not** need to hand-write SLURM scripts or match our node names — the CLI
generates the job from your config block.

---

## 2. Configure (edit one block)

Open [`run_cluster.sh`](run_cluster.sh) and edit the marked `EDIT THIS BLOCK`
section for your cluster:

| Variable | What to set |
|---|---|
| `CLUSTER_NAME` | a label for your results, e.g. `l40s-siteX` |
| `SLURM_NODES` / `SLURM_GPUS_PER_NODE` | how many nodes × GPUs to grab |
| `SLURM_TIME` | walltime (e.g. `48:00:00`) |
| `SLURM_PARTITION` / `SLURM_ACCOUNT` / `SLURM_QOS` | your scheduler's requirements (leave `""` if not needed) |
| `SLURM_NODELIST` | pin specific nodes, or `""` to let SLURM choose |
| `CARLA_SIF` | path to the image if not at the repo root |

**Keep `AGENTS` and `ROUTES_KIND` at their defaults** (`tcp interfuser cilrs neat
roach`, `short_tiny`) — that's the A100 baseline, and equal inputs are what make the
comparison fair.

> **GPU driver:** you do **not** configure the driver. The server manager
> auto-detects `libnvidia-gpucomp.so.<version>` for whatever driver your nodes run
> (`carla_server_manager._find_gpucomp`), so L40S's driver is handled automatically.
>
> **`--gres` vs `--gpus-per-node`:** the CLI emits `--gpus-per-node=N`. If your SLURM
> only accepts `--gres=gpu:N`, add it via `--slurm-extra` in `_slurm_args`, or use the
> hand-written [`example_slurm_job.sh`](example_slurm_job.sh) as a template.

---

## 3. Setup check, then smoke, then run

```bash
./examples/run_cluster.sh setup    # verifies SIF + SLURM + singularity, fetches weights, validates configs
./examples/run_cluster.sh smoke    # tiny ~1h validation run — confirm CARLA renders & agents drive
./examples/run_cluster.sh monitor  # watch it (Ctrl-C to stop watching; the run keeps going)
```

Only once the smoke completes cleanly (a few routes per agent, no import/weight
errors) do the real run:

```bash
./examples/run_cluster.sh run
./examples/run_cluster.sh monitor
```

The run collects `agents × routes × weathers` on a shared queue drained by one
CARLA server + one agent per GPU. It's **resumable**: if the allocation ends, just
`run_cluster.sh run` again (it reuses completed work) or add nodes.

---

## 4. Export & compare

**What you're actually collecting.** Each route *file* is a **suite** of many short
routes (a `_tiny` file can hold 300–450), run in sequence and **checkpointed per
route**. On unstable hardware a server may crash after a handful of routes, so whole
*files* rarely finish — but every route completed before that is saved. **The metric
is per-route, not per-file.** `tools/harvest_results.py` pulls those per-route scores
out of every job's `results.json` (including crashed/timed-out files), so your real
yield is the **route-eval count** — far higher than the queue's "completed jobs"
number. Track that, not file-completions.

```bash
./examples/run_cluster.sh summary   # this cluster's numbers (route-evals first)
./examples/run_cluster.sh export    # writes results/<CLUSTER_NAME>/ — share this dir
```

The bundle's **primary artifact is `per_route_results.csv`** (one row per route-eval:
agent, route, weather, driving score, infractions). It also includes
`collection_results.json`, per-agent / outcome / sim-ratio reports, the
difficulty-vs-performance validation (computed at per-route granularity), and
`cluster_meta.txt` (GPU model, node count, config). Send that directory back for the
A100-vs-your-cluster comparison.

---

## 5. What to expect on L40S vs A100 (the interesting part)

| Dimension | A100 (baseline) | L40S (expected) |
|---|---|---|
| **CARLA server stability** | intermittent `Signal 11` at GL init; the resilience layer (kill/health-check/park) fires often | RTX rasteriser → should boot & render cleanly; parks/restarts rarely if ever |
| **Throughput** | sim ~8–12× slower than real-time; low route-evals/GPU-hr | should be markedly faster |
| **Routes completed per file** | a few before a server crash → whole files rarely finish; the harvester recovers the per-route data | a stable server can run whole suites → many more route-evals, and actual full-file completions |
| **`LAV`** | crashes the server on `load_world` → excluded | may run — add `lav` to `AGENTS` to test (a real comparison finding) |
| **Agent driving scores** | the baseline `score_composed` per agent/route/weather | should be *similar* (agent skill is hardware-independent); large divergence would flag a rendering-timing artifact worth investigating |

The headline comparison is **infra behaviour** (stability + throughput), and a
**sanity check** that agent scores are consistent across hardware. If L40S is stable
and fast, it also becomes the better platform for future large sweeps.

---

## 6. Handy references
- Full architecture / parameters: [`PAPER_REFERENCE.md`](../PAPER_REFERENCE.md)
- Module/agent internals & the difficulty scheduler: [`PAPER_REFERENCE.md`](../PAPER_REFERENCE.md) §4–§9
- The by-hand SLURM tutorial (to learn the mechanics): [`example_slurm_job.sh`](example_slurm_job.sh)
- Pipeline-module reference (to add/modify an agent): [`../leaderboard/team_code/PIPELINE_MODULES.md`](../leaderboard/team_code/PIPELINE_MODULES.md)
