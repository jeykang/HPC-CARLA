# Verification & Dev-Experience Roadmap

The project's purpose is **distributed verification of driving agents** (run agents
across the route × weather × scenario space and characterise them) plus a
**modular agent-development framework**. This plan covers five improvements that
turn it from "runs agents and writes scattered output" into "produces
trustworthy, comparable verification results and is pleasant to develop against."

Ordering reflects dependency and value: #1 underpins #2; #3 is the standalone
dev-experience win; #4/#5 harden fairness and ergonomics.

---

## #1 — Run-outcome classifier (infra vs agent) — FOUNDATION

**Problem.** A verification number is only meaningful if "the agent drove badly"
is separated from "the infrastructure broke." On this cluster ~92% of jobs failed
for *infra* reasons (server segfault, 600s connect timeout, scenario-spawn error,
walltime kill). Counting those as agent failures scores the cluster, not the agent.

**Taxonomy** (derived from the leaderboard's own status strings):
| Category | Meaning | Source signal |
|---|---|---|
| `valid_pass` | agent completed the route | route `status == 'Completed'` |
| `valid_fail` | agent evaluated, drove poorly | route `status` startswith `Failed - Agent {timed out,got blocked,deviated…}` |
| `agent_error` | agent code/config broke | `entry_status` `Rejected`/`Finished with agent errors`; crash_message `Agent's sensors were invalid`/`Agent couldn't be set up`/`Agent crashed` |
| `infra_fail` | cluster/sim broke or run cut short | no `results.json`; `entry_status` `Started`/`Crashed`; crash_message `Simulation crashed`; missing route records (`progress[0] < progress[1]`) |

**Data sources.** Per job: `<save_path>/results.json` (`entry_status`, `_checkpoint.records[].status`, `progress`), `<save_path>/run_summary.json` (`global_steps`), job `rc`/duration in `completed_jobs.json`, worker-log markers (600s timeout) as a fallback when `results.json` is absent.

**Deliverables.**
- `tools/classify_outcomes.py` — library (`classify_route`, `classify_job`, `aggregate`) + CLI.
- Per-agent summary: valid evals, pass-rate **over valid evals only**, mean composed/route score over valid, plus separately-reported `agent_error` and `infra_fail` rates (system health, excluded from agent score).
- Optional: annotate `completed_jobs.json` with an `outcome` field for downstream use.

**Risks.** "Agent timed out" is game-time, so the cluster's 10× slowdown does *not* misclassify it (slowness costs real time, not game time) — it's a genuine agent outcome. Jobs with no `results.json` must be cross-checked against `run_summary.global_steps` to distinguish "server never came up" from "wrote elsewhere."

**Effort:** ~½ day. Pure analysis, no cluster needed (runs on existing output).

---

## #2 — Results aggregation & agent-comparison report

**Problem.** Output is 1,386 scattered `results.json` files; there's no
cross-agent comparison — the actual product of a verification harness.

**Design.** Build on #1's per-route classification. Aggregate over the valid-eval
set into a multi-index table: **agent × town × weather × route-type × scenario-type**
→ {n_valid, pass_rate, mean score_composed, mean score_route, infraction rates per
km}. Emit:
- A Markdown/CSV comparison table (per-agent headline + per-dimension breakdowns).
- Side-by-side agent ranking with infra-failures excluded and reported separately.
- Reuse/extend `genfig.py` for plots (check what it already aggregates first to
  avoid duplication); add per-scenario-type and per-weather difficulty curves.

**Deliverables.** `tools/verification_report.py` (imports `classify_outcomes`),
emitting `report/verification_<ts>.{md,csv,json}`.

**Effort:** ~1 day. Depends on #1.

---

## #3 — Local pipeline test harness (dev-experience win)

**Problem.** ~8 reimplementation bugs (TCP Beta params, LAV ×5, BEV NMS…) were
found via 10-minute cluster round-trips on a node that fails 92% of jobs. There is
no way to exercise an agent pipeline without CARLA + SLURM.

**Design.** A CPU-only harness that feeds **canned sensor inputs** (synthetic or
a few recorded frames) into a `PipelineEngine` built from an agent's YAML, runs N
ticks, and asserts a sane `carla.VehicleControl` (finite, in-range steer/throttle/
brake; no NaNs; shapes correct through each module). Mock the `carla` module where
needed so it imports off-cluster.
- `tests/conftest.py` — canned input fixtures (RGB/LiDAR/measurement tensors).
- `tests/test_pipeline_<agent>.py` — per-agent: config loads, each module's
  declared read-keys are present before it runs, control output valid.
- `tests/test_pipeline_engine.py` — engine contract (context key flow).
- Make it `pytest`-runnable locally and in CI.

**Deliverables.** `tests/` suite + a `make test` / README section.

**Effort:** ~1–2 days. Highest iteration-speed ROI; independent of #1/#2.

---

## #4 — Reproducibility & deterministic evaluation

**Problem.** For fair agent comparison, every agent must face identical scenarios;
results must be reproducible/citable.

**Design.**
- Thread fixed seeds through: TrafficManager seed, `carlaProviderSeed`, weather,
  scenario spawn RNG. Surface as launcher flags (`--seed`) forwarded to workers.
- Per-run **provenance manifest**: agent + config hash, CARLA/SIF version, seeds,
  route/scenario files, git SHA, timestamp → written next to `results.json`.
- A `verify --reproduce <manifest>` path that re-runs an identical config.

**Deliverables.** seed plumbing in `manage_continuous.py`/evaluator args;
`manifest.json` writer in `consolidated_agent` or the worker; manifest surfaced in
#2's report.

**Effort:** ~1 day. Mostly plumbing + a writer.

---

## #5 — Pipeline DX polish

**Problem.** The pipeline module contract (context read/write keys) is implicit;
wiring mistakes surface only at runtime on the cluster (the class of bug we hit).

**Design.**
- **Config validation**: on load, check each module's declared `reads`/`writes`
  keys form a consistent DAG (no module reads a key nothing has written); fail
  fast with a clear message. Requires modules to declare their key contract
  (lightweight: class attributes or a registry).
- **Dry-run/introspection**: `continuous_cli.py validate-config <agent>` prints
  the resolved module list, the key flow, and any gaps — no CARLA.
- **Scaffolding**: `new-agent <name>` emits a starter YAML + module stubs.
- Document the context-key contract in `PIPELINE_MODULES.md`.

**Deliverables.** a config validator (reused by #3's tests), CLI subcommands,
docs.

**Effort:** ~1–2 days. Composes with #3 (validator powers both tests and CLI).

---

## Sequencing
1. **#1** (this PR) — makes all numbers honest; no cluster needed.
2. **#2** — the verification product, on top of #1.
3. **#3** — parallelisable anytime; biggest dev-speed win.
4. **#4**, **#5** — fairness + ergonomics once the core reporting exists.
