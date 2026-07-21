# Metrics refresh — results from the finished full sweep

*Answers the METRICS_REFRESH_REQUEST against the completed 2,835-job stratified sweep
(job 167271, pod09+pod17, 2026-07-16→07-21). Harvest snapshot: **2026-07-21**. The
data-built numbers here **supersede** `PAPER_OFFLINE_ANALYSIS_2026-07-15.md` (1,648/86%).
All physical-ceiling checks pass (see §E).*

**The bundle (§A)** is `paper_figures_data_260721/` — same file names/schema as the 07-15
bundle **plus the new `seed` column** on `per_route_results.csv`. Regenerate with
`tools/harvest_results.py` → `tools/figure_data_export.py`.

---

## ⚑ Claims that CHANGE with the balanced 13k data (prose gating)

1. **Scalar difficulty is now *backwards*, not just washed out.** Pooled per-route Spearman
   vs `score_composed` went **0.04 (n=1648) → +0.443 (n=13059)** — positive/wrong-signed,
   driven by the scenario term (+0.488). All 5 agents positive. TCP's positive sign persists
   *and is no longer special*. Stronger anti-evidence for the scalar than before.
2. **The per-agent λ table can finally enter §6.3 (EDITS #7).** `illum_dark` is now
   **identified AND significant for all 5 agents** (was 0/35 significant at n=1648).
3. **"InterFuser has zero infractions" is no longer true.** On the balanced set InterFuser
   shows 3.3 red-light / 5.5 layout-collision / 11.4 vehicle-collision / 18.8 off-road per 100
   evals. Drop the zero-infraction claim (it was an easy-condition artifact).
4. **Per-agent ranking reshuffled** (see §C) — TCP top, InterFuser mid; the old
   InterFuser-top/all-clustered-86-89 was coverage-collapsed.
5. **Persistence savings must be scoped.** 24 boots/83% was the *original low-crash window*;
   the full 4.5-day run had **2,363 boots** (segfault-driven) → only **16.6% of per-job boots
   eliminated, 1.20 jobs/boot** (§C). Report per-run, labelled.
6. **§4.3 "sharded writer not yet validated" UPGRADES; EDITS #24 closes** — it ran clean
   under 4.5 days of sustained load (§D).
7. **`n_distinct_junctions` holds** as the per-route axis: −0.254 (was −0.295).

---

## Bundle-file corrections (Fig 7 un-scoping — verify items #26)

Two bundle files were re-exported after the first pass; **both are corrected in the shipped
bundle and in `tools/figure_data_export.py`:**

- **`figL4_server_boots_current.csv`** — was NOT stale, but its schema invited a misread: the boot
  count lives in the (former) `healthy`/`launch_attempts` columns (Σ=**2,363**), while
  `relaunch_triggers` (Σ=52) is only the no-listener sub-path. **Fixed schema:** primary column
  renamed **`server_boots`** (= 2,363; P4-clean 2,322), with `relaunch_triggers_nolistener` kept
  clearly secondary. Fig 7b plots `server_boots`. → **unscoped to the full sweep.**
- **`figL5_exceptions_from_logs.csv`** — the worker logs **accumulate across runs** (no per-line
  timestamps), so the raw tally mixed the original campaign (AgentError 100 / ValueError 50 /
  Timeout 5 are pre-P4 residue) with P4 (only RuntimeError grew, 37→565). **Fixed:** the exporter
  now subtracts a prior-run baseline (`EXC_BASELINE` env). Full-sweep-attributable exceptions:
  **RuntimeError 528, AttributeError 1, all others 0** — i.e. the sweep's failures are
  CARLA/infra (timeouts, GL), **not agent-code** (0 new AgentErrors over 13k evals — a clean Fig 7a
  story). The CSV now carries `count_full_sweep`, `count_cumulative`, and a `note`. → **unscoped.**

*(The cleaner Fig 7a source remains the outcome taxonomy `figL5_outcome_taxonomy.csv`, which is
built from the 13k harvest and was full-sweep all along.)*

---

## §C — Campaign-level counts (stated explicitly)

- **Total valid route-evals n = 13,059** (was 1,648). **94.3% recovered from queue-`failed`
  jobs** (12,316 of 13,059; only 743 from cleanly-`completed` jobs).
- **Jobs executed = 2,835** (ALL terminal: 511 completed + 2,324 failed; 0 pending/never-ran —
  the sweep fully drained). **Route-evals per job ≈ 4.6.**
- **GPU-hours (both, labelled):**
  - **Compute ≈ 1,485 GPU-h** = Σ per-job walltime capped at the 1 h `JOB_TIMEOUT` (mean
    0.52 h/job; ceiling 2,835 h — passes).
  - **Occupancy ≈ 1,716 GPU-h** = Σ terminal-job (start→end) spans (mean 0.61 h, median 0.52 h
    — spans ≈ compute this run because there were **no outages**, unlike the original where
    spans were idle-contaminated). ≈ 16 GPUs × 108.6 h wall-clock (1,738) → ~85% duty cycle.
- **Server boots = 2,363** (worker-log, per node/gpu, 133–168/GPU) — **2,322 P4-clean** from the
  job-scoped SLURM `.out` (the ~41 difference is pre-P4 residue in the accumulated worker logs).
  Dominated by GL-segfault recoveries. 1.20 jobs/boot; 16.6% of per-job boots eliminated vs
  restart-per-job. **In `figL4_server_boots_current.csv` this is the `server_boots` column — NOT
  `relaunch_triggers_nolistener` (Σ=52), which is only the no-listener sub-path and undercounts ~45×
  (most boots follow a segfault). See "Bundle-file corrections" below.**
- **Coverage: 168 distinct town×weather cells (8 towns × 21 weathers), FULLY covered by every
  one of the 5 agents** (was 37 cells). Full factorial.

**Per-agent table (Table 3) — n=13,059:**

| agent | n | mean driving score | mean route-completion (distance) | Completed-status share | timeout share |
|---|---:|---:|---:|---:|---:|
| TCP | 2,515 | 66.8 | 80.8 | 53.4% | 38.0% |
| Roach | 3,304 | 61.8 | 87.5 | 76.5% | 10.6% |
| InterFuser | 1,238 | 56.6 | 71.8 | 50.2% | 40.6% |
| NEAT | 1,860 | 55.1 | 79.4 | 66.1% | 24.0% |
| CILRS | 4,142 | 24.6 | 54.1 | 21.0% | 29.9% |

*(The two completion metrics are deliberately distinct and DIVERGE — e.g. TCP drives 80.8% of
the distance but only 53.4% of routes reach `Completed` status because it times out 38% of the
time.)*

**Infractions per 100 route-evals (feeds §7.2 signatures):**

| agent | red-light | layout-collision | off-road | vehicle-collision |
|---|---:|---:|---:|---:|
| Roach | **24.9** | 6.0 | 25.0 | 24.4 |
| CILRS | 16.4 | **44.4** | **54.8** | 17.9 |
| NEAT | 13.0 | 13.7 | 23.9 | 22.0 |
| InterFuser | 3.3 | 5.5 | 18.8 | 11.4 |
| TCP | 3.0 | 7.1 | 18.0 | 6.2 |

Signature reads: **Roach runs red lights** (24.9/100, up from 17.8); **CILRS layout-collision +
off-road heavy** (44.4 / 54.8) with only 21% Completed; **TCP times out** (38% share, lowest
infractions); **InterFuser is NOT infraction-free** anymore.

> **⚠ Comparability caveat for Table 3:** per-agent n differs up to 3× (CILRS 4,142 vs
> InterFuser 1,238) because faster agents finish more routes per timeout-capped job, so the
> *conditions actually scored* aren't identically distributed. Coverage is 168/168 cells for
> all agents, but eval density per cell varies. Means + the CILRS-vs-rest gap are solid; a
> load-bearing ranking should be condition-matched (per cell).

---

## §B — Analysis outputs (on the refreshed harvest)

**B1 · difficulty_validation --per-route** (Table 2 / §6.2): scalar difficulty vs `score_composed`,
per-route, n=13,059.

| population | n | Spearman ρ | p |
|---|---:|---:|---:|
| pooled | 13,059 | **+0.443** | <1e-4 |
| cilrs | 4,142 | +0.531 | <1e-4 |
| neat | 1,860 | +0.447 | <1e-4 |
| roach | 3,304 | +0.396 | <1e-4 |
| interfuser | 1,238 | +0.356 | <1e-4 |
| tcp | 2,515 | +0.294 | <1e-4 |

Component (pooled): route +0.353, **scenario +0.488**, weather −0.075. → the scalar's
positive sign is the degenerate geometry/scenario terms; only weather has the (weakly) correct
sign. **Does not support the prune assumption.**

**B2 · sensitivity_matrix** (§6.4 + the §6.3 λ table, EDITS #7): per-agent noisy-OR λ, 95% CIs.
`illum_dark` **identified + significant for all 5 agents**; `road_water`, `cloud` also
significant; `geom`/`scen` confounded (r≈0.96), `precip`/`fog` n.s.

| agent | λ(illum_dark) [95% CI] | λ(road_water) | λ(cloud) |
|---|---|---|---|
| CILRS | **1.07 [0.81, 1.33]** | 0.85 | 1.05 |
| InterFuser | **0.61 [0.35, 0.88]** | 0.39 | 0.48 |
| NEAT | **0.58 [0.38, 0.79]** | 0.15 | 0.24 |
| TCP | **0.47 [0.30, 0.63]** | 0.33 | 0.39 |
| Roach | **0.46 [0.29, 0.63]** | 0.19 | 0.22 |

Illumination sensitivity is ordered weak-baseline-first: CILRS ≫ InterFuser ≈ NEAT > TCP ≈
Roach. Full table written to `paper_artifacts/sensitivity_matrix.md`.

**B3 · difficulty_model_comparison (AUC)** (§6.3 / abstract): the tool doesn't scale to 13k (the
per-eval scalar re-parse × 5-fold CV times out); run on a **representative 3,265-eval subsample**
(every 4th row):

| model | AUC (13k subsample) | AUC (old 1.6k) |
|---|---:|---:|
| scalar difficulty | **0.464** | 0.53 |
| illumination + geometry (multi-axis) | **0.527** | 0.62 |

Multi-axis beats scalar on held-out **log-loss 6/6 agents**, AUC 5/6. Two readings: (a) the scalar
AUC fell **below 0.5** — confirming B1's "backwards" (its correlation is positive); (b) the
multi-axis AUC also dropped (0.62→0.53) because the balanced set is harder and, per P2, a real slice
of outcomes are seed-noise — so **the ceiling not only holds but the achievable AUC is lower on the
full-difficulty set**, strengthening the irreducible-stochasticity reading. *(Caveat: subsample, not
full 13k — the tool needs vectorizing to run at full n; direction and margin are stable.)*

**B4 · per-route density vs score** (Spearman, n=12,343 matched):

| feature | ρ (13k) | ρ (1.6k) |
|---|---:|---:|
| `n_distinct_junctions` | **−0.254** | −0.295 |
| `junctions_per_km` | −0.117 | +0.02 |
| `heading_deg_per_km` | −0.016 | −0.186 |
| `path_len_m` | +0.038 | −0.02 |

`n_distinct_junctions` remains the real per-route difficulty axis; curvature collapsed to ~0 on
the balanced set.

**B5 · P1b triad**: not re-grown as a dedicated experiment (the sweep re-ran the `_illum` routes
at all 21 weathers/seed 2000; the weather-0/1/14 subset could be recomputed but the dedicated
n=4/cell triad result stands — the Table 4 small-n caveat does **not** lift yet).

**B6 · P2 repeats**: not grown — P2's 12-seed study is a separate dataset (seeds 3001–3012, in
`_rep*` dirs); the sweep added no new seeds. §6.3 seed paragraph stays (12 seeds; 22/24; s.d. 33).

---

## §D — Run-conditions facts (one line each)

- **Nodes:** original **pod09 + pod17** (post-admin-reboot), 16 A100. No replacements/mix.
- **Sharded sensor writer under sustained load: YES — ran clean for the full 4.5 days.** Every
  route dir carries `shard_manifest.json` + `shards/`. **WekaFS fence / drain / D-state events:
  ZERO** on pod09/17 for the whole run; **0 GPUs parked**; SLURM job reached a clean `COMPLETED`
  (not `NODE_FAIL`). Run duration **108.6 h (4d 12h 37m)**. → **§4.3 upgrades from "not yet
  validated"; EDITS #24 closes.**
- **GL segfaults:** ~1,300–1,900 `Signal 11` events absorbed over the run (~130–165/GPU, ≈ the
  2,363 boots) — the host GL instability persists but is fully recovered; consistent with §4.3/§8.

---

## §E — Provenance

- **Seed split: all 13,059 evals are `RUN_SEED=2000`** (uniform — the sweep used the fixed
  campaign seed). The P2 seed-varied study (12 seeds 3001–3012) is a **separate** dataset in the
  `_rep*` dirs, not in this harvest. → §8's per-seed caveat **simplifies**: the main dataset is
  homogeneous; no pre/post-P2 mixing (EDITS #19 — reword to "the campaign is single-seed; a
  dedicated 12-seed study (§6.3) probes seed variance separately").
- **Units labelled:** "jobs" = executed (2,835, all terminal). "GPU-hours" split into **compute
  (1,485)** vs **occupancy (1,716)**, both stated.
- **Physical-ceiling checks:** 2,835 jobs × 1 h cap = 2,835 GPU-h ceiling; compute 1,485 < ceiling
  ✓. n=13,059 over 2,835 jobs = 4.6 evals/job (plausible) ✓.
- **Supersession:** these numbers supersede `PAPER_OFFLINE_ANALYSIS_2026-07-15.md` (log in
  EDITS.md). Where the 07-15 doc disagrees, the data-built export here wins.
