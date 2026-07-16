# Offline analysis outputs for the SC26 draft — 2026-07-15

Produced on the login node (no CARLA execution, no matplotlib/numpy) from the existing state
files. Numbers/tables/text are ready to paste into `main.tex`; **PNG figures must be rendered on
your laptop** (matplotlib) from the `genfig` scripts + the fresh data noted per item. Harvest is
now **n=1648 route-evals** (was 769 in the tracker), **86% recovered from "failed" jobs**.

---

## Item #1 — difficulty validation on the full harvest (§6.2, §7.3)  ⚠️ REFRAME, not a number swap

Per-route Spearman of scalar difficulty vs `score_composed`:

| scope | n | Spearman ρ | p | note |
|---|--:|--:|--:|---|
| **pooled (all agents)** | 1648 | **+0.005** | 0.83 | washes out (was +0.036 @204) |
| cilrs | 477 | −0.188 | <1e-4 | significant negative |
| interfuser | 140 | −0.149 | 0.078 | **n.s.** (was −0.642 @204) |
| neat | 223 | −0.047 | 0.48 | n.s. |
| roach | 539 | +0.075 | 0.083 | n.s. |
| tcp | 269 | **+0.138** | 0.024 | significant **positive** (was −0.311 @204) |

Component (pooled): route ρ=−0.088 (p=0.0004, weak), scenario −0.001 (n.s.), weather −0.017 (n.s.).

**The old paper numbers (InterFuser −0.642, TCP −0.311) were small-sample artifacts and do not
survive.** TCP's correlation flips *positive* (its cautious control times out on the geometrically
"easy" routes); InterFuser's strong negative vanishes. **Action:** drop the "significant negative
per-agent correlations" framing in §6.2/§7.3. The honest, stronger story: the scalar difficulty is
inadequate (pooled ≈0; per-agent inconsistent, even sign-flipping) — which motivates the multi-axis
rework and the "difficulty is agent-relative" finding (see the map-density and 2-waypoint results in
`PAPER_REFERENCE.md §7`).

---

## Item #4 — Table 2, harvest-based (§7.3)

| agent | n | mean score | mean completion | towns | weathers |
|---|--:|--:|--:|--:|--:|
| roach | 539 | 86.2 | 93.4 | 4 (01,02,03,05) | 11 |
| interfuser | 140 | 88.7 | 88.7 | 4 (01,02,04,05) | 11 |
| neat | 223 | 86.5 | 90.3 | 4 (01,02,03,05) | 12 |
| tcp | 269 | 86.3 | 87.1 | 3 (01,02,05) | 11 |
| cilrs | 477 | 41.4 | 55.4 | 5 (01–05) | 13 |

Total **1648** route-evals across **37** distinct (town × weather) cells. Add these `n` and the
coverage columns to the caption; strike "replace with clean run."

---

## Item #5 — outcomes + infractions (Fig 6 data; §7.3)

**Outcome taxonomy** (per-route status; valid despite crashes — §4.4):

| agent | n | Completed% | timeout | blocked | deviated | crashed |
|---|--:|--:|--:|--:|--:|--:|
| cilrs | 477 | 26 | 259 | 28 | 68 | 0 |
| interfuser | 140 | 77 | 30 | 0 | 0 | 2 |
| neat | 223 | 84 | 35 | 0 | 0 | 0 |
| roach | 539 | 86 | 67 | 1 | 8 | 0 |
| tcp | 269 | 58 | 110 | 4 | 0 | 0 |

**Infraction totals** (event counts):

| agent | collVeh | collPed | collLayout | redLight | offRoad | routeDev |
|---|--:|--:|--:|--:|--:|--:|
| cilrs | 22 | 5 | 281 | 58 | 208 | 68 |
| interfuser | 0 | 0 | 0 | 0 | 0 | 0 |
| neat | 12 | 0 | 0 | 20 | 0 | 0 |
| roach | 33 | 4 | 0 | 96 | 18 | 8 |
| tcp | 11 | 0 | 1 | 0 | 0 | 0 |

Story for the caption: distinct failure *signatures* — CILRS crashes into layout/off-road (weak
perception), Roach runs red lights (aggressive but completes), TCP times out (cautious), InterFuser
is infraction-free (times out rather than risk). Render the grouped bars on your side.

---

## Item #2 — Fig 4 re-scoped: persistence + scaling (§7.2)

> **CORRECTED 2026-07-17 — the earlier numbers here conflated three "job" units and don't survive
> the arithmetic a reviewer does (1694×3030 s ≈ 1,425 GPU-h vs the 1,006 elsewhere). Ground truth
> recomputed from the campaign backup (`collection_state/*.prezerowsmoke_20260715.json`):**

**Definitions (keep explicit in the caption — this is exactly the multiplication a reviewer runs):**
- *Job* = one `(agent, weather, route-file)` evaluation execution, walltime capped at the 1 h
  `JOB_TIMEOUT`.
- The queue held **1,680 jobs**; the node outages meant only **135 ran to a terminal state**
  (44 completed + 91 failed) + 8 running; **1,553 stayed pending and never executed.** Report **135
  jobs executed**, not the queue size. (The old "N = 1694" was ≈ the queue size, mislabeled.)

**Compute actually consumed:**
- **≈100 GPU-hours** = Σ per-job walltime, each capped at 1 h (mean **0.74 h/job** over the 135
  executed; physical ceiling 135 GPU-h). This is the honest "GPU-hours of job compute" — report this.
- The **mean 3,030 s / median 3,600 s** walltime is the `duration` field, which exists on the **91
  failed jobs only** (they run to the timeout); completed jobs have `duration: null`. **Do not
  multiply 3,030 s by the queue size** — that is the 1,425 GPU-h error.
- The **1,006 GPU-h** figure = Σ of the 135 jobs' `(start→end)` wall-clock spans (mean 7.4 h but
  **median 1.0 h**); the long tail is node-crash/requeue idle inside the span, so it overcounts
  compute while missing non-terminal jobs. Occupancy proxy, not compute — don't equate it to job time.

**Persistence (measured, campaign-as-run):**
- **24 server boots served the 135 executed jobs → ~83% of per-job boots eliminated (~3.7 GPU-h
  saved).** Boot budget ~120 s/start (some boots hit that ceiling — upper bound).
- The **98.6% / 55.8 GPU-h** figure is `(1680−24)/1680` — a **full-sweep projection** that credits
  the 1,553 never-run jobs; use only if explicitly labelled "projected."

**Scaling:** campaign wall-clock 2026-07-07 → 07-15 (**183 h**), mostly outage/idle; ~135 jobs at
~100 GPU-compute-h. x-axis = effective GPU count (8 single-node / 16 two-node, minus parked/drained),
from job start/end timestamps + parked markers.

---

## Item #16 — §8 Discussion sentences (draft, factual)

> During final data collection the primary cluster's A100 nodes entered a recurring hardware-failure
> state, repeatedly crashing under sustained GPU load. The harvested dataset nonetheless survived
> intact: 86% of the 1,648 per-route evaluations were recovered from the on-disk checkpoints of jobs
> the scheduler had marked *failed* (§4.4) — unplanned but direct evidence for the recovery-and-
> accounting design the system is built around.

---

## Item #11 — abstract numbers (for after 1/2/4, + L40S if it lands)

- **5** driving agents (TCP, InterFuser, CILRS, NEAT, Roach); LAV pending (collaborator/L40S).
- **1,648** per-route evaluations, **86%** recovered from failed jobs, over **37** town×weather cells.
- Persistence: **~83%** of per-job server boots eliminated as measured (24 boots / 135 executed jobs;
  ~100 GPU-h of compute); 98.6% only as a full-sweep projection.
- Difficulty: a single scalar washes out (pooled ρ≈0.00); motivates per-model, map-aware scoring.

---

## Not done here (blocked / not mine)

- **#3, #6** (L40S portability, LAV) — collaborator's `handoff_L40S/`; independent of this cluster.
- **#10** (authors/affiliations/acks/URL) — needs you.
- **#7, #13** (per-axis λ table; clean re-run) — deferred per the tracker; do not block submission.
- **Figures (#2, #5) + `main.tex` edits** — render/paste on your laptop (matplotlib + paper source).
