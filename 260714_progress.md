# HPC-CARLA — Progress Report

**Period:** since last Tuesday (2026-07-07) · 13 commits, 43 files, **+6.5k lines**, all landed on `main`.

## TL;DR
- **Agent roster expanded 2 → 5 productive agents** — added CILRS, NEAT, and Roach.
- **New per-route metric + harvester** — recovers the real dataset (per-route driving scores) even from jobs that crash mid-run; **84% of our data comes from otherwise-discarded "failed" jobs**.
- **Difficulty scoring validated, then reworked** into a per-model, multi-axis sensitivity model (physical weather decomposition + noisy-OR), validated on synthetic ground truth.
- **Cross-cluster (L40S) comparison package** ready to hand to a collaborator.
- **769 route-evals collected** across the five agents so far.
- Whole codebase consolidated onto `main`; README + technical reference rewritten.

---

## 1. Agent roster: 2 → 5 productive
Added **CILRS, NEAT, and Roach** as modular inference pipelines (YAML-driven stages, not monolithic
agents), joining TCP and InterFuser. All five run end-to-end.

| | Method | Sensors |
|---|---|---|
| TCP | Trajectory-guided control + PID (NeurIPS'22) | camera |
| InterFuser | Multi-sensor fusion transformer (CoRL'22) | 3× camera + LiDAR |
| CILRS | Conditional imitation learning (ICCV'19) | camera |
| NEAT | Neural attention fields (ICCV'21) | 3× camera |
| Roach | RL-coached imitation (ICCV'21) | camera |

*(LAV — a LiDAR-primary agent — is implemented but currently server-limited on our hardware; it is a
first-class target for the L40S run below.)*

## 2. Data collection & the per-route metric (key methodological advance)
Each route *file* is actually a **suite of hundreds of short routes**, checkpointed **per route**. The
important realization: **the metric is the per-route evaluation, not the completed file** — and every
route completed before a mid-run crash is still on disk, even in jobs the queue marks "failed."

Built **`harvest_results.py`** to recover those per-route scores from every checkpoint. On current data
it pulls **769 route-evals, 84% of them from "failed" jobs** — i.e. we'd have thrown away ~5/6 of the
usable data under file-level accounting. This is what makes the dataset viable on imperfect hardware.

**Results so far** (mean CARLA driving score, 0–100):

| Agent | route-evals | mean driving score | route completion |
|---|--:|--:|--:|
| roach | 210 | 88.8 | 95.5 |
| interfuser | 64 | 87.2 | 87.2 |
| neat | 143 | 85.4 | 89.3 |
| tcp | 99 | 83.7 | 84.8 |
| cilrs | 253 | 38.9 | 54.1 |

## 3. Difficulty scoring: validated, then reworked (main research contribution)
We can now **empirically validate** the scheduler's difficulty score against agent performance
(`difficulty_validation.py`). Two findings:

1. A **single scalar difficulty washes out** against performance (pooled Spearman ≈ +0.036) — because
   *different architectures fail on different axes*, so one number that fuses them cancels.
2. So difficulty was **reworked as a vector**:
   - **Physical weather decomposition** (`weather_axes.py`): the 0–20 weather index is split into
     independent axes — illumination, precipitation, fog, etc.
   - **Per-model sensitivity fit** (`sensitivity_matrix.py`): a per-agent noisy-OR failure model gives
     each axis's hazard weight per agent, with confidence intervals and honest significance reporting.
   - **Validated on synthetic ground truth** (`noisy_or_sanity.py`): the method provably recovers
     per-model axis sensitivity where a scalar cannot.
   - **Illumination-stratified scheduling** added so the sweep actually samples across the axes
     (hardest-first alone had collapsed the sample onto night/rain).

## 4. CILRS low score — audited, genuine (not a bug)
CILRS scores far below the others (~39). We **audited it end-to-end** — channel order, normalization,
control gating, command mapping, checkpoint load — and confirmed it is **genuine weak-baseline
behavior, not an integration bug** (CILRS is the 2019 baseline; being the floor is expected, and it
usefully anchors the low end of the comparison). Contrast with a historical InterFuser color bug, whose
signature was *uniform* collapse — CILRS's is functional-but-weak, the opposite.

## 5. Scheduling & reliability
- **Job-first / agent-interleaved scheduling** — metrics now accrue **evenly across all five agents**,
  so an early cutoff never leaves the last agents un-measured (replaced an older priority scheme that
  starved the newer agents).
- **Reliability hardening** for the A100's weak real-time rasteriser (recover-not-crashloop, per-GPU
  isolation, health-check/park) so a bad GPU no longer takes the run down.

## 6. Cross-cluster (L40S ↔ A100) comparison package
A one-file portable launcher + setup guide (`examples/`) lets a collaborator reproduce the exact run on
**L40S** hardware. Purpose: quantify the A100-vs-L40S difference and — importantly — **L40S may run the
LiDAR-primary LAV agent**, which our current hardware cannot.

## 7. Docs
README rewritten to the current state; `PAPER_REFERENCE.md` (the living technical reference) updated
across roster, difficulty model, and reliability.

---

## Findings & open questions
- **Camera-vs-LiDAR illumination sensitivity does not separate on the current roster.** Likely because
  InterFuser is camera+LiDAR *fusion* (camera-primary), not LiDAR-primary — so darkness still hurts it.
  A clean test needs **LAV** → another reason the L40S run matters.
- **InterFuser is throughput-limited** (~1.7 routes/job vs roach's ~14.6) due to cautious/stationary
  control + heavy sensors; it's underrepresented in the data (a known characteristic, not a fault).
- Per-model per-axis significance needs **more illumination coverage** to firm up — the stratified
  scheduler is now collecting it.

## Next steps
- Continue the sweep to firm up per-model difficulty sensitivity.
- Hand the L40S package to the collaborator; get LAV running there for the LiDAR comparison.
- Fold LAV into the roster analysis once available.
