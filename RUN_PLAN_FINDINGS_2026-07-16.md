# Priority-run findings (P0a–P3)

*Results from executing the priority run plan on the rebooted pod09/pod17 (2026-07-16).
The ongoing P4 sweep is excluded — this file is findings only, not the task list.
Each finding lists its data, its verdict, and where it lands in the paper / PAPER_REFERENCE.*

---

## P0a — Node qualification (post-reboot)

**Verdict:** pod09/pod17 are healthy after the admin reboot that cleared the WekaFS fence.

- Bounded probes: `uid=1064(autodr_test)` resolves, `/scratch` read+write OK, container + GPU OK, no D-state.
- Persistent-mode smoke: the 5 camera agents (cilrs/neat/roach 4/4, interfuser/tcp 3+/4) completed
  with **0 agent errors, 0 GPUs parked**; both nodes stayed `allocated`; `scancel` returned them
  straight to `idle` with no D-state wedge.
- The host **GL Signal-11 segfault still reproduces** (≈108 events in the short smoke) — it is a host
  quirk unrelated to the reboot — and the restart-hardening recovers it (jobs complete).

**Paper destination:** §8 "Cluster idiosyncrasy" — first same-cluster/different-reboot test; the
GL instability is host-intrinsic and recoverable, the WekaFS fence is the fatal-but-now-cleared mode.

---

## P1a — Per-route map density (upgraded from per-town)

**Verdict:** delivered a **new, validated per-route difficulty axis**; and it's computable offline.

- `tools/route_map_density.py` reproduces the leaderboard's runtime interpolation
  (`GlobalRoutePlanner` over the map topology) **offline** via `carla.Map(name, xodr)` — **no server,
  no GL, no segfault** — over all **2,345 routes**, along the true driven path.
- Joined to the 1,648 harvested route-evals (**100% match**), pooled Spearman vs `score_composed`:

  | per-route feature | Spearman (n=1648) | reading |
  |---|---|---|
  | **`n_distinct_junctions`** (intersections driven through) | **−0.295** | real difficulty axis |
  | `heading_deg_per_km` (curvature) | −0.186 | weaker second |
  | `junctions_per_km` (rate) | +0.022 | washes out |
  | `path_len_m` | −0.022 | none |

- Key nuance: the **absolute intersection count** predicts; the **per-km rate does not**. This beats
  the old scalar difficulty (~0) and needs the interpolated path — impossible from the 2-waypoint
  endpoints alone.

**Paper destination:** strengthens Fig. 3 and the noisy-OR inputs; **deletes** the §8 limitation
"per-route map density needs the sim / endpoint approximation too coarse." (PAPER_REFERENCE §7.)

---

## P1b — Controlled illumination triads (resolves item #18)

**Verdict:** "dark = hard" is **real but agent-specific**, not universal — a controlled comparison,
not just a conditioned fit.

Fixed agent+route, vary **only** illumination on matched clear presets (ClearNoon / ClearSunset /
ClearNight = weather 0/1/14, no precip confound); 4 routes (Town02 dense + Town05 open, low/high
junction); n=4 routes/cell.

| agent | ClearNoon | ClearSunset | ClearNight | noon→night Δ |
|---|---|---|---|---|
| **neat** (camera-only) | 81.6 | 68.4 | 54.6 | **−27.0** |
| **interfuser** (camera+LiDAR) | 74.7 | 80.8 | 68.3 | **−6.4** |
| roach | 54.0 | 68.7 | 53.5 | −0.5 |
| tcp | 70.5 | 70.6 | 72.6 | +2.1 |

- Camera-only **NEAT collapses in the dark (−27)**; camera+LiDAR **InterFuser is far less affected
  (−6.4)** — LiDAR is the plausible reason and matches the sister project's "LiDAR is not
  illumination-biased." **Roach flat, TCP slightly inverted.**
- Turns "dark = hard" from conditioned-fit-only into a controlled result. (Small per-cell n:
  suggestive, not a significance claim.)

**Paper destination:** §6.3 — a 3-row table / small figure; resolves the marginal-vs-conditioned
soft spot (#18). (PAPER_REFERENCE §7.)

---

## P2 — Repeat-eval variance (why the ceiling is a ceiling)

**Verdict:** the ~0.65 AUC ceiling is **partly irreducible closed-loop stochasticity**, not merely
missing features.

Confirmed the precondition first: **`RUN_SEED` is pinned (2000)** and drives both traffic-manager
and scenario RNG, so all 1,648 prior evals share one seed. Added a scheduler feature (per-job `seed`
+ `_repNN` output tagging) and re-ran identical `(agent, route, weather)` triples under **12 distinct
seeds** (2 agents × 3 routes × MidRainyNoon).

- Variance concentrates **at the competence boundary**:
  - **NEAT / Town05 (mid):** 12 seeds → `score_composed` `[9.5, 11.8, 100, 100, ×8]` — **bimodal,
    std 33**. 10 clean successes, **2 catastrophic seed-triggered failures under identical
    conditions.**
  - **Town03 (easy):** 100 every seed (std 0). **Town02 (hard, dense):** 22/24 genuine "agent
    deviated / timed out" failures every seed.
- No condition feature can predict which seed collides — the collision is set by closed-loop RNG (NPC
  spawn/behaviour). So a real fraction of outcomes are seed coin-flips.
- Run was **infra-clean**: 0 parked GPUs, no drain/fence.

**Paper destination:** §6.3 "two readings" / §8 scope — outcome-variance stat + the one sentence
resolving the open question. (PAPER_REFERENCE §7.)

---

## P3 — LAV boot (P0b) — settled, mini-sweep skipped

**Verdict:** LAV's server-limitation is **intrinsic (LAV ↔ A100/host-GL), not a crashed-node
artifact** — a clean reboot does not fix it.

- A dedicated LAV-only run on the freshly-rebooted pod09 reproduced the failure **identically**:
  a crash-retry loop of **Signal 11 + "failed to connect to newly created map"** at `load_world`.
- Restart-hardening does eventually force a boot through, but throughput is ≈0 and the few routes
  that ran **failed** ("agent deviated" / "timed out").
- Therefore P3's LAV mini-sweep is **not a viable data source**; LAV stays excluded pending an L40S
  (or a CARLA build that survives its `load_world`). The camera-vs-LiDAR illumination contrast (P1b)
  is carried by InterFuser's LiDAR fusion, not LAV.

**Paper destination:** §9 / §5.2 — one sentence: LAV server-limitation is not node-specific.
(PAPER_REFERENCE §9.)

---

### Provenance
- Per-route density: `paper_artifacts/route_map_density.csv` (2,345 rows) · script `tools/route_map_density.py`
- Illumination triad: `routes_town0{2,5}_illum.xml`, harvested from `dataset/*/weather_{0,1,14}/…`
- Variance study: `dataset/*/weather_6/map_*/routes_town0*_p2_rep*/results.json` · scheduler seed/repeat in `manage_continuous.py`
- Commits on `main`: 1006175 (P1a), e3bd875 (P1b routes), b561a9a (P2 feature), db193b7 (P2 result), b750929 (P3).
