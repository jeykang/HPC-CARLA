# Technical Reference — HPC-CARLA Persistent Data Collection System

*Comprehensive reference for paper writing. Contains exact parameter values,
architecture specifications, implementation details, and engineering decisions.*

*Last updated: **2026-07-15**. State of the project: five productive agents
(TCP, InterFuser, CILRS, NEAT, Roach; **LAV** server-limited) evaluated as modular
pipelines. **1,648 per-route evaluations** harvested, **86% recovered from crashed
/ "failed" jobs** — the recovery-and-accounting thesis in action (§6, §8). The
difficulty model was substantially reworked and honestly **re-validated at n=1648**:
the single scalar **washes out**, its geometry/scenario terms are **degenerate** for
the routes actually collected (2-waypoint endpoints), and the recoverable signal is
**illumination + map urban-density** (§7, §11). Cluster reliability degraded from
intermittent GL segfaults to **recurrent whole-node crashes** under sustained A100
load; a **park-on-unkillable** mitigation was added and the run is **currently paused**
pending admin node stabilization (§9). Sections **4–5** (agent internals), **10**, and
the **appendices** are stable reference; **§1–3, 6–9, 11** carry the current-state
narrative and were rewritten this pass.*

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware and HPC Infrastructure](#2-hardware-and-hpc-infrastructure)
3. [CARLA Simulation Environment](#3-carla-simulation-environment)
4. [Modular Agent Architecture](#4-modular-agent-architecture)
5. [Agent Implementations](#5-agent-implementations)
   - 5.1 [TCP](#51-tcp---trajectory-guided-control-with-pid)
   - 5.2 [LAV](#52-lav---learning-from-all-vehicles)
   - 5.3 [InterFuser](#53-interfuser---interpretable-multi-sensor-fusion-transformer)
   - 5.4 [CILRS](#54-cilrs--conditional-imitation-learning)
   - 5.5 [NEAT](#55-neat--neural-attention-fields)
   - 5.6 [Roach](#56-roach--rl-coached-imitation-learning)
6. [Data Collection Pipeline](#6-data-collection-pipeline)
7. [Job Scheduling and Difficulty Estimation](#7-job-scheduling-and-difficulty-estimation)
8. [Dataset Structure and Statistics](#8-dataset-structure-and-statistics)
9. [Implementation Challenges and Solutions](#9-implementation-challenges-and-solutions)
10. [Metrics and Evaluation](#10-metrics-and-evaluation)
11. [Key Findings](#11-key-findings)
12. [Appendix A — Weather Presets](#appendix-a--weather-presets)
13. [Appendix B — Scenario Type Difficulty Weights](#appendix-b--scenario-type-difficulty-weights)
14. [Appendix C — Pipeline Module Reference](#appendix-c--pipeline-module-reference)

---

## 1. System Overview

The system evaluates published imitation-learning driving agents at scale by running them across
the combinatorial space of CARLA towns, routes, and weather on an HPC A100 cluster. Every agent is
re-implemented as a declarative YAML pipeline of composable stages (§4), so sensor wiring, model
parameters, and control logic are reconfigured independently of agent code — a single
`ConsolidatedAgent` wrapper *is* every agent, distinguished only by its config.

Three findings shape everything downstream and recur through this document:

1. **The reportable unit is the per-route evaluation, not the completed job.** CARLA's leaderboard
   checkpoints `results.json` *per route within a route-file suite*, so a server that crashes
   mid-suite still leaves every route it finished on disk — even though SLURM/queue accounting
   marks the whole job `failed`. A harvester (§6, §9) recovers these. Of the **1,648 route-evals**
   collected to date, **≈86% were recovered from "failed" jobs** that file-level accounting
   discards entirely.
2. **Difficulty is agent-relative and the single scalar difficulty score does not predict
   performance.** Re-validated honestly at n=1648 it *washes out* (pooled Spearman ≈ 0); the
   earlier encouraging per-agent correlations (n≈204) were small-sample artifacts that reverse or
   vanish at scale. The recoverable signal is **illumination ("dark = hard") + map urban-density**,
   near a **~0.65 AUC ceiling** independently corroborated by a sister AV-scene-scoring project
   (§7, §11).
3. **Reliability is a host-infrastructure story, not an agent-code story.** The A100 GL stack
   segfaults intermittently and, under sustained multi-day load, escalates to whole-node crashes.
   The pipeline was hardened to *recover* rather than *collapse* (segfault restart → park-on-
   unkillable), which is what makes the 86% recovery possible; the node-crash mode is beyond any
   user-side fix and currently gates the run (§9).

**Agent roster (six agents, five productive):**
- **TCP** (§5.1) — trajectory + control imitation learning
- **InterFuser** (§5.3) — camera+LiDAR sensor-fusion transformer IL
- **CILRS** (§5.4) — conditional imitation learning (genuine weak baseline; audited, §5.4)
- **NEAT** (§5.5) — neural attention fields
- **Roach** (§5.6) — RL-coached imitation learning
- **LAV** (§5.2) — camera+LiDAR IL from all vehicles — **server-limited** on A100 (crashes during
  `load_world`, yields ≈0 metrics); code/config retained, excluded from throughput runs (§9)

All five productive agents are camera-centric except InterFuser (camera+LiDAR fusion); LAV is the
only LiDAR-primary agent, which is precisely the one A100 cannot serve — so the collected data
cannot yet contrast camera-only vs LiDAR illumination sensitivity from the LAV side (§7, §11).

**Scope of a full collection cycle:**
- 5 productive agents × 22 route files × 21 weather presets ≈ 2,300 jobs (LAV would add a 6th)
- 8 CARLA towns (Town01–Town07, Town10HD)
- Route types: **long** (16–75 real waypoints, ~0.8–2.5 km) vs **short**/**tiny** (stored as **2
  waypoints — endpoints only**; the driven path is interpolated at runtime — see §3, §7)
- 21 CARLA weather presets, ClearNoon (0) → HardRainNight (20)

**Core design decisions:**
- **Persistent CARLA servers** (one per GPU) eliminate the ~60 s boot per job — ≈98.6% of boots
  removed at steady state (§10).
- **One wrapper, many agents:** agent identity is a YAML config over a shared runner (§4).
- **Job-first scheduling:** each agent advances through the queue in lockstep, so metrics
  accumulate *interleaved across agents* — if the run is cut short, no agent is left un-sampled
  (§7).
- **Server + GPU resilience:** a crashed CARLA is killed, health-checked, relaunched; a GPU whose
  process survives SIGKILL (uninterruptible D-state) is *parked* so it neither burns the queue nor
  drains the node (§9).
- **Illumination-stratified coverage:** the scheduler guarantees noon/sunset/night samples per
  agent before reverting to hardest-first, so the identifiable difficulty axis is actually
  sampled (§7).
- Leaderboard scores are parsed from `results.json` per route and unioned across the queue by the
  harvester (§6).

---

## 2. Hardware and HPC Infrastructure

### Cluster configuration

| Parameter | Value |
|-----------|-------|
| Scheduler | SLURM |
| Nodes assigned to this project | `hpc-pr-a-pod09`, `hpc-pr-a-pod17` (8 A100 each) |
| GPUs per node | 8 × NVIDIA A100 (driver 575.57.08) |
| Max parallelism | 16 concurrent CARLA evaluations (1 per GPU) when both nodes are up |
| Job time limit | 336 h (14 days) |
| Per-route wall-clock (`JOB_TIMEOUT`) | 3600 s — a route exceeding it is killed and re-queued |
| Node allocation | Exclusive |

**Effective capacity is well below 16 GPUs.** Under sustained multi-day load these A100 nodes crashed
(§9): over the recent runs **pod09 went fully "Not responding"** and **pod17 repeatedly drained**
("Kill task failed"), traced to WekaFS storage fencing (§9). After an admin reboot (2026-07-15) that
cleared the fence and re-attached the WekaFS clients, both nodes were re-validated end-to-end (a
persistent-mode smoke completed cleanly with no drain/fence/park recurrence) and returned to service.
Even so, the run has historically spent significant time on a **single node or paused entirely**
rather than the nominal 16-way. Throughput figures in this document are single-node-realistic, not
16-GPU-ideal. The simulation itself runs **~8–12× slower than real time** with no per-*sweep*
wall-clock cap, so a full sweep does not drain quickly — the run is treated as a continuous harvest,
not a fixed batch (§7, §8).

### Container runtime

Agents run inside a Singularity/Apptainer container (`carla_official.sif`) that bundles the
CARLA Python API, PyTorch, and all agent dependencies. The project root
(`/scratch/autodr_test/HPC-CARLA-persistent`) is bind-mounted into the container as
`/workspace`. Environment variables are forwarded via `SINGULARITYENV_*` / `APPTAINERENV_*`.

The container does **not** include `torch_scatter`; the LAV `PointPillarNet` scatter operations
were replaced with pure-PyTorch equivalents (see §9).

### Port allocation

Each GPU slot uses three ports:

| Purpose | Node 0 base | Node 1 base | Per-GPU offset |
|---------|------------|------------|----------------|
| CARLA RPC | 2000 | 3000 | +100 per GPU |
| Traffic Manager | 7000 | 8000 | +100 per GPU |

Example: node 0, GPU 4 → RPC 2400, TM 7400.

### Multi-node coordination

A master node (NODE_ID=0) initialises the queue and shared state; secondary nodes wait 10 s
then begin consuming jobs. Each node runs a separate coordinator process that claims GPU slots
and dispatches `singularity exec` subprocesses. A file-level lock (`.scheduler.lock`) serialises
queue writes across nodes.

### State files

| File | Purpose |
|------|---------|
| `collection_state/job_queue.json` | Live queue (running + pending) |
| `collection_state/completed_jobs.json` | Completed/failed job archive with scores |
| `collection_state/runtime_estimates.json` | Empirical runtime estimates per (agent, route) |
| `collection_state/gpu_status.json` | Per-GPU health snapshots |
| `collection_state/health/<gpu>.json` | Per-GPU live status |
| `collection_state/metrics/` | Node-level GPU/system utilisation time-series |

---

## 3. CARLA Simulation Environment

**CARLA version:** 0.9.10 (leaderboard evaluator 1.0)

**Python version inside container:** 3.7

### Towns

| Town | Description | Scenario types available |
|------|-------------|--------------------------|
| Town01 | Grid-layout urban, 4-way intersections | Scenario1,3,4,7,8,9,10 |
| Town02 | Residential, smaller scale | Scenario1,3,4,7,8,9,10 |
| Town03 | Complex multi-lane, roundabout | Scenario1,3,4,7,8,9,10 |
| Town04 | Highway + ramps, complex interchanges | Scenario1,3,4,7,8,9,10 |
| Town05 | Multi-lane city + large roundabout | Scenario1,3,4,7,8,9,10 |
| Town06 | European-style, tram tracks | Scenario1,3,4,7,8,9,10 |
| Town07 | Rural roads, narrow lanes | Scenario1,3,4,7,8,9,10 |
| Town10HD | High-detail urban (HD version) | Scenario1,3,4,7,8,9,10 |

### Routes

22 XML route files in `leaderboard/data/training_routes/`. Each file contains multiple
`<route>` elements with `<waypoint>` nodes (x, y, z, yaw attributes, CARLA world coordinates).

| Type | Towns covered | Stored waypoints/route | Driven path |
|------|--------------|------------------------|-------------|
| Long | Town01–07 | **16–75** (real geometry) | ~0.8–2.5 km |
| Short | Town01–07, Town10 | **2 (endpoints only)** | interpolated at runtime |
| Tiny | Town01–07, Town10 | **2 (endpoints only)** | interpolated at runtime |

> **Important correction (2-waypoint degeneracy).** The `_short` and `_tiny` route files — which
> constitute the *entire current sweep* — store each route as just its **start and end waypoint**.
> The actual driven path is reconstructed at runtime by CARLA's `GlobalRoutePlanner` (A\* over the
> map topology). Any offline metric that parses these XMLs (route length, sharp-turn count, heading
> change, along-path scenario density) therefore measures only **endpoint displacement**, not the
> real route — it is near-noise. This single fact explains why two of the three inputs to the
> scalar difficulty score (§7) are uninformative for the routes actually being collected, and why
> the earlier route-geometry correlations did not survive at scale. Only the `_long` files carry
> real, parseable geometry.

**Difficulty score range** (see §7): `_long` files score high (`routes_town04_long` = 73.26);
`_short`/`_tiny` files collapse to a narrow, mostly weather-driven band because their geometry and
scenario terms are degenerate per the note above.

### Scenarios

Scenario triggers are defined in `leaderboard/data/scenarios/town*_all_scenarios.json`. All
scenario types share **identical spawn locations** within a town — confirmed by comparing position
sets across all 7 types: zero differing positions (486 unique positions in Town01, 119 unique
20 m-grid cells).

| Scenario ID | Description | Difficulty weight |
|-------------|-------------|-------------------|
| Scenario1 | Slow leading vehicle | 1.0 |
| Scenario3 | Cut-in vehicle | 3.0 |
| Scenario4 | Stationary obstacle in lane | 2.0 |
| Scenario7 | Pedestrian at marked crossing | 2.5 |
| Scenario8 | Jaywalking pedestrian | 3.5 |
| Scenario9 | Sudden appearance from occlusion | 4.5 |
| Scenario10 | Slow vehicle + secondary hazard | 2.0 |

Mean type weight across all 7 types: **2.64**.

### Weather presets

See Appendix A for the full 21-preset table with difficulty scores.

---

## 4. Modular Agent Architecture

### ConsolidatedAgent

`leaderboard/team_code/consolidated_agent.py` — a universal `AutonomousAgent` subclass that
operates in two modes:

**Legacy mode:** delegates `run_step()` to an original agent class (e.g., `InterfuserAgent`).

**Pipeline mode:** reads a `pipeline:` block from the agent YAML and executes it each tick via
`PipelineEngine`.

Each tick follows four stages:
1. `_ensure_pipeline_or_inner_loaded()` — lazy-initialises the pipeline on first call
2. `_save_sensor_data(input_data, timestamp)` — writes per-frame sensor data when
   `COLLECT_DATA=1`
3. Pipeline execution: `pipeline.run(ctx)` → `ctx['control']`
4. `_postprocess_control(control)` — coerces output to `carla.VehicleControl`

Context dict keys injected before pipeline execution:

| Key | Type | Description |
|-----|------|-------------|
| `input_data` | dict | Leaderboard sensor dict `{sensor_id: (frame, raw)}` |
| `timestamp` | float | Simulation timestamp |
| `global_step` | int | 0-based frame counter |
| `last_control` | carla.VehicleControl | Previous tick's output |
| `config` | dict | Parsed YAML config |
| `agent` | ConsolidatedAgent | Agent instance (carries `_global_plan`) |

### PipelineEngine

`leaderboard/team_code/pipeline_engine.py` — iterates the module list, calling `module.run(ctx)`
on each. Stops early if `ctx['__pipeline_stop__'] = True` (used by `WarmupAndFrameSkip`).
If a module has a `setup(agent, config)` method it is called once on first tick.

### pipeline_modules.py

Single file (~2,500 lines) containing all 40+ pipeline stage classes organised into:
- Sensor extraction (5 classes)
- Routing/planning (2 classes)
- Image processing (3 classes)
- LiDAR processing — InterFuser-style (1 class)
- EKF localisation + LAV LiDAR pipeline (4 classes)
- BEV detection (1 class)
- Control modules (5 classes)
- TCP-specific (6 classes)
- LAV-specific (5 classes)
- InterFuser-specific (3 classes)
- Torch utilities (3 classes)
- Glue / state (5 classes)

---

## 5. Agent Implementations

### 5.1 TCP — Trajectory-guided Control with PID

**Reference:** [Chen et al., 2022, "Think Twice Before Driving"](https://arxiv.org/abs/2305.06022)

**Original codebase:** `leaderboard/team_code/tcp/`

**Checkpoint:** `tcp_model.ckpt` (PyTorch Lightning checkpoint)

#### Sensors

| Sensor | Position (x,y,z) | Resolution | FOV | Notes |
|--------|-----------------|------------|-----|-------|
| `sensor.camera.rgb` (rgb) | (−1.5, 0, 2.0) | 900×256 | 100° | Rear-facing camera |
| `sensor.other.imu` (imu) | (0, 0, 0) | — | — | tick 0.05 s |
| `sensor.other.gnss` (gps) | (0, 0, 0) | — | — | tick 0.01 s |
| `sensor.speedometer` (speed) | — | — | — | 20 Hz |

#### Model architecture

**Backbone:** ResNet34 (ImageNet pretrained)
- Input: 900×256 RGB (no resize/crop — full resolution fed directly)
- Output: 1000-dim embedding (`feature_emb`) + spatial features `cnn_feature` (8×29×512)

**State head:** Linear(9→128) → ReLU → Linear(128→128)
- State vector: `[speed/12, target_x, target_y, cmd_one_hot(6)]`, shape (1, 9)

**Trajectory path (waypoint prediction):**
- `join_traj`: Linear(1128→512) → ReLU → Linear(512→512) → ReLU → Linear(512→256)
- `GRUCell(4, 256)` — autoregressive over `pred_len` steps, initial hidden = join_traj output
- Output: `pred_wp` — waypoint deltas, shape (pred_len, 2), metres

**Control path (Beta-distribution):**
- `join_ctrl`: Linear(640→512) → ReLU → Linear(512→512) → ReLU → Linear(512→256)
- `policy_head`: Linear(256→256) → ReLU → Linear(256→256) → Dropout → ReLU
- `GRUCell(260, 256)` — autoregressive
- `dist_mu`: Linear(256→2) → **Softplus** → `mu_branches` (α parameters, range (0,∞))
- `dist_sigma`: Linear(256→2) → **Softplus** → `sigma_branches` (β parameters, range (0,∞))

**Auxiliary heads:**
- `pred_speed`: Linear(1000→256) → ReLU → Dropout → ReLU → Linear(256→1)
- Attention-weighted sum over `cnn_feature` spatial positions (8×29 → 1)

#### Dual-path control

The pipeline implements two control paths blended by turning status:

**Beta-distribution path (`TCPBetaControl`):**

The model's `mu_branches` and `sigma_branches` are Softplus outputs — they are the α and β
parameters of a Beta distribution directly, not mean and sigma. Action is computed as the
distribution **mode** using the `_get_action_beta` formula from the original codebase:

```
x = 0.5 (default)
if α > 1 and β > 1:  x = (α − 1) / (α + β − 2)    # mode
if α ≤ 1 and β > 1:  x = 0.0                          # minimum
if α > 1 and β ≤ 1:  x = 1.0                          # maximum
if α ≤ 1 and β ≤ 1:  x = α / (α + β)                 # mean (bimodal case)
action = x * 2 − 1  ∈ [−1, 1]
```

Outputs: `acc = action[0]`, `steer = clip(action[1], −1, 1)`.

**PID path (`TCPPIDControl`):**

Reimplementation of `TCP.control_pid()` including the three-angle outlier-rejection:

```
angle       = arctan2(wps[aim_idx][1], wps[aim_idx][0])
angle_last  = arctan2(wps[aim_idx−1][1], wps[aim_idx−1][0])
angle_target = arctan2(target[1], target[0])      # GPS-derived angle

use_target = (|angle_target| < |angle|) OR
             (|angle_target − angle_last| > angle_thresh AND target_dist < dist_thresh)
final_angle = angle_target if use_target else angle
```

Y-axis is flipped (`wps[:,1] *= −1` and `target[1] *= −1`) to match CARLA forward-negative
convention. Speed delta: `clip(desired_speed − speed, 0, clip_delta)`.

PID parameters: `turn_KP=1.25, turn_KI=0.75, turn_KD=0.3, turn_n=40`,
`speed_KP=5.0, speed_KI=0.5, speed_KD=1.0, speed_n=40`.

**Blending (`TCPBlendControl`):**

Turning status is detected using a 20-frame rolling window of `|steer|`:
- Turning if at least 10 of the last 20 frames have `|steer| > 0.1`

```
straight: control = 0.3 × Beta + 0.7 × PID
turning:  control = 0.7 × Beta + 0.3 × PID
```

Output is clamped: `steer ∈ [−1, 1]`, `throttle ∈ [0, 0.75]`, `brake ∈ [0, 1]`.

**Brake/throttle split:**
```
acc ≥ 0  → throttle = min(acc, max_throttle), brake = 0
acc < 0  → throttle = 0, brake = |acc|
if brake < 0.05: brake = 0
```

---

### 5.2 LAV — Learning from All Vehicles

**Reference:** [Chen et al., 2022, "Learning from All Vehicles"](https://arxiv.org/abs/2203.11934)

**Original codebase:** `leaderboard/team_code/lav/`

**Checkpoints (all in `/workspace/leaderboard/team_code/lav/weights/`):**

| File | Component | Version |
|------|-----------|---------|
| `seg_1.th` | ERFNet segmentation | v1 |
| `lidar_v2_7.th` | PointPillarNet detection | v2 iter 7 |
| `bra_v2_9.th` | Brake predictor | v2 iter 9 |
| `uniplanner_v2_7.th` | UniPlanner trajectory | v2 iter 7 |
| `bev_v2_64.th` | BEV expert planner | v2, 64 features |

#### Sensors

| Sensor | id | Position (x,y,z) | Resolution | FOV | Yaw |
|--------|-----|-----------------|------------|-----|-----|
| `sensor.speedometer` | EGO | — | — | — | — |
| `sensor.other.gnss` | GPS | (0, 0, 2.4) | — | — | — |
| `sensor.other.imu` | IMU | (0, 0, 2.4) | — | — | — |
| `sensor.lidar.ray_cast` | LIDAR | (0, 0, 2.4) | — | — | 0° |
| `sensor.camera.rgb` | RGB_0 | (1.5, 0, 2.4) | 256×288 | 64° | −60° |
| `sensor.camera.rgb` | RGB_1 | (1.5, 0, 2.4) | 256×288 | 64° | 0° |
| `sensor.camera.rgb` | RGB_2 | (1.5, 0, 2.4) | 256×288 | 64° | +60° |
| `sensor.camera.rgb` | TEL_RGB | (1.5, 0, 2.4) | 480×288 | 40° | 0° |

#### Pipeline (execution order)

**Localisation:**
1. `EKFEgoLocalizer` — kinematic bicycle model EKF fusing GPS + IMU + speed. Parameters:
   `lf=1.477531 m`, `lr=1.3936 m` (front/rear axle to CoG),
   `gnss_noise=5×10⁻⁶`, `compass_noise=1×10⁻⁷`, `max_steer=70°`, `freq=20 Hz`.

**Perception — RGB:**
2. `MultiCameraToTorchBatch` — stacks RGB_0, RGB_1, RGB_2 → (3, 288, 256) float tensor batch
3. `LAVRGBSegmentationRunner` — ERFNet, outputs seg maps for channels [4, 6, 7, 10]

**Perception — LiDAR:**
4. `LidarVehicleBodyFilter` — removes points in ego footprint:
   x ∈ (−2.4, 0), y ∈ (−0.8, 0.8), z ∈ (−1.5, −1.0)
5. `PointPaintingModule` — projects LiDAR into each segmentation map, appends 4 semantic
   channels. Camera geometry: `cam_yaws=(−60°, 0°, +60°)`, `lidar_xyz=(0,0,2.4)`,
   `cam_xyz=(1.5,0,2.4)`, `rgb_h=288`, `rgb_w=256`, `fov=64°`
6. `TemporalLidarAccumulator` — ego-motion-compensated 3-frame stacking:
   `num_frame_stack=2`, `gap=5` (every 5th frame), `concat_with_prev=True`.
   Appends one-hot time encoding of length `num_frame_stack+1=3`.

**Feature pipeline:**

*Input to LiDAR model:* 11 features per point (xyz + intensity + 4 painted + 3 time one-hot).
After `PointPillarNet.decorate()` adds 5 features (3 cluster-offset + xp + yp): **16 total**.

7. `LAVLiDARModelRunner` — `PointPillarNet` + `ConvBackbone` + 4 heads.
   BEV: min_x=−10, max_x=70, min_y=−40, max_y=40 (@ 4 ppm → **320×320 canvas**).
   `num_features=[64, 64]`.

8. `BEVHeatmapNMS` — maxpool NMS (kernel=7) on center heatmap. Min score=0.1, max_det=15.
   Ego filter: radius 2 px around pixel (160, 280).
   Size filter: `(i==1 and w < 0.4) or h < 0.8` (in metres, ppm=4).
   Output: `[(x, y, w, h, cos, sin), ...]` per class in BEV pixel coordinates.

**Brake prediction:**
9. `HorizontalCameraConcat` — concatenates RGB_0,RGB_1,RGB_2 horizontally
10. `LAVBrakePredictionRunner` — dual ResNet18 + cross-attention on wide + telephoto crops.
    TEL_RGB bottom-cropped by 96 px.

**Planning:**
11. `LAVUniPlannerRunner` — GRU-based multi-command trajectory planner.
    - Crop size: 96 px, feature_x_jitter=1.5, feature_angle_jitter=20°
    - `num_plan=20` waypoints, `num_plan_iter=5` refinement iterations, `num_cmds=6`
    - Lane-change state machine: command 4 or 5 only activates after **300 consecutive frames**
    - Outputs: ego plan (20×2), cast trajectories for other agents, command probabilities

12. `LAVCollisionCheck` — checks if predicted other-vehicle trajectories intersect ego plan:
    `dist_threshold_static=1.0 m`, `dist_threshold_moving=2.5 m`, `cmd_thresh=0.2`

**Control:**
13. `WaypointTrackingPID` — command-conditioned PID.
    Aim points by command: `[4, 4, 4, 3, 6, 6]` (waypoint index into 20-step plan).
    Speed ratios: `[0.8, 0.8, 0.8, 0.6, 0.8, 0.8]`.
    `turn_KP=0.8, turn_KI=0.5, turn_KD=0.2`. `speed_KP=5.0, speed_KI=0.5, speed_KD=1.0`.
    `brake_speed=0.2 m/s`, `max_throttle=0.8`.

14. `EmergencyBrakeOverride` — applies: brake if collision flag or brake_pred > 0.1,
    caps speed to 35 km/h, anti-stuck after `stop_limit=600` frames.

#### PointPillarNet architecture detail

```
DynamicPointNet:
  net: Linear(16, 64) → BN → ReLU → Linear(64, 64) → BN → ReLU
  scatter_max: max-pool features over pillar (pure PyTorch, replaces torch_scatter)

PointPillarNet:
  decorate(): appends (cluster_xyz, xp, yp) → 11+5=16 features
  grid_locations(): filters to BEV bounds, computes pixel coords
  pillar_generation(): unique pillar coords + inverse indices
  scatter_points(): scatter features to 320×320 canvas

ConvBackbone:
  conv1: Conv2d(64,64,3,stride=2) + 4× Conv2d(64,64,3,stride=1) + BN+ReLU
  conv2: Conv2d(64,128,3,stride=2) + 6× Conv2d(128,128,3,stride=1) + BN+ReLU
  conv3: Conv2d(128,128,3,stride=2) + 6× Conv2d(128,128,3,stride=1) + BN+ReLU
  upconv1: ConvTranspose2d(64,128,1,stride=1)
  upconv2: ConvTranspose2d(128,128,4,stride=2)
  upconv3: ConvTranspose2d(128,128,4,stride=4,padding=1,out_padding=2)
  output: cat[u1, u2, u3] → 384 channels
```

Detection heads (applied to 384-channel BEV feature map):
```
Head(384→2→output_size): Conv2d(384,64,3) → BN → ReLU → ConvTranspose2d(64,2,3,stride=2)
center_head: 2-channel heatmap (vehicle, pedestrian)
box_head:    2-channel size map (width, height)
ori_head:    2-channel orientation (cos, sin)
seg_head:    3-channel segmentation + sigmoid
```

---

### 5.3 InterFuser — Interpretable Multi-sensor Fusion Transformer

**Reference:** [Shao et al., 2023, "Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer"](https://arxiv.org/abs/2207.14024)

**Original codebase:** `leaderboard/team_code/interfuser/`

**Checkpoint:** `interfuser.pth.tar` (key: `state_dict`, loaded with `strict=False`)

#### Sensors

| Sensor | id | Position (x,y,z) | Resolution | FOV | Yaw |
|--------|-----|-----------------|------------|-----|-----|
| `sensor.camera.rgb` | rgb | (1.3, 0, 2.3) | 800×600 | 100° | 0° |
| `sensor.camera.rgb` | rgb_left | (1.3, 0, 2.3) | 400×300 | 100° | −60° |
| `sensor.camera.rgb` | rgb_right | (1.3, 0, 2.3) | 400×300 | 100° | +60° |
| `sensor.lidar.ray_cast` | lidar | (1.3, 0, 2.5) | — | — | −90° |
| `sensor.other.imu` | imu | (0, 0, 0) | — | — | 0° |
| `sensor.other.gnss` | gps | (0, 0, 0) | — | — | 0° |
| `sensor.speedometer` | speed | — | 20 Hz | — | — |

#### Image preprocessing

Matches `create_carla_rgb_transform()` from the original agent exactly:

| Stream | Input resolution | Resize (W×H) | Crop | Output |
|--------|-----------------|--------------|------|--------|
| rgb (front) | 800×600 | 341×256 | 224×224 | (1,3,224,224) |
| rgb_left | 400×300 | 195×146 | 128×128 | (1,3,128,128) |
| rgb_right | 400×300 | 195×146 | 128×128 | (1,3,128,128) |
| rgb_center | 800×600 | none | 128×128 | (1,3,128,128) |

All images normalised: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

#### LiDAR preprocessing

```
lidar_xyz = input[:, :3]      # xyz only (flip_y=True → y *= −1)
transform_2d_points(xyz, π/2 − compass, −pos[0], −pos[1], ...)
lidar_hist = lidar_to_histogram_features(transformed_xyz, crop=224)
```

LiDAR histogram cached every 2 frames (`reuse_every_n=2`, warmup: always update for first 4).

#### Input tensor assembly

```
measurements = [cmd_one_hot(6), speed]  → float32 (1, 7)
target_point  = [x, y]                  → float32 (1, 2)
lidar_hist    = histogram               → float32 (1, C, 224, 224)
```

`measurements` construction: `cmd_one_hot[command − 1] = 1` (1-based command → 0-based index),
then `append(speed_m_s)`.

#### Model architecture summary

**InterFuser (transformer-based multi-modal fusion):**

Backbone encoders (one per modality stream):
- Front RGB: `HybridEmbed` (ResNet → patch embed), output tokens (56×56 → flattened)
- Left/right RGB: separate `HybridEmbed` encoders
- Center RGB: separate `HybridEmbed`
- LiDAR: `HybridEmbed`

Position encodings: sinusoidal, per-modality view embeddings.

Transformer encoder: multi-layer cross-attention over concatenated multi-modal token sequence.

Decoder query embeddings (`query_embed`): `(num_queries, B, embed_dim)`.
Outputs `hs` of shape `(B, N_tokens, embed_dim)`:
- `hs[:, :400]` → `traffic_feature` (20×20 BEV agent grid)
- `hs[:, 400]` → shared feature for junction/traffic_light/stop_sign heads
- `hs[:, 401:411]` → waypoint prediction features (10 tokens)

Output heads:
- `traffic_pred_head`: BEV occupancy (400 × 7) — `[present, x, y, cos_h, sin_h, speed, extent]`
- `waypoints_generator`: GRU or linear decoder → 10 future waypoints × 2
- `junction_pred_head`: Linear(embed_dim, 2) — class 0=not-junction, class 1=junction
- `traffic_light_pred_head`: Linear(embed_dim, 2) — class 0=red/yellow, class 1=green
- `stop_sign_head`: Linear(embed_dim, 2) — class 0=no-sign, class 1=stop-sign

**Softmax extraction (critical for correct interpretation):**

```python
is_junction     = softmax(logits, dim=1)[0, 0]  # p(not at junction)
traffic_light   = softmax(logits, dim=1)[0, 0]  # p(red or yellow light)
stop_sign       = softmax(logits, dim=1)[0, 0]  # p(no stop sign)
```

Training label conventions (from `carla_dataset.py`):
- `is_junction`: 1 if at junction (class 1 = junction, class 0 = not junction)
- `traffic_light`: 0 if red light present, 1 if absent — **note: class 0 = red/yellow**
- `stop_sign`: `int(affordances['stop_sign'])` — 1 if stop sign, so class 0 = no stop sign

#### Traffic-meta tracking

`TrafficMetaTracker` wraps the original `Tracker` class:
- Updates on even global steps and first 4 steps (`update_every_n=2`)
- `momentum=0.0` — no EMA smoothing (each update fully replaces previous)
- GPS coordinate system: planner-calibrated (same as route planner's `pos` output)
- Timestamp: `global_step // 2` (tracks at half the control frequency)

#### InterfuserController

Direct import of the reference `InterfuserController` class (unchanged). Key logic:

| Condition | Action | Anti-deadlock |
|-----------|--------|---------------|
| `d_0 < max(3, speed)` | `brake=True, desired_speed=0` | `stop_steps > 1200` → 12 forced frames |
| `junction > 0 and traffic_light > 0.3` | `brake=True` | `red_light_steps > 1000` → 80 unblocked frames |
| `stop_sign < 0.6` | `in_stop_sign_effect=True` (3 cycles of 2m each) | `stop_steps > 1200` |

Collision detection threshold: `detect_threshold=0.04`, `collision_buffer=[2.5, 1.2] m`.
Max speed: `5.0 m/s`. `stop_limit=1200 frames`.

**Known behaviour:** On CARLA routes with dense stop signs (virtually all town intersections),
the 3-cycle brake sequence and slow `block_stop_sign_distance` clearance (decreases by
`0.05 × speed`, so effectively zero during braking) causes repeated `stop_steps > 1200`
anti-deadlock triggers (12 frames of forced movement every ~1200 stopped frames).
This is model-inherent — the reference `InterfuserAgent` exhibits identical behaviour.

---

### 5.4 CILRS — Conditional Imitation Learning

**Paradigm:** conditional imitation learning (the classic CARLA baseline).
**Source:** `autonomousvision/transfuser` @ `cvpr2021` branch.
**Vendored:** `leaderboard/team_code/cilrs/` — `model.py` (`ImageCNN`/`Controller`/`CILRS` verbatim
+ a `CILRSInterface` dict-in/dict-out adapter), `modules.py` (`CILRSControl`), `config.py`.
**Checkpoint:** `best_model.pth` (48 MB, ResNet-18 encoder; bare `OrderedDict`, `strict=True`);
`fetch_weights.sh` pulls it from the transfuser model-zoo S3 bundle.
**Config:** `configs/cilrs.yaml` (8 steps).

**Sensors:** front RGB 400×300 fov100 at (x=1.3, z=2.3). *(The reference's 3 side/rear cameras are
dropped — the model consumes only the front camera, so they add render/crash exposure for nothing.)*

**Pipeline:** `ExtractCameraRGB → ExtractSpeed → ExtractGNSS → RoutePlannerNextCommand →
ImageHWCToTorchCHW → TorchModelRunner(CILRSInterface) → CILRSControl`.

**Load-bearing detail:** the image is fed as float32 in **[0,255] with NO `/255`** — ImageNet
normalization happens *inside* the encoder (`ImageCNN(normalize=True)`). `command` is the raw
1-based RoadOption value; the `Controller` selects branch `command−1`. Control:
`steer = 2σ−1`, `throttle = σ·0.75`, `brake = σ`, then reference brake gating
(`brake<0.05→0`; `throttle>brake→brake=0`).

**Performance note (verified 2026-07-07):** CILRS's mean `score_composed` (≈39) is the roster
floor — well below the others — but this is *genuine model weakness, not an integration artifact*.
Audited end-to-end after the low score: channel order (`bgr_to_rgb`, the same helper the working
agents use), the [0,255]/ImageNet-norm path (train-eval consistent), brake/throttle gating (verbatim
`cilrs_agent.py:209-210`), and the `next_command` source (1-based RoadOption — the *same*
`RoutePlannerNextCommand` output the known-good TCP consumes) all check out; the checkpoint loads
`strict=True` with no missing/unexpected keys. The dominant failure is **timeout (59%)** — slow,
conservative driving — not the collisions/route-deviations a perception bug produces, and CILRS
still scores 100 on 15 routes (perception demonstrably works). ≈39 DS is in-line-to-above published
CILRS baselines (~7–30). This is the *opposite* signature to the historical InterFuser color bug
(which caused a **uniform** collapse); CILRS's bimodal functional-but-weak profile is what a real
weak baseline looks like, and it usefully anchors the low end of the agent comparison.

### 5.5 NEAT — Neural Attention Fields

**Paradigm:** implicit BEV attention fields → waypoints (camera-only).
**Source:** `autonomousvision/neat` (`MultiTaskAgent`).
**Vendored:** `leaderboard/team_code/neat/` — `architectures/` (`AttentionField`, `encoder.py`,
`decoder.py`, `controller.py`), `modules.py` (`NEATImagePreprocess`, `NEATModelRunner`,
`NEATControlPID`), `config.py`.
**Checkpoints:** `best_encoder.pth` (113 MB) + `best_decoder.pth` (13 MB); fetched from the AVG S3
bundle.
**Config:** `configs/neat.yaml` (13 steps).

**Sensors:** 3 RGB cameras (yaw 0/−60/+60), 400×300 fov100 at (x=1.3, z=2.3) — **all
model-consumed** (this is the heaviest of the new agents' sensor rigs). *(The save-only `rgb_front`
800×600 and `bev` cameras are omitted.)*

**Architecture:** encoder = ResNet-34 + AdaptiveAvgPool(8,8) + velocity embed + learnable pos-emb
+ 2-layer transformer (n_embd=512, n_head=4). Decoder = iterative attention field
(`attention_iters=2`, `onet_blocks=5`, `num_class=5`). `plan()` samples a plan grid and iteratively
re-weights attention, accumulating waypoint offsets **in place across ticks** (deliberate — mirrors
the released agent). Waypoints → `control_pid` (aim_dist=4, turn/speed PID). Uses
`resnet34(pretrained=False)` (weights come from the checkpoint; avoids a fragile runtime torch-hub
download).

### 5.6 Roach — RL-Coached Imitation Learning

**Paradigm:** camera IL policy distilled from a privileged RL coach — the **sensorimotor** agent,
**not** the BEV coach (which needs rendered birdview input we don't provide).
**Source:** `zhejz/carla-roach` (`agents/cilrs`, `CoILICRA`).
**Vendored:** `leaderboard/team_code/roach/` — `cilrs_model.py` (`CoILICRA` verbatim + a
`RoachILPolicy` adapter), `networks/` (`resnet`/`fc`/`join`/`branching`), `modules.py`
(`RoachRouteTarget`, `CilrsStateVector`, `CilrsActionFromBranches`).
**Checkpoint:** `cilrs_ckpt_lk_12uzu2lu.pth` (287 MB, L_K LeaderBoard run;
`{policy_init_kwargs, policy_state_dict}` format); fetched from Weights & Biases (public API).
**Config:** `configs/roach.yaml` (11 steps).

**Sensors:** front RGB 900×256 fov100 at (x=−1.5, z=2.0) + imu/gnss/speedometer.

**Measurements:** state = `[forward_speed/12, loc_in_ev.x, loc_in_ev.y]` (S=3), where `loc_in_ev`
is the ego-frame vector to the next route GPS target (Mercator `gps_to_location` +
`vec_global_to_ref`, yaw=`compass−90°`). *Verified 0.0 m error against Roach's own carla transform;
deliberately does **not** reuse `RoutePlannerNextCommand`, whose planar frame differs.*

**Architecture:** ResNet-34 perception + measurement MLP → join → 6 command branches,
`action_distribution=beta_shared`. Command clamps + selects a branch; Beta **mode** → action via
`_get_action_beta` (`[-1,1]`); `process_act` maps `acc≥0→throttle`, `acc<0→brake`. Control via
`ControlFromAccSteer` + `ClampControl`.

---

## 6. Data Collection Pipeline

### Per-frame sensor saving

When `COLLECT_DATA=1`, `ConsolidatedAgent._save_sensor_data()` runs **before** pipeline
inference each tick. This guarantees data is saved even if inference crashes (allowing diagnosis
via `global_steps=0` in run_summary.json).

Data is saved to:
```
$DATASET_DIR/{agent}/{weather_N}/map_{NN}/{route_stem}/
  {sensor_id}/
    {frame_number}.npy   (or .jpg for cameras if configured)
  metadata.json
  run_summary.json       (written at destroy())
```

Sensor types saved by default: all sensors in the agent's `sensors()` list.

### WekaFS-friendly sharded writes (default since 2026-07-15)

The naïve layout above writes ~7 tiny files per tick (one per sensor) — tens of thousands of
small files per route. On the cluster's **WekaFS** `/scratch` each create is a metadata op + file
lease; that small-file storm is a direct contributor to the Weka `HangingIos`/node-fencing that
crashed the nodes (§9). To avoid it, `ConsolidatedAgent` now writes sensor frames to a **node-local
staging dir** (`/dev/shm`, never Weka) and rolls them into a few large **tar shards** on `/scratch`
(`leaderboard/team_code/sensor_stager.py`):
```
$DATASET_DIR/{agent}/{weather_N}/map_{NN}/{route_stem}/
  shards/shard_00000.tar        # each holds {sensor_id}/{frame}.{ext}
  shards/shard_00001.tar        # rolled every HPC_CARLA_SHARD_SIZE ticks (default 64)
  shard_manifest.json
  metadata.json / run_summary.json / results.json   ← still direct small writes to /scratch
```
This turns ~77k small-file/lease ops per route into a few dozen large sequential writes (the access
pattern Weka handles well). Knobs: `HPC_CARLA_SHARD_SENSORS=0` disables it (legacy direct writes),
`HPC_CARLA_SHARD_SIZE`, `HPC_CARLA_STAGE_ROOT`. It **fails safe** — if node-local staging isn't
writable it reverts to direct `/scratch` writes. Bounded loss on a mid-route crash = only the
current partial shard (< shard_size ticks). Crucially, **`results.json` stays a live file on
`/scratch`** so the per-route harvester (§9) is unaffected. Reconstruct the classic per-frame layout
with `tools/unpack_shards.py`.

### run_summary.json

Written on `destroy()`. Contains:

```json
{
  "run_id": "SLURM_JOB_ID",
  "job_id": "HPC_CARLA_JOB_ID",
  "run_tag": "route_stem",
  "node": "hostname",
  "gpu_id": "GPU_ID",
  "global_steps": N,
  "frames_saved_by_sensor": {"GPS": N, "LIDAR": N, ...},
  "data_collection_started_at": "ISO8601",
  "data_collection_ended_at": "ISO8601"
}
```

`global_steps=0` with 1 frame per sensor indicates a crash on the first inference step.

### Leaderboard results.json

The CARLA evaluator writes `results.json` to `$CHECKPOINT_ENDPOINT` = `$SAVE_PATH/results.json`.
After each job, `_finish()` parses this and records into `completed_jobs.json`:

```json
{
  "score_composed": float,    // mean composite driving score (0–100)
  "score_route": float,       // mean route completion (0–100)
  "score_n_routes": int,      // number of route evaluations
  "route_statuses": [str]     // "Completed" / "Failed" per route
}
```

Leaderboard composite score formula (CARLA 1.0):
```
score_composed = score_route × score_penalty
```
where `score_penalty` is a product of infraction multipliers (collision with vehicle,
collision with layout, red light violation, stop sign violation, off-road).

---

## 7. Job Scheduling and Difficulty Estimation

### Queue structure

Each job entry in `job_queue.json`:

```json
{
  "id": int,
  "agent": "tcp" | "lav" | "interfuser",
  "weather": int (0–20),
  "route": "routes_town04_long.xml",
  "town": "04",
  "status": "pending" | "running" | "completed" | "failed" | "skipped",
  "attempts": int,
  "gpu": int,
  "node": str,
  "start_time": "ISO8601Z",
  "end_time": "ISO8601Z",
  "duration": int (seconds)
}
```

### Scheduling priority

`_reserve_next` sorts pending jobs by this key (ascending) and the worker takes the first:

```
(attempts, coverage_deficit, coverage_count, −difficulty_score, agent, −estimated_runtime_s)
```

1. **Fewest attempts first** — retries deprioritised relative to fresh jobs.
2. **Illumination coverage first** — `0` while the job's `(agent, illumination-bin)` is below
   `COVERAGE_QUOTA` finished jobs, else `1` (see *Illumination-stratified coverage* below).
3. **Least-covered bin first** (coverage phase only) — interleaves noon/sunset/night.
4. **Highest difficulty first** — hardest `(route+scenario+weather)` early, so `prune` can drop
   the easier same-route variants as redundant.
5. **Agent** (tiebreak) — interleaves agents within a difficulty tier so none monopolises the
   fleet (replaced the old hard-coded `{interfuser:0,…}` priority that starved the newer agents).
6. **Longest estimated runtime** (tiebreak) — fills GPU time efficiently.

**Illumination-stratified coverage.** Pure hardest-first marches down the `_WEATHER_DIFF`
ranking; on the tiny/short suites (small route+scenario difficulty) it effectively sorts by
weather alone, so the completed sample collapses onto the darkest+rainiest presets. On the A100
run **100% of the first completions were night** (weathers 17–20), leaving illumination
unsampled and its per-model sensitivity unidentifiable (see *Multi-axis difficulty*).
`COVERAGE_QUOTA` (env, default 3) guarantees that many finished jobs per `(agent,
illumination-bin ∈ {noon, sunset, night})` before reverting to pure hardest-first;
`COVERAGE_QUOTA=0` restores the original sort exactly. Bright bins are front-loaded and are the
faster / less-crash-prone jobs, so they yield data sooner.

### Difficulty scoring

**Route geometric difficulty** (memoised per XML file):

```
geo_score(route) = mean over routes in file of:
    sharp_turns × 2.0 + path_length_m / 500 + total_heading_change_deg / 180
```

where:
- `sharp_turns` = count of consecutive-waypoint heading jumps > 45°
- `path_length_m` = Euclidean sum of inter-waypoint distances
- `total_heading_change_deg` = sum of absolute heading deltas (wrapped to [0, 180])

**Scenario density** (memoised per town):

```
scen_score(route) = mean over routes of:
    unique_hit_cells × mean_type_weight × 0.25
```

where:
- `unique_hit_cells` = unique 20 m-grid cells containing at least one scenario trigger
  within 25 m of any route waypoint
- `mean_type_weight` = 2.64 (mean of all 7 type weights)
- `SCALE = 0.25` (calibrated so scenario contribution is comparable to geometry score)

**Weather difficulty:**

Lookup in `_WEATHER_DIFF[0..20]` — see Appendix A.

**Total job difficulty:**

```
difficulty = geo_score + scen_score + weather_diff
```

**Observed score ranges** (from offline computation on all route files):
- Highest: `routes_town04_long.xml` = 73.26 (geo=29.69, scen=43.57)
- Lowest: `routes_town07_short.xml` = 0.14 (geo=0.14, scen=0.00)

### Redundant pruning

`ContinuousManager.prune_redundant()` marks pending jobs as `skipped` if a harder variant
for the same `(agent, route)` pair has already completed. "Harder" is defined by the same
difficulty scoring formula. Invocable via `python3 manage_continuous.py prune [--dry-run]`.

### Difficulty validation — and why the scalar score does not survive

The scalar difficulty score is only useful if it predicts agent performance: a harder-scored route
should yield a *lower* driving score. `tools/difficulty_validation.py` tests this by correlating
per-job difficulty against `score_composed`, with `tools/harvest_results.py` supplying the
fine-grained per-route sample (§6/§9).

**At small n it looked validated; at scale it does not.** An early per-route check (n ≈ 204) gave
the right sign and apparent significance — InterFuser Spearman ρ = −0.642, TCP ρ = −0.311. Those
did **not** survive the larger harvest. At **n = 1,648** the pooled correlation is ≈ 0 (the score
*washes out*), and the per-agent picture is inconsistent rather than merely weaker:

| Agent | ρ at n≈204 | ρ at n=1648 | Verdict |
|-------|-----------|-------------|---------|
| InterFuser | −0.642 (sig) | ≈ −0.15 (n.s.) | collapsed to noise |
| TCP | −0.311 (sig) | ≈ +0.14 (**sign flipped**, sig) | *anti*-correlated |

The TCP sign flip is not a rounding effect — it is real and diagnostic: TCP's cautious control
**times out in dense grid towns (Town02)**, which the geometry proxy rates *easy*, so higher
"difficulty" as scored actually tracks *higher* TCP success. **This is a paper reframe, not a
number swap:** the honest conclusion is that a single route+scenario+weather scalar is the wrong
model, for two independently diagnosed reasons (next subsection). It is reported this way
deliberately — the old n≈204 figures should not be cited.

Note the methodological point that still holds: **per-route granularity is what makes any
validation possible at all** — per-*file* aggregation (n = 13) is far too coarse. The harvester's
per-route recovery is the enabling instrument; the negative result is a property of the *scalar
model*, not of the measurement.

### Multi-axis difficulty and per-model sensitivity

A single scalar difficulty **washes out** against performance (pooled Spearman **+0.036**), a
lesson echoing a sister project (illumination-difficulty for AV scene scoring). Two independent
causes, both since diagnosed.

**1 — Illumination dominates, against a low ceiling.** Decomposing the 0–20 weather ordinal into
physical axes (`tools/weather_axes.py`: `illum_dark / precip / road_water / cloud / fog`; Night
params exact from `consolidated_agent.py`) and fitting a per-agent noisy-OR
`P(fail)=1−exp(−Σ λ_j x_j)` (`tools/sensitivity_matrix.py`, Newton MLE + Hessian CIs; validated on
synthetic ground truth in `tools/noisy_or_sanity.py`) shows the recoverable signal is
**illumination** — "dark = hard." A cross-validated head-to-head
(`tools/difficulty_model_comparison.py`) puts the old scalar at AUC ≈ **0.53** and a parsimonious
`illumination + geometry` model at ≈ **0.62**, near the **~0.65 ceiling** the sister project found
robust even to reasoning foundation models; the extra weather axes are noise. An independent
second-domain replication.

**2 — The geometry & scenario terms are degenerate.** The `_tiny`/`_short` route files (the entire
current sweep) store each route as **2 waypoints — endpoints only**; the driven path is
interpolated at runtime by `GlobalRoutePlanner`. So `route_difficulty`/`scenario_difficulty`, which
parse those XMLs, measure only endpoint displacement — near-noise. Two of the scalar's three inputs
are thus uninformative for the routes being collected. (Only `_long` routes carry real geometry,
16–75 waypoints.)

**Map is a dominant, missed factor.** Fail rates split hard by town — Town02 (dense grid) 72–100%
vs Town05 (open) 10–19% — yet the model scores all towns ~identically. The fix is a map-intrinsic
**urban-density axis** (`tools/map_density.py`): junctions per km of road / per km², mean segment
length, curvature — read from **OpenDRIVE**, offline from a `.xodr` or in-sim via
`carla.Map.to_opendrive()`, so **custom user maps score by the same logic with no hardcoding**.
Validated on the collected towns: `junctions_per_road_km` ranks Town02 (4.00) > Town01 (2.85) >
Town05 (2.27), matching observed difficulty.

**Per-route map density is now computed (not blocked).** `tools/route_map_density.py` reproduces the
leaderboard's runtime interpolation (`GlobalRoutePlanner` over the map topology) **offline** — it
builds `carla.Map(name, xodr)` from the town `.xodr` with **no server and no GL** (so it sidesteps
the segfault entirely) — and measures, along the true driven path of all 2,345 routes: interpolated
length, junction waypoints, distinct junctions traversed, and curvature. Joined to the 1,648
route-evals (100% match), the useful feature is the **count of distinct intersections driven
through**, `n_distinct_junctions` (pooled Spearman **−0.295** vs `score_composed`) — a real per-route
difficulty axis and stronger than the old scalar (~0). Note the *rate* form washes out
(`junctions_per_km` +0.022); it is the absolute intersection **count** that predicts, with curvature
`heading_deg_per_km` a weaker second (−0.186). This supersedes the earlier "needs the sim / endpoint
approximation too coarse" limitation and gives the noisy-OR model a geometry input that survives at
n=1648.

**Per-agent nuance.** Difficulty predicts failure for the condition-dependent agents (InterFuser /
NEAT / Roach, near the ceiling) but **not** CILRS (fails near-uniformly) or TCP (failures run
*opposite* to route geometry — its cautious control times out in dense Town02, which the geometry
proxy rates easy). Net: the real difficulty signal is **illumination + map urban-density**, not the
geometry/scenario terms the model was built around.

### Runtime estimation

`collection_state/runtime_estimates.json` stores empirical runtimes:
- Long routes: 5400 s
- Short routes: 1800 s
- Tiny routes: 3600 s (default)
- Updated by `optimize_runtime_estimates()` after ≥2 completed runs per combination

---

## 8. Dataset Structure and Statistics

### Directory layout

```
dataset/
  {agent}/
    weather_{N}/
      map_{NN}/
        {route_stem}/
          {sensor_id}/        ← per-sensor directory
            {frame}.npy       ← data array
          metadata.json
          run_summary.json
```

### Current collection status (as of 2026-07-15)

**Headline: 1,648 per-route evaluations across the 5 productive agents, ≈86% recovered from
"failed" jobs** by `tools/harvest_results.py` (which unions `job_queue.json` with
`completed_jobs.json` and reads every expected `results.json`, §6/§9). File-level completion
accounting would report only the ~14% from cleanly-finished jobs; the per-route harvest is what
turns crash-truncated suites into usable data. **The route-eval is the reportable unit**, not the
completed job.

Per-agent mean `score_composed` from the canonical harvest (`tools/harvest_results.py` over the live
`completed_jobs.json` + `job_queue.json`, n = 1,648):

| Agent | n (route-evals) | Mean score_composed | Mean route completion | Character |
|-------|----------------:|--------------------:|----------------------:|-----------|
| InterFuser | 140 | **88.7** | 88.7 | highest mean; camera+LiDAR fusion (smallest n — see below) |
| NEAT | 223 | 86.5 | 90.3 | strong; neural attention fields |
| TCP | 269 | 86.3 | 87.1 | strong; but fails by *timeout* in dense towns (§7) |
| Roach | 539 | 86.2 | 93.4 | strong; RL-coached |
| CILRS | 477 | 41.4 | 55.4 | genuine weak baseline (audited — not an integration bug, §5.4) |

The four non-CILRS agents cluster tightly at **86–89**; CILRS sits far below as the intended weak
baseline. **Caution on ranking within the top cluster:** the ordering is not robust — InterFuser
leads on the *smallest* sample (n = 140) and the top four are within ~2.5 points. An earlier n ≈ 204
harvest had InterFuser *last* of the five (71.8) and Roach top (90.4); those figures were themselves
small-sample artifacts that reshuffled at scale — the same lesson §7 draws for the difficulty
correlations. Cite the n = 1,648 means with their `n`, and treat the CILRS-vs-rest gap (not the
intra-cluster order) as the load-bearing separation.

**Validation smokes (post-resilience):** interfuser 8/8 and tcp 8/8 routes; CILRS/NEAT/Roach 4/4
each — zero agent-code errors, zero GPUs parked. LAV 0/4 (server crash at `load_world`). These
confirm the five productive agents are integration-clean; residual failures are host/server, not
agent code.

**Historical (earliest partial single-agent run, retained for frame-count reference):** InterFuser
only — 16 routes × 12 weather over Town03/Town04, ~179,520 frames, **3,413,973** `.npy` files on
disk.

### Run history and current disposition

The active sweep is **~2,300 jobs** (5 agents × 22 route files × 21 weather; LAV excluded, §9),
scheduled **job-first** so metrics accumulate interleaved across agents — if the run is cut short,
every agent has coverage. Because the sim runs 8–12× slower than real time (§2), a full cycle does
not drain; it is a continuous harvest.

Recent runs did not proceed uninterrupted. Under sustained load the assigned A100 nodes degraded
(§9): pod17 repeatedly **drained** ("Kill task failed" on an unkillable CARLA), and pod09 went
fully **"Not responding."** The `park-on-unkillable` mitigation (§9, commit 5037b4b) contains the
*drain* mode by isolating a wedged GPU instead of letting it take the node down, but it cannot
prevent a full kernel-level node crash. After repeated pod crashes the run was **paused pending
admin stabilization** (node reboot + a SLURM `UnkillableStepProgram`) — its current state. The
1,648 route-evals above are what the resilience + harvest machinery preserved *through* those
outages, which is itself the reliability evidence (§9).

### InterFuser detailed frame statistics

Observed frames per run (`global_steps`): 11,150–11,342 (mean ~11,220).
Each run covers all routes in the route XML file (routes_town04_long has 5 sub-routes).

### Sensors saved per InterFuser run

| Sensor key | Content | Shape per frame |
|------------|---------|-----------------|
| `gps` | GNSS lat/lon | (2,) |
| `lidar` | 3D point cloud | (N, 4) |
| `imu` | IMU data | (6,) |
| `speed` | Vehicle speed | scalar |
| `rgb` | Front camera (800×600) | (600, 800, 4) |
| `rgb_left` | Left camera (400×300) | (300, 400, 4) |
| `rgb_right` | Right camera (400×300) | (300, 400, 4) |

### Sensors saved per LAV run (when available)

| Sensor key | Content |
|------------|---------|
| GPS | GNSS (2,) |
| LIDAR | 3D point cloud (N, 4) |
| IMU | IMU data |
| EGO | Speed scalar |
| RGB_0, RGB_1, RGB_2 | Three cameras (256×288×4) |
| TEL_RGB | Telephoto camera (480×288×4) |

### Sensors saved per TCP run (when available)

| Sensor key | Content |
|------------|---------|
| `rgb` | Front camera (900×256×4) |
| `gps` | GNSS (2,) |
| `imu` | IMU |
| `speed` | Speed scalar |

---

## 9. Implementation Challenges and Solutions

This section documents the non-trivial engineering problems encountered when re-implementing each
agent as a modular pipeline. These may be relevant as related-work discussion or as evidence of
the complexity of faithful agent reimplementation.

### TCP

#### Bug 1: Beta-distribution parameter misinterpretation

**Symptom:** Car steered with near-zero steer regardless of model predictions.

**Root cause:** `mu_branches` and `sigma_branches` from the model are produced by `nn.Softplus`
and are therefore the α and β parameters of the Beta distribution with range (0, ∞). The
initial pipeline implementation clamped them to [0, 1] and applied method-of-moments conversion
(treating them as mean/sigma), producing `dist.mean ≈ 0.5` → steer ≈ 0 always.

**Fix:** Replace with `_get_action_beta` mode formula (see §5.1).

#### Bug 2: PID missing angle_target outlier rejection

**Symptom:** Hard turn on straight road — car pulled immediately into barriers.

**Root cause:** The three-angle outlier rejection in `TCP.control_pid()` was missing. Without it,
noisy waypoint angles on straight roads produce large steer commands. The GPS-derived angle
(`angle_target`) must override the waypoint angle when waypoints are noisier than the route.

**Fix:** Implement full three-angle logic with `use_target` condition (see §5.1).

#### Bug 3: PID steer clipping bound

**Symptom:** Steer magnitude capped at 0.25 instead of 1.0.

**Root cause:** `clip_delta` (throttle clipping, 0.25) was mistakenly applied to steer.

**Fix:** Separate steer clip to `[−1, 1]`.

### LAV

#### Bug 1: Missing `torch_scatter` dependency

**Symptom:** `ModuleNotFoundError: No module named 'torch_scatter'` at first inference step.

**Root cause:** `PointPillarNet` used `scatter_mean` and `scatter_max` from `torch_scatter`,
which is not bundled in the CARLA container.

**Fix:** Pure-PyTorch replacements:
- `_scatter_mean`: `scatter_add_` + count normalisation (O(1) per call, any PyTorch version)
- `_scatter_max`: `scatter_reduce_` with `reduce='amax'` (PyTorch ≥1.12) with loop fallback

#### Bug 2: Wrong `num_input` for PointPillarNet

**Symptom:** `RuntimeError: size mismatch for point_pillar_net.point_net.net.0.weight:
copying a param with shape torch.Size([64, 16]) from checkpoint, the shape in current model
is torch.Size([64, 11]).`

**Root cause:** `num_input=11` was set as the raw feature count, but `DynamicPointNet`'s
`nn.Linear(num_input, 64)` receives the **post-`decorate()`** feature count (raw + 5).
Raw=11, decorate adds 5, so `num_input` must be 16.

**Fix:** `num_input: 16` in `lav.yaml` and as the class default.

#### Bug 3: `ExtractLidarXYZ` dropping intensity

**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied (9103×15 and 16×64)`.

**Root cause:** `ExtractLidarXYZ` sliced `arr[:, :3]` (xyz only), discarding intensity. With
only 3 raw features: 3 + 4 painted + 3 time = 10 raw → decorate → 15. Checkpoint expects 16.

**Fix:** Added `num_cols` parameter; `lav.yaml` uses `num_cols: 4` to retain XYZΙ.

#### Bug 4: `PointPillarNet.nx`/`ny` stored as float

**Symptom:** `TypeError: zeros(): argument 'size' must be tuple of ints, but found element
of type float at pos 3`.

**Root cause:** `self.nx = (max_x − min_x) × pixels_per_meter` produces a Python float.

**Fix:** `int((max_x − min_x) × pixels_per_meter)`.

#### Bug 5: BEVHeatmapNMS height filter operator precedence

**Symptom:** False-positive vehicle detections passed to UniPlanner (silent wrong output).

**Root cause:** Reference filter is `if i==1 and w < 0.1*ppm or h < 0.2*ppm:` which Python
parses as `(i==1 and w<0.1*ppm) or (h<0.2*ppm)` — filtering ANY class with h < 0.8 m.
The pipeline wrapped the `or` inside the `i==1` condition, restricting height filtering to
pedestrians only.

**Fix:** `(i == 1 and w < 0.1 × ppm) or h < 0.2 × ppm`.

#### Bug 6: `LAVCollisionCheck` trajectory shape

**Symptom:** `ValueError: The truth value of an array with more than one element is ambiguous`
whenever another vehicle was present (long masked by the server crashes; surfaced once the
resilience work let LAV reach the collision check).

**Root cause:** `LAVCollisionCheck` assumed `trajs` was `(T, 2)` and did
`init_x, init_y = trajs[0,0], trajs[0,1]`; the real shape is `(num_cmds, T, 2)`, so `init_y` became
a `(2,)` array and `if init_y > threshold` raised.

**Fix:** `init_x, init_y = trajs[0, 0]` (unpack the `(x,y)` of cmd-0/step-0), matching
`lav_agent.plan_collide`. *(This fix was itself initially masked by stale container bytecode — see
"Cluster reliability" below — so it only took effect after `PYTHONDONTWRITEBYTECODE` was enabled.)*

### InterFuser

#### No code bugs found

The reimplementation is faithful to the reference. The observed "stuck for thousands of frames
then moves briefly" behaviour is model-inherent — identical to running the original
`InterfuserAgent`. See §5.3 for the full controller analysis.

### Cluster reliability — intermittent CARLA GL segfault (host issue) + resilience

**Symptom:** Persistent CARLA servers segfault (`Signal 11`) intermittently at **GL/EGL context
creation**, uniformly across all GPUs (each server crashes 4–17× per multi-hour run). Before
mitigation this collapsed full runs (e.g. 2 completed of 45).

**Root cause (host, not project code):** a UE4-4.24 / NVIDIA-driver-575 GL-init instability on
these A100 nodes. Ruled out with direct probes (`tools/gl_probe.sh`, `tools/gl_version_probe.sh`):
the NVIDIA GL/EGL stack **is** present and version-consistent (575) inside the container — it is
*not* a missing-lib / Mesa-software-fallback / driver-version-mismatch / render-on-GPU-0 problem.
Per-process `nvidia-smi` confirms each server renders `C+G` on its **own** GPU (so
`CUDA_VISIBLE_DEVICES` + the C+G coupling already pins EGL here — no `-graphicsadapter` needed). The
crash precedes UE4's own logging, so `-stdout` capture shows only the crash handler.

**Mitigation — makes it recoverable, not fatal** (`carla_server_manager.py`,
`persistent_carla_worker.sh`):
- **Clean restart:** before relaunch, kill any lingering/segfaulted server bound to the GPU's RPC
  port (its own process group), and wait for the port/process to free.
- **Boot health-check:** a server counts as up only if the port opens **and** the process is still
  alive a few seconds later (catches segfault-right-after-bind).
- **Retry + park:** retry `CARLA_BOOT_ATTEMPTS` (default 3); if every boot segfaults, **park** the
  GPU — the worker stops pulling jobs (which would fast-fail `rc=3` and burn the queue) and
  periodically re-attempts recovery.
- **Per-GPU HOME isolation** (`/carla_home`): the UE4 instances no longer share `~/.cache` shader
  cache / lock files; first-render is staggered by GPU index.

**Result:** servers still crash but recover — 2-GPU smokes: interfuser 8/8, tcp 8/8, and the three
new (camera-lighter) agents 4/4 each, with 0 GPUs parked and no queue-burn. **LAV** is the
exception: it triggers a server crash during `load_world` ("failed to connect to newly created
map") that recovery cannot pre-empt, so LAV stays server-limited.

**Per-route data recovery.** Because `results.json` is checkpointed *per route* within a
route-file *suite* (§6), a server that segfaults mid-suite still leaves every route it
finished on disk — even though the job is marked `failed`. `tools/harvest_results.py` unions
`job_queue.json` with `completed_jobs.json` and reads each expected `results.json`, recovering
those per-route evals. Across the run this recovered **≈86% of the 1,648 route-evals from jobs the
queue calls "failed"** — data the file-level completion count discards entirely. The reportable
unit is therefore the **route-eval**, not the completed job/file.

### Escalation: from recoverable segfault to unrecoverable node crash + park-on-unkillable

Under sustained multi-day load the failure mode escalated past what the segfault-restart logic
above can handle.

**Failure chain.** When a route exceeds `JOB_TIMEOUT` (3600 s, §2), `manage_continuous.py` kills the
eval's process group (`SIGTERM` → 60 s → `SIGKILL`) while deliberately leaving the *persistent*
CARLA server alive for the next job. But if that CARLA is mid-render on the wedged GL driver, the
kill leaves it in **uninterruptible D-state** — SIGKILL cannot reap it. SLURM then reports
**"Kill task failed"** and **drains the node**; a run of these escalated on pod09 to a full
**"Not responding"** kernel hang. Three distinct severities result:

| Severity | Trigger | Recoverable? |
|----------|---------|--------------|
| GL segfault at context creation | UE4-4.24 / driver-575 instability | **Yes** — restart-hardening (above) |
| GPU wedged in D-state after timeout-kill | CARLA killed mid-GL-render | **Partly** — park-on-unkillable (below) |
| Whole-node "Not responding" | kernel hang under sustained load | **No** — admin reboot required |

**Mitigation — park-on-unkillable** (`carla_server_manager.py`, commit 5037b4b). The manager now
reads `/proc/<pid>/stat` process state and treats a process that survives SIGKILL as a wedge, not a
transient:
- `_proc_state(pid)` — reads the state char (`D` = uninterruptible, `Z` = zombie).
- `_wedged_pid_on_port(rpc)` — flags a process bound to a GPU's RPC port only if it stays `D`/`Z`
  across a ~1.5 s confirmation window (fast-path; `R`/`S` are never flagged — unit-tested).
- After the kill loop in `_kill_gpu_server`, any surviving PID on the port → **park the GPU**
  (`_park_gpu`) rather than looping on an unkillable process.
- `ensure()` no longer *trusts an open port*: if the listener is a wedged D/Z process it recycles +
  parks instead of dispatching a job into a dead server.

The point of parking is to **stop one wedged GPU from draining the whole node** — the worker on
that GPU quiesces (it does not fast-fail jobs `rc=3` and burn the queue), while the other GPUs keep
producing. This contains the *drain* severity. It does **not** and cannot prevent a full
kernel-level "Not responding" crash, which needs an admin reboot and a SLURM `UnkillableStepProgram`
to clean up the D-state step. As of this writing the run is **paused** on that admin dependency
(§8).

**Root cause of the whole-node crashes — confirmed: WekaFS storage fencing (not memory, not the GL
segfault).** Diagnosed 2026-07-15 from login-node evidence (`NODE_FAILURE_EVIDENCE_2026-07-15.md`).
`/scratch` is **WekaFS**, a network filesystem, and pod09/pod17 are *converged* Weka-backend +
compute nodes. `weka events` show each node's Weka process going `NodeUnreachable → FENCING →
Rejoined` at the **exact** SLURM `NODE_FAIL` timestamps (job 167116: pod17 19:10 / pod09 20:22 ↔
NODE_FAIL 19:13–20:23), bracketed by CRITICAL `HangingDriverFrontendIos` alerts. When Weka fences a
node's storage client, `/scratch` I/O dies and every writer wedges in uninterruptible **D-state**
(I/O-wait — which is why SIGKILL cannot reap it, and why `park-on-unkillable` can contain the *drain*
but never the *fence*). The likely trigger is heavy CARLA GL/CPU load on a converged node starving
the Weka process until it misses its cluster heartbeat, with metadata-heavy small-`.npy` writes
adding file-lease pressure. **Memory is excluded**: ~1 TB RAM/node at <5% use, zero `OUT_OF_MEMORY`
states, no OOM traces. This retro-explains the earlier storage-maintenance suspicion — right
subsystem (Weka), even though that specific maintenance window didn't line up. Our-side mitigation:
shard/stage the per-frame writes to cut Weka metadata pressure; the rest is admin-side (weka client
isolation, `UnkillableStepProgram`).

### Cross-cutting: stale container bytecode silently ignored source edits

**Symptom:** A correct agent-code edit (the LAV collision fix) kept producing the *old* error, and
the traceback pointed at a **comment line** — the tell-tale of stale compiled bytecode running
against newer source.

**Root cause:** the container's Python did not reliably invalidate `__pycache__` across the
bind-mounted `/workspace`, so edits under `leaderboard/team_code/*` could execute an old `.pyc`.

**Fix (`continuous_cli.py`):** set `PYTHONDONTWRITEBYTECODE=1` in the eval command (container never
writes `.pyc`) + a one-time `__pycache__` purge at run start. **Implication:** any earlier
agent-side fix that "seemed ineffective" may have been masked the same way.

---

## 10. Metrics and Evaluation

### Leaderboard metrics (per run)

The CARLA Leaderboard 1.0 evaluates each route independently and computes:

| Metric | Formula / Description |
|--------|----------------------|
| Route Completion | % of route waypoints reached |
| Driving Score | Route Completion × Infraction Penalty |
| Infraction Penalty | Product of per-category multipliers (0–1 each) |
| Collisions w/ vehicle | Per-km count, multiplier: 0.6 per event |
| Collisions w/ layout | Per-km count, multiplier: 0.6 per event |
| Red light violations | Per-km count, multiplier: 0.7 per event |
| Stop sign violations | Per-km count, multiplier: 0.7 per event |
| Off-road driving | % of off-road frames, multiplier: 0.7 |

These are parsed from `results.json._checkpoint.records[].scores` and averaged into
`score_composed` and `score_route` in `completed_jobs.json`.

### Collection-level metrics

Recorded continuously into `collection_state/metrics/`:

| Metric | Location | Frequency |
|--------|----------|-----------|
| Job start/end events | `metrics/events.jsonl` | Per job |
| CARLA server startup time | `metrics/servers/<node>/carla_pool.jsonl` | Per server start |
| GPU utilisation | `metrics/node/<node>/gpu.jsonl` | Periodic |
| System utilisation | `metrics/node/<node>/system.jsonl` | Periodic |
| Node hardware config | `metrics/node/<node>/static.json` | Once |

### Persistence savings

Each job would otherwise boot a fresh CARLA server (~60 s of UE4 startup) before evaluating. With
one **persistent** server per GPU, a boot is incurred only on first launch and after a crash/park,
not per job — at steady state this eliminates **≈98.6%** of server boots across a run (measured from
`metrics/servers/<node>/carla_pool.jsonl` boot events vs jobs dispatched). This is what makes a
slower-than-real-time sweep (§2) tractable at all, and it compounds with the per-route harvest (§6):
the machinery that keeps servers alive is the same machinery that leaves finished routes on disk
when one finally crashes.

### Figure generation

```bash
python3 genfig.py --state-root collection_state \
                  --dataset-root dataset \
                  --outdir paper_figures
```

---

## 11. Key Findings

Consolidated, paper-facing summary of the results this system produced. Each links back to the
section with the mechanism and the code.

**F1 — Crash-truncated evaluation is recoverable, and recovery is most of the data.** Because CARLA
checkpoints `results.json` per route within a suite, a mid-suite crash still leaves finished routes
on disk. Unioning the queue with completed-jobs and re-reading every checkpoint recovered **≈86% of
1,648 route-evals** from jobs the queue calls "failed." The methodological claim: **report the
route-eval, not the job** — file-level accounting throws away the majority of a distributed CARLA
evaluation's yield under realistic (crashy) infrastructure. (§6, §9)

**F2 — A single scalar difficulty score does not predict agent performance.** Re-validated at
n = 1,648 it washes out (pooled Spearman ≈ 0). The earlier encouraging correlations (n ≈ 204;
InterFuser −0.642, TCP −0.311) were small-sample artifacts — at scale InterFuser collapses to
noise and **TCP's sign flips positive** (its cautious control times out in dense grid towns the
geometry proxy scores "easy"). Difficulty is **agent-relative**, not a scene property. (§7)

**F3 — Two of the scalar's three inputs are structurally degenerate for the routes collected.** The
`_short`/`_tiny` route files — the entire current sweep — store only endpoints; the driven path is
interpolated at runtime by `GlobalRoutePlanner`. Any XML-parsed geometry/scenario term measures
endpoint displacement, i.e. near-noise. Only `_long` files carry real geometry. (§3, §7)

**F4 — The recoverable difficulty signal is illumination + map, near a ~0.65 AUC ceiling.**
Decomposing weather into physical axes and fitting a per-agent noisy-OR
`P(fail)=1−exp(−Σ λ_j x_j)` (Newton MLE + Hessian CIs, validated on synthetic ground truth) shows
the identifiable axis is **illumination — "dark = hard"**. A cross-validated head-to-head puts the
old scalar at AUC ≈ 0.53 and a parsimonious **illumination + geometry** model at ≈ 0.62 — near the
**~0.65 ceiling** a sister AV-scene-scoring project found robust even to reasoning foundation
models (Nvidia Cosmos Reason). Extra weather axes add nothing. This is an **independent
second-domain replication** of that ceiling. (§7)

**F5 — Map urban-density is a dominant, previously-missed factor, and it can be computed
map-agnostically.** Fail rates split hard by town (Town02 dense grid 72–100% vs Town05 open
10–19%) while the scalar scored all towns alike. `tools/map_density.py` derives an urban-density
axis (junctions per road-km / per km², segment length, curvature) from **OpenDRIVE** — offline from
a `.xodr` or in-sim via `carla.Map.to_opendrive()` — so **custom user maps score by the same logic
with no hardcoding**. `junctions_per_road_km` ranks Town02 (4.00) > Town01 (2.85) > Town05 (2.27),
matching observed difficulty. (§7)

**F6 — Per-agent nuance matters for the paper's framing.** Difficulty (illumination) predicts
failure for the condition-dependent agents (InterFuser / NEAT / Roach, near the ceiling) but not
CILRS (fails near-uniformly — a genuine weak baseline, audited clean in §5.4) or TCP (anti-correlated
via the timeout mechanism). A per-agent multi-axis model, not one global scalar, is the correct
object.

**F7 — Persistent servers + resilience are what make the throughput real.** One persistent CARLA per
GPU eliminates ≈98.6% of process boots (§10); the restart-hardening + park-on-unkillable machinery
(§9) is what let the 1,648 route-evals survive intermittent GL segfaults and node drains. The
unresolved ceiling is host-side: whole-node "Not responding" crashes under sustained A100 load,
which no user-space fix reaches. (§9)

**Caveat / open thread.** The camera-only vs LiDAR illumination contrast that motivated the
multi-axis approach cannot be closed from the collected data alone, because the only LiDAR-primary
agent (LAV) is exactly the one A100 cannot serve (§5.2, §9). Real per-*route* map density needs the
sim to interpolate the path (blocked on the login node); the offline endpoint approximation is too
coarse. (§7)

---

## Appendix A — Weather Presets

Order matches `_WEATHER_IDS` in `consolidated_agent.py` (0-indexed).

| Index | Name | Category | Difficulty Score |
|-------|------|----------|-----------------|
| 0 | ClearNoon | Clear / Day | 0.0 |
| 1 | ClearSunset | Clear / Dusk | 0.5 |
| 2 | CloudyNoon | Cloudy / Day | 0.5 |
| 3 | CloudySunset | Cloudy / Dusk | 1.0 |
| 4 | WetNoon | Wet / Day | 1.5 |
| 5 | WetSunset | Wet / Dusk | 2.0 |
| 6 | MidRainyNoon | Rain / Day | 2.5 |
| 7 | MidRainSunset | Rain / Dusk | 3.0 |
| 8 | WetCloudyNoon | Wet+Cloudy / Day | 1.5 |
| 9 | WetCloudySunset | Wet+Cloudy / Dusk | 2.0 |
| 10 | HardRainNoon | Heavy Rain / Day | 3.5 |
| 11 | HardRainSunset | Heavy Rain / Dusk | 4.0 |
| 12 | SoftRainNoon | Light Rain / Day | 2.0 |
| 13 | SoftRainSunset | Light Rain / Dusk | 2.5 |
| 14 | ClearNight | Clear / Night | 3.0 |
| 15 | CloudyNight | Cloudy / Night | 3.5 |
| 16 | WetNight | Wet / Night | 4.0 |
| 17 | WetCloudyNight | Wet+Cloudy / Night | 4.5 |
| 18 | SoftRainNight | Light Rain / Night | 4.5 |
| 19 | MidRainyNight | Rain / Night | 5.0 |
| 20 | HardRainNight | Heavy Rain / Night | 5.5 |

Night presets (14–20) use custom `carla.WeatherParameters` with sun altitude angle −90°.
Standard presets (0–13) use named CARLA constants.

---

## Appendix B — Scenario Type Difficulty Weights

Used in scenario density scoring. Mean weight = **2.64** (across 7 types).

| Scenario | Description | Weight | Rationale |
|----------|-------------|--------|-----------|
| Scenario1 | Slow leading vehicle | 1.0 | Low urgency, predictable |
| Scenario3 | Cut-in vehicle | 3.0 | Lateral reaction required |
| Scenario4 | Stationary obstacle | 2.0 | Lane change or stop |
| Scenario7 | Pedestrian at crosswalk | 2.5 | Right-of-way decision |
| Scenario8 | Jaywalking pedestrian | 3.5 | Unexpected trajectory |
| Scenario9 | Sudden appearance (occluded) | 4.5 | Maximum surprise; minimal reaction time |
| Scenario10 | Slow vehicle + secondary | 2.0 | Complex but not fast-reacting |

All 7 scenario types share identical spawn locations within each town (verified: 0 differing
positions across all type pairs in Town01, 486 unique positions).

---

## Appendix C — Pipeline Module Reference

Quick-reference table of all pipeline stage classes in `pipeline_modules.py`.

| Class | Category | Key args |
|-------|----------|---------|
| `ExtractCameraRGB` | Extraction | sensor_id, out_key, bgr_to_rgb |
| `ExtractSpeed` | Extraction | sensor_id, out_key |
| `ExtractGNSS` | Extraction | sensor_id, out_key, take |
| `ExtractCompass` | Extraction | sensor_id, out_key |
| `ExtractLidarXYZ` | Extraction | sensor_id, out_key, flip_y, **num_cols** |
| `RoutePlannerNextCommand` | Routing | gps_key, out_pos/wp/cmd_key, min/max_distance |
| `TargetPointFromNextWaypoint` | Routing | pos_key, compass_key, wp_key, out_key |
| `ImageHWCToTorchCHW` | Image | in_key, resize_wh, center_crop, mean, std |
| `HorizontalCameraConcat` | Image | in_keys, out_key |
| `MultiCameraToTorchBatch` | Image | in_keys, out_key, divide_by_255 |
| `LidarHistogramFromXYZ` | LiDAR/IF | lidar_xyz_key, compass_key, pos_key, crop, reuse_every_n |
| `LidarVehicleBodyFilter` | LiDAR/LAV | lidar_key, out_key, min/max_x/y/z |
| `EKFEgoLocalizer` | LiDAR/LAV | gps_key, compass_key, speed_key, lf, lr, freq |
| `PointPaintingModule` | LiDAR/LAV | lidar_key, seg_key, cam_yaws, lidar_xyz, cam_xyz |
| `TemporalLidarAccumulator` | LiDAR/LAV | lidar_key, pos_key, compass_key, num_frame_stack, gap |
| `BEVHeatmapNMS` | Detection | heatmaps/sizemaps/orimaps_key, kernel_size, min_score, ego_pixel_x/y |
| `ControlFromAccSteer` | Control | acc_key, steer_key, out_key |
| `PIDFromWaypoints` | Control | waypoints_key, speed_key, config |
| `WaypointTrackingPID` | Control/LAV | waypoints_key, aim_point, speed_ratio, pixels_per_meter |
| `ClampControl` | Control | control_key, steer/throttle/brake_clip |
| `BlendControls` | Control | a_key, b_key, out_key, alpha |
| `TCPStateAssemble` | TCP | speed_key, target_point_key, cmd_key, out_key |
| `TCPModelRunner` | TCP | checkpoint_path, img_key, state_key, target_point_key |
| `TCPBetaControl` | TCP | pred_key, brake_speed, brake_ratio |
| `TCPPIDControl` | TCP | pred_key, speed_key, target_point_key, aim_dist, angle_thresh |
| `TurningStatusDetector` | TCP | last_control_key, window, threshold, count_thresh |
| `TCPBlendControl` | TCP | ctrl_key, traj_key, status_key, out_key |
| `LAVRGBSegmentationRunner` | LAV | checkpoint_path, in_key, seg_channels |
| `LAVBrakePredictionRunner` | LAV | checkpoint_path, wide_key, tel_key, crop_tel_bottom |
| `LAVLiDARModelRunner` | LAV | checkpoint_path, lidar_key, num_input (=16), num_features |
| `LAVUniPlannerRunner` | LAV | checkpoint_path, bev_checkpoint_path, num_plan, num_cmds |
| `LAVCollisionCheck` | LAV | ego_plan_key, other_cast_key, dist_threshold_static/moving |
| `EmergencyBrakeOverride` | LAV | brake_threshold, max_speed_kmh, stop_limit, force_frames |
| `InterfuserOutputUnpack` | InterFuser | model_output_key, traffic_meta_key, pred_waypoints_key |
| `TrafficMetaTracker` | InterFuser | traffic_meta_key, gps_key, compass_key, momentum, update_every_n |
| `InterfuserControllerModule` | InterFuser | speed/waypoints/junction/traffic_light/stop_sign_key, detect_threshold |
| `NumpyToTorch` | Utility | in_key, out_key, device, dtype, add_batch_dim |
| `TorchModelRunner` | Utility | model spec, checkpoint_path, inputs dict, output_key |
| `WarmupAndFrameSkip` | Utility | warmup_steps, every_n |
| `CommandOneHotFromNextCommand` | Glue | cmd_key, num_cmds, one_based |
| `NormalizeScalar` | Glue | in_key, out_key, denom |
| `AssembleVector` | Glue | keys, out_key |
| `SetValue` | Glue | key, value |
| `RenameKeys` | Glue | mapping, keep_source |

**Bold** args indicate a non-obvious default or one that differs from the original agent.
