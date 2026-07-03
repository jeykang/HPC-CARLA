# Technical Reference — HPC-CARLA Persistent Data Collection System

*Comprehensive reference for paper writing. Contains exact parameter values,
architecture specifications, implementation details, and engineering decisions.*

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
6. [Data Collection Pipeline](#6-data-collection-pipeline)
7. [Job Scheduling and Difficulty Estimation](#7-job-scheduling-and-difficulty-estimation)
8. [Dataset Structure and Statistics](#8-dataset-structure-and-statistics)
9. [Implementation Challenges and Solutions](#9-implementation-challenges-and-solutions)
10. [Metrics and Evaluation](#10-metrics-and-evaluation)
11. [Appendix A — Weather Presets](#appendix-a--weather-presets)
12. [Appendix B — Scenario Type Difficulty Weights](#appendix-b--scenario-type-difficulty-weights)
13. [Appendix C — Pipeline Module Reference](#appendix-c--pipeline-module-reference)

---

## 1. System Overview

The system performs scalable autonomous-driving data collection by running three published
imitation-learning agents — **TCP**, **LAV**, and **InterFuser** — across the full combinatorial
space of CARLA towns, routes, and weather conditions on an HPC GPU cluster. Each agent has been
re-implemented as a declarative YAML pipeline of composable stages, enabling sensor wiring, model
parameters, and control logic to be reconfigured independently of agent code.

**Scope of collection:**
- 3 agents × 22 routes × 21 weather conditions = 1,386 total jobs per run cycle
- 8 CARLA towns (Town01–Town07, Town10HD)
- Route types: long (~30 waypoints, ~1–2 km), short (~8 waypoints, ~300 m), tiny (~4 waypoints)
- 21 CARLA weather presets from ClearNoon (index 0) to HardRainNight (index 20)

**Core design decisions:**
- Persistent CARLA servers (one per GPU) eliminate the ~60 s startup overhead per job
- All agents share a single `ConsolidatedAgent` wrapper; agent identity is a YAML config
- A difficulty-aware scheduler prioritises hard combinations (complex routes + harsh weather +
  dense adversarial scenarios) to maximise signal from early runs
- Leaderboard scores from `results.json` are parsed into `completed_jobs.json` after each run

---

## 2. Hardware and HPC Infrastructure

### Cluster configuration

| Parameter | Value |
|-----------|-------|
| Scheduler | SLURM |
| Nodes in use | 2 (`hpc-pr-a-pod09`, `hpc-pr-a-pod17`) |
| GPUs per node | 8 × NVIDIA GPU (each with 575.57.08 driver) |
| Total GPUs | 16 |
| Job time limit | 336 h (14 days) |
| Node allocation | Exclusive |
| Parallelism | 16 concurrent CARLA evaluations (1 per GPU) |

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

| Type | Towns covered | Approx waypoints/route | Approx path length |
|------|--------------|------------------------|-------------------|
| Long | Town01–07 | 20–35 | 800 m – 2.5 km |
| Short | Town01–07, Town10 | 5–12 | 200–500 m |
| Tiny | Town01–07, Town10 | 3–6 | 100–250 m |

**Difficulty score range** (see §7): long routes score 35–75; short routes score 2–6; tiny routes
score 4–8.

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

Jobs are sorted by this 4-tuple key (ascending):

```
(attempts, agent_priority, −difficulty_score, −estimated_runtime_s)
```

1. **Fewest attempts first** — retries are deprioritised relative to fresh jobs
2. **Agent priority** — currently `{interfuser:0, tcp:1, lav:2}` for validation ordering
3. **Highest difficulty first** — ensures most informative combinations run early
4. **Longest estimated runtime as tiebreak** — fills GPU time efficiently

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

### Current collection status (as of 2026-06-05)

| Agent | Routes completed | Weather conditions | Towns | Frames |
|-------|------------------|--------------------|-------|--------|
| InterFuser | 16 | 12 | Town03, Town04 | ~179,520 |
| TCP | 0 | — | — | 0 |
| LAV | 0 | — | — | 0 |

Total `.npy` files on disk: **3,413,973** (InterFuser data only).

### Queue coverage

| Metric | Value |
|--------|-------|
| Total jobs | 1,386 |
| Running | 16 |
| Pending | 1,370 |
| Jobs per agent | 462 |
| Weather conditions queued | 21 (0–20) |

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

### InterFuser

#### No code bugs found

The reimplementation is faithful to the reference. The observed "stuck for thousands of frames
then moves briefly" behaviour is model-inherent — identical to running the original
`InterfuserAgent`. See §5.3 for the full controller analysis.

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

### Figure generation

```bash
python3 genfig.py --state-root collection_state \
                  --dataset-root dataset \
                  --outdir paper_figures
```

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
