# Pipeline Modules Reference

All agents in this repo run through `ConsolidatedAgent` in one of two modes:

- **Legacy mode** — delegates to an original agent class unchanged (e.g. the original `lav_agent.py`).
- **Pipeline mode** — runs a list of modular classes defined in a YAML `pipeline:` block.

This document covers pipeline mode. Each module implements `run(context) -> context`. The context
dict flows through the list in order; modules read and write named keys. The final module must
write `context['control']`.

**Accepted control formats** (coerced by `ConsolidatedAgent._coerce_control`):
- `carla.VehicleControl`
- `dict` with keys `steer`, `throttle`, `brake`
- `dict` with keys `steer`, `acc` (acc ≥ 0 → throttle, acc < 0 → brake)
- `tuple/list` of length 3

**Fixed context keys injected before pipeline runs:**

| Key | Type | Description |
|-----|------|-------------|
| `input_data` | dict | Raw Leaderboard sensor dict `{sensor_id: (frame, raw)}` |
| `timestamp` | float | Simulation timestamp |
| `global_step` | int | Step counter (0-based) |
| `last_control` | carla.VehicleControl | Previous step's output control |
| `config` | dict | Parsed YAML config |
| `agent` | ConsolidatedAgent | Agent instance (carries `_global_plan`) |

---

## Sensor Extraction

### `ExtractCameraRGB`
Extract a camera image from `input_data` and write it as an HxWx3 uint8 numpy array.

- **Args**: `sensor_id`, `out_key`, `bgr_to_rgb=True`
- **Writes**: `context[out_key]` — `np.ndarray (H, W, 3)` uint8

### `ExtractSpeed`
Extract speed (m/s) from the Leaderboard speedometer sensor dict.

- **Args**: `sensor_id='speed'`, `out_key='speed'`, `dict_key='speed'`
- **Writes**: `context[out_key]` — `float`

### `ExtractGNSS`
Extract GNSS lat/lon (or x/y) into a 2-vector.

- **Args**: `sensor_id='gps'`, `out_key='gps'`, `take=2`
- **Writes**: `context[out_key]` — `np.ndarray (2,)`

### `ExtractCompass`
Extract compass heading (radians) from the IMU sensor (last element of the IMU array).

- **Args**: `sensor_id='imu'`, `out_key='compass'`
- **Writes**: `context[out_key]` — `float`

### `ExtractLidarXYZ`
Extract a LiDAR point cloud and keep the first `num_cols` columns.

Default `num_cols=3` keeps only XYZ. Set `num_cols=4` to retain intensity — required by LAV's
PointPillarNet, whose checkpoint was trained on XYZI + painted features. `flip_y` negates column 1
(Y-axis only) to match agent coordinate conventions.

- **Args**: `sensor_id='lidar'`, `out_key='lidar_xyz'`, `flip_y=True`, `num_cols=3`
- **Writes**: `context[out_key]` — `np.ndarray (N, num_cols)` float32

---

## Routing and Planning

### `RoutePlannerNextCommand`
Advance the route planner using the current GNSS position and emit the next high-level command.

- **Args**: `gps_key='gps'`, `out_pos_key='pos'`, `out_wp_key='next_waypoint'`,
  `out_cmd_key='next_command'`, `min_distance=4.0`, `max_distance=50.0`, `gps_in_degrees=True`
- **Reads**: `context['agent']._global_plan`
- **Writes**:
  - `context[out_pos_key]` — planner-local position `(x, y)`
  - `context[out_wp_key]` — next waypoint absolute position `(x, y)`
  - `context[out_cmd_key]` — int command (1=Left, 2=Right, 3=Straight, 4=LaneFollow, …)

### `TargetPointFromNextWaypoint`
Convert absolute next waypoint + current pose into a vehicle-frame target point.

$$\text{target} = R^T(\theta)\,(wp - pos), \quad \theta = \text{compass} + \pi/2$$

- **Args**: `pos_key='pos'`, `compass_key='compass'`, `next_waypoint_key='next_waypoint'`,
  `out_key='target_point'`
- **Writes**: `context[out_key]` — `np.ndarray (2,)` in vehicle-frame metres

---

## Image Processing

### `ImageHWCToTorchCHW`
Convert an HxWxC numpy image to a (1, C, H, W) float32 CUDA tensor with optional resize, crop, and ImageNet normalisation.

- **Args**: `in_key`, `out_key=None` (defaults to `in_key`), `device='cuda'`, `divide_by_255=True`,
  `mean=None`, `std=None`, `add_batch_dim=True`, `resize_wh=None`, `center_crop=None`
- **Writes**: `context[out_key]` — `torch.Tensor (1, C, H, W)` float32

### `HorizontalCameraConcat`
Concatenate multiple camera images horizontally (along axis 1) into one wide array.

- **Args**: `in_keys` (list of context keys), `out_key='rgb_wide_raw'`
- **Writes**: `context[out_key]` — `np.ndarray (H, W_total, C)`

### `MultiCameraToTorchBatch`
Stack N camera images (H, W, C) into a batched CUDA tensor (N, C, H, W).

- **Args**: `in_keys`, `out_key='rgb_batch_t'`, `device='cuda'`, `divide_by_255=False`
- **Writes**: `context[out_key]` — `torch.Tensor (N, C, H, W)` float32

---

## LiDAR Processing (InterFuser-style)

### `LidarHistogramFromXYZ`
Build a BEV LiDAR histogram (InterFuser input format) from an XYZ point cloud with planner-local
ego-motion compensation.

- **Args**: `lidar_xyz_key='lidar_xyz'`, `compass_key='compass'`, `pos_key='pos'`,
  `out_key='lidar_hist'`, `crop=224`, `reuse_every_n=2`, `warmup_reuse_steps=4`
- **Writes**: `context[out_key]` — `np.ndarray (C, H, W)` float32

---

## EKF Localisation and LiDAR Pipeline (LAV)

These modules replicate the LAV agent's perception front-end exactly. They must run in order.

### `LidarVehicleBodyFilter`
Remove LiDAR points inside the ego-vehicle bounding box (sensor self-returns).

Default bounds match LAV's `preprocess()` ego-body footprint.

- **Args**: `lidar_key='lidar_raw'`, `out_key='lidar_filtered'`,
  `min_x=-2.4`, `max_x=0.0`, `min_y=-0.8`, `max_y=0.8`, `min_z=-1.5`, `max_z=-1.0`
- **Writes**: `context[out_key]` — filtered point cloud `np.ndarray (N', num_cols)`

### `EKFEgoLocalizer`
Kinematic bicycle model EKF fusing GPS, compass, and vehicle speed. Outputs the previous-step
corrected pose (used for LiDAR ego-motion compensation) and updates internal state for the next
tick.

- **Args**: `gps_key='gps_raw'`, `compass_key='compass'`, `speed_key='speed'`,
  `last_control_key='control'`, `out_pos_key='ekf_pos'`, `out_compass_key='ekf_compass'`,
  `lf=1.477531`, `lr=1.3936`, `gnss_noise=5e-6`, `compass_noise=1e-7`,
  `max_steer_angle=70.0`, `freq=20.0`
- **Writes**: `context[out_pos_key]` `(x, y)`, `context[out_compass_key]` float radians

### `PointPaintingModule`
Project LiDAR points onto semantic segmentation maps from multiple cameras and append the painted
semantic features to each point. Wraps LAV's `CoordConverter` and `point_painting` utilities.

- **Args**: `lidar_key='lidar_filtered'`, `seg_key='rgb_seg'`, `out_key='lidar_fused'`,
  `cam_yaws=(-60, 0, 60)`, `lidar_xyz=(0, 0, 2.4)`, `cam_xyz=(1.5, 0, 2.4)`,
  `rgb_h=288`, `rgb_w=256`, `fov=64.0`
- **Writes**: `context[out_key]` — `np.ndarray (N, lidar_cols + n_seg_channels)`

### `TemporalLidarAccumulator`
Accumulate point-painted LiDAR frames over time with ego-motion compensation. Appends a one-hot
temporal encoding of length `num_frame_stack + 1` to each frame's features, then concatenates all
retained frames into a single array.

- **Args**: `lidar_key='lidar_fused'`, `pos_key='ekf_pos'`, `compass_key='ekf_compass'`,
  `out_key='lidar_stacked'`, `num_frame_stack=2`, `gap=5`, `concat_with_prev=True`
- **Writes**: `context[out_key]` — `np.ndarray (N_all, feat_dim)` float32

### `BEVHeatmapNMS`
Non-maximum suppression on Bird's-Eye-View heatmaps from `LAVLiDARModelRunner`. Returns detections
as a list-of-lists `[[vehicles...], [pedestrians...]]` where each detection is `(x, y, w, h, cos, sin)`
in BEV pixel coordinates.

Filters applied (replicating `lav_agent.det_inference` exactly):
- Ego-proximity filter: removes peaks within `ego_filter_radius` pixels of `(ego_pixel_x, ego_pixel_y)`
- Size filter: `(i==1 and w < 0.1*ppm) or h < 0.2*ppm` — removes tiny-footprint pedestrians AND
  any detection with unrealistically small height regardless of class.

- **Args**: `heatmaps_key`, `sizemaps_key`, `orimaps_key`, `out_key='lav_detections'`,
  `kernel_size=7`, `min_score=0.1`, `max_det=15`, `pixels_per_meter=4.0`,
  `ego_pixel_x=160`, `ego_pixel_y=280`, `ego_filter_radius=2.0`
- **Writes**: `context[out_key]` — `list[list[tuple]]`

---

## Control Modules

### `ControlFromAccSteer`
Convert separate `acc` and `steer` scalars into a `{steer, throttle, brake}` dict.

- **Args**: `acc_key='acc'`, `steer_key='steer'`, `out_key='control'`,
  `throttle_clip=1.0`, `brake_clip=1.0`

### `PIDFromWaypoints`
Compute `{steer, throttle, brake}` from a predicted waypoint path and current speed using numpy
PID controllers. Configurable via a `_PIDCfg`-compatible dict.

- **Args**: `waypoints_key='waypoints'`, `speed_key='speed'`, `out_key='control'`, `config=None`
- **Reads**: waypoints `(N, 2)` in metres, speed in m/s

### `WaypointTrackingPID`
Command-conditioned PID controller that tracks LAV UniPlanner waypoints. Selects a command-specific
aim point, computes desired speed from inter-waypoint spacing, and drives steer and throttle PIDs.
Mirrors `lav_agent.pid_control()`.

- **Args**: `waypoints_key='lav_ego_plan'`, `speed_key='speed'`, `cmd_key='next_command'`,
  `out_key='control_base'`, `pixels_per_meter=4.0`, `aim_point=[...]`, `speed_ratio=[...]`,
  `turn_KP=0.8`, `turn_KI=0.5`, `turn_KD=0.2`, `turn_n=40`, `speed_KP=5.0`, `speed_KI=0.5`,
  `speed_KD=1.0`, `speed_n=40`, `brake_speed=0.2`, `clip_delta=0.25`, `max_throttle=0.8`

### `ClampControl`
Clamp and sanitise a control dict in-place.

- **Args**: `control_key='control'`, `steer_clip=1.0`, `throttle_clip=1.0`, `brake_clip=1.0`,
  `zero_throttle_when_braking_over=0.5`, `brake_wins_over_throttle=True`

### `BlendControls`
Blend two control dicts: `out = alpha*a + (1-alpha)*b`.

- **Args**: `a_key`, `b_key`, `out_key='control'`, `alpha=0.3`

---

## TCP-specific Modules

These modules implement the dual-path Beta+PID control loop from [TCP](https://github.com/OpenDriveLab/TCP).

### `TCPStateAssemble`
Build the `(1, 9)` state tensor: `[speed/12, target_x, target_y, cmd_one_hot(6)]`.

- **Args**: `speed_key='speed'`, `target_point_key='target_point'`, `cmd_key='next_command'`,
  `out_key='tcp_state'`, `device='cuda'`
- **Writes**: `context[out_key]` — `torch.Tensor (1, 9)`

### `TCPModelRunner`
Load the TCP ResNet34+GRU model and run `forward(img, state, target_point)`.

- **Args**: `checkpoint_path`, `img_key='rgb_t'`, `state_key='tcp_state'`,
  `target_point_key='target_point_t'`, `out_key='tcp_pred'`, `device='cuda'`
- **Writes**: `context[out_key]` — raw model prediction dict with keys
  `mu_branches`, `sigma_branches`, `pred_wp`

### `TCPBetaControl`
Derive a control action from the Beta-distribution head of TCP.

`mu_branches` and `sigma_branches` are Softplus outputs (range (0,∞)) — they are the α and β
parameters directly, not mean/sigma. Implements `_get_action_beta` from the original model: mode
`= (α-1)/(α+β-2)` when both > 1, scaled to [-1, 1].

- **Args**: `pred_key='tcp_pred'`, `out_key='ctrl_ctrl'`, `brake_speed=0.4`,
  `brake_ratio=1.1`, `clip_delta=0.25`, `max_throttle=0.75`
- **Writes**: `context[out_key]` — `{steer, throttle, brake}` dict

### `TCPPIDControl`
PID control on TCP's predicted waypoints. Faithful reimplementation of `TCP.control_pid()`,
including the three-angle outlier-rejection logic that suppresses noisy waypoints on straight roads
by falling back to the GPS-derived angle.

- **Args**: `pred_key='tcp_pred'`, `speed_key='speed'`, `target_point_key='target_point'`,
  `out_key='ctrl_traj'`, `aim_dist=4.0`, `angle_thresh=0.3`, `dist_thresh=10.0`,
  `brake_speed=0.4`, `brake_ratio=1.1`, `clip_delta=0.25`, `max_throttle=0.75`,
  `turn_KP=1.25`, `turn_KI=0.75`, `turn_KD=0.3`, `turn_n=40`,
  `speed_KP=5.0`, `speed_KI=0.5`, `speed_KD=1.0`, `speed_n=40`
- **Writes**: `context[out_key]` — `{steer, throttle, brake}` dict

### `TurningStatusDetector`
Classify the current driving situation as turning (1) or straight (0) using a rolling window of
the previous control's steer magnitude.

- **Args**: `last_control_key='control'`, `out_key='turning_status'`, `window=20`,
  `threshold=0.1`, `count_thresh=10`
- **Writes**: `context[out_key]` — `int` (0 or 1)

### `TCPBlendControl`
Blend the Beta-distribution control and the PID-trajectory control based on turning status.
Straight: 0.3×Beta + 0.7×PID. Turning: 0.7×Beta + 0.3×PID. Clamps all outputs to valid ranges.

- **Args**: `ctrl_key='ctrl_ctrl'`, `traj_key='ctrl_traj'`, `status_key='turning_status'`,
  `out_key='control'`
- **Writes**: `context[out_key]` — `{steer, throttle, brake}` dict

---

## LAV-specific Modules

These modules implement the [LAV](https://github.com/dotchen/LAV) agent's inference pipeline:
segmentation → point painting → temporal LiDAR accumulation → PointPillar detection →
UniPlanner trajectory → collision check → PID control.

### `LAVRGBSegmentationRunner`
Run LAV's ERFNet semantic segmentation model on a batch of camera images. Lazy-loads on first call.

- **Args**: `checkpoint_path`, `in_key='rgb_batch_t'`, `out_key='rgb_seg'`,
  `seg_channels=(4, 6, 7, 10)`, `device='cuda'`
- **Writes**: `context[out_key]` — `np.ndarray (N_cams, C_seg, H, W)` float32

### `LAVBrakePredictionRunner`
Run LAV's cross-attention brake prediction model (dual ResNet18 + attention) on concatenated wide
and telephoto RGB. Lazy-loads on first call.

- **Args**: `checkpoint_path`, `wide_key='rgb_wide_raw'`, `tel_key='rgb_tel_raw'`,
  `out_key='brake_pred'`, `crop_tel_bottom=96`, `device='cuda'`
- **Writes**: `context[out_key]` — `float` brake probability

### `LAVLiDARModelRunner`
Run LAV's PointPillar-based BEV object detection model. Outputs feature maps and
detection heatmaps/sizemaps/orimaps for downstream use by `BEVHeatmapNMS` and `LAVUniPlannerRunner`.

`num_input` is the **post-`decorate()`** feature count fed into `DynamicPointNet`'s first
`nn.Linear`. `decorate()` appends 5 features (3 cluster-offset + xp + yp) to the raw point, so:
`num_input = raw_dims + 5`. For the standard checkpoint (`lidar_v2_7.th`): raw = XYZI(4) +
painted(4) + time_one_hot(3) = 11, so `num_input = 16`.

- **Args**: `checkpoint_path`, `lidar_key='lidar_stacked'`, `out_features_key='lav_features'`,
  `out_heatmaps_key='lav_heatmaps'`, `out_sizemaps_key='lav_sizemaps'`,
  `out_orimaps_key='lav_orimaps'`, `num_input=16`, `backbone='cnn'`,
  `num_features=(64, 64)`, `min_x=-10.0`, `max_x=70.0`, `min_y=-40.0`, `max_y=40.0`,
  `pixels_per_meter=4.0`, `device='cuda'`
- **Writes**: four context keys with BEV feature/detection tensors

### `LAVUniPlannerRunner`
Run LAV's GRU-based multi-command UniPlanner. Produces an ego trajectory, cast (predicted future)
locations for other agents, and their predicted commands. Handles the lane-change state machine:
a lane-change command (4 or 5) is only acted on after 300 consecutive frames, matching the
reference agent.

- **Args**: `checkpoint_path`, `bev_checkpoint_path`, `features_key='lav_features'`,
  `detections_key='lav_detections'`, `cmd_key='next_command'`,
  `target_point_key='target_point'`, `out_ego_plan_key='lav_ego_plan'`,
  `out_cast_locs_key='lav_cast_locs'`, `out_other_cast_key='lav_other_cast'`,
  `out_other_cmds_key='lav_other_cmds'`, `pixels_per_meter=4.0`, `crop_size=96`,
  `num_cmds=6`, `num_plan=20`, `num_plan_iter=5`, `num_frame_stack=2`,
  `num_features=(64, 64)`, `min_x=-10.0`, `device='cuda'`
- **Writes**: ego plan `(N, 2)`, cast locations, other cast trajectories, other command probs

### `LAVCollisionCheck`
Check whether any high-confidence predicted other-vehicle trajectory intersects the ego plan
within a distance threshold. Outputs a boolean collision flag used by `EmergencyBrakeOverride`.

- **Args**: `ego_plan_key='lav_ego_plan'`, `other_cast_key='lav_other_cast'`,
  `other_cmds_key='lav_other_cmds'`, `out_key='lav_collision'`,
  `pixels_per_meter=4.0`, `cmd_thresh=0.2`, `brake_speed=0.2`,
  `dist_threshold_static=1.0`, `dist_threshold_moving=2.5`
- **Writes**: `context[out_key]` — `bool`

### `EmergencyBrakeOverride`
Apply safety overrides on top of a base control output: brake on collision flag, cap max speed,
recover from prolonged stops.

- **Args**: `control_key='control_base'`, `brake_pred_key='brake_pred'`,
  `collision_key='lav_collision'`, `speed_key='speed'`, `out_key='control'`,
  `brake_threshold=0.1`, `max_speed_kmh=35.0`, `stop_limit=600`,
  `force_throttle=0.4`, `force_frames=20`
- **Writes**: `context[out_key]` — `{steer, throttle, brake}` dict

---

## InterFuser-specific Modules

These modules implement the [InterFuser](https://github.com/opendilab/InterFuser) agent's
transformer-based inference and traffic-scene controller.

### `InterfuserOutputUnpack`
Unpack the 6-element tuple output of the `interfuser_baseline` forward pass into named context keys.

- **Args**: `model_output_key='model_output'`, `traffic_meta_key='traffic_meta_raw'`,
  `pred_waypoints_key='pred_waypoints'`, `is_junction_key='is_junction'`,
  `traffic_light_key='traffic_light_state'`, `stop_sign_key='stop_sign'`

### `TrafficMetaTracker`
Apply InterFuser's agent Tracker to spatially-stabilise object detections, then apply exponential
moving average smoothing. Updates every `update_every_n` steps (skips updates during warmup).

- **Args**: `traffic_meta_key='traffic_meta_raw'`, `gps_key='pos'`, `compass_key='compass'`,
  `out_key='traffic_meta'`, `momentum=0.0`, `update_every_n=2`, `warmup_always_update=4`

### `InterfuserControllerModule`
Wrap `InterfuserController` (safety-aware PID with traffic light / stop-sign / junction logic)
for use in the modular pipeline.

- **Args**: `speed_key`, `waypoints_key`, `junction_key`, `traffic_light_key`, `stop_sign_key`,
  `traffic_meta_key`, `out_key='control'`, `turn_KP=1.25`, `turn_KI=0.75`, `turn_KD=0.3`,
  `turn_n=40`, `speed_KP=5.0`, `speed_KI=0.5`, `speed_KD=1.0`, `speed_n=40`,
  `max_throttle=0.75`, `brake_speed=0.1`, `brake_ratio=1.1`, `clip_delta=0.35`,
  `max_speed=5.0`, `detect_threshold=0.04`

---

## Torch Utilities

### `NumpyToTorch`
Convert a numpy-like value to a `torch.Tensor`, optionally adding a batch dimension.

- **Args**: `in_key`, `out_key=None` (defaults to `in_key`), `device='cuda'`, `dtype='float32'`,
  `add_batch_dim=False`

### `TorchModelRunner`
Instantiate an arbitrary torch model from config and run it every tick. Suitable for building
fully YAML-driven inference pipelines without writing any Python.

- **Args**:
  - Model spec: `model={module, class_name, args}` (or via `model_module`/`model_class_name`/`model_args`)
  - Checkpoint: `checkpoint_path`, `checkpoint_state_dict_key='state_dict'`, `checkpoint_prefix_strip`
  - Runtime: `device='cuda'`, `eval_mode=True`, `strict=False`, `no_grad=True`
  - Wiring: `inputs` (dict mapping model kwarg name → context key)
  - Outputs: `output_key` (store whole output) or `output_map` (map output dict keys to context keys)

---

## Pipeline Scheduling / Guards

### `WarmupAndFrameSkip`
Optionally short-circuit the rest of the pipeline for a warmup period or on every-N-steps
frame-skipping. Sets `context['control']` to a hold value and sets `context[stop_key] = True`,
which causes `PipelineEngine` to stop processing the remaining modules for that tick.

- **Args**: `warmup_steps=0`, `every_n=1`, `warmup_control=None`, `stop_key='__pipeline_stop__'`

---

## Glue and State Modules

### `CommandOneHotFromNextCommand`
Convert `next_command` to a one-hot float32 vector (1-based by default, 6-way).

- **Args**: `cmd_key='next_command'`, `out_key='cmd_one_hot'`, `num_cmds=6`,
  `one_based=True`, `clamp=True`, `negative_to=4`

### `NormalizeScalar`
Normalize a scalar: `out = float(in) / denom`.

- **Args**: `in_key`, `out_key`, `denom=1.0`

### `AssembleVector`
Concatenate scalars and/or 1-D arrays from multiple context keys into a single 1-D float32 numpy
vector.

- **Args**: `keys` (list), `out_key='state'`

### `SetValue`
Write a constant value into the context.

- **Args**: `key`, `value`

### `RenameKeys`
Rename or mirror keys in the context according to a mapping dict.

- **Args**: `mapping` (dict `{old: new}`), `keep_source=True`

---

## Minimal Pipeline Example

```yaml
sensors:
  - type: sensor.camera.rgb
    id: rgb
    width: 900
    height: 256
    fov: 100
    x: -1.5
    y: 0.0
    z: 2.0
  - type: sensor.other.gnss
    id: gps
  - type: sensor.other.imu
    id: imu
  - type: sensor.speedometer
    id: speed

pipeline:
  - module: team_code.pipeline_modules
    class: ExtractCameraRGB
    args: {sensor_id: rgb, out_key: rgb_raw, bgr_to_rgb: true}

  - module: team_code.pipeline_modules
    class: ExtractGNSS
    args: {sensor_id: gps, out_key: gps_raw, take: 2}

  - module: team_code.pipeline_modules
    class: ExtractCompass
    args: {sensor_id: imu, out_key: compass}

  - module: team_code.pipeline_modules
    class: ExtractSpeed
    args: {sensor_id: speed, out_key: speed}

  - module: team_code.pipeline_modules
    class: RoutePlannerNextCommand
    args:
      gps_key: gps_raw
      out_pos_key: pos
      out_wp_key: next_waypoint
      out_cmd_key: next_command
      gps_in_degrees: true

  - module: team_code.pipeline_modules
    class: TargetPointFromNextWaypoint
    args: {pos_key: pos, compass_key: compass, next_waypoint_key: next_waypoint, out_key: target_point}

  - module: team_code.pipeline_modules
    class: ImageHWCToTorchCHW
    args:
      in_key: rgb_raw
      out_key: rgb_t
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]

  # ...your TorchModelRunner sets `waypoints`...

  - module: team_code.pipeline_modules
    class: PIDFromWaypoints
    args: {waypoints_key: waypoints, speed_key: speed, out_key: control}
```

---

## Context-key contract & offline validation

Each tick the agent seeds the pipeline context with these keys (see
`consolidated_agent.run_step`):

`agent`, `input_data`, `timestamp`, `global_step`, `last_control`, `config`,
`external_config`

Modules then communicate by reading/writing **named context keys**, wired
explicitly through their `args` — typically `*_key` / `in_key` / `out_key`
fields (e.g. `ExtractSpeed(out_key='speed')` writes `speed`; a later
`TCPStateAssemble(speed_key='speed')` reads it). The final control must be
written to `context['control']` (a `carla.VehicleControl` or
`{steer, throttle, brake}` dict).

Because the wiring lives in the YAML args, you can review a pipeline's data flow
**without CARLA, torch, or the cluster**:

```bash
python3 continuous_cli.py validate-config leaderboard/team_code/configs/tcp.yaml
python3 continuous_cli.py validate-config --all      # validate every config
python3 continuous_cli.py new-agent myagent          # scaffold a starter config
```

`validate-config` hard-fails on malformed shape, a missing `module`/`class`, bad
`args`, or a step whose class doesn't exist in its module (catching typos before
a cluster round-trip), and prints each step with its args so the key flow is
reviewable. The same checks run in CI via `tests/test_config_schema.py`.
