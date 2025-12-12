# Pipeline Modules Reference

This document describes the reusable modules available for **pipeline-mode** agents (i.e., configs that use `pipeline:` in `ConsolidatedAgent`).

Conventions:
- Each module implements `run(context) -> context`.
- `context` is a mutable dict; modules read/write keys on it.
- `context['input_data']` is Leaderboard `input_data` shaped like `{sensor_id: (frame, raw)}`.
- A pipeline should ultimately produce `context['control']`.

Control formats accepted by `ConsolidatedAgent`:
- `carla.VehicleControl`
- `dict {steer, throttle, brake}`
- `dict {steer, acc}` where `acc>=0 -> throttle`, `acc<0 -> brake`
- `tuple/list` of length 3 (ordering configurable via `pipeline_control_tuple_order`)

---

## Extraction Modules

### `ExtractCameraRGB`
Extract a camera image from `input_data` and write it into the context.

- **Args**: `sensor_id`, `out_key`, `bgr_to_rgb` (default true)
- **Reads**: `context['input_data'][sensor_id]`
- **Writes**: `context[out_key]` (typically `np.ndarray` HxWx3)

### `ExtractSpeed`
Extract speed (m/s) from the speedometer sensor.

- **Args**: `sensor_id` (default `speed`), `out_key` (default `speed`), `dict_key` (default `speed`)
- **Writes**: `context[out_key]` as `float`

### `ExtractGNSS`
Extract GNSS into a 2-vector.

- **Args**: `sensor_id` (default `gps`), `out_key` (default `gps`), `take` (default 2)
- **Writes**: `context[out_key]` as `np.ndarray` shape `(2,)`

### `ExtractCompass`
Extract compass (radians) from IMU (last element).

- **Args**: `sensor_id` (default `imu`), `out_key` (default `compass`)
- **Writes**: `context[out_key]` as `float`

### `ExtractLidarXYZ`
Extract LiDAR XYZ (Nx3) from a LiDAR sensor.

- **Args**: `sensor_id` (default `lidar`), `out_key` (default `lidar_xyz`), `flip_y` (default true)
- **Writes**: `context[out_key]` as `np.ndarray` shape `(N,3)` float32

---

## Routing / Planning Modules

### `RoutePlannerNextCommand`
Compute `next_command` from the global plan and current GNSS.

- **Args**: `gps_key`, `out_pos_key` (default `pos`), `out_wp_key` (default `next_waypoint`), `out_cmd_key` (default `next_command`), `min_distance`, `max_distance`, `gps_in_degrees`
- **Reads**: `context['agent']._global_plan` (set via Leaderboard `set_global_plan()`)
- **Writes**:
  - `context[out_pos_key]` planner-local position
  - `context[out_wp_key]` next waypoint in planner-local coords
  - `context[out_cmd_key]` int command

### `TargetPointFromNextWaypoint`
Compute InterFuser/TCP-style `target_point`:

$$target\_point = R^T(\theta) \cdot (next\_wp - pos), \quad \theta = compass + \pi/2$$

- **Args**: `pos_key`, `compass_key`, `next_waypoint_key`, `out_key` (default `target_point`)
- **Writes**: `context[out_key]` as `np.ndarray` shape `(2,)`

---

## LiDAR Processing (InterFuser-like)

### `LidarHistogramFromXYZ`
Convert LiDAR xyz into histogram features using `team_code.utils.lidar_to_histogram_features` and a planner-local transform.

- **Args**: `lidar_xyz_key`, `compass_key`, `pos_key`, `out_key`, `crop`, `reuse_every_n`, `warmup_reuse_steps`
- **Writes**: `context[out_key]` as `np.ndarray` float32 shape `(C,H,W)`
- **Notes**: internally caches the last histogram to optionally reuse every N ticks.

---

## Control Modules

### `ControlFromAccSteer`
Convert `{acc, steer}` into `{steer, throttle, brake}`.

- **Args**: `acc_key`, `steer_key`, `out_key`, `throttle_clip`, `brake_clip`

### `PIDFromWaypoints`
Compute `{steer, throttle, brake}` from predicted waypoints + current speed using a numpy PID controller.

- **Args**: `waypoints_key`, `speed_key`, `out_key`, `config` (PID gains and thresholds)
- **Reads**: waypoints shaped `(N,2)` in meters and speed in m/s

### `ClampControl`
Clamp and sanitize a control dict.

- **Args**: `control_key`, `steer_clip`, `throttle_clip`, `brake_clip`, `zero_throttle_when_braking_over`, `brake_wins_over_throttle`

### `BlendControls`
Blend two control dicts: `alpha*a + (1-alpha)*b`.

- **Args**: `a_key`, `b_key`, `out_key`, `alpha`

---

## Scheduling / Performance Modules

### `WarmupAndFrameSkip`
Optional warmup + frame skipping that can **short-circuit** the rest of the pipeline.

- **Args**: `warmup_steps`, `every_n`, `warmup_control`, `stop_key` (default `__pipeline_stop__`)
- **Reads**: `context['global_step']`, `context['last_control']`
- **Writes**: `context['control']` and `context[stop_key]=True` when skipping
- **Notes**: Works because `PipelineEngine` stops when it sees `__pipeline_stop__`.

---

## TCP-style Command/State Assembly

### `CommandOneHotFromNextCommand`
Convert `next_command` to a one-hot vector (default 6-way, 1-based).

- **Args**: `cmd_key`, `out_key`, `num_cmds`, `one_based`, `clamp`, `negative_to`

### `NormalizeScalar`
Normalize a scalar: `out = float(in)/denom`.

- **Args**: `in_key`, `out_key`, `denom`

### `AssembleVector`
Concatenate scalar/vector parts into a single 1D float32 vector.

- **Args**: `keys` (list), `out_key`

---

## Composition Glue

### `SetValue`
Set a constant into the context.

- **Args**: `key`, `value`

### `RenameKeys`
Rename or mirror keys in the context.

- **Args**: `mapping` (dict), `keep_source`

---

## Optional Torch Modules (Lazy Import)

These modules import `torch` only when they run (or in `setup()`), so just importing pipeline modules stays lightweight.

### `NumpyToTorch`
Convert a numpy-like value to a `torch.Tensor`.

- **Args**: `in_key`, `out_key` (defaults to `in_key`), `device` (default `cuda`), `dtype` (default `float32`), `add_batch_dim`
- **Reads**: `context[in_key]`
- **Writes**: `context[out_key]` as `torch.Tensor`

### `ImageHWCToTorchCHW`
Convert an HxWxC numpy image into a torch CHW float tensor (optionally batched and normalized).

- **Args**: `in_key`, `out_key` (defaults to `in_key`), `device`, `divide_by_255`, `mean`, `std`, `add_batch_dim`
- **Writes**: `torch.Tensor` shaped `(1,3,H,W)` by default

### `TorchModelRunner`
Instantiate a torch model and run it every tick.

- **Key idea**: you can keep your “agent” entirely YAML-driven: extraction → tensor conversion → model inference → control postprocess.
- **Args**:
  - Model spec: either `model: {module, class_name, args}` or `model_module`/`model_class_name`/`model_args`
  - Checkpoint: `checkpoint_path`, `checkpoint_state_dict_key` (default `state_dict`), `checkpoint_prefix_strip` (optional)
  - Runtime: `device`, `eval_mode`, `strict`
  - Wiring: `inputs` (mapping model_input_name → context_key)
  - Outputs: `output_key` (store whole output) OR `output_map` (map output dict keys to context keys)

#### Minimal YAML example

```yaml
pipeline:
  - module: leaderboard.team_code.pipeline_modules
    class_name: ExtractCameraRGB
    args: {sensor_id: rgb, out_key: rgb}

  - module: leaderboard.team_code.pipeline_modules
    class_name: ImageHWCToTorchCHW
    args:
      in_key: rgb
      out_key: rgb_t
      device: cuda
      divide_by_255: true
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]

  - module: leaderboard.team_code.pipeline_modules
    class_name: TorchModelRunner
    args:
      model:
        module: my_pkg.models
        class_name: MyNet
        args: {hidden: 256}
      checkpoint_path: /path/to/ckpt.pth
      inputs:
        rgb: rgb_t
      output_key: model_output

  # ...then a custom postprocess module would turn model_output into `control`...
```

---

## Minimal Example (routing + target point)

```yaml
pipeline:
  - module: leaderboard.team_code.pipeline_modules
    class_name: ExtractGNSS
    args: {sensor_id: gps, out_key: gps}
  - module: leaderboard.team_code.pipeline_modules
    class_name: ExtractCompass
    args: {sensor_id: imu, out_key: compass}
  - module: leaderboard.team_code.pipeline_modules
    class_name: RoutePlannerNextCommand
    args: {gps_key: gps, out_cmd_key: next_command}
  - module: leaderboard.team_code.pipeline_modules
    class_name: TargetPointFromNextWaypoint
    args: {pos_key: pos, compass_key: compass, next_waypoint_key: next_waypoint, out_key: target_point}

  # ...your model module sets `waypoints`...

  - module: leaderboard.team_code.pipeline_modules
    class_name: PIDFromWaypoints
    args: {waypoints_key: waypoints, speed_key: speed, out_key: control}
```
