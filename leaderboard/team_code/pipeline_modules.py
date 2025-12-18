"""Reusable pipeline modules for config-defined (composed) agents.

These are intended for *new* agents that run through the `pipeline:` mode of
`ConsolidatedAgent`. Legacy agents should keep using their native implementations.

Design goals:
- Minimal assumptions about model philosophy (direct controls vs waypoints vs acc/steer)
- Works with Leaderboard `input_data` format: {sensor_id: (frame, raw)}
- Avoid importing heavy deps (torch/carla/cv2) at import time

Convention:
- Modules read/write a mutable `context` dict.
- `context['input_data']` is the raw Leaderboard input_data.
- `context['control']` may be produced as:
    - dict {steer, throttle, brake}
    - dict {steer, acc}
    - tuple/list of len 3
    - (or directly VehicleControl; ConsolidatedAgent will coerce)

"""
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _get_sensor(input_data: Mapping[str, Tuple[int, Any]], sensor_id: str) -> Any:
    if sensor_id not in input_data:
        raise KeyError(f"Missing sensor_id={sensor_id!r} in input_data")
    return input_data[sensor_id][1]


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    # CARLA images typically come as BGRA/BGR; most models want RGB.
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    return img[:, :, :3][:, :, ::-1]


# ---------------------------------------------------------------------------
# Extraction modules
# ---------------------------------------------------------------------------


class ExtractCameraRGB:
    """Extract a camera image and store it under context[out_key] as RGB uint8."""

    def __init__(self, sensor_id: str, out_key: str, bgr_to_rgb: bool = True):
        self.sensor_id = sensor_id
        self.out_key = out_key
        self.bgr_to_rgb = bool(bgr_to_rgb)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_data = context["input_data"]
        img = _get_sensor(input_data, self.sensor_id)
        if isinstance(img, np.ndarray) and self.bgr_to_rgb:
            img = _bgr_to_rgb(img)
        context[self.out_key] = img
        return context


class ExtractSpeed:
    """Extract speed (m/s) from a speedometer dict."""

    def __init__(self, sensor_id: str = "speed", out_key: str = "speed", dict_key: str = "speed"):
        self.sensor_id = sensor_id
        self.out_key = out_key
        self.dict_key = dict_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_data = context["input_data"]
        raw = _get_sensor(input_data, self.sensor_id)
        if isinstance(raw, dict):
            context[self.out_key] = float(raw.get(self.dict_key, 0.0))
            return context
        # Some agents pass speed as scalar/array.
        try:
            context[self.out_key] = float(raw)
        except Exception:
            context[self.out_key] = 0.0
        return context


class ExtractGNSS:
    """Extract GNSS (lat, lon) or (x,y) into a 2-vector."""

    def __init__(self, sensor_id: str = "gps", out_key: str = "gps", take: int = 2):
        self.sensor_id = sensor_id
        self.out_key = out_key
        self.take = int(take)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_data = context["input_data"]
        raw = _get_sensor(input_data, self.sensor_id)
        arr = np.array(raw)
        context[self.out_key] = arr[: self.take]
        return context


class ExtractCompass:
    """Extract compass from IMU (convention: last element is compass radians)."""

    def __init__(self, sensor_id: str = "imu", out_key: str = "compass"):
        self.sensor_id = sensor_id
        self.out_key = out_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_data = context["input_data"]
        raw = _get_sensor(input_data, self.sensor_id)
        arr = np.array(raw)
        compass = float(arr[-1]) if arr.size else 0.0
        if np.isnan(compass):
            compass = 0.0
        context[self.out_key] = compass
        return context


class ExtractLidarXYZ:
    """Extract LiDAR point cloud and store Nx3 (float32) under context[out_key].

    Leaderboard typically provides LiDAR as an (N,4) array-like (x,y,z,intensity).
    This module keeps only xyz and optionally flips y to match many agent conventions.
    """

    def __init__(self, sensor_id: str = "lidar", out_key: str = "lidar_xyz", flip_y: bool = True):
        self.sensor_id = sensor_id
        self.out_key = out_key
        self.flip_y = bool(flip_y)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_data = context["input_data"]
        raw = _get_sensor(input_data, self.sensor_id)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"LiDAR must be (N,>=3), got {arr.shape}")
        xyz = arr[:, :3].copy()
        if self.flip_y and xyz.shape[1] >= 2:
            xyz[:, 1] *= -1.0
        context[self.out_key] = xyz
        return context


# ---------------------------------------------------------------------------
# Route planner (matches InterFuser/TCP style)
# ---------------------------------------------------------------------------


class RoutePlannerNextCommand:
    """Compute next high-level command from global plan + current GNSS.

    Writes:
      - context['pos'] (planner-local position)
      - context['next_command'] (int)

    Assumes GNSS is stored in context[gps_key] as a 2-vector.
    """

    def __init__(
        self,
        gps_key: str = "gps",
        out_pos_key: str = "pos",
        out_wp_key: str = "next_waypoint",
        out_cmd_key: str = "next_command",
        min_distance: float = 4.0,
        max_distance: float = 50.0,
        gps_in_degrees: bool = True,
    ):
        self.gps_key = gps_key
        self.out_pos_key = out_pos_key
        self.out_wp_key = out_wp_key
        self.out_cmd_key = out_cmd_key
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.gps_in_degrees = bool(gps_in_degrees)
        self._planner = None

    def setup(self, agent: Any, full_config: Dict[str, Any]) -> None:
        # We lazily init on first run because Leaderboard usually calls
        # set_global_plan() AFTER agent.setup().
        return None

    def _ensure_planner(self, agent: Any):
        if self._planner is not None:
            return
        from team_code.planner import RoutePlanner

        self._planner = RoutePlanner(self.min_distance, self.max_distance)

        global_plan = getattr(agent, "_global_plan", None)
        if global_plan is None:
            raise RuntimeError("Global plan not set yet; RoutePlannerNextCommand needs set_global_plan()")
        self._planner.set_route(global_plan, gps=self.gps_in_degrees)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        agent = context.get("agent")
        if agent is None:
            raise KeyError("context['agent'] is required for RoutePlannerNextCommand")

        self._ensure_planner(agent)

        gps = np.array(context[self.gps_key])
        # Match InterFuser/TCP conversion
        pos = (gps - self._planner.mean) * self._planner.scale

        wp, cmd = self._planner.run_step(pos)

        context[self.out_pos_key] = pos
        context[self.out_wp_key] = np.array(wp)
        try:
            context[self.out_cmd_key] = int(cmd.value)
        except Exception:
            context[self.out_cmd_key] = int(cmd)
        return context


class TargetPointFromNextWaypoint:
    """Compute target_point from (pos, compass, next_waypoint).

    Matches the InterFuser/TCP convention:
      theta = compass + pi/2
      target_point = R^T * (next_wp - pos)

    Writes:
      - context[out_key] = np.ndarray shape (2,)
    """

    def __init__(
        self,
        pos_key: str = "pos",
        compass_key: str = "compass",
        next_waypoint_key: str = "next_waypoint",
        out_key: str = "target_point",
    ):
        self.pos_key = pos_key
        self.compass_key = compass_key
        self.next_waypoint_key = next_waypoint_key
        self.out_key = out_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pos = np.array(context[self.pos_key], dtype=np.float32)
        next_wp = np.array(context[self.next_waypoint_key], dtype=np.float32)
        compass = float(context.get(self.compass_key, 0.0))
        if np.isnan(compass):
            compass = 0.0

        theta = compass + np.pi / 2.0
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]], dtype=np.float32)
        target_point = R.T.dot(local_command_point)

        context[self.out_key] = target_point
        return context


# ---------------------------------------------------------------------------
# LiDAR processing (InterFuser-like)
# ---------------------------------------------------------------------------


class LidarHistogramFromXYZ:
    """Convert LiDAR xyz into histogram features (InterFuser-style).

    This mirrors the InterFuser preprocessing:
      - transform points into planner-local frame based on compass+pos
      - run `team_code.utils.lidar_to_histogram_features`
      - optionally reuse a previous histogram for stability

    Writes:
      - context[out_key] = np.ndarray float32 with shape (C,H,W)
    """

    def __init__(
        self,
        lidar_xyz_key: str = "lidar_xyz",
        compass_key: str = "compass",
        pos_key: str = "pos",
        out_key: str = "lidar_hist",
        crop: int = 224,
        reuse_every_n: int = 2,
        warmup_reuse_steps: int = 4,
    ):
        self.lidar_xyz_key = lidar_xyz_key
        self.compass_key = compass_key
        self.pos_key = pos_key
        self.out_key = out_key
        self.crop = int(crop)
        self.reuse_every_n = max(1, int(reuse_every_n))
        self.warmup_reuse_steps = max(0, int(warmup_reuse_steps))

        self._step = -1
        self._prev = None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1

        lidar_xyz = np.asarray(context[self.lidar_xyz_key], dtype=np.float32)
        compass = float(context.get(self.compass_key, 0.0))
        if np.isnan(compass):
            compass = 0.0
        pos = np.asarray(context[self.pos_key], dtype=np.float32)
        if pos.shape[0] < 2:
            raise ValueError(f"pos must be a 2-vector, got {pos}")

        from team_code.utils import lidar_to_histogram_features, transform_2d_points

        xyz = np.zeros((lidar_xyz.shape[0], 3), dtype=np.float32)
        xyz[:, :3] = lidar_xyz[:, :3]
        xyz[:, 2] = lidar_xyz[:, 2]

        full_lidar = transform_2d_points(
            xyz,
            np.pi / 2.0 - compass,
            -float(pos[0]),
            -float(pos[1]),
            np.pi / 2.0 - compass,
            -float(pos[0]),
            -float(pos[1]),
        )
        feats = lidar_to_histogram_features(full_lidar, crop=self.crop)

        if (self._step % self.reuse_every_n) == 0 or self._step < self.warmup_reuse_steps:
            self._prev = feats
        context[self.out_key] = self._prev if self._prev is not None else feats
        return context


# ---------------------------------------------------------------------------
# Command/state assembly (TCP-style)
# ---------------------------------------------------------------------------


class CommandOneHotFromNextCommand:
    """Convert next_command into a one-hot vector.

    TCP uses 6 commands with 1-based ids coming from planner. Some logs contain
    command<0; TCP maps those to 4.

    Writes:
      - context[out_key] = np.ndarray shape (num_cmds,)
    """

    def __init__(
        self,
        cmd_key: str = "next_command",
        out_key: str = "cmd_one_hot",
        num_cmds: int = 6,
        one_based: bool = True,
        clamp: bool = True,
        negative_to: Optional[int] = 4,
    ):
        self.cmd_key = cmd_key
        self.out_key = out_key
        self.num_cmds = int(num_cmds)
        self.one_based = bool(one_based)
        self.clamp = bool(clamp)
        self.negative_to = negative_to

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cmd = int(context.get(self.cmd_key, 0))
        if cmd < 0 and self.negative_to is not None:
            cmd = int(self.negative_to)

        idx = cmd - 1 if self.one_based else cmd
        if self.clamp:
            idx = int(np.clip(idx, 0, self.num_cmds - 1))
        if idx < 0 or idx >= self.num_cmds:
            raise ValueError(f"Command index out of range: cmd={cmd} idx={idx} num_cmds={self.num_cmds}")

        one_hot = np.zeros((self.num_cmds,), dtype=np.float32)
        one_hot[idx] = 1.0
        context[self.out_key] = one_hot
        return context


class NormalizeScalar:
    """Normalize a scalar: out = float(in)/denom."""

    def __init__(self, in_key: str, out_key: str, denom: float = 1.0):
        self.in_key = in_key
        self.out_key = out_key
        self.denom = float(denom) if float(denom) != 0.0 else 1.0

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context[self.out_key] = float(context.get(self.in_key, 0.0)) / self.denom
        return context


class AssembleVector:
    """Concatenate scalars/vectors into a single 1D float32 numpy vector."""

    def __init__(self, keys, out_key: str = "state"):
        self.keys = list(keys)
        self.out_key = out_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        parts = []
        for key in self.keys:
            v = context.get(key)
            if v is None:
                raise KeyError(f"Missing context[{key!r}] for AssembleVector")
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            parts.append(arr)
        context[self.out_key] = np.concatenate(parts, axis=0).astype(np.float32)
        return context


# ---------------------------------------------------------------------------
# Control utilities
# ---------------------------------------------------------------------------


class ClampControl:
    """Clamp and sanitize a control-like dict in context[control_key]."""

    def __init__(
        self,
        control_key: str = "control",
        steer_clip: float = 1.0,
        throttle_clip: float = 1.0,
        brake_clip: float = 1.0,
        zero_throttle_when_braking_over: float = 0.5,
        brake_wins_over_throttle: bool = True,
    ):
        self.control_key = control_key
        self.steer_clip = float(steer_clip)
        self.throttle_clip = float(throttle_clip)
        self.brake_clip = float(brake_clip)
        self.zero_throttle_when_braking_over = float(zero_throttle_when_braking_over)
        self.brake_wins_over_throttle = bool(brake_wins_over_throttle)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctrl = context.get(self.control_key)
        if not isinstance(ctrl, dict):
            return context

        steer = float(ctrl.get("steer", 0.0))
        throttle = float(ctrl.get("throttle", 0.0))
        brake = float(ctrl.get("brake", 0.0))

        steer = float(np.clip(steer, -self.steer_clip, self.steer_clip))
        throttle = float(np.clip(throttle, 0.0, self.throttle_clip))
        brake = float(np.clip(brake, 0.0, self.brake_clip))

        if self.brake_wins_over_throttle and throttle > brake:
            brake = 0.0
        if brake > self.zero_throttle_when_braking_over:
            throttle = 0.0

        context[self.control_key] = {"steer": steer, "throttle": throttle, "brake": brake}
        return context


class BlendControls:
    """Blend two control dicts: out = alpha*a + (1-alpha)*b."""

    def __init__(
        self,
        a_key: str,
        b_key: str,
        out_key: str = "control",
        alpha: float = 0.3,
    ):
        self.a_key = a_key
        self.b_key = b_key
        self.out_key = out_key
        self.alpha = float(alpha)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        a = context.get(self.a_key) or {}
        b = context.get(self.b_key) or {}
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise TypeError("BlendControls expects dict controls")

        def g(d, k):
            try:
                return float(d.get(k, 0.0))
            except Exception:
                return 0.0

        out = {
            "steer": self.alpha * g(a, "steer") + (1.0 - self.alpha) * g(b, "steer"),
            "throttle": self.alpha * g(a, "throttle") + (1.0 - self.alpha) * g(b, "throttle"),
            "brake": self.alpha * g(a, "brake") + (1.0 - self.alpha) * g(b, "brake"),
        }
        context[self.out_key] = out
        return context


# ---------------------------------------------------------------------------
# Small general-purpose modules (composition glue)
# ---------------------------------------------------------------------------


class SetValue:
    """Set a constant value into the context."""

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context[self.key] = self.value
        return context


class RenameKeys:
    """Rename/mirror context keys according to a mapping."""

    def __init__(self, mapping: Dict[str, str], keep_source: bool = True):
        self.mapping = dict(mapping)
        self.keep_source = bool(keep_source)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        for src, dst in self.mapping.items():
            if src in context:
                context[dst] = context[src]
                if not self.keep_source and dst != src:
                    try:
                        del context[src]
                    except Exception:
                        pass
        return context


# ---------------------------------------------------------------------------
# Optional Torch helpers (lazy-import torch)
# ---------------------------------------------------------------------------


def _import_symbol(module_path: str, class_name: str):
    import importlib

    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, class_name)
    except AttributeError as exc:
        raise ImportError(f"{class_name!r} not found in {module_path!r}") from exc


class NumpyToTorch:
    """Convert a numpy-like value in context[in_key] into a torch.Tensor.

    This module imports torch lazily at runtime.
    """

    def __init__(
        self,
        in_key: str,
        out_key: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float32",
        add_batch_dim: bool = False,
    ):
        self.in_key = in_key
        self.out_key = out_key or in_key
        self.device = device
        self.dtype = dtype
        self.add_batch_dim = bool(add_batch_dim)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        v = context.get(self.in_key)
        if isinstance(v, torch.Tensor):
            t = v
        else:
            arr = np.asarray(v)
            t = torch.from_numpy(arr)

        # dtype handling
        if self.dtype:
            try:
                t = t.to(getattr(torch, self.dtype))
            except Exception:
                t = t.float()

        if self.add_batch_dim and t.ndim >= 1:
            t = t.unsqueeze(0)

        if self.device:
            t = t.to(self.device)
        context[self.out_key] = t
        return context


class ImageHWCToTorchCHW:
    """Convert an HxWxC numpy image to torch CHW float tensor.

    - Optionally divides by 255.
    - Optionally normalizes with mean/std (RGB order).
    """

    def __init__(
        self,
        in_key: str,
        out_key: Optional[str] = None,
        device: str = "cuda",
        divide_by_255: bool = True,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        add_batch_dim: bool = True,
    ):
        self.in_key = in_key
        self.out_key = out_key or in_key
        self.device = device
        self.divide_by_255 = bool(divide_by_255)
        self.mean = mean
        self.std = std
        self.add_batch_dim = bool(add_batch_dim)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        img = np.asarray(context.get(self.in_key))
        if img.ndim != 3:
            raise ValueError(f"Expected HxWxC image for {self.in_key!r}, got shape={img.shape}")
        if img.shape[2] < 3:
            raise ValueError(f"Expected at least 3 channels for {self.in_key!r}, got shape={img.shape}")

        img = img[:, :, :3]
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        t = t.to(dtype=torch.float32)
        if self.divide_by_255:
            t = t / 255.0

        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)
            t = (t - mean) / std

        if self.add_batch_dim:
            t = t.unsqueeze(0)

        if self.device:
            t = t.to(self.device)
        context[self.out_key] = t
        return context


class TorchModelRunner:
    """Instantiate and run a torch model from config.

    This is intentionally minimal and opinionated:
    - torch is imported lazily
    - `setup()` instantiates the model once and loads an optional checkpoint
    - `run()` builds a dict of model inputs from context and calls the model

    Typical usage is: sensor extraction -> tensor conversion -> model -> postprocess.
    """

    def __init__(
        self,
        model: Optional[Dict[str, Any]] = None,
        model_module: Optional[str] = None,
        model_class_name: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_state_dict_key: str = "state_dict",
        checkpoint_prefix_strip: Optional[str] = None,
        device: str = "cuda",
        eval_mode: bool = True,
        strict: bool = False,
        inputs: Optional[Dict[str, str]] = None,
        output_key: str = "model_output",
        output_map: Optional[Dict[str, str]] = None,
    ):
        self.model_spec = model
        self.model_module = model_module
        self.model_class_name = model_class_name
        self.model_args = model_args or {}
        self.checkpoint_path = checkpoint_path
        self.checkpoint_state_dict_key = checkpoint_state_dict_key
        self.checkpoint_prefix_strip = checkpoint_prefix_strip
        self.device = device
        self.eval_mode = bool(eval_mode)
        self.strict = bool(strict)
        self.inputs = inputs or {}
        self.output_key = output_key
        self.output_map = output_map

        self._model = None

    def setup(self, agent: Any, full_config: Dict[str, Any]) -> None:
        import torch

        spec = self.model_spec or {}
        module_path = spec.get("module") or self.model_module
        class_name = spec.get("class_name") or spec.get("class") or self.model_class_name
        args = spec.get("args") or self.model_args or {}

        if not module_path or not class_name:
            raise ValueError("TorchModelRunner requires model.module and model.class_name")

        ModelClass = _import_symbol(module_path, class_name)
        self._model = ModelClass(**args)

        if self.device:
            self._model = self._model.to(self.device)

        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            state = ckpt
            if isinstance(ckpt, dict) and self.checkpoint_state_dict_key in ckpt:
                state = ckpt[self.checkpoint_state_dict_key]

            if isinstance(state, dict) and self.checkpoint_prefix_strip:
                prefix = str(self.checkpoint_prefix_strip)
                new_state = {}
                for k, v in state.items():
                    if isinstance(k, str) and k.startswith(prefix):
                        new_state[k[len(prefix) :]] = v
                    else:
                        new_state[k] = v
                state = new_state

            if not isinstance(state, dict):
                raise ValueError("Checkpoint did not contain a state_dict-like mapping")
            self._model.load_state_dict(state, strict=self.strict)

        if self.eval_mode:
            self._model.eval()
        return None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError("TorchModelRunner.setup() was not called")

        # Build inputs
        model_inputs = {}
        for model_key, ctx_key in self.inputs.items():
            if ctx_key not in context:
                raise KeyError(f"Missing context[{ctx_key!r}] for model input {model_key!r}")
            model_inputs[str(model_key)] = context[ctx_key]

        # Allow no explicit mapping: if inputs is empty, try using context['model_inputs'].
        if not model_inputs:
            v = context.get("model_inputs")
            if not isinstance(v, dict):
                raise ValueError("TorchModelRunner needs `inputs` mapping or context['model_inputs'] dict")
            model_inputs = v

        out = self._model(model_inputs)

        if self.output_map and isinstance(out, dict):
            for out_key, ctx_key in self.output_map.items():
                if out_key not in out:
                    raise KeyError(f"Model output missing key {out_key!r}")
                context[ctx_key] = out[out_key]
            return context

        context[self.output_key] = out
        return context


class WarmupAndFrameSkip:
    """Optionally short-circuit the pipeline for warmup and/or frame skipping.

    Requires ConsolidatedAgent to pass:
      - context['global_step'] (int)
      - context['last_control'] (optional)

    Behavior:
      - For first warmup_steps ticks: output warmup_control and stop.
      - Thereafter, if every_n > 1 and global_step % every_n != 0:
          output last_control (or warmup_control) and stop.
      - Otherwise: allow pipeline to continue.
    """

    def __init__(
        self,
        warmup_steps: int = 0,
        every_n: int = 1,
        warmup_control: Optional[Dict[str, float]] = None,
        stop_key: str = "__pipeline_stop__",
    ):
        self.warmup_steps = int(warmup_steps)
        self.every_n = max(1, int(every_n))
        self.warmup_control = warmup_control or {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        self.stop_key = str(stop_key)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        step = int(context.get("global_step", 0))

        if self.warmup_steps > 0 and step < self.warmup_steps:
            context["control"] = context.get("last_control") or dict(self.warmup_control)
            context[self.stop_key] = True
            return context

        if self.every_n > 1 and (step % self.every_n) != 0:
            context["control"] = context.get("last_control") or dict(self.warmup_control)
            context[self.stop_key] = True
            return context

        return context


# ---------------------------------------------------------------------------
# Control-format modules
# ---------------------------------------------------------------------------


class ControlFromAccSteer:
    """Convert acc+steer to control dict.

    Inputs:
      - context[acc_key]: acceleration-like scalar
      - context[steer_key]: steer scalar

    Output:
      - context['control'] = {steer, throttle, brake}

    Convention:
      acc >= 0 => throttle=acc, brake=0
      acc < 0  => throttle=0, brake=abs(acc)
    """

    def __init__(
        self,
        acc_key: str = "acc",
        steer_key: str = "steer",
        out_key: str = "control",
        throttle_clip: float = 1.0,
        brake_clip: float = 1.0,
    ):
        self.acc_key = acc_key
        self.steer_key = steer_key
        self.out_key = out_key
        self.throttle_clip = float(throttle_clip)
        self.brake_clip = float(brake_clip)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        acc = float(context.get(self.acc_key, 0.0))
        steer = float(context.get(self.steer_key, 0.0))

        if acc >= 0.0:
            throttle = min(self.throttle_clip, acc)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self.brake_clip, abs(acc))

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": brake}
        return context


class _PIDCfg(object):
    def __init__(
        self,
        turn_KP=1.0,
        turn_KI=0.0,
        turn_KD=0.0,
        turn_n=20,
        speed_KP=1.0,
        speed_KI=0.0,
        speed_KD=0.0,
        speed_n=20,
        brake_speed=0.4,
        brake_ratio=1.1,
        clip_delta=0.25,
        max_throttle=0.75,
        **kwargs
    ):
        # Keep config strict (unknown keys usually indicate typos).
        if kwargs:
            raise TypeError("Unknown PID config keys: {}".format(", ".join(sorted(kwargs.keys()))))

        self.turn_KP = float(turn_KP)
        self.turn_KI = float(turn_KI)
        self.turn_KD = float(turn_KD)
        self.turn_n = int(turn_n)

        self.speed_KP = float(speed_KP)
        self.speed_KI = float(speed_KI)
        self.speed_KD = float(speed_KD)
        self.speed_n = int(speed_n)

        self.brake_speed = float(brake_speed)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.max_throttle = float(max_throttle)


class PIDFromWaypoints:
    """Compute control from predicted waypoints + current speed.

    Inputs:
      - context[waypoints_key]: array-like (N,2) in meters
      - context[speed_key]: float speed (m/s)

    Output:
      - context[out_key] = {steer, throttle, brake}

    This is a lightweight numpy implementation modeled after the repository's
    `team_code/controller.py` logic, but without torch dependencies.
    """

    def __init__(
        self,
        waypoints_key: str = "waypoints",
        speed_key: str = "speed",
        out_key: str = "control",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.waypoints_key = waypoints_key
        self.speed_key = speed_key
        self.out_key = out_key
        self.cfg = _PIDCfg(**(config or {}))

        self._turn_window = [0.0] * int(self.cfg.turn_n)
        self._speed_window = [0.0] * int(self.cfg.speed_n)

    def _pid_step(self, window, error: float, kp: float, ki: float, kd: float) -> float:
        window.pop(0)
        window.append(float(error))
        integral = float(np.mean(window)) if len(window) >= 2 else 0.0
        derivative = float(window[-1] - window[-2]) if len(window) >= 2 else 0.0
        return kp * float(error) + ki * integral + kd * derivative

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wps = np.array(context[self.waypoints_key], dtype=np.float32)
        if wps.ndim != 2 or wps.shape[0] < 2 or wps.shape[1] < 2:
            raise ValueError(f"waypoints must be (N,2) with N>=2, got {wps.shape}")

        # Match the repo's PID convention (forward is negative y)
        wps = wps.copy()
        wps[:, 1] *= -1.0

        speed = float(context.get(self.speed_key, 0.0))

        desired_speed = float(np.linalg.norm(wps[0] - wps[1]) * 2.0)
        brake = bool(desired_speed < self.cfg.brake_speed or (speed / max(desired_speed, 1e-3)) > self.cfg.brake_ratio)

        aim = (wps[1] + wps[0]) / 2.0
        angle = float(np.degrees(np.pi / 2.0 - np.arctan2(aim[1], aim[0])) / 90.0)
        if speed < 0.01:
            angle = 0.0

        steer = self._pid_step(self._turn_window, angle, self.cfg.turn_KP, self.cfg.turn_KI, self.cfg.turn_KD)
        steer = float(np.clip(steer, -1.0, 1.0))

        delta = float(np.clip(desired_speed - speed, 0.0, self.cfg.clip_delta))
        throttle = self._pid_step(self._speed_window, delta, self.cfg.speed_KP, self.cfg.speed_KI, self.cfg.speed_KD)
        throttle = float(np.clip(throttle, 0.0, self.cfg.max_throttle))
        throttle = float(throttle if not brake else 0.0)

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": float(brake)}
        return context
