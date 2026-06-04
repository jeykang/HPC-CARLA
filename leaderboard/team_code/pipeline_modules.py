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

    - Optionally resizes via PIL before cropping (resize_wh = (width, height)).
    - Optionally center-crops to a square (center_crop = int side length).
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
        resize_wh: Optional[Tuple[int, int]] = None,
        center_crop: Optional[int] = None,
    ):
        self.in_key = in_key
        self.out_key = out_key or in_key
        self.device = device
        self.divide_by_255 = bool(divide_by_255)
        self.mean = mean
        self.std = std
        self.add_batch_dim = bool(add_batch_dim)
        self.resize_wh = tuple(resize_wh) if resize_wh is not None else None
        self.center_crop = int(center_crop) if center_crop is not None else None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        img = np.asarray(context.get(self.in_key))
        if img.ndim != 3:
            raise ValueError(f"Expected HxWxC image for {self.in_key!r}, got shape={img.shape}")
        if img.shape[2] < 3:
            raise ValueError(f"Expected at least 3 channels for {self.in_key!r}, got shape={img.shape}")

        img = img[:, :, :3]

        if self.resize_wh is not None or self.center_crop is not None:
            from PIL import Image as _PILImage
            pil = _PILImage.fromarray(img.astype(np.uint8))
            if self.resize_wh is not None:
                pil = pil.resize(self.resize_wh, _PILImage.BILINEAR)
            if self.center_crop is not None:
                w, h = pil.size
                c = self.center_crop
                left = (w - c) // 2
                top = (h - c) // 2
                pil = pil.crop((left, top, left + c, top + c))
            img = np.asarray(pil)

        t = torch.from_numpy(img.copy()).permute(2, 0, 1).contiguous()
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
        no_grad: bool = True,
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
        self.no_grad = bool(no_grad)

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

        import torch
        if self.no_grad:
            with torch.no_grad():
                out = self._model(model_inputs)
        else:
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


# ---------------------------------------------------------------------------
# InterFuser-specific modules
# ---------------------------------------------------------------------------


class InterfuserOutputUnpack:
    """Unpack the 6-tuple output of interfuser_baseline into named context keys.

    The model returns (traffic_meta, pred_waypoints, is_junction,
    traffic_light_state, stop_sign, aux). This module converts each tensor to
    numpy and applies softmax to the three classification heads.
    """

    def __init__(
        self,
        model_output_key: str = "model_output",
        traffic_meta_key: str = "traffic_meta_raw",
        pred_waypoints_key: str = "pred_waypoints",
        is_junction_key: str = "is_junction",
        traffic_light_key: str = "traffic_light_state",
        stop_sign_key: str = "stop_sign",
    ):
        self.model_output_key = model_output_key
        self.traffic_meta_key = traffic_meta_key
        self.pred_waypoints_key = pred_waypoints_key
        self.is_junction_key = is_junction_key
        self.traffic_light_key = traffic_light_key
        self.stop_sign_key = stop_sign_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch.nn.functional as F

        out = context[self.model_output_key]
        traffic_meta, pred_waypoints, is_junction, traffic_light_state, stop_sign, _ = out

        context[self.traffic_meta_key] = traffic_meta.detach().cpu().numpy()[0]
        context[self.pred_waypoints_key] = pred_waypoints.detach().cpu().numpy()[0]
        context[self.is_junction_key] = float(
            F.softmax(is_junction, dim=1).detach().cpu().numpy().reshape(-1)[0]
        )
        context[self.traffic_light_key] = float(
            F.softmax(traffic_light_state, dim=1).detach().cpu().numpy().reshape(-1)[0]
        )
        context[self.stop_sign_key] = float(
            F.softmax(stop_sign, dim=1).detach().cpu().numpy().reshape(-1)[0]
        )
        return context


class TrafficMetaTracker:
    """Apply InterFuser's Tracker + exponential moving average to traffic_meta.

    Mirrors the original agent's per-step logic:
      - Update tracker and EMA on even steps and during warmup.
      - Always output the current EMA, so the controller always has valid data.
    """

    def __init__(
        self,
        traffic_meta_key: str = "traffic_meta_raw",
        gps_key: str = "pos",
        compass_key: str = "compass",
        out_key: str = "traffic_meta",
        momentum: float = 0.0,
        update_every_n: int = 2,
        warmup_always_update: int = 4,
    ):
        self.traffic_meta_key = traffic_meta_key
        self.gps_key = gps_key
        self.compass_key = compass_key
        self.out_key = out_key
        self.momentum = float(momentum)
        self.update_every_n = max(1, int(update_every_n))
        self.warmup_always_update = max(0, int(warmup_always_update))
        self._tracker = None
        self._avg: Optional[np.ndarray] = None
        self._step = -1

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1

        if self._tracker is None:
            from team_code.tracker import Tracker
            self._tracker = Tracker()
            self._avg = np.zeros((400, 7), dtype=np.float32)

        should_update = (
            (self._step % self.update_every_n == 0)
            or (self._step < self.warmup_always_update)
        )

        if should_update:
            traffic_meta = np.asarray(context[self.traffic_meta_key], dtype=np.float32)
            gps = np.array(context[self.gps_key], dtype=np.float64)
            compass = float(context.get(self.compass_key, 0.0))
            if np.isnan(compass):
                compass = 0.0

            updated = self._tracker.update_and_predict(
                traffic_meta.reshape(20, 20, -1),
                gps,
                compass,
                self._step // self.update_every_n,
            )
            updated = updated.reshape(400, -1).astype(np.float32)
            self._avg = (
                self.momentum * self._avg + (1.0 - self.momentum) * updated
            )

        context[self.out_key] = self._avg
        return context


class InterfuserControllerModule:
    """Wrap InterfuserController for use in the modular pipeline.

    Accepts the same config knobs as GlobalConfig so all PID and safety
    parameters can be tuned directly from the YAML without a separate
    config file.
    """

    def __init__(
        self,
        speed_key: str = "speed",
        waypoints_key: str = "pred_waypoints",
        junction_key: str = "is_junction",
        traffic_light_key: str = "traffic_light_state",
        stop_sign_key: str = "stop_sign",
        traffic_meta_key: str = "traffic_meta",
        out_key: str = "control",
        turn_KP: float = 1.25,
        turn_KI: float = 0.75,
        turn_KD: float = 0.3,
        turn_n: int = 40,
        speed_KP: float = 5.0,
        speed_KI: float = 0.5,
        speed_KD: float = 1.0,
        speed_n: int = 40,
        max_throttle: float = 0.75,
        brake_speed: float = 0.1,
        brake_ratio: float = 1.1,
        clip_delta: float = 0.35,
        max_speed: float = 5.0,
        collision_buffer: Any = (2.5, 1.2),
        detect_threshold: float = 0.04,
    ):
        self.speed_key = speed_key
        self.waypoints_key = waypoints_key
        self.junction_key = junction_key
        self.traffic_light_key = traffic_light_key
        self.stop_sign_key = stop_sign_key
        self.traffic_meta_key = traffic_meta_key
        self.out_key = out_key
        self._cfg_kwargs = dict(
            turn_KP=float(turn_KP), turn_KI=float(turn_KI),
            turn_KD=float(turn_KD), turn_n=int(turn_n),
            speed_KP=float(speed_KP), speed_KI=float(speed_KI),
            speed_KD=float(speed_KD), speed_n=int(speed_n),
            max_throttle=float(max_throttle), brake_speed=float(brake_speed),
            brake_ratio=float(brake_ratio), clip_delta=float(clip_delta),
            max_speed=float(max_speed),
            collision_buffer=list(collision_buffer),
            detect_threshold=float(detect_threshold),
        )
        self._ctrl = None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._ctrl is None:
            from team_code.interfuser.interfuser_config import GlobalConfig
            from team_code.interfuser.interfuser_controller import InterfuserController
            cfg = GlobalConfig(**self._cfg_kwargs)
            self._ctrl = InterfuserController(cfg)

        speed = float(context[self.speed_key])
        waypoints = np.asarray(context[self.waypoints_key])
        junction = float(context[self.junction_key])
        light = float(context[self.traffic_light_key])
        stop = float(context[self.stop_sign_key])
        meta = np.asarray(context[self.traffic_meta_key])

        steer, throttle, brake, _ = self._ctrl.run_step(
            speed, waypoints, junction, light, stop, meta
        )

        steer = float(steer)
        throttle = float(throttle)
        brake = float(brake)
        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": brake}
        return context


# ─────────────────────────────────────────────────────────────────────────────
# TCP modules
# ─────────────────────────────────────────────────────────────────────────────

class TCPModelRunner:
    """Load the TCP model and run forward(img, state, target_point).

    The TCP checkpoint stores weights under a "model." prefix; we strip it.
    GlobalConfig is constructed with pred_len=4, seq_len=1 (TCP defaults).
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_key: str = "rgb_t",
        state_key: str = "tcp_state",
        target_point_key: str = "target_point_t",
        out_key: str = "tcp_pred",
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.img_key = img_key
        self.state_key = state_key
        self.target_point_key = target_point_key
        self.out_key = out_key
        self.device = device
        self._model = None

    def _load_model(self):
        import torch
        from team_code.tcp.TCP.model import TCP
        from team_code.tcp.TCP.config import GlobalConfig

        cfg = GlobalConfig()
        cfg.pred_len = 4
        cfg.seq_len = 1

        model = TCP(cfg)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix added by pytorch-lightning
        cleaned = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(cleaned, strict=False)
        model.to(self.device)
        model.eval()
        self._model = model

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        if self._model is None:
            self._load_model()

        img = context[self.img_key]
        state = context[self.state_key]
        target_point = context[self.target_point_key]

        with torch.no_grad():
            pred = self._model(img, state, target_point)

        context[self.out_key] = pred
        return context


class TCPStateAssemble:
    """Build the (1, 9) state tensor for TCP: [speed/12, target_x, target_y, cmd_one_hot(6)].

    Command is 0-based (TCP convention); negative commands map to index 3 (follow-lane).
    """

    def __init__(
        self,
        speed_key: str = "speed",
        target_point_key: str = "target_point",
        cmd_key: str = "next_command",
        out_key: str = "tcp_state",
        device: str = "cuda",
    ):
        self.speed_key = speed_key
        self.target_point_key = target_point_key
        self.cmd_key = cmd_key
        self.out_key = out_key
        self.device = device

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        speed = float(context[self.speed_key]) / 12.0
        tp = np.asarray(context[self.target_point_key], dtype=np.float32).flatten()
        cmd = int(context[self.cmd_key])
        # tcp_agent.py: negative → 4 (LANEFOLLOW, 1-based) → -1 → 3; positive → subtract 1
        if cmd < 0:
            cmd = 3  # 0-based LANEFOLLOW
        else:
            cmd = cmd - 1  # convert 1-based RoadOption value to 0-based index
        one_hot = np.zeros(6, dtype=np.float32)
        one_hot[min(cmd, 5)] = 1.0

        state_np = np.concatenate([[speed, tp[0], tp[1]], one_hot]).astype(np.float32)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        context[self.out_key] = state_t
        return context


class TCPBetaControl:
    """Sample Beta-distribution action from TCP pred dict → {steer, throttle, brake}.

    Mirrors TCP.process_action() using deterministic Beta mean for evaluation.
    """

    def __init__(
        self,
        pred_key: str = "tcp_pred",
        out_key: str = "ctrl_ctrl",
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta: float = 0.25,
        max_throttle: float = 0.75,
    ):
        self.pred_key = pred_key
        self.out_key = out_key
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.max_throttle = max_throttle

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        pred = context[self.pred_key]
        # mu_branches and sigma_branches are Softplus outputs → they ARE α and β directly,
        # not mean/sigma. Implement _get_action_beta from tcp/TCP/model.py lines 231-249.
        alpha = pred["mu_branches"][0].clamp(min=1e-4)    # shape (2,), α ∈ (0, ∞)
        beta_p = pred["sigma_branches"][0].clamp(min=1e-4)  # shape (2,), β ∈ (0, ∞)

        x = torch.full_like(alpha, 0.5)
        # mode = (α-1)/(α+β-2) when both > 1
        mask1 = (alpha > 1) & (beta_p > 1)
        x[mask1] = (alpha[mask1] - 1.0) / (alpha[mask1] + beta_p[mask1] - 2.0)
        # boundary cases
        mask2 = (alpha <= 1) & (beta_p > 1);  x[mask2] = 0.0
        mask3 = (alpha > 1) & (beta_p <= 1);  x[mask3] = 1.0
        # both ≤ 1: use mean as tiebreak (mode is bimodal / undefined)
        mask4 = (alpha <= 1) & (beta_p <= 1)
        x[mask4] = alpha[mask4] / (alpha[mask4] + beta_p[mask4]).clamp(min=1e-5)

        action = (x * 2.0 - 1.0).cpu().numpy()   # scale [0,1] → [-1,1]
        acc = float(action[0])
        steer = float(np.clip(action[1], -1.0, 1.0))

        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(acc)

        throttle = min(throttle, self.max_throttle)
        if brake < 0.05:
            brake = 0.0

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": brake}
        return context


class TCPPIDControl:
    """PID control on TCP predicted waypoints → {steer, throttle, brake}.

    Faithful re-implementation of TCP.control_pid(), including the angle_target
    outlier-rejection logic that suppresses waypoint noise on straight roads.
    """

    def __init__(
        self,
        pred_key: str = "tcp_pred",
        speed_key: str = "speed",
        target_point_key: str = "target_point",
        out_key: str = "ctrl_traj",
        aim_dist: float = 4.0,
        angle_thresh: float = 0.3,
        dist_thresh: float = 10.0,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta: float = 0.25,
        max_throttle: float = 0.75,
        turn_KP: float = 1.25,
        turn_KI: float = 0.75,
        turn_KD: float = 0.3,
        turn_n: int = 40,
        speed_KP: float = 5.0,
        speed_KI: float = 0.5,
        speed_KD: float = 1.0,
        speed_n: int = 40,
    ):
        self.pred_key = pred_key
        self.speed_key = speed_key
        self.target_point_key = target_point_key
        self.out_key = out_key
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.max_throttle = max_throttle
        self._turn_KP = turn_KP
        self._turn_KI = turn_KI
        self._turn_KD = turn_KD
        self._turn_n = turn_n
        self._speed_KP = speed_KP
        self._speed_KI = speed_KI
        self._speed_KD = speed_KD
        self._speed_n = speed_n
        self._turn_window = None
        self._speed_window = None

    def _ensure_windows(self):
        import collections
        if self._turn_window is None:
            self._turn_window = collections.deque(maxlen=self._turn_n)
        if self._speed_window is None:
            self._speed_window = collections.deque(maxlen=self._speed_n)

    def _pid(self, window, KP, KI, KD, error):
        window.append(error)
        if len(window) >= 2:
            integral = sum(window) / len(window)
            derivative = window[-1] - window[-2]
        else:
            integral = error
            derivative = 0.0
        return KP * error + KI * integral + KD * derivative

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_windows()

        pred = context[self.pred_key]
        speed = float(context[self.speed_key])

        wps = pred["pred_wp"][0].cpu().numpy()  # (pred_len, 2)
        wps[:, 1] *= -1  # forward is negative y in TCP waypoints

        target = np.asarray(context[self.target_point_key], dtype=np.float64).flatten()[:2].copy()
        target[1] *= -1  # same y-flip as waypoints

        num_pairs = len(wps) - 1
        desired_speed_dyn = 0.0
        best_norm = 1e5
        aim = wps[0]
        for i in range(num_pairs):
            desired_speed_dyn += np.linalg.norm(wps[i + 1] - wps[i]) * 2.0 / num_pairs
            norm = np.linalg.norm((wps[i + 1] + wps[i]) / 2.0)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = wps[i]
                best_norm = norm

        aim_last = wps[-1] - wps[-2]

        angle = float(np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90.0)
        angle_last = float(np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90.0)
        angle_target = float(np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90.0)

        # Outlier rejection: prefer GPS target angle when waypoints are noisier
        # (reduces straight-road drift; helps with sudden turn commands)
        use_target = abs(angle_target) < abs(angle) or (
            abs(angle_target - angle_last) > self.angle_thresh and target[1] < self.dist_thresh
        )
        angle_final = angle_target if use_target else angle

        steer_delta = self._pid(self._turn_window, self._turn_KP, self._turn_KI,
                                self._turn_KD, angle_final)
        steer = float(np.clip(steer_delta, -1.0, 1.0))

        delta = float(np.clip(desired_speed_dyn - speed, 0.0, self.clip_delta))
        throttle_delta = self._pid(self._speed_window, self._speed_KP, self._speed_KI,
                                   self._speed_KD, delta)
        throttle = float(np.clip(throttle_delta, 0.0, self.max_throttle))
        brake = bool(desired_speed_dyn < self.brake_speed or (
            desired_speed_dyn > 1e-3 and speed / desired_speed_dyn > self.brake_ratio
        ))

        if brake:
            throttle = 0.0

        context[self.out_key] = {
            "steer": steer,
            "throttle": throttle,
            "brake": float(brake),
        }
        return context


class TurningStatusDetector:
    """Classify current driving as turning (1) or straight (0).

    Rolling 20-frame window of |steer| from the previously emitted control dict.
    Status=1 when >10 of the last 20 frames have |steer| > 0.1.
    """

    def __init__(
        self,
        last_control_key: str = "control",
        out_key: str = "turning_status",
        window: int = 20,
        threshold: float = 0.1,
        count_thresh: int = 10,
    ):
        self.last_control_key = last_control_key
        self.out_key = out_key
        self.window_size = window
        self.threshold = threshold
        self.count_thresh = count_thresh
        self._steer_window = None

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import collections
        if self._steer_window is None:
            self._steer_window = collections.deque(maxlen=self.window_size)

        ctrl = context.get(self.last_control_key)
        if ctrl is not None:
            if isinstance(ctrl, dict):
                steer = abs(float(ctrl.get("steer", 0.0)))
            else:
                steer = abs(float(getattr(ctrl, "steer", 0.0)))
        else:
            steer = 0.0
        self._steer_window.append(steer)

        turning = int(sum(1 for s in self._steer_window if s > self.threshold) > self.count_thresh)
        context[self.out_key] = turning
        return context


class TCPBlendControl:
    """Blend Beta-control and trajectory-PID outputs based on turning status.

    straight (status=0): 0.3*ctrl + 0.7*traj
    turning  (status=1): 0.7*ctrl + 0.3*traj
    Post-blend: brake > 0.5 → throttle = 0.
    """

    def __init__(
        self,
        ctrl_key: str = "ctrl_ctrl",
        traj_key: str = "ctrl_traj",
        status_key: str = "turning_status",
        out_key: str = "control",
    ):
        self.ctrl_key = ctrl_key
        self.traj_key = traj_key
        self.status_key = status_key
        self.out_key = out_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctrl = context[self.ctrl_key]
        traj = context[self.traj_key]
        status = int(context[self.status_key])

        if status == 1:
            w_ctrl, w_traj = 0.7, 0.3
        else:
            w_ctrl, w_traj = 0.3, 0.7

        steer = float(np.clip(w_ctrl * ctrl["steer"] + w_traj * traj["steer"], -1.0, 1.0))
        throttle = float(np.clip(w_ctrl * ctrl["throttle"] + w_traj * traj["throttle"], 0.0, 0.75))
        brake = float(np.clip(w_ctrl * ctrl["brake"] + w_traj * traj["brake"], 0.0, 1.0))

        if brake > 0.5:
            throttle = 0.0

        context[self.out_key] = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
        }
        return context


# ─────────────────────────────────────────────────────────────────────────────
# LAV modules
#
# Generic modules (reusable with other models):
#   LidarVehicleBodyFilter, HorizontalCameraConcat, MultiCameraToTorchBatch,
#   EKFEgoLocalizer, TemporalLidarAccumulator, PointPaintingModule,
#   BEVHeatmapNMS, WaypointTrackingPID, EmergencyBrakeOverride
#
# LAV-specific wrappers (architecture-tied; noted with CAVEAT in docstrings):
#   LAVRGBSegmentationRunner, LAVBrakePredictionRunner,
#   LAVLiDARModelRunner, LAVUniPlannerRunner, LAVCollisionCheck
# ─────────────────────────────────────────────────────────────────────────────


class LidarVehicleBodyFilter:
    """Remove LiDAR points inside a configurable ego-vehicle bounding box.

    Filters points satisfying ALL of: x in (min_x, max_x), y in (min_y, max_y),
    z in (min_z, max_z). The defaults match the LAV ego-vehicle body footprint.
    """

    def __init__(
        self,
        lidar_key: str = "lidar_raw",
        out_key: str = "lidar_filtered",
        min_x: float = -2.4,
        max_x: float = 0.0,
        min_y: float = -0.8,
        max_y: float = 0.8,
        min_z: float = -1.5,
        max_z: float = -1.0,
    ):
        self.lidar_key = lidar_key
        self.out_key = out_key
        self.bounds = (min_x, max_x, min_y, max_y, min_z, max_z)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        lidar = np.asarray(context[self.lidar_key])
        min_x, max_x, min_y, max_y, min_z, max_z = self.bounds
        mask = (
            (lidar[:, 0] > min_x) & (lidar[:, 0] < max_x) &
            (lidar[:, 1] > min_y) & (lidar[:, 1] < max_y) &
            (lidar[:, 2] > min_z) & (lidar[:, 2] < max_z)
        )
        context[self.out_key] = np.delete(lidar, np.argwhere(mask), axis=0)
        return context


class HorizontalCameraConcat:
    """Concatenate multiple camera images horizontally (axis=1) into a single array."""

    def __init__(self, in_keys: list, out_key: str = "rgb_wide_raw"):
        self.in_keys = in_keys
        self.out_key = out_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        imgs = [np.asarray(context[k]) for k in self.in_keys]
        context[self.out_key] = np.concatenate(imgs, axis=1)
        return context


class MultiCameraToTorchBatch:
    """Stack N camera images (H, W, C) into a batched float tensor (N, C, H, W).

    No normalization — suitable for models that expect raw uint8-scaled floats
    (e.g., LAV's segmentation model). Set divide_by_255=True if needed.
    """

    def __init__(
        self,
        in_keys: list,
        out_key: str = "rgb_batch_t",
        device: str = "cuda",
        divide_by_255: bool = False,
    ):
        self.in_keys = in_keys
        self.out_key = out_key
        self.device = device
        self.divide_by_255 = divide_by_255

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        imgs = [np.asarray(context[k]) for k in self.in_keys]
        arr = np.stack(imgs, axis=0).astype(np.float32)  # (N, H, W, C)
        if self.divide_by_255:
            arr /= 255.0
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        context[self.out_key] = t
        return context


class EKFEgoLocalizer:
    """Kinematic bicycle model EKF fusing GPS, compass, and speed.

    Outputs a smoothed ego-position and heading for use in temporal LiDAR
    accumulation. Reads the previous frame's steer from context[last_control_key].

    On the first call the EKF is initialised from the GPS fix; the first output
    position is the raw GPS position (no dynamics applied yet). The EKF state is
    updated at the end of each call so the NEXT call reflects the current motion.

    Note: wraps lav/ekf.py EKF implementation. The interface — GPS + compass +
    speed → filtered pose — is generic for any kinematic bicycle model EKF.
    """

    def __init__(
        self,
        gps_key: str = "gps_raw",
        compass_key: str = "compass",
        speed_key: str = "speed",
        last_control_key: str = "control",
        out_pos_key: str = "ekf_pos",
        out_compass_key: str = "ekf_compass",
        cos0: float = 1.0,
        lf: float = 1.477531,
        lr: float = 1.393600,
        gnss_noise: float = 0.000005,
        compass_noise: float = 1e-7,
        max_steer_angle: float = 70.0,
        freq: float = 20.0,
    ):
        self.gps_key = gps_key
        self.compass_key = compass_key
        self.speed_key = speed_key
        self.last_control_key = last_control_key
        self.out_pos_key = out_pos_key
        self.out_compass_key = out_compass_key
        self._ekf_kwargs = dict(
            cos0=cos0, lf=lf, lr=lr,
            gnss_noise=gnss_noise, compass_noise=compass_noise,
            max_steer_angle=max_steer_angle, freq=freq,
        )
        self._ekf = None
        self._initialized = False

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import math
        from team_code.lav.ekf import EKF

        if self._ekf is None:
            self._ekf = EKF(**self._ekf_kwargs)

        gps = np.asarray(context[self.gps_key])
        compass_raw = float(context[self.compass_key])
        if np.isnan(compass_raw):
            compass_raw = 0.0
        # LAV offsets compass by -π/2 to align with EKF convention
        compass = compass_raw - math.pi / 2.0
        speed = float(context[self.speed_key])

        if not self._initialized:
            self._ekf.init(float(gps[0]), float(gps[1]), compass)
            self._initialized = True

        # Output current (previous-step-updated) state
        pos = self._ekf.x[:2].copy()
        ori = float(self._ekf.x[2])
        context[self.out_pos_key] = pos
        context[self.out_compass_key] = ori

        # Get previous steer for motion model update
        ctrl = context.get(self.last_control_key)
        if ctrl is not None:
            steer = float(ctrl.get("steer", 0.0) if isinstance(ctrl, dict) else getattr(ctrl, "steer", 0.0))
        else:
            steer = 0.0

        # Update EKF with current measurements (available for NEXT frame)
        self._ekf.step(speed, steer, float(gps[0]), float(gps[1]), compass)
        return context


class TemporalLidarAccumulator:
    """Accumulate LiDAR frames over time with ego-motion compensation.

    Maintains a rolling FIFO of (lidar_array, ekf_pos, ekf_compass) tuples.
    Every `gap` frames, a historical frame is sampled. Each historical frame's
    points are transformed to the current ego-frame via rotation+translation.
    A one-hot time channel (length = num_frame_stack + 1) is appended to each
    frame's feature columns, then all frames are concatenated into a single array.

    When concat_with_prev=True, the current and previous raw lidar arrays are
    concatenated before being stored, doubling density per step (LAV behaviour).
    """

    def __init__(
        self,
        lidar_key: str = "lidar_fused",
        pos_key: str = "ekf_pos",
        compass_key: str = "ekf_compass",
        out_key: str = "lidar_stacked",
        num_frame_stack: int = 2,
        gap: int = 5,
        concat_with_prev: bool = True,
    ):
        self.lidar_key = lidar_key
        self.pos_key = pos_key
        self.compass_key = compass_key
        self.out_key = out_key
        self.num_frame_stack = num_frame_stack
        self.gap = gap
        self.concat_with_prev = concat_with_prev

        num_frame_keep = (num_frame_stack + 1) * gap
        self._fifo_lidar: "collections.deque" = None
        self._fifo_pos: "collections.deque" = None
        self._fifo_compass: "collections.deque" = None
        self._num_frame_keep = num_frame_keep
        self._prev_lidar = None

    def _ensure_fifos(self):
        import collections
        if self._fifo_lidar is None:
            self._fifo_lidar = collections.deque(maxlen=self._num_frame_keep)
            self._fifo_pos = collections.deque(maxlen=self._num_frame_keep)
            self._fifo_compass = collections.deque(maxlen=self._num_frame_keep)

    @staticmethod
    def _move_points(lidar_xyz, dloc, ori0, ori1):
        # Mirrors lav/ekf.py move_lidar_points exactly.
        # dloc = hist_loc - cur_loc; ori0 = current orientation; ori1 = historical orientation.
        dloc = dloc @ np.array([
            [np.cos(ori0), -np.sin(ori0)],
            [np.sin(ori0),  np.cos(ori0)],
        ])
        ori = ori1 - ori0
        lidar_xyz = lidar_xyz @ np.array([
            [np.cos(ori),  np.sin(ori), 0],
            [-np.sin(ori), np.cos(ori), 0],
            [0,            0,           1],
        ])
        lidar_xyz = lidar_xyz.copy()
        lidar_xyz[:, :2] += dloc
        return lidar_xyz

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import collections
        self._ensure_fifos()

        lidar = np.asarray(context[self.lidar_key])
        pos = np.asarray(context[self.pos_key])
        compass = float(context[self.compass_key])

        # Double density: concatenate current with previous frame
        if self.concat_with_prev:
            if self._prev_lidar is not None:
                lidar = np.concatenate([lidar, self._prev_lidar], axis=0)
            self._prev_lidar = np.asarray(context[self.lidar_key])

        self._fifo_lidar.append(lidar)
        self._fifo_pos.append(pos.copy())
        self._fifo_compass.append(compass)

        cur_loc = self._fifo_pos[-1]
        cur_ori = self._fifo_compass[-1]

        rel_lidars = []
        n_time = self.num_frame_stack + 1
        for i, t in enumerate(range(len(self._fifo_lidar) - 1, -1, -self.gap)):
            if i >= n_time:
                break
            hist_lidar = self._fifo_lidar[t]
            hist_loc = self._fifo_pos[t]
            hist_ori = self._fifo_compass[t]

            xyz = hist_lidar[:, :3]
            feats = hist_lidar[:, 3:]
            xyz_transformed = self._move_points(xyz, hist_loc - cur_loc, cur_ori, hist_ori)

            time_enc = np.zeros((len(xyz), n_time), dtype=xyz.dtype)
            time_enc[:, i] = 1.0

            rel_lidars.append(np.concatenate([xyz_transformed, feats, time_enc], axis=-1))

        context[self.out_key] = np.concatenate(rel_lidars, axis=0)
        return context


class PointPaintingModule:
    """Project LiDAR points onto semantic segmentation maps and append painted features.

    For each camera, projects each LiDAR point into the image plane and samples
    the semantic label at the projected pixel. Features from all cameras are
    accumulated (last-write wins for overlapping points). The painted features are
    concatenated to the full LiDAR array along the last axis.

    Note: wraps lav/point_painting.py CoordConverter + point_painting.
    The interface (lidar + per-camera segmentation maps → feature-enriched lidar)
    is generic; the coordinate-projection maths are LAV's implementation.
    Camera geometry is fully parametric via cam_yaws, lidar_xyz, cam_xyz, etc.
    """

    def __init__(
        self,
        lidar_key: str = "lidar_filtered",
        seg_key: str = "rgb_seg",
        out_key: str = "lidar_fused",
        cam_yaws: list = (-60, 0, 60),
        lidar_xyz: list = (0, 0, 2.4),
        cam_xyz: list = (1.5, 0, 2.4),
        rgb_h: int = 288,
        rgb_w: int = 256,
        fov: float = 64.0,
    ):
        self.lidar_key = lidar_key
        self.seg_key = seg_key
        self.out_key = out_key
        self.cam_yaws = list(cam_yaws)
        self.lidar_xyz = list(lidar_xyz)
        self.cam_xyz = list(cam_xyz)
        self.rgb_h = rgb_h
        self.rgb_w = rgb_w
        self.fov = fov
        self._converters = None

    def _ensure_converters(self):
        from team_code.lav.point_painting import CoordConverter, point_painting as _pp
        if self._converters is None:
            self._converters = [
                CoordConverter(
                    yaw,
                    lidar_xyz=self.lidar_xyz,
                    cam_xyz=self.cam_xyz,
                    rgb_h=self.rgb_h,
                    rgb_w=self.rgb_w,
                    fov=self.fov,
                )
                for yaw in self.cam_yaws
            ]
        return _pp

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        point_painting = self._ensure_converters()

        lidar = np.asarray(context[self.lidar_key])
        seg = context[self.seg_key]  # (B, C, H, W) numpy or list of (C, H, W)

        if hasattr(seg, "numpy"):
            seg = seg.detach().cpu().numpy()
        # seg expected as array (num_cams, C, H, W)
        sems = [seg[i] for i in range(len(self._converters))]

        painted = point_painting(lidar, sems, self._converters)
        context[self.out_key] = np.concatenate([lidar, painted], axis=-1)
        return context


class BEVHeatmapNMS:
    """Non-maximum suppression on BEV heatmaps → list of detections per class.

    Uses maxpool NMS: a pixel is a peak iff it equals the local max within a
    (kernel_size × kernel_size) neighbourhood and exceeds min_score.

    Returns context[out_key] as a list of lists:
      [[(x, y, w, h, cos, sin), ...],   # class 0 (vehicles)
       [(x, y, w, h, cos, sin), ...]]   # class 1 (pedestrians)

    Mirrors lav_agent.py det_inference() / extract_peak().
    """

    def __init__(
        self,
        heatmaps_key: str = "lav_heatmaps",
        sizemaps_key: str = "lav_sizemaps",
        orimaps_key: str = "lav_orimaps",
        out_key: str = "lav_detections",
        kernel_size: int = 7,
        min_score: float = 0.1,
        max_det: int = 15,
        pixels_per_meter: float = 4.0,
        ego_pixel_x: int = 160,
        ego_pixel_y: int = 280,
        ego_filter_radius: float = 2.0,
    ):
        self.heatmaps_key = heatmaps_key
        self.sizemaps_key = sizemaps_key
        self.orimaps_key = orimaps_key
        self.out_key = out_key
        self.kernel_size = kernel_size
        self.min_score = min_score
        self.max_det = max_det
        self.pixels_per_meter = pixels_per_meter
        self.ego_pixel_x = ego_pixel_x
        self.ego_pixel_y = ego_pixel_y
        self.ego_filter_radius = ego_filter_radius

    @staticmethod
    def _extract_peak(heatmap, kernel_size, min_score, max_det):
        import torch
        from torch.nn import functional as F
        kp = kernel_size // 2
        max_cls = F.max_pool2d(heatmap[None, None], kernel_size=kernel_size, padding=kp, stride=1)[0, 0]
        possible = heatmap - (max_cls > heatmap).float() * 1e5
        n = min(max_det, possible.numel())
        score, loc = torch.topk(possible.view(-1), n)
        W = heatmap.size(1)
        return [
            (float(s), int(l) % W, int(l) // W)
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score
        ]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        heatmaps = torch.sigmoid(context[self.heatmaps_key][0])  # (2, H, W)
        sizemaps = context[self.sizemaps_key][0]                  # (2, H, W)
        orimaps = context[self.orimaps_key][0]                    # (2, H, W)

        dets = []
        for i, c in enumerate(heatmaps):
            cls_dets = []
            for s, x, y in self._extract_peak(c, self.kernel_size, self.min_score, self.max_det):
                w = float(sizemaps[0, y, x])
                h = float(sizemaps[1, y, x])
                # Filter detections at the ego vehicle pixel location (sensor self-returns)
                if np.linalg.norm([x - self.ego_pixel_x, y - self.ego_pixel_y]) <= self.ego_filter_radius:
                    continue
                # Filter undersized pedestrian detections (class 1)
                if i == 1 and (w < 0.1 * self.pixels_per_meter or h < 0.2 * self.pixels_per_meter):
                    continue
                cos = float(orimaps[0, y, x])
                sin = float(orimaps[1, y, x])
                cls_dets.append((x, y, w, h, cos, sin))
            dets.append(cls_dets)

        context[self.out_key] = dets
        return context


class WaypointTrackingPID:
    """Command-conditioned PID controller tracking planned waypoints.

    Computes desired_speed from the mean inter-waypoint displacement, selects a
    command-specific aim point, and drives PID controllers for steer and speed.
    Mirrors LAV's pid_control() method but is fully self-contained.

    aim_point: list of waypoint indices per command (length = num_cmds).
    speed_ratio: per-command speed scaling applied to desired_speed before delta
                 computation (length = num_cmds).
    pixels_per_meter: scale factor — UniPlanner waypoints are in pixel space.
    """

    def __init__(
        self,
        waypoints_key: str = "lav_ego_plan",
        speed_key: str = "speed",
        cmd_key: str = "next_command",
        out_key: str = "control_base",
        pixels_per_meter: float = 4.0,
        aim_point: list = (4, 4, 4, 3, 6, 6),
        speed_ratio: list = (0.8, 0.8, 0.8, 0.6, 0.8, 0.8),
        turn_KP: float = 0.8,
        turn_KI: float = 0.5,
        turn_KD: float = 0.2,
        turn_n: int = 40,
        speed_KP: float = 5.0,
        speed_KI: float = 0.5,
        speed_KD: float = 1.0,
        speed_n: int = 40,
        brake_speed: float = 0.2,
        clip_delta: float = 0.25,
        max_throttle: float = 0.8,
    ):
        self.waypoints_key = waypoints_key
        self.speed_key = speed_key
        self.cmd_key = cmd_key
        self.out_key = out_key
        self.pixels_per_meter = pixels_per_meter
        self.aim_point = list(aim_point)
        self.speed_ratio = list(speed_ratio)
        self.brake_speed = brake_speed
        self.clip_delta = clip_delta
        self.max_throttle = max_throttle
        self._turn_params = (turn_KP, turn_KI, turn_KD, turn_n)
        self._speed_params = (speed_KP, speed_KI, speed_KD, speed_n)
        self._turn_ctrl = None
        self._speed_ctrl = None

    def _ensure_controllers(self):
        import collections
        if self._turn_ctrl is None:
            KP, KI, KD, n = self._turn_params
            self._turn_ctrl = _SimplePID(KP, KI, KD, n)
            KP, KI, KD, n = self._speed_params
            self._speed_ctrl = _SimplePID(KP, KI, KD, n)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_controllers()

        wps = np.asarray(context[self.waypoints_key]).copy()
        speed = float(context[self.speed_key])
        cmd = int(context[self.cmd_key])
        cmd = max(0, min(cmd, len(self.aim_point) - 1))

        if np.isnan(wps).any():
            context[self.out_key] = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
            return context

        # Scale to pixel space and flip y to match LAV convention
        wps *= self.pixels_per_meter
        wps[:, 1] *= -1

        desired_speed = np.linalg.norm(wps[1:] - wps[:-1], axis=1).mean()

        aim_idx = min(self.aim_point[cmd], len(wps) - 1)
        aim = wps[aim_idx]
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90.0
        steer = float(np.clip(self._turn_ctrl.step(angle), -1.0, 1.0))

        brake = bool(desired_speed < self.brake_speed * self.pixels_per_meter)
        delta = float(np.clip(desired_speed * self.speed_ratio[cmd] - speed, 0.0, self.clip_delta))
        throttle = float(np.clip(self._speed_ctrl.step(delta), 0.0, self.max_throttle))
        if brake:
            throttle = 0.0

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": float(brake)}
        return context


class _SimplePID:
    """Internal stateful PID used by WaypointTrackingPID."""
    def __init__(self, KP, KI, KD, n):
        import collections
        self.KP, self.KI, self.KD = KP, KI, KD
        self._w = collections.deque([0.0] * n, maxlen=n)

    def step(self, error):
        self._w.append(error)
        integral = float(np.mean(self._w))
        derivative = (self._w[-1] - self._w[-2]) if len(self._w) >= 2 else 0.0
        return self.KP * error + self.KI * integral + self.KD * derivative


class EmergencyBrakeOverride:
    """Apply safety overrides on top of a planned control output.

    Handles: brake-prediction threshold, collision flag, max speed, and
    stuck-vehicle recovery (force-throttle after stop_counter exceeds a limit).
    """

    def __init__(
        self,
        control_key: str = "control_base",
        brake_pred_key: str = "brake_pred",
        collision_key: str = "lav_collision",
        speed_key: str = "speed",
        out_key: str = "control",
        brake_threshold: float = 0.1,
        max_speed_kmh: float = 35.0,
        stop_limit: int = 600,
        force_throttle: float = 0.4,
        force_frames: int = 20,
    ):
        self.control_key = control_key
        self.brake_pred_key = brake_pred_key
        self.collision_key = collision_key
        self.speed_key = speed_key
        self.out_key = out_key
        self.brake_threshold = brake_threshold
        self.max_speed_kmh = max_speed_kmh
        self.stop_limit = stop_limit
        self.force_throttle = force_throttle
        self.force_frames = force_frames
        self._stop_counter = 0
        self._force_move = 0

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ctrl = dict(context[self.control_key])
        speed = float(context[self.speed_key])
        steer = ctrl["steer"]
        throt = ctrl["throttle"]
        brake = ctrl["brake"]

        brake_pred = context.get(self.brake_pred_key)
        if brake_pred is not None and float(brake_pred) > self.brake_threshold:
            throt, brake = 0.0, 1.0

        collision = context.get(self.collision_key)
        if collision:
            throt, brake = 0.0, 1.0

        if speed * 3.6 > self.max_speed_kmh:
            throt = 0.0

        if speed < 0.1:
            self._stop_counter += 1
        else:
            self._stop_counter = 0

        if self._stop_counter >= self.stop_limit:
            self._force_move = self.force_frames

        if self._force_move > 0:
            throt = max(self.force_throttle, throt)
            brake = 0.0
            self._force_move -= 1

        context[self.out_key] = {"steer": steer, "throttle": throt, "brake": brake}
        return context


# ── LAV-specific model wrappers ───────────────────────────────────────────────

class LAVRGBSegmentationRunner:
    """Run LAV's ERFNet segmentation model on a batch of camera images.

    CAVEAT: wraps team_code.lav.models.rgb.RGBSegmentationModel — architecture
    is LAV-specific (ERFNet with fixed semantic channel indices [4,6,7,10]).

    Expects input tensor (B, 3, H, W) float from MultiCameraToTorchBatch.
    Outputs foreground semantics: softmax → keep channels [1:] weighted by
    (1 - background), resulting in (B, C-1, H, W) numpy array.
    """

    def __init__(
        self,
        checkpoint_path: str,
        in_key: str = "rgb_batch_t",
        out_key: str = "rgb_seg",
        seg_channels: list = (4, 6, 7, 10),
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.in_key = in_key
        self.out_key = out_key
        self.seg_channels = list(seg_channels)
        self.device = device
        self._model = None

    def _load(self):
        import torch
        from team_code.lav.models.rgb import RGBSegmentationModel
        m = RGBSegmentationModel(self.seg_channels).to(self.device)
        m.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        m.eval()
        self._model = m

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        if self._model is None:
            self._load()
        x = context[self.in_key]
        with torch.no_grad():
            logits = self._model(x)
            pred = torch.softmax(logits, dim=1)
        pred_np = pred.detach().cpu().numpy()
        # Foreground-weighted semantics (matches lav_agent.py line 270)
        context[self.out_key] = pred_np[:, 1:] * (1.0 - pred_np[:, :1])
        return context


class LAVBrakePredictionRunner:
    """Run LAV's cross-attention brake predictor.

    CAVEAT: wraps team_code.lav.models.rgb.RGBBrakePredictionModel — architecture
    is LAV-specific (dual ResNet18 + 8-head cross-attention).

    Concatenates in_keys images horizontally for the wide-view input; applies
    crop_tel_bottom row crop to the telephoto image. No per-pixel normalization
    is applied (matches lav_agent.py raw float conversion).
    """

    def __init__(
        self,
        checkpoint_path: str,
        wide_key: str = "rgb_wide_raw",
        tel_key: str = "rgb_tel_raw",
        out_key: str = "brake_pred",
        crop_tel_bottom: int = 96,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.wide_key = wide_key
        self.tel_key = tel_key
        self.out_key = out_key
        self.crop_tel_bottom = crop_tel_bottom
        self.device = device
        self._model = None

    def _load(self):
        import torch
        from team_code.lav.models.rgb import RGBBrakePredictionModel
        m = RGBBrakePredictionModel([4, 10, 18]).to(self.device)
        m.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        m.eval()
        self._model = m

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        if self._model is None:
            self._load()

        wide = np.asarray(context[self.wide_key], dtype=np.float32)
        tel = np.asarray(context[self.tel_key], dtype=np.float32)
        if self.crop_tel_bottom > 0:
            tel = tel[:-self.crop_tel_bottom]

        wide_t = torch.from_numpy(wide[None]).permute(0, 3, 1, 2).to(self.device)
        tel_t = torch.from_numpy(tel[None]).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            pred = self._model(wide_t, tel_t)
        context[self.out_key] = float(pred)
        return context


class LAVLiDARModelRunner:
    """Run LAV's PointPillar-based LiDAR object detection model.

    CAVEAT: wraps team_code.lav.models.lidar.LiDARModel — architecture is
    LAV-specific (PointPillarNet + multi-scale CNN + detection/segmentation heads).

    Input: (N, num_input) float tensor of stacked + painted LiDAR points.
    Outputs: feature map and raw (pre-sigmoid) heatmaps, sizemaps, orimaps.
    """

    def __init__(
        self,
        checkpoint_path: str,
        lidar_key: str = "lidar_stacked",
        out_features_key: str = "lav_features",
        out_heatmaps_key: str = "lav_heatmaps",
        out_sizemaps_key: str = "lav_sizemaps",
        out_orimaps_key: str = "lav_orimaps",
        num_input: int = 16,  # raw(11) + decorate(5); checkpoint lidar_v2_7.th expects 16
        backbone: str = "cnn",
        num_features: list = (64, 64),
        min_x: float = -10.0,
        max_x: float = 70.0,
        min_y: float = -40.0,
        max_y: float = 40.0,
        pixels_per_meter: float = 4.0,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.lidar_key = lidar_key
        self.out_features_key = out_features_key
        self.out_heatmaps_key = out_heatmaps_key
        self.out_sizemaps_key = out_sizemaps_key
        self.out_orimaps_key = out_orimaps_key
        self._model_kwargs = dict(
            num_input=num_input, backbone=backbone,
            num_features=list(num_features),
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            pixels_per_meter=pixels_per_meter,
        )
        self.device = device
        self._model = None

    def _load(self):
        import torch
        from team_code.lav.models.lidar import LiDARModel
        m = LiDARModel(**self._model_kwargs).to(self.device)
        m.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        m.eval()
        self._model = m

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        if self._model is None:
            self._load()
        pts = np.asarray(context[self.lidar_key], dtype=np.float32)
        pts_t = torch.from_numpy(pts).to(self.device)
        with torch.no_grad():
            features, heatmaps, sizemaps, orimaps, _ = self._model([pts_t], [len(pts_t)])
        context[self.out_features_key] = features
        context[self.out_heatmaps_key] = heatmaps
        context[self.out_sizemaps_key] = sizemaps
        context[self.out_orimaps_key] = orimaps
        return context


class LAVUniPlannerRunner:
    """Run LAV's GRU-based multi-command motion planner.

    CAVEAT: wraps team_code.lav.models.uniplanner.UniPlanner + BEVPlanner —
    architecture is LAV-specific (iterative GRU refinement, command-conditioned
    trajectories, multi-vehicle trajectory forecasting).

    Outputs ego trajectory and cast/other trajectories for collision checking.
    When cmd is a lane-change command (4 or 5), lav_ego_plan is replaced by
    lav_cast_locs to match LAV's lane-change override logic.
    """

    def __init__(
        self,
        checkpoint_path: str,
        bev_checkpoint_path: str,
        features_key: str = "lav_features",
        detections_key: str = "lav_detections",
        cmd_key: str = "next_command",
        target_point_key: str = "target_point",
        out_ego_plan_key: str = "lav_ego_plan",
        out_cast_locs_key: str = "lav_cast_locs",
        out_other_cast_key: str = "lav_other_cast",
        out_other_cmds_key: str = "lav_other_cmds",
        pixels_per_meter: float = 4.0,
        crop_size: int = 96,
        num_cmds: int = 6,
        num_plan: int = 20,
        num_plan_iter: int = 5,
        num_frame_stack: int = 2,
        num_features: list = (64, 64),
        feature_x_jitter: float = 1.5,
        feature_angle_jitter: float = 20.0,
        min_x: float = -10.0,
        device: str = "cuda",
    ):
        self.checkpoint_path = checkpoint_path
        self.bev_checkpoint_path = bev_checkpoint_path
        self.features_key = features_key
        self.detections_key = detections_key
        self.cmd_key = cmd_key
        self.target_point_key = target_point_key
        self.out_ego_plan_key = out_ego_plan_key
        self.out_cast_locs_key = out_cast_locs_key
        self.out_other_cast_key = out_other_cast_key
        self.out_other_cmds_key = out_other_cmds_key
        self.num_cmds = num_cmds
        self.min_x = min_x
        self._model_kwargs = dict(
            pixels_per_meter=pixels_per_meter,
            crop_size=crop_size,
            feature_x_jitter=feature_x_jitter,
            feature_angle_jitter=feature_angle_jitter,
            x_offset=0,
            y_offset=1 + min_x / ((70.0 - min_x) / 2),
            num_cmds=num_cmds,
            num_plan=num_plan,
            num_input_feature=num_features[-1] * 6,
            num_plan_iter=num_plan_iter,
        )
        self._bev_kwargs = dict(
            pixels_per_meter=pixels_per_meter,
            crop_size=crop_size,
            feature_x_jitter=feature_x_jitter,
            feature_angle_jitter=feature_angle_jitter,
            x_offset=0,
            y_offset=1 + min_x / ((70.0 - min_x) / 2),
            num_cmds=num_cmds,
            num_plan=num_plan,
            num_plan_iter=num_plan_iter,
            num_frame_stack=num_frame_stack,
        )
        self.device = device
        self._model = None
        # Lane-change state machine (mirrors lav_agent.py lines 157-160, 301-312)
        self._lane_change_counter = 0
        self._lane_changed = None  # set to cmd value once counter exceeds threshold

    def _load(self):
        import torch
        from team_code.lav.models.uniplanner import UniPlanner
        from team_code.lav.models.bev_planner import BEVPlanner
        bev = BEVPlanner(**self._bev_kwargs).to(self.device)
        bev.load_state_dict(torch.load(self.bev_checkpoint_path, map_location=self.device))
        bev.eval()
        m = UniPlanner(bev, **self._model_kwargs).to(self.device)
        m.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        m.eval()
        self._model = m

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        if self._model is None:
            self._load()

        features = context[self.features_key]
        dets = context[self.detections_key]
        cmd = int(context[self.cmd_key])
        cmd = max(0, min(cmd, self.num_cmds - 1))

        # Lane-change state machine: mirrors lav_agent.py lines 301-312.
        # Hold lane-change commands for 300 frames before reverting to LANEFOLLOW (3).
        if cmd in [4, 5]:
            if self._lane_changed is not None and cmd != self._lane_changed:
                self._lane_change_counter = 0
            self._lane_change_counter += 1
            self._lane_changed = cmd if self._lane_change_counter > 300 else None
        else:
            self._lane_change_counter = 0
            self._lane_changed = None
        if cmd == self._lane_changed:
            cmd = 3  # revert to LANEFOLLOW once lane change is completed

        tp = np.asarray(context[self.target_point_key], dtype=np.float32).flatten()[:2]
        nxps = torch.from_numpy(np.array([-tp[0], -tp[1]], dtype=np.float32)).to(self.device)

        with torch.no_grad():
            ego_plan, ego_cast, other_cast, other_cmds = self._model.infer(
                features[0], dets[1], cmd, nxps
            )

        ego_plan_np = ego_plan.detach().cpu().numpy()
        ego_cast_np = ego_cast.detach().cpu().numpy()
        other_cast_np = other_cast.detach().cpu().numpy()
        other_cmds_np = other_cmds.detach().cpu().numpy()

        # Lane-change override: use cast trajectory for the active command
        if cmd in [4, 5]:
            ego_plan_np = ego_cast_np

        context[self.out_ego_plan_key] = ego_plan_np
        context[self.out_cast_locs_key] = ego_cast_np
        context[self.out_other_cast_key] = other_cast_np
        context[self.out_other_cmds_key] = other_cmds_np
        return context


class LAVCollisionCheck:
    """Check if any high-confidence other-vehicle trajectory intersects the ego plan.

    CAVEAT: input format (other_cast_locs, other_cast_cmds) is specific to
    LAVUniPlannerRunner's output — trajectories are in pixel space, cmd scores
    are confidences in [0, 1].

    Mirrors lav_agent.py plan_collide() with configurable thresholds.
    """

    def __init__(
        self,
        ego_plan_key: str = "lav_ego_plan",
        other_cast_key: str = "lav_other_cast",
        other_cmds_key: str = "lav_other_cmds",
        out_key: str = "lav_collision",
        pixels_per_meter: float = 4.0,
        cmd_thresh: float = 0.2,
        brake_speed: float = 0.2,
        dist_threshold_static: float = 1.0,
        dist_threshold_moving: float = 2.5,
    ):
        self.ego_plan_key = ego_plan_key
        self.other_cast_key = other_cast_key
        self.other_cmds_key = other_cmds_key
        self.out_key = out_key
        self.pixels_per_meter = pixels_per_meter
        self.cmd_thresh = cmd_thresh
        self.brake_speed = brake_speed
        self.dist_static = dist_threshold_static
        self.dist_moving = dist_threshold_moving

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ego = np.asarray(context[self.ego_plan_key])
        other_cast = np.asarray(context.get(self.other_cast_key, []))
        other_cmds = np.asarray(context.get(self.other_cmds_key, []))

        collision = False
        for trajs, cmds in zip(other_cast, other_cmds):
            init_x, init_y = trajs[0, 0], trajs[0, 1]  # noqa: F841
            if init_y > 0.5 * self.pixels_per_meter:
                continue
            for traj, cmd_score in zip(trajs, cmds):
                if cmd_score < self.cmd_thresh:
                    continue
                spd = np.linalg.norm(traj[1:] - traj[:-1], axis=-1).mean()
                threshold = self.dist_static if spd < self.brake_speed else self.dist_moving
                dist = np.linalg.norm(traj - ego, axis=-1).min()
                if dist < threshold:
                    collision = True
                    break
            if collision:
                break

        context[self.out_key] = collision
        return context
