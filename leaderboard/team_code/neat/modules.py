"""NEAT (Neural Attention Fields) pipeline glue modules.

These classes reproduce the inference path of the original NEAT leaderboard agent
(`autonomousvision/neat` -> leaderboard/team_code/neat_agent.py, class MultiTaskAgent)
as modular pipeline steps of the form `run(context) -> context`.

The original agent's per-tick flow (neat_agent.py:MultiTaskAgent.run_step) is:

  tick(input_data)                      -> BGR->RGB 3 cameras, gps, speed, compass,
                                           next_command, target_point   (lines 158-195)
  scale_and_crop_image(...) per camera  -> (1,3,256,256) float32 in [0,255]   (line 29-41, 226-236)
  buffer frames (deque, len seq_len)    -> input_buffer                        (57, 204-236)
  encoding = net.encoder(images, v)     -> transformer encoding                (245)
  net.plan(target, encoding, ...)       -> pred_waypoint_mean, red_light_occ   (247)
  net.control_pid(wp[:, seq_len:], ...) -> steer, throttle, brake              (248)
  brake<0.05->0 ; throttle>brake->brake=0                                      (258-259)

Design notes / fidelity:
  * NO /255 and NO ImageNet normalization happens here: the vendored encoder
    (architectures/encoder.py ImageCNN, normalize=True) applies ImageNet
    normalization *internally to the [0,255] tensor* exactly as at training time
    (data.py:scale_and_crop_image returns uint8 [0,255]; encoder.normalize_imagenet
    divides by 0.229 etc. without a prior /255). Mirroring this is critical.
  * The plan grid is created ONCE (net.create_plan_grid) and reused every tick,
    exactly like the original agent (setup line 64, passed each run_step line 247).
    net.plan() mutates plan_grid[:,:,:2] in place across ticks -- this is the
    original behaviour and is deliberately preserved.
  * next_command is computed by RoutePlannerNextCommand (needed to advance the
    route + produce the target point) but, as in the original, is NOT fed to the
    network (the agent computes `command` at run_step line 222 but never passes it
    to plan()/control_pid()).

All heavy imports (torch, torchvision, the vendored architectures) are lazy.
"""

from collections import deque
from typing import Any, Dict, List, Optional


PIPELINE_STOP_KEY = "__pipeline_stop__"


# ---------------------------------------------------------------------------
# Image preprocessing (mirror of neat_agent.py:scale_and_crop_image, lines 29-41)
# ---------------------------------------------------------------------------


def scale_and_crop_image(image, scale=1, crop=256):
    """Scale and center-crop a PIL image, returning a channels-first numpy array.

    Verbatim port of neat_agent.py:scale_and_crop_image (lines 29-41). With the
    NEAT config (scale=1) the resize is a no-op; the crop takes the central
    crop x crop region. Returns a (3, crop, crop) uint8 array.
    """
    import numpy as np

    (width, height) = (image.width // scale, image.height // scale)
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


class NEATImagePreprocess:
    """Scale + center-crop one RGB camera into a NEAT input tensor.

    Reads an HxWx3 uint8 RGB array (as produced by ExtractCameraRGB) and writes a
    (1, 3, crop, crop) float32 tensor in the [0,255] range on `device` -- exactly
    matching the original agent:
        torch.from_numpy(scale_and_crop_image(Image.fromarray(rgb))).unsqueeze(0)
             .to('cuda', dtype=torch.float32)                    (neat_agent.py:226-228)

    IMPORTANT: values are left in [0,255]; ImageNet normalization is applied later
    *inside* the vendored encoder (ImageCNN.normalize_imagenet), never here.
    """

    def __init__(
        self,
        in_key: str,
        out_key: str,
        scale: int = 1,
        crop: int = 256,
        device: str = "cuda",
    ):
        self.in_key = in_key
        self.out_key = out_key
        self.scale = int(scale)
        self.crop = int(crop)
        self.device = device

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        import numpy as np
        from PIL import Image

        rgb = context[self.in_key]
        # rgb is HxWx3 uint8 (RGB). Match the original: PIL round-trip then crop.
        arr = scale_and_crop_image(
            Image.fromarray(np.asarray(rgb).astype(np.uint8)),
            scale=self.scale,
            crop=self.crop,
        )
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device, dtype=torch.float32)
        context[self.out_key] = tensor
        return context


# ---------------------------------------------------------------------------
# NEAT network runner: encoder + iterative attention-field plan
# ---------------------------------------------------------------------------


class NEATModelRunner:
    """Run the NEAT AttentionField network: encoder + iterative `plan` decode.

    Mirrors neat_agent.py:MultiTaskAgent (setup lines 59-68, run_step lines 197-252).

    Builds `AttentionField(GlobalConfig(), device)` once, loads the two separate
    checkpoints into net.encoder / net.decoder, and creates the plan + light
    sampling grids once (reused every tick, matching the original).

    Per tick:
      * push each preprocessed camera tensor into a per-camera deque of length
        `seq_len` (deque(maxlen=seq_len) reproduces the original append / popleft
        buffering, neat_agent.py:204-236);
      * for the first `seq_len` ticks emit a zero hold-control and stop the
        pipeline (original returns a zero VehicleControl while the buffer fills,
        lines 204-217);
      * otherwise assemble the image list interleaved per timestep
        (front, left, right) exactly as lines 238-243, run the encoder, then the
        iterative attention-field `plan()` (lines 245-247).

    Writes (for the controller):
      context[out_waypoints_key] : full pred_waypoint_mean tensor (1, tot_len, 2)
      context[out_redlight_key]  : red_light_occ (scalar tensor or 0)
      context[out_velocity_key]  : gt_velocity tensor (1,)
      context[out_target_key]    : target_point tensor (2, 1)
    """

    def __init__(
        self,
        encoder_checkpoint: str,
        decoder_checkpoint: str,
        image_keys: List[str],
        speed_key: str = "speed",
        target_point_key: str = "target_point",
        out_waypoints_key: str = "neat_pred_waypoints",
        out_redlight_key: str = "neat_red_light",
        out_velocity_key: str = "neat_velocity",
        out_target_key: str = "neat_target",
        device: str = "cuda",
        hold_control: Optional[Dict[str, float]] = None,
        stop_key: str = PIPELINE_STOP_KEY,
    ):
        self.encoder_checkpoint = encoder_checkpoint
        self.decoder_checkpoint = decoder_checkpoint
        self.image_keys = list(image_keys)
        self.speed_key = speed_key
        self.target_point_key = target_point_key
        self.out_waypoints_key = out_waypoints_key
        self.out_redlight_key = out_redlight_key
        self.out_velocity_key = out_velocity_key
        self.out_target_key = out_target_key
        self.device = device
        self.hold_control = hold_control or {"steer": 0.0, "throttle": 0.0, "brake": 0.0}
        self.stop_key = stop_key

        self.net = None
        self.config = None
        self.plan_grid = None
        self.light_grid = None
        self._buffers = None
        self._step = -1

    def _ensure_net(self):
        if self.net is not None:
            return
        import torch
        from team_code.neat.config import GlobalConfig
        from team_code.neat.architectures import AttentionField

        self.config = GlobalConfig()

        if len(self.image_keys) != self.config.num_camera:
            raise ValueError(
                "NEATModelRunner: %d image_keys given but config.num_camera=%d"
                % (len(self.image_keys), self.config.num_camera)
            )

        self.net = AttentionField(self.config, self.device)
        self.net.encoder.load_state_dict(
            torch.load(self.encoder_checkpoint, map_location=self.device)
        )
        self.net.decoder.load_state_dict(
            torch.load(self.decoder_checkpoint, map_location=self.device)
        )

        # Sampling grids: created ONCE and reused every tick (original setup 64-65).
        self.plan_grid = self.net.create_plan_grid(
            self.config.plan_scale, self.config.plan_points, 1
        )
        self.light_grid = self.net.create_light_grid(
            self.config.light_x_steps, self.config.light_y_steps, 1
        )

        self.net.to(self.device)
        self.net.eval()

        # One frame buffer per camera view (deque(maxlen=seq_len)).
        self._buffers = {k: deque(maxlen=self.config.seq_len) for k in self.image_keys}

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        self._ensure_net()
        self._step += 1

        # ---- push current preprocessed frames into per-camera buffers ----
        for k in self.image_keys:
            self._buffers[k].append(context[k])

        # ---- warmup: buffer not yet full -> zero hold-control, stop pipeline ----
        if self._step < self.config.seq_len:
            context["control"] = dict(self.hold_control)
            context[self.stop_key] = True
            return context

        # ---- state tensors (mirror run_step lines 219-224) ----
        speed = float(context[self.speed_key])
        gt_velocity = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

        tp = context[self.target_point_key]  # np.ndarray (2,) in vehicle frame
        target_point = torch.stack(
            [torch.FloatTensor([float(tp[0])]), torch.FloatTensor([float(tp[1])])]
        ).to(self.device, dtype=torch.float32)  # (2, 1)

        # ---- assemble image list: per timestep (front, left, right) ----
        # neat_agent.py:238-243
        images = []
        for i in range(self.config.seq_len):
            for k in self.image_keys:
                images.append(self._buffers[k][i])

        # ---- encoder + iterative attention-field plan ----
        with torch.no_grad():
            encoding = self.net.encoder(images, gt_velocity)
            pred_waypoint_mean, red_light_occ = self.net.plan(
                target_point,
                encoding,
                self.plan_grid,
                self.light_grid,
                self.config.plan_points,
                self.config.plan_iters,
            )

        context[self.out_waypoints_key] = pred_waypoint_mean
        context[self.out_redlight_key] = red_light_occ
        context[self.out_velocity_key] = gt_velocity
        context[self.out_target_key] = target_point
        return context


# ---------------------------------------------------------------------------
# NEAT controller: faithful reimplementation of AttentionField.control_pid
# ---------------------------------------------------------------------------


class NEATControlPID:
    """PID controller on NEAT waypoints -- faithful port of
    AttentionField.control_pid (architectures/__init__.py:140-219) plus the
    agent's post-processing (neat_agent.py:254-259).

    Holds its own persistent turn / speed PIDController instances (vendored
    architectures/controller.py) so integral/derivative history carries across
    ticks, exactly like the net's own controllers in the original.

    Reads:
      waypoints_key : full pred_waypoint_mean (1, tot_len, 2); the first
                      `seq_len` entries are dropped here (original slices
                      pred_waypoint_mean[:, seq_len:] at the call site, line 248)
      velocity_key  : gt_velocity tensor (1,)
      target_key    : target_point tensor (2, 1)
      redlight_key  : red_light_occ (scalar tensor or int)

    Writes context[out_key] = {steer, throttle, brake}.
    """

    def __init__(
        self,
        waypoints_key: str = "neat_pred_waypoints",
        velocity_key: str = "neat_velocity",
        target_key: str = "neat_target",
        redlight_key: str = "neat_red_light",
        out_key: str = "control",
        seq_len: int = 1,
        # controller / affordance params (defaults = neat/config.py GlobalConfig)
        aim_dist: float = 4.0,
        angle_thresh: float = 0.3,
        dist_thresh: float = 10.0,
        red_light_mult: float = 0.0,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta: float = 0.25,
        max_throttle: float = 0.75,
        turn_KP: float = 0.75,
        turn_KI: float = 0.75,
        turn_KD: float = 0.3,
        turn_n: int = 40,
        speed_KP: float = 5.0,
        speed_KI: float = 0.5,
        speed_KD: float = 1.0,
        speed_n: int = 40,
    ):
        self.waypoints_key = waypoints_key
        self.velocity_key = velocity_key
        self.target_key = target_key
        self.redlight_key = redlight_key
        self.out_key = out_key
        self.seq_len = int(seq_len)

        self.aim_dist = float(aim_dist)
        self.angle_thresh = float(angle_thresh)
        self.dist_thresh = float(dist_thresh)
        self.red_light_mult = float(red_light_mult)
        self.brake_speed = float(brake_speed)
        self.brake_ratio = float(brake_ratio)
        self.clip_delta = float(clip_delta)
        self.max_throttle = float(max_throttle)

        self._turn_cfg = (turn_KP, turn_KI, turn_KD, int(turn_n))
        self._speed_cfg = (speed_KP, speed_KI, speed_KD, int(speed_n))
        self.turn_controller = None
        self.speed_controller = None

    def _ensure_controllers(self):
        if self.turn_controller is not None:
            return
        from team_code.neat.architectures.controller import PIDController

        self.turn_controller = PIDController(*self._turn_cfg)
        self.speed_controller = PIDController(*self._speed_cfg)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        import torch

        self._ensure_controllers()

        # Slice off the seq_len "current" positions -> future waypoints only.
        waypoints = context[self.waypoints_key][:, self.seq_len:]
        velocity = context[self.velocity_key]
        target = context[self.target_key]
        red_light = context[self.redlight_key]

        # ---- AttentionField.control_pid (architectures/__init__.py:140-201) ----
        assert waypoints.size(0) == 1
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()
        if torch.is_tensor(red_light):
            red_light = red_light.data.cpu().numpy()

        # flip y (forward is negative in our waypoints)
        waypoints[:, 1] *= -1
        target[1] *= -1

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            desired_speed += np.linalg.norm(
                waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs
            norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        # slow if red light affordance is active
        if red_light:
            desired_speed *= self.red_light_mult

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # outlier rejection: prefer target point when its angle is smaller, or when
        # the last-segment angle disagrees strongly and the target is close.
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (
            np.abs(angle_target - angle_last) > self.angle_thresh
            and target[1] < self.dist_thresh
        )
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        # ---- agent post-processing (neat_agent.py:254-259) ----
        steer = float(steer)
        throttle = float(throttle)
        brake = float(brake)
        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        context[self.out_key] = {"steer": steer, "throttle": throttle, "brake": brake}
        return context
