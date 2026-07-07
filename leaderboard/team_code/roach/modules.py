"""Roach (CILRS) - specific pipeline glue modules.

These implement the parts of the original Roach IL agent that are NOT the network
forward pass itself: the measurement-vector assembly and the Beta-distribution ->
action decoding. The network is run by the generic ``TorchModelRunner`` on the
vendored ``team_code.roach.cilrs_model.RoachILPolicy`` wrapper.

Every class implements ``run(context) -> context`` and wires keys via ``args``, per
``leaderboard/team_code/PIPELINE_MODULES.md``.

Fidelity references (original repo = zhejz/carla-roach):
  * measurement vector / command mapping ... agents/cilrs/cilrs_wrapper.py:37-89
  * action decode (process_act) ............ agents/cilrs/cilrs_wrapper.py:91-107
  * Beta -> action / branch select ......... agents/cilrs/models/cilrs_model.py:212-288
  * route-target advancement + command ..... carla_gym/core/obs_manager/navigation/gnss.py:76-123
  * ego-frame target vector (loc_in_ev) .... carla_gym/utils/transforms.py:21-33,53-81
  * Mercator gps_to_location ............... carla_gym/core/task_actor/common/navigation/route_manipulation.py:32-44

torch is imported lazily inside ``run`` (login node has no torch; container does).
"""
from typing import Any, Dict, Sequence

import math
import numpy as np


# route_manipulation.py:20
_EARTH_RADIUS_EQUA = 6378137.0

# agents.navigation.local_planner.RoadOption values used by the leaderboard global
# plan (VOID=-1, LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, CHANGELANELEFT=5,
# CHANGELANERIGHT=6). Matches the branch-index mapping in cilrs_wrapper.py:47-59.
_CHANGELANELEFT = 5
_CHANGELANERIGHT = 6


def _gps_to_xy(lat: float, lon: float):
    """Mercator projection, verbatim from route_manipulation.gps_to_location (:40-42).

    Returns (x, y) in metres; altitude only affects z there and is irrelevant to the
    ego-frame (x, y) target vector, so it is dropped.
    """
    x = lon / 180.0 * (math.pi * _EARTH_RADIUS_EQUA)
    y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * _EARTH_RADIUS_EQUA
    return x, y


class RoachRouteTarget:
    """Reproduce Roach's GNSS obs-manager route tracking and the CILRS ego-frame
    target vector ("vec") + high-level command (the CILRS branch selector).

    This mirrors ``carla_gym/.../navigation/gnss.py:get_observation`` (idx advancement
    with a 12 m reach threshold, plus the change-lane command rule) followed by
    ``cilrs_wrapper.process_obs``'s ``loc_in_ev`` computation. It deliberately does NOT
    reuse the repo's ``RoutePlannerNextCommand`` because that tracker uses a different
    advancement heuristic AND a different planar coordinate convention
    ((lat*scale, lon*scale)) than Roach's Mercator + (compass-90 deg) rotation, which
    would feed the learned "vec" input in the wrong frame.

    Writes:
      * context[out_vec_key] : np.ndarray (2,) float32 = (loc_in_ev.x, loc_in_ev.y) [m]
      * context[out_cmd_key] : int, 0-based branch index in [0, 5]

    Reads context['agent']._global_plan (list of (gps, RoadOption)), the raw GNSS
    (lat, lon) at gps_key, and the compass (rad) at compass_key.
    """

    def __init__(
        self,
        gps_key: str = "gps_raw",
        compass_key: str = "compass",
        out_vec_key: str = "roach_vec",
        out_cmd_key: str = "roach_command_idx",
        reach_threshold: float = 12.0,
    ):
        self.gps_key = gps_key
        self.compass_key = compass_key
        self.out_vec_key = out_vec_key
        self.out_cmd_key = out_cmd_key
        self.reach_threshold = float(reach_threshold)
        self._idx = -1          # gnss.py:42 (self._idx = -1)
        self._plan = None       # cached [(lat, lon, road_option_value), ...]

    def _ensure_plan(self, agent: Any) -> None:
        if self._plan is not None:
            return
        gp = getattr(agent, "_global_plan", None)
        if not gp:
            raise RuntimeError(
                "RoachRouteTarget needs agent._global_plan (set_global_plan not called yet)"
            )
        plan = []
        for gps, road_option in gp:
            if isinstance(gps, dict):
                lat = float(gps["lat"])
                lon = float(gps["lon"])
            else:
                lat = float(gps[0])
                lon = float(gps[1])
            rv = int(getattr(road_option, "value", road_option))
            plan.append((lat, lon, rv))
        if len(plan) < 2:
            raise RuntimeError("RoachRouteTarget needs a global plan with >= 2 points")
        self._plan = plan

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        agent = context.get("agent")
        if agent is None:
            raise KeyError("context['agent'] is required for RoachRouteTarget")
        self._ensure_plan(agent)
        plan = self._plan
        n = len(plan)

        gps = np.asarray(context[self.gps_key], dtype=np.float64).reshape(-1)
        ev_lat, ev_lon = float(gps[0]), float(gps[1])

        compass = float(context.get(self.compass_key, 0.0))
        if np.isnan(compass):
            compass = 0.0

        # vec_global_to_ref with a yaw-only reference rotation of (compass - 90 deg):
        # R = yaw(psi); loc_in_ev = R^T @ (target_xy - ev_xy).  (transforms.py:21-33,64-68)
        psi = compass - math.pi / 2.0
        cos_p, sin_p = math.cos(psi), math.sin(psi)
        ev_x, ev_y = _gps_to_xy(ev_lat, ev_lon)

        def loc_in_ev(lat, lon):
            tx, ty = _gps_to_xy(lat, lon)
            dx, dy = tx - ev_x, ty - ev_y
            x = cos_p * dx + sin_p * dy
            y = -sin_p * dx + cos_p * dy
            return x, y

        # --- idx advancement (gnss.py:98-107) ---
        nidx = min(self._idx + 1, n - 1)
        ax, ay = loc_in_ev(plan[nidx][0], plan[nidx][1])
        if math.sqrt(ax * ax + ay * ay) < self.reach_threshold and ax < 0.0:
            self._idx += 1
        self._idx = min(self._idx, n - 2)

        # --- command via change-lane rule (gnss.py:109-116) ---
        road_option_0 = plan[max(0, self._idx)][2]
        t_lat, t_lon, road_option_1 = plan[self._idx + 1]
        if road_option_0 in (_CHANGELANELEFT, _CHANGELANERIGHT) and road_option_1 not in (
            _CHANGELANELEFT,
            _CHANGELANERIGHT,
        ):
            command_raw = road_option_1
        else:
            command_raw = road_option_0

        # --- ego-frame target vector for the CURRENT target gps (cilrs_wrapper.py:42-45) ---
        vx, vy = loc_in_ev(t_lat, t_lon)
        context[self.out_vec_key] = np.array([vx, vy], dtype=np.float32)

        # --- command -> 0-based branch index (cilrs_wrapper.py:54-58) ---
        c = int(command_raw)
        if c < 0:
            c = 4
        c = c - 1
        context[self.out_cmd_key] = int(c)
        return context


class CilrsStateVector:
    """Assemble the CILRS measurement vector and emit it as a (1, S) torch tensor.

    Mirrors cilrs_wrapper.process_obs (:61-88): concatenation order is speed, then vec,
    then the 6-way command one-hot, restricted to whichever of [speed, vec, cmd] are in
    ``input_states``. speed is normalized by ``speed_factor`` (12.0).

    For the released L_K LeaderBoard checkpoint (12uzu2lu) input_states = [speed, vec],
    giving state = [forward_speed/12, loc_in_ev.x, loc_in_ev.y] with S = 3 (matches the
    checkpoint's measurements Linear(3 -> 128)).
    """

    def __init__(
        self,
        speed_key: str = "speed",
        vec_key: str = "roach_vec",
        cmd_key: str = "roach_command_idx",
        input_states: Sequence[str] = ("speed", "vec"),
        out_key: str = "roach_state",
        speed_factor: float = 12.0,
        num_cmds: int = 6,
        device: str = "cuda",
    ):
        self.speed_key = speed_key
        self.vec_key = vec_key
        self.cmd_key = cmd_key
        self.input_states = list(input_states)
        self.out_key = out_key
        self.speed_factor = float(speed_factor)
        self.num_cmds = int(num_cmds)
        self.device = device

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        parts = []
        if "speed" in self.input_states:
            parts.append(float(context[self.speed_key]) / self.speed_factor)
        if "vec" in self.input_states:
            vec = np.asarray(context[self.vec_key], dtype=np.float32).reshape(-1)
            parts.append(float(vec[0]))
            parts.append(float(vec[1]))
        if "cmd" in self.input_states:
            one_hot = [0.0] * self.num_cmds
            idx = int(context[self.cmd_key])
            idx = int(np.clip(idx, 0, self.num_cmds - 1))
            one_hot[idx] = 1.0
            parts += one_hot

        state = np.asarray(parts, dtype=np.float32)
        t = torch.from_numpy(state).unsqueeze(0)  # (1, S)
        if self.device:
            t = t.to(self.device)
        context[self.out_key] = t
        return context


class CilrsActionFromBranches:
    """Select the command branch and decode it into (acc, steer) scalars.

    Faithful to CoILICRA.forward_branch (cilrs_model.py:212-224): clamp the command to
    a valid branch, and -- for a Beta action head -- convert the (alpha, beta) Softplus
    branch outputs to a deterministic action via ``_get_action_beta`` (Beta mode scaled
    to [-1, 1]) before selecting the branch with ``extract_branch``. Those two static
    methods are reused directly from the vendored model so the math is identical.

    Then mirrors cilrs_wrapper.process_act (:91-105): action = (acc, steer); steer is
    clipped to [-1, 1] here, and (acc -> throttle/brake) is delegated to the downstream
    ``ControlFromAccSteer`` module.

    Writes context[acc_out_key] (float) and context[steer_out_key] (float).
    """

    def __init__(
        self,
        mu_key: str = "roach_mu",
        sigma_key: str = "roach_sigma",
        action_branches_key: str = "roach_action_branches",
        cmd_key: str = "roach_command_idx",
        action_distribution: str = "beta_shared",
        number_of_branches: int = 6,
        acc_out_key: str = "roach_acc",
        steer_out_key: str = "roach_steer",
    ):
        self.mu_key = mu_key
        self.sigma_key = sigma_key
        self.action_branches_key = action_branches_key
        self.cmd_key = cmd_key
        self.action_distribution = action_distribution
        self.number_of_branches = int(number_of_branches)
        self.acc_out_key = acc_out_key
        self.steer_out_key = steer_out_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from team_code.roach.cilrs_model import CoILICRA

        cmd = int(context[self.cmd_key])
        command_tensor = torch.tensor([cmd], dtype=torch.int8)
        command_tensor.clamp_(0, self.number_of_branches - 1)  # cilrs_model.py:217

        if self.action_distribution in ("beta", "beta_shared"):
            mu = context[self.mu_key]
            sigma = context[self.sigma_key]
            action_branches = CoILICRA._get_action_beta(mu, sigma)  # :220
        else:
            action_branches = context[self.action_branches_key]

        action = CoILICRA.extract_branch(action_branches, command_tensor)  # :221/223
        act = action[0].detach().cpu().numpy()  # (2,) = (acc, steer); cilrs_model.py:224

        acc = float(act[0])
        steer = float(np.clip(act[1], -1.0, 1.0))  # cilrs_wrapper.py:104
        context[self.acc_out_key] = acc
        context[self.steer_out_key] = steer
        return context
