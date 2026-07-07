"""CILRS network definition (+ a thin adapter for the generic TorchModelRunner).

The three network classes below (ImageCNN, Controller, CILRS) and the
normalize_imagenet() helper are vendored VERBATIM from
autonomousvision/transfuser @ cvpr2021 : cilrs/model.py, with a single,
behaviour-preserving change flagged inline:

  * ImageCNN uses `pretrained=False` for the ResNet-18 backbone (the reference
    hardcodes `pretrained=True`). At evaluation the entire encoder is overwritten
    by `load_state_dict(best_model.pth)`, so this changes nothing about the loaded
    weights — it only avoids a torchvision ImageNet download inside the offline
    container. See FIDELITY note in configs/cilrs.yaml.

`CILRSInterface` (at the bottom) is NOT part of the reference; it is a small
dict-in / dict-out `nn.Module` adapter so the repo's generic
`team_code.pipeline_modules.TorchModelRunner` can drive the network. It subclasses
CILRS (adding zero parameters, so the reference checkpoint loads with strict=True)
and only reproduces the per-camera encoding + tensor-packing that the reference
agent does inline in run_step().

torch/torchvision are imported at module top exactly as the reference model file
does; this module is only imported at runtime (offline validate-config uses AST
and never imports it).
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from team_code.cilrs.config import GlobalConfig


# ─────────────────────────────────────────────────────────────────────────────
# Reference network (cilrs/model.py, cvpr2021) — vendored verbatim.
# ─────────────────────────────────────────────────────────────────────────────


class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=False, pretrained=False):
        super().__init__()
        self.normalize = normalize
        # Reference: models.resnet18(pretrained=True). We default pretrained=False
        # because the eval checkpoint fully overwrites these weights (see module
        # docstring); behaviour after load_state_dict is identical.
        self.features = models.resnet18(pretrained=pretrained)
        self.features.fc = nn.Sequential()

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class Controller(nn.Module):
    """ Decoder with velocity input, velocity prediction and conditional control outputs.
    Args:
        num_branch (int): number of conditional branches
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of each decoder branch
        input_velocity (bool): whether to add input velocity information to encoding
        predict_velocity (bool): whether to output a velocity branch prediction
    """

    def __init__(self, num_branch=6, dim=1, c_dim=512, hidden_size=256,
                                input_velocity=True, predict_velocity=True):
        super().__init__()
        self.num_branch = num_branch
        self.input_velocity = input_velocity
        self.predict_velocity = predict_velocity

        # Project input velocity measurement to feature size
        if input_velocity:
            self.vel_in = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, c_dim),
            )

        # Project feature to velocity prediction
        if predict_velocity:
            self.vel_out = nn.Sequential(
                nn.Linear(c_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, dim),
            )

        # Control branches
        fc_branch_list = []
        for i in range(num_branch):
            fc_branch_list.append(nn.Sequential(
            nn.Linear(c_dim, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid(),
        ))

        self.branches = nn.ModuleList(fc_branch_list)

    def forward(self, c, velocity, command):
        batch_size = c.size(0)
        encoding = c

        if self.input_velocity:
            encoding += self.vel_in(velocity.unsqueeze(1))

        control_pred = 0.
        for i, branch in enumerate(self.branches):
            # Choose control for branch of only active command
            # We check for (command - 1) since navigational command 0 is ignored
            control_pred += branch(encoding) * (i == (command - 1)).unsqueeze(1).expand(batch_size,3)

        if self.predict_velocity:
            velocity_pred = self.vel_out(c)
            return control_pred, velocity_pred

        return control_pred


class CILRS(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = ImageCNN(512, normalize=True).to(self.device)
        self.controller = Controller(num_branch=6, dim=1, c_dim=512, hidden_size=256,
                                input_velocity=True, predict_velocity=True).to(self.device)

    def forward(self, c, velocity, command):
        ''' Predicts vehicle control.
        Args:
            c (tensor): latent conditioned code c
            velocity (tensor): speedometer input
            command (tensor): high-level navigational command
        '''
        c = sum(c)
        control_pred, velocity_pred = self.controller(c, velocity, command)
        steer = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle
        brake = control_pred[:,2]
        return steer, throttle, brake, velocity_pred


# ─────────────────────────────────────────────────────────────────────────────
# Adapter for the generic pipeline runner (NOT in the reference).
# ─────────────────────────────────────────────────────────────────────────────


class CILRSInterface(CILRS):
    """Dict-in / dict-out adapter so `TorchModelRunner` can drive CILRS.

    Mirrors CILRSAgent.run_step() (cvpr2021 leaderboard/team_code/cilrs_agent.py
    lines 177-203) EXACTLY for the shipped config (seq_len=1, ignore_sides=True,
    ignore_rear=True):

      * per-camera latent = self.encoder([img])          (agent line 184)
      * the list of per-camera latents is summed inside CILRS.forward (`c=sum(c)`)
      * velocity / command tensors built inline, shape (1,)  (agent lines 177-178)
      * command branch is selected inside Controller.forward
      * steer/throttle/brake produced by CILRS.forward

    Adds zero parameters over CILRS, so the reference `best_model.pth` state_dict
    loads with strict=True. Only overrides forward() to accept a dict of inputs.
    """

    def __init__(self, max_throttle=0.75, device='cuda', seq_len=1,
                 ignore_sides=True, ignore_rear=True):
        cfg = GlobalConfig(
            max_throttle=max_throttle,
            seq_len=seq_len,
            ignore_sides=ignore_sides,
            ignore_rear=ignore_rear,
        )
        super().__init__(cfg, device)

    def forward(self, inputs):
        """inputs is a dict:
            front   : (1, 3, H, W) float32 tensor in [0, 255]  (ImageNet-norm is
                      applied INSIDE the encoder, matching the reference — the
                      image must NOT be divided by 255 or pre-normalized)
            left/right/rear : same, required only when ignore_sides/ignore_rear=False
            velocity: python float, speed in m/s
            command : python int, RoadOption value (1-based; branch = command-1)
        """
        # Per-camera encodings, mirroring run_step's encoding.append(...) order.
        encoding = [self.encoder([inputs['front']])]
        if not self.config.ignore_sides:
            encoding.append(self.encoder([inputs['left']]))
            encoding.append(self.encoder([inputs['right']]))
        if not self.config.ignore_rear:
            encoding.append(self.encoder([inputs['rear']]))

        # Built inline exactly as the reference agent does (shape (1,), float32).
        velocity = torch.FloatTensor([float(inputs['velocity'])]).to(self.device, dtype=torch.float32)
        command = torch.FloatTensor([float(inputs['command'])]).to(self.device, dtype=torch.float32)

        steer, throttle, brake, velocity_pred = super().forward(encoding, velocity, command)
        return {
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'velocity': velocity_pred,
        }
