#!/usr/bin/env python3
"""In-container probe to localize LAV's first-route crash — no pytest, no CARLA server.

Builds the LAV pipeline from lav.yaml, which instantiates every module INCLUDING
the model runners that load the ERFNet seg, PointPillar LiDAR, UniPlanner, and
brake checkpoints onto the GPU. The real run dies during the first route with an
empty results.json (rc=245 / SIGSEGV, no Python traceback), which is consistent
with a crash in this load/init path. `faulthandler` prints the Python frame even
on a hard fault.

Run inside the SIF on a GPU node (e.g. pod17 per the node-sharing rule):

  cd /scratch/autodr_test/HPC-CARLA-persistent
  srun -w hpc-pr-a-pod17 --gres=gpu:1 singularity exec --nv \
      -B "$PWD":/workspace carla_official.sif \
      python3 /workspace/tools/lav_probe.py

Interpreting the result:
  * SIGSEGV / fault here  -> the bug is in LAV model load/init (no CARLA involved);
    faulthandler shows the exact frame (prime suspects: PointPillar init, the pure
    -PyTorch _scatter_* in lav/models/point_pillar.py, or a checkpoint mismatch).
  * "BUILD OK"            -> models load fine; the crash is at inference or
    server-side. Re-run me with --infer for a one-tick forward (added next).
"""
import faulthandler
import os
import sys

faulthandler.enable()

# Resolve repo root from this file (tools/ -> repo). Add both leaderboard/ (so
# `team_code.*` resolves) and leaderboard/team_code/ (so the LAV package's own
# absolute `from lav.*` imports resolve, as the leaderboard agent-loader does).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "leaderboard"))
sys.path.insert(0, os.path.join(_REPO, "leaderboard", "team_code"))

import yaml  # noqa: E402
from team_code.pipeline_engine import PipelineEngine  # noqa: E402

cfg_path = os.path.join(_REPO, "leaderboard", "team_code", "configs", "lav.yaml")
cfg = yaml.safe_load(open(cfg_path))

print(f"python: {sys.version.split()[0]}", flush=True)
try:
    import torch
    print(f"torch: {torch.__version__}  cuda_available={torch.cuda.is_available()}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"torch import failed: {e}", flush=True)

eng = PipelineEngine(cfg["pipeline"])
print(f"building LAV pipeline: {len(cfg['pipeline'])} steps "
      "(loads ERFNet + PointPillar + UniPlanner + brake) ...", flush=True)
eng.build()   # if this faults, faulthandler prints the frame; rc != 0
print(f"BUILD OK: {len(eng._modules)} modules instantiated", flush=True)
for i, m in enumerate(eng._modules):
    print(f"  [{i:2}] {type(m).__name__}", flush=True)

if "--infer" not in sys.argv:
    print("\nBUILD-only. Re-run with --infer to exercise the LiDAR/PointPillar "
          "forward (the inference-crash suspect).", flush=True)
    sys.exit(0)

# ---- inference probe: exercise the prime suspects with cuda.synchronize() ----
# cuda.synchronize() forces async CUDA faults to surface at the offending call
# rather than later, so faulthandler/the traceback points at the real line.
import numpy as np  # noqa: E402
import torch  # noqa: E402

print("\n=== isolating the scatter ops (torch", torch.__version__,
      "vs scatter_reduce_ which needs >=1.12) ===", flush=True)
from team_code.lav.models.point_pillar import _scatter_max, _scatter_mean  # noqa: E402
src = torch.randn(2000, 32, device="cuda")
idx = torch.randint(0, 256, (2000,), device="cuda")
print("  _scatter_mean ...", flush=True)
_ = _scatter_mean(src, idx); torch.cuda.synchronize(); print("    OK", flush=True)
print("  _scatter_max  ...", flush=True)
_ = _scatter_max(src, idx); torch.cuda.synchronize(); print("    OK", flush=True)

print("\n=== LiDAR model forward on a synthetic in-range point cloud ===", flush=True)
lidar_runner = next(m for m in eng._modules if type(m).__name__ == "LAVLiDARModelRunner")
N = 24000
pts = np.empty((N, 16 - 5), dtype=np.float32)   # raw=11; decorate() appends 5 -> 16
pts[:, 0] = np.random.uniform(-10, 70, N)        # x in [min_x, max_x]
pts[:, 1] = np.random.uniform(-40, 40, N)        # y in [min_y, max_y]
pts[:, 2] = np.random.uniform(-3, 3, N)          # z
pts[:, 3] = np.random.uniform(0, 1, N)           # intensity
pts[:, 4:] = np.random.uniform(0, 1, (N, pts.shape[1] - 4))  # painted/temporal feats
ctx = {"lidar_stacked": pts}
print(f"  feeding {pts.shape} points to LAVLiDARModelRunner ...", flush=True)
lidar_runner.run(ctx); torch.cuda.synchronize()
print("  LiDAR forward OK on well-formed input.", flush=True)

# ---- real chain: filter -> camera batch -> ERFNet seg -> point-painting ->
# temporal stacking (FIRST-frame: empty history) -> PointPillar -> NMS.
# This exercises the actual data flow (and first-frame edge cases) the isolated
# test skipped, without needing a CARLA server or the route planner.
print("\n=== real LiDAR chain on synthetic raw sensors (steps 11..17) ===", flush=True)
names = [type(m).__name__ for m in eng._modules]
def _idx(n):
    return names.index(n)
ID_RGB = (256, 288)   # (width, height) for RGB_0/1/2  -> array (H, W, 3)
chain = ["LidarVehicleBodyFilter", "MultiCameraToTorchBatch", "LAVRGBSegmentationRunner",
         "PointPaintingModule", "TemporalLidarAccumulator", "LAVLiDARModelRunner",
         "BEVHeatmapNMS"]
N2 = 30000
lidar_raw = np.empty((N2, 4), dtype=np.float32)
lidar_raw[:, 0] = np.random.uniform(-10, 70, N2)
lidar_raw[:, 1] = np.random.uniform(-40, 40, N2)
lidar_raw[:, 2] = np.random.uniform(-3, 3, N2)
lidar_raw[:, 3] = np.random.uniform(0, 1, N2)
def _rgb():
    return (np.random.rand(ID_RGB[1], ID_RGB[0], 3) * 255).astype(np.float32)
ctx2 = {
    "lidar_raw": lidar_raw,
    "rgb_0_raw": _rgb(), "rgb_1_raw": _rgb(), "rgb_2_raw": _rgb(),
    "ekf_pos": np.array([0.0, 0.0], dtype=np.float32), "ekf_compass": 0.0,
    "control": None,
}
for cname in chain:
    mod = eng._modules[_idx(cname)]
    print(f"  running {cname} ...", flush=True)
    out = mod.run(ctx2)
    if isinstance(out, dict):
        ctx2 = out
    torch.cuda.synchronize()
    print(f"    OK", flush=True)
print("\nFull LiDAR chain OK on synthetic data. If the real run still crashes, the "
      "trigger is real-sensor-specific (e.g. zero points in view) or in the seg/"
      "brake/uniplanner path or server-side. The eval template now sets "
      "PYTHONFAULTHANDLER=1, so the next real run dumps the crash frame to the worker log.",
      flush=True)
