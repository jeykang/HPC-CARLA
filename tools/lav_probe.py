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
print("\nModels loaded without crashing -> the first-route crash is at inference "
      "or server-side, not model load. Tell me and I'll add a one-tick --infer stage.",
      flush=True)
