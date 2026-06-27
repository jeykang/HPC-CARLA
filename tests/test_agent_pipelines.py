"""Tier-2 agent-pipeline tests — heavy; run in the container (`pytest -m heavy`).

Builds each agent's real pipeline from its YAML, instantiating every module
including the model runners that load checkpoints. This catches the construction
bugs we hit by hand (e.g. LAV `num_input` state_dict size mismatch, missing
classes, bad args) — the kind that previously only surfaced after a 10-minute
cluster round-trip.

Requires numpy + torch (+ the model weights); auto-skipped where unavailable, so
it is a no-op on the login node. Excluded from the default run by the `heavy`
marker (see pytest.ini); run explicitly inside the container:

    singularity exec --nv carla_official.sif bash -lc \\
      'cd /workspace && PYTHONPATH=leaderboard pytest -m heavy tests/'
"""
import glob
import os

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from team_code.pipeline_engine import PipelineEngine  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS = sorted(glob.glob(os.path.join(_REPO, "leaderboard", "team_code", "configs", "*.yaml")))
_IDS = [os.path.basename(p) for p in _CONFIGS]


@pytest.mark.heavy
@pytest.mark.parametrize("cfg_path", _CONFIGS, ids=_IDS)
def test_agent_pipeline_builds(cfg_path):
    """Every module in the agent's pipeline instantiates (incl. model weight load)."""
    cfg = yaml.safe_load(open(cfg_path))
    pipeline = cfg.get("pipeline")
    assert isinstance(pipeline, list) and pipeline

    eng = PipelineEngine(pipeline)
    try:
        eng.build()
    except FileNotFoundError as e:
        pytest.skip(f"model weights / assets not present in this environment: {e}")

    # A construction/shape bug (e.g. state_dict mismatch) would have raised above;
    # confirm every declared step produced a module instance.
    assert len(eng._modules) == len(pipeline), \
        f"built {len(eng._modules)} modules for {len(pipeline)} steps"
    for step in eng._modules:
        assert hasattr(step, "run") and callable(step.run), \
            f"module {type(step).__name__} has no callable run()"
