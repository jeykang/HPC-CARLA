"""Shared pytest setup for the agent-pipeline test suite.

- Puts `leaderboard/` on sys.path so `team_code.*` imports as a namespace package.
- Installs a permissive mock `carla` when the real egg isn't importable (off-cluster).
- Registers the `heavy` marker (real model loading; in-container only).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "leaderboard"))   # team_code.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # _mock_carla

try:
    import carla  # noqa: F401  (real CARLA egg, inside the container)
except Exception:
    import _mock_carla
    sys.modules["carla"] = _mock_carla


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: requires torch/numpy and model weights; run in the container "
        "with `pytest -m heavy` (excluded by default).",
    )
