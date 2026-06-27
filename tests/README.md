# Tests

Two tiers, so most checks run anywhere (no cluster, no CARLA) and the expensive
ones are opt-in inside the container.

## Tier 1 — fast, dependency-light (runs anywhere, incl. the login node)

Pure-Python pipeline-engine contract + config-contract checks. Needs only the
stdlib and `pyyaml`. No torch/CARLA/GPU.

```bash
# without pytest:
python3 tests/test_pipeline_engine.py
python3 tests/test_config_schema.py

# or with pytest (runs tier-1 only by default):
pytest
```

`test_config_schema.py` statically validates every agent YAML (pipeline/sensor
shape) and that each step's class actually exists in `pipeline_modules.py` —
catching typo'd/renamed steps without importing anything heavy.

## Tier 2 — heavy, in-container (`heavy` marker, excluded by default)

Builds each agent's real pipeline from its YAML, instantiating every module
including model-weight loading. Catches construction/shape bugs (e.g. a
state_dict size mismatch) without a cluster round-trip. Needs `numpy`, `torch`,
and the model weights, so run it inside the SIF:

```bash
singularity exec --nv carla_official.sif bash -lc \
  'cd /workspace && PYTHONPATH=leaderboard pytest -m heavy tests/'
```

Off-cluster it auto-skips (numpy/torch absent), so it never burdens the login node.
A permissive mock `carla` (see `_mock_carla.py`) is installed automatically when
the real egg isn't importable.
