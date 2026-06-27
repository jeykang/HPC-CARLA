"""Tier-1 config-contract tests — yaml + ast only, no torch/carla.

Validates every agent YAML's pipeline/sensor shape and, crucially, that each
pipeline step's class actually EXISTS in pipeline_modules.py (statically, via ast
— no import, no torch). This catches the "class not found" / typo'd-step class of
config bug that previously only surfaced at runtime on the cluster.

Run directly:   python3 tests/test_config_schema.py
Or via pytest:  pytest tests/test_config_schema.py
"""
import ast
import glob
import os
import sys

import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS = os.path.join(_REPO, "leaderboard", "team_code", "configs")
_MODULES_PY = os.path.join(_REPO, "leaderboard", "team_code", "pipeline_modules.py")
_ENGINE_PY = os.path.join(_REPO, "leaderboard", "team_code", "pipeline_engine.py")


def _classes_in(path):
    tree = ast.parse(open(path).read())
    return {n.name for n in tree.body if isinstance(n, ast.ClassDef)}


def _defined_classes():
    classes = {}
    for path, mod in ((_MODULES_PY, "team_code.pipeline_modules"),
                      (_ENGINE_PY, "team_code.pipeline_engine")):
        for c in _classes_in(path):
            classes.setdefault(mod, set()).add(c)
    return classes


def _agent_configs():
    return sorted(glob.glob(os.path.join(_CONFIGS, "*.yaml")))


def _check_config(path, defined):
    cfg = yaml.safe_load(open(path))
    name = os.path.basename(path)
    assert isinstance(cfg, dict), f"{name}: top-level must be a mapping"

    pipeline = cfg.get("pipeline")
    assert isinstance(pipeline, list) and pipeline, f"{name}: 'pipeline' must be a non-empty list"

    sensors = cfg.get("sensors")
    assert isinstance(sensors, list) and sensors, f"{name}: 'sensors' must be a non-empty list"
    for s in sensors:
        assert isinstance(s, dict) and ("id" in s or "type" in s), \
            f"{name}: each sensor must be a dict with id/type: {s!r}"

    for i, step in enumerate(pipeline):
        assert isinstance(step, dict), f"{name}[{i}]: step must be a dict"
        module = step.get("module")
        klass = step.get("class_name") or step.get("class")
        args = step.get("args", {})
        assert isinstance(module, str) and module, f"{name}[{i}]: missing 'module'"
        assert isinstance(klass, str) and klass, f"{name}[{i}]: missing 'class'/'class_name'"
        assert isinstance(args, dict), f"{name}[{i}]: 'args' must be a mapping"
        if module in defined:
            assert klass in defined[module], \
                f"{name}[{i}]: class '{klass}' not found in {module}"
    return len(pipeline), len(sensors)


def test_all_agent_configs_valid():
    defined = _defined_classes()
    configs = _agent_configs()
    assert configs, "no agent configs found"
    for path in configs:
        _check_config(path, defined)


def _run_all():
    defined = _defined_classes()
    failed = 0
    configs = _agent_configs()
    if not configs:
        print("  FAIL: no agent configs found")
        return 1
    for path in configs:
        try:
            n_steps, n_sensors = _check_config(path, defined)
            print(f"  PASS {os.path.basename(path):16} ({n_steps} steps, {n_sensors} sensors)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {os.path.basename(path):16} {type(e).__name__}: {e}")
    print(f"\n{len(configs)-failed}/{len(configs)} configs valid")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
