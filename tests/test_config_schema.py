"""Tier-1 config-contract tests — yaml + ast only, no torch/carla.

Delegates to tools/validate_config.py (the same validator behind
`continuous_cli.py validate-config`) so the test and the dev CLI never drift.
Asserts every agent YAML is structurally valid and that each pipeline step's
class actually exists in pipeline_modules.py (static, no import, no torch).

Run directly:   python3 tests/test_config_schema.py
Or via pytest:  pytest tests/test_config_schema.py
"""
import glob
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tools"))

import validate_config as vc  # noqa: E402

_CONFIGS = sorted(glob.glob(os.path.join(_REPO, "leaderboard", "team_code", "configs", "*.yaml")))


def test_all_agent_configs_valid():
    assert _CONFIGS, "no agent configs found"
    defined = vc.defined_classes()
    for path in _CONFIGS:
        res = vc.validate_config(path, defined)
        assert res["ok"], f"{os.path.basename(path)}: {res['errors']}"


def _run_all():
    if not _CONFIGS:
        print("  FAIL: no agent configs found")
        return 1
    defined = vc.defined_classes()
    failed = 0
    for path in _CONFIGS:
        res = vc.validate_config(path, defined)
        name = os.path.basename(path)
        if res["ok"]:
            print(f"  PASS {name:16} ({len(res['steps'])} steps, {res['sensors']} sensors)")
        else:
            failed += 1
            print(f"  FAIL {name:16} {res['errors']}")
    print(f"\n{len(_CONFIGS)-failed}/{len(_CONFIGS)} configs valid")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
