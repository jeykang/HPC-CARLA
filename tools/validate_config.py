#!/usr/bin/env python3
"""Validate / introspect an agent pipeline YAML without CARLA or torch (ast only).

Hard errors (exit 1):
  - malformed pipeline/sensor shape
  - a step whose class doesn't exist in its module
  - bad args type

Introspection: prints each step's class and its args (the actual context-key
wiring, e.g. input_key/output_key) so the pipeline's data flow is reviewable
off-cluster. (Full read/written-key DAG validation is left for when modules
declare their key contract; static inference is unreliable here because modules
wire keys dynamically through args, not string literals.)

Usage:
  python3 tools/validate_config.py leaderboard/team_code/configs/tcp.yaml
  python3 tools/validate_config.py --all
"""
import argparse
import ast
import glob
import os
import sys

import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEAM = os.path.join(_REPO, "leaderboard", "team_code")
_MODULE_FILES = {
    "team_code.pipeline_modules": os.path.join(_TEAM, "pipeline_modules.py"),
    "team_code.pipeline_engine": os.path.join(_TEAM, "pipeline_engine.py"),
}
def _module_ast(path):
    return ast.parse(open(path).read())


def defined_classes():
    """{module_path: {class_name, ...}} for the known pipeline module files."""
    out = {}
    for mod, path in _MODULE_FILES.items():
        if os.path.exists(path):
            out[mod] = {n.name for n in _module_ast(path).body if isinstance(n, ast.ClassDef)}
    return out


def _fmt_args(args):
    if not args:
        return ""
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def validate_config(path, defined=None):
    """Return {ok, errors, steps, sensors}."""
    if defined is None:
        defined = defined_classes()
    errors, steps = [], []
    cfg = yaml.safe_load(open(path))
    if not isinstance(cfg, dict):
        return {"ok": False, "errors": ["top-level YAML is not a mapping"],
                "steps": [], "sensors": 0}

    pipeline = cfg.get("pipeline")
    sensors = cfg.get("sensors")
    if not (isinstance(pipeline, list) and pipeline):
        errors.append("'pipeline' must be a non-empty list")
    if not (isinstance(sensors, list) and sensors):
        errors.append("'sensors' must be a non-empty list")

    for i, step in enumerate(pipeline or []):
        if not isinstance(step, dict):
            errors.append(f"step[{i}] must be a dict"); continue
        module = step.get("module")
        klass = step.get("class_name") or step.get("class")
        args = step.get("args", {})
        if not (isinstance(module, str) and module):
            errors.append(f"step[{i}] missing 'module'")
        if not (isinstance(klass, str) and klass):
            errors.append(f"step[{i}] missing 'class'/'class_name'")
        if not isinstance(args, dict):
            errors.append(f"step[{i}] 'args' must be a mapping"); args = {}
        if module in defined and klass and klass not in defined[module]:
            errors.append(f"step[{i}]: class '{klass}' not found in {module}")
        steps.append({"i": i, "class": klass, "args": args})

    return {"ok": not errors, "errors": errors, "steps": steps,
            "sensors": len(sensors) if isinstance(sensors, list) else 0}


def _print(path, res):
    name = os.path.basename(path)
    print(f"\n=== {name} ===  ({len(res['steps'])} steps, {res['sensors']} sensors)")
    for s in res["steps"]:
        argstr = _fmt_args(s["args"])
        print(f"  [{s['i']:2}] {s['class']}" + (f"  ({argstr})" if argstr else ""))
    for e in res["errors"]:
        print(f"  ERROR {e}")
    print(f"  -> {'OK' if res['ok'] else 'INVALID'}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="path to an agent YAML")
    ap.add_argument("--all", action="store_true", help="validate every config in configs/")
    args = ap.parse_args(argv)

    if args.all or not args.config:
        paths = sorted(glob.glob(os.path.join(_TEAM, "configs", "*.yaml")))
    else:
        paths = [args.config]
    defined = defined_classes()
    bad = 0
    for p in paths:
        res = validate_config(p, defined)
        _print(p, res)
        bad += 0 if res["ok"] else 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
