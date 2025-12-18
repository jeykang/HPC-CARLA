"""Lightweight, config-defined pipeline engine for new agents.

This module is intentionally minimal: it provides a way to compose a new agent
from a list of modules declared in YAML, without writing a new Leaderboard agent
class.

Legacy agents (InterFuser/LAV/TCP/...) should continue to run through the
compatibility proxy path in `consolidated_agent.py`.

Pipeline modules are regular Python classes with optional:
- setup(agent, full_config)
- run(context) -> context or any value

The pipeline is executed per-tick. By convention, modules can write:
- context['control'] as either a carla.VehicleControl or a dict with keys
  {steer, throttle, brake}.

The engine does not import heavy deps (torch/carla) to keep import-time light.
"""
import importlib
from typing import Any, Dict, List


PIPELINE_STOP_KEY = "__pipeline_stop__"


def dynamic_import(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'") from exc


class PipelineEngine:
    def __init__(self, specs: List[Dict[str, Any]]):
        if not isinstance(specs, list) or not specs:
            raise ValueError("pipeline must be a non-empty list")
        self._specs = specs
        self._modules: List[Any] = []

    def build(self) -> None:
        modules: List[Any] = []
        for spec in self._specs:
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid pipeline step (expected dict): {spec!r}")

            module_path = spec.get("module")
            class_name = spec.get("class_name") or spec.get("class")
            args = spec.get("args") or {}

            if not module_path or not class_name:
                raise ValueError(f"Pipeline step must include module and class_name: {spec!r}")
            if not isinstance(args, dict):
                raise ValueError(f"Pipeline step args must be a dict: {spec!r}")

            StepClass = dynamic_import(module_path, class_name)
            try:
                step = StepClass(**args)
            except TypeError:
                step = StepClass()
                for k, v in args.items():
                    setattr(step, k, v)
            modules.append(step)

        self._modules = modules

    def setup(self, agent: Any, full_config: Dict[str, Any]) -> None:
        if not self._modules:
            self.build()
        for step in self._modules:
            fn = getattr(step, "setup", None)
            if callable(fn):
                fn(agent, full_config)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self._modules:
            raise RuntimeError("PipelineEngine not built")

        ctx = context
        for step in self._modules:
            fn = getattr(step, "run", None)
            if not callable(fn):
                raise TypeError(f"Pipeline step has no run(context): {step!r}")
            out = fn(ctx)
            if out is None:
                continue
            if isinstance(out, dict):
                ctx = out
                if ctx.get(PIPELINE_STOP_KEY):
                    break
            else:
                # Allow a module to directly return a control-like object.
                ctx["control"] = out
        return ctx


# ---------------------------------------------------------------------------
# Minimal built-in modules (for testing / templates)
# ---------------------------------------------------------------------------


class FixedControl:
    """Always returns the same control dict.

    Useful to validate the pipeline wiring end-to-end.
    """

    def __init__(self, steer: float = 0.0, throttle: float = 0.0, brake: float = 0.0):
        self.steer = float(steer)
        self.throttle = float(throttle)
        self.brake = float(brake)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context = dict(context)
        context["control"] = {"steer": self.steer, "throttle": self.throttle, "brake": self.brake}
        return context
