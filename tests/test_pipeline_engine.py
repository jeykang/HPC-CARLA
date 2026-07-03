"""Tier-1 pipeline-engine contract tests — pure Python, no torch/carla/numpy.

Runs anywhere (including the login node) with just the stdlib. Exercises the
PipelineEngine build/run contract and the ctx-flow semantics that wiring bugs hit.

Run directly (no pytest needed):   python3 tests/test_pipeline_engine.py
Or via pytest:                      pytest tests/test_pipeline_engine.py
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "leaderboard"))

from team_code.pipeline_engine import (  # noqa: E402
    PipelineEngine, FixedControl, dynamic_import, PIPELINE_STOP_KEY,
)


# --- stub modules used to drive run() without importing heavy deps -----------
class _Writer:
    def run(self, ctx):
        ctx = dict(ctx)
        ctx["written"] = 42
        return ctx


class _Reader:
    def run(self, ctx):
        assert "written" in ctx, "Reader ran before Writer wrote its key"
        ctx = dict(ctx)
        ctx["read"] = ctx["written"] + 1
        return ctx


class _Stopper:
    def run(self, ctx):
        ctx = dict(ctx)
        ctx[PIPELINE_STOP_KEY] = True
        return ctx


class _ShouldNotRun:
    def run(self, ctx):
        raise AssertionError("module after a stopper must not run")


class _ReturnsControlObj:
    def run(self, ctx):
        return ("steer", 0.1)  # non-dict -> engine assigns to ctx['control']


def test_rejects_empty_or_nonlist_specs():
    for bad in ([], "nope", None, {}):
        try:
            PipelineEngine(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for specs={bad!r}")


def test_run_before_build_raises():
    eng = PipelineEngine([{"module": "x", "class": "y"}])
    try:
        eng.run({})
    except RuntimeError:
        return
    raise AssertionError("run() before build() must raise RuntimeError")


def test_build_and_run_fixedcontrol():
    eng = PipelineEngine([{
        "module": "team_code.pipeline_engine", "class": "FixedControl",
        "args": {"steer": 0.2, "throttle": 0.5, "brake": 0.0},
    }])
    eng.build()
    ctx = eng.run({})
    assert ctx["control"] == {"steer": 0.2, "throttle": 0.5, "brake": 0.0}, ctx["control"]


def test_ctx_flow_writer_then_reader():
    eng = PipelineEngine([{"module": "team_code.pipeline_engine", "class": "FixedControl"}])
    eng._modules = [_Writer(), _Reader()]
    ctx = eng.run({})
    assert ctx["read"] == 43, ctx


def test_stop_key_halts_pipeline():
    eng = PipelineEngine([{"module": "team_code.pipeline_engine", "class": "FixedControl"}])
    eng._modules = [_Stopper(), _ShouldNotRun()]
    eng.run({})  # must not raise


def test_nondict_return_sets_control():
    eng = PipelineEngine([{"module": "team_code.pipeline_engine", "class": "FixedControl"}])
    eng._modules = [_ReturnsControlObj()]
    ctx = eng.run({})
    assert ctx["control"] == ("steer", 0.1), ctx


def test_build_rejects_missing_fields():
    for bad in ([{"module": "m"}], [{"class": "c"}], [{"args": {}}], ["notadict"]):
        eng = PipelineEngine(bad)
        try:
            eng.build()
        except (ValueError, ImportError):
            continue
        raise AssertionError(f"expected build() to reject {bad!r}")


def test_dynamic_import_missing_class():
    try:
        dynamic_import("team_code.pipeline_engine", "NoSuchClass")
    except ImportError:
        return
    raise AssertionError("dynamic_import must raise ImportError for missing class")


def test_args_setattr_fallback():
    # FixedControl.__init__ accepts kwargs, but ensure unknown attrs via setattr
    # path don't crash the engine for a no-arg-ctor class.
    class NoArgs:
        def run(self, ctx):
            return ctx
    eng = PipelineEngine([{"module": "team_code.pipeline_engine", "class": "FixedControl"}])
    eng._modules = [NoArgs()]
    eng.run({})  # no exception


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
