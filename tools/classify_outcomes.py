#!/usr/bin/env python3
"""Classify verification-run outcomes into infra-failure vs agent-outcome.

A verification number is only meaningful if "the agent drove badly" is separated
from "the infrastructure broke." On a degraded cluster the latter dominates, and
counting it as agent performance scores the cluster, not the agent. This module
classifies every route into one bucket and aggregates per agent over the *valid*
evaluations only, reporting infra/agent-error rates separately as system health.

Taxonomy
--------
  valid_pass   route status == 'Completed'                      -> agent metric
  valid_fail   'Failed - Agent {timed out,got blocked,...}'     -> agent metric
  agent_error  agent code/config broke (sensors invalid, setup, -> flagged
               crash, entry_status 'Rejected')                     (agent bug)
  infra_fail   no results.json; entry 'Started'/'Crashed';       -> EXCLUDED
               missing route records; global_steps<=0               from scoring

Data sources (per run dir <agent>/weather_N/map_NN/<route>/):
  results.json     leaderboard checkpoint: entry_status, _checkpoint.records[]
                   (status, scores), _checkpoint.progress = [done, total]
  run_summary.json fallback when results.json is absent: global_steps
                   (>0 = the agent ticked; <=0/empty = never ran = infra)

Usage:
  python3 tools/classify_outcomes.py [--dataset DIR] [--json OUT] [--annotate]
  python3 tools/classify_outcomes.py --paths results/simulation_results.2n16g.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

CATEGORIES = ("valid_pass", "valid_fail", "agent_error", "infra_fail", "ran_no_metrics")

# Substrings that mark an agent *code* error (a bug in the agent), vs a driving
# failure (a valid evaluation the agent lost). Order matters: checked first.
_AGENT_ERROR_MARKERS = ("agent crashed", "sensors were invalid", "couldn't be set up",
                        "could not be set up")


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def classify_route(status):
    """Classify a single leaderboard route record `status` string."""
    s = (status or "").strip()
    low = s.lower()
    if s == "Completed":
        return "valid_pass"
    if any(m in low for m in _AGENT_ERROR_MARKERS):
        return "agent_error"
    if low.startswith("failed"):
        return "valid_fail"   # timed out / got blocked / deviated / collision / ...
    # Unknown/empty status on a recorded route — treat conservatively as infra.
    return "infra_fail"


def classify_results(results):
    """Classify a parsed results.json dict.

    Returns dict: outcomes Counter, scores list (composed over valid routes),
    entry_status, expected, recorded.
    """
    out = Counter()
    scores = []
    entry = (results.get("entry_status") or "").strip()
    cp = results.get("_checkpoint", {}) or {}
    records = cp.get("records", []) or []
    progress = cp.get("progress") or []
    expected = progress[1] if len(progress) == 2 and isinstance(progress[1], int) else len(records)

    # Whole-run agent-setup rejection: no usable evaluations.
    if entry == "Rejected":
        out["agent_error"] += max(expected, 1)
        return {"outcomes": out, "scores": scores, "entry_status": entry,
                "expected": expected, "recorded": len(records)}

    for r in records:
        cat = classify_route(r.get("status"))
        out[cat] += 1
        if cat in ("valid_pass", "valid_fail"):
            sc = (r.get("scores") or {}).get("score_composed")
            if isinstance(sc, (int, float)):
                scores.append(float(sc))

    # Routes that should have run but have no record => run was cut short (infra).
    missing = max(0, expected - len(records))
    if missing:
        out["infra_fail"] += missing
    return {"outcomes": out, "scores": scores, "entry_status": entry,
            "expected": expected, "recorded": len(records)}


def classify_run_dir(run_dir):
    """Classify one run directory, preferring results.json, falling back to
    run_summary.json (coarse ran-vs-infra) when the leaderboard wrote no checkpoint.
    """
    results = _read_json(os.path.join(run_dir, "results.json"))
    if results is not None:
        info = classify_results(results)
        info["source"] = "results.json"
        return info

    # Degraded mode: no leaderboard checkpoint. Use run_summary global_steps.
    summary = _read_json(os.path.join(run_dir, "run_summary.json"))
    out = Counter()
    if summary is None:
        out["infra_fail"] += 1
        return {"outcomes": out, "scores": [], "entry_status": "no_output",
                "expected": 1, "recorded": 0, "source": "none"}
    gs = summary.get("global_steps", -1)
    if isinstance(gs, int) and gs > 0:
        out["ran_no_metrics"] += 1            # agent ticked but no scored result
    else:
        out["infra_fail"] += 1                # never ticked => infra
    return {"outcomes": out, "scores": [], "entry_status": "run_summary_only",
            "expected": 1, "recorded": 1 if out.get("ran_no_metrics") else 0,
            "source": "run_summary.json"}


def _agent_of(run_dir, dataset_root):
    rel = os.path.relpath(run_dir, dataset_root)
    parts = rel.split(os.sep)
    return parts[0] if parts and parts[0] != ".." else "unknown"


def discover_run_dirs(dataset_root):
    """Bounded glob of <agent>/weather_*/map_*/<route>/ dirs holding output."""
    dirs = set()
    for name in ("results.json", "run_summary.json"):
        for p in glob.glob(os.path.join(dataset_root, "*", "weather_*", "map_*", "*", name)):
            dirs.add(os.path.dirname(p))
    return sorted(dirs)


def aggregate(dataset_root):
    """Classify every run dir under dataset_root; return per-agent aggregates."""
    per_agent = defaultdict(lambda: {"outcomes": Counter(), "scores": [], "runs": 0})
    for run_dir in discover_run_dirs(dataset_root):
        agent = _agent_of(run_dir, dataset_root)
        info = classify_run_dir(run_dir)
        per_agent[agent]["outcomes"] += info["outcomes"]
        per_agent[agent]["scores"].extend(info["scores"])
        per_agent[agent]["runs"] += 1
    return per_agent


def _fmt_agent(agent, agg):
    o = agg["outcomes"]
    vp, vf = o.get("valid_pass", 0), o.get("valid_fail", 0)
    valid = vp + vf
    ae, inf, rnm = o.get("agent_error", 0), o.get("infra_fail", 0), o.get("ran_no_metrics", 0)
    total = valid + ae + inf + rnm
    pass_rate = (100.0 * vp / valid) if valid else float("nan")
    mean_score = (sum(agg["scores"]) / len(agg["scores"])) if agg["scores"] else float("nan")
    return (f"{agent:<12} runs={agg['runs']:<4} routes={total:<5} "
            f"valid={valid:<4} pass={vp:<4} fail={vf:<4} "
            f"pass%={pass_rate:6.1f}  meanScore={mean_score:6.1f}  "
            f"| agent_err={ae:<3} infra={inf:<4} ran_no_metrics={rnm:<4}")


def print_report(per_agent):
    print("=" * 110)
    print("VERIFICATION OUTCOMES  (agent metrics computed over VALID evals only; "
          "infra/ran_no_metrics excluded)")
    print("=" * 110)
    if not per_agent:
        print("No run output found.")
        return
    for agent in sorted(per_agent):
        print(_fmt_agent(agent, per_agent[agent]))
    # System-health rollup
    allo = Counter()
    for agg in per_agent.values():
        allo += agg["outcomes"]
    tot = sum(allo.values()) or 1
    print("-" * 110)
    print("System health (all agents): " + ", ".join(
        f"{k}={allo.get(k,0)} ({100*allo.get(k,0)//tot}%)" for k in CATEGORIES))
    print("=" * 110)
    print("NOTE: ran_no_metrics = agent ticked but the leaderboard wrote no scored "
          "results.json (pre---checkpoint-fix runs). Future runs classify fully.")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", "dataset"),
                    help="dataset root to scan (default: $DATASET_DIR or ./dataset)")
    ap.add_argument("--paths", nargs="+", help="classify specific results.json files instead")
    ap.add_argument("--json", help="write the per-agent summary as JSON to this path")
    args = ap.parse_args(argv)

    if args.paths:
        # Ad-hoc: classify explicit results.json files (e.g. results/ archives).
        for p in args.paths:
            r = _read_json(p)
            if r is None:
                print(f"{p}: unreadable"); continue
            info = classify_results(r)
            o = info["outcomes"]
            print(f"{p}: entry={info['entry_status']!r} expected={info['expected']} "
                  f"recorded={info['recorded']} -> " +
                  ", ".join(f"{k}={o.get(k,0)}" for k in CATEGORIES if o.get(k)))
        return 0

    per_agent = aggregate(args.dataset)
    print_report(per_agent)
    if args.json:
        serial = {a: {"outcomes": dict(v["outcomes"]), "runs": v["runs"],
                      "n_scores": len(v["scores"]),
                      "mean_score": (sum(v["scores"]) / len(v["scores"])) if v["scores"] else None}
                  for a, v in per_agent.items()}
        with open(args.json, "w") as f:
            json.dump(serial, f, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
