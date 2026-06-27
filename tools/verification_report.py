#!/usr/bin/env python3
"""Aggregate classified run outcomes into an agent-comparison verification report.

Builds on tools/classify_outcomes.py (#1): every run is classified into
valid_pass / valid_fail / agent_error / infra_fail / ran_no_metrics, then rolled
up per agent and per dimension (route-type, town, weather day/night). Agent
performance (pass-rate, mean score) is computed over VALID evaluations only;
infra and ran_no_metrics are reported separately as coverage/health so a degraded
cluster never inflates or deflates an agent's score.

Outputs report/verification_<ts>.{md,csv,json} (and prints the Markdown).

Usage:
  python3 tools/verification_report.py [--dataset DIR] [--outdir report] [--ts TAG]
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_outcomes as co  # noqa: E402

ROUTE_TYPES = ("tiny", "short", "long", "smoke")


def _dims_from_path(run_dir, dataset_root):
    """Extract (agent, town, weather, route_type, route_stem) from a run dir path
    of the form <dataset>/<agent>/weather_<N>/map_<NN>/<route_stem>/.
    """
    rel = os.path.relpath(run_dir, dataset_root).split(os.sep)
    agent = rel[0] if len(rel) > 0 else "unknown"
    weather = town = route_stem = "?"
    for part in rel[1:]:
        if part.startswith("weather_"):
            weather = part.split("_", 1)[1]
        elif part.startswith("map_"):
            town = part.split("_", 1)[1]
    route_stem = rel[-1] if len(rel) > 1 else "?"
    rtype = next((t for t in ROUTE_TYPES if t in route_stem), "other")
    return agent, town, weather, rtype, route_stem


def _weather_bucket(weather):
    """CARLA presets 0-13 are day/sunset, 14-20 are night (per project convention)."""
    try:
        return "night" if int(weather) >= 14 else "day"
    except (TypeError, ValueError):
        return "?"


def collect_rows(dataset_root):
    """One row per run dir: dims + classified outcomes + valid-route scores."""
    rows = []
    for run_dir in co.discover_run_dirs(dataset_root):
        agent, town, weather, rtype, stem = _dims_from_path(run_dir, dataset_root)
        info = co.classify_run_dir(run_dir)
        rows.append({
            "agent": agent, "town": town, "weather": weather,
            "weather_bucket": _weather_bucket(weather),
            "route_type": rtype, "route_stem": stem,
            "outcomes": info["outcomes"], "scores": info["scores"],
        })
    return rows


def summarize(outcomes, scores):
    """Metrics for a group. Performance is over VALID evals only."""
    vp, vf = outcomes.get("valid_pass", 0), outcomes.get("valid_fail", 0)
    ae, inf, rnm = (outcomes.get("agent_error", 0), outcomes.get("infra_fail", 0),
                    outcomes.get("ran_no_metrics", 0))
    valid = vp + vf
    total = valid + ae + inf + rnm
    return {
        "total": total, "valid": valid, "valid_pass": vp, "valid_fail": vf,
        "agent_error": ae, "infra_fail": inf, "ran_no_metrics": rnm,
        "pass_rate": (vp / valid) if valid else None,
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "infra_rate": (inf / total) if total else None,
        "scored_coverage": (len(scores) / total) if total else None,
    }


def group(rows, *keys):
    """Aggregate rows by the given dim keys -> {key: summary}.

    Single key -> scalar key (e.g. 'tcp'); multiple keys -> tuple key.
    """
    acc = defaultdict(lambda: {"outcomes": Counter(), "scores": []})
    for r in rows:
        k = r[keys[0]] if len(keys) == 1 else tuple(r[key] for key in keys)
        acc[k]["outcomes"] += r["outcomes"]
        acc[k]["scores"].extend(r["scores"])
    return {k: summarize(v["outcomes"], v["scores"]) for k, v in acc.items()}


# ---------- rendering ----------
def _pct(x):
    return "  -  " if x is None else f"{100*x:5.1f}%"


def _score(x):
    return "  -  " if x is None else f"{x:5.1f}"


def _headline_table(by_agent):
    lines = ["| agent | scored | valid | pass% | mean score | agent_err | infra% | ran_no_metrics |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for agent in sorted(by_agent):
        s = by_agent[agent]
        lines.append(f"| {agent} | {s['valid']+s['agent_error']} | {s['valid']} | "
                     f"{_pct(s['pass_rate'])} | {_score(s['mean_score'])} | "
                     f"{s['agent_error']} | {_pct(s['infra_rate'])} | {s['ran_no_metrics']} |")
    return "\n".join(lines)


def _breakdown_table(grouped, dim_label):
    lines = [f"| agent | {dim_label} | valid | pass% | mean score | infra% |",
             "|---|---|--:|--:|--:|--:|"]
    for (agent, dim) in sorted(grouped):
        s = grouped[(agent, dim)]
        lines.append(f"| {agent} | {dim} | {s['valid']} | {_pct(s['pass_rate'])} | "
                     f"{_score(s['mean_score'])} | {_pct(s['infra_rate'])} |")
    return "\n".join(lines)


def _ranking(by_agent):
    # Rank by pass-rate over valid evals, then mean score; agents with no valid
    # evals are listed last (unranked).
    def key(a):
        s = by_agent[a]
        pr = s["pass_rate"] if s["pass_rate"] is not None else -1
        ms = s["mean_score"] if s["mean_score"] is not None else -1
        return (-(s["valid"] > 0), -pr, -ms)
    lines = ["| rank | agent | pass% | mean score | n valid |", "|--:|---|--:|--:|--:|"]
    for i, a in enumerate(sorted(by_agent, key=key), 1):
        s = by_agent[a]
        rank = str(i) if s["valid"] else "—"
        lines.append(f"| {rank} | {a} | {_pct(s['pass_rate'])} | {_score(s['mean_score'])} | {s['valid']} |")
    return "\n".join(lines)


def render_markdown(rows, ts):
    by_agent = group(rows, "agent")
    md = [f"# Verification Report — {ts}", "",
          "Agent performance is computed over **valid evaluations only** "
          "(`valid_pass`+`valid_fail`). `infra_fail` (cluster/sim broke) and "
          "`ran_no_metrics` (ran but wrote no scored result) are coverage/health, "
          "excluded from pass-rate and score.", "",
          "## Headline", "", _headline_table(by_agent), "",
          "## Ranking (valid evals only)", "", _ranking(by_agent), "",
          "## By route type", "", _breakdown_table(group(rows, "agent", "route_type"), "route_type"), "",
          "## By town", "", _breakdown_table(group(rows, "agent", "town"), "town"), "",
          "## By day/night", "", _breakdown_table(group(rows, "agent", "weather_bucket"), "day/night"), ""]
    return "\n".join(md), by_agent


def write_csv(path, rows):
    grouped = group(rows, "agent", "route_type", "town", "weather_bucket")
    cols = ["agent", "route_type", "town", "day_night", "total", "valid",
            "valid_pass", "valid_fail", "pass_rate", "mean_score", "agent_error",
            "infra_fail", "ran_no_metrics"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (agent, rtype, town, dn) in sorted(grouped):
            s = grouped[(agent, rtype, town, dn)]
            w.writerow([agent, rtype, town, dn, s["total"], s["valid"],
                        s["valid_pass"], s["valid_fail"],
                        "" if s["pass_rate"] is None else round(s["pass_rate"], 4),
                        "" if s["mean_score"] is None else round(s["mean_score"], 3),
                        s["agent_error"], s["infra_fail"], s["ran_no_metrics"]])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", "dataset"))
    ap.add_argument("--outdir", default="report")
    ap.add_argument("--ts", default=None, help="timestamp tag for filenames")
    args = ap.parse_args(argv)

    rows = collect_rows(args.dataset)
    if not rows:
        print(f"No run output found under {args.dataset}")
        return 1

    ts = args.ts
    if ts is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    md, by_agent = render_markdown(rows, ts)
    print(md)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, f"verification_{ts}")
    with open(base + ".md", "w") as f:
        f.write(md + "\n")
    write_csv(base + ".csv", rows)
    with open(base + ".json", "w") as f:
        json.dump({"timestamp": ts,
                   "by_agent": {a: by_agent[a] for a in by_agent},
                   "by_agent_route_type": {f"{a}|{d}": s for (a, d), s in
                                           group(rows, "agent", "route_type").items()}},
                  f, indent=2)
    print(f"\nWrote {base}.md / .csv / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
