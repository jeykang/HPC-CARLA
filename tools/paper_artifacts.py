#!/usr/bin/env python3
"""Generate the paper's data tables from existing run output — one command.

Emits CSV/JSON tables (the data behind every figure) into an output dir, using
only the stdlib + pyyaml, so it runs anywhere — including the login node. It
reuses the verification tooling (classify_outcomes / verification_report /
parse_sim_ratio) plus the raw metric streams (gpu.jsonl, completed_jobs.json).

Figures (PNGs) are intentionally NOT rendered here: matplotlib isn't available
on the login node, and `genfig.py` already owns plotting. Render plots from these
CSVs with genfig (in the container / on a workstation). This tool guarantees the
*numbers* are always reproducible regardless of environment.

Usage:
  python3 tools/paper_artifacts.py [--dataset dataset] [--state-root collection_state]
                                   [--logs logs] [--outdir paper_artifacts]
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_outcomes as co          # noqa: E402
import verification_report as vr         # noqa: E402
import parse_sim_ratio as psr            # noqa: E402

CATS = ("valid_pass", "valid_fail", "agent_error", "infra_fail", "ran_no_metrics")


def _pctiles(vals, ps=(0, 10, 50, 90, 100)):
    vals = sorted(vals)
    if not vals:
        return {p: None for p in ps}
    n = len(vals)
    return {p: vals[min(n - 1, int(p / 100 * n))] for p in ps}


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---- tables -----------------------------------------------------------------
def t_outcomes(rows, outdir):
    """Outcome taxonomy per agent (feeds the failure-taxonomy figure)."""
    by_agent = vr.group(rows, "agent")
    out = []
    for a in sorted(by_agent):
        s = by_agent[a]
        out.append([a, s["total"], s["valid_pass"], s["valid_fail"],
                    s["agent_error"], s["infra_fail"], s["ran_no_metrics"]])
    _write_csv(os.path.join(outdir, "outcomes_by_agent.csv"),
               ["agent", "total", "valid_pass", "valid_fail", "agent_error",
                "infra_fail", "ran_no_metrics"], out)
    return len(out)


def t_agent_comparison(rows, outdir):
    """Per-agent headline + per-dimension comparison (feeds the results table)."""
    def emit(name, *keys):
        g = vr.group(rows, *keys)
        hdr = list(keys) + ["total", "valid", "pass_rate", "mean_score",
                            "agent_error", "infra_fail", "ran_no_metrics"]
        body = []
        for k in sorted(g):
            s = g[k]
            kk = list(k) if isinstance(k, tuple) else [k]
            body.append(kk + [s["total"], s["valid"],
                              "" if s["pass_rate"] is None else round(s["pass_rate"], 4),
                              "" if s["mean_score"] is None else round(s["mean_score"], 3),
                              s["agent_error"], s["infra_fail"], s["ran_no_metrics"]])
        _write_csv(os.path.join(outdir, name), hdr, body)
    emit("agent_headline.csv", "agent")
    emit("agent_by_town.csv", "agent", "town")
    emit("agent_by_weather.csv", "agent", "weather")
    emit("agent_by_daynight.csv", "agent", "weather_bucket")
    emit("agent_by_routetype.csv", "agent", "route_type")


def t_sim_ratio(logs_glob, outdir):
    """Sim System/Game-time ratio distribution (feeds the throughput figure)."""
    files = sorted(glob.glob(logs_glob))
    allv = []
    per = []
    for f in files:
        v = psr.parse(f)
        if not v:
            continue
        allv.extend(v)
        p = _pctiles(v)
        per.append([os.path.basename(f), len(v), round(sum(v) / len(v), 4),
                    p[0], p[10], p[50], p[90], p[100]])
    _write_csv(os.path.join(outdir, "sim_ratio_by_log.csv"),
               ["log", "n", "mean", "min", "p10", "p50", "p90", "max"], per)
    if allv:
        p = _pctiles(allv)
        _write_csv(os.path.join(outdir, "sim_ratio_overall.csv"),
                   ["n", "mean", "min", "p10", "p50", "p90", "max"],
                   [[len(allv), round(sum(allv) / len(allv), 4),
                     p[0], p[10], p[50], p[90], p[100]]])
    # raw values for histogram rendering
    _write_csv(os.path.join(outdir, "sim_ratio_values.csv"), ["ratio"], [[v] for v in allv])
    return len(allv)


def t_gpu_util(state_root, outdir):
    """GPU utilization / memory / power distribution (feeds the utilization figure)."""
    files = glob.glob(os.path.join(state_root, "metrics", "node", "*", "gpu*.jsonl"))
    fields = ("util_sm", "tempC", "power_W")
    acc = {k: [] for k in fields}
    mem_pct = []
    n = 0
    for f in files:
        try:
            for line in open(f):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                n += 1
                for k in fields:
                    if isinstance(d.get(k), (int, float)):
                        acc[k].append(float(d[k]))
                if d.get("mem_total_MiB"):
                    mem_pct.append(100.0 * d.get("mem_used_MiB", 0) / max(1, d["mem_total_MiB"]))
        except Exception:
            continue
    rows = []
    for k, vals in list(acc.items()) + [("mem_pct", mem_pct)]:
        if vals:
            p = _pctiles(vals)
            rows.append([k, len(vals), round(sum(vals) / len(vals), 2),
                         p[0], p[10], p[50], p[90], p[100]])
    _write_csv(os.path.join(outdir, "gpu_util_summary.csv"),
               ["metric", "n", "mean", "min", "p10", "p50", "p90", "max"], rows)
    return n


def t_job_durations(state_root, outdir):
    """Job-duration distribution + runaway concentration (feeds the duration figure)."""
    path = os.path.join(state_root, "completed_jobs.json")
    try:
        jobs = json.load(open(path)).get("jobs", [])
    except Exception:
        jobs = []
    durs = sorted((j.get("duration") or 0) for j in jobs if j.get("duration"))
    if not durs:
        _write_csv(os.path.join(outdir, "job_durations.csv"), ["note"], [["no durations"]])
        return 0
    total = sum(durs)
    # concentration: share of compute in the top-k longest jobs
    desc = sorted(durs, reverse=True)
    conc = []
    for k in (1, 5, 10, 20, 50):
        if k <= len(desc):
            conc.append([k, round(100.0 * sum(desc[:k]) / total, 1)])
    _write_csv(os.path.join(outdir, "job_duration_concentration.csv"),
               ["top_k_jobs", "pct_of_total_compute"], conc)
    _write_csv(os.path.join(outdir, "job_duration_values.csv"), ["seconds"], [[d] for d in durs])
    return len(durs)


def _write_index(outdir, summary):
    lines = ["# Paper artifacts (data tables)\n",
             "Generated by `tools/paper_artifacts.py` from existing run output. "
             "Render figures from these CSVs with `genfig.py` (needs matplotlib).\n",
             "| file | feeds figure/table |", "|---|---|",
             "| outcomes_by_agent.csv | failure taxonomy (infra vs agent) |",
             "| agent_headline.csv / agent_by_*.csv | agent comparison table + per-dimension figures |",
             "| sim_ratio_*.csv | sim throughput (System/Game ratio) distribution |",
             "| gpu_util_summary.csv | GPU utilization / mem / power |",
             "| job_duration_*.csv | job-duration distribution + runaway concentration |", ""]
    lines.append("## Counts at generation\n")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(outdir, "INDEX.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", "dataset"))
    ap.add_argument("--state-root", default="collection_state")
    ap.add_argument("--logs", default="logs/worker_*.log")
    ap.add_argument("--outdir", default="paper_artifacts")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    rows = vr.collect_rows(args.dataset)
    summary = {}
    summary["run_dirs"] = len(rows)
    t_outcomes(rows, args.outdir)
    t_agent_comparison(rows, args.outdir)
    summary["sim_ratio_routes"] = t_sim_ratio(args.logs, args.outdir)
    summary["gpu_samples"] = t_gpu_util(args.state_root, args.outdir)
    summary["jobs_with_duration"] = t_job_durations(args.state_root, args.outdir)
    _write_index(args.outdir, summary)

    print(f"Wrote tables to {args.outdir}/  (run_dirs={summary['run_dirs']}, "
          f"sim_ratio_routes={summary['sim_ratio_routes']}, "
          f"gpu_samples={summary['gpu_samples']}, "
          f"jobs_with_duration={summary['jobs_with_duration']})")
    print(f"See {args.outdir}/INDEX.md. Render figures with genfig.py (matplotlib).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
