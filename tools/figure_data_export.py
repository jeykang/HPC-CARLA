#!/usr/bin/env python3
"""
figure_data_export.py — pull + pre-aggregate raw cluster telemetry into small,
portable CSVs so the paper figures can be BUILT ELSEWHERE (no cluster data, no
CARLA, matplotlib on a laptop).

Runs on the login node (stdlib only; no numpy/pandas). Reads the live state +
worker logs, writes a self-contained bundle directory of CSVs + a README spec +
a matplotlib build script.

Covers the two open tracker items:
  * 2b  — Fig. 4 companion: route-evals/hour vs effective active-GPU count
  * 14  — Fig. 7 (appendix): failure taxonomy (figL5) + server stability (figL4)

Usage:
  python3 tools/figure_data_export.py [--out paper_figures_data] \
      [--state collection_state] [--logs logs] [--harvest paper_artifacts/per_route_results.csv]

If the harvest CSV is missing, run `python3 tools/harvest_results.py` first
(it writes paper_artifacts/per_route_results.csv).
"""
import argparse, csv, glob, json, os, re, collections
from datetime import datetime, timedelta, timezone


def _parse_ts(s):
    """ISO8601 (with trailing Z or offset, optional fractional secs) -> aware datetime."""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # tolerate space separator / no offset
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hour_floor(dt):
    return dt.replace(minute=0, second=0, microsecond=0)


def load_terminal_jobs(state_dir):
    """Union completed_jobs + job_queue; keep terminal (completed/failed) jobs
    that have both start and end timestamps."""
    jobs = {}
    for fn in ("completed_jobs.json", "job_queue.json"):
        p = os.path.join(state_dir, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["jobs"] if isinstance(d, dict) and "jobs" in d else d
        for j in rows:
            if j.get("status") not in ("completed", "failed"):
                continue
            st, en = _parse_ts(j.get("start_time")), _parse_ts(j.get("end_time"))
            if not (st and en) or en <= st:
                continue
            jid = j["id"]
            # completed_jobs wins over job_queue for the same id (has scores)
            if jid in jobs and fn == "job_queue.json":
                continue
            jobs[jid] = {
                "job_id": jid, "agent": j.get("agent"), "node": j.get("node"),
                "gpu": j.get("gpu"), "status": j.get("status"),
                "start": st, "end": en, "duration_s": int((en - st).total_seconds()),
            }
    return jobs


def load_harvest(csv_path):
    """Return (evals_per_job dict, outcome Counter, outcome_by_agent, rows)."""
    per_job = collections.Counter()
    outcome = collections.Counter()
    by_agent = collections.defaultdict(collections.Counter)
    rows = []
    if not os.path.exists(csv_path):
        return per_job, outcome, by_agent, rows
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
            try:
                per_job[int(r["job_id"])] += 1
            except (KeyError, ValueError):
                pass
            outcome[r.get("status", "?")] += 1
            by_agent[r.get("agent", "?")][r.get("status", "?")] += 1
    return per_job, outcome, by_agent, rows


# ----------------------------------------------------------------------------
# Fig 2b — throughput vs effective active-GPU count
# ----------------------------------------------------------------------------
def build_2b(jobs, evals_per_job, out_dir):
    if not jobs:
        return 0, 0
    js = list(jobs.values())
    t0 = min(j["start"] for j in js)
    t1 = max(j["end"] for j in js)

    # minute-resolution concurrency sweep
    step = timedelta(minutes=1)
    per_hour_conc = collections.defaultdict(list)      # hour -> [concurrency samples]
    per_hour_nodes = collections.defaultdict(set)
    t = t0
    while t <= t1:
        active = [j for j in js if j["start"] <= t < j["end"]]
        h = _hour_floor(t)
        per_hour_conc[h].append(len(active))
        for j in active:
            per_hour_nodes[h].add(j["node"])
        t += step

    # route-evals + jobs attributed to the hour a job ENDED
    per_hour_evals = collections.Counter()
    per_hour_jobs = collections.Counter()
    for j in js:
        h = _hour_floor(j["end"])
        per_hour_evals[h] += evals_per_job.get(j["job_id"], 0)
        per_hour_jobs[h] += 1

    hours = sorted(set(per_hour_conc) | set(per_hour_evals))
    hp = os.path.join(out_dir, "fig2b_hourly.csv")
    with open(hp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour_utc", "jobs_ended", "route_evals",
                    "active_gpu_mean", "active_gpu_max", "nodes_active_max"])
        for h in hours:
            conc = per_hour_conc.get(h, [0])
            w.writerow([
                h.strftime("%Y-%m-%dT%H:00Z"),
                per_hour_jobs.get(h, 0),
                per_hour_evals.get(h, 0),
                round(sum(conc) / len(conc), 3),
                max(conc),
                len(per_hour_nodes.get(h, ())),
            ])

    # raw per-job intervals so the builder can recompute concurrency any way
    ip = os.path.join(out_dir, "fig2b_job_intervals.csv")
    with open(ip, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "agent", "node", "gpu", "status",
                    "start_utc", "end_utc", "duration_s", "route_evals"])
        for j in sorted(js, key=lambda x: x["start"]):
            w.writerow([j["job_id"], j["agent"], j["node"], j["gpu"], j["status"],
                        j["start"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                        j["end"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                        j["duration_s"], evals_per_job.get(j["job_id"], 0)])
    return len(hours), len(js)


# ----------------------------------------------------------------------------
# Fig L5 — failure taxonomy
# ----------------------------------------------------------------------------
def build_L5(outcome, by_agent, log_glob, out_dir):
    # (a) leaderboard route-outcome taxonomy (primary)
    tp = os.path.join(out_dir, "figL5_outcome_taxonomy.csv")
    total = sum(outcome.values()) or 1
    with open(tp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "count", "pct_of_route_evals"])
        for s, n in outcome.most_common():
            w.writerow([s, n, round(100 * n / total, 2)])

    # (b) outcome x agent matrix
    statuses = [s for s, _ in outcome.most_common()]
    mp = os.path.join(out_dir, "figL5_outcome_by_agent.csv")
    with open(mp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent"] + statuses + ["total"])
        for a in sorted(by_agent):
            row = [by_agent[a].get(s, 0) for s in statuses]
            w.writerow([a] + row + [sum(row)])

    # (c) python-exception taxonomy from worker logs (supplementary)
    exc = collections.Counter()
    pat = re.compile(r"\b([A-Z][A-Za-z_]*(?:Error|Exception|Timeout))\b")
    for lg in glob.glob(log_glob):
        try:
            with open(lg, errors="ignore") as f:
                for line in f:
                    for m in pat.findall(line):
                        exc[m] += 1
        except OSError:
            pass
    # Worker logs ACCUMULATE across runs (no per-line timestamps), so the raw tally
    # mixes campaigns. Subtract a prior-run baseline (env EXC_BASELINE as JSON, e.g.
    # the 2026-07-15 counts) to attribute exceptions to the current sweep. Without a
    # baseline, count_full_sweep == count_cumulative.
    baseline = {}
    _b = os.environ.get("EXC_BASELINE", "")
    if _b:
        try:
            baseline = {k: int(v) for k, v in json.loads(_b).items()}
        except Exception:
            baseline = {}
    ep = os.path.join(out_dir, "figL5_exceptions_from_logs.csv")
    with open(ep, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exception_type", "count_full_sweep", "count_cumulative", "note"])
        note = "delta vs baseline (logs accumulate)" if baseline else "cumulative (no baseline)"
        for s, n in sorted(exc.items(), key=lambda kv: -kv[1]):
            w.writerow([s, max(0, n - baseline.get(s, 0)), n, note])
    return total, dict(exc)


# ----------------------------------------------------------------------------
# Fig L4 — server stability
# ----------------------------------------------------------------------------
def build_L4(log_glob, state_dir, out_dir):
    # (a) current run: boot attempts / healthy events per (node,gpu) from worker logs
    boots = collections.defaultdict(lambda: collections.Counter())
    node_re = re.compile(r"worker_(?P<node>[^_]+(?:-[^_]+)*)_gpu(?P<gpu>\d+)\.log$")
    launch_re = re.compile(r"\[server_manager\]\s*gpu(\d+):\s*launch attempt")
    healthy_re = re.compile(r"\[server_manager\]\s*gpu(\d+):\s*healthy")
    nolisten_re = re.compile(r"\[server_manager\]\s*gpu(\d+):\s*no listener")
    for lg in glob.glob(log_glob):
        m = node_re.search(os.path.basename(lg))
        node = m.group("node") if m else os.path.basename(lg)
        gpu = m.group("gpu") if m else "?"
        key = (node, gpu)
        try:
            with open(lg, errors="ignore") as f:
                for line in f:
                    if launch_re.search(line):
                        boots[key]["launch_attempts"] += 1
                    if healthy_re.search(line):
                        boots[key]["healthy"] += 1
                    if nolisten_re.search(line):
                        boots[key]["relaunch_triggers"] += 1
        except OSError:
            pass
    cp = os.path.join(out_dir, "figL4_server_boots_current.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        # server_boots (= 'healthy' events) is THE boot count and the primary column;
        # relaunch_triggers_nolistener is only the no-listener sub-path (most boots
        # follow a segfault, not a no-listener) — kept secondary so it can't be
        # misread as the boot total (it undercounts ~45x).
        w.writerow(["node", "gpu", "server_boots", "relaunch_triggers_nolistener"])
        for (node, gpu), c in sorted(boots.items()):
            w.writerow([node, gpu, c["healthy"], c["relaunch_triggers"]])

    # (b) historical run: rich carla_pool.jsonl (ready_seconds + events over time)
    hist_rows = []
    for pool in glob.glob(os.path.join(state_dir, "metrics/servers/*/carla_pool.jsonl")):
        try:
            with open(pool, errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    hist_rows.append(d)
        except (OSError, json.JSONDecodeError):
            pass
    hp = os.path.join(out_dir, "figL4_server_pool_historical.csv")
    with open(hp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_utc", "node", "gpu_id", "event", "ready_seconds", "listening", "slurm_job_id"])
        for d in sorted(hist_rows, key=lambda r: r.get("ts", "")):
            w.writerow([d.get("ts"), d.get("node"), d.get("gpu_id"), d.get("event"),
                        d.get("ready_seconds"), d.get("listening"), d.get("slurm_job_id")])
    return dict((f"{k[0]}/gpu{k[1]}", dict(v)) for k, v in boots.items()), len(hist_rows)


README = """# Paper figure data bundle

Pre-aggregated, portable telemetry for building two figures OFF the cluster
(matplotlib on a laptop — no CARLA, no cluster access). Generated by
`tools/figure_data_export.py` on the login node.

Provenance summary (read before plotting):
- **fig2b_*** and **figL5_outcome_*** describe the **CURRENT** run (2026-07, n=1648
  route-evals). Authoritative.
- **figL5_exceptions_from_logs.csv** is a supplementary Python-exception tally from
  the current worker logs.
- **figL4_server_boots_current.csv** is the current run, but the worker logs carry
  no per-boot timestamps or `ready_seconds`, so it is boot/health COUNTS only.
- **figL4_server_pool_historical.csv** is the RICH timestamped boot/ready/crash
  series, but from the **earlier Dec-2025 run** (the only run with `carla_pool.jsonl`
  telemetry). Use it for the ready-time distribution / stability-over-time shape, and
  LABEL it as the earlier run — or cut Fig. 7 and let §7.3 prose carry it (tracker
  item 14 explicitly offers this).

## Files

### Fig 2b — throughput vs effective active-GPU count (tracker item 2b, Fig. 4 companion)
- `fig2b_hourly.csv` — one row per UTC hour: `jobs_ended, route_evals,
  active_gpu_mean, active_gpu_max, nodes_active_max`. Concurrency is a 1-minute-
  resolution sweep over terminal-job [start,end] intervals (a job occupies one GPU),
  averaged per hour — this IS the "effective active GPU count" (it drops when GPUs
  are down/parked, without needing park markers).
- `fig2b_job_intervals.csv` — raw per-job intervals (`start_utc,end_utc,duration_s,
  node,gpu,status,route_evals`) so you can recompute concurrency any other way.
- **Suggested plot:** scatter `active_gpu_mean` (x) vs `route_evals` (y), one point
  per hour; optionally color by `nodes_active_max`. A ~linear cloud = throughput
  scales with effective GPUs. Overlay `jobs_ended` as a second panel if useful.

### Fig L5 — failure taxonomy (tracker item 14)
- `figL5_outcome_taxonomy.csv` — leaderboard per-route outcome (`Completed`,
  `Failed - Agent timed out`, `... deviated`, `... got blocked`, `... Simulation
  crashed`) with counts and % of the 1648 route-evals. **Primary taxonomy.**
- `figL5_outcome_by_agent.csv` — outcome x agent matrix (stacked-bar ready).
- `figL5_exceptions_from_logs.csv` — Python exception classes seen in worker logs.
- **Suggested plot:** horizontal bar of `figL5_outcome_taxonomy`, or a per-agent
  stacked bar from the matrix (shows CILRS's timeout-heavy profile vs the rest).

### Fig L4 — server stability (tracker item 14)
- `figL4_server_boots_current.csv` — current run: per (node,gpu) relaunch triggers /
  launch attempts / healthy count.
- `figL4_server_pool_historical.csv` — Dec-2025 run: timestamped boot/ready events
  with `ready_seconds` (~120 s boot).
- **Suggested plot:** ready_seconds histogram (historical), or boots-per-GPU bar
  (current). See provenance caveat above re: which run.

### Reference
- `per_route_results.csv` — the full n=1648 harvest (one row per route-eval) this
  bundle was aggregated from, in case you want a different cut.

## Building
`build_figures.py` (matplotlib) renders draft PNGs from these CSVs:
    pip install matplotlib
    python3 build_figures.py            # writes *.png next to the CSVs
It is a STARTING POINT — restyle to match the paper's other figures.
"""

BUILD = '''#!/usr/bin/env python3
"""Draft figure builder — run OFF-cluster. Needs matplotlib only.
Reads the CSVs in this directory, writes draft PNGs alongside them."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(name):
    with open(os.path.join(HERE, name)) as f:
        return list(csv.DictReader(f))

# --- Fig 2b: throughput vs effective GPUs ---
h = rd("fig2b_hourly.csv")
x = [float(r["active_gpu_mean"]) for r in h]
y = [int(r["route_evals"]) for r in h]
plt.figure(figsize=(5, 4))
plt.scatter(x, y, s=28, alpha=0.75, edgecolor="k", linewidth=0.4)
plt.xlabel("Effective active GPU count (mean concurrent jobs / hour)")
plt.ylabel("Route-evals completed / hour")
plt.title("Throughput vs effective GPU count")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig2b_throughput_vs_gpus.png"), dpi=160)
plt.close()

# --- Fig L5: outcome taxonomy ---
t = rd("figL5_outcome_taxonomy.csv")
labels = [r["outcome"].replace("Failed - Agent ", "").replace("Failed - ", "") for r in t]
vals = [int(r["count"]) for r in t]
plt.figure(figsize=(6, 3.2))
plt.barh(range(len(vals)), vals, color="#c44")
plt.yticks(range(len(vals)), labels); plt.gca().invert_yaxis()
plt.xlabel("Route-evals"); plt.title("Per-route outcome taxonomy (n=%d)" % sum(vals))
plt.tight_layout(); plt.savefig(os.path.join(HERE, "figL5_outcome_taxonomy.png"), dpi=160)
plt.close()

# --- Fig L5b: per-agent stacked ---
m = rd("figL5_outcome_by_agent.csv")
cols = [c for c in m[0].keys() if c not in ("agent", "total")]
agents = [r["agent"] for r in m]
plt.figure(figsize=(6.5, 3.6))
bottom = [0] * len(agents)
for c in cols:
    v = [int(r[c]) for r in m]
    plt.bar(agents, v, bottom=bottom, label=c.replace("Failed - Agent ", "").replace("Failed - ", ""))
    bottom = [b + x for b, x in zip(bottom, v)]
plt.ylabel("Route-evals"); plt.legend(fontsize=7, loc="upper right")
plt.title("Outcome by agent"); plt.tight_layout()
plt.savefig(os.path.join(HERE, "figL5_outcome_by_agent.png"), dpi=160)
plt.close()

# --- Fig L4: historical ready_seconds histogram (if present) ---
try:
    p = rd("figL4_server_pool_historical.csv")
    rs = [float(r["ready_seconds"]) for r in p if r.get("ready_seconds") not in (None, "", "None")]
    if rs:
        plt.figure(figsize=(5, 3.4))
        plt.hist(rs, bins=20, color="#48a")
        plt.xlabel("CARLA boot ready_seconds"); plt.ylabel("boots")
        plt.title("Server boot time (Dec-2025 run)"); plt.tight_layout()
        plt.savefig(os.path.join(HERE, "figL4_server_ready_seconds.png"), dpi=160)
        plt.close()
except FileNotFoundError:
    pass
print("wrote draft PNGs to", HERE)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_figures_data")
    ap.add_argument("--state", default="collection_state")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--harvest", default="paper_artifacts/per_route_results.csv")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log_glob = os.path.join(a.logs, "worker_*.log")

    jobs = load_terminal_jobs(a.state)
    evals_per_job, outcome, by_agent, rows = load_harvest(a.harvest)

    n_hours, n_jobs = build_2b(jobs, evals_per_job, a.out)
    total_evals, exc = build_L5(outcome, by_agent, log_glob, a.out)
    boots, n_hist = build_L4(log_glob, a.state, a.out)

    # copy the harvest for reference
    if os.path.exists(a.harvest):
        with open(a.harvest) as src, open(os.path.join(a.out, "per_route_results.csv"), "w") as dst:
            dst.write(src.read())

    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(README)
    with open(os.path.join(a.out, "build_figures.py"), "w") as f:
        f.write(BUILD)

    print("=== figure data bundle written to %s/ ===" % a.out)
    print("Fig 2b : %d hourly rows, %d terminal jobs, %d route-evals attributed"
          % (n_hours, n_jobs, sum(evals_per_job.values())))
    print("Fig L5 : %d route-evals across %d outcome classes; log exceptions=%s"
          % (total_evals, len(outcome), exc))
    print("Fig L4 : current boots per gpu=%s" % boots)
    print("Fig L4 : %d historical carla_pool events" % n_hist)


if __name__ == "__main__":
    main()
