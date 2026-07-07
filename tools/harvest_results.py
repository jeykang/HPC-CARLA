#!/usr/bin/env python3
"""Per-route results harvester for the CARLA continuous-collection project.

WHY THIS EXISTS
---------------
Each collection *job* runs a whole route FILE — a suite of 9-442 short
point-to-point routes — through the leaderboard's ``RouteIndexer``, which
checkpoints **per route** into a ``results.json``. The job queue only records a
*file-level* aggregate score, and only when the whole file finishes. On the slow
A100 most files never finish (server segfault / timeout after a handful of
routes), so the queue's file-level metric is missing for the majority of work.

**But every completed route's score is already in the per-job ``results.json``.**
So the real, complete metric set is *per route* inside those checkpoint files —
including inside the checkpoints of jobs the queue marked ``failed``. This tool
harvests that per-route data. That is what makes the collected data usable for
the paper even though very few whole *files* complete.

WHERE THE DATA LIVES
--------------------
Each job writes ``save_path/results.json`` where (mirrored verbatim from
``manage_continuous.py`` ~L1299-1304 and the sibling builder ~L465-468)::

    save_path = <dataset>/<agent>/weather_<int(weather)>/map_<int(town):02d>/<Path(route).stem>
    checkpoint = save_path/results.json

We enumerate the expected paths from the job records (NOT by walking the huge
``dataset/`` tree). Finished jobs are read from two state files:

  * ``job_queue.json``       — the live queue (``completed`` + ``pending``).
  * ``completed_jobs.json``  — the archive of FINISHED jobs, and the ONLY place
    that records ``failed`` jobs. A failed job is reset to ``pending`` in the
    queue for a future retry, so its per-route data is invisible unless we also
    read ``completed_jobs.json`` (sibling of the queue file). This is the same
    two-file union that ``tools/difficulty_validation.py::load_completed`` uses.

RESULTS.JSON FORMAT (confirmed against real files on 2026-07-07)
---------------------------------------------------------------
Inspected e.g. ``dataset/tcp/weather_20/map_02/routes_town02_tiny/results.json``
(a queue-``failed`` job: progress [5,129] yet 5 real per-route records)::

    {"_checkpoint": {"progress": [done,total],
                     "global_record": {...},
                     "records": [ {"index": 0,
                                   "route_id": "RouteScenario_0",
                                   "status": "Completed" | "Failed - Agent timed out" | ...,
                                   "scores": {"score_composed", "score_route", "score_penalty"},
                                   "infractions": {"collisions_vehicle": [...],
                                                   "collisions_pedestrian": [...],
                                                   "collisions_layout": [...],
                                                   "red_light": [...], "route_dev": [...],
                                                   "outside_route_lanes": [...],
                                                   "stop_infraction": [...], "route_timeout": [...],
                                                   "vehicle_blocked": [...]},   # each a LIST of events
                                   "meta": {"duration_game","duration_system","route_length"}},
                                 ... ]},
     "entry_status": ..., "eligible": ..., "labels": [...], "sensors": [...], "values": [...]}

Robust: stdlib-only (pandas/numpy optional, not required); skips
missing/partial/corrupt ``results.json`` with counters; never crashes on empty
input.

Usage:
  python3 tools/harvest_results.py [--queue collection_state/job_queue.json]
                                   [--out-dir paper_artifacts]
                                   [--dataset-dir dataset]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Stdlib-only by design: the harvest is small and the CSV writer is
# deterministic, so numpy/pandas are neither imported nor required.

# Sibling module: weather-preset -> physical-axis decomposition (illum/precip/fog),
# so illumination can be analysed apart from precipitation/fog downstream.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_axes  # noqa: E402


# =============================================================================
# Path construction — REPLICATED from manage_continuous.py so a harvested path
# is byte-for-byte the checkpoint the worker wrote.
#   _label_weather / _label_map : manage_continuous.py ~ L435-441 (and L1286-1294)
#   save_path / results.json    : manage_continuous.py ~ L465-468, L1300-1304
# =============================================================================
def _label_weather(idx):
    try:
        return f"weather_{int(idx)}"
    except Exception:
        return f"weather_{idx}"


def _label_map(tn):
    try:
        return f"map_{int(tn):02d}"
    except Exception:
        return f"map_{tn or 'unknown'}"


def results_path_for(dataset_dir, job):
    """Expected results.json path for a job dict (agent/weather/town/route)."""
    route_stem = Path(str(job.get("route", ""))).stem
    save_path = os.path.join(
        str(dataset_dir),
        str(job.get("agent", "")),
        _label_weather(job.get("weather", 0)),
        _label_map(job.get("town", "")),
        route_stem,
    )
    return os.path.join(save_path, "results.json")


# =============================================================================
# Job loading — union job_queue.json + sibling completed_jobs.json, keep only
# FINISHED jobs (status in {completed, failed}). The completed_jobs.json record
# is authoritative for the finished status (a failed job reads as 'pending' in
# the queue), so it wins on a per-id conflict.
# =============================================================================
FINISHED = ("completed", "failed")


def _load_jobs(path):
    """Return the list of job dicts from a state file, or [] on any problem."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    return jobs if isinstance(jobs, list) else []


def load_finished_jobs(queue_path):
    """Union of finished (completed|failed) jobs across the queue and its
    sibling completed_jobs.json, deduped by id.

    completed_jobs.json is authoritative (it is the only source of 'failed'),
    so its record overrides a queue record for the same id.
    """
    queue_path = Path(queue_path)
    completed_path = queue_path.parent / "completed_jobs.json"

    by_id = {}
    # completed_jobs.json first so its finished status is authoritative...
    for src in (completed_path, queue_path):
        for job in _load_jobs(src):
            if job.get("status") not in FINISHED:
                continue
            jid = job.get("id")
            key = jid if jid is not None else id(job)
            # ...but let a later 'completed' upgrade never be shadowed: only
            # fill if absent, so the authoritative completed_jobs.json wins.
            by_id.setdefault(key, job)
    return list(by_id.values())


# =============================================================================
# Record extraction
# =============================================================================
# Curated "key" infractions surfaced as their own columns. Each results.json
# infraction value is a LIST of event descriptors, so the count is its length.
INFRACTION_COLUMNS = [
    ("collisions_vehicle", "infr_collisions_vehicle"),
    ("collisions_pedestrian", "infr_collisions_pedestrian"),
    ("collisions_layout", "infr_collisions_layout"),
    ("red_light", "infr_red_light"),
    ("outside_route_lanes", "infr_outside_route_lanes"),
    ("route_dev", "infr_route_dev"),
]

# Physical weather axes (tools/weather_axes.py) — let illumination be analysed
# separately from precipitation/fog, which the single 0-20 ordinal fuses together.
WEATHER_AXIS_COLUMNS = ["weather_name", "time_of_day", "sun_altitude",
                        "illum_dark", "precip", "road_water", "cloud", "fog"]

FIELDS = [
    "agent", "route_file", "town", "weather",
] + WEATHER_AXIS_COLUMNS + [
    "route_id",
    "score_composed", "score_route", "score_penalty", "status",
] + [col for _, col in INFRACTION_COLUMNS] + [
    "route_length", "job_status", "job_id",
]


def _infraction_count(val):
    """Count events for one infraction key (value is normally a list)."""
    if isinstance(val, list):
        return len(val)
    if isinstance(val, (int, float)):
        return int(val)
    return 0


def _num(x):
    """Coerce to float, else None (so empty/garbage never crashes stats)."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _weather_cols(weather):
    """Physical-axis decomposition columns for a weather preset index. Blank
    cells for an out-of-range / non-int index (never crashes)."""
    try:
        return weather_axes.axis_row(int(weather))
    except Exception:
        return {c: "" for c in WEATHER_AXIS_COLUMNS}


def rows_from_checkpoint(data, job):
    """Yield one flat row dict per record in a parsed results.json.

    Returns (rows, n_records). ``rows`` is empty if the file has no records.
    """
    cp = data.get("_checkpoint") if isinstance(data, dict) else None
    if not isinstance(cp, dict):
        return [], 0
    records = cp.get("records")
    if not isinstance(records, list):
        return [], 0

    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        scores = rec.get("scores") or {}
        infr = rec.get("infractions") or {}
        meta = rec.get("meta") or {}
        row = {
            "agent": job.get("agent", ""),
            "route_file": job.get("route", ""),
            "town": job.get("town", ""),
            "weather": job.get("weather", ""),
            "route_id": rec.get("route_id"),
            "score_composed": _num(scores.get("score_composed")),
            "score_route": _num(scores.get("score_route")),
            "score_penalty": _num(scores.get("score_penalty")),
            "status": rec.get("status"),
            "route_length": _num(meta.get("route_length")),
            "job_status": job.get("status"),
            "job_id": job.get("id"),
        }
        for key, col in INFRACTION_COLUMNS:
            row[col] = _infraction_count(infr.get(key))
        rows.append(row)
    return rows, len(records)


# =============================================================================
# Harvest driver
# =============================================================================
class Counters:
    def __init__(self):
        self.jobs_finished = 0        # finished jobs enumerated
        self.paths_expected = 0       # unique results.json paths
        self.files_found = 0
        self.files_missing = 0        # path does not exist
        self.files_corrupt = 0        # unreadable / bad JSON / no _checkpoint
        self.files_empty = 0          # valid but zero records
        self.records_seen = 0         # records before dedupe
        self.dupes_dropped = 0        # records dropped by resume dedupe


def harvest(jobs, dataset_dir):
    """Harvest per-route rows from every finished job's results.json.

    Dedupe key: (agent, weather, town, route_file, route_id). When the same
    route-eval recurs across resumes/retries (same condition -> same file, or a
    retry that overwrote the file), keep the record from the FURTHEST-PROGRESSED
    file (largest ``_checkpoint.progress[0]``), tie-broken by more records.

    Returns (rows, counters).
    """
    counters = Counters()
    counters.jobs_finished = len(jobs)

    # Parse each unique path once (several jobs can map to one condition/path).
    parse_cache = {}   # abs path -> (rows, progress_done, n_records) | None(corrupt/missing)

    by_key = {}        # dedupe key -> {"progress": int, "nrec": int, "row": row}

    for job in jobs:
        path = results_path_for(dataset_dir, job)
        abspath = os.path.abspath(path)

        if abspath not in parse_cache:
            counters.paths_expected += 1
            parsed = _parse_file(abspath, counters)
            parse_cache[abspath] = parsed
        parsed = parse_cache[abspath]
        if parsed is None:
            continue  # missing/corrupt already counted

        # Re-extract rows with THIS job's status/id (the file content is shared,
        # but job_status/job_id are per-job — matters when a path is referenced
        # by more than one finished job).
        data_rows, progress_done, n_rec = parsed
        for base in data_rows:
            counters.records_seen += 1
            row = dict(base)
            row["agent"] = job.get("agent", row["agent"])
            row["route_file"] = job.get("route", row["route_file"])
            row["town"] = job.get("town", row["town"])
            row["weather"] = job.get("weather", row["weather"])
            row["job_status"] = job.get("status")
            row["job_id"] = job.get("id")
            row.update(_weather_cols(row["weather"]))

            key = (row["agent"], str(row["weather"]), str(row["town"]),
                   row["route_file"], row["route_id"])
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = {"progress": progress_done, "nrec": n_rec, "row": row}
            else:
                # Keep the furthest-progressed file (more done routes = more
                # authoritative); tie-break on record count.
                better = (progress_done > prev["progress"] or
                          (progress_done == prev["progress"] and n_rec > prev["nrec"]))
                counters.dupes_dropped += 1
                if better:
                    by_key[key] = {"progress": progress_done, "nrec": n_rec, "row": row}

    rows = [v["row"] for v in by_key.values()]
    # Stable, human-friendly ordering.
    rows.sort(key=lambda r: (str(r["agent"]), str(r["route_file"]),
                             str(r["town"]), str(r["weather"]),
                             _route_id_sort(r["route_id"])))
    return rows, counters


def _route_id_sort(rid):
    """Sort RouteScenario_<n> numerically, falling back to the raw string."""
    s = str(rid)
    if "_" in s:
        tail = s.rsplit("_", 1)[1]
        if tail.isdigit():
            return (0, int(tail))
    return (1, s)


def _parse_file(abspath, counters):
    """Parse one results.json. Returns (rows, progress_done, n_records) or None."""
    if not os.path.isfile(abspath):
        counters.files_missing += 1
        return None
    try:
        with open(abspath) as f:
            data = json.load(f)
    except Exception:
        counters.files_corrupt += 1
        return None
    if not isinstance(data, dict) or not isinstance(data.get("_checkpoint"), dict):
        counters.files_corrupt += 1
        return None

    # We build rows from a synthetic empty job here (job fields are re-applied by
    # the caller per referencing job), so pass a stub.
    rows, n_rec = rows_from_checkpoint(data, {})
    counters.files_found += 1
    if n_rec == 0:
        counters.files_empty += 1

    progress = data["_checkpoint"].get("progress")
    if isinstance(progress, (list, tuple)) and progress:
        try:
            progress_done = int(progress[0])
        except Exception:
            progress_done = n_rec
    else:
        progress_done = n_rec
    return rows, progress_done, n_rec


# =============================================================================
# Output
# =============================================================================
def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "per_route_results.csv")
    json_path = os.path.join(out_dir, "per_route_results.json")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})

    with open(json_path, "w") as f:
        json.dump([{k: r.get(k) for k in FIELDS} for r in rows], f, indent=2)

    return csv_path, json_path


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else None


def _fmt(v):
    return "  n/a" if v is None else f"{v:6.2f}"


def print_summary(rows, counters, out_paths):
    total = len(rows)
    from_failed = sum(1 for r in rows if r.get("job_status") == "failed")
    from_completed = sum(1 for r in rows if r.get("job_status") == "completed")

    print("=" * 74)
    print("PER-ROUTE RESULTS HARVEST")
    print("=" * 74)
    print(f"Finished jobs enumerated (completed+failed) : {counters.jobs_finished}")
    print(f"Unique results.json paths expected          : {counters.paths_expected}")
    print(f"  found                                     : {counters.files_found}")
    print(f"  missing (never ran / no checkpoint)       : {counters.files_missing}")
    print(f"  corrupt / no _checkpoint                  : {counters.files_corrupt}")
    print(f"  empty (0 records)                         : {counters.files_empty}")
    print(f"Records seen (pre-dedupe)                   : {counters.records_seen}")
    print(f"Duplicate route-evals dropped (resumes)     : {counters.dupes_dropped}")
    print("-" * 74)
    print(f"TOTAL ROUTE-EVALS HARVESTED                 : {total}")
    print("=" * 74)

    if total == 0:
        print("No route-evals harvested. (No finished jobs, or no readable "
              "results.json yet.)")
        print(f"\nWrote {out_paths[0]}\nWrote {out_paths[1]}")
        return

    # ---- per-agent breakdown ----
    agents = sorted(set(str(r.get("agent", "")) for r in rows))
    print("\nPer-agent (route-eval count | mean score_composed | mean route completion):")
    print(f"  {'agent':14s} {'n':>5s}  {'score_composed':>14s}  {'route_completion':>16s}")
    for a in agents:
        sub = [r for r in rows if str(r.get("agent", "")) == a]
        msc = _mean([r.get("score_composed") for r in sub])
        mrc = _mean([r.get("score_route") for r in sub])
        print(f"  {a:14s} {len(sub):5d}  {_fmt(msc):>14s}  {_fmt(mrc):>16s}")

    # ---- headline: how much did we recover from queue-'failed' jobs? ----
    print("\n" + "-" * 74)
    print("RECOVERY — route-evals by originating queue-job status:")
    print(f"  from queue-'completed' jobs : {from_completed}")
    print(f"  from queue-'failed'    jobs : {from_failed}   <-- extra data the "
          f"harvester recovers")
    if total:
        pct = 100.0 * from_failed / total
        print(f"  failed-job share of all harvested route-evals : {pct:.1f}%")
    # Context: the queue records only ONE file-level aggregate per WHOLE file,
    # and only when it finishes — so the per-route harvest yields many metrics
    # where the queue would have yielded at most a handful.
    print("-" * 74)

    # ---- illumination coverage (the axis hardest-first tends to collapse) ----
    tod = {}
    for r in rows:
        t = r.get("time_of_day") or "?"
        tod[t] = tod.get(t, 0) + 1
    print("\nIllumination coverage (time-of-day of harvested route-evals):")
    for t in ("noon", "sunset", "night", "?"):
        if t in tod:
            print(f"  {t:8s}: {tod[t]}")
    if len([t for t in tod if t != "?"]) <= 1:
        print("  -> a SINGLE illumination bin: the illumination axis is UNSAMPLED, so"
              "\n     its per-model sensitivity is unidentifiable (see tools/sensitivity_matrix.py).")
    print("-" * 74)

    print(f"\nWrote {out_paths[0]}")
    print(f"Wrote {out_paths[1]}")


# =============================================================================
def main(argv):
    ap = argparse.ArgumentParser(
        description="Harvest per-route leaderboard scores from every finished "
                    "job's results.json checkpoint (completed AND failed jobs).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="collection_state/job_queue.json",
                    help="job queue JSON (default: collection_state/job_queue.json). "
                         "Its sibling completed_jobs.json is read automatically "
                         "for 'failed' jobs.")
    ap.add_argument("--out-dir", default="paper_artifacts",
                    help="output dir for per_route_results.csv/.json (default: paper_artifacts)")
    ap.add_argument("--dataset-dir", default="dataset",
                    help="dataset root where results.json live (default: dataset)")
    args = ap.parse_args(argv)

    jobs = load_finished_jobs(args.queue)
    rows, counters = harvest(jobs, args.dataset_dir)
    out_paths = write_outputs(rows, args.out_dir)
    print_summary(rows, counters, out_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
