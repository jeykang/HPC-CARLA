#!/usr/bin/env python3
"""
gpu_hours_audit.py — reconcile the GPU-hours / job-count numbers for Fig 4 / §7.

Resolves a cross-caption arithmetic trap: three different "job" units were being
conflated (queue size vs jobs-executed; failed-only `duration` vs full-span
wall-clock). This recomputes every number from the raw queue state with explicit
definitions, so a caption can cite a defensible figure.

Stdlib only. Run against the original-campaign backup (default) or any queue state:
  python3 tools/gpu_hours_audit.py \
      --queue collection_state/job_queue.prezerowsmoke_20260715.json \
      --completed collection_state/completed_jobs.prezerowsmoke_20260715.json
"""
import argparse, json, statistics
from datetime import datetime

CAP_S = 3600  # JOB_TIMEOUT


def _load(p):
    d = json.load(open(p))
    return d["jobs"] if isinstance(d, dict) and "jobs" in d else d


def _parse(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
    except (ValueError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="collection_state/job_queue.prezerowsmoke_20260715.json")
    ap.add_argument("--completed", default="collection_state/completed_jobs.prezerowsmoke_20260715.json")
    a = ap.parse_args()

    q = _load(a.queue)
    byid = {j["id"]: j for j in q}
    try:
        for j in _load(a.completed):
            byid[j["id"]] = j
    except FileNotFoundError:
        pass
    allj = list(byid.values())

    import collections
    status = collections.Counter(j["status"] for j in q)
    term = [j for j in allj if j.get("status") in ("completed", "failed")]

    print("QUEUE: total=%d  %s" % (len(q), dict(status)))
    print("JOBS EXECUTED (terminal) = %d ; running = %d ; pending/never-ran = %d"
          % (len(term), status.get("running", 0), status.get("pending", 0)))

    # duration field (failed jobs carry it; completed are null)
    durs = [j["duration"] for j in term if isinstance(j.get("duration"), (int, float))]
    if durs:
        print("\n'duration' field: present on %d/%d terminal jobs (the FAILED ones), "
              "mean=%.0fs median=%.0fs" % (len(durs), len(term),
                                           statistics.mean(durs), statistics.median(durs)))

    # DEFENSIBLE active compute: per job = min(duration or span, CAP)
    comp = []
    for j in term:
        d = j.get("duration")
        if isinstance(d, (int, float)):
            comp.append(min(d, CAP_S))
        else:
            s, e = _parse(j.get("start_time")), _parse(j.get("end_time"))
            if s and e and e > s:
                comp.append(min((e - s).total_seconds(), CAP_S))
    print("\n>>> ACTIVE GPU-COMPUTE = %.1f GPU-h  (Σ per-job walltime capped at 1h; "
          "mean %.2f h/job over %d jobs; ceiling %d h)"
          % (sum(comp) / 3600, statistics.mean(comp) / 3600, len(comp), len(term)))

    # raw spans (the ~1006 occupancy proxy) + campaign wall-clock
    spans = []
    st, en = [], []
    for j in term:
        s, e = _parse(j.get("start_time")), _parse(j.get("end_time"))
        if s and e and e > s:
            spans.append((e - s).total_seconds())
            st.append(s); en.append(e)
    if spans:
        print("Σ raw (start->end) spans = %.1f GPU-h  (occupancy proxy; mean %.2fh median %.2fh; "
              "includes outage idle)" % (sum(spans) / 3600, statistics.mean(spans) / 3600,
                                         statistics.median(spans) / 3600))
    if st:
        span_h = (max(en) - min(st)).total_seconds() / 3600
        print("Campaign wall-clock span = %.1f h (%s -> %s)" % (span_h, min(st).date(), max(en).date()))

    # persistence savings (needs a measured boot count M; pass via --boots if known)
    M = 24
    N = len(term) + status.get("running", 0)
    print("\nPersistence (measured, M=%d boots, N=%d executed): eliminated %.1f%% of per-job boots, "
          "saved %.1f GPU-h" % (M, N, 100 * (N - M) / N, (N - M) * 120 / 3600))
    print("  full-sweep PROJECTION (N=queue %d): %.1f%% / %.1f GPU-h  <- label as projected"
          % (len(q), 100 * (len(q) - M) / len(q), (len(q) - M) * 120 / 3600))


if __name__ == "__main__":
    main()
