#!/usr/bin/env python3
"""Aggregate CARLA 'System Time / Game Time' ratios from worker logs.

The leaderboard prints a result table per route containing:
    | Ratio (System Time / Game Time) | 0.082 |
This ratio is the sim speed: 1.0 == real-time, higher == faster. It's the
metric that determines collection throughput, and it's per-route (so robust to
the cluster's stochastic job failures — only completed routes contribute).

Usage:
    python3 tools/parse_sim_ratio.py LOG [LOG ...]
    python3 tools/parse_sim_ratio.py 'bench_legacy/worker_*.log'
    python3 tools/parse_sim_ratio.py 'bench_colocate/worker_*.log'

Reports per-file and overall n / mean / median / p10 / p90 / min / max.
Compare the OVERALL mean between two runs to get the co-location speedup.
"""
import sys
import re
import glob

# Label, then (across the box-drawing separator) the first float.
RATIO_RE = re.compile(r'Ratio \(System Time / Game Time\).*?([0-9]+\.[0-9]+)')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKH]')


def parse(path):
    vals = []
    try:
        with open(path, errors='ignore') as f:
            for line in f:
                m = RATIO_RE.search(ANSI_RE.sub('', line))
                if m:
                    try:
                        vals.append(float(m.group(1)))
                    except ValueError:
                        pass
    except (FileNotFoundError, IsADirectoryError):
        pass
    return vals


def stats(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return None
    return dict(n=n, mean=sum(vals) / n, median=vals[n // 2],
                p10=vals[int(n * 0.1)], p90=vals[min(n - 1, int(n * 0.9))],
                min=vals[0], max=vals[-1])


def fmt(s):
    if not s:
        return 'no data'
    return (f"n={s['n']:4d}  mean={s['mean']:.3f}  median={s['median']:.3f}  "
            f"p10={s['p10']:.3f}  p90={s['p90']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}")


def main(argv):
    files = []
    for a in argv:
        files.extend(sorted(glob.glob(a)) or [a])
    if not files:
        print(__doc__)
        return 1
    allv = []
    for f in files:
        v = parse(f)
        allv.extend(v)
        print(f"{f}: {fmt(stats(v))}")
    print("-" * 70)
    print(f"OVERALL: {fmt(stats(allv))}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
