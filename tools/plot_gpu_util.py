#!/usr/bin/env python3
"""Plot GPU utilization over time from collection_state metrics GPU JSONL files.

Usage:
  python3 tools/plot_gpu_util.py --input <path/to/gpu.jsonl> --out plots/gpu_util.png

The script streams the input file (line-by-line), detects timestamps and per-GPU
utilization fields using heuristics, and produces a PNG and CSV with the timeseries.
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
import math

plt = None



def parse_ts(obj):
    # common timestamp keys
    for k in ("ts", "timestamp", "time", "t"):
        if k in obj:
            s = obj[k]
            if isinstance(s, (int, float)):
                try:
                    return datetime.fromtimestamp(s)
                except Exception:
                    continue
            if isinstance(s, str):
                try:
                    # accept ending Z
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    return datetime.fromisoformat(s)
                except Exception:
                    # fallback: try parse basic float
                    try:
                        return datetime.fromtimestamp(float(s))
                    except Exception:
                        continue
    return None


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def extract_from_gpu_entry(entry):
    # entry is a dict; try to find an utilization value
    util_keys = [k for k in entry.keys() if "util" in k.lower() or "usage" in k.lower()]
    for k in util_keys:
        v = entry.get(k)
        if numeric(v):
            return v
        # some utilities nested like {'gpu_util': {'gpu': 10}}
        if isinstance(v, dict):
            for vv in v.values():
                if numeric(vv):
                    return vv
    # special keys
    for k in ("gpu_util", "utilization", "gpu_utilization", "util"): 
        if k in entry and numeric(entry[k]):
            return entry[k]
    return None


def walk(obj, gpu_count=None):
    """Walk JSON object to find per-gpu utils. Returns list of (index, util) pairs."""
    found = []

    if isinstance(obj, dict):
        # if dict contains 'gpus' list
        if 'gpus' in obj and isinstance(obj['gpus'], list):
            for e in obj['gpus']:
                if isinstance(e, dict):
                    idx = e.get('index')
                    util = extract_from_gpu_entry(e)
                    if util is not None:
                        if idx is None:
                            # try to find 'id' or 'gpu'
                            idx = e.get('gpu') or e.get('id')
                        found.append((int(idx) if idx is not None else None, float(util)))
            if found:
                return found

        # otherwise scan keys for per-gpu named keys like gpu_0_util
        for k, v in obj.items():
            if isinstance(v, dict):
                sub = walk(v, gpu_count=gpu_count)
                if sub:
                    found.extend(sub)
            elif isinstance(v, list):
                # maybe list of numbers representing GPUs
                if all(numeric(x) for x in v) and gpu_count and len(v) == gpu_count:
                    for i, x in enumerate(v):
                        found.append((i, float(x)))
            else:
                # key pattern gpu0_util, gpu_0_util, gpu-0-util
                import re
                m = re.match(r"gpu[_-]?(\d+)[_\-]?.*util", k.lower())
                if m and numeric(v):
                    found.append((int(m.group(1)), float(v)))

    return found


def parse_file(path, gpu_count_hint=None, max_lines=None):
    per_gpu = defaultdict(list)
    total = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if max_lines and total >= max_lines:
                break
            ln = ln.strip()
            if not ln:
                continue
            total += 1
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            ts = parse_ts(obj)
            if ts is None:
                # try to find ts inside nested
                for v in obj.values():
                    if isinstance(v, dict) and 'ts' in v:
                        ts = parse_ts(v)
                        if ts:
                            break
            entries = walk(obj, gpu_count=gpu_count_hint)
            if not entries:
                # as fallback, try top-level numeric keys
                for k, v in obj.items():
                    if numeric(v) and 0 <= v <= 100:
                        # ambiguous; assign to gpu 0
                        entries.append((0, float(v)))
            for idx, util in entries:
                if ts is None:
                    # skip if no timestamp
                    continue
                if idx is None:
                    idx = 0
                per_gpu[int(idx)].append((ts, util))
    return per_gpu


def save_csv(per_gpu, out_csv):
    import csv
    rows = []
    for g, series in per_gpu.items():
        for ts, util in series:
            rows.append((g, ts.isoformat(), util))
    rows.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['gpu', 'ts', 'util'])
        w.writerows(rows)


def plot(per_gpu, out_png, title=None, downsample=None):
    # lazy import matplotlib so script can run to produce CSV on systems
    # where matplotlib isn't installed
    global plt
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    except Exception as e:
        print('matplotlib not available; skipping plotting:', e)
        return
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.figure(figsize=(12, 6))
    for g in sorted(per_gpu.keys()):
        series = per_gpu[g]
        series.sort(key=lambda x: x[0])
        if downsample and len(series) > downsample:
            step = max(1, len(series)//downsample)
            series = series[::step]
        ts = [s[0] for s in series]
        utils = [s[1] for s in series]
        plt.plot(ts, utils, label=f'GPU {g}')
    plt.xlabel('Time')
    plt.ylabel('GPU Utilization (%)')
    if title:
        plt.title(title)
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    print('Wrote', out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=False,
                    default='collection_state/metrics/node/hpc-pr-a-pod09/gpu.jsonl',
                    help='Path to gpu.jsonl (can be large)')
    ap.add_argument('--out', '-o', required=False, default='plots/gpu_util.png')
    ap.add_argument('--csv', required=False, default='plots/gpu_util.csv')
    ap.add_argument('--max-lines', type=int, default=None, help='Max lines to parse (for quick tests)')
    ap.add_argument('--downsample', type=int, default=2000, help='Downsample points per GPU for plotting')
    args = ap.parse_args()

    path = args.input
    if not os.path.exists(path):
        print('Input file not found:', path)
        return
    print('Parsing', path)
    per_gpu = parse_file(path, gpu_count_hint=None, max_lines=args.max_lines)
    if not per_gpu:
        print('No GPU utilization data found.')
        return
    print('Found GPUs:', sorted(per_gpu.keys()))
    save_csv(per_gpu, args.csv)
    plot(per_gpu, args.out, title=os.path.basename(path), downsample=args.downsample)


if __name__ == '__main__':
    main()
