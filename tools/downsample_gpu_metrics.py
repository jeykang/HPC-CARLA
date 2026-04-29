#!/usr/bin/env python3
"""Downsample GPU metrics JSONL to a small CSV for local plotting.

Reads the large gpu.jsonl, bins data into 1-minute windows per GPU,
and outputs a small CSV with columns: gpu, ts, util_sm, power_W, mem_used_MiB

Usage:
  python3 tools/downsample_gpu_metrics.py \
      -i collection_state/metrics/node/hpc-pr-a-pod09/gpu.jsonl \
      -o plots/gpu_util_small.csv
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta


def parse_ts(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def round_to_minute(dt):
    return dt.replace(second=0, microsecond=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='Path to gpu.jsonl')
    ap.add_argument('-o', '--output', default='plots/gpu_util_small.csv')
    ap.add_argument('--bin-seconds', type=int, default=60, help='Bin size in seconds')
    args = ap.parse_args()

    bin_sec = args.bin_seconds
    # Accumulate per (gpu, bin_start) -> list of (util_sm, power_W, mem_used_MiB)
    buckets = defaultdict(list)

    print(f'Reading {args.input} ...')
    count = 0
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts_str = obj.get('ts')
            gpu_idx = obj.get('index')
            util_sm = obj.get('util_sm')
            if ts_str is None or gpu_idx is None or util_sm is None:
                continue
            try:
                ts = parse_ts(ts_str)
            except Exception:
                continue
            # Round timestamp to nearest bin
            bin_start = ts.replace(microsecond=0)
            bin_start = bin_start - timedelta(seconds=bin_start.second % bin_sec)

            power = obj.get('power_W')
            mem = obj.get('mem_used_MiB')
            buckets[(gpu_idx, bin_start)].append((util_sm, power, mem))
            count += 1
            if count % 500000 == 0:
                print(f'  processed {count} lines ...')

    print(f'Total lines processed: {count}')
    print(f'Unique (gpu, bin) buckets: {len(buckets)}')

    # Aggregate each bucket (mean)
    rows = []
    for (gpu, bin_start), vals in buckets.items():
        util_mean = sum(v[0] for v in vals) / len(vals)
        power_vals = [v[1] for v in vals if v[1] is not None]
        power_mean = sum(power_vals) / len(power_vals) if power_vals else None
        mem_vals = [v[2] for v in vals if v[2] is not None]
        mem_mean = sum(mem_vals) / len(mem_vals) if mem_vals else None
        rows.append((gpu, bin_start.isoformat(), util_mean, power_mean, mem_mean))

    rows.sort(key=lambda r: (r[0], r[1]))

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('gpu,ts,util_sm,power_W,mem_used_MiB\n')
        for r in rows:
            power_str = f'{r[3]:.2f}' if r[3] is not None else ''
            mem_str = f'{r[4]:.0f}' if r[4] is not None else ''
            f.write(f'{r[0]},{r[1]},{r[2]:.2f},{power_str},{mem_str}\n')

    print(f'Wrote {len(rows)} rows to {args.output}')


if __name__ == '__main__':
    main()
