#!/usr/bin/env python3
# metrics_daemon.py
import os, sys, time, argparse
from pathlib import Path
from hpc_metrics import GPUSampler, SystemSampler, MetricsWriter, NVMLUnavailable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.environ.get("STATE_DIR") or (Path(os.environ.get("PROJECT_ROOT",".")).resolve()/"collection_state"))
    ap.add_argument("--interval", type=float, default=float(os.environ.get("METRICS_INTERVAL", "2.0")))
    ap.add_argument("--burst", type=int, default=1, help="number of gpu polls per tick (kept for future smoothing)")
    args = ap.parse_args()

    Path(args.state_dir).mkdir(parents=True, exist_ok=True)
    writer = MetricsWriter(Path(args.state_dir))
    sys_s = SystemSampler()
    try:
        gpu_s = GPUSampler()
    except NVMLUnavailable as e:
        print(f"[metrics] GPU metrics unavailable: {e}", file=sys.stderr)
        gpu_s = None

    print(f"[metrics] daemon up: state={args.state_dir} interval={args.interval}s")
    while True:
        try:
            if gpu_s:
                g = gpu_s.sample()
                writer.write_gpu_samples(g)
            s = sys_s.sample()
            writer.write_system_sample(s)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[metrics] warn: {e}", file=sys.stderr)
        time.sleep(max(args.interval, 0.5))

if __name__ == "__main__":
    main()
