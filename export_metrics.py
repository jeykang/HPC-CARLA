#!/usr/bin/env python3
# export_metrics.py
import os, json, argparse, csv
from pathlib import Path

def load_jsonl(p: Path):
    for line in p.open():
        line=line.strip()
        if not line: continue
        try: yield json.loads(line)
        except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.environ.get("STATE_DIR") or (Path(os.environ.get("PROJECT_ROOT",".")).resolve()/"collection_state"))
    ap.add_argument("--out-prefix", default="metrics_export")
    args = ap.parse_args()

    node_dir = Path(args.state_dir)/"metrics"/"node"
    out_gpu = Path(f"{args.out_prefix}_gpu.csv")
    out_sys = Path(f"{args.out_prefix}_system.csv")

    # GPU CSV
    with out_gpu.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts","node","gpu","job_id","sm","mem_used_MiB","mem_total_MiB","power_W","tempC","pstate","clock_sm","clock_mem"])
        for nd in sorted(node_dir.glob("*")):
            gj = nd/"gpu.jsonl"
            if not gj.exists(): continue
            for r in load_jsonl(gj):
                w.writerow([r.get("ts"), r.get("node"), r.get("index"), r.get("job_id"),
                            r.get("util_sm"), r.get("mem_used_MiB"), r.get("mem_total_MiB"),
                            r.get("power_W"), r.get("tempC"), r.get("pstate"),
                            r.get("clock_sm_MHz"), r.get("clock_mem_MHz")])

    # System CSV
    with out_sys.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts","node","cpu_pct","load_1","ram_used_MiB","ram_total_MiB","swap_used_MiB","swap_total_MiB","net_bytes_sent","net_bytes_recv","disk_read_bytes","disk_write_bytes"])
        for nd in sorted(node_dir.glob("*")):
            sj = nd/"system.jsonl"
            if not sj.exists(): continue
            for r in load_jsonl(sj):
                w.writerow([r.get("ts"), r.get("node"), r.get("cpu_pct"), r.get("load_1"),
                            r.get("ram_used_MiB"), r.get("ram_total_MiB"),
                            r.get("swap_used_MiB"), r.get("swap_total_MiB"),
                            r.get("net_bytes_sent"), r.get("net_bytes_recv"),
                            r.get("disk_read_bytes"), r.get("disk_write_bytes")])

    print(f"Wrote:\n  {out_gpu}\n  {out_sys}")

if __name__ == "__main__":
    main()
