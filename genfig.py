#!/usr/bin/env python3
# file: generate_all_figures.py
# Faster, resumable figure generator with cached coverage counting and progress logs.
# Generates Figs 1–9, Table 2, and the 3D topology map.
#
# Quick tips:
#   - To skip the heavy coverage scan: add --skip-coverage
#   - To precompute coverage (parallel + cached): run with --only 8
#   - To resume: re-run; completed figures are skipped unless --force
#
# Requires: matplotlib pandas numpy pillow (Graphviz binary optional for DOT->PNG)
# No extra deps (uses stdlib concurrency).

import argparse, json, os, re, sys, shutil, textwrap, subprocess, itertools, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# Utilities
# ---------------------------

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_marker(outdir: Path, name: str):
    (outdir / f".done_{name}").write_text(time.strftime("%F %T"))

def done_marker(outdir: Path, name: str) -> bool:
    return (outdir / f".done_{name}").exists()

def save_or_dot(dot_str: str, out_png: Path, out_dot: Path):
    out_dot.write_text(dot_str)
    try:
        subprocess.run(["dot", "-Tpng", str(out_dot), "-o", str(out_png)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        # Graphviz not available; PNG may be missing but DOT is saved
        pass

def human_tree(root: Path, max_depth=5, max_items_per_dir=20):
    lines, prefix_stack = [], []
    root = root.resolve()
    def walk(d: Path, depth: int):
        if depth > max_depth: return
        try:
            entries = sorted(list(d.iterdir()))
        except Exception:
            return
        hidden = 0
        if len(entries) > max_items_per_dir:
            hidden = len(entries) - max_items_per_dir
            entries = entries[:max_items_per_dir]
        for i, e in enumerate(entries):
            is_last = (i == len(entries) - 1 and hidden == 0)
            connector = "└── " if is_last else "├── "
            lines.append("".join(prefix_stack) + connector + e.name + ("/" if e.is_dir() else ""))
            if e.is_dir():
                prefix_stack.append("    " if is_last else "│   ")
                walk(e, depth+1)
                prefix_stack.pop()
        if hidden:
            lines.append("".join(prefix_stack) + f"└── ... (+{hidden} more)")
    lines.append(root.name + "/")
    walk(root, 1)
    return "\n".join(lines)

def find_first_image(root: Path, names=("0000.png","0001.png","0002.png")):
    for n in names:
        p = root / n
        if p.exists():
            return p
    for p in root.glob("*.png"):
        return p
    return None

# ---------------------------
# Fig. 1 – System teaser
# ---------------------------

def fig01_system_teaser(outdir: Path):
    dot = r'''
digraph G {
  rankdir=LR; node [shape=rounded];
  subgraph cluster_user { label="Client / Login Node"; style=dashed; CLI [label="continuous_cli.py"]; }
  subgraph cluster_slurm { label="SLURM"; style=dashed; SCHED [label="Scheduler"]; }
  subgraph cluster_nodes {
    label="GPU Nodes (xN)"; style=dashed;
    subgraph cluster_node1 {
      label="Node A"; style=dotted;
      W1 [label="persistent_carla_worker.sh\n(CARLA + Healthcheck)", shape=box];
      A1 [label="Agent Wrapper\n(Leaderboard V1)"];
      W1 -> A1 [label="attach"];
    }
    subgraph cluster_node2 {
      label="Node B"; style=dotted;
      W2 [label="persistent_carla_worker.sh\n(CARLA + Healthcheck)", shape=box];
      A2 [label="Agent Wrapper\n(Leaderboard V1)"];
      W2 -> A2 [label="attach"];
    }
  }
  MON [label="Monitor\n(collection_state/)"];
  OUT [label="dataset/<agent>/weather/map/route/\n(sensors)", shape=folder];

  CLI -> SCHED [label="submit jobs"];
  SCHED -> W1 [label="launch"];
  SCHED -> W2 [label="launch"];
  A1 -> MON [label="state/metrics"];
  A2 -> MON [label="state/metrics"];
  A1 -> OUT [label="save()"];
  A2 -> OUT [label="save()"];
}
'''
    save_or_dot(dot, outdir/"fig01_system_teaser.png", outdir/"fig01_system_teaser.dot")

# ---------------------------
# Fig. 2 – Sequence
# ---------------------------

def fig02_sequence(outdir: Path):
    dot = r'''
digraph G {
  rankdir=LR; node [shape=plaintext];
  CLI [label="continuous_cli.py"]; SLURM [label="SLURM Scheduler"];
  CONT [label="Singularity Container"]; WORK [label="persistent_carla_worker.sh"];
  AGENT [label="Agent Wrapper"]; MON [label="Monitor"]; DS [label="dataset/..."];
  CLI -> SLURM [label="submit"]; SLURM -> CONT [label="spawn"];
  CONT -> WORK [label="start"]; AGENT -> WORK [label="RPC attach"];
  AGENT -> DS [label="standardized save()"]; WORK -> MON [label="health JSON"]; AGENT -> MON [label="state JSON"];
}
'''
    save_or_dot(dot, outdir/"fig02_sequence.png", outdir/"fig02_sequence.dot")

# ---------------------------
# Fig. 3 – Lifecycle (state machine)
# ---------------------------

def fig03_lifecycle(outdir: Path):
    dot = r'''
digraph G {
  rankdir=LR; node [shape=circle];
  RESET [label="reset\n(combos)"]; START [label="start\n(persistent)"];
  MON [label="monitor\n(health loop)"]; REC [label="recover\n(restart)"];
  RESET -> START -> MON; MON -> REC -> MON; MON -> RESET [style=dashed,label="all done / rerun"];
}
'''
    save_or_dot(dot, outdir/"fig03_lifecycle.png", outdir/"fig03_lifecycle.dot")

# ---------------------------
# Fig. 4 – Container card (text)
# ---------------------------

def fig04_container_card(outdir: Path):
    txt = textwrap.dedent("""
    Container Definition (excerpt)
    ------------------------------
    Base: NVIDIA/CARLA 0.9.10 (Singularity)
    Adds:
      - Python deps per agents (carla, numpy, pandas, matplotlib, pillow, torch, etc.)
      - OpenCV/codecs (libjpeg*) and CUDA compatibility
      - Leaderboard V1 + agent wrapper + health monitor
    """).strip()
    (outdir/"fig04_container_card.txt").write_text(txt)

# ---------------------------
# Fig. 5 – Dataset tree (text + DOT)
# ---------------------------

def fig05_dataset_tree(dataset_root: Path, outdir: Path, max_depth=5):
    (outdir/"fig05_dataset_tree.txt").write_text(human_tree(dataset_root, max_depth=max_depth))
    dot_lines = ['digraph T {', 'node [shape=folder];', 'rankdir=LR;']
    nid = 0
    def nidgen():
        nonlocal nid; nid += 1; return f"n{nid}"
    def add_dir(d: Path, parent=None, depth=0):
        if depth>max_depth: return
        me = nidgen(); dot_lines.append(f'{me} [label="{d.name}"];')
        if parent: dot_lines.append(f"{parent} -> {me};")
        try:
            subs = [e for e in d.iterdir() if e.is_dir()]
        except Exception:
            subs = []
        for e in sorted(subs)[:20]:
            add_dir(e, me, depth+1)
    add_dir(dataset_root)
    dot_lines.append("}")
    save_or_dot("\n".join(dot_lines), outdir/"fig05_dataset_tree.png", outdir/"fig05_dataset_tree.dot")

# ---------------------------
# Fig. 6 – Throughput scaling
# ---------------------------

def load_job_durations(state_root: Path):
    durs = []
    comp = state_root/"completed_jobs.json"
    if comp.exists():
        try:
            data = json.loads(comp.read_text())
            if isinstance(data, dict) and "jobs" in data: data = data["jobs"]
            for j in data:
                d = j.get("duration_s")
                if d is None:
                    st = j.get("start_ts") or j.get("start_time"); et = j.get("end_ts") or j.get("end_time")
                    if st and et:
                        try: d = float(et) - float(st)
                        except Exception: d = None
                if d is not None: durs.append(float(d))
        except Exception:
            pass
    if not durs:
        est = state_root/"runtime_estimates.json"
        if est.exists():
            try:
                data = json.loads(est.read_text())
                if isinstance(data, dict):
                    for v in data.values():
                        try: durs.append(float(v))
                        except Exception: pass
            except Exception:
                pass
    return durs

def simulate_makespan(durations, workers):
    slots = [0.0]*workers
    for t in sorted(durations, reverse=True):
        i = min(range(workers), key=lambda k: slots[k])
        slots[i] += t
    return max(slots) if slots else 0.0

def fig06_throughput_scaling(state_root: Path, outdir: Path, worker_grid=(1,2,4,8,16)):
    durs = load_job_durations(state_root)
    if not durs:
        log("Fig06: no durations found; using placeholder (100×300s).")
        durs = [300.0]*100
    xs, ys_pred = [], []
    for w in worker_grid:
        xs.append(w); ys_pred.append(simulate_makespan(durs, w)/3600.0)
    ideal = []
    if ys_pred:
        base = ys_pred[0]
        ideal = [base/max(1,w) for w in xs]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys_pred, marker="o", label="Predicted makespan")
    plt.plot(xs, ideal, linestyle="--", marker="x", label="Ideal linear")
    plt.xlabel("Workers (GPUs)"); plt.ylabel("Wall time (hours)")
    plt.title("Throughput scaling from job durations"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"fig06_throughput_scaling.png", dpi=200); plt.close()

# ---------------------------
# Fig. 7 – Persistence savings
# ---------------------------

def estimate_jobs_count(state_root: Path):
    comp = state_root/"completed_jobs.json"
    if comp.exists():
        try:
            data = json.loads(comp.read_text())
            if isinstance(data, dict) and "jobs" in data: data = data["jobs"]
            return len(data)
        except Exception:
            pass
    jq = state_root/"job_queue.json"
    if jq.exists():
        try:
            data = json.loads(jq.read_text())
            if isinstance(data, dict) and "completed" in data:
                return len(data.get("completed", []))
        except Exception:
            pass
    return 0

def find_launch_overhead_seconds(default=60.0):
    # If you later track overhead explicitly, plug it here.
    return default

def fig07_persistence_savings(state_root: Path, outdir: Path):
    n_jobs = estimate_jobs_count(state_root)
    overhead_s = find_launch_overhead_seconds()
    xs = list(range(1, (n_jobs or 20)+1))
    per_job_min = overhead_s/60.0
    cum_minutes = np.cumsum([per_job_min]*len(xs))
    plt.figure(figsize=(6,4))
    plt.plot(xs, cum_minutes)
    plt.xlabel("Jobs"); plt.ylabel("Cumulative minutes saved")
    plt.title("Time saved via persistent CARLA servers")
    plt.grid(True, axis="y", linestyle=":")
    plt.tight_layout(); plt.savefig(outdir/"fig07_persistence_savings.png", dpi=200); plt.close()

# ---------------------------
# Fig. 8 – Coverage heatmap (FAST: parallel + cached)
# ---------------------------

def fast_count_png(dir_path: Path) -> int:
    # Avoid building giant Python lists; iterate via os.scandir (C-level readdir)
    cnt = 0
    try:
        with os.scandir(dir_path) as it:
            for e in it:
                # cheap suffix check, no stat calls
                n = e.name
                if len(n) >= 4 and n[-4:].lower() == ".png":
                    cnt += 1
    except Exception:
        return 0
    return cnt

def collect_rgb_dirs(dataset_root: Path, agents):
    work = []  # list of (agent, weather, map, route, rgb_dir, mtime)
    for agent in agents:
        aroot = dataset_root/agent
        if not aroot.exists(): continue
        for wdir in sorted(aroot.glob("weather_*")):
            if not wdir.is_dir(): continue
            w = wdir.name.replace("weather_","")
            for mdir in sorted(wdir.glob("map_*")):
                if not mdir.is_dir(): continue
                m = mdir.name
                for rdir in sorted(mdir.glob("routes_*")):
                    if not rdir.is_dir(): continue
                    rgb_dir = rdir/"rgb"
                    if rgb_dir.exists():
                        try:
                            mt = rgb_dir.stat().st_mtime
                        except Exception:
                            mt = 0.0
                        work.append((agent,w,m,rdir.name,rgb_dir,mt))
    return work

def load_coverage_cache(cache_csv: Path):
    if not cache_csv or not cache_csv.exists():
        return {}
    df = pd.read_csv(cache_csv)
    # expected columns: agent,weather,map,route,mtime,frames
    key = lambda r: (r["agent"], str(r["weather"]), str(r["map"]), str(r["route"]))
    return { key(r): (float(r["mtime"]), int(r["frames"])) for _, r in df.iterrows() }

def save_coverage_cache(cache_csv: Path, rows):
    df = pd.DataFrame(rows, columns=["agent","weather","map","route","mtime","frames"])
    df.to_csv(cache_csv, index=False)

def fig08_coverage_heatmap(dataset_root: Path, outdir: Path, agents, io_threads=8,
                           coverage_cache: Path=None, rebuild_cache=False):
    work = collect_rgb_dirs(dataset_root, agents)
    if not work:
        plt.figure(figsize=(6,4)); plt.title("No dataset found")
        plt.savefig(outdir/"fig08_scenario_coverage_heatmap.png", dpi=200); plt.close(); return

    cache = {} if rebuild_cache else load_coverage_cache(coverage_cache) if coverage_cache else {}
    todo, kept = [], []
    for agent,w,m,route,rgb_dir,mt in work:
        k = (agent,w,m,route)
        if k in cache and abs(cache[k][0] - mt) < 1e-6:
            kept.append((agent,w,m,route,mt,cache[k][1]))
        else:
            todo.append((agent,w,m,route,rgb_dir,mt))

    log(f"Fig08: coverage scan — {len(work)} routes (reuse {len(kept)} cached, scan {len(todo)}).")
    results = list(kept)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, io_threads)) as ex:
            futs = {}
            for agent,w,m,route,rgb_dir,mt in todo:
                futs[ex.submit(fast_count_png, rgb_dir)] = (agent,w,m,route,mt)
            done = 0; last_print = time.time()
            for f in as_completed(futs):
                cnt = f.result()
                agent,w,m,route,mt = futs[f]
                results.append((agent,w,m,route,mt,cnt))
                done += 1
                now = time.time()
                if now - last_print > 2.0:
                    log(f"Fig08: scanned {done}/{len(todo)} routes...")
                    last_print = now

    # Save/refresh cache
    if coverage_cache:
        save_coverage_cache(coverage_cache, results)

    # Build pivot and plot
    df = pd.DataFrame(results, columns=["agent","weather","map","route","mtime","frames"])
    df["route_id"] = df["map"].astype(str) + "/" + df["route"].astype(str)
    pivot = df.pivot_table(index="route_id", columns="weather", values="frames", aggfunc="sum", fill_value=0)

    plt.figure(figsize=(max(6, 0.3*len(pivot.columns)+4), max(6, 0.22*len(pivot.index)+3)))
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Frames captured")
    plt.title("Scenario coverage (route × weather) by frames")
    plt.tight_layout()
    plt.savefig(outdir/"fig08_scenario_coverage_heatmap.png", dpi=200)
    plt.close()

# ---------------------------
# Fig. 9 – Sensor mosaic
# ---------------------------

def fig09_sensor_mosaic(dataset_root: Path, outdir: Path, agent_choice=None):
    agent_dirs = [d for d in (dataset_root.iterdir() if dataset_root.exists() else []) if d.is_dir()]
    agent_dirs.sort()
    if agent_choice:
        agent_dirs = [dataset_root/agent_choice] + [d for d in agent_dirs if d.name != agent_choice]
    img_paths = []
    for ad in agent_dirs:
        for w in sorted(ad.glob("weather_*")):
            for m in sorted(w.glob("map_*")):
                for r in sorted(w.glob("routes_*")):
                    rgb = r/"rgb"; rgb_l = r/"rgb_left"; rgb_r = r/"rgb_right"
                    cands = [p for p in [find_first_image(rgb), find_first_image(rgb_l), find_first_image(rgb_r)] if p]
                    if cands: img_paths = cands; break
                if img_paths: break
            if img_paths: break
        if img_paths: break
    if not img_paths:
        plt.figure(figsize=(6,2)); plt.text(0.5,0.5,"No images found", ha="center", va="center")
        plt.axis("off"); plt.savefig(outdir/"fig09_sensor_mosaic.png", dpi=200, bbox_inches="tight"); plt.close(); return
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    h = max(im.size[1] for im in imgs); scaled=[]
    for im in imgs:
        w = int(im.size[0]*(h/im.size[1])); scaled.append(im.resize((w,h), Image.BILINEAR))
    total_w = sum(im.size[0] for im in scaled); canvas = Image.new("RGB", (total_w, h), (255,255,255))
    x=0
    for im in scaled:
        canvas.paste(im, (x,0)); x += im.size[0]
    canvas.save(outdir/"fig09_sensor_mosaic.png")

# ---------------------------
# Table 2 – Hardware config (LaTeX)
# ---------------------------

def table02_hardware_config(outdir: Path):
    rows = [
        {"Cluster":"pod10","GPU/Node":"8×A100 40GB","Nodes":"1","Driver":"535","CUDA":"12.2"},
        {"Cluster":"pod11","GPU/Node":"8×A100 40GB","Nodes":"1","Driver":"535","CUDA":"12.2"},
    ]
    df = pd.DataFrame(rows, columns=["Cluster","GPU/Node","Nodes","Driver","CUDA"])
    (outdir/"table02_hardware_config.tex").write_text(df.to_latex(index=False, escape=True, column_format="lcccc"))

# ---------------------------
# 3D topology – route×weather → score
# ---------------------------

def load_scores(scores_csv: Path, state_root: Path):
    rows = []
    if scores_csv and scores_csv.exists():
        df = pd.read_csv(scores_csv)
        lower = {c.lower():c for c in df.columns}
        need = ["agent","weather","map","route","score"]
        for n in need:
            if n not in lower: raise ValueError(f"--scores missing column: {n}")
        df = df.rename(columns={lower[c]:c for c in need})
        rows = df.to_dict(orient="records")
    else:
        comp = state_root/"completed_jobs.json"
        if comp.exists():
            try:
                data = json.loads(comp.read_text())
                if isinstance(data, dict) and "jobs" in data: data = data["jobs"]
                for j in data:
                    sc = j.get("score") or j.get("driving_score")
                    if sc is None: continue
                    rows.append({"agent": j.get("agent") or j.get("agent_name","unknown"),
                                 "weather": str(j.get("weather","")),
                                 "map": j.get("map") or j.get("town") or "",
                                 "route": j.get("route") or j.get("route_id") or "",
                                 "score": float(sc)})
            except Exception:
                pass
    return pd.DataFrame(rows)

def topology_3d(scores_df: pd.DataFrame, outdir: Path, agent=None):
    if scores_df.empty:
        plt.figure(figsize=(6,4)); plt.title("No scores found")
        plt.savefig(outdir/"topology_3d_route_weather_score.png", dpi=200); plt.close(); return
    sdf = scores_df[scores_df["agent"]==agent] if agent else scores_df
    if sdf.empty: sdf = scores_df
    sdf = sdf.copy(); sdf["route_id"] = sdf["map"].astype(str) + "/" + sdf["route"].astype(str)
    piv = sdf.pivot_table(index="route_id", columns="weather", values="score", aggfunc="mean")
    routes = list(piv.index); weathers = list(piv.columns); Z = piv.values
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    _x = np.arange(len(weathers)); _y = np.arange(len(routes))
    _xx, _yy = np.meshgrid(_x, _y); x, y = _xx.ravel(), _yy.ravel(); tops = Z.ravel()
    bottoms = np.zeros_like(tops); dx = np.ones_like(tops)*0.6; dy = np.ones_like(tops)*0.6
    ax.bar3d(x, y, bottoms, dx, dy, tops)
    ax.set_xlabel("Weather"); ax.set_ylabel("Route (map/route)"); ax.set_zlabel("Driving score")
    ax.set_xticks(_x); ax.set_xticklabels(weathers, rotation=90); ax.set_yticks(_y); ax.set_yticklabels(routes)
    plt.tight_layout(); plt.savefig(outdir/"topology_3d_route_weather_score.png", dpi=200); plt.close(fig)

# ---------------------------
# Main driver with selectors
# ---------------------------

FIGS = {
    "1":"teaser","teaser":"teaser",
    "2":"sequence","sequence":"sequence",
    "3":"lifecycle","lifecycle":"lifecycle",
    "4":"container","container":"container",
    "5":"dataset","dataset":"dataset",
    "6":"throughput","throughput":"throughput",
    "7":"persistence","persistence":"persistence",
    "8":"coverage","coverage":"coverage",
    "9":"mosaic","mosaic":"mosaic",
    "table2":"table2","hardware":"table2",
    "topology":"topology",
}

def main():
    ap = argparse.ArgumentParser(description="Generate HPC-CARLA paper figures (fast + cached)")
    ap.add_argument("--state-root", type=Path, required=True, help="Path to collection_state/")
    ap.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset/")
    ap.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    ap.add_argument("--agents", nargs="*", default=[], help="Agent subfolders (default: all under dataset/)")
    ap.add_argument("--scores", type=Path, default=None, help="Optional CSV: agent,weather,map,route,score")
    ap.add_argument("--agent-for-mosaic", default=None)
    ap.add_argument("--max-tree-depth", type=int, default=5)
    ap.add_argument("--workers", nargs="*", type=int, default=[1,2,4,8,16])
    ap.add_argument("--io-threads", type=int, default=8, help="Parallelism for coverage scan")
    ap.add_argument("--coverage-cache", type=Path, default=Path("coverage_cache.csv"))
    ap.add_argument("--rebuild-coverage-cache", action="store_true")
    ap.add_argument("--skip-coverage", action="store_true", help="Skip Fig.8")
    ap.add_argument("--only", nargs="*", default=[], help="Generate only these (IDs or names). Ex: --only 6 7 topology")
    ap.add_argument("--force", action="store_true", help="Ignore .done markers and regenerate")
    args = ap.parse_args()

    outdir = args.outdir; ensure_outdir(outdir)

    # discover agents
    agents = args.agents or [d.name for d in args.dataset_root.iterdir() if d.is_dir()]

    # Selection
    selected = set(FIGS.values()) if not args.only else set(FIGS[x] for x in args.only if x in FIGS)
    if not selected:
        log("Nothing selected (--only matched nothing)."); sys.exit(0)

    def run_once(key, fn):
        if not args.force and done_marker(outdir, key):
            log(f"Skip {key} (already done).")
            return
        log(f"Start {key} …")
        fn()
        save_marker(outdir, key)
        log(f"Done {key}.")

    # Fig 1
    if "teaser" in selected:
        run_once("fig01_teaser", lambda: fig01_system_teaser(outdir))
    # Fig 2
    if "sequence" in selected:
        run_once("fig02_sequence", lambda: fig02_sequence(outdir))
    # Fig 3
    if "lifecycle" in selected:
        run_once("fig03_lifecycle", lambda: fig03_lifecycle(outdir))
    # Fig 4
    if "container" in selected:
        run_once("fig04_container", lambda: fig04_container_card(outdir))
    # Fig 5
    if "dataset" in selected:
        run_once("fig05_dataset_tree", lambda: fig05_dataset_tree(args.dataset_root, outdir, max_depth=args.max_tree_depth))
    # Fig 6
    if "throughput" in selected:
        run_once("fig06_throughput", lambda: fig06_throughput_scaling(args.state_root, outdir, worker_grid=tuple(args.workers)))
    # Fig 7
    if "persistence" in selected:
        run_once("fig07_persistence", lambda: fig07_persistence_savings(args.state_root, outdir))
    # Fig 8 (heavy)
    if "coverage" in selected and not args.skip_coverage:
        run_once("fig08_coverage", lambda: fig08_coverage_heatmap(
            args.dataset_root, outdir, agents,
            io_threads=args.io_threads,
            coverage_cache=args.coverage_cache,
            rebuild_cache=args.rebuild_coverage_cache))
    elif "coverage" in selected and args.skip_coverage:
        log("Skip fig08_coverage (per --skip-coverage).")
    # Fig 9
    if "mosaic" in selected:
        run_once("fig09_mosaic", lambda: fig09_sensor_mosaic(args.dataset_root, outdir, agent_choice=args.agent_for_mosaic))
    # Table 2
    if "table2" in selected:
        run_once("table02_hardware", lambda: table02_hardware_config(outdir))
    # Topology
    if "topology" in selected:
        def _run_topology():
            scores_df = load_scores(args.scores, args.state_root)
            topology_3d(scores_df, outdir)
        run_once("topology_3d", _run_topology)

    log(f"All selected figures done -> {outdir}")

if __name__ == "__main__":
    main()
