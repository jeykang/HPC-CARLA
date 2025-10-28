#!/usr/bin/env python3
"""
Topology map (3D surface) of driving score vs. weather and route.

- Scans: <DATASET_ROOT>/<agent>/weather_*/map_*/routes_*_*/metadata.json
- Extracts: agent, weather id, town, route variant (long/short/tiny), composed/driving score
- Outputs:
    1) CSV of all found scores
    2) One or more 3D surface PNGs (per agent x town)
Usage:
    python make_topology_map.py \
        --dataset-root /path/to/dataset \
        --outdir /path/to/out \
        --agent interfuser --town town01

Notes:
- Works for InterFuser/TCP-style dirs (rgb/rgb_left/...) and LAV-style dirs (RGB_0/TEL_RGB/...) — both keep metadata.json in the *route* dir.  # see fig. 5 dataset tree
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import matplotlib.pyplot as plt

ROUTE_RE = re.compile(r"routes_(town(\d+))_(long|short|tiny)", re.IGNORECASE)

def _first_number(x: Any) -> Optional[float]:
    try:
        # accept ints/floats or numeric strings
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            # handle "95.0", "95", etc.
            return float(x)
    except Exception:
        return None
    return None

def _dfs_find_numeric_by_keys(obj: Any, key_substrings: List[str]) -> Optional[float]:
    """
    Depth-first search through nested dict/list and return the first numeric value
    whose key contains any of key_substrings (case-insensitive).
    Search order matters; provide most-specific keys first.
    """
    def _iter_items(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k, v
        elif isinstance(node, list):
            for idx, v in enumerate(node):
                yield str(idx), v

    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                lk = k.lower()
                for sub in key_substrings:
                    if sub in lk:
                        num = _first_number(v)
                        if num is not None:
                            return num
                stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                stack.append(v)
    return None

def extract_score(meta: Dict[str, Any]) -> Optional[float]:
    """
    Try common Leaderboard/agent metadata layouts.
    Priority order:
      - 'score_composed' / 'composed_score' (Leaderboard composed score)
      - 'driving_score'
      - generic 'score' (fallback)
    """
    for keys in [
        ["score_composed"],
        ["composed_score"],
        ["driving_score"],
        # Nested patterns often used: result.scores.score_composed, scores.score_composed, etc.
        ["result", "scores", "score_composed"],
        ["scores", "score_composed"],
        ["scores", "composed_score"],
    ]:
        # direct key-chain attempt
        node = meta
        ok = True
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok:
            num = _first_number(node)
            if num is not None:
                return num

    # Fuzzy fallback via DFS
    fuzzy = _dfs_find_numeric_by_keys(
        meta,
        key_substrings=["score_composed", "composed_score", "driving_score", "score"]
    )
    return fuzzy

def parse_weather_id(s: str) -> Optional[int]:
    m = re.search(r"weather_(\d+)", s, re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_map_id(s: str) -> Optional[int]:
    m = re.search(r"map_(\d+)", s, re.IGNORECASE)
    return int(m.group(1)) if m else None

def parse_route_bits(route_dirname: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Returns (town_name like 'town01', town_id int, variant in {long,short,tiny})
    """
    m = ROUTE_RE.search(route_dirname)
    if not m:
        return None, None, None
    full_town, town_id_str, variant = m.group(1), m.group(2), m.group(3)
    town_name = f"town{int(town_id_str):02d}"
    return town_name, int(town_id_str), variant

def walk_metadata(dataset_root: Path) -> pd.DataFrame:
    """
    Find all metadata.json files and extract rows: agent, weather_id, map_id, town, variant, route_dir, score
    Expected path pattern:
        <root>/<agent>/weather_<id>/map_<id>/<routes_*_*>/metadata.json
    """
    rows = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        if "metadata.json" not in filenames:
            continue
        route_dir = Path(dirpath)
        try:
            rel = route_dir.relative_to(dataset_root)
        except Exception:
            continue
        parts = rel.parts
        # Expect at least: agent, weather_*, map_*, routes_*
        if len(parts) < 4:
            continue
        agent, weather_str, map_str, route_str = parts[0], parts[1], parts[2], parts[3]
        weather_id = parse_weather_id(weather_str)
        map_id = parse_map_id(map_str)
        town_name, town_id, variant = parse_route_bits(route_str)

        meta_path = route_dir / "metadata.json"
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        score = extract_score(meta)

        rows.append(
            dict(
                agent=agent,
                weather_id=weather_id,
                map_id=map_id,
                town=town_name,
                town_id=town_id,
                variant=variant,
                route=route_str,
                score=score,
                meta_path=str(meta_path),
            )
        )
    df = pd.DataFrame(rows)
    # Drop rows with no parsed essentials
    if not df.empty:
        df = df.dropna(subset=["weather_id", "town_id", "score"])
    return df

def plot_surface(df: pd.DataFrame, out_png: Path, title: str):
    """
    Build a surface where X=route index (ordered by route name), Y=weather id, Z=score.
    """
    routes = sorted(df["route"].unique())
    weathers = sorted(df["weather_id"].unique())

    r_index = {r: i for i, r in enumerate(routes)}
    w_index = {w: i for i, w in enumerate(weathers)}

    Z = np.full((len(weathers), len(routes)), np.nan, dtype=float)
    for _, row in df.iterrows():
        ri = r_index[row["route"]]
        wi = w_index[row["weather_id"]]
        Z[wi, ri] = row["score"]

    X, Y = np.meshgrid(np.arange(len(routes)), np.arange(len(weathers)))

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Surface; missing values will appear as gaps
    ax.plot_surface(X, Y, Z, linewidth=0.5, antialiased=True)
    ax.set_xticks(np.arange(len(routes)))
    ax.set_xticklabels(routes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(weathers)))
    ax.set_yticklabels([str(w) for w in weathers])
    ax.set_xlabel("Route")
    ax.set_ylabel("Weather ID")
    ax.set_zlabel("Driving Score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--agent", type=str, default=None,
                    help="Filter to a single agent (e.g., 'interfuser', 'lav', 'tcp').")
    ap.add_argument("--town", type=str, default=None,
                    help="Filter to a single town (e.g., 'town01').")
    ap.add_argument("--min-runs", type=int, default=1,
                    help="Keep (agent,town) groups with at least this many rows.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = walk_metadata(args.dataset_root)
    if df.empty:
        raise SystemExit("No metadata.json files with usable scores were found.")

    # Optional filters
    if args.agent:
        df = df[df["agent"].str.lower() == args.agent.lower()]
    if args.town:
        df = df[df["town"].str.lower() == args.town.lower()]

    # Save the raw table of scores
    csv_path = args.outdir / "topology_scores.csv"
    df.to_csv(csv_path, index=False)

    # Build a surface per (agent, town) group
    grp = df.groupby(["agent", "town"])
    made_any = False
    for (agent, town), sub in grp:
        if len(sub) < args.min_runs:
            continue
        title = f"Topology: score vs weather/route — {agent} · {town}"
        out_png = args.outdir / f"topology_{agent}_{town}.png"
        plot_surface(sub, out_png, title=title)
        made_any = True

    if not made_any:
        # As a fallback, aggregate across towns per agent (use route names as-is)
        for agent, sub in df.groupby("agent"):
            if len(sub) < args.min_runs:
                continue
            title = f"Topology: score vs weather/route — {agent} (all towns)"
            out_png = args.outdir / f"topology_{agent}_alltowns.png"
            plot_surface(sub, out_png, title=title)

    print(f"[OK] Wrote: {csv_path}")
    for p in sorted(args.outdir.glob("topology_*.png")):
        print(f"[OK] Wrote: {p}")

if __name__ == "__main__":
    main()
