#!/usr/bin/env python3
"""
route_map_density.py — PER-ROUTE map density from the real interpolated driven path.

Motivation (P1a): the `_short`/`_tiny` route files store only 2 waypoints (endpoints);
the actual driven path is reconstructed at runtime by the leaderboard's
GlobalRoutePlanner. Offline endpoint geometry is therefore near-noise, and the
existing map-density axis (tools/map_density.py) is only per-TOWN. This upgrades it
to per-ROUTE by reproducing the exact runtime interpolation
(leaderboard/.../route_manipulation.py:interpolate_trajectory) but OFFLINE:
`carla.Map(name, xodr)` needs no server, no GL, no segfaults — so it runs as a plain
container job.

For every <route> in the training route files it counts, along the interpolated path:
  - n_interp_wps, path_len_m
  - n_junction_wps, n_distinct_junctions   (intersections actually driven through)
  - junctions_per_km                        (the per-route density axis)
  - heading_change_deg, heading_deg_per_km  (curvature proxy)

Run INSIDE the CARLA container (needs `carla` + agents.navigation):
  singularity exec --nv -B "$PWD":/workspace carla_official.sif bash -lc \
    'cd /workspace && PYTHONPATH=/home/carla/PythonAPI/carla:$PYTHONPATH \
     python3 tools/route_map_density.py --out paper_artifacts/route_map_density.csv'
"""
import argparse
import csv
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

XODR_DIR = "carla_maps/OpenDrive"


def xodr_path_for(town):
    """Map a route's town attribute (e.g. 'Town10HD'/'Town10') to a .xodr file."""
    cands = [town]
    if town in ("Town10", "Town10HD"):
        cands = ["Town10HD", "Town10"]  # CARLA loads the HD variant for town10 routes
    for c in cands:
        p = os.path.join(XODR_DIR, c + ".xodr")
        if os.path.exists(p):
            return p
    return None


def parse_routes(routes_dir):
    """Yield (route_file, route_id, town, [carla.Location,...]) per <route>."""
    files = sorted(glob.glob(os.path.join(routes_dir, "routes_*.xml")))
    files = [f for f in files if not f.endswith(("_smoke.xml",)) and ".test" not in f]
    for f in files:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as e:
            print("[warn] parse %s: %s" % (f, e), file=sys.stderr)
            continue
        for r in root.findall("route"):
            town = r.get("town")
            locs = []
            for wp in r.findall("waypoint"):
                locs.append(carla.Location(
                    x=float(wp.get("x")), y=float(wp.get("y")), z=float(wp.get("z"))))
            if town and len(locs) >= 2:
                yield os.path.basename(f), r.get("id"), town, locs


def metrics_for_route(grp, locs):
    """Reproduce interpolate_trajectory: chain trace_route through the waypoints."""
    trace = []
    for i in range(len(locs) - 1):
        try:
            seg = grp.trace_route(locs[i], locs[i + 1])
        except Exception as e:
            print("[warn] trace_route seg %d failed: %s" % (i, e), file=sys.stderr)
            continue
        trace.extend(seg)
    if not trace:
        return None

    path_len = 0.0
    heading_change = 0.0
    n_junc = 0
    jids = set()
    prev_loc = None
    prev_yaw = None
    for wp, _opt in trace:
        tf = wp.transform
        loc = tf.location
        if wp.is_junction:
            n_junc += 1
            try:
                jids.add(wp.get_junction().id)
            except Exception:
                pass
        if prev_loc is not None:
            path_len += math.sqrt((loc.x - prev_loc.x) ** 2 + (loc.y - prev_loc.y) ** 2)
        if prev_yaw is not None:
            d = abs((tf.rotation.yaw - prev_yaw + 180.0) % 360.0 - 180.0)
            heading_change += d
        prev_loc, prev_yaw = loc, tf.rotation.yaw

    km = path_len / 1000.0 if path_len > 0 else float("nan")
    return {
        "n_interp_wps": len(trace),
        "path_len_m": round(path_len, 2),
        "n_junction_wps": n_junc,
        "n_distinct_junctions": len(jids),
        "junctions_per_km": round(len(jids) / km, 4) if km and km == km and km > 0 else "",
        "heading_change_deg": round(heading_change, 1),
        "heading_deg_per_km": round(heading_change / km, 2) if km and km == km and km > 0 else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes-dir", default="leaderboard/data/training_routes")
    ap.add_argument("--out", default="paper_artifacts/route_map_density.csv")
    ap.add_argument("--resolution", type=float, default=1.0)
    a = ap.parse_args()

    # Group routes by town so each map + GRP is built once.
    by_town = {}
    for rf, rid, town, locs in parse_routes(a.routes_dir):
        by_town.setdefault(town, []).append((rf, rid, locs))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["town", "route_file", "route_id", "n_interp_wps", "path_len_m",
            "n_junction_wps", "n_distinct_junctions", "junctions_per_km",
            "heading_change_deg", "heading_deg_per_km"]
    n_written = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for town in sorted(by_town):
            xp = xodr_path_for(town)
            if not xp:
                print("[skip] no .xodr for %s" % town, file=sys.stderr)
                continue
            cmap = carla.Map(town, open(xp).read())  # OFFLINE — no server
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(cmap, a.resolution))
            grp.setup()
            n_town = 0
            for rf, rid, locs in by_town[town]:
                m = metrics_for_route(grp, locs)
                if m is None:
                    continue
                row = {"town": town, "route_file": rf, "route_id": rid}
                row.update(m)
                w.writerow(row)
                n_written += 1
                n_town += 1
            print("[%s] %d routes -> density (xodr=%s)" % (town, n_town, os.path.basename(xp)))
    print("wrote %d per-route density rows to %s" % (n_written, a.out))


if __name__ == "__main__":
    main()
