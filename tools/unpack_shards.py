#!/usr/bin/env python3
"""
unpack_shards.py — reconstruct the per-frame sensor layout from tar shards.

The collector now writes sensor frames as tar shards (see
leaderboard/team_code/sensor_stager.py) to spare WekaFS the small-file storm:

    <route_dir>/shards/shard_00000.tar   # holds <sensor>/<frame>.<ext>
    <route_dir>/shard_manifest.json

This tool expands those shards back to the classic layout expected by
downstream training/preprocessing:

    <route_dir>/<sensor>/<frame>.<ext>

Stdlib only. Safe to re-run (idempotent). Does NOT delete shards unless --purge.

Usage:
  # one route
  python3 tools/unpack_shards.py dataset/tcp/weather_0/map_01/routes_town01_tiny
  # whole dataset (every dir that has a shards/ subdir)
  python3 tools/unpack_shards.py --recursive dataset
  # expand and then remove the tar shards to reclaim inodes
  python3 tools/unpack_shards.py --recursive --purge dataset
"""
import argparse
import json
import os
import sys
import tarfile


def _is_within(base, target):
    """Guard against path traversal in tar members."""
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    return target == base or target.startswith(base + os.sep)


def unpack_route(route_dir, purge=False, verbose=True):
    shard_dir = os.path.join(route_dir, "shards")
    if not os.path.isdir(shard_dir):
        return 0
    shards = sorted(f for f in os.listdir(shard_dir)
                    if f.startswith("shard_") and (f.endswith(".tar") or f.endswith(".tar.gz")))
    if not shards:
        return 0
    n_files = 0
    for name in shards:
        path = os.path.join(shard_dir, name)
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                dest = os.path.join(route_dir, member.name)
                if not _is_within(route_dir, dest):
                    if verbose:
                        print("  ! skipping unsafe member: %s" % member.name)
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with tar.extractfile(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
                n_files += 1
    if verbose:
        print("unpacked %-55s %5d files from %2d shards" % (route_dir, n_files, len(shards)))
    if purge:
        for name in shards:
            try:
                os.unlink(os.path.join(shard_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(shard_dir)
        except OSError:
            pass
    return n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="route dir, or a root to search with --recursive")
    ap.add_argument("--recursive", action="store_true",
                    help="unpack every dir under PATH that contains a shards/ subdir")
    ap.add_argument("--purge", action="store_true",
                    help="delete the tar shards after successful expansion")
    a = ap.parse_args()

    routes = []
    if a.recursive:
        for root, dirs, _files in os.walk(a.path):
            if "shards" in dirs:
                routes.append(root)
    else:
        routes.append(a.path)

    total_files, total_routes = 0, 0
    for r in routes:
        n = unpack_route(r, purge=a.purge)
        if n:
            total_files += n
            total_routes += 1
    print("---\nunpacked %d routes, %d files total%s"
          % (total_routes, total_files, " (shards purged)" if a.purge else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
