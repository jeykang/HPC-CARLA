"""Tier-1 SensorStager tests — stdlib only, no torch/carla/numpy.

The leaderboard builds a fresh agent, and so a fresh SensorStager, for every
route of a route file, and all of them write into the one job directory. These
tests pin the behaviour that depends on: a later route must add to the earlier
routes' shards and manifest, never replace them.

Run directly (no pytest needed):   python3 tests/test_sensor_stager.py
Or via pytest:                      pytest tests/test_sensor_stager.py
"""
import json
import os
import shutil
import sys
import tarfile
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "leaderboard"))

from team_code.sensor_stager import SensorStager  # noqa: E402


def _drive_route(final_dir, stage_root, route, ticks, shard_size=2):
    """Simulate one route: one new stager, `ticks` frames, then finalize."""
    stager = SensorStager(final_dir, stage_root=stage_root, shard_size=shard_size)
    assert stager.enabled
    for tick in range(ticks):
        sensor_dir = stager.dir_for("rgb_front")
        with open(os.path.join(sensor_dir, "r%d_%06d.png" % (route, tick)), "w") as f:
            f.write("frame")
        stager.after_tick()
    return stager.finalize()


def _shard_members(final_dir):
    shard_dir = os.path.join(final_dir, "shards")
    members = []
    for name in sorted(os.listdir(shard_dir)):
        with tarfile.open(os.path.join(shard_dir, name)) as tar:
            members.extend(tar.getnames())
    return sorted(members)


def _workdir():
    root = tempfile.mkdtemp(prefix="stager_test_")
    return root, os.path.join(root, "final"), os.path.join(root, "stage")


def test_single_route_rolls_every_frame_into_shards():
    root, final_dir, stage_root = _workdir()
    try:
        manifest = _drive_route(final_dir, stage_root, route=1, ticks=5)
        assert _shard_members(final_dir) == ["rgb_front/r1_%06d.png" % t for t in range(5)]
        # 5 ticks at shard_size 2 -> two full shards + the remainder flushed by finalize()
        assert manifest["n_shards"] == 3
        assert manifest["total_ticks"] == 5
        assert not os.listdir(stage_root), "finalize() must drop the node-local stage"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_second_route_does_not_overwrite_first_routes_shards():
    root, final_dir, stage_root = _workdir()
    try:
        _drive_route(final_dir, stage_root, route=1, ticks=4)
        _drive_route(final_dir, stage_root, route=2, ticks=4)
        expected = sorted("rgb_front/r%d_%06d.png" % (r, t) for r in (1, 2) for t in range(4))
        assert _shard_members(final_dir) == expected
        assert sorted(os.listdir(os.path.join(final_dir, "shards"))) == [
            "shard_%05d.tar" % i for i in range(4)]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_manifest_accumulates_across_routes():
    root, final_dir, stage_root = _workdir()
    try:
        _drive_route(final_dir, stage_root, route=1, ticks=4)
        _drive_route(final_dir, stage_root, route=2, ticks=3)
        with open(os.path.join(final_dir, "shard_manifest.json")) as f:
            manifest = json.load(f)
        on_disk = sorted(os.listdir(os.path.join(final_dir, "shards")))
        assert [e["shard"] for e in manifest["shards"]] == on_disk
        assert manifest["n_shards"] == len(on_disk) == 4
        assert manifest["total_ticks"] == 7
        assert sum(e["n_files"] for e in manifest["shards"]) == 7
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unusable_stage_root_disables_staging():
    root, final_dir, _stage_root = _workdir()
    try:
        blocker = os.path.join(root, "not_a_dir")
        with open(blocker, "w") as f:
            f.write("x")
        stager = SensorStager(final_dir, stage_root=blocker)
        assert not stager.enabled
        # Caller falls back to writing straight into the final dir.
        assert stager.dir_for("rgb_front") == os.path.join(final_dir, "rgb_front")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
