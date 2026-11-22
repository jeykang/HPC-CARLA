"""
sensor_stager.py — WekaFS-friendly sensor data writer.

Problem: the per-frame sensor dump writes ~7 tiny files per simulation tick
(one PNG/NPY/JSON per sensor), i.e. tens of thousands of small files per route.
On the cluster's WekaFS `/scratch` each create is a separate metadata op +
file-lease acquisition; that small-file storm is the load that drives Weka's
`HangingIos` / node-fencing (see NODE_FAILURE_EVIDENCE_2026-07-15.md).

Fix: write the per-frame files to a NODE-LOCAL staging directory (default
`/dev/shm`, which is never Weka), and periodically roll them into a single large
`.tar` shard on `/scratch`. This turns ~77k small-file/lease ops per route into a
few dozen large sequential writes — the access pattern Weka handles well.

Design notes:
- Stdlib only (no numpy/cv2), so it is importable + testable off the GPU nodes.
- `results.json` / `run_summary.json` / `metadata.json` are NOT handled here —
  they stay as direct small writes to `/scratch` (they are tiny and, crucially,
  `results.json` is the harvester's per-route recovery source; it must remain a
  live file on `/scratch`).
- Bounded loss on crash: only the current partial shard (< `shard_size` ticks)
  lives on node-local storage; everything rolled is already on `/scratch`.
- Fail-safe: if node-local staging can't be set up, `enabled` stays False and the
  caller transparently falls back to writing directly to `/scratch` (legacy path).

On-disk result under a route's `/scratch` dir:
    shards/shard_00000.tar        # each tar holds `<sensor_folder>/<frame>.<ext>`
    shards/shard_00001.tar
    ...
    shard_manifest.json           # index: shard_size, n_shards, per-shard file counts

One job dir is shared by every route of a route file, and the leaderboard builds
a fresh agent (hence a fresh stager) per route. Shard numbering and the manifest
therefore continue from whatever is already in the dir instead of restarting at
0, which would overwrite the earlier routes' shards.
Unpack with `tools/unpack_shards.py` (reconstructs the original per-frame layout).
"""

import os
import re
import json
import shutil
import tarfile

_SHARD_RE = re.compile(r"^shard_(\d+)\.tar(\.gz)?$")


def _safe_name(path):
    """Stable, filesystem-safe token derived from the final /scratch dir."""
    return path.strip("/").replace("/", "__").replace(" ", "_") or "route"


class SensorStager:
    def __init__(self, final_dir, stage_root="/dev/shm/hpc_carla_stage",
                 shard_size=64, compress=False, logger=None):
        self.final_dir = final_dir
        self.shard_dir = os.path.join(final_dir, "shards")
        self.shard_size = max(1, int(shard_size))
        self.compress = bool(compress)
        self._log = logger if callable(logger) else (lambda _m: None)
        self.stage = os.path.join(stage_root, _safe_name(final_dir))
        self._tick = 0
        self._shard_idx = 0
        self._manifest = []
        self.enabled = False
        try:
            # unique per-route stage (clear any stale leftovers from a re-run)
            shutil.rmtree(self.stage, ignore_errors=True)
            os.makedirs(self.stage, exist_ok=True)
            os.makedirs(self.shard_dir, exist_ok=True)
            self._shard_idx = self._next_shard_index()
            self.enabled = True
            self._log("[stager] enabled: stage=%s shard_size=%d -> %s"
                      % (self.stage, self.shard_size, self.shard_dir))
        except OSError as exc:
            self._log("[stager] disabled (stage setup failed: %s); "
                      "falling back to direct /scratch writes" % exc)
            self.enabled = False

    # -- first free shard number in shard_dir -------------------------------
    def _next_shard_index(self):
        """One past the highest shard already present (0 for a fresh dir)."""
        highest = -1
        for name in os.listdir(self.shard_dir):
            m = _SHARD_RE.match(name)
            if m:
                highest = max(highest, int(m.group(1)))
        return highest + 1

    # -- directory handed to the per-sensor writer --------------------------
    def dir_for(self, folder_name):
        """Node-local sensor dir when enabled; else the /scratch dir (legacy)."""
        base = self.stage if self.enabled else self.final_dir
        path = os.path.join(base, folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    # -- called once per simulation tick ------------------------------------
    def after_tick(self):
        if not self.enabled:
            return
        self._tick += 1
        if self._tick % self.shard_size == 0:
            self._roll()

    # -- roll everything staged so far into one tar on /scratch -------------
    def _roll(self):
        files = []
        for root, _dirs, names in os.walk(self.stage):
            for name in names:
                files.append(os.path.join(root, name))
        if not files:
            return
        suffix = ".tar.gz" if self.compress else ".tar"
        shard_name = "shard_%05d%s" % (self._shard_idx, suffix)
        shard_path = os.path.join(self.shard_dir, shard_name)
        tmp_path = shard_path + ".tmp"
        mode = "w:gz" if self.compress else "w"
        try:
            with tarfile.open(tmp_path, mode) as tar:
                for f in sorted(files):
                    tar.add(f, arcname=os.path.relpath(f, self.stage))
            os.replace(tmp_path, shard_path)  # atomic within /scratch
        except OSError as exc:
            self._log("[stager] shard write failed (%s); keeping staged files" % exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return
        for f in files:
            try:
                os.unlink(f)
            except OSError:
                pass
        self._manifest.append({"shard": shard_name, "n_files": len(files),
                               "tick_end": self._tick})
        self._log("[stager] wrote %s (%d files)" % (shard_name, len(files)))
        self._shard_idx += 1

    # -- flush remainder + write the index, then drop the local stage -------
    def finalize(self):
        if not self.enabled:
            return
        self._roll()
        manifest_path = os.path.join(self.final_dir, "shard_manifest.json")
        # Earlier routes of this job already indexed their shards here; extend
        # that index (dropping entries whose shard is gone) rather than replace it.
        prior_shards, prior_ticks = [], 0
        try:
            with open(manifest_path) as f:
                prior = json.load(f)
            prior_shards = [e for e in prior.get("shards", [])
                            if isinstance(e, dict) and os.path.exists(
                                os.path.join(self.shard_dir, str(e.get("shard"))))]
            prior_ticks = int(prior.get("total_ticks", 0)) if prior_shards else 0
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        shards = prior_shards + self._manifest
        manifest = {
            "shard_size": self.shard_size,
            "compress": self.compress,
            "n_shards": len(shards),
            "total_ticks": prior_ticks + self._tick,
            "shards": shards,
        }
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except OSError as exc:
            self._log("[stager] manifest write failed: %s" % exc)
        shutil.rmtree(self.stage, ignore_errors=True)
        self.enabled = False
        return manifest
