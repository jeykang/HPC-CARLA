"""Tier-1 tests for queue recovery and the results tools — stdlib only.

Covers the rules that decide what a job's on-disk results mean:
  * `manage_continuous.py reclaim` — an orphaned job counts as completed only
    if the leaderboard evaluated its whole route file;
  * `tools/harvest_results.py` — a job is labelled by its latest finished
    attempt, and repeat-variance jobs are read from their `_repNN` directory;
  * `tools/classify_outcomes.py` — a simulator crash is an infrastructure
    failure, not a driving failure.

Run directly (no pytest needed):   python3 tests/test_queue_and_results.py
Or via pytest:                      pytest tests/test_queue_and_results.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))

import classify_outcomes as co  # noqa: E402
import harvest_results as hr  # noqa: E402
import manage_continuous as mc  # noqa: E402


def _job(job_id, route, status, **extra):
    job = {"id": job_id, "agent": "tcp", "weather": 0, "route": route, "town": "02",
           "status": status, "attempts": 1, "gpu": 0, "node": "n0",
           "start_time": "2026-07-01T00:00:00Z", "end_time": None, "duration": None}
    job.update(extra)
    return job


def _write_checkpoint(dataset_dir, stem, done, total, entry_status, run_summary=False):
    """A results.json as the leaderboard leaves it after `done` of `total` routes."""
    run_dir = os.path.join(dataset_dir, "tcp", "weather_0", "map_02", stem)
    os.makedirs(run_dir)
    records = [{"route_id": "RouteScenario_%d" % i, "status": "Completed",
                "scores": {"score_composed": 100.0, "score_route": 100.0, "score_penalty": 1.0}}
               for i in range(done)]
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"entry_status": entry_status,
                   "_checkpoint": {"progress": [done, total], "records": records}}, f)
    if run_summary:
        with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
            json.dump({"global_steps": 100}, f)


@contextlib.contextmanager
def _workspace():
    root = tempfile.mkdtemp(prefix="queue_test_")
    saved = os.environ.get("DATASET_DIR")
    os.environ["DATASET_DIR"] = os.path.join(root, "dataset")
    try:
        yield os.path.join(root, "state"), os.path.join(root, "dataset")
    finally:
        if saved is None:
            os.environ.pop("DATASET_DIR", None)
        else:
            os.environ["DATASET_DIR"] = saved
        shutil.rmtree(root, ignore_errors=True)


def _reclaim(state_dir, queue, archive=()):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "job_queue.json"), "w") as f:
        json.dump({"jobs": queue}, f)
    with open(os.path.join(state_dir, "completed_jobs.json"), "w") as f:
        json.dump({"jobs": list(archive)}, f)
    with contextlib.redirect_stdout(io.StringIO()):
        mc.ContinuousManager(state_dir).reclaim_stale(stale_hours=0)
    with open(os.path.join(state_dir, "job_queue.json")) as f:
        new_queue = {j["id"]: j for j in json.load(f)["jobs"]}
    with open(os.path.join(state_dir, "completed_jobs.json")) as f:
        new_archive = json.load(f)["jobs"]
    return new_queue, new_archive


# --- reclaim ---------------------------------------------------------------
def test_reclaim_completes_job_whose_whole_route_file_was_evaluated():
    with _workspace() as (state_dir, dataset_dir):
        _write_checkpoint(dataset_dir, "routes_a", 3, 3, "Finished")
        queue, archive = _reclaim(state_dir, [_job(0, "routes_a.xml", "running")])
        assert queue[0]["status"] == "completed"
        assert queue[0]["score_n_routes"] == 3
        assert [(j["id"], j["status"]) for j in archive] == [(0, "completed")]


def test_reclaim_requeues_partially_evaluated_job_even_with_run_summary():
    # run_summary.json is written at the end of EVERY route, so it exists after
    # one route of 300; it must not be read as "the job finished".
    with _workspace() as (state_dir, dataset_dir):
        _write_checkpoint(dataset_dir, "routes_b", 1, 300, "", run_summary=True)
        queue, archive = _reclaim(state_dir, [_job(0, "routes_b.xml", "running")])
        assert queue[0]["status"] == "pending"
        assert archive == []


def test_reclaim_requeues_job_with_no_output():
    with _workspace() as (state_dir, _dataset_dir):
        queue, _archive = _reclaim(state_dir, [_job(0, "routes_c.xml", "running")])
        assert queue[0]["status"] == "pending"


def test_reclaim_reads_repeat_jobs_from_their_suffixed_dir():
    with _workspace() as (state_dir, dataset_dir):
        _write_checkpoint(dataset_dir, "routes_p2_rep03", 1, 1, "Finished")
        queue, _archive = _reclaim(state_dir, [_job(0, "routes_p2.xml", "running", repeat=3)])
        assert queue[0]["status"] == "completed"


def test_reclaim_archives_completion_after_an_earlier_failed_attempt():
    with _workspace() as (state_dir, dataset_dir):
        _write_checkpoint(dataset_dir, "routes_d", 2, 2, "Finished with agent errors")
        earlier = _job(0, "routes_d.xml", "failed")
        _queue, archive = _reclaim(state_dir, [_job(0, "routes_d.xml", "running", attempts=2)],
                                   archive=[earlier])
        assert [j["status"] for j in archive] == ["failed", "completed"]


def test_reclaim_leaves_other_statuses_alone():
    with _workspace() as (state_dir, _dataset_dir):
        queue, _archive = _reclaim(state_dir, [_job(0, "routes_e.xml", "failed"),
                                               _job(1, "routes_f.xml", "pending")])
        assert (queue[0]["status"], queue[1]["status"]) == ("failed", "pending")


# --- harvest ---------------------------------------------------------------
def _finished(state_dir, queue, archive):
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, "job_queue.json"), "w") as f:
        json.dump({"jobs": queue}, f)
    with open(os.path.join(state_dir, "completed_jobs.json"), "w") as f:
        json.dump({"jobs": archive}, f)
    jobs = hr.load_finished_jobs(os.path.join(state_dir, "job_queue.json"))
    return {j["id"]: j["status"] for j in jobs}


def test_harvest_labels_job_by_latest_attempt():
    with _workspace() as (state_dir, _dataset_dir):
        statuses = _finished(
            state_dir,
            queue=[_job(0, "a.xml", "completed"),    # failed, then completed on retry
                   _job(1, "b.xml", "failed"),       # failed once
                   _job(2, "c.xml", "pending"),      # failed, then reset for retry
                   _job(3, "d.xml", "pending")],     # never ran
            archive=[_job(0, "a.xml", "failed"), _job(1, "b.xml", "failed"),
                     _job(2, "c.xml", "failed"), _job(0, "a.xml", "completed")])
        assert statuses == {0: "completed", 1: "failed", 2: "failed"}


def test_harvest_takes_live_queue_status_over_stale_archive():
    # `reclaim` can complete a job in the queue whose only archive entry is an
    # earlier failed attempt.
    with _workspace() as (state_dir, _dataset_dir):
        statuses = _finished(state_dir, queue=[_job(0, "a.xml", "completed")],
                             archive=[_job(0, "a.xml", "failed")])
        assert statuses == {0: "completed"}


def test_harvest_results_path_for_plain_and_repeat_jobs():
    plain = hr.results_path_for("ds", _job(0, "routes_town05_tiny.xml", "failed", town="05", weather=6))
    assert plain == os.path.join("ds", "tcp", "weather_6", "map_05", "routes_town05_tiny", "results.json")
    rep = hr.results_path_for("ds", _job(1, "routes_town05_p2.xml", "completed", town="05",
                                         weather=6, repeat=7))
    assert rep == os.path.join("ds", "tcp", "weather_6", "map_05", "routes_town05_p2_rep07", "results.json")


def test_harvest_keeps_repeats_as_separate_evals():
    with _workspace() as (state_dir, dataset_dir):
        jobs = []
        for rep in range(3):
            _write_checkpoint(dataset_dir, "routes_p2_rep%02d" % rep, 1, 1, "Finished")
            jobs.append(_job(rep, "routes_p2.xml", "completed", repeat=rep, seed=3000 + rep))
        rows, counters = hr.harvest(jobs, dataset_dir)
        assert counters.files_missing == 0
        assert sorted(r["seed"] for r in rows) == [3000, 3001, 3002]


# --- outcome classification -----------------------------------------------
def test_classify_route_separates_driving_from_infrastructure():
    assert co.classify_route("Completed") == "valid_pass"
    assert co.classify_route("Failed - Agent timed out") == "valid_fail"
    assert co.classify_route("Failed - Agent got blocked") == "valid_fail"
    assert co.classify_route("Failed - Simulation crashed") == "infra_fail"
    assert co.classify_route("Failed - Agent couldn't be set up") == "agent_error"
    assert co.classify_route("") == "infra_fail"


def test_is_driving_outcome():
    assert co.is_driving_outcome("Completed")
    assert co.is_driving_outcome("Failed - Agent deviated from the route")
    assert not co.is_driving_outcome("Failed - Simulation crashed")
    assert not co.is_driving_outcome("Failed - Agent's sensors were invalid")
    assert not co.is_driving_outcome(None)


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
