#!/usr/bin/env python3
"""Validate that the scheduler's difficulty score predicts real agent performance.

The continuous-collection scheduler ranks pending jobs HARDEST-first by an
agent-independent *difficulty score* (route geometry + scenario density +
weather), so a later `prune` step can drop the easier same-route variants as
redundant — on the assumption that an agent which clears a hard condition also
clears the easy one. That assumption only holds if difficulty actually
*anti-correlates* with agent performance. This tool measures that correlation
(Spearman + Pearson, per-agent and pooled, with p-values and a per-component
breakdown) so the result can go in the paper.

Expected result: a NEGATIVE correlation (harder condition -> lower driving
score). A near-zero or positive correlation would undermine the prune step.

Runs anywhere: stdlib only. numpy / scipy / matplotlib are all OPTIONAL and
guarded — Spearman/Pearson and their p-values are implemented by hand so the
tool degrades gracefully on a bare login node.

Usage:
  python3 tools/difficulty_validation.py [--project-root .] [--state-root collection_state]
                                         [--routes-dir DIR] [--scenarios-dir DIR]
                                         [--outdir paper_artifacts] [--no-write]

# =============================================================================
# FIDELITY NOTE — the difficulty model below is REPLICATED VERBATIM from
# manage_continuous.py::ContinuousManager.run_next_job (nested functions).
# It MUST be kept in sync with the scheduler; if the scheduler's scoring
# changes, this file silently goes stale and the paper number is wrong.
# Replicated (line numbers as of this writing):
#   * _route_difficulty      manage_continuous.py ~ lines 997-1042
#   * _WEATHER_DIFF (21 elt) manage_continuous.py ~ lines 1044-1072
#   * _scenario_difficulty   manage_continuous.py ~ lines 1074-1164
#   * _job_difficulty        manage_continuous.py ~ lines 1166-1171
#   * routes_dir/scenarios_dir  manage_continuous.py __init__ ~ lines 43-44
# RECOMMENDED REFACTOR: extract the four functions above into a shared module
# (e.g. difficulty_model.py) and import it from BOTH the scheduler and this
# tool, so there is a single source of truth instead of a copy that can drift.
# =============================================================================
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---- optional acceleration / stats / plotting (ALL guarded) -----------------
try:
    import numpy as _np  # noqa: F401
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

try:
    import scipy.stats as _spstats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as _plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# =============================================================================
# Difficulty model — REPLICATED from manage_continuous.py (see FIDELITY NOTE).
# The scheduler defines these as nested functions of run_next_job with
# per-process memo caches (_cache / _town_cache). We keep the exact arithmetic
# and the memoisation, wrapped in a small class holding routes_dir/scenarios_dir
# to mirror `self.routes_dir` / `self.scenarios_dir`.
# =============================================================================

# manage_continuous.py ~ lines 1050-1072 (copied verbatim, indices unchanged).
# Weather difficulty table indexed by the job's integer `weather` value:
# [ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset,
#  MidRainyNoon, MidRainSunset, WetCloudyNoon, WetCloudySunset,
#  HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset,
#  ClearNight, CloudyNight, WetNight, WetCloudyNight,
#  SoftRainNight, MidRainyNight, HardRainNight]
_WEATHER_DIFF = [
    0.0,  # 0  ClearNoon        — baseline, full visibility
    0.5,  # 1  ClearSunset      — low sun, mild glare
    0.5,  # 2  CloudyNoon
    1.0,  # 3  CloudySunset
    1.5,  # 4  WetNoon          — slippery road surface
    2.0,  # 5  WetSunset
    2.5,  # 6  MidRainyNoon     — rain + reduced visibility
    3.0,  # 7  MidRainSunset
    1.5,  # 8  WetCloudyNoon
    2.0,  # 9  WetCloudySunset
    3.5,  # 10 HardRainNoon     — heavy rain, low visibility
    4.0,  # 11 HardRainSunset
    2.0,  # 12 SoftRainNoon
    2.5,  # 13 SoftRainSunset
    3.0,  # 14 ClearNight       — darkness, perception-heavy
    3.5,  # 15 CloudyNight
    4.0,  # 16 WetNight
    4.5,  # 17 WetCloudyNight
    4.5,  # 18 SoftRainNight
    5.0,  # 19 MidRainyNight
    5.5,  # 20 HardRainNight    — worst case
]


class DifficultyModel:
    """Mirror of the scheduler's difficulty functions (agent-independent)."""

    def __init__(self, routes_dir, scenarios_dir):
        self.routes_dir = Path(routes_dir)
        self.scenarios_dir = Path(scenarios_dir)
        self._route_cache = {}
        self._scen_cache = {}
        self._town_cache = {}

    # ---- manage_continuous.py ~ lines 997-1042 -----------------------------
    def route_difficulty(self, route_name):
        """Parse route XML -> mean per-route geometric difficulty score.

        Score = sharp_turns*2 + path_length_m/500 + total_heading_change_deg/180
        (memoised — each XML parsed at most once per process).
        """
        if route_name in self._route_cache:
            return self._route_cache[route_name]
        xml_path = self.routes_dir / route_name
        try:
            root = ET.parse(str(xml_path)).getroot()
            route_scores = []
            for route_el in root.findall('route'):
                wps = route_el.findall('waypoint')
                if len(wps) < 2:
                    route_scores.append(0.0)
                    continue
                xs = [float(w.get('x', 0.0)) for w in wps]
                ys = [float(w.get('y', 0.0)) for w in wps]
                yaws = [float(w.get('yaw', 0.0)) % 360 for w in wps]

                path_len = sum(
                    math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
                    for i in range(len(xs) - 1)
                )
                heading_deltas = []
                for i in range(1, len(yaws)):
                    d = abs(yaws[i] - yaws[i - 1])
                    if d > 180:
                        d = 360 - d
                    heading_deltas.append(d)
                total_heading = sum(heading_deltas)
                sharp_turns = sum(1 for d in heading_deltas if d > 45)

                route_scores.append(sharp_turns * 2.0 + path_len / 500.0 + total_heading / 180.0)

            score = sum(route_scores) / len(route_scores) if route_scores else 0.0
        except Exception:
            score = 0.0
        self._route_cache[route_name] = score
        return score

    # ---- manage_continuous.py ~ lines 1074-1164 ----------------------------
    def scenario_difficulty(self, route_name):
        """Count adversarial trigger locations within 25 m of the route,
        weighted by mean scenario-type hazard weight. (memoised)."""
        if route_name in self._scen_cache:
            return self._scen_cache[route_name]

        m = re.search(r'town(\d+)', route_name, re.IGNORECASE)
        if not m:
            self._scen_cache[route_name] = 0.0
            return 0.0

        town_tag = f"town{m.group(1).zfill(2)}"

        if town_tag not in self._town_cache:
            TYPE_WEIGHT = {
                'Scenario1':  1.0,
                'Scenario3':  3.0,
                'Scenario4':  2.0,
                'Scenario7':  2.5,
                'Scenario8':  3.5,
                'Scenario9':  4.5,
                'Scenario10': 2.0,
            }
            try:
                spath = self.scenarios_dir / f"{town_tag}_all_scenarios.json"
                with open(str(spath)) as _f:
                    sdata = json.load(_f)
                town_key = list(sdata['available_scenarios'][0].keys())[0]
                slist = sdata['available_scenarios'][0][town_key]

                # Unique positions (all types share the same set — verified).
                evt_pos = list({
                    (round(ev['transform']['x'], 1), round(ev['transform']['y'], 1))
                    for ev in slist[0]['available_event_configurations']
                })

                ws = [TYPE_WEIGHT.get(s['scenario_type'], 2.0) for s in slist]
                mean_w = sum(ws) / len(ws) if ws else 2.0
                self._town_cache[town_tag] = (evt_pos, mean_w)
            except Exception:
                self._town_cache[town_tag] = ([], 2.0)

        evt_pos, mean_w = self._town_cache[town_tag]
        if not evt_pos:
            self._scen_cache[route_name] = 0.0
            return 0.0

        RADIUS = 25.0   # metres — distance from waypoint to trigger location
        SCALE = 0.25    # per hit-cell contribution

        try:
            root = ET.parse(str(self.routes_dir / route_name)).getroot()
            route_scores = []
            for route_el in root.findall('route'):
                wps = [
                    (float(w.get('x', 0.0)), float(w.get('y', 0.0)))
                    for w in route_el.findall('waypoint')
                ]
                hit_cells = set()
                for ex, ey in evt_pos:
                    for wx, wy in wps:
                        if math.hypot(wx - ex, wy - ey) <= RADIUS:
                            hit_cells.add((int(ex // 20), int(ey // 20)))
                            break   # one waypoint match is enough per event
                route_scores.append(len(hit_cells) * mean_w * SCALE)
            score = sum(route_scores) / len(route_scores) if route_scores else 0.0
        except Exception:
            score = 0.0

        self._scen_cache[route_name] = score
        return score

    # ---- manage_continuous.py ~ lines 1166-1171 ----------------------------
    def weather_difficulty(self, weather):
        weather_idx = int(weather)
        return _WEATHER_DIFF[weather_idx] if weather_idx < len(_WEATHER_DIFF) else 2.5

    def job_difficulty(self, job):
        """_job_difficulty(job) = route + scenario + weather (verbatim)."""
        route = job.get('route', '')
        route_score = self.route_difficulty(route) + self.scenario_difficulty(route)
        weather_score = self.weather_difficulty(job.get('weather', 0))
        return route_score + weather_score

    def components(self, job):
        """Return the three difficulty components separately, for the breakdown."""
        route = job.get('route', '')
        return {
            "route": self.route_difficulty(route),
            "scenario": self.scenario_difficulty(route),
            "weather": self.weather_difficulty(job.get('weather', 0)),
        }


# =============================================================================
# Data loading — union completed_jobs.json + completed jobs in job_queue.json,
# deduped by job id (both files carry the same completed records today).
# =============================================================================
def _load_jobs(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    return jobs if isinstance(jobs, list) else []


def _has_score(job):
    return isinstance(job.get("score_composed"), (int, float)) and not _isnan(job["score_composed"])


def _isnan(x):
    try:
        return math.isnan(x)
    except Exception:
        return False


def load_completed(state_root):
    """Union both sources, keep status=='completed', dedupe by id.

    On duplicate id, prefer the record that actually carries a numeric
    score_composed (so a bare queue stub never shadows a scored completion).
    """
    completed_file = os.path.join(state_root, "completed_jobs.json")
    queue_file = os.path.join(state_root, "job_queue.json")

    by_id = {}
    for src in (completed_file, queue_file):
        for job in _load_jobs(src):
            if job.get("status") != "completed":
                continue
            jid = job.get("id")
            key = jid if jid is not None else id(job)
            if key not in by_id:
                by_id[key] = job
            elif not _has_score(by_id[key]) and _has_score(job):
                by_id[key] = job
    return list(by_id.values())


def _to_float(x):
    """Parse a CSV cell to float, or None (empty / non-numeric)."""
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def load_per_route(csv_path):
    """Load a harvested per-route CSV (tools/harvest_results.py output) as
    job-like dicts, so build_records() computes difficulty-vs-performance at
    PER-ROUTE granularity — one data point per route eval — instead of one
    point per completed FILE. This yields far more points: on the current data
    the harvester recovers ~200 route-evals (most from queue-'failed' jobs)
    versus only a handful of completed file-level jobs.

    The harvested CSV has one row per route eval; we map its columns onto the
    same shape build_records() consumes. Crucially 'route_file' -> 'route',
    because the DifficultyModel keys route/scenario difficulty off the route XML
    filename — so every route eval within a file shares that file's difficulty,
    paired here with its own per-route score_composed. Rows without a numeric
    score_composed are dropped downstream by _has_score(). Returns [] on any
    read problem (mirrors load_completed's graceful failure).
    """
    jobs = []
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                weather = row.get("weather")
                try:
                    weather = int(weather)
                except (TypeError, ValueError):
                    pass
                jobs.append({
                    # composite id keeps each route eval distinct in the appendix
                    "id": f"{row.get('agent', '')}|{row.get('route_file', '')}"
                          f"|w{row.get('weather', '')}|{row.get('route_id', '')}",
                    "agent": row.get("agent", "unknown"),
                    "route": row.get("route_file", ""),
                    "town": row.get("town", ""),
                    "weather": weather,
                    "score_composed": _to_float(row.get("score_composed")),
                    "score_route": _to_float(row.get("score_route")),
                })
    except Exception:
        return []
    return jobs


# =============================================================================
# Correlation statistics — pure-Python Spearman/Pearson + t-approx p-values.
# Prefer scipy.stats when importable; otherwise use the hand-rolled versions.
# Every function returns None (never raises) on degenerate input.
# =============================================================================
def _mean(xs):
    return sum(xs) / len(xs)


def _rankdata(vals):
    """Average (fractional) ranks, 1-based, ties share the mean of their ranks."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i, n = 0, len(vals)
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson_r(xs, ys):
    """Pearson r, or None if n<2 or either variable has zero variance."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy)
    return max(-1.0, min(1.0, r))


def _spearman_r(xs, ys):
    if len(xs) < 2:
        return None
    return _pearson_r(_rankdata(xs), _rankdata(ys))


# ---- Student-t two-sided p-value for a correlation coefficient --------------
# p = I_{df/(df+t^2)}(df/2, 1/2) with t = r*sqrt(df/(1-r^2)), df = n-2.
# Uses the regularized incomplete beta function (Numerical Recipes betacf).
def _gammaln(x):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _gammaln(a + b) - _gammaln(a) - _gammaln(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _pval_from_r(r, n):
    """Two-sided p-value for H0: rho=0 via the t-approximation. df = n-2.

    This is the same approximation scipy uses for pearsonr and (by default) for
    spearmanr; it is only reliable for n greater than ~10 — small-n callers get a
    caveat elsewhere. Returns None when df<=0.
    """
    if r is None:
        return None
    df = n - 2
    if df <= 0:
        return None
    if abs(r) >= 1.0:
        return 0.0
    t2 = r * r * df / (1.0 - r * r)  # = t^2
    return _betai(0.5 * df, 0.5, df / (df + t2))


def correlate(xs, ys, method):
    """Return (r, p, n, backend). Never raises; None r/p on degenerate input."""
    n = len(xs)
    if HAVE_SCIPY and n >= 3:
        try:
            if method == "spearman":
                res = _spstats.spearmanr(xs, ys)
                r = float(getattr(res, "correlation", res[0]))
                p = float(getattr(res, "pvalue", res[1]))
            else:
                res = _spstats.pearsonr(xs, ys)
                r, p = float(res[0]), float(res[1])
            if not (_isnan(r) or _isnan(p)):
                return r, p, n, "scipy"
        except Exception:
            pass  # fall through to pure-python
    r = _spearman_r(xs, ys) if method == "spearman" else _pearson_r(xs, ys)
    p = _pval_from_r(r, n)
    return r, p, n, "builtin"


# =============================================================================
# Variation / small-n guards
# =============================================================================
def _pstdev(xs):
    if len(xs) < 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def variation_note(xs):
    """Classify the variation in a difficulty vector.

    Returns (is_constant, is_low, detail_str). 'is_constant' means the
    correlation is mathematically undefined (zero variance).
    """
    if not xs:
        return True, True, "no data"
    distinct = sorted(set(round(x, 9) for x in xs))
    sd = _pstdev(xs)
    m = _mean(xs)
    cov = (sd / abs(m)) if m else float("inf")
    if len(distinct) <= 1:
        return True, True, "constant (zero variance)"
    is_low = len(distinct) < 3 or cov < 0.05
    detail = f"{len(distinct)} distinct values, stdev={sd:.4f}, CoV={cov:.3f}"
    return False, is_low, detail


# =============================================================================
# Analysis
# =============================================================================
def build_records(jobs, model):
    """Attach difficulty + components to each scored completed job."""
    recs = []
    for j in jobs:
        if not _has_score(j):
            continue
        comp = model.components(j)
        recs.append({
            "id": j.get("id"),
            "agent": j.get("agent", "unknown"),
            "route": j.get("route", ""),
            "town": j.get("town", ""),
            "weather": j.get("weather"),
            "score_composed": float(j["score_composed"]),
            "score_route": j.get("score_route"),
            "difficulty": comp["route"] + comp["scenario"] + comp["weather"],
            "d_route": comp["route"],
            "d_scenario": comp["scenario"],
            "d_weather": comp["weather"],
        })
    return recs


def analyze(recs):
    """Compute pooled + per-agent (difficulty vs score) and pooled component corrs."""
    result = {"pooled": {}, "per_agent": {}, "components": {}, "n_scored": len(recs)}

    ys = [r["score_composed"] for r in recs]
    xs = [r["difficulty"] for r in recs]

    # pooled difficulty vs score
    const, low, detail = variation_note(xs)
    sp = correlate(xs, ys, "spearman")
    pe = correlate(xs, ys, "pearson")
    result["pooled"] = {
        "n": len(recs), "spearman": sp, "pearson": pe,
        "x_constant": const, "x_low_var": low, "x_detail": detail,
    }

    # per-agent
    agents = sorted(set(r["agent"] for r in recs))
    for a in agents:
        sub = [r for r in recs if r["agent"] == a]
        axs = [r["difficulty"] for r in sub]
        ays = [r["score_composed"] for r in sub]
        c, l, d = variation_note(axs)
        result["per_agent"][a] = {
            "n": len(sub),
            "spearman": correlate(axs, ays, "spearman"),
            "pearson": correlate(axs, ays, "pearson"),
            "x_constant": c, "x_low_var": l, "x_detail": d,
        }

    # pooled per-component vs score
    for comp_key, label in (("d_route", "route"), ("d_scenario", "scenario"),
                            ("d_weather", "weather"), ("difficulty", "total")):
        cxs = [r[comp_key] for r in recs]
        const_c, low_c, detail_c = variation_note(cxs)
        result["components"][label] = {
            "n": len(recs),
            "spearman": correlate(cxs, ys, "spearman"),
            "pearson": correlate(cxs, ys, "pearson"),
            "x_constant": const_c, "x_low_var": low_c, "x_detail": detail_c,
        }
    return result


# =============================================================================
# Rendering
# =============================================================================
def _fmt_r(v):
    return " n/a " if v is None else f"{v:+.3f}"


def _fmt_p(v):
    if v is None:
        return " n/a "
    if v < 1e-4:
        return "<1e-4"
    return f"{v:.4f}"


def _corr_row(label, n, sp, pe, note):
    # sp / pe are (r, p, n, backend) tuples
    return (f"| {label} | {n} | {_fmt_r(sp[0])} | {_fmt_p(sp[1])} | "
            f"{_fmt_r(pe[0])} | {_fmt_p(pe[1])} | {note} |")


def render_markdown(recs, result, backend_note, plot_path,
                    granularity="completed file-level job"):
    L = []
    A = L.append
    A("# Difficulty vs. Performance Validation")
    A("")
    A("Does the scheduler's **difficulty score** predict actual agent "
      "performance? The prune step assumes it does (an agent clearing a hard "
      "condition clears the easy one), so difficulty should **anti-correlate** "
      "with `score_composed` (the 0-100 leaderboard driving score). "
      "**Expected sign: NEGATIVE** (harder -> lower score).")
    A("")
    A(f"- Analysis unit: **{granularity}**")
    A(f"- Scored records analysed: **{result['n_scored']}**")
    A(f"- Stats backend: **{backend_note}**")
    A(f"- p-values: two-sided, Student-t approximation (df=n-2); reliable only "
      f"for n larger than ~10.")
    A("")

    if not recs:
        A("> **No scored completed jobs found.** Re-run once collection has "
          "produced completions with a numeric `score_composed`.")
        return "\n".join(L)

    # ---- pooled + per-agent difficulty vs score ----
    A("## Difficulty vs. score_composed")
    A("")
    A("| scope | n | Spearman r | Spearman p | Pearson r | Pearson p | difficulty variation |")
    A("|---|--:|--:|--:|--:|--:|---|")
    p = result["pooled"]
    A(_corr_row("**pooled (all agents)**", p["n"], p["spearman"], p["pearson"], p["x_detail"]))
    for a in sorted(result["per_agent"]):
        s = result["per_agent"][a]
        A(_corr_row(a, s["n"], s["spearman"], s["pearson"], s["x_detail"]))
    A("")

    # ---- component breakdown ----
    A("## Which component drives the signal? (pooled, vs score_composed)")
    A("")
    A("| component | n | Spearman r | Spearman p | Pearson r | Pearson p | variation |")
    A("|---|--:|--:|--:|--:|--:|---|")
    for label in ("route", "scenario", "weather", "total"):
        c = result["components"][label]
        A(_corr_row(label, c["n"], c["spearman"], c["pearson"], c["x_detail"]))
    A("")

    # ---- interpretation ----
    pooled_sp = p["spearman"][0]
    A("## Interpretation")
    A("")
    if pooled_sp is None:
        A("- Pooled Spearman is **undefined** — the difficulty values have no "
          "usable variation across the scored jobs, so no monotonic "
          "relationship can be measured.")
    else:
        direction = ("NEGATIVE (harder -> worse) — consistent with the prune "
                     "assumption" if pooled_sp < 0 else
                     "POSITIVE/zero (harder -> same-or-better) — does NOT yet "
                     "support the prune assumption")
        A(f"- Pooled Spearman r = **{pooled_sp:+.3f}**: {direction}.")
    A("")

    # ---- caveats / small-n handling ----
    caveats = []
    if result["n_scored"] < 10:
        caveats.append(
            f"Pooled n = {result['n_scored']} (< 10): far too few completions "
            f"for a significant estimate.")
    if p["x_low_var"]:
        caveats.append(
            f"Pooled difficulty has near-zero variation ({p['x_detail']}): the "
            f"correlation is dominated by a handful of distinct route/weather "
            f"conditions and is not yet meaningful.")
    wc = result["components"]["weather"]
    if wc["x_constant"]:
        caveats.append(
            "Weather-difficulty is **constant** across all scored jobs (early "
            "data is all weather=20 -> fixed 5.5), so weather-vs-score is "
            "undefined; only route/scenario components vary. This resolves as "
            "collection samples more weather presets.")
    small_agents = [a for a, s in result["per_agent"].items() if s["n"] < 10]
    if small_agents:
        caveats.append(
            "Per-agent n < 10 for: " + ", ".join(
                f"{a} (n={result['per_agent'][a]['n']})" for a in sorted(small_agents)
            ) + " — per-agent r/p are not interpretable yet.")

    if caveats:
        A("## Caveats — insufficient variation / data for significance")
        A("")
        A("> **Re-run this tool as collection grows.** The numbers above are "
          "recorded for provenance but should not be cited until these clear:")
        A("")
        for c in caveats:
            A(f"- {c}")
        A("")

    # ---- per-record appendix (small n -> show every point) ----
    if len(recs) <= 40:
        A("## Scored jobs used (appendix)")
        A("")
        A("| id | agent | route | weather | difficulty | route | scenario | weather | score_composed |")
        A("|--:|---|---|--:|--:|--:|--:|--:|--:|")
        for r in sorted(recs, key=lambda r: (r["agent"], r["route"])):
            A(f"| {r['id']} | {r['agent']} | {r['route']} | {r['weather']} | "
              f"{r['difficulty']:.4f} | {r['d_route']:.4f} | {r['d_scenario']:.4f} | "
              f"{r['d_weather']:.4f} | {r['score_composed']:.4f} |")
        A("")

    if plot_path:
        A(f"Scatter plot written to `{plot_path}`.")
    elif not HAVE_MPL:
        A("_matplotlib not available — scatter plot skipped._")
    A("")
    return "\n".join(L)


def make_scatter(recs, result, out_png):
    """Scatter: difficulty (x) vs score_composed (y), colour per agent, pooled
    trend line. Returns the path written, or None on any failure / no matplotlib."""
    if not HAVE_MPL or not recs:
        return None
    try:
        fig, ax = _plt.subplots(figsize=(7, 5))
        agents = sorted(set(r["agent"] for r in recs))
        cmap = _plt.get_cmap("tab10")
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for i, a in enumerate(agents):
            sub = [r for r in recs if r["agent"] == a]
            ax.scatter([r["difficulty"] for r in sub],
                       [r["score_composed"] for r in sub],
                       color=cmap(i % 10), marker=markers[i % len(markers)],
                       s=70, edgecolor="black", linewidth=0.4, label=a, zorder=3)
        # pooled least-squares trend line (only if x has variance)
        xs = [r["difficulty"] for r in recs]
        ys = [r["score_composed"] for r in recs]
        if _pstdev(xs) > 0 and len(xs) >= 2:
            mx, my = _mean(xs), _mean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            a0 = my - b * mx
            xr = [min(xs), max(xs)]
            ax.plot(xr, [a0 + b * x for x in xr], "--", color="gray",
                    linewidth=1.5, zorder=2, label="pooled trend")
        sp = result["pooled"]["spearman"][0]
        sub = "" if sp is None else f"  (pooled Spearman r={sp:+.3f})"
        ax.set_xlabel("scheduler difficulty score")
        ax.set_ylabel("score_composed (leaderboard driving score)")
        ax.set_title("Difficulty vs. agent performance" + sub)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=130)
        _plt.close(fig)
        return out_png
    except Exception as e:
        sys.stderr.write(f"[warn] scatter plot failed: {e}\n")
        return None


# =============================================================================
def main(argv):
    ap = argparse.ArgumentParser(
        description="Validate difficulty-score vs agent-performance correlation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    ap.add_argument("--project-root", default=default_root,
                    help="repo root (mirrors ContinuousManager PROJECT_ROOT)")
    ap.add_argument("--state-root", default=None,
                    help="collection_state dir (default: <project-root>/collection_state)")
    ap.add_argument("--routes-dir", default=None,
                    help="override routes dir (default mirrors manager: "
                         "<project-root>/leaderboard/data/training_routes)")
    ap.add_argument("--scenarios-dir", default=None,
                    help="override scenarios dir (default mirrors manager: "
                         "<project-root>/leaderboard/data/scenarios)")
    ap.add_argument("--outdir", default="paper_artifacts",
                    help="where to write the .md report / .png (default paper_artifacts)")
    ap.add_argument("--per-route", default=None, metavar="CSV",
                    help="harvested per-route CSV (tools/harvest_results.py output). "
                         "When given, compute difficulty-vs-performance at PER-ROUTE "
                         "granularity (one point per route eval, far more data) from "
                         "this file instead of the default per-file completed jobs.")
    ap.add_argument("--no-write", action="store_true",
                    help="print the report only; do not write files")
    args = ap.parse_args(argv)

    proj = Path(args.project_root)
    state_root = args.state_root or str(proj / "collection_state")
    routes_dir = args.routes_dir or str(proj / "leaderboard/data/training_routes")
    scenarios_dir = args.scenarios_dir or str(proj / "leaderboard/data/scenarios")

    model = DifficultyModel(routes_dir, scenarios_dir)
    # Default: one point per completed FILE-level job. With --per-route: one
    # point per harvested route eval (far more data, incl. queue-'failed' jobs).
    if args.per_route:
        jobs = load_per_route(args.per_route)
        granularity = "per-route eval (harvested)"
    else:
        jobs = load_completed(state_root)
        granularity = "completed file-level job"
    recs = build_records(jobs, model)
    result = analyze(recs)

    backend = ("scipy" if HAVE_SCIPY else "pure-python (scipy absent)") + \
              (", numpy present" if HAVE_NUMPY else ", numpy absent")

    plot_path = None
    if not args.no_write and recs:
        plot_path = make_scatter(recs, result, os.path.join(args.outdir, "difficulty_vs_score.png"))

    md = render_markdown(recs, result, backend, plot_path, granularity)
    print(md)

    if not args.no_write:
        os.makedirs(args.outdir, exist_ok=True)
        md_path = os.path.join(args.outdir, "difficulty_validation.md")
        with open(md_path, "w") as f:
            f.write(md + "\n")
        sys.stderr.write(f"\nWrote {md_path}"
                         + (f" and {plot_path}" if plot_path else "") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
