#!/usr/bin/env python3
"""Per-axis x per-agent difficulty-SENSITIVITY matrix.

Instead of one scalar difficulty correlated against performance (which fuses
axes that hit different architectures differently — see the A100 finding that a
single difficulty washes out to Spearman +0.036), this fits, PER AGENT, a
noisy-OR / competing-hazards model of route failure:

    P(fail | route, agent) = 1 - exp( - sum_j  lambda_j(agent) * x_j )

where x_j are the normalised difficulty axes (route geometry, scenario density,
and the physically-decomposed weather axes illum_dark / precip / road_water /
cloud / fog from tools/weather_axes.py). Each lambda_j(agent) >= 0 is that axis's
hazard weight for that agent. The fit (Newton MLE + inverse-Hessian CIs) was
validated on synthetic data with known ground truth: it recovers per-model axis
sensitivity and cleanly separates, e.g., illumination-sensitive camera models
from illumination-insensitive LiDAR models — the property a scalar cannot show.

IDENTIFIABILITY IS REPORTED FIRST. An axis with ~zero variance in the sample
(e.g. illum_dark when every eval is at night) is UNIDENTIFIABLE and excluded from
the fit. Axes that co-move almost perfectly (|r|>0.95, e.g. precip and fog when
only heavy-rain-night presets ran) are CONFOUNDED: a fitted weight is really the
group's joint effect, not that axis's. On the current night-only sample most
weather axes fall into one of these buckets — which is exactly why the scheduler
needs illumination-stratified coverage (see manage_continuous.py).

Pure stdlib (no numpy/scipy). Reuses DifficultyModel (geometry/scenario) from
tools/difficulty_validation.py and the axis table from tools/weather_axes.py.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_axes  # noqa: E402
from difficulty_validation import DifficultyModel  # noqa: E402

# Fit axes: route geometry + scenario density (from the route XML), then the
# physical weather axes (already normalised [0,1] by the harvester).
GEOM_AXES = ["geom", "scen"]
WX_AXES = ["illum_dark", "precip", "road_water", "cloud", "fog"]
FIT_AXES = GEOM_AXES + WX_AXES

CONFOUND_R = 0.95     # |corr| above which two axes are treated as unseparable
VAR_EPS = 1e-9        # variance below which an axis is "constant" (unidentifiable)


# ----------------------------------------------------------------- data load
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(csv_path, model):
    """Read the harvested per-route CSV -> list of {agent, feat{axis:val}, fail}.

    geom/scen come from the route XML via DifficultyModel; the weather axes are
    read from the harvester's columns, falling back to recomputing from the
    weather index if an older CSV lacks them.
    """
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            route = r.get("route_file", "")
            feat = {
                "geom": model.route_difficulty(route),
                "scen": model.scenario_difficulty(route),
            }
            have_wx = all(r.get(k) not in (None, "") for k in WX_AXES)
            if have_wx:
                for k in WX_AXES:
                    feat[k] = _num(r.get(k))
            else:  # older CSV: recompute from the weather index
                try:
                    ax = weather_axes.axes(int(r.get("weather")))
                    for k in WX_AXES:
                        feat[k] = ax[k]
                except Exception:
                    for k in WX_AXES:
                        feat[k] = None
            if any(feat[k] is None for k in FIT_AXES):
                continue
            status = (r.get("status") or "").strip().lower()
            fail = 0.0 if status.startswith("completed") else 1.0
            rows.append({"agent": r.get("agent", "?"), "feat": feat, "fail": fail})
    return rows


def normalize_geom(rows):
    """Min-max the raw geom/scen axes to [0,1] (weather axes already normalised),
    so lambda are comparable across axes and agents."""
    for a in GEOM_AXES:
        vals = [row["feat"][a] for row in rows]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        for row in rows:
            row["feat"][a] = (row["feat"][a] - lo) / span


# ------------------------------------------------------------- linear algebra
def _solve_inv(H, g):
    n = len(H)
    A = [H[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        d = A[col][col] or 1e-12
        A[col] = [v / d for v in A[col]]
        for r in range(n):
            if r != col and A[r][col]:
                fr = A[r][col]
                A[r] = [A[r][k] - fr * A[col][k] for k in range(2 * n)]
    inv = [[A[i][n + j] for j in range(n)] for i in range(n)]
    x = [sum(inv[i][j] * g[j] for j in range(n)) for i in range(n)]
    return x, inv


# --------------------------------------------------------- noisy-OR Newton MLE
def _nll(lam, X, y):
    tot = 0.0
    for xi, yi in zip(X, y):
        eta = min(max(sum(lam[j] * xi[j] for j in range(len(lam))), 1e-9), 50.0)
        pf = min(max(1.0 - math.exp(-eta), 1e-12), 1 - 1e-12)
        tot -= yi * math.log(pf) + (1 - yi) * math.log(1 - pf)
    return tot


def fit_noisy_or(X, y, iters=60, ridge=1e-4):
    """Newton MLE of lambda>=0 for P(fail)=1-exp(-X.lambda). Returns (lam, se)."""
    k = len(X[0])
    lam = [0.3] * k
    inv = [[0.0] * k for _ in range(k)]
    for _ in range(iters):
        G = [0.0] * k
        H = [[0.0] * k for _ in range(k)]
        for xi, yi in zip(X, y):
            eta = min(max(sum(lam[j] * xi[j] for j in range(k)), 1e-9), 50.0)
            s = math.exp(-eta); pf = min(max(1.0 - s, 1e-9), 1 - 1e-9)
            gi = (1.0 - yi) - yi * s / pf
            hi = yi * s / (pf * pf)
            for j in range(k):
                G[j] += gi * xi[j]
                if hi:
                    row = H[j]; xij = xi[j]
                    for kk in range(k):
                        row[kk] += hi * xij * xi[kk]
        for j in range(k):
            H[j][j] += ridge
        step, inv = _solve_inv(H, G)
        base = _nll(lam, X, y); alpha = 1.0
        for _bt in range(24):
            cand = [max(0.0, lam[j] - alpha * step[j]) for j in range(k)]
            if _nll(cand, X, y) <= base + 1e-9:
                lam = cand; break
            alpha *= 0.5
        else:
            lam = [max(0.0, lam[j] - alpha * step[j]) for j in range(k)]
    se = [math.sqrt(inv[j][j]) if inv[j][j] > 0 else float("nan") for j in range(k)]
    return lam, se


# ------------------------------------------------------------- diagnostics
def _stdev(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def analyze_agent(rows, min_n):
    """Fit one agent. Returns a dict describing per-axis identifiability + lambda."""
    n = len(rows)
    fr = sum(r["fail"] for r in rows) / n if n else 0.0
    out = {"n": n, "fail_rate": fr, "axes": {a: {} for a in FIT_AXES}}
    if n < min_n:
        out["skip"] = f"n={n} < min_n={min_n}"
        return out
    if fr <= 0.0 or fr >= 1.0:
        out["skip"] = f"no outcome variation (fail_rate={fr:.2f})"
        return out

    cols = {a: [r["feat"][a] for r in rows] for a in FIT_AXES}
    std = {a: _stdev(cols[a]) for a in FIT_AXES}
    active = [a for a in FIT_AXES if std[a] > VAR_EPS]
    for a in FIT_AXES:
        if std[a] <= VAR_EPS:
            out["axes"][a] = {"status": "unident", "reason": "constant in sample"}

    # confounding among active axes (max |corr| to another active axis)
    confound = {}
    for a in active:
        mx = 0.0; partner = None
        for b in active:
            if b == a:
                continue
            r = abs(_corr(cols[a], cols[b]))
            if r > mx:
                mx, partner = r, b
        confound[a] = (mx, partner)

    # fit on active axes only (keeps the Hessian well-conditioned)
    X = [[r["feat"][a] for a in active] for r in rows]
    y = [r["fail"] for r in rows]
    lam, se = fit_noisy_or(X, y)
    for idx, a in enumerate(active):
        mxr, partner = confound[a]
        entry = {"lambda": lam[idx], "se": se[idx],
                 "ci": (lam[idx] - 1.96 * se[idx], lam[idx] + 1.96 * se[idx]),
                 "max_r": mxr, "partner": partner}
        entry["status"] = "confounded" if mxr > CONFOUND_R else "ok"
        # Significant only if the whole 95% CI is positive (λ>=0, so ci_lo>0 means
        # the axis provably raises this agent's failure hazard). Confounded weights
        # are a joint effect, so they don't count as an individual-axis result.
        entry["significant"] = (entry["status"] == "ok" and entry["ci"][0] > 0.0)
        out["axes"][a] = entry
    return out


# ------------------------------------------------------------------- render
def _cell(entry):
    st = entry.get("status")
    if st == "unident":
        return "  —  "
    if st in ("ok", "confounded"):
        s = f"{entry['lambda']:.2f}"
        return s + ("*" if st == "confounded" else " ")
    return "  ?  "


def render(results, out_path=None):
    lines = []
    P = lines.append
    P("# Per-axis × per-agent difficulty-sensitivity matrix\n")
    P("Fitted hazard weights **λ_axis(agent)** from a per-agent noisy-OR failure "
      "model (higher λ = that axis drives *this* agent's failures harder). "
      "`—` = unidentifiable (axis constant in the sample); `*` = confounded "
      "(|r|>0.95 with another axis — the weight is the group's joint effect).\n")

    agents = sorted(results)
    header = "| agent | n | fail% | " + " | ".join(FIT_AXES) + " |"
    P(header)
    P("|" + "---|" * (len(FIT_AXES) + 3))
    for ag in agents:
        r = results[ag]
        if r.get("skip"):
            P(f"| {ag} | {r['n']} | {r['fail_rate']*100:.0f} | "
              + " | ".join([f"_{r['skip']}_"] + [""] * (len(FIT_AXES) - 1)) + " |")
            continue
        cells = [_cell(r["axes"][a]) for a in FIT_AXES]
        P(f"| {ag} | {r['n']} | {r['fail_rate']*100:.0f} | " + " | ".join(cells) + " |")

    # identifiability + significance summary
    P("\n## Identifiability & significance in this sample\n")
    fitted = [(ag, a) for ag in agents if not results[ag].get("skip")
              for a in FIT_AXES if results[ag]["axes"][a].get("status") in ("ok", "confounded")]
    n_sig_total = sum(1 for ag, a in fitted if results[ag]["axes"][a].get("significant"))
    P(f"**{n_sig_total} of {len(fitted)} (agent × axis) fits are statistically significant** "
      f"(95% CI excludes 0). `identified` below means *estimable*, not precise — a wide CIs that "
      f"straddles 0 is directional only, and tightens with more per-agent coverage.\n")
    any_sig = n_sig_total > 0
    for a in FIT_AXES:
        entries = [results[ag]["axes"][a] for ag in agents if not results[ag].get("skip")]
        n_ok = sum(1 for e in entries if e.get("status") == "ok")
        n_conf = sum(1 for e in entries if e.get("status") == "confounded")
        n_uni = sum(1 for e in entries if e.get("status") == "unident")
        n_sig = sum(1 for e in entries if e.get("significant"))
        tag = ("**identified**" if n_ok else
               ("confounded" if n_conf else "unidentified"))
        P(f"- `{a}`: {tag} — ok={n_ok} (**significant={n_sig}**), confounded={n_conf}, unident={n_uni}")
    if not any_sig:
        P("\n> **No axis is significant for any agent yet.** The axes are now estimable "
          "(illumination-stratified coverage broke the night-only collinearity), but the "
          "per-agent samples are still too small — CIs straddle 0. More coverage, especially "
          "for the thinly-sampled LiDAR agent(s), is needed before any per-model claim.")

    # per-agent CIs (detail)
    P("\n## λ with 95% CIs (identified/confounded axes)\n")
    for ag in agents:
        r = results[ag]
        if r.get("skip"):
            P(f"- **{ag}**: skipped ({r['skip']})")
            continue
        parts = []
        for a in FIT_AXES:
            e = r["axes"][a]
            if e.get("status") in ("ok", "confounded"):
                lo, hi = e["ci"]
                mark = f" (confounded w/ {e['partner']}, r={e['max_r']:.2f})" \
                    if e["status"] == "confounded" else ""
                parts.append(f"{a}={e['lambda']:.2f} [{lo:.2f},{hi:.2f}]{mark}")
        P(f"- **{ag}** (n={r['n']}, fail={r['fail_rate']*100:.0f}%): "
          + ("; ".join(parts) if parts else "_no identifiable axis_"))

    text = "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text


# --------------------------------------------------------------------- main
def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    proj = os.environ.get("PROJECT_ROOT", os.getcwd())
    ap.add_argument("--csv", default="paper_artifacts/per_route_results.csv",
                    help="harvested per-route CSV (tools/harvest_results.py output)")
    ap.add_argument("--routes-dir", default=os.path.join(proj, "leaderboard/data/training_routes"))
    ap.add_argument("--scenarios-dir", default=os.path.join(proj, "leaderboard/data/scenarios"))
    ap.add_argument("--out", default="paper_artifacts/sensitivity_matrix.md",
                    help="markdown output (default: paper_artifacts/sensitivity_matrix.md)")
    ap.add_argument("--min-n", type=int, default=20,
                    help="skip agents with fewer than this many route-evals")
    ap.add_argument("--no-write", action="store_true", help="print only; do not write")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.csv):
        print(f"error: {args.csv} not found — run tools/harvest_results.py first", file=sys.stderr)
        return 2

    model = DifficultyModel(args.routes_dir, args.scenarios_dir)
    rows = load_rows(args.csv, model)
    if not rows:
        print("No usable rows in the CSV.", file=sys.stderr)
        return 1
    normalize_geom(rows)

    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r)
    results = {ag: analyze_agent(rs, args.min_n) for ag, rs in by_agent.items()}

    text = render(results, None if args.no_write else args.out)
    print(text)
    if not args.no_write:
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
