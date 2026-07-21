#!/usr/bin/env python3
"""Fair head-to-head: does the MULTI-AXIS difficulty representation predict real
agent failures better than the OLD SCALAR difficulty — out of sample?

The scalar washing out (pooled Spearman +0.036) shows the old system is weak; it
does NOT show the new one is good. This tests the new one directly, on the real
harvested data, with k-fold cross-validation so the multi-axis model's extra
parameters can't win by overfitting.

For each agent (and pooled), predict per-route FAILURE (status != Completed) with
a noisy-OR model from two representations:
  * SCALAR   : the old difficulty  = route_geom + scenario + weather_diff  (1 feature)
  * MULTI    : [geom, scen, illum_dark, precip, road_water, cloud, fog]     (7 features)
and compare held-out log-loss (lower=better) and AUC (higher=better). A NULL model
(predict the base failure rate) is the floor.

Pure stdlib; reuses tools/sensitivity_matrix.py (fit) + weather_axes + difficulty_validation.
"""
import argparse, csv, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weather_axes
from difficulty_validation import DifficultyModel, _WEATHER_DIFF
from sensitivity_matrix import fit_noisy_or, WX_AXES

MULTI_AXES = ["geom", "scen"] + WX_AXES
FOLDS = 5


def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def load(csv_path, model):
    rows = []
    # Memoize the route/scenario difficulty (parses route XML) by route file and the
    # weather axes by index — both are O(distinct) not O(rows). Without this the parse
    # ran once per route-eval (13k×) and the tool timed out at full n; with ~27 route
    # files and 21 weathers it now runs in seconds. (verify #27: vectorize to full 13k)
    _rd_cache = {}
    _ax_cache = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            route = r.get("route_file", "")
            try: w = int(r.get("weather"))
            except (TypeError, ValueError): continue
            if route not in _rd_cache:
                _rd_cache[route] = (model.route_difficulty(route), model.scenario_difficulty(route))
            geom, scen = _rd_cache[route]
            wd = _WEATHER_DIFF[w] if 0 <= w < len(_WEATHER_DIFF) else 2.5
            if w not in _ax_cache:
                _ax_cache[w] = weather_axes.axes(w)
            ax = _ax_cache[w]
            feat = {"geom": geom, "scen": scen,
                    "scalar": geom + scen + wd,               # the OLD difficulty
                    "illum_dark": ax["illum_dark"], "precip": ax["precip"],
                    "road_water": ax["road_water"], "cloud": ax["cloud"], "fog": ax["fog"]}
            status = (r.get("status") or "").strip().lower()
            fail = 0.0 if status.startswith("completed") else 1.0
            key = f"{r.get('agent')}|{route}|w{w}|{r.get('route_id')}"
            fold = sum(ord(c) for c in key) % FOLDS      # deterministic CV fold
            rows.append({"agent": r.get("agent", "?"), "feat": feat, "fail": fail, "fold": fold})
    return rows


def normalize(rows, keys):
    for k in keys:
        vs = [r["feat"][k] for r in rows]
        lo, hi = min(vs), max(vs); span = (hi - lo) or 1.0
        for r in rows:
            r["feat"][k] = (r["feat"][k] - lo) / span


def _auc(pairs):
    """Mann-Whitney AUC from (pred, label) pairs; None if one class absent."""
    pos = [p for p, y in pairs if y == 1.0]; neg = [p for p, y in pairs if y == 0.0]
    if not pos or not neg: return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs); i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[order[j + 1]][0] == pairs[order[i]][0]: j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): ranks[order[k]] = avg
        i = j + 1
    rpos = sum(ranks[i] for i in range(len(pairs)) if pairs[i][1] == 1.0)
    n1, n0 = len(pos), len(neg)
    return (rpos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _logloss(pairs):
    tot = 0.0
    for p, y in pairs:
        p = min(max(p, 1e-9), 1 - 1e-9)
        tot -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return tot / len(pairs) if pairs else float("nan")


def cv_predict(rows, keys):
    """5-fold CV; return held-out (pred, label) pairs for the given feature keys."""
    pairs = []
    for f in range(FOLDS):
        train = [r for r in rows if r["fold"] != f]
        test = [r for r in rows if r["fold"] == f]
        if not test: continue
        if not train or len({r["fail"] for r in train}) < 2:
            base = sum(r["fail"] for r in train) / len(train) if train else 0.5
            pairs += [(base, r["fail"]) for r in test]; continue
        X = [[r["feat"][k] for k in keys] for r in train]; y = [r["fail"] for r in train]
        lam, _ = fit_noisy_or(X, y)
        for r in test:
            eta = sum(lam[i] * r["feat"][k] for i, k in enumerate(keys))
            pairs.append((1.0 - math.exp(-min(max(eta, 0.0), 50.0)), r["fail"]))
    return pairs


def evaluate(rows, label):
    n = len(rows); fr = sum(r["fail"] for r in rows) / n if n else 0.0
    null_pairs = [(fr, r["fail"]) for r in rows]
    sc = cv_predict(rows, ["scalar"])
    mu = cv_predict(rows, MULTI_AXES)
    ll_null, ll_sc, ll_mu = _logloss(null_pairs), _logloss(sc), _logloss(mu)
    auc_sc, auc_mu = _auc(sc), _auc(mu)
    def fa(a): return "  n/a" if a is None else f"{a:.3f}"
    print(f"  {label:12s} n={n:4d} fail={fr*100:4.0f}%  |  "
          f"logloss null={ll_null:.3f} scalar={ll_sc:.3f} multi={ll_mu:.3f}  |  "
          f"AUC scalar={fa(auc_sc)} multi={fa(auc_mu)}")
    return {"label": label, "n": n, "ll_null": ll_null, "ll_sc": ll_sc, "ll_mu": ll_mu,
            "auc_sc": auc_sc, "auc_mu": auc_mu}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    proj = os.environ.get("PROJECT_ROOT", os.getcwd())
    ap.add_argument("--csv", default="paper_artifacts/per_route_results.csv")
    ap.add_argument("--routes-dir", default=os.path.join(proj, "leaderboard/data/training_routes"))
    ap.add_argument("--scenarios-dir", default=os.path.join(proj, "leaderboard/data/scenarios"))
    ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args(argv)

    model = DifficultyModel(args.routes_dir, args.scenarios_dir)
    rows = load(args.csv, model)
    normalize(rows, ["geom", "scen", "scalar"])   # weather axes already [0,1]
    print(f"Loaded {len(rows)} route-evals.  Predicting FAILURE; 5-fold CV.")
    print("Lower log-loss = better; AUC 0.5 = no better than base rate.\n")

    print("POOLED:")
    res = [evaluate(rows, "all-agents")]
    print("\nPER-AGENT:")
    by = {}
    for r in rows: by.setdefault(r["agent"], []).append(r)
    for a in sorted(by):
        if len(by[a]) >= args.min_n: res.append(evaluate(by[a], a))
        else: print(f"  {a:12s} n={len(by[a])} < {args.min_n} — skipped")

    # verdict
    wins = sum(1 for r in res if r["auc_mu"] is not None and r["auc_sc"] is not None
               and r["auc_mu"] > r["auc_sc"] + 0.01)
    lls = sum(1 for r in res if r["ll_mu"] < r["ll_sc"] - 1e-3)
    print("\nVERDICT")
    print(f"  multi-axis beats scalar on held-out AUC (>+0.01): {wins}/{len(res)} groups")
    print(f"  multi-axis beats scalar on held-out log-loss    : {lls}/{len(res)} groups")
    both = sum(1 for r in res if r['auc_mu'] and r['auc_sc'] and r['auc_mu'] > r['auc_sc']+0.01
               and r['ll_mu'] < r['ll_sc']-1e-3)
    print(f"  multi-axis beats scalar on BOTH                 : {both}/{len(res)} groups")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
