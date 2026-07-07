#!/usr/bin/env python3
"""Sanity check: can a per-model noisy-OR (competing-hazards) fit RECOVER
per-axis difficulty sensitivity, and in particular SEPARATE illumination
sensitivity by sensor modality (camera vs LiDAR)?

Generative model (ground truth):
    P(fail | route, model) = 1 - exp( - sum_j lambda_j(model) * x_j )
where x_j in [0,1] are per-axis hazards. This IS noisy-OR in the continuous
limit (each lambda_j = -ln(1-p_j) >= 0 is that axis's hazard weight).

We encode the hypothesis in the ground-truth lambdas:
  * camera-only agents  -> lambda_illum LARGE
  * LiDAR-aided agents  -> lambda_illum ~ 0 , lambda_precip larger (rain scatter)
then fit each model independently and ask whether the fit recovers it.

Pure stdlib (no numpy/scipy), matching the repo's tools/ convention.
"""
import math, random

AXES  = ['geom', 'scen', 'illum', 'precip', 'traffic']
K     = len(AXES)
ILLUM = AXES.index('illum')

# Ground-truth hazard weights (lambda >= 0). Failure rate kept mid-range.
TRUE = {
    # camera-only: illumination dominates the hazard
    'cilrs':      [0.35, 0.25, 0.95, 0.20, 0.15],
    'neat':       [0.30, 0.22, 0.85, 0.18, 0.15],
    'tcp':        [0.28, 0.20, 0.75, 0.15, 0.12],
    'roach':      [0.30, 0.20, 0.70, 0.15, 0.12],
    # LiDAR-aided: illumination ~0, precipitation matters more
    'interfuser': [0.40, 0.25, 0.05, 0.45, 0.15],
    'lav':        [0.38, 0.22, 0.05, 0.40, 0.15],
}
MODALITY = {'cilrs': 'camera', 'neat': 'camera', 'tcp': 'camera', 'roach': 'camera',
            'interfuser': 'lidar', 'lav': 'lidar'}


# ---------------------------------------------------------------- generative
def simulate(lam, n, rng, illum_var=True):
    """Return X (n x K), y (fail 1/0), score (0-100). illum_var=False pins the
    illumination axis to ~constant (mimics the real daytime-only sample)."""
    X, y, score = [], [], []
    for _ in range(n):
        xi = [rng.random() for _ in range(K)]
        if not illum_var:
            xi[ILLUM] = 0.92 + 0.02 * rng.random()   # pinned bright (like presets 19-20)
        eta = sum(lam[j] * xi[j] for j in range(K))
        pf  = 1.0 - math.exp(-eta)
        y.append(1.0 if rng.random() < pf else 0.0)
        X.append(xi)
        score.append(max(0.0, min(100.0, 100.0 * (1.0 - pf) + rng.gauss(0, 6))))
    return X, y, score


# ------------------------------------------------------------------- solver
def _solve_inv(H, g):
    """Solve H x = g and return (x, H^{-1}) for K x K via Gauss-Jordan."""
    n = len(H)
    A = [H[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        d = A[col][col] or 1e-12
        A[col] = [v / d for v in A[col]]
        for r in range(n):
            if r != col and A[r][col]:
                f = A[r][col]
                A[r] = [A[r][k] - f * A[col][k] for k in range(2 * n)]
    inv = [[A[i][n + j] for j in range(n)] for i in range(n)]
    x   = [sum(inv[i][j] * g[j] for j in range(n)) for i in range(n)]
    return x, inv


def _nll(lam, X, y):
    tot = 0.0
    for xi, yi in zip(X, y):
        eta = min(max(sum(lam[j] * xi[j] for j in range(K)), 1e-9), 50.0)
        pf  = min(max(1.0 - math.exp(-eta), 1e-12), 1 - 1e-12)
        tot -= yi * math.log(pf) + (1 - yi) * math.log(1 - pf)
    return tot


def fit(X, y, iters=40, ridge=1e-6):
    """Newton MLE of the noisy-OR hazard weights (lambda >= 0). Returns
    (lambda_hat, standard_errors) with SEs from the inverse Hessian at the MLE."""
    lam = [0.3] * K
    inv = [[0.0] * K for _ in range(K)]
    for _ in range(iters):
        G = [0.0] * K
        H = [[0.0] * K for _ in range(K)]
        for xi, yi in zip(X, y):
            eta = min(max(sum(lam[j] * xi[j] for j in range(K)), 1e-9), 50.0)
            s   = math.exp(-eta); pf = min(max(1.0 - s, 1e-9), 1 - 1e-9)
            gi  = (1.0 - yi) - yi * s / pf
            hi  = yi * s / (pf * pf)
            for j in range(K):
                G[j] += gi * xi[j]
                if hi:
                    row = H[j]; xij = xi[j]
                    for k in range(K):
                        row[k] += hi * xij * xi[k]
        for j in range(K):
            H[j][j] += ridge
        step, inv = _solve_inv(H, G)
        # damped Newton with backtracking to stay in the descent regime
        base = _nll(lam, X, y); alpha = 1.0
        for _bt in range(20):
            cand = [max(0.0, lam[j] - alpha * step[j]) for j in range(K)]
            if _nll(cand, X, y) <= base + 1e-9:
                lam = cand; break
            alpha *= 0.5
        else:
            lam = [max(0.0, lam[j] - alpha * step[j]) for j in range(K)]
    se = [math.sqrt(inv[j][j]) if inv[j][j] > 0 else float('nan') for j in range(K)]
    return lam, se


# ------------------------------------------------------------- rank stats
def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


# ================================================================= experiments
def expA():
    print("=" * 78)
    print("A. RECOVERY + MODALITY SEPARATION  (n=600/model, illumination well sampled)")
    print("=" * 78)
    rng = random.Random(1)
    lo_cam, hi_lid = [], []
    print(f"  {'model':11s} {'mod':6s} {'failrate':8s}  {'lam_illum true':>14s}  {'lam_illum hat [95% CI]':>26s}")
    for m in TRUE:
        X, y, _ = simulate(TRUE[m], 600, rng)
        lam, se = fit(X, y)
        fr = sum(y) / len(y)
        lo = lam[ILLUM] - 1.96 * se[ILLUM]; hi = lam[ILLUM] + 1.96 * se[ILLUM]
        print(f"  {m:11s} {MODALITY[m]:6s} {fr:8.2f}  {TRUE[m][ILLUM]:>14.2f}  "
              f"{lam[ILLUM]:>8.3f} [{lo:6.3f},{hi:6.3f}]")
        (lo_cam if MODALITY[m] == 'camera' else hi_lid).append((lo, hi, lam[ILLUM]))
    cam_lo_min = min(c[0] for c in lo_cam)          # lowest camera CI lower bound
    lid_hi_max = max(l[1] for l in hi_lid)          # highest lidar CI upper bound
    print(f"\n  camera lam_illum CIs all >= {cam_lo_min:.3f}; lidar CIs all <= {lid_hi_max:.3f}")
    sep = cam_lo_min > lid_hi_max
    print(f"  -> 95% CIs {'DO NOT overlap' if sep else 'OVERLAP'}: "
          f"camera-vs-LiDAR illumination sensitivity is "
          f"{'cleanly separated' if sep else 'NOT separated'}.")
    # full-vector recovery for one camera model
    X, y, _ = simulate(TRUE['cilrs'], 600, rng); lam, se = fit(X, y)
    print("\n  full-vector recovery (cilrs, n=600):")
    for j, a in enumerate(AXES):
        ok = abs(lam[j] - TRUE['cilrs'][j]) <= 1.96 * se[j]
        print(f"    {a:8s} true={TRUE['cilrs'][j]:.2f}  hat={lam[j]:.3f} +/- {1.96*se[j]:.3f}  "
              f"{'ok' if ok else 'OFF'}")
    return sep


def expB():
    print("\n" + "=" * 78)
    print("B. THE NAIVE SCALAR FAILS  (D = unweighted sum of axes, like geo+scen+weather)")
    print("=" * 78)
    rng = random.Random(2)
    pooled_D, pooled_S = [], []
    print(f"  {'model':11s} {'mod':6s}  Spearman(scalar D, score)")
    for m in TRUE:
        X, y, sc = simulate(TRUE[m], 600, rng)
        D = [sum(xi) for xi in X]
        r = _spearman(D, sc)
        print(f"  {m:11s} {MODALITY[m]:6s}  {r:+.3f}")
        pooled_D += D; pooled_S += sc
    rp = _spearman(pooled_D, pooled_S)
    print(f"\n  POOLED across all models: Spearman(D, score) = {rp:+.3f}")
    print(f"  (compare real run's pooled total-difficulty Spearman = +0.036 -> same wash-out)")
    return rp


def expC():
    print("\n" + "=" * 78)
    print("C. HOW MUCH COVERAGE?  reliability of camera>LiDAR separation vs evals/model")
    print("=" * 78)
    print(f"  {'n/model':8s}  {'cam lam_illum':>13s}  {'lid lam_illum':>13s}  {'sep. reliability':>16s}")
    R = 20
    thresh_n = None
    for n in (50, 100, 200, 400, 800):
        cam_vals, lid_vals, wins = [], [], 0
        for rep in range(R):
            rng = random.Random(1000 + rep)
            Xc, yc, _ = simulate(TRUE['cilrs'], n, rng)
            Xl, yl, _ = simulate(TRUE['interfuser'], n, rng)
            lc, _ = fit(Xc, yc); ll, _ = fit(Xl, yl)
            cam_vals.append(lc[ILLUM]); lid_vals.append(ll[ILLUM])
            wins += (lc[ILLUM] > ll[ILLUM])
        rel = wins / R
        if thresh_n is None and rel >= 0.95:
            thresh_n = n
        print(f"  {n:8d}  {sum(cam_vals)/R:>13.3f}  {sum(lid_vals)/R:>13.3f}  {rel:>15.0%}")
    print(f"\n  -> reliable (>=95%) separation from ~{thresh_n} evals/model "
          f"WITH illumination variation.")
    return thresh_n


def expD():
    print("\n" + "=" * 78)
    print("D. MISSPECIFICATION CHECK  (generate under LOGISTIC link, fit noisy-OR)")
    print("=" * 78)
    rng = random.Random(7)

    def sim_logit(lam, n):
        X, y = [], []
        for _ in range(n):
            xi = [rng.random() for _ in range(K)]
            z  = -1.2 + sum(3.0 * lam[j] * xi[j] for j in range(K))   # logistic
            pf = 1.0 / (1.0 + math.exp(-z))
            y.append(1.0 if rng.random() < pf else 0.0); X.append(xi)
        return X, y

    Xc, yc = sim_logit(TRUE['cilrs'], 600); lc, sc_ = fit(Xc, yc)
    Xl, yl = sim_logit(TRUE['interfuser'], 600); ll, sl_ = fit(Xl, yl)
    print(f"  camera cilrs      lam_illum_hat = {lc[ILLUM]:.3f} +/- {1.96*sc_[ILLUM]:.3f}")
    print(f"  lidar  interfuser lam_illum_hat = {ll[ILLUM]:.3f} +/- {1.96*sl_[ILLUM]:.3f}")
    survives = lc[ILLUM] - 1.96 * sc_[ILLUM] > ll[ILLUM] + 1.96 * sl_[ILLUM]
    print(f"  -> separation {'SURVIVES' if survives else 'BREAKS'} under model mismatch.")
    return survives


def expE():
    print("\n" + "=" * 78)
    print("E. THE REAL-DATA TRAP  (illumination pinned bright, as in the 97% daytime sample)")
    print("=" * 78)
    rng = random.Random(11)
    Xc, yc, _ = simulate(TRUE['cilrs'], 600, rng, illum_var=False)
    lc, sc_ = fit(Xc, yc)
    print(f"  camera cilrs, illumination ~constant: lam_illum_hat = {lc[ILLUM]:.3f} "
          f"+/- {1.96*sc_[ILLUM]:.3f}")
    print(f"  -> with no illumination variation the axis is UNIDENTIFIABLE "
          f"(huge CI); coverage, not method, is the current blocker.")


if __name__ == '__main__':
    sep = expA()
    rp  = expB()
    tn  = expC()
    sv  = expD()
    expE()
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  * per-model noisy-OR recovers illumination sensitivity & separates "
          f"modality: {'YES' if sep else 'NO'}")
    print(f"  * naive scalar washes out (pooled Spearman ~ {rp:+.3f}, real was +0.036): "
          f"{'YES' if abs(rp) < 0.15 else 'NO'}")
    print(f"  * separation robust to link misspecification: {'YES' if sv else 'NO'}")
    print(f"  * coverage needed: ~{tn} illumination-varied evals/model "
          f"(real sample currently ~5)")
