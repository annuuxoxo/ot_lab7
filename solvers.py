"""
solvers.py -- the single-objective sub-problem solvers used by the classical methods (algorithms 1-9).

The lab sheet allows scipy/linprog for the sub-problems; we wrote our own so the whole code base is
dependency-free (NumPy only):

  * linprog()   two-phase dense simplex (Bland's rule)          -> goal programming / lexicographic LPs
  * nelder_mead() bounded Nelder-Mead                           -> nonlinear sub-problems
  * minimize()  multi-start Nelder-Mead + exact L1 penalty for constraints g(x) <= 0
  * minimize_integer_1d()  enumeration for one integer variable (P2 integer)
The scalarisation, the sweeps over weights / eps / targets and the deviation variables are NOT here -- they
live in the algorithm modules.
"""
import numpy as np


# =================================================================== simplex
def _pivot(T, r, c):
    T[r] /= T[r, c]
    for i in range(T.shape[0]):
        if i != r and T[i, c] != 0:
            T[i] -= T[i, c] * T[r]


def _run_simplex(T, basis, allowed, tol=1e-9, max_iter=10000):
    """Minimise: last row of T holds reduced costs, last column the rhs.  Bland's rule prevents cycling."""
    m = T.shape[0] - 1
    for _ in range(max_iter):
        z = T[-1, :-1]
        cand = [j for j in range(len(z)) if allowed[j] and z[j] < -tol]
        if not cand:
            return "optimal"
        c = cand[0]
        col = T[:m, c]
        best, r = np.inf, -1
        for i in range(m):
            if col[i] > tol:
                ratio = T[i, -1] / col[i]
                if ratio < best - 1e-12 or (abs(ratio - best) <= 1e-12 and basis[i] < basis[r]):
                    best, r = ratio, i
        if r < 0:
            return "unbounded"
        _pivot(T, r, c)
        basis[r] = c
    return "iteration limit"


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """min c@x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  lb <= x <= ub.   Returns (x, fun, status)."""
    c = np.asarray(c, float); n = len(c)
    lb = np.zeros(n); ub = np.full(n, np.inf)
    if bounds is not None:
        for i, (lo, hi) in enumerate(bounds):
            lb[i] = 0.0 if lo is None else lo
            ub[i] = np.inf if hi is None else hi
    rows, rhs, kinds = [], [], []                 # kind: 'le' or 'eq'
    A_ub = np.zeros((0, n)) if A_ub is None else np.atleast_2d(np.asarray(A_ub, float))
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, float)
    A_eq = np.zeros((0, n)) if A_eq is None else np.atleast_2d(np.asarray(A_eq, float))
    b_eq = np.zeros(0) if b_eq is None else np.asarray(b_eq, float)
    for a, b in zip(A_ub, b_ub):
        rows.append(a); rhs.append(b - a @ lb); kinds.append("le")
    for i in range(n):
        if np.isfinite(ub[i]):
            e = np.zeros(n); e[i] = 1; rows.append(e); rhs.append(ub[i] - lb[i]); kinds.append("le")
    for a, b in zip(A_eq, b_eq):
        rows.append(a); rhs.append(b - a @ lb); kinds.append("eq")
    m = len(rows)
    n_slack = sum(k == "le" for k in kinds)
    ncol = n + n_slack
    A = np.zeros((m, ncol)); b = np.zeros(m); s = 0
    for i, (a, r, k) in enumerate(zip(rows, rhs, kinds)):
        A[i, :n] = a; b[i] = r
        if k == "le":
            A[i, n + s] = 1.0; s += 1
        if b[i] < 0:
            A[i] *= -1; b[i] *= -1
    # phase 1 with one artificial variable per row
    T = np.zeros((m + 1, ncol + m + 1))
    T[:m, :ncol] = A; T[:m, ncol:ncol + m] = np.eye(m); T[:m, -1] = b
    basis = list(range(ncol, ncol + m))
    T[-1, :ncol] = -A.sum(0); T[-1, -1] = -b.sum()
    allowed = np.ones(ncol + m, bool)
    _run_simplex(T, basis, allowed)
    if -T[-1, -1] > 1e-7:
        return None, None, "infeasible"
    for i in range(m):                              # drive artificials out of the basis
        if basis[i] >= ncol:
            nz = [j for j in range(ncol) if abs(T[i, j]) > 1e-9]
            if nz:
                _pivot(T, i, nz[0]); basis[i] = nz[0]
    # phase 2
    cost = np.zeros(ncol + m); cost[:n] = c
    T[-1, :-1] = cost
    T[-1, -1] = 0.0
    for i in range(m):
        if basis[i] < ncol + m and cost[basis[i]] != 0:
            T[-1] -= cost[basis[i]] * T[i]
    allowed = np.zeros(ncol + m, bool); allowed[:ncol] = True
    status = _run_simplex(T, basis, allowed)
    y = np.zeros(ncol + m)
    for i in range(m):
        y[basis[i]] = T[i, -1]
    x = lb + y[:n]
    return x, float(c @ x), status


# =================================================================== Nelder-Mead
def nelder_mead(fun, x0, lb, ub, maxfev=3000, xtol=1e-10, ftol=1e-13, step=0.05):
    """Bounded Nelder-Mead (points are clipped into the box)."""
    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    n = len(x0)
    clip = lambda x: np.minimum(np.maximum(x, lb), ub)
    S = [clip(np.asarray(x0, float))]
    for i in range(n):
        p = S[0].copy(); d = step * (ub[i] - lb[i])
        p[i] = p[i] + d if p[i] + d <= ub[i] else p[i] - d
        S.append(clip(p))
    S = np.array(S); fv = np.array([fun(p) for p in S]); nfev = n + 1
    while nfev < maxfev:
        o = np.argsort(fv); S, fv = S[o], fv[o]
        if np.max(np.abs(S[1:] - S[0])) < xtol and abs(fv[-1] - fv[0]) < ftol:
            break
        cen = S[:-1].mean(0)
        xr = clip(cen + (cen - S[-1])); fr = fun(xr); nfev += 1
        if fr < fv[0]:
            xe = clip(cen + 2 * (cen - S[-1])); fe = fun(xe); nfev += 1
            S[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            S[-1], fv[-1] = xr, fr
        else:
            if fr < fv[-1]:
                xc = clip(cen + 0.5 * (xr - cen))
            else:
                xc = clip(cen + 0.5 * (S[-1] - cen))
            fc = fun(xc); nfev += 1
            if fc < min(fr, fv[-1]):
                S[-1], fv[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    S[i] = clip(S[0] + 0.5 * (S[i] - S[0])); fv[i] = fun(S[i])
                nfev += n
    i = int(np.argmin(fv))
    return S[i], float(fv[i])


def minimize(fun, lb, ub, cons=(), x0=None, n_starts=6, maxfev=3000, seed=0, rho_schedule=(1e1, 1e3, 1e5)):
    """min fun(x)  s.t.  g(x) <= 0 for every g in cons, lb <= x <= ub.
    Exact L1 penalty + multi-start Nelder-Mead.  Returns (x, fun(x), max_violation)."""
    lb = np.asarray(lb, float); ub = np.asarray(ub, float); n = len(lb)
    rng = np.random.default_rng(seed)
    viol = lambda x: sum(max(0.0, float(g(x))) for g in cons)
    pool = lb + rng.random((60 * n + 40, n)) * (ub - lb)
    if n == 1:
        pool = np.linspace(lb[0], ub[0], 801)[:, None]
    if x0 is not None:
        pool = np.vstack([np.atleast_2d(x0), pool])
    scores = np.array([fun(p) + 1e3 * viol(p) * (1 + abs(fun(p))) for p in pool])
    starts = pool[np.argsort(scores)[:n_starts]]
    best = None
    for s in starts:
        x = s
        for rho in rho_schedule:
            pen = lambda z, rho=rho: fun(z) + rho * viol(z)
            x, _ = nelder_mead(pen, x, lb, ub, maxfev=maxfev // len(rho_schedule) + 200)
        v = viol(x); f = fun(x)
        key = (v > 1e-6, f if v <= 1e-6 else v)
        if best is None or key < best[0]:
            best = (key, x.copy(), f, v)
    return best[1], best[2], best[3]


def minimize_integer_1d(fun, lo, hi, cons=()):
    """Enumerate every integer in [lo, hi] (one integer decision variable)."""
    best = None
    for v in range(int(lo), int(hi) + 1):
        x = np.array([float(v)])
        if all(g(x) <= 1e-9 for g in cons):
            f = fun(x)
            if best is None or f < best[1]:
                best = (x, f)
    return (best[0], best[1], 0.0) if best else (None, None, np.inf)
