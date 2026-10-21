"""
moo_utils.py -- shared multiobjective toolkit (Task A1), written from scratch with NumPy only.

Everything is MINIMISATION.  Convert a maximisation objective with  Max f == Min -f  first.

Contents
--------
dominates / constrained_dominates      Pareto dominance and Deb's three feasibility rules (Part 1, slide 23)
dominance_matrix                       vectorised pairwise dominance (optionally constrained)
non_dominated_mask / pareto_filter_2d  extract the non-dominated subset
fast_non_dominated_sort                Deb's n_p / S_p procedure (Part 1, slide 22)
crowding_distance                      Part 1, slide 24
normalize                              (f - ideal) / (nadir - ideal)
hv2d / hv_mc / hypervolume             exact 2-D hypervolume, Monte-Carlo estimate for >= 3 objectives
hv_contributions_2d                    exclusive hypervolume contribution of each point (SMS-EMOA)
igd / gd / spacing                     quality indicators (5E slide 41-42; Schott spacing)
das_dennis                             structured reference points on the unit simplex (NSGA-III, RVEA, MOEA/D)
"""
import numpy as np
from math import comb


# --------------------------------------------------------------------------- dominance
def dominates(a, b):
    """True if a Pareto-dominates b (minimisation): a<=b everywhere and a<b somewhere."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return bool(np.all(a <= b) and np.any(a < b))


def constrained_dominates(fa, cva, fb, cvb, tol=0.0):
    """Deb's constrained-dominance (Part 1 slide 23).
    Rule 1: feasible beats infeasible.  Rule 2: both feasible -> Pareto dominance.
    Rule 3: both infeasible -> smaller constraint violation wins (objectives ignored)."""
    fa_ok, fb_ok = cva <= tol, cvb <= tol
    if fa_ok and not fb_ok:
        return True
    if not fa_ok and fb_ok:
        return False
    if fa_ok and fb_ok:
        return dominates(fa, fb)
    return cva < cvb


def dominance_matrix(F, CV=None):
    """D[i, j] = True when i dominates j.  With CV given, Deb's constrained dominance is used."""
    F = np.asarray(F, float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    D = le & lt
    if CV is not None:
        CV = np.asarray(CV, float)
        feas = CV <= 0
        both_f = feas[:, None] & feas[None, :]
        one = feas[:, None] & ~feas[None, :]
        none = ~feas[:, None] & ~feas[None, :]
        D = (both_f & D) | one | (none & (CV[:, None] < CV[None, :]))
    return D


def non_dominated_mask(F, CV=None):
    """Boolean mask of the rows that nobody dominates.  Duplicates are all kept."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return np.zeros(0, bool)
    return ~dominance_matrix(F, CV).any(axis=0)


def pareto_filter_2d(F):
    """O(N log N) non-dominated filter for two objectives (used on million-point reference samples)."""
    F = np.asarray(F, float)
    order = np.lexsort((F[:, 1], F[:, 0]))
    best = np.inf
    keep = []
    for i in order:
        if F[i, 1] < best:
            keep.append(i)
            best = F[i, 1]
    return np.array(keep, int)


def fast_non_dominated_sort(F, CV=None):
    """Deb's fast non-dominated sorting.  Returns (fronts, rank): fronts is a list of index arrays
    (front 1 first); rank[i] is the 0-based front number of individual i."""
    n = len(F)
    D = dominance_matrix(F, CV)
    n_p = D.sum(axis=0).astype(int)            # how many dominate p
    rank = np.full(n, -1, int)
    fronts = []
    current = np.where(n_p == 0)[0]
    r = 0
    while len(current):
        fronts.append(current)
        rank[current] = r
        n_p = n_p - D[current].sum(axis=0)      # remove this front's dominance
        n_p[current] = -1
        current = np.where(n_p == 0)[0]
        r += 1
    return fronts, rank


# --------------------------------------------------------------------------- diversity
def crowding_distance(F):
    """Crowding distance of every row of F (one front).  Boundary points get inf (Part 1 slide 24)."""
    F = np.asarray(F, float)
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        d[:] = np.inf
        return d
    for j in range(m):
        idx = np.argsort(F[:, j], kind="stable")
        rng_ = F[idx[-1], j] - F[idx[0], j]
        d[idx[0]] = d[idx[-1]] = np.inf
        if rng_ <= 0:
            continue
        d[idx[1:-1]] += (F[idx[2:], j] - F[idx[:-2], j]) / rng_
    return d


# --------------------------------------------------------------------------- normalisation
def normalize(F, ideal, nadir):
    ideal = np.asarray(ideal, float); nadir = np.asarray(nadir, float)
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    return (np.asarray(F, float) - ideal) / span


# --------------------------------------------------------------------------- hypervolume
def hv2d(F, ref):
    """Exact hypervolume for two minimised objectives, bounded by ref."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[pareto_filter_2d(F)]
    F = F[np.argsort(F[:, 0])]
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in F:
        hv += (ref[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return float(hv)


_MC_CACHE = {}


def hv_mc(F, ref, lower=None, n_samples=200000, seed=12345):
    """Monte-Carlo hypervolume for >= 3 objectives.  The same sample cloud (fixed seed) is reused
    so that two fronts are always compared on identical samples."""
    F = np.asarray(F, float)
    ref = np.asarray(ref, float)
    m = len(ref)
    lower = np.zeros(m) if lower is None else np.asarray(lower, float)
    if len(F) == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    key = (m, n_samples, seed)
    if key not in _MC_CACHE:
        _MC_CACHE[key] = np.random.default_rng(seed).random((n_samples, m))
    S = lower + _MC_CACHE[key] * (ref - lower)
    dominated = np.zeros(n_samples, bool)
    for i in range(0, len(F), 64):
        blk = F[i:i + 64]
        dominated |= np.any(np.all(blk[None, :, :] <= S[:, None, :], axis=2), axis=1)
    return float(dominated.mean() * np.prod(ref - lower))


def hypervolume(F, ref, **kw):
    """Exact in 2-D, Monte-Carlo otherwise (allowed by the lab sheet for >= 3 objectives)."""
    ref = np.asarray(ref, float)
    return hv2d(F, ref) if len(ref) == 2 else hv_mc(F, ref, **kw)


def hv_contributions_2d(F, ref):
    """Exclusive hypervolume contribution of every row (2-D).  Dominated points contribute 0."""
    F = np.asarray(F, float)
    n = len(F)
    contrib = np.zeros(n)
    inside = np.where(np.all(F < ref, axis=1))[0]
    if len(inside) == 0:
        return contrib
    sub = F[inside]
    nd_local = pareto_filter_2d(sub)
    nd = inside[nd_local]
    P = F[nd]
    order = np.argsort(P[:, 0], kind="stable")
    P_sorted = P[order]
    k = len(P_sorted)
    for t in range(k):
        right = P_sorted[t + 1, 0] if t + 1 < k else ref[0]
        upper = P_sorted[t - 1, 1] if t > 0 else ref[1]
        contrib[nd[order[t]]] = (right - P_sorted[t, 0]) * (upper - P_sorted[t, 1])
    # exact duplicates of a front point share (and therefore lose) their contribution
    return contrib


# --------------------------------------------------------------------------- indicators
def _pairwise_dist(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    return np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2), 0.0))


def igd(A, Z):
    """Inverted generational distance: mean distance from each true-front point z to the nearest obtained point."""
    A = np.asarray(A, float)
    if len(A) == 0:
        return np.inf
    out = 0.0
    for i in range(0, len(Z), 2000):
        out += _pairwise_dist(Z[i:i + 2000], A).min(axis=1).sum()
    return float(out / len(Z))


def gd(A, Z):
    """Generational distance: mean distance from each obtained point to the nearest true-front point."""
    A = np.asarray(A, float)
    if len(A) == 0:
        return np.inf
    return float(_pairwise_dist(A, Z).min(axis=1).mean())


def spacing(A):
    """Schott spacing: standard deviation of nearest-neighbour distances (0 = perfectly even)."""
    A = np.asarray(A, float)
    if len(A) < 3:
        return float("nan")
    D = _pairwise_dist(A, A)
    np.fill_diagonal(D, np.inf)
    d = D.min(axis=1)
    return float(np.sqrt(((d - d.mean()) ** 2).sum() / (len(d) - 1)))


# --------------------------------------------------------------------------- reference points
def das_dennis(m, p):
    """All points of the form (k_1..k_m)/p with k_i integer >= 0 and sum k_i = p.   H = C(m+p-1, p)."""
    out = []

    def rec(prefix, left, slots):
        if slots == 1:
            out.append(prefix + [left])
            return
        for k in range(left, -1, -1):
            rec(prefix + [k], left - k, slots - 1)

    rec([], p, m)
    pts = np.array(out, float) / p
    assert len(pts) == comb(m + p - 1, p)
    return pts


def two_layer_das_dennis(m, p_boundary, p_inside):
    """Boundary layer plus an inside layer shrunk toward the centroid (Deb & Jain, many-objective case)."""
    a = das_dennis(m, p_boundary)
    b = das_dennis(m, p_inside)
    b = b / 2.0 + 1.0 / (2.0 * m)
    return np.vstack([a, b])


def hv_contributions_mc(F, ref, lower=None, n_samples=100000, seed=12345):
    """Monte-Carlo exclusive hypervolume contribution of every row (any number of objectives)."""
    F = np.asarray(F, float); ref = np.asarray(ref, float)
    m = len(ref)
    lower = np.zeros(m) if lower is None else np.asarray(lower, float)
    n = len(F)
    key = (m, n_samples, seed)
    if key not in _MC_CACHE:
        _MC_CACHE[key] = np.random.default_rng(seed).random((n_samples, m))
    S = lower + _MC_CACHE[key] * (ref - lower)
    D = np.zeros((n, n_samples), bool)
    for i in range(n):
        D[i] = np.all(F[i] <= S, axis=1) & np.all(F[i] < ref)
    cnt = D.sum(0)
    only = (cnt == 1)
    contrib = (D & only).sum(1) / n_samples * np.prod(ref - lower)
    return contrib
