"""
Algorithm 35 -- R-NSGA-II, Reference-point based NSGA-II (Deb, Sundar, Udaya Bhaskara Rao & Chaudhuri 2006; preference based, 5E slides 20-22)

Lecture steps
  1. The decision maker supplies reference point(s) in objective space (natural units, converted here to minimisation form).
  2. Min-max normalise objectives (here: by the problem's ideal/nadir range).
  3. Euclidean distance d_i of every solution to the (nearest) reference point.
  4. Sort by d_i ascending and accept a solution only if it is at least eps away (normalised space) from every already accepted one
     (eps-clearing keeps a spread near the preference); the rejected ones follow, again by distance (back-fill).
  5. NSGA-II frame: parent + offspring are first split into non-dominated fronts; the eps-clearing order replaces crowding distance
     inside the front that does not fit completely.  Tournament: lower front rank, then smaller distance to the reference point.
Output: the final population's non-dominated set (a FOCUSED subset, not the whole front).
Defaults: N = 100, eps = 0.01, pc = 0.9.   ref_points = list of reference points in natural units.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, eps=0.01, pc=0.9)
REF_DEFAULT = {"P3": [(0.5, 0.5)], "P4": [(0.02, 0.12)], "P14": [(1.8, 12.0)], "P7": [(40.0, 50.0)], "P5": [(20.0, 0.002)]}


def to_min(problem, pts):
    pts = np.array(pts, float)
    if problem.maximize is not None:
        for j, mx in enumerate(problem.maximize):
            if mx:
                pts[:, j] = -pts[:, j]
    return pts


def clearing_order(Fn, Rn, eps):
    """Order the rows: eps-cleared representatives (by distance) first, then the rest (by distance)."""
    d = np.sqrt(((Fn[:, None, :] - Rn[None, :, :]) ** 2).sum(2)).min(1)
    order = np.argsort(d, kind="stable")
    accepted, rejected = [], []
    for i in order:
        if all(np.linalg.norm(Fn[i] - Fn[a]) >= eps for a in accepted):
            accepted.append(i)
        else:
            rejected.append(i)
    return np.array(accepted + rejected), d


def initialize(problem, seed=0, max_evals=10000, N=100, eps=0.01, pc=0.9, ref_points=None, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.eps, st.pc = eps, pc
    st.R = to_min(problem, REF_DEFAULT[problem.name] if ref_points is None else ref_points)
    st.init_population()
    _update_rank(st)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def _update_rank(st):
    fronts, rank = mu.fast_non_dominated_sort(st.S)
    st.rank = rank
    Fn, Rn = st.problem.norm(st.S), st.problem.norm(st.R)
    st.dist = np.sqrt(((Fn[:, None, :] - Rn[None, :, :]) ** 2).sum(2)).min(1)


def step(st):
    N = st.N
    a = st.rng.integers(N, size=N + N % 2); b = st.rng.integers(N, size=N + N % 2)
    win_a = (st.rank[a] < st.rank[b]) | ((st.rank[a] == st.rank[b]) & (st.dist[a] <= st.dist[b]))
    win = np.where(win_a, a, b)
    Q = ops.variation(st.problem, st.X[win], st.rng, pc=st.pc)[:N]
    Fq, CVq, Sq = st.evaluate(Q)
    X = np.vstack([st.X, Q]); F = np.vstack([st.F, Fq]); CV = np.concatenate([st.CV, CVq]); S = np.vstack([st.S, Sq])
    fronts, _ = mu.fast_non_dominated_sort(S)
    Fn, Rn = st.problem.norm(S), st.problem.norm(st.R)
    chosen = []
    for fr in fronts:
        if len(chosen) + len(fr) <= N:
            chosen.extend(fr.tolist())
        else:
            order, _ = clearing_order(Fn[fr], Rn, st.eps)
            chosen.extend(fr[order[: N - len(chosen)]].tolist())
            break
    keep = np.array(chosen)
    st.X, st.F, st.CV, st.S = X[keep], F[keep], CV[keep], S[keep]
    _update_rank(st)


run = make_run(initialize, step)
