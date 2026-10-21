"""
Algorithm 14 -- NSGA-II (Deb, Pratap, Agarwal & Meyarivan 2002; elitist, Part 3 slides 4-11)

Lecture steps
  1. Initialise population P0 randomly and evaluate.
  2. Fast non-dominated sorting into fronts F1, F2, ... (n_p / S_p bookkeeping, O(k N^2)).
  3. Crowding distance inside every front (boundary points = infinity).
  4. Crowded-comparison binary tournament: lower rank wins, ties -> larger crowding distance.
  5. SBX crossover + polynomial mutation create offspring Q.
  6. Combine R = P u Q (elitism); sort R by front and crowding; keep the best N.     7. Repeat.
Constraints: constraint='penalty' (default, objectives pushed away by rho*CV), 'ignore' (no handling at all) or
'deb' (Deb's feasibility rules inside the sorting -- this is exactly Algorithm 21, see alg21_constrained_nsga2.py).
Defaults: N = 100, pc = 0.9, pm = 1/n, eta_c = 15, eta_m = 20.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, pc=0.9, eta_c=15.0, eta_m=20.0)


def initialize(problem, seed=0, max_evals=10000, N=100, pc=0.9, eta_c=15.0, eta_m=20.0, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.pc, st.eta_c, st.eta_m = pc, eta_c, eta_m
    st.init_population()
    _rank(st)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def sort_args(st, S, CV):
    return (S, CV) if st.constraint == "deb" else (S, None)


def _rank(st):
    fronts, rank = mu.fast_non_dominated_sort(*sort_args(st, st.S, st.CV))
    st.rank = rank
    st.crowd = ops.front_crowding(st.S, fronts)


def environmental_selection(S, CV, N, deb):
    """Fill the next population front by front, truncating the last front by crowding distance."""
    fronts, rank = mu.fast_non_dominated_sort(S, CV if deb else None)
    chosen = []
    for fr in fronts:
        if len(chosen) + len(fr) <= N:
            chosen.extend(fr.tolist())
        else:
            cd = mu.crowding_distance(S[fr])
            order = np.argsort(-cd, kind="stable")
            chosen.extend(fr[order[: N - len(chosen)]].tolist())
            break
    return np.array(chosen)


def step(st):
    N = st.N
    mate = ops.crowded_tournament(st.rank, st.crowd, st.rng, N + N % 2)
    Q = ops.variation(st.problem, st.X[mate], st.rng, pc=st.pc, eta_c=st.eta_c, eta_m=st.eta_m)[:N]
    Fq, CVq, Sq = st.evaluate(Q)
    X = np.vstack([st.X, Q]); F = np.vstack([st.F, Fq]); CV = np.concatenate([st.CV, CVq]); S = np.vstack([st.S, Sq])
    keep = environmental_selection(S, CV, N, st.constraint == "deb")
    st.X, st.F, st.CV, st.S = X[keep], F[keep], CV[keep], S[keep]
    _rank(st)


run = make_run(initialize, step)
