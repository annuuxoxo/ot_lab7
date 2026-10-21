"""
Algorithm 30 -- MOMA, Multi-Objective Memetic Algorithm (NSGA-II + local search; hybrid, Part 4 slide 69)

Lecture steps
  1. One NSGA-II generation (crowded tournament, SBX / permutation operators, parent + offspring elitism).
  2. Take the n_ls LEAST CROWDED members (largest crowding distance) of the first front.
  3. Hill climbing on each: try `ls_steps` random neighbours (+-1 step / swap or inversion / bit flip); a neighbour is accepted if it
     dominates the current point, or if it improves a random-weighted sum of the normalised objectives without being dominated
     (the 'combined score' of the lecture).   The improved point replaces the original in the population (Lamarckian).
  4. Re-rank the population and repeat.   Local-search evaluations are counted in the budget.
Defaults: N = 100, n_ls = 10, ls_steps = 3 (the NSGA-II defaults otherwise).
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms import alg14_nsga2 as nsga2

DEFAULTS = dict(nsga2.DEFAULTS, n_ls=10, ls_steps=3)


def initialize(problem, seed=0, max_evals=10000, n_ls=10, ls_steps=3, **kw):
    st = nsga2.initialize(problem, seed=seed, max_evals=max_evals, **kw)
    st.n_ls, st.ls_steps = n_ls, ls_steps
    return st


def evaluate(st, X):
    return st.evaluate(X)


def local_search(st, i, w):
    p, rng = st.problem, st.rng
    x, f, cv, s = st.X[i].copy(), st.F[i].copy(), st.CV[i], st.S[i].copy()
    for _ in range(st.ls_steps):
        if st.evals >= st.max_evals:
            break
        y = ops.neighbour(p, x, rng, 0.05)
        Fy, CVy, Sy = st.evaluate(y[None])
        dom = np.all(Sy[0] <= s) and np.any(Sy[0] < s)
        worse = np.all(s <= Sy[0]) and np.any(s < Sy[0])
        score_gain = float(w @ p.norm(s)) - float(w @ p.norm(Sy[0]))
        if dom or (not worse and score_gain > 0):
            x, f, cv, s = y, Fy[0], CVy[0], Sy[0]
    st.X[i], st.F[i], st.CV[i], st.S[i] = x, f, cv, s


def step(st):
    nsga2.step(st)
    first = np.where(st.rank == 0)[0]
    cd = st.crowd[first]
    order = first[np.argsort(-np.where(np.isinf(cd), 1e9, cd), kind="stable")]
    for i in order[: st.n_ls]:
        w = st.rng.random(st.problem.n_obj); w /= w.sum()
        local_search(st, i, w)
    nsga2._rank(st)


run = make_run(initialize, step)
