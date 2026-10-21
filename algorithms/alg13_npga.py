"""
Algorithm 13 -- NPGA, Niched Pareto Genetic Algorithm (Horn, Nafpliotis & Goldberg 1994; non-elitist, Part 2 slide 27)

Lecture steps (one tournament)
  1. Pick two candidates i, j at random (binary tournament).
  2. SEPARATELY pick a comparison set of size t_dom at random (used only to measure crowding).
  3. If i dominates j select i; if j dominates i select j.
  4. Otherwise compute the niche counts NC(i), NC(j) = sum over the comparison set of sh(d) (normalised objective space).
  5. Select the candidate with the LOWER niche count (ties: random).  Repeat until the mating pool is full; then crossover/mutation.
Defaults: N = 100, t_dom = 10 (10 % of N), sigma_share = 0.1.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, t_dom=10, sigma=0.1, pc=0.9)


def initialize(problem, seed=0, max_evals=10000, N=100, t_dom=None, sigma=None, pc=0.9, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.t_dom = max(2, N // 10) if t_dom is None else t_dom
    st.sigma = 0.1 * np.sqrt(problem.n_obj / 2) if sigma is None else sigma
    st.pc = pc
    st.init_population()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def tournament(S, Fn, i, j, comp, sigma, rng):
    """Winner of one Pareto tournament between candidates i and j given the comparison set `comp`."""
    if mu.dominates(S[i], S[j]):
        return i
    if mu.dominates(S[j], S[i]):
        return j
    def nc(a):
        d = np.sqrt(((Fn[comp] - Fn[a]) ** 2).sum(1))
        return np.where(d < sigma, 1 - (d / sigma) ** 2, 0.0).sum()
    ni, nj = nc(i), nc(j)
    if ni == nj:
        return i if rng.random() < 0.5 else j
    return i if ni < nj else j


def step(st):
    Fn = st.problem.norm(st.S)
    pool = []
    for _ in range(st.N - st.N % 2):
        i, j = st.rng.choice(st.N, 2, replace=False)
        comp = st.rng.choice(st.N, st.t_dom, replace=False)
        pool.append(tournament(st.S, Fn, i, j, comp, st.sigma, st.rng))
    st.X = ops.variation(st.problem, st.X[np.array(pool)], st.rng, pc=st.pc)
    if len(st.X) < st.N:
        st.X = np.vstack([st.X, st.problem.random_x(st.N - len(st.X), st.rng)])
    st.F, st.CV, st.S = st.evaluate(st.X)


run = make_run(initialize, step)
