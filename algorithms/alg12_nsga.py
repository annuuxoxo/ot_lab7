"""
Algorithm 12 -- NSGA, Non-dominated Sorting Genetic Algorithm (Srinivas & Deb 1994; non-elitist, Part 2 slide 21)

Lecture steps
  1. Sort the population into fronts F1, F2, ... by repeated non-domination.
  2. Dummy fitness  F(i) = Fmax - (rank(i)-1) dF  : every front gets a fitness gap (Fmax = N).  In the original paper dF is
     adaptive: the dummy fitness of front r+1 is set just below (x0.9) the SMALLEST shared fitness of front r, so sharing can
     never reorder the fronts.  We use that rule (it is the 'big gap' of the lecture in practice).
  3. Inside EACH front compute the niche count NC(i) with the sharing function (normalised objective space) and
     F_shared(i) = F(i) / NC(i).
  4. Roulette selection, crossover, mutation (generational).  5. Re-sort next generation.
Defaults: N = 100, sigma_share = 0.1, pc = 0.9.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState, roulette, sharing_matrix

DEFAULTS = dict(N=100, sigma=0.1, pc=0.9)


def initialize(problem, seed=0, max_evals=10000, N=100, sigma=None, pc=0.9, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.sigma = 0.1 * np.sqrt(problem.n_obj / 2) if sigma is None else sigma
    st.pc = pc
    st.init_population()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def shared_fitness(S, Fn, sigma, Fmax):
    fronts, rank = mu.fast_non_dominated_sort(S)
    fit = np.zeros(len(S)); dummy = Fmax
    for fr in fronts:
        sh, _ = sharing_matrix(Fn[fr], sigma)
        nc = sh.sum(1)
        fit[fr] = dummy / nc
        dummy = 0.9 * fit[fr].min()
    return fit, rank


def step(st):
    Fn = st.problem.norm(st.S)
    fit, rank = shared_fitness(st.S, Fn, st.sigma, float(st.N))
    st.rank = rank
    pool = roulette(fit, st.N - st.N % 2, st.rng)
    st.X = ops.variation(st.problem, st.X[pool], st.rng, pc=st.pc)
    if len(st.X) < st.N:
        st.X = np.vstack([st.X, st.problem.random_x(st.N - len(st.X), st.rng)])
    st.F, st.CV, st.S = st.evaluate(st.X)


run = make_run(initialize, step)
