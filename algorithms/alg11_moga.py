"""
Algorithm 11 -- MOGA, Multi-Objective Genetic Algorithm (Fonseca & Fleming 1993; non-elitist, Part 2 slide 14)

Lecture steps
  1. Evaluate all objectives.
  2. rank(i) = 1 + (number of individuals that dominate i).
  3. Raw fitness F(i) = 1 / rank(i).
  4. Pairwise distances in NORMALISED objective space, sharing sh(d) = 1 - (d/sigma)^2 for d < sigma.
  5. Niche count NC(i) = sum_j sh(d_ij) (includes itself);  F_shared(i) = F(i) / NC(i).
  6. Roulette-wheel selection on F_shared, crossover, mutation (generational, no elitism).
Defaults: N = 100, sigma_share = 0.1 (normalised units; 0.1 * sqrt(M/2) for M objectives), pc = 0.9.
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


def shared_fitness(S, Fn, sigma, CV=None):
    D = mu.dominance_matrix(S)
    rank = 1 + D.sum(0)
    raw = 1.0 / rank
    sh, _ = sharing_matrix(Fn, sigma)
    nc = sh.sum(1)                                    # diagonal contributes sh(0) = 1
    return raw / nc, rank, nc


def step(st):
    Fn = st.problem.norm(st.S)
    fit, rank, nc = shared_fitness(st.S, Fn, st.sigma)
    st.rank, st.nc = rank, nc
    pool = roulette(fit, st.N - st.N % 2, st.rng)
    st.X = ops.variation(st.problem, st.X[pool], st.rng, pc=st.pc)
    if len(st.X) < st.N:
        st.X = np.vstack([st.X, st.problem.random_x(st.N - len(st.X), st.rng)])
    st.F, st.CV, st.S = st.evaluate(st.X)


run = make_run(initialize, step)
