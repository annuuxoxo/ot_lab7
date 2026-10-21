"""
Algorithm 10 -- VEGA, Vector Evaluated Genetic Algorithm (Schaffer 1985; non-elitist, Part 2 slide 8)

Lecture steps
  1. Evaluate every objective f_1..f_k for each individual.
  2. Split the mating pool into k equal sub-populations.
  3. Select sub-population m using objective f_m ONLY (proportional selection, P(i,m) = f_m(x_i)/sum_j f_m(x_j)
     after turning the minimised objective into a fitness  fmax_m - f_m).
  4. Shuffle the sub-populations together into one mating pool.
  5. Crossover (SBX) and mutation (polynomial) create the next generation; repeat.
Non-elitist: the next population is the offspring only, and the returned front is the non-dominated part of the FINAL population.
Defaults: N = 100, pc = 0.9, eta_c = 15, eta_m = 20.
"""
import numpy as np
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState, roulette

DEFAULTS = dict(N=100, pc=0.9)


def initialize(problem, seed=0, max_evals=10000, N=100, pc=0.9, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.pc = pc
    st.init_population()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    N, k = st.N, st.problem.n_obj
    sizes = np.full(k, N // k); sizes[: N - sizes.sum()] += 1
    pool = []
    for m in range(k):
        fit = st.S[:, m].max() - st.S[:, m] + 1e-12            # proportional fitness for a minimised objective
        pool.append(roulette(fit, sizes[m], st.rng))
    pool = np.concatenate(pool)
    st.rng.shuffle(pool)                                         # step 4: shuffle sub-populations together
    off = ops.variation(st.problem, st.X[pool[: N - N % 2]], st.rng, pc=st.pc)
    if len(off) < N:
        off = np.vstack([off, st.problem.random_x(N - len(off), st.rng)])
    st.X = off
    st.F, st.CV, st.S = st.evaluate(st.X)


run = make_run(initialize, step)
