"""
Algorithm 33 -- SMS-EMOA, S-Metric Selection EMOA (Beume, Naujoks & Emmerich 2007; indicator based, 5E slides 14-16)

Lecture steps
  1. Fix a reference point r worse than every feasible point (1.1 in the normalised space).
  2. Initialise a population of mu individuals.
  3. STEADY STATE: create exactly ONE offspring per iteration (two random parents, SBX + polynomial mutation).
  4. Add it (mu + 1 individuals) and non-dominated-sort them; the worst front is the candidate set.
  5. If the worst front has more than one member delete the one with the smallest hypervolume contribution
     dHV(i) = HV(P) - HV(P \\ {i}) restricted to that front (exact in 2-D, Monte-Carlo for k >= 3); otherwise delete its single member.
  6. Repeat until the budget is spent.
Defaults: mu = 100, pc = 0.9, one offspring per iteration.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, pc=0.9)


def initialize(problem, seed=0, max_evals=10000, N=100, pc=0.9, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.pc = pc
    st.init_population()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def contributions(Fn, ref):
    if Fn.shape[1] == 2:
        return mu.hv_contributions_2d(Fn, ref)
    return mu.hv_contributions_mc(Fn, ref, lower=Fn.min(0).clip(max=0.0), n_samples=20000)


def worst_index(S, problem):
    """Index (into S) of the individual SMS-EMOA deletes."""
    fronts, _ = mu.fast_non_dominated_sort(S)
    last = fronts[-1]
    if len(last) == 1:
        return int(last[0])
    c = contributions(problem.norm(S[last]), problem.hv_ref)
    return int(last[np.argmin(c)])


def step(st):
    i, j = st.rng.choice(st.N, 2, replace=False)
    child = ops.variation(st.problem, st.X[[i, j]], st.rng, pc=st.pc)[:1]
    Fc, CVc, Sc = st.evaluate(child)
    X = np.vstack([st.X, child]); F = np.vstack([st.F, Fc]); CV = np.concatenate([st.CV, CVc]); S = np.vstack([st.S, Sc])
    w = worst_index(S, st.problem)
    keep = np.ones(len(X), bool); keep[w] = False
    st.X, st.F, st.CV, st.S = X[keep], F[keep], CV[keep], S[keep]


run = make_run(initialize, step)
