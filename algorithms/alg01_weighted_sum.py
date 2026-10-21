"""
Algorithm 1 -- Weighted Sum (classical, Part 1 slide 30)

Lecture steps
  1. Choose weights w_1..w_k (w_i >= 0, sum w_i = 1) reflecting importance.
  2. Solve the single objective problem  min  F(x) = sum_i w_i f_i(x)  for x*.
  3. Vary the weights and re-solve to trace out several Pareto points.
Defaults: n_weights = 11 equally spaced w_1 in [0,1] (k = 2), normalize = True (objectives scaled to [0,1]
by ideal/nadir; normalize=False reproduces the raw-unit behaviour of the lecture, which is scale-sensitive).
"""
import numpy as np
from algorithms.base import make_run
from algorithms.classical_common import SweepState, solve

DEFAULTS = dict(n_weights=11, normalize=True)


def initialize(problem, seed=0, max_evals=np.inf, n_weights=11, normalize=True, weights=None):
    if weights is None:
        w1 = np.linspace(0, 1, n_weights)
        weights = np.stack([w1, 1 - w1], 1) if problem.n_obj == 2 else _simplex_weights(problem.n_obj, n_weights)
    st = SweepState(problem, seed, weights, max_evals)
    st.normalize = normalize
    return st


def _simplex_weights(k, n):
    from moo_utils import das_dennis
    p = 1
    while len(das_dennis(k, p)) < n:
        p += 1
    return das_dennis(k, p)


def evaluate(st, X):
    return st.problem.evaluate(X)


def step(st):
    w = np.asarray(st.settings[st.i]); p = st.problem
    ideal, scale = (p.ideal, p.scale) if st.normalize else (0.0, 1.0)
    x, F, v = solve(st.counter, lambda f: float(w @ ((f - ideal) / scale)), seed=st.seed + st.i)
    st.X.append(x); st.F.append(F); st.info.append(dict(w=w.tolist(), x=x.tolist(), F=F.tolist(), viol=v))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
