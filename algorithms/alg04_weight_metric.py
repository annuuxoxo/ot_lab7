"""
Algorithm 4 -- Weight Metric method, p-norm distance to the ideal point (classical, Part 1 slide 38)

Lecture steps
  1. Find the ideal point z* (best value of each objective alone: k extra single-objective solves).
  2. Choose the norm order p = 1 (Manhattan), 2 (Euclidean) or infinity (Tchebycheff).
  3. Choose weights w_i and minimise  ( sum_i w_i |f_i - z*_i|^p )^(1/p)   (p = inf: max_i w_i |f_i - z*_i|).
Defaults: p = 2, n_weights = 11, deviations scaled by (nadir - ideal) so the weights are comparable.
"""
import numpy as np
from algorithms.base import make_run
from algorithms.classical_common import SweepState, solve, Counter, ideal_point

DEFAULTS = dict(p=2, n_weights=11)


def initialize(problem, seed=0, max_evals=np.inf, p=2, n_weights=11, normalize=True):
    c = Counter(problem)
    ideal, pay, _ = ideal_point(c, seed)
    w1 = np.linspace(0.02, 0.98, n_weights)
    weights = [np.array([a, 1 - a]) for a in w1] if problem.n_obj == 2 else [np.ones(problem.n_obj) / problem.n_obj]
    st = SweepState(problem, seed, weights, max_evals)
    st.counter.n += c.n; st.p = p; st.ideal = ideal; st.normalize = normalize
    return st


def evaluate(st, X):
    return st.problem.evaluate(X)


def distance(f, w, z, p, scale):
    d = np.abs(f - z) / scale
    if np.isinf(p):
        return float(np.max(w * d))
    return float(np.sum(w * d ** p) ** (1.0 / p))


def step(st):
    w = st.settings[st.i]; sc = st.problem.scale if st.normalize else np.ones(st.problem.n_obj)
    x, F, v = solve(st.counter, lambda f: distance(f, w, st.ideal, st.p, sc), seed=st.seed + st.i)
    st.X.append(x); st.F.append(F)
    st.info.append(dict(w=w.tolist(), p=float(st.p), x=x.tolist(), F=F.tolist(), viol=v))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
