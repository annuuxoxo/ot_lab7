"""
Algorithm 3 -- Benson's method, reference-point form of the lecture (Part 1 slides 36-37)

Lecture steps
  1. Choose a reference point z* (one target per objective, in the objective's own units).
  2. s_i = z*_i - f_i(x) for a minimised objective (the 'achievement past the target', always >= 0 when met).
  3. Choose direction weights lambda_i.
  4. Maximise  sum_i lambda_i s_i(x)  subject to  s_i(x) >= 0  for every i.
Modes (the two ways of generating a front)
  'lambda'    : fixed z*, vary lambda (lambda_1 from 0 to 1)
  'reference' : fixed lambda = (1/k,..), move z* along the anti-diagonal of the [ideal, nadir] box (z_scale times the box)
Defaults: mode = 'lambda', n_points = 11, z* = nadir (everything is 'past target'), z_scale = 1.3.
"""
import numpy as np
from algorithms.base import make_run
from algorithms.classical_common import SweepState, solve, Counter, ideal_point

DEFAULTS = dict(mode="lambda", n_points=11, z=None, z_scale=1.3)


def initialize(problem, seed=0, max_evals=np.inf, mode="lambda", n_points=11, z=None, z_scale=1.3, lambdas=None, settings=None):
    custom = settings
    k = problem.n_obj
    ideal, nadir = problem.ideal, problem.nadir
    zref = np.asarray(nadir if z is None else z, float)
    if mode == "lambda":
        l1 = np.linspace(0.02, 0.98, n_points)
        settings = [(np.array([a, 1 - a]), zref) for a in l1] if k == 2 else [(np.ones(k) / k, zref)]
    else:
        t = np.linspace(0.0, 1.0, n_points)
        settings = [(np.ones(k) / k, ideal + (nadir - ideal) * z_scale * np.array([a, 1 - a])) for a in t]
    if lambdas is not None:
        settings = [(np.asarray(l), zref) for l in lambdas]
    if custom is not None:                         # explicit list of (lambda, z*) pairs, e.g. for the hand trace
        settings = [(np.asarray(l, float), np.asarray(zz, float)) for l, zz in custom]
    st = SweepState(problem, seed, settings, max_evals)
    st.mode = mode
    return st


def evaluate(st, X):
    return st.problem.evaluate(X)


def step(st):
    lam, z = st.settings[st.i]
    p = st.problem; sc = p.scale
    # maximise sum lam_i s_i  == minimise  -sum lam_i (z_i - f_i)/scale_i ;   s_i >= 0  <=>  f_i - z_i <= 0
    cons = [(lambda f, j=j: (f[j] - z[j]) / sc[j]) for j in range(p.n_obj)]
    x, F, v = solve(st.counter, lambda f: -float(lam @ ((z - f) / sc)), cons, seed=st.seed + st.i)
    ok = v <= 1e-6                               # an empty region f <= z* has no Benson solution
    if ok:
        st.X.append(x); st.F.append(F)
    st.info.append(dict(lam=lam.tolist(), z=z.tolist(), x=x.tolist(), F=F.tolist(), feasible=bool(ok), viol=v,
                        s=(z - F).tolist()))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
