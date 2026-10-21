"""
Algorithm 2 -- epsilon-Constraint (classical, Part 1 slide 33)

Lecture steps
  1. Choose the primary objective f_1 (optimised directly).
  2. Fix an upper bound eps_i for every other objective.
  3. Solve  min f_1(x)  s.t.  f_i(x) <= eps_i  (i = 2..k).
  4. Vary eps_i and re-solve to sweep out the front.
Defaults: primary = objective 1, n_eps = 11 values per constrained objective, equally spaced between the ideal value
and the nadir value of the problem (so the sweep covers the whole range of that objective).
"""
import itertools
import numpy as np
from algorithms.base import make_run
from algorithms.classical_common import SweepState, solve, ideal_point

DEFAULTS = dict(n_eps=11, primary=0)


def initialize(problem, seed=0, max_evals=np.inf, n_eps=11, primary=0, eps_list=None):
    from algorithms.classical_common import Counter
    c = Counter(problem)
    ideal, pay, _ = ideal_point(c, seed)
    others = [j for j in range(problem.n_obj) if j != primary]
    if eps_list is None:
        grids = [np.linspace(problem.ideal[j], problem.nadir[j], n_eps) for j in others]
        eps_list = [dict(zip(others, combo)) for combo in itertools.product(*grids)]
    st = SweepState(problem, seed, eps_list, max_evals)
    st.counter.n += c.n; st.primary = primary; st.ideal = ideal; st.payoff = pay
    return st


def evaluate(st, X):
    return st.problem.evaluate(X)


def step(st):
    eps = st.settings[st.i]; pr = st.primary
    cons = [(lambda f, j=j, e=e: (f[j] - e) / st.problem.scale[j]) for j, e in eps.items()]
    x, F, v = solve(st.counter, lambda f: f[pr], cons, seed=st.seed + st.i)
    st.X.append(x); st.F.append(F); st.info.append(dict(eps={int(k): float(e) for k, e in eps.items()}, x=x.tolist(), F=F.tolist(), viol=v))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
