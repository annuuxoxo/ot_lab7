"""
Algorithm 7 -- Lexicographic method (classical, Part 1 slide 42 and 5E slides 26-28)

Lecture steps
  1. Rank the objectives strictly by priority.
  2. Optimise the priority-1 objective alone  -> f_1*.
  3. Add the constraint  f_1(x) >= f_1* - eps_1  (maximised objective)  and optimise the priority-2 objective.
  4. Repeat down the list: one 'hold the previous optimum' constraint per stage; k sequential LP solves.
Defaults: priority = objective order of the problem (profit first for P11, yield first for P7), eps = 0 at every stage.
Each setting in the sweep is (priority order, eps) so the effect of the tolerance can be traced.
"""
import numpy as np
import solvers as sv
from algorithms.base import make_run
from algorithms.classical_common import SweepState

DEFAULTS = dict(order=None, eps=0.0)


def lex_solve(problem, order, eps):
    C, A_ub, B_ub = problem.C, problem.A_ub, problem.B_ub
    n = C.shape[1]
    A = [list(r) for r in A_ub]; B = list(B_ub); stages = []
    x = None
    for t, j in enumerate(order):
        x, fun, status = sv.linprog(-C[j], np.array(A), np.array(B), bounds=[(lo, None) for lo in problem.lb])
        best = -fun
        stages.append(dict(objective=int(j), optimum=float(best), x=x.tolist()))
        tol = eps[t] if hasattr(eps, "__len__") else eps
        A.append(list(-C[j])); B.append(-(best - tol))           # f_j >= best - eps
    return x, stages


def initialize(problem, seed=0, max_evals=np.inf, order=None, eps=0.0, eps_list=None):
    order = list(range(problem.n_obj)) if order is None else list(order)
    settings = [(order, e) for e in (eps_list if eps_list is not None else [eps])]
    return SweepState(problem, seed, settings, max_evals)


def evaluate(st, X):
    return st.problem.evaluate(X)


def step(st):
    order, eps = st.settings[st.i]
    x, stages = lex_solve(st.problem, order, eps)
    F, _ = st.problem.evaluate(x[None, :]); st.counter.n += 1
    st.X.append(x); st.F.append(F[0])
    st.info.append(dict(order=list(order), eps=eps, x=x.tolist(), F_natural=st.problem.natural(F[0]).tolist(), stages=stages))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
