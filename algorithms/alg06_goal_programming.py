"""
Algorithm 6 -- Goal Programming (classical, Part 1 slide 40 and 5E slides 23-25)

Lecture steps
  1. Set a target T_i for every objective.
  2. Introduce deviation variables d_i^- (undershoot) and d_i^+ (overshoot), both >= 0.
  3. Minimise the weighted undesired deviation subject to  f_i(x) + d_i^- - d_i^+ = T_i  and the original constraints.
For the linear problems P11 and P7 the whole thing is ONE linear program, solved by our simplex (solvers.linprog) with the
deviation variables as extra LP columns.  Goals are stated in the natural direction ('at least' = maximised objective,
penalise d^-).  Defaults: unit weights; weights='normalized' divides each deviation by its target.
"""
import numpy as np
import solvers as sv
from algorithms.base import make_run, Result
from algorithms.classical_common import SweepState

DEFAULTS = dict(weights="unit")
GOALS = {"P11": dict(targets=(2400.0, 600.0), sense=("ge", "ge")),
         "P7": dict(targets=(40.0, 60.0), sense=("ge", "ge"))}


def solve_gp(problem, targets, sense, w):
    """LP with variables [x (n), d- (k), d+ (k)].   Returns x, d-, d+, objective."""
    C, A_ub, B_ub = problem.C, problem.A_ub, problem.B_ub
    n, k = C.shape[1], C.shape[0]
    nv = n + 2 * k
    A_eq = np.zeros((k, nv)); b_eq = np.array(targets, float)
    for j in range(k):
        A_eq[j, :n] = C[j]; A_eq[j, n + j] = 1.0; A_eq[j, n + k + j] = -1.0       # f_j + d- - d+ = T_j
    A = np.hstack([A_ub, np.zeros((len(A_ub), 2 * k))])
    c = np.zeros(nv)
    for j in range(k):
        if sense[j] in ("ge", "eq"):
            c[n + j] = w[j]                           # undershoot is undesired
        if sense[j] in ("le", "eq"):
            c[n + k + j] = w[j]                       # overshoot is undesired
    lb = list(problem.lb) + [0.0] * (2 * k)
    bounds = [(lo, None) for lo in lb]
    x, fun, status = sv.linprog(c, A, B_ub, A_eq, b_eq, bounds)
    return x[:n], x[n:n + k], x[n + k:], fun, status


def initialize(problem, seed=0, max_evals=np.inf, weights="unit", targets=None, sense=None, weight_list=None):
    g = GOALS[problem.name]
    T = np.array(g["targets"] if targets is None else targets, float)
    sense = g["sense"] if sense is None else sense
    if weight_list is None:
        weight_list = [np.ones(len(T))] if weights == "unit" else [1.0 / np.abs(T)]
    st = SweepState(problem, seed, weight_list, max_evals)
    st.T, st.sense = T, sense
    return st


def evaluate(st, X):
    return st.problem.evaluate(X)


def step(st):
    w = np.asarray(st.settings[st.i], float)
    x, dm, dp, fun, status = solve_gp(st.problem, st.T, st.sense, w)
    F, _ = st.problem.evaluate(x[None, :]); st.counter.n += 1
    st.X.append(x); st.F.append(F[0])
    st.info.append(dict(w=w.tolist(), x=x.tolist(), d_minus=dm.tolist(), d_plus=dp.tolist(), objective=fun,
                        F_natural=st.problem.natural(F[0]).tolist(), status=status))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
