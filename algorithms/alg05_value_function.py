"""
Algorithm 5 -- Value Function method (classical, Part 1 slide 39)

Lecture steps
  1. Elicit the decision maker's value function v(f_1..f_k).
  2. Substitute F(x) into v and solve  max v(F(x))  as ONE single-objective problem.
  3. The optimum x* is the DM's most preferred point -- no ranking of a front is needed.
The value functions below are our stand-ins for an elicited DM:
  P4  : v = return - (gamma/2) risk   (mean-variance utility; gamma = risk aversion: conservative 60, balanced 10, aggressive 1.5)
  P14 : v = -[ ((weight-1.8)/0.2)^2 + ((runtime-12)/2)^2 ]   (wants ~1.8 kg and ~12 h)
Defaults: P4 profiles {conservative, balanced, aggressive}; P14 single DM.  `gammas` can be swept to trace a front.
"""
import numpy as np
from algorithms.base import make_run
from algorithms.classical_common import SweepState, solve

DEFAULTS = dict(gammas=(60.0, 10.0, 1.5))
PROFILES = {"conservative": 60.0, "balanced": 10.0, "aggressive": 1.5}


def value_function(problem, param):
    nm = problem.name
    if nm.startswith("P4"):
        # F = (risk, -return) in minimisation form
        return lambda f: -(-f[1] - 0.5 * param * f[0])
    if nm == "P14":
        t = np.array([1.8, 12.0]); s = np.array([0.2, 2.0])
        return lambda f: float(np.sum(((np.array([f[0], -f[1]]) - t) / s) ** 2))
    raise ValueError("no value function defined for " + nm)


def initialize(problem, seed=0, max_evals=np.inf, gammas=(60.0, 10.0, 1.5)):
    settings = list(gammas) if problem.name.startswith("P4") else [1.0]
    return SweepState(problem, seed, settings, max_evals)


def evaluate(st, X):
    return st.problem.evaluate(X)


def step(st):
    g = st.settings[st.i]
    vf = value_function(st.problem, g)                 # returns the NEGATIVE value (we minimise)
    x, F, v = solve(st.counter, vf, seed=st.seed + st.i)
    if st.problem.name.startswith("P4"):
        x = x / x.sum() if x.sum() > 0 else x
    st.X.append(x); st.F.append(F)
    st.info.append(dict(param=g, x=x.tolist(), F=F.tolist(), value=-float(vf(F))))
    st.i += 1


run = make_run(initialize, step, default_max_evals=np.inf)   # a sweep always finishes
