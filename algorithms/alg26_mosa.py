"""
Algorithm 26 -- MOSA, Multi-Objective Simulated Annealing (Kirkpatrick-style SA with Pareto acceptance; Part 4 slide 41)

Lecture steps
  1. Start from a solution x with temperature T0.
  2. Draw a random neighbour y.
  3. If y is non-dominated with respect to x (or dominates it) ALWAYS accept.
  4. If y is dominated by x accept with the Boltzmann probability  exp(-dE / T),  dE = sum over the WORSENED objectives of
     (f_m(y) - f_m(x))   (objectives normalised to [0,1] so one temperature fits all).
  5. Cool:  T <- alpha T.   6. Every accepted non-dominated point is offered to the archive.  Repeat.
The lecture's agent is a single walker.  With a 10,000 evaluation budget one walker is far too weak, so `chains` independent walkers
share one archive (chains = 1 gives the pure lecture algorithm).   alpha is chosen so that T falls from T0 to T_end exactly at the end
of the budget (geometric schedule).   Defaults: chains = 10, T0 = 1.0, T_end = 1e-3, step = 0.1 of the variable range, archive = 100.
"""
import numpy as np
import ops
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(chains=10, T0=1.0, T_end=1e-3, step=0.1)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def accept_probability(dE, T):
    """Boltzmann probability of accepting a DOMINATED neighbour:  exp(-dE / T)."""
    return float(np.exp(-dE / T))


def initialize(problem, seed=0, max_evals=10000, chains=10, T0=1.0, T_end=1e-3, step=0.1, archive=100,
               constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.chains, st.T, st.step_size = chains, T0, step
    iters = max(1, max_evals // chains)
    st.alpha = (T_end / T0) ** (1.0 / iters)
    st.X = problem.random_x(chains, st.rng)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    p, rng = st.problem, st.rng
    Y = np.array([ops.neighbour(p, x, rng, st.step_size) for x in st.X])
    Fy, CVy, Sy = st.evaluate(Y)
    Sn, Syn = p.norm(st.S), p.norm(Sy)
    new_dom = np.all(Syn <= Sn, axis=1) & np.any(Syn < Sn, axis=1)
    old_dom = np.all(Sn <= Syn, axis=1) & np.any(Sn < Syn, axis=1)
    dE = np.where(Syn > Sn, Syn - Sn, 0.0).sum(1)                       # only the worsened objectives
    accept = ~old_dom | (rng.random(st.chains) < np.exp(-dE / st.T))      # = accept_probability(dE, T) per chain
    st.X[accept], st.F[accept], st.CV[accept], st.S[accept] = Y[accept], Fy[accept], CVy[accept], Sy[accept]
    st.arch.add(Y[accept], Fy[accept], CVy[accept])
    st.T *= st.alpha


run = make_run(initialize, step)
