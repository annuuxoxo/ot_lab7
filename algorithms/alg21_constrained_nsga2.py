"""
Algorithm 21 -- Constrained NSGA-II (Deb 2000/2002; constrained, Part 4 slides 3-9)

Lecture steps (the three changes that turn NSGA-II into a constrained optimiser)
  1. CV(x) = sum_i max(0, g_i(x)) + sum_j |h_j(x)| for every individual.
  2. Constrained domination (Deb's three rules) replaces the plain dominance test everywhere in the algorithm:
        feasible beats infeasible;  both feasible -> Pareto dominance;  both infeasible -> smaller CV wins.
  3. Non-dominated sorting therefore ranks every feasible front above ALL infeasible points; infeasible points are ordered by CV.
Everything else is NSGA-II (crowded tournament, SBX + polynomial mutation, parent+offspring elitism).
Defaults: those of NSGA-II.
"""
from algorithms import alg14_nsga2 as nsga2
from algorithms.base import make_run

DEFAULTS = dict(nsga2.DEFAULTS)


def initialize(problem, seed=0, max_evals=10000, **kw):
    kw["constraint"] = "deb"
    return nsga2.initialize(problem, seed=seed, max_evals=max_evals, **kw)


def evaluate(st, X):
    return st.evaluate(X)


step = nsga2.step
run = make_run(initialize, step)
