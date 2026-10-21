"""
Algorithm 28 -- MO-TLBO, Multi-Objective Teaching-Learning-Based Optimisation (Rao & Patel 2013; human-inspired, Part 4 slide 55)

Lecture steps
  1. Initialise a class of N learners and an archive of non-dominated solutions.
  2. TEACHER phase: each learner's teacher is an archive member chosen by the crowding-distance tournament; the class mean M is computed;
     every learner moves   X_new = X + r (X_teacher - T_F M),  r ~ U(0,1), T_F = round(1 + rand) in {1, 2}.
  3. Greedy dominance acceptance: X_new replaces X if it dominates it (random choice if mutually non-dominated).
  4. LEARNER phase: each learner i meets a random learner j; it moves toward the better one (if i dominates j: away from j,
     X_i + r (X_i - X_j); otherwise X_i + r (X_j - X_i)); accepted by the same greedy rule.
  5. Every non-dominated result is offered to the archive.   No algorithm specific parameters beyond N.
Defaults: N = 50 learners (a teacher phase and a learner phase cost 2N evaluations per iteration), archive = 100.
"""
import numpy as np
import ops
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(N=50, archive=100)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def teacher_move(X, teacher, mean, TF, r):
    """X_new = X + r (X_teacher - T_F * X_mean)"""
    return X + r * (teacher - TF * mean)


def learner_move(Xi, Xj, i_dominates_j, r):
    """Away from j if i is better, toward j otherwise."""
    return Xi + r * ((Xi - Xj) if i_dominates_j else (Xj - Xi))


def initialize(problem, seed=0, max_evals=10000, N=50, archive=100, constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.N = N
    st.X = problem.random_x(N, st.rng).astype(float)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def _accept(st, V):
    p, rng = st.problem, st.rng
    V = np.minimum(np.maximum(V, p.lb), p.ub)
    if p.kind == "int":
        V = np.rint(V)
    Fv, CVv, Sv = st.evaluate(V)
    new_dom = np.all(Sv <= st.S, axis=1) & np.any(Sv < st.S, axis=1)
    old_dom = np.all(st.S <= Sv, axis=1) & np.any(st.S < Sv, axis=1)
    take = new_dom | (~new_dom & ~old_dom & (rng.random(len(V)) < 0.5))
    st.X[take], st.F[take], st.CV[take], st.S[take] = V[take], Fv[take], CVv[take], Sv[take]
    st.arch.add(V, Fv, CVv)


def step(st):
    rng, N = st.rng, st.N
    # teacher phase
    teacher = st.arch.X[st.arch.leader(rng, N)].astype(float)            # every learner is taught by a crowding-selected leader
    mean = st.X.mean(0)
    TF = rng.integers(1, 3, size=(N, 1))
    r = rng.random((N, st.X.shape[1]))
    _accept(st, teacher_move(st.X, teacher, mean, TF, r))
    # learner phase
    j = rng.integers(N - 1, size=N); j = j + (j >= np.arange(N))
    i_dom = np.all(st.S <= st.S[j], axis=1) & np.any(st.S < st.S[j], axis=1)
    direction = np.where(i_dom[:, None], st.X - st.X[j], st.X[j] - st.X)
    _accept(st, st.X + rng.random((N, st.X.shape[1])) * direction)


run = make_run(initialize, step)
