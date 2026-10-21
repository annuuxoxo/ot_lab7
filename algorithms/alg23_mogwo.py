"""
Algorithm 23 -- MOGWO, Multi-Objective Grey Wolf Optimiser (Mirjalili, Saremi, Mirjalili & Coelho 2016; swarm, Part 4 slide 20)

Lecture steps
  1. Initialise the wolf pack and the archive of non-dominated solutions.
  2. The three leaders alpha, beta, delta are re-selected from the archive every iteration (crowding-distance tournament,
     so sparse regions of the front are preferred) -- not by a single-objective rank.
  3. a decreases linearly 2 -> 0; for every wolf and leader: A = 2 a r1 - a, C = 2 r2, D = |C X_leader - X|, X_k = X_leader - A D.
  4. New position X = (X_alpha + X_beta + X_delta) / 3 (encircling), clipped to the box.
  5. Evaluate; update the archive (crowding truncation); repeat.
Binary variables (P9): positions stay continuous in [0,1] and the evaluated mask is  position > 0.5  (with a sigmoid-flavoured
random flip to keep exploring).   Defaults: N = 100 wolves, archive = 100.
"""
import numpy as np
import ops
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(N=100, archive=100)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def decode(problem, Xc):
    if problem.kind == "bin":
        return (Xc > 0.5).astype(int)
    return Xc


def encircle(X, leaders, a, r1, r2):
    """X_k = X_leader - A |C X_leader - X| for the three leaders, new position = (X_1 + X_2 + X_3) / 3.
    leaders: list of 3 arrays like X; r1, r2: lists of 3 random arrays like X."""
    new = np.zeros_like(X, dtype=float)
    for Xl, a1, a2 in zip(leaders, r1, r2):
        A = 2 * a * a1 - a
        C = 2 * a2
        new += Xl - A * np.abs(C * Xl - X)
    return new / 3.0


def initialize(problem, seed=0, max_evals=10000, N=100, archive=100, constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.N = N
    if problem.kind == "bin":
        st.Xc = st.rng.random((N, problem.n_var)); lo, hi = np.zeros(problem.n_var), np.ones(problem.n_var)
    else:
        st.Xc = problem.random_x(N, st.rng).astype(float); lo, hi = problem.lb, problem.ub
    st.lo, st.hi = lo, hi
    st.X = decode(problem, st.Xc)
    if problem.kind == "bin":
        st.X = ops.repair_bin(st.X, st.rng)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    st.T = max(1, max_evals // N)
    st.archXc = st.Xc.copy()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    rng, N = st.rng, st.N
    a = 2.0 * (1.0 - st.gen / st.T)
    n_arch = len(st.arch)
    # leader positions: archive members are decoded vectors; for 'bin' use them as 0/1 positions
    L = st.arch.X.astype(float)
    picks = [st.arch.leader(rng, N) for _ in range(3)]            # alpha, beta, delta for every wolf
    Xn = encircle(st.Xc, [L[picks[k]] for k in range(3)], a, [rng.random(st.Xc.shape) for _ in range(3)],
                  [rng.random(st.Xc.shape) for _ in range(3)])
    Xn = np.minimum(np.maximum(Xn, st.lo), st.hi)
    st.Xc = Xn
    if st.problem.kind == "bin":
        s = 1.0 / (1.0 + np.exp(-10 * (Xn - 0.5)))
        st.X = ops.repair_bin((rng.random(Xn.shape) < s).astype(int), rng)
    else:
        st.X = np.rint(Xn) if st.problem.kind == "int" else Xn
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.arch.add(st.X, st.F, st.CV)


run = make_run(initialize, step)
