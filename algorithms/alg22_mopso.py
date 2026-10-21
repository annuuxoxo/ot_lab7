"""
Algorithm 22 -- MOPSO, Multi-Objective Particle Swarm Optimisation (Coello & Lechuga 2002; swarm, Part 4 slide 13)

Lecture steps
  1. Initialise particles x_i with zero velocity; personal best pbest_i = x_i; external archive of non-dominated solutions.
  2. For every particle pick a global guide gbest from the archive (binary tournament on archive crowding distance: the less
     crowded leader wins).
  3. v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x) ;   x <- x + v   (positions clipped to the box, velocity reversed at a wall).
  4. Turbulence (mutation) of a shrinking fraction of the swarm: probability (1 - t/T)^5.
  5. Evaluate; pbest is replaced if the new position dominates it (random choice when mutually non-dominated).
  6. Add non-dominated particles to the archive (crowding-distance truncation to its size limit); repeat.
Defaults: N = 100 particles, archive = 100, w = 0.4, c1 = c2 = 1.5 (the 5-slide example uses w = 0.7; 0.4 is Coello's value and converges better).
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(N=100, archive=100, w=0.4, c1=1.5, c2=1.5)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def velocity_update(v, x, pbest, gbest, w, c1, c2, r1, r2):
    """v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)"""
    return w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)


def initialize(problem, seed=0, max_evals=10000, N=100, archive=100, w=0.4, c1=1.5, c2=1.5, constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.N, st.w, st.c1, st.c2 = N, w, c1, c2
    st.X = problem.random_x(N, st.rng)
    st.V = np.zeros_like(st.X, dtype=float)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.PX, st.PF, st.PS = st.X.copy(), st.F.copy(), st.S.copy()
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    st.T = max(1, max_evals // N)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    p, rng, N = st.problem, st.rng, st.N
    lead = st.arch.leader(rng, N)
    G = st.arch.X[lead].astype(float)
    r1, r2 = rng.random(st.X.shape), rng.random(st.X.shape)
    st.V = velocity_update(st.V, st.X, st.PX, G, st.w, st.c1, st.c2, r1, r2)
    Xn = st.X + st.V
    out = (Xn < p.lb) | (Xn > p.ub)
    st.V = np.where(out, -st.V, st.V)
    Xn = np.minimum(np.maximum(Xn, p.lb), p.ub)
    pm = (1.0 - st.gen / st.T) ** 5
    mut = rng.random(N) < pm
    if mut.any():
        Xn[mut] = ops.poly_mutation(Xn[mut], p.lb, p.ub, rng, 20.0, 1.0 / p.n_var)
    st.X = Xn
    st.F, st.CV, st.S = st.evaluate(st.X)
    dom_new = np.all(st.S <= st.PS, axis=1) & np.any(st.S < st.PS, axis=1)
    dom_old = np.all(st.PS <= st.S, axis=1) & np.any(st.PS < st.S, axis=1)
    swap = dom_new | (~dom_new & ~dom_old & (rng.random(N) < 0.5))
    st.PX[swap], st.PF[swap], st.PS[swap] = st.X[swap], st.F[swap], st.S[swap]
    st.arch.add(st.X, st.F, st.CV)


run = make_run(initialize, step)
