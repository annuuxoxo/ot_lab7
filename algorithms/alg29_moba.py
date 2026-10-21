"""
Algorithm 29 -- MOBA, Multi-Objective Bat Algorithm (Yang 2011, extended to MOO; swarm, Part 4 slide 62)

Lecture steps
  1. Initialise N bats (positions x_i, velocities 0, loudness A_i = A0, pulse rate r_i) and the archive.
  2. Frequency  f_i = f_min + (f_max - f_min) beta,  beta ~ U(0,1).
  3. x_best = an archive member chosen by the crowding-distance tournament;  v_i <- v_i + (x_best - x_i) f_i ;  x_new = x_i + v_i.
  4. With probability (1 - r_i) replace x_new by a small local walk around x_best:  x_best + eps A_mean (eps ~ U(-1,1), scaled by 10 % of the range).
  5. Accept x_new if it dominates x_i, or if it is non-dominated and rand < A_i; then loudness A_i <- alpha A_i and the pulse
     rate r_i <- r0 (1 - exp(-gamma t)).     6. Offer every non-dominated point to the archive.
Defaults: N = 50, f_min = 0, f_max = 2, A0 = 1, alpha = 0.9 (loudness decay), r0 = 0.5, gamma = 0.9, archive = 100.
"""
import numpy as np
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(N=50, fmin=0.0, fmax=2.0, A0=1.0, alpha=0.9, r0=0.5, gamma=0.9)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def frequency(fmin, fmax, beta):
    return fmin + (fmax - fmin) * beta


def velocity_update(v, x, x_best, f):
    """lecture form  v <- v + (x_best - x) f_i"""
    return v + (x_best - x) * f


def initialize(problem, seed=0, max_evals=10000, N=50, fmin=0.0, fmax=2.0, A0=1.0, alpha=0.9, r0=0.5, gamma=0.9,
               archive=100, constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.N, st.fmin, st.fmax, st.alpha, st.r0, st.gamma = N, fmin, fmax, alpha, r0, gamma
    st.lo, st.hi = (np.zeros(problem.n_var), np.ones(problem.n_var)) if problem.kind == "bin" else (problem.lb, problem.ub)
    st.Xc = st.rng.uniform(st.lo, st.hi, (N, problem.n_var))
    st.V = np.zeros_like(st.Xc)
    st.A = np.full(N, A0); st.r = np.zeros(N)
    st.X = _decode(st)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    return st


def _decode(st, Xc=None):
    Xc = st.Xc if Xc is None else Xc
    if st.problem.kind == "bin":
        import ops
        return ops.repair_bin((Xc > 0.5).astype(int), st.rng)
    return np.rint(Xc) if st.problem.kind == "int" else Xc


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    rng, N, p = st.rng, st.N, st.problem
    span = st.hi - st.lo
    best = st.arch.X[st.arch.leader(rng, N)].astype(float)
    if p.kind == "bin":
        best = best.copy()
    fr = frequency(st.fmin, st.fmax, rng.random((N, 1)))
    st.V = np.clip(velocity_update(st.V, st.Xc, best, fr), -span, span)
    Xn = st.Xc + st.V
    walk = rng.random(N) > st.r
    eps = rng.uniform(-1, 1, (N, Xn.shape[1]))
    Xn[walk] = best[walk] + eps[walk] * st.A.mean() * 0.1 * span
    Xn = np.minimum(np.maximum(Xn, st.lo), st.hi)
    Xd = _decode(st, Xn)
    Fn, CVn, Sn = st.evaluate(Xd)
    new_dom = np.all(Sn <= st.S, axis=1) & np.any(Sn < st.S, axis=1)
    old_dom = np.all(st.S <= Sn, axis=1) & np.any(st.S < Sn, axis=1)
    take = new_dom | (~old_dom & (rng.random(N) < st.A))
    st.Xc[take], st.X[take], st.F[take], st.CV[take], st.S[take] = Xn[take], Xd[take], Fn[take], CVn[take], Sn[take]
    st.A[take] *= st.alpha
    st.r[take] = st.r0 * (1 - np.exp(-st.gamma * (st.gen + 1)))
    st.arch.add(Xd, Fn, CVn)


run = make_run(initialize, step)
