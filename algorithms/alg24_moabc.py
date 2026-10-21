"""
Algorithm 24 -- MOABC, Multi-Objective Artificial Bee Colony (Akbari et al. 2012; swarm, Part 4 slide 27)

Lecture steps
  1. Initialise SN food sources (solutions) with trial counters = 0 and an external archive of non-dominated solutions.
  2. EMPLOYED bees: v = x_i + phi (x_i - x_k) on one random coordinate, phi in [-1,1], k != i.  Greedy dominance choice:
     v replaces x_i if it dominates it (random choice if mutually non-dominated); otherwise the trial counter of i grows.
  3. ONLOOKER bees: choose a source with the crowded-comparison tournament on the sources' fronts (less crowded sources are
     favoured), then search around it exactly like an employed bee.
  4. SCOUT bees: a source whose trial counter exceeds `limit` is abandoned and replaced by a random solution.
  5. Every evaluated non-dominated point is offered to the archive (crowding truncation); repeat.
Defaults: SN = 50 sources, limit = 30, archive = 100 (phi is drawn uniformly; the 5-slide example fixes phi = 0.5).
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(SN=50, limit=30, archive=100)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def employed_move(xi, xk, j, phi):
    """v = x_i with coordinate j replaced by x_ij + phi (x_ij - x_kj)."""
    v = np.array(xi, float).copy()
    v[j] = xi[j] + phi * (xi[j] - xk[j])
    return v


def compare(s_new, s_old):
    """'new' if the trial dominates, 'old' if the source dominates, 'tie' if mutually non-dominated."""
    if np.all(s_new <= s_old) and np.any(s_new < s_old):
        return "new"
    if np.all(s_old <= s_new) and np.any(s_old < s_new):
        return "old"
    return "tie"


def initialize(problem, seed=0, max_evals=10000, SN=50, limit=30, archive=100, constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.SN, st.limit = SN, limit
    st.X = problem.random_x(SN, st.rng)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.trial = np.zeros(SN, int)
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def _neighbour_of(st, src_idx):
    p, rng = st.problem, st.rng
    V = st.X[src_idx].astype(float).copy()
    for t, i in enumerate(src_idx):
        k = rng.integers(st.SN - 1); k = k + 1 if k >= i else k
        j = rng.integers(p.n_var)
        V[t] = employed_move(st.X[i], st.X[k], j, rng.uniform(-1, 1))
    V = np.minimum(np.maximum(V, p.lb), p.ub)
    return np.rint(V) if p.kind == "int" else V


def _greedy(st, src_idx, V):
    Fv, CVv, Sv = st.evaluate(V)
    for t, i in enumerate(src_idx):
        c = compare(Sv[t], st.S[i])
        if c == "new" or (c == "tie" and st.rng.random() < 0.5):
            st.X[i], st.F[i], st.CV[i], st.S[i] = V[t], Fv[t], CVv[t], Sv[t]
            st.trial[i] = 0
        else:
            st.trial[i] += 1
    st.arch.add(V, Fv, CVv)


def step(st):
    SN = st.SN
    idx = np.arange(SN)
    _greedy(st, idx, _neighbour_of(st, idx))                                   # employed bees
    fronts, rank = mu.fast_non_dominated_sort(st.S)
    crowd = ops.front_crowding(st.S, fronts)
    onl = ops.crowded_tournament(rank, crowd, st.rng, SN)                      # onlookers pick less-crowded sources
    _greedy(st, onl, _neighbour_of(st, onl))
    scouts = np.where(st.trial > st.limit)[0]                                  # scouts
    if len(scouts):
        Xs = st.problem.random_x(len(scouts), st.rng)
        Fs, CVs, Ss = st.evaluate(Xs)
        st.X[scouts], st.F[scouts], st.CV[scouts], st.S[scouts] = Xs, Fs, CVs, Ss
        st.trial[scouts] = 0
        st.arch.add(Xs, Fs, CVs)


run = make_run(initialize, step)
