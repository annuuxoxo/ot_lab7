"""
Algorithm 34 -- HypE, Hypervolume Estimation Algorithm (Bader & Zitzler 2011; indicator based, 5E slides 17-19)

Lecture steps
  1. Bounding box between the ideal point and the reference point r (normalised space: r = 1.1 on every axis).
  2. Draw S random sample points uniformly inside the box.
  3. For every sample find the individuals that dominate it and split that sample's volume share equally among them.
  4. An individual's fitness / contribution = sum of its shares over the S samples.
  5. Environmental selection: after non-dominated sorting, remove from the last front the individual with the LOWEST estimate, then
     RE-ESTIMATE, until the population has size N.   Mating: random pairs; SBX + polynomial mutation.
Defaults: N = 100 (72 for k=5 in the comparisons), S = 2000 samples, pc = 0.9.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, samples=2000, pc=0.9)


def hype_remove(Fn, n_remove, ref, samples, rng):
    """Iteratively delete `n_remove` individuals of Fn (rows) with the lowest estimated contribution; returns kept local indices."""
    n, k = Fn.shape
    lower = np.minimum(Fn.min(0), 0.0)
    pts = lower + rng.random((samples, k)) * (ref - lower)
    dom = np.all(Fn[None, :, :] <= pts[:, None, :], axis=2) & np.all(Fn < ref, axis=1)[None, :]    # (S, n)
    alive = np.ones(n, bool)
    for _ in range(n_remove):
        cnt = (dom & alive[None, :]).sum(1)
        share = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
        fit = (dom & alive[None, :]).astype(float).T @ share
        fit = np.where(alive, fit, np.inf)
        alive[int(np.argmin(fit))] = False
    return np.where(alive)[0]


def environmental_selection(S, Sn_fn, N, ref, samples, rng):
    fronts, _ = mu.fast_non_dominated_sort(S)
    chosen = []
    for fr in fronts:
        if len(chosen) + len(fr) <= N:
            chosen.extend(fr.tolist())
        else:
            need = N - len(chosen)
            keep_local = hype_remove(Sn_fn[fr], len(fr) - need, ref, samples, rng)
            chosen.extend(fr[keep_local].tolist())
            break
    return np.array(chosen)


def initialize(problem, seed=0, max_evals=10000, N=100, samples=2000, pc=0.9, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.samples, st.pc = samples, pc
    st.init_population()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    N = st.N
    idx = st.rng.integers(N, size=N + N % 2)
    Q = ops.variation(st.problem, st.X[idx], st.rng, pc=st.pc)[:N]
    Fq, CVq, Sq = st.evaluate(Q)
    X = np.vstack([st.X, Q]); F = np.vstack([st.F, Fq]); CV = np.concatenate([st.CV, CVq]); S = np.vstack([st.S, Sq])
    keep = environmental_selection(S, st.problem.norm(S), N, st.problem.hv_ref, st.samples, st.rng)
    st.X, st.F, st.CV, st.S = X[keep], F[keep], CV[keep], S[keep]


run = make_run(initialize, step)
