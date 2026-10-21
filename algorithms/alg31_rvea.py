"""
Algorithm 31 -- RVEA, Reference Vector guided Evolutionary Algorithm (Cheng, Jin, Olhofer & Sendhoff 2016; many-objective, 5E slides 8-10)

Lecture steps
  1. Unit reference vectors v_1..v_H over the k-D simplex (the NSGA-III points, normalised to unit length).
  2. Objectives are translated by the ideal point (and scaled by the problem's ideal/nadir range) every generation.
  3. Each individual is assigned to the reference vector with the smallest ANGLE theta.
  4. Angle-penalised distance  APD = ||f'|| [1 + alpha (t / t_max) theta / gamma_v] ,  gamma_v = smallest angle between v and the other vectors
     (alpha = 2 is a fixed constant:  weak pressure toward the axes early, strong convergence pressure late).
  5. Inside each vector's sector keep only the individual with the smallest APD (one survivor per sector, so |P| <= H).
  6. Every fr * t_max generations the vectors are adapted to the population's current objective ranges.   Mating: random pairs, SBX + PM.
Defaults: H from the Das-Dennis set (p = 99 / 12 / 4 for k = 2 / 3 / 5), alpha = 2, fr = 0.1, pc = 1.0, eta_c = 30, eta_m = 20.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState
from algorithms.alg15_nsga3 import P_DEFAULT

DEFAULTS = dict(p=None, alpha=2.0, fr=0.1)


def unit(V):
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def initialize(problem, seed=0, max_evals=10000, p=None, alpha=2.0, fr=0.1, pc=1.0, eta_c=30.0, eta_m=20.0,
               constraint="penalty", **kw):
    k = problem.n_obj
    V0 = mu.das_dennis(k, P_DEFAULT.get(k, 6) if p is None else p)
    V0 = unit(np.where(V0 == 0, 1e-6, V0))
    st = PopState(problem, seed, max_evals, len(V0), constraint)
    st.V0, st.V, st.alpha, st.fr, st.pc, st.eta_c, st.eta_m = V0, V0.copy(), alpha, fr, pc, eta_c, eta_m
    st.tmax = max(1, max_evals // len(V0))
    st.init_population()
    st.t = 0
    return st


def evaluate(st, X):
    return st.evaluate(X)


def gamma_of(V):
    C = np.clip(V @ V.T, -1, 1)
    np.fill_diagonal(C, -1)
    return np.arccos(C.max(1))                      # smallest angle between v_i and any other vector


def apd_select(Fp, V, alpha, t, tmax):
    """Return indices of the survivors (one per non-empty sector) and the assignment info."""
    norms = np.linalg.norm(Fp, axis=1)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    cosv = np.clip((Fp / norms[:, None]) @ V.T, -1, 1)
    assign = cosv.argmax(1)
    theta = np.arccos(cosv[np.arange(len(Fp)), assign])
    gam = gamma_of(V)
    apd = norms * (1 + alpha * (t / tmax) * theta / np.maximum(gam[assign], 1e-9))
    keep = []
    for j in np.unique(assign):
        members = np.where(assign == j)[0]
        keep.append(members[np.argmin(apd[members])])
    return np.array(keep), assign, theta, apd


def step(st):
    n = len(st.X); H = len(st.V0)
    idx = st.rng.integers(n, size=H + H % 2)
    Q = ops.variation(st.problem, st.X[idx], st.rng, pc=st.pc, eta_c=st.eta_c, eta_m=st.eta_m)[:H]
    Fq, CVq, Sq = st.evaluate(Q)
    X = np.vstack([st.X, Q]); F = np.vstack([st.F, Fq]); CV = np.concatenate([st.CV, CVq]); S = np.vstack([st.S, Sq])
    Sn = st.problem.norm(S)
    zmin = Sn.min(0)
    Fp = Sn - zmin
    keep, assign, theta, apd = apd_select(Fp, st.V, st.alpha, st.t, st.tmax)
    st.X, st.F, st.CV, st.S = X[keep], F[keep], CV[keep], S[keep]
    st.t += 1
    if st.t % max(1, int(st.fr * st.tmax)) == 0:                      # reference vector adaptation
        Pn = st.problem.norm(st.S) - zmin
        rng_ = Pn.max(0) - Pn.min(0)
        st.V = unit(st.V0 * np.where(rng_ > 1e-12, rng_, 1.0))


run = make_run(initialize, step)
