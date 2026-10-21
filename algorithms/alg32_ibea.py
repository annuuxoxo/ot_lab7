"""
Algorithm 32 -- IBEA, Indicator-Based Evolutionary Algorithm (Zitzler & Kunzli 2004; indicator based, 5E slides 11-13)

Lecture steps
  1. Additive epsilon indicator  I(a,b) = max_i (a_i - b_i)   (I <= 0  <=>  a weakly dominates b); objectives normalised to [0,1] first
     (the 5E verification slide stresses that un-normalised objectives bias the indicator).
  2. Fitness (lecture)  F(i) = sum_{j != i} -exp(-I(i,j) / kappa) ;  environmental selection repeatedly deletes the individual with the
     LARGEST (worst) F and RE-COMPUTES the fitness of the rest after each deletion until N remain.
     convention='original' uses the paper's  F(x) = sum_y -exp(-I(y,x)/kappa)  (delete the smallest) -- check_convention in the unit test.
  3. Binary tournament on fitness, SBX + polynomial mutation, merge, repeat.   There is NO explicit diversity step.
Defaults: N = 100, kappa = 0.05 (normalised objectives), convention = 'original', pc = 0.9.
"""
import numpy as np
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, kappa=0.05, convention="original", pc=0.9)


def eps_indicator_matrix(Fn):
    """I[i, j] = max_k (Fn[i,k] - Fn[j,k])."""
    return (Fn[:, None, :] - Fn[None, :, :]).max(2)


def fitness_all(I, kappa, convention):
    E = np.exp(np.clip(-I / kappa, -700, 50))                # clip: penalised (infeasible) points give huge |I|
    np.fill_diagonal(E, 0.0)
    return -E.sum(1) if convention == "lecture" else -E.sum(0)      # lecture: sum_j over I(i,j); original: sum over I(y,x)


def environmental_selection(Fn, N, kappa, convention="original"):
    n = len(Fn)
    I = eps_indicator_matrix(Fn)
    E = np.exp(np.clip(-I / kappa, -700, 50)); np.fill_diagonal(E, 0.0)
    alive = np.ones(n, bool)
    fit = -E.sum(1) if convention == "lecture" else -E.sum(0)
    order = []
    while alive.sum() > N:
        f = np.where(alive, fit, np.nan)
        worst = int(np.nanargmax(f)) if convention == "lecture" else int(np.nanargmin(f))
        alive[worst] = False; order.append(worst)
        if convention == "lecture":
            fit = fit + E[:, worst]                          # F(i) lost the term -exp(-I(i,worst))
        else:
            fit = fit + E[worst, :]                          # F(x) lost the term -exp(-I(worst,x))
    return np.where(alive)[0], fit, order


def initialize(problem, seed=0, max_evals=10000, N=100, kappa=0.05, convention="original", pc=0.9, constraint="penalty", **kw):
    st = PopState(problem, seed, max_evals, N, constraint)
    st.kappa, st.convention, st.pc = kappa, convention, pc
    st.init_population()
    st.fit = fitness_all(eps_indicator_matrix(problem.norm(st.S)), kappa, convention)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    N = st.N
    a = st.rng.integers(N, size=N + N % 2); b = st.rng.integers(N, size=N + N % 2)
    better_a = st.fit[a] > st.fit[b] if st.convention == "original" else st.fit[a] < st.fit[b]
    win = np.where(better_a, a, b)
    Q = ops.variation(st.problem, st.X[win], st.rng, pc=st.pc)[:N]
    Fq, CVq, Sq = st.evaluate(Q)
    X = np.vstack([st.X, Q]); F = np.vstack([st.F, Fq]); CV = np.concatenate([st.CV, CVq]); S = np.vstack([st.S, Sq])
    Fn = st.problem.norm(S)
    keep, fit, _ = environmental_selection(Fn, N, st.kappa, st.convention)
    st.X, st.F, st.CV, st.S = X[keep], F[keep], CV[keep], S[keep]
    st.fit = fit[keep]


run = make_run(initialize, step)
