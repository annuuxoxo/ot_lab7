"""
Algorithm 19 -- MOEA/D-DE (Li & Zhang 2009; elitist, decomposition + differential evolution; Part 3 slides 44-50)

Lecture steps (as MOEA/D, but the variation operator is differential evolution)
  1-3. Weight vectors, neighbourhoods B(i), initial population and ideal point as in MOEA/D.
  4. For sub-problem i pick three distinct donors r1, r2, r3 from B(i) (prob delta) or from the whole population.
  5. DE mutation   V = X_r1 + F (X_r2 - X_r3);   DE binomial crossover  U_j = V_j if rand <= CR (or j = j_rand) else X_i,j ;
     followed by polynomial mutation of U.     6. Update ideal point and at most n_r neighbours by Tchebycheff comparison.
  7. External archive of non-dominated solutions.
Defaults: N = 100, T = 20, delta = 0.9, n_r = 2, F = 0.5, CR = 1.0, eta_m = 20 (CR = 1 is the value recommended by Li & Zhang).
"""
import numpy as np
import ops
from algorithms.base import make_run, Archive
from algorithms.moead_common import DecompState, weight_vectors, neighbourhoods, tcheby
from algorithms import alg18_moead as base

DEFAULTS = dict(N=100, T=20, delta=0.9, nr=2, F=0.5, CR=1.0)


def initialize(problem, seed=0, max_evals=10000, N=100, T=20, delta=0.9, nr=2, F=0.5, CR=1.0, eta_m=20.0,
               constraint="penalty", **kw):
    st = base.initialize(problem, seed=seed, max_evals=max_evals, N=N, T=T, delta=delta, nr=nr, constraint=constraint)
    st.Fde, st.CR, st.eta_m = F, CR, eta_m
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    p = st.problem
    for i in st.rng.permutation(st.N):
        pool = st.B[i] if st.rng.random() < st.delta else np.arange(st.N)
        if len(pool) >= 3:
            r = st.rng.choice(pool, 3, replace=False)
        else:
            r = st.rng.choice(st.N, 3, replace=False)
        U = ops.de_trial(p, st.X, i, r, st.Fde, st.CR, st.rng)
        if p.kind in ("real", "int"):
            U = ops.poly_mutation(U[None], p.lb, p.ub, st.rng, st.eta_m)[0]
        y = U[None]
        Fy, CVy, Sy = st.evaluate(y)
        yn = p.norm(Sy)[0]
        st.z = np.minimum(st.z, yn)
        base._update(st, y[0], Fy[0], CVy[0], Sy[0], yn, pool)
        if st.evals >= st.max_evals:
            break
    base.flush(st)


run = make_run(initialize, step)
