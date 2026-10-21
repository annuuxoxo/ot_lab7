"""
Algorithm 18 -- MOEA/D (Zhang & Li 2007; elitist, decomposition; Part 3 slides 36-43)

Lecture steps
  1. Initialise N weight vectors lambda_1..lambda_N spread over the objective space.
  2. Neighbourhood B(i) = the T weight vectors closest to lambda_i.
  3. Initialise one solution per sub-problem and the ideal point z*.
  4. For each sub-problem i: pick two parents from B(i) (with probability delta, otherwise from the whole population).
  5. SBX + polynomial mutation produce one child y.            6. Update the ideal point z* with y.
  7. Update neighbours: for at most n_r sub-problems j in the chosen set, replace x_j by y if g^te(y|lambda_j,z*) <= g^te(x_j|lambda_j,z*).
  8. Update the external archive (non-dominated solutions); repeat.
Scalarisation: Tchebycheff on objectives scaled by (nadir - ideal) of the problem.
Defaults: N = 100 (k=2) / smallest Das-Dennis set >= N otherwise, T = 20, delta = 0.9, n_r = 2, pc = 1.0, eta_c = 20, eta_m = 20.
"""
import numpy as np
import ops
from algorithms.base import make_run, Archive
from algorithms.moead_common import DecompState, weight_vectors, neighbourhoods, tcheby

DEFAULTS = dict(N=100, T=20, delta=0.9, nr=2)


def initialize(problem, seed=0, max_evals=10000, N=100, T=20, delta=0.9, nr=2, pc=1.0, eta_c=20.0, eta_m=20.0,
               constraint="penalty", **kw):
    st = DecompState(problem, seed, max_evals, constraint)
    st.W = weight_vectors(problem.n_obj, N)
    st.N = len(st.W)
    st.T = min(T, st.N); st.B = neighbourhoods(st.W, st.T)
    st.delta, st.nr, st.pc, st.eta_c, st.eta_m = delta, nr, pc, eta_c, eta_m
    st.X = problem.random_x(st.N, st.rng)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.Sn = problem.norm(st.S)                      # normalised selection objectives of the current solutions
    st.z = st.Sn.min(0)
    st.arch = Archive(st.N)
    st.arch.add(st.X, st.F, st.CV)
    st.buf = []                                     # children waiting to be offered to the archive (flushed every sweep)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def make_child(st, i, pool):
    a, b = st.rng.choice(pool, 2, replace=False) if len(pool) > 1 else (pool[0], pool[0])
    return ops.variation(st.problem, st.X[[a, b]], st.rng, pc=st.pc, eta_c=st.eta_c, eta_m=st.eta_m)[0:1]


def step(st):
    for i in st.rng.permutation(st.N):
        pool = st.B[i] if st.rng.random() < st.delta else np.arange(st.N)
        y = make_child(st, i, pool)
        Fy, CVy, Sy = st.evaluate(y)
        yn = st.problem.norm(Sy)[0]
        st.z = np.minimum(st.z, yn)
        _update(st, y[0], Fy[0], CVy[0], Sy[0], yn, pool)
        if st.evals >= st.max_evals:
            break
    flush(st)


def flush(st):
    if st.buf:
        xs, fs, cs = zip(*st.buf)
        st.arch.add(np.array(xs), np.array(fs), np.array(cs))
        st.buf = []


def _update(st, x, f, cv, s, yn, pool):
    gy = tcheby(yn, st.W[pool], st.z)
    gx = tcheby(st.Sn[pool], st.W[pool], st.z)
    better = np.where(gy <= gx)[0]
    st.rng.shuffle(better)
    for t in better[: st.nr]:
        j = pool[t]
        st.X[j], st.F[j], st.CV[j], st.S[j], st.Sn[j] = x, f, cv, s, yn
    st.buf.append((x, f, cv))


run = make_run(initialize, step)
