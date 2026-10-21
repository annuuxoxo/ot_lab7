"""
Algorithm 16 -- SPEA2 (Zitzler, Laumanns & Thiele 2001; elitist, external archive; Part 3 slides 20-27)

Lecture steps
  1. Initialise a population P and an empty archive A.
  2. Strength  S(i) = |{ j : i dominates j }|  over P u A.
  3. Raw fitness  R(i) = sum of S(j) over all j that dominate i  (0 = non-dominated).
  4. Density  D(i) = 1 / (sigma_i^k + 2), sigma_i^k = distance to the k-th nearest neighbour, k = sqrt(|P| + |A|)
     (distances in normalised objective space).      5. Fitness F(i) = R(i) + D(i)  (lower is better).
  6. Environmental selection: copy all F < 1 into the archive; if it overflows truncate iteratively (remove the member with the
     smallest distance to its nearest neighbour, ties -> next nearest ...); if it underflows fill with the best dominated ones.
  7. Binary tournament on F over the archive, SBX + polynomial mutation -> next population.   8. Repeat.
Defaults: N = 100, archive = 100, pc = 0.9.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(N=100, archive=100, pc=0.9)


def spea2_fitness(S, Fn, k):
    n = len(S)
    D = mu.dominance_matrix(S)
    strength = D.sum(1)
    raw = (D * strength[:, None]).sum(0)               # sum of strengths of the dominators of each column
    dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(2))
    np.fill_diagonal(dist, np.inf)
    kk = int(min(max(1, k), n - 1)) if n > 1 else 1
    sig = np.sort(dist, axis=1)[:, kk - 1] if n > 1 else np.zeros(n)
    dens = 1.0 / (sig + 2.0)
    return raw + dens, raw, dens


def truncate(Fn, keep_n):
    idx = list(range(len(Fn)))
    dist = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(2))
    np.fill_diagonal(dist, np.inf)
    while len(idx) > keep_n:
        sub = dist[np.ix_(idx, idx)]
        srt = np.sort(sub, axis=1)
        order = np.lexsort(srt.T[::-1])                # lexicographically smallest distance vector first
        idx.pop(int(order[0]))
    return np.array(idx)


class _State(PopState):
    def output(self):
        return self.AX, self.AF, self.ACV


def evaluate(st, X):
    return st.evaluate(X)


def _env_select(st):
    X = np.vstack([st.X, st.AX]); F = np.vstack([st.F, st.AF]); CV = np.concatenate([st.CV, st.ACV]); S = np.vstack([st.S, st.AS])
    Fn = st.problem.norm(S)
    k = int(np.sqrt(len(X)))
    fit, raw, dens = spea2_fitness(S, Fn, k)
    nd = np.where(fit < 1)[0]
    if len(nd) < st.n_arch:
        rest = np.where(fit >= 1)[0]
        take = np.concatenate([nd, rest[np.argsort(fit[rest])[: st.n_arch - len(nd)]]])
    elif len(nd) > st.n_arch:
        take = nd[truncate(Fn[nd], st.n_arch)]
    else:
        take = nd
    st.AX, st.AF, st.ACV, st.AS = X[take], F[take], CV[take], S[take]
    st.Afit = fit[take]


def step(st):
    # binary tournament on the archive fitness -> mating pool -> offspring population
    n = len(st.AX)
    a = st.rng.integers(n, size=st.N + st.N % 2); b = st.rng.integers(n, size=st.N + st.N % 2)
    win = np.where(st.Afit[a] <= st.Afit[b], a, b)
    Q = ops.variation(st.problem, st.AX[win], st.rng, pc=st.pc)[: st.N]
    st.X = Q
    st.F, st.CV, st.S = st.evaluate(Q)
    _env_select(st)


def initialize(problem, seed=0, max_evals=10000, N=100, archive=100, pc=0.9, constraint="penalty", **kw):
    st = _State(problem, seed, max_evals, N, constraint)
    st.n_arch, st.pc = archive, pc
    st.init_population()
    st.AX = st.X[:0]; st.AF = st.F[:0]; st.ACV = st.CV[:0]; st.AS = st.S[:0]
    _env_select(st)
    return st


run = make_run(initialize, step)
