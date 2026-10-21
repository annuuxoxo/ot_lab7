"""
Algorithm 17 -- PESA-II (Corne, Jerram, Knowles & Oates 2001; elitist, grid based; Part 3 slides 28-35)

Lecture steps
  1. Divide the (archive's) objective space into a regular hyper-grid of hyper-boxes (div divisions per objective).
  2. Place every archive member into its box and count the occupants of every box (the 'squeeze factor').
  3. REGION BASED selection: choose a box by binary tournament (the box with the LOWER squeeze factor wins), then pick a random
     archive member inside it.  This is repeated for both parents of every child.
  4. SBX + polynomial mutation produce a small internal population; every child is evaluated.
  5. Archive update: a child enters only if no archive member dominates it; members it dominates are removed; if the archive
     exceeds its limit a random member of the MOST crowded box is deleted.
Output: the archive.   Defaults: archive = 100, internal population = 20, div = 16 divisions per objective, pc = 0.9.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import Core, make_run

DEFAULTS = dict(archive=100, ip=20, div=16, pc=0.9)


class State(Core):
    def output(self):
        return self.AX, self.AF, self.ACV


def grid_cells(F, div):
    lo, hi = F.min(0), F.max(0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    idx = np.minimum(((F - lo) / span * div).astype(int), div - 1)
    return idx


def cell_keys(F, div):
    idx = grid_cells(F, div)
    return [tuple(r) for r in idx]


def initialize(problem, seed=0, max_evals=10000, archive=100, ip=20, div=16, pc=0.9, constraint="penalty", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.n_arch, st.ip, st.div, st.pc = archive, ip, div, pc
    st.X = problem.random_x(ip, st.rng)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.AX = st.X[:0]; st.AF = st.F[:0]; st.ACV = st.CV[:0]; st.AS = st.S[:0]
    for i in range(ip):
        _insert(st, st.X[i], st.F[i], st.CV[i], st.S[i])
    return st


def evaluate(st, X):
    return st.evaluate(X)


def _insert(st, x, f, cv, s):
    if len(st.AS):
        dom_by = np.all(st.AS <= s, axis=1) & np.any(st.AS < s, axis=1)
        if dom_by.any() or np.any(np.all(st.AS == s, axis=1)):
            return False
        dominated = np.all(s <= st.AS, axis=1) & np.any(s < st.AS, axis=1)
        keep = ~dominated
        st.AX, st.AF, st.ACV, st.AS = st.AX[keep], st.AF[keep], st.ACV[keep], st.AS[keep]
    st.AX = np.vstack([st.AX, x[None]]) if len(st.AX) else x[None].copy()
    st.AF = np.vstack([st.AF, f[None]]) if len(st.AF) else f[None].copy()
    st.ACV = np.concatenate([st.ACV, [cv]]); st.AS = np.vstack([st.AS, s[None]]) if len(st.AS) else s[None].copy()
    if len(st.AX) > st.n_arch:                              # remove a random member of the most crowded box
        keys = cell_keys(st.AS, st.div)
        counts = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        worst = max(counts, key=counts.get)
        members = [i for i, k in enumerate(keys) if k == worst]
        drop = members[st.rng.integers(len(members))]
        keep = np.ones(len(st.AX), bool); keep[drop] = False
        st.AX, st.AF, st.ACV, st.AS = st.AX[keep], st.AF[keep], st.ACV[keep], st.AS[keep]
    return True


def select_by_region(st, n_pick):
    keys = cell_keys(st.AS, st.div)
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    cells = list(counts)
    members = {k: [] for k in cells}
    for i, k in enumerate(keys):
        members[k].append(i)
    out = []
    for _ in range(n_pick):
        a = cells[st.rng.integers(len(cells))]; b = cells[st.rng.integers(len(cells))]
        c = a if counts[a] <= counts[b] else b                     # lower squeeze factor wins
        out.append(members[c][st.rng.integers(len(members[c]))])
    return np.array(out)


def step(st):
    par = select_by_region(st, st.ip + st.ip % 2)
    Q = ops.variation(st.problem, st.AX[par], st.rng, pc=st.pc)[: st.ip]
    Fq, CVq, Sq = st.evaluate(Q)
    st.X, st.F, st.CV, st.S = Q, Fq, CVq, Sq
    for i in range(len(Q)):
        _insert(st, Q[i], Fq[i], CVq[i], Sq[i])


run = make_run(initialize, step)
