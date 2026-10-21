"""
ops.py -- variation operators shared by the evolutionary / swarm modules (all hand-written).

real / int : simulated binary crossover (SBX) + polynomial mutation, DE/rand/1/bin
perm       : order crossover (OX) + swap / inversion mutation, 2-opt style neighbour moves
bin        : uniform crossover + bit-flip mutation (a repair keeps at least one bit set)
"""
import numpy as np
from moo_utils import crowding_distance


def clip_x(problem, X):
    if problem.kind in ("perm",):
        return X
    if problem.kind == "bin":
        return (X > 0.5).astype(int)
    X = np.minimum(np.maximum(X, problem.lb), problem.ub)
    return np.rint(X) if problem.kind == "int" else X


def repair_bin(X, rng):
    X = X.copy()
    empty = np.where(X.sum(1) == 0)[0]
    for i in empty:
        X[i, rng.integers(X.shape[1])] = 1
    return X


# ----------------------------------------------------------------------------- real coded
def sbx(P, lb, ub, rng, eta=15.0, pc=0.9):
    """Simulated binary crossover on consecutive row pairs of P."""
    p1, p2 = P[0::2].astype(float), P[1::2].astype(float)
    n, d = p1.shape
    u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    do = (rng.random((n, 1)) < pc) & (rng.random((n, d)) < 0.5)
    c1 = np.where(do, c1, p1); c2 = np.where(do, c2, p2)
    out = np.empty_like(P, dtype=float)
    out[0::2], out[1::2] = c1, c2
    return np.minimum(np.maximum(out, lb), ub)


def poly_mutation(X, lb, ub, rng, eta=20.0, pm=None):
    X = X.astype(float).copy()
    n, d = X.shape
    pm = 1.0 / d if pm is None else pm
    span = np.where(ub - lb > 0, ub - lb, 1.0)
    mask = rng.random((n, d)) < pm
    u = rng.random((n, d))
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    X = X + mask * delta * span
    return np.minimum(np.maximum(X, lb), ub)


# ----------------------------------------------------------------------------- permutations
CFG = {"perm_mut": "mixed", "perm_cx": "ox"}      # experiment switches: 'mixed' | 'swap' | 'inversion' ; 'ox' | 'none'


def order_crossover(a, b, rng):
    n = len(a)
    i, j = sorted(rng.choice(n, 2, replace=False))
    def child(p, q):
        c = -np.ones(n, int); c[i:j + 1] = p[i:j + 1]
        fill = [g for g in q if g not in set(p[i:j + 1])]
        pos = [k for k in list(range(j + 1, n)) + list(range(0, i))]
        for k, g in zip(pos, fill):
            c[k] = g
        return c
    return child(a, b), child(b, a)


def perm_mutation(x, rng):
    x = x.copy(); n = len(x)
    i, j = sorted(rng.choice(n, 2, replace=False))
    mode = CFG["perm_mut"]
    if mode == "swap" or (mode == "mixed" and rng.random() < 0.5):
        x[i], x[j] = x[j], x[i]                       # swap
    else:
        x[i:j + 1] = x[i:j + 1][::-1]                 # inversion
    return x


# ----------------------------------------------------------------------------- generic
def variation(problem, P, rng, pc=0.9, pm=None, eta_c=15.0, eta_m=20.0):
    """Offspring from the rows of P (consecutive rows are mating pairs; len(P) must be even)."""
    k = problem.kind
    if k in ("real", "int"):
        C = sbx(P, problem.lb, problem.ub, rng, eta_c, pc)
        C = poly_mutation(C, problem.lb, problem.ub, rng, eta_m, pm)
        return np.rint(C) if k == "int" else C
    if k == "bin":
        n, d = P.shape
        C = P.copy()
        for t in range(0, n - 1, 2):
            if rng.random() < pc:
                m = rng.random(d) < 0.5
                C[t, m], C[t + 1, m] = P[t + 1, m], P[t, m]
        flip = rng.random(C.shape) < ((1.0 / d) if pm is None else pm)
        C = np.where(flip, 1 - C, C)
        return repair_bin(C, rng)
    if k == "perm":
        C = P.copy()
        for t in range(0, len(P) - 1, 2):
            if CFG["perm_cx"] == "ox" and rng.random() < pc:
                C[t], C[t + 1] = order_crossover(P[t], P[t + 1], rng)
        for t in range(len(C)):
            if rng.random() < (0.3 if pm is None else pm):
                C[t] = perm_mutation(C[t], rng)
        return C
    raise ValueError(k)


def neighbour(problem, x, rng, step=0.1):
    """Small random move used by simulated annealing, hill climbing and bat local walks."""
    k = problem.kind
    if k == "perm":
        return perm_mutation(x, rng)
    if k == "bin":
        y = x.copy(); y[rng.integers(len(y))] ^= 1
        return repair_bin(y[None], rng)[0]
    span = problem.ub - problem.lb
    if k == "int":
        y = x.copy(); j = rng.integers(len(y)); y[j] += rng.choice([-1, 1])
        return np.minimum(np.maximum(y, problem.lb), problem.ub)
    y = x + rng.normal(0, step, len(x)) * span * (rng.random(len(x)) < max(1.0 / len(x), 0.3))
    return np.minimum(np.maximum(y, problem.lb), problem.ub)


def de_trial(problem, pop, i, r, F, CR, rng, jrand=None, rand=None):
    """DE/rand/1/bin trial vector for target i using donors r=(r1,r2,r3)  (V = X_r1 + F (X_r2 - X_r3))."""
    d = pop.shape[1]
    V = pop[r[0]] + F * (pop[r[1]] - pop[r[2]])
    rand = rng.random(d) if rand is None else np.asarray(rand)
    jrand = rng.integers(d) if jrand is None else jrand
    take = (rand <= CR); take[jrand] = True
    U = np.where(take, V, pop[i])
    U = np.minimum(np.maximum(U, problem.lb), problem.ub)
    return np.rint(U) if problem.kind == "int" else U


# ----------------------------------------------------------------------------- selection
def crowded_tournament(rank, crowd, rng, n_pick):
    """Binary tournament with the crowded-comparison operator (lower rank, then larger crowding)."""
    N = len(rank)
    a = rng.integers(N, size=n_pick); b = rng.integers(N, size=n_pick)
    better_a = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    pick_a = np.where(tie, rng.random(n_pick) < 0.5, better_a)
    return np.where(pick_a, a, b)


def front_crowding(F, fronts):
    cd = np.zeros(len(F))
    for fr in fronts:
        cd[fr] = crowding_distance(F[fr])
    return cd


def pair_up(idx):
    idx = np.asarray(idx)
    return idx if len(idx) % 2 == 0 else idx[:-1]
