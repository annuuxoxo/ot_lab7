"""
Algorithm 8 -- TOPSIS (ranking / MCDM, Part 1 slide 43 and 5E slides 29-31)

Lecture steps
  1. Build the m x k decision matrix of already generated alternatives; vector-normalise each column r_ij = x_ij / sqrt(sum x_ij^2).
  2. Weighted matrix v_ij = w_j r_ij.
  3. Ideal A+ (best per column, respecting min/max) and anti-ideal A- (worst per column).
  4. Euclidean distances d_i+ and d_i- ; closeness C_i = d_i- / (d_i+ + d_i-) ; rank by C_i descending.
TOPSIS never searches: it is NOT an optimiser, it only ranks a given set.
Defaults: weights from the caller (equal if None).
"""
import numpy as np

DEFAULTS = dict(weights=None)


def topsis(M, minimize, weights=None):
    """Return (closeness, ranking) ; ranking[0] is the index of the best alternative."""
    M = np.asarray(M, float); minimize = np.asarray(minimize, bool)
    m, k = M.shape
    w = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
    R = M / np.sqrt((M ** 2).sum(0))
    V = R * w
    best = np.where(minimize, V.min(0), V.max(0)); worst = np.where(minimize, V.max(0), V.min(0))
    dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1))
    C = dm / np.where(dp + dm > 0, dp + dm, 1.0)
    return C, np.argsort(-C, kind="stable")


# interface of the shared module contract (no iterations: one call ranks the set)
def initialize(M, minimize, weights=None):
    return dict(M=np.asarray(M, float), minimize=np.asarray(minimize, bool), weights=weights)


def step(st):
    st["C"], st["rank"] = topsis(st["M"], st["minimize"], st["weights"])
    return st


def evaluate(st, X=None):
    return step(st)["C"]


def run(M, minimize, weights=None):
    return step(initialize(M, minimize, weights))
