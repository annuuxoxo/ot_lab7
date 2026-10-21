"""
Algorithm 9 -- VIKOR (ranking / MCDM, Part 1 slide 44 and 5E slides 32-34)

Lecture steps
  1. Same decision matrix and weights as TOPSIS.
  2. Best f_j* and worst f_j- per criterion (respecting min/max).
  3. Group utility S_i = sum_j w_j (f_j* - f_ij)/(f_j* - f_j-)  and individual regret R_i = max_j of the same terms.
  4. Q_i = v (S_i - S*)/(S- - S*) + (1 - v)(R_i - R*)/(R- - R*), S* = min S, S- = max S, R* = min R, R- = max R.
  5. Rank by Q ascending (lowest = best compromise); 'acceptable advantage' Q(2nd) - Q(1st) >= 1/(m-1).
Defaults: v = 0.5.  Like TOPSIS, VIKOR ranks a given set; it does not optimise.
"""
import numpy as np

DEFAULTS = dict(v=0.5)


def vikor(M, minimize, weights=None, v=0.5):
    M = np.asarray(M, float); minimize = np.asarray(minimize, bool)
    m, k = M.shape
    w = np.ones(k) / k if weights is None else np.asarray(weights, float) / np.sum(weights)
    best = np.where(minimize, M.min(0), M.max(0)); worst = np.where(minimize, M.max(0), M.min(0))
    span = np.where(np.abs(best - worst) > 0, best - worst, 1.0)
    T = w * (best - M) / span                       # normalised regret of every alternative on every criterion
    S = T.sum(1); R = T.max(1)
    def rng(a):
        d = a.max() - a.min(); return d if d > 0 else 1.0
    Q = v * (S - S.min()) / rng(S) + (1 - v) * (R - R.min()) / rng(R)
    order = np.argsort(Q, kind="stable")
    advantage = (Q[order[1]] - Q[order[0]]) >= 1.0 / (m - 1) if m > 1 else True
    return dict(S=S, R=R, Q=Q, ranking=order, acceptable_advantage=bool(advantage))


def initialize(M, minimize, weights=None, v=0.5):
    return dict(M=np.asarray(M, float), minimize=np.asarray(minimize, bool), weights=weights, v=v)


def step(st):
    st.update(vikor(st["M"], st["minimize"], st["weights"], st["v"]))
    return st


def evaluate(st, X=None):
    return step(st)["Q"]


def run(M, minimize, weights=None, v=0.5):
    return step(initialize(M, minimize, weights, v))
