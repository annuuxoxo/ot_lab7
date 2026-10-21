"""
Algorithm 20 -- MOEA/D-ACO (Ke, Zhang & Battiti 2013; elitist, decomposition + ant colony; Part 3 slides 51-57)
Permutation (routing) problems, here the bi-objective TSP P8.

Lecture steps
  1. N sub-problems with weight vectors lambda_i (Tchebycheff scalarisation) and neighbourhoods B(i); every sub-problem owns a
     pheromone matrix tau_i and a heuristic matrix  eta_i = eta_1^lambda_i1 * eta_2^lambda_i2  (eta_k = 1 / cost_k).
  2. An ant of sub-problem i builds a tour city by city with probability  p_cd ~ tau_i(c,d)^alpha * eta_i(c,d)^beta.
  3. The new tour is evaluated and offered to the sub-problems of B(i): it replaces x_j when it is better for lambda_j
     (at most n_r replacements) -- exactly the MOEA/D update.
  4. Pheromone update  tau_i <- (1 - rho) tau_i + delta_tau : deposits come from the current solutions of the neighbourhood B(i).
  5. External archive of non-dominated tours; repeat.
Simplification: one ant per sub-problem per iteration; deposit = Q / (normalised Tchebycheff value); tau clipped to [0.02, 20].
Defaults: N = 20 sub-problems, T = 5, alpha = 1, beta = 2, rho = 0.3, Q = 1, n_r = 2.
"""
import numpy as np
from algorithms.base import make_run, Archive
from algorithms.moead_common import DecompState, neighbourhoods, tcheby

DEFAULTS = dict(N=20, T=5, alpha=1.0, beta=2.0, rho=0.3, Q=1.0, nr=2)


def transition_probs(tau_row, eta_row, alpha, beta, allowed):
    """p_cd = tau^alpha eta^beta / sum over the cities still allowed."""
    w = (np.asarray(tau_row, float) ** alpha) * (np.asarray(eta_row, float) ** beta) * np.asarray(allowed, float)
    return w / w.sum()


def initialize(problem, seed=0, max_evals=10000, N=20, T=5, alpha=1.0, beta=2.0, rho=0.3, Q=1.0, nr=2, **kw):
    st = DecompState(problem, seed, max_evals, "ignore")
    n = problem.n_var
    a = np.linspace(0, 1, N)
    st.W = np.stack([a, 1 - a], 1); st.N = N
    st.B = neighbourhoods(st.W, T); st.alpha, st.beta, st.rho, st.Q, st.nr = alpha, beta, rho, Q, nr
    D, Tm = problem.D.copy(), problem.T.copy()
    np.fill_diagonal(D, 1.0); np.fill_diagonal(Tm, 1.0)
    st.eta1, st.eta2 = 1.0 / D, 1.0 / Tm
    st.eta = [(st.eta1 ** w[0]) * (st.eta2 ** w[1]) for w in st.W]      # eta_i = eta1^l1 * eta2^l2
    st.tau = np.ones((N, n, n))
    st.X = problem.random_x(N, st.rng)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.Sn = problem.norm(st.S); st.z = st.Sn.min(0)
    st.arch = Archive(100); st.arch.add(st.X, st.F, st.CV)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def build_tour(st, i):
    n = st.problem.n_var
    tau, eta = st.tau[i], st.eta[i]
    cur = int(st.rng.integers(n)); tour = [cur]; left = np.ones(n, bool); left[cur] = False
    for _ in range(n - 1):
        w = (tau[cur] ** st.alpha) * (eta[cur] ** st.beta) * left
        s = w.sum()
        nxt = int(st.rng.choice(n, p=w / s)) if s > 0 else int(np.flatnonzero(left)[0])
        tour.append(nxt); left[nxt] = False; cur = nxt
    return np.array(tour)


def step(st):
    p = st.problem; n = p.n_var
    tours = []
    for i in range(st.N):
        t = build_tour(st, i)
        F, CV, S = st.evaluate(t[None])
        yn = p.norm(S)[0]
        st.z = np.minimum(st.z, yn)
        pool = st.B[i]
        gy = tcheby(yn, st.W[pool], st.z); gx = tcheby(st.Sn[pool], st.W[pool], st.z)
        better = np.where(gy < gx)[0]; st.rng.shuffle(better)
        for q in better[: st.nr]:
            j = pool[q]
            st.X[j], st.F[j], st.CV[j], st.S[j], st.Sn[j] = t, F[0], CV[0], S[0], yn
        tours.append((t, F[0]))
    xs, fs = zip(*tours)
    st.arch.add(np.array(xs), np.array(fs), np.zeros(len(xs)))
    # pheromone update: evaporation + deposits of the neighbourhood's current solutions
    for i in range(st.N):
        tau = st.tau[i] * (1 - st.rho)
        for j in st.B[i]:
            g = tcheby(st.Sn[j], st.W[i], st.z) + 1e-3
            dep = st.Q / (1.0 + 10 * g)
            x = st.X[j]; nx = np.roll(x, -1)
            tau[x, nx] += dep; tau[nx, x] += dep
        st.tau[i] = np.clip(tau, 0.02, 20.0)


run = make_run(initialize, step)
