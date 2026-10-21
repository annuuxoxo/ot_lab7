"""
Algorithm 25 -- MOACO, Multi-Objective Ant Colony Optimisation (bi-criterion ant system; swarm, Part 4 slide 34)
Permutation problems (the bi-objective TSP P8).

Lecture steps
  1. Two pheromone matrices tau_1, tau_2 (one per objective), heuristics eta_k = 1 / cost_k.
  2. Every ant draws a weight l in [0,1] and builds a tour city by city with
        p_cd ~ [tau_1^l tau_2^(1-l)]^alpha * [eta_1^l eta_2^(1-l)]^beta .
  3. Evaluate the tours; the non-dominated ones (against this iteration's ants and the archive) are the only ones that deposit.
  4. Evaporation  tau <- (1 - rho) tau  and deposit  d_tau_k = Q / f_k(route)  (here f_k is divided by the nearest-neighbour cost so that
     Q is dimensionless) for every depositing ant on BOTH matrices.
  5. Archive of non-dominated tours; repeat.
Defaults: ants = 20, alpha = 1, beta = 2, rho = 0.5, Q = 10 (deposit shared between the depositing ants), tau in [0.01, 10].
"""
import numpy as np
import moo_utils as mu
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(ants=20, alpha=1.0, beta=2.0, rho=0.5, Q=10.0)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def update_pheromone(tau, tours, F, nd, rho, Q, ref_cost, tau_min=0.01, tau_max=10.0):
    """tau_k <- (1-rho) tau_k ; every NON-DOMINATED ant adds  Q / (f_k / ref_k) / n_depositors  on both matrices."""
    tau = [t * (1 - rho) for t in tau]
    dep = np.where(nd)[0]
    for i in dep:
        x = tours[i]; nx = np.roll(x, -1)
        for k in range(2):
            d = Q / (F[i, k] / ref_cost[k]) / max(len(dep), 1)
            tau[k][x, nx] += d; tau[k][nx, x] += d
    return [np.clip(t, tau_min, tau_max) for t in tau]


def initialize(problem, seed=0, max_evals=10000, ants=20, alpha=1.0, beta=2.0, rho=0.5, Q=10.0, **kw):
    st = State(problem, seed, max_evals, "ignore")
    n = problem.n_var
    st.m, st.alpha, st.beta, st.rho, st.Q = ants, alpha, beta, rho, Q
    D, T = problem.D.copy(), problem.T.copy()
    np.fill_diagonal(D, 1.0); np.fill_diagonal(T, 1.0)
    st.eta = [1.0 / D, 1.0 / T]
    nn = [problem.nearest_neighbour_tour(M) for M in (problem.D, problem.T)]
    st.ref_cost = np.array([problem.D[t, np.roll(t, -1)].sum() for t in (nn[0], nn[1])[:1]] + [problem.T[nn[1], np.roll(nn[1], -1)].sum()])
    st.tau = [np.ones((n, n)), np.ones((n, n))]
    st.arch = Archive(100)
    return st


def evaluate(st, X):
    return st.evaluate(X)


def build_tour(st, lam):
    n = st.problem.n_var; rng = st.rng
    tau = (st.tau[0] ** lam) * (st.tau[1] ** (1 - lam))
    eta = (st.eta[0] ** lam) * (st.eta[1] ** (1 - lam))
    w_all = (tau ** st.alpha) * (eta ** st.beta)
    cur = int(rng.integers(n)); tour = [cur]; left = np.ones(n, bool); left[cur] = False
    for _ in range(n - 1):
        w = w_all[cur] * left
        nxt = int(rng.choice(n, p=w / w.sum()))
        tour.append(nxt); left[nxt] = False; cur = nxt
    return np.array(tour)


def step(st):
    lam = st.rng.random(st.m)
    tours = np.array([build_tour(st, l) for l in lam])
    F, CV, S = st.evaluate(tours)
    pool_X = tours if st.arch.X is None else np.vstack([tours, st.arch.X])
    pool_F = F if st.arch.X is None else np.vstack([F, st.arch.F])
    nd = mu.non_dominated_mask(pool_F)[: st.m]
    st.tau = update_pheromone(st.tau, tours, F, nd, st.rho, st.Q, st.ref_cost)
    st.arch.add(tours, F, CV)


run = make_run(initialize, step)
