"""Shared pieces for the population based algorithms: population state, sharing function, sorting helpers."""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import Core, Archive


class PopState(Core):
    """Generic population state: X, F, CV, S (selection objectives).  `output()` returns the whole population,
    i.e. the behaviour of a NON-elitist algorithm; elitist algorithms override it with their archive."""

    def __init__(self, problem, seed, max_evals, N=100, constraint="penalty", rho=1e3, **kw):
        super().__init__(problem, seed, max_evals, constraint, rho)
        self.N = N
        self.params = kw

    def init_population(self):
        self.X = self.problem.random_x(self.N, self.rng)
        self.F, self.CV, self.S = self.evaluate(self.X)

    def output(self):
        return self.X, self.F, self.CV


def sharing_matrix(Fn, sigma):
    """sh(d_ij) = 1 - (d_ij / sigma)^2 for d_ij < sigma else 0 on normalised objective vectors (Part 2 slide 6)."""
    d = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(2))
    sh = np.where(d < sigma, 1.0 - (d / sigma) ** 2, 0.0)
    return sh, d


def roulette(fitness, n, rng):
    f = np.asarray(fitness, float)
    f = f - min(0.0, f.min())
    s = f.sum()
    p = np.full(len(f), 1.0 / len(f)) if s <= 0 else f / s
    return rng.choice(len(f), size=n, p=p)


def mate_and_vary(st, parent_idx, **kw):
    """Offspring decision vectors from an index array of parents (consecutive pairs mate)."""
    idx = ops.pair_up(parent_idx)
    return ops.variation(st.problem, st.X[idx], st.rng, **kw)


def norm_for_distance(st, F):
    return st.problem.norm(F)
