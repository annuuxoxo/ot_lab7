"""Pieces shared by MOEA/D, MOEA/D-DE and MOEA/D-ACO: weight vectors, neighbourhoods, Tchebycheff scalarisation."""
import numpy as np
import moo_utils as mu
from algorithms.base import Core, Archive


def weight_vectors(M, N):
    """N uniformly spread weight vectors (exact for M=2; the smallest Das-Dennis set with >= N points otherwise)."""
    if M == 2:
        a = np.linspace(0, 1, N)
        return np.stack([a, 1 - a], 1)
    p = 1
    while len(mu.das_dennis(M, p)) < N:
        p += 1
    W = mu.das_dennis(M, p)
    return W


def neighbourhoods(W, T):
    d = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(2))
    return np.argsort(d, axis=1)[:, :T]


def tcheby(Fn, lam, z):
    """g^te(x | lambda, z*) = max_i lambda_i |f_i - z_i|  (lambda_i = 0 is replaced by 1e-6, Zhang & Li 2007)."""
    lam = np.where(lam < 1e-6, 1e-6, lam)
    return np.max(lam * np.abs(Fn - z), axis=-1)


class DecompState(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV
