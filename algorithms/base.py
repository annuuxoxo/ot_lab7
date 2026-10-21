"""
algorithms/base.py -- bookkeeping shared by every algorithm module.

Each algorithm module exposes
    DEFAULTS                      default parameters
    initialize(problem, seed, max_evals, **params) -> state
    evaluate(state, X)            counts function evaluations, returns (F, CV, S)
    step(state)                   ONE iteration (generation / sweep point / ant cycle ...)
    run(problem, seed, max_evals, **params) -> Result
Everything is minimisation.  S is the 'selection' version of F: with constraint='penalty' (default for the
algorithms that have no feasibility rule of their own) S = F + rho*CV*(nadir-ideal), so an infeasible point is
pushed away; constraint='ignore' gives S = F (used to show what happens without any constraint handling);
Constrained NSGA-II (21) and the algorithms that say so use Deb's rules on (F, CV) directly.
"""
import time
import numpy as np
import moo_utils as mu


class Result:
    def __init__(self, X, F, CV, evals, seconds, history, extra=None):
        self.X, self.F, self.CV = X, F, CV
        self.evals, self.seconds, self.history = evals, seconds, history
        self.extra = extra or {}


def nondominated_feasible(X, F, CV, feasible_only=True):
    X = np.asarray(X); F = np.asarray(F, float); CV = np.asarray(CV, float)
    if len(F) == 0:
        return X, F, CV
    if feasible_only:
        ok = CV <= 0
        X, F, CV = X[ok], F[ok], CV[ok]
        if len(F) == 0:
            return X, F, CV
    m = mu.non_dominated_mask(F)
    # drop exact duplicates in objective space
    X, F, CV = X[m], F[m], CV[m]
    _, u = np.unique(np.round(F, 12), axis=0, return_index=True)
    u = np.sort(u)
    return X[u], F[u], CV[u]


class Core:
    """State container with evaluation counting, penalty handling and convergence history."""

    def __init__(self, problem, seed=0, max_evals=10000, constraint="penalty", rho=1e3, record_every=1):
        self.problem = problem
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.max_evals = max_evals
        self.constraint = constraint
        self.rho = rho
        self.evals = 0
        self.history = []                     # (evals, HV of current output set)
        self.record_every = record_every
        self.gen = 0
        self._t0 = time.time()
        self.extra = {}

    # -- evaluation ---------------------------------------------------------------------
    def evaluate(self, X):
        X = np.atleast_2d(X)
        F, CV = self.problem.evaluate(X)
        self.evals += len(X)
        return F, CV, self.sel(F, CV)

    def sel(self, F, CV):
        if self.constraint == "penalty":
            return F + self.rho * CV[:, None] * self.problem.scale
        return F

    # -- output set + history -------------------------------------------------------------
    def output(self):
        """(X, F, CV) of the set this algorithm returns -- override in the algorithm's state class."""
        raise NotImplementedError

    def record(self):
        self.gen += 1
        if self.record_every and self.problem.n_obj == 2 and (self.gen % self.record_every == 0):
            X, F, CV = self.output()
            Xn, Fn, _ = nondominated_feasible(X, F, CV)
            self.history.append((self.evals, self.problem.hv(Fn) if len(Fn) else 0.0))

    def result(self):
        X, F, CV = self.output()
        feas = self.constraint != "ignore"
        n_in = len(F)
        n_infeasible = int((np.asarray(CV) > 0).sum())
        Xn, Fn, CVn = nondominated_feasible(X, F, CV, feasible_only=feas)
        extra = dict(self.extra)
        extra.update(n_returned=n_in, n_infeasible_returned=n_infeasible)
        return Result(Xn, Fn, CVn, self.evals, time.time() - self._t0, self.history, extra)


def make_run(initialize, step, default_max_evals=10000):
    def run(problem, seed=0, max_evals=None, **params):
        max_evals = default_max_evals if max_evals is None else max_evals
        st = initialize(problem, seed=seed, max_evals=max_evals, **params)
        while st.evals < st.max_evals and not getattr(st, "done", False):
            step(st)
            st.record()
        return st.result()
    return run


class Archive:
    """External archive of feasible non-dominated solutions, truncated by crowding distance."""

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.X = None; self.F = np.zeros((0, 0)); self.CV = np.zeros(0)

    def __len__(self):
        return 0 if self.X is None else len(self.X)

    def add(self, X, F, CV):
        X = np.asarray(X); F = np.asarray(F, float); CV = np.asarray(CV, float)
        ok = CV <= 0
        if not ok.any():
            return
        X, F, CV = X[ok], F[ok], CV[ok]
        if self.X is not None and len(self.X):
            X = np.concatenate([self.X, X]); F = np.concatenate([self.F, F]); CV = np.concatenate([self.CV, CV])
        m = mu.non_dominated_mask(F)
        X, F, CV = X[m], F[m], CV[m]
        _, u = np.unique(np.round(F, 12), axis=0, return_index=True)       # remove duplicates
        u = np.sort(u); X, F, CV = X[u], F[u], CV[u]
        while len(X) > self.max_size:
            cd = mu.crowding_distance(F)
            worst = int(np.argmin(cd))
            keep = np.ones(len(X), bool); keep[worst] = False
            X, F, CV = X[keep], F[keep], CV[keep]
        self.X, self.F, self.CV = X, F, CV

    def crowding(self):
        return mu.crowding_distance(self.F) if len(self) else np.zeros(0)

    def leader(self, rng, k=1):
        """Pick archive leaders: binary tournament on crowding distance (larger = less crowded wins)."""
        n = len(self)
        cd = self.crowding()
        a = rng.integers(n, size=k); b = rng.integers(n, size=k)
        return np.where(cd[a] >= cd[b], a, b)


def normalise_pop(problem, F):
    return problem.norm(F)
