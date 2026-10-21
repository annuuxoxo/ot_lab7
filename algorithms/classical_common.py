"""Helpers for the classical (scalarising) methods 1-7: evaluation counting and the sub-problem builder."""
import numpy as np
import solvers as sv
from algorithms.base import Result
import time


class Counter:
    """Wraps a problem; counts evaluations and memoises the last point (objective + constraints share one call)."""

    def __init__(self, problem):
        self.p = problem
        self.n = 0
        self._key = None
        self._val = None

    def __call__(self, x):
        x = np.asarray(x, float)
        key = x.tobytes()
        if key != self._key:
            F, CV = self.p.evaluate(x[None, :])
            self.n += 1
            self._key, self._val = key, (F[0], float(CV[0]))
        return self._val

    def F(self, x):
        return self(x)[0]

    def cv(self, x):
        return self(x)[1]


def solve(counter, scalar_of_F, extra_cons=(), seed=0, x0=None, maxfev=3000, n_starts=6):
    """min scalar_of_F(F(x)) s.t. CV(x) <= 0 and every g(F(x)) <= 0 in extra_cons.  Returns x, F(x), violation."""
    p = counter.p
    fun = lambda x: float(scalar_of_F(counter.F(x)))
    cons = [lambda x: counter.cv(x)] + [(lambda x, g=g: g(counter.F(x))) for g in extra_cons]
    if p.kind == "int" and p.n_var == 1:
        x, f, v = sv.minimize_integer_1d(fun, p.lb[0], p.ub[0], cons)
    else:
        x, f, v = sv.minimize(fun, p.lb, p.ub, cons, x0=x0, seed=seed, maxfev=maxfev, n_starts=n_starts)
    return x, (counter.F(x) if x is not None else None), v


def ideal_point(counter, seed=0):
    """k single-objective solves -> ideal point z* and the payoff table (needed by weight metric, eps-constraint)."""
    p = counter.p
    M = p.n_obj
    pay = np.zeros((M, M)); xs = []
    for j in range(M):
        x, F, v = solve(counter, lambda f, j=j: f[j], seed=seed + j)
        pay[j] = F; xs.append(x)
    return pay.diagonal().copy(), pay, xs


def finish(counter, X, F, t0, extra):
    X = np.array(X); F = np.array(F)
    from algorithms.base import nondominated_feasible
    CV = np.zeros(len(F))
    Xn, Fn, _ = nondominated_feasible(X, F, CV)
    extra = dict(extra)
    extra["all_X"] = X; extra["all_F"] = F
    return Result(Xn, Fn, np.zeros(len(Fn)), counter.n, time.time() - t0, [], extra)


class SweepState:
    """Minimal state for sweep-type methods: a list of settings, solved one per step()."""

    def __init__(self, problem, seed, settings, max_evals=np.inf):
        self.problem = problem; self.seed = seed; self.settings = list(settings); self.i = 0
        self.counter = Counter(problem); self.X = []; self.F = []; self.info = []
        self.t0 = time.time(); self.max_evals = max_evals
        self.history = []; self.extra = {}

    @property
    def evals(self):
        return self.counter.n

    @property
    def done(self):
        return self.i >= len(self.settings)

    def record(self):
        pass

    def result(self):
        ex = dict(self.extra); ex["settings"] = self.settings; ex["info"] = self.info
        return finish(self.counter, self.X, self.F, self.t0, ex)
