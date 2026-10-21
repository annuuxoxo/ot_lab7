"""
Algorithm 36 -- MGDA, Multiple-Gradient Descent Algorithm (Desideri 2012; gradient based, 5E slides 35-37)

Lecture steps
  1. At x compute the gradient g_i = grad f_i(x) of every objective.
  2. Find the minimum-norm point of the convex hull of {g_1..g_k}:  for k = 2 closed form
        alpha* = clip( (g2 - g1).g2 / |g1 - g2|^2 , 0, 1 ),  d* = alpha* g1 + (1 - alpha*) g2
     (k > 2: Frank-Wolfe on the simplex, our own code).
  3. If |d*| ~ 0 then x is Pareto-stationary: stop.
  4. Otherwise  x <- x - eta d*  -- every objective decreases (weakly) at the same time.
`run()` solves the toy P13a from several starting points (the end points lie on the Pareto set, the segment [a,b]).
`min_norm_point` is reused by the multi-task network training in mtl_net / experiments.
Defaults: eta = 0.1, tol = 1e-6, max 200 steps per start, 11 starts.
"""
import numpy as np
from algorithms.base import Result

DEFAULTS = dict(eta=0.1, tol=1e-6, steps=200, starts=11)


def min_norm_point(G, iters=200):
    """Minimum-norm point in conv{rows of G}.  Returns (weights alpha, d)."""
    G = np.asarray(G, float)
    k = len(G)
    if k == 2:
        g1, g2 = G
        den = float((g1 - g2) @ (g1 - g2))
        a = 0.5 if den < 1e-20 else float(np.clip((g2 - g1) @ g2 / den, 0.0, 1.0))
        return np.array([a, 1 - a]), a * g1 + (1 - a) * g2
    a = np.ones(k) / k                                  # Frank-Wolfe with exact line search
    M = G @ G.T
    for _ in range(iters):
        grad = M @ a
        j = int(np.argmin(grad))
        e = np.zeros(k); e[j] = 1
        d = e - a
        den = float(d @ M @ d)
        if den < 1e-20:
            break
        step = float(np.clip(-(a @ M @ d) / den, 0.0, 1.0))
        if step < 1e-12:
            break
        a = a + step * d
    return a, a @ G


def initialize(problem, seed=0, max_evals=None, eta=0.1, tol=1e-6, steps=200, starts=11):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, starts, endpoint=False) + rng.uniform(0, 0.3)
    X0 = np.stack([5 * np.cos(ang), 5 * np.sin(ang)], 1) + np.array([0.0, 2.0])
    X0 = np.vstack([[0.0, 5.0], X0])                       # the lab's start (0,5) first
    return dict(problem=problem, X=X0, eta=eta, tol=tol, steps=steps, evals=0, traj=[X0.copy()], d_norm=np.full(len(X0), np.inf),
                active=np.ones(len(X0), bool), alpha_hist=[], seed=seed)


def evaluate(st, X):
    st["evals"] += len(X)
    return st["problem"].evaluate(X)


def step(st):
    p = st["problem"]
    for i in np.where(st["active"])[0]:
        G = p.grad(st["X"][i])
        a, d = min_norm_point(G)
        st["d_norm"][i] = np.linalg.norm(d)
        if st["d_norm"][i] < st["tol"]:
            st["active"][i] = False                         # Pareto stationary
            continue
        st["X"][i] = st["X"][i] - st["eta"] * d
    st["traj"].append(st["X"].copy())
    evaluate(st, st["X"])


def run(problem, seed=0, max_evals=None, **params):
    st = initialize(problem, seed, max_evals, **params)
    for _ in range(st["steps"]):
        step(st)
        if not st["active"].any():
            break
    F, _ = problem.evaluate(st["X"])
    return Result(st["X"], F, np.zeros(len(F)), st["evals"], 0.0, [], dict(traj=np.array(st["traj"]), d_norm=st["d_norm"]))


def train_network(**kw):
    """Part (b) of P13: train the hand-coded two-head network with this method (see mtl_net.train)."""
    import mtl_net
    return mtl_net.train("mgda", **kw)
