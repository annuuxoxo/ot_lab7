"""
Algorithm 37 -- GradNorm, gradient normalisation for adaptive loss balancing (Chen, Badrinarayanan, Lee & Rabinovich 2018; 5E slides 38-40)

Lecture steps
  1. One trainable weight w_i per task loss (init 1); the shared parameters theta are updated with  sum_i w_i grad L_i.
  2. After each pass measure G_i = || grad_theta (w_i L_i) || = w_i ||g_i|| and the loss ratio L_i(t)/L_i(0).
  3. Average gradient norm  Gbar = mean_i G_i  and relative inverse training rate  r_i = [L_i(t)/L_i(0)] / mean_j [L_j(t)/L_j(0)].
  4. Target  target_i = Gbar r_i^alpha  (alpha = asymmetry, fixed input).
  5. L_grad = sum_i |G_i - target_i| ; gradient step ONLY on the weights w (dL_grad/dw_i = sign(G_i - target_i) ||g_i||), then
     renormalise  sum_i w_i = k.
Defaults: alpha = 0.5, lr_w = 0.025, lr (model) = 0.05.   `gradnorm_update` is also used by the network experiment.
"""
import numpy as np
from algorithms.base import Result

DEFAULTS = dict(alpha=0.5, lr_w=0.025, eta=0.05)


def gradnorm_update(w, gnorms, L, L0, alpha, lr_w):
    """One GradNorm weight update.  Returns (new weights, L_grad, targets)."""
    w = np.asarray(w, float); k = len(w)
    G = w * gnorms
    Gbar = G.mean()
    ratio = np.asarray(L, float) / np.asarray(L0, float)
    r = ratio / ratio.mean()
    target = Gbar * r ** alpha
    Lgrad = float(np.abs(G - target).sum())
    dw = np.sign(G - target) * gnorms
    w_new = np.maximum(w - lr_w * dw, 1e-3)
    w_new = w_new * k / w_new.sum()
    return w_new, Lgrad, target


def initialize(problem, seed=0, max_evals=None, alpha=0.5, lr_w=0.025, eta=0.05, steps=300, starts=11):
    from algorithms.alg36_mgda import initialize as mi
    base = mi(problem, seed, max_evals, starts=starts)
    base.update(alpha=alpha, lr_w=lr_w, eta=eta, steps=steps, w=np.ones((len(base["X"]), 2)))
    F0, _ = problem.evaluate(base["X"]); base["L0"] = F0.copy()
    return base


def evaluate(st, X):
    st["evals"] += len(X)
    return st["problem"].evaluate(X)


def step(st):
    p = st["problem"]
    F, _ = p.evaluate(st["X"])
    for i in range(len(st["X"])):
        G = p.grad(st["X"][i])
        gn = np.linalg.norm(G, axis=1)
        w, Lg, _ = gradnorm_update(st["w"][i], gn, F[i], st["L0"][i], st["alpha"], st["lr_w"])
        st["w"][i] = w
        st["X"][i] = st["X"][i] - st["eta"] * (w @ G)
    st["traj"].append(st["X"].copy())
    st["evals"] += len(st["X"])


def run(problem, seed=0, max_evals=None, **params):
    st = initialize(problem, seed, max_evals, **params)
    for _ in range(st["steps"]):
        step(st)
    F, _ = problem.evaluate(st["X"])
    return Result(st["X"], F, np.zeros(len(F)), st["evals"], 0.0, [], dict(traj=np.array(st["traj"]), w=st["w"]))


def train_network(**kw):
    """Part (b) of P13: train the hand-coded two-head network with this method (see mtl_net.train)."""
    import mtl_net
    return mtl_net.train("gradnorm", **kw)
