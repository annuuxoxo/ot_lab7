"""
mtl_net.py -- P13(b): a small shared-body network with a classification head and a regression head, forward and backward passes
written by hand in NumPy (no autograd).

    x --> [W1,b1] tanh --> [W2,b2] tanh --> h  (shared body)
    h --> sigmoid(Wc.h + bc)            : task 1, binary cross-entropy
    h --> Wr.h + br                      : task 2, half mean squared error
`grads()` returns the loss of each task and the gradient of EACH task loss with respect to the shared parameters and to its
own head, which is what MGDA and GradNorm need.
"""
import numpy as np


def make_data(n=256, d=8, reg_scale=10.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    yc = ((X[:, 0] + X[:, 1] * X[:, 2]) > 0).astype(float)                 # classification target
    tr = reg_scale * (np.sin(X[:, 3]) + 0.5 * X[:, 4] - 0.3 * X[:, 5] * X[:, 6]) + 0.1 * rng.normal(size=n)   # regression target
    return X, yc, tr


class MTLNet:
    def __init__(self, d_in=8, h=16, seed=0):
        r = np.random.default_rng(seed)
        self.sh = dict(W1=r.normal(0, 1 / np.sqrt(d_in), (d_in, h)), b1=np.zeros(h),
                       W2=r.normal(0, 1 / np.sqrt(h), (h, h)), b2=np.zeros(h))
        self.hc = dict(Wc=r.normal(0, 1 / np.sqrt(h), h), bc=np.zeros(1))
        self.hr = dict(Wr=r.normal(0, 1 / np.sqrt(h), h), br=np.zeros(1))

    # -- flatten helpers --------------------------------------------------------------
    SH_KEYS = ("W1", "b1", "W2", "b2")

    def flat(self, d, keys):
        return np.concatenate([d[k].ravel() for k in keys])

    def apply(self, d, keys, vec, lr):
        i = 0
        for k in keys:
            n = d[k].size
            d[k] -= lr * vec[i:i + n].reshape(d[k].shape); i += n

    def forward(self, X):
        A1 = np.tanh(X @ self.sh["W1"] + self.sh["b1"])
        H = np.tanh(A1 @ self.sh["W2"] + self.sh["b2"])
        zc = H @ self.hc["Wc"] + self.hc["bc"][0]
        yr = H @ self.hr["Wr"] + self.hr["br"][0]
        return A1, H, zc, yr

    def losses(self, X, yc, tr):
        _, _, zc, yr = self.forward(X)
        p = 1 / (1 + np.exp(-zc))
        eps = 1e-12
        Lc = float(-np.mean(yc * np.log(p + eps) + (1 - yc) * np.log(1 - p + eps)))
        Lr = float(0.5 * np.mean((yr - tr) ** 2))
        return Lc, Lr

    def _body_backward(self, X, A1, H, dH):
        dZ2 = dH * (1 - H ** 2)
        gW2 = A1.T @ dZ2; gb2 = dZ2.sum(0)
        dA1 = dZ2 @ self.sh["W2"].T
        dZ1 = dA1 * (1 - A1 ** 2)
        gW1 = X.T @ dZ1; gb1 = dZ1.sum(0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

    def grads(self, X, yc, tr):
        n = len(X)
        A1, H, zc, yr = self.forward(X)
        p = 1 / (1 + np.exp(-zc))
        Lc, Lr = self.losses(X, yc, tr)
        dzc = (p - yc) / n
        dyr = (yr - tr) / n
        out = dict(Lc=Lc, Lr=Lr)
        out["g_sh_c"] = self._body_backward(X, A1, H, np.outer(dzc, self.hc["Wc"]))
        out["g_sh_r"] = self._body_backward(X, A1, H, np.outer(dyr, self.hr["Wr"]))
        out["g_hc"] = np.concatenate([H.T @ dzc, [dzc.sum()]])           # (Wc, bc)
        out["g_hr"] = np.concatenate([H.T @ dyr, [dyr.sum()]])           # (Wr, br)
        return out

    def set_head(self, which, vec, lr):
        d = self.hc if which == "c" else self.hr
        k = ("Wc", "bc") if which == "c" else ("Wr", "br")
        self.apply(d, k, vec, lr)

    def n_shared(self):
        return sum(self.sh[k].size for k in self.SH_KEYS)


def train(method="fixed", steps=500, lr=0.05, reg_scale=10.0, seed=0, alpha=0.5, lr_w=0.001, n=256, w0=(1.0, 1.0), hidden=16):
    """Full-batch gradient descent on the two-head network.
    method = 'fixed'    : total loss w1 L1 + w2 L2 with constant weights w0   (default: equal weights)
             'mgda'     : shared-body direction = min-norm point of {grad L1, grad L2} (alg36), heads keep their own gradients
             'gradnorm' : adaptive weights (alg37), shared direction = sum_i w_i grad L_i, head i scaled by w_i
    Returns a dict with the loss and weight histories."""
    from algorithms.alg36_mgda import min_norm_point
    from algorithms.alg37_gradnorm import gradnorm_update
    X, yc, tr = make_data(n, reg_scale=reg_scale, seed=seed)
    net = MTLNet(seed=seed, h=hidden)
    w = np.array(w0, float)
    L0 = None
    hist = dict(Lc=[], Lr=[], w=[], alpha=[], gnorm_c=[], gnorm_r=[])
    for t in range(steps):
        g = net.grads(X, yc, tr)
        if L0 is None:
            L0 = np.array([g["Lc"], g["Lr"]])
        hist["Lc"].append(g["Lc"]); hist["Lr"].append(g["Lr"]); hist["w"].append(w.copy())
        gc, gr = g["g_sh_c"], g["g_sh_r"]
        hist["gnorm_c"].append(float(np.linalg.norm(gc))); hist["gnorm_r"].append(float(np.linalg.norm(gr)))
        if method == "fixed":
            d_sh = w[0] * gc + w[1] * gr; sc_c, sc_r = w[0], w[1]
        elif method == "mgda":
            a, d_sh = min_norm_point(np.stack([gc, gr])); hist["alpha"].append(float(a[0])); sc_c = sc_r = 1.0
        elif method == "gradnorm":
            gn = np.array([np.linalg.norm(gc), np.linalg.norm(gr)])
            w, _, _ = gradnorm_update(w, gn, [g["Lc"], g["Lr"]], L0, alpha, lr_w)
            d_sh = w[0] * gc + w[1] * gr; sc_c, sc_r = w[0], w[1]
        else:
            raise ValueError(method)
        net.apply(net.sh, net.SH_KEYS, d_sh, lr)
        net.set_head("c", sc_c * g["g_hc"], lr); net.set_head("r", sc_r * g["g_hr"], lr)
    Lc, Lr = net.losses(X, yc, tr)
    hist["Lc"].append(Lc); hist["Lr"].append(Lr); hist["w"].append(w.copy())
    for k in ("Lc", "Lr", "w", "alpha", "gnorm_c", "gnorm_r"):
        hist[k] = np.array(hist[k])
    return hist
