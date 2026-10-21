"""
problems.py -- the 14-problem bank of Lab 7 (Section 4), all written from scratch.

Every problem is a `Problem` object.  `evaluate(X)` takes an (N, n) array and returns
    F  (N, M)  objective values in MINIMISATION form (a maximised objective is stored as -f)
    CV (N,)    total constraint violation  (0 = feasible)
`natural(F)` converts back to the natural direction of the scenario for reporting.

Encodings (kind):  'real' continuous vector, 'int' integer vector, 'perm' permutation, 'bin' 0/1 mask.
Normalisation / hypervolume use the per-problem `ideal` and `nadir`; the hypervolume reference point is
1.1 in the normalised space for every problem (`HV_REF`).
"""
import os
import numpy as np
import moo_utils as mu

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "results", "ref")
HV_REF = 1.1


class Problem:
    name = "problem"
    n_var = 1
    n_obj = 2
    kind = "real"
    lb = None
    ub = None
    maximize = None            # tuple of bool, natural direction of every objective
    obj_names = None
    analytic_bounds = None     # (ideal, nadir) when known in closed form

    def __init__(self):
        self._ref = None
        self._bounds = None

    # -- to be provided by sub-classes ------------------------------------------------
    def evaluate(self, X):
        raise NotImplementedError

    def true_front(self, n=500):
        """Dense sample of the true Pareto front in minimisation form, or None if unknown."""
        return None

    # -- helpers ------------------------------------------------------------------------
    def natural(self, F):
        F = np.asarray(F, float).copy()
        if self.maximize is not None:
            for j, mx in enumerate(self.maximize):
                if mx:
                    F[..., j] = -F[..., j]
        return F

    def _ref_path(self):
        return os.path.join(REF_DIR, self.name + ".npy")

    def reference_front(self, n=400):
        """Best-known / analytic front used for IGD (minimisation form, unnormalised)."""
        if self._ref is None:
            tf = self.true_front(2000)
            if tf is not None:
                self._ref = tf
            elif os.path.exists(self._ref_path()):
                self._ref = np.load(self._ref_path())
            else:
                self._ref = None
        return self._ref

    @property
    def has_true_front(self):
        return self.true_front(10) is not None or os.path.exists(self._ref_path())

    @property
    def bounds(self):
        if self._bounds is None:
            ref = self.reference_front()
            if ref is not None and len(ref) > 1:
                self._bounds = (ref.min(0), ref.max(0))
            elif self.analytic_bounds is not None:
                self._bounds = tuple(np.asarray(b, float) for b in self.analytic_bounds)
            else:
                raise RuntimeError("no normalisation bounds for " + self.name)
        return self._bounds

    @property
    def ideal(self):
        return self.bounds[0]

    @property
    def nadir(self):
        return self.bounds[1]

    @property
    def scale(self):
        return np.where(self.nadir - self.ideal > 1e-12, self.nadir - self.ideal, 1.0)

    def norm(self, F):
        return mu.normalize(F, self.ideal, self.nadir)

    @property
    def hv_ref(self):
        return np.full(self.n_obj, HV_REF)

    def hv(self, F):
        """Hypervolume of the NORMALISED front (larger is better)."""
        if len(F) == 0:
            return 0.0
        return mu.hypervolume(self.norm(F), self.hv_ref)

    def igd(self, F):
        ref = self.reference_front()
        if ref is None or len(F) == 0:
            return float("nan")
        return mu.igd(self.norm(F), self.norm(ref))

    def spacing(self, F):
        return mu.spacing(self.norm(F)) if len(F) else float("nan")

    def random_x(self, N, rng):
        if self.kind == "perm":
            return np.array([rng.permutation(self.n_var) for _ in range(N)])
        if self.kind == "bin":
            X = (rng.random((N, self.n_var)) < 0.5).astype(int)
            X[X.sum(1) == 0, rng.integers(self.n_var)] = 1
            return X
        X = self.lb + rng.random((N, self.n_var)) * (self.ub - self.lb)
        return np.rint(X) if self.kind == "int" else X


def thin_front(F, n):
    """Pick n points spread evenly along the arc length of a 2-objective front."""
    F = np.asarray(F, float)
    if len(F) <= n:
        return F
    F = F[np.argsort(F[:, 0])]
    sc = np.where(F.max(0) - F.min(0) > 0, F.max(0) - F.min(0), 1)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(F / sc, axis=0), axis=1))]
    targets = np.linspace(0, seg[-1], n)
    idx = np.unique(np.searchsorted(seg, targets).clip(0, len(F) - 1))
    return F[idx]


# =========================================================================================
class P1(Problem):
    """Factory production speed: min (x-3)^2 and (x-7)^2, 1<=x<=10.  Pareto set 3<=x<=7."""
    name = "P1"
    n_var, n_obj = 1, 2
    lb, ub = np.array([1.0]), np.array([10.0])
    maximize = (False, False)
    obj_names = ("cost (x-3)^2", "quality loss (x-7)^2")
    analytic_bounds = ((0, 0), (16, 16))

    def evaluate(self, X):
        x = np.atleast_2d(X)[:, 0]
        return np.stack([(x - 3) ** 2, (x - 7) ** 2], 1), np.zeros(len(x))

    def true_front(self, n=500):
        x = np.linspace(3, 7, n)
        return np.stack([(x - 3) ** 2, (x - 7) ** 2], 1)


class P2(Problem):
    """Delivery fleet sizing: min 100/x and x^2, 1<=x<=10 (real or integer).  Every x is Pareto optimal."""
    n_var, n_obj = 1, 2
    lb, ub = np.array([1.0]), np.array([10.0])
    maximize = (False, False)
    obj_names = ("delivery cost 100/x", "delay x^2")
    analytic_bounds = ((10, 1), (100, 100))

    def __init__(self, integer=False):
        super().__init__()
        self.kind = "int" if integer else "real"
        self.name = "P2int" if integer else "P2"

    def evaluate(self, X):
        x = np.atleast_2d(X)[:, 0]
        return np.stack([100.0 / x, x ** 2], 1), np.zeros(len(x))

    def true_front(self, n=500):
        x = np.arange(1, 11, dtype=float) if self.kind == "int" else np.linspace(1, 10, n)
        return np.stack([100.0 / x, x ** 2], 1)


class P3(Problem):
    """ZDT2 with n = 10: f1 = x1, g = 1 + sum_{i>=2} x_i, f2 = g (1 - (f1/g)^2).  Non-convex front."""
    name = "P3"
    n_var, n_obj = 10, 2
    lb, ub = np.zeros(10), np.ones(10)
    maximize = (False, False)
    obj_names = ("f1", "f2")
    analytic_bounds = ((0, 0), (1, 1))

    def evaluate(self, X):
        X = np.atleast_2d(X)
        f1 = X[:, 0]
        g = 1.0 + X[:, 1:].sum(1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.stack([f1, f2], 1), np.zeros(len(X))

    def true_front(self, n=500):
        f1 = np.linspace(0, 1, n)
        return np.stack([f1, 1 - f1 ** 2], 1)


class P4(Problem):
    """Four-asset portfolio: min risk w'Sw, max return mu'w, w>=0, sum w = 1.
    handling='repair' : w = x / sum(x) before evaluation (equality satisfied by construction)
    handling='penalty': w = x, CV = max(0, |sum x - 1| - 1e-3)."""
    n_var, n_obj = 4, 2
    lb, ub = np.zeros(4), np.ones(4)
    maximize = (False, True)
    obj_names = ("risk w'Sw", "return mu'w")
    MU = np.array([0.08, 0.12, 0.15, 0.10])
    SIG = np.array([[0.010, 0.002, 0.001, 0.003],
                    [0.002, 0.040, 0.006, 0.004],
                    [0.001, 0.006, 0.090, 0.005],
                    [0.003, 0.004, 0.005, 0.020]])

    def __init__(self, handling="repair"):
        super().__init__()
        self.handling = handling
        self.name = "P4" if handling == "repair" else "P4pen"

    def weights(self, X):
        X = np.atleast_2d(X)
        if self.handling == "repair":
            s = X.sum(1, keepdims=True)
            W = np.where(s > 1e-12, X / np.maximum(s, 1e-12), 0.25)
            return W
        return X

    def evaluate(self, X):
        X = np.atleast_2d(X)
        W = self.weights(X)
        risk = np.einsum("ni,ij,nj->n", W, self.SIG, W)
        ret = W @ self.MU
        cv = np.zeros(len(X)) if self.handling == "repair" else np.maximum(0, np.abs(X.sum(1) - 1) - 1e-3)
        return np.stack([risk, -ret], 1), cv

    def true_front(self, n=500):
        return None

    def reference_front(self, n=400):
        if self._ref is None:
            p = self._ref_path().replace("P4pen", "P4")
            if os.path.exists(p):
                self._ref = np.load(p)
        return self._ref

    @property
    def has_true_front(self):
        return os.path.exists(self._ref_path().replace("P4pen", "P4"))


class P5(Problem):
    """Cantilever beam: min mass, min tip deflection, stress 6PL/(b h^2) <= 250 MPa  <=>  b h^2 >= 2.4e-5."""
    name = "P5"
    n_var, n_obj = 2, 2
    lb, ub = np.array([0.01, 0.02]), np.array([0.10, 0.20])
    maximize = (False, False)
    obj_names = ("mass [kg]", "deflection [m]")
    L, P, E, RHO = 1.0, 1000.0, 200e9, 7850.0
    BH2_MIN = 2.4e-5

    def evaluate(self, X):
        X = np.atleast_2d(X)
        b, h = X[:, 0], X[:, 1]
        mass = self.RHO * self.L * b * h
        defl = 4 * self.P * self.L ** 3 / (self.E * b * h ** 3)
        cv = np.maximum(0.0, (self.BH2_MIN - b * h ** 2) / self.BH2_MIN)      # relative violation
        return np.stack([mass, defl], 1), cv

    def stress(self, X):
        X = np.atleast_2d(X)
        return 6 * self.P * self.L / (X[:, 0] * X[:, 1] ** 2)


class P6(Problem):
    """Binh-Korn: min 4x1^2+4x2^2 and (x1-5)^2+(x2-5)^2 with two nonlinear constraints."""
    name = "P6"
    n_var, n_obj = 2, 2
    lb, ub = np.array([0.0, 0.0]), np.array([5.0, 3.0])
    maximize = (False, False)
    obj_names = ("f1", "f2")

    def evaluate(self, X):
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]
        f1 = 4 * x1 ** 2 + 4 * x2 ** 2
        f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
        g1 = (x1 - 5) ** 2 + x2 ** 2 - 25                 # <= 0
        g2 = 7.7 - ((x1 - 8) ** 2 + (x2 + 3) ** 2)        # <= 0
        cv = np.maximum(0, g1) + np.maximum(0, g2)
        return np.stack([f1, f2], 1), cv


class P7(Problem):
    """Water allocation: max 0.8 x1 and 1.2 x2, x1+x2<=90, x1>=10, x2>=20 (stored as minimisation)."""
    name = "P7"
    n_var, n_obj = 2, 2
    lb, ub = np.array([10.0, 20.0]), np.array([90.0, 90.0])
    maximize = (True, True)
    obj_names = ("agricultural yield", "industrial output")
    analytic_bounds = ((-56, -96), (-8, -24))
    # linear data  (used by the LP based goal / lexicographic methods)
    C = np.array([[0.8, 0.0], [0.0, 1.2]])               # objective j = C[j] @ x   (to be maximised)
    A_ub = np.array([[1.0, 1.0]]); B_ub = np.array([90.0])

    def evaluate(self, X):
        X = np.atleast_2d(X)
        f = np.stack([0.8 * X[:, 0], 1.2 * X[:, 1]], 1)
        cv = np.maximum(0, X.sum(1) - 90.0)
        return -f, cv

    def true_front(self, n=500):
        x1 = np.linspace(10, 70, n)
        return -np.stack([0.8 * x1, 1.2 * (90 - x1)], 1)


class P8(Problem):
    """Bi-objective TSP, 15 cities (seed 7): total distance and total travel time."""
    name = "P8"
    n_var, n_obj = 15, 2
    kind = "perm"
    maximize = (False, False)
    obj_names = ("total distance", "total time")

    def __init__(self):
        super().__init__()
        np.random.seed(7)
        self.xy = np.random.rand(15, 2) * 100
        U = np.triu(np.random.uniform(0.5, 2.0, (15, 15)), 1)
        U = U + U.T
        d = np.linalg.norm(self.xy[:, None] - self.xy[None, :], axis=2)
        self.D = d
        self.T = d * np.where(U > 0, U, 1.0)
        self.lb = np.zeros(15); self.ub = np.full(15, 14.0)
        self._provisional_bounds()

    @staticmethod
    def _two_opt(M, tour):
        n = len(tour); best = M[tour, np.roll(tour, -1)].sum(); improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    cand = tour.copy(); cand[i + 1:j + 1] = cand[i + 1:j + 1][::-1]
                    c = M[cand, np.roll(cand, -1)].sum()
                    if c < best - 1e-12:
                        tour, best, improved = cand, c, True
        return tour, best

    def _provisional_bounds(self):
        """Ideal / nadir from 2-opt local search on each single objective (used until the pooled reference front exists)."""
        rng = np.random.default_rng(0)
        bd = bt = None
        for _ in range(40):
            t, c = self._two_opt(self.D, rng.permutation(15))
            if bd is None or c < bd[1]: bd = (t, c)
            t, c = self._two_opt(self.T, rng.permutation(15))
            if bt is None or c < bt[1]: bt = (t, c)
        Fd, _ = self.evaluate(bd[0][None]); Ft, _ = self.evaluate(bt[0][None])
        self.analytic_bounds = ((Fd[0, 0], Ft[0, 1]), (Ft[0, 0], Fd[0, 1]))

    def evaluate(self, X):
        X = np.atleast_2d(X).astype(int)
        nxt = np.roll(X, -1, axis=1)
        return np.stack([self.D[X, nxt].sum(1), self.T[X, nxt].sum(1)], 1), np.zeros(len(X))

    def nearest_neighbour_tour(self, M, start=0):
        n = len(M); tour = [start]; left = set(range(n)) - {start}
        while left:
            c = tour[-1]; j = min(left, key=lambda k: M[c, k]); tour.append(j); left.remove(j)
        return np.array(tour)


class P9(Problem):
    """Feature selection (Wisconsin breast cancer): min 1-accuracy of own 5-NN / 5-fold CV and min #features."""
    name = "P9"
    n_var, n_obj = 30, 2
    kind = "bin"
    maximize = (False, False)
    obj_names = ("CV error", "# features")
    analytic_bounds = ((0.0, 1), (0.40, 30))

    def __init__(self):
        super().__init__()
        from sklearn.datasets import load_breast_cancer       # data loading only (allowed by the sheet)
        d = load_breast_cancer()
        X = d.data.astype(float)
        self.X = (X - X.mean(0)) / X.std(0)
        self.y = d.target.astype(int)
        self.feature_names = list(d.feature_names)
        n = len(self.y)
        perm = np.random.RandomState(0).permutation(n)
        self.fold = np.empty(n, int); self.fold[perm] = np.arange(n) % 5
        self.same_fold = self.fold[:, None] == self.fold[None, :]
        self._cache = {}
        self.lb = np.zeros(30); self.ub = np.ones(30)

    def cv_error(self, mask):
        key = mask.tobytes()
        if key in self._cache:
            return self._cache[key]
        Z = self.X * mask
        sq = (Z ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
        D[self.same_fold] = np.inf                    # an item is classified using the other four folds only
        nn = np.argpartition(D, 5, axis=1)[:, :5]
        pred = (self.y[nn].sum(1) >= 3).astype(int)   # majority vote of the 5 neighbours
        err = float((pred != self.y).mean())
        self._cache[key] = err
        return err

    def evaluate(self, X):
        X = (np.atleast_2d(X) > 0.5).astype(int)
        nf = X.sum(1)
        err = np.array([self.cv_error(m.astype(float)) if k > 0 else 1.0 for m, k in zip(X, nf)])
        return np.stack([err, nf.astype(float)], 1), (nf == 0).astype(float)

    def baseline_error(self):
        return self.cv_error(np.ones(30))


class P10(Problem):
    """DTLZ2 with M objectives and n = M + 9 variables.  True front: sum f_m^2 = 1."""
    kind = "real"
    maximize = None

    def __init__(self, M=3):
        super().__init__()
        self.n_obj = M
        self.n_var = M + 9
        self.lb, self.ub = np.zeros(self.n_var), np.ones(self.n_var)
        self.name = f"P10_M{M}"
        self.analytic_bounds = (np.zeros(M), np.ones(M))

    def evaluate(self, X):
        X = np.atleast_2d(X); M = self.n_obj
        g = ((X[:, M - 1:] - 0.5) ** 2).sum(1)
        F = np.empty((len(X), M))
        c = np.cos(X[:, :M - 1] * np.pi / 2); s = np.sin(X[:, :M - 1] * np.pi / 2)
        for m in range(1, M + 1):                      # objective index m (1-based) as in the sheet
            val = (1 + g) * np.prod(c[:, :M - m], axis=1)
            if m > 1:
                val = val * s[:, M - m]
            F[:, m - 1] = val
        return F, np.zeros(len(X))

    def true_front(self, n=500):
        p = {3: 20, 5: 8}.get(self.n_obj, 6)
        W = mu.das_dennis(self.n_obj, p)
        return W / np.linalg.norm(W, axis=1, keepdims=True)

    def frac_nondominated_random(self, N, rng):
        X = rng.random((N, self.n_var)); F, _ = self.evaluate(X)
        return mu.non_dominated_mask(F).mean()


class P11(Problem):
    """Production planning: max profit 4A+6B and units A+B, labour 2A+3B<=1200, material 4A+2B<=1600."""
    name = "P11"
    n_var, n_obj = 2, 2
    lb, ub = np.array([0.0, 0.0]), np.array([400.0, 400.0])
    maximize = (True, True)
    obj_names = ("profit", "units")
    analytic_bounds = ((-2400, -500), (-1600, -400))
    C = np.array([[4.0, 6.0], [1.0, 1.0]])                              # objectives (maximised)
    A_ub = np.array([[2.0, 3.0], [4.0, 2.0]]); B_ub = np.array([1200.0, 1600.0])

    def evaluate(self, X):
        X = np.atleast_2d(X)
        f = X @ self.C.T
        cv = np.maximum(0, (X @ self.A_ub.T - self.B_ub) / self.B_ub).sum(1)
        return -f, cv

    def true_front(self, n=500):
        return None


class P14(Problem):
    """Laptop design: min weight 0.4+0.05 x1+0.012 x2, max runtime 0.2 x2/(0.5+0.03 x1)."""
    name = "P14"
    n_var, n_obj = 2, 2
    lb, ub = np.array([13.0, 40.0]), np.array([17.0, 90.0])
    maximize = (False, True)
    obj_names = ("weight [kg]", "runtime [h]")
    analytic_bounds = ((1.53, -20.2247), (2.13, -8.9888))
    PREF = np.array([1.8, 12.0])                                       # DM reference point (natural units)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        w = 0.4 + 0.05 * X[:, 0] + 0.012 * X[:, 1]
        rt = 0.2 * X[:, 1] / (0.5 + 0.03 * X[:, 0])
        return np.stack([w, -rt], 1), np.zeros(len(X))

    def true_front(self, n=500):
        x2 = np.linspace(40, 90, n)
        return self.evaluate(np.stack([np.full(n, 13.0), x2], 1))[0]


class P13a(Problem):
    """Multi-task toy: f1 = |x-a|^2, f2 = |x-b|^2, a=(3,3), b=(-3,-3).  Pareto set = segment [a,b]."""
    name = "P13a"
    n_var, n_obj = 2, 2
    lb, ub = np.array([-6.0, -6.0]), np.array([6.0, 6.0])
    maximize = (False, False)
    obj_names = ("task 1 loss", "task 2 loss")
    a = np.array([3.0, 3.0]); b = np.array([-3.0, -3.0])
    analytic_bounds = ((0, 0), (72, 72))

    def evaluate(self, X):
        X = np.atleast_2d(X)
        return np.stack([((X - self.a) ** 2).sum(1), ((X - self.b) ** 2).sum(1)], 1), np.zeros(len(X))

    def grad(self, x):
        """Rows are the gradients of f1 and f2 at x."""
        x = np.asarray(x, float)
        return np.stack([2 * (x - self.a), 2 * (x - self.b)])

    def true_front(self, n=500):
        t = np.linspace(0, 1, n)
        return np.stack([72 * t ** 2, 72 * (1 - t) ** 2], 1)


def all_problems():
    return {"P1": P1(), "P2": P2(), "P2int": P2(True), "P3": P3(), "P4": P4(), "P4pen": P4("penalty"),
            "P5": P5(), "P6": P6(), "P7": P7(), "P8": P8(), "P9": P9(), "P10_M3": P10(3), "P10_M5": P10(5),
            "P11": P11(), "P13a": P13a(), "P14": P14()}


# =========================================================================================
#  reference fronts for the problems without a closed form (dense sampling, Task A / D use these for IGD)
# =========================================================================================
def build_dense_reference(name, save=True):
    """Dense-sampling reference front for P4, P5, P6 (2-4 variables, so brute force is affordable)."""
    os.makedirs(REF_DIR, exist_ok=True)
    if name == "P4":
        p = P4("repair"); step = 200
        pts = [(i, j, k, step - i - j - k) for i in range(step + 1) for j in range(step + 1 - i)
               for k in range(step + 1 - i - j)]
        W = np.array(pts, float) / step
        F, _ = p.evaluate(W)
    elif name == "P5":
        p = P5()
        b = np.linspace(p.lb[0], p.ub[0], 1800); h = np.linspace(p.lb[1], p.ub[1], 1800)
        B, H = np.meshgrid(b, h)
        X = np.stack([B.ravel(), H.ravel()], 1)
        hb = np.sqrt(p.BH2_MIN / b).clip(p.lb[1], p.ub[1])
        X = np.vstack([X, np.stack([b, hb], 1)])
        F, cv = p.evaluate(X); F = F[cv <= 0]
    elif name == "P6":
        p = P6()
        x1 = np.linspace(0, 5, 3000); x2 = np.linspace(0, 3, 1800)
        A, B = np.meshgrid(x1, x2)
        X = np.stack([A.ravel(), B.ravel()], 1)
        F, cv = p.evaluate(X); F = F[cv <= 0]
    else:
        raise ValueError(name)
    keep = pareto_idx = mu.pareto_filter_2d(F)
    front = thin_front(F[keep], 1500)
    if save:
        np.save(os.path.join(REF_DIR, name + ".npy"), front)
    return front


if __name__ == "__main__":
    for nm in ("P4", "P5", "P6"):
        f = build_dense_reference(nm)
        print(nm, len(f), f.min(0), f.max(0))


class P12:
    """Supplier selection (MCDM ranking) -- a decision matrix, not an optimisation problem."""
    name = "P12"
    suppliers = ["S1", "S2", "S3", "S4", "S5"]
    criteria = ["Cost $k", "Quality /10", "Delivery days", "Reliability %"]
    minimize = np.array([True, False, True, False])
    matrix = np.array([[120, 8, 10, 92],
                       [100, 7, 14, 88],
                       [140, 9, 7, 95],
                       [90, 6, 18, 80],
                       [130, 8, 9, 90]], float)
    weights = np.array([0.3, 0.3, 0.2, 0.2])


_MAKE = {"P1": lambda: P1(), "P2": lambda: P2(), "P2int": lambda: P2(True), "P3": lambda: P3(), "P4": lambda: P4(),
         "P4pen": lambda: P4("penalty"), "P5": lambda: P5(), "P6": lambda: P6(), "P7": lambda: P7(), "P8": lambda: P8(),
         "P9": lambda: P9(), "P10_M3": lambda: P10(3), "P10_M5": lambda: P10(5), "P11": lambda: P11(), "P13a": lambda: P13a(),
         "P14": lambda: P14()}
_CACHE = {}


def make(key):
    """Build (and cache) one problem by key -- cheaper than all_problems() because P9 loads a dataset."""
    if key not in _CACHE:
        _CACHE[key] = _MAKE[key]()
    return _CACHE[key]
