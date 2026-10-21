"""
Algorithm 15 -- NSGA-III (Deb & Jain 2014; elitist, many-objective; Part 3 slides 12-19 and 5E slides 5-7)

Lecture steps
  1. Generate H = C(k + p - 1, p) structured reference points (Das-Dennis) on the normalised hyperplane, p divisions per axis.
  2. Initialise N ~ H individuals (N = H rounded up to a multiple of 4) and evaluate the k objectives.
  3. Merge parents + offspring, non-dominated sort into F1, F2, ... and take whole fronts until |S_t| >= N.
  4. Normalise: ideal point = per-objective minimum; nadir estimate (normalization='extreme': ASF extreme points + hyperplane
     intercepts of the paper; normalization='max': translated per-objective maximum, the 5E teaching simplification).
  5. Associate each member of S_t with the reference LINE of smallest perpendicular distance; count niches rho_j using only
     the members already accepted (F1..F_{l-1}).
  6. Fill the last front: repeatedly take the reference point with the fewest members (random tie), then the closest associated
     member of the last front if rho_j = 0, a random one otherwise; drop reference points that have no candidates.
Mating: random pairs (no pressure needed -- the niching supplies it), SBX + polynomial mutation.
Defaults: p = 12 for k=3 (H=91 -> N=92), p = 4 for k=5 (H=70 -> N=72); k=2: p = 99 (N = 100). pc = 1.0, eta_c = 30, eta_m = 20.
"""
import numpy as np
import moo_utils as mu
import ops
from algorithms.base import make_run
from algorithms.ea_common import PopState

DEFAULTS = dict(p=None, normalization="extreme", pc=1.0, eta_c=30.0, eta_m=20.0)
P_DEFAULT = {2: 99, 3: 12, 5: 4}


def reference_points(k, p):
    return mu.das_dennis(k, p)


def normalize_extreme(Fs, mode="extreme"):
    """Return (normalised objectives, ideal, intercepts) for the candidate set Fs."""
    ideal = Fs.min(0)
    Ft = Fs - ideal
    k = Fs.shape[1]
    if mode == "max":
        icpt = Ft.max(0)
    else:
        ext = np.empty((k, k))
        for j in range(k):
            w = np.full(k, 1e-6); w[j] = 1.0
            asf = (Ft / w).max(1)
            ext[j] = Ft[int(np.argmin(asf))]
        try:
            a = np.linalg.solve(ext, np.ones(k))
            icpt = 1.0 / a
            if not np.all(np.isfinite(icpt)) or np.any(icpt < 1e-6) or np.any(a <= 0):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            icpt = Ft.max(0)
    icpt = np.where(icpt < 1e-12, 1.0, icpt)
    return Ft / icpt, ideal, icpt


def associate(Fnorm, Z):
    """Perpendicular distance of every point to every reference line through the origin; returns (nearest, dist_to_nearest, D)."""
    W = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    proj = Fnorm @ W.T                                      # (n, H)
    d2 = (Fnorm ** 2).sum(1)[:, None] - proj ** 2
    D = np.sqrt(np.maximum(d2, 0.0))
    near = D.argmin(1)
    return near, D[np.arange(len(Fnorm)), near], D


def niching(near, dist, n_accepted, last_idx, K, H, rng):
    """Steps 5-6: choose K members of the last front.  `near/dist` are indexed like S_t; n_accepted = #members of F1..F_{l-1}
    (they come first in S_t); last_idx = positions in S_t of the last front."""
    rho = np.zeros(H, int)
    for j in near[:n_accepted]:
        rho[j] += 1
    avail = set(last_idx.tolist()); excluded = set(); chosen = []
    while len(chosen) < K:
        cand_refs = [j for j in range(H) if j not in excluded]
        mn = min(rho[j] for j in cand_refs)
        js = [j for j in cand_refs if rho[j] == mn]
        j = js[0] if rng is None else js[rng.integers(len(js))]           # rng=None: deterministic (hand trace) tie-breaks
        members = sorted(i for i in avail if near[i] == j)
        if not members:
            excluded.add(j); continue
        if rho[j] == 0:
            pick = min(members, key=lambda i: dist[i])
        else:
            pick = members[0] if rng is None else members[rng.integers(len(members))]
        chosen.append(pick); avail.discard(pick); rho[j] += 1
    return np.array(chosen)


def select_survivors(S, N, Z, rng, CV=None, normalization="extreme"):
    """Environmental selection of NSGA-III on the union S; returns (indices, info dict for traces)."""
    fronts, rank = mu.fast_non_dominated_sort(S, CV)
    St = []; l = 0
    for l, fr in enumerate(fronts):
        St.extend(fr.tolist())
        if len(St) >= N:
            break
    St = np.array(St)
    if len(St) == N:
        return St, dict(fronts=[f.tolist() for f in fronts], note="no niching needed")
    n_acc = len(St) - len(fronts[l])
    Fnorm, ideal, icpt = normalize_extreme(S[St], normalization)
    near, dist, D = associate(Fnorm, Z)
    K = N - n_acc
    last_local = np.arange(n_acc, len(St))
    pick = niching(near, dist, n_acc, last_local, K, len(Z), rng)
    chosen = np.concatenate([St[:n_acc], St[pick]])
    info = dict(fronts=[f.tolist() for f in fronts], St=St.tolist(), n_accepted=int(n_acc), ideal=ideal, intercepts=icpt,
                Fnorm=Fnorm, near=near, dist=dist, picked=St[pick].tolist())
    return chosen, info


def initialize(problem, seed=0, max_evals=10000, p=None, N=None, normalization="extreme", pc=1.0, eta_c=30.0,
               eta_m=20.0, constraint="penalty", **kw):
    k = problem.n_obj
    p = P_DEFAULT.get(k, 6) if p is None else p
    Z = reference_points(k, p)
    H = len(Z)
    N = int(np.ceil(H / 4.0) * 4) if N is None else N
    st = PopState(problem, seed, max_evals, N, constraint)
    st.Z, st.normalization, st.pc, st.eta_c, st.eta_m = Z, normalization, pc, eta_c, eta_m
    st.init_population()
    return st


def evaluate(st, X):
    return st.evaluate(X)


def step(st):
    N = st.N
    perm = st.rng.permutation(N)
    perm = np.concatenate([perm, st.rng.permutation(N)])[: N + N % 2]
    Q = ops.variation(st.problem, st.X[perm], st.rng, pc=st.pc, eta_c=st.eta_c, eta_m=st.eta_m)[:N]
    Fq, CVq, Sq = st.evaluate(Q)
    X = np.vstack([st.X, Q]); F = np.vstack([st.F, Fq]); CV = np.concatenate([st.CV, CVq]); S = np.vstack([st.S, Sq])
    idx, info = select_survivors(S, N, st.Z, st.rng, None, st.normalization)
    st.X, st.F, st.CV, st.S = X[idx], F[idx], CV[idx], S[idx]
    st.last_info = info


run = make_run(initialize, step)
