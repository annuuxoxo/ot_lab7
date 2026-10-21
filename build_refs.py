"""
build_refs.py -- reference fronts for the problems without a closed-form front (used for IGD and for the normalisation bounds).
  P4, P5, P6 : dense sampling (problems.build_dense_reference)
  P8, P9     : the non-dominated union of LONG runs (200k / 30k evaluations, several seeds) of several of our own algorithms.
Pooled fronts are a best-known approximation, so IGD on P8 / P9 is flagged with * in the report.
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import problems as pb, moo_utils as mu, algorithms as A

os.makedirs(pb.REF_DIR, exist_ok=True)
for nm in ("P4", "P5", "P6"):
    f = pb.build_dense_reference(nm); print(nm, len(f), flush=True)

def pool(problem, jobs):
    allF = []
    for alg, seeds, evals, kw in jobs:
        m = A.load(alg)
        for s in seeds:
            r = m.run(problem, seed=s, max_evals=evals, **kw)
            allF.append(r.F); print("  ", A.NAME[alg], s, len(r.F), flush=True)
    F = np.vstack(allF)
    F = F[mu.non_dominated_mask(F)]
    _, u = np.unique(np.round(F, 9), axis=0, return_index=True)
    return F[np.sort(u)]

P8 = pb.P8()
f8 = pool(P8, [(14, range(5), 200000, {}), (16, range(2), 100000, {}), (20, range(3), 100000, {}), (30, range(3), 100000, {}),
               (18, range(2), 100000, {"N": 100})])
np.save(os.path.join(pb.REF_DIR, "P8.npy"), f8); print("P8 reference points:", len(f8), f8.min(0), f8.max(0), flush=True)

P9 = pb.P9()
f9 = pool(P9, [(14, range(3), 30000, {}), (30, range(2), 30000, {}), (16, range(1), 20000, {}), (23, range(1), 20000, {})])
np.save(os.path.join(pb.REF_DIR, "P9.npy"), f9); print("P9 reference points:", len(f9), f9.min(0), f9.max(0), "baseline err", P9.baseline_error(), flush=True)
