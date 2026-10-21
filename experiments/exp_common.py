"""Common machinery for the experiment scripts: parallel execution of (algorithm, problem, seed) jobs and metric computation."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process: no oversubscription
import sys, time, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import multiprocessing as mp
import problems as pb, moo_utils as mu, algorithms as A

RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)


def run_job(job):
    """job = (alg_number, problem_key, seed, max_evals, kwargs, tag)  ->  metrics + the front."""
    alg, key, seed, evals, kw, tag = job
    P = pb.make(key); m = A.load(alg)
    t0 = time.time()
    r = m.run(P, seed=seed, max_evals=evals, **kw)
    F = np.asarray(r.F)
    out = dict(alg=alg, name=A.NAME[alg], prob=key, seed=seed, tag=tag, evals=int(r.evals), seconds=time.time() - t0,
               n=int(len(F)), hv=P.hv(F), igd=P.igd(F), spacing=P.spacing(F), F=F.tolist(),
               history=[list(h) for h in r.history], n_infeasible_returned=int(r.extra.get("n_infeasible_returned", 0)))
    ex = r.extra
    for k in ("n_returned",):
        if k in ex:
            out[k] = int(ex[k])
    return out


def run_many(jobs, procs=9, verbose=True):
    t0 = time.time(); res = []
    with mp.get_context("fork").Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1)):
            res.append(r)
            if verbose and (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(jobs)} jobs  {time.time()-t0:.0f}s", flush=True)
    return res


def summarize(results, keys=("alg", "prob", "tag")):
    """mean/std of the metrics over seeds for every (alg, prob, tag) group."""
    groups = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    rows = []
    for g, rs in groups.items():
        row = dict(zip(keys, g)); row["name"] = rs[0]["name"]; row["runs"] = len(rs)
        for met in ("hv", "igd", "spacing", "seconds", "n", "evals"):
            v = np.array([x[met] for x in rs], float)
            row[met + "_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[met + "_std"] = float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")
        rows.append(row)
    return rows


def save_json(obj, name):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)
