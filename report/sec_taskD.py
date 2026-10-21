"""Task D: head-to-head comparison tables and discussion."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl import *
import numpy as np
import algorithms as A

RES = os.path.join(ROOT, "results")


def load():
    return json.load(open(os.path.join(RES, "taskD.json")))


def group(runs, key=("alg", "tag")):
    g = {}
    for r in runs:
        g.setdefault(tuple(r[k] for k in key), []).append(r)
    return g


def stats(rs, met):
    v = np.array([r[met] for r in rs], float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"), float(np.nanstd(v)) if np.isfinite(v).any() else float("nan")


def ranking_tables(D1):
    """Returns per-problem table rows and the combined ranking."""
    res = {}
    for prob in ("P3", "P5"):
        g = group([r for r in D1 if r["prob"] == prob])
        rows = []
        for (alg, tag), rs in g.items():
            rows.append(dict(alg=alg, tag=tag, name=A.NAME[alg], hv=stats(rs, "hv"), igd=stats(rs, "igd"), sp=stats(rs, "spacing"), n=stats(rs, "n")[0],
                             t=stats(rs, "seconds")[0], ev=stats(rs, "evals")[0]))
        for met, rk in (("hv", "rhv"), ("igd", "rigd"), ("sp", "rsp")):
            vals = [r[met][0] for r in rows]
            order = sorted(range(len(rows)), key=lambda i: (np.isnan(vals[i]), -vals[i] if met == "hv" else vals[i]))
            for rank, i in enumerate(order):
                rows[i][rk] = rank + 1
        res[prob] = rows
    # combined
    comb = {}
    for prob, rows in res.items():
        for r in rows:
            comb.setdefault((r["alg"], r["tag"]), []).append(r)
    final = []
    for k, rs in comb.items():
        if len(rs) < 2:
            continue
        score = np.mean([x["rhv"] for x in rs] + [x["rigd"] for x in rs])
        final.append((score, k, rs))
    final.sort(key=lambda t: t[0])
    return res, final


def label(r):
    return f"#{r['alg']} {r['name']}" + (f" [{r['tag']}]" if r["tag"] else "")


def d1(D):
    res, final = ranking_tables(D["D1"])
    o = [H2("D1  Continuous bi-objective: P3 (ZDT2, unconstrained) and P5 (beam, constrained)"),
         P("Every algorithm of our set that applies to a continuous problem was run with 10,000 evaluations and 10 seeds (classical methods: 21 weights / eps values / reference points, "
           "evaluations reported). HV and IGD are in the normalised space (HV reference 1.1); rank 1 is best. R-NSGA-II is a preference method (its front is a focused subset around a "
           "reference point), so its rank is shown for completeness only. MODE appears twice: the lecture rule and the NSGA-II-survival variant.", "p")]
    for prob, title in (("P3", "P3 (ZDT2, n = 10) -- unconstrained, non-convex"), ("P5", "P5 (cantilever beam) -- constrained")):
        rows = [["rank (HV)", "algorithm", "HV mean &plusmn; std", "IGD mean &plusmn; std", "Spacing mean &plusmn; std", "#pts", "evals", "time [s]", "rank IGD"]]
        for r in sorted(res[prob], key=lambda x: x["rhv"]):
            rows.append([str(r["rhv"]), label(r), pm(*r["hv"]), pm(*r["igd"]), pm(*r["sp"]), f"{r['n']:.0f}", f"{r['ev']:,.0f}", f"{r['t']:.1f}", str(r["rigd"])])
        o += [H3(title), tbl(rows, [11, 46, 28, 28, 28, 8, 12, 10, 10], True)]
    rows = [["overall rank", "algorithm", "mean of the 4 ranks (HV, IGD on P3 and P5)", "HV rank P3 / P5", "IGD rank P3 / P5"]]
    for i, (sc, k, rs) in enumerate(final):
        d = {x["alg"]: x for x in rs}
        by = {}
        for prob, rr in res.items():
            for x in rr:
                if (x["alg"], x["tag"]) == k:
                    by[prob] = x
        rows.append([str(i + 1), label(by["P3"]), f"{sc:.1f}", f"{by['P3']['rhv']} / {by['P5']['rhv']}", f"{by['P3']['rigd']} / {by['P5']['rigd']}"])
    o += [H3("Combined ranking (algorithms that ran on both problems)"), tbl(rows, [18, 60, 42, 28, 28], True)]
    return o, res, final


def d2(D):
    o = [H2("D2  Many-objective: DTLZ2 with M = 5 (and M = 3)")]
    for key, N in (("P10_M5", "N = 72 (NSGA-II, NSGA-III, HypE), 70 reference vectors / weight vectors (MOEA/D, RVEA)"), ("P10_M3", "N = 92 / 91 / H = 91")):
        g = group([r for r in D["D2"] if r["prob"] == key], key=("alg",))
        rows = [["algorithm", "HV (MC) mean &plusmn; std", "IGD mean &plusmn; std", "Spacing mean &plusmn; std", "#pts", "time [s]"]]
        items = sorted(g.items(), key=lambda kv: stats(kv[1], "igd")[0])
        for (alg,), rs in items:
            rows.append([f"#{alg} {A.NAME[alg]}", pm(*stats(rs, "hv")), pm(*stats(rs, "igd")), pm(*stats(rs, "spacing")), f"{stats(rs, 'n')[0]:.0f}", f"{stats(rs, 'seconds')[0]:.1f}"])
        o += [H3(("M = 5" if key.endswith("M5") else "M = 3") + f" -- {N}; sorted by IGD (best first)"), tbl(rows, [40, 38, 38, 38, 10, 12], True)]
    return o


def d3(D):
    g = group(D["D3"], key=("alg",))
    rows = [["algorithm", "HV mean &plusmn; std", "Spacing mean &plusmn; std", "IGD* mean &plusmn; std", "#pts (non-dominated tours)", "time [s]"]]
    for (alg,), rs in sorted(g.items(), key=lambda kv: -stats(kv[1], "hv")[0]):
        rows.append([f"#{alg} {A.NAME[alg]}", pm(*stats(rs, "hv")), pm(*stats(rs, "spacing")), pm(*stats(rs, "igd")), f"{stats(rs, 'n')[0]:.1f}", f"{stats(rs, 'seconds')[0]:.1f}"])
    return [H2("D3  Discrete: bi-objective TSP (P8), permutation operators"), tbl(rows, [40, 36, 36, 36, 26, 12], True)]
