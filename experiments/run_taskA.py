"""
Task A -- run every one of the 37 algorithms (fixed seed 0, 10,000 evaluations for the population methods) on the problems of its
'Test problems' column; save metrics (HV, IGD, Spacing, runtime) and one front plot per run.
Outputs: results/taskA.json, figures/A_<nn>_<alg>_<problem>.png
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
EV = 10000
INF = float("inf")
# (alg, [(problem key, kwargs, tag)])
PLAN = {
    1: [("P1", {}, ""), ("P3", {}, ""), ("P4", {}, "")],
    2: [("P2", {}, ""), ("P3", {}, ""), ("P5", {}, "")],
    3: [("P1", {}, "lambda sweep"), ("P7", {}, "lambda sweep")],
    4: [("P1", {"p": 2}, "p=2"), ("P3", {"p": INF}, "p=inf")],
    5: [("P4", {}, ""), ("P14", {}, "")],
    6: [("P11", {}, "unit w"), ("P7", {}, "unit w")],
    7: [("P11", {}, ""), ("P7", {"eps_list": [0, 2, 4, 6, 8, 10]}, "eps sweep")],
    10: [("P2", {}, ""), ("P3", {}, "")],
    11: [("P3", {}, ""), ("P5", {}, "")],
    12: [("P5", {}, ""), ("P7", {}, "")],
    13: [("P3", {}, ""), ("P5", {}, "")],
    14: [("P3", {}, ""), ("P5", {}, ""), ("P6", {}, "")],
    15: [("P10_M3", {}, ""), ("P10_M5", {}, "")],
    16: [("P4", {}, ""), ("P5", {}, "")],
    17: [("P3", {}, ""), ("P5", {}, "")],
    18: [("P3", {}, ""), ("P10_M3", {}, "")],
    19: [("P3", {}, ""), ("P5", {}, "")],
    20: [("P8", {}, "")],
    21: [("P5", {}, ""), ("P6", {}, ""), ("P7", {}, "")],
    22: [("P4", {}, ""), ("P5", {}, "")],
    23: [("P5", {}, ""), ("P9", {}, "")],
    24: [("P3", {}, ""), ("P6", {}, "")],
    25: [("P8", {}, "")],
    26: [("P3", {}, ""), ("P5", {}, "")],
    27: [("P3", {}, "lecture rule"), ("P5", {}, "lecture rule")],
    28: [("P3", {}, ""), ("P5", {}, "")],
    29: [("P3", {}, ""), ("P4", {}, "")],
    30: [("P5", {}, ""), ("P9", {}, "")],
    31: [("P10_M3", {}, ""), ("P10_M5", {}, "")],
    32: [("P3", {}, ""), ("P14", {}, "")],
    33: [("P3", {}, ""), ("P14", {}, "")],
    34: [("P10_M3", {}, ""), ("P10_M5", {}, "")],
    35: [("P5", {}, ""), ("P14", {}, "")],
}
CLASSICAL = {1, 2, 3, 4, 5, 6, 7}


def front_plot(n, alg_name, key, r, fname):
    P = pb.make(key)
    fig, ax = plt.subplots(figsize=(4.3, 3.3), dpi=110)
    ref = P.reference_front()
    M = P.n_obj
    F = np.array(r["F"])
    if M == 2:
        if ref is not None:
            R = P.natural(ref); o = np.argsort(R[:, 0])
            ax.plot(R[o, 0], R[o, 1], "-", color="0.65", lw=2.2, label="true / reference front", zorder=1)
        if len(F):
            N = P.natural(F)
            ax.scatter(N[:, 0], N[:, 1], s=14 if len(F) > 30 else 40, c="#d95f02", zorder=3, label=f"obtained ({len(F)} pts)")
        if key == "P14" or key == "P7":
            pass
        ax.set_xlabel(P.obj_names[0] if P.obj_names else "f1"); ax.set_ylabel(P.obj_names[1] if P.obj_names else "f2")
    else:
        # parallel view: f1-f2 projection plus the sphere constraint check
        if len(F):
            ax.scatter(F[:, 0], F[:, 1], s=10, c="#d95f02", label=f"obtained ({len(F)} pts)")
        if ref is not None:
            ax.scatter(ref[::7, 0], ref[::7, 1], s=3, c="0.7", label="true front", zorder=0)
        ax.set_xlabel("f1"); ax.set_ylabel("f2")
    ax.set_title(f"#{n} {alg_name} on {key}\nHV={r['hv']:.4f}" + ("" if np.isnan(r['igd']) else f"  IGD={r['igd']:.4f}"), fontsize=8.5)
    ax.legend(fontsize=6.5, loc="best"); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def main():
    only_p9 = "p9" in sys.argv[1:]
    jobs = []
    for n, lst in PLAN.items():
        for key, kw, tag in lst:
            if (key == "P9") != only_p9:
                continue
            jobs.append((n, key, SEED, None if n in CLASSICAL else EV, kw, tag))
    res = run_many(jobs)
    res.sort(key=lambda r: (r["alg"], r["prob"]))
    for r in res:
        fn = f"A_{r['alg']:02d}_{r['name'].replace('/','-').replace(' ','_')}_{r['prob']}.png"
        r["figure"] = fn
        front_plot(r["alg"], r["name"], r["prob"], r, fn)
    save_json(res, "taskA_p9.json" if only_p9 else "taskA.json")
    print("Task A done:", len(res), "runs")
    for r in res:
        print(f"{r['alg']:2d} {r['name']:20s} {r['prob']:7s} n={r['n']:4d} evals={r['evals']:6d} HV={r['hv']:.4f} IGD={r['igd']:.4f} sp={r['spacing']:.4f} t={r['seconds']:.1f}s")


if __name__ == "__main__":
    main()
