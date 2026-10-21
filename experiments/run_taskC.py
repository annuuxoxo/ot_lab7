"""
Task C3 -- sensitivity of the most important parameter of the three deep-dive algorithms (roll no. 77): 3 values x 10 seeds.
  Algorithm 3  Benson's method : reference point scale z_scale (position of z* along the anti-diagonal of the ideal-nadir box), on P3
  Algorithm 15 NSGA-III        : divisions p (reference points H = C(k+p-1,p)), on P10 with M = 5, budget 10,000 evaluations
  Algorithm 27 MODE            : scale factor F, on P3 (lecture selection rule), budget 10,000 evaluations
Outputs results/taskC.json and figures/C_*.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = list(range(10)); EV = 10000


def jobs():
    J = []
    for z in (1.0, 1.3, 1.8):
        for s in SEEDS:
            J.append((3, "P3", s, None, {"mode": "reference", "n_points": 21, "z_scale": z}, f"z_scale={z}"))
    for p in (2, 4, 6):
        for s in SEEDS:
            J.append((15, "P10_M5", s, EV, {"p": p}, f"p={p}"))
    for F in (0.2, 0.5, 0.9):
        for s in SEEDS:
            J.append((27, "P3", s, EV, {"F": F}, f"F={F}"))
    return J


def boxplot(ax, groups, labels, ylabel, title):
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)


def main():
    res = run_many(jobs())
    out = dict(runs=res)
    summ = {}
    for alg, prob in ((3, "P3"), (15, "P10_M5"), (27, "P3")):
        tags = sorted({r["tag"] for r in res if r["alg"] == alg})
        summ[f"{alg}"] = {}
        for t in tags:
            rs = [r for r in res if r["alg"] == alg and r["tag"] == t]
            summ[f"{alg}"][t] = dict(hv_mean=float(np.mean([r["hv"] for r in rs])), hv_std=float(np.std([r["hv"] for r in rs])),
                                     igd_mean=float(np.nanmean([r["igd"] for r in rs])), igd_std=float(np.nanstd([r["igd"] for r in rs])),
                                     n_mean=float(np.mean([r["n"] for r in rs])), spacing_mean=float(np.nanmean([r["spacing"] for r in rs])),
                                     mid_mean=float(np.mean([sum(0.3 < f[0] < 0.7 for f in r["F"]) for r in rs])) if alg != 15 else float("nan"),
                                     seconds=float(np.mean([r["seconds"] for r in rs])))
    out["summary"] = summ
    save_json(out, "taskC.json")
    # figures --------------------------------------------------------------------------------------------
    for alg, prob, title, fname in ((3, "P3", "Benson (z-sweep) on P3: reference-point scale", "C_alg03_sensitivity.png"),
                                    (15, "P10_M5", "NSGA-III on DTLZ2 M=5: divisions p", "C_alg15_sensitivity.png"),
                                    (27, "P3", "MODE (lecture rule) on P3: scale factor F", "C_alg27_sensitivity.png")):
        tags = sorted({r["tag"] for r in res if r["alg"] == alg}, key=lambda t: float(t.split("=")[1]))
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3), dpi=110)
        boxplot(axs[0], [[r["hv"] for r in res if r["alg"] == alg and r["tag"] == t] for t in tags], tags, "Hypervolume (higher better)", "HV, 10 seeds")
        boxplot(axs[1], [[r["igd"] for r in res if r["alg"] == alg and r["tag"] == t] for t in tags], tags, "IGD (lower better)", "IGD, 10 seeds")
        P = pb.make(prob); ref = P.reference_front(); ax = axs[2]
        cols = ["#1b9e77", "#d95f02", "#7570b3"]
        if ref is not None and P.n_obj == 2:
            ax.plot(ref[:, 0], ref[:, 1], color="0.75", lw=3, label="true front")
        for c, t in zip(cols, tags):
            r = [x for x in res if x["alg"] == alg and x["tag"] == t][0]
            Fm = np.array(r["F"])
            if len(Fm):
                ax.scatter(Fm[:, 0], Fm[:, 1], s=10, c=c, label=f"{t} ({len(Fm)} pts)")
        ax.set_title("fronts of seed 0", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
        fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
