"""Figures for Task D from results/taskD.json."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = load_json("taskD.json")


def label(r):
    return r["name"] + (f" [{r['tag']}]" if r["tag"] else "")


def d1_small_multiples(key, fname):
    P = pb.make(key); runs = [r for r in D["D1"] if r["prob"] == key and r["seed"] == 0]
    runs.sort(key=lambda r: (r["alg"], r["tag"]))
    n = len(runs); cols = 6; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(14, 2.3 * rows), dpi=100)
    ref = P.reference_front(); R = P.natural(ref); o = np.argsort(R[:, 0])
    for ax, r in zip(axs.ravel(), runs):
        ax.plot(R[o, 0], R[o, 1], "-", color="0.75", lw=2.5)
        F = np.array(r["F"])
        if len(F):
            N = P.natural(F); ax.scatter(N[:, 0], N[:, 1], s=6, c="#d95f02")
        ax.set_title(f"#{r['alg']} {label(r)}\nHV={r['hv']:.3f}", fontsize=6.5); ax.tick_params(labelsize=5); ax.grid(alpha=0.2)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Fronts of seed 0 on {key} (grey = true front)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def d1_overlay(key, fname, algs):
    P = pb.make(key); fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=120)
    ref = P.reference_front(); R = P.natural(ref); o = np.argsort(R[:, 0]); ax.plot(R[o, 0], R[o, 1], "-", color="0.8", lw=5, label="true front")
    cm = plt.get_cmap("tab10")
    for i, (a, tag) in enumerate(algs):
        rs = [r for r in D["D1"] if r["prob"] == key and r["alg"] == a and r["tag"] == tag and r["seed"] == 0]
        if not rs: continue
        r = rs[0]; F = np.array(r["F"])
        if len(F):
            N = P.natural(F); ax.scatter(N[:, 0], N[:, 1], s=14, color=cm(i % 10), marker="os^vDpP*hX"[i % 10], label=f"#{a} {r['name']}" + (f" [{tag}]" if tag else ""), alpha=0.85)
    ax.set_xlabel(P.obj_names[0]); ax.set_ylabel(P.obj_names[1]); ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3)
    ax.set_title(f"{key}: fronts of selected algorithms (seed 0)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def d1_box(key, fname):
    runs = [r for r in D["D1"] if r["prob"] == key]
    groups = {}
    for r in runs: groups.setdefault((r["alg"], r["tag"]), []).append(r)
    items = sorted(groups.items(), key=lambda kv: -np.mean([x["hv"] for x in kv[1]]))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), dpi=100)
    for ax, met, ttl in ((axs[0], "hv", "hypervolume (higher better)"), (axs[1], "igd", "IGD (lower better, log scale)")):
        data = [[x[met] for x in v] for k, v in items]
        ax.boxplot(data, tick_labels=[f"#{k[0]}" + ("*" if k[1] else "") for k, v in items], showmeans=True)
        ax.set_title(f"{key}: {ttl}, 10 seeds", fontsize=9); ax.tick_params(labelsize=6.5, axis="x", rotation=90); ax.grid(alpha=0.3)
        if met == "igd": ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


def d2_figures():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5), dpi=105)
    for key, ax_i, M in (("P10_M5", 0, 5), ("P10_M3", 1, 3)):
        runs = [r for r in D["D2"] if r["prob"] == key]
        algs = [14, 15, 18, 31, 34]
        data = [[r["igd"] for r in runs if r["alg"] == a] for a in algs]
        axs[ax_i].boxplot(data, tick_labels=["NSGA-II", "NSGA-III", "MOEA/D", "RVEA", "HypE"], showmeans=True)
        axs[ax_i].set_title(f"DTLZ2 M={M}: IGD, 10 seeds (lower better)", fontsize=9); axs[ax_i].grid(alpha=0.3); axs[ax_i].tick_params(labelsize=8)
    axs[2].plot(D["D2_front1_fraction_M5"], label="M = 5 (N = 72)"); axs[2].plot(D["D2_front1_fraction_M3"], label="M = 3 (N = 92)")
    axs[2].set_xlabel("generation"); axs[2].set_ylabel("fraction of NSGA-II population in front 1"); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
    axs[2].set_title("selection pressure of NSGA-II", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "D2_manyobj.png")); plt.close(fig)


def d3_figures():
    P = pb.make("P8"); ref = P.reference_front()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7), dpi=105)
    cm = {20: "#1b9e77", 25: "#d95f02", 14: "#7570b3", 30: "#e7298a"}
    nm = {20: "MOEA/D-ACO", 25: "MOACO", 14: "NSGA-II", 30: "MOMA"}
    axs[0].scatter(ref[:, 0], ref[:, 1], s=70, facecolors="none", edgecolors="k", label="best-known front (10 tours)")
    for a in (20, 25, 14, 30):
        # union of the non-dominated points over the 10 seeds
        F = np.vstack([np.array(r["F"]) for r in D["D3"] if r["alg"] == a])
        F = F[mu.non_dominated_mask(F)]
        axs[0].scatter(F[:, 0], F[:, 1], s=14, c=cm[a], label=f"{nm[a]} (union of 10 seeds)", alpha=0.8)
    axs[0].set_xlabel("total distance"); axs[0].set_ylabel("total time"); axs[0].legend(fontsize=6.5); axs[0].grid(alpha=0.3); axs[0].set_title("P8 fronts", fontsize=9)
    data = [[r["hv"] for r in D["D3"] if r["alg"] == a] for a in (20, 25, 14, 30)]
    axs[1].boxplot(data, tick_labels=[nm[a] for a in (20, 25, 14, 30)], showmeans=True); axs[1].set_title("P8: hypervolume, 10 seeds", fontsize=9); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "D3_tsp.png")); plt.close(fig)


def d5_figures():
    G = D["D5"]; net = G["net"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=105)
    for ax, sc in zip(axs[:2], ("1.0", "10.0")):
        for meth, c in (("fixed", "k"), ("mgda", "#d95f02"), ("gradnorm", "#1b9e77")):
            d = net[f"{meth}|scale={sc}"]
            ax.plot(d["Lc"], c=c, ls="-", label=f"{meth}: classification"); ax.plot(d["Lr"], c=c, ls="--", label=f"{meth}: regression")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title(f"mean of 10 data seeds, regression target scale {float(sc):g}", fontsize=8.5)
        ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    d = net["gradnorm_lrw0.025|scale=10"]
    w = np.array(d["w"]); axs[2].plot(w[:, 0], label="w_cls (lr_w = 0.025)"); axs[2].plot(w[:, 1], label="w_reg (lr_w = 0.025)")
    axs[2].set_xlabel("epoch"); axs[2].set_title("GradNorm weights with the lecture's lr_w = 0.025", fontsize=8.5); axs[2].legend(fontsize=7); axs[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "D5_gradient.png")); plt.close(fig)


if __name__ == "__main__":
    d1_small_multiples("P3", "D1_P3_fronts.png"); d1_small_multiples("P5", "D1_P5_fronts.png")
    d1_overlay("P3", "D1_P3_overlay.png", [(14, ""), (19, ""), (16, ""), (29, ""), (1, "WS 21w"), (2, "eps 21"), (4, "WM p=inf 21w"), (10, ""), (27, ""), (26, "")])
    d1_overlay("P5", "D1_P5_overlay.png", [(14, ""), (21, ""), (19, ""), (16, ""), (22, ""), (1, "WS 21w"), (2, "eps 21"), (10, ""), (13, ""), (35, "")])
    d1_box("P3", "D1_P3_box.png"); d1_box("P5", "D1_P5_box.png")
    d2_figures(); d3_figures(); d5_figures()
    print("figures D done")
