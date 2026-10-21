"""Evidence figures (Task B / Section 4 investigations) from results/evidence.json plus a few quick re-runs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *
import ops
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = load_json("evidence.json")
INF = float("inf")


def classical_p3():
    P3 = pb.make("P3"); ref = P3.true_front(300)
    runs = [("Weighted sum (21 weights)", A.load(1).run(P3, n_weights=21)),
            ("epsilon-constraint (21 eps)", A.load(2).run(P3, n_eps=21)),
            ("Benson: lambda sweep (z*=nadir)", A.load(3).run(P3, n_points=21)),
            ("Benson: z* sweep (scale 1.3)", A.load(3).run(P3, mode="reference", n_points=21, z_scale=1.3)),
            ("Weight metric p = 1", A.load(4).run(P3, p=1, n_weights=21)), ("Weight metric p = 2", A.load(4).run(P3, p=2, n_weights=21)),
            ("Weight metric p = inf (Tchebycheff)", A.load(4).run(P3, p=INF, n_weights=21))]
    fig, axs = plt.subplots(2, 4, figsize=(13.5, 6.0), dpi=100)
    for ax, (t, r) in zip(axs.ravel(), runs):
        ax.plot(ref[:, 0], ref[:, 1], color="0.75", lw=3)
        ax.scatter(r.F[:, 0], r.F[:, 1], s=26, c="#d95f02", zorder=3)
        ax.set_title(f"{t}\n{len(r.F)} distinct pts, HV = {P3.hv(r.F):.3f}", fontsize=7.5); ax.grid(alpha=0.25); ax.tick_params(labelsize=6)
    axs.ravel()[-1].axis("off")
    fig.suptitle("P3 (ZDT2, non-convex): what each classical sweep can reach", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_classical_P3.png")); plt.close(fig)


def classical_convex():
    P1 = pb.make("P1"); ref = P1.true_front(300)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), dpi=100)
    r = A.load(1).run(P1, n_weights=11); axs[0].plot(ref[:, 0], ref[:, 1], color="0.75", lw=3); axs[0].scatter(np.array(r.extra["all_F"])[:, 0], np.array(r.extra["all_F"])[:, 1], c="#d95f02", s=26)
    axs[0].set_title("P1: weighted sum, 11 equal steps in w1:\nspacing is NOT equal", fontsize=8)
    for ax, p in ((axs[1], 1), (axs[2], INF)):
        rr = A.load(4).run(P1, p=p, n_weights=11); ax.plot(ref[:, 0], ref[:, 1], color="0.75", lw=3); ax.scatter(rr.F[:, 0], rr.F[:, 1], c="#1b9e77", s=26)
        ax.set_title(f"P1: weight metric p = {'inf' if p == INF else p}, 11 weights", fontsize=8)
    for ax in axs: ax.grid(alpha=0.25); ax.set_xlabel("f1 = (x-3)^2"); ax.set_ylabel("f2 = (x-7)^2")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_classical_P1.png")); plt.close(fig)


def mcdm_plot():
    P11 = pb.make("P11"); P7 = pb.make("P7")
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=100)
    ax = axs[0]
    ax.fill([0, 400, 300, 0], [0, 0, 200, 400], color="#cfe2f3", label="feasible region")
    ax.scatter([300], [200], c="r", s=70, zorder=4, label="GP optimum = lexicographic = (300,200)"); ax.scatter([0], [400], c="k", marker="x", s=60, label="tie vertex (0,400): same profit 2400")
    ax.set_xlabel("xA"); ax.set_ylabel("xB"); ax.legend(fontsize=6.5); ax.set_title("P11: profit tie along the edge (300,200)-(0,400)", fontsize=8); ax.grid(alpha=0.3)
    ax = axs[1]
    for it in E["lex_p7"]:
        ax.scatter(*it["F"], s=50, label=f"eps = {it['eps']}")
    tf = P7.natural(P7.true_front(100)); ax.plot(tf[:, 0], tf[:, 1], color="0.8", lw=3, zorder=0)
    ax.set_xlabel("agricultural yield"); ax.set_ylabel("industrial output"); ax.set_title("P7: lexicographic with tolerance eps on the first priority", fontsize=8); ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    ax = axs[2]
    p12 = E["p12"]; sup = ["S1", "S2", "S3", "S4", "S5"]; x = np.arange(5)
    ax.bar(x - 0.2, p12["topsis_C"], 0.4, label="TOPSIS C (weights 0.3/0.3/0.2/0.2)"); ax.bar(x + 0.2, p12["topsis_shift_C"], 0.4, label="TOPSIS C (cost 0.2, quality 0.4)")
    ax.set_xticks(x); ax.set_xticklabels(sup); ax.legend(fontsize=6.5); ax.set_title("P12: a 0.1 weight shift changes TOPSIS scores", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_mcdm.png")); plt.close(fig)


def premature_plot():
    H = E["premature_hist"]; fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), dpi=100)
    for k, v in H.items():
        h = np.array(v["history"])
        if len(h): axs[0].plot(h[:, 0], h[:, 1], label=f"{k} (final HV {v['final_hv']:.3f})")
    axs[0].set_xlabel("evaluations"); axs[0].set_ylabel("hypervolume of the current front"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)
    axs[0].set_title("P3: HV history -- a flat curve far below the true HV (0.538) = premature convergence", fontsize=8)
    P3 = pb.make("P3"); ref = P3.true_front(200); axs[1].plot(ref[:, 0], ref[:, 1], color="0.8", lw=4, label="true front")
    for alg, nm, c in ((14, "NSGA-II", "#1b9e77"), (27, "MODE (lecture)", "#d95f02")):
        r = A.load(alg).run(P3, seed=0, max_evals=10000); axs[1].scatter(r.F[:, 0], r.F[:, 1], s=14, c=c, label=nm)
    axs[1].legend(fontsize=7); axs[1].grid(alpha=0.3); axs[1].set_title("diversity lost: all MODE points sit at f1 = 0", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_premature.png")); plt.close(fig)


def constraints_plot():
    P6 = pb.make("P6"); P5 = pb.make("P5")
    from algorithms import alg14_nsga2 as n2
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=100)
    ref = P6.reference_front(); ax = axs[0]
    ax.plot(ref[:, 0], ref[:, 1], color="0.75", lw=4, label="true front")
    for mode, c, mk in (("ignore", "#e41a1c", "x"), ("deb", "#1b9e77", "o")):
        r = n2.run(P6, seed=0, max_evals=10000, constraint=mode)
        # for 'ignore' show everything it returned (feasible or not)
        st = n2.initialize(P6, seed=0, max_evals=10000, constraint=mode)
        while st.evals < 10000: n2.step(st)
        F, CV = st.F, st.CV
        nd = mu.non_dominated_mask(st.S); ax.scatter(F[nd, 0], F[nd, 1], c=c, marker=mk, s=18, label=f"{mode}: {int((CV[nd] > 0).sum())} of {int(nd.sum())} returned points infeasible")
    ax.set_title("P6 Binh-Korn: no feasibility rule vs Deb's rules", fontsize=8); ax.legend(fontsize=6.3); ax.grid(alpha=0.3); ax.set_xlabel("f1"); ax.set_ylabel("f2")
    ax = axs[1]
    ig = E["p5_ignore_limit"]; fa = np.array(ig["F_nolimit"]); fb = np.array(ig["F_limit"]); ra = P5.reference_front()
    ax.scatter(fa[:, 0], fa[:, 1], s=12, c="#e41a1c", label="stress limit ignored"); ax.scatter(fb[:, 0], fb[:, 1], s=12, c="#1b9e77", label="stress limit enforced")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7); ax.set_xlabel("mass [kg]"); ax.set_ylabel("deflection [m]"); ax.grid(alpha=0.3)
    ax.set_title(f"P5: ignoring the stress limit (min mass {ig['min_mass_nolimit']:.2f} vs {ig['min_mass_limit']:.2f} kg)", fontsize=8)
    ax = axs[2]
    rng = np.random.default_rng(0); X = P5.random_x(20000, rng); F, cv = P5.evaluate(X)
    ax.scatter(X[cv <= 0, 0], X[cv <= 0, 1], s=2, c="#1b9e77", label="feasible"); ax.scatter(X[cv > 0, 0], X[cv > 0, 1], s=2, c="#e41a1c", label=f"infeasible ({E['p5_infeasible_fraction']*100:.1f} %)")
    ax.set_xlabel("width b [m]"); ax.set_ylabel("height h [m]"); ax.legend(fontsize=7, markerscale=4); ax.set_title("P5 random designs", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_constraints.png")); plt.close(fig)


def p4_plot():
    P4 = pb.make("P4"); ref = P4.reference_front(); R = P4.natural(ref)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=110)
    ax.plot(R[:, 0], R[:, 1], color="0.75", lw=4, label="efficient frontier")
    cols = {"conservative": "#1f77b4", "balanced": "#2ca02c", "aggressive": "#d62728"}
    for k, v in E["p4"]["profiles"].items():
        ax.scatter(v["risk"], v["ret"], s=70, c=cols[k], zorder=4, label=f"{k}: w = {np.round(v['w'], 2).tolist()}")
    for v in E["vf_p4"]:
        ax.scatter(v["risk"], v["ret"], marker="*", s=90, c="k", zorder=5)
    ax.scatter([], [], marker="*", c="k", s=60, label="value-function optima (gamma = 60, 10, 1.5)")
    ax.set_xlabel("risk w'Sw"); ax.set_ylabel("expected return"); ax.legend(fontsize=6); ax.grid(alpha=0.3); ax.set_title("P4: investor profiles on the NSGA-II front", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_p4.png")); plt.close(fig)


def p8_plot():
    t = E["p8_tours"]; xy = np.array(t["coords"]); td = t["tour_dist"]; tt = t["tour_time"]
    P8 = pb.make("P8"); ref = P8.reference_front()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), dpi=100)
    for ax, tour, ttl in ((axs[0], td, f"shortest-distance tour ({t['best_dist']:.1f})"), (axs[1], tt, f"shortest-time tour ({t['best_time']:.1f})")):
        tr = np.array(tour + [tour[0]]); ax.plot(xy[tr, 0], xy[tr, 1], "-o", ms=4); [ax.text(xy[i, 0] + 1, xy[i, 1] + 1, str(i), fontsize=6) for i in range(15)]
        ax.set_title(ttl, fontsize=8); ax.grid(alpha=0.3)
    ax = axs[2]
    ax.scatter(ref[:, 0], ref[:, 1], s=40, c="k", label="best-known front (10 tours)")
    ax.scatter([t["nn_distance_tour"][0]], [t["nn_distance_tour"][1]], marker="s", c="r", label="nearest-neighbour tour (distance)")
    ax.scatter([t["nn_time_tour"][0]], [t["nn_time_tour"][1]], marker="^", c="orange", label="nearest-neighbour tour (time)")
    ax.scatter([E["p8_random"]["mean_dist"]], [E["p8_random"]["mean_time"]], marker="x", c="gray", label="mean of random tours")
    ax.set_xlabel("distance"); ax.set_ylabel("time"); ax.legend(fontsize=6.5); ax.grid(alpha=0.3); ax.set_title("P8 objective space", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_p8.png")); plt.close(fig)


def p9_plot():
    if "p9" not in E: return
    d = E["p9"]; F = np.array(d["front"]); fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.5), dpi=100)
    ax = axs[0]; ax.step(F[:, 1], 1 - F[:, 0], where="post", marker="o"); ax.axhline(d["baseline_acc"], color="r", ls="--", label=f"all 30 features: {d['baseline_acc']*100:.2f} %")
    ax.axhline(d["baseline_acc"] - 0.01, color="orange", ls=":", label="baseline - 1 percentage point"); ax.set_xlabel("number of features"); ax.set_ylabel("5-fold CV accuracy"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("P9: accuracy of the best-known front", fontsize=9)
    ax = axs[1]; fr = np.array(d["freq"]); o = np.argsort(-fr)[:12]; names = pb.make("P9").feature_names
    ax.barh([names[i] for i in o][::-1], fr[o][::-1], color="#4c78a8"); ax.set_xlabel("fraction of front solutions that use the feature"); ax.tick_params(labelsize=7); ax.set_title("most frequent features", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_p9.png")); plt.close(fig)


def sensitivity_panels():
    keys = [("moga_sigma", "MOGA: sigma_share", "hv"), ("nsga_sigma", "NSGA: sigma_share", "hv"), ("npga_sigma", "NPGA: sigma_share", "hv"),
            ("pesa_div", "PESA-II: grid divisions", "hv"), ("moead_T", "MOEA/D: neighbourhood T", "hv"), ("mopso_w", "MOPSO: inertia w", "hv"),
            ("moabc_limit", "MOABC: limit", "hv"), ("mosa_chains", "MOSA: chains", "hv"), ("ibea_kappa", "IBEA: kappa", "hv"), ("nsga2_N", "NSGA-II: population N", "igd")]
    fig, axs = plt.subplots(2, 5, figsize=(14.5, 5.2), dpi=100)
    for ax, (k, ttl, met) in zip(axs.ravel(), keys):
        d = E[k]; labs = list(d); vals = [d[l][met] for l in labs]
        ax.bar(range(len(labs)), vals, color="#4c78a8"); ax.set_xticks(range(len(labs))); ax.set_xticklabels([l.split("=")[-1] for l in labs], fontsize=7)
        ax.set_title(f"{ttl}\n({met.upper()} on P3, mean of 5-10 seeds)", fontsize=7.5); ax.grid(alpha=0.3, axis="y")
        if met == "hv": ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_sensitivity.png")); plt.close(fig)


def p10_plot():
    fr = E["p10_nd_fraction"]; fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=110)
    ax.bar(["M = 2\n(ZDT2)", "M = 3", "M = 5"], [fr["2"], fr["3"], fr["5"]], color=["#4c78a8", "#f58518", "#e45756"])
    for i, v in enumerate([fr["2"], fr["3"], fr["5"]]): ax.text(i, v + 0.02, f"{v*100:.0f} %", ha="center", fontsize=8)
    ax.set_ylabel("fraction of a random population\nthat is non-dominated"); ax.set_ylim(0, 1.1); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_p10.png")); plt.close(fig)


def rnsga_plot():
    P14 = pb.make("P14"); tf = P14.natural(P14.true_front(200)); fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=110)
    ax.plot(tf[:, 0], tf[:, 1], color="0.75", lw=4, label="true front (x1 = 13)")
    cols = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for (lab, ref), c in zip((("A (1.8 kg, 12 h)", (1.8, 12.0)), ("B (1.6 kg, 18 h)  attainable", (1.6, 18.0)), ("C (1.2 kg, 25 h) beyond the front", (1.2, 25.0)), ("D (2.1 kg, 6 h)  poor trade-off", (2.1, 6.0))), cols):
        r = A.load(35).run(P14, seed=0, max_evals=10000, ref_points=[ref]); N = P14.natural(r.F)
        ax.scatter(N[:, 0], N[:, 1], s=9, c=c, label=lab); ax.scatter(*ref, marker="*", s=120, c=c, edgecolor="k", zorder=5)
    ax.set_xlabel("weight [kg]"); ax.set_ylabel("runtime [h]"); ax.legend(fontsize=6); ax.grid(alpha=0.3); ax.set_title("R-NSGA-II: stars = reference points, dots = final population", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E_rnsga.png")); plt.close(fig)


if __name__ == "__main__":
    for f in (classical_p3, classical_convex, mcdm_plot, premature_plot, constraints_plot, p4_plot, p8_plot, p9_plot, sensitivity_panels, p10_plot, rnsga_plot):
        try:
            f(); print("ok", f.__name__, flush=True)
        except Exception as e:
            print("FAILED", f.__name__, type(e).__name__, e, flush=True)
