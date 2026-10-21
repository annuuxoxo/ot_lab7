"""Task A for the algorithms that do not return a front by themselves: TOPSIS (8), VIKOR (9), MGDA (36), GradNorm (37)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mtl_net
from algorithms import alg08_topsis as tp, alg09_vikor as vk, alg36_mgda as mg, alg37_gradnorm as gn

out = {}
P12 = pb.P12
# ---------------- 8 / 9 : P12 and the NSGA-II front of P4
P4 = pb.make("P4"); front = A.load(14).run(P4, seed=0, max_evals=10000)
M = P4.natural(front.F); mn = np.array([True, False]); w = np.array([0.5, 0.5])
C, rk = tp.topsis(M, mn, w); o = vk.vikor(M, mn, w, 0.5)
out["topsis_p12"] = dict(C=tp.topsis(P12.matrix, P12.minimize, P12.weights)[0].tolist())
out["P4_front_topsis_top3"] = M[rk[:3]].tolist(); out["P4_front_vikor_top3"] = M[o["ranking"][:3]].tolist()
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4), dpi=110)
Cp, rkp = tp.topsis(P12.matrix, P12.minimize, P12.weights)
axs[0].bar(P12.suppliers, Cp, color="#4c78a8"); axs[0].set_title("TOPSIS closeness C_i, P12 (higher = better)", fontsize=9); axs[0].set_ylabel("C_i")
ordx = np.argsort(M[:, 0])
for ax, rank, ttl in ((axs[1], rk, "TOPSIS ranking of the NSGA-II front of P4"), (axs[2], o["ranking"], "VIKOR (v=0.5) ranking of the same front")):
    ax.plot(M[ordx, 0], M[ordx, 1], "-", color="0.7", lw=2); ax.scatter(M[:, 0], M[:, 1], s=8, c="0.5")
    for k, (c, lab) in enumerate(zip(["#d62728", "#ff7f0e", "#2ca02c"], ["rank 1", "rank 2", "rank 3"])):
        ax.scatter(M[rank[k], 0], M[rank[k], 1], s=70, c=c, label=lab, zorder=4, edgecolor="k")
    ax.set_xlabel("risk w'Sw"); ax.set_ylabel("return"); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "A_08_09_TOPSIS_VIKOR.png")); plt.close(fig)
# ---------------- 36 / 37 : P13a toy
P = pb.make("P13a")
r36 = mg.run(P, seed=0); r37 = gn.run(P, seed=0, steps=300)
x = np.array([0., 5.]); fx = []
for _ in range(300):
    x = x - 0.05 * P.grad(x).sum(0)
fixed_end = x.copy()
out["toy"] = dict(mgda_end_from_0_5=r36.X[0].tolist(), gradnorm_end_from_0_5=r37.X[0].tolist(), fixed_end=fixed_end.tolist(),
                  mgda_dist_to_segment=float(np.abs(r36.X[:, 0] - r36.X[:, 1]).max()))
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5), dpi=110)
ax = axs[0]
ax.plot([-3, 3], [-3, 3], "-", color="0.6", lw=4, label="Pareto set [a,b]")
for t in r36.extra["traj"][:, :4].transpose(1, 0, 2): ax.plot(t[:, 0], t[:, 1], "-", lw=1)
ax.scatter(*P.a, c="k", marker="*", s=90); ax.scatter(*P.b, c="k", marker="*", s=90); ax.text(3.1, 3.1, "a", fontsize=8); ax.text(-3.4, -3.4, "b", fontsize=8)
ax.scatter(r36.X[:, 0], r36.X[:, 1], s=12, c="#d95f02", zorder=4, label="MGDA end points"); ax.scatter(*fixed_end, c="g", s=60, marker="s", label="fixed w=(1,1) end", zorder=5)
ax.set_title("MGDA trajectories on P13(a)", fontsize=9); ax.legend(fontsize=6.5); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax = axs[1]; tf = P.true_front(200); ax.plot(tf[:, 0], tf[:, 1], color="0.7", lw=3, label="true front")
ax.scatter(r36.F[:, 0], r36.F[:, 1], c="#d95f02", s=14, label="MGDA (11 starts)"); ax.scatter(r37.F[:, 0], r37.F[:, 1], c="#1b9e77", s=14, marker="x", label="GradNorm (11 starts)")
ax.set_xlabel("f1"); ax.set_ylabel("f2"); ax.set_title("objective space", fontsize=9); ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axs[2]
for meth, c in (("fixed", "k"), ("mgda", "#d95f02"), ("gradnorm", "#1b9e77")):
    h = mtl_net.train(meth, steps=500, reg_scale=1.0, seed=0)
    ax.plot(h["Lc"], c=c, ls="-", label=f"{meth} L_cls"); ax.plot(h["Lr"], c=c, ls="--", label=f"{meth} L_reg")
    out[f"net_{meth}_scale1"] = dict(Lc=float(h["Lc"][-1]), Lr=float(h["Lr"][-1]))
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title("two-head network (P13b), equal-size tasks", fontsize=9); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "A_36_37_MGDA_GradNorm.png")); plt.close(fig)
save_json(out, "taskA_special.json")
print(json.dumps(out, indent=1)[:1500])
