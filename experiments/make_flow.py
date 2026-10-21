"""Flow diagrams for the three study cards (Task C)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import FIG
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def flow(steps, fname, title, loop=None, colors=None):
    n = len(steps)
    fig, ax = plt.subplots(figsize=(11, 1.7 + (0.4 if loop else 0)), dpi=130)
    ax.set_xlim(0, n * 2.2); ax.set_ylim(-0.9 if loop else -0.2, 1.6); ax.axis("off")
    for i, s in enumerate(steps):
        x = i * 2.2 + 0.05
        c = (colors or ["#dfe7f3"] * n)[i]
        ax.add_patch(FancyBboxPatch((x, 0.2), 1.9, 1.1, boxstyle="round,pad=0.03", fc=c, ec="#1f3b66", lw=1.1))
        ax.text(x + 0.95, 0.75, s, ha="center", va="center", fontsize=7.3, wrap=True)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + 1.93, 0.75), (x + 2.2, 0.75), arrowstyle="-|>", mutation_scale=11, color="#1f3b66"))
    if loop:
        ax.add_patch(FancyArrowPatch((n * 2.2 - 0.4, 0.15), (0.5, 0.15), connectionstyle="arc3,rad=-0.0", arrowstyle="-|>", mutation_scale=10,
                                     color="#c0392b", linestyle="--", lw=1.1))
        ax.text(n * 1.1, -0.2, loop, ha="center", fontsize=7.5, color="#c0392b")
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)


flow(["reference point z*<br>and weights &lambda;".replace("<br>", "\n"), "achievements\ns_i = z*_i - f_i(x)", "constraints\ns_i(x) >= 0",
      "maximise\nSum lambda_i s_i(x)\n(one NLP / LP solve)", "one efficient\npoint x*", "change (z*, lambda)\nand re-solve"],
     "C_flow_alg03.png", "Algorithm 3 - Benson's method (reference-point form)", loop="sweep: new lambda (convex part) or new z* (also reaches dents)")
flow(["Das-Dennis\nreference points\nH = C(k+p-1,p)", "merge P u Q,\nnon-dominated sort\nF1, F2, ...", "fill whole fronts\nuntil |S_t| >= N", "normalise: ideal,\nASF extreme points,\nintercepts",
      "associate each\nmember with nearest\nreference LINE", "niche counts rho_j;\nfill last front from\nsparsest niches", "next generation\nP (size N)"],
     "C_flow_alg15.png", "Algorithm 15 - NSGA-III", loop="SBX + polynomial mutation of random pairs -> offspring Q")
flow(["population X_i\n+ archive", "mutation\nV = X_r1 + F (X_r2 - X_r3)", "binomial crossover\nU_j = V_j if rand<=CR\nelse X_ij", "clip to bounds,\nevaluate U",
      "U dominates X_i ?\nyes: replace\nno: keep X_i", "non-dominated U\n-> archive\n(crowding trim)"],
     "C_flow_alg27.png", "Algorithm 27 - MODE", loop="next generation: every target gets a new trial vector")
print("flows done")
