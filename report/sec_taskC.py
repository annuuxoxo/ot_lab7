"""Task C: study cards, hand traces and sensitivity studies for the algorithms assigned to roll number 77 (3, 15, 27)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl import *
import numpy as np

RES = os.path.join(ROOT, "results")


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def card_table(rows):
    return tbl([["Item", "Content"]] + rows, [34, 142], True)


def study_card_3():
    rows = [
     ["Full form, year, authors", "<b>Benson's method</b> (H. P. Benson, 1978, J. Optim. Theory Appl. 26 &mdash; an efficiency-testing / efficient-point-generating scalarisation). "
                                  "The lecture (Part 1, slides 36-37) presents the reference-point form used here."],
     ["Problem it was invented to fix", "Weighted sums only reach the convex-hull part of a front and give no control over WHERE on the front the answer lies; "
                                        "a decision maker usually has a target (reference) in mind. Benson's construction asks: how far can we go <i>past</i> the target in every objective at once?"],
     ["Core idea (5 lines)", "1. Pick a reference point z* in objective units (a target for each objective).<br/>2. The achievement s<sub>i</sub> = z*<sub>i</sub> - f<sub>i</sub>(x) is the distance by which "
                             "x beats the target on objective i (all objectives minimised).<br/>3. Only x with every s<sub>i</sub> &ge; 0 are admissible.<br/>4. Among those, "
                             "maximise the weighted total &Sigma; &lambda;<sub>i</sub> s<sub>i</sub> &mdash; one single-objective solve.<br/>5. Change (z*, &lambda;) and repeat to obtain more efficient points."],
     ["Flow diagram", img("C_flow_alg03.png", 172)],
     ["Numbered steps and formulas", "1. Choose z* &isin; R<sup>k</sup> and &lambda; &isin; R<sup>k</sup><sub>+</sub>.<br/>2. s<sub>i</sub>(x) = z*<sub>i</sub> - f<sub>i</sub>(x) (minimised f<sub>i</sub>); "
                                     "s<sub>i</sub>(x) = f<sub>i</sub>(x) - z*<sub>i</sub> (maximised f<sub>i</sub>).<br/>3. Solve  max<sub>x</sub> &Sigma;<sub>i</sub> &lambda;<sub>i</sub> s<sub>i</sub>(x)  "
                                     "s.t.  s<sub>i</sub>(x) &ge; 0 (i = 1..k), g(x) &le; 0, bounds.<br/>4. Report x*, f(x*), s(x*). If no x satisfies all s<sub>i</sub> &ge; 0 the setting is infeasible "
                                     "(z* is better than anything attainable).<br/>5. New (z*, &lambda;) &rarr; step 2.   Equivalent form: min &Sigma;&lambda;<sub>i</sub> f<sub>i</sub> s.t. f<sub>i</sub> &le; z*<sub>i</sub>."],
     ["Parameters", "<b>z*</b> (reference point): too <i>loose</i> (e.g. the nadir) &rArr; the s<sub>i</sub> &ge; 0 constraints never bind and the method collapses to a weighted sum; too <i>tight</i> (better than "
                    "the ideal) &rArr; empty feasible region, no answer; exactly on the front &rArr; the only admissible point is z* itself.  <b>&lambda;</b>: tilts the optimum inside the admissible box; "
                    "too extreme (&lambda;<sub>i</sub> &rarr; 0) &rArr; objective i is only kept at its target s<sub>i</sub> = 0.  <b>sub-solver</b> tolerance and restarts (our Nelder-Mead with exact L1 penalty)."],
     ["Convergence / diversity / constraints", "Each solve is exact if the sub-problem is convex; the quality of the FRONT is only as good as the sweep of (z*, &lambda;): diversity is under the user's control, not automatic. "
                                               "Constraints g(x) &le; 0 enter the sub-problem directly."],
     ["Non-convex fronts", "&lambda;-sweeps alone behave like a weighted sum (only the convex hull / end points). Moving z* makes the box constraint bind <i>on the front itself</i> and reaches the dent "
                           "(see the sensitivity table in C3: the z* sweep places points in the dent)."],
     ["More than three objectives", "Nothing in the formulation limits k, but a good sweep needs k-1 free dimensions of z* and of &lambda;: the number of solves grows exponentially; also no way to visualise."],
     ["Complexity", "One constrained single-objective solve per (z*, &lambda;) pair; for linear data a simplex LP (polynomial in practice); in our code O(maxfev) function evaluations per setting "
                    "(&asymp; 2,000-27,000 evaluations on P1/P3)."],
    ]
    return card_table(rows)


def study_card_15():
    rows = [
     ["Full form, year, authors", "<b>Non-dominated Sorting Genetic Algorithm III</b> &mdash; K. Deb and H. Jain, 2014, IEEE Trans. Evolutionary Computation 18(4):577-601 (the lecture: Part 3 slides 12-19 and 5E slides 5-7)."],
     ["Problem it was invented to fix", "NSGA-II breaks down for k &ge; 4 objectives: almost every individual of a random population is non-dominated, so Pareto rank no longer discriminates, "
                                        "and crowding distance (designed for 2-3 objectives) neither spreads the solutions nor pushes them to the front."],
     ["Core idea (5 lines)", "Keep NSGA-II's elitist non-dominated sorting, but replace crowding distance by <b>niching around a set of structured reference points</b> spread on the normalised unit simplex. "
                             "Objectives are normalised every generation. Each candidate is associated with the closest reference LINE; the last front is filled from the least-crowded reference points. "
                             "Diversity is therefore built into the geometry and does not degrade with k."],
     ["Flow diagram", img("C_flow_alg15.png", 172)],
     ["Numbered steps and formulas", "1. H = C(k+p-1, p) Das-Dennis points w with &Sigma;w<sub>i</sub> = 1, step 1/p.   2. N = H rounded up to a multiple of 4; random initial population.<br/>"
        "3. Offspring Q (SBX + polynomial mutation of random pairs); R = P &cup; Q; non-dominated sort F<sub>1</sub>, F<sub>2</sub>, ...; S<sub>t</sub> = F<sub>1</sub> &cup; ... &cup; F<sub>l</sub> with |S<sub>t</sub>| &ge; N.<br/>"
        "4. Normalise: z<sub>min</sub>,<sub>i</sub> = min f<sub>i</sub>; f' = f - z<sub>min</sub>; extreme point j = argmin ASF<sub>j</sub>, ASF<sub>j</sub>(x) = max<sub>i</sub> f'<sub>i</sub>/w<sub>ji</sub> "
        "(w<sub>jj</sub> = 1, others 10<sup>-6</sup>); intercepts a = 1 / (E<sup>-1</sup> 1) from the hyperplane through the extreme points (fallback: max f'); f<sup>n</sup> = f'/a.<br/>"
        "5. d<sub>&perp;</sub>(s, w) = || f<sup>n</sup> - (w'f<sup>n</sup> / ||w||<sup>2</sup>) w ||; associate s with the w of smallest d<sub>&perp;</sub>.<br/>"
        "6. &rho;<sub>j</sub> = #members of F<sub>1..l-1</sub> associated with w<sub>j</sub>. Repeat K = N - |F<sub>1..l-1</sub>| times: take j with minimum &rho;<sub>j</sub> (random ties); candidates = "
        "members of F<sub>l</sub> associated with j; none &rArr; discard j; &rho;<sub>j</sub> = 0 &rArr; take the closest to the line, else a random one; &rho;<sub>j</sub> += 1."],
     ["Parameters", "<b>p</b> (divisions) sets H and N: too small &rArr; few reference points, coarse front (p = 2, H = 15: 16 solutions); too large &rArr; N huge (p = 6, H = 210, N = 212) so at a fixed budget "
                    "there are only 47 generations and convergence suffers (Task C3 numbers).  p<sub>c</sub> = 1.0, &eta;<sub>c</sub> = 30 (small &eta;<sub>c</sub> &rArr; large jumps / disruption, large &eta;<sub>c</sub> &rArr; "
                    "offspring stick to parents), p<sub>m</sub> = 1/n, &eta;<sub>m</sub> = 20. Two-layer reference sets are advised from k = 8."],
     ["Convergence / diversity / constraints", "Convergence from elitist non-dominated sorting; diversity from niching; constraints by Deb's constrained domination in the sort (we use the penalty S = F + &rho;CV&middot;scale)."],
     ["Non-convex fronts", "No convexity assumption: reference lines are directions, not weights, so a dented or concave front is covered. Irregular (degenerate, disconnected) fronts waste reference points whose niches stay empty."],
     ["More than three objectives", "Designed for it: H and the normalisation scale with k, not with n. The price is N &asymp; H (210 reference points for k = 5, p = 6)."],
     ["Complexity", "Sorting O(N<sup>2</sup> k) + association O(N H k) + niching O(N H) per generation; memory O(N H)."],
    ]
    return card_table(rows)


def study_card_27():
    rows = [
     ["Full form, year, authors", "<b>Multi-Objective Differential Evolution</b> &mdash; Xue, Sanderson &amp; Graves, 2003 (lecture Part 4, slide 48), built on Differential Evolution (R. Storn &amp; K. Price, 1997)."],
     ["Problem it was invented to fix", "Differential evolution is a very fast optimiser for continuous <i>single</i>-objective problems, but its greedy 'trial replaces target if better' rule needs a definition of 'better' "
                                        "when there are several objectives. MODE substitutes Pareto dominance and keeps the non-dominated trials in an archive."],
     ["Core idea (5 lines)", "Each member X<sub>i</sub> is a <i>target</i>. A <i>mutant</i> V = X<sub>r1</sub> + F (X<sub>r2</sub> - X<sub>r3</sub>) uses the scaled difference of two random members as a self-adapting step; "
                             "binomial crossover mixes V with X<sub>i</sub> into a trial U. The trial replaces X<sub>i</sub> only if it Pareto-dominates it; otherwise X<sub>i</sub> survives. "
                             "Every non-dominated trial is offered to an external archive that holds the answer."],
     ["Flow diagram", img("C_flow_alg27.png", 172)],
     ["Numbered steps and formulas", "1. Random population X<sub>1..NP</sub>, archive A = non-dominated(X).<br/>2. For each i draw distinct r1, r2, r3 &ne; i:  V = X<sub>r1</sub> + F (X<sub>r2</sub> - X<sub>r3</sub>).<br/>"
        "3. Draw j<sub>rand</sub>; U<sub>j</sub> = V<sub>j</sub> if rand<sub>j</sub> &le; CR or j = j<sub>rand</sub>, else X<sub>i,j</sub>; clip U to [lb, ub].<br/>4. Evaluate U. "
        "If U &prec; X<sub>i</sub> (constrained dominance when constrained) then X<sub>i</sub> &larr; U, else keep X<sub>i</sub>.<br/>5. A &larr; non-dominated(A &cup; {U}) with crowding-distance truncation to 100.<br/>"
        "6. Repeat; output A."],
     ["Parameters", "<b>F</b> (scale factor): too small (0.2) &rArr; tiny steps, slow but the population stays spread; too large (0.9) &rArr; mutants overshoot, steps destroy good structure (see sensitivity). "
                    "<b>CR</b>: small &rArr; trials differ from the target in few coordinates (good for separable problems), 1.0 &rArr; trial = mutant (rotation invariant). <b>NP</b>: too small &rArr; "
                    "few distinct differences, premature convergence; too large &rArr; slow. Archive size: truncation by crowding distance."],
     ["Convergence / diversity / constraints", "Very fast convergence on smooth continuous problems. Diversity comes ONLY from the archive and the initial population, because the dominance-only replacement rule "
                                               "has no spreading mechanism. Constraints: Deb's rules or penalty in the dominance test."],
     ["Non-convex fronts", "Dominance handles any shape in principle, but on ZDT2 the lecture rule collapsed to f<sub>1</sub> = 0 (all trials with lower f<sub>1</sub> and lower g dominate; trials that trade f<sub>1</sub> for f<sub>2</sub> are rejected)."],
     ["More than three objectives", "A trial is dominating less and less often as k grows (mostly mutually non-dominated), so replacements stop and the archive grows without pressure: not suited without an indicator or reference-point selection."],
     ["Complexity", "O(NP n) per generation for variation + O(NP k) for the dominance tests + O((NP + |A|)<sup>2</sup> k) for the archive update; memory O(NP + |A|)."],
    ]
    return card_table(rows)


# ------------------------------------------------------------------------------------------ hand traces
def trace_benson(h):
    rows = [["Setting", "z* , &lambda;", "objective &Phi;(x) = &Sigma; &lambda;<sub>i</sub> (z*<sub>i</sub> - f<sub>i</sub>)", "admissible x (s<sub>i</sub> &ge; 0)", "optimum", "f(x*), s(x*), &Phi;"]]
    text = [
        ("z* = (16,16), &lambda; = (0.5,0.5)", "&Phi; = 0.5(16-(x-3)<sup>2</sup>) + 0.5(16-(x-7)<sup>2</sup>) = 16 - (x<sup>2</sup> - 10x + 29)<br/>&Phi;' = -2x + 10", "(x-3)<sup>2</sup> &le; 16 &rArr; -1 &le; x &le; 7;  (x-7)<sup>2</sup> &le; 16 &rArr; 3 &le; x &le; 11;  box [1,10] &rArr; <b>[3,7]</b>",
         "&Phi;' = 0 at x = 5 (inside)"),
        ("z* = (16,16), &lambda; = (0.8,0.2)", "&Phi; = 16 - (0.8(x-3)<sup>2</sup> + 0.2(x-7)<sup>2</sup>) = 16 - (x<sup>2</sup> - 7.6x + 17)<br/>&Phi;' = -2x + 7.6", "same interval [3,7]", "&Phi;' = 0 at x = 3.8 (inside)"),
        ("z* = (4,10), &lambda; = (0.5,0.5)", "&Phi; = 0.5(4-(x-3)<sup>2</sup>) + 0.5(10-(x-7)<sup>2</sup>) = 7 - (x<sup>2</sup> - 10x + 29)<br/>&Phi;' = -2x + 10", "(x-3)<sup>2</sup> &le; 4 &rArr; 1 &le; x &le; 5;  (x-7)<sup>2</sup> &le; 10 &rArr; x &ge; 7-&radic;10 = 3.838 &rArr; <b>[3.838, 5]</b>",
         "&Phi;' &ge; 0 on the whole interval &rArr; optimum at the right end x = 5 where s<sub>1</sub> = 0 (constraint binds)")]
    for t, hh in zip(text, h):
        f = hh["f"]; s = hh["s"]
        rows.append([t[0], "", t[1], t[2], t[3], f"f = ({f[0]:.2f}, {f[1]:.2f})<br/>s = ({s[0]:.2f}, {s[1]:.2f})<br/>&Phi; = {hh['phi']:.2f}"])
    rows = [[r[0], r[2], r[3], r[4], r[5]] for r in rows]
    rows[0] = ["Setting (z*, &lambda;)", "objective and derivative", "admissible interval from s<sub>i</sub> &ge; 0 and the box", "optimum", "result"]
    return tbl(rows, [27, 48, 45, 30, 26], True)


def fmt_vec(v, d=2):
    return "(" + ", ".join(f"{x:.{d}f}" for x in v) + ")"


def trace_nsga3(it, title, parents, offspring):
    names = it["names"]
    rows = [["point", "x = (wingspan, power)", "f<sub>1</sub> weight", "f<sub>2</sub> cost", "f<sub>3</sub> -flight time", "front"]]
    fr = {}
    for k, f in enumerate(it["fronts"]):
        for n in f:
            fr[n] = k + 1
    allpts = {**parents, **offspring}
    for n, F in zip(names, it["F"]):
        rows.append([("<b>%s</b>" % n) + (" (parent)" if n in parents else " (offspring)"), fmt_vec(allpts[n], 2), f"{F[0]:.3f}", f"{F[1]:.1f}", f"{F[2]:.2f}", f"F{fr[n]}"])
    out = [tbl(rows, [27, 38, 24, 24, 30, 15], True)]
    St = it["St"]
    rows = [["member of S<sub>t</sub>", "translated f' = f - ideal", "normalised f<sup>n</sup> = f' / intercepts", "d<sub>&perp;</sub> to w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>", "nearest"]]
    for n in St:
        rows.append([n, fmt_vec(it["translated"][n], 3), fmt_vec(it["normalised"][n], 3), fmt_vec(it["dist"][n], 3), f"w{it['nearest'][n] + 1}"])
    out.append(Spacer(1, 1.5 * mm))
    out.append(P(f"Fronts: {', '.join('F%d = {%s}' % (k + 1, ', '.join(f)) for k, f in enumerate(it['fronts']))}. S<sub>t</sub> = {{{', '.join(St)}}}, "
                 f"{it['n_accepted']} members accepted whole (F<sub>1..l-1</sub>), K = {4 - it['n_accepted']} to be chosen from F<sub>l</sub>. "
                 f"ideal = {fmt_vec(it['ideal'], 3)}, intercepts (max f') = {fmt_vec(it['intercepts'], 3)}.", "small"))
    out.append(tbl(rows, [26, 44, 46, 44, 16], True))
    out.append(Spacer(1, 1.5 * mm))
    out.append(P("<b>Niching:</b> " + "; ".join(it["log"]) + f".  Survivors: <b>{', '.join(it['survivors'])}</b>.", "small"))
    return out


def trace_mode(m):
    out = [P(f"Problem P1 (f<sub>1</sub> = (x-3)<sup>2</sup>, f<sub>2</sub> = (x-7)<sup>2</sup>, 1 &le; x &le; 10), NP = 4, initial population x = {m['pop0']} with f = "
             f"{[tuple(round(a, 2) for a in f) for f in m['F0']]}. Initial archive (non-dominated) = {m['archive0']} (x = 1 is dominated by x = 4, x = 10 by x = 4). "
             "Donor indices r<sub>1</sub>, r<sub>2</sub>, r<sub>3</sub> are fixed (they are random in the algorithm) so the trace is reproducible; one variable &rArr; U = V.", "small")]
    for k, it in enumerate(m["iterations"]):
        rows = [["i", "target x<sub>i</sub> (f)", "r<sub>1</sub>,r<sub>2</sub>,r<sub>3</sub>", "F", "V = x<sub>r1</sub> + F(x<sub>r2</sub> - x<sub>r3</sub>)", "U (clipped)", "f(U)", "outcome"]]
        for r in it["rows"]:
            rows.append([str(r["i"]), f"{r['x']:g} ({r['fX'][0]:g}, {r['fX'][1]:g})", ",".join(map(str, r["r"])), f"{r['F']:g}",
                         f"{r['V']:g}", f"{r['U']:g}", f"({r['fU'][0]:g}, {r['fU'][1]:g})", r["outcome"]])
        out.append(Spacer(1, 1.5 * mm))
        out.append(P(f"<b>Iteration {k + 1}</b> (F = {it['rows'][0]['F']})", "small"))
        out.append(tbl(rows, [6, 27, 15, 8, 40, 16, 24, 40], True))
        out.append(P(f"Population after iteration {k + 1}: x = {it['pop_after']}; archive = {it['archive']}.", "small"))
    return out


def sensitivity(c):
    out = []
    if c is None:
        return [P("(sensitivity results missing)")]
    s = c["summary"]
    def table(alg, label):
        rows = [[label, "HV mean &plusmn; std", "IGD mean &plusmn; std", "#points", "points with 0.3&lt;f<sub>1</sub>&lt;0.7" if alg != "15" else "runtime [s]"]]
        for t, v in s[alg].items():
            rows.append([t, pm(v["hv_mean"], v["hv_std"]), pm(v["igd_mean"], v["igd_std"]), fmt(v["n_mean"], 1), fmt(v["mid_mean"], 1) if alg != "15" else fmt(v["seconds"], 1)])
        return tbl(rows, [32, 40, 40, 24, 36], True)
    return s, table
