"""Report front matter: title page, conventions, problem formulations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl import *
import numpy as np
import problems as pb, algorithms as A


def title_page(test_summary):
    out = [Spacer(1, 25 * mm), P("Lab Assignment 7", "title"),
           P("Multiobjective Optimization: 37 Algorithms, Trade-offs and Applications", "h1"),
           P("Optimization Techniques Lab &middot; based on Chapter 5 (Parts 1-4 and Bonus Module 5E)", "pl"), Spacer(1, 8 * mm)]
    rows = [["Roll number R", "77"],
            ["Deep-dive algorithms (Task C)", "Algorithm 1 = ((77-1) mod 37)+1 = <b>3 Benson's Method</b><br/>"
                                              "Algorithm 2 = ((77+11) mod 37)+1 = <b>15 NSGA-III</b><br/>"
                                              "Algorithm 3 = ((77+23) mod 37)+1 = <b>27 MODE</b>"],
            ["Language / libraries", "Python 3.13, NumPy + Matplotlib + standard library only for every algorithm, baseline and metric. "
                                     "scikit-learn is used ONLY to load the Wisconsin breast-cancer data (P9); reportlab only builds this PDF."],
            ["Seeds", "Task A: seed 0.  Tasks C and D: seeds 0-9 (10 independent runs, mean &plusmn; std).  Reference-front builds: seeds 0-4."],
            ["Budget", "10,000 function evaluations for every population / swarm algorithm (classical sweeps: 11-21 settings, evaluations reported)."],
            ["Unit tests", test_summary],
            ["Code", "moo_utils.py, problems.py, solvers.py, ops.py, mtl_net.py, algorithms/alg01..alg37, tests/, experiments/, handtrace/, README.md"]]
    out.append(tbl([["Item", "Value"]] + rows, [48, 128], True))
    out += [Spacer(1, 6 * mm), P("<b>Honesty note.</b> Everything reported here was produced by the code in the accompanying zip; the raw numbers behind every table and plot are in "
                                "<i>results/*.json</i>. Where an algorithm only partly works, or where our implementation deliberately differs from a lecture formula, "
                                "this is stated next to the result (see the implementation log in Section 3 and the list of deviations in Appendix B).", "note")]
    return out


def conventions():
    o = [H1("1  Conventions, shared toolkit and verification")]
    o.append(P("<b>Minimisation everywhere.</b> A maximised objective f is stored as -f (Max f = Min -f); <i>Problem.natural()</i> converts back for plots and tables. "
               "<b>Normalisation.</b> Before computing hypervolume (HV), IGD or Spacing every front is mapped with (f - ideal)/(nadir - ideal), using for each problem the ideal "
               "and nadir of its true / best-known front (so the same scale is used for all algorithms). The HV reference point is 1.1 on every normalised axis. "
               "<b>Metrics.</b> HV: exact sweep for 2 objectives, a fixed-seed Monte-Carlo estimate (200,000 samples) for 3 or more; IGD: mean distance from points of the true / "
               "reference front to the nearest obtained point; Spacing: Schott's standard deviation of nearest-neighbour distances. IGD on P4, P5, P6 uses a dense-sampling front, on "
               "P8 and P9 the pooled best-known front of long runs (marked * in tables; P8 has only 10 non-dominated tours, P9 only 7).", "p"))
    o.append(P("<b>Shared toolkit (moo_utils.py, Task A1).</b> dominance test, Deb's constrained dominance, vectorised dominance matrix, fast non-dominated sorting (n<sub>p</sub>/S<sub>p</sub>), "
               "crowding distance, normalisation, 2-D hypervolume (+ Monte-Carlo, + 2-D/MC hypervolume contributions), IGD, GD, Spacing, Das-Dennis reference points. "
               "They are tested on the lecture examples: the car Pareto-front table (A, B, C, D non-dominated; E, F dominated), Deb's constrained-dominance table "
               "(A, D feasible; B beats C by smaller violation), the crowding-distance table of Part 4 (1.00, 0.70, 1.00), and the 5E metric example "
               "(HV = 0.7283 / 0.7234, IGD &asymp; 0.077). Our own simplex LP, Nelder-Mead and integer solvers (solvers.py) serve the classical methods, so no scipy call is made by any algorithm.", "p"))
    o.append(P("<b>Verification summary.</b> tests/test_moo_utils.py (9 tests), test_problems.py (13: every 'sanity check' number of Section 4), test_solvers.py (5), "
               "test_algorithms.py (<b>37 tests, one per algorithm</b>, each reproduces a number printed on a lecture slide: e.g. Weighted-Sum x* = 5 and 3.8, "
               "eps-constraint x* = 2 and 3, TOPSIS C<sub>i</sub> = 0.466/0.459/0.541/0.629, VIKOR Q = 0.607/1.0/0.571/0, IBEA fitness -4.05 ... -4.87, "
               "SMS-EMOA total HV 9.10, MGDA x<sub>1</sub> = (4.6, 4.6), GradNorm w = (1.063, 0.937), ...), and test_handtraces.py (3 tests: the deep-dive hand traces of Task C). "
               "All 67 tests pass (results/test_report.txt).", "p"))
    o.append(P("<b>Slide discrepancies found while testing</b> (kept, not hidden): (i) 5E slide 15 quotes an SMS-EMOA contribution of 0.70 for L2 but the exclusive rectangle is "
               "0.3 x 3 = 0.90 (the other five contributions and the total 9.10 are right); (ii) Part 2 slide 40: x = 2 is dominated by x = 0 <i>and</i> by x = 1, so its MOGA rank is 3, not 2; "
               "(iii) the 5E IBEA fitness formula F(i) = &Sigma;<sub>j</sub> -exp(-I(i,j)/&kappa;) (delete the largest) reproduces the slide numbers exactly but is the 'dual' of the original paper's "
               "F(x) = &Sigma;<sub>y</sub> -exp(-I(y,x)/&kappa;) (delete the smallest); the lecture convention collapses to a single point on ZDT2 in our runs (Table B-IBEA), so the default is the paper's.", "small"))
    return o


PROB_TEXT = {
 "P1": dict(form="min F(x) = [ (x-3)<sup>2</sup>, (x-7)<sup>2</sup> ]  s.t. 1 &le; x &le; 10", enc="real scalar x", cons="bounds only (clip)",
            note="Pareto set 3&le;x&le;7; convex front f<sub>2</sub> = (4-&radic;f<sub>1</sub>)<sup>2</sup>. Checked: x=5 -> (4,4); x=3 -> (0,16)."),
 "P2": dict(form="min F(x) = [ 100/x, x<sup>2</sup> ]  s.t. 1 &le; x &le; 10  (real, and integer)", enc="real scalar, or integer (rounded after variation)", cons="bounds only",
            note="every x is Pareto optimal; front f<sub>2</sub> = 10000/f<sub>1</sub><sup>2</sup>. Checked: x=2 -> (50,4); x=10 -> (10,100). Integer: 10 distinct points."),
 "P3": dict(form="ZDT2: f<sub>1</sub> = x<sub>1</sub>, g = 1 + &Sigma;<sub>i=2..10</sub> x<sub>i</sub>, f<sub>2</sub> = g (1 - (f<sub>1</sub>/g)<sup>2</sup>); x &isin; [0,1]<sup>10</sup>", enc="real vector, n = 10",
            cons="bounds only", note="non-convex front f<sub>2</sub> = 1 - f<sub>1</sub><sup>2</sup> (x<sub>2..10</sub> = 0)."),
 "P4": dict(form="min [ w'&Sigma;w , -&mu;'w ]  s.t. &Sigma;w<sub>i</sub> = 1, w &ge; 0  (4 assets, data of Section 4)", enc="x &isin; [0,1]<sup>4</sup>; w = x / &Sigma;x (repair)",
            cons="equality by <b>repair</b> (w = x/&Sigma;x): satisfied exactly, no evaluation is wasted and the search space keeps its dimension. The penalty variant (CV = |&Sigma;x - 1|, tolerance 1e-3) is compared in Section 4 evidence.",
            note="Checked: w=(0,0,1,0) -> risk 0.09, return 0.15. Min-risk end = mixture (risk 0.00704, return 0.0926)."),
 "P5": dict(form="min [ 7850 L b h , 4 P L<sup>3</sup>/(E b h<sup>3</sup>) ]  s.t. b h<sup>2</sup> &ge; 2.4e-5;  b &isin; [0.01,0.10], h &isin; [0.02,0.20]", enc="real vector (b, h)",
            cons="relative violation CV = max(0, (2.4e-5 - b h<sup>2</sup>)/2.4e-5); Deb's rules (alg 21) or penalty rho&middot;CV&middot;(nadir-ideal) (default for the others)",
            note="Checked: (0.01, 0.05) -> mass 3.925 kg, deflection 0.016 m, stress 240 MPa, feasible."),
 "P6": dict(form="Binh-Korn: min [ 4x<sub>1</sub><sup>2</sup>+4x<sub>2</sub><sup>2</sup>, (x<sub>1</sub>-5)<sup>2</sup>+(x<sub>2</sub>-5)<sup>2</sup> ] s.t. (x<sub>1</sub>-5)<sup>2</sup>+x<sub>2</sub><sup>2</sup> &le; 25, (x<sub>1</sub>-8)<sup>2</sup>+(x<sub>2</sub>+3)<sup>2</sup> &ge; 7.7; x<sub>1</sub>&isin;[0,5], x<sub>2</sub>&isin;[0,3]",
            enc="real vector", cons="CV = max(0,g<sub>1</sub>) + max(0,g<sub>2</sub>); Deb's rules / penalty / none (experiment)", note="Checked: (0,0) -> (0,50), (5,3) -> (136,4), both feasible."),
 "P7": dict(form="max [0.8x<sub>1</sub>, 1.2x<sub>2</sub>] s.t. x<sub>1</sub>+x<sub>2</sub> &le; 90, x<sub>1</sub> &ge; 10, x<sub>2</sub> &ge; 20  (stored as min of the negatives)", enc="real vector",
            cons="bounds + CV = max(0, x<sub>1</sub>+x<sub>2</sub>-90)", note="front = the line x<sub>1</sub>+x<sub>2</sub>=90: (10,80) -> (8,96); (70,20) -> (56,24)."),
 "P8": dict(form="bi-objective TSP, 15 cities (numpy seed 7), min total distance and total time t<sub>ij</sub> = d<sub>ij</sub> u<sub>ij</sub>, u<sub>ij</sub> ~ U(0.5,2)", enc="permutation of 0..14 (closed tour)",
            cons="every city once -- guaranteed by construction (OX crossover, swap / inversion mutation; ants build permutations)", note="coordinates printed in Section 4; nearest-neighbour tours used as baselines."),
 "P9": dict(form="min [ 1 - accuracy(5-NN, 5-fold CV), #features ] over 0/1 masks of the 30 standardised features", enc="binary mask, repair keeps &ge; 1 bit",
            cons="at least one feature (CV = 1 if empty, repaired by the operators)", note="own 5-NN and own 5-fold CV (fixed fold assignment); all-ones baseline error 0.0369 (96.3 % accuracy)."),
 "P10": dict(form="DTLZ2, M = 3 and 5, n = M+9: g = &Sigma;<sub>i&ge;M</sub>(x<sub>i</sub>-0.5)<sup>2</sup>, f<sub>m</sub> = (1+g) &Pi; cos(x<sub>i</sub>&pi;/2) sin(x<sub>M-m+1</sub>&pi;/2)", enc="real vector",
            cons="bounds only", note="g = 0 &rArr; &Sigma; f<sub>m</sub><sup>2</sup> = 1 (checked for M = 3, 5); true front = Das-Dennis points projected on the unit sphere."),
 "P11": dict(form="max [4A+6B, A+B] s.t. 2A+3B &le; 1200, 4A+2B &le; 1600, A,B &ge; 0 (goals: profit &ge; 2400, units &ge; 600; priority: profit then units)", enc="real vector",
            cons="relative violations summed; LP data used by goal programming / lexicographic", note="corners (0,0),(400,0),(300,200),(0,400); profit 2400 on the whole edge (300,200)-(0,400); units max 500 at (300,200)."),
 "P12": dict(form="5 suppliers x 4 criteria (cost min, quality max, delivery min, reliability max), weights (0.3,0.3,0.2,0.2)", enc="decision matrix", cons="-", note="ranking problem for TOPSIS / VIKOR (algorithms 8, 9)."),
 "P13": dict(form="(a) f<sub>1</sub>=|x-a|<sup>2</sup>, f<sub>2</sub>=|x-b|<sup>2</sup>, a=(3,3), b=(-3,-3), start (0,5);  (b) two-head network (classification BCE + regression MSE) on a shared 2-layer tanh body, hand-coded back-propagation",
            enc="x &isin; R<sup>2</sup>;  network weights", cons="none", note="Pareto set of (a) = segment [a,b]; gradient check of (b) agrees with finite differences to 4e-7."),
 "P14": dict(form="min weight 0.4+0.05x<sub>1</sub>+0.012x<sub>2</sub>, max runtime 0.2x<sub>2</sub>/(0.5+0.03x<sub>1</sub>); x<sub>1</sub>&isin;[13,17], x<sub>2</sub>&isin;[40,90]", enc="real vector",
            cons="bounds only", note="Checked: (13,40) -> 1.53 kg, 8.99 h; (13,90) -> 2.13 kg, 20.2 h; all Pareto points have x<sub>1</sub>=13."),
}


def bounds_text(key):
    try:
        p = pb.make(key)
        ide, nad = p.ideal, p.nadir
        nat_i = p.natural(ide); nat_n = p.natural(nad)
        return f"ideal {np.round(nat_i, 4).tolist()}, nadir {np.round(nat_n, 4).tolist()} (natural units)"
    except Exception as e:
        return "-"


def problem_formulations():
    o = [H1("2  Problem bank: one-page formulations"),
         P("For every problem used in this report the table states the standard form, the encoding of a solution, how constraints are handled, how objectives are normalised, "
           "the hypervolume reference point and the termination rule. All objectives are converted to minimisation before dominance is used.", "p")]
    keys = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14"]
    for k in keys:
        t = PROB_TEXT[k]
        pk = {"P10": "P10_M3", "P13": "P13a", "P12": None}.get(k, k)
        norm = "-" if pk is None else bounds_text(pk)
        if k == "P10":
            norm = "ideal 0, nadir 1 on every axis (M = 3 and M = 5)"
        hv = "-" if k in ("P12",) else "(1.1, ..., 1.1) in normalised space (the same for every algorithm)"
        term = {"P11": "LP optimum (exact)", "P12": "n/a (ranking)", "P13": "300 gradient steps (a), 500 epochs (b)"}.get(k, "10,000 evaluations for EAs / swarms; end of the sweep for classical methods")
        rows = [["Standard form", t["form"]], ["Encoding", t["enc"]], ["Constraints", t["cons"]], ["Normalisation", norm],
                ["HV reference point", hv], ["Termination", term], ["Sanity check / remark", t["note"]]]
        o.append(KeepTogether([H3(k + " " + {"P1": "Factory production speed", "P2": "Delivery fleet sizing", "P3": "ZDT2 benchmark", "P4": "Investment portfolio",
                                           "P5": "Cantilever beam", "P6": "Binh-Korn", "P7": "Water allocation", "P8": "Bi-objective TSP", "P9": "Feature selection",
                                           "P10": "DTLZ2 many-objective", "P11": "Production planning", "P12": "Supplier selection", "P13": "Multi-task learning",
                                           "P14": "Laptop design"}[k]),
                               tbl([["Item", "Specification"]] + rows, [32, 144], True)]))
        o.append(Spacer(1, 2 * mm))
    p8 = pb.make("P8")
    o.append(P("<b>P8 coordinates used (numpy.random.seed(7), 15 x 2, scaled to [0,100]):</b> " +
               "; ".join(f"{i}:({x:.1f},{y:.1f})" for i, (x, y) in enumerate(p8.xy)) +
               ". Symmetric u<sub>ij</sub> ~ U(0.5,2.0) drawn from the same stream (upper triangle).", "small"))
    return o
