"""Task E (algorithm selection), Task F (viva preparation)."""
import values as V
from values import EV, A, D1, D2, D3, run, rank
from rl import *


def f(x, d=3):
    return f"{x:.{d}f}"


def task_e():
    hv3 = lambda a, tag="": D1(a, "P3", "hv", tag, 3, True)
    ig5 = lambda a: D1(a, "P5", "igd", "", 4, True)
    cm = EV["p6_modes"]
    rows = [["problem", "two suitable algorithms", "one poor choice", "justification (convexity, #objectives, variables, constraints, preferences, budget) and evidence"],
     ["<b>P3</b> ZDT2: non-convex, 2 objectives, 10 real variables, no constraints, budget 10,000",
      "29 MOBA, 19 MOEA/D-DE (also 33 SMS-EMOA, 32 IBEA)", "1 Weighted Sum",
      f"The front is concave, so any weighted sum reaches only the end points (Weighted Sum: HV {hv3(1,'WS 21w')} vs {hv3(29)} for MOBA; 8 points, 0 in the middle). Two objectives and a smooth continuous space favour population methods with a good spread "
      f"mechanism; the budget is enough (D1: MOBA rank {rank(29,'P3')[0]}, MOEA/D-DE rank {rank(19,'P3')[0]} of {rank(29,'P3')[1]}). Evidence: Task D1, Fig. E-1."],
     ["<b>P5</b> beam: 2 objectives, 2 real variables, one nonlinear inequality constraint",
      "21 Constrained NSGA-II, 16 SPEA2 (or 27 MODE with survival)", "10 VEGA (or any method with constraints ignored)",
      f"One constraint that excludes {EV['p5_infeasible_fraction']*100:.0f} % of random designs: Deb's rules return only feasible designs (IGD {ig5(21)}). Ignoring the stress limit lowers the 'optimal' mass from {EV['p5_ignore_limit']['min_mass_limit']:.2f} to {EV['p5_ignore_limit']['min_mass_nolimit']:.2f} kg &mdash; an unsafe beam (Fig. E-8). "
      f"VEGA has neither a rule nor spread (IGD {ig5(10)}). Evidence: Task D1 (P5) and Section 4 investigation."],
     ["<b>P10</b> DTLZ2 with M = 5: many objectives, 14 real variables, budget 10,000",
      "18 MOEA/D, 31 RVEA / 15 NSGA-III (with a larger budget)", "14 NSGA-II",
      f"With 5 objectives {EV['p10_nd_fraction']['5']*100:.0f} % of a random population is non-dominated, so Pareto rank cannot select and crowding distance does not spread. At 10,000 evaluations the IGD ranking is RVEA {D2(31,'P10_M5','igd',3)} &lt; MOEA/D {D2(18,'P10_M5','igd',3)} &lt; HypE {D2(34,'P10_M5','igd',3)} &lt; NSGA-III {D2(15,'P10_M5','igd',3)} &lt; NSGA-II {D2(14,'P10_M5','igd',3)} "
      f"(NSGA-III overtakes MOEA/D at 100,000 evaluations on M = 3: 0.052 vs 0.072). Evidence: Task D2, Fig. D-2."],
     ["<b>P8</b> TSP: permutation encoding, 2 objectives, no constraint",
      "20 MOEA/D-ACO, 30 MOMA", "27 MODE (or 22 MOPSO)",
      f"Permutations need order crossover / swaps or constructive ants; continuous arithmetic (DE, PSO velocity) has no meaning. MOEA/D-ACO HV {D3(20,'hv',3,True)}, MOMA {D3(30,'hv',3,True)}, NSGA-II {D3(14,'hv',3,True)}, MOACO {D3(25,'hv',3,True)}. Evidence: Task D3."],
     ["<b>P14</b> laptop: 2 objectives, DM prefers about (1.8 kg, 12 h)",
      "35 R-NSGA-II, 5 Value Function", "18 MOEA/D (no preference mechanism) - or a full-front method when only that region matters",
      f"A preference exists, so spend evaluations there: R-NSGA-II concentrates around the reference (centroid {EV['rnsga_ref']['A (1.8 kg, 12 h)']['centroid'][0]:.2f} kg / {EV['rnsga_ref']['A (1.8 kg, 12 h)']['centroid'][1]:.1f} h); a utility optimum needs one solve ({run(5,'P14')['evals']:,} evaluations) but gives one point only. Careful: an unattainable reference (1.2 kg, 25 h) distorts the result. Evidence: Fig. E-9."],
     ["<b>P13</b> multi-task network: differentiable, shared parameters",
      "36 MGDA, 37 GradNorm", "14 NSGA-II (population search over network weights)",
      f"Gradients exist and the number of parameters is large (a population method would need far more than 10,000 evaluations). With tasks of very different gradient scale equal weights hurt the small-gradient task; GradNorm (small lr<sub>w</sub>) lowers the classification loss at the same regression loss: classification {sum(V.TD['D5']['net']['gradnorm|scale=10.0']['Lc_final'])/10:.2f}, regression {sum(V.TD['D5']['net']['gradnorm|scale=10.0']['Lr_final'])/10:.2f} versus {sum(V.TD['D5']['net']['fixed|scale=10.0']['Lc_final'])/10:.2f} / {sum(V.TD['D5']['net']['fixed|scale=10.0']['Lr_final'])/10:.2f} (Fig. D-5)."]]
    o = [H1("8  Task E &mdash; algorithm selection reasoning"),
         P("Six problems (one many-objective: P10; one constrained: P5). Choices backed by our own results: P3, P5, P10, P8 and P13 (Tasks D1-D5).", "p"),
         tbl(rows, [28, 33, 25, 90], True)]
    return o


def task_f():
    hv3 = lambda a, tag="": D1(a, "P3", "hv", tag, 3, True)
    c = EV["p6_candidates"]
    qa = [
     ("1. Why can Weighted Sum never reach a dented part of a non-convex front?",
      "A weighted sum w<sub>1</sub>f<sub>1</sub> + w<sub>2</sub>f<sub>2</sub> is minimised where a straight line of slope -w<sub>1</sub>/w<sub>2</sub> first touches the feasible set from below-left. On a concave (dented) part every such line touches an end point first, so no weight selects an interior point: "
      f"ZDT2 with 21 weights gave only end points, {EV['ws_p3']['mid']} points in the middle, HV {f(EV['ws_p3']['hv'])} versus 0.538 (Fig. E-1)."),
     ("2. Why did VEGA collapse towards the extremes in your run?",
      f"Each sub-population is selected by one objective only, so a balanced compromise (average in both objectives) is never the best for any single one and is selected by nobody; repeated generations amplify the extremes. ZDT2: {f(EV['vega_p3']['VEGA']['n'],1)} non-dominated points, HV {f(EV['vega_p3']['VEGA']['hv'])} (Fig. A-10). It has no elitism either."),
     ("3. What does elitism guarantee, and which of your algorithms have it?",
      "That a non-dominated solution once found is not lost: the next population is chosen from parents plus offspring (or an archive), so the best front is monotone (HV non-decreasing for archive methods). Elitist: 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 (archive), 27, 28, 29, 30, 31 (per sector), 32, 33, 34, 35. "
      "Non-elitist: 10 VEGA, 11 MOGA, 12 NSGA, 13 NPGA (we report the final population only). RVEA and HypE can keep a dominated point but never lose the best of a sector / front."),
     ("4. Why does crowding distance stop working well with many objectives, and what replaces it?",
      f"With k = 5, {EV['p10_nd_fraction']['5']*100:.0f} % of a random population is non-dominated (k = 2: {EV['p10_nd_fraction']['2']*100:.0f} %), so after a few generations the whole population sits in front 1 (NSGA-II: {V.TD['D2_front1_fraction_M5'][20]*100:.0f} % at generation 20) and the only remaining criterion, crowding, rewards boundary points and does not push to the front. "
      f"Reference points / vectors (NSGA-III, RVEA, MOEA/D) or indicators (IBEA, SMS-EMOA, HypE) replace it; NSGA-II IGD {D2(14,'P10_M5','igd',3)} vs {D2(15,'P10_M5','igd',3)} (NSGA-III)."),
     ("5. Which of your algorithms needs a preference from the decision maker, and what happens if it is wrong?",
      f"5 Value Function (utility), 6 Goal Programming (targets), 7 Lexicographic (priorities), 3 Benson (reference point), 35 R-NSGA-II (reference point), 8/9 TOPSIS/VIKOR (weights, v), and weighted-sum-type methods (weights). Wrong preference: P4 with risk aversion 5 / 10 / 20 moves the portfolio from return {EV['vf_p4_sens'][0]['ret']:.3f} to {EV['vf_p4_sens'][2]['ret']:.3f}; an unattainable R-NSGA-II reference collapses the population to one end; "
      "a 0.1 weight shift flips the TOPSIS winner on P12."),
     ("6. Explain Deb's three constraint rules with one numerical example.",
      f"Rule 1: feasible beats infeasible; rule 2: two feasible points compare by Pareto dominance; rule 3: two infeasible points compare by smaller total violation CV. Example (lecture): A(3,6), D(4,5) feasible; B(1,2) CV 2, C(0.5,1) CV 5. A and D dominate B and C although B, C have better objectives; A vs D is a trade-off; B dominates C (CV 2 &lt; 5) although C's objectives are better. "
      f"On Binh-Korn: (0,0) is feasible with CV {c['A (0,0)']['cv']:.1f}, (3.2,0.5) has CV {c['E (3.2,0.5)']['cv']:.2f}, (5,0.2) has CV {c['F (5,0.2)']['cv']:.2f}: the feasible point beats both, and between the infeasible ones the smaller CV wins. Without rules NSGA-II returned {cm_inf()*100:.0f} % infeasible points."),
     ("7. How would you detect that your algorithm has prematurely converged?",
      f"Track the hypervolume of the current front against evaluations (Fig. E-3): a plateau far below the reachable value (MODE lecture rule stays at 0.11 while NSGA-II reaches 0.53), and a collapsing spread (f<sub>1</sub>-range of the archive {[round(x,3) for x in EV['premature_hist']['MODE (lecture)']['f1_range']]}, Spacing NaN), or population variance going to zero."),
     ("8. Why do TOPSIS and VIKOR not count as optimisers?",
      "They never search the decision space: they only rank alternatives that already exist (a matrix). They cannot create a better point, their output depends on the supplied front and on the weights (a 0.1 shift changes the order on P12), and they have no objective function to improve."),
    ]
    o = [H1("9  Task F &mdash; viva preparation: answers to the eight sample questions"), P("Short answers with the evidence from our own runs; the code walkthrough questions (change one line, predict the effect) are prepared through the docstrings and unit tests of each module.", "p")]
    for q, a in qa:
        o.append(KeepTogether([H3(q), P(a, "p")]))
    return o


def cm_inf():
    return EV["p6_modes"]["ignore"]["infeasible_fraction_returned"]
