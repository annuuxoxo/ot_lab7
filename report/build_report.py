"""Builds Lab7_Report.pdf from the code, tests and results."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from rl import *
import sec_setup, sec_taskA, sec_taskB, sec_taskC, sec_taskD, sec_taskEF, sec_appendix
import values as V


def figure_block(title, items):
    out = [H3(title)]
    for fn, cap, w in items:
        if os.path.exists(os.path.join(FIG, fn)):
            out += [KeepTogether([img(fn, w), P(cap, "cap")])]
    return out


def task_c():
    hand = json.load(open(os.path.join(ROOT, "results", "handtrace.json")))
    C = V.TC
    o = [PageBreak(), H1("6  Task C &mdash; deep dive: algorithms 3, 15 and 27 (roll no. 77)"),
         P("R = 77: Algorithm 1 = ((77-1) mod 37)+1 = 3, Algorithm 2 = ((77+11) mod 37)+1 = 15, Algorithm 3 = ((77+23) mod 37)+1 = 27.", "p"),
         H2("C1  Study cards"), H3("Algorithm 3 &mdash; Benson's method"), sec_taskC.study_card_3(),
         H3("Algorithm 15 &mdash; NSGA-III"), sec_taskC.study_card_15(), H3("Algorithm 27 &mdash; MODE"), sec_taskC.study_card_27(),
         H2("C2  Hand traces (reproduced by the code: tests/test_handtraces.py)"),
         H3("Algorithm 3 &mdash; Benson on P1, three (z*, &lambda;) settings"), sec_taskC.trace_benson(hand["benson"]),
         P("Settings 1-2 are interior optima (the same x as the weighted sum of lecture slide 32: 5 and 3.8); setting 3 shows the reference point binding: s<sub>1</sub> = 0.", "small"),
         H3("Algorithm 15 &mdash; NSGA-III, drone problem (3 objectives, N = 4, p = 1 so H = 3 reference points = the unit axes, 'max' normalisation)")]
    ns = hand["nsga3"]
    o.append(P("f<sub>1</sub> = 0.5 + 0.6x<sub>1</sub> + 0.004x<sub>2</sub>, f<sub>2</sub> = 20 + 40x<sub>1</sub> + 0.3x<sub>2</sub>, f<sub>3</sub> = -(15 + 0.05x<sub>2</sub> - 8x<sub>1</sub>). "
               "Niche ties are broken by lowest index (the paper draws at random). Perpendicular distance to axis j = norm of the other two normalised coordinates.", "small"))
    o.append(P("<b>Iteration 1</b>", "pl")); o += sec_taskC.trace_nsga3(ns["it1"], "it1", ns["P0"], ns["Q0"])
    P1 = {n: ({**ns["P0"], **ns["Q0"]})[n] for n in ns["it1"]["survivors"]}
    o.append(P("<b>Iteration 2</b> (parents = survivors of iteration 1)", "pl")); o += sec_taskC.trace_nsga3(ns["it2"], "it2", P1, ns["Q1"])
    o.append(H3("Algorithm 27 &mdash; MODE, two iterations")); o += sec_taskC.trace_mode(hand["mode"])
    o.append(H2("C3  Sensitivity of the most important parameter (3 values x 10 seeds)"))
    if C:
        s = C["summary"]
        def t(alg, label, mid=True):
            rows = [[label, "HV mean &plusmn; std", "IGD mean &plusmn; std", "#points", "in dent 0.3&lt;f<sub>1</sub>&lt;0.7" if mid else "runtime [s]"]]
            for k, v in s[alg].items():
                rows.append([k, pm(v["hv_mean"], v["hv_std"]), pm(v["igd_mean"], v["igd_std"]), fmt(v["n_mean"], 1), fmt(v["mid_mean"], 1) if mid else fmt(v["seconds"], 1)])
            return tbl(rows, [32, 40, 40, 24, 36], True)
        o += [H3("Benson: reference-point scale z_scale (P3, 21 settings)"), t("3", "z_scale"), img("C_alg03_sensitivity.png", 175),
              P(CTEXT3(s["3"]), "p"),
              H3("NSGA-III: divisions p (DTLZ2, M = 5, 10,000 evaluations)"), t("15", "p", False), img("C_alg15_sensitivity.png", 175), P(CTEXT15(s["15"]), "p"),
              H3("MODE (lecture rule): scale factor F (P3)"), t("27", "F"), img("C_alg27_sensitivity.png", 175), P(CTEXT27(s["27"]), "p")]
    return o


def CTEXT3(s):
    ks = list(s)
    return ("<b>Effect.</b> " + "; ".join(f"{k}: {s[k]['n_mean']:.1f} points, {s[k]['mid_mean']:.1f} in the dent, HV {s[k]['hv_mean']:.3f}" for k in ks) +
            ". The scale decides whether the box constraints s<sub>i</sub> &ge; 0 bind: when the reference box is tight enough to cut the front, the optimum is pushed onto the front itself, including the dent; "
            "when it is loose the constraints are inactive and the method degenerates to a weighted sum (end points); when too tight many settings are empty (fewer points).")


def CTEXT15(s):
    ks = list(s)
    return ("<b>Effect.</b> " + "; ".join(f"{k}: IGD {s[k]['igd_mean']:.3f}, HV {s[k]['hv_mean']:.3f}, {s[k]['n_mean']:.0f} solutions" for k in ks) +
            ". Few reference points give a coarse front; many give a fine front but, at a fixed budget, too few generations to converge (N = 212 means 47 generations).")


def CTEXT27(s):
    ks = list(s)
    return ("<b>Effect.</b> " + "; ".join(f"{k}: HV {s[k]['hv_mean']:.3f}, IGD {s[k]['igd_mean']:.3f}, {s[k]['n_mean']:.1f} archive points" for k in ks) +
            ". With the lecture's dominance-only replacement the population collapses towards f<sub>1</sub> = 0 for all F; F only changes how fast and how many intermediate archive points are collected before the collapse.")


def main():
    ts = open(os.path.join(ROOT, "results", "test_report.txt")).read().strip().splitlines()[-1]
    out_path = os.path.join(ROOT, "Lab7_Report.pdf")
    doc = make_doc(out_path)
    st = []
    st += sec_setup.title_page(f"{ts} (results/test_report.txt)")
    st += [PageBreak()] + sec_setup.conventions() + sec_setup.problem_formulations()
    st += [PageBreak()] + sec_taskA.task_a()
    # evidence figures
    ev = figure_block("Evidence figures referred to in Task B", [
        ("E_classical_P3.png", "Fig. E-1: what each classical sweep can reach on the non-convex ZDT2.", 175),
        ("E_classical_P1.png", "Fig. E-2: P1 (convex): equal weight steps do not give equal spacing; effect of p.", 170),
        ("E_premature.png", "Fig. E-3: premature convergence: HV histories and the collapsed MODE front.", 175),
        ("E_p4.png", "Fig. E-4: P4 investor profiles on the NSGA-II front and value-function optima.", 100),
        ("E_mcdm.png", "Fig. E-5: P11 tie, P7 lexicographic tolerance sweep, P12 TOPSIS weight shift.", 175),
        ("E_sensitivity.png", "Fig. E-6: sensitivity of the main parameter of ten algorithms (P3).", 178),
        ("E_p10.png", "Fig. E-7: fraction of a random population that is non-dominated.", 80),
        ("E_constraints.png", "Fig. E-8: constraint handling: no rule vs Deb's rules (P6), ignoring the stress limit (P5), infeasible designs.", 178),
        ("E_rnsga.png", "Fig. E-9: R-NSGA-II with four different reference points on P14.", 110),
        ("E_p8.png", "Fig. E-10: P8 tours and objective space.", 178), ("E_p9.png", "Fig. E-11: P9 accuracy vs number of features and most frequent features.", 170)])
    st += [NextPageTemplate("landscape"), PageBreak(), H1("4  Task B &mdash; master comparison table (all 37 algorithms)"),
           P("Advantages / disadvantages: at least four of each, at least two of each backed by a number or figure from our own runs; the rest by short arguments from the algorithm's steps. "
             "Runtimes quote Task A (seed 0); runtimes inside the 10-seed Task D runs were inflated by laptop power throttling and are not used.", "small"),
           sec_taskB.table(), NextPageTemplate("portrait"), PageBreak()] + sec_taskB.family_paragraph()
    st += ev
    st += [PageBreak(), H1("5  Section 4 investigations: answers to the problem questions")] + investigations()
    st += task_c()
    st += [PageBreak(), H1("7  Task D &mdash; head-to-head comparison (10 seeds, equal budgets)")]
    d1, res, final = sec_taskD.d1(V.TD)
    st += d1 + [img("D1_P3_overlay.png", 150), P("Fig. D-0: fronts of selected algorithms on P3 (seed 0).", "cap"), img("D1_P5_overlay.png", 150), P("Fig. D-0b: P5.", "cap"),
                img("D1_P3_fronts.png", 178), P("Fig. D-1a: seed-0 front of every algorithm on P3.", "cap"), img("D1_P5_fronts.png", 178), P("Fig. D-1b: P5.", "cap"),
                img("D1_P3_box.png", 178), img("D1_P5_box.png", 178), P("Box plots of HV and IGD over 10 seeds (* = variant).", "cap")]
    st += [P(D1_TEXT(final, res), "p")]
    st += sec_taskD.d2(V.TD) + [img("D2_manyobj.png", 178), P("Fig. D-2: IGD of the five many-objective methods and the share of NSGA-II's population in front 1.", "cap"), P(D2_TEXT(), "p")]
    st += sec_taskD.d3(V.TD) + [img("D3_tsp.png", 170), P("Fig. D-3: P8 fronts (union of 10 seeds) and HV box plot.", "cap"), P(D3_TEXT(), "p")]
    st += D4() + [H2("D5  MGDA and GradNorm against fixed equal weights on P13"), img("D5_gradient.png", 178), P("Fig. D-5: network losses (mean of 10 data seeds) and GradNorm weights with the lecture's lr<sub>w</sub>.", "cap"), P(D5_TEXT(), "p")]
    st += [PageBreak()] + sec_taskEF.task_e() + sec_taskEF.task_f()
    st += [PageBreak()] + sec_appendix.appendix()
    if os.environ.get('LAB7_DOCX'):
        return st
    doc.build(st)
    print("wrote", out_path)


def investigations():
    E = V.EV; o = []
    o.append(P(f"<b>P1.</b> {E['ws_p1']['n_weights']} weights give {E['ws_p1']['distinct']} distinct points; equal steps in w give gaps from {E['ws_p1']['gap_min']:.3f} to {E['ws_p1']['gap_max']:.3f} (not equal). "
               f"Weight metric on P1: HV p=1 {E['wm']['1']['p1_hv']:.3f}, p=2 {E['wm']['2']['p1_hv']:.3f}, p=&infin; {E['wm']['inf']['p1_hv']:.3f} (Fig. E-2).", "p"))
    p2 = E["p2"]
    o.append(P(f"<b>P2.</b> Real x: NSGA-II returns {p2['P2|P2']['n']:.0f} points, Spacing {p2['P2|P2']['spacing']:.4f}; VEGA {p2['P2|VEGA']['n']:.0f} points, Spacing {p2['P2|VEGA']['spacing']:.4f}. "
               f"Integer x: NSGA-II {p2['P2int|P2int']['n']:.0f} points (there are only 10 integers), &epsilon;-constraint {E['eps_p2']['distinct_int']} distinct vs {E['eps_p2']['distinct_real']} for real x.", "p"))
    p4 = E["p4"]
    o.append(P(f"<b>P4 (repair vs penalty).</b> NSGA-II, 10 seeds: repair HV {p4['P4']['hv']:.4f}, IGD {p4['P4']['igd']:.4f}; penalty HV {p4['P4pen']['hv']:.4f}, IGD {p4['P4pen']['igd']:.4f}, and the penalty front still violates "
               f"&Sigma;w = 1 by up to {p4['P4pen']['max_sum_error']:.4f}. We therefore use repair. Profiles on the front: " +
               "; ".join(f"{k}: risk {v['risk']:.4f}, return {v['ret']:.4f}, w = {[round(x,2) for x in v['w']]}" for k, v in p4["profiles"].items()) +
               f". The minimum-risk end is a mixture w = {[round(x,2) for x in p4['ends']['w_minrisk']]} (Fig. E-4).", "p"))
    ig = E["p5_ignore_limit"]
    o.append(P(f"<b>P5.</b> {E['p5_infeasible_fraction']*100:.1f} % of random designs violate the stress limit. Constraint handling: Deb's rules (alg 21) / penalty. Ignoring the limit lowers the minimum mass from {ig['min_mass_limit']:.2f} to {ig['min_mass_nolimit']:.2f} kg and while the hypervolume of the whole front hardly changes ({ig['hv_nolimit']:.3f} vs {ig['hv_limit']:.3f}): the unsafe designs appear only in the light, flexible end of the front (Fig. E-8). Sanity: {E['p5_sanity']['F']} kg/m, stress {E['p5_sanity']['stress_MPa']:.0f} MPa.", "p"))
    import problems as _pb, numpy as _np
    _P6 = _pb.make("P6"); _pts = {"A (0,0)": (0, 0), "B (5,3)": (5, 3), "C (0,3)": (0, 3), "D (0.5,2.5)": (0.5, 2.5), "E (4,3)": (4, 3), "F (1,3)": (1, 3)}
    _F, _CV = _P6.evaluate(_np.array(list(_pts.values()), float))
    cand = {k: dict(f=_F[i].tolist(), cv=float(_CV[i])) for i, k in enumerate(_pts)}; pm_ = E["p6_modes"]
    o.append(P("<b>P6.</b> Deb's rules by hand on six candidates (f, CV): " + "; ".join(f"{k}: f = ({v['f'][0]:.1f}, {v['f'][1]:.1f}), CV {v['cv']:.2f}" for k, v in cand.items()) +
               f". Rule 1: A, B, E, F (feasible) beat C and D (infeasible) whatever their objectives; rule 2: A, B, E, F are mutually non-dominated (a genuine trade-off, ordinary dominance); rule 3: D (CV 1.5) beats C (CV 9). NSGA-II over 10 seeds returns {pm_['ignore']['infeasible_fraction_returned']*100:.0f} % infeasible points with no rule, "
               f"{pm_['penalty']['infeasible_fraction_returned']*100:.0f} % with a penalty and {pm_['deb']['infeasible_fraction_returned']*100:.0f} % with Deb's rules (Fig. E-8).", "p"))
    o.append(P(f"<b>P7.</b> Constrained NSGA-II front: {E['p7_line_error']['n']} points, maximum deviation from x<sub>1</sub>+x<sub>2</sub> = 90 is {E['p7_line_error']['max_abs']:.2e}. "
               f"For the reference (40, 50) the closest front point is {[round(x,1) for x in E['p7_pref']['closest_true']]} (x = {[round(x,1) for x in E['p7_pref']['x']]}); R-NSGA-II centres at {[round(x,1) for x in E['p7_rnsga']['centroid']]}.", "p"))
    t = E["p8_tours"]; ops = E["p8_operators"]
    o.append(P("<b>P8.</b> NSGA-II with different permutation operators (10 seeds, HV): " + "; ".join(f"{k}: {v['hv']:.3f}" for k, v in ops.items()) +
               f". Best distance {t['best_dist']:.1f} (time of that tour {t['time_of_best_dist_tour']:.1f}); best time {t['best_time']:.1f} (distance of that tour {t['dist_of_best_time_tour']:.1f}); the two tours share {t['common_edges']} of 15 edges. "
               f"Nearest-neighbour tours: distance {t['nn_distance_tour'][0]:.1f}, time {t['nn_time_tour'][1]:.1f}; mean random tour {E['p8_random']['mean_dist']:.0f} / {E['p8_random']['mean_time']:.0f} (Fig. E-10).", "p"))
    if "p9" in E:
        d = E["p9"]
        o.append(P(f"<b>P9.</b> All-ones baseline accuracy {d['baseline_acc']*100:.2f} %. Smallest subset within 1 percentage point of the baseline: <b>{d['smallest_within_1pp']} features</b> (error {d['err_at_smallest']:.4f}: {d['subset_smallest']}). "
                   f"Best error {d['best_error']:.4f} with {d['best_error_features']} features. Most frequent features: " + ", ".join(f"{a} ({b*100:.0f} %)" for a, b in d["top_features"][:6]) + " (Fig. E-11).", "p"))
    fr = E["p10_nd_fraction"]
    o.append(P(f"<b>P10.</b> Non-dominated fraction of a random population of 100: {fr['2']*100:.0f} % (2 objectives), {fr['3']*100:.0f} % (M = 3), {fr['5']*100:.0f} % (M = 5): selection pressure from Pareto rank disappears (Fig. E-7).", "p"))
    gp = E["gp_p11"]
    o.append(P(f"<b>P11.</b> The Pareto set is the single point (300, 200) (profit and units are both maximal there). Goal programming with unit and with normalised weights both give x = {[round(v) for v in gp['unit']['x']]}, shortfall {gp['unit']['dm'][1]:.0f} on the units goal. "
               f"Lexicographic: stage 1 returns profit 2400 on an edge (max x<sub>A</sub>: {E['lex_p11_tie']['max xA']['x']}, units {E['lex_p11_tie']['max xA']['units']:.0f}; min x<sub>A</sub>: {E['lex_p11_tie']['min xA']['x']}, units {E['lex_p11_tie']['min xA']['units']:.0f}); "
               "only the second priority separates them.", "p"))
    p12 = E["p12"]
    o.append(P(f"<b>P12.</b> Pareto-optimal suppliers: {p12['pareto']} (all five). TOPSIS order S{p12['topsis_rank']} (C = {[round(c,3) for c in p12['topsis_C']]}); after moving 0.1 weight from cost to quality: {p12['topsis_shift_rank']}. "
               f"VIKOR order for v = 0.25 / 0.5 / 0.75: {p12['vikor']['0.25']['rank']} / {p12['vikor']['0.5']['rank']} / {p12['vikor']['0.75']['rank']}; after the shift (v = 0.5): {p12['vikor_shift']['0.5']['rank']}. "
               f"VIKOR protects against one very bad criterion through the regret term R (Task B rows 8-9).", "p"))
    p14 = E["p14_x1"]
    o.append(P(f"<b>P14.</b> NSGA-II front: x<sub>1</sub> in [{p14['min_x1']:.3f}, {p14['max_x1']:.3f}], {p14['frac_at_13']*100:.0f} % of points at x<sub>1</sub> = 13 (the algorithm discovers it). The reference (1.8 kg, 12 h) lies "
               f"{E['p14_pref']['dist']:.2f} (normalised) away from the front: the front offers {E['p14_pref']['front_runtime_at_1p8']:.1f} h at 1.8 kg, so the reference is dominated; R-NSGA-II focuses near weight {E['rnsga_ref']['A (1.8 kg, 12 h)']['centroid'][0]:.2f} kg (Fig. E-9).", "p"))
    return o


def D1_TEXT(final, res):
    top = ", ".join(f"#{k[0]} {sec_taskA_name(k[0])}" for _, k, _ in final[:5])
    bot = ", ".join(f"#{k[0]} {sec_taskA_name(k[0])}" for _, k, _ in final[-5:])
    return (f"<b>Reading the ranking.</b> Best combined ranks: {top}. Worst: {bot}. The top of the table is occupied by methods that combine a good diversity mechanism with strong variation (MOBA, SMS-EMOA, IBEA, MOEA/D-DE, MOGWO, MODE with survival); "
            "on ZDT2 differences between these are in the third decimal of HV, so ranks 1-8 are statistically close (std 0.000-0.003). The bottom is explained by mechanism: Weighted Sum reaches only the end points of a non-convex front; "
            "VEGA and NPGA never converge; RVEA keeps few sectors; the lecture-rule MODE collapses; MO-TLBO collapses in some seeds (large std). On the constrained P5 almost all methods reach HV 1.195, so IGD separates them: methods that rely on the penalty "
            "with an indicator or reference structure (IBEA, NSGA-III, HypE, RVEA) are the worst of the EAs. NSGA-II / Constrained NSGA-II give identical results on P5, as the penalty already separates feasible from infeasible points. "
            "Classical sweeps use 2-27x more evaluations (21-270k) for 11-21 points and are not better. Runtimes in this table were inflated by laptop power throttling; the clean runtimes are those of Task A.")


def sec_taskA_name(n):
    import algorithms as A
    return A.NAME[n]


def D2_TEXT():
    g = V.TD["D2_front1_fraction_M5"]; h = V.TD["D2_front1_fraction_M3"]
    return (f"<b>What degrades in NSGA-II.</b> At M = 5 {g[0]*100:.0f} % of the initial population is already in front 1 and it is {g[20]*100:.0f} % after 20 generations (M = 3: {h[0]*100:.0f} % and {h[20]*100:.0f} %). "
            f"Pareto rank no longer discriminates, crowding distance rewards boundary points, and the result is the worst IGD ({V.D2(14,'P10_M5','igd',3,True)}) and a hypervolume of {V.D2(14,'P10_M5','hv',3)} against {V.D2(18,'P10_M5','hv',3)} for MOEA/D. "
            "RVEA has the lowest IGD but a small, partly empty set (about 44 of 70 vectors), HypE the highest hypervolume.")


def D3_TEXT():
    return (f"MOMA ({V.D3(30,'hv',3,True)}) and MOEA/D-ACO ({V.D3(20,'hv',3,True)}) find nearly the whole 10-tour front with low variance; NSGA-II is close on average but with larger variance ({V.D3(14,'hv',3,True)}); "
            f"MOACO is clearly worse and unstable ({V.D3(25,'hv',3,True)}). Permutation operators used: order crossover + swap/inversion mutation (Section 4 evidence: P8).")


def D4():
    D = V.TD["D4"]; o = [H2("D4  TOPSIS and VIKOR on the NSGA-II front of P4 (100 portfolios) and on P12")]
    rows = [["method / weights", "top-3 (risk, return)"]]
    for k, v in D.items():
        if k.startswith("topsis") or k.startswith("vikor"):
            rows.append([k, "; ".join(f"({p[0]:.4f}, {p[1]:.4f})" for p in v["pts"])])
    o.append(tbl(rows, [50, 126], True))
    o.append(P("Moving the weight vector by 0.1 changes the top-3 of both methods (compare w = (0.5,0.5) with (0.6,0.4) and (0.4,0.6)): the ranking <b>flips</b> for TOPSIS and for VIKOR. On P12: " +
               f"TOPSIS {D['P12']['topsis']['rank']} -> {D['P12']['topsis_shift']['rank']} after the shift.", "p"))
    return o


def D5_TEXT():
    n = V.TD["D5"]["net"]; t = V.TD["D5"]["toy"]
    m = lambda k, a: sum(n[k][a]) / len(n[k][a])
    return (f"<b>Toy P13(a)</b> from (0,5): fixed equal weights end at f = {[round(x,1) for x in t['fixed equal']['f']]} (the middle of the segment, whatever the gradient sizes), MGDA at {[round(x,1) for x in t['mgda']['f']]} (the foot of the perpendicular on [a,b], a Pareto-stationary point), "
            f"GradNorm at {[round(x,1) for x in t['gradnorm']['f']]}. <b>Network, equal-size tasks</b> (scale 1): final (classification, regression) = fixed ({m('fixed|scale=1.0','Lc_final'):.3f}, {m('fixed|scale=1.0','Lr_final'):.3f}), MGDA ({m('mgda|scale=1.0','Lc_final'):.3f}, {m('mgda|scale=1.0','Lr_final'):.3f}), GradNorm ({m('gradnorm|scale=1.0','Lc_final'):.3f}, {m('gradnorm|scale=1.0','Lr_final'):.3f}). "
            f"<b>One task with ~50x larger gradients</b> (regression target scale 10; gradient norms {V.TD['D5']['net']['grad_norms_scale10']['cls'][0]:.2f} vs {V.TD['D5']['net']['grad_norms_scale10']['reg'][0]:.1f}): fixed ({m('fixed|scale=10.0','Lc_final'):.3f}, {m('fixed|scale=10.0','Lr_final'):.3f}), MGDA ({m('mgda|scale=10.0','Lc_final'):.3f}, {m('mgda|scale=10.0','Lr_final'):.3f}), GradNorm ({m('gradnorm|scale=10.0','Lc_final'):.3f}, {m('gradnorm|scale=10.0','Lr_final'):.3f}). "
            "With equal weights the large-gradient task dominates; MGDA follows the small-gradient task and lets the regression loss lag; GradNorm with a small lr<sub>w</sub> keeps the regression loss equal to the equal-weight run while lowering the classification loss, whereas the lecture's lr<sub>w</sub> = 0.025 saturates the weights (Fig. D-5, right).")


if __name__ == "__main__":
    main()
