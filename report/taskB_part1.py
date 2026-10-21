"""Task B master table, entries 1-9 (classical and ranking methods).  Every number is read from results/*.json at build time."""
import values as V
from values import EV, A, D1, run, rank


def n(x, d=2):
    return f"{x:.{d}f}"


def entries():
    e = {}
    ws1, ws3, wsn, wsr = EV["ws_p1"], EV["ws_p3"], EV["ws_p5_normTrue"], EV["ws_p5_normFalse"]
    e[1] = dict(
        what="Collapse the objectives into one number F(x) = &Sigma; w<sub>i</sub> f<sub>i</sub> (w<sub>i</sub> &ge; 0, &Sigma;w<sub>i</sub> = 1), minimise it with a single-objective solver, "
             "then change the weights and solve again; every solve gives one point of the front.",
        adv=[f"One cheap single-objective solve per point: {ws1['n_weights']} weights on P1 cost {ws1['evals']:,} evaluations and 0.2 s [log, Fig. A-1].",
             f"On the convex front of P1 every weight gives a Pareto-optimal point (HV {n(ws1['hv'],3)}, {ws1['distinct']} distinct points for {ws1['n_weights']} weights).",
             "Any off-the-shelf optimiser can be used inside (we used Nelder-Mead; an LP solver would do for linear problems) &mdash; argument: the sub-problem is a plain scalar function.",
             "Weights are easy to explain to a decision maker and the extreme weights give the single-objective optima (anchor points) &mdash; argument from the formula.",
             "Memory O(1), nothing stored but the points found."],
        dis=[f"Cannot reach a dent: on ZDT2 ({ws3['n_weights']} weights) every returned point is (within 0.006) one of the two end points (f<sub>1</sub> = 0 or 1), {ws3['mid']} lie in 0.3&lt;f<sub>1</sub>&lt;0.7, HV {n(ws3['hv'],3)} (true 0.538) [Fig. E-1].",
             f"Equal steps in w do not give exactly equal spacing: on P1 the gaps between consecutive points (normalised) range from {n(ws1['gap_min'],3)} to {n(ws1['gap_max'],3)}, i.e. up to {n((ws1['gap_ratio']-1)*100,0)} % apart, and the end gaps differ most [Fig. E-2].",
             f"Very sensitive to objective scaling: on P5 in raw units (mass ~ 100 kg, deflection ~ 0.01 m) 11 weights give {wsr['distinct']} distinct design(s), with normalised objectives {wsn['distinct']}.",
             "Needs meaningful weights before the run, and each solve is independent (no information is shared between settings) &mdash; argument.",
             "Linear fronts (P7) are an extreme case: every weight lands on one of the two end points (argument; see Benson, Fig. A-3)."],
        best="Convex front, roughly known preferences, an expensive-to-code but cheap-to-run single-objective solver (e.g. portfolio P4: HV " + A(1, "P4", "hv") + ").",
        avoid="Non-convex or disconnected fronts, objectives of very different scale unless normalised, linear trade-offs.")
    ep3, epr, ep2 = EV["eps_p3"], EV["eps_p3_range"], EV["eps_p2"]
    e[2] = dict(
        what="Pick one objective as the goal and turn the others into constraints f<sub>i</sub>(x) &le; &epsilon;<sub>i</sub>; solve the constrained single-objective problem for a grid of &epsilon; values.",
        adv=[f"Reaches non-convex parts: on ZDT2 21 &epsilon; values give {ep3['distinct']} distinct points, {ep3['mid']} in the dent, HV {n(ep3['hv'],3)} vs weighted sum {n(ws3['hv'],3)} [Fig. E-1].",
             f"Control per objective: the bound &epsilon;<sub>i</sub> is in the units of the objective; on the integer fleet problem P2 40 &epsilon; values give {ep2['distinct_int']} distinct points (one per integer vehicle count).",
             "Handles real and integer decision variables unchanged (enumeration for one integer variable) &mdash; run on P2 and P2int.",
             f"Spacing follows the &epsilon; grid, so an even grid gives an even front (Spacing {n(run(2,'P3')['spacing'],3)} on ZDT2, Fig. A-2).",
             "Deterministic and interpretable: each point comes with the guarantee f<sub>i</sub> &le; &epsilon;<sub>i</sub> &mdash; argument from the formulation."],
        dis=[f"Needs an &epsilon; range: a grid wider than the front's range wastes solves, {epr['infeasible']} of {epr['n']} &epsilon; values were infeasible when it was extended 0.3 beyond ideal and nadir.",
             f"Expensive: {run(2,'P3')['evals']:,} evaluations for 11 points on ZDT2 (population methods use 10,000 for 100 points) [Fig. A-2].",
             "The number of solves grows as (grid size)<sup>k-1</sup>: 21 values per bound need 441 solves for 3 objectives (argument).",
             "Constraints on f<sub>i</sub> may be weakly active: the returned point can be only weakly Pareto-optimal (needs a second pass) &mdash; argument.",
             "Each solve needs a solver able to handle the extra constraints (our penalty Nelder-Mead gets slow on 10 variables) &mdash; runtime 11 s for P3."],
        best="Non-convex front with one objective that has priority, few objectives (2-3), integer decision variables.",
        avoid="Many objectives (grid explodes), or when &epsilon; ranges are unknown and solves are expensive.")
    b1, b3, b7 = EV["benson_p1"], EV["benson_p3"], EV["benson_p7"]
    e[3] = dict(
        what="Choose a reference point z* in objective units and weights &lambda;. s<sub>i</sub> = z*<sub>i</sub> - f<sub>i</sub> measures how far x is past the target; maximise &Sigma;&lambda;<sub>i</sub> s<sub>i</sub> "
             "subject to s<sub>i</sub> &ge; 0. Vary &lambda; or z* for more points.",
        adv=[f"Anchored search: the answer is never worse than the target in any objective (checked: s<sub>i</sub> &ge; 0 for every returned point, unit test 3).",
             f"Moving the reference point reaches the dent of ZDT2: z*-sweep gives {b3['ref_distinct']} distinct points, {b3['ref_mid']} in 0.3&lt;f<sub>1</sub>&lt;0.7, HV {n(b3['ref_hv'],3)} vs {n(b3['lambda_hv'],3)} for a &lambda;-sweep [Fig. E-1, Task C3].",
             f"On convex P1 it matches weighted sum (HV {n(b1['lambda_hv'],3)}, {b1['lambda_distinct']} points) and uses the same cheap solves (22,654 evaluations for 11 settings).",
             "Intuitive for decision makers who think in targets; the s<sub>i</sub> values tell how much slack remains in each objective &mdash; argument.",
             "Handles any constraints the sub-solver accepts."],
        dis=[f"A &lambda;-sweep alone is a weighted sum in disguise: on ZDT2 it gives {b3['lambda_distinct']} distinct points, {b3['lambda_mid']} in the dent; on the linear front P7 only {b7['distinct']} distinct points.",
             f"Needs a sensible z*: on ZDT2 only {b3['ref_feasible']} of {b3['ref_settings']} z* settings on the anti-diagonal had a non-empty admissible region [Task C3].",
             "Needs weights AND a reference point, i.e. preferences must be partly known &mdash; argument.",
             f"{b3['ref_evals']:,} evaluations for 21 settings on ZDT2 (sub-solver NM, 10 variables).",
             "Many objectives: a good sweep needs k-1 degrees of freedom in z* and in &lambda; (argument)."],
        best="A target is available (e.g. hospital cost 100 / 50 patients) and a handful of preference points are needed; convex or linear problems.",
        avoid="No reasonable reference point, or a front that must be mapped completely with &lambda; only.")
    wm = EV["wm"]
    e[4] = dict(
        what="Find the ideal point z* (k single-objective solves), then minimise the weighted p-norm distance (&Sigma; w<sub>i</sub> |f<sub>i</sub> - z*<sub>i</sub>|<sup>p</sup>)<sup>1/p</sup> for p = 1, 2 or &infin; and several weight vectors.",
        adv=[f"p = &infin; (Tchebycheff) reaches the whole non-convex front: on ZDT2 {wm['inf']['p3_distinct']} distinct points, {wm['inf']['p3_mid']} in the dent, HV {n(wm['inf']['p3_hv'],3)}; p = 2 only {wm['2']['p3_mid']} in the dent (HV {n(wm['2']['p3_hv'],3)}) [Fig. E-1].",
             f"Flexible: the norm reshapes the search; on convex P1 p = 1, 2, &infin; give HV {n(wm['1']['p1_hv'],3)}, {n(wm['2']['p1_hv'],3)}, {n(wm['inf']['p1_hv'],3)}.",
             "The ideal point gives a natural normalisation of the objectives (the deviation is measured in 'distance to the best possible') &mdash; argument.",
             "p = 1 is linear (weighted sum with the ideal point removed), so it can use LP solvers.",
             "Weights keep their meaning when objectives are rescaled by the ideal-nadir range."],
        dis=[f"Needs the ideal point: {run(4,'P3')['evals'] - 0:,} evaluations on ZDT2 include 2 extra solves; with expensive objectives these are not free.",
             f"Solution depends on p: HV on ZDT2 {n(wm['1']['p3_hv'],3)} (p=1), {n(wm['2']['p3_hv'],3)} (p=2), {n(wm['inf']['p3_hv'],3)} (p=&infin;).",
             "p = &infin; is non-smooth (max of terms): gradient solvers fail; weak Pareto optimality possible &mdash; argument.",
             f"Slow with a derivative-free sub-solver: {run(4,'P3')['seconds']:.0f} s for 11 points on ZDT2.",
             "Spacing is not controlled: equal weight steps give unequal gaps (same as weighted sum)."],
        best="Fine control over trade-off shape with a known ideal point; non-convex fronts (use p = &infin;).",
        avoid="Ideal point unknown or very expensive; many objectives (more weights to sweep).")
    v4, vs = EV["vf_p4"], EV["vf_p4_sens"]
    e[5] = dict(
        what="Write the decision maker's utility v(f<sub>1</sub>..f<sub>k</sub>), substitute F(x) and maximise it as one single-objective problem; the optimum is the most preferred point.",
        adv=[f"One solve and no front to rank: P14 with v = closeness to (1.8 kg, 12 h) returns exactly ({EV['vf_p14']['natural'][0]:.2f} kg, {EV['vf_p14']['natural'][1]:.2f} h) in {run(5,'P14')['evals']:,} evaluations.",
             f"Gives investor-specific answers: P4 with risk aversion 60 / 10 / 1.5 returns risk {v4[0]['risk']:.4f} / {v4[1]['risk']:.4f} / {v4[2]['risk']:.4f} and return {v4[0]['ret']:.4f} / {v4[1]['ret']:.4f} / {v4[2]['ret']:.4f} (Fig. E-4).",
             "Theoretically the 'correct' answer: it optimises what the decision maker really values &mdash; argument.",
             "Works with any single-objective optimiser and any shape of front (non-convex too) provided v is quasi-concave &mdash; argument.",
             "Cheap: &lt; 3,000 evaluations on P14."],
        dis=[f"The utility is a guess: halving / doubling the risk aversion on P4 moves the answer from return {vs[0]['ret']:.4f} to {vs[2]['ret']:.4f} (&gamma; = 5 / 10 / 20 gives risks {vs[0]['risk']:.4f} / {vs[1]['risk']:.4f} / {vs[2]['risk']:.4f}).",
             "Returns ONE point: no knowledge of the trade-offs that were given up &mdash; run on P14 returns a single solution (HV is meaningless).",
             "Eliciting v from a human is hard and error-prone (lecture slide 38) &mdash; argument.",
             "A badly chosen v (non-concave) gives local optima and needs multi-start (our solver used 6 starts) &mdash; argument.",
             "Cannot be used to explore: wrong preferences are invisible until the answer is inspected."],
        best="A believable utility exists (portfolio theory, cost models) and a single recommendation is wanted.",
        avoid="Preferences unknown or unstable; exploratory design where the front itself is the product.")
    g11, g7 = EV["gp_p11"], EV["gp_p7"]
    e[6] = dict(
        what="Set a target T<sub>i</sub> for each objective, add deviation variables d<sub>i</sub><sup>-</sup>, d<sub>i</sub><sup>+</sup> with f<sub>i</sub> + d<sub>i</sub><sup>-</sup> - d<sub>i</sub><sup>+</sup> = T<sub>i</sub>, "
             "and minimise the weighted undesired deviations (an LP for linear problems).",
        adv=[f"Exact LP solution in one call: P11 gives x = ({g11['unit']['x'][0]:.0f}, {g11['unit']['x'][1]:.0f}) with shortfall d<sub>2</sub><sup>-</sup> = {g11['unit']['dm'][1]:.0f} (the unit goal 600 can never be met, max 500) and 1 evaluation.",
             f"Reports WHICH goal fails and by how much (d<sub>i</sub><sup>-</sup> = {g11['unit']['dm']}) &mdash; unit and normalised weights agree on P11 ({g11['normalized']['dm']}).",
             f"Reachable targets are met exactly: P11 with T = (2400, 500) has all deviations {EV['gp_p11_reachable']['dm']} (objective {EV['gp_p11_reachable']['obj']:.0f}).",
             "Many goals with different units fit in one model &mdash; argument (deviation variables add 2k columns).",
             "Fast: LP, 0.00 s runtime."],
        dis=[f"Weights decide infeasible trade-offs: P7 with targets (50,70) gives {g7['targets (50,70) unit']['F']} with unit weights, {g7['targets (50,70) norm']['F']} with normalised weights and {g7['targets (50,70) w=(3,1)']['F']} with w=(3,1).",
             "Needs targets in advance; unreachable targets just report a shortfall instead of a trade-off curve (P11: shortfall 100) &mdash; one compromise point only.",
             "Satisficing, not optimising: with all goals met the solution is not necessarily Pareto-optimal &mdash; argument (profit 1200 &gt; goal 1000 in the 5E slide leaves 200 unused).",
             "Only linear goals stay an LP; non-linear fall back to a general solver &mdash; argument.",
             "No picture of the front is produced."],
        best="Clear numeric targets for several goals of different units (production planning P11).",
        avoid="Targets unknown, or a full trade-off curve is wanted.")
    lx = EV["lex_p11_tie"]
    e[7] = dict(
        what="Rank the objectives; optimise the first alone, add the constraint f<sub>1</sub> &ge; f<sub>1</sub>* - &epsilon;<sub>1</sub>, optimise the second subject to it, and so on (k sequential solves).",
        adv=[f"No weights at all; breaks ties correctly: P11 profit 2400 is attained on a whole edge &mdash; max x<sub>A</sub> gives {lx['max xA']['x']} (units {lx['max xA']['units']:.0f}), min x<sub>A</sub> gives {lx['min xA']['x']} (units {lx['min xA']['units']:.0f}); the second priority picks the better one, {EV['lex_p11']}.",
             "Exact (LP) and takes k solves, not a sweep: 6 evaluations for 6 settings of &epsilon; on P7.",
             f"The tolerance &epsilon; opens a controlled trade-off: P7 with &epsilon; = 0, 2, 5, 10, 20 gives yield {[round(i['F'][0]) for i in EV['lex_p7']]} and output {[round(i['F'][1]) for i in EV['lex_p7']]}.",
             "Natural for safety-first problems (never trade away a higher priority) &mdash; argument.",
             "Trivial to implement on top of an LP solver."],
        dis=[f"Priorities swap the answer entirely: P11 with units first gives {EV['lex_p11_swapped']}, profit first {EV['lex_p11']}.",
             "With &epsilon; = 0 the lower priorities only act among exact ties: P7 yields the end point (56, 24) &mdash; no trade-off (Fig. E-5).",
             "Needs a strict total order; roughly equal objectives cannot be expressed &mdash; argument.",
             "Output is one point per priority list and &epsilon;: no front.",
             "Sequential solves cannot be parallelised across stages &mdash; argument."],
        best="Strict, non-negotiable priorities (safety before cost).",
        avoid="Objectives of comparable importance; when a trade-off picture is needed.")
    p12 = EV["p12"]; pb_ = EV["p12_bad"]
    e[8] = dict(
        what="Rank an already generated set of alternatives: vector-normalise the decision matrix, weight it, find the ideal A<sup>+</sup> and anti-ideal A<sup>-</sup>, compute C<sub>i</sub> = d<sub>i</sub><sup>-</sup>/(d<sub>i</sub><sup>+</sup> + d<sub>i</sub><sup>-</sup>) and sort.",
        adv=[f"Single, intuitive ranking in O(mk): P12 gives C = {[round(c,3) for c in p12['topsis_C']]} and the order S{p12['topsis_rank']} (rank list)." ,
             "Works on any decision matrix with min/max criteria &mdash; reproduces the 5E slide values (0.466, 0.459, 0.541, 0.629) in unit test 8.",
             f"Applied to the NSGA-II front of P4 it ranks 100 portfolios in &lt; 0.01 s; the top three lie at risk {', '.join(f'{p[0]:.4f}' for p in V.TD['D4']['topsis w=(0.5,0.5)']['pts']) if V.TD else 'n/a'} (Task D4).",
             "No extra evaluation of the problem; consumes only the front &mdash; argument.",
             "Fully deterministic."],
        dis=[f"Weight-sensitive: moving 0.1 of weight from cost to quality changes the TOPSIS order from {p12['topsis_rank']} to {p12['topsis_shift_rank']} on P12.",
             f"Not protected against one terrible criterion: adding a supplier with cost 80 / quality 9 / delivery 40 d / reliability 96 gives it TOPSIS rank {pb_['topsis_rank'][-1]} of 6 whereas VIKOR (v = 0.25) puts it at rank {pb_['vikor_rank']['0.25'][-1]}.",
             "Sensitive to the normalisation method (vector vs min-max) &mdash; argument.",
             "Is not an optimiser: cannot create better alternatives (needs a generated front) &mdash; viva question 8.",
             "Rank reversal can occur when an alternative is added or removed &mdash; known property (argument)."],
        best="Ranking a short list of already generated Pareto solutions with known weights.",
        avoid="Weights unknown, or when one very bad criterion must disqualify an alternative.")
    e[9] = dict(
        what="For every alternative compute group utility S<sub>i</sub> (weighted sum of normalised gaps to the best) and individual regret R<sub>i</sub> (largest weighted gap), combine them into "
             "Q<sub>i</sub> = v(S - S*)/(S<sup>-</sup> - S*) + (1 - v)(R - R*)/(R<sup>-</sup> - R*) and rank by Q (small = good).",
        adv=[f"Explicitly balances average performance and worst-case regret: with the bad-criterion supplier added, VIKOR (v = 0.25) ranks it {pb_['vikor_rank']['0.25'][-1]} while TOPSIS ranks it {pb_['topsis_rank'][-1]}.",
             f"Shows when a single winner is not justified: on P12 the 'acceptable advantage' test Q(2nd) - Q(1st) &ge; 1/(m-1) is {p12['vikor']['0.5']['adv']} for v = 0.5 (Q = {[round(q,2) for q in p12['vikor']['0.5']['Q']]}).",
             f"Parameter v exposes the compromise: rankings for v = 0.25 / 0.5 / 0.75 are {p12['vikor']['0.25']['rank']} / {p12['vikor']['0.5']['rank']} / {p12['vikor']['0.75']['rank']}.",
             "Reproduces the 5E numbers (S = .533/.6/.4/.367, Q = .607/1/.571/0) &mdash; unit test 9.",
             "O(mk) and no problem evaluations."],
        dis=[f"Weight-sensitive as well: after the 0.1 shift the VIKOR (v = 0.5) order becomes {p12['vikor_shift']['0.5']['rank']}.",
             "Needs v (utility/regret trade-off) &mdash; the ranking changes with v (see above).",
             "More computation and more parameters than TOPSIS &mdash; lecture slide 44.",
             "Not an optimiser (ranks only).",
             "The compromise 'set' may contain several alternatives, so no unique answer is guaranteed."],
        best="A compromise is wanted that never sacrifices one criterion badly.",
        avoid="When pure closeness-to-ideal is the goal and no regret protection is needed.")
    return e
