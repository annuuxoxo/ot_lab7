"""Task B entries 22-37."""
import values as V
from values import EV, A, D1, D2, D3, run, rank


def f(x, d=3):
    return f"{x:.{d}f}"


def S(name, label, met, d=3):
    return f(EV[name][label][met], d)


def entries():
    e = {}
    hv3 = lambda a, tag="": D1(a, "P3", "hv", tag, 3, True)
    ig3 = lambda a, tag="": D1(a, "P3", "igd", tag, 4, True)
    hv5 = lambda a, tag="": D1(a, "P5", "hv", tag, 3, True)
    ig5 = lambda a, tag="": D1(a, "P5", "igd", tag, 4, True)
    rk = lambda a, p, m="hv", tag="": "%d/%d" % rank(a, p, m, tag)
    sec = lambda a, p: f"{run(a,p)['seconds']:.1f}"
    mw = EV["mopso_w"]
    e[22] = dict(
        what="Particles move with v = wv + c<sub>1</sub>r<sub>1</sub>(pbest - x) + c<sub>2</sub>r<sub>2</sub>(gbest - x); the guide is an archive leader chosen by crowding distance; pbest updates by dominance; a shrinking turbulence mutation is added.",
        adv=[f"Fast and accurate on continuous problems: P5 IGD {ig5(22)} (rank {rk(22,'P5','igd')}), P4 IGD {A(22,'P4','igd',4)}, {sec(22,'P5')} s per run.",
             f"Good spread from the archive: Spacing {D1(22,'P5','spacing','',4)} on P5.",
             "Two-line update rule, easy to implement &mdash; argument.",
             f"Reasonable on ZDT2: HV {hv3(22)} (rank {rk(22,'P3')}).",
             "Archive keeps the elite solutions (elitist)."],
        dis=[f"Sensitive to the inertia weight: P3 HV = {S('mopso_w','w=0.2','hv')} / {S('mopso_w','w=0.4','hv')} / {S('mopso_w','w=0.7','hv')} / {S('mopso_w','w=0.9','hv')} for w = 0.2 / 0.4 / 0.7 / 0.9 [Fig. E-6].",
             "Prone to premature convergence on multimodal landscapes &mdash; lecture slide 13 (not stressed by our unimodal problems).",
             "Continuous variables only (argument).",
             "Three real parameters (w, c<sub>1</sub>, c<sub>2</sub>) plus archive size.",
             f"Lower HV than the best on ZDT2 ({hv3(22)} vs {hv3(29)} for MOBA)."],
        best="Continuous, smooth problems with a limited budget.", avoid="Discrete / combinatorial problems.")
    e[23] = dict(
        what="Three leaders (alpha, beta, delta) are drawn from the archive by crowding distance each iteration; every wolf moves to the mean of three encircling positions X<sub>k</sub> = X<sub>leader</sub> - A|CX<sub>leader</sub> - X|.",
        adv=[f"Almost parameter-free yet strong: ZDT2 HV {hv3(23)} (rank {rk(23,'P3')}), P5 IGD {ig5(23)}.",
             f"Even spread: Spacing {D1(23,'P3','spacing','',4)} on P3.",
             f"Works for feature selection with a sigmoid mask: P9 HV {A(23,'P9','hv')} with {run(23,'P9')['n']} subsets.",
             "Only the archive size is a parameter &mdash; lecture slide 20.",
             f"Fast: {sec(23,'P5')} s."],
        dis=[f"On P9 behind MOMA (HV {A(23,'P9','hv')} vs {A(30,'P9','hv')}, IGD* {A(23,'P9','igd',3)} vs {A(30,'P9','igd',3)}) &mdash; the binary mapping is crude.",
             "Can converge prematurely on multimodal fronts (lecture) &mdash; the average-of-three-leaders update shrinks the swarm.",
             "Position update has no memory of a personal best &mdash; argument.",
             "Continuous design: binary / permutation need a transfer function.",
             "Quality depends on the archive truncation rule."],
        best="Continuous MOO with smooth landscapes, few parameters wanted.", avoid="Highly multimodal problems.")
    ml = EV["moabc_limit"]
    e[24] = dict(
        what="Employed bees perturb one coordinate of their source with a greedy dominance choice; onlookers pick less-crowded sources; scouts replace sources that fail more than `limit` times.",
        adv=[f"Robust on the constrained Binh-Korn: HV {A(24,'P6','hv')}, IGD {A(24,'P6','igd',4)} (best-class).",
             f"Scouts add exploration: limit = 5 / 30 / 300 gives HV {S('moabc_limit','limit=5','hv')} / {S('moabc_limit','limit=30','hv')} / {S('moabc_limit','limit=300','hv')} on P3.",
             "Few parameters (SN, limit) &mdash; lecture slide 27.",
             f"Good on P5: IGD {ig5(24)}.",
             "Good for multimodal landscapes (argument &mdash; not stressed here)."],
        dis=[f"Slower to converge on ZDT2: HV {hv3(24)} (rank {rk(24,'P3')}), {D1(24,'P3','n','',0)} points.",
             "Changes only one coordinate per move &mdash; slow for high-dimensional problems (argument).",
             f"Sensitive to the limit parameter (HV range {S('moabc_limit','limit=5','hv')}-{S('moabc_limit','limit=300','hv')}).",
             "Continuous variables.",
             f"Spacing {D1(24,'P3','spacing','',4)} worse than MOBA / MOGWO."],
        best="Multimodal MOO and diversity-critical problems.", avoid="Tight budgets.")
    e[25] = dict(
        what="Two pheromone matrices (distance, time); each ant draws a weight, builds a tour with the blended pheromone/heuristic, and only the non-dominated ants deposit on both matrices.",
        adv=[f"Builds valid tours directly: P8 HV {D3(25,'hv',3,True)}, no repair needed.",
             f"Occasionally finds the full front: best seed HV near {max(r['hv'] for r in V.TD['D3'] if r['alg']==25):.3f}.",
             "Two trails keep both objectives alive &mdash; lecture slide 34.",
             "Handles combinatorial structure natively.",
             f"Cheap per iteration: {sec(25,'P8')} s."],
        dis=[f"High variance: HV std {D3(25,'hv',3,True).split('&plusmn;')[1].strip()} and only {D3(25,'n',1)} tours on average (best-known 10), IGD* {D3(25,'igd',3)}.",
             "Slow convergence and sensitive to &alpha;, &beta;, &rho; &mdash; lecture slide 34.",
             "Strong evaporation (&rho; = 0.5) forgets structure quickly.",
             "Continuous problems impossible.",
             f"Rank last of the four on P8 (HV {D3(25,'hv',3)})."],
        best="TSP-like routing with several costs.", avoid="Continuous decision spaces.")
    mc = EV["mosa_chains"]
    e[26] = dict(
        what="Walkers propose random neighbours; non-dominated moves are always accepted, dominated moves with probability exp(-&Delta;E/T); T is cooled geometrically; accepted points feed an archive.",
        adv=[f"Excellent on easy fronts: P5 HV {hv5(26)} and IGD {ig5(26)} despite the simple moves.",
             f"Escapes local optima by design: acceptance of worse moves; archive spread Spacing {D1(26,'P5','spacing','',4)}.",
             "Works with any neighbourhood (swap, bit flip, Gaussian) &mdash; argument (ops.neighbour).",
             "Tiny memory: one point per walker.",
             "Few lines of code."],
        dis=[f"Slow convergence on ZDT2: HV {hv3(26)} vs 0.538, IGD {ig3(26)} (rank {rk(26,'P3')}).",
             f"One walker is far too weak: chains = 1 / 10 / 50 gives HV {S('mosa_chains','chains=1','hv')} / {S('mosa_chains','chains=10','hv')} / {S('mosa_chains','chains=50','hv')}.",
             "Cooling schedule needs tuning (T<sub>0</sub>, &alpha;) &mdash; lecture.",
             "No population information is shared (except through the archive).",
             "Boltzmann acceptance depends on objective scaling &mdash; we had to normalise."],
        best="Highly multimodal problems with cheap moves.", avoid="Small budgets where fast convergence is required.")
    mv = EV["mode_variants"]
    e[27] = dict(
        what="Each target gets a DE trial V = X<sub>r1</sub> + F(X<sub>r2</sub> - X<sub>r3</sub>) (binomial crossover); the trial replaces the target only if it dominates it; non-dominated trials go to the archive.",
        adv=[f"Very fast and precise on smooth continuous fronts: P5 HV {hv5(27)} IGD {ig5(27)}, {sec(27,'P5')} s.",
             f"With NSGA-II survival (variant) ZDT2 HV {hv3(27,'survival variant')}, rank {rk(27,'P3','hv','survival variant')} &mdash; DE variation is excellent.",
             "Self-scaling step size (difference of members) &mdash; argument.",
             "Only F, CR and NP &mdash; simple.",
             f"Archive keeps {D1(27,'P5','n','',0)} points on P5."],
        dis=[f"Collapse with the lecture rule on ZDT2: HV {hv3(27)}, {D1(27,'P3','n','',0)} point, IGD {ig3(27)} [Fig. E-3, Task C].",
             f"The cause is selection: all {EV['premature_hist']['MODE (lecture)']['n']} archive point(s) have f<sub>1</sub> in {[round(x,3) for x in EV['premature_hist']['MODE (lecture)']['f1_range']]}.",
             "Needs F, CR tuned (Task C3).", "Real variables only.",
             "Many objectives: trials are rarely dominating, so replacements stall &mdash; argument."],
        best="Continuous, smooth, low-dimensional objective problems (with survival selection).", avoid="Non-convex fronts with the lecture rule; discrete variables.")
    tb = EV["tlbo_p3"]["MO-TLBO"]
    e[28] = dict(
        what="Learners are taught by archive leaders (x + r(T - T<sub>F</sub> mean)), then learn from random classmates (move toward the better one); acceptance by dominance; no algorithm-specific parameters.",
        adv=[f"No tuning parameters beyond N &mdash; strong on P5 (HV {hv5(28)}, IGD {ig5(28)}).",
             f"Excellent spacing when it works: {D1(28,'P3','spacing','',4)} on P3.",
             "Two simple phases &mdash; easy to code.",
             "Archive-based, elitist.",
             f"Cheap: {sec(28,'P5')} s."],
        dis=[f"Unreliable on ZDT2: HV {hv3(28)} over 10 seeds (std {D1(28,'P3','hv','',3,True).split('&plusmn;')[1].strip()}), IGD {ig3(28)} &mdash; some seeds collapse to f<sub>1</sub> = 0.",
             "Population collapses against the bound when T<sub>F</sub> M exceeds the teacher (clipping) &mdash; verified on seed 1.",
             "Slower convergence per lecture slide 55.",
             "Needs an external archive (the MO version).",
             "Continuous only."],
        best="When no parameter tuning is possible.", avoid="Fast results or ZDT2-type fronts.")
    ms = EV["moma_ls"]
    e[29] = dict(
        what="Bats fly toward an archive leader with frequency-scaled velocity, do local random walks around it with probability 1 - r, and accept moves by dominance or loudness A; A decays, r rises.",
        adv=[f"Best result on ZDT2 in the 10-seed comparison: HV {hv3(29)}, IGD {ig3(29)} (rank {rk(29,'P3')}).",
             f"Excellent spread: Spacing {D1(29,'P3','spacing','',4)}.",
             f"Also strong on the portfolio P4: IGD {A(29,'P4','igd',4)}.",
             "Exploration-to-exploitation shift via loudness &mdash; lecture slide 62.",
             f"Fast: {sec(29,'P3')} s."],
        dis=[f"Sensitive to frequency range and loudness decay &mdash; lecture slide 62 (we did not scan them; defaults chosen as slides).",
             "Continuous variables (binary through thresholding).",
             "Local walk scale (10 % of range) is a hidden parameter &mdash; our design choice.",
             "Accepts worse moves with probability A &mdash; can lose good points (archive protects them).",
             "Little theory."],
        best="Continuous MOO, balanced exploration/exploitation.", avoid="Discrete problems.")
    e[30] = dict(
        what="One NSGA-II generation, then hill climbing (few neighbour moves) on the least-crowded front members; improved points replace the originals.",
        adv=[f"Best of the four on the TSP: HV {D3(30,'hv',3,True)}, IGD* {D3(30,'igd',3)} (NSGA-II {D3(14,'hv',3,True)}).",
             f"On feature selection it finds the 7-subset front: P9 HV {A(30,'P9','hv')}, IGD* {A(30,'P9','igd',3)}.",
             "Works with permutation, binary and real encodings through ops.neighbour.",
             "Combines global and local search.",
             f"Local search ablation on P5: HV {S('moma_ls','n_ls=10','hv')} vs {S('moma_ls','n_ls=0 (plain NSGA-II)','hv')} without it."],
        dis=[f"Local search costs evaluations: with 10,000 evaluations MOMA is not better than NSGA-II on ZDT2 (HV {hv3(30)} vs {hv3(14)}, high variance {D1(30,'P3','hv','',3,True).split('&plusmn;')[1].strip()}).",
             f"On P5 IGD {ig5(30)} vs {ig5(14)} for NSGA-II: no gain on easy continuous problems.",
             "More parameters (n<sub>ls</sub>, steps) and code.",
             "Improved points reduce diversity if n<sub>ls</sub> too large (n<sub>ls</sub> = 30 gives HV " + S('moma_ls','n_ls=30','hv') + ").",
             "Cost-benefit depends on the budget &mdash; lecture slide 69."],
        best="Combinatorial problems where local moves are cheap and effective.", avoid="Tight budgets.")
    ra = EV["rvea_alpha"]
    e[31] = dict(
        what="Unit reference vectors; each individual is assigned to the vector with the smallest angle; within a sector the individual with the smallest angle-penalised distance (1 + &alpha;(t/t<sub>max</sub>)&theta;/&gamma;) survives; vectors adapt to the objective range.",
        adv=[f"Very cheap per generation: {sec(31,'P10_M5')} s for 10,000 evaluations.",
             f"On DTLZ2 M = 5 it has the lowest IGD of the five methods in the comparison ({D2(31,'P10_M5','igd',3,True)}, NSGA-II {D2(14,'P10_M5','igd',3)}).",
             "Angle penalty trades exploration for convergence over the run &mdash; argument.",
             "Reference vectors adapt to the front's scale.",
             "Only &alpha; and the adaptation frequency."],
        dis=[f"Sectors stay empty: only {D2(31,'P10_M5','n',0)} survivors of 70 vectors (M = 5), {D2(31,'P10_M3','n',0)} of 91 (M = 3).",
             f"Weak with few objectives and at this budget: M = 3 IGD {D2(31,'P10_M3','igd',3,True)} vs MOEA/D {D2(18,'P10_M3','igd',3)}; ZDT2 HV {hv3(31)}.",
             f"&alpha; = 0 / 2 / 8 gives IGD {S('rvea_alpha','alpha=0.0','igd')} / {S('rvea_alpha','alpha=2.0','igd')} / {S('rvea_alpha','alpha=8.0','igd')} (M = 3).",
             "Can keep dominated points (exclusive sectors) &mdash; 5E slide 10.",
             "More complexity (adaptation, &gamma;)."],
        best="5+ objectives, strong convergence needed late.", avoid="2 objectives or small budgets.")
    ik, ic = EV["ibea_kappa"], EV["ibea_conv"]
    e[32] = dict(
        what="Fitness from the additive epsilon indicator I(a,b) = max<sub>i</sub>(a<sub>i</sub> - b<sub>i</sub>) summed with exp(-I/&kappa;); the worst individual is deleted repeatedly with fitness recomputed; no diversity step.",
        adv=[f"No explicit diversity mechanism and no niche radius &mdash; strong on ZDT2: HV {hv3(32)} (rank {rk(32,'P3')}), 3rd best.",
             f"Excellent on the laptop problem P14: IGD {A(32,'P14','igd',4)}.",
             f"Not sensitive to &kappa; in a wide range: HV {S('ibea_kappa','kappa=0.005','hv')} / {S('ibea_kappa','kappa=0.05','hv')} / {S('ibea_kappa','kappa=0.5','hv')} / {S('ibea_kappa','kappa=5.0','hv')} for &kappa; = 0.005 ... 5.",
             "Works for any number of objectives (indicator is O(k)) &mdash; argument.",
             f"Fast: {sec(32,'P3')} s."],
        dis=[f"Fitness convention matters: the lecture's formula gives HV {S('ibea_conv','lecture','hv')} (collapse) vs {S('ibea_conv','original','hv')} for the paper's.",
             "Objectives must be normalised or the indicator is dominated by the largest-range objective (5E slide 13).",
             f"Constrained problems with a penalty cause huge indicator values: P5 IGD {ig5(32)} (rank {rk(32,'P5','igd')}).",
             "O(N<sup>2</sup>) indicator matrix and recomputation after each deletion.",
             "&kappa; scale must match the objective scale."],
        best="When a quality indicator drives the result and objectives are normalised.", avoid="Unnormalised objectives, constrained problems without a repair.")
    e[33] = dict(
        what="Steady state: one offspring per iteration; the worst non-dominated front loses the member with the smallest hypervolume contribution.",
        adv=[f"Top performer on ZDT2 (rank {rk(33,'P3')}, HV {hv3(33)}) and on the constrained beam (rank {rk(33,'P5')}).",
             f"Directly optimises the hypervolume: the hypervolume of the final set is highest: P14 HV {A(33,'P14','hv',4)}.",
             "Excellent spread in 2 objectives: Spacing " + A(33,'P14','spacing',4) + " on P14.",
             "Total hypervolume never decreases &mdash; monotone (5E slide 16).",
             "One offspring per step, so no generation lag."],
        dis=[f"By far the slowest: {sec(33,'P3')} s (ZDT2) and {sec(33,'P14')} s (P14) for 10,000 evaluations vs {sec(14,'P3')} s for NSGA-II.",
             "Exact hypervolume is exponential in k; we use Monte-Carlo beyond 2 objectives &mdash; argument.",
             "Needs a reference point (1.1 normalised) &mdash; results depend on it.",
             "Steady-state loop in Python: one iteration per evaluation.",
             f"IGD {ig5(33)} on P5 is mid-ranked despite top HV (rank {rk(33,'P5','igd')})."],
        best="2-3 objectives with a small budget where quality per evaluation is key.", avoid="4+ objectives or when runtime matters.")
    hs = EV["hype_S"]
    e[34] = dict(
        what="Hypervolume contributions are estimated by Monte-Carlo sampling in a bounding box (each sample shared among its dominators); the lowest estimate is removed repeatedly with re-estimation.",
        adv=[f"Scales to many objectives: DTLZ2 M = 5 HV {D2(34,'P10_M5','hv',3,True)}, the highest of the five methods; M = 3 HV {D2(34,'P10_M3','hv',3)}.",
             f"Accuracy / cost trade-off is explicit: S = 50 / 500 / 2000 gives IGD {S('hype_S','S=50','igd')} / {S('hype_S','S=500','igd')} / {S('hype_S','S=2000','igd')} on M = 5.",
             "No reference vectors or niche radius.",
             "Works for any k.",
             f"Fast enough: {sec(34,'P10_M5')} s."],
        dis=[f"Sampling noise: on ZDT2 it gives HV {hv3(34)} with std {D1(34,'P3','hv','',3,True).split('&plusmn;')[1].strip()}, rank {rk(34,'P3')}.",
             f"IGD on M = 5 {D2(34,'P10_M5','igd',3,True)} is not better than MOEA/D {D2(18,'P10_M5','igd',3)}.",
             "A dominated solution can survive a generation (5E slide 19).",
             "Needs enough samples S (cost grows with S N k).",
             f"Constrained P5: IGD {ig5(34)}, rank {rk(34,'P5','igd')}."],
        best="Many objectives where exact hypervolume is too costly.", avoid="Small budgets and constrained problems.")
    rn = EV["rnsga_ref"]
    labs = list(rn)
    e[35] = dict(
        what="The decision maker gives a reference point; after non-dominated sorting the last front is ordered by normalised distance to the reference with eps-clearing so the survivors stay spread near it.",
        adv=[f"Focuses the search: for the reference (1.8 kg, 12 h) the population concentrates at centroid ({rn[labs[0]]['centroid'][0]:.2f} kg, {rn[labs[0]]['centroid'][1]:.1f} h), weight range {rn[labs[0]]['weight_range']:.2f} kg of the 0.60 kg front [Fig. E-9].",
             f"Different references move the focus: B centroid ({rn[labs[1]]['centroid'][0]:.2f}, {rn[labs[1]]['centroid'][1]:.1f}), D ({rn[labs[3]]['centroid'][0]:.2f}, {rn[labs[3]]['centroid'][1]:.1f}).",
             "Saves evaluations in the region of interest &mdash; argument.",
             f"On the water problem the closest point to (40, 50) is found: population centroid {EV['p7_rnsga']['centroid']}.",
             "Keeps NSGA-II machinery."],
        dis=[f"Needs a reference point: an unattainable one (C: 1.2 kg, 25 h) pulls the population to the extreme at centroid ({rn[labs[2]]['centroid'][0]:.2f}, {rn[labs[2]]['centroid'][1]:.1f}).",
             f"Returns a subset: its IGD on P5 is {ig5(35)} (rank {rk(35,'P5','igd')}) against the full front [Fig. D-1].",
             "&epsilon; must be tuned; shrinks as k grows (5E slide 20).",
             "Trades Pareto-optimality for preference relevance: dominated solutions can remain (5E slide 22).",
             "Needs a decision maker's input."],
        best="The decision maker already knows the region of interest.", avoid="Unknown preferences, or when the whole front is the product.")
    e[36] = dict(
        what="At x compute the gradients of all objectives; the min-norm point of their convex hull is a common descent direction; step x &larr; x - &eta;d until it vanishes (Pareto-stationary).",
        adv=[f"Converges from every start to the Pareto set of the toy: max |x<sub>1</sub> - x<sub>2</sub>| = {V.TAS['toy']['mgda_dist_to_segment']:.1e}; from (0,5) ends at {[round(v,2) for v in V.TAS['toy']['mgda_end_from_0_5']]} (equal weights ends at {[round(v,2) for v in V.TAS['toy']['fixed_end']]}).",
             "Both objectives never increase in a step (guaranteed descent) &mdash; unit test 36 reproduces the 5E numbers.",
             "Cost O(nk) + O(k<sup>2</sup>), independent of n &mdash; scales to networks.",
             "No weights needed.",
             f"Network with large-gradient task: classification loss {V.TD['D5']['net']['mgda|scale=10.0']['Lc_final'][0]:.2f}, lowest of the three methods."],
        dis=[f"Favours the task with small gradients: regression loss {sum(V.TD['D5']['net']['mgda|scale=10.0']['Lr_final'])/10:.2f} vs {sum(V.TD['D5']['net']['fixed|scale=10.0']['Lr_final'])/10:.2f} for equal weights (scale 10, 10 seeds) [Fig. D-5].",
             "Needs gradients (differentiable objectives only).",
             "Reaches only Pareto-STATIONARY points, which may not be globally optimal &mdash; lecture.",
             "One point per start: a front needs many starts.",
             "k > 2 needs a small QP (Frank-Wolfe here)."],
        best="Differentiable multi-task learning with moderate imbalance.", avoid="Non-differentiable or discrete problems.")
    e[37] = dict(
        what="Task weights w<sub>i</sub> are trained so that the weighted gradient norms G<sub>i</sub> = w<sub>i</sub>||g<sub>i</sub>|| match Gbar r<sub>i</sub><sup>&alpha;</sup>, where r<sub>i</sub> is the relative inverse training rate; weights are renormalised to sum k.",
        adv=[f"On the imbalanced network, with a small lr<sub>w</sub>, it lowers the classification loss at an unchanged regression loss relative to equal weights: classification {V.TD['D5']['net']['gradnorm|scale=10.0']['Lc_final'][0]:.2f} vs fixed {V.TD['D5']['net']['fixed|scale=10.0']['Lc_final'][0]:.2f}, regression {sum(V.TD['D5']['net']['gradnorm|scale=10.0']['Lr_final'])/10:.2f} vs {sum(V.TD['D5']['net']['fixed|scale=10.0']['Lr_final'])/10:.2f}.",
             f"Balances training automatically: weights end at {[round(v,2) for v in V.TD['D5']['net']['gradnorm|scale=10.0']['w_final']]} on scale-10 data.",
             "Reproduces the 5E numbers (w = 1.063, 0.937).",
             "Extra cost: one gradient-norm per task per step; k scalars trained.",
             "Works for any number of tasks."],
        dis=[f"Saturates with the lecture's lr<sub>w</sub> = 0.025 when gradients are large: weights run to {[round(v,3) for v in V.TD['D5']['net']['gradnorm_lrw0.025|scale=10']['w'][-1]]} [Fig. D-5].",
             "Needs &alpha;, lr<sub>w</sub> tuned and a second optimisation loop.",
             "Needs gradients.", "Does not give a front, just one trade-off per run.",
             f"On the toy it moves away from the equal-weight midpoint to {[round(v,1) for v in V.TD['D5']['toy']['gradnorm']['f']]} (f1,f2) &mdash; the balance is defined by gradient norms, not by losses."],
        best="Multi-task networks with differing gradient scales.", avoid="Non-differentiable problems, or when lr<sub>w</sub> cannot be tuned.")
    return e
