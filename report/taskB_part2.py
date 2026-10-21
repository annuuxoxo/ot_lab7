"""Task B entries 10-21 (evolutionary methods)."""
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
    vg2, vg3, n2p2 = EV["vega_p2"]["VEGA"], EV["vega_p3"]["VEGA"], EV["nsga2_p2"]["NSGA-II"]
    e[10] = dict(
        what="Split the mating pool into k parts; part m is filled by proportional selection on objective f<sub>m</sub> alone; shuffle the parts together, apply crossover and mutation, repeat. No archive, no dominance.",
        adv=[f"Fastest per generation: {run(10,'P3')['seconds']:.1f} s for 10,000 evaluations (O(kN)), the cheapest of all population methods [Task A log].",
             f"Finds the single-objective optima quickly: on P2 it returns {f(vg2['n'],0)} points with HV {f(vg2['hv'])} (NSGA-II {f(n2p2['hv'])}).",
             "Trivial to implement on top of any GA: only the selection step changes &mdash; argument.",
             "Needs no parameters besides the GA ones (N, p<sub>c</sub>, p<sub>m</sub>) &mdash; no niche radius.",
             "Works for any encoding that a GA supports."],
        dis=[f"Collapses to the extremes: on ZDT2 only {f(vg3['n'],1)} non-dominated points survive, HV {f(vg3['hv'])} (true 0.538), IGD {D1(10,'P3','igd','',3)} [Fig. A-10].",
             f"Rank {rk(10,'P3')} on P3 and {rk(10,'P5')} on P5 in the 10-seed comparison (HV {hv3(10)} / {hv5(10)}).",
             "No elitism: the best point found can vanish in the next generation (argument, lecture slide 8).",
             "Cannot represent balanced compromises: a compromise is mediocre in every single objective and is never selected &mdash; argument verified by the collapse above.",
             "No constraint handling of its own (penalty used)."],
        best="Teaching, or quickly locating the single-objective optima.",
        avoid="Whenever a spread of balanced trade-offs is required.")
    mg, ng, np_ = EV["moga_sigma"], EV["nsga_sigma"], EV["npga_sigma"]
    e[11] = dict(
        what="Rank each individual by 1 + the number of individuals dominating it, set fitness 1/rank, divide by the niche count of a sharing function in normalised objective space, select by roulette wheel, vary.",
        adv=[f"Dominance-based fitness gives real spread: on ZDT2 it returns {D1(11,'P3','n','',0)} points with HV {hv3(11)} versus VEGA's {f(vg3['n'],1)} points, HV {f(vg3['hv'])}.",
             f"Sharing keeps crowded ranks alive: Spacing {D1(11,'P3','spacing','',4)} on P3, only mild degradation on the constrained P5 (HV {hv5(11)}).",
             f"Cheap: {run(11,'P3')['seconds']:.1f} s for 10,000 evaluations (O(kN<sup>2</sup>)).",
             "Simple to code and works for real, integer, binary and permutation encodings &mdash; argument (uses only dominance and distances).",
             "A first-generation Pareto method that needs no archive."],
        dis=[f"Depends strongly on &sigma;<sub>share</sub>: HV on P3 = {S('moga_sigma','sigma=0.02','hv')} / {S('moga_sigma','sigma=0.1','hv')} / {S('moga_sigma','sigma=0.3','hv')} / {S('moga_sigma','sigma=0.6','hv')} for &sigma; = 0.02 / 0.1 / 0.3 / 0.6 [Fig. E-6].",
             f"Not elitist and converges slowly: rank {rk(11,'P3')} on P3 (IGD {ig3(11)}) against NSGA-II's {ig3(14)}.",
             "All rank-1 individuals get the same raw fitness, so selection pressure toward the front vanishes once the population is mostly non-dominated &mdash; argument.",
             "O(N<sup>2</sup>) niche counts every generation: grows quickly with N.",
             "Many objectives: almost everybody is rank 1 (Fig. E-7)."],
        best="Moderate diversity needed and the dominance-rank idea is to be taught.",
        avoid="Elitism mandatory, no time to tune &sigma;<sub>share</sub>, or more than 3 objectives.")
    e[12] = dict(
        what="Peel the population into non-dominated fronts, give each front a dummy fitness lower than the previous one, share fitness inside each front, select by roulette and vary.",
        adv=[f"Explicit front structure plus sharing gives good spread: P5 gives {D1(12,'P5','n','',0)} points, HV {hv5(12)}, P3 {D1(12,'P3','n','',0)} points, HV {hv3(12)}.",
             f"Better spread than MOGA on P3 (HV {hv3(12)} vs {hv3(11)}).",
             "Conceptually clean: the front index orders fronts strictly, sharing only reorders inside a front &mdash; argument (adaptive dummy fitness).",
             "Directly inspired NSGA-II, so it is a good stepping stone &mdash; lecture.",
             f"Handles the water allocation P7 (linear front on a constraint boundary) with HV {A(12,'P7','hv')}, IGD {A(12,'P7','igd',4)}."],
        dis=[f"Not elitist and sensitive to &sigma;: P3 HV = {S('nsga_sigma','sigma=0.02','hv')} / {S('nsga_sigma','sigma=0.1','hv')} / {S('nsga_sigma','sigma=0.3','hv')} / {S('nsga_sigma','sigma=0.6','hv')} for &sigma; = 0.02 / 0.1 / 0.3 / 0.6.",
             f"Behind modern methods: rank {rk(12,'P3')} on P3 (IGD {ig3(12)}), rank {rk(12,'P5')} on P5.",
             "Repeated full sorting plus per-front sharing is O(kN<sup>3</sup>) in the original analysis &mdash; slowest of the non-elitist group (lecture slide 33).",
             f"Fewer points than NSGA-II ({D1(12,'P3','n','',0)} vs {D1(14,'P3','n','',0)} on P3).",
             "No archive: good solutions can be lost."],
        best="Small populations and teaching the origin of non-dominated sorting.",
        avoid="Large populations, tight time budgets, or elitism needed.")
    e[13] = dict(
        what="Binary tournament: draw two candidates and a comparison set; if one dominates the other it wins, otherwise the candidate with the smaller niche count (measured against the comparison set) wins.",
        adv=[f"No fitness computation at all: a tournament is 2 dominance tests + at most 2 niche counts; {run(13,'P3')['seconds']:.1f} s per 10,000 evaluations.",
             f"Reasonable on the constrained beam: P5 HV {hv5(13)} ({D1(13,'P5','n','',0)} points).",
             "Plugs into any GA as a selection operator &mdash; argument.",
             "Direct use of dominance, no scalarisation.",
             "Works with any encoding."],
        dis=[f"Fails to converge on ZDT2 in 10,000 evaluations: only {D1(13,'P3','n','',0)} non-dominated points, HV {hv3(13)}, IGD {ig3(13)} [Fig. A-13].",
             f"Needs &sigma;<sub>share</sub> and the comparison-set size: P3 HV for t<sub>dom</sub> = 2 / 10 / 40 is {S('npga_tdom','t_dom=2','hv')} / {S('npga_tdom','t_dom=10','hv')} / {S('npga_tdom','t_dom=40','hv')}.",
             "Non-elitist: tournaments can discard front members (argument).",
             f"IGD rank {rk(13,'P5','igd')} on P5 (IGD {ig5(13)}).",
             "Result quality depends on random comparison sets &mdash; high variance between seeds."],
        best="Fast selection with moderate diversity, as an add-on selection operator.",
        avoid="Maximum diversity or guaranteed elitism.")
    nm = EV["nsga2_N"]; nb = EV["nsga2_budget"]
    e[14] = dict(
        what="Parents + offspring are sorted into fronts; the next population takes whole fronts and fills the last one by crowding distance; parents come from binary tournaments (lower rank, then larger crowding).",
        adv=[f"Strong, fast, parameter-light: P3 HV {hv3(14)} (IGD {ig3(14)}), P5 IGD {ig5(14)}, {run(14,'P3')['seconds']:.1f} s per run.",
             f"Good spread on the true-front problems: Spacing {D1(14,'P3','spacing','',4)} on P3.",
             f"Behaves well with population size: IGD on P3 = {S('nsga2_N','N=20','igd',4)} (N = 20), {S('nsga2_N','N=100','igd',4)} (N = 100) at a fixed 10,000 evaluations [Fig. E-6].",
             f"Keeps improving with budget: IGD {f(nb['1000']['igd'],4)} at 1,000, {f(nb['10000']['igd'],4)} at 10,000, {f(nb['40000']['igd'],4)} at 40,000 evaluations.",
             "Elitism guarantees the best front can never get worse &mdash; argument (parents compete with offspring)."],
        dis=[f"Degrades with objectives: DTLZ2 M = 5 IGD {D2(14,'P10_M5','igd',3,True)} versus {D2(15,'P10_M5','igd',3,True)} for NSGA-III; {V.TD['D2_front1_fraction_M5'][20]*100:.0f} % of its population is in front 1 after 20 generations [Fig. D-2].",
             f"A few bad seeds on ZDT2: HV {hv3(14)}, so the mean is below the best methods (rank {rk(14,'P3')}).",
             "Crowding distance is a crude spacing estimate and only uses neighbours along sorted objectives &mdash; argument.",
             "No built-in preference or constraint rule (penalty); the constrained version is algorithm 21.",
             "Convergence stalls on deceptive landscapes &mdash; lecture slide 6."],
        best="2-3 objectives, a simple reliable general-purpose elitist method.",
        avoid="4+ objectives, or when a precise archive is required.")
    mn = EV["rvea_pop"]
    e[15] = dict(
        what="NSGA-II sorting is kept, but the last front is filled using reference points on a normalised simplex: each solution is associated with its nearest reference line and the least-crowded reference points are served first.",
        adv=[f"Better than NSGA-II with many objectives: DTLZ2 M = 5 IGD {D2(15,'P10_M5','igd',3,True)} vs {D2(14,'P10_M5','igd',3,True)}; M = 3 {D2(15,'P10_M3','igd',3,True)} vs {D2(14,'P10_M3','igd',3,True)} [Fig. D-2].",
             "Guaranteed spread by construction: one niche per reference point, 72 solutions for H = 70 (M = 5) [Task C].",
             f"Fast: {run(15,'P10_M5')['seconds']:.1f} s per 10,000 evaluations (O(kN<sup>2</sup>)).",
             "No convexity assumption (reference lines are directions) &mdash; argument.",
             f"Converges with enough budget: IGD 0.052 at 100,000 evaluations on DTLZ2 M = 3 versus NSGA-II 0.099 (one extra seed-0 run, not part of the tables)."],
        dis=[f"Needs H decided in advance and N &asymp; H: p = 6 gives N = 212 and only 47 generations at 10,000 evaluations (Task C3).",
             f"Weak on 2 objectives: ZDT2 rank {rk(15,'P3')} (HV {hv3(15)}, IGD {ig3(15)}) &mdash; one reference point per front slot is a poor fit.",
             f"Constrained problems with a penalty: P5 IGD {ig5(15)}, rank {rk(15,'P5','igd')}.",
             "More moving parts than NSGA-II (normalisation, association, niching) &mdash; code is 3x longer.",
             "Regular reference sets bias the result when the front is irregular &mdash; lecture slide 14."],
        best="3-10 objectives with a regular front.",
        avoid="2 objectives or tiny budgets relative to H.")
    sp = EV["pesa_div"]
    e[16] = dict(
        what="Keep an external archive; fitness = strength of the dominated set plus raw fitness of the dominators plus a k-th nearest-neighbour density; truncate the archive by removing the closest pair; select parents from the archive by tournament.",
        adv=[f"Best-in-class spread: Spacing {A(16,'P4','spacing',4)} on P4 and IGD {A(16,'P4','igd',4)} (rank among the best), {A(16,'P5','igd',4)} on P5.",
             f"Elitist with a fixed archive size: P5 IGD {ig5(16)}, {D1(16,'P5','n','',0)} points.",
             "Density is fine-grained (k-th neighbour) &mdash; sharper than a grid &mdash; argument.",
             "Archive truncation preserves boundary points (lexicographic nearest-neighbour rule).",
             "Few parameters (N, archive, k = &radic;N)."],
        dis=[f"Slow: {run(16,'P5')['seconds']:.1f} s vs {run(14,'P5')['seconds']:.1f} s for NSGA-II on P5 (distance matrix every generation) [Fig. A-16].",
             f"Variance on ZDT2: HV {hv3(16)} (rank {rk(16,'P3')}).",
             "Memory O(N<sup>2</sup>) for distances.",
             "More code (archive truncation) &mdash; lecture slide 22.",
             "Not designed for many objectives (k-th neighbour distances concentrate) &mdash; argument."],
        best="2-3 objectives where a precise, well-spread archive matters.",
        avoid="Quick prototypes, large populations, many objectives.")
    e[17] = dict(
        what="Archive plus a hyper-grid over the archive's objective range; parents are chosen by picking a grid box (sparser box wins) and a random member in it; a new point enters only if non-dominated and a member of the fullest box is dropped on overflow.",
        adv=[f"Cheap bookkeeping: {run(17,'P3')['seconds']:.1f} s per run, grid counting is O(N).",
             f"Solid result at low cost: P3 HV {hv3(17)} (rank {rk(17,'P3')}), P5 HV {hv5(17)}.",
             f"Insensitive to resolution in a wide range: P3 HV = {S('pesa_div','div=4','hv')} / {S('pesa_div','div=8','hv')} / {S('pesa_div','div=16','hv')} / {S('pesa_div','div=32','hv')} for 4 / 8 / 16 / 32 divisions.",
             "Elitist; scales to more objectives since a grid needs no pairwise distances &mdash; argument.",
             "Simple to explain."],
        dis=[f"Less precise spread than SPEA2/NSGA-II: Spacing {D1(17,'P3','spacing','',4)} vs {D1(14,'P3','spacing','',4)}.",
             f"Grid resolution matters at the extremes: IGD {S('pesa_div','div=4','igd',4)} (4 divisions) vs {S('pesa_div','div=16','igd',4)} (16).",
             "Grid cells exist only where the archive is: a fixed grid cannot represent irregular fronts well &mdash; lecture slide 30.",
             "Number of cells grows as div<sup>k</sup>: memory explodes for many objectives (dictionary here) &mdash; argument.",
             "Only 20 offspring per iteration: many small steps, sequential loop in Python."],
        best="Limited budget and many objectives where speed matters.",
        avoid="High-precision diversity control.")
    mt = EV["moead_T"]
    e[18] = dict(
        what="N weight vectors define N Tchebycheff sub-problems; each sub-problem mates within its T nearest neighbours, and a child replaces up to n<sub>r</sub> neighbours that it improves.",
        adv=[f"Best on DTLZ2 with 3 objectives at this budget: IGD {D2(18,'P10_M3','igd',3,True)}, HV {D2(18,'P10_M3','hv',3)} (NSGA-III {D2(15,'P10_M3','igd',3)}) and strong at M = 5 (HV {D2(18,'P10_M5','hv',3)}).",
             f"Cheap: scalar comparisons only; {run(18,'P3')['seconds']:.1f} s for 10,000 evaluations on P3.",
             "Tchebycheff reaches non-convex fronts (ZDT2 IGD " + ig3(18) + ") &mdash; the dent is covered.",
             "Neighbourhood mating focuses search; each child improves up to n<sub>r</sub> sub-problems.",
             f"Scales with objectives (only weight vectors grow): H = 70 (M = 5)."],
        dis=[f"Variance on ZDT2: HV {hv3(18)} (rank {rk(18,'P3')}).",
             f"Neighbourhood size matters: P3 HV = {S('moead_T','T=5','hv')} / {S('moead_T','T=10','hv')} / {S('moead_T','T=20','hv')} / {S('moead_T','T=50','hv')} for T = 5 / 10 / 20 / 50.",
             "Quality depends on the weight-vector layout matching the front shape &mdash; irregular fronts leave sub-problems with duplicates (lecture slide 38).",
             "Needs normalised objectives and an ideal-point estimate &mdash; we use the problem's scale.",
             "GA operators converge slower than DE (algorithm 19)."],
        best="Many objectives / large scale with a regular front.",
        avoid="Disconnected or very irregular fronts, trivial 2-objective cases.")
    e[19] = dict(
        what="MOEA/D with a differential-evolution child: V = X<sub>r1</sub> + F(X<sub>r2</sub> - X<sub>r3</sub>) from neighbours, binomial crossover, polynomial mutation, then the usual neighbour update.",
        adv=[f"Best decomposition method on continuous 2-objective problems: ZDT2 HV {hv3(19)} (rank {rk(19,'P3')}), IGD {ig3(19)} vs MOEA/D {ig3(18)}.",
             f"Stable across seeds (std {D1(19,'P3','hv','',4,True).split('&plusmn;')[1].strip()}).",
             f"On DTLZ2 M = 3 it improves on MOEA/D (one extra seed-1 run: IGD 0.049 vs 0.067, not part of the tables).",
             "DE steps adapt to the spread of the neighbourhood &mdash; argument.",
             f"Good on constrained P5: HV {hv5(19)}, IGD {ig5(19)}."],
        dis=["Needs F and CR (0.5 / 1.0 here) &mdash; DE is sensitive to them (see MODE sensitivity, Task C).",
             "Real-valued only: DE arithmetic has no meaning for permutations (argument).",
             f"Same T / weight-vector dependency as MOEA/D.",
             f"Slower per evaluation than NSGA-II: {run(19,'P3')['seconds']:.1f} s vs {run(14,'P3')['seconds']:.1f} s.",
             "Population of 100 sub-problems requires 100 evaluations per sweep &mdash; slow feedback at small budgets."],
        best="Continuous problems with complicated Pareto sets.",
        avoid="Discrete or combinatorial variables.")
    p8 = D3
    e[20] = dict(
        what="MOEA/D where each sub-problem keeps its own pheromone matrix and heuristic; an ant builds a tour with probabilities &tau;<sup>&alpha;</sup>&eta;<sup>&beta;</sup>, and the tour updates neighbouring sub-problems.",
        adv=[f"Among the best on the bi-objective TSP (lowest IGD*, lowest variance): HV {p8(20,'hv',3,True)} (best-known front has 10 tours; it finds {p8(20,'n',1)} on average), IGD* {p8(20,'igd',3)}.",
             f"Low variance across seeds (std {p8(20,'hv',4,True).split('&plusmn;')[1].strip()}) compared with NSGA-II's {p8(14,'hv',3,True).split('&plusmn;')[1].strip()}.",
             "Constructive: every tour is valid by construction (no repair) &mdash; argument.",
             "Pheromone memory transfers structure between neighbouring sub-problems.",
             "Inherits MOEA/D decomposition."],
        dis=[f"Slow: {run(20,'P8')['seconds']:.1f} s vs {run(14,'P5')['seconds']:.1f} s for NSGA-II (python loops over cities and sub-problems).",
             "Four parameters (&alpha;, &beta;, &rho;, Q) &mdash; slower convergence than operators-based GA (lecture slide 52).",
             "Cannot be used for continuous variables (argument).",
             "N matrices of size n<sup>2</sup> &mdash; memory O(N n<sup>2</sup>).",
             f"Our simplified version has a slightly lower mean HV than MOMA ({p8(30,'hv',3,True)}): difference is within noise."],
        best="Discrete / routing problems.",
        avoid="Continuous variables.")
    cm = EV["p6_modes"]
    e[21] = dict(
        what="NSGA-II in which dominance is replaced by Deb's constrained dominance: feasible beats infeasible, feasible pairs by Pareto dominance, infeasible pairs by smaller violation; infeasible points sit behind all feasible fronts.",
        adv=[f"Returns only feasible points: on Binh-Korn all {run(21,'P6')['n']} returned points are feasible, HV {A(21,'P6','hv')}, IGD {A(21,'P6','igd',4)}.",
             f"Without any rule NSGA-II returns {cm['ignore']['infeasible_fraction_returned']*100:.0f} % infeasible points (max CV {cm['ignore']['max_cv']:.2f}), with the rule {cm['deb']['infeasible_fraction_returned']*100:.0f} % [Fig. E-8].",
             "No penalty factor to tune &mdash; argument.",
             f"Cantilever beam: HV {hv5(21)}, IGD {ig5(21)}, same quality as the penalty version at no extra cost.",
             f"Handles the boundary-optimal water problem P7 ({run(21,'P7')['n']} feasible points, IGD {A(21,'P7','igd',4)})."],
        dis=[f"No gain over a good penalty on easy constraints: P5 IGD {ig5(21)} equals NSGA-II's {ig5(14)}.",
             "Infeasible individuals are ranked only by CV, so the objectives of infeasible points are ignored &mdash; it can stall when the feasible region is tiny (argument).",
             "Needs a CV scale: sums of unlike constraints are compared directly (we normalise P5).",
             "Inherits NSGA-II's weakness for 4+ objectives.",
             "Equality constraints need a tolerance."],
        best="Constrained problems with moderately large feasible regions.",
        avoid="Many objectives or tiny feasible regions without repair.")
    return e
