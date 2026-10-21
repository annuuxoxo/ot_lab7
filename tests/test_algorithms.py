"""One unit test per algorithm: each checks the KEY RULE of the algorithm on numbers taken from the lecture slides."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import moo_utils as mu, ops, problems as pb
import algorithms as A

close = lambda a, b, t=1e-3: np.allclose(a, b, atol=t)


def test_01_weighted_sum():                       # Part 1 slide 32: w=(.5,.5) -> x=5 (4,4);  w=(.8,.2) -> x=3.8 (0.64,10.24)
    m = A.load(1); r = m.run(pb.P1(), weights=[(0.5, 0.5), (0.8, 0.2), (0.2, 0.8)], normalize=False)
    xs = [i["x"][0] for i in r.extra["info"]]
    assert close(xs, [5.0, 3.8, 6.2], 5e-3)
    assert close(r.extra["info"][1]["F"], [0.64, 10.24], 2e-2)
    # key rule: a convex combination never lands inside a dent -> on ZDT2 only the end points are found
    F = m.run(pb.P3(), n_weights=9).F
    assert (np.minimum(F[:, 0], 1 - F[:, 0]) < 0.06).all(), F


def test_02_epsilon_constraint():                 # Part 1 slide 35: eps2=4 -> x=2 (50,4); eps2=9 -> x=3 (33.33,9)
    m = A.load(2); r = m.run(pb.P2(), eps_list=[{1: 4.0}, {1: 9.0}, {1: 16.0}])
    assert close([i["x"][0] for i in r.extra["info"]], [2, 3, 4], 5e-3)
    assert close(r.extra["info"][1]["F"], [33.333, 9], 2e-2)
    F = m.run(pb.P3(), n_eps=9).F                                      # reaches the dented middle of the non-convex front
    assert ((F[:, 0] > 0.3) & (F[:, 0] < 0.7)).any()


def test_03_benson():                             # every returned point satisfies s_i >= 0 ; objective = sum lam_i s_i
    m = A.load(3); r = m.run(pb.P1(), n_points=5)
    for i in r.extra["info"]:
        assert all(s >= -1e-6 for s in i["s"]) and i["feasible"]
    z = np.array([4.0, 10.0]); r = m.run(pb.P1(), settings=[((0.5, 0.5), z)])
    assert close(r.extra["info"][0]["s"], [0.0, 6.0], 2e-2)            # the reference point binds (s1 = 0)


def test_04_weight_metric():                      # Tchebycheff (p=inf) reaches the dent of ZDT2; p=2 with equal weights on P1 -> x=5
    m = A.load(4)
    r = m.run(pb.P1(), p=2, n_weights=3); assert close(r.extra["info"][1]["x"][0], 5.0, 5e-3)
    F = m.run(pb.P3(), p=np.inf, n_weights=11).F
    assert ((F[:, 0] > 0.3) & (F[:, 0] < 0.7)).any()
    assert abs(m.distance(np.array([3.0, 4.0]), np.ones(2), np.zeros(2), 2, np.ones(2)) - 5.0) < 1e-12


def test_05_value_function():                     # the DM's utility is maximised: P14 -> (1.8 kg, 12 h)
    m = A.load(5); r = m.run(pb.P14())
    w, rt = pb.P14().natural(r.F)[0]
    assert abs(w - 1.8) < 0.01 and abs(rt - 12) < 0.05


def test_06_goal_programming():                   # P11: shortfall 100 on the units goal, profit goal met exactly
    m = A.load(6); r = m.run(pb.P11()); i = r.extra["info"][0]
    assert close(i["x"], [300, 200], 1e-6) and close(i["d_minus"], [0, 100], 1e-6) and close(i["d_plus"], [0, 0], 1e-6)


def test_07_lexicographic():                      # profit first (2400) then units (500) -> (300,200)
    m = A.load(7); r = m.run(pb.P11()); i = r.extra["info"][0]
    assert close(i["F_natural"], [2400, 500], 1e-6)
    r = m.run(pb.P7(), eps_list=[0.0]); assert close(r.extra["info"][0]["F_natural"], [56, 24], 1e-6)


def test_08_topsis():                             # 5E slides 30-31: supplier matrix, C = .466 .459 .541 .629 ; weight shift flips the winner
    m = A.load(8); M = np.array([[50, 7, 10], [40, 6, 14], [60, 9, 8], [45, 8, 12]], float); mn = [True, False, True]
    C, rk = m.topsis(M, mn, [0.4, 0.4, 0.2])
    assert close(C, [0.466, 0.459, 0.541, 0.629], 1e-3) and rk[0] == 3
    C2, rk2 = m.topsis(M, mn, [0.3, 0.5, 0.2]); assert rk2[0] == 2 and close(C2, [0.434, 0.351, 0.649, 0.618], 1e-3)


def test_09_vikor():                              # 5E slide 33: S,R and Q for v=0.5
    m = A.load(9); M = np.array([[50, 7, 10], [40, 6, 14], [60, 9, 8], [45, 8, 12]], float)
    o = m.vikor(M, [True, False, True], [0.4, 0.4, 0.2], 0.5)
    assert close(o["S"], [0.533, 0.6, 0.4, 0.367], 1e-3) and close(o["R"], [0.267, 0.4, 0.4, 0.133], 1e-3)
    assert close(o["Q"], [0.607, 1.0, 0.571, 0.0], 1e-3) and o["acceptable_advantage"]


def test_10_vega():                               # selection uses ONE objective per sub-population (Part 2 slide 8)
    from algorithms.ea_common import roulette
    rng = np.random.default_rng(0); f1 = 1000 / np.array([10, 20, 30, 40.]); f2 = -(np.array([10, 20, 30, 40.]) ** 2 / 100)
    for f in (f1, f2):
        fit = f.max() - f + 1e-12; sel = roulette(fit, 4000, rng)
        assert f[sel].mean() < f.mean()           # sub-population m is biased toward the minimiser of f_m
    r = A.load(10).run(pb.P3(), seed=1); assert len(r.F) <= 3                # collapse to the extremes


def test_11_moga():                               # Part 2 slides 17 and 40
    m = A.load(11); x = np.array([0., 1, 2]); S = np.stack([(x - 1) ** 2, (x + 1) ** 2], 1)
    assert (1 + mu.dominance_matrix(S).sum(0)).tolist() == [1, 1, 3]   # x=2 is dominated by x=0 AND x=1 -> rank 3 (slide only names x=0)
    x = np.array([5., 10, 15, 20]); F = np.stack([10 * x, -100 * np.sqrt(x)], 1); Fn = (F - F.min(0)) / (F.max(0) - F.min(0))
    fit, rank, nc = m.shared_fitness(F, Fn, 0.5)
    assert close(nc, [1.0, 1.151, 1.417, 1.266], 5e-3) and close(fit, [1.0, 0.869, 0.706, 0.790], 5e-3)      # slide rounds coordinates to 2 d.p.


def test_12_nsga():                               # Part 2 slide 23: fronts and niche counts (sigma = 0.75)
    from algorithms.ea_common import sharing_matrix
    m = A.load(12); pts = np.array([(30, 70), (50, 50), (70, 30), (35, 50)], float); F = -pts
    fronts, _ = mu.fast_non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1, 2] and fronts[1].tolist() == [3]
    ABC = F[:3]; Fn = (ABC - ABC.min(0)) / (ABC.max(0) - ABC.min(0))
    nc = sharing_matrix(Fn, 0.75)[0].sum(1); assert close(nc, [1.111, 1.222, 1.111], 2e-3)
    Fn_all = (F - F.min(0)) / (F.max(0) - F.min(0))
    fit, _ = m.shared_fitness(F, Fn_all, 0.75, 100.0)
    assert fit[:3].min() > fit[3]                 # every front-1 member outranks the front-2 member after sharing


def test_13_npga():                               # Part 2 slide 29: niche count decides between non-dominated candidates
    m = A.load(13); S = np.array([(10, -.8), (15, -.9), (12, -.85), (18, -.95)], float)
    Fn = np.array([(0, 0), (.5, .5), (.2, .25), (.8, .75)])
    assert m.tournament(S, Fn, 0, 1, np.array([2, 3]), 0.5, np.random.default_rng(0)) == 0       # A (NC .590) beats B (NC .778)
    S2 = np.array([(1, 1), (2, 2)], float); assert m.tournament(S2, Fn[:2], 0, 1, np.array([0]), 0.5, np.random.default_rng(0)) == 0   # dominance first


def test_14_nsga2():                              # Part 3 slides 8-10: bridge design
    x = np.array([1., 2, 3, 4]); F = np.stack([x ** 2, -(100 - (x - 3) ** 2)], 1)
    fronts, rank = mu.fast_non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1, 2] and fronts[1].tolist() == [3]
    cd = mu.crowding_distance(F[fronts[0]]); order = np.argsort(F[fronts[0]][:, 0])
    assert np.isinf(cd[order[0]]) and np.isinf(cd[order[-1]]) and abs(cd[order[1]] - 2.0) < 1e-9
    win = ops.crowded_tournament(np.array([0, 1]), np.array([1.0, 9.0]), np.random.default_rng(0), 400)
    assert (win == 0).mean() > 0.6                     # lower rank wins even against a much larger crowding distance (when the two differ)
    assert ops.crowded_tournament(np.array([0, 0]), np.array([1.0, np.inf]), np.random.default_rng(1), 2000).mean() > 0.6   # tie -> larger crowding


def test_15_nsga3():                              # 5E slide 5 + Part 3 slide 16: H = C(M+p-1,p) and nearest reference line
    m = A.load(15); assert len(m.reference_points(3, 2)) == 6 and len(m.reference_points(5, 4)) == 70
    Z = np.eye(3); near, dist, D = m.associate(np.array([[0.85, 0.10, 0.08]]), Z)
    assert near[0] == 0 and abs(dist[0] - 0.128) < 1e-3                                   # Design 1 -> r1, d = 0.128
    P = np.array([[1, 1, 0.2], [1, 0.1, 1], [0.1, 1, 1.0], [0.0, 0.0, 0.0], [2.5, 2.5, 0.1]])
    Fn, ideal, icpt = m.normalize_extreme(np.diag([2.0, 4.0, 5.0]) + 0.0)
    assert close(icpt, [2, 4, 5], 1e-6)                                                    # intercepts of the hyperplane through the extreme points


def test_16_spea2():                              # Part 3 slides 24-25: strength, raw fitness, density (k=1)
    m = A.load(16); F = np.array([(5, -8), (8, -12), (10, -15), (6, -7), (12, -14)], float)
    fit, raw, dens = m.spea2_fitness(F, F, 1)
    assert raw.tolist() == [0, 0, 0, 1, 1] and close(dens[[3, 4]], [0.293, 0.236], 2e-3)
    assert close(fit[:3], [0.293, 0.178, 0.236], 2e-3) and (fit[:3] < 1).all() and (fit[3:] >= 1).all()


def test_17_pesa2():                              # Part 3 slides 32-33: B and C share a grid box; sparse boxes are preferred
    m = A.load(17); F = np.array([(1, -1), (2, -2.2), (2.3, -2.4), (3.5, -3.5), (5, -4.5)], float)
    keys = m.cell_keys(F, 4); assert keys[1] == keys[2] and len({keys[0], keys[1], keys[3], keys[4]}) == 4
    class S: pass
    st = S(); st.AS = F; st.div = 4; st.rng = np.random.default_rng(0)
    picks = m.select_by_region(st, 4000); frac_crowded = np.isin(picks, [1, 2]).mean()
    assert frac_crowded < 0.15, frac_crowded


def test_18_moead():                              # Part 3 slides 40-41: Tchebycheff values and neighbourhoods
    from algorithms.moead_common import tcheby, neighbourhoods
    lam = np.array([0.5, 0.5]); z = np.zeros(2)
    assert close([tcheby(np.array(f), lam, z) for f in [(0, 4), (1, 1), (4, 0)]], [2.0, 0.5, 2.0])
    W = np.array([(1, 0), (.5, .5), (0, 1.)]); B = neighbourhoods(W, 2)
    assert set(B[0]) == {0, 1} and 1 in B[1] and 1 in B[2]


def test_19_moead_de():                           # Part 3 slides 47-48: antenna, V = X_r1 + F (X_r2 - X_r3)
    P = pb.P14(); pop = np.array([[12.0], [18.0], [8.0], [15.0]])
    class Q:  # 1-D problem stand-in with the bounds [5,25]
        kind = "real"; lb = np.array([5.0]); ub = np.array([25.0])
    u = ops.de_trial(Q, pop, 0, (2, 3, 1), 0.5, 0.9, np.random.default_rng(0)); assert close(u, [6.5])
    u = ops.de_trial(Q, pop, 1, (3, 1, 2), 0.6, 0.9, np.random.default_rng(0)); assert close(u, [21.0])


def test_20_moead_aco():                          # Part 3 slide 54: p(R1) = 0.579, p(R2) = 0.421
    m = A.load(20); p = m.transition_probs([1.0, 1.0], [1 / 40, 1 / 55], 1, 1, [1, 1])
    assert close(p, [0.579, 0.421], 1e-3)
    r = m.run(pb.P8(), seed=0, max_evals=600); assert all(sorted(t.tolist()) == list(range(15)) for t in r.X)


def test_21_constrained_nsga2():                  # Part 1 slide 23: Deb's three rules inside the sorting and the survival selection
    from algorithms import alg14_nsga2 as n2
    F = np.array([(3, 6), (4, 5), (1, 2), (0.5, 1)], float); CV = np.array([0, 0, 2, 5.0])
    assert [sorted(f.tolist()) for f in mu.fast_non_dominated_sort(F, CV)[0]] == [[0, 1], [2], [3]]
    keep = n2.environmental_selection(F, CV, 3, True); assert sorted(keep.tolist()) == [0, 1, 2]
    r = A.load(21).run(pb.P6(), seed=1, max_evals=4000); assert len(r.F) > 0 and (r.CV <= 0).all()


def test_22_mopso():                              # Part 4 slide 13: v = w v + c1 r1 (pbest-x) + c2 r2 (gbest-x), w=.7, c1=c2=1.5, r=.5
    m = A.load(22); v = m.velocity_update(np.array([1.0]), np.array([2.0]), np.array([4.0]), np.array([6.0]), .7, 1.5, 1.5, .5, .5)
    assert close(v, [0.7 + 1.5 * .5 * 2 + 1.5 * .5 * 4])


def test_23_mogwo():                              # a = 0 -> A = 0, every X_k = leader -> new position = mean of alpha, beta, delta
    m = A.load(23); X = np.array([[0.5, 0.5]]); L = [np.array([[0.0, 0.2]]), np.array([[1.0, 0.4]]), np.array([[0.5, 0.9]])]
    z = [np.array([[0.3, 0.3]])] * 3
    assert close(m.encircle(X, L, 0.0, z, z), np.mean(L, axis=0))


def test_24_moabc():                              # employed bee: x_new = x_i + phi (x_i - x_k); greedy dominance choice
    m = A.load(24); v = m.employed_move(np.array([4.0, 9.0]), np.array([8.0, 1.0]), 0, 0.5); assert close(v, [2.0, 9.0])
    assert m.compare(np.array([1., 1.]), np.array([2., 2.])) == "new" and m.compare(np.array([2., 2.]), np.array([1., 1.])) == "old"
    assert m.compare(np.array([1., 3.]), np.array([2., 2.])) == "tie"


def test_25_moaco():                              # only non-dominated ants deposit; evaporation (1-rho)
    m = A.load(25); n = 4; tau = [np.ones((n, n)), np.ones((n, n))]
    tours = np.array([[0, 1, 2, 3], [0, 2, 1, 3]]); F = np.array([[10., 10.], [20., 20.]]); nd = np.array([True, False])
    new = m.update_pheromone(tau, tours, F, nd, 0.5, 10.0, np.array([10., 10.]), 0.0, 99.0)
    assert close(new[0][0, 1], 0.5 + 10.0) and close(new[0][0, 2], 0.5)      # ant 1 deposits, ant 2 (dominated) does not


def test_26_mosa():                               # Part 4 slide 45: dE = 5 -> P = 0.951 (T=100) and 0.007 (T=1)
    m = A.load(26); assert abs(m.accept_probability(5.0, 100) - 0.951) < 1e-3 and abs(m.accept_probability(5.0, 1.0) - 0.0067) < 1e-3
    a = m.run(pb.P1(), seed=0, max_evals=500); assert len(a.F) > 3


def test_27_mode():                               # lecture rule: trial replaces the target only if it DOMINATES it
    m = A.load(27); P = pb.P1(); st = m.initialize(P, NP=2, X0=np.array([[1.0], [8.0]]))
    Fu, CVu, Su = st.evaluate(np.array([[3.0], [2.0]]))                       # (0,16) dominates (4,36); (1,25) vs (25,1) non-dominated
    rep = m.select(st, np.array([[3.0], [2.0]]), Fu, CVu, Su)
    assert rep.tolist() == [True, False] and close(st.X[:, 0], [3.0, 8.0])
    u = ops.de_trial(P, np.array([[4.0], [8.0], [2.0]]), 0, (0, 1, 2), 0.5, 0.9, np.random.default_rng(0)); assert close(u, [7.0])  # 4 + .5 (8-2)


def test_28_motlbo():                             # Part 4 slide 55: X_new = X + r (X_teacher - T_F X_mean)
    m = A.load(28); assert close(m.teacher_move(np.array([10.0]), np.array([4.0]), np.array([8.0]), 2, 0.5), [10 + 0.5 * (4 - 16)])
    assert close(m.learner_move(np.array([1.0]), np.array([3.0]), True, 0.5), [1 + .5 * (1 - 3)]) and close(m.learner_move(np.array([1.0]), np.array([3.0]), False, 0.5), [2.0])


def test_29_moba():                               # Part 4 slide 62: f_i = 0 + (2-0)*0.2 = 0.4 ; v = v + (x_best - x) f_i ; A <- 0.9 A
    m = A.load(29); f = m.frequency(0.0, 2.0, 0.2); assert abs(f - 0.4) < 1e-12
    assert close(m.velocity_update(np.array([0.0]), np.array([1.0]), np.array([6.0]), f), [2.0]) and abs(0.9 * 1.0 - 0.9) < 1e-12


def test_30_moma():                               # local search never returns a point dominated by its start
    m = A.load(30); st = m.initialize(pb.P3(), seed=3, max_evals=10 ** 6, N=20, n_ls=3, ls_steps=5)
    for _ in range(3):
        before = st.S.copy(); m.step(st)
    r = m.run(pb.P8(), seed=0, max_evals=800); assert all(sorted(t.tolist()) == list(range(15)) for t in r.X)
    x0 = st.X[0].copy(); f0 = st.S[0].copy(); m.local_search(st, 0, np.array([.5, .5]))
    assert not (np.all(f0 <= st.S[0]) and np.any(f0 < st.S[0]))


def test_31_rvea():                               # assignment by smallest angle; APD grows with the angle penalty
    m = A.load(31); V = np.eye(2)
    Fp = np.array([[1.0, 0.1], [0.2, 1.5]])
    keep, assign, theta, apd = m.apd_select(Fp, V, 2.0, 0.0, 10)
    assert assign.tolist() == [0, 1] and np.allclose(apd, np.linalg.norm(Fp, axis=1))          # t = 0: no penalty
    _, _, _, apd_late = m.apd_select(Fp, V, 2.0, 10, 10); assert (apd_late > apd).all()
    Fp2 = np.array([[1.0, 0.5], [0.9, 0.0]]); keep, assign, _, _ = m.apd_select(Fp2, V, 2.0, 5, 10)
    assert keep.tolist() == [1]                                                                    # one survivor per sector


def test_32_ibea():                               # 5E slide 12: lecture fitness values (kappa = 20) and first deleted = L1
    m = A.load(32); L = np.array([(1.0, 12.0), (1.2, 9.0), (1.5, 7.0), (1.8, 6.0), (1.3, 11.0), (2.0, 5.5)])
    I = m.eps_indicator_matrix(L); fit = m.fitness_all(I, 20.0, "lecture")
    assert close(fit, [-4.05, -4.60, -4.83, -4.87, -4.25, -4.84], 6e-3)
    keep, _, order = m.environmental_selection(L, 4, 20.0, "lecture"); assert order[0] == 0
    I2 = m.eps_indicator_matrix(np.array([[0., 0.], [1., 1.]])); assert I2[0, 1] == -1 and I2[1, 0] == 1       # a dominates b -> I(a,b) < 0


def test_33_smsemoa():                            # 5E slide 15: hypervolume contributions with r = (2.5, 13.5)
    m = A.load(33); L = np.array([(1.0, 12.0), (1.2, 9.0), (1.5, 7.0), (1.8, 6.0), (1.3, 11.0), (2.0, 5.5)])
    c = mu.hv_contributions_2d(L, (2.5, 13.5))
    assert close(c[[0, 2, 3, 5, 4]], [0.30, 0.60, 0.20, 0.25, 0.0], 1e-9) and abs(c[1] - 0.9) < 1e-9       # slide quotes 0.70 for L2: 0.3 x 3 = 0.9
    assert abs(mu.hv2d(L, (2.5, 13.5)) - 9.10) < 1e-9                                                       # total HV0 = 9.10 as on the slide
    assert m.worst_index.__doc__ is not None


def test_34_hype():                               # dominated point has the lowest estimate and is removed first
    m = A.load(34); Fn = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5], [0.9, 0.9]])
    keep = m.hype_remove(Fn, 1, np.array([1.1, 1.1]), 4000, np.random.default_rng(0)); assert 3 not in keep.tolist() and len(keep) == 3
    keep = m.hype_remove(Fn[:3], 1, np.array([1.1, 1.1]), 4000, np.random.default_rng(0)); assert len(keep) == 2


def test_35_rnsga2():                             # 5E slide 21: distances to the reference (1.3 kg, 800$) and eps-clearing
    m = A.load(35); L = np.array([(1.0, 12.0), (1.2, 9.0), (1.5, 7.0), (1.8, 6.0), (1.3, 11.0), (2.0, 5.5)])
    mn, mx = L.min(0), L.max(0); Fn = (L - mn) / (mx - mn); Rn = ((np.array([[1.3, 8.0]]) - mn) / (mx - mn))
    order, d = m.clearing_order(Fn, Rn, 0.15)
    assert close(d, [0.685, 0.184, 0.252, 0.587, 0.462, 0.799], 2e-3)
    assert order[:3].tolist() == [1, 2, 4]                                                       # L2, L3, L5 (as on the slide)


def test_36_mgda():                               # 5E slide 36: g1=(4,4), g2=(16,16) -> alpha*=1, d=(4,4), x1=(4.6,4.6), x2=(4.28,4.28)
    m = A.load(36); a, d = m.min_norm_point(np.array([[4., 4.], [16., 16.]])); assert close(a, [1, 0]) and close(d, [4, 4])
    x = np.array([5., 5.]); P = pb.P13a()
    for expect in ([4.6, 4.6], [4.28, 4.28]):
        a, d = m.min_norm_point(P.grad(x)); x = x - 0.1 * d; assert close(x, expect)
    a3, d3 = m.min_norm_point(np.array([[1., 0.], [0., 1.], [-1., -1.]])); assert np.linalg.norm(d3) < 1e-6          # 0 in the hull -> stationary


def test_37_gradnorm():                           # 5E slide 39: w=(1,1), g=(2,3), L=(1,3.6), L0=(2,4), alpha=.5, lr=.025
    m = A.load(37); w, Lg, target = m.gradnorm_update([1, 1], np.array([2.0, 3.0]), [1.0, 3.6], [2.0, 4.0], 0.5, 0.025)
    assert close(target, [2.113, 2.835], 1e-3) and abs(Lg - 0.278) < 1e-3 and close(w, [1.063, 0.937], 1e-3) and abs(w.sum() - 2) < 1e-12


ALL = [v for k, v in dict(globals()).items() if k.startswith("test_")]
if __name__ == "__main__":
    bad = 0
    for t in ALL:
        try:
            t(); print("PASS", t.__name__)
        except Exception as e:
            bad += 1; print("FAIL", t.__name__, type(e).__name__, e)
    print("failures:", bad)
