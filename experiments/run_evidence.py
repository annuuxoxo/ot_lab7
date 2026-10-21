"""
Evidence experiments for Task B (advantages / disadvantages with numbers) and for the 'Investigate' questions of Section 4.
Everything here is our own runs; results/evidence.json is read by the report builder, which quotes the numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *
import ops
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from algorithms import alg06_goal_programming as gp, alg07_lexicographic as lex, alg08_topsis as tp, alg09_vikor as vk
import solvers as sv

E = {}
INF = float("inf")
EV = 10000


def ndistinct(F, tol=1e-3):
    F = np.asarray(F)
    if len(F) == 0:
        return 0
    return int(len(np.unique(np.round(F / tol).astype(np.int64), axis=0)))


def mid_count(F, lo=0.3, hi=0.7):
    F = np.asarray(F)
    return int(((F[:, 0] > lo) & (F[:, 0] < hi)).sum()) if len(F) else 0


def many(alg, key, seeds, evals=EV, **kw):
    return run_many([(alg, key, s, evals, kw, "") for s in seeds], verbose=False)


def mean(rs, k):
    v = np.array([r[k] for r in rs], float); return float(np.nanmean(v))


# =================================================================== classical methods
def classical():
    P1, P3, P5, P4 = (pb.make(k) for k in ("P1", "P3", "P5", "P4"))
    # --- weighted sum
    r = A.load(1).run(P1, n_weights=11)
    F = P1.norm(np.array(r.extra["all_F"]))
    gaps = np.linalg.norm(np.diff(F, axis=0), axis=1)
    E["ws_p1"] = dict(n_weights=11, distinct=ndistinct(r.extra["all_F"]), gap_min=float(gaps.min()), gap_max=float(gaps.max()),
                      gap_ratio=float(gaps.max() / max(gaps.min(), 1e-12)), spacing=P1.spacing(r.F), hv=P1.hv(r.F), evals=r.evals,
                      first_gaps=gaps.tolist())
    r = A.load(1).run(P3, n_weights=21)
    E["ws_p3"] = dict(n_weights=21, distinct=ndistinct(r.F), mid=mid_count(r.F), hv=P3.hv(r.F), igd=P3.igd(r.F), evals=r.evals, seconds=r.seconds,
                      F=r.F.tolist())
    for norm in (False, True):
        r = A.load(1).run(P5, n_weights=11, normalize=norm)
        E[f"ws_p5_norm{norm}"] = dict(distinct=ndistinct(r.F, 1e-4), hv=P5.hv(r.F), igd=P5.igd(r.F), n=len(r.F),
                                      F=r.F.tolist())
    r = A.load(1).run(P4, n_weights=11); E["ws_p4"] = dict(hv=P4.hv(r.F), igd=P4.igd(r.F), n=len(r.F))
    # --- epsilon constraint
    r = A.load(2).run(P3, n_eps=21)
    E["eps_p3"] = dict(distinct=ndistinct(r.F), mid=mid_count(r.F), hv=P3.hv(r.F), igd=P3.igd(r.F), evals=r.evals, seconds=r.seconds)
    ideal2, nad2 = P3.ideal[1], P3.nadir[1]
    eps_list = [{1: e} for e in np.linspace(ideal2 - 0.3, nad2 + 0.3, 14)]
    r = A.load(2).run(P3, eps_list=eps_list)
    inf = [i["viol"] > 1e-6 for i in r.extra["info"]]
    E["eps_p3_range"] = dict(n=len(eps_list), infeasible=int(sum(inf)), eps=[float(list(e.values())[0]) for e in eps_list],
                             viol=[i["viol"] for i in r.extra["info"]])
    P2i, P2r = pb.make("P2int"), pb.make("P2")
    ri = A.load(2).run(P2i, n_eps=40); rr = A.load(2).run(P2r, n_eps=40)
    E["eps_p2"] = dict(distinct_int=ndistinct(ri.F, 1e-6), distinct_real=ndistinct(rr.F, 1e-6), n_eps=40,
                       F_int=ri.F.tolist(), hv_int=P2i.hv(ri.F), hv_real=P2r.hv(rr.F))
    # --- Benson
    ra = A.load(3).run(P3, n_points=21)
    rb = A.load(3).run(P3, mode="reference", n_points=21, z_scale=1.3)
    E["benson_p3"] = dict(lambda_distinct=ndistinct(ra.F), lambda_mid=mid_count(ra.F), lambda_hv=P3.hv(ra.F),
                          ref_distinct=ndistinct(rb.F), ref_mid=mid_count(rb.F), ref_hv=P3.hv(rb.F),
                          ref_feasible=int(sum(i["feasible"] for i in rb.extra["info"])), ref_settings=len(rb.extra["info"]),
                          ref_evals=rb.evals)
    ra = A.load(3).run(P1, n_points=11); rb = A.load(3).run(P1, mode="reference", n_points=11, z_scale=1.3)
    E["benson_p1"] = dict(lambda_hv=P1.hv(ra.F), lambda_distinct=ndistinct(ra.F), ref_hv=P1.hv(rb.F), ref_distinct=ndistinct(rb.F),
                          ref_feasible=int(sum(i["feasible"] for i in rb.extra["info"])), n=11)
    P7 = pb.make("P7"); r7 = A.load(3).run(P7, n_points=11)
    E["benson_p7"] = dict(F=P7.natural(r7.F).tolist(), distinct=ndistinct(r7.F), hv=P7.hv(r7.F))
    # --- weight metric
    wm = {}
    for p in (1, 2, INF):
        rr = A.load(4).run(P3, p=p, n_weights=21)
        r1 = A.load(4).run(P1, p=p, n_weights=21)
        wm[str(p)] = dict(p3_mid=mid_count(rr.F), p3_distinct=ndistinct(rr.F), p3_hv=P3.hv(rr.F), p1_hv=P1.hv(r1.F), p1_distinct=ndistinct(r1.F),
                          p3_evals=rr.evals, p1_spacing=P1.spacing(r1.F))
    E["wm"] = wm
    # --- value function
    r = A.load(5).run(P4)
    E["vf_p4"] = [dict(gamma=i["param"], risk=i["F"][0], ret=-i["F"][1], w=i["x"]) for i in r.extra["info"]]
    r = A.load(5).run(P4, gammas=(5.0, 10.0, 20.0))
    E["vf_p4_sens"] = [dict(gamma=i["param"], risk=i["F"][0], ret=-i["F"][1]) for i in r.extra["info"]]
    r = A.load(5).run(pb.make("P14")); E["vf_p14"] = dict(natural=pb.make("P14").natural(r.F)[0].tolist(), x=r.extra["info"][0]["x"])
    # --- goal programming
    P11, P7 = pb.make("P11"), pb.make("P7")
    out = {}
    for lab, kw in (("unit", dict(weights="unit")), ("normalized", dict(weights="normalized"))):
        i = A.load(6).run(P11, **kw).extra["info"][0]
        out[lab] = dict(x=i["x"], dm=i["d_minus"], objective=i["objective"], F=i["F_natural"])
    E["gp_p11"] = out
    sol = {}
    for lab, T in (("targets (40,60)", (40, 60)), ("targets (50,70)", (50, 70))):
        for wl, w in (("unit", [1.0, 1.0]), ("norm", [1 / T[0], 1 / T[1]]), ("w=(3,1)", [3.0, 1.0])):
            i = A.load(6).run(P7, targets=T, weight_list=[np.array(w)]).extra["info"][0]
            sol[f"{lab} {wl}"] = dict(x=i["x"], dm=i["d_minus"], F=i["F_natural"])
    E["gp_p7"] = sol
    i = A.load(6).run(P11, targets=(2400, 500)).extra["info"][0]; E["gp_p11_reachable"] = dict(dm=i["d_minus"], obj=i["objective"])
    # --- lexicographic
    ex = {}
    for sign, nm in ((1, "max xA"), (-1, "min xA")):
        c = np.array([-sign, 0.0])
        A_ub = np.vstack([P11.A_ub, -P11.C[0]]); b_ub = np.append(P11.B_ub, -2400.0)
        x, f, st = sv.linprog(c, A_ub, b_ub, bounds=[(0, None)] * 2)
        ex[nm] = dict(x=x.tolist(), profit=float(P11.C[0] @ x), units=float(P11.C[1] @ x))
    E["lex_p11_tie"] = ex
    E["lex_p11"] = A.load(7).run(P11).extra["info"][0]["F_natural"]
    E["lex_p11_swapped"] = A.load(7).run(P11, order=[1, 0]).extra["info"][0]["F_natural"]
    rr = A.load(7).run(P7, eps_list=[0, 2, 5, 10, 20])
    E["lex_p7"] = [dict(eps=i["eps"], F=i["F_natural"]) for i in rr.extra["info"]]
    # --- TOPSIS / VIKOR
    P12 = pb.P12
    C, rk = tp.topsis(P12.matrix, P12.minimize, P12.weights)
    w2 = P12.weights.copy(); w2[0] -= 0.1; w2[1] += 0.1
    C2, rk2 = tp.topsis(P12.matrix, P12.minimize, w2)
    Fm = P12.matrix * np.where(P12.minimize, 1.0, -1.0)
    E["p12"] = dict(pareto=[P12.suppliers[i] for i in np.where(mu.non_dominated_mask(Fm))[0]], topsis_C=C.tolist(), topsis_rank=(rk + 1).tolist(),
                    topsis_shift_C=C2.tolist(), topsis_shift_rank=(rk2 + 1).tolist(), shift_w=w2.tolist(),
                    vikor={str(v): dict(S=vk.vikor(P12.matrix, P12.minimize, P12.weights, v)["S"].tolist(), R=vk.vikor(P12.matrix, P12.minimize, P12.weights, v)["R"].tolist(),
                                        Q=vk.vikor(P12.matrix, P12.minimize, P12.weights, v)["Q"].tolist(),
                                        rank=(vk.vikor(P12.matrix, P12.minimize, P12.weights, v)["ranking"] + 1).tolist(),
                                        adv=vk.vikor(P12.matrix, P12.minimize, P12.weights, v)["acceptable_advantage"]) for v in (0.25, 0.5, 0.75)},
                    vikor_shift={str(v): dict(rank=(vk.vikor(P12.matrix, P12.minimize, w2, v)["ranking"] + 1).tolist(), Q=vk.vikor(P12.matrix, P12.minimize, w2, v)["Q"].tolist())
                                 for v in (0.25, 0.5, 0.75)})
    # one very bad criterion: add S6 = best cost, best quality, 40-day delivery, best reliability
    M6 = np.vstack([P12.matrix, [80, 9, 40, 96]])
    C6, rk6 = tp.topsis(M6, P12.minimize, P12.weights)
    o6 = {str(v): vk.vikor(M6, P12.minimize, P12.weights, v) for v in (0.25, 0.5, 0.75)}
    E["p12_bad"] = dict(topsis_rank=(rk6 + 1).tolist(), topsis_C=C6.tolist(), vikor_rank={v: (o["ranking"] + 1).tolist() for v, o in o6.items()},
                        vikor_R={v: o["R"].tolist() for v, o in o6.items()})
    # a point that is bad on ONE criterion only: compare R (regret) and C
    # --- P11 explain tie
    E["p11_front"] = "single point (300,200): profit 2400 and units 500 are both maximal there"


# =================================================================== evolutionary methods: sensitivity runs
def sens(alg, key, name, grid, seeds=range(5), evals=EV, extra=None):
    jobs = []
    for label, kw in grid:
        k2 = dict(extra or {}); k2.update(kw)
        for s in seeds:
            jobs.append((alg, key, s, evals, k2, label))
    res = run_many(jobs, verbose=False)
    rows = {}
    for label, _ in grid:
        rs = [r for r in res if r["tag"] == label]
        rows[label] = dict(hv=mean(rs, "hv"), igd=mean(rs, "igd"), spacing=mean(rs, "spacing"), n=mean(rs, "n"), seconds=mean(rs, "seconds"),
                           mid=float(np.mean([mid_count(r["F"]) for r in rs])), hv_std=float(np.std([r["hv"] for r in rs])))
    E[name] = rows


def evolutionary():
    # VEGA collapse
    sens(10, "P2", "vega_p2", [("VEGA", {})])
    sens(14, "P2", "nsga2_p2", [("NSGA-II", {})])
    sens(10, "P3", "vega_p3", [("VEGA", {})])
    sens(14, "P3", "nsga2_p3_ref", [("NSGA-II", {})], seeds=range(10))
    # sharing radius
    for alg, nm in ((11, "moga"), (12, "nsga"), (13, "npga")):
        sens(alg, "P3", f"{nm}_sigma", [(f"sigma={s}", {"sigma": s}) for s in (0.02, 0.1, 0.3, 0.6)])
    sens(13, "P3", "npga_tdom", [(f"t_dom={t}", {"t_dom": t}) for t in (2, 10, 40)])
    # population size / generations (P3) : IGD vs N at fixed budget, IGD vs budget at fixed N
    sens(14, "P3", "nsga2_N", [(f"N={n}", {"N": n}) for n in (10, 20, 50, 100, 200)], seeds=range(10))
    E["nsga2_budget"] = {}
    for ev in (1000, 2500, 5000, 10000, 20000, 40000):
        rs = many(14, "P3", range(10), ev); E["nsga2_budget"][str(ev)] = dict(hv=mean(rs, "hv"), igd=mean(rs, "igd"), gens=ev // 100)
    # other parameters
    sens(17, "P3", "pesa_div", [(f"div={d}", {"div": d}) for d in (4, 8, 16, 32)])
    sens(18, "P3", "moead_T", [(f"T={t}", {"T": t}) for t in (5, 10, 20, 50)])
    sens(22, "P3", "mopso_w", [(f"w={w}", {"w": w}) for w in (0.2, 0.4, 0.7, 0.9)])
    sens(24, "P3", "moabc_limit", [(f"limit={l}", {"limit": l}) for l in (5, 30, 300)])
    sens(26, "P3", "mosa_chains", [(f"chains={c}", {"chains": c}) for c in (1, 10, 50)])
    sens(32, "P3", "ibea_kappa", [(f"kappa={k}", {"kappa": k}) for k in (0.005, 0.05, 0.5, 5.0)])
    sens(32, "P3", "ibea_conv", [("original", {"convention": "original"}), ("lecture", {"convention": "lecture"})])
    sens(34, "P10_M5", "hype_S", [(f"S={s}", {"samples": s, "N": 72}) for s in (50, 500, 2000)], seeds=range(3))
    sens(31, "P10_M3", "rvea_alpha", [(f"alpha={a}", {"alpha": a}) for a in (0.0, 2.0, 8.0)], seeds=range(5))
    sens(27, "P3", "mode_variants", [("lecture rule", {}), ("survival", {"selection": "survival"})], seeds=range(10))
    sens(28, "P3", "tlbo_p3", [("MO-TLBO", {})], seeds=range(10))
    # MOMA ablation: local search on / off
    sens(30, "P5", "moma_ls", [("n_ls=0 (plain NSGA-II)", {"n_ls": 0}), ("n_ls=10", {"n_ls": 10}), ("n_ls=30", {"n_ls": 30})])
    # RVEA population shrink
    rs = many(31, "P10_M3", range(5)); E["rvea_pop"] = dict(n=mean(rs, "n"), H=91)
    # R-NSGA-II : different reference points on P14 (natural units: weight kg, runtime h)
    P14 = pb.make("P14"); rn = {}
    for lab, ref in (("A (1.8 kg, 12 h)", (1.8, 12.0)), ("B (1.6 kg, 18 h)  attainable", (1.6, 18.0)), ("C (1.2 kg, 25 h) beyond the front", (1.2, 25.0)),
                     ("D (2.1 kg, 6 h)  poor trade-off", (2.1, 6.0))):
        rs = run_many([(35, "P14", s, EV, {"ref_points": [ref]}, lab) for s in range(5)], verbose=False)
        Fn = [P14.natural(np.array(r["F"])) for r in rs]
        cent = np.mean([f.mean(0) for f in Fn], 0)
        spread = float(np.mean([f[:, 0].max() - f[:, 0].min() for f in Fn]))
        rn[lab] = dict(centroid=cent.tolist(), weight_range=spread, n=mean(rs, "n"), hv=mean(rs, "hv"))
    E["rnsga_ref"] = rn


# =================================================================== problem investigations
def investigations():
    rng = np.random.default_rng(0)
    # P10 non-dominated fraction of a random population
    fr = {}
    for M in (2, 3, 5):
        p = pb.P10(M) if M > 2 else None
        vals = []
        for _ in range(30):
            if p is None:
                F = pb.P3().evaluate(rng.random((100, 10)))[0]
            else:
                F = p.evaluate(rng.random((100, p.n_var)))[0]
            vals.append(mu.non_dominated_mask(F).mean())
        fr[str(M)] = float(np.mean(vals))
    E["p10_nd_fraction"] = fr
    # P5 infeasible fraction + effect of ignoring the stress limit
    P5 = pb.make("P5"); X = P5.random_x(100000, rng); F, cv = P5.evaluate(X)
    E["p5_infeasible_fraction"] = float((cv > 0).mean())
    class P5nc(pb.P5):
        name = "P5nc"
        def evaluate(self, X):
            F, cv = super().evaluate(X); return F, np.zeros_like(cv)
    P5n = P5nc(); P5n._bounds = P5.bounds; P5n._ref = P5.reference_front()
    pb._CACHE["P5nc"] = P5n
    pb._MAKE["P5nc"] = lambda: P5n
    rs = run_many([(14, "P5nc", s, EV, {}, "no stress limit") for s in range(5)], verbose=False)
    rc = run_many([(21, "P5", s, EV, {}, "with limit") for s in range(5)], verbose=False)
    Fnc = np.vstack([np.array(r["F"]) for r in rs]); Fc = np.vstack([np.array(r["F"]) for r in rc])
    # evaluate the 'no limit' designs with the true constraint
    E["p5_ignore_limit"] = dict(min_mass_nolimit=float(Fnc[:, 0].min()), min_mass_limit=float(Fc[:, 0].min()),
                                hv_nolimit=pb.make("P5").hv(Fnc[mu.non_dominated_mask(Fnc)]), hv_limit=pb.make("P5").hv(Fc[mu.non_dominated_mask(Fc)]),
                                F_nolimit=Fnc[mu.non_dominated_mask(Fnc)].tolist(), F_limit=Fc[mu.non_dominated_mask(Fc)].tolist())
    # P5 sanity
    E["p5_sanity"] = dict(F=P5.evaluate(np.array([[0.01, 0.05]]))[0][0].tolist(), stress_MPa=float(P5.stress([[0.01, 0.05]])[0] / 1e6))
    # P6: Deb's rules, four candidate points
    P6 = pb.make("P6")
    cand = {"A (0,0)": (0, 0), "B (5,3)": (5, 3), "C (4,3)": (4, 3), "D (1,3)": (1, 3), "E (3.2,0.5)": (3.2, 0.5), "F (5,0.2)": (5, 0.2)}
    Fc_, CVc_ = P6.evaluate(np.array(list(cand.values()), float))
    E["p6_candidates"] = {k: dict(f=Fc_[i].tolist(), cv=float(CVc_[i])) for i, k in enumerate(cand)}
    # P6: no feasibility rule
    modes = {}
    for mode in ("ignore", "penalty", "deb"):
        infs = []; nfs = []
        for s in range(10):
            from algorithms import alg14_nsga2 as n2
            r = n2.run(P6, seed=s, max_evals=EV, constraint=mode)
            infs.append(r.extra["n_infeasible_returned"] / max(r.extra["n_returned"], 1)); nfs.append(len(r.F))
        modes[mode] = dict(infeasible_fraction_returned=float(np.mean(infs)), n_feasible_front=float(np.mean(nfs)))
    # for 'ignore' mode report max CV of the raw returned set
    r = n2.run(P6, seed=0, max_evals=EV, constraint="ignore")
    modes["ignore"]["max_cv"] = float(np.max(r.CV)) if len(r.CV) else 0.0
    E["p6_modes"] = modes
    # P7: front exactness and DM preference (40,50)
    P7 = pb.make("P7"); r = A.load(21).run(P7, seed=0)
    N = P7.natural(r.F); x1 = N[:, 0] / 0.8; x2 = N[:, 1] / 1.2
    E["p7_line_error"] = dict(max_abs=float(np.abs(x1 + x2 - 90).max()), n=len(N), range_f1=[float(N[:, 0].min()), float(N[:, 0].max())])
    tf = P7.natural(P7.true_front(2000)); d = np.linalg.norm((tf - [40, 50]) / (P7.nadir - P7.ideal), axis=1)
    E["p7_pref"] = dict(closest_true=tf[np.argmin(d)].tolist(), x=[float(tf[np.argmin(d), 0] / 0.8), float(tf[np.argmin(d), 1] / 1.2)])
    rr = A.load(35).run(P7, seed=0); Nn = P7.natural(rr.F)
    E["p7_rnsga"] = dict(centroid=Nn.mean(0).tolist(), range_f1=[float(Nn[:, 0].min()), float(Nn[:, 0].max())], n=len(Nn))
    # P4 repair vs penalty
    out = {}
    for key in ("P4", "P4pen"):
        rs = run_many([(14, key, s, EV, {}, key) for s in range(10)], verbose=False)
        P = pb.make(key); cvs = []
        out[key] = dict(hv=mean(rs, "hv"), igd=mean(rs, "igd"), spacing=mean(rs, "spacing"), n=mean(rs, "n"), seconds=mean(rs, "seconds"))
    from algorithms import alg14_nsga2 as n2
    rp = n2.run(pb.make("P4pen"), seed=0, max_evals=EV)
    out["P4pen"]["max_sum_error"] = float(np.abs(rp.X.sum(1) - 1).max())
    rq = n2.run(pb.make("P4"), seed=0, max_evals=EV)
    W = pb.make("P4").weights(rq.X); order = np.argsort(rq.F[:, 0])
    prof = {}
    for lab, q in (("conservative", 0.1), ("balanced", 0.5), ("aggressive", 0.9)):
        i = order[int(q * (len(order) - 1))]
        prof[lab] = dict(risk=float(rq.F[i, 0]), ret=float(-rq.F[i, 1]), w=W[i].tolist())
    out["profiles"] = prof
    out["ends"] = dict(max_return=float(-rq.F[:, 1].max()), min_risk=float(rq.F[:, 0].min()), w_minrisk=W[np.argmin(rq.F[:, 0])].tolist())
    E["p4"] = out
    # P2 : evenly spread or clustered?
    P2 = pb.make("P2"); P2i = pb.make("P2int")
    sp = {}
    for key in ("P2", "P2int"):
        rs = run_many([(14, key, s, EV, {}, key) for s in range(5)] + [(10, key, s, EV, {}, "VEGA") for s in range(5)], verbose=False)
        for tag in (key, "VEGA"):
            q = [r for r in rs if r["tag"] == tag]
            sp[f"{key}|{tag}"] = dict(n=mean(q, "n"), spacing=mean(q, "spacing"), hv=mean(q, "hv"))
    E["p2"] = sp
    # P8 operators
    P8 = pb.make("P8"); from algorithms import alg14_nsga2 as n2
    ops_res = {}
    for lab, cx, mt in (("OX + mixed(swap/inv)", "ox", "mixed"), ("OX + swap", "ox", "swap"), ("OX + inversion", "ox", "inversion"),
                        ("mutation only (inversion)", "none", "inversion"), ("OX only", "ox", "none")):
        ops.CFG["perm_cx"], ops.CFG["perm_mut"] = cx, mt
        if mt == "none":
            ops.CFG["perm_mut"] = "swap"
            old = ops.perm_mutation; ops.perm_mutation = lambda x, rng: x.copy()
        hvs = []; igds = []
        for s in range(10):
            r = n2.run(P8, seed=s, max_evals=EV); hvs.append(P8.hv(r.F)); igds.append(P8.igd(r.F))
        if mt == "none":
            ops.perm_mutation = old
        ops_res[lab] = dict(hv=float(np.mean(hvs)), hv_std=float(np.std(hvs)), igd=float(np.mean(igds)))
    ops.CFG["perm_cx"], ops.CFG["perm_mut"] = "ox", "mixed"
    E["p8_operators"] = ops_res
    ref = P8.reference_front(); E["p8_ref_points"] = int(len(ref))
    nn_d = P8.nearest_neighbour_tour(P8.D); nn_t = P8.nearest_neighbour_tour(P8.T)
    Fnn = P8.evaluate(np.array([nn_d, nn_t]))[0]
    rr = n2.run(P8, seed=0, max_evals=40000)
    xd = rr.X[np.argmin(rr.F[:, 0])]; xt = rr.X[np.argmin(rr.F[:, 1])]
    edges = lambda t: {frozenset((int(t[i]), int(t[(i + 1) % len(t)]))) for i in range(len(t))}
    E["p8_tours"] = dict(nn_distance_tour=Fnn[0].tolist(), nn_time_tour=Fnn[1].tolist(), best_dist=float(rr.F[:, 0].min()), best_time=float(rr.F[:, 1].min()),
                         dist_of_best_time_tour=float(rr.F[np.argmin(rr.F[:, 1]), 0]), time_of_best_dist_tour=float(rr.F[np.argmin(rr.F[:, 0]), 1]),
                         common_edges=len(edges(xd) & edges(xt)), front_size=len(rr.F), coords=P8.xy.tolist(),
                         tour_dist=xd.tolist(), tour_time=xt.tolist())
    rand = P8.random_x(1000, np.random.default_rng(1)); Fr = P8.evaluate(rand)[0]
    E["p8_random"] = dict(mean_dist=float(Fr[:, 0].mean()), mean_time=float(Fr[:, 1].mean()))
    # P14: x1 = 13 on all front points?
    P14 = pb.make("P14"); r = A.load(14).run(P14, seed=0, max_evals=EV)
    E["p14_x1"] = dict(max_x1=float(r.X[:, 0].max()), min_x1=float(r.X[:, 0].min()), frac_at_13=float((r.X[:, 0] < 13.01).mean()), n=len(r.X))
    tf = P14.true_front(2000); Rn = P14.norm(np.array([[1.8, -12.0]]))
    d = np.linalg.norm(P14.norm(tf) - Rn, axis=1)
    E["p14_pref"] = dict(front_pt_closest=P14.natural(tf[np.argmin(d)]).tolist(), dist=float(d.min()),
                         front_runtime_at_1p8=float(-tf[np.argmin(np.abs(tf[:, 0] - 1.8)), 1]))
    # premature convergence detection: HV plateau and diversity
    hist = {}
    for alg, nm, kw in ((14, "NSGA-II", {}), (27, "MODE (lecture)", {}), (28, "MO-TLBO", {}), (26, "MOSA", {}), (22, "MOPSO", {"w": 0.9})):
        r = A.load(alg).run(pb.make("P3"), seed=0, max_evals=EV, **kw)
        hist[nm] = dict(history=[list(h) for h in r.history], final_hv=pb.make("P3").hv(r.F), n=len(r.F), f1_range=[float(r.F[:, 0].min()), float(r.F[:, 0].max())])
    E["premature_hist"] = hist


def _p9_job(args):
    alg, seed = args
    P = pb.make("P9"); r = A.load(alg).run(P, seed=seed, max_evals=20000)
    return alg, seed, r.F.tolist(), r.X.tolist()


def p9_investigation():
    import multiprocessing as mp
    jobs = [(a, s) for a in (14, 30) for s in range(5)]
    with mp.get_context("fork").Pool(9) as pool:
        out = pool.map(_p9_job, jobs)
    P9 = pb.make("P9")
    F = np.vstack([np.array(o[2]) for o in out]); X = np.vstack([np.array(o[3]) for o in out])
    nd = mu.non_dominated_mask(F); F, X = F[nd], X[nd]
    _, u = np.unique(np.round(F, 9), axis=0, return_index=True)
    F, X = F[u], X[u]
    o = np.argsort(F[:, 1]); F, X = F[o], X[o]
    base = P9.baseline_error()
    ok = F[:, 0] <= base + 0.01
    smallest = int(F[ok, 1].min())
    # frequency of every feature over ALL front solutions of all runs (pooled, de-duplicated)
    allX = np.vstack([np.array(o_[3]) for o_ in out]); allF = np.vstack([np.array(o_[2]) for o_ in out])
    keep = np.vstack([ (np.array(o_[3])) for o_ in out])
    nd_all = mu.non_dominated_mask(allF); fx = allX[nd_all]
    freq = (fx > 0.5).mean(0)
    order = np.argsort(-freq)
    E["p9"] = dict(baseline_error=base, baseline_acc=1 - base, front=F.tolist(), n_features=F[:, 1].tolist(),
                   smallest_within_1pp=smallest, err_at_smallest=float(F[ok][np.argmin(F[ok, 1]), 0]),
                   best_error=float(F[:, 0].min()), best_error_features=int(F[np.argmin(F[:, 0]), 1]),
                   top_features=[(P9.feature_names[i], float(freq[i])) for i in order[:8]],
                   freq=freq.tolist(), n_front_solutions=int(len(fx)),
                   best_subset=[P9.feature_names[i] for i in np.where(X[np.argmin(F[:, 0])] > 0.5)[0]],
                   subset_smallest=[P9.feature_names[i] for i in np.where(X[ok][np.argmin(F[ok, 1])] > 0.5)[0]])


def main():
    which = sys.argv[1:] or ["classical", "evolutionary", "investigations"]
    path = os.path.join(RES, "evidence.json")
    if os.path.exists(path):
        E.update(json.load(open(path)))
    for w in which:
        print("running", w, flush=True)
        {"classical": classical, "evolutionary": evolutionary, "investigations": investigations, "p9": p9_investigation}[w]()
        save_json(E, "evidence.json")
    print("evidence keys:", len(E))


if __name__ == "__main__":
    main()
