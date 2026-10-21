"""
Task D -- head-to-head comparison with equal budgets (10,000 evaluations) and 10 seeds.
  D1  P3 (unconstrained) and P5 (constrained): every algorithm of our set that applies
  D2  P10 with M = 5 (and 3): NSGA-II, NSGA-III, MOEA/D, RVEA, HypE
  D3  P8: MOEA/D-ACO, MOACO, NSGA-II, MOMA
  D4  TOPSIS / VIKOR on the NSGA-II front of P4 (top three, weight shift of 0.1)
  D5  MGDA, GradNorm vs fixed equal weights on P13 (toy + network)
Outputs results/taskD.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_common import *

SEEDS = list(range(10)); EV = 10000; INF = float("inf")


def d1_jobs():
    jobs = []
    sweeps = [(1, {"n_weights": 21}, "WS 21w"), (2, {"n_eps": 21}, "eps 21"), (3, {"mode": "reference", "n_points": 21}, "Benson z-sweep 21"),
              (4, {"p": INF, "n_weights": 21}, "WM p=inf 21w")]
    eas = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    for key in ("P3", "P5"):
        for n, kw, tag in sweeps:
            for s in SEEDS:
                jobs.append((n, key, s, None, kw, tag))
        for n in eas:
            for s in SEEDS:
                jobs.append((n, key, s, EV, {}, ""))
        for s in SEEDS:
            jobs.append((27, key, s, EV, {"selection": "survival"}, "survival variant"))
        if key == "P5":
            for s in SEEDS:
                jobs.append((21, key, s, EV, {}, ""))
    return jobs


def d2_jobs():
    jobs = []
    for key in ("P10_M5", "P10_M3"):
        N = 72 if key == "P10_M5" else 92
        for s in SEEDS:
            jobs.append((14, key, s, EV, {"N": N}, ""))
            jobs.append((15, key, s, EV, {}, ""))
            jobs.append((18, key, s, EV, {"N": 70 if key == "P10_M5" else 91}, ""))
            jobs.append((31, key, s, EV, {}, ""))
            jobs.append((34, key, s, EV, {"N": N}, ""))
    return jobs


def d3_jobs():
    return [(n, "P8", s, EV, {}, "") for n in (20, 25, 14, 30) for s in SEEDS]


def nsga2_front1_fraction(key="P10_M5", N=72, gens=140, seeds=5):
    """Fraction of the population in the first non-dominated front of NSGA-II, per generation (many-objective degradation)."""
    from algorithms import alg14_nsga2 as a14
    P = pb.make(key); out = []
    for s in range(seeds):
        st = a14.initialize(P, seed=s, max_evals=10 ** 9, N=N)
        fr = []
        for g in range(gens):
            fr.append(float((st.rank == 0).mean())); a14.step(st)
        out.append(fr)
    return np.mean(out, axis=0).tolist()


def d4_mcdm():
    from algorithms import alg08_topsis as tp, alg09_vikor as vk
    P = pb.make("P4")
    r = A.load(14).run(P, seed=0, max_evals=EV)
    front = P.natural(r.F)                                  # (risk, return)
    M = front; mn = np.array([True, False])
    out = dict(n=len(M), front=M.tolist())
    for name, w in (("w=(0.5,0.5)", [0.5, 0.5]), ("w=(0.6,0.4)", [0.6, 0.4]), ("w=(0.4,0.6)", [0.4, 0.6])):
        C, rk = tp.topsis(M, mn, w)
        out["topsis " + name] = dict(top3=rk[:3].tolist(), pts=M[rk[:3]].tolist(), C=C[rk[:3]].tolist())
        for v in (0.25, 0.5, 0.75):
            o = vk.vikor(M, mn, w, v)
            out[f"vikor {name} v={v}"] = dict(top3=o["ranking"][:3].tolist(), pts=M[o["ranking"][:3]].tolist(), Q=o["Q"][o["ranking"][:3]].tolist())
    # supplier selection P12
    P12 = pb.P12
    sup = dict()
    C, rk = tp.topsis(P12.matrix, P12.minimize, P12.weights); sup["topsis"] = dict(C=C.tolist(), rank=(rk + 1).tolist())
    w2 = P12.weights.copy(); w2[0] -= 0.1; w2[1] += 0.1
    C2, rk2 = tp.topsis(P12.matrix, P12.minimize, w2); sup["topsis_shift"] = dict(C=C2.tolist(), rank=(rk2 + 1).tolist(), w=w2.tolist())
    for v in (0.25, 0.5, 0.75):
        o = vk.vikor(P12.matrix, P12.minimize, P12.weights, v)
        sup[f"vikor v={v}"] = dict(S=o["S"].tolist(), R=o["R"].tolist(), Q=o["Q"].tolist(), rank=(o["ranking"] + 1).tolist(), adv=o["acceptable_advantage"])
        o2 = vk.vikor(P12.matrix, P12.minimize, w2, v)
        sup[f"vikor_shift v={v}"] = dict(Q=o2["Q"].tolist(), rank=(o2["ranking"] + 1).tolist())
    # Pareto-optimal suppliers (cost min, quality max, delivery min, reliability max)
    Fm = P12.matrix * np.where(P12.minimize, 1.0, -1.0)
    sup["pareto_optimal"] = [P12.suppliers[i] for i in np.where(mu.non_dominated_mask(Fm))[0]]
    out["P12"] = sup
    return out


def d5_gradient():
    from algorithms import alg36_mgda as mg, alg37_gradnorm as gn
    import mtl_net
    P = pb.make("P13a")
    toy = {}
    x0 = np.array([0.0, 5.0])
    for name, step_fn in (("fixed equal", lambda x: P.grad(x).sum(0)),
                          ("mgda", lambda x: mg.min_norm_point(P.grad(x))[1])):
        x = x0.copy(); tr = [x.copy()]
        for _ in range(300):
            x = x - 0.05 * step_fn(x); tr.append(x.copy())
        toy[name] = dict(traj=np.array(tr).tolist(), f=P.evaluate(x[None])[0][0].tolist())
    x = x0.copy(); w = np.ones(2); tr = [x.copy()]; ws = [w.copy()]; F0 = P.evaluate(x[None])[0][0]
    for _ in range(300):
        G = P.grad(x); F = P.evaluate(x[None])[0][0]
        w, _, _ = gn.gradnorm_update(w, np.linalg.norm(G, axis=1), F, F0, 0.5, 0.025)
        x = x - 0.05 * (w @ G); tr.append(x.copy()); ws.append(w.copy())
    toy["gradnorm"] = dict(traj=np.array(tr).tolist(), w=np.array(ws).tolist(), f=P.evaluate(x[None])[0][0].tolist())
    net = {}
    for scale in (1.0, 10.0):
        for meth in ("fixed", "mgda", "gradnorm"):
            runs = [mtl_net.train(meth, steps=500, reg_scale=scale, seed=s) for s in range(10)]
            net[f"{meth}|scale={scale}"] = dict(Lc=np.mean([h["Lc"] for h in runs], 0).tolist(), Lr=np.mean([h["Lr"] for h in runs], 0).tolist(),
                                              Lc_final=[float(h["Lc"][-1]) for h in runs], Lr_final=[float(h["Lr"][-1]) for h in runs],
                                              w_final=np.mean([h["w"][-1] for h in runs], 0).tolist(),
                                              Lc0=float(np.mean([h["Lc"][0] for h in runs])), Lr0=float(np.mean([h["Lr"][0] for h in runs])))
    # plain GradNorm with the lecture's lr_w = 0.025 on the imbalanced network
    h = mtl_net.train("gradnorm", steps=500, reg_scale=10.0, seed=0, lr_w=0.025)
    net["gradnorm_lrw0.025|scale=10"] = dict(Lc=h["Lc"].tolist(), Lr=h["Lr"].tolist(), w=h["w"].tolist())
    h = mtl_net.train("fixed", steps=3, reg_scale=10.0, seed=0)
    net["grad_norms_scale10"] = dict(cls=h["gnorm_c"].tolist(), reg=h["gnorm_r"].tolist())
    return dict(toy=toy, net=net)


def main():
    out = {}
    print("D1", flush=True); out["D1"] = run_many(d1_jobs())
    print("D2", flush=True); out["D2"] = run_many(d2_jobs())
    print("D3", flush=True); out["D3"] = run_many(d3_jobs())
    print("D2 front-1 fraction", flush=True); out["D2_front1_fraction_M5"] = nsga2_front1_fraction("P10_M5", 72)
    out["D2_front1_fraction_M3"] = nsga2_front1_fraction("P10_M3", 92)
    print("D4", flush=True); out["D4"] = d4_mcdm()
    print("D5", flush=True); out["D5"] = d5_gradient()
    save_json(out, "taskD.json")
    print("saved taskD.json")


if __name__ == "__main__":
    main()
