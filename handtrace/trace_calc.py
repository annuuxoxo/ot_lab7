"""
Hand traces for the three deep-dive algorithms of roll number 77:  3 (Benson), 15 (NSGA-III), 27 (MODE).

This file does the arithmetic with the `math` module only -- it does NOT import any algorithm module -- so it plays the role of the
paper-and-pencil calculation.  tests/test_handtraces.py then checks that our implementations reproduce every number printed here.
Run:  python3 handtrace/trace_calc.py   (writes results/handtrace.json)
"""
import json, math, os

R = lambda v, d=4: round(v, d)


# ===================================================================== Algorithm 3: Benson, problem P1
def benson():
    f1 = lambda x: (x - 3) ** 2
    f2 = lambda x: (x - 7) ** 2
    out = []
    # Setting 1: z* = (16,16), lambda = (0.5,0.5).   Objective Phi(x) = 0.5(16-f1)+0.5(16-f2) = 16 - (x^2 - 10x + 29)
    # Phi'(x) = -2x + 10 = 0 -> x = 5.   Constraints s1 = 16-(x-3)^2 >= 0 -> -1<=x<=7 ; s2 = 16-(x-7)^2>=0 -> 3<=x<=11 ; box [1,10]
    x = 5.0
    out.append(dict(z=(16, 16), lam=(0.5, 0.5), x=x, f=(f1(x), f2(x)), s=(16 - f1(x), 16 - f2(x)),
                    phi=0.5 * (16 - f1(x)) + 0.5 * (16 - f2(x)), feasible_interval=(3, 7)))
    # Setting 2: z* = (16,16), lambda = (0.8,0.2).  Phi = 16 - (0.8 f1 + 0.2 f2) = 16 - (x^2 - 7.6 x + 17) ; Phi' = -2x + 7.6 = 0 -> x = 3.8
    x = 3.8
    out.append(dict(z=(16, 16), lam=(0.8, 0.2), x=x, f=(f1(x), f2(x)), s=(16 - f1(x), 16 - f2(x)),
                    phi=0.8 * (16 - f1(x)) + 0.2 * (16 - f2(x)), feasible_interval=(3, 7)))
    # Setting 3 (reference point BINDS): z* = (4,10), lambda=(0.5,0.5).  s1>=0 -> (x-3)^2<=4 -> 1<=x<=5 ; s2>=0 -> (x-7)^2<=10 -> 3.838<=x<=10.16
    # feasible interval [3.838, 5]; Phi' = -2x+10 >= 0 on the interval so the optimum is the right end x = 5 where s1 = 0.
    lo = 7 - math.sqrt(10)
    x = 5.0
    out.append(dict(z=(4, 10), lam=(0.5, 0.5), x=x, f=(f1(x), f2(x)), s=(4 - f1(x), 10 - f2(x)),
                    phi=0.5 * (4 - f1(x)) + 0.5 * (10 - f2(x)), feasible_interval=(R(lo), 5.0)))
    return out


# ===================================================================== Algorithm 15: NSGA-III on the drone problem (3 objectives)
def drone(x1, x2):
    return (0.5 + 0.6 * x1 + 0.004 * x2, 20 + 40 * x1 + 0.3 * x2, -(15 + 0.05 * x2 - 8 * x1))


def dominates(a, b):
    return all(p <= q for p, q in zip(a, b)) and any(p < q for p, q in zip(a, b))


def fronts_of(F):
    n = len(F); left = set(range(n)); fr = []
    while left:
        cur = [i for i in sorted(left) if not any(dominates(F[j], F[i]) for j in left if j != i)]
        fr.append(cur); left -= set(cur)
    return fr


def nsga3_iteration(names, pts, N=4):
    F = [drone(*pts[n]) for n in names]
    fr = fronts_of(F)
    St = []; l = 0
    for l, f in enumerate(fr):
        St += f
        if len(St) >= N:
            break
    n_acc = len(St) - len(fr[l])
    ideal = [min(F[i][m] for i in St) for m in range(3)]
    T = {i: [F[i][m] - ideal[m] for m in range(3)] for i in St}                 # translated objectives f'
    icpt = [max(T[i][m] for i in St) for m in range(3)]                          # 'max' normalisation = nadir estimate
    Nn = {i: [T[i][m] / icpt[m] for m in range(3)] for i in St}
    # p = 1 -> reference points are the three unit axes; perpendicular distance to axis m = norm of the other two coordinates
    D = {i: [math.sqrt(sum(Nn[i][k] ** 2 for k in range(3) if k != m)) for m in range(3)] for i in St}
    near = {i: min(range(3), key=lambda m: (D[i][m], m)) for i in St}
    rho = [0, 0, 0]
    for i in St[:n_acc]:
        rho[near[i]] += 1
    chosen = list(St[:n_acc]); avail = list(St[n_acc:]); excluded = set(); log = []
    while len(chosen) < N:
        refs = [m for m in range(3) if m not in excluded]
        mn = min(rho[m] for m in refs)
        j = [m for m in refs if rho[m] == mn][0]
        mem = sorted(i for i in avail if near[i] == j)
        if not mem:
            excluded.add(j); log.append(f"ref {j}: no candidate -> excluded"); continue
        pick = min(mem, key=lambda i: D[i][j]) if rho[j] == 0 else mem[0]
        log.append(f"ref {j} (rho={rho[j]}): candidates {[names[i] for i in mem]} -> {names[pick]}")
        chosen.append(pick); avail.remove(pick); rho[j] += 1
    return dict(names=names, F=[[R(v, 3) for v in f] for f in F], fronts=[[names[i] for i in f] for f in fr],
                St=[names[i] for i in St], n_accepted=n_acc, ideal=[R(v, 3) for v in ideal],
                translated={names[i]: [R(v, 3) for v in T[i]] for i in St}, intercepts=[R(v, 3) for v in icpt],
                normalised={names[i]: [R(v, 4) for v in Nn[i]] for i in St},
                dist={names[i]: [R(v, 4) for v in D[i]] for i in St}, nearest={names[i]: near[i] for i in St},
                rho_final=rho, log=log, survivors=[names[i] for i in chosen])


def nsga3():
    P0 = {"D1": (0.6, 120), "D2": (0.8, 150), "D5": (0.7, 250), "D6": (1.4, 300)}
    Q0 = {"a": (0.65, 110), "b": (0.75, 240), "c": (0.95, 195), "d": (1.1, 260)}
    pts = {**P0, **Q0}
    it1 = nsga3_iteration(list(P0) + list(Q0), pts)
    P1 = {n: pts[n] for n in it1["survivors"]}
    Q1 = {"e": (1.37, 145), "f": (1.34, 200), "g": (1.32, 255), "h": (0.62, 245)}
    pts2 = {**P1, **Q1}
    it2 = nsga3_iteration(list(P1) + list(Q1), pts2)
    return dict(P0=P0, Q0=Q0, it1=it1, Q1=Q1, it2=it2)


# ===================================================================== Algorithm 27: MODE on P1, population 4
def mode():
    f = lambda x: ((x - 3) ** 2, (x - 7) ** 2)
    pop = [1.0, 4.0, 8.0, 10.0]
    lo, hi = 1.0, 10.0
    F = [0.5, 0.6]
    donors = [[(1, 2, 3), (2, 3, 0), (1, 0, 3), (2, 1, 0)],
              [(1, 2, 3), (0, 2, 3), (1, 3, 0), (2, 0, 1)]]
    arch = []
    def nd_filter(pts):
        keep = []
        for p in pts:
            fp = f(p)
            if not any(dominates(f(q), fp) for q in pts if q != p):
                keep.append(p)
        return sorted(set(keep))
    arch = nd_filter(pop)
    rec = dict(pop0=pop[:], F0=[f(x) for x in pop], archive0=arch[:], iterations=[])
    for it in range(2):
        trials = []; rows = []
        for i, x in enumerate(pop):
            r1, r2, r3 = donors[it][i]
            v = pop[r1] + F[it] * (pop[r2] - pop[r3])
            u = min(max(v, lo), hi)
            fu, fx = f(u), f(x)
            if dominates(fu, fx):
                outcome = "replace"
            elif dominates(fx, fu):
                outcome = "keep (target dominates trial)"
            else:
                outcome = "keep (non-dominated, trial -> archive)"
            rows.append(dict(i=i + 1, x=x, r=(r1 + 1, r2 + 1, r3 + 1), F=F[it], V=R(v), U=R(u), fU=[R(a) for a in fu], fX=[R(a) for a in fx], outcome=outcome))
            trials.append((i, u, outcome))
        for i, u, o in trials:
            if o == "replace":
                pop[i] = u
        arch = nd_filter(arch + [u for _, u, _ in trials])
        rec["iterations"].append(dict(rows=rows, pop_after=pop[:], F_after=[[R(a) for a in f(x)] for x in pop], archive=[R(a) for a in arch]))
    return rec


if __name__ == "__main__":
    res = dict(benson=benson(), nsga3=nsga3(), mode=mode())
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "handtrace.json")
    json.dump(res, open(path, "w"), indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    print(json.dumps(res["benson"], indent=1, default=str))
    for k in ("it1", "it2"):
        t = res["nsga3"][k]
        print(k, t["fronts"], "St", t["St"], "ideal", t["ideal"], "icpt", t["intercepts"], "survivors", t["survivors"], t["log"], "rho", t["rho_final"])
    for it in res["mode"]["iterations"]:
        for r in it["rows"]:
            print(r)
        print(it["pop_after"], it["archive"])
