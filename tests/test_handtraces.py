"""Deep-dive algorithms 3, 15 and 27 (roll no. 77): the code must reproduce the hand traces of handtrace/trace_calc.py."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "handtrace"))
import numpy as np
import trace_calc as tc
import problems as pb
import moo_utils as mu
from algorithms import alg03_benson as a3, alg15_nsga3 as a15, alg27_mode as a27


def test_trace_benson_algorithm3():
    hand = tc.benson()
    P = pb.P1()
    st = a3.initialize(P, settings=[(h["lam"], h["z"]) for h in hand])
    while not st.done:
        a3.step(st)
    for h, got in zip(hand, st.info):
        assert np.allclose(got["x"][0], h["x"], atol=2e-3), (got["x"], h["x"])
        assert np.allclose(got["F"], h["f"], atol=2e-2), (got["F"], h["f"])
        assert np.allclose(got["s"], h["s"], atol=2e-2)
        lam = np.array(h["lam"]); assert abs(float(lam @ np.array(got["s"])) - h["phi"]) < 2e-2


def _nsga3_check(it, name_pts):
    names = it["names"]
    F = np.array([tc.drone(*name_pts[n]) for n in names])
    Z = mu.das_dennis(3, 1)
    idx, info = a15.select_survivors(F, 4, Z, None, None, "max")          # rng=None: deterministic tie-breaks like the hand trace
    assert [sorted(names[i] for i in f) for f in info["fronts"]] == [sorted(f) for f in it["fronts"]]
    assert sorted(names[i] for i in idx) == sorted(it["survivors"]), ([names[i] for i in idx], it["survivors"])
    assert np.allclose(info["ideal"], it["ideal"], atol=1e-3) and np.allclose(info["intercepts"], it["intercepts"], atol=1e-3)
    St = [names[i] for i in info["St"]]
    for k, n in enumerate(St):                                           # nearest reference line and its distance
        d = it["dist"][n]
        assert info["near"][k] == it["nearest"][n]
        assert abs(info["dist"][k] - d[it["nearest"][n]]) < 2e-3


def test_trace_nsga3_algorithm15():
    hand = tc.nsga3()
    pts = {**hand["P0"], **hand["Q0"]}
    _nsga3_check(hand["it1"], pts)
    P1 = {n: pts[n] for n in hand["it1"]["survivors"]}
    pts2 = {**P1, **hand["Q1"]}
    _nsga3_check(hand["it2"], pts2)


def test_trace_mode_algorithm27():
    hand = tc.mode()
    P = pb.P1()
    st = a27.initialize(P, seed=0, max_evals=10 ** 6, NP=4, CR=0.9, X0=np.array(hand["pop0"])[:, None])
    assert sorted(st.arch.X[:, 0].tolist()) == hand["archive0"]
    for it, Fval in zip(hand["iterations"], (0.5, 0.6)):
        st.Fde = Fval
        script = dict(r=[tuple(a - 1 for a in row["r"]) for row in it["rows"]])
        a27.step(st, script)
        assert np.allclose(st.X[:, 0], it["pop_after"]), (st.X[:, 0], it["pop_after"])
        assert np.allclose(sorted(st.arch.X[:, 0]), it["archive"], atol=1e-4)
        outcomes = ["replace" if r else "keep" for r in st.last_rep]
        assert outcomes == [row["outcome"].split(" ")[0] for row in it["rows"]]


ALL = [v for k, v in dict(globals()).items() if k.startswith("test_")]
if __name__ == "__main__":
    for t in ALL:
        t(); print("PASS", t.__name__)
