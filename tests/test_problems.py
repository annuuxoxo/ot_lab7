"""Hand-checkable sanity values from Section 4 of the lab sheet."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import problems as pb
import moo_utils as mu

close = lambda a, b, t=1e-6: np.allclose(a, b, atol=t, rtol=1e-4)

def test_P1():
    p = pb.P1(); F, _ = p.evaluate([[5.0]]); assert close(F[0], (4, 4))
    F, _ = p.evaluate([[3.0]]); assert close(F[0], (0, 16))
    f = p.true_front(50); assert close(f[:, 1], (4 - np.sqrt(f[:, 0])) ** 2)

def test_P2():
    p = pb.P2(); assert close(p.evaluate([[2.0]])[0][0], (50, 4)); assert close(p.evaluate([[10.0]])[0][0], (10, 100))
    f = p.true_front(30); assert close(f[:, 1], 10000 / f[:, 0] ** 2)

def test_P3():
    p = pb.P3(); x = np.zeros((1, 10)); x[0, 0] = 0.5
    assert close(p.evaluate(x)[0][0], (0.5, 0.75))

def test_P4():
    p = pb.P4(); F, cv = p.evaluate([[0, 0, 1.0, 0]]); assert close(F[0], (0.09, -0.15))
    ref = p.reference_front(); i = np.argmin(ref[:, 0]); assert ref[i, 1] > -0.15 and ref[i, 0] < 0.0071   # mixture

def test_P5():
    p = pb.P5(); F, cv = p.evaluate([[0.01, 0.05]])
    assert close(F[0], (3.925, 0.016), 1e-6) and cv[0] == 0 and close(p.stress([[0.01, 0.05]]), 2.4e8, 1)

def test_P6():
    p = pb.P6(); F, cv = p.evaluate([[0, 0], [5, 3]])
    assert close(F, [(0, 50), (136, 4)]) and (cv == 0).all()

def test_P7():
    p = pb.P7(); F, cv = p.evaluate([[10, 80], [70, 20]]); assert close(p.natural(F), [(8, 96), (56, 24)]) and (cv == 0).all()

def test_P8():
    p = pb.P8(); assert p.xy.shape == (15, 2)
    F, _ = p.evaluate(np.array([np.arange(15)])); assert F[0, 0] > 0 and F[0, 1] > 0
    assert np.allclose(p.T, p.T.T)

def test_P9():
    p = pb.P9(); e = p.baseline_error(); assert 0.0 < e < 0.06, e       # all-ones baseline accuracy ~ 97%
    F, cv = p.evaluate(np.zeros((1, 30))); assert cv[0] == 1

def test_P10():
    for M in (3, 5):
        p = pb.P10(M); x = np.random.rand(20, p.n_var); x[:, M - 1:] = 0.5
        F, _ = p.evaluate(x); assert close((F ** 2).sum(1), 1.0)       # g = 0  ->  sum f^2 = 1

def test_P11():
    p = pb.P11(); F, cv = p.evaluate([[300, 200], [0, 400], [400, 0]])
    assert close(p.natural(F), [(2400, 500), (2400, 400), (1600, 400)]) and (cv <= 1e-12).all()

def test_P14():
    p = pb.P14(); F, _ = p.evaluate([[13, 40], [13, 90]]); N = p.natural(F)
    assert close(N[0, 0], 1.53) and abs(N[0, 1] - 8.99) < 0.01 and close(N[1, 0], 2.13) and abs(N[1, 1] - 20.2) < 0.1

def test_P13a():
    p = pb.P13a(); g = p.grad([0, 5]); assert close(g[0], [-6, 4]) and close(g[1], [6, 16])

ALL = [v for k, v in dict(globals()).items() if k.startswith("test_")]
if __name__ == "__main__":
    for t in ALL:
        t(); print("PASS", t.__name__)
