import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, solvers as sv

def test_lp_production():
    # P11: max 4A+6B s.t. 2A+3B<=1200, 4A+2B<=1600  -> profit 2400
    x, f, st = sv.linprog([-4, -6], [[2, 3], [4, 2]], [1200, 1600], bounds=[(0, None)] * 2)
    assert st == "optimal" and abs(-f - 2400) < 1e-6, (x, f, st)

def test_lp_equality_and_lower_bounds():
    x, f, st = sv.linprog([1, 1], A_eq=[[1, 1]], b_eq=[5], bounds=[(1, 4), (2, 6)])
    assert st == "optimal" and abs(f - 5) < 1e-9 and 1 - 1e-9 <= x[0] <= 3 + 1e-9

def test_lp_infeasible():
    _, _, st = sv.linprog([1], A_ub=[[1], [-1]], b_ub=[1, -3], bounds=[(0, None)])
    assert st == "infeasible"

def test_nm_constrained():
    f = lambda x: (x[0] - 3) ** 2 + (x[1] - 3) ** 2
    g = lambda x: x[0] + x[1] - 4            # x+y <= 4  -> optimum (2,2)
    x, fv, v = sv.minimize(f, [0, 0], [10, 10], [g], seed=1)
    assert np.allclose(x, [2, 2], atol=1e-3) and v < 1e-6, (x, v)

def test_integer_1d():
    x, f, v = sv.minimize_integer_1d(lambda x: abs(x[0] - 3.4), 1, 10)
    assert x[0] == 3

ALL = [v for k, v in dict(globals()).items() if k.startswith("test_")]
if __name__ == "__main__":
    for t in ALL:
        t(); print("PASS", t.__name__)
