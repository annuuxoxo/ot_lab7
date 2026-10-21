"""Unit tests for moo_utils on the lecture examples (answers verifiable by hand)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import moo_utils as mu


def test_car_pareto_front():
    # Part 1 slide 9: min cost, max mileage.  Convert mileage to -mileage.
    cars = {"A": (5, 15), "B": (8, 22), "C": (6, 18), "D": (10, 25), "E": (7, 12), "F": (9, 20)}
    names = list(cars)
    F = np.array([(c, -m) for c, m in cars.values()], float)
    mask = mu.non_dominated_mask(F)
    front = {n for n, k in zip(names, mask) if k}
    assert front == {"A", "B", "C", "D"}, front
    D = mu.dominance_matrix(F)
    assert D[names.index("A"), names.index("E")] and D[names.index("B"), names.index("F")]


def test_debs_constrained_table():
    # Part 1 slide 23: A(feas,3,6) D(feas,4,5) B(infeas CV2,1,2) C(infeas CV5,0.5,1)
    A, Dd, B, C = ((3, 6), 0), ((4, 5), 0), ((1, 2), 2), ((0.5, 1), 5)
    cd = lambda x, y: mu.constrained_dominates(x[0], x[1], y[0], y[1])
    assert cd(A, B) and cd(A, C) and cd(Dd, B) and cd(Dd, C)          # rule 1
    assert not cd(A, Dd) and not cd(Dd, A)                              # rule 2: trade-off
    assert cd(B, C) and not cd(C, B)                                    # rule 3: smaller CV wins
    F = np.array([x[0] for x in (A, Dd, B, C)], float)
    CV = np.array([x[1] for x in (A, Dd, B, C)], float)
    fronts, rank = mu.fast_non_dominated_sort(F, CV)
    assert [sorted(f.tolist()) for f in fronts] == [[0, 1], [2], [3]]


def test_fast_nds_fronts():
    # Part 2 slide 4 / Part 3 bridge example: x=1,2,3,4 -> x=4 is dominated
    F = np.array([(1, -96), (4, -99), (9, -100), (16, -99)], float)
    fronts, rank = mu.fast_non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1, 2] and fronts[1].tolist() == [3]


def test_crowding_distance_lecture():
    # Part 4 foundations slide 5
    F = np.array([(32, 48), (34, 45), (36, 42), (36.8, 40.8), (40, 36)], float)
    d = mu.crowding_distance(F)
    assert np.isinf(d[0]) and np.isinf(d[4])
    assert np.allclose(d[1:4], [1.00, 0.70, 1.00], atol=0.02), d


def test_hypervolume_5E_example():
    ref = (1.1, 1.1)
    true = [(0, 1), (.25, .5), (.5, 1 - np.sqrt(.5)), (.75, 1 - np.sqrt(.75)), (1, 0)]
    obt = [(.02, .87), (.24, .55), (.48, .31), (.80, .20), (.95, .08)]
    assert abs(mu.hv2d(np.array(true), ref) - 0.7283) < 5e-4
    assert abs(mu.hv2d(np.array(obt), ref) - 0.7234) < 5e-4
    assert abs(mu.hv_mc(np.array(obt), ref, n_samples=400000) - 0.7234) < 0.01   # MC agrees


def test_igd_5E_example():
    true = np.array([(0, 1), (.25, .5), (.5, 1 - np.sqrt(.5)), (.75, 1 - np.sqrt(.75)), (1, 0)], float)
    obt = np.array([(.02, .87), (.24, .55), (.48, .31), (.80, .20), (.95, .08)], float)
    # slide quotes 0.0772 for the matched 1-to-1 distances; nearest-neighbour IGD is <= that
    assert 0.07 < mu.igd(obt, true) < 0.0775


def test_hv_contribution_2d():
    F = np.array([(1, 3), (2, 2), (3, 1), (3, 3)], float)
    c = mu.hv_contributions_2d(F, (4, 4))
    assert np.allclose(c, [1, 1, 1, 0]), c


def test_das_dennis_count_and_sum():
    P = mu.das_dennis(3, 2)
    assert len(P) == 6 and np.allclose(P.sum(1), 1)          # C(4,2)=6 (5E slide 5)
    assert len(mu.das_dennis(5, 4)) == 70


def test_spacing_even_vs_clustered():
    even = np.array([(i / 10, 1 - i / 10) for i in range(11)])
    clus = np.array([(0, 1), (0.05, .95), (0.1, .9), (0.9, .1), (1, 0)])
    assert mu.spacing(even) < 1e-9 < mu.spacing(clus)


ALL = [v for k, v in dict(globals()).items() if k.startswith("test_")]
if __name__ == "__main__":
    for t in ALL:
        t(); print("PASS", t.__name__)
