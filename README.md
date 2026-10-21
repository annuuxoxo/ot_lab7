# Lab 7 – Multiobjective Optimization: 37 algorithms (roll no. 77)

All 37 algorithms of the Chapter 5 catalogue, the shared toolkit, the 14-problem bank, the experiments and the report
generator. Language: **Python 3.10+**. Algorithms, baselines and metrics use **only NumPy and the standard library**
(Matplotlib for plots). No optimisation / evolutionary-computation library is used or imported by any algorithm
(no pymoo, DEAP, scipy.optimize, …). Two non-algorithmic exceptions:

* `scikit-learn` is imported once, in `problems.P9`, **only to load the Wisconsin breast-cancer data**
  (the 5-NN classifier and the 5-fold cross-validation are our own code);
* `reportlab` builds the PDF report (`report/build_report.py`).

Roll number R = 77 → deep-dive algorithms **3 (Benson's method), 15 (NSGA-III), 27 (MODE)**.

## Layout

| path | content |
|---|---|
| `moo_utils.py` | dominance, Deb's constrained dominance, fast non-dominated sort, crowding distance, normalisation, 2-D hypervolume (+ Monte-Carlo, + contributions), IGD, GD, Spacing, Das–Dennis points |
| `problems.py` | P1–P14 (`make("P3")` …), reference fronts, normalisation bounds |
| `solvers.py` | own simplex LP, bounded Nelder–Mead + penalty, 1-D integer enumeration (used by the classical methods 1–7) |
| `ops.py` | SBX, polynomial mutation, DE, order crossover, swap/inversion, neighbour moves, tournaments |
| `mtl_net.py` | P13(b): two-head network with hand-written forward/backward passes and training loops |
| `algorithms/alg01_…alg37_…` | one module per algorithm: `DEFAULTS`, `initialize`, `evaluate`, `step` (one iteration), `run`, docstring with the lecture steps |
| `tests/` | `test_moo_utils`, `test_problems`, `test_solvers`, `test_algorithms` (37 tests, one per algorithm), `test_handtraces` (Task C traces) |
| `handtrace/trace_calc.py` | paper-and-pencil arithmetic of the three hand traces (no algorithm imports) |
| `experiments/` | `run_taskA.py`, `run_taskA_special.py`, `run_evidence.py`, `run_taskC.py`, `run_taskD.py`, figure scripts |
| `results/` | raw result files behind every table and plot (`*.json`, reference fronts in `results/ref/`, logs) |
| `figures/` | all plots used in the report |
| `report/` | `build_report.py` → `Lab7_Report.pdf` |

## How to run

```bash
pip install numpy matplotlib scikit-learn reportlab      # scikit-learn only loads P9's data, reportlab only builds the PDF
python3 run_all_tests.py                                  # 67 unit tests + the 3 hand traces (≈ 20 s)
python3 handtrace/trace_calc.py                           # prints / stores the hand-trace numbers

# reproduce the results (about 30–40 min on a 10-core laptop)
python3 build_refs.py                                     # reference fronts (dense sampling P4–P6, pooled long runs P8–P9)
python3 experiments/run_taskA.py && python3 experiments/run_taskA.py p9 && python3 experiments/run_taskA_special.py
python3 experiments/run_evidence.py                       # evidence numbers for Task B
python3 experiments/run_taskC.py                          # sensitivity studies of algorithms 3, 15, 27
python3 experiments/run_taskD.py && python3 experiments/make_figures_D.py
python3 report/build_report.py                            # → Lab7_Report.pdf
```

Single algorithm, e.g. NSGA-II on ZDT2 with the fixed seed 0:

```python
import problems as pb, algorithms as A
P = pb.make("P3")
r = A.load(14).run(P, seed=0, max_evals=10000)      # r.F = non-dominated front (minimisation form), r.X = decision vectors
print(P.hv(r.F), P.igd(r.F), P.spacing(r.F))
```

## Seeds and budgets

* Task A: seed 0, 10,000 function evaluations for every population / swarm algorithm; classical methods run their sweep to the end.
* Tasks C and D: seeds 0–9 (10 independent runs, mean ± std). Reference-front builds: seeds 0–4.
* P8 data: `numpy.random.seed(7)`; coordinates are printed in the report.

Fixed seeds make every number reproducible on the same NumPy version (default_rng streams are stable across platforms
for the generators used).

## Notes

* The code was written for the lecture versions of the algorithms; deliberate deviations (e.g. IBEA convention, RVEA
  penalty form, MOSA parallel chains, MODE variant) are documented in each module's docstring and in Appendix B of the report.
* Library output is never used as a result.
