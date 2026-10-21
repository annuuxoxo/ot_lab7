"""Runs every unit test (moo_utils, problems, solvers, 37 algorithms, 3 hand traces) and writes results/test_report.txt."""
import os, sys, time, importlib
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
lines = []; bad = 0
for mod in ("test_moo_utils", "test_problems", "test_solvers", "test_algorithms", "test_handtraces"):
    m = importlib.import_module(mod)
    for t in m.ALL:
        t0 = time.time()
        try:
            t(); status = "PASS"
        except Exception as e:
            status = "FAIL"; bad += 1; msg = f"{type(e).__name__}: {e}"
        lines.append(f"{status}  {mod}.{t.__name__}  ({time.time()-t0:.2f}s)" + ("" if status == "PASS" else "  " + msg))
        print(lines[-1])
summary = f"{len(lines) - bad} passed, {bad} failed, {len(lines)} tests"
print(summary)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
open(os.path.join(ROOT, "results", "test_report.txt"), "w").write("\n".join(lines + [summary]) + "\n")
sys.exit(1 if bad else 0)
