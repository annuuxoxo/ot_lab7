"""
Algorithm 27 -- MODE, Multi-Objective Differential Evolution (Xue, Sanderson & Graves 2003; evolutionary, Part 4 slide 48)

Lecture steps
  1. Initialise NP target vectors X_i and an archive of non-dominated solutions.
  2. DE mutation   V = X_r1 + F (X_r2 - X_r3)   (r1, r2, r3 distinct, != i).
  3. Binomial crossover  U_j = V_j if rand_j <= CR or j = j_rand, else X_i,j   (one variable -> U = V); clip U to the bounds.
  4. Evaluate U.  The trial REPLACES its target only if it dominates it (Deb's rules when constrained); if it is merely non-dominated
     the target survives and the trial is only offered to the archive.
  5. Archive: non-dominated feasible solutions, crowding-distance truncation.     6. Repeat.
Scripted randomness: step(st, script=dict(r=..., rand=..., jrand=...)) lets the hand trace of Task C inject the random numbers.
Variant selection='survival' (NOT the lecture rule): targets and trials are merged and NSGA-II survival (rank + crowding) keeps NP,
which is what rescues MODE from the collapse shown by the lecture rule on non-convex fronts.
Defaults: NP = 100, F = 0.5, CR = 0.9, archive = 100, selection = 'dominance' (the lecture rule).
"""
import numpy as np
import ops
from algorithms.base import Core, make_run, Archive

DEFAULTS = dict(NP=100, F=0.5, CR=0.9, archive=100)


class State(Core):
    def output(self):
        return self.arch.X, self.arch.F, self.arch.CV


def initialize(problem, seed=0, max_evals=10000, NP=100, F=0.5, CR=0.9, archive=100, constraint="penalty", X0=None,
               selection="dominance", **kw):
    st = State(problem, seed, max_evals, constraint)
    st.NP, st.Fde, st.CR, st.selection = NP, F, CR, selection
    st.X = problem.random_x(NP, st.rng) if X0 is None else np.asarray(X0, float)
    st.NP = len(st.X)
    st.F, st.CV, st.S = st.evaluate(st.X)
    st.arch = Archive(archive); st.arch.add(st.X, st.F, st.CV)
    st.replaced = 0
    return st


def evaluate(st, X):
    return st.evaluate(X)


def make_trials(st, script=None):
    NP = st.NP; U = np.empty_like(st.X, dtype=float)
    for i in range(NP):
        if script is not None:
            r = script["r"][i]; rand = script.get("rand", [None] * NP)[i]; jr = script.get("jrand", [None] * NP)[i]
        else:
            cand = [j for j in range(NP) if j != i]
            r = st.rng.choice(cand, 3, replace=False); rand = jr = None
        U[i] = ops.de_trial(st.problem, st.X, i, r, st.Fde, st.CR, st.rng, jrand=jr, rand=rand)
    return U


def select(st, U, Fu, CVu, Su):
    """Trial replaces target only if it dominates (in the selection objectives S)."""
    rep = np.all(Su <= st.S, axis=1) & np.any(Su < st.S, axis=1)
    st.X[rep], st.F[rep], st.CV[rep], st.S[rep] = U[rep], Fu[rep], CVu[rep], Su[rep]
    st.replaced += int(rep.sum())
    return rep


def step(st, script=None):
    U = make_trials(st, script)
    Fu, CVu, Su = st.evaluate(U)
    if st.selection == "survival":
        from algorithms import alg14_nsga2 as nsga2
        X = np.vstack([st.X, U]); Fa = np.vstack([st.F, Fu]); CV = np.concatenate([st.CV, CVu]); S = np.vstack([st.S, Su])
        keep = nsga2.environmental_selection(S, CV, st.NP, False)
        rep = keep >= st.NP
        st.X, st.F, st.CV, st.S = X[keep], Fa[keep], CV[keep], S[keep]
        st.replaced += int(rep.sum()); rep = np.zeros(st.NP, bool)
    else:
        rep = select(st, U, Fu, CVu, Su)
    st.arch.add(U, Fu, CVu)
    st.last_rep = rep
    return rep


run = make_run(initialize, step)
