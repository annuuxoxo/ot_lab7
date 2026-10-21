"""Appendix: deviations from the lecture versions, file list, how to reproduce."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl import *


def appendix():
    o = [H1("Appendix A &mdash; How to reproduce"),
         P("<font face='Courier'>python3 run_all_tests.py</font> (67 tests, 3 hand traces) &middot; <font face='Courier'>python3 build_refs.py</font> &middot; "
           "<font face='Courier'>python3 experiments/run_taskA.py [p9]</font> &middot; <font face='Courier'>run_taskA_special.py</font> &middot; <font face='Courier'>run_evidence.py</font> &middot; "
           "<font face='Courier'>run_taskC.py</font> &middot; <font face='Courier'>run_taskD.py</font> &middot; <font face='Courier'>make_figures_D.py / make_figures_E.py / make_flow.py</font> &middot; "
           "<font face='Courier'>report/build_report.py</font>. Seeds: Task A seed 0, Tasks C/D seeds 0-9, reference fronts seeds 0-4; P8 data numpy seed 7. The raw numbers behind every "
           "table and plot are in <i>results/*.json</i> (taskA, taskA_p9, taskA_special, evidence, taskC, taskD, handtrace) and the reference fronts in <i>results/ref/</i>.", "p"),
         H1("Appendix B &mdash; Deliberate deviations from the lecture versions and known limitations"),
         tbl([["alg", "what we do", "why"],
              ["1, 4", "objectives scaled by (nadir - ideal) before weighting (normalize = True); raw units available (normalize = False)", "the lecture lists scale-sensitivity as a weakness; we measure it (Task B) but use the fair version in comparisons"],
              ["3", "reference-point form of the lecture (maximise &Sigma;&lambda;s, s &ge; 0); two sweeps: &lambda; and z*", "the lecture's form equals a box-constrained weighted sum, so z* must move to reach non-convex parts"],
              ["5", "utilities are our stand-ins for an elicited decision maker (mean-variance utility for P4, target-closeness for P14)", "no real DM available"],
              ["6, 7", "own simplex LP (Bland's rule), deviation variables as LP columns", "scipy.linprog not used"],
              ["12 (NSGA)", "dummy fitness of front r+1 = 0.9 x smallest shared fitness of front r (Srinivas-Deb) instead of a fixed &Delta;F", "keeps the front order whatever the niche counts"],
              ["13 (NPGA)", "lecture variant: the two candidates are compared with each other; the comparison set (10 % of N) only measures niche counts", "as on lecture slide 27; Horn's original compares both against the set"],
              ["15", "extreme-point normalisation (paper) by default; 'max' normalisation (5E) for the hand trace; Das-Dennis p = 12 / 4 for k = 3 / 5 (single layer)", "hand trace must be checkable by pencil"],
              ["20, 25", "simplified ant models: one ant per sub-problem (20); deposit = Q/(f/f<sub>NN</sub>)/#depositors, &tau; clipped (25)", "stability within 10,000 evaluations"],
              ["21 and others", "algorithms other than 21 use an exact-style penalty S = F + 1000 &middot; CV &middot; (nadir - ideal); 21 uses Deb's rules; 'ignore' mode exists for the experiment", "so that every algorithm can run on constrained problems"],
              ["22", "leader = archive member picked by a crowding-distance tournament (Coello's adaptive grid not used); w = 0.4", "lecture says crowding distance"],
              ["26 (MOSA)", "10 independent walkers share one archive; objectives normalised, T<sub>0</sub> = 1, geometric cooling to 10<sup>-3</sup>; chains = 1 gives the lecture algorithm", "one walker cannot cover a front in 10,000 evaluations"],
              ["27 (MODE)", "default = lecture rule; selection = 'survival' is an extra variant for the comparison", "to show the collapse and its cure"],
              ["31 (RVEA)", "penalty form of the 5E slide, 1 + &alpha;(t/t<sub>max</sub>)&theta;/&gamma;, &alpha; = 2 (the paper uses M (t/t<sub>max</sub>)<sup>&alpha;</sup>)", "follow the lecture; we checked the paper's form too: IGD 0.145 vs 0.174 (M = 3)"],
              ["32 (IBEA)", "paper's fitness F(x) = &Sigma;<sub>y</sub> -exp(-I(y,x)/&kappa;), delete the smallest; objectives normalised; &kappa; = 0.05", "the lecture's dual form reproduces the slide numbers (unit test) but collapses in practice"],
              ["33, 34", "hypervolume contributions exact for k = 2, Monte-Carlo (20,000 / 2,000 samples) otherwise; HypE shares each sample equally among its dominators (no &alpha;<sub>i</sub> weights)", "as on the 5E slides"],
              ["35", "NSGA-II fronts first, eps-clearing order replaces crowding in the last front; eps = 0.01 (normalised)", "lecture eps = 0.15 is for 6 points"],
              ["37", "in the network experiment lr<sub>w</sub> = 0.001 (the lecture's 0.025 saturates the weights because the gradient norms are 50x larger); both are reported", "see Task D5"],
              ["P8, P9, P4-P6", "IGD uses a best-known front (dense sampling for P4-P6, pooled long runs for P8, P9; marked *)", "no closed form"],
              ["P9", "features standardised with full-data mean / std (a mild leakage into the CV estimate that is the same for every algorithm)", "simplicity"],
              ["Hypervolume", "Monte-Carlo (200,000 fixed samples) for k &ge; 3", "allowed by the lab sheet; &plusmn; 0.003 sampling error"]],
             [20, 90, 66], True)]
    return o
