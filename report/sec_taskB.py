"""Task B master table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl import *
import algorithms as A
import taskB_part1, taskB_part2, taskB_part3
import values as V

FAMILY = {1: "Classical", 8: "Ranking / MCDM", 10: "Non-elitist MOEAs", 14: "Elitist MOEAs", 21: "Constrained", 22: "Swarm / heuristics",
          31: "Many-objective, indicator, preference", 36: "Gradient-based"}


def bullets(lst):
    return "<br/>".join("&bull; " + x for x in lst)


def table():
    ent = {}
    ent.update(taskB_part1.entries()); ent.update(taskB_part2.entries()); ent.update(taskB_part3.entries())
    rows = [["#", "algorithm", "what it does (own words)", "advantages (evidence: Fig./number from our runs, or argument)", "disadvantages", "best choice", "avoid"]]
    extra = []
    for n, name, _, fam in A.CATALOGUE:
        e = ent[n]
        assert len(e["adv"]) >= 4 and len(e["dis"]) >= 4, n
        rows.append([str(n), f"<b>{name}</b><br/><font size=5.5>{fam}</font>", e["what"], bullets(e["adv"]), bullets(e["dis"]), e["best"], e["avoid"]])
        if n in FAMILY:
            extra.append(("SPAN", len(rows) - 1, FAMILY[n]))
    # family header rows inserted as separate rows
    out_rows = [rows[0]]
    marks = {r: t for _, r, t in extra}
    for i, r in enumerate(rows[1:], 1):
        if i in marks:
            out_rows.append([f"<b>{marks[i]}</b>", "", "", "", "", "", ""])
        out_rows.append(r)
    style = []
    for i, r in enumerate(out_rows):
        if i and r[1] == "" and r[0].startswith("<b>"):
            style += [("SPAN", (0, i), (-1, i)), ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#c9d6ea"))]
    t = tbl([[Paragraph(c, S["tiny"]) if isinstance(c, str) else c for c in r] for r in out_rows], [6, 18, 46, 80, 70, 30, 27], True, extra=style, zebra=False)
    return t


def family_paragraph():
    return [H3("Half-page synthesis: how the families differ"),
            P("<b>Diversity.</b> Classical methods have none of their own: spread equals the quality of the sweep (equal weight steps gave gaps up to "
              f"{(V.EV['ws_p1']['gap_ratio']-1)*100:.0f} % apart on P1). Non-elitist EAs use fitness sharing or niche tournaments, which work only with a tuned niche radius (MOGA's P3 HV ranges from "
              f"{V.EV['moga_sigma']['sigma=0.02']['hv']:.2f} to {V.EV['moga_sigma']['sigma=0.6']['hv']:.2f}); VEGA has none and collapses. Elitist EAs build it into selection: crowding distance (NSGA-II), "
              "k-th neighbour density (SPEA2), grid (PESA-II), reference points / vectors (NSGA-III, RVEA), weight vectors (MOEA/D). Swarm / heuristic methods reuse an archive with crowding-distance truncation, so their spread is almost uniform (Spacing 0.002-0.005 on P3). "
              "Indicator methods get it from the indicator itself (SMS-EMOA best Spacing on P14, IBEA none at all).", "p"),
            P("<b>Convergence.</b> Classical methods converge exactly per solve but need many solves (11-21 points cost 230,000-270,000 evaluations on ZDT2 with a derivative-free sub-solver). Elitist and swarm methods converge steadily "
              "and reach IGD &lt; 0.01 on P3/P5 in 10,000 evaluations; DE-based methods (MOEA/D-DE, MODE with survival) and MOBA converge fastest. Non-elitist methods converge slowly or not at all (NPGA, VEGA). "
              "Hybrid and indicator methods buy accuracy with runtime (SMS-EMOA 15 s vs 0.6 s).", "p"),
            P("<b>Constraints.</b> Only one algorithm has its own rule (21); all others in our runs use a penalty, which worked well on the two-variable P5 but made indicator-based methods (IBEA, HypE) and reference-point methods (NSGA-III) "
              "much worse on P5 (IGD 0.12-0.25 versus 0.005). Without any rule NSGA-II returns infeasible points on Binh-Korn; classical methods hand constraints to the sub-solver.", "p"),
            P("<b>Non-convex fronts.</b> Weighted sum and the lambda-form of Benson cannot reach a dent; epsilon-constraint, Tchebycheff weight metric, Benson with a moving reference point, and all dominance / decomposition-Tchebycheff based EAs can. "
              "Nonetheless, selection rules can still fail: MODE (lecture rule) and MO-TLBO collapsed on ZDT2. <b>Many objectives.</b> Dominance-and-crowding (NSGA-II) fails beyond 3 objectives; decomposition (MOEA/D), reference points / vectors, and "
              "hypervolume estimation (HypE) are the remedies, with different costs (H grows combinatorially; HypE needs sampling).", "p")]
