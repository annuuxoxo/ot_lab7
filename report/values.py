"""Accessors for the numbers quoted in the report (everything comes from results/*.json)."""
import os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def _j(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default


TA = (_j("taskA.json", []) or []) + (_j("taskA_p9.json", []) or [])
TD = _j("taskD.json", {}) or {}
EV = _j("evidence.json", {}) or {}
TC = _j("taskC.json", {}) or {}
TAS = _j("taskA_special.json", {}) or {}


def run(alg, prob, tag=None):
    for r in TA:
        if r["alg"] == alg and r["prob"] == prob and (tag is None or r["tag"] == tag):
            return r
    raise KeyError((alg, prob))


def A(alg, prob, met, d=3):
    r = run(alg, prob)
    v = r[met]
    return f"{v:.{d}f}"


def _grp(runs, alg, tag, prob):
    return [r for r in runs if r["alg"] == alg and r["prob"] == prob and (tag is None or r["tag"] == tag)]


def D1(alg, prob, met, tag="", d=3, std=False):
    rs = _grp(TD["D1"], alg, tag, prob)
    v = np.array([r[met] for r in rs], float)
    m, s = float(np.nanmean(v)), float(np.nanstd(v))
    return f"{m:.{d}f} &plusmn; {s:.{d}f}" if std else f"{m:.{d}f}"


def D1v(alg, prob, met, tag=""):
    rs = _grp(TD["D1"], alg, tag, prob)
    return float(np.nanmean([r[met] for r in rs]))


def rank(alg, prob, met="hv", tag=""):
    """rank (1 = best) of the algorithm among all D1 algorithms on this problem."""
    groups = {}
    for r in TD["D1"]:
        if r["prob"] == prob:
            groups.setdefault((r["alg"], r["tag"]), []).append(r[met])
    mean = {k: float(np.nanmean(v)) for k, v in groups.items()}
    order = sorted(mean, key=lambda k: (np.isnan(mean[k]), -mean[k] if met == "hv" else mean[k]))
    return order.index((alg, tag)) + 1, len(order)


def D2(alg, key, met, d=3, std=False):
    rs = _grp(TD["D2"], alg, None, key)
    v = np.array([r[met] for r in rs], float)
    m, s = float(np.nanmean(v)), float(np.nanstd(v))
    return f"{m:.{d}f} &plusmn; {s:.{d}f}" if std else f"{m:.{d}f}"


def D3(alg, met, d=3, std=False):
    rs = _grp(TD["D3"], alg, None, "P8")
    v = np.array([r[met] for r in rs], float)
    m, s = float(np.nanmean(v)), float(np.nanstd(v))
    return f"{m:.{d}f} &plusmn; {s:.{d}f}" if std else f"{m:.{d}f}"
