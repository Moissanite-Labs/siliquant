"""The predictions of PREREG.md as code, so that every verdict is computed and none is read off a table by eye.

Each function returns (holds, text). The marks are those of the plan. This file is committed
with the plan, before the run, and run.py refuses to start if it has changed since.
"""
from __future__ import annotations

import math

import numpy as np

REAL = ("whisper", "qwen")
TIDY = ("G", "L", "T3", "U", "RS", "C", "K")          # the synthetic families with no lump at zero
INDEPENDENT = ("G", "L", "T3", "U")                    # entries drawn independently
FINE = ("4", "4+")                                     # the rates from 4 bits per weight up
G_BALL8 = 0.070454
VOL_BALL8_QUARTER = (math.pi ** 4 / 24.0) ** 0.25


def db(x):
    return 10.0 * math.log10(x)


def _expected_top_chi2_8(blocks):
    t = np.linspace(0.0, 160.0, 800001)
    h = t / 2.0
    cdf = 1.0 - np.exp(-h) * (1.0 + h + h * h / 2.0 + h ** 3 / 6.0)
    f = 1.0 - cdf ** blocks
    return float(np.sum((f[1:] + f[:-1]) * 0.5) * (t[1] - t[0]))


def _residuals(rows, families, rates, kinds=("cut", "scalar"), lattice=None):
    return np.array([db(r["meas"] / r["L3"]) for r in rows if r["kind"] in kinds and r["family"] in families and r["rate"] in rates
                     and (lattice is None or r.get("lattice") == lattice)])


def _cells(by_cell, fam_of, families, rates):
    return [k for k in by_cell if fam_of[k[0]] in families and k[2] in rates]


def verdicts(rows, by_cell, fam_of, probe, choices):
    out = []

    def add(tag, holds, text):
        out.append(f"{tag} | {'holds' if holds else 'FAILS'} | {text}")

    fams = {r["family"] for r in rows}
    real = [f for f in REAL if f in fams]
    tidy = [f for f in TIDY if f in fams]

    # 1. The formula on real matrices, from 4 bits up
    res = _residuals(rows, real, FINE)
    mean_size, within = float(np.abs(res).mean()), float(np.mean(np.abs(res) <= 0.3))
    per = "; ".join(f"{f}: {np.abs(_residuals(rows, (f,), FINE)).mean():.3f} and {np.mean(np.abs(_residuals(rows, (f,), FINE)) <= 0.3):.3f}" for f in real)
    add("1", mean_size <= 0.12 and within >= 0.90,
        f"real matrices, rates 4 and 4+: mean size of the residual {mean_size:.3f} dB (mark 0.12), share within 0.3 dB {within:.3f} (mark 0.90), {len(res)} rows. By model: {per}")

    # 2. The formula on the tidy synthetic families, from 4 bits up
    res = _residuals(rows, tidy, FINE)
    mean_size, within = float(np.abs(res).mean()), float(np.mean(np.abs(res) <= 0.3))
    add("2a", mean_size <= 0.12 and within >= 0.90,
        f"synthetic families {', '.join(tidy)}, rates 4 and 4+: mean size {mean_size:.3f} dB (mark 0.12), share within 0.3 dB {within:.3f} (mark 0.90), {len(res)} rows")
    if "BS" in fams:
        e8 = _residuals(rows, ("BS",), FINE, kinds=("cut",), lattice="E8")
        z8 = _residuals(rows, ("BS",), FINE, kinds=("cut",), lattice="Z8")
        add("2b", float(e8.mean()) <= -0.05 and float(np.mean(np.abs(z8) <= 0.3)) >= 0.90,
            f"family BS (near-zero columns): E8 arms' mean residual {e8.mean():+.3f} dB (predicted at or under -0.05: the formula is high), "
            f"integer arms within 0.3 dB on {np.mean(np.abs(z8) <= 0.3):.3f} of rows (mark 0.90)")

    # 3. At 2 bits the formula is low
    cut_real = _residuals(rows, real, ("2",), kinds=("cut",))
    signs = {f: float(_residuals(rows, (f,), ("2",), kinds=("cut",)).mean()) for f in sorted(fams)}
    within = float(np.mean(np.abs(cut_real) <= 0.3))
    add("3", all(v > 0 for v in signs.values()) and 0.4 <= float(cut_real.mean()) <= 1.5 and within < 0.5,
        f"2 bits, cut arms: mean residual on real matrices {cut_real.mean():+.3f} dB (predicted +0.4 to +1.5), share within 0.3 dB {within:.3f} (predicted under 0.5). "
        f"Mean by family, predicted positive in each: " + ", ".join(f"{f} {v:+.2f}" for f, v in signs.items()))

    # 4. The choice of cell and shape at a fixed group and rate, from 4 bits up
    keys = _cells(by_cell, fam_of, real, FINE)
    ch = [choices(by_cell[k]) for k in keys]
    regret = float(np.mean([c["formula"]["regret"] for c in ch]))
    hits = float(np.mean([c["formula"]["hit"] for c in ch]))
    rank = float(np.mean([c["formula"]["rank"] for c in ch]))
    add("4a", regret <= 0.05 and hits >= 0.80 and rank >= 0.95,
        f"real matrices, rates 4 and 4+, {len(keys)} choices among 13 arms: mean regret {regret:.4f} dB (mark 0.05), the pick is the measured best on {hits:.3f} (mark 0.80), "
        f"mean rank correlation {rank:.3f} (mark 0.95), largest regret {max(c['formula']['regret'] for c in ch):.2f} dB")
    worst = {}
    for f in sorted(fams - set(REAL)):
        k2 = _cells(by_cell, fam_of, (f,), FINE)
        worst[f] = float(np.mean([choices(by_cell[k])["formula"]["regret"] for k in k2]))
    add("4b", all(v <= 0.10 for v in worst.values()),
        "synthetic families, rates 4 and 4+, mean regret by family (mark 0.10 dB in each): " + ", ".join(f"{f} {v:.3f}" for f, v in worst.items()))

    # 5. The best shape on real weights lies between the cross-polytope and the ball
    parts, ok = [], True
    for g in ("32", "128", "R"):
        keys = [k for k in by_cell if fam_of[k[0]] in real and k[1] == g and any(a["budget"] for a in by_cell[k])]
        inner, over = 0, []
        for k in keys:
            arms = {a["arm"]: a["meas"] for a in by_cell[k]}
            best = min(arms, key=arms.get)
            inner += best.endswith(("p5/4", "p3/2"))
            over.append(db(min(arms["E8:p2"], arms["E8:p1"]) / arms[best]))
        share, gain = inner / len(keys), float(np.mean(over))
        ok &= share > 0.5 and 0.05 <= gain <= 0.40
        parts.append(f"group {g}: best arm has p = 5/4 or 3/2 on {inner} of {len(keys)}, and beats the better of E8 in a ball and E8 in a cross-polytope by {gain:.3f} dB in the mean")
    add("5", ok, "real matrices, the three cells of the budget (predicted: more than half, and 0.05 to 0.40 dB). " + "; ".join(parts))

    # 6. What the map has to hold
    parts, ok = [], True
    for g, rate in sorted({(k[1], k[2]) for k in by_cell if k[2] in FINE}):
        keys = [k for k in by_cell if fam_of[k[0]] in real and k[1] == g and k[2] == rate]
        ch = [choices(by_cell[k]) for k in keys]
        formula = float(np.mean([c["formula"]["regret"] for c in ch]))
        shuffled = float(np.mean([c["shuffled"]["regret"] for c in ch]))
        rnd = []
        for k in keys:
            arms = {a["arm"]: a["meas"] for a in by_cell[k]}
            rnd.append(db(arms["E8:p2"] / min(arms.values())))
        rnd = float(np.mean(rnd))
        ok &= rnd >= 0.2 and formula < shuffled <= rnd
        parts.append(f"{g}/{rate}: formula {formula:.3f}, shuffled {shuffled:.3f}, round {rnd:.3f}")
    add("6a", ok, "real matrices, mean regret in dB by cell (predicted: round at least 0.2, and formula < shuffled <= round, in every cell). " + "; ".join(parts))
    ind = [f for f in INDEPENDENT if f in fams]
    if ind:
        keys = _cells(by_cell, fam_of, ind, FINE)
        gap = max(abs(choices(by_cell[k])["shuffled"]["regret"] - choices(by_cell[k])["formula"]["regret"]) for k in keys)
        add("6b", gap <= 0.10, f"families with independent entries ({', '.join(ind)}): the shuffled copy's pick costs within 0.10 dB of the formula's in every choice: largest gap {gap:.3f} dB over {len(keys)}")
    if "RS" in fams:
        keys = [k for k in _cells(by_cell, fam_of, ("RS",), FINE)]
        ch = [choices(by_cell[k]) for k in keys]
        best_ball = sum(c["best"] == "E8:p2" for c in ch)
        pick_ball = sum(c["formula"]["pick"] == "E8:p2" for c in ch)
        row = [choices(by_cell[k]) for k in keys if k[1] == "R"]
        shuf = float(np.mean([c["shuffled"]["regret"] for c in row]))
        add("6c", best_ball == len(ch) and pick_ball == len(ch) and shuf >= 0.1,
            f"family RS (every row has its own scale; a round cloud in each group): E8 in a ball is the measured best in {best_ball} of {len(ch)} choices and the formula's pick in {pick_ball} "
            f"(predicted: all). Per row the shuffled copy's pick costs {shuf:.3f} dB in the mean (predicted at least 0.1)")

    # 7. The size knob in the budget
    same, regrets, best128, n = 0, [], 0, 0
    for mname in sorted({k[0] for k in by_cell if fam_of[k[0]] in real}):
        arms = [dict(a, arm=f"{a['arm']}@{a['group']}") for k, v in by_cell.items() if k[0] == mname for a in v if a["budget"]]
        c = choices(arms)
        same += c["formula"]["group"] == c["best_group"]
        best128 += c["best_group"] == "128"
        regrets.append(c["formula"]["regret"])
        n += 1
    add("7a", same / n >= 0.80 and float(np.mean(regrets)) <= 0.05 and best128 / n > 0.5,
        f"real matrices, 39 arms at 4.5 bits: the formula's pick has the measured best group on {same} of {n} (mark 0.80), mean regret {np.mean(regrets):.4f} dB (mark 0.05), "
        f"and groups of 128 are the measured best on {best128} (predicted more than half)")
    if "G" in fams:
        g128 = []
        for mname in sorted({k[0] for k in by_cell if fam_of[k[0]] == "G"}):
            arms = [dict(a, arm=f"{a['arm']}@{a['group']}") for k, v in by_cell.items() if k[0] == mname for a in v if a["budget"]]
            c = choices(arms)
            g128.append(c["best"] == "E8:p2@128" and c["formula"]["pick"] == "E8:p2@128")
        add("7b", all(g128), f"Gaussian matrices at 4.5 bits: E8 in a ball with groups of 128 is the measured best and the formula's pick on {sum(g128)} of {len(g128)} (predicted: all)")

    # 8. Kept directions
    rank = [r for r in rows if r["kind"] == "rank"]
    full_best = {}
    full_best_formula = {}
    for k, v in by_cell.items():
        for a in v:
            if a["budget"]:
                full_best[k[0]] = min(full_best.get(k[0], 1e9), a["meas"])
                full_best_formula[k[0]] = min(full_best_formula.get(k[0], 1e9), a["L3"])
    rr = [r for r in rank if r["family"] in real]
    ahead = sum(r["meas"] < full_best[r["matrix"]] for r in rr)
    ahead_f = sum(r["L3"] < full_best_formula[r["matrix"]] for r in rr)
    res = np.array([db(r["meas"] / r["L3"]) for r in rr])
    add("8a", ahead == 0 and ahead_f == 0 and float(np.mean(np.abs(res) <= 1.5)) >= 0.90,
        f"real matrices, {len(rr)} rank arms: ahead of the best full arm at 4.5 bits on {ahead}, and by the formula on {ahead_f} (predicted: none, both). "
        f"Measured error within 1.5 dB of the formula on {np.mean(np.abs(res) <= 1.5):.3f} (mark 0.90). Smallest gap to the best full arm {min(db(r['meas'] / full_best[r['matrix']]) for r in rr):+.2f} dB")
    kk = [r for r in rank if r["family"] == "K" and r["share"] == "1/8"]
    if kk:
        gains = [db(full_best[r["matrix"]] / r["meas"]) for r in kk]
        says = sum(r["L3"] < full_best_formula[r["matrix"]] for r in kk)
        add("8b", min(gains) >= 3.0 and says == len(kk),
            f"family K (64 directions hold all but a thousandth): the arm that keeps 64 directions beats the best full arm by {min(gains):.2f} to {max(gains):.2f} dB (predicted at least 3 on each), "
            f"and the formula puts it ahead on {says} of {len(kk)}")

    # 9. The ceiling for a round cloud
    low, eb, best = [], [], []
    for k in _cells(by_cell, fam_of, ("G",), FINE):
        arms = [a for a in by_cell[k] if a["kind"] == "cut"]
        floor = lambda a: G_BALL8 * VOL_BALL8_QUARTER * _expected_top_chi2_8(a["size"] // 8) * a["codewords"] ** -0.25
        low.append(min(db(a["meas"] / floor(a)) for a in arms))
        eb += [db(a["meas"] / floor(a)) for a in arms if a["arm"] == "E8:p2"]
        best.append(min(by_cell[k], key=lambda a: a["meas"])["arm"] == "E8:p2")
    if low:
        add("9", min(low) >= -0.05 and all(best) and max(eb) <= 0.45,
            f"Gaussian matrices, rates 4 and 4+: smallest measured error over the floor of a round cloud {min(low):+.3f} dB (predicted not under -0.05). "
            f"E8 in a ball is the measured best in {sum(best)} of {len(best)} choices (predicted: all) and at most {max(eb):.3f} dB above the floor (mark 0.45)")

    # 10. How like noise the rounding error is
    def ratio(families, group, kind):
        v = [r["spectral_ratio"] for r in rows if r["kind"] == kind and r["family"] in families and r["group"] == group and r.get("spectral_ratio") is not None]
        return float(np.mean(v)) if v else float("nan")
    xs = [ratio(real, g, "scalar") for g in ("32", "128", "R")]
    es = [ratio(real, g, "cut") for g in ("32", "128", "R")]
    gs = [ratio(("G",), g, k) for g in ("32", "128") for k in ("scalar", "cut")]
    rises = xs[0] < xs[1] < xs[2] and es[0] < es[1] < es[2]
    below = all(e < x for e, x in zip(es, xs))
    flat = all(abs(v - 1.0) <= 0.02 for v in gs) if "G" in fams else True
    add("10", rises and below and flat,
        "spectral ratio in the three cells of the budget. Real matrices, groups of 32, 128, a row: scalar rule " + ", ".join(f"{v:.3f}" for v in xs)
        + "; E8 in a ball " + ", ".join(f"{v:.3f}" for v in es) + " (predicted: both rise with the group, and E8 in a ball is under the scalar rule in each). "
        + "Gaussian matrices with groups of 32 and 128: " + ", ".join(f"{v:.3f}" for v in gs) + " (predicted within 0.02 of 1)")
    return out
