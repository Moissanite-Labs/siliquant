#!/usr/bin/env python3
"""Tables for the study in this folder, from pilot.json or results.json. Rounds nothing.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/shape-formula-held-out-2026-11-17/summary.py pilot > pilot.out
  python3 experiments/shape-formula-held-out-2026-11-17/summary.py results > results.out
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GROUPS = ("32", "128", "R")
RATES = ("2", "4", "4+")
G_E8 = 929.0 / 12960.0
G_BALL8 = 0.070454                      # bounds/bounds.out, section 2
VOL_BALL8_QUARTER = (math.pi ** 4 / 24.0) ** 0.25


def db(x):
    return 10.0 * math.log10(x)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def expected_top_chi2_8(blocks):
    """Mean of the largest of `blocks` squared lengths of Gaussian blocks of 8: the integral of 1 - F^blocks."""
    t = np.linspace(0.0, 160.0, 800001)
    h = t / 2.0
    cdf = 1.0 - np.exp(-h) * (1.0 + h + h * h / 2.0 + h ** 3 / 6.0)
    f = 1.0 - cdf ** blocks
    return float(np.sum((f[1:] + f[:-1]) * 0.5) * (t[1] - t[0]))


def family_groups(rows):
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    return fams


def cells_of(rows):
    """The cells present, in a fixed order: (group, rate)."""
    seen = {(r["group"], r["rate"]) for r in rows if r["kind"] in ("cut", "scalar")}
    return [(g, rate) for rate in RATES for g in GROUPS if (g, rate) in seen]


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():+.2f} | {np.abs(v).mean():.2f} | {v.min():+.2f} | {v.max():+.2f} | {np.mean(np.abs(v) <= 0.1):.2f} | {np.mean(np.abs(v) <= 0.3):.2f} | {len(v)}"


def choices(arms):
    """For the arms of one matrix in one cell: what each way of choosing picks, and what the pick costs.

    A way of choosing sees a prediction for every arm and takes the smallest. Its regret is the
    measured error of its pick over the smallest measured error, in dB.
    """
    meas = np.array([a["meas"] for a in arms])
    best = int(meas.argmin())
    out = {"best": arms[best]["arm"], "best_group": arms[best]["group"]}
    for way, key in (("formula", "L3"), ("level 1", "L1"), ("level 2", "L2"), ("shuffled", "L3_shuffled")):
        if any(key not in a for a in arms):
            continue
        pred = np.array([a[key] for a in arms])
        pick = int(pred.argmin())
        out[way] = {"pick": arms[pick]["arm"], "group": arms[pick]["group"], "regret": db(meas[pick] / meas[best]),
                    "hit": pick == best, "rank": spearman(pred, meas) if len(arms) > 2 else float("nan")}
    return out


def render(data, with_predictions):
    """The tables as a list of lines, and the per-matrix figures behind three of them."""
    meta, rows = data["meta"], data["rows"]
    out = []
    P = out.append
    cut = [r for r in rows if r["kind"] == "cut"]
    sca = [r for r in rows if r["kind"] == "scalar"]
    rank = [r for r in rows if r["kind"] == "rank"]
    probe = {r["matrix"]: r for r in rows if r["kind"] == "probe"}
    fams = family_groups(rows)
    cells = cells_of(rows)
    by_cell = defaultdict(list)                                  # (matrix, group, rate) -> arms
    for r in cut + sca:
        by_cell[(r["matrix"], r["group"], r["rate"])].append(r)
    fam_of = {r["matrix"]: r["family"] for r in rows}
    real = [f for f in fams if f in ("whisper", "qwen")]
    synth = [f for f in fams if f not in real]

    P(f"Set: {meta['set']}. Matrices: {meta['matrices']}. Rows: {len(rows)}. Seconds: {meta['seconds']}. NumPy {meta['numpy']}, Python {meta['python']}.")
    P("")
    P("0. Checks")
    for ok, text in meta["checks"]:
        P(f"   {'ok  ' if ok else 'FAIL'} {text}")
    shr = [r for r in cut if r["shrunk"] > 0]
    P(f"   runs where a block lay outside the codebook at its group's scale and was shrunk: {len(shr)} of {len(cut)}"
      + (f", largest share of blocks {max(r['shrunk'] for r in shr):.2e}" if shr else ""))
    worst = max(p["receipt_worst_relative_difference"] for p in probe.values())
    P(f"   the region part of the cube, the ball and the cross-polytope from whole-number arithmetic against 64-bit floats: largest relative difference {worst:.1e} over {len(probe)} matrices and 3 groups")
    P("")
    P("1. The codebooks")
    P("   The last column is the bits per weight if the index of each block is stored on its own in a whole number of bits.")
    P("   lattice | shape | cap | radius as a p-norm | codewords | codewords over cap | bits per weight without the scale | with a whole number of bits per block")
    for b in meta["codebooks"]:
        P(f"   {b['lattice']} | {b['shape']} | 2^{math.log2(b['cap']):.4f} | {b['radius_as_norm']:.4f} | {b['codewords']} | {b['codewords'] / b['cap']:.6f} | {b['bits']:.5f} | {math.ceil(math.log2(b['codewords'])) / 8.0:.3f}")
    P("")
    P("2. How close the formula is: measured error over the formula's, in dB. 0 means exact, + means the formula is low.")
    P("   level 1: cell, region and count. level 2: and the block on the wall. level 3: and the cell at zero.")
    P("   family | rate | arms | level | mean | mean size | min | max | share within 0.1 | within 0.3 | rows")
    for f in fams:
        for rate in RATES:
            for label, sel in (("integers", [r for r in cut if r["lattice"] == "Z8"]), ("E8", [r for r in cut if r["lattice"] == "E8"]), ("scalar rule", sca)):
                rs = [r for r in sel if r["family"] == f and r["rate"] == rate]
                if not rs:
                    continue
                for level in ("L1", "L2", "L3"):
                    P(f"   {f:>7} | {rate:>2} | {label:>11} | {level} | {stats([db(r['meas'] / r[level]) for r in rs])}")
    P("")
    P("   The same at level 3 by shape, cut arms only, rates 4 and 4+")
    P("   set | shape | mean | mean size | min | max | share within 0.1 | within 0.3 | rows")
    for label, fs in (("synthetic", synth), ("real", real)):
        for p in sorted({r["p"] for r in cut}, key=lambda s: eval(s)):
            rs = [r for r in cut if r["family"] in fs and r["p"] == p and r["rate"] != "2"]
            if rs:
                P(f"   {label:>9} | {p:>3} | {stats([db(r['meas'] / r['L3']) for r in rs])}")
    P("")
    P("3. Choosing the cell and the shape at a fixed group and rate: 12 cut arms and the scalar rule")
    P("   regret = measured error of the pick over the smallest measured error, in dB. hit = the pick is the measured best.")
    P("   formula: level 3 on the matrix. level 1: without the wall and the cell at zero. shuffled: level 3 on the same entries in random places.")
    P("   round: always E8 in a ball, the best region for a round cloud (proved in the plan). ball or cross: the better of E8 in a ball and E8 in a cross-polytope, chosen by measuring both.")
    P("   family | group | rate | formula: mean regret, max, hits, rank corr. | level 1: mean, max, hits | shuffled: mean, max, hits | round: mean, max | best over ball or cross: mean, max | matrices")
    regrets = defaultdict(list)
    for f in fams:
        for g, rate in cells:
            keys = [k for k in by_cell if fam_of[k[0]] == f and k[1] == g and k[2] == rate]
            if not keys:
                continue
            ch = [choices(by_cell[k]) for k in keys]
            line = f"   {f:>7} | {g:>3} | {rate:>2} | "
            line += f"{np.mean([c['formula']['regret'] for c in ch]):.3f}, {max(c['formula']['regret'] for c in ch):.2f}, {sum(c['formula']['hit'] for c in ch)}, {np.mean([c['formula']['rank'] for c in ch]):.3f} | "
            for way in ("level 1", "shuffled"):
                line += f"{np.mean([c[way]['regret'] for c in ch]):.3f}, {max(c[way]['regret'] for c in ch):.2f}, {sum(c[way]['hit'] for c in ch)} | "
            rnd, two = [], []
            for k in keys:
                arms = {a["arm"]: a["meas"] for a in by_cell[k]}
                best = min(arms.values())
                rnd.append(db(arms["E8:p2"] / best))
                two.append(db(min(arms["E8:p2"], arms["E8:p1"]) / best))
            line += f"{np.mean(rnd):.3f}, {max(rnd):.2f} | {np.mean(two):.3f}, {max(two):.2f} | {len(keys)}"
            P(line)
            for c, k, a, b in zip(ch, keys, rnd, two):
                regrets[(f, g, rate)].append({"matrix": k[0], "formula": c["formula"]["regret"], "shuffled": c["shuffled"]["regret"],
                                              "level1": c["level 1"]["regret"], "round": a, "two": b, "hit": c["formula"]["hit"]})
    P("")
    P("4. Which arm is best, measured, and which the formula picks")
    P("   family | group | rate | measured best: arm and count | the formula's pick: arm and count")
    for f in fams:
        for g, rate in cells:
            keys = [k for k in by_cell if fam_of[k[0]] == f and k[1] == g and k[2] == rate]
            if not keys:
                continue
            ch = [choices(by_cell[k]) for k in keys]
            mb = Counter(c["best"] for c in ch).most_common()
            fp = Counter(c["formula"]["pick"] for c in ch).most_common()
            P(f"   {f:>7} | {g:>3} | {rate:>2} | " + ", ".join(f"{a} {n}" for a, n in mb) + " | " + ", ".join(f"{a} {n}" for a, n in fp))
    P("")
    P("5. Gain of every arm over the scalar rule in the same cell, measured, in dB: mean over the matrices of a family")
    arms_order = [f"{lat}:{sh}" for lat in ("Z8", "E8") for sh in ("p2/3", "p1", "p5/4", "p3/2", "p2", "p3")]
    P("   family | group | rate | " + " | ".join(arms_order) + " | scalar rule dB | its bits | cut arms' bits")
    for f in fams:
        for g, rate in cells:
            keys = [k for k in by_cell if fam_of[k[0]] == f and k[1] == g and k[2] == rate]
            if not keys:
                continue
            gains = defaultdict(list)
            xs, xb, cb = [], [], []
            for k in keys:
                arms = {a["arm"]: a for a in by_cell[k]}
                xs.append(db(arms["X"]["meas"]))
                xb.append(arms["X"]["bits"])
                cb.append(arms["E8:p2"]["bits"])
                for a in arms_order:
                    gains[a].append(db(arms["X"]["meas"] / arms[a]["meas"]))
            P(f"   {f:>7} | {g:>3} | {rate:>2} | " + " | ".join(f"{np.mean(gains[a]):+.2f}" for a in arms_order) + f" | {np.mean(xs):.2f} | {np.mean(xb):.3f} | {np.mean(cb):.3f}")
    P("")
    P("6. The size knob: the three cells of the budget, 4.5 bits per weight with the scales. 13 arms in each of 3 groups.")
    P("   family | measured best group: counts | the formula's group: counts | same group on | regret of the formula's pick: mean, max | regret of the best arm of each group, measured: 32, 128, row | matrices")
    budget_regret = defaultdict(list)
    for f in fams:
        mats = sorted({k[0] for k in by_cell if fam_of[k[0]] == f})
        res = []
        for mname in mats:
            arms = [dict(a, arm=f"{a['arm']}@{a['group']}") for k, v in by_cell.items() if k[0] == mname for a in v if a["budget"]]
            if not arms:
                continue
            c = choices(arms)
            best = min(a["meas"] for a in arms)
            per_group = {g: db(min(a["meas"] for a in arms if a["group"] == g) / best) for g in GROUPS}
            res.append((c, per_group))
            budget_regret[f].append({"matrix": mname, "regret": c["formula"]["regret"], "same_group": c["formula"]["group"] == c["best_group"]})
        if not res:
            continue
        mb = Counter(c["best_group"] for c, _ in res)
        fp = Counter(c["formula"]["group"] for c, _ in res)
        P(f"   {f:>7} | " + ", ".join(f"{g}: {mb.get(g, 0)}" for g in GROUPS) + " | " + ", ".join(f"{g}: {fp.get(g, 0)}" for g in GROUPS)
          + f" | {sum(c['formula']['group'] == c['best_group'] for c, _ in res)} | {np.mean([c['formula']['regret'] for c, _ in res]):.3f}, {max(c['formula']['regret'] for c, _ in res):.2f} | "
          + ", ".join(f"{np.mean([pg[g] for _, pg in res]):.2f}" for g in GROUPS) + f" | {len(res)}")
    P("")
    P("7. Kept directions: the k largest directions of the matrix in two rounded factors, at the budget of 4.5 bits")
    P("   'behind' = error of the rank arm over the smallest error of a full arm in the budget, in dB. dropped = the share of the squared weights in the directions left out.")
    P("   family | kept | levels | bits | dropped, dB | measured, dB | measured over formula: mean (min, max) | behind the best full arm, measured: mean (min, max) | by the formula: mean | rank arm ahead on | formula says ahead on | matrices")
    rank_rows = defaultdict(list)
    for f in fams:
        for share in ("1/8", "1/4", "1/2"):
            rs = [r for r in rank if r["family"] == f and r["share"] == share]
            if not rs:
                continue
            behind, behind_f, res = [], [], []
            for r in rs:
                full = [a for k, v in by_cell.items() if k[0] == r["matrix"] for a in v if a["budget"]]
                behind.append(db(r["meas"] / min(a["meas"] for a in full)))
                behind_f.append(db(r["L3"] / min(a["L3"] for a in full)))
                res.append(db(r["meas"] / r["L3"]))
                rank_rows[f].append({"matrix": r["matrix"], "share": share, "behind": behind[-1], "behind_formula": behind_f[-1], "residual": res[-1]})
            P(f"   {f:>7} | {share} | {int(np.median([r['q'] for r in rs]))} | {np.mean([r['bits'] for r in rs]):.3f} | {np.mean([db(max(r['tail'], 1e-30)) for r in rs]):.2f} | {np.mean([db(r['meas']) for r in rs]):.2f} | "
              f"{np.mean(res):+.2f} ({min(res):+.2f}, {max(res):+.2f}) | {np.mean(behind):+.2f} ({min(behind):+.2f}, {max(behind):+.2f}) | {np.mean(behind_f):+.2f} | "
              f"{sum(b < 0 for b in behind)} | {sum(b < 0 for b in behind_f)} | {len(rs)}")
    P("")
    P("8. The map: what the probe says about each family, in dB")
    P("   region gain = 10 log10 of the cube's region part over the shape's: what the shape is worth against the scalar rule before the cell and the count.")
    P("   spiky = top 1 percent of the entries' share of the squared weights (0.070 for a Gaussian). peak = a block's largest entry over its length (0.650 Gaussian).")
    P("   stretch = what Shannon's rule gains at 4 bits from the spread of the mass over the directions of the matrix, its columns and its rows.")
    P("   together = 10 log10 of the region part of the shuffled copy over that of the matrix, for the ball per row: + means the entries' places help.")
    P("   zero lump = share of the entries under a tenth of the root mean square (0.080 for a Gaussian).")
    P("   family | spiky | peak | region gain per row: p2/3, p1, p5/4, p3/2, p2, p3 | best p per row: counts | together: 32, row | stretch: directions, columns, rows | directions dropped at 4 bits | zero lump | matrices")
    shapes = ("2/3", "1", "5/4", "3/2", "2", "3")
    for f in fams:
        ps = [p for p in probe.values() if p["family"] == f]
        gain = np.array([[db(p["phi"]["R"]["cube"] / p["phi"]["R"][s]) for s in shapes] for p in ps])
        bestp = Counter(shapes[i] for i in gain.argmax(axis=1)).most_common()
        tog = [np.mean([db(p["phi_shuffled"][g]["2"] / p["phi"][g]["2"]) for p in ps]) for g in ("32", "R")]
        st = [np.mean([p["stretch"][k]["gain_db_at_4_bits"] for p in ps]) for k in ("directions", "columns", "rows")]
        P(f"   {f:>7} | {np.mean([p['top_1_percent_share_of_mass'] for p in ps]):.3f} | {np.mean([p['peak'] for p in ps]):.3f} | "
          + ", ".join(f"{v:+.2f}" for v in gain.mean(axis=0)) + " | " + ", ".join(f"{a}: {n}" for a, n in bestp)
          + f" | {tog[0]:+.2f}, {tog[1]:+.2f} | {st[0]:.2f}, {st[1]:.2f}, {st[2]:.2f} | {np.mean([p['stretch']['directions']['share_dropped_at_4_bits'] for p in ps]):.3f} | "
          f"{np.mean([p['share_under_a_tenth_of_rms'] for p in ps]):.3f} | {len(ps)}")
    P("")
    P("9. Gaussian matrices against the ceiling for a round cloud")
    P("   No region of a code in 8 dimensions does better on a round cloud than a ball, and no cell than a ball's (the plan gives the proof).")
    P("   floor = G of the ball in 8 dimensions * region part of the ball for Gaussian blocks, from its integral * codewords^(-1/4).")
    P("   group | rate | smallest measured error over the floor, in dB, over all cut arms and Gaussian matrices | the arm | E8 in a ball: mean | rows")
    gm = sorted({r["matrix"] for r in cut if r["family"] == "G"})
    for g, rate in cells:
        rs = [r for r in cut if r["family"] == "G" and r["group"] == g and r["rate"] == rate]
        if not rs:
            continue
        vals = []
        for r in rs:
            blocks = r["size"] // 8
            phi_ball = VOL_BALL8_QUARTER * expected_top_chi2_8(blocks)
            vals.append((db(r["meas"] / (G_BALL8 * phi_ball * r["codewords"] ** -0.25)), r["arm"]))
        eb = [v for v, a in vals if a == "E8:p2"]
        lo = min(vals)
        P(f"   {g:>3} | {rate:>2} | {lo[0]:+.3f} | {lo[1]} | {np.mean(eb):+.3f} | {len(rs)}")
    P("")
    P("10. Is the rounding error like noise? Its largest singular value over that of a matrix of independent entries of the same total square.")
    P("   The second record found that the largest singular value of the weight change ranked the damage to a model's output best of five measures.")
    P("   1 means the error has no direction of its own. The three cells of the budget.")
    P("   family | group | scalar rule: mean (min, max) | E8 in a ball: mean (min, max) | matrices")
    for f in fams:
        for g in GROUPS:
            xs = [r["spectral_ratio"] for r in sca if r["family"] == f and r["group"] == g and r.get("spectral_ratio") is not None]
            es = [r["spectral_ratio"] for r in cut if r["family"] == f and r["group"] == g and r.get("spectral_ratio") is not None]
            if xs and es:
                P(f"   {f:>7} | {g:>3} | {np.mean(xs):.3f} ({min(xs):.3f}, {max(xs):.3f}) | {np.mean(es):.3f} ({min(es):.3f}, {max(es):.3f}) | {len(xs)}")
    P("")
    detail = {"regrets": {"|".join(k): v for k, v in regrets.items()}, "budget": budget_regret, "rank": rank_rows}
    if with_predictions:
        import predictions
        P("11. The predictions of the plan")
        for line in predictions.verdicts(rows, by_cell, fam_of, probe, choices):
            P("   " + line)
        P("")
    return out, detail


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "results"
    data = json.loads((HERE / f"{name}.json").read_text())
    out, detail = render(data, with_predictions=(name == "results"))
    (HERE / f"{name}.detail.json").write_text(json.dumps(detail, indent=1) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
