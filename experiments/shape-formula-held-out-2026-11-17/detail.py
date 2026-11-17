#!/usr/bin/env python3
"""Tables written after the run was read. None was planned. Reads results.json, safe.json and bounds/bounds.out and rounds nothing.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/shape-formula-held-out-2026-11-17/detail.py > detail.out
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summary  # noqa: E402

REAL = ("whisper", "qwen")
FINE = ("4", "4+")


def db(x):
    return 10.0 * math.log10(x)


def main():
    data = json.loads((HERE / "results.json").read_text())
    rows = data["rows"]
    safe = json.loads((HERE / "safe.json").read_text())["rows"]
    probe = {r["matrix"]: r for r in rows if r["kind"] == "probe"}
    by_cell = defaultdict(list)
    for r in rows:
        if r["kind"] in ("cut", "scalar"):
            by_cell[(r["matrix"], r["group"], r["rate"])].append(r)
    fam_of = {r["matrix"]: r["family"] for r in rows}
    out = []
    P = out.append

    P("1. The rank arms that are ahead of the best full arm at 4.5 bits, on real matrices")
    P("   matrix | kept | levels | bits | dropped, dB | rank arm dB | by the formula dB | best full arm dB | ahead by | top eighth of the directions' share of the squared weights | Shannon's gain from the directions at 4 bits | zero lump")
    full = defaultdict(lambda: 1e9)
    for k, v in by_cell.items():
        for a in v:
            if a["budget"]:
                full[k[0]] = min(full[k[0]], a["meas"])
    for r in rows:
        if r["kind"] == "rank" and r["family"] in REAL and r["meas"] < full[r["matrix"]]:
            p = probe[r["matrix"]]
            P(f"   {r['matrix']} | {r['share']} | {r['q']} | {r['bits']:.3f} | {db(r['tail']):.2f} | {db(r['meas']):.2f} | {db(r['L3']):.2f} | {db(full[r['matrix']]):.2f} | {db(full[r['matrix']] / r['meas']):.2f} | "
              f"{p['stretch']['directions']['top_eighth_share_of_mass']:.4f} | {p['stretch']['directions']['gain_db_at_4_bits']:.2f} | {p['share_under_a_tenth_of_rms']:.3f}")
    P("   The five real matrices where a rank arm comes nearest without being ahead:")
    near = sorted(((db(r["meas"] / full[r["matrix"]]), r) for r in rows if r["kind"] == "rank" and r["family"] in REAL and r["meas"] >= full[r["matrix"]]), key=lambda t: t[0])[:5]
    for gap, r in near:
        P(f"   {r['matrix']} | {r['share']} | behind by {gap:.2f} dB | top eighth {probe[r['matrix']]['stretch']['directions']['top_eighth_share_of_mass']:.4f}")
    P("")

    P("2. By model: the three ways of choosing, from 4 bits up, and the best shape in the budget")
    P("   model | group | rate | formula: mean regret | shuffled | round | matrices")
    for f in REAL:
        for g, rate in (("32", "4"), ("128", "4"), ("R", "4"), ("128", "4+"), ("R", "4+")):
            keys = [k for k in by_cell if fam_of[k[0]] == f and k[1] == g and k[2] == rate]
            ch = [summary.choices(by_cell[k]) for k in keys]
            rnd = []
            for k in keys:
                arms = {a["arm"]: a["meas"] for a in by_cell[k]}
                rnd.append(db(arms["E8:p2"] / min(arms.values())))
            P(f"   {f:>7} | {g:>3} | {rate:>2} | {np.mean([c['formula']['regret'] for c in ch]):.3f} | {np.mean([c['shuffled']['regret'] for c in ch]):.3f} | {np.mean(rnd):.3f} | {len(keys)}")
    P("   model | group of the budget | best arm has p = 5/4 or 3/2 on | ahead of the better of E8 in a ball and E8 in a cross-polytope: mean (max) | ahead of E8 in a ball: mean | matrices")
    for f in REAL:
        for g in ("32", "128", "R"):
            keys = [k for k in by_cell if fam_of[k[0]] == f and k[1] == g and any(a["budget"] for a in by_cell[k])]
            inner, two, ball = 0, [], []
            for k in keys:
                arms = {a["arm"]: a["meas"] for a in by_cell[k]}
                best = min(arms, key=arms.get)
                inner += best.endswith(("p5/4", "p3/2"))
                two.append(db(min(arms["E8:p2"], arms["E8:p1"]) / arms[best]))
                ball.append(db(arms["E8:p2"] / arms[best]))
            P(f"   {f:>7} | {g:>3} | {inner} of {len(keys)} | {np.mean(two):.3f} ({max(two):.2f}) | {np.mean(ball):.3f} | {len(keys)}")
    P("")

    P("3. Where the formula is furthest off from 4 bits up: the ten rows of real matrices with the largest residual, and the matrix's lump at zero")
    P("   matrix | arm | group | rate | measured over formula, dB | share of the entries under a tenth of the root mean square | Shannon's gain from the columns at 4 bits")
    worst = sorted((r for r in rows if r["kind"] in ("cut", "scalar") and r["family"] in REAL and r["rate"] in FINE), key=lambda r: -abs(db(r["meas"] / r["L3"])))[:10]
    for r in worst:
        p = probe[r["matrix"]]
        P(f"   {r['matrix']} | {r['arm']} | {r['group']} | {r['rate']} | {db(r['meas'] / r['L3']):+.2f} | {p['share_under_a_tenth_of_rms']:.3f} | {p['stretch']['columns']['gain_db_at_4_bits']:.2f}")
    res = defaultdict(list)
    for r in rows:
        if r["kind"] in ("cut", "scalar") and r["family"] in REAL and r["rate"] in FINE:
            res[r["matrix"]].append(abs(db(r["meas"] / r["L3"])))
    off = sorted(((max(v), m) for m, v in res.items()), reverse=True)
    P(f"   real matrices with any row over 0.3 dB: {sum(v > 0.3 for v, _ in off)} of {len(off)}. The same with the star p = 2/3 left out: "
      + str(sum(max(abs(db(r['meas'] / r['L3'])) for r in rows if r['kind'] in ('cut', 'scalar') and r['matrix'] == m and r['rate'] in FINE and r.get('p') != '2/3') > 0.3 for _, m in off)))
    P("")

    P("4. A gate on the formula (crossing X5 of the note in PR #5): the share of runs a gate would pass if it refused a rounding whose error is more than a mark over the formula's")
    P("   set | rate | mark 0.1 dB | 0.2 | 0.3 | 0.5 | runs")
    for label, fs in (("real", REAL), ("synthetic", tuple(f for f in dict.fromkeys(fam_of.values()) if f not in REAL))):
        for rate in ("2", "4", "4+"):
            v = np.array([db(r["meas"] / r["L3"]) for r in rows if r["kind"] in ("cut", "scalar") and r["family"] in fs and r["rate"] == rate])
            P(f"   {label:>9} | {rate:>2} | " + " | ".join(f"{np.mean(v <= m):.3f}" for m in (0.1, 0.2, 0.3, 0.5)) + f" | {len(v)}")
    P("")

    P("5. The safe rule: the matrices where the arm with the smallest bound is furthest from the measured best of the 12")
    P("   matrix | arm with the smallest bound | measured best | regret, dB | zero lump")
    by = defaultdict(list)
    for r in safe:
        by[r["matrix"]].append(r)
    reg = []
    for m, rs in by.items():
        if rs[0]["family"] not in REAL:
            continue
        b = min(rs, key=lambda r: r["bound"])
        best = min(rs, key=lambda r: r["meas"])
        reg.append((db(b["meas"] / best["meas"]), m, f"{b['arm']}@{b['group']}", f"{best['arm']}@{best['group']}"))
    for g, m, a, b in sorted(reg, reverse=True)[:5]:
        P(f"   {m} | {a} | {b} | {g:.2f} | {probe[m]['share_under_a_tenth_of_rms']:.3f}")
    v = np.array([g for g, *_ in reg])
    P(f"   regret is 0 on {int(np.sum(v == 0))} of {len(v)}, under 0.05 dB on {int(np.sum(v < 0.05))}, and its mean without the largest one is {np.mean(np.sort(v)[:-1]):.4f} dB")
    P("")

    P("6. The scalar rule on uniform weights: the rule clips on the side opposite to the largest entry")
    P("   The largest entry goes to level -q/2 and the levels stop at q/2 - 1. A weight within half a step of the far end is pulled back a whole level.")
    P("   Its squared error is 7/12 of a squared step in the mean, not 1/12. A uniform weight is there with chance 1 / (2 q). So the error is 1 + 3/q times the formula's.")
    P("   group | rate | levels q | measured over formula, dB | 10 log10(1 + 3/q)")
    for g, rate in (("32", "4"), ("128", "4"), ("R", "4"), ("128", "4+"), ("R", "4+"), ("32", "2"), ("R", "2")):
        v = [db(r["meas"] / r["L3"]) for r in rows if r["kind"] == "scalar" and r["family"] == "U" and r["group"] == g and r["rate"] == rate]
        q = [r["q"] for r in rows if r["kind"] == "scalar" and r["family"] == "U" and r["group"] == g and r["rate"] == rate][0]
        P(f"   {g:>3} | {rate:>2} | {q} | {np.mean(v):+.2f} | {db(1.0 + 3.0 / q):+.2f}")
    P("")

    P("7. Figures the record quotes that no other table shows")
    star_off = defaultdict(float)
    for r in rows:
        if r["kind"] in ("cut", "scalar") and r["family"] in REAL and r["rate"] in FINE and r.get("p") != "2/3":
            star_off[r["matrix"]] = max(star_off[r["matrix"]], abs(db(r["meas"] / r["L3"])))
    P("   Real matrices with a run more than 0.3 dB off from 4 bits up, the star p = 2/3 left out: matrix | largest residual | zero lump")
    for m, v in sorted(star_off.items(), key=lambda t: -t[1]):
        if v > 0.3:
            P(f"   {m} | {v:.2f} | {probe[m]['share_under_a_tenth_of_rms']:.3f}")
    P("   The five arms with an exact formula, at the three group sizes of the budget: where the pick of level 2 is not the pick of level 3, and where it is not the measured best")
    exact = defaultdict(list)
    for r in rows:
        if r["kind"] in ("cut", "scalar") and r["budget"] and r["arm"] in ("X", "Z8:p1", "Z8:p2", "E8:p1", "E8:p2"):
            exact[r["matrix"]].append(r)
    for m, rs in exact.items():
        l2, l3, mb = min(rs, key=lambda r: r["L2"]), min(rs, key=lambda r: r["L3"]), min(rs, key=lambda r: r["meas"])
        if l2 is not l3 or l2 is not mb:
            P(f"   {m} | level 2: {l2['arm']}@{l2['group']} | level 3: {l3['arm']}@{l3['group']} | measured best: {mb['arm']}@{mb['group']} | level 2's pick over the measured best: {db(l2['meas'] / mb['meas']):.3f} dB | zero lump {probe[m]['share_under_a_tenth_of_rms']:.3f}")
    P("   Runs of a cut arm in which a block was shrunk, by rate: rate | runs with one | runs | largest share of blocks")
    for rate in ("2", "4", "4+"):
        v = [r["shrunk"] for r in rows if r["kind"] == "cut" and r["rate"] == rate]
        P(f"   {rate:>2} | {sum(x > 0 for x in v)} | {len(v)} | {max(v):.4f}")
    best_rank = defaultdict(lambda: 1e9)
    for r in rows:
        if r["kind"] == "rank" and r["family"] in REAL:
            best_rank[r["matrix"]] = min(best_rank[r["matrix"]], r["meas"])
    gaps = sorted(db(best_rank[m] / full[m]) for m in best_rank)
    P(f"   The best rank arm of a real matrix against its best full arm at 4.5 bits: ahead on {sum(g < 0 for g in gaps)} of {len(gaps)}, by {-gaps[0]:.2f} and {-gaps[1]:.2f} dB. Behind on the rest by {min(g for g in gaps if g > 0):.2f} to {gaps[-1]:.2f} dB")
    res2 = [db(r["meas"] / r["L3"]) for r in rows if r["kind"] in ("cut", "scalar") and r["family"] in REAL and r["rate"] == "2"]
    P(f"   The largest residual at 2 bits on a real matrix: {max(res2):+.2f} dB. Over every family: {max(db(r['meas'] / r['L3']) for r in rows if r['kind'] in ('cut', 'scalar') and r['rate'] == '2'):+.2f} dB")
    sys.path.insert(0, str(HERE.parents[1] / "shape"))
    import certified as C
    import shapelib as S
    P("   The safe scale at 2 bits, not run: 20 log10(r_eff / (r - m)) for a codebook of at most 2^16 points")
    for lattice, p, name in (("E8", 2, "E8 in a ball"), ("Z8", 2, "the integers in a ball"), ("E8", 1, "E8 in a cross-polytope"), ("Z8", 1, "the integers in a cross-polytope")):
        code = S.Codebook(lattice, S.Shape(p), 1 << 16)
        P(f"   {name}: {20.0 * math.log10(code.r_eff / C.safe_radius(code)):.2f} dB")
    lines = (HERE.parents[1] / "bounds" / "bounds.out").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("10. "))
    phi = {}
    for ln in lines[start:]:
        parts = [x.strip() for x in ln.split("|")]
        if len(parts) == 7 and parts[0].isdigit():
            phi[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[4]))
    P("   Gaussian weights, from the limits table: what the step from groups of 8 to groups of 2048 costs in the region part")
    P(f"   cube {db(phi[2048][0] / phi[8][0]):.2f} dB | ball {db(phi[2048][1] / phi[8][1]):.2f} dB | cross-polytope {db(phi[2048][2] / phi[8][2]):.2f} dB")
    print("\n".join(out))


if __name__ == "__main__":
    main()
