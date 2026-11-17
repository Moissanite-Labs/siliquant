#!/usr/bin/env python3
"""Tables written after the run was read. None was planned. Reads results.json and safe.json and rounds nothing.

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
    print("\n".join(out))


if __name__ == "__main__":
    main()
