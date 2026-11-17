#!/usr/bin/env python3
"""Tables for the record that results.out does not hold. Written after the run. None of them was planned.

Reads results.json of this run and of e8-gain-largest-entry-rule-2026-11-17.

Usage, from the root of a checkout:
  python3 experiments/region-shapes-no-clip-rule-2026-11-17/detail.py > detail.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "e8-gain-largest-entry-rule-2026-11-17" / "results.json"
CELLS = (("32", 16), ("R", 16), ("R", 32))


def db(x):
    return 10.0 * math.log10(x)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    rows = json.loads((HERE / "results.json").read_text())
    prior = {(r["matrix"], r["group"], r["q"]): r for r in json.loads(PRIOR.read_text()) if "arm" not in r}
    by = {(r["matrix"], r["arm"], r["group"], r["q"]): r for r in rows}
    real = sorted({r["matrix"] for r in rows if r["matrix"].startswith("real/")})

    print("1. The share of the E8 Voronoi code's gain over the scalar rule that integer levels in a ball reach, in dB")
    print("   family | group | q | VE8 over X | ZB over X | ZB over X as a share of VE8 over X")
    for fam in ("G", "T", "real"):
        ms = sorted({r["matrix"] for r in rows if r["matrix"].split("/")[0] == fam})
        for group, q in CELLS:
            ve = np.mean([db(by[(m, "ZB", group, q)]["X"] / by[(m, "ZB", group, q)]["VE8"]) for m in ms])
            zb = np.mean([db(by[(m, "ZB", group, q)]["X"] / by[(m, "ZB", group, q)]["rel_err"]) for m in ms])
            print(f"   {fam:>4} | {group:>2} | {q} | {ve:+.2f} | {zb:+.2f} | {zb / ve:.2f}")
    print()
    print("2. On the real matrices: the better of the two E8 regions on each matrix, against the E8 Voronoi code, in dB")
    print("   One more bit per matrix would say which region it uses.")
    print("   group | q | EB over VE8 | EC over VE8 | better of the two over VE8: mean (min, max) | ahead of VE8 on | EC is the better one on | matrices")
    for group, q in CELLS:
        eb = [db(by[(m, "EB", group, q)]["VE8"] / by[(m, "EB", group, q)]["rel_err"]) for m in real]
        ec = [db(by[(m, "EC", group, q)]["VE8"] / by[(m, "EC", group, q)]["rel_err"]) for m in real]
        best = [max(a, b) for a, b in zip(eb, ec)]
        print(f"   {group:>2} | {q} | {np.mean(eb):+.2f} | {np.mean(ec):+.2f} | {np.mean(best):+.2f} ({min(best):+.2f}, {max(best):+.2f}) | {sum(x > 0 for x in best)} | {sum(b > a for a, b in zip(eb, ec))} | {len(real)}")
    print()
    print("3. Which real matrices the cross-polytope suits: EC over VE8 against the mean peak of a block of the matrix")
    print("   peak = the largest entry of a block over its length, as in the record of the E8 gain: 0.650 for Gaussian blocks, 1 for a single weight")
    print("   group | q | rank correlation of (EC over VE8) with the peak | mean peak where EC is ahead | where it is behind")
    for group, q in CELLS:
        ec = np.array([db(by[(m, "EC", group, q)]["VE8"] / by[(m, "EC", group, q)]["rel_err"]) for m in real])
        pk = np.array([prior[(m, group, q)]["all_peak"] for m in real])
        print(f"   {group:>2} | {q} | {spearman(ec, pk):+.2f} | {pk[ec > 0].mean():.3f} | {pk[ec <= 0].mean():.3f}")
    print()
    print("4. The real matrices in order of EC over VE8 with one group per row at q = 16, in dB")
    print("   matrix | EC over VE8 | EB over VE8 | ZC over VE8 | mean peak of a block")
    order = sorted(real, key=lambda m: db(by[(m, "EC", "R", 16)]["VE8"] / by[(m, "EC", "R", 16)]["rel_err"]))
    for m in order[:5] + order[-5:]:
        g = {a: db(by[(m, a, "R", 16)]["VE8"] / by[(m, a, "R", 16)]["rel_err"]) for a in ("EC", "EB", "ZC")}
        print(f"   {m[5:]:<42} | {g['EC']:+.2f} | {g['EB']:+.2f} | {g['ZC']:+.2f} | {prior[(m, 'R', 16)]['all_peak']:.3f}")
    print()
    print("5. Blocks shrunk: runs where a block lay outside the codebook at its group's scale")
    for arm in ("ZB", "ZC", "EB", "EC"):
        rs = [r for r in rows if r["arm"] == arm]
        bad = [r for r in rs if r["shrunk"] > 0]
        print(f"   {arm}: {len(bad)} of {len(rs)} runs" + (f", largest share of blocks {max(r['shrunk'] for r in bad):.1e}" if bad else ""))


if __name__ == "__main__":
    main()
