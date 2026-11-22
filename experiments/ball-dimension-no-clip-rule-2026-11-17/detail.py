#!/usr/bin/env python3
"""Tables for the record that results.out does not hold: the counts the predictions need, matrix by matrix.

Written after the run. Table 1 scores predictions 6 to 8 of the plan. Tables 2 and 3 were not planned.

Usage, from the root of a checkout:
  python3 experiments/ball-dimension-no-clip-rule-2026-11-17/detail.py > detail.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELLS = (("32", 16), ("128", 16), ("R", 16), ("R", 32))
LARGEST = {"32": 32, "128": 128, "R": 128}


def db(x):
    return 10.0 * math.log10(x)


def main():
    rows = json.loads((HERE / "results.json").read_text())
    by = {(r["matrix"], r["arm"], r["n"], r["group"], r["q"]): r for r in rows}
    fams = ("G", "T", "whisper", "qwen")

    def mats(f):
        return sorted({r["matrix"] for r in rows if r["matrix"].split("/")[0] == f})

    def gain(m, arm, n, group, q, ref=("X", 1)):
        return db(by[(m, ref[0], ref[1], group, q)]["rel_err"] / by[(m, arm, n, group, q)]["rel_err"])

    print("1. Predictions 6 to 8 on the real matrices, in dB. n is the largest block of the cell: 32 with groups of 32, 128 otherwise.")
    print("   family | group | q | EB at n over EB at 8: mean (min, max) | ahead on | EB at n over VE8: mean (min, max) | ahead on | ZB at n over VE8: mean (min, max) | ahead on | matrices")
    for f in ("whisper", "qwen"):
        for group, q in CELLS:
            n = LARGEST[group]
            a = [gain(m, "EB", n, group, q, ("EB", 8)) for m in mats(f)]
            b = [gain(m, "EB", n, group, q, ("VE8", 8)) for m in mats(f)]
            c = [gain(m, "ZB", n, group, q, ("VE8", 8)) for m in mats(f)]
            print(f"   {f:>7} | {group:>3} | {q} | {np.mean(a):+.2f} ({min(a):+.2f}, {max(a):+.2f}) | {sum(x > 0 for x in a)} | "
                  f"{np.mean(b):+.2f} ({min(b):+.2f}, {max(b):+.2f}) | {sum(x > 0 for x in b)} | {np.mean(c):+.2f} ({min(c):+.2f}, {max(c):+.2f}) | {sum(x > 0 for x in c)} | {len(a)}")
    print()
    print("2. One ball for each group: the relative squared error in dB over all the matrices of a family, mean (min, max)")
    print("   With one block to a group the scale divides each group by its own length, so the error of a group is a share of its own size.")
    print("   family | EB, n = 32, groups of 32, q = 16 | EB, n = 128, groups of 128, q = 16 | ZB, n = 128, groups of 128, q = 16 | X, groups of 32 | X, groups of 128 | matrices")
    for f in fams:
        cols = []
        for arm, n, group in (("EB", 32, "32"), ("EB", 128, "128"), ("ZB", 128, "128"), ("X", 1, "32"), ("X", 1, "128")):
            v = [db(by[(m, arm, n, group, 16)]["rel_err"]) for m in mats(f)]
            cols.append(f"{np.mean(v):.2f} ({min(v):.2f}, {max(v):.2f})")
        print(f"   {f:>7} | " + " | ".join(cols) + f" | {len(mats(f))}")
    print()
    print("3. Ball against cross-polytope with E8, at the largest block of each cell, on the real matrices: EC over EB in dB")
    print("   family | group | q | mean (min, max) | the cross-polytope is ahead on | matrices")
    for f in ("whisper", "qwen"):
        for group, q in CELLS:
            n = LARGEST[group]
            v = [gain(m, "EC", n, group, q, ("EB", n)) for m in mats(f)]
            print(f"   {f:>7} | {group:>3} | {q} | {np.mean(v):+.2f} ({min(v):+.2f}, {max(v):+.2f}) | {sum(x > 0 for x in v)} | {len(v)}")
    print()
    cut = [r for r in rows if r["arm"] not in ("X", "VE8")]
    print("4. Blocks shrunk, by arm: runs with any, and the largest share of blocks")
    for arm in ("ZB", "EB", "ZC", "EC"):
        rs = [r for r in cut if r["arm"] == arm]
        bad = [r for r in rs if r["shrunk"] > 0]
        print(f"   {arm}: {len(bad)} of {len(rs)} runs" + (f", largest share {max(r['shrunk'] for r in bad):.1e}" if bad else ""))


if __name__ == "__main__":
    main()
