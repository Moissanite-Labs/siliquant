#!/usr/bin/env python3
"""Further tables from results.json, written after the run. Nothing here was in the plan.

Usage, from the root of a checkout:
  python3 experiments/lattice-vs-scalar-rounding-2026-11-15/detail.py > experiments/lattice-vs-scalar-rounding-2026-11-15/detail.out
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def db(x):
    return 10.0 * math.log10(x)


def main():
    rows = json.loads((HERE / "results.json").read_text())
    by = {(r["matrix"], r["q"], r["arm"]): r for r in rows}

    def gain(m, q, arm, base):
        return db(by[(m, q, base)]["rel_err"] / by[(m, q, arm)]["rel_err"])

    print("1. Arm S on the Gaussian family, each seed, in dB")
    for q in (3, 4, 8, 16):
        print(f"   q = {q:>2}: " + ", ".join(f"{db(by[(f'G/{s}', q, 'S')]['rel_err']):.4f}" for s in range(3)))
    print("   Scalar floors from bounds/bounds.out: -7.2086 (q = 3), -9.3003 (q = 4). One standard error of the error on one")
    print("   matrix of 262,144 weights is about 0.016 dB at q = 3.")
    print()
    print("2. Exact search against the shrink rule on the same matrix, q = 4")
    print("   matrix | arm | scalar | shrink rule | exact search | lost to the rule | blocks out of range at the chosen scale")
    for m in ("G/0", "RC/0", "real/encoder.blocks.0.attn.query.weight", "real/decoder.blocks.5.attn.out.weight"):
        for pre in ("", "D+"):
            for lat in ("A8*", "E8"):
                a, b, s = by[(m, 4, pre + lat)], by[(m, 4, pre + lat + "-opt")], by[(m, 4, pre + "S")]
                print(f"   {m.replace('real/', '')} | {pre + lat} | {db(s['rel_err']):.2f} | {db(a['rel_err']):.2f} | {db(b['rel_err']):.2f} | "
                      f"{db(a['rel_err'] / b['rel_err']):.2f} dB | {a['overload']:.2f}")
    print()
    print("3. Real matrices by kind, mean gain in dB")
    print("   q | kind | matrices | D+A8* over D+S | D+E8 over D+S | D+S over S | E8 over S | D+E8 over S")
    for q in (4, 8):
        kinds = collections.defaultdict(list)
        for (m, qq, arm) in by:
            if m.startswith("real/") and qq == q and arm == "S":
                kind = "MLP" if ".mlp" in m else ("cross-attention" if "cross_attn" in m else "self-attention")
                kinds[kind].append([gain(m, q, "D+A8*", "D+S"), gain(m, q, "D+E8", "D+S"), gain(m, q, "D+S", "S"), gain(m, q, "E8", "S"), gain(m, q, "D+E8", "S")])
        every = [v for vs in kinds.values() for v in vs]
        for kind, vs in sorted(kinds.items()) + [("all", every)]:
            print(f"   {q} | {kind} | {len(vs)} | " + " | ".join(f"{sum(v[i] for v in vs) / len(vs):+.2f}" for i in range(5)))
        print(f"   {q} | D+E8 over S across all {len(every)}: least {min(v[4] for v in every):+.2f}, most {max(v[4] for v in every):+.2f}")
    worse = sorted((m.replace("real/", ""), round(gain(m, 4, "D+A8*", "D+S"), 3)) for (m, q, arm) in by
                   if m.startswith("real/") and q == 4 and arm == "S" and gain(m, 4, "D+A8*", "D+S") <= 0)
    print(f"   real matrices at q = 4 where D+A8* is not better than D+S: {worse}")
    print()
    print("4. Share of blocks (weights for S) out of range at the chosen scale, Gaussian family, mean of three seeds")
    for q in (4, 16):
        print(f"   q = {q:>2}: " + ", ".join(f"{a} {sum(by[(f'G/{s}', q, a)]['overload'] for s in range(3)) / 3:.3f}" for a in ("S", "Z8", "A8*", "E8")))
    print()
    print("5. Bits per weight of the balanced arms at q = 4, with the two scale vectors counted: "
          + ", ".join(f"{b:.4f}" for b in sorted({r["bits"] for r in rows if r["arm"] == "D+S" and r["q"] == 4})))
    print(f"   Largest imbalance left after balancing on a real matrix: {max(r.get('imbalance_after', 0) for r in rows if r['matrix'].startswith('real/')):.3f}")


if __name__ == "__main__":
    main()
