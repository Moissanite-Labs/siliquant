#!/usr/bin/env python3
"""What a region other than a ball would be worth when one region spans a whole group. Arithmetic on weights: nothing is rounded.

With one ball per group the region part of the formula of issue #10 reads no weight. With another
region it does: for the ball of a p-norm over a group of n weights it is

    Phi_p  =  volume(unit ball of the p-norm in n dimensions)^(2/n) * n * (sum over groups of the squared p-norm) / (sum of squared weights).

This prints 10 log10(Phi_ball / Phi_p) for p = 1, 5/4, 3/2 and 3: what the formula says such a
region would gain over the ball, before the count of its codebook and before any run. No such
codebook over more than 8 weights has been built for p other than 1 and 2.

The matrices are the held-out ones of the record shape-formula-held-out-2026-11-17, read by its
program: three synthetic families, 64 matrices of whisper-base.en and 21 of Qwen2.5-0.5B.

Usage, from the root of a checkout, with NumPy:
  python3 shape/group_shape.py --whisper weights.npz --qwen model.safetensors > shape/group_shape.out
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(ROOT / "experiments" / "shape-formula-held-out-2026-11-17"))
import run  # noqa: E402
import shapelib as S  # noqa: E402

SHAPES = (Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(2), Fraction(3))
GROUPS = (8, 32, 128)
FAMILIES = ("G", "L", "T3", "whisper", "qwen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", required=True)
    ap.add_argument("--qwen", required=True)
    args = ap.parse_args()
    run._files["whisper"], run._files["qwen"] = args.whisper, args.qwen
    tasks = [t for t in run.tasks_for("held", args.whisper, args.qwen) if t["family"] in FAMILIES]
    res = {}
    for t in tasks:
        W = np.ascontiguousarray(run.load(t), dtype=np.float32).astype(np.float64)
        for n in GROUPS:
            X = W.reshape(-1, n)
            total = float((X ** 2).sum())
            phi = {}
            for p in SHAPES:
                shape = S.Shape(p)
                phi[str(p)] = shape.unit_volume(n) ** (2.0 / n) * n * float((shape.norm(X) ** 2).sum()) / total
            res.setdefault((t["family"], n), []).append({k: 10.0 * math.log10(phi["2"] / v) for k, v in phi.items()})
    print("What a p-norm region would gain over a ball with one region per group of n weights, by the formula, in dB. + means the p-norm region is ahead.")
    print("family | n | p = 1: mean (min, max) | p = 5/4 | p = 3/2 | p = 3 | the best p: counts | the best p over the ball: mean | matrices")
    for fam in FAMILIES:
        for n in GROUPS:
            rows = res[(fam, n)]
            cells = []
            for p in ("1", "5/4", "3/2", "3"):
                v = [r[p] for r in rows]
                cells.append(f"{np.mean(v):+.2f} ({min(v):+.2f}, {max(v):+.2f})")
            best = Counter(max(r, key=r.get) for r in rows).most_common()
            print(f"{fam:>7} | {n:3d} | " + " | ".join(cells) + " | " + ", ".join(f"{k}: {c}" for k, c in best)
                  + f" | {np.mean([max(r.values()) for r in rows]):+.2f} | {len(rows)}")


if __name__ == "__main__":
    main()
