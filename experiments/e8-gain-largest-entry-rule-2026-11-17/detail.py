#!/usr/bin/env python3
"""Tables for the record that results.out does not hold. Written after the run. None of them was planned.

Reads results.json. Table 4 also reads the weights file, when it is given.

Usage, from the root of a checkout:
  python3 experiments/e8-gain-largest-entry-rule-2026-11-17/detail.py --weights /path/to/weights.npz > detail.out
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = 10.0 * math.log10((1.0 / 12.0) / (929.0 / 12960.0))


def db(x):
    return 10.0 * math.log10(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    args = ap.parse_args()
    rows = [r for r in json.loads((HERE / "results.json").read_text()) if "arm" not in r]

    def cell(fam, group, q):
        return [r for r in rows if r["matrix"].split("/")[0] == fam and r["group"] == group and r["q"] == q]

    def gain(r):
        return db(r["X"] / r["VE8"])

    def left(r):
        return gain(r) - SHAPE - r["VE8_scale_term"]

    four = (("32", 16), ("R", 16), ("32", 32), ("R", 32))
    print("1. Predictions 2 and 3 on the Gaussian family, matrix by matrix, in dB")
    print("   matrix | group | q | gain of E8 over X | scale term | gain minus shape minus scale | Z8 over X")
    for group, q in four:
        for r in cell("G", group, q):
            print(f"   {r['matrix']} | {group:>2} | {q} | {gain(r):+.4f} | {r['VE8_scale_term']:+.4f} | {left(r):+.4f} | {db(r['X'] / r['VZ8']):+.4f}")
    print()
    print("2. Prediction 5 on the real matrices: how many have shape plus scale within a margin of the measured gain")
    print("   group | q | mean gain | within 0.5 dB | within 0.3 dB | matrices")
    for group, q in (("32", 16), ("R", 16)):
        rs = cell("real", group, q)
        ls = [abs(left(r)) for r in rs]
        print(f"   {group:>2} | {q} | {np.mean([gain(r) for r in rs]):+.3f} | {sum(x <= 0.5 for x in ls)} | {sum(x <= 0.3 for x in ls)} | {len(rs)}")
    print()
    print("3. The block on the wall. If one block of each group has (1 + e) times the mean error of a block and the rest have the mean,")
    print("   measured error over formula is 1 + e over the number of blocks in a group. The table gives e.")
    print("   For the scalar rule: if one weight of each group has no error, measured over formula is 1 - f over the group size. The table gives f.")
    print("   family | group | blocks in a group | e for Z8 | e for E8 | f for X")
    for fam in ("G", "T", "real"):
        for group in ("8", "32", "128", "R"):
            rs = cell(fam, group, 16)
            if not rs:
                continue
            ez = np.mean([(r["VZ8"] / r["VZ8_model"] - 1.0) * (r["size"] // 8) for r in rs])
            ee = np.mean([(r["VE8"] / r["VE8_model"] - 1.0) * (r["size"] // 8) for r in rs])
            fx = np.mean([(1.0 - r["X"] / r["X_model"]) * r["size"] for r in rs])
            sizes = sorted({r["size"] // 8 for r in rs})
            print(f"   {fam:>4} | {group:>3} | {'/'.join(str(s) for s in sizes)} | {ez:+.3f} | {ee:+.3f} | {fx:+.2f}")
    print("   A point on the surface of a cell in n dimensions is, in the mean, (n + 2) / n times as far in squared distance as a point inside it: e = 2/8 = 0.25.")
    print()
    print("4. The real matrices in order of their gain with groups of 32 at q = 16, lowest first, in dB")
    print("   matrix | gain | scale term | X over its formula | E8 over its formula | Z8 over its formula" + (" | columns with a standard deviation under a tenth of the median | entries under 1/16 of their group's largest" if args.weights else ""))
    z = np.load(args.weights) if args.weights else None
    rs = sorted(cell("real", "32", 16), key=gain)
    for r in rs[:4] + rs[-2:]:
        extra = ""
        if z is not None:
            W = z[r["matrix"][5:]].astype(np.float64)
            cs = W.std(axis=0)
            g = np.abs(W).reshape(-1, 32)
            extra = f" | {float((cs < 0.1 * np.median(cs)).mean()) * 100:.1f} % | {float((g < g.max(axis=1, keepdims=True) / 16.0).mean()) * 100:.1f} %"
        print(f"   {r['matrix'][5:]:<42} | {gain(r):+.2f} | {r['VE8_scale_term']:+.2f} | {db(r['X'] / r['X_model']):+.2f} | {db(r['VE8'] / r['VE8_model']):+.2f} | {db(r['VZ8'] / r['VZ8_model']):+.2f}{extra}")
    if z is not None:
        g = np.abs(np.random.default_rng(1).normal(size=(40000, 32)))
        print(f"   For Gaussian groups of 32 the last column would be {float((g < g.max(axis=1, keepdims=True) / 16.0).mean()) * 100:.1f} %.")
    print()
    print("5. The gains of the second record on gemma beside those of this run, in dB")
    print("   cell | gemma (second record) | Gaussian | Student t | whisper")
    for label, group, q, gemma in (("groups of 32, q = 16", "32", 16, 1.51), ("one group per row, q = 16", "R", 16, 2.52), ("one group per row, q = 32", "R", 32, 2.55)):
        print(f"   {label:<26} | {gemma:.2f} | " + " | ".join(f"{np.mean([gain(r) for r in cell(f, group, q)]):.2f}" for f in ("G", "T", "real")))
    print("   Rows are 2,048 long on the synthetic families and 512 or 2,048 on whisper. The second record does not list the row lengths of gemma.")


    print()
    print("6. The scale term for regions of other shapes with the volume of the E8 code's region, in dB. Not a run of any codebook:")
    print("   each line is 10 log10 of (sum of squared scalar steps over sum of squared scales), with the scale the smallest at which the region holds every block of the group.")
    print("   ball: the scale is set by the longest block. cross-polytope: by the block with the largest sum of sizes of its entries. The cube is the scalar rule, 0 by definition.")
    print("   family | group | E8 region (measured in the run) | ball | cross-polytope | matrices")
    import importlib.util
    spec = importlib.util.spec_from_file_location("study", HERE / "run.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    r_ball = study.R_BALL8
    r_cross = (math.factorial(8) / 2.0 ** 8) ** (1.0 / 8.0)      # the cross-polytope of volume 1 reaches this far along a weight axis
    mats = {f"{k}/{sd}": study.synthetic(k, sd) for k in ("G", "T") for sd in range(3)}
    if z is not None:
        for r in cell("real", "32", 16):
            mats[r["matrix"]] = z[r["matrix"][5:]]
    q = 16
    for fam in ("G", "T", "real"):
        for group in ("32", "R"):
            acc = {"ball": [], "cross": [], "e8": []}
            for r in cell(fam, group, q):
                if r["matrix"] not in mats:
                    continue
                W = np.asarray(mats[r["matrix"]], dtype=np.float64)
                size = W.shape[1] if group == "R" else int(group)
                d = np.abs(W).reshape(-1, size).max(axis=1) / (q / 2.0)
                b = W.reshape(-1, size // 8, 8)
                t_ball = np.sqrt((b ** 2).sum(axis=2)).max(axis=1) / (r_ball * q)
                t_cross = np.abs(b).sum(axis=2).max(axis=1) / (r_cross * q)
                acc["ball"].append(db(float(np.sum(d ** 2)) / float(np.sum(t_ball ** 2))))
                acc["cross"].append(db(float(np.sum(d ** 2)) / float(np.sum(t_cross ** 2))))
                acc["e8"].append(r["VE8_scale_term"])
            if acc["e8"]:
                print(f"   {fam:>4} | {group:>2} | {np.mean(acc['e8']):+.2f} | {np.mean(acc['ball']):+.2f} | {np.mean(acc['cross']):+.2f} | {len(acc['e8'])}")
    print(f"   Reach along a weight axis at volume 1: cube 0.5, ball {r_ball:.3f}, E8 cell 1.0, cross-polytope {r_cross:.3f}. Along the diagonal: cube 1.414, ball {r_ball:.3f}, E8 cell 0.707, cross-polytope {r_cross / math.sqrt(8.0):.3f}.")


if __name__ == "__main__":
    main()
