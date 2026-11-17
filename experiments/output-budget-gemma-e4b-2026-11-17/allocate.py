#!/usr/bin/env python3
"""The allocation rule of Part B of PREREG.md: bits by equal gain, exhaustive over 9^7 choices.

Reads the results of Part A (the lines X32:Q@KIND for Q = 8, 16, 32) and prints the best arm and
the reversed arm at each budget with the mean KL predicted for each. Nothing is scored here.

The rule, as in the plan:
  - a kind may get 4, 6, 8, 12, 16, 22, 32, 46 or 64 levels, one scale per 32 weights;
    its bits are log2(levels) + 0.5;
  - the KL of a kind at a number of bits is read off its three points (3.5, 4.5, 5.5 bits): a straight
    line in the logarithm of KL between them, the end slopes beyond them;
  - the KL of a mixed arm is the sum over kinds;
  - the best arm has the smallest predicted KL among the choices whose mean bits per weight, over
    all the block weights, is at most the budget. Ties go to the choice with fewer bits, then to
    the first in the order of the levels;
  - the reversed arm has the largest predicted KL among the choices that use the same levels as the
    best arm in any order over the kinds, with mean bits between the budget less 0.05 and the budget.
    If there is none the window is widened by 0.05 at a time, and the output says so.

With --scale-bits 8 (the joined arm of the plan) a kind's bits are log2(levels) + 8/32 plus 16 for
each row of the kind's matrices, shared over the kind's weights, and only the best arm at the first
budget is printed. The KL points are still those of Part A.

Usage: python3 allocate.py --results results.jsonl [--budgets 4.5 3.5] [--scale-bits 8]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
LEVELS = (4, 6, 8, 12, 16, 22, 32, 46, 64)
POINTS = (8, 16, 32)
EPS = 1e-9


def predicted_kl(kl_points, levels):
    """KL at each of `levels` for one kind. kl_points: its KL at Q = 8, 16, 32. Straight in log KL against log2(levels)."""
    x0, x1, x2 = (math.log2(q) for q in POINTS)
    y0, y1, y2 = (math.log(k) for k in kl_points)
    out = []
    for q in levels:
        x = math.log2(q)
        if x <= x1:
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        else:
            y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        out.append(math.exp(y))
    return np.array(out)


def best_choice(kl, weights, bits, budget):
    """The choice of one option per kind with the smallest summed KL whose weighted mean bits is at most the budget.

    kl, bits: arrays (kinds, options): the KL and the bits per weight of each option of each kind.
    weights: the number of weights of each kind. Returns (indices per kind, summed KL, mean bits).
    Exhaustive. Ties go to the choice with fewer bits, then to the first in index order.
    """
    k, n = kl.shape
    total_w = float(np.sum(weights))
    tot_kl = np.zeros(1)
    tot_bits = np.zeros(1)
    for i in range(k):
        tot_kl = (tot_kl[:, None] + kl[i][None, :]).reshape(-1)
        tot_bits = (tot_bits[:, None] + (weights[i] * bits[i])[None, :]).reshape(-1)
    mean = tot_bits / total_w
    ok = np.nonzero(mean <= budget + EPS)[0]
    if len(ok) == 0:
        raise SystemExit("no choice fits the budget")
    lowest = tot_kl[ok].min()
    near = ok[tot_kl[ok] <= lowest * (1 + 1e-12)]
    order = np.lexsort((near, mean[near]))
    pick = int(near[order[0]])
    return np.unravel_index(pick, (n,) * k), float(tot_kl[pick]), float(mean[pick])


def reversed_choice(kl, weights, bits, option_levels, best_idx, budget, step=0.05):
    """The largest summed KL among the orderings of the best choice's levels with mean bits in [budget - w, budget].

    w starts at 0.05 and is widened by 0.05 until there is an ordering. Returns (indices, KL, mean bits, window).
    """
    k = kl.shape[0]
    total_w = float(np.sum(weights))
    chosen = tuple(best_idx)
    orderings = sorted(set(itertools.permutations(chosen)))
    scored = []
    for p in orderings:
        s = float(sum(kl[i][p[i]] for i in range(k)))
        mean = float(sum(weights[i] * bits[i][p[i]] for i in range(k))) / total_w
        scored.append((p, s, mean))
    window = step
    for _ in range(200):
        inside = [x for x in scored if budget - window - EPS <= x[2] <= budget + EPS]
        if inside:
            top = max(x[1] for x in inside)
            near = sorted(x for x in inside if x[1] >= top * (1 - 1e-12))
            return near[0][0], near[0][1], near[0][2], window
        window += step
    raise SystemExit("no reversed arm")


def read_part_a(path):
    rows = {}
    for ln in Path(path).read_text().splitlines():
        r = json.loads(ln)
        if "kl_mean" in r:
            rows[r["variant"]] = r
    kl, weights, row_count = {}, {}, {}
    for kind in KINDS:
        try:
            kl[kind] = [rows[f"X32:{q}@{kind}"]["kl_mean"] for q in POINTS]
            weights[kind] = rows[f"X32:16@{kind}"]["weights_changed"]
            row_count[kind] = rows[f"X32:16@{kind}"]["rows_changed"]
        except KeyError as e:
            raise SystemExit(f"Part A is not complete: {e} is missing")
    return kl, weights, row_count


def variant_name(levels, scale_bits):
    mod = "" if scale_bits == 16 else f"s{scale_bits}"
    return f"X32{mod}mix:" + ",".join(f"{k}={levels[k]}" for k in KINDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[4.5, 3.5])
    ap.add_argument("--scale-bits", type=int, default=16)
    args = ap.parse_args()
    kl_pts, weights, row_count = read_part_a(args.results)
    w = np.array([weights[k] for k in KINDS], dtype=float)
    print("Part A: KL of each kind alone at 8, 16 and 32 levels, and its weights")
    for k in KINDS:
        print(f"   {k}: {kl_pts[k][0]:.6f} {kl_pts[k][1]:.6f} {kl_pts[k][2]:.6f}   weights {weights[k]}   rows {row_count[k]}")
    print(f"   scales stored in {args.scale_bits} bits" + ("" if args.scale_bits == 16 else "; a row's scale is counted too"))
    kl = np.array([predicted_kl(kl_pts[k], LEVELS) for k in KINDS])
    if args.scale_bits == 16:
        bits = np.array([[math.log2(q) + 0.5 for q in LEVELS] for _ in KINDS])
    else:
        bits = np.array([[math.log2(q) + args.scale_bits / 32.0 + 16.0 * row_count[k] / weights[k] for q in LEVELS] for k in KINDS])
    budgets = args.budgets[:1] if args.scale_bits != 16 else args.budgets
    for budget in budgets:
        print()
        print(f"Budget {budget} bits per weight over all {int(w.sum())} block weights")
        idx, s, mean = best_choice(kl, w, bits, budget)
        levels = {k: LEVELS[i] for k, i in zip(KINDS, idx)}
        print(f"   best arm: " + " ".join(f"{k}={levels[k]}" for k in KINDS) + f" | mean bits {mean:.4f} | predicted KL {s:.6f}")
        print("JSON " + json.dumps({"budget": budget, "role": "best", "variant": variant_name(levels, args.scale_bits), "levels": levels,
                                    "mean_bits": mean, "predicted_kl": s, "scale_bits": args.scale_bits, "score": True}))
        if args.scale_bits == 16:
            ridx, rs, rmean, window = reversed_choice(kl, w, bits, LEVELS, idx, budget)
            rlevels = {k: LEVELS[i] for k, i in zip(KINDS, ridx)}
            if tuple(ridx) == tuple(idx):
                print("   NOTE: the reversed arm is the same as the best arm")
            note = "" if abs(window - 0.05) < 1e-9 else f" (the window had to be widened to {window:.2f})"
            print(f"   reversed arm: " + " ".join(f"{k}={rlevels[k]}" for k in KINDS) + f" | mean bits {rmean:.4f} | predicted KL {rs:.6f}{note}")
            print("JSON " + json.dumps({"budget": budget, "role": "reversed", "variant": variant_name(rlevels, 16), "levels": rlevels,
                                        "mean_bits": rmean, "predicted_kl": rs, "window": window, "scale_bits": 16, "score": True,
                                        "same_as_best": tuple(ridx) == tuple(idx)}))


if __name__ == "__main__":
    main()
