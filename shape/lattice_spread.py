#!/usr/bin/env python3
"""Where integer levels and E8 put their error, with one ball per group, on a whole model. Weight error only: nothing is scored.

On the third model the run of model-output-smollm2-2026-11-17 scored integer levels in one ball
per group of 128 under E8 in the same ball, though E8 has 0.66 dB less weight error. On Qwen2.5
E8 was ahead by what its weight error says. This program asks what, in the weights alone, differs
between the two lattices and between the two models. It was written after those rows were seen.

Integer levels send a weight under half a step to zero: its error is the weight, however small.
E8 has two cosets, the whole numbers with an even sum and the same moved by a half. In the
second every weight of a block of eight lands on a half, the small ones too. So E8 spends error on
weights that integer levels leave alone, and saves it elsewhere. The sum is what the formula
gives. Where it falls is not in the formula.

For every block matrix and each of the two arms, rounded in full as the harness rounds them:

  matrix    squared error over squared weights
  columns   the mean over the columns of (squared error of a column over its squared weights).
            A column is one input of the matrix. If a trained model keeps the weights on an
            input small where the input is large, this is nearer to the error of the output.
  rows      the same over the rows
  gain      the rounded matrix along the matrix: <rounded, W> / <W, W>. Under one, it is shrunk.
  zeros     the part of the rounded weights that are exactly zero
  small     for the weights under half a step: their part of all weights, and of the error
  by size   the mean squared error, in squared steps, of the weights by their size in steps

and, of the matrix alone: how unevenly its columns carry the weight.

Usage, from the root of a checkout, with NumPy:
  python3 shape/lattice_spread.py MODEL.gguf [STRIDE] > OUT
STRIDE keeps one block of the model in every STRIDE (default 1, all of them).
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(ROOT / "lattice"))
import design  # noqa: E402
import ggufread  # noqa: E402

balldim = design.balldim
KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")
ARMS = (("ZBG128:b560s16", "Z8"), ("EBG128:b560s16", "E8"))
N, BITS = 128, 560
EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))    # sizes of a weight, in steps of its group
_path = None
_bounds = {}


def db(x):
    return 10.0 * math.log10(x)


def ball_round(W64, lattice, bound):
    """One ball per group of 128, the scale the smallest that fits: the rounded matrix and the scale of every weight."""
    rows, cols = W64.shape
    X = W64.reshape(-1, N)
    t = balldim.block_scales(X, lattice, "ball", bound)
    Y = balldim.closest(lattice, X / t[:, None])
    out = balldim.size_of("ball", Y) > bound + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(balldim.SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = balldim.size_of("ball", balldim.closest(lattice, xo * mid[:, None])) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = balldim.closest(lattice, xo * lo[:, None])
    Wh = (Y * t[:, None]).reshape(rows, cols).astype(np.float16).astype(np.float64)
    return Wh, np.repeat(t, N).reshape(rows, cols)


def one(name):
    W = ggufread.Reader(_path).matrix(name).astype(np.float64)
    w2 = W ** 2
    c_w = w2.sum(axis=0)
    col = c_w / c_w.mean()
    res = {"shape": W.shape,
           "col_uneven": float(np.sqrt(np.mean(col ** 2))),          # root mean square of the columns' squared weight over its mean
           "col_small": float(np.mean(col < 0.25)),                  # columns with under a quarter of the mean squared weight
           "col_min": float(col.min())}
    for arm, lattice in ARMS:
        Wh, T = ball_round(W, lattice, _bounds[lattice])
        err = (W - Wh) ** 2
        total = float(err.sum() / w2.sum())
        c_e, r_e, r_w = err.sum(axis=0), err.sum(axis=1), w2.sum(axis=1)
        size = np.abs(W) / T
        small = size < 0.5
        e_steps = err / T ** 2
        bins = []
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            m = (size >= a) & (size < b)
            bins.append((float(m.mean()), float(e_steps[m].mean()) if m.any() else float("nan"), float(err[m].sum() / err.sum())))
        # the columns with the least weight: the tenth of the columns with the smallest squared weight
        low = c_w <= np.quantile(c_w, 0.1)
        low1 = c_w <= np.quantile(c_w, 0.01)
        least = int(np.argmin(c_w))
        res[arm] = {"matrix": total,
                    "columns": float(np.mean(c_e[c_w > 0] / c_w[c_w > 0])),
                    "rows": float(np.mean(r_e[r_w > 0] / r_w[r_w > 0])),
                    "gain": float((Wh * W).sum() / w2.sum()),
                    "zeros": float(np.mean(Wh == 0)),
                    "small": float(small.mean()),
                    "small_err": float(err[small].sum() / err.sum()),
                    "low_cols": float(c_e[low].sum() / c_w[low].sum()),
                    "low1_cols": float(c_e[low1].sum() / c_w[low1].sum()),
                    "least_col": float(c_e[least] / c_w[least]),
                    "least_steps": float(np.sqrt(np.mean(size[:, least] ** 2))),
                    "low1_steps": float(np.sqrt(np.mean(size[:, low1] ** 2))),
                    "rms_steps": float(np.sqrt(np.mean(size ** 2))),
                    "bins": bins}
    return name, res


def main():
    global _path
    _path = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    for _, lattice in ARMS:
        _bounds[lattice] = design.ball_bound(lattice, N, BITS)[0]
    rd = ggufread.Reader(_path)
    names = [n for n in rd.names if BLOCK.match(n) and len(rd.tensors[n]["dims"]) == 2 and int(BLOCK.match(n).group(1)) % stride == 0]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(one, names, chunksize=1)
    weights = sum(r["shape"][0] * r["shape"][1] for _, r in res)
    print(f"File: {Path(_path).name}. One block in every {stride}. Block matrices: {len(res)}. Weights: {weights}. Seconds: {time.time() - t0:.0f}.")
    print(f"Bounds of the two balls: " + ", ".join(f"{lat} {b}" for lat, b in _bounds.items()) + ".")
    print()
    print("Of the matrices alone, by kind: how unevenly the columns carry the squared weight (root mean square over mean: 1 is even) | "
          "columns with under a quarter of the mean | the smallest column over the mean")
    for k in KINDS:
        v = [r for n, r in res if BLOCK.match(n).group(2) == k]
        print(f"{k} | {np.mean([r['col_uneven'] for r in v]):.3f} | {np.mean([r['col_small'] for r in v]):.4f} | {np.mean([r['col_min'] for r in v]):.4f}")
    print()
    print("Mean over the matrices. Errors in dB.")
    print("arm | matrix | columns | rows | columns over matrix | rows over matrix | the tenth of the columns with the least weight, over matrix | gain | "
          "rounded weights that are zero | weights under half a step: their part of the weights, of the error | size of a weight in steps, root mean square")
    for arm, _ in ARMS:
        g = lambda key: [r[arm][key] for _, r in res]                                 # noqa: E731
        m, c, rw, lc = (np.mean([db(x) for x in g(k)]) for k in ("matrix", "columns", "rows", "low_cols"))
        print(f"{arm} | {m:.3f} | {c:.3f} | {rw:.3f} | {c - m:+.2f} | {rw - m:+.2f} | {lc - m:+.2f} | {np.mean(g('gain')):.5f} | {np.mean(g('zeros')):.4f} | "
              f"{np.mean(g('small')):.4f}, {np.mean(g('small_err')):.4f} | {np.mean(g('rms_steps')):.2f}")
    print()
    print("E8 less integer levels, in dB, mean over the matrices of a kind: matrix | columns | the tenth of the columns with the least weight | the hundredth | "
          "the one column with the least (largest over the matrices) | size of a weight in steps, root mean square: in the hundredth, in the one column (smallest over the matrices)")
    (za, _), (ea, _) = ARMS
    for k in KINDS + ("all",):
        v = [r for n, r in res if k == "all" or BLOCK.match(n).group(2) == k]
        d = lambda key: np.mean([db(r[ea][key]) - db(r[za][key]) for r in v])         # noqa: E731
        worst = max(db(r[ea]["least_col"]) - db(r[za]["least_col"]) for r in v)
        print(f"{k} | {d('matrix'):+.2f} | {d('columns'):+.2f} | {d('low_cols'):+.2f} | {d('low1_cols'):+.2f} | {d('least_col'):+.2f} ({worst:+.2f}) | "
              f"{np.mean([r[za]['low1_steps'] for r in v]):.2f}, {np.mean([r[za]['least_steps'] for r in v]):.2f} ({min(r[za]['least_steps'] for r in v):.2f})")
    print()
    print("By the size of a weight in steps of its group: part of the weights | mean squared error in squared steps | part of the error")
    for arm, _ in ARMS:
        print(arm)
        for i, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:])):
            v = [r[arm]["bins"][i] for _, r in res]
            e = [x[1] for x in v if x[1] == x[1]]
            print(f"  {a:g} to {b:g} | {np.mean([x[0] for x in v]):.4f} | {np.mean(e) if e else float('nan'):.4f} | {np.mean([x[2] for x in v]):.4f}")


if __name__ == "__main__":
    main()
