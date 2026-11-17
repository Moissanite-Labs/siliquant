#!/usr/bin/env python3
"""How evenly a rule spreads its error over the rows of a matrix, on a whole model. Weight error only: nothing is scored.

The run on Qwen2.5 found that with one scale per group of 32 or 128 weights the mean KL of a model
is one constant times the weight error, and that with one scale per row it is 2 to 2.7 times
that. This program asks whether one number computed from the weights says which is which.

For every block matrix of a model and every arm it gives three figures:

  matrix   the squared error of the matrix over its squared weights: the usual weight error
  rows     the mean over the rows of (squared error of a row over its squared weights)
  columns  the same over the columns

Rows are the outputs of a matrix and columns its inputs. If a rule's error is the same share of
every row, "rows" is "matrix". If it gathers in a few rows that are small, "rows" is the larger.
Each is then averaged over the matrices in dB, as the model-output records average weight error.

The arms: the scalar largest-entry rule of shape/shapelib.py at any levels and group, rounded in
full; and, by the formula with nothing rounded, a ball over 128 weights with one scale per row
("B128@R") and one ball per group, whose three figures are equal by construction.

Usage, from the root of a checkout, with NumPy:
  python3 shape/row_spread.py MODEL.gguf X32:16 X128:16 XR:16 XR:32 B128@R > OUT
"""
from __future__ import annotations

import math
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggufread  # noqa: E402

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")
_path = None


def db(x):
    return 10.0 * math.log10(x)


def scalar_error(W, q, group):
    """Squared error of every weight under the scalar largest-entry rule, as shapelib.scalar_rule rounds it."""
    rows, cols = W.shape
    size = cols if group == "R" else int(group)
    X = W.reshape(-1, size)
    j = np.abs(X).argmax(axis=1)[:, None]
    d = np.take_along_axis(X, j, axis=1) / np.float32(-q / 2.0)
    safe = np.where(d == 0, np.float32(1.0), d)
    R = np.clip(np.rint(X / safe), -q // 2, q // 2 - 1).astype(np.float32) * d
    Wh = R.reshape(rows, cols).astype(np.float16).astype(np.float64)
    return (W.astype(np.float64) - Wh) ** 2


def uneven(share, total):
    """How unevenly the rows carry the error: (root mean square of the rows' shares over their mean, the worst row over
    the matrix's share, the part of the rows whose share is over four times the matrix's)."""
    return (float(np.sqrt(np.mean(share ** 2)) / np.mean(share)), float(share.max() / total), float(np.mean(share > 4.0 * total)))


def spread(err, w2):
    """(matrix, rows, columns, and three figures of unevenness) from the squared error and the squared weight of every entry."""
    total = float(err.sum() / w2.sum())
    r_e, r_w = err.sum(axis=1), w2.sum(axis=1)
    c_e, c_w = err.sum(axis=0), w2.sum(axis=0)
    share = r_e[r_w > 0] / r_w[r_w > 0]
    cols = float(np.mean(c_e[c_w > 0] / c_w[c_w > 0]))
    return (total, float(np.mean(share)), cols) + uneven(share, total)


def one(task):
    name, arms = task
    W = ggufread.Reader(_path).matrix(name)
    w2 = W.astype(np.float64) ** 2
    out = {}
    for arm in arms:
        m = re.match(r"^X(\d+|R):(\d+)$", arm)
        if m:
            g = "R" if m.group(1) == "R" else int(m.group(1))
            out[arm] = spread(scalar_error(W, int(m.group(2)), g), w2)
        elif arm == "B128@R":
            # a ball over 128 weights, one scale per row: by the formula the error of a row is the same for each of its
            # blocks, and is set by the longest block. So a row's share is (blocks * longest squared) / (sum of squared).
            b2 = w2.reshape(W.shape[0], -1, 128).sum(axis=2)
            per_row_err = b2.shape[1] * b2.max(axis=1)                 # up to a constant that is the same for every row
            r_w = b2.sum(axis=1)
            ok = r_w > 0
            total = float(per_row_err[ok].sum() / r_w[ok].sum())
            share = per_row_err[ok] / r_w[ok]
            out[arm] = (total, float(np.mean(share)), float("nan")) + uneven(share, total)
        else:
            raise SystemExit(f"unknown arm {arm}")
    return name, W.shape, out


def main():
    global _path
    _path = sys.argv[1]
    arms = sys.argv[2:]
    t0 = time.time()
    rd = ggufread.Reader(_path)
    names = [n for n in rd.names if BLOCK.match(n) and len(rd.tensors[n]["dims"]) == 2]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(one, [(n, arms) for n in names], chunksize=1)
    weights = sum(s[0] * s[1] for _, s, _ in res)
    print(f"File: {Path(_path).name}. Block matrices: {len(res)}. Weights: {weights}. Seconds: {time.time() - t0:.0f}.")
    print("Mean over the matrices, in dB. For B128@R the figures are relative: only 'rows over matrix' means anything.")
    print("uneven = root mean square of the rows' shares over their mean: 1 when every row has the same share of error.")
    print("arm | matrix | rows | columns | rows over matrix | columns over matrix | uneven: mean (largest matrix) | worst row over the matrix's share: mean (largest) | "
          "rows over four times the matrix's share: mean | uneven by kind: " + ", ".join(KINDS))
    for arm in arms:
        m = [db(o[arm][0]) for _, _, o in res]
        r = [db(o[arm][1]) for _, _, o in res]
        c = [db(o[arm][2]) if o[arm][2] == o[arm][2] else float("nan") for _, _, o in res]
        kinds = []
        for k in KINDS:
            v = [o[arm][3] for n, _, o in res if BLOCK.match(n).group(2) == k]
            kinds.append(f"{np.mean(v):.2f}")
        u = [o[arm][3] for _, _, o in res]
        w = [o[arm][4] for _, _, o in res]
        f = [o[arm][5] for _, _, o in res]
        print(f"{arm} | {np.mean(m):.3f} | {np.mean(r):.3f} | {np.mean(c):.3f} | {np.mean(r) - np.mean(m):+.2f} | {np.mean(c) - np.mean(m):+.2f} | "
              f"{np.mean(u):.3f} ({max(u):.2f}) | {np.mean(w):.1f} ({max(w):.0f}) | {np.mean(f):.4f} | " + ", ".join(kinds))


if __name__ == "__main__":
    main()
