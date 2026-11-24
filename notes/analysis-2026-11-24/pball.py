#!/usr/bin/env python3
"""Exploratory, weights only: integer levels in a p-norm ball over 128 weights, at 4.5 bits a weight.

shape-formula-held-out-2026-11-17 found a p-norm ball with p = 5/4 or 3/2 best over 8 weights,
0.2 to 0.3 dB ahead of the ball. Over long blocks only p = 1 and p = 2 were run
(ball-dimension-no-clip-rule-2026-11-17). This checks p = 2, 3/2, 5/4 and 1 over 128 weights.

The region: integer points whose entries' costs round(8 |v|^p) sum to at most B, B the largest
with at most 2^560 points, counted in floating point (a relative error near 1e-13, enough to set
the bound; nothing here is a format). One 16-bit scale per group, set by bisection so the rounded
point is in the region, as the ball arm does. Made-up Laplace and Gaussian matrices and three
matrices of SmolLM2-1.7B, first 256 rows.

Usage, from the root: python3 notes/analysis-2026-11-24/pball.py SmolLM2-1.7B-Instruct-f16.gguf
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hl_run", ROOT / "experiments/hamming-lattices-real-weights-2026-11-24/run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["hl_run"] = R
spec.loader.exec_module(R)
Q = R.Q

N, BITS, K = 128, 560, 8.0
MATRICES = ("blk.1.ffn_down.weight", "blk.5.ffn_down.weight", "blk.10.attn_q.weight")


def costs(p, vmax):
    return np.rint(K * np.arange(vmax + 1) ** p).astype(np.int64)


def log2count(p, B):
    vmax = int((B / K) ** (1 / p)) + 1
    c = costs(p, vmax)
    c = c[c <= B]
    row = np.zeros(B + 1)
    row[0] = 1.0
    scale = 0.0
    for _ in range(N):
        new = row.copy()
        for v in range(1, len(c)):
            new[c[v]:] += 2 * row[:B + 1 - c[v]]
        mx = new.max()
        new /= mx
        scale += math.log2(mx)
        row = new
    return math.log2(np.cumsum(row)[B]) + scale


def bound_for(p):
    lo, hi = 1, 1
    while log2count(p, hi) <= BITS:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if log2count(p, mid) <= BITS:
            lo = mid
        else:
            hi = mid
    return lo, log2count(p, lo)


def round_p(W, p, B):
    X = W.reshape(-1, N)
    g = np.maximum(np.abs(X).max(axis=1), 1e-30)
    vmax = int((B / K) ** (1 / p)) + 2
    c = costs(p, vmax + 50)

    def cost(Y):
        return c[np.minimum(np.abs(Y).astype(np.int64), len(c) - 1)].sum(axis=1)

    lo, hi = g / (vmax + 1) * 0.5, g * 4.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        ok = cost(np.rint(X / mid[:, None])) <= B
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    Y = np.rint(X / hi[:, None])
    assert np.all(cost(Y) <= B)
    return (Y * hi[:, None]).reshape(W.shape)


def main():
    rng = np.random.default_rng(0)
    mats = {"Laplace": rng.laplace(size=(256, 4096)), "Gaussian": rng.normal(size=(256, 4096))}
    Q._model_path = sys.argv[1]
    mine = {t.name: t for t in Q.reader().tensors}
    for name in MATRICES:
        mats["SmolLM2 " + name] = Q.matrix_of(mine[name]).astype(np.float64)[:256]
    print(f"Weight error (dB) of integer levels in a p-norm ball over {N} weights, {BITS} bits for the point and 16 for the scale\n")
    print("| p | Bound | Bits a weight | " + " | ".join(mats) + " |")
    print("|---|---|---|" + "---|" * len(mats))
    for p in (2.0, 1.5, 1.25, 1.0):
        B, lb = bound_for(p)
        errs = []
        for W in mats.values():
            Wh = round_p(W, p, B)
            errs.append(10 * math.log10(float(np.sum((W - Wh) ** 2) / np.sum(W ** 2))))
        print(f"| {p:g} | {B} | {(lb + 16) / N:.4f} | " + " | ".join(f"{e:.3f}" for e in errs) + " |", flush=True)


if __name__ == "__main__":
    main()
