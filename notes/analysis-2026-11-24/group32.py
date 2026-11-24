#!/usr/bin/env python3
"""Exploratory, weights only: the lattices of hamming-lattices-real-weights-2026-11-24 in the block of the packed format.

The packed format (packed-ball-format-2026-11-18, shape/ballpack.py) stores 32 weights in 18 bytes:
a one-byte scale code and a 136-bit number of a point of Z^32 in a ball. Could the same 136 bits
number a point of A_16 x A_16, A_32 or A_8^4 (E8 turned) in a ball, and what would the weight
error be?

1. Weight error of each lattice in a ball over 32 weights with 136 bits for the point, by the
   ball arm's rounding (run.py's ball_round, with its 16-bit scale: the one-byte scale code of the
   format is not modelled, so compare the lattices with one another, not with the file). Made-up
   Laplace and Gaussian matrices and three matrices of SmolLM2-1.7B, first 256 rows.
2. The numbering of each ball (shape/ballindex.py for Z^32, shape/hammingindex.py for A_16, A_32):
   bound, log2 of the count, table size, and the table look-ups to number and to read back a
   point, over 2,000 points drawn at random from the ball.

Usage, from the root: python3 notes/analysis-2026-11-24/group32.py SmolLM2-1.7B-Instruct-f16.gguf
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "experiments/hamming-lattices-real-weights-2026-11-24"
sys.path.insert(0, str(ROOT / "shape"))
spec = importlib.util.spec_from_file_location("hl_run", REC / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["hl_run"] = R
spec.loader.exec_module(R)
Q, hamming = R.Q, R.hamming
from ballindex import BallIndex  # noqa: E402
from hammingindex import HammingBallIndex  # noqa: E402

BITS, N = 136, 32
MATRICES = ("blk.1.ffn_down.weight", "blk.5.ffn_down.weight", "blk.10.attn_q.weight")


def arm(lat):
    bd = Q.ball_module()
    if lat in ("Z8", "E8"):
        b, p = Q.ball_bound(lat, N, BITS)
        return (lambda X: bd.closest(lat, X)), b, p, math.sqrt(N) / 2
    if lat == "A8":
        b, p = Q.ball_bound("E8", N, BITS)
        return (lambda X: hamming.closest(X * math.sqrt(2), 3) / math.sqrt(2)), b, p, math.sqrt(N) / 2
    m = {"A16": 4, "A32": 5}[lat]
    b, p = hamming.ball_bound(m, N, BITS)
    return (lambda X: hamming.closest(X, m)), b, p, math.sqrt(hamming.covering_sq(m, N))


def main():
    rng = np.random.default_rng(0)
    mats = {"Laplace": rng.laplace(size=(256, 4096)), "Gaussian": rng.normal(size=(256, 4096))}
    Q._model_path = sys.argv[1]
    mine = {t.name: t for t in Q.reader().tensors}
    for name in MATRICES:
        mats["SmolLM2 " + name] = Q.matrix_of(mine[name]).astype(np.float64)[:256]
    lats = ("Z8", "E8", "A8", "A16", "A32")
    print(f"## 1. Weight error (dB) in a ball over {N} weights, {BITS} bits for the point\n")
    print("| Matrix | " + " | ".join(lats) + " | A16 over Z8 (dB) |")
    print("|---|" + "---|" * (len(lats) + 1))
    for name, W in mats.items():
        errs = []
        for lat in lats:
            c, b, p, mg = arm(lat)
            Wh, _, _ = R.ball_round(W, N, None, c, b, p, mg, 14, 12)
            errs.append(10 * math.log10(float(np.sum((W - Wh) ** 2) / np.sum(W ** 2))))
        print(f"| {name} | " + " | ".join(f"{e:.3f}" for e in errs) + f" | {errs[0] - errs[3]:.3f} |", flush=True)
    print("\n## 2. Numbering a ball over 32 weights in 136 bits\n")
    print("| Lattice | Bound | log2 N | Table entries | Look-ups to read back a point | Look-ups to number a point |")
    print("|---|---|---|---|---|---|")
    cases = [("Z^32 (the packed format)", BallIndex("Z8", N, Q.ball_bound("Z8", N, BITS)[0]))]
    for m in (4, 5):
        b, _ = hamming.ball_bound(m, N, BITS)
        cases.append((f"A{1 << m} over 32", HammingBallIndex(m, N, b)))
    rng = np.random.default_rng(2)
    for name, idx in cases:
        ks = [int.from_bytes(rng.bytes(24), "little") % idx.count for _ in range(2000)]
        idx.lookups = 0
        ys = [idx.unrank(k) for k in ks]
        lu = idx.lookups / len(ks)
        idx.lookups = 0
        assert [idx.rank(y) for y in ys] == ks
        lr = idx.lookups / len(ks)
        print(f"| {name} | {idx.bound} | {math.log2(idx.count):.3f} | {idx.table_entries()} | {lu:.0f} | {lr:.0f} |", flush=True)


if __name__ == "__main__":
    main()
