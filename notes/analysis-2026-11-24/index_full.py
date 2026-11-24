#!/usr/bin/env python3
"""Numbering of the record's balls at full size, with shape/hammingindex.py.

For each arm of hamming-lattices-real-weights-2026-11-24 (A_16 or A_32 repeated over a group of
128 weights; A_8, which is E8 turned times the root of 2, in the units where its points are
integers), at 4.5 and 3.0 bits: the bound, log2 of the count of points, the bits a stored number
takes, the size of the tables, the time to build them, and the time and table look-ups to read
back and to number one group, over 30 numbers drawn at random. Every number is read back to a
point of the ball and numbered again to itself. The count is checked against the record's
hamming.ball_bound. Python, one core.

Usage, from the root: python3 notes/analysis-2026-11-24/index_full.py
"""
import importlib.util
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shape"))
from hammingindex import HammingBallIndex  # noqa: E402

spec = importlib.util.spec_from_file_location("H", ROOT / "experiments/hamming-lattices-real-weights-2026-11-24/hamming.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def main():
    rng = np.random.default_rng(1)
    print("| Lattice | Bits for the point | Bound | log2 N | Stored bits + 16 for the scale, a weight | Table entries | Build (s) | Read back (ms) | Number (ms) | Look-ups to read back |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m, bits in ((4, 368), (5, 368), (3, 368), (4, 560), (5, 560), (3, 560)):
        bound, points = H.ball_bound(m, 128, bits)
        t0 = time.time()
        idx = HammingBallIndex(m, 128, bound)
        tb = time.time() - t0
        assert idx.count == points
        ks = [int.from_bytes(rng.bytes(80), "little") % idx.count for _ in range(30)]
        idx.lookups = 0
        t0 = time.time()
        ys = [idx.unrank(k) for k in ks]
        tu = (time.time() - t0) / len(ks)
        lu = idx.lookups / len(ks)
        t0 = time.time()
        rs = [idx.rank(y) for y in ys]
        tr = (time.time() - t0) / len(ks)
        assert rs == ks and all(idx.contains(y) for y in ys)
        print(f"| A{1 << m} | {bits} | {bound} | {math.log2(idx.count):.3f} | {idx.bits()} + 16 = {(idx.bits() + 16) / 128:.5f} | {idx.table_entries()} | "
              f"{tb:.0f} | {tu * 1e3:.1f} | {tr * 1e3:.1f} | {lu:.0f} |", flush=True)
    print(f"\nLargest resident memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


if __name__ == "__main__":
    main()
