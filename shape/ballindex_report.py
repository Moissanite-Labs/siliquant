#!/usr/bin/env python3
"""What numbering the points of the ball-cut codebooks of the records costs: bits, table size, look-ups.

For each codebook it builds the table of shape/ballindex.py, checks its count against the count
the record ball-dimension-no-clip-rule-2026-11-17 made another way (by powers of a polynomial),
rounds Gaussian blocks into the codebook, numbers them and reads them back.

Usage, from the root of a checkout:
  python3 shape/ballindex_report.py > shape/ballindex.out
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(ROOT / "lattice"))
import ballindex as B  # noqa: E402
import lattices as L  # noqa: E402

_spec = importlib.util.spec_from_file_location("balldim", ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17" / "run.py")
balldim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(balldim)

CODEBOOKS = (("Z8", 8, 16), ("Z8", 32, 16), ("Z8", 128, 16), ("Z8", 32, 32), ("Z8", 128, 32),
             ("E8", 8, 16), ("E8", 16, 16), ("E8", 32, 16), ("E8", 128, 16), ("E8", 8, 32), ("E8", 32, 32))
BLOCKS = 200


def round_into(index, x, e8):
    scale = math.sqrt(float((x ** 2).sum()) / index.bound)
    while True:
        y = np.rint(x / scale) if index.lattice == "Z8" else e8.closest((x / scale).reshape(-1, 8)).reshape(-1)
        if index.contains(y):
            return y
        scale *= 1.01


def main():
    e8 = L.E8()
    print("Numbering the points of ball-cut codebooks (shape/ballindex.py)")
    print("lattice | n | q | bound | points equal to the record's count | bits of a number | bits per weight | n log2(q) | table entries | bits of the largest entry | table in megabytes if held exactly | "
          "look-ups per weight to number a block | to read it back | blocks read back equal | seconds to build the table")
    for lattice, n, q in CODEBOOKS:
        bound, points = balldim.codebook(lattice, "ball", n, q)
        t0 = time.time()
        ix = B.BallIndex(lattice, n, bound)
        built = time.time() - t0
        rng = np.random.default_rng(1000 + n + q)
        ok = 0
        ix.lookups = 0
        ys = [round_into(ix, rng.normal(size=n), e8) for _ in range(BLOCKS)]
        ks = [ix.rank(y) for y in ys]
        up = ix.lookups
        ix.lookups = 0
        for y, k in zip(ys, ks):
            ok += bool(np.array_equal(ix.unrank(k), y))
        down = ix.lookups
        width = ix.count.bit_length()
        print(f"{lattice} | {n} | {q} | {bound} | {ix.count == points} | {ix.bits()} | {ix.bits() / n:.4f} | {n * math.log2(q):.0f} | {ix.table_entries()} | {width} | "
              f"{ix.table_entries() * math.ceil(width / 8) / 1e6:.2f} | {up / (BLOCKS * n):.2f} | {down / (BLOCKS * n):.2f} | {ok} of {BLOCKS} | {built:.1f}")
    print()
    print("A look-up is one entry of the table, a whole number of up to the width shown, compared with the number being decoded and taken from it.")
    print("For E8 the table has a row for each entry of a block, each half of E8 and each parity, and sizes are counted in quarters, so it is larger.")


if __name__ == "__main__":
    main()
