"""Tests of hammingindex.py: python3 -m unittest shape/test_hammingindex.py"""
import importlib.util
import itertools
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hammingindex import HammingBallIndex, checks  # noqa: E402
from ballindex import BallIndex  # noqa: E402

REC = os.path.join(HERE, "..", "experiments", "hamming-lattices-real-weights-2026-11-24", "hamming.py")


def brute(m, group, bound):
    """Every point of A_n^k with squared length at most bound, in the order hammingindex defines."""
    n, k = 1 << m, group // (1 << m)
    h = checks(m)
    r = int(math.isqrt(bound))
    order = [0] + [s * v for v in range(1, r + 1) for s in (1, -1)]
    blocks = {}
    for blk in itertools.product(order, repeat=n):
        t = sum(v * v for v in blk)
        if t > bound:
            continue
        s = 0
        for j, v in enumerate(blk):
            if v & 1:
                s ^= h[j]
        if s == 0:
            blocks.setdefault(t, []).append(blk)
    pos = {v: i for i, v in enumerate(order)}
    for t in blocks:
        blocks[t].sort(key=lambda b: [pos[v] for v in b])
    keyed = [(t, i, b) for t in sorted(blocks) for i, b in enumerate(blocks[t])]
    pts = []
    for combo in itertools.product(keyed, repeat=k):
        if sum(c[0] for c in combo) <= bound:
            pts.append(sum((list(c[2]) for c in combo), []))
    return pts


class TestHammingIndex(unittest.TestCase):

    def test_order_and_bijection_small(self):
        for m, group, bound in ((3, 8, 6), (3, 16, 4), (2, 8, 5)):
            idx = HammingBallIndex(m, group, bound)
            pts = brute(m, group, bound)
            self.assertEqual(idx.count, len(pts))
            for i, p in enumerate(pts):
                self.assertEqual(idx.rank(p), i)
                self.assertEqual(list(idx.unrank(i)), p)

    def test_e8_count(self):
        # A_8 is E8 turned by H8/sqrt(8), times sqrt(2): same theta series with squared lengths doubled
        for group, bound in ((16, 6), (32, 5), (128, 3)):
            a = HammingBallIndex(3, group, 2 * bound)
            e = BallIndex("E8", group, bound)
            self.assertEqual(a.count, e.count)

    def test_shell_counts(self):
        idx = HammingBallIndex(4, 16, 8)
        # A_16: one point at 0, none at 1 to 3 (the code has no word of weight under 4), and at 4 the
        # 32 points +-2 e_i and 16 sign patterns on each of the 140 words of weight 4: 2272
        self.assertEqual(idx.block_count[:5], [1, 0, 0, 0, 32 + 140 * 16])

    def test_counts_match_record(self):
        spec = importlib.util.spec_from_file_location("hamming_rec", REC)
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
        for m, bits in ((4, 368), (5, 368)):
            bound, points = H.ball_bound(m, 128, bits)
            idx = HammingBallIndex(m, 128, bound)
            self.assertEqual(idx.count, points)
            self.assertLessEqual(idx.bits(), bits)

    def test_round_trip_full_size(self):
        rng = np.random.default_rng(0)
        idx = HammingBallIndex(4, 128, 651)
        for _ in range(40):
            k = int(rng.integers(0, 2 ** 62)) * (idx.count // 2 ** 62) + int(rng.integers(0, 2 ** 30))
            k %= idx.count
            y = idx.unrank(k)
            self.assertTrue(idx.contains(y))
            self.assertEqual(idx.rank(y), k)
        for k in (0, 1, idx.count - 1):
            self.assertEqual(idx.rank(idx.unrank(k)), k)

    def test_refuses(self):
        idx = HammingBallIndex(3, 8, 6)
        with self.assertRaises(ValueError):
            idx.rank([1, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            idx.unrank(idx.count)


if __name__ == "__main__":
    unittest.main()
