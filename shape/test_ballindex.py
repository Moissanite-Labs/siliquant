"""Tests of shape/ballindex.py: the numbering of a ball-cut codebook is one-to-one and onto.

Usage, from the root of a checkout:
  python3 -m unittest shape/test_ballindex.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(ROOT / "lattice"))
import ballindex as B  # noqa: E402
import lattices as L  # noqa: E402


def round_into(index, x):
    """Round a block to the lattice at the smallest of a ladder of scales at which the point is in the codebook."""
    e8 = L.E8()
    scale = np.sqrt((x ** 2).sum() / index.bound)
    while True:
        y = np.rint(x / scale) if index.lattice == "Z8" else e8.closest((x / scale).reshape(-1, 8)).reshape(-1)
        if index.contains(y):
            return y
        scale *= 1.01


class Exhaustive(unittest.TestCase):
    def check(self, lattice, n, bound, count):
        ix = B.BallIndex(lattice, n, bound)
        self.assertEqual(ix.count, count)
        points = [ix.unrank(k) for k in range(ix.count)]
        self.assertTrue(all(ix.contains(p) for p in points))
        self.assertEqual(len({tuple(p) for p in points}), ix.count)
        for k, p in enumerate(points):
            self.assertEqual(ix.rank(p), k)

    def test_integers(self):
        self.check("Z8", 1, 9, 7)
        self.check("Z8", 2, 5, 21)                 # the points of the plane within the circle of squared radius 5
        self.check("Z8", 3, 6, 81)
        self.check("Z8", 4, 4, 89)

    def test_e8_shells(self):
        self.check("E8", 8, 0, 1)
        self.check("E8", 8, 2, 241)                # 240 points of squared length 2
        self.check("E8", 8, 4, 2401)               # and 2160 of squared length 4
        self.check("E8", 16, 2, 481)               # two blocks: the origin, or a short vector in one of them

    def test_e8_points_are_in_the_lattice(self):
        ix = B.BallIndex("E8", 8, 4)
        e8 = L.E8()
        for k in range(0, ix.count, 7):
            p = ix.unrank(k)
            self.assertTrue(np.array_equal(e8.closest(p[None, :])[0], p))


class Refusals(unittest.TestCase):
    def test_points_outside_are_refused(self):
        ix = B.BallIndex("Z8", 4, 4)
        with self.assertRaises(ValueError):
            ix.rank(np.array([2.0, 1.0, 0.0, 0.0]))          # squared length 5
        with self.assertRaises(ValueError):
            ix.rank(np.array([0.5, 0.5, 0.5, 0.5]))          # not whole
        e = B.BallIndex("E8", 8, 4)
        with self.assertRaises(ValueError):
            e.rank(np.array([1.0, 0, 0, 0, 0, 0, 0, 0]))     # odd sum: not in E8
        with self.assertRaises(ValueError):
            e.rank(np.array([0.5] * 7 + [-0.5]))             # half entries with an odd sum of whole parts
        with self.assertRaises(ValueError):
            e.unrank(e.count)


class RoundedBlocks(unittest.TestCase):
    def roundtrip(self, lattice, n, bound, blocks, seed):
        ix = B.BallIndex(lattice, n, bound)
        rng = np.random.default_rng(seed)
        seen = set()
        for _ in range(blocks):
            y = round_into(ix, rng.normal(size=n))
            k = ix.rank(y)
            self.assertTrue(0 <= k < ix.count)
            self.assertTrue(np.array_equal(ix.unrank(k), y))
            seen.add(k)
        self.assertGreater(len(seen), blocks // 2)
        return ix

    def test_integers_32(self):
        ix = self.roundtrip("Z8", 32, 553, 40, 1)
        self.assertLessEqual(ix.count, 16 ** 32)
        self.assertLessEqual(ix.bits(), 128)

    def test_e8_8(self):
        ix = self.roundtrip("E8", 8, 178, 60, 2)
        self.assertEqual(ix.count, 4158269521)       # the size the record of the region shapes gives
        self.assertLessEqual(ix.bits(), 32)

    def test_e8_16(self):
        ix = self.roundtrip("E8", 16, 304, 30, 3)
        self.assertLessEqual(ix.bits(), 64)


if __name__ == "__main__":
    unittest.main()
