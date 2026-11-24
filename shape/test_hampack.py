"""Tests of hampack.py: python3 -m unittest shape/test_hampack.py"""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ballpack  # noqa: E402
from hampack import HamFormat, checks  # noqa: E402
from hammingindex import HammingBallIndex  # noqa: E402


def in_lattice(y, L, bound):
    y = [int(v) for v in y]
    if sum(v * v for v in y) > bound:
        return False
    h = checks(L)
    for b in range(0, len(y), L):
        s = 0
        for j in range(L):
            if y[b + j] & 1:
                s ^= h[j]
        if s:
            return False
    return True


def key(y):
    return [2 * abs(int(v)) - (1 if v > 0 else 0) if v else 0 for v in y]      # 0, 1, -1, 2, -2, ... as 0, 1, 2, 3, 4, ...


class TestHamPack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.f16 = HamFormat(16, 32, 136)
        cls.f32 = HamFormat(32, 32, 136)
        cls.f8 = HamFormat(8, 32, 136)

    def test_counts_agree_with_hammingindex(self):
        for f in (self.f8, self.f16, self.f32):
            idx = HammingBallIndex(f.m, f.n, f.bound)
            self.assertEqual(f.count, idx.count)
            self.assertLessEqual(f.count, 1 << 136)
            self.assertGreater(HammingBallIndex(f.m, f.n, f.bound + 1).count if f.bound + 1 <= 4 * f.bound else 0, 0)

    def test_bound_is_largest(self):
        for f in (self.f16, self.f32):
            r = f.bound + 1
            while HammingBallIndex(f.m, f.n, r).count == f.count:
                r += 1
            self.assertGreater(HammingBallIndex(f.m, f.n, r).count, 1 << 136)

    def test_whole_order_small(self):
        # every number from 0 to N - 1, read back: points of the ball, in strictly rising order, numbered back to themselves
        for L, n, bits in ((8, 8, 8), (8, 16, 16), (16, 16, 16)):
            f = HamFormat(L, n, bits)
            self.assertEqual(f.count, HammingBallIndex(f.m, n, f.bound).count)
            prev = None
            for k in range(f.count):
                y = f.ref_unrank(k)
                self.assertTrue(in_lattice(y, L, f.bound))
                self.assertEqual(f.ref_rank(y), k)
                if prev is not None:
                    self.assertLess(key(prev), key(y))
                prev = y
            ks = np.arange(f.count, dtype=np.uint64)
            nums = np.stack([[(int(k) >> (8 * b)) & 255 for b in range(f.bytes)] for k in ks]).astype(np.uint8)
            Y = f.unrank(nums)
            for k in range(f.count):
                self.assertEqual(list(Y[k]), list(f.ref_unrank(k)))
            self.assertTrue(np.array_equal(f.rank(Y), nums))

    def test_c_against_python_full_size(self):
        rng = np.random.default_rng(0)
        for f in (self.f8, self.f16, self.f32):
            ks = [int.from_bytes(rng.bytes(24), "little") % f.count for _ in range(300)] + [0, 1, f.count - 1]
            nums = np.array([[(k >> (8 * b)) & 255 for b in range(f.bytes)] for k in ks], dtype=np.uint8)
            Y = f.unrank(nums)
            for i, k in enumerate(ks):
                self.assertEqual(list(Y[i]), list(f.ref_unrank(k)))
                self.assertTrue(in_lattice(Y[i], f.L, f.bound))
            self.assertTrue(np.array_equal(f.rank(Y), nums))

    def test_refuses(self):
        f = self.f16
        y = np.zeros((1, 32), dtype=np.int8)
        y[0, 0] = 1                                    # parity not a word of the code
        with self.assertRaises(ValueError):
            f.rank(y)
        too_big = f.count
        with self.assertRaises(ValueError):
            f.unrank(np.array([[(too_big >> (8 * b)) & 255 for b in range(f.bytes)]], dtype=np.uint8))

    def test_closest_matches_record(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("hrec", os.path.join(HERE, "..", "experiments", "hamming-lattices-real-weights-2026-11-24", "hamming.py"))
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
        X = np.random.default_rng(1).normal(scale=3.0, size=(500, 32))
        for f in (self.f8, self.f16, self.f32):
            self.assertTrue(np.array_equal(f.closest(X), H.closest(X, f.m)))

    def test_matrix_and_file_round_trip(self):
        rng = np.random.default_rng(2)
        W = rng.laplace(size=(64, 256)).astype(np.float32)
        z = ballpack.Format(32, 136)
        _, Wz, _ = z.encode(W)
        ez = 10 * math.log10(float(np.sum((W - Wz.astype(np.float64)) ** 2) / np.sum(W.astype(np.float64) ** 2)))
        for f, gain in ((self.f16, 0.6), (self.f32, 0.5), (self.f8, 0.5)):
            stored, Wh, shrunk = f.encode(W, keep=np.array([3, 70], dtype=np.int64))
            self.assertTrue(np.array_equal(f.decode(stored), Wh))
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "m.sqhp")
                size = f.write(p, [("w", stored)])
                g, back = HamFormat.read(p)
                self.assertEqual(size, os.path.getsize(p))
                self.assertEqual((g.L, g.n, g.bits, g.bound), (f.L, f.n, f.bits, f.bound))
                self.assertTrue(np.array_equal(g.decode(back[0][1]), Wh))
                self.assertEqual(size, 28 + len(f.tensor_bytes("w", stored)))
                blob = open(p, "rb").read()
                for cut in (20, size - 1):
                    with open(p, "wb") as h:
                        h.write(blob[:cut])
                    with self.assertRaises(ValueError):
                        HamFormat.read(p)
            self.assertTrue(np.array_equal(Wh[:, [3, 70]], W[:, [3, 70]].astype(np.float16)))
            _, Wn, _ = f.encode(W)
            e = 10 * math.log10(float(np.sum((W - Wn.astype(np.float64)) ** 2) / np.sum(W.astype(np.float64) ** 2)))
            self.assertLess(e, ez - gain)
            self.assertEqual(stored["records"].shape[1], 18)


if __name__ == "__main__":
    unittest.main()
