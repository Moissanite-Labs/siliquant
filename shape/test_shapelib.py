"""Tests of shape/shapelib.py. Run from the root of a checkout: python3 -m unittest shape/test_shapelib.py"""
import math
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shapelib as S  # noqa: E402


def scan(lattice, shape, R, box):
    """The number of points of the lattice of size <= R by listing every vector of 8 entries in a box.

    Lists the halves of 4 entries directly and pairs them, without the tables of counts in shapelib.
    """
    t = shape.table(R)
    total = 0
    vals = np.arange(-box, box + 1, dtype=np.float64)
    for off in ((0.0,) if lattice == "Z8" else (0.0, 0.5)):
        g = np.stack(np.meshgrid(*([vals + off] * 4), indexing="ij"), -1).reshape(-1, 4)
        size = S.sizes(t, g)
        if lattice == "Z8":
            a = np.sort(size)
            total += int(np.searchsorted(a, R - a, side="right").sum())
        else:
            parity = np.rint(np.floor(g).sum(axis=1)).astype(np.int64) & 1
            for par in (0, 1):
                a = np.sort(size[parity == par])
                total += int(np.searchsorted(a, R - a, side="right").sum())
    return total


class TestRoots(unittest.TestCase):
    def test_iroot(self):
        for b in (1, 2, 3, 4, 7, 38):
            for x in list(range(0, 300)) + [10 ** 30 + 7, 2 ** 200 - 1, 2 ** 200, 2 ** 200 + 1]:
                r = S.iroot(x, b)
                self.assertTrue(r ** b <= x < (r + 1) ** b, (x, b, r))

    def test_tables_are_exact_for_whole_p(self):
        for p in (1, 2, 3):
            t = S.Shape(p).table(10 << S.S_BITS)
            for k in range(len(t) - 1):
                self.assertEqual(F(int(t[k]), 1 << S.S_BITS), F(k, 2) ** p)

    def test_tables_round_down(self):
        for p in (F(2, 3), F(5, 4), F(3, 2)):
            sh = S.Shape(p)
            for k in range(1, 60):
                v = sh.size_of_index(k)
                # v <= (k/2)^p 2^S < v + 1, tested without floats: raise both sides to the power b
                lhs = F(k, 2) ** sh.a * (1 << (S.S_BITS * sh.b))
                self.assertTrue(v ** sh.b <= lhs < (v + 1) ** sh.b)

    def test_volume(self):
        self.assertAlmostEqual(S.Shape(2).unit_volume(), math.pi ** 4 / 24.0, places=12)
        self.assertAlmostEqual(S.Shape(1).unit_volume(), 256.0 / math.factorial(8), places=15)
        self.assertAlmostEqual(S.Shape(2).unit_volume(3), 4.0 * math.pi / 3.0, places=12)


class TestCounts(unittest.TestCase):
    def test_against_a_scan(self):
        for p in (F(2, 3), F(1), F(5, 4), F(3, 2), F(2), F(3)):
            sh = S.Shape(p)
            for radius in (1.0, 2.0, 3.3):
                R = int(radius ** float(p) * (1 << S.S_BITS))
                counter = S.Counter(sh, R)
                for lattice in ("Z8", "E8"):
                    self.assertEqual(counter.count(lattice, R), scan(lattice, sh, R, int(radius) + 1), (p, radius, lattice))

    def test_e8_shells(self):
        counter = S.Counter(S.Shape(2), 8 << S.S_BITS)
        totals = [counter.count("E8", (2 * m) << S.S_BITS) for m in range(5)]
        self.assertEqual(totals, [1, 241, 2401, 9121, 26641])       # 1, 240, 2160, 6720 and 17520 points of squared length 0, 2, 4, 6, 8

    def test_codebooks_of_the_region_record(self):
        expected = {("Z8", 1, 16): (29, 3789356689), ("Z8", 2, 16): (179, 4206764513), ("E8", 2, 16): (178, 4158269521),
                    ("E8", 1, 16): (29, 3418475137), ("Z8", 2, 32): (720, 1094300611393), ("Z8", 1, 32): (59, 1007248645697),
                    ("E8", 2, 32): (720, 1097438938801), ("E8", 1, 32): (59, 965614200833)}
        for (lattice, p, q), (radius, count) in expected.items():
            c = S.Codebook(lattice, S.Shape(p), q ** 8)
            self.assertEqual((c.R, c.N), (radius << S.S_BITS, count))

    def test_cap_and_next_point(self):
        for lattice in ("Z8", "E8"):
            for p in (F(2, 3), F(3, 2)):
                cap = 1 << 16
                c = S.Codebook(lattice, S.Shape(p), cap)
                self.assertLessEqual(c.N, cap)
                counter = S.Counter(c.shape, 2 * c.R)
                self.assertEqual(counter.count(lattice, c.R), c.N)
                # the next size up would pass the cap
                lo, hi = c.R, 2 * c.R
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if counter.count(lattice, mid) > c.N:
                        hi = mid
                    else:
                        lo = mid
                self.assertGreater(counter.count(lattice, hi), cap)


class TestRounding(unittest.TestCase):
    def setUp(self):
        self.W32 = np.random.default_rng(11).standard_t(5, size=(16, 256)).astype(np.float32)
        self.W = self.W32.astype(np.float64)

    def test_rounded_blocks_are_codewords_and_nearest(self):
        X = self.W.reshape(-1, 8)
        for lattice in ("Z8", "E8"):
            for p in (F(2, 3), F(1), F(2), F(3)):
                code = S.Codebook(lattice, S.Shape(p), 1 << 16)
                tb = S.block_scales(X, code)
                Y = code.closest(X / tb[:, None])
                self.assertTrue(np.all(code.inside(Y)), (lattice, p))
                # at a scale a little smaller the block is outside for nearly every block: the scale is tight
                Y2 = code.closest(X / (tb * 0.97)[:, None])
                self.assertGreater(np.mean(~code.inside(Y2)), 0.5)
                rel, tg, shrunk = S.round_groups(self.W, code, 32, tb)
                self.assertTrue(0 < rel < 1)
                self.assertEqual(len(tg), self.W.size // 32)

    def test_members_of_e8(self):
        X = self.W.reshape(-1, 8) * 3.0
        Y = S._E8.closest(X)
        twice = np.rint(2 * Y)
        self.assertTrue(np.all(np.abs(2 * Y - twice) < 1e-12))
        halves = (twice % 2 != 0)
        self.assertTrue(np.all(halves.all(axis=1) | (~halves).all(axis=1)))       # all whole or all halves
        self.assertTrue(np.all(np.rint(Y.sum(axis=1)) % 2 == 0))                   # even sum

    def test_scalar_rule(self):
        rel, bits = S.scalar_rule(self.W32, 16, 32)
        self.assertAlmostEqual(bits, 4.5)
        # written out again in 64-bit floats: the largest entry on level -8, the rest rounded to -8..7
        Xg = self.W.reshape(-1, 32)
        j = np.abs(Xg).argmax(axis=1)[:, None]
        d = np.take_along_axis(Xg, j, axis=1) / -8.0
        ref = np.clip(np.rint(Xg / d), -8, 7) * d
        ref_rel = float(np.sum((Xg - ref) ** 2) / np.sum(Xg ** 2))
        self.assertAlmostEqual(rel, ref_rel, delta=1e-4 * ref_rel)


class TestFormula(unittest.TestCase):
    def test_formula_on_gaussian_rows(self):
        W = np.random.default_rng(3).normal(size=(64, 2048)).astype(np.float32).astype(np.float64)
        X = W.reshape(-1, 8)
        for lattice, p in (("Z8", F(2)), ("E8", F(2)), ("E8", F(1)), ("E8", F(3, 2))):
            code = S.Codebook(lattice, S.Shape(p), 1 << 32)
            rel, _, _ = S.round_groups(W, code, "R", S.block_scales(X, code))
            for level in (1, 2, 3):
                self.assertLess(abs(S.db(rel / S.predict(W, code, "R", level))), 0.2, (lattice, p, level))
        rel, _ = S.scalar_rule(W.astype(np.float32), 16, "R")
        self.assertLess(abs(S.db(rel / S.predict_scalar(W, 16, "R", 3))), 0.1)

    def test_level_one_is_the_product(self):
        W = np.random.default_rng(4).laplace(size=(8, 256))
        code = S.Codebook("E8", S.Shape(F(3, 2)), 1 << 16)
        for g in (32, "R"):
            product = code.G * S.phi(W, code.shape, g) * code.N ** (-2.0 / 8.0)
            self.assertAlmostEqual(S.predict(W, code, g, 1), product, delta=1e-12 * product)
        self.assertAlmostEqual(S.predict_scalar(W, 16, 32, 1), S.phi_cube(W, 32) / (12.0 * 256.0), places=14)

    def test_ball_is_best_for_a_round_cloud(self):
        W = np.random.default_rng(5).normal(size=(256, 2048))
        for g in (8, 32, "R"):
            ball = S.phi(W, S.Shape(2), g)
            for p in (F(2, 3), F(1), F(5, 4), F(3, 2), F(3)):
                self.assertGreater(S.phi(W, S.Shape(p), g), ball)
            self.assertGreater(S.phi_cube(W, g), ball)


class TestExact(unittest.TestCase):
    def test_integers(self):
        for dtype in (np.float16, np.float32, np.float64):
            W = np.random.default_rng(6).normal(size=(4, 16)).astype(dtype)
            W[0, 0] = 0.0
            W[1, 1] = np.finfo(dtype).tiny                      # a very small entry stretches the common power of two
            Z, e = S.to_integers(W)
            back = np.array([[F(int(z)) * F(2) ** e for z in row] for row in Z])
            want = np.array([[F(float(w)) for w in row] for row in W])
            self.assertTrue(np.all(back == want), dtype)

    def test_exact_terms_agree_with_floats(self):
        W = np.random.default_rng(7).standard_t(4, size=(32, 256)).astype(np.float32)
        for g in (8, 32, "R"):
            terms = S.exact_terms(W, g)
            exact = S.phi_from_exact(terms)
            for name, value in (("cube", S.phi_cube(W, g)), ("ball", S.phi(W, S.Shape(2), g)), ("cross", S.phi(W, S.Shape(1), g))):
                frac, const = exact[name]
                self.assertAlmostEqual(const * float(frac) / value, 1.0, places=12)

    def test_exact_terms_by_hand(self):
        W = np.array([[1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, -3.0]], dtype=np.float32)
        t = S.exact_terms(W, 8)
        unit = F(2) ** t["exponent"]
        self.assertEqual(t["sum_sq"] * unit ** 2, F(21, 4) + F(7, 16) + 9)
        self.assertEqual(t["top_cube"] * unit ** 2, 4 + 9)
        self.assertEqual(t["top_ball"] * unit ** 2, F(21, 4) + F(7, 16) + 9)
        self.assertEqual(t["top_cross"] * unit ** 2, F(7, 2) ** 2 + F(19, 4) ** 2)
        self.assertEqual(S.exact_terms(W, 8), S.exact_terms(W.astype(np.float64), 8))

    def test_digest_names_the_bytes(self):
        W = np.arange(16, dtype=np.float32).reshape(2, 8)
        self.assertEqual(S.matrix_digest(W), S.matrix_digest(W.copy()))
        self.assertNotEqual(S.matrix_digest(W), S.matrix_digest(W.astype(np.float64)))
        V = W.copy()
        V[1, 7] += 1
        self.assertNotEqual(S.matrix_digest(W), S.matrix_digest(V))


if __name__ == "__main__":
    unittest.main()
