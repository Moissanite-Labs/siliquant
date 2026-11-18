"""Tests of shape/ballpack.py: the C coder against ballindex.py, exact round trips, and the bytes of a file.

Run: python3 -m unittest shape/test_ballpack.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ballpack  # noqa: E402


def number(row):
    return int.from_bytes(bytes(row), "little")


def random_points(fmt, rng, count):
    """Points of the codebook: blocks of Gaussian noise rounded at a scale that leaves them inside the ball."""
    out = []
    while len(out) < count:
        x = rng.standard_normal(fmt.n) * rng.uniform(0.05, 1.0) * np.sqrt(fmt.bound / fmt.n)
        y = np.rint(x)
        if (y ** 2).sum() <= fmt.bound:
            out.append(y)
    return np.array(out, dtype=np.int8)


class SmallFormat(unittest.TestCase):
    """8 weights and 16 bits: small enough to walk the whole codebook."""

    @classmethod
    def setUpClass(cls):
        cls.fmt = ballpack.Format(8, 16, 16)

    def test_every_number_names_one_point(self):
        fmt = self.fmt
        self.assertLessEqual(fmt.count, 1 << 16)
        numbers = np.zeros((fmt.count, fmt.bytes), dtype=np.uint8)
        for k in range(fmt.count):
            numbers[k] = list(k.to_bytes(fmt.bytes, "little"))
        Y = fmt.unrank(numbers)
        self.assertTrue(np.all((Y.astype(np.int64) ** 2).sum(axis=1) <= fmt.bound))
        self.assertEqual(len({bytes(y) for y in Y}), fmt.count)                # all different
        self.assertTrue(np.array_equal(fmt.rank(Y), numbers))                  # and rank is the inverse
        for k in range(0, fmt.count, 97):                                      # the same points as ballindex.py
            self.assertTrue(np.array_equal(fmt.index.unrank(k), Y[k].astype(np.float64)))

    def test_numbers_past_the_codebook_are_refused(self):
        fmt = self.fmt
        if fmt.count < 1 << 16:
            with self.assertRaises(ValueError):
                fmt.unrank(np.array([list(fmt.count.to_bytes(fmt.bytes, "little"))], dtype=np.uint8))

    def test_a_point_outside_the_ball_is_refused(self):
        y = np.zeros((1, 8), dtype=np.int8)
        y[0, 0] = 127
        with self.assertRaises(ValueError):
            self.fmt.rank(y)

    def test_the_bound_is_the_largest(self):
        fmt = self.fmt
        self.assertLessEqual(fmt.count, 1 << fmt.bits)
        import ballindex
        self.assertGreater(ballindex.BallIndex("Z8", fmt.n, fmt.bound + 1).count, 1 << fmt.bits)


class Format32(unittest.TestCase):
    """The format of the record: 32 weights, 136 bits."""

    @classmethod
    def setUpClass(cls):
        cls.fmt = ballpack.Format(32, 136, 16)
        cls.rng = np.random.default_rng(20261118)

    def test_c_coder_is_ballindex(self):
        fmt = self.fmt
        Y = random_points(fmt, self.rng, 300)
        numbers = fmt.rank(Y)
        for y, row in zip(Y, numbers):
            self.assertEqual(number(row), fmt.index.rank(y.astype(np.float64)))
        self.assertTrue(np.array_equal(fmt.unrank(numbers), Y))
        zero = fmt.rank(np.zeros((1, 32), dtype=np.int8))
        self.assertEqual(number(zero[0]), 0)                                   # zero is the first point
        last = np.array([list((fmt.count - 1).to_bytes(fmt.bytes, "little"))], dtype=np.uint8)
        self.assertTrue(np.array_equal(fmt.unrank(last)[0].astype(np.float64), fmt.index.unrank(fmt.count - 1)))
        with self.assertRaises(ValueError):
            fmt.unrank(np.array([list(fmt.count.to_bytes(fmt.bytes, "little"))], dtype=np.uint8))

    def test_bits(self):
        fmt = self.fmt
        self.assertEqual(fmt.bytes, 17)
        self.assertLessEqual(fmt.count, 1 << 136)
        self.assertGreater(fmt.count, 1 << 135)                                # the ball fills more than half of the numbers

    def matrices(self):
        rng = self.rng
        gauss = rng.standard_normal((24, 256)).astype(np.float32) * 0.02
        heavy = (rng.standard_t(3, size=(24, 256)) * 0.01).astype(np.float32)
        spike = gauss.copy()
        spike[3, 17] = 5.0                                                     # one entry far larger than the rest
        holes = gauss.copy()
        holes[5] = 0.0                                                         # a row of zeros
        holes[:, 64:96] = 0.0                                                  # and blocks of zeros
        return {"gauss": gauss, "heavy": heavy, "spike": spike, "holes": holes, "zero": np.zeros((4, 64), dtype=np.float32)}

    def test_round_trip_is_exact(self):
        fmt = self.fmt
        cols = np.exp(self.rng.standard_normal(256) * 0.5)
        for name, W in self.matrices().items():
            for c in (None, cols[: W.shape[1]]):
                stored, Wh, _ = fmt.encode(W, c)
                self.assertEqual(Wh.dtype, np.float16)
                self.assertTrue(np.array_equal(fmt.decode(stored), Wh), name)
                self.assertEqual(stored["records"].shape, (W.size // 32, 18))
                Y = fmt.unrank(stored["records"][:, 1:])
                self.assertTrue(np.all((Y.astype(np.int64) ** 2).sum(axis=1) <= fmt.bound), name)

    def test_error_is_that_of_the_ball(self):
        fmt = self.fmt
        W = self.matrices()["gauss"]
        _, Wh, shrunk = fmt.encode(W)
        rel = float(((W.astype(np.float64) - Wh.astype(np.float64)) ** 2).sum() / (W.astype(np.float64) ** 2).sum())
        self.assertLess(10 * np.log10(rel), -23.0)                             # 4.25 bits a weight in a ball over 32
        self.assertGreater(10 * np.log10(rel), -27.0)
        self.assertLess(shrunk, 0.05)
        self.assertTrue(np.array_equal(fmt.encode(np.zeros((2, 32), dtype=np.float32))[1], np.zeros((2, 32), dtype=np.float16)))

    def test_zero_weights_stay_zero_without_column_weights(self):
        fmt = self.fmt
        W = self.matrices()["holes"]
        _, Wh, _ = fmt.encode(W)
        self.assertTrue(np.all(Wh[5] == 0))
        self.assertTrue(np.all(Wh[:, 64:96] == 0))

    def test_file(self):
        fmt = self.fmt
        mats = self.matrices()
        cols = np.exp(self.rng.standard_normal(256) * 0.5)
        packed, want, size = [], {}, 28
        for i, (name, W) in enumerate(mats.items()):
            c = cols[: W.shape[1]] if i % 2 else None
            stored, Wh, _ = fmt.encode(W, c)
            packed.append((name, stored))
            want[name] = Wh
            size += 2 + len(name) + 9 + 2 + (2 * W.shape[1] if c is not None else 0) + 18 * (W.size // 32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.sqbp")
            self.assertEqual(fmt.write(path, packed), size)
            self.assertEqual(os.path.getsize(path), size)
            fmt2, tensors = ballpack.Format.read(path)
            self.assertEqual([n for n, _ in tensors], list(mats))
            for name, stored in tensors:
                self.assertTrue(np.array_equal(fmt2.decode(stored), want[name]), name)
            with open(path, "ab") as f:
                f.write(b"x")
            with self.assertRaises(ValueError):
                ballpack.Format.read(path)

    def test_a_row_that_blocks_do_not_tile_is_refused(self):
        with self.assertRaises(ValueError):
            self.fmt.encode(np.zeros((2, 40), dtype=np.float32))

    def test_timing_decodes_the_same_points(self):
        fmt = self.fmt
        W = self.matrices()["heavy"]
        stored, _, _ = fmt.encode(W)
        seconds, out = fmt.timing(stored["records"], top=float(stored["top"]), repeats=1)
        Y = fmt.unrank(stored["records"][:, 1:])
        t = np.float32(1.0) * (float(stored["top"]) * 2.0 ** (-stored["records"][:, 0].astype(np.float64) / 16)).astype(np.float32)
        self.assertTrue(np.array_equal(out, Y.astype(np.float32) * t[:, None]))
        self.assertGreaterEqual(seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
