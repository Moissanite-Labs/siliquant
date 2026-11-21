"""Tests of shape/ballpack.py: the C coder against ballindex.py, exact round trips, kept columns, and the bytes of a file.

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

    def test_kept_columns_come_back_as_they_were(self):
        fmt = self.fmt
        keep = np.array([0, 17, 100, 255])
        cols = np.exp(self.rng.standard_normal(256) * 0.5)
        for name in ("gauss", "heavy", "spike"):
            W = self.matrices()[name]
            for c in (None, cols):
                stored, Wh, _ = fmt.encode(W, c, keep=keep)
                self.assertTrue(np.array_equal(Wh[:, keep], W[:, keep].astype(np.float16)), name)      # as stored: 16-bit floats of the input
                self.assertTrue(np.array_equal(fmt.decode(stored), Wh), name)
                Y = fmt.unrank(stored["records"][:, 1:]).reshape(W.shape)
                self.assertTrue(np.all(Y[:, keep] == 0), name)                                         # a kept column takes no room in a block
                self.assertEqual(stored["records"].shape, (W.size // 32, 18))
        W = self.matrices()["spike"]                                                                   # the large entry is in column 17
        plain, kept = fmt.encode(W)[1].astype(np.float64), fmt.encode(W, keep=np.array([17]))[1].astype(np.float64)
        err = lambda A: float(((W.astype(np.float64) - A)[3] ** 2).sum())
        self.assertLess(err(kept), 0.1 * err(plain))                                                   # with it kept, its row is rounded at the scale of the rest

    def test_bytes_and_file_with_kept_columns(self):
        fmt = self.fmt
        W = self.matrices()["heavy"]
        keep = np.array([3, 64, 65])
        cols = np.exp(self.rng.standard_normal(256) * 0.5)
        packed, want, size = [], {}, 28
        for name, c, k in (("plain", None, None), ("kept", None, keep), ("both", cols, keep)):
            stored, Wh, _ = fmt.encode(W, c, keep=k)
            packed.append((name, stored))
            want[name] = Wh
            size += 2 + len(name) + 9 + 2 + (2 * 256 if c is not None else 0) + (4 + 4 * 3 + 2 * 24 * 3 if k is not None else 0) + 18 * (W.size // 32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "k.sqbp")
            self.assertEqual(fmt.write(path, packed), size)
            fmt2, tensors = ballpack.Format.read(path)
            for name, stored in tensors:
                self.assertTrue(np.array_equal(fmt2.decode(stored), want[name]), name)
            self.assertIsNone(tensors[0][1]["keep"])
            self.assertTrue(np.array_equal(tensors[1][1]["keep"], keep))

    def test_kept_columns_must_be_columns_in_rising_order(self):
        W = self.matrices()["gauss"]
        for bad in (np.array([5, 5]), np.array([9, 3]), np.array([256]), np.array([-1]), np.array([], dtype=np.int64), np.array([1.5])):
            with self.assertRaises(ValueError):
                self.fmt.encode(W, keep=bad)

    def test_bytes_without_kept_columns_are_as_before(self):
        import hashlib
        fmt = self.fmt
        rng = np.random.default_rng(7)
        W = (rng.standard_normal((16, 256)) * 0.02).astype(np.float32)
        c = np.exp(rng.standard_normal(256) * 0.5)
        sha = lambda stored: hashlib.sha256(fmt.tensor_bytes("m", stored)).hexdigest()
        self.assertEqual(sha(fmt.encode(W)[0]), "998404863c3f766a7941e3034c2b2d9652593ec0e8b690e209914d48b1903a91")     # the bytes before kept columns existed
        self.assertEqual(sha(fmt.encode(W, c)[0]), "73883d41b04521a0222a22a81c4ddbf63a8f42ff9d3445ee389699aa06808ee9")

    def test_two_formats_alive_do_not_read_each_others_table(self):
        fmt = self.fmt
        W = self.matrices()["heavy"]
        stored, Wh, _ = fmt.encode(W)
        small = ballpack.Format(8, 16, 16)                                      # the C coder's one table is now this format's
        Y8 = random_points(small, self.rng, 50)
        n8 = small.rank(Y8)
        self.assertTrue(np.array_equal(fmt.decode(stored), Wh))                 # and now the first format's again
        self.assertTrue(np.array_equal(small.unrank(n8), Y8))
        Y = random_points(fmt, self.rng, 20)
        self.assertEqual([number(r) for r in fmt.rank(Y)], [fmt.index.rank(y.tolist()) for y in Y])
        seconds, out = fmt.timing(stored["records"], top=float(stored["top"]), repeats=1)
        self.assertEqual(out.shape, (W.size // 32, 32))
        self.assertTrue(np.array_equal(small.unrank(n8), Y8))

    def test_a_bad_file_is_refused_with_one_kind_of_error(self):
        import struct
        fmt = self.fmt
        W = self.matrices()["gauss"]
        cols = np.exp(self.rng.standard_normal(256) * 0.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "b.sqbp")
            fmt.write(path, [("a", fmt.encode(W, cols, keep=np.array([3, 9]))[0]), ("b", fmt.encode(W)[0])])
            with open(path, "rb") as f:
                good = f.read()
            ballpack.Format.read(path)

            def refused(blob):
                with open(path, "wb") as f:
                    f.write(blob)
                with self.assertRaises(ValueError):
                    ballpack.Format.read(path)

            for cut in (0, 7, 20, 28, 30, 33, 41, 300, 560, 700, len(good) // 2, len(good) - 1):     # cut short at every kind of place
                refused(good[:cut])
            head = lambda n, bits, bound, steps: ballpack.MAGIC + struct.pack("<5I", n, bits, bound, steps, 2) + good[28:]
            refused(head(32, 136, fmt.bound, 0))                                # a scale of no steps
            refused(head(0, 136, fmt.bound, 16))                                # a block of no weights
            refused(head(32, 136, fmt.bound + 1, 16))                           # a bound that is not the format's
            at = 28 + 2 + 1                                                     # rows and cols of the first matrix
            rows, cols_ = struct.unpack_from("<II", good, at)
            refused(good[:at] + struct.pack("<II", rows, cols_ + 1) + good[at + 8:])   # a row the blocks do not tile
            refused(good[:at + 8] + bytes([4]) + good[at + 9:])                 # a flag this reader does not know

    def test_the_compiled_coder_is_loaded_from_a_directory_of_this_user_alone(self):
        d = ballpack._build_dir()
        st = os.stat(d)
        self.assertEqual(st.st_uid, os.getuid())
        self.assertEqual(st.st_mode & 0o022, 0)
        with tempfile.TemporaryDirectory() as t:
            os.chmod(t, 0o777)
            self.assertFalse(ballpack._private(t))                              # open to others: not a place to load a library from
            os.chmod(t, 0o700)
            self.assertTrue(ballpack._private(t))
        self.assertFalse(ballpack._private(os.path.join(str(d), "no-such-file")))

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
