"""Tests of shape/ggufread.py and of the row figures of shape/row_spread.py. Run from the root of a checkout: python3 -m unittest shape/test_ggufread.py"""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ggufread  # noqa: E402
import row_spread  # noqa: E402
import shapelib as S  # noqa: E402


def _string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, align=32):
    """A small GGUF file, version 3, with one string, one array and the alignment as metadata."""
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), 3)
    head += _string("general.architecture") + struct.pack("<I", 8) + _string("toy")
    head += _string("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", align)
    head += _string("toy.scores") + struct.pack("<I", 9) + struct.pack("<IQ", 6, 5) + struct.pack("<5f", 1, 2, 3, 4, 5)
    blobs, offset = [], 0
    for name, arr in tensors.items():
        kind = {np.dtype("float32"): 0, np.dtype("float16"): 1}[arr.dtype]
        rows, cols = arr.shape
        head += _string(name) + struct.pack("<I", 2) + struct.pack("<QQ", cols, rows) + struct.pack("<IQ", kind, offset)
        raw = arr.tobytes()
        raw += b"\0" * (-len(raw) % align)
        blobs.append(raw)
        offset += len(raw)
    head += b"\0" * (-len(head) % align)
    Path(path).write_bytes(head + b"".join(blobs))


class TestReader(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(3, 8)).astype(np.float16)
        b = rng.normal(size=(5, 16)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "toy.gguf"
            write_gguf(path, {"blk.0.attn_q.weight": a, "blk.0.ffn_up.weight": b})
            rd = ggufread.Reader(str(path))
            self.assertEqual(rd.names, ["blk.0.attn_q.weight", "blk.0.ffn_up.weight"])
            self.assertEqual(rd.meta["general.architecture"], "toy")
            self.assertTrue(np.array_equal(rd.matrix("blk.0.attn_q.weight"), a.astype(np.float32)))
            self.assertTrue(np.array_equal(rd.matrix("blk.0.ffn_up.weight"), b))
            self.assertEqual(rd.matrix("blk.0.ffn_up.weight").shape, (5, 16))

    def test_not_gguf(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.bin"
            path.write_bytes(b"NOPE" + b"\0" * 64)
            with self.assertRaises(ValueError):
                ggufread.Reader(str(path))


class TestRowFigures(unittest.TestCase):
    def test_scalar_error_is_that_of_shapelib(self):
        W = np.random.default_rng(2).standard_t(4, size=(8, 256)).astype(np.float32)
        for q, g in ((16, 32), (16, "R"), (32, 128)):
            err = row_spread.scalar_error(W, q, g)
            rel, _ = S.scalar_rule(W, q, g)
            self.assertAlmostEqual(float(err.sum() / (W.astype(np.float64) ** 2).sum()), rel, places=14)

    def test_even_and_uneven(self):
        w2 = np.ones((4, 8))
        even = row_spread.spread(0.01 * w2, w2)
        self.assertAlmostEqual(even[0], 0.01)
        self.assertAlmostEqual(even[1], 0.01)
        self.assertAlmostEqual(even[3], 1.0)             # every row has the same share
        err = 0.01 * w2.copy()
        err[0] *= 9.0                                     # one row carries nine times the rest
        total, rows, cols, uneven, worst, over = row_spread.spread(err, w2)
        self.assertAlmostEqual(total, 0.03)
        self.assertAlmostEqual(rows, 0.03)               # the mean over rows is still the matrix's share
        self.assertGreater(uneven, 1.5)
        self.assertAlmostEqual(worst, 3.0)
        self.assertEqual(over, 0.0)


if __name__ == "__main__":
    unittest.main()
