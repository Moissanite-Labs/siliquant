#!/usr/bin/env python3
"""Tests of the turned lattice and of the program, on made-up matrices. No model is read.

  python3 -m unittest experiments/e8-hadamard-real-weights-2026-11-22/test_turn.py

The first class needs NumPy and lattice/lattices.py. The second needs what run.py needs to be imported:
torch and llama.cpp's gguf package on GGUF_PY. Where those are missing it is skipped, and says so.
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lattice"))
import lattices as L  # noqa: E402
from turned import H, U, turn  # noqa: E402

E8 = L.E8()


def roots():
    """The 240 shortest points of E8 as it is usually written."""
    out = []
    for i, j in itertools.combinations(range(8), 2):
        for a, b in itertools.product((1.0, -1.0), repeat=2):
            v = np.zeros(8)
            v[i], v[j] = a, b
            out.append(v)
    for signs in itertools.product((0.5, -0.5), repeat=8):
        if sum(s < 0 for s in signs) % 2 == 0:
            out.append(np.array(signs))
    return np.array(out)


def whole(points):
    """Turned points times the root of 2, which the tests hold to whole numbers."""
    return turn(points) * math.sqrt(2.0)


class TestTurnedLattice(unittest.TestCase):
    def test_the_matrix_is_symmetric_and_its_own_inverse(self):
        self.assertTrue(np.array_equal(H, H.T))
        self.assertTrue(np.allclose(U @ U, np.eye(8), atol=1e-15))
        x = np.random.default_rng(1).standard_normal((50, 64))
        self.assertTrue(np.allclose(turn(turn(x)), x, atol=1e-12))

    def test_turning_keeps_the_length_of_every_block_of_8(self):
        x = np.random.default_rng(2).standard_normal((40, 128))
        a = (x.reshape(-1, 8) ** 2).sum(axis=1)
        b = (turn(x).reshape(-1, 8) ** 2).sum(axis=1)
        self.assertTrue(np.allclose(a, b, rtol=1e-12))

    def test_the_240_shortest_points_turned_are_16_on_an_axis_and_224_on_four(self):
        r = roots()
        self.assertEqual(len(r), 240)
        self.assertTrue(np.allclose(E8.closest(r), r))
        z = whole(r)
        self.assertTrue(np.allclose(z, np.rint(z), atol=1e-9))
        counts = np.bincount(np.count_nonzero(np.rint(z), axis=1), minlength=9)
        self.assertEqual((int(counts[1]), int(counts[4]), int(counts.sum())), (16, 224, 240))

    def test_turned_points_are_whole_numbers_whose_remainders_form_the_hamming_code(self):
        rng = np.random.default_rng(3)
        y = E8.closest(rng.standard_normal((20000, 8)) * 3.0)
        z = whole(y)
        self.assertTrue(np.allclose(z, np.rint(z), atol=1e-9))
        words = {tuple(int(v) for v in row) for row in (np.rint(z).astype(np.int64) % 2)}
        self.assertEqual(len(words), 16)
        self.assertEqual(sorted(sum(w) for w in words), [0] + [4] * 14 + [8])
        for a in words:
            for b in words:
                self.assertIn(tuple((p + q) % 2 for p, q in zip(a, b)), words)

    def test_the_nearest_turned_point_is_the_nearest_point_of_that_code_lattice(self):
        """An independent decoder: for each of the 16 words, the nearest point of the word plus twice the whole numbers."""
        rng = np.random.default_rng(4)
        y = E8.closest(rng.standard_normal((20000, 8)) * 3.0)
        words = np.array(sorted({tuple(int(v) for v in row) for row in (np.rint(whole(y)).astype(np.int64) % 2)}), dtype=np.float64)
        x = rng.standard_normal((3000, 8)) * 2.0
        mine = turn(E8.closest(turn(x)))
        z = x * math.sqrt(2.0)
        best, best_d = None, None
        for w in words:
            cand = (2.0 * np.rint((z - w) / 2.0) + w) / math.sqrt(2.0)
            d = ((x - cand) ** 2).sum(axis=1)
            if best is None:
                best, best_d = cand, d
            else:
                better = d < best_d
                best, best_d = np.where(better[:, None], cand, best), np.where(better, d, best_d)
        self.assertTrue(np.allclose(((x - mine) ** 2).sum(axis=1), best_d, atol=1e-9))


try:
    spec = importlib.util.spec_from_file_location("e8had_run", HERE / "run.py")
    RUN = importlib.util.module_from_spec(spec)
    sys.modules["e8had_run"] = RUN
    spec.loader.exec_module(RUN)
    WHY = ""
except Exception as e:                                            # torch or the gguf package is not here
    RUN, WHY = None, f"run.py cannot be imported here: {type(e).__name__}: {e}"


@unittest.skipIf(RUN is None, WHY)
class TestProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.W = (rng.standard_normal((6, 256)) * 0.02).astype(np.float32)
        cls.F = (rng.standard_normal((16, 256)) * 0.02).astype(np.float32)
        cls.F[:, 77] *= 60.0                                      # one loud column

    def test_the_plain_arms_are_the_base_program_bit_for_bit(self):
        Q = RUN.Q
        for arm, lattice in (("ZBG", "Z8"), ("EBG", "E8")):
            for bits in (560, 368):
                Wh, info = RUN.round_matrix(f"{arm}128:b{bits}s16", "attn_q", self.W)
                base, binfo = Q.group_ball_round(self.W, lattice, 128, bits, 16)
                self.assertTrue(np.array_equal(Wh, base))
                self.assertEqual((info["bits"], info["rel_err"], info["overload"]), (binfo["bits"], binfo["rel_err"], binfo["overload"]))
        self.assertEqual(Q.finish.__module__, Q.__name__)           # the base program's function is back in its place

    def test_the_turned_arm_has_the_bits_of_e8_and_an_error_of_its_size(self):
        for bits in (560, 368):
            _, e = RUN.round_matrix(f"EBG128:b{bits}s16", "attn_q", self.W)
            Wh, h = RUN.round_matrix(f"HBG128:b{bits}s16", "attn_q", self.W)
            self.assertEqual(h["bits"], e["bits"])
            self.assertLess(abs(10.0 * math.log10(h["rel_err"] / e["rel_err"])), 1.0)
            self.assertEqual(Wh.dtype, np.float16)
        self.assertLessEqual(RUN.round_matrix("HBG128:b560s16", "attn_q", self.W)[1]["bits"], 4.5)
        self.assertLessEqual(RUN.round_matrix("HBG128:b368s16", "attn_q", self.W)[1]["bits"], 3.0)

    def test_the_turned_arm_is_the_e8_arm_on_the_turned_matrix_turned_back(self):
        Y, bits, shrunk = RUN.ball64(turn(self.W.astype(np.float64)), "E8", 560)
        want, _ = RUN.Q.finish(self.W, turn(Y), bits, shrunk)
        self.assertTrue(np.array_equal(RUN.round_matrix("HBG128:b560s16", "attn_q", self.W)[0], want))

    def test_kept_columns_are_those_of_the_loud_columns_record(self):
        LC, F = RUN.LC, self.F
        cols = LC.loud(F, LC.K)
        self.assertIn(77, cols)
        rest = F.copy()
        rest[:, cols] = 0.0
        want, winfo = LC.ball("ZBG128:b560s16", "blk.0.ffn_down.weight", rest)
        want[:, cols] = F[:, cols].astype(np.float16)
        Wh, info = RUN.round_matrix("ZBG128:b560s16+K10", "ffn_down", F)
        self.assertTrue(np.array_equal(Wh, want))
        self.assertEqual(info["kept_columns"], 10)
        self.assertAlmostEqual(info["bits"], winfo["bits"] + (10 * 16 * 16.0 + 10 * 16.0) / F.size, places=12)
        for arm in ("EBG", "HBG"):
            Wh, info = RUN.round_matrix(f"{arm}128:b560s16+K10", "ffn_down", F)
            self.assertTrue(np.array_equal(Wh[:, cols], F[:, cols].astype(np.float16)))
            self.assertEqual(info["kept_columns"], 10)
        self.assertEqual(RUN.round_matrix("HBG128:b560s16+K10", "attn_q", F)[1]["kept_columns"], 0)

    def test_the_counts_of_small_weights_are_counts(self):
        Wh, info = RUN.round_matrix("HBG128:b368s16", "ffn_up", self.W)
        W64, Wh64 = self.W.astype(np.float64), Wh.astype(np.float64)
        rms = math.sqrt(float(np.mean(W64 ** 2)))
        self.assertEqual(info["zeros"], int(np.count_nonzero(Wh64 == 0)))
        for d in RUN.SMALL:
            mask = np.abs(W64) < d * rms
            n, z, e, w = info["small"][str(d)]
            self.assertEqual((n, z), (int(mask.sum()), int(np.count_nonzero(Wh64[mask] == 0))))
            self.assertAlmostEqual(e, float(((W64 - Wh64)[mask] ** 2).sum()), places=15)
            self.assertAlmostEqual(w, float((W64[mask] ** 2).sum()), places=15)
        pooled = RUN.small_stats({"blk.0.ffn_up.weight": info, "blk.0.attn_q.weight": info})
        self.assertAlmostEqual(pooled["stored_zero_percent"]["all"], 100.0 * info["zeros"] / info["size"], places=12)
        self.assertAlmostEqual(pooled["stored_zero_percent"]["mlp"], pooled["stored_zero_percent"]["attention"], places=12)

    def test_the_plain_voronoi_arm_is_the_base_program_bit_for_bit(self):
        import torch
        import tq
        code = tq.Code("E8", 16)
        kept = code.centre
        code.centre = torch.zeros_like(kept)                      # the lines of the base program for its arm VE8Z
        try:
            base, binfo = tq.maxscale_voronoi(self.W, 16, 32, name="E8")
        finally:
            code.centre = kept
        Wh, info = RUN.round_matrix("VE8Z32:16", "attn_q", self.W)
        self.assertTrue(np.array_equal(Wh, base))
        self.assertEqual((info["bits"], info["rel_err"], info["overload"]), (4.5, binfo["rel_err"], binfo["overload"]))
        self.assertTrue(torch.equal(tq.Code("E8", 16).centre, kept))    # the harness's centre is back in its place

    def test_the_turned_voronoi_arm_has_the_same_bits_and_keeps_zero_at_zero(self):
        Wh, info = RUN.round_matrix("VH8Z32:16", "attn_q", self.W)
        _, plain = RUN.round_matrix("VE8Z32:16", "attn_q", self.W)
        self.assertEqual(info["bits"], 4.5)
        self.assertLess(abs(10.0 * math.log10(info["rel_err"] / plain["rel_err"])), 1.5)
        zeros = np.zeros((2, 64), dtype=np.float32)
        zeros[0, 5] = 1.0                                         # one weight, and nothing else in its row
        for v in ("VE8Z32:16", "VH8Z32:16"):
            out = RUN.round_matrix(v, "attn_q", zeros)[0]
            self.assertTrue(np.all(out[1] == 0))                  # a group of zeros stays zero
            self.assertTrue(np.all(out[0, 32:] == 0))

    def test_names_outside_the_plan_are_refused(self):
        for v in ("HBG128:b560s8", "HBG64:b560s16", "hHBG128:b560s16", "HBG128:b560s16+K1", "ZBG128:b500s16", "VE832:16", "VH8Z128:16", "VH8Z32:16+K10"):
            self.assertIsNone(RUN.VARIANT.match(v))
            self.assertIsNone(RUN.VORONOI.match(v))


if __name__ == "__main__":
    unittest.main()
