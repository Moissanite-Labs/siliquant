#!/usr/bin/env python3
"""Checks of the programs of this record: allocate.py, measures.py and the variant names of run.py.

Needs NumPy and torch. Usage, from the root of a checkout:
  python3 experiments/output-budget-gemma-e4b-2026-11-17/test_programs.py
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
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "persist"))


def load(name):
    spec = importlib.util.spec_from_file_location("ob_" + name, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


allocate = load("allocate")
measures = load("measures")
ob_run = measures.ob_run


class AllocateTest(unittest.TestCase):
    def test_three_kinds_by_hand(self):
        # Options 8, 16, 32 levels (3.5, 4.5, 5.5 bits). KL of a kind = c * 4^-(bits - 3.5).
        # Equal weights, c = 1, 1, 16, budget 4.5 bits: the total of the three kinds' bits is 13.5.
        # (16,16,16) gives 0.25 + 0.25 + 4 = 4.5; (8,16,32) and (16,8,32) give 1 + 0.25 + 1 = 2.25,
        # (8,8,32) gives 3. The best is 2.25 and the first of the two ties in index order is (8,16,32).
        c = np.array([1.0, 1.0, 16.0])
        bits = np.tile(np.array([3.5, 4.5, 5.5]), (3, 1))
        kl = c[:, None] * 4.0 ** -(bits - 3.5)
        idx, s, mean = allocate.best_choice(kl, np.array([1.0, 1.0, 1.0]), bits, 4.5)
        self.assertEqual(tuple(idx), (0, 1, 2))
        self.assertAlmostEqual(s, 2.25)
        self.assertAlmostEqual(mean, 4.5)
        # The reversed arm uses the same levels in another order: the worst is the heaviest kind at 8 levels.
        ridx, rs, rmean, window = allocate.reversed_choice(kl, np.ones(3), bits, None, idx, 4.5)
        self.assertEqual(tuple(ridx), (1, 2, 0))              # c = 1 gets 16, c = 1 gets 32, c = 16 gets 8
        self.assertAlmostEqual(rs, 0.25 + 0.0625 + 16.0)
        self.assertAlmostEqual(window, 0.05)

    def test_weights_matter(self):
        # Two kinds, the first has three times the weights. Budget 4.5. Equal KL curves.
        # (16,16): 2 * 0.25 = 0.5 at mean 4.5. (32,8) costs (3*5.5 + 3.5)/4 = 5.0 > 4.5. (8,32): (3*3.5+5.5)/4 = 4.0, KL 1 + 1/16.
        # (16, 32): (13.5 + 5.5)/4 = 4.75 > 4.5. (32, 16) is (16.5 + 4.5)/4 = 5.25. (16,8): (13.5+3.5)/4 = 4.25, KL 0.25 + 1.
        bits = np.tile(np.array([3.5, 4.5, 5.5]), (2, 1))
        kl = 4.0 ** -(bits - 3.5)
        idx, s, mean = allocate.best_choice(kl, np.array([3.0, 1.0]), bits, 4.5)
        self.assertEqual(tuple(idx), (1, 1))
        self.assertAlmostEqual(s, 0.5)

    def test_seven_kinds_full_option_set(self):
        # One kind is enormously sensitive and the others do not move with the levels. The best gives
        # the sensitive one the most levels (64, by the end slope) and the rest the fewest (4, fewer bits breaks the tie).
        kl_pts = {k: [1e-4, 1e-4, 1e-4] for k in allocate.KINDS}
        kl_pts["ffn_down"] = [100.0, 25.0, 6.25]
        kl = np.array([allocate.predicted_kl(kl_pts[k], allocate.LEVELS) for k in allocate.KINDS])
        bits = np.array([[math.log2(q) + 0.5 for q in allocate.LEVELS]] * 7)
        idx, s, mean = allocate.best_choice(kl, np.ones(7), bits, 4.5)
        levels = {k: allocate.LEVELS[i] for k, i in zip(allocate.KINDS, idx)}
        self.assertEqual(levels["ffn_down"], 64)
        self.assertTrue(all(levels[k] == 4 for k in allocate.KINDS if k != "ffn_down"))
        self.assertAlmostEqual(s, 6 * 1e-4 + 6.25 * 0.25 ** math.log2(2), places=6)
        self.assertEqual(len(allocate.LEVELS) ** 7, 4782969)

    def test_interpolation(self):
        # Straight in log KL between the points, the end slopes beyond them.
        pts = [8.0, 4.0, 1.0]                                  # at 8, 16, 32 levels
        v = allocate.predicted_kl(pts, (8, 16, 32, 64, 4, 12))
        self.assertTrue(np.allclose(v[:3], [8.0, 4.0, 1.0]))
        self.assertAlmostEqual(v[3], 0.25)                      # the last slope, again
        self.assertAlmostEqual(v[4], 16.0)                      # the first slope, extended down
        self.assertAlmostEqual(v[5], 8.0 * 0.5 ** math.log2(12 / 8.0))


class MeasuresTest(unittest.TestCase):
    def test_along_equals_direct_loop(self):
        rng = np.random.default_rng(3)
        W = rng.normal(size=(7, 11))
        W[2] = 0.0                                              # a row of zeros counts nothing
        Wh = W + rng.normal(scale=0.1, size=W.shape)
        E = Wh - W
        rows = sum(float(E[i] @ W[i]) ** 2 / float(W[i] @ W[i]) for i in range(7) if W[i].any()) / float(np.sum(W ** 2))
        cols = sum(float(E[:, j] @ W[:, j]) ** 2 / float(W[:, j] @ W[:, j]) for j in range(11)) / float(np.sum(W ** 2))
        self.assertAlmostEqual(measures.along(W, Wh, 1), rows, places=12)
        self.assertAlmostEqual(measures.along(W, Wh, 0), cols, places=12)
        self.assertNotEqual(rows, cols)

    def test_along_cases(self):
        W = np.arange(1.0, 13.0).reshape(3, 4)
        self.assertAlmostEqual(measures.along(W, 2.0 * W, 1), 1.0)           # a change along every row: all of it
        self.assertAlmostEqual(measures.along(W, W, 1), 0.0)
        Wh = W.copy()
        Wh[0] += np.array([-W[0, 1], W[0, 0], 0.0, 0.0])                       # perpendicular to row 0 in its first two entries ...
        Wh[0, 2:] = W[0, 2:]
        self.assertAlmostEqual(measures.along(W, Wh, 1), (-W[0, 1] * W[0, 0] + W[0, 0] * W[0, 1]) ** 2)

    def test_tree_shift(self):
        import persistlib
        rng = np.random.default_rng(5)
        W = rng.normal(size=(6, 9))
        Wh = W + rng.normal(scale=0.2, size=W.shape)
        tw = persistlib.tree_weights(W, wmax=1.0)
        d = float(np.linalg.norm(tw - persistlib.tree_weights(Wh, wmax=1.0)))
        self.assertAlmostEqual(measures.tree_shift(W, Wh), d * d / float(np.sum(tw * tw)), places=12)
        self.assertAlmostEqual(measures.tree_shift(W, W), 0.0)

    def test_spread(self):
        W = np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [0.0, 0.0]])
        s, zero = measures.spread(W, 1)
        self.assertEqual(zero, 1)
        self.assertAlmostEqual(s, float(np.std(np.log([1.0, 2.0, 4.0]))))


class VariantTest(unittest.TestCase):
    def test_parse(self):
        p = ob_run.parse("X32:16@ffn_down#0,8,40")
        self.assertEqual((p["group"], p["mod"], p["kind"], p["layers"]), ("32", None, "ffn_down", (0, 8, 40)))
        self.assertEqual(ob_run.parse("XRo:16~3.45")["c"], 3.45)
        self.assertEqual(ob_run.parse("X32s6:20")["mod"], "s6")
        self.assertEqual(ob_run.parse("clip:5@ffn_down#8")["layers"], (8,))
        mix = "X32mix:" + ",".join(f"{k}={q}" for k, q in zip(ob_run.KINDS, (4, 6, 8, 12, 16, 22, 32)))
        self.assertEqual(ob_run.parse(mix)["levels"]["ffn_up"], 22)
        self.assertEqual(ob_run.parse(mix.replace("X32mix", "X32s8mix"))["mod"], "s8")
        for bad in ("XRo:16", "X32s8:16~0.5", "X32:16@nothing", "X32mix:attn_q=8", "Q:16"):
            with self.assertRaises(SystemExit):
                ob_run.parse(bad)

    def test_selected(self):
        p = ob_run.parse("X32:16@ffn_down#0,8")
        self.assertTrue(ob_run.selected(p, "blk.8.ffn_down.weight"))
        self.assertFalse(ob_run.selected(p, "blk.9.ffn_down.weight"))
        self.assertFalse(ob_run.selected(p, "blk.8.ffn_up.weight"))
        self.assertFalse(ob_run.selected(p, "blk.8.attn_norm.weight"))

    def test_parts(self):
        self.assertEqual(len(ob_run.PART_A), 22)
        self.assertEqual(len(ob_run.PART_C), 8)
        self.assertEqual(len(ob_run.PART_D), 4)
        self.assertEqual(len(ob_run.PART_E), 14)
        for v in ob_run.PART_A + ob_run.PART_C + ob_run.PART_D + ob_run.PART_E + ob_run.FIRST[1:]:
            ob_run.parse(v)

    def test_mixed_arm_uses_each_kinds_levels(self):
        rng = np.random.default_rng(8)
        W = rng.normal(size=(4, 64)).astype(np.float32)
        levels = dict(zip(ob_run.KINDS, (4, 6, 8, 12, 16, 22, 32)))
        mix = "X32mix:" + ",".join(f"{k}={q}" for k, q in levels.items())
        spec = ob_run.parse(mix)
        import tq
        for kind, q in levels.items():
            Wh, info = ob_run.round_one(spec, W, kind)
            self.assertTrue(np.array_equal(Wh, tq.maxscale_round(W, q, 32)[0]))
            self.assertAlmostEqual(info["bits"], math.log2(q) + 0.5)
        spec8 = ob_run.parse(mix.replace("X32mix", "X32s8mix"))
        Wh, info = ob_run.round_one(spec8, W, "ffn_down")
        self.assertTrue(np.array_equal(Wh, tq.maxscale_round_sbit(W, 32, 32, 8)[0]))
        self.assertAlmostEqual(info["bits"], math.log2(32) + 0.25 + 16.0 / 64)


if __name__ == "__main__":
    unittest.main()
