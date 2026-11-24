#!/usr/bin/env python3
"""Tests of the Hamming-code lattices and of the program, on made-up matrices. No model is read.

  python3 -m unittest experiments/hamming-lattices-real-weights-2026-11-24/test_hamming.py

The first class needs NumPy only. The second needs what run.py needs to be imported: torch and llama.cpp's
gguf package on GGUF_PY. Where those are missing it is skipped, and says so.
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
sys.path.insert(0, str(ROOT / "experiments" / "e8-hadamard-real-weights-2026-11-22"))
import hamming  # noqa: E402
import lattices as L  # noqa: E402
from turned import turn  # noqa: E402


def code_words(m):
    """Every word of the extended Hamming code of length 2^m, by checking all 2^n vectors' syndromes
    through the parity-check vectors (m = 3 or 4)."""
    n = 1 << m
    h = hamming.checks(m)
    words = []
    for v in range(1 << n):
        s = 0
        for i in range(n):
            if v >> i & 1:
                s ^= int(h[i])
        if s == 0:
            words.append([(v >> i) & 1 for i in range(n)])
    return np.array(words, dtype=np.float64)


def brute_closest(X, words):
    best = np.full(len(X), np.inf)
    out = np.zeros_like(X)
    for c in words:
        P = c + 2.0 * np.round((X - c) / 2.0)
        d = ((X - P) ** 2).sum(axis=1)
        better = d < best
        best[better], out[better] = d[better], P[better]
    return out, best


class TestCodesAndLattices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = {3: code_words(3), 4: code_words(4)}

    def test_the_codes_have_their_size_and_distance_4(self):
        for m, (k,) in ((3, (4,)), (4, (11,))):
            w = self.words[m]
            self.assertEqual(len(w), 1 << k)
            weights = w.sum(axis=1)
            self.assertEqual(int(weights[weights > 0].min()), 4)

    def test_the_weight_enumerator_counts_the_words(self):
        for m in (3, 4):
            counts = np.bincount(self.words[m].sum(axis=1).astype(int), minlength=(1 << m) + 1)
            self.assertEqual(list(counts), hamming.weight_enumerator(m))
        self.assertEqual(hamming.weight_enumerator(4)[4], 140)
        A32 = hamming.weight_enumerator(5)
        self.assertEqual(sum(A32), 1 << 26)
        self.assertEqual(A32[4], 1240)                                   # n(n-1)(n-2)/24 for an extended Hamming code

    def test_the_length_8_code_is_the_reed_muller_code_of_the_turned_lattice(self):
        # rows of the Sylvester-Hadamard matrix as bits, and their complements
        Hm = np.array([[1.0]])
        for _ in range(3):
            Hm = np.block([[Hm, Hm], [Hm, -Hm]])
        rm = {tuple(((1 - r) / 2).astype(int)) for r in Hm} | {tuple(((1 + r) / 2).astype(int)) for r in Hm}
        self.assertEqual(rm, {tuple(w.astype(int)) for w in self.words[3]})

    def test_shell_counts(self):
        s3 = hamming.shells(3, 12)
        self.assertEqual([s3[r] for r in (0, 4, 8, 12)], [1, 240, 2160, 6720])  # E8 over the root of 2
        self.assertEqual(sum(s3) - sum(s3[r] for r in (0, 4, 8, 12)), 0)
        s4 = hamming.shells(4, 6)
        self.assertEqual(s4[4], 32 + 140 * 16)                                  # 2e_i, and +-1 on a word of weight 4
        # every point of norm <= 6 counted by brute force over the words
        n, total = 16, [0] * 7
        for c in self.words[4]:
            w = int(c.sum())
            if w > 6:
                continue
            # coordinates off the word are even (0 or +-2), on it odd (+-1 here, since 9 > 6)
            for k2 in range(0, 2):                                           # number of +-2 off the word
                r = w + 4 * k2
                if r <= 6:
                    total[r] += 2 ** w * math.comb(n - w, k2) * 2 ** k2
        self.assertEqual(total, s4[:7])

    def test_the_decoder_finds_the_nearest_point(self):
        rng = np.random.default_rng(3)
        for m in (3, 4):
            X = rng.standard_normal((3000, 1 << m)) * 2.5
            mine = hamming.closest(X, m)
            ref, best = brute_closest(X, self.words[m])
            self.assertTrue(np.allclose(((X - mine) ** 2).sum(axis=1), best, atol=1e-9))
            self.assertTrue(np.array_equal(mine, ref))
            # the points are in the lattice: whole numbers whose remainders are words
            par = np.mod(mine, 2.0)
            self.assertTrue(np.all((par == 0) | (par == 1)))
            words = {tuple(w.astype(int)) for w in self.words[m]}
            self.assertTrue(all(tuple(p.astype(int)) in words for p in par))

    def test_the_decoder_of_length_32_is_within_the_covering_bound_and_in_the_lattice(self):
        rng = np.random.default_rng(4)
        X = rng.uniform(-6, 6, size=(4000, 32))
        Y = hamming.closest(X, 5)
        d = ((X - Y) ** 2).sum(axis=1)
        self.assertTrue(np.all(d <= hamming.covering_sq(5, 32) + 1e-9))
        h = hamming.checks(5)
        bits = np.mod(Y, 2.0).astype(int)
        syn = np.bitwise_xor.reduce(np.where(bits == 1, h[None, :], 0), axis=1)
        self.assertTrue(np.all(syn == 0))
        # no single move to another point of the lattice nearby is closer: try +-2 at each coordinate
        for i in range(32):
            for s in (2.0, -2.0):
                Z = Y.copy()
                Z[:, i] += s
                self.assertTrue(np.all(((X - Z) ** 2).sum(axis=1) >= d - 1e-9))

    def test_the_c_decoder_is_the_numpy_decoder_bit_for_bit(self):
        if hamming._library() is None:
            self.skipTest("no C compiler here")
        rng = np.random.default_rng(6)
        for m in (3, 4, 5):
            n = 1 << m
            X = np.concatenate([rng.standard_normal((3000, n)) * 3.0,
                                rng.laplace(0, 0.3, size=(3000, n)),
                                np.round(rng.standard_normal((500, n)) * 4) / 2,        # ties on half-integers
                                np.zeros((10, n))])
            self.assertTrue(np.array_equal(hamming.closest(X, m, use_c=True), hamming.closest(X, m, use_c=False)))

    def test_a8_over_the_root_of_2_is_turned_e8(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((5000, 8)) * 3.0
        a8 = hamming.closest(X * math.sqrt(2.0), 3) / math.sqrt(2.0)
        e8 = turn(L.E8().closest(turn(X)))
        self.assertTrue(np.allclose(a8, e8, atol=1e-12))

    def test_the_ball_bound_of_a8_is_that_of_e8_doubled(self):
        bd_path = ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17" / "run.py"
        spec = importlib.util.spec_from_file_location("balldim_t", bd_path)
        bd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bd)
        for bits in (368, 560):
            cap = 1 << bits
            # E8's bound by the ball module's counts, the limit grown until the count passes the cap
            limit = 300
            while True:
                total = bd.poly_pow(bd.shells8("E8", "ball", limit), 16, limit)
                run, best = 0, None
                for r, c in enumerate(total):
                    run += c
                    if run > cap:
                        break
                    if r % 2 == 0:
                        best = (r, run)
                if run > cap:
                    break
                limit = int(limit * 1.3) + 8
            b8 = hamming.ball_bound(3, 128, bits)
            self.assertEqual(b8[1], best[1])
            self.assertIn(b8[0], (2 * best[0], 2 * best[0] + 1, 2 * best[0] + 2, 2 * best[0] + 3))

    def test_the_new_balls_have_no_more_points_than_the_bits(self):
        for m in (4, 5):
            for bits in (368, 560):
                bound, points = hamming.ball_bound(m, 128, bits)
                self.assertLessEqual(points, 1 << bits)
                self.assertGreater(points, 1 << (bits - 2))
                self.assertGreater(sum(hamming.poly_pow(hamming.shells(m, bound + 4), 128 >> m, bound + 4)), 1 << bits)


try:
    spec = importlib.util.spec_from_file_location("hl_run", HERE / "run.py")
    RUN = importlib.util.module_from_spec(spec)
    sys.modules["hl_run"] = RUN
    spec.loader.exec_module(RUN)
    WHY = ""
except Exception as e:                                            # torch or the gguf package is not here
    RUN, WHY = None, f"run.py cannot be imported here: {type(e).__name__}: {e}"


@unittest.skipIf(RUN is None, WHY)
class TestProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.W = (rng.standard_normal((6, 256)) * 0.02).astype(np.float32)
        cls.L = (rng.laplace(0, 0.01, size=(8, 256))).astype(np.float32)

    def test_ball_round_is_the_base_program_on_integers_and_e8(self):
        Q, bd = RUN.Q, RUN.Q.ball_module()
        for lattice in ("Z8", "E8"):
            for bits in (560, 368):
                base, binfo = Q.group_ball_round(self.W, lattice, 128, bits, 16)
                closest, bound, points, margin = RUN.pieces(lattice, bits)
                Y, b, shrunk = RUN.ball_round(self.W.astype(np.float64), 128, bits, closest, bound, points, margin,
                                              bd.BISECT_STEPS, bd.SHRINK_STEPS)
                self.assertTrue(np.array_equal(Y.astype(np.float16), base))
                self.assertEqual((b, shrunk), (binfo["bits"], binfo["overload"]))

    def test_a8_is_the_turned_arm(self):
        for M in (self.W, self.L):
            for bits in (560, 368):
                a, ai = RUN.round_any(f"A8BG128:b{bits}s16", "attn_q", M)
                h, hi = RUN.round_any(f"HBG128:b{bits}s16", "attn_q", M)
                self.assertEqual(ai["bits"], hi["bits"])
                same = np.mean(a == h)
                self.assertGreaterEqual(same, 0.999)
                self.assertAlmostEqual(ai["rel_err"], hi["rel_err"], delta=1e-4 * hi["rel_err"])

    def test_the_new_arms_count_their_bits_and_keep_small_weights_at_zero(self):
        for bits, want in ((560, 4.5), (368, 3.0)):
            for n in (16, 32):
                Wh, info = RUN.round_any(f"A{n}BG128:b{bits}s16", "attn_q", self.L)
                self.assertLessEqual(info["bits"], want + 1e-12)
                self.assertGreater(info["bits"], want - 0.02)
                self.assertLess(info["rel_err"], 0.2)
        _, z = RUN.round_any("ZBG128:b560s16", "attn_q", self.L)
        _, a16 = RUN.round_any("A16BG128:b560s16", "attn_q", self.L)
        _, h = RUN.round_any("HBG128:b560s16", "attn_q", self.L)
        small = lambda i: i["small"]["0.05"][1] / max(i["small"]["0.05"][0], 1)
        self.assertGreater(small(a16), small(h))
        self.assertLess(a16["rel_err"], z["rel_err"])

    def test_kept_columns_are_left_whole(self):
        rng = np.random.default_rng(9)
        F = (rng.standard_normal((16, 256)) * 0.02).astype(np.float32)
        F[:, 77] *= 60.0
        Wh, info = RUN.round_any("A16BG128:b560s16+K10", "ffn_down", F)
        cols = RUN.LC.loud(F, RUN.LC.K)
        self.assertTrue(np.array_equal(Wh[:, cols], F[:, cols].astype(np.float16)))
        self.assertEqual(info["kept_columns"], len(cols))

    def test_nf4_rounds_to_the_nearest_of_its_16_values_over_the_block_maximum(self):
        spec = importlib.util.spec_from_file_location("hl_base", HERE / "baselines.py")
        B = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(B)
        self.assertEqual(len(B.NF4), 16)
        Y, bits = B.nf4(self.L)
        self.assertEqual(bits, 4.5)
        X = self.L.reshape(-1, 64).astype(np.float32)
        a = np.abs(X).max(axis=1)
        Z = X / a[:, None]
        want = B.NF4[np.abs(Z[..., None] - B.NF4[None, None, :]).argmin(axis=2)] * a[:, None]
        self.assertTrue(np.allclose(Y.reshape(-1, 64), want, atol=0, rtol=1e-6))
        # the largest weight of each block is kept exactly, up to its sign
        self.assertTrue(np.allclose(np.abs(Y.reshape(-1, 64)).max(axis=1), a, rtol=1e-7))
        self.assertEqual(set(B.Q4KM), set(RUN.EH.ACCEPTED))

    def test_names_outside_the_plan_are_refused(self):
        for v in ("A64BG128:b560s16", "A16BG128:b500s16", "A16BG64:b560s16", "A16BG128:b560s8"):
            self.assertFalse(RUN.known(v))
        for v in ("A8BG128:b560s16", "A16BG128:b368s16+K10", "A32BG128:b560s16", "HBG128:b560s16"):
            self.assertTrue(RUN.known(v))


if __name__ == "__main__":
    unittest.main()
