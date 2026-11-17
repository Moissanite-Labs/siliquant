"""Tests of shape/ballformula.py. Run from the root of a checkout: python3 -m unittest shape/test_ballformula.py"""
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ballformula as B  # noqa: E402

QWEN = HERE.parent / "experiments" / "model-output-qwen25-2026-11-17" / "results.jsonl"


class TestClosedForm(unittest.TestCase):
    def test_volume(self):
        self.assertAlmostEqual(math.exp(B.log_ball_volume(2)), math.pi, places=12)
        self.assertAlmostEqual(math.exp(B.log_ball_volume(8)), math.pi ** 4 / 24.0, places=12)

    def test_region_part(self):
        # one ball per block of 8: the figure of the limits table, section 10
        self.assertAlmostEqual(B.phi_ball(8), 11.355, places=3)
        # it rises with the group and stays under 2 pi e
        values = [B.phi_ball(n) for n in (8, 32, 128, 512, 2048, 2 ** 16)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[-1], 2.0 * math.pi * math.e)
        self.assertAlmostEqual(values[-1] / (2.0 * math.pi * math.e), 1.0, places=3)

    def test_scale_factor(self):
        self.assertAlmostEqual(B.scale_factor(128, None), 1.0 + 2.0 / 128.0)
        # a grid of m steps per factor of two: the mean of 2^(2u/m) over u in (0, 1), by a sum
        m = 16
        mean = sum(2.0 ** (2.0 * (k + 0.5) / 1000.0 / m) for k in range(1000)) / 1000.0
        self.assertAlmostEqual(B.scale_factor(128, m), mean, places=6)
        self.assertGreater(B.scale_factor(128, 8), B.scale_factor(128, 16))

    def test_parse(self):
        self.assertEqual(B.parse("EBG128:b560s16"), ("E", 128, 560, None, 4.5))
        self.assertEqual(B.parse("EBG128:b504s8"), ("E", 128, 504, 16, 4.0))
        self.assertEqual(B.parse("ZB32@32:16"), ("Z", 32, 128.0, None, 4.5))
        for other in ("X32:16", "VE8Z32:16", "EB128@R:16", "EC32@32:16", "file:x.gguf", "unchanged"):
            self.assertIsNone(B.parse(other))

    def test_one_bit_is_six_decibels(self):
        a, b = B.ball_error("E", 128, 560), B.ball_error("E", 128, 560 + 128)
        self.assertAlmostEqual(B.db(a / b), 20.0 * math.log10(2.0), places=9)
        self.assertAlmostEqual(B.db(B.ball_error("Z", 128, 560) / a), B.db((1 / 12) / (929 / 12960)), places=9)

    @unittest.skipUnless(QWEN.exists(), "the rows of the Qwen run are not in this checkout")
    def test_against_the_qwen_run(self):
        """The closed form against the weight errors measured on 196 matrices of Qwen2.5-1.5B: within 0.03 dB on every arm it covers."""
        seen = 0
        for line in QWEN.read_text().splitlines():
            r = json.loads(line)
            want = B.predicted_error_db(r["variant"])
            if want is None:
                continue
            seen += 1
            self.assertLess(abs(r["weight_err_db_mean"] - want), 0.03, r["variant"])
            self.assertLess(abs(r["bits_per_weight"] - B.parse(r["variant"])[4]), 0.002, r["variant"])
        self.assertGreaterEqual(seen, 8)

    def test_anchor(self):
        self.assertAlmostEqual(B.predicted_kl(-25.0, 0.1, -22.0), 0.1 * 10 ** -0.3)


if __name__ == "__main__":
    unittest.main()


class TestDesign(unittest.TestCase):
    def test_table_size_formula_is_that_of_the_index(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("design", HERE / "design.py")
        design = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(design)
        import ballindex as BI
        for lattice, n, bound in (("Z8", 8, 179), ("Z8", 32, 553), ("E8", 8, 178), ("E8", 16, 304)):
            self.assertEqual(design.table_entries(lattice, n, bound), BI.BallIndex(lattice, n, bound).table_entries())
        # the codebooks of two scored arms: the bound and the count give the bits the run reports
        for lattice, n, bits, want in (("E8", 32, 136, 4.4994), ("Z8", 128, 560, 4.4998)):
            bound, points = design.ball_bound(lattice, n, bits)
            self.assertLessEqual(points, 1 << bits)
            scale = 8 if n == 32 else 16
            self.assertAlmostEqual((math.log2(points) + scale) / n, want, places=3)
