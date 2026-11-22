"""Tests of shape/certified.py and shape/check_certificate.py. Run from the root of a checkout: python3 -m unittest shape/test_certified.py"""
import copy
import importlib.util
import json
import math
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import certified as C  # noqa: E402
import check_certificate as K  # noqa: E402
import shapelib as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("region", HERE.parent / "experiments" / "region-shapes-no-clip-rule-2026-11-17" / "run.py")
region = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(region)


def candidates(W, caps):
    """The five exact arms at three group sizes for a matrix. caps maps a group size to the cap of its codebooks."""
    out = []
    for size, cap in caps.items():
        terms = S.exact_terms(W, size)
        q = S.iroot(cap, 8)
        out.append({"arm": "X", "size": size, "codewords": q ** 8, "terms": terms})
        for arm, lattice, p in (("Z8:p1", "Z8", 1), ("Z8:p2", "Z8", 2), ("E8:p1", "E8", 1), ("E8:p2", "E8", 2)):
            out.append({"arm": arm, "size": size, "codewords": S.Codebook(lattice, S.Shape(p), cap).N, "terms": terms})
    return out


class TestExactFormula(unittest.TestCase):
    def setUp(self):
        self.W = np.random.default_rng(21).standard_t(5, size=(16, 256)).astype(np.float32)
        self.caps = {32: 1 << 32, 128: 1 << 35, 256: S.iroot(1 << 71, 2)}        # 4.5 bits per weight with the scales: 2^35.5 for rows of 256

    def test_pi_bounds(self):
        self.assertTrue(K.pi_bounds_hold())
        self.assertLess(abs(float(C.PI_LOW) - math.pi), 1e-15)
        self.assertEqual(C.PI_HIGH - C.PI_LOW, F(1, 10 ** 20))
        self.assertEqual((K.PI_LOW_N, K.PI_HIGH_N, K.PI_D), (C.PI_LOW.numerator * (K.PI_D // C.PI_LOW.denominator), C.PI_HIGH.numerator * (K.PI_D // C.PI_HIGH.denominator), 10 ** 20))

    def test_exact_value_is_level_two_of_the_float_formula(self):
        W64 = self.W.astype(np.float64)
        for c in candidates(self.W, self.caps):
            r, b = C.fourth_power(c["arm"], c["size"], c["codewords"], c["terms"])
            exact = (float(r) * math.pi ** (4 * b)) ** 0.25
            if c["arm"] == "X":
                want = S.predict_scalar(W64, S.iroot(c["codewords"], 8), c["size"], 2)
            else:
                lattice, p = c["arm"].split(":p")
                want = S.predict(W64, S.Codebook(lattice, S.Shape(int(p)), self.caps[c["size"]]), c["size"], 2)
            self.assertAlmostEqual(exact / want, 1.0, places=11, msg=(c["arm"], c["size"]))

    def test_pick_is_the_float_pick_and_the_checker_agrees(self):
        cands = candidates(self.W, self.caps)
        best, cert = C.pick(cands, budget=F(9, 2))
        values = [(float(C.fourth_power(c["arm"], c["size"], c["codewords"], c["terms"])[0])
                   * math.pi ** (4 * C.fourth_power(c["arm"], c["size"], c["codewords"], c["terms"])[1])) for c in cands]
        self.assertEqual(best, int(np.argmin(values)))
        self.assertEqual(K.check(cert), [])

    def test_checker_catches_a_wrong_pick_and_a_changed_number(self):
        cands = candidates(self.W, self.caps)
        best, cert = C.pick(cands, budget=F(9, 2))
        for other in range(len(cands)):
            if other != best:
                self.assertTrue(K.check(dict(cert, pick=other)), other)
        import copy
        forged = copy.deepcopy(cert)
        forged["candidates"][best]["codewords"] = str(int(forged["candidates"][best]["codewords"]) // 1000)   # fewer codewords: over-states the error
        self.assertTrue(K.check(forged))
        forged = copy.deepcopy(cert)
        forged["candidates"][0]["terms"]["top_cube"] = str(int(forged["candidates"][0]["terms"]["top_ball"]) + 1)
        self.assertTrue(K.check(forged))

    def test_budget(self):
        self.assertTrue(C.within_budget(1 << 32, 32, F(9, 2)))
        self.assertFalse(C.within_budget((1 << 32) + 1, 32, F(9, 2)))
        self.assertTrue(C.within_budget(1 << 35, 128, F(9, 2)))
        self.assertFalse(C.within_budget((1 << 35) + 1, 128, F(9, 2)))
        for n in (512, 896, 2048, 4864):
            cap = S.iroot(1 << (8 * (F(9, 2) - F(16, n))).numerator, (8 * (F(9, 2) - F(16, n))).denominator)
            for N, want in ((cap, True), (cap + 1, False)):
                self.assertEqual(C.within_budget(N, n, F(9, 2)), want)
                self.assertEqual(K.in_budget({"codewords": str(N), "size": n}, [9, 2]), want)

    def test_comparison_with_pi(self):
        # a ball arm against a cube arm whose value sits just under and just over pi^4 times the ball's
        ball = (F(1), 1)
        self.assertEqual(C.compare(ball, (C.PI_HIGH ** 4 + F(1, 10 ** 30), 0)), -1)
        self.assertEqual(C.compare(ball, (C.PI_LOW ** 4 - F(1, 10 ** 30), 0)), 1)
        self.assertIsNone(C.compare(ball, ((C.PI_LOW ** 4 + C.PI_HIGH ** 4) / 2, 0)))
        self.assertEqual(C.compare((F(2), 1), (F(3), 1)), -1)
        self.assertEqual(C.compare((F(3), 0), (F(3), 0)), 0)


class TestBridgeReceipts(unittest.TestCase):
    def setUp(self):
        W = np.random.default_rng(21).standard_t(5, size=(16, 256)).astype(np.float32)
        self.best, self.cert = C.pick(candidates(W, {32: 1 << 32, 128: 1 << 35}), budget=F(9, 2))

    def test_boolean_pick_is_not_integer_zero(self):
        one = {"budget": None, "candidates": [self.cert["candidates"][self.best]], "pick": 0}
        self.assertEqual(K.check(one), [])
        self.assertEqual(K.check_bound(one, K.certificate_sha256(one)), [])
        for invalid in (False, 0.0, "0", None):
            forged = dict(one, pick=invalid)
            self.assertTrue(K.check(forged), invalid)

    def test_receipt_digest_binds_the_whole_claim(self):
        digest = K.certificate_sha256(self.cert)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        reordered = {"pick": self.cert["pick"], "candidates": self.cert["candidates"], "budget": self.cert["budget"]}
        self.assertEqual(K.certificate_sha256(reordered), digest)
        self.assertEqual(K.check_bound(reordered, digest), [])
        self.assertTrue(K.check_bound(self.cert, "0" * 64))
        forged = copy.deepcopy(self.cert)
        forged["candidates"][self.best]["codewords"] = str(int(forged["candidates"][self.best]["codewords"]) + 1)
        self.assertNotEqual(K.certificate_sha256(forged), digest)
        self.assertTrue(K.check_bound(forged, digest))

    def test_malformed_bridge_inputs_fail_closed(self):
        for path, value in ((("pick",), False), (("candidates", 0, "size"), True), (("candidates", 0, "codewords"), 1.5)):
            forged = copy.deepcopy(self.cert)
            target = forged
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self.assertTrue(K.check(forged), path)
            self.assertTrue(K.check_bound(forged, "0" * 64), path)
        self.assertTrue(K.check(dict(self.cert, unbound_evidence={})))
        self.assertTrue(K.check_bound(self.cert, "not-a-digest"))

    def test_real_g0_receipt_and_boolean_mutation(self):
        path = HERE.parent / "experiments" / "shape-formula-held-out-2026-11-17" / "results.certificates.json"
        cert = json.loads(path.read_text())["G/0"]
        self.assertEqual(K.check(cert), [])
        self.assertEqual(K.certificate_sha256(cert), "2afb503bedb75184ada25dbd3023d398655c48fb6257d4998d1a59cbd80be791")
        one = dict(cert, candidates=[cert["candidates"][cert["pick"]]], pick=0)
        self.assertEqual(K.check(one), [])
        self.assertTrue(K.check(dict(one, pick=False)))

    def test_negative_matrix_metadata_is_rejected(self):
        one = {"budget": None, "candidates": [copy.deepcopy(self.cert["candidates"][self.best])], "pick": 0}
        candidate = one["candidates"][0]
        count = int(candidate["terms"]["count"])
        candidate["size"] = -8
        candidate["terms"]["groups"] = str(-count // 8)
        self.assertTrue(K.check(one))


class TestForAnyWeights(unittest.TestCase):
    def test_identities(self):
        self.assertTrue(K.identities())

    def test_inequalities_on_extreme_blocks(self):
        # one large weight per block: the three norms are equal, and the ceilings of the region gains are reached
        one = np.zeros((4, 64), dtype=np.float32)
        one[:, ::8] = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
        t = S.exact_terms(one, 8)
        self.assertEqual(t["top_cube"], t["top_ball"])
        self.assertEqual(t["top_ball"], t["top_cross"])
        # eight equal weights per block: the other end
        flat = np.ones((4, 64), dtype=np.float32)
        t = S.exact_terms(flat, 8)
        self.assertEqual(t["top_ball"], 8 * t["top_cube"])
        self.assertEqual(t["top_cross"], 8 * t["top_ball"])
        for seed in range(20):
            W = np.random.default_rng(seed).standard_t(2, size=(8, 64)).astype(np.float32)
            for g in (8, 32, 64):
                t = S.exact_terms(W, g)
                self.assertTrue(t["top_cube"] <= t["top_ball"] <= t["top_cross"] <= 8 * t["top_ball"] and t["top_ball"] <= 8 * t["top_cube"])


class TestSafeScale(unittest.TestCase):
    def test_the_two_claims_on_heavy_tails(self):
        """No block outside the codebook, and no block over its bound, for weights with very heavy tails and for a matrix with zeros."""
        mats = [np.random.default_rng(31).standard_t(1.5, size=(32, 256)), np.random.default_rng(32).normal(size=(32, 256))]
        mats.append(mats[1] * (np.random.default_rng(33).uniform(size=(32, 256)) < 0.2))
        mats.append(np.zeros((8, 64)) + np.eye(8, 64))
        for W in mats:
            W = W.astype(np.float32).astype(np.float64)
            for lattice in ("Z8", "E8"):
                for p in (1, 2):
                    for cap in (1 << 16, 1 << 32):
                        code = S.Codebook(lattice, S.Shape(p), cap)
                        for g in (8, 32, "R"):
                            r = C.safe_round(W, code, g)
                            self.assertEqual(r["outside"], 0.0, (lattice, p, cap, g))
                            self.assertLessEqual(r["worst_block_over_its_bound"], 1.0 + 1e-9)
                            self.assertLessEqual(r["meas_before_storage"], r["bound"] * (1 + 1e-9))

    def test_bound_over_estimate_is_fixed(self):
        W = np.random.default_rng(34).laplace(size=(16, 256))
        for lattice, want in (("E8", 1.0 / (8 * 929 / 12960)), ("Z8", 2.0 / (8 / 12))):
            r = C.safe_round(W, S.Codebook(lattice, S.Shape(2), 1 << 32), 32)
            self.assertAlmostEqual(r["bound"] / r["estimate"], want, places=9)

    def test_margin(self):
        # the largest sum of sizes of a vector of length rho is sqrt(8) rho, reached by eight equal entries
        self.assertAlmostEqual(C.norm_margin("Z8", 1), 4.0)
        self.assertAlmostEqual(C.norm_margin("E8", 1), 2.0 * math.sqrt(2.0))
        self.assertAlmostEqual(C.norm_margin("E8", 2), 1.0)
        # the covering radius of the integers: the point with every entry 1/2 is sqrt 2 from the lattice
        x = np.full((1, 8), 0.5)
        self.assertAlmostEqual(float(np.sqrt(((x - np.rint(x)) ** 2).sum())), math.sqrt(2.0))
        # E8: no sampled point is further than 1 from its nearest lattice point, and a known deep hole is at 1
        X = np.random.default_rng(35).uniform(-4, 4, size=(200000, 8))
        self.assertLessEqual(float(np.sqrt(((X - S._E8.closest(X)) ** 2).sum(axis=1)).max()), 1.0)
        hole = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]])
        self.assertAlmostEqual(float(np.sqrt(((hole - S._E8.closest(hole)) ** 2).sum())), 1.0)


class TestCountsTwice(unittest.TestCase):
    def test_two_counters_agree_on_every_codebook_of_the_study(self):
        """shapelib counts by pairing two halves of 4 entries. The region record counts by adding one entry at a time.
        The two share no code. They must give the same codebook for every cap of the study."""
        caps = [1 << 16, 1 << 32, 1 << 35] + [S.iroot(1 << (36 * b - a), b) for a, b in ((1, 4), (1, 16), (1, 7), (1, 38))]
        for cap in caps:
            for lattice in ("Z8", "E8"):
                for p, shape in ((1, "cross"), (2, "ball")):
                    c = S.Codebook(lattice, S.Shape(p), cap)
                    radius = c.R >> S.S_BITS
                    table = region.counts(lattice, shape, radius + 4)
                    self.assertEqual(table[radius], c.N, (cap, lattice, p))
                    bigger = [r for r in range(radius + 1, radius + 4) if table[r] > c.N]
                    self.assertTrue(bigger and table[bigger[0]] > cap, (cap, lattice, p))


class TestReviewEdgeCases(unittest.TestCase):
    def setUp(self):
        W = np.random.default_rng(21).standard_t(5, size=(16, 256)).astype(np.float32)
        self.best, self.cert = C.pick(candidates(W, {32: 1 << 32, 128: 1 << 35}), budget=F(9, 2))

    def test_oversized_whole_numbers_return_findings(self):
        huge_string = copy.deepcopy(self.cert)
        huge_string["candidates"][0]["codewords"] = "9" * 5000
        self.assertTrue(K.check(huge_string))
        self.assertTrue(K.check_bound(huge_string, "0" * 64))

        huge_int = copy.deepcopy(self.cert)
        huge_int["candidates"][0]["terms"]["sum_sq"] = 10 ** 5000
        self.assertTrue(K.check_bound(huge_int, "0" * 64))

    def test_lone_surrogate_arm_is_a_finding(self):
        forged = copy.deepcopy(self.cert)
        forged["candidates"][0]["arm"] = "\ud800"
        self.assertTrue(K.check(forged))
        with self.assertRaises(ValueError):
            K.certificate_sha256(forged)

    def test_zero_cube_bound_is_rejected(self):
        forged = copy.deepcopy(self.cert)
        terms = forged["candidates"][self.best]["terms"]
        for key in ("top_cube", "top_ball", "top_cross"):
            terms[key] = "0"
        self.assertTrue(any("cube bound" in finding for finding in K.check(forged)))

    def test_mixed_integer_and_string_representations(self):
        receipt = copy.deepcopy(self.cert)
        candidate = receipt["candidates"][self.best]
        candidate["codewords"] = str(int(candidate["codewords"]))
        for key in ("count", "groups", "sum_sq", "top_cube", "top_ball", "top_cross"):
            candidate["terms"][key] = str(int(candidate["terms"][key]))
        self.assertEqual(K.check(receipt), [])

    def test_huge_budget_numerator_fails_closed(self):
        forged = copy.deepcopy(self.cert)
        forged["budget"] = [10 ** 9, 2]
        self.assertTrue(K.check(forged))


if __name__ == "__main__":
    unittest.main()
