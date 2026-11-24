#!/usr/bin/env python3
"""Assertion-free R5 verification, mutation-killing & Zone-B adversarial test suite
for Track S47 (Portal 52 / Moissanite-Labs/siliquant#77: Extended Hamming Construction-A
Lattices A_8, A_16, A_32, Syndrome Decoding Optimality, Voronoi Zero-Retention &
INT8/INT16/INT32/f64 Accumulator Bounds).

100% pure Python 3 stdlib unittest, zero bare asserts, zero AST floats, zero float(),
zero .isdigit(), zero islink/is_symlink, zero mathematical RNG.
"""
from __future__ import annotations

import ast
import copy
import ctypes
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import hamming_construction_a_audit as audit  # noqa: E402
import hamming_construction_a_oracle as oracle  # noqa: E402


def _resign_payload(payload: dict[str, object]) -> dict[str, object]:
    p_sha = audit._sha256_canonical(payload)
    outer = {
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": p_sha,
        "payload": payload,
    }
    c_sha = audit._sha256_canonical(outer)
    return {
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
        "payload": payload,
    }


def _call_main(argv: list[str]) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with mock.patch.object(sys, "stdout", out_buf), mock.patch.object(sys, "stderr", err_buf):
        rc = audit.main(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


class TestHammingConstructionAR5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = audit.build_standalone_certificate()
        cls.payload = cls.cert["payload"]

    def _expect_err(self, code: str, fn, *args, **kwargs) -> oracle.HammingLatticeError:
        with self.assertRaises(oracle.HammingLatticeError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)
        return ctx.exception

    # ------------------------------------------------------------------
    # 1. Static AST & Hygiene
    # ------------------------------------------------------------------
    def test_01_static_ast_hygiene(self) -> None:
        files = (
            _THIS_DIR / "hamming_construction_a_oracle.py",
            _THIS_DIR / "hamming_construction_a_audit.py",
            _THIS_DIR / "test_hamming_construction_a_r5.py",
        )
        for fp in files:
            src = fp.read_text(encoding="utf-8")
            self.assertNotIn("\r", src)
            tree = ast.parse(src, filename=str(fp))
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert, f"bare assert in {fp.name}")
                if isinstance(node, ast.Constant):
                    self.assertIsNot(type(node.value), float, f"float literal in {fp.name}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, "float", f"float() call in {fp.name}")
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ("isdigit", "islink", "is_symlink"), f"forbidden attr in {fp.name}")

    # ------------------------------------------------------------------
    # 2. Strict Primitives, Bit Caps & Exact Formatters
    # ------------------------------------------------------------------
    def test_02_strict_primitives_and_formatters(self) -> None:
        oracle.require_ok(True, "OK", "ok")
        self._expect_err("BOOM", oracle.require_ok, False, "BOOM", "fail")
        self._expect_err("INVALID_CONDITION_TYPE", oracle.require_ok, 1, "X", "bad")
        self.assertIs(oracle.require_exact_bool(True), True)
        self.assertIs(oracle.require_exact_bool(False), False)
        self._expect_err("INVALID_BOOL_TYPE", oracle.require_exact_bool, 1)

        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        self.assertEqual(oracle.require_exact_int(max_ok, "m", lo=0, hi=max_ok), max_ok)
        self.assertEqual(oracle.require_exact_int(0, "z", lo=0, hi=0), 0)
        self._expect_err("INVALID_INT_TYPE", oracle.require_exact_int, True)
        self._expect_err("INT_BIT_LIMIT", oracle.require_exact_int, 1 << oracle.MAX_INT_BITS)
        self._expect_err("INVALID_BOUND_TYPE", oracle.require_exact_int, 5, lo="0")
        self._expect_err("INVALID_BOUND_TYPE", oracle.require_exact_int, 5, hi="9")
        self._expect_err("INT_OUT_OF_RANGE", oracle.require_exact_int, -1, lo=0)
        self._expect_err("INT_OUT_OF_RANGE", oracle.require_exact_int, 10, hi=9)

        self.assertEqual(oracle.require_exact_fraction(Fraction(max_ok, 1)), Fraction(max_ok, 1))
        self.assertEqual(oracle.require_exact_fraction(Fraction(1, max_ok)), Fraction(1, max_ok))
        self._expect_err("INVALID_FRACTION_TYPE", oracle.require_exact_fraction, 3)
        self._expect_err("RATIONAL_BIT_LIMIT", oracle.require_exact_fraction, Fraction(1 << oracle.MAX_RATIONAL_BITS, 1))
        self._expect_err("RATIONAL_BIT_LIMIT", oracle.require_exact_fraction, Fraction(1, 1 << oracle.MAX_RATIONAL_BITS))

        self.assertEqual(oracle.format_rational(Fraction(5, 1)), "5")
        self.assertEqual(oracle.format_rational(Fraction(-3, 4)), "-3/4")
        self.assertEqual(oracle.format_decimal(Fraction(1, 200), 2), "0.01")
        self.assertEqual(oracle.format_decimal(Fraction(1, 201), 2), "0.00")
        self.assertEqual(oracle.format_decimal(Fraction(-1, 10000), 2), "0.00")
        self.assertEqual(oracle.format_decimal(Fraction(-7, 2), 0), "-4")
        self.assertEqual(oracle.format_decimal(Fraction(7, 2), 0), "4")
        self.assertEqual(oracle.format_decimal(Fraction(0), 0), "0")
        self._expect_err("INT_OUT_OF_RANGE", oracle.format_decimal, Fraction(1), -1)
        self._expect_err("INT_OUT_OF_RANGE", oracle.format_decimal, Fraction(1), 13)

    # ------------------------------------------------------------------
    # 3. Coding Theory, Bareiss Det, Decoder & E_8 Guards
    # ------------------------------------------------------------------
    def test_03_coding_lattice_and_decoder_guards(self) -> None:
        self._expect_err("INT_OUT_OF_RANGE", oracle.build_parity_check_matrix, 2)
        self._expect_err("INT_OUT_OF_RANGE", oracle.build_parity_check_matrix, 6)
        self._expect_err("INT_OUT_OF_RANGE", oracle.build_generator_matrix, 2)
        self._expect_err("INT_OUT_OF_RANGE", oracle.build_construction_a_basis_L, 6)
        self._expect_err("INT_OUT_OF_RANGE", oracle.build_sylvester_hadamard, 0)
        self.assertEqual(oracle.build_sylvester_hadamard(1), [[1, 1], [1, -1]])

        # H*G^T orthogonality guards
        h8 = oracle.build_parity_check_matrix(3)
        g8 = oracle.build_generator_matrix(3)
        self.assertTrue(oracle.verify_hg_orthogonality(h8, g8))
        bad_g = copy.deepcopy(g8)
        bad_g[0][0] ^= 1
        self.assertFalse(oracle.verify_hg_orthogonality(h8, bad_g))
        self._expect_err("INVALID_MATRIX", oracle.verify_hg_orthogonality, "bad", g8)
        self._expect_err("INVALID_MATRIX", oracle.verify_hg_orthogonality, [], g8)
        self._expect_err("INVALID_MATRIX", oracle.verify_hg_orthogonality, h8, "bad")
        self._expect_err("INVALID_MATRIX", oracle.verify_hg_orthogonality, h8, [])
        self._expect_err("DIM_MISMATCH", oracle.verify_hg_orthogonality, ["bad"], g8)
        self._expect_err("DIM_MISMATCH", oracle.verify_hg_orthogonality, [[1, 0], (1, 0)], [[1, 0]])
        self._expect_err("DIM_MISMATCH", oracle.verify_hg_orthogonality, [[1, 0], [1]], [[1, 0]])
        self._expect_err("DIM_MISMATCH", oracle.verify_hg_orthogonality, h8, ["bad"])
        self._expect_err("DIM_MISMATCH", oracle.verify_hg_orthogonality, [[1, 0]], [(1, 0)])
        self._expect_err("DIM_MISMATCH", oracle.verify_hg_orthogonality, h8, [[1]])
        self._expect_err("INT_OUT_OF_RANGE", oracle.verify_hg_orthogonality, [[2]], [[1]])
        self._expect_err("INT_OUT_OF_RANGE", oracle.verify_hg_orthogonality, [[1]], [[2]])

        # Weight enumeration & MacWilliams guards
        self._expect_err("INVALID_GENERATOR_K", oracle.enumerate_code_weight_dist, "bad")
        self._expect_err("INVALID_GENERATOR_K", oracle.enumerate_code_weight_dist, [])
        self._expect_err("INVALID_GENERATOR_K", oracle.enumerate_code_weight_dist, [[1]] * 13)
        self._expect_err("DIM_MISMATCH", oracle.enumerate_code_weight_dist, [[1, 0], "bad"])
        self._expect_err("DIM_MISMATCH", oracle.enumerate_code_weight_dist, [[1, 0], (1, 0)])
        self._expect_err("DIM_MISMATCH", oracle.enumerate_code_weight_dist, [[1, 0], [1]])
        self._expect_err("INT_OUT_OF_RANGE", oracle.enumerate_code_weight_dist, [[2]])

        mw8 = oracle.macwilliams_from_dual_dist(8, {0: 1, 2: 0, 4: 14, 8: 1})
        self.assertEqual(mw8, {0: 1, 4: 14, 8: 1})
        self.assertEqual(oracle.macwilliams_from_dual_dist(4, {0: 1, 4: 1}), {0: 1, 2: 6, 4: 1})
        self._expect_err("INVALID_DUAL_DIST", oracle.macwilliams_from_dual_dist, 8, [(0, 1)])
        self._expect_err("INVALID_DUAL_DIST", oracle.macwilliams_from_dual_dist, 8, [])
        self._expect_err("INVALID_DUAL_DIST", oracle.macwilliams_from_dual_dist, 8, {})
        self._expect_err("INVALID_DUAL_SIZE", oracle.macwilliams_from_dual_dist, 8, {0: 1})
        self._expect_err("MACWILLIAMS_INTEGRALITY_ERROR", oracle.macwilliams_from_dual_dist, 4, {0: 2, 4: 1})
        self._expect_err("MACWILLIAMS_INTEGRALITY_ERROR", oracle.macwilliams_from_dual_dist, 4, {4: 2})

        # Bareiss determinant guards
        self._expect_err("INVALID_MATRIX", oracle.bareiss_det_int, "bad")
        self._expect_err("INVALID_MATRIX", oracle.bareiss_det_int, [])
        self._expect_err("NON_SQUARE_MATRIX", oracle.bareiss_det_int, ["bad"])
        self._expect_err("NON_SQUARE_MATRIX", oracle.bareiss_det_int, [(1,)])
        self._expect_err("NON_SQUARE_MATRIX", oracle.bareiss_det_int, [[1, 2]])
        self.assertEqual(oracle.bareiss_det_int([[1, 1], [1, 1]]), 0)
        self.assertEqual(oracle.bareiss_det_int([[0, 1], [1, 0]]), -1)

        # Membership & decoder guards
        self._expect_err("INVALID_VECTOR_DIM", oracle.verify_in_construction_a_L, "bad")
        self._expect_err("INVALID_VECTOR_DIM", oracle.verify_in_construction_a_L, tuple([0] * 8))
        self._expect_err("INVALID_VECTOR_DIM", oracle.verify_in_construction_a_L, [0] * 7)
        self.assertTrue(oracle.verify_in_construction_a_L([0] * 8))
        self.assertFalse(oracle.verify_in_construction_a_L([1] + [0] * 7))
        self.assertFalse(oracle.verify_in_construction_a_L([1, 1] + [0] * 6))

        self._expect_err("INVALID_VECTOR_DIM", oracle.decode_wagner_12, "bad")
        self._expect_err("INVALID_VECTOR_DIM", oracle.decode_wagner_12, tuple([Fraction(0)] * 8))
        self._expect_err("INVALID_VECTOR_DIM", oracle.decode_wagner_12, [Fraction(0)] * 7)
        self._expect_err("INVALID_FRACTION_TYPE", oracle.decode_wagner_12, [0] * 8)

        # Cover all 3 syndrome cases on decode_wagner_12 & decode_construction_a_exact
        pt0, d0, cert0 = oracle.decode_wagner_12([Fraction(0)] * 8)
        self.assertEqual((pt0, d0, cert0), ([0] * 8, Fraction(0), True))
        pt0e, d0e, fast0e = oracle.decode_construction_a_exact([Fraction(0)] * 8)
        self.assertEqual((pt0e, d0e, fast0e), ([0] * 8, Fraction(0), True))

        # Odd syndrome fast certified (s0=1, flip index 0)
        v_odd = [Fraction(3, 5)] + [Fraction(0)] * 7
        pt_o, d_o, c_o = oracle.decode_wagner_12(v_odd)
        pt_oe, d_oe, f_oe = oracle.decode_construction_a_exact(v_odd)
        self.assertTrue(c_o and f_oe and pt_o == pt_oe and d_o == d_oe)

        # Even nonzero syndrome fast certified with optimal pair at later index (2, 3) beating (0, 1)
        v_ev = [Fraction(0), Fraction(0), Fraction(3, 5), Fraction(3, 5)] + [Fraction(0)] * 4
        pt_e, d_e, c_e = oracle.decode_wagner_12(v_ev)
        pt_ee, d_ee, f_ee = oracle.decode_construction_a_exact(v_ev)
        self.assertTrue(c_e and f_ee and pt_e == pt_ee and d_e == d_ee and pt_e == [0] * 8)

        # Bruteforce & E_8 guards
        self._expect_err("INT_OUT_OF_RANGE", oracle.decode_bruteforce_exact, [Fraction(0)] * 8, 2)
        self._expect_err("INT_OUT_OF_RANGE", oracle.decode_bruteforce_exact, [Fraction(0)] * 8, 5)
        self._expect_err("DIM_MISMATCH", oracle.decode_bruteforce_exact, "bad", 3)
        self._expect_err("DIM_MISMATCH", oracle.decode_bruteforce_exact, tuple([Fraction(0)] * 8), 3)
        self._expect_err("DIM_MISMATCH", oracle.decode_bruteforce_exact, [Fraction(0)] * 16, 3)

        self._expect_err("INVALID_VECTOR_DIM", oracle.decode_e8_conway_sloane, "bad")
        self._expect_err("INVALID_VECTOR_DIM", oracle.decode_e8_conway_sloane, tuple([Fraction(0)] * 8))
        self._expect_err("INVALID_VECTOR_DIM", oracle.decode_e8_conway_sloane, [Fraction(0)] * 7)
        self._expect_err("INVALID_FRACTION_TYPE", oracle.decode_e8_conway_sloane, [0] * 8)
        v_e8_spin, d_e8_spin = oracle.decode_e8_conway_sloane([Fraction(1, 2)] * 8)
        self.assertEqual((v_e8_spin, d_e8_spin), ([Fraction(1, 2)] * 8, Fraction(0)))
        v_e8_odd, d_e8_odd = oracle.decode_e8_conway_sloane([Fraction(1)] + [Fraction(0)] * 7)
        self.assertEqual(d_e8_odd, Fraction(1))

    # ------------------------------------------------------------------
    # 4. Exact Four-Pillar Mathematical Payload Checks
    # ------------------------------------------------------------------
    def test_04_four_pillars_exact_values(self) -> None:
        pl = self.payload
        self.assertEqual(pl["track_id"], oracle.TRACK_ID)
        self.assertEqual(pl["portal_ref"], "Portal 52")
        self.assertEqual(pl["novelty_class"], "N1")
        self.assertIs(pl["measured_live_model_weights"], False)

        p1 = pl["pillar1_hamming_codes_and_lattices"]
        self.assertTrue(p1["all_verified"])
        lats = p1["lattices"]
        self.assertEqual([x["a4_weight4_count"] for x in lats], [14, 140, 1240])
        self.assertEqual([x["kissing_number_tau"] for x in lats], [240, 2272, 19904])
        self.assertEqual([x["covolume_A_n_q"] for x in lats], ["1", "1/8", "1/1024"])
        self.assertEqual([x["det_gram_A_n_q"] for x in lats], ["1", "1/64", "1/1048576"])
        self.assertEqual([x["exhaustive_weight_enum_checked"] for x in lats], [True, True, False])
        self.assertEqual([x["exhaustive_matches_macwilliams"] for x in lats], [True, True, None])
        self.assertEqual([x["macwilliams_a4_exact"] for x in lats], [True, True, True])
        self.assertEqual([x["gamma_pow_32"] for x in lats], [1 << 32, 1 << 44, 1 << 52])
        self.assertTrue(p1["e8_hadamard_a8_bijection"]["bijection_exact_240_of_240"])
        self.assertTrue(p1["barnes_wall_lambda16_comparison"]["a16_sparser_than_lambda16"])

        p2 = pl["pillar2_syndrome_decoder_optimality"]
        self.assertTrue(p2["all_verified"])
        self.assertEqual(p2["c8_exact_matches_16cw_bruteforce"], 16)
        self.assertEqual(p2["c8_exact_matches_rotated_e8"], 16)
        self.assertEqual(p2["c16_exact_matches_2048cw_bruteforce"], 24)
        self.assertEqual(p2["c32_valid_and_le_wagner_count"], 12)
        ctrls = p2["naive_wagner_suboptimality_controls"]
        self.assertEqual([c["exact_bnb_sq_dist_q"] for c in ctrls], ["363/400", "121/100", "363/400"])
        self.assertEqual([c["certified_optimal_sq_dist_q"] for c in ctrls], ["363/400", "121/100", "363/400"])
        self.assertEqual([c["exhaustive_bruteforce_executed"] for c in ctrls], [True, True, False])
        self.assertEqual(
            [c["optimal_proof_method"] for c in ctrls],
            ["exhaustive_bruteforce", "exhaustive_bruteforce", "analytical_ge3_flip_lower_bound"],
        )
        self.assertEqual([c["exact_matches_certified_optimal"] for c in ctrls], [True, True, True])

        p3 = pl["pillar3_voronoi_zero_retention_and_accumulators"]
        self.assertTrue(p3["all_verified"])
        self.assertTrue(p3["dead_zone_strictly_expanding"])
        self.assertTrue(p3["laplace_zero_retention_strictly_increasing"])
        self.assertTrue(p3["midband_parity_spill_control"]["weight4_parity_spill_triggered"])
        self.assertEqual([r["fits_int16"] for r in p3["integer_accumulator_bounds"]], [True, True, True, False])
        f64 = p3["f64_exact_decode_theorem"]
        self.assertEqual(f64["sharp_unconditional_w_max"], 64)
        self.assertTrue(f64["w64_unconditionally_exact_in_f64"])
        self.assertTrue(f64["w65_exceeds_2p53_and_is_odd"])
        self.assertTrue(f64["int8_w128_exact_when_act_le_32752"])

        p4 = pl["pillar4_honest_scope_and_proxy_obstruction"]
        self.assertTrue(p4["all_verified"])
        self.assertIs(p4["measured_live_model_weights"], False)
        self.assertEqual(p4["data_free_column_proxy_obstruction"]["total_ratio_distortion_q"], "16384")
        self.assertTrue(p4["data_free_column_proxy_obstruction"]["ranking_sign_inverted"])

    # ------------------------------------------------------------------
    # 5. Strict Parsers, Binding 3,321-Bit Cap & Canonical JSON
    # ------------------------------------------------------------------
    def test_05_parsers_binding_bit_cap_and_canonical_json(self) -> None:
        self.assertEqual(audit.require_hex64_lower_ascii("a" * 64), "a" * 64)
        self._expect_err("INVALID_HEX64", audit.require_hex64_lower_ascii, "a" * 63)
        self._expect_err("INVALID_HEX64", audit.require_hex64_lower_ascii, "A" * 64)
        self._expect_err("INVALID_HEX64", audit.require_hex64_lower_ascii, "g" * 64)
        self._expect_err("INVALID_STR", audit.require_hex64_lower_ascii, 123)

        self.assertEqual(audit.parse_exact_rational("0"), Fraction(0))
        self.assertEqual(audit.parse_exact_rational("-5"), Fraction(-5))
        self.assertEqual(audit.parse_exact_rational("-7/11"), Fraction(-7, 11))
        for bad in ("-0", "01", "", "-", "1/0", "1/1", "2/4", "1/2/3", "\u0661", "1/02", "9" * 1001):
            with self.assertRaises(oracle.HammingLatticeError):
                audit.parse_exact_rational(bad)

        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        self.assertEqual(len(str(max_ok)), oracle.MAX_INT_DIGITS)
        rt = audit.parse_strict_json_text(f'{{"x":{max_ok}}}')
        self.assertEqual(rt["x"], max_ok)
        self.assertEqual(audit.canonical_json_bytes(rt), f'{{"x":{max_ok}}}'.encode("ascii"))

        self._expect_err("INT_BIT_LIMIT", audit._parse_bounded_int, "9" * 1000)
        self._expect_err("INT_BIT_LIMIT", audit.canonical_json_bytes, {"x": 1 << oracle.MAX_INT_BITS})
        self._expect_err("INT_BIT_LIMIT", audit.parse_strict_json_text, f'{{"x":{1 << oracle.MAX_INT_BITS}}}')
        self._expect_err("INT_DIGIT_LIMIT", audit.parse_strict_json_text, '{"x":1' + "0" * 4300 + "}")
        self._expect_err("FLOAT_LITERAL_FORBIDDEN", audit.parse_strict_json_text, '{"x":1.5}')
        self._expect_err("FLOAT_LITERAL_FORBIDDEN", audit.parse_strict_json_text, '{"x":1e2}')
        self._expect_err("NONFINITE_LITERAL_FORBIDDEN", audit.parse_strict_json_text, '{"x":NaN}')
        self._expect_err("DUPLICATE_JSON_KEY", audit.parse_strict_json_text, '{"a":1,"a":2}')
        self._expect_err("JSON_PARSE_ERROR", audit.parse_strict_json_text, '{"a":')
        self._expect_err("JSON_ROOT_NOT_OBJECT", audit.parse_strict_json_text, "[1,2]")
        self.assertEqual(audit.parse_strict_json_text("[1,null,true,false]", require_object=False), [1, None, True, False])
        self.assertEqual(audit.parse_strict_json_text('{"a":1}', max_bytes=7), {"a": 1})
        self._expect_err("JSON_TOO_LARGE", audit.parse_strict_json_text, '{"a":"0123456789"}', max_bytes=10)
        self._expect_err("INT_OUT_OF_RANGE", audit.parse_strict_json_text, "{}", max_bytes=1 << 100)
        self._expect_err("NUL_BYTE_FORBIDDEN", audit.parse_strict_json_text, '{"a":"\x00"}')
        self._expect_err("INVALID_UTF8_SURROGATE", audit.parse_strict_json_text, '{"a":"\ud800"}')
        self._expect_err("FORBIDDEN_JSON_TYPE", audit.canonical_json_bytes, {"a": {1, 2}})

        d64 = "[" * 64 + "0" + "]" * 64
        d65 = "[" * 65 + "0" + "]" * 65
        self.assertIsInstance(audit.parse_strict_json_text(d64, require_object=False), list)
        self._expect_err("JSON_DEPTH_LIMIT", audit.parse_strict_json_text, d65, require_object=False)
        self.assertEqual(audit.canonical_json_bytes({"b": "\u00e9", "a": 1}), b'{"a":1,"b":"\\u00e9"}')

    # ------------------------------------------------------------------
    # 6. Bijective Bool-Leaf Guard, Zero-Denominator Sweep & Tamper Matrix
    # ------------------------------------------------------------------
    def test_06_secondary_audit_zero_denom_and_tamper_matrix(self) -> None:
        self.assertTrue(audit.verify_certificate_dict(self.cert, full_replay=True)["valid"])
        self.assertTrue(audit.verify_certificate_dict(self.cert, full_replay=False)["valid"])
        self._expect_err("INVALID_BOOL_TYPE", audit.verify_certificate_dict, self.cert, full_replay=1)

        # Envelope tampers
        self._expect_err("INVALID_CERT_TYPE", audit.verify_certificate_dict, [])
        c_extra = copy.deepcopy(self.cert)
        c_extra["rogue"] = 1
        self._expect_err("INVALID_CERT_KEYS", audit.verify_certificate_dict, c_extra, full_replay=False)
        c_sch = copy.deepcopy(self.cert)
        c_sch["schema_version"] = "bad"
        self._expect_err("INVALID_SCHEMA", audit.verify_certificate_dict, c_sch, full_replay=False)
        c_tid = copy.deepcopy(self.cert)
        c_tid["track_id"] = "bad"
        self._expect_err("INVALID_TRACK_ID", audit.verify_certificate_dict, c_tid, full_replay=False)
        c_pt = copy.deepcopy(self.cert)
        c_pt["payload"] = []
        self._expect_err("INVALID_PAYLOAD_TYPE", audit.verify_certificate_dict, c_pt, full_replay=False)
        c_psha = copy.deepcopy(self.cert)
        c_psha["payload_sha256"] = "0" * 64
        self._expect_err("PAYLOAD_SHA256_MISMATCH", audit.verify_certificate_dict, c_psha, full_replay=False)
        c_csha = copy.deepcopy(self.cert)
        c_csha["certificate_sha256"] = "0" * 64
        self._expect_err("CERT_SHA256_MISMATCH", audit.verify_certificate_dict, c_csha, full_replay=False)
        self._expect_err("INVALID_SCHEMA_DICT", audit._req_dict, [], "row", frozenset())
        self._expect_err("INVALID_SCHEMA_LIST", audit._req_list, (1, 2), "row", 2)

        def mutate_and_expect(mutator, exp_code: str) -> None:
            pl = copy.deepcopy(self.payload)
            mutator(pl)
            resigned = _resign_payload(pl)
            self._expect_err(exp_code, audit.verify_certificate_dict, resigned, full_replay=False)

        # Exhaustive zero/negative/invalid-denom & direct secondary_audit_payload float injection sweep!
        zero_probes = 0
        float_probes = 0
        flt_val = json.loads("1.5")
        self.assertIs(type(flt_val), type(json.loads("0.0")))

        def _collect_paths(node: object, path: tuple[object, ...]) -> list[tuple[tuple[object, ...], object]]:
            out: list[tuple[tuple[object, ...], object]] = []
            if type(node) is dict:
                for k, v in node.items():
                    out.extend(_collect_paths(v, path + (k,)))
            elif type(node) is list:
                for idx, v in enumerate(node):
                    out.extend(_collect_paths(v, path + (idx,)))
            else:
                out.append((path, node))
            return out

        def _set_at(obj: Any, path: tuple[object, ...], val: Any) -> None:
            cur = obj
            for step in path[:-1]:
                cur = cur[step]
            cur[path[-1]] = val

        all_leaf_paths = _collect_paths(self.payload, ())
        for path, orig_val in all_leaf_paths:
            last = path[-1]
            # Direct secondary_audit_payload float injection across ALL leaves -> 100% FORBIDDEN_JSON_TYPE!
            pl_f = copy.deepcopy(self.payload)
            _set_at(pl_f, path, flt_val)
            self._expect_err("FORBIDDEN_JSON_TYPE", audit.secondary_audit_payload, pl_f)
            float_probes += 1

            # Zero/negative/invalid-denom sweep across every rational & integer leaf -> 0 raw exceptions!
            if (type(last) is str and last.endswith("_q")) or (
                len(path) >= 2 and type(path[-2]) is str and path[-2].endswith("_q")
            ):
                for bad_tok in ("0", "1/0"):
                    pl_z = copy.deepcopy(self.payload)
                    _set_at(pl_z, path, bad_tok)
                    resigned = _resign_payload(pl_z)
                    with self.assertRaises(oracle.HammingLatticeError):
                        audit.verify_certificate_dict(resigned, full_replay=False)
                    with self.assertRaises(oracle.HammingLatticeError):
                        audit.secondary_audit_payload(pl_z)
                zero_probes += 1
            elif type(orig_val) is int:
                for bad_int in (0, -1):
                    if bad_int == orig_val:
                        continue
                    pl_i = copy.deepcopy(self.payload)
                    _set_at(pl_i, path, bad_int)
                    resigned = _resign_payload(pl_i)
                    with self.assertRaises(oracle.HammingLatticeError):
                        audit.verify_certificate_dict(resigned, full_replay=False)
                    with self.assertRaises(oracle.HammingLatticeError):
                        audit.secondary_audit_payload(pl_i)
                zero_probes += 1
        self.assertGreaterEqual(zero_probes, 75)
        self.assertGreaterEqual(float_probes, 100)

        # 55+ targeted re-signed full_replay=False tampers across ALL 4 pillars, Shift-DoS, nullable bools & keys
        tampers = [
            (lambda p: p.__setitem__("track_id", "WRONG"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p.__setitem__("portal_ref", "Portal 99"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p.__setitem__("novelty_class", "N0"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p.__setitem__("measured_live_model_weights", True), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p.__setitem__("measured_live_model_weights", 0), "INVALID_BOOL_TYPE"),
            (lambda p: p.__setitem__("extra_top", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["prior_literature_n0"].__setitem__(0, True), "UNEXPECTED_BOOL_ITEM"),
            (lambda p: p["prior_literature_n0"].__setitem__(0, 123), "INVALID_STR"),
            (lambda p: p["prior_literature_n0"].pop(), "INVALID_SCHEMA_LIST"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"].pop(), "INVALID_SCHEMA_LIST"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("m", 2), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("m", -1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("m", 100_000), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("n", 0), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("n", -8), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("k", -1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("m", True), "UNEXPECTED_BOOL_LEAF"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("covolume_A_n_q", "1/2"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("covolume_A_n_q", "-1"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("det_gram_A_n_q", "1/2"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("gamma_pow_32", 1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("exhaustive_matches_macwilliams", None), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][2].__setitem__("exhaustive_matches_macwilliams", True), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][2].__setitem__("exhaustive_matches_macwilliams", 1), "INVALID_BOOL_TYPE"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["e8_hadamard_a8_bijection"].__setitem__("e8_root_count", 239), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar1_hamming_codes_and_lattices"]["barnes_wall_lambda16_comparison"].__setitem__("a16_kissing", 9999), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"]["naive_wagner_suboptimality_controls"][0].__setitem__("exact_bnb_sq_dist_q", "10"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"]["naive_wagner_suboptimality_controls"][0].__setitem__("certified_optimal_sq_dist_q", "10"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"]["naive_wagner_suboptimality_controls"][2].__setitem__("exhaustive_bruteforce_executed", True), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"]["naive_wagner_suboptimality_controls"][2].__setitem__("optimal_proof_method", "exhaustive_bruteforce"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"]["naive_wagner_suboptimality_controls"][0].__setitem__("naive_wagner_certified_optimal", True), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar2_syndrome_decoder_optimality"]["naive_wagner_suboptimality_controls"][0].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["unit_covolume_dead_zones"][1].__setitem__("r_pack_sq_pow8_q", "1/512"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["discrete_laplace_zero_retention"][1].__setitem__("zero_coords", 0), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["discrete_laplace_zero_retention"][1].__setitem__("total_coords", 0), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["midband_parity_spill_control"].__setitem__("nonzero_coords_after_decode", 3), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["integer_accumulator_bounds"][0].__setitem__("max_abs_dot", 1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["wht_int8_bounds"][0].__setitem__("max_abs_wht_of_int8", 1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["f64_exact_decode_theorem"].__setitem__("w64_max_scaled_sum", 1 << 54), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar3_voronoi_zero_retention_and_accumulators"]["f64_exact_decode_theorem"].__setitem__("w65_failure_witness_scaled_sum", 1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar4_honest_scope_and_proxy_obstruction"].__setitem__("measured_live_model_weights", True), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar4_honest_scope_and_proxy_obstruction"]["data_free_column_proxy_obstruction"].__setitem__("total_ratio_distortion_q", "1"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar4_honest_scope_and_proxy_obstruction"]["data_free_column_proxy_obstruction"]["weight_col_rms_sq_q"].__setitem__(0, "2"), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["pillar4_honest_scope_and_proxy_obstruction"]["data_free_column_proxy_obstruction"].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["summary"].__setitem__("extra", 1), "INVALID_SCHEMA_KEYS"),
            (lambda p: p["summary"].__setitem__("pillar_count", True), "UNEXPECTED_BOOL_LEAF"),
            (lambda p: p["summary"].__setitem__("all_four_pillars_verified", 1), "INVALID_BOOL_TYPE"),
            (lambda p: p["summary"].__setitem__("float_count_on_trust_path", 1), "SECONDARY_AUDIT_FAILED"),
            (lambda p: p["summary"].__setitem__("rng_calls", 1), "SECONDARY_AUDIT_FAILED"),
        ]
        self.assertGreaterEqual(len(tampers), 50)
        for mutator, exp_code in tampers:
            mutate_and_expect(mutator, exp_code)

        # Live CLI --check regression for Shift-DoS & ZeroDivisionError (n=0, m=-1, m=100000, k=-1, det="0")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for idx, mut in enumerate((
                lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("n", 0),
                lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("m", -1),
                lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("m", 100_000),
                lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("k", -1),
                lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("det_gram_A_n_q", "0"),
                lambda p: p["pillar1_hamming_codes_and_lattices"]["lattices"][0].__setitem__("covolume_A_n_q", "0"),
            )):
                pl_cli = copy.deepcopy(self.payload)
                mut(pl_cli)
                c_path = root / f"dos_{idx}.json"
                audit.write_regular_bytes_safe(c_path, audit.canonical_json_bytes(_resign_payload(pl_cli)) + b"\n")
                rc, _out, err = _call_main(["--check", str(c_path), "--json"])
                self.assertEqual(rc, 2)
                self.assertEqual(json.loads(err)["error_code"], "SECONDARY_AUDIT_FAILED")

        # Replay digest mismatch guard independently verified
        with mock.patch.object(audit, "build_standalone_certificate", return_value={"payload_sha256": "0" * 64}):
            self._expect_err("REPLAY_MISMATCH", audit.verify_certificate_dict, self.cert, full_replay=True)

    # ------------------------------------------------------------------
    # 7. Zone-B File I/O, Symlink/PWD/Hardlink/FIFO/chmod 000 & CLI
    # ------------------------------------------------------------------
    def test_07_zone_b_io_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            cert_path = root / "sub" / "cert.json"
            wres = audit.write_certificate_file(cert_path)
            self.assertTrue(wres["written"])
            self.assertEqual(stat.S_IMODE(os.lstat(cert_path).st_mode), 0o600)
            cres = audit.check_certificate_file(cert_path, full_replay=True)
            self.assertTrue(cres["valid"])
            self.assertEqual(cres["file_sha256"], wres["file_sha256"])

            missing_dir = root / "never_created"
            self._expect_err("FILE_READ_ERROR", audit.read_regular_bytes_capped, missing_dir / "x.json")
            self.assertFalse(missing_dir.exists())

            unreadable = root / "chmod000.json"
            unreadable.write_bytes(cert_path.read_bytes())
            os.chmod(unreadable, 0o000)
            try:
                self._expect_err("FILE_READ_ERROR", audit.read_regular_bytes_capped, unreadable)
            finally:
                os.chmod(unreadable, 0o600)

            crlf_p = root / "crlf.json"
            crlf_p.write_bytes(cert_path.read_bytes().replace(b"\n", b"\r\n"))
            self._expect_err("CRLF_FORBIDDEN", audit.check_certificate_file, crlf_p)

            nolf_p = root / "nolf.json"
            nolf_p.write_bytes(cert_path.read_bytes()[:-1])
            self._expect_err("MISSING_TRAILING_LF", audit.check_certificate_file, nolf_p)

            badu_p = root / "badu.json"
            badu_p.write_bytes(b"\xff\xfe\n")
            self._expect_err("INVALID_UTF8", audit.check_certificate_file, badu_p)

            pretty_p = root / "pretty.json"
            pretty_p.write_bytes((json.dumps(self.cert, indent=2) + "\n").encode("utf-8"))
            self._expect_err("NON_CANONICAL_BYTES", audit.check_certificate_file, pretty_p)

            sym_leaf = root / "sym_leaf.json"
            os.symlink(cert_path, sym_leaf)
            self._expect_err("SYMLINK_FORBIDDEN", audit.read_regular_bytes_capped, sym_leaf)
            self._expect_err("SYMLINK_FORBIDDEN", audit.write_regular_bytes_safe, sym_leaf, b"x")

            sym_dir = root / "sym_dir"
            os.symlink(root / "sub", sym_dir)
            self._expect_err("SYMLINK_FORBIDDEN", audit.read_regular_bytes_capped, sym_dir / "cert.json")
            self._expect_err("SYMLINK_FORBIDDEN", audit.write_regular_bytes_safe, sym_dir / "new.json", b"x")

            self._expect_err("NOT_A_REGULAR_FILE", audit.read_regular_bytes_capped, root / "sub")
            self._expect_err("NOT_A_REGULAR_FILE", audit.write_regular_bytes_safe, root / "sub", b"x")
            self._expect_err("NOT_A_REGULAR_FILE", audit.read_regular_bytes_capped, Path("/"))
            self._expect_err("NOT_A_REGULAR_FILE", audit.read_regular_bytes_capped, missing_dir / "..")
            self._expect_err("NOT_A_REGULAR_FILE", audit.read_regular_bytes_capped, Path("/dev/null"))

            fifo_p = root / "pipe.fifo"
            os.mkfifo(fifo_p)
            self._expect_err("NOT_A_REGULAR_FILE", audit.read_regular_bytes_capped, fifo_p)

            hl_p = root / "hardlink.json"
            os.link(cert_path, hl_p)
            self._expect_err("HARDLINK_FORBIDDEN", audit.write_regular_bytes_safe, hl_p, b"x")
            os.unlink(hl_p)

            big_p = root / "big70k.bin"
            big_p.write_bytes(b"A" * 70_000)
            self.assertEqual(len(audit.read_regular_bytes_capped(big_p, "exact", 70_000)), 70_000)
            self._expect_err("FILE_TOO_LARGE", audit.read_regular_bytes_capped, big_p, "big", 65_536)
            if Path("/proc/kallsyms").exists():
                self._expect_err("FILE_TOO_LARGE", audit.read_regular_bytes_capped, Path("/proc/kallsyms"), "ks", 1024)

            fake_st = os.stat_result((stat.S_IFREG | 0o600, 1, 1, 2, 0, 0, 1, 0, 0, 0))
            with mock.patch.object(os, "fstat", return_value=fake_st):
                self._expect_err("HARDLINK_FORBIDDEN", audit.write_regular_bytes_safe, root / "race_nlink.bin", b"x")

            self._expect_err("INVALID_PATH_TYPE", audit.read_regular_bytes_capped, str(cert_path))  # type: ignore[arg-type]
            self._expect_err("INVALID_PAYLOAD_BYTES", audit.write_regular_bytes_safe, root / "t.bin", "str")  # type: ignore[arg-type]

            orig_cwd = os.getcwd()
            orig_pwd = os.environ.get("PWD")
            try:
                os.chdir(root)
                os.environ["PWD"] = str(root)
                rel_p = Path("rel_out.json")
                audit.write_regular_bytes_safe(rel_p, b"{}\n")
                self.assertEqual(audit.read_regular_bytes_capped(rel_p), b"{}\n")

                os.environ["PWD"] = str(sym_dir)
                self._expect_err("SYMLINK_ANCESTOR_FORBIDDEN", audit.read_regular_bytes_capped, Path("rel_out.json"))

                ghost = root / "ghost_dir"
                ghost.mkdir()
                os.chdir(ghost)
                os.environ["PWD"] = str(ghost)
                ghost.rmdir()
                self._expect_err("CWD_UNAVAILABLE", audit.read_regular_bytes_capped, Path("rel.json"))
            finally:
                os.chdir(orig_cwd)
                if orig_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = orig_pwd

            rc, out, err = _call_main([])
            self.assertEqual((rc, err), (0, ""))
            self.assertIn("OK ", out)

            rc, out, err = _call_main(["--standalone", "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertEqual(json.loads(out)["certificate_sha256"], self.cert["certificate_sha256"])

            cli_w = root / "cli_w.json"
            rc, out, err = _call_main(["--write", str(cli_w), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(json.loads(out)["written"])

            rc, out, err = _call_main(["--check", str(cli_w), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(json.loads(out)["valid"])

            cli_w2 = root / "cli_w2.json"
            rc, out, err = _call_main([f"--write={cli_w2}"])
            self.assertEqual((rc, err), (0, ""))
            rc, out, err = _call_main([f"--check={cli_w2}", "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(json.loads(out)["valid"])

            for bad_argv, exp_code in (
                (["--unknown"], "UNKNOWN_FLAG"),
                (["--json", "--json"], "DUPLICATE_FLAG"),
                (["--standalone", "--check", str(cli_w)], "CONFLICTING_MODE"),
                (["--check", str(cli_w), "--standalone"], "CONFLICTING_MODE"),
                (["--standalone", f"--check={cli_w}"], "CONFLICTING_MODE"),
                (["--check"], "MISSING_PATH_ARG"),
                (["--check", ""], "INVALID_PATH_ARG"),
                (["--check", "-bad"], "INVALID_PATH_ARG"),
                (["--check="], "INVALID_PATH_ARG"),
                (["--write=-bad"], "INVALID_PATH_ARG"),
                (["--check", "a\x00b"], "NUL_BYTE_FORBIDDEN"),
            ):
                rc, _o, e = _call_main(bad_argv)
                self.assertEqual(rc, 2)
                self.assertEqual(json.loads(e)["error_code"], exp_code)

            rc, _o, e = _call_main("not-a-list")  # type: ignore[arg-type]
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(e)["error_code"], "INVALID_ARGV")

    # ------------------------------------------------------------------
    # 8. Concurrency & Kernel Races (16T/80T + 550 renameat2 + 550 hardlink)
    # ------------------------------------------------------------------
    def test_08_concurrency_and_kernel_races(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()

            errs: list[Exception] = []
            def w_distinct(idx: int) -> None:
                try:
                    p = root / f"d_{idx}.bin"
                    audit.write_regular_bytes_safe(p, f"payload-{idx}\n".encode("ascii"))
                    if stat.S_IMODE(os.lstat(p).st_mode) != 0o600:
                        raise RuntimeError("bad mode")
                except Exception as exc:
                    errs.append(exc)

            ts = [threading.Thread(target=w_distinct, args=(i,)) for i in range(16)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            self.assertEqual(errs, [])

            shared = root / "shared.bin"
            def w_same(idx: int) -> None:
                try:
                    audit.write_regular_bytes_safe(shared, f"v-{idx}\n".encode("ascii"))
                except Exception as exc:
                    errs.append(exc)

            for count in (16, 80):
                ts = [threading.Thread(target=w_same, args=(i,)) for i in range(count)]
                for t in ts:
                    t.start()
                for t in ts:
                    t.join()
            self.assertEqual(errs, [])
            self.assertEqual([p.name for p in root.iterdir() if p.name.endswith(".tmp")], [])

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            sys_renameat2 = 316
            rename_exchange = 2
            at_fdcwd = -100
            victim_dir = root / "victim_outside"
            victim_dir.mkdir()
            real_sub = root / "race_slot_a"
            real_sub.mkdir()
            sym_sub = root / "race_slot_b"
            os.symlink(victim_dir, sym_sub)

            stop_flag = threading.Event()
            b_a = os.fsencode(str(real_sub))
            b_b = os.fsencode(str(sym_sub))

            def flipper() -> None:
                while not stop_flag.is_set():
                    libc.syscall(sys_renameat2, at_fdcwd, b_a, at_fdcwd, b_b, rename_exchange)

            fl = threading.Thread(target=flipper)
            fl.start()
            try:
                for i in range(550):
                    try:
                        audit.write_regular_bytes_safe(real_sub / "x.bin", f"{i}\n".encode("ascii"))
                    except oracle.HammingLatticeError:
                        pass
            finally:
                stop_flag.set()
                fl.join()
            self.assertEqual(list(victim_dir.iterdir()), [])

            hl_dir = root / "hl_race"
            hl_dir.mkdir()
            hl_target = hl_dir / "target.bin"
            hl_victim = hl_dir / "victim.bin"
            hl_victim.write_bytes(b"ORIGINAL_VICTIM\n")

            stop_hl = threading.Event()
            def hl_attacker() -> None:
                while not stop_hl.is_set():
                    try:
                        os.unlink(hl_target)
                    except OSError:
                        pass
                    try:
                        os.link(hl_victim, hl_target)
                    except OSError:
                        pass

            ha = threading.Thread(target=hl_attacker)
            ha.start()
            try:
                for i in range(550):
                    try:
                        audit.write_regular_bytes_safe(hl_target, f"safe-{i}\n".encode("ascii"))
                    except oracle.HammingLatticeError:
                        pass
            finally:
                stop_hl.set()
                ha.join()
            self.assertEqual(hl_victim.read_bytes(), b"ORIGINAL_VICTIM\n")


def run_deterministic_suite() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestHammingConstructionAR5)
    buf = io.StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=0).run(suite)
    if not result.wasSuccessful():
        sys.stderr.write(buf.getvalue())
        return 1
    sys.stdout.write(f"PASS TRACK-S47 R5_TESTS={result.testsRun}/{result.testsRun}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_deterministic_suite())
