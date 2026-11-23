#!/usr/bin/env python3
"""Track S30 (Portal 26): R5 Pure-Q Adversarial, Algebraic, Mutation & Zone-B Test Suite.

Pure Python 3 standard library: zero floats, zero bare asserts, zero RNG.
Covers all 5 exact-Q pillars grounded in pinned M1 Max siliquant Gemma 3 receipts,
every exported primitive/guard boundary, JSONL provenance tampering, secondary
algebraic audit tampering, and race-free dir_fd / atomic O_EXCL + os.replace / CLI hardening.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import gemma_m1_tomography_audit as audit
import gemma_m1_tomography_oracle as oracle


class _PathSubclass(type(Path("."))):
    pass


def _resign_cert(cert: dict) -> dict:
    out = copy.deepcopy(cert)
    payload_sha = oracle.canonical_sha256(out["payload"])
    out["payload_sha256"] = payload_sha
    env = {
        "schema": out["schema"],
        "track_id": out["track_id"],
        "payload_sha256": payload_sha,
        "payload": out["payload"],
    }
    out["certificate_sha256"] = oracle.canonical_sha256(env)
    return out


class TestFivePillarsCertified(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.build_standalone_certificate()
        cls.payload = cls.cert["payload"]

    def test_top_level_and_determinism(self) -> None:
        cert2 = oracle.build_standalone_certificate()
        self.assertEqual(self.cert["payload_sha256"], cert2["payload_sha256"])
        self.assertEqual(self.cert["certificate_sha256"], cert2["certificate_sha256"])
        self.assertEqual(self.payload["oracle_class"], "exact")
        self.assertEqual(self.payload["novelty_tier"], "N1")
        self.assertIs(self.payload["zero_trust_path_floats"], True)
        self.assertIs(self.payload["zero_rng"], True)
        self.assertIs(self.payload["zero_bare_asserts"], True)
        self.assertEqual(self.payload["pinned_source_commit"], "dc10d4bbca")
        res = oracle.validate_certificate_payload(self.cert, full_replay=True)
        self.assertIs(res["valid"], True)
        sec = audit.secondary_algebraic_audit(self.payload)
        self.assertIs(sec["secondary_audit_passed"], True)
        self.assertEqual(sec["models_checked"], 3)
        self.assertEqual(sec["triples_checked"], 12)
        self.assertEqual(sec["e8_roots_checked"], 240)

    def test_pillar1_provenance_and_prereg_p1_to_p10(self) -> None:
        p1 = self.payload["pillar1_provenance_and_prereg"]
        self.assertEqual(p1["total_rows_validated"], 102)
        for m in oracle.MODELS:
            self.assertEqual(p1["pinned_jsonl_sha256"][f"{m}.jsonl"], oracle.PINNED_FILE_SHA256[f"{m}.jsonl"])

        self.assertEqual(p1["p1_hzbg_vs_q4km_imat"]["gemma-3-270m-it"]["verdict_wiki"], "HOLD")
        self.assertEqual(p1["p1_hzbg_vs_q4km_imat"]["gemma-3-1b-it"]["verdict_wiki"], "FAIL")
        self.assertEqual(p1["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["verdict_wiki"], "HOLD")
        self.assertEqual(p1["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["verdict_python"], "HOLD")
        self.assertEqual(p1["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["wikitext"]["z_signed_2dp"], "+3.78")
        self.assertEqual(p1["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["python"]["z_signed_2dp"], "+2.31")

        self.assertEqual(p1["p2_plain_zbg_vs_q4km"]["gemma-3-270m-it"]["verdict_wiki"], "HOLD")
        self.assertEqual(p1["p2_plain_zbg_vs_q4km"]["gemma-3-1b-it"]["verdict_wiki"], "HOLD")
        self.assertEqual(p1["p2_plain_zbg_vs_q4km"]["gemma-3-4b-it"]["verdict_wiki"], "FAIL_REVERSED")
        self.assertEqual(p1["p2_plain_zbg_vs_q4km"]["gemma-3-4b-it"]["wikitext_q4km_beats_zbg"]["z_signed_2dp"], "-6.11")

        for m in oracle.MODELS:
            self.assertEqual(p1["p3_hzbg_vs_zbg"][m]["verdict"], "HOLD")
            self.assertEqual(p1["p4_hzbg_k10_vs_hzbg"][m]["verdict"], "FAIL")
            self.assertEqual(p1["p5_rand_retention"][m]["verdict_wiki"], "HOLD")
            self.assertEqual(p1["p6_zbg_k10_vs_zbg"][m]["verdict"], "FAIL")
            self.assertEqual(p1["p8_5bit_improvement_and_rank1"][m]["verdict"], "HOLD")
            self.assertEqual(p1["p9_4bit_rise_and_beats_q4_0"][m]["verdict"], "HOLD")

        self.assertIs(p1["p5_wiki_monotone_scale_decay"], True)
        self.assertIs(p1["p5_4b_python_collapse_below_10pct"], True)
        self.assertEqual(p1["p5_rand_retention"]["gemma-3-4b-it"]["python"]["retention_pct_2dp"], "8.46")
        self.assertEqual(p1["p7_e4b_bridge_status"], "NOT_APPLICABLE_GEMMA3")
        self.assertIs(p1["p10_spearman_holds_all_6"], True)
        self.assertEqual(p1["p10_inversion_le_0_2_passes_of_6"], 1)
        self.assertIs(p1["p10_top1_ranking"]["gemma-3-270m-it"]["wikitext"]["p10_full_hold"], True)
        self.assertEqual(p1["p10_top1_ranking"]["gemma-3-270m-it"]["wikitext"]["spearman_rho_exact"], "135/136")
        self.assertEqual(p1["p10_top1_ranking"]["gemma-3-4b-it"]["wikitext"]["spearman_rho_exact"], "203/204")
        # S30-VER-02: tied 4B python uses exact Pearson-of-midranks rho^2 = 49923/55352 -> 4dp "0.9496"!
        p10_4b_py = p1["p10_top1_ranking"]["gemma-3-4b-it"]["python"]
        self.assertEqual(p10_4b_py["spearman_rho_sq"], "49923/55352")
        self.assertEqual(p10_4b_py["spearman_rho_exact"], "IRRATIONAL_TIED_SQRT")
        self.assertEqual(p10_4b_py["spearman_rho_4dp"], "0.9496")
        self.assertIs(p10_4b_py["spearman_has_ties"], True)
        # S30-DS-05: 4B P4 Wiki zero-truncated z_signed_2dp never emits "-0.00"
        self.assertEqual(p1["p4_hzbg_k10_vs_hzbg"]["gemma-3-4b-it"]["wikitext"]["z_signed_2dp"], "0.00")
        self.assertIs(p1["bf16_4b_token_embd_asymmetry_certified"], True)

    def test_pillar2_rate_distortion_and_jensen_bracket(self) -> None:
        p2 = self.payload["pillar2_rate_distortion_convexity"]
        self.assertEqual(p2["triples_count"], 12)
        self.assertIs(p2["all_12_strictly_convex"], True)
        self.assertIs(p2["all_12_geo_beats_linear"], True)
        self.assertIs(p2["all_6_wiki_geo_within_2_5_pct"], True)
        self.assertIs(p2["all_3_zbg_slopes_match_6_02"], True)
        self.assertEqual(p2["shannon_20_log10_2_lo_4dp"], "6.0205")
        self.assertEqual(p2["shannon_20_log10_2_hi_4dp"], "6.0206")

        iso_w = p2["gemma3_4b_iso_kl_bracket"]["wikitext"]
        iso_p = p2["gemma3_4b_iso_kl_bracket"]["python"]
        self.assertEqual(iso_w["b_lin_4dp"], "4.3644")
        self.assertEqual(iso_w["b_log_4dp"], "4.3314")
        self.assertEqual(iso_w["savings_log_lo_pct_2dp"], "10.04")
        self.assertIs(iso_w["b_log_width_lt_1e_12"], True)
        self.assertIs(iso_w["jensen_strict_gap_b_log_lt_b_lin"], True)
        self.assertEqual(iso_p["b_lin_4dp"], "4.3789")
        self.assertEqual(iso_p["b_log_4dp"], "4.3335")
        self.assertEqual(iso_p["savings_log_lo_pct_2dp"], "10.00")
        self.assertIs(iso_p["jensen_strict_gap_b_log_lt_b_lin"], True)

    def test_pillar3_q4km_fallback_decoder(self) -> None:
        p3 = self.payload["pillar3_q4km_fallback_decoder"]
        self.assertIs(p3["fallback_270m_1b_exact"], True)
        self.assertIs(p3["normal_4b_zero_fallback_exact"], True)
        self.assertIs(p3["q4_0_negative_control_exact"], True)
        self.assertEqual(p3["decoded_models"]["gemma-3-270m-it"]["file:Q4_K_M"]["q8_0_kind_count"], 1)
        self.assertEqual(p3["decoded_models"]["gemma-3-270m-it"]["file:Q4_K_M"]["q4_k_kind_count"], 1)
        self.assertEqual(p3["decoded_models"]["gemma-3-1b-it"]["file:Q4_K_M"]["q5_kind_count"], 4)
        self.assertEqual(p3["decoded_models"]["gemma-3-4b-it"]["file:Q4_K_M"]["q4_k_kind_count"], 5)
        self.assertEqual(p3["decoded_models"]["gemma-3-4b-it"]["file:Q4_K_M"]["q6_k_kind_count"], 2)

    def test_pillar4_e8_h8_anisotropy_and_pred_m1_01(self) -> None:
        p4 = self.payload["pillar4_e8_h8_anisotropy"]
        emp = p4["empirical_paradox"]
        self.assertIs(emp["mse_identical_within_0_002_db"], True)
        self.assertIs(emp["e8_packing_gain_ge_0_65_db_all_21"], True)
        self.assertIs(emp["hbg_lower_ppl_wiki_all_3_models"], True)
        self.assertIs(emp["kl_flips_270m_hbg_to_4b_ebg"], True)

        op = p4["exact_e8_h8_operator_theorem"]
        self.assertEqual(op["e8_minimal_roots_count"], 240)
        self.assertIs(op["h8_symmetric_and_sq_8I"], True)
        self.assertIs(op["e8_all_240_roots_norm2_and_fixed"], True)
        self.assertIs(op["qh_not_automorphism_of_e8_proved"], True)
        self.assertIs(op["generic_isotropic_trace_invariance_exact"], True)
        self.assertEqual(op["witness_tr_sigma_e"], "1/4")
        self.assertEqual(op["witness_tr_sigma_e_h"], "1/4")
        self.assertIs(op["witness_identical_l2_error"], True)
        self.assertEqual(op["h_act_1_axis_kl_e8"], "1/4")
        self.assertEqual(op["h_act_1_axis_kl_h8"], "1/32")
        self.assertEqual(op["h_act_2_dc_kl_e8"], "1/32")
        self.assertEqual(op["h_act_2_dc_kl_h8"], "1/4")
        self.assertIs(op["exact_8x_opposite_sign_kl_flip"], True)
        self.assertEqual(op["isometric_qh_e1_d2_e8"], "1")
        self.assertEqual(op["isometric_qh_e1_d2_qh_qsqrt2"], ["3", "-2"])
        self.assertIs(op["isometric_qh_e1_unique_nearest_and_lt_1_5"], True)
        self.assertEqual(op["isometric_qh_root_d2_e8"], "0")
        self.assertEqual(op["isometric_qh_root_d2_qh_qsqrt2"], ["6", "-4"])
        self.assertIs(op["isometric_qh_root_unique_nearest_and_gt_1_3"], True)
        self.assertIs(op["isometric_qh_single_vector_opposite_flip"], True)

        pred = p4["pred_m1_01_hebg128_steelman"]
        self.assertEqual(pred["pred_4b_wiki_kl_theo_lo_6dp"], "0.061685")
        self.assertEqual(pred["pred_4b_py_kl_theo_lo_6dp"], "0.026777")
        self.assertEqual(pred["pred_4b_wiki_kl_emp_transfer_6dp"], "0.059990")
        self.assertIs(pred["pred_savings_strictly_above_12pct"], True)

    def test_pillar5_run_py_ast_audit(self) -> None:
        p5 = self.payload["pillar5_run_py_ast_audit"]
        self.assertEqual(p5["run_py_sha256"], oracle.PINNED_FILE_SHA256["run.py"])
        self.assertIs(p5["weighted_ball_re_supports_all_6"], True)
        self.assertEqual(p5["run_bug_01_ast_extracted_prefixes"], ["hZBG"])
        self.assertEqual(p5["run_bug_01_hardcoded_hzbg_lines"], [967, 1035])
        self.assertIs(p5["run_bug_01_detected_exact_lines"], True)
        self.assertEqual(p5["run_bug_01_missed_count"], 5)
        self.assertIs(p5["run_bug_01_fixed_predicate_verified"], True)
        self.assertEqual(p5["run_bug_02_hardcoded_f16_lines"], [733])
        self.assertIs(p5["run_bug_02_detected_exact_line"], True)
        self.assertIs(p5["run_bug_02_fixed_selector_verified"], True)


class TestOraclePrimitivesAndPinnedValidation(unittest.TestCase):
    def _assert_err(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.GemmaM1Error) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_scalar_rational_decimal_and_json_guards(self) -> None:
        self.assertEqual(oracle.require_exact_int(0, "x", min_val=0, max_val=0), 0)
        self.assertEqual(oracle.require_exact_int(10 ** 1000 - 1, "x"), 10 ** 1000 - 1)
        self._assert_err("INVALID_INT", oracle.require_exact_int, True, "x")
        self._assert_err("INT_DIGITS_EXCEEDED", oracle.require_exact_int, 10 ** 1000, "x")
        self._assert_err("INT_DIGITS_EXCEEDED", oracle.require_exact_int, 1 << 20000, "x")
        self._assert_err("INT_TOO_SMALL", oracle.require_exact_int, 4, "x", min_val=5)
        self._assert_err("INT_TOO_LARGE", oracle.require_exact_int, 6, "x", max_val=5)

        self.assertIs(oracle.require_exact_bool(True, "b"), True)
        self._assert_err("INVALID_BOOL", oracle.require_exact_bool, 1, "b")
        self.assertEqual(oracle.require_exact_str("", "s", non_empty=False), "")
        self._assert_err("INVALID_STR", oracle.require_exact_str, b"a", "s")
        self._assert_err("EMPTY_STR", oracle.require_exact_str, "", "s")
        self._assert_err("INVALID_UTF8_SURROGATE", oracle.require_exact_str, "a\ud800", "s")
        self._assert_err("INVALID_FRACTION", oracle.require_exact_fraction, 1, "q")

        good_sha = "0123456789abcdef" * 4
        self.assertEqual(oracle.require_sha256_hex(good_sha, "h"), good_sha)
        self._assert_err("INVALID_SHA256", oracle.require_sha256_hex, good_sha[:-1], "h")
        self._assert_err("INVALID_SHA256", oracle.require_sha256_hex, "A" * 64, "h")

        self.assertEqual(oracle.format_exact_rational(Fraction(3, 1)), "3")
        self.assertEqual(oracle.format_exact_rational(Fraction(-3, 4)), "-3/4")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-5, 2), 0), "-2")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-5, 2), 2), "-2.50")
        self.assertEqual(oracle.parse_exact_rational("0"), Fraction(0, 1))
        self.assertEqual(oracle.parse_exact_rational("-7/3"), Fraction(-7, 3))
        self.assertEqual(oracle.parse_exact_rational("9" * 1000), Fraction(int("9" * 1000), 1))
        self.assertEqual(oracle.parse_exact_rational("1" + "0" * 999 + "/" + "9" + "0" * 998 + "1"), Fraction(10 ** 999, 9 * 10 ** 999 + 1))
        for bad_q in ("1/2/3", "a/2", "1/b", "/2", "1/", "-/2", ":/1", "1/:", "²", "²/2", "1/²", "①", "1/①", "١٢/٣", "１２/３", "3/-2"):
            self._assert_err("INVALID_RATIONAL", oracle.parse_exact_rational, bad_q)
        for noncan_q in ("0/9", "2/4", "1/1", "-0", "-0/3", "01", "01/2", "1/02"):
            self._assert_err("NON_CANONICAL_RATIONAL", oracle.parse_exact_rational, noncan_q)
        self._assert_err("INT_DIGITS_EXCEEDED", oracle.parse_exact_rational, "9" * 1001)
        self._assert_err("INT_DIGITS_EXCEEDED", oracle.parse_exact_rational, "9" * 1001 + "/1")
        self._assert_err("INT_DIGITS_EXCEEDED", oracle.parse_exact_rational, "1/" + "9" * 1001)
        self._assert_err("ZERO_DENOMINATOR", oracle.parse_exact_rational, "1/0")

        # parse_exact_decimal_token exact boundaries
        self.assertEqual(oracle.parse_exact_decimal_token("0"), Fraction(0, 1))
        self.assertEqual(oracle.parse_exact_decimal_token("-12.34"), Fraction(-1234, 100))
        self.assertEqual(oracle.parse_exact_decimal_token("1.5e+2"), Fraction(150, 1))
        self.assertEqual(oracle.parse_exact_decimal_token("1.5E-2"), Fraction(15, 1000))
        self.assertEqual(oracle.parse_exact_decimal_token("1e00001"), Fraction(10, 1))
        self.assertEqual(oracle.parse_exact_decimal_token("1e512"), Fraction(10 ** 512, 1))
        self.assertEqual(oracle.parse_exact_decimal_token("1e-512"), Fraction(1, 10 ** 512))
        self.assertEqual(oracle.parse_exact_decimal_token("9" * 256), Fraction(int("9" * 256), 1))
        for bad_d in ("1e2e3", "1e", "1e+", "1e²", "1e123456", "1.2.3", ".5", "5.", "1.²", "².1", "1a", "²"):
            self._assert_err("INVALID_DECIMAL", oracle.parse_exact_decimal_token, bad_d)
        self._assert_err("DECIMAL_EXPONENT_EXCEEDED", oracle.parse_exact_decimal_token, "1e513")
        self._assert_err("DECIMAL_EXPONENT_EXCEEDED", oracle.parse_exact_decimal_token, "1e-513")
        self._assert_err("DECIMAL_MANTISSA_EXCEEDED", oracle.parse_exact_decimal_token, "9" * 257)

        # JSON tree & strict JSON parser
        oracle.validate_json_tree_iterative({"": ["", True, False, None, 0]}, max_depth=2)
        oracle.validate_json_tree_iterative({"a": 1}, max_depth=1)
        self._assert_err("JSON_DEPTH_EXCEEDED", oracle.validate_json_tree_iterative, {"a": {"b": 1}}, max_depth=1)
        self._assert_err("FORBIDDEN_JSON_TYPE", oracle.validate_json_tree_iterative, {"a": (1, 2)})

        pad = oracle.MAX_JSON_BYTES - len('{"a":""}')
        exact_cap = '{"a":"' + ("x" * pad) + '"}'
        self.assertEqual(len(oracle.parse_strict_json_text(exact_cap)["a"]), pad)
        self._assert_err("JSON_TOO_LARGE", oracle.parse_strict_json_text, '{"a":"' + ("x" * (pad + 1)) + '"}')
        self._assert_err("FLOAT_FORBIDDEN", oracle.parse_strict_json_text, '{"a": 1.0}')
        self._assert_err("NONFINITE_JSON_FORBIDDEN", oracle.parse_strict_json_text, '{"a": NaN}')
        self._assert_err("INT_DIGITS_EXCEEDED", oracle.parse_strict_json_text, '{"a": ' + ("9" * 5000) + '}')
        self.assertEqual(oracle._strict_parse_int_token("9" * 1000), int("9" * 1000))
        self.assertEqual(oracle._strict_parse_int_token("0"), 0)
        self.assertEqual(oracle._strict_parse_int_token("-42"), -42)
        self._assert_err("INT_DIGITS_EXCEEDED", oracle._strict_parse_int_token, "9" * 1001)
        self._assert_err("INVALID_STR", oracle._strict_parse_int_token, 123)
        for bad_it in ("١٢", "²", "+5", "01", "-0", ""):
            self._assert_err("INVALID_INT_TOKEN" if bad_it else "EMPTY_STR", oracle._strict_parse_int_token, bad_it)
        self._assert_err("DUPLICATE_JSON_KEY", oracle.parse_strict_json_text, '{"a": 1, "a": 2}')
        self._assert_err("INVALID_JSON", oracle.parse_strict_json_text, "{bad")
        self._assert_err("ROOT_NOT_OBJECT", oracle.parse_strict_json_text, "[1]", require_object=True)
        self.assertEqual(oracle.parse_strict_json_text("[1]", require_object=False), [1])
        with mock.patch.object(oracle.json, "dumps", side_effect=ValueError("boom")):
            self._assert_err("CANONICAL_ENCODE_ERROR", oracle.canonical_json_bytes, {"a": 1})

    def test_math_enclosures_e8_h8_and_patch_helper_guards(self) -> None:
        # ln_bounds_exact & exp_neg_bounds_exact & exp_neg_interval_exact
        lo_half, hi_half = oracle.ln_bounds_exact(Fraction(1, 2), 16)
        lo_two, hi_two = oracle.ln_bounds_exact(Fraction(2, 1), 16)
        self.assertEqual(lo_half, -hi_two)
        self.assertEqual(hi_half, -lo_two)
        self.assertLess(hi_two - lo_two, Fraction(1, 10 ** 12))
        self._assert_err("LOG_DOMAIN_ERROR", oracle.ln_bounds_exact, Fraction(0, 1))
        self._assert_err("LOG_DOMAIN_ERROR", oracle.ln_bounds_exact, Fraction(-1, 2))
        self._assert_err("INVALID_FRACTION", oracle.ln_bounds_exact, 2)

        e0_lo, e0_hi = oracle.exp_neg_bounds_exact(Fraction(0, 1), 4)
        self.assertEqual((e0_lo, e0_hi), (Fraction(1, 1), Fraction(1, 1)))
        e1_lo, e1_hi = oracle.exp_neg_bounds_exact(Fraction(1, 1), 10)
        self.assertLess(e1_lo, e1_hi)
        self._assert_err("EXP_DOMAIN_ERROR", oracle.exp_neg_bounds_exact, Fraction(-1, 10))
        self._assert_err("EXP_DOMAIN_ERROR", oracle.exp_neg_bounds_exact, Fraction(11, 10))
        self._assert_err("EXP_ORDER_ODD", oracle.exp_neg_bounds_exact, Fraction(1, 2), 3)
        eq_lo, eq_hi = oracle.exp_neg_interval_exact(Fraction(1, 2), Fraction(1, 2), 4)
        self.assertLess(eq_lo, eq_hi)
        self._assert_err("INVALID_INTERVAL", oracle.exp_neg_interval_exact, Fraction(1, 2), Fraction(1, 4))

        # Matrix, vec8, E8/D8, classifier, and patch helper guards
        self._assert_err("INVALID_MATRIX", oracle.mat_trace_exact, 123)
        self._assert_err("INVALID_MATRIX", oracle.mat_trace_exact, "ab")
        self._assert_err("INVALID_MATRIX", oracle.mat_trace_exact, [])
        self._assert_err("INVALID_MATRIX", oracle.mat_trace_exact, [123])
        self._assert_err("INVALID_MATRIX", oracle.mat_trace_exact, ["a"])
        self._assert_err("INVALID_MATRIX_SHAPE", oracle.mat_trace_exact, [[Fraction(1, 1), Fraction(2, 1)]])
        self._assert_err("INVALID_FRACTION", oracle.mat_trace_exact, [[1]])
        eye2 = [[Fraction(1, 1), Fraction(0, 1)], [Fraction(0, 1), Fraction(1, 1)]]
        self._assert_err("INVALID_MATRIX", oracle.mat_vec8_exact, "ab", [Fraction(0, 1)] * 8)
        self._assert_err("INVALID_MATRIX_DIM", oracle.mat_vec8_exact, eye2, [Fraction(0, 1)] * 8)
        self._assert_err("INVALID_VEC8", oracle.mat_vec8_exact, oracle.hadamard8_unnorm_exact(), 123)
        self._assert_err("INVALID_VEC8", oracle.mat_vec8_exact, oracle.hadamard8_unnorm_exact(), "12345678")
        self._assert_err("INVALID_VEC8", oracle.mat_vec8_exact, oracle.hadamard8_unnorm_exact(), [Fraction(0, 1)] * 4)
        self._assert_err("INVALID_FRACTION", oracle.mat_vec8_exact, oracle.hadamard8_unnorm_exact(), [0] * 8)

        # D8 flip both upward and downward
        self.assertEqual(oracle.nearest_d8_exact([Fraction(3, 5)] + [Fraction(0, 1)] * 7), [Fraction(0, 1)] * 8)
        self.assertEqual(oracle.nearest_d8_exact([Fraction(6, 5)] + [Fraction(0, 1)] * 7), [Fraction(2, 1)] + [Fraction(0, 1)] * 7)

        # classify_llama_quant_kind exact boundary tests
        self.assertEqual(oracle.classify_llama_quant_kind(Fraction(-33, 1)), "Q8_0")
        self.assertEqual(oracle.classify_llama_quant_kind(Fraction(-278, 10)), "Q6_K")
        self.assertEqual(oracle.classify_llama_quant_kind(Fraction(-25, 1)), "Q5_0_or_Q5_K")
        self.assertEqual(oracle.classify_llama_quant_kind(Fraction(-215, 10)), "Q4_K")
        self.assertEqual(oracle.classify_llama_quant_kind(Fraction(-214, 10)), "Q4_0_or_lower")
        self._assert_err("INVALID_FRACTION", oracle.classify_llama_quant_kind, -25)

        # quantize_tensor_type_flags_fixed guards
        self._assert_err("INVALID_TENSOR_NAMES", oracle.quantize_tensor_type_flags_fixed, 123, "BF16")
        self._assert_err("INVALID_TENSOR_NAMES", oracle.quantize_tensor_type_flags_fixed, "token_embd.weight", "BF16")
        self._assert_err("UNSUPPORTED_EMBD_DTYPE", oracle.quantize_tensor_type_flags_fixed, ["token_embd.weight"], "Q4_0")

        # _pair_z_sq zero-gain, negative-gain, non-positive KL/SE guards & Spearman guards
        r_a = {"variant": "a", "kl_mean": Fraction(1, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(80, 1)}
        r_b = {"variant": "b", "kl_mean": Fraction(1, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(80, 1)}
        r_c = {"variant": "c", "kl_mean": Fraction(2, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(70, 1)}
        r_zero_kl = {"variant": "z", "kl_mean": Fraction(0, 1), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(80, 1)}
        r_zero_se = {"variant": "s", "kl_mean": Fraction(1, 10), "kl_mean_se": Fraction(0, 1), "same_top": Fraction(80, 1)}
        self.assertEqual(oracle._pair_z_sq(r_a, r_b)["z_signed_2dp"], "0.00")
        self._assert_err("NON_POSITIVE_KL", oracle._pair_z_sq, r_zero_kl, r_a)
        self._assert_err("NON_POSITIVE_KL", oracle._pair_z_sq, r_a, r_zero_kl)
        self._assert_err("NON_POSITIVE_SE", oracle._pair_z_sq, r_zero_se, r_a)
        self._assert_err("NON_POSITIVE_SE", oracle._pair_z_sq, r_a, r_zero_se)
        sp = oracle._exact_spearman_and_inversions([r_a, r_b, r_c])
        self.assertEqual(sp["spearman_rho_sq"], "1")
        self.assertEqual(sp["spearman_rho_exact"], "1")
        self.assertIs(sp["spearman_has_ties"], True)
        self._assert_err("INSUFFICIENT_ROWS", oracle._exact_spearman_and_inversions, [r_a])
        self._assert_err("INSUFFICIENT_ROWS", oracle._exact_spearman_and_inversions, "bad")
        self._assert_err("DEGENERATE_RANKS", oracle._exact_spearman_and_inversions, [r_a, r_b])
        r_same_top = {"variant": "d", "kl_mean": Fraction(3, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(80, 1)}
        self._assert_err("DEGENERATE_RANKS", oracle._exact_spearman_and_inversions, [r_a, r_same_top])
        r_same_kl = {"variant": "e", "kl_mean": Fraction(1, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(70, 1)}
        self._assert_err("DEGENERATE_RANKS", oracle._exact_spearman_and_inversions, [r_a, r_same_kl])
        # Negative correlation test (covers cov < 0 formatting)
        r_neg1 = {"variant": "n1", "kl_mean": Fraction(1, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(70, 1)}
        r_neg2 = {"variant": "n2", "kl_mean": Fraction(2, 10), "kl_mean_se": Fraction(1, 100), "same_top": Fraction(80, 1)}
        sp_neg = oracle._exact_spearman_and_inversions([r_neg1, r_neg2])
        self.assertEqual(sp_neg["spearman_rho_4dp"], "-1.0000")
        self.assertIs(sp_neg["rho_ge_9_10"], False)

    def test_pinned_reader_jsonl_provenance_and_cert_guards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdir = root / "pinned"
            shutil.copytree(oracle.PINNED_INPUTS_DIR, pdir)

            # _read_pinned_file_bytes guards
            self._assert_err("INVALID_PINNED_DIR", oracle._read_pinned_file_bytes, str(pdir), "run.py")
            self._assert_err("INVALID_PINNED_DIR", oracle._read_pinned_file_bytes, Path("a\x00b"), "run.py")
            self._assert_err("INVALID_PINNED_FILENAME", oracle._read_pinned_file_bytes, pdir, "a/b")
            self._assert_err("INVALID_PINNED_FILENAME", oracle._read_pinned_file_bytes, pdir, "a\x00b")
            self._assert_err("INVALID_PINNED_FILENAME", oracle._read_pinned_file_bytes, pdir, "..")
            self._assert_err("INVALID_STR", oracle._read_pinned_file_bytes, pdir, "run.py", 123)
            self._assert_err("PINNED_READ_ERROR", oracle._read_pinned_file_bytes, root / "missing_dir", "run.py")
            self._assert_err("PINNED_READ_ERROR", oracle._read_pinned_file_bytes, pdir, "missing.txt")
            self._assert_err("PINNED_INPUT_SHA_MISMATCH", oracle._read_pinned_file_bytes, pdir, "run.py", "0" * 64)

            sym_d = root / "sym_d"
            sym_d.symlink_to(pdir)
            self._assert_err("SYMLINK_FORBIDDEN", oracle._read_pinned_file_bytes, sym_d, "run.py")
            sym_f = pdir / "sym_run.py"
            sym_f.symlink_to(pdir / "run.py")
            self._assert_err("SYMLINK_FORBIDDEN", oracle._read_pinned_file_bytes, pdir, "sym_run.py")
            sym_f.unlink()
            fifo_p = pdir / "pipe.fifo"
            os.mkfifo(fifo_p)
            self._assert_err("NON_REGULAR_FILE", oracle._read_pinned_file_bytes, pdir, "pipe.fifo")
            fifo_p.unlink()
            exact_f = pdir / "exact.bin"
            exact_f.write_bytes(b"x" * oracle.MAX_JSON_BYTES)
            self.assertEqual(len(oracle._read_pinned_file_bytes(pdir, "exact.bin")), oracle.MAX_JSON_BYTES)
            exact_f.unlink()
            big_f = pdir / "big.bin"
            big_f.write_bytes(b"x" * (oracle.MAX_JSON_BYTES + 1))
            self._assert_err("FILE_TOO_LARGE", oracle._read_pinned_file_bytes, pdir, "big.bin")
            big_f.unlink()
            with mock.patch.object(oracle.os, "open", side_effect=OSError("anchor fail")):
                self._assert_err("PINNED_READ_ERROR", oracle._read_pinned_file_bytes, pdir, "run.py")

            # Raw JSONL parser & dataset provenance failure modes
            self._assert_err("INVALID_JSONL", oracle._parse_raw_jsonl_line, "{bad")
            self._assert_err("INVALID_JSONL", oracle._parse_raw_jsonl_line, "[1, 2]")
            self._assert_err("DUPLICATE_JSON_KEY", oracle._parse_raw_jsonl_line, '{"a": 1, "a": 2}')
            self._assert_err("NONFINITE_JSON_FORBIDDEN", oracle._parse_raw_jsonl_line, '{"a": NaN}')
            self._assert_err("INVALID_NUMERIC_FIELD", oracle._as_fraction_num, "1.0", "x")
            self._assert_err("NON_POSITIVE_METRIC", oracle._as_fraction_num, Fraction(0, 1), "kl_mean", positive=True)
            self._assert_err("NEGATIVE_METRIC", oracle._as_fraction_num, Fraction(-1, 10), "same_top", non_negative=True)

            # Mutate one JSONL file cleanly in temp dir and test every loader guard
            j270 = pdir / "gemma-3-270m-it.jsonl"
            orig_lines = j270.read_text(encoding="utf-8").splitlines()

            def _run_tampered_jsonl(mut_fn, expected_code: str) -> None:
                lines = list(orig_lines)
                mut_fn(lines)
                j270.write_text("\n".join(lines) + "\n", encoding="utf-8")
                try:
                    self._assert_err(expected_code, oracle.load_pinned_jsonl_dataset, pdir, verify_pin_hashes=False)
                finally:
                    j270.write_text("\n".join(orig_lines) + "\n", encoding="utf-8")

            # Invalid UTF-8
            j270.write_bytes(b"\xff\xfe\n")
            self._assert_err("INVALID_UTF8", oracle.load_pinned_jsonl_dataset, pdir, verify_pin_hashes=False)
            j270.write_text("\n".join(orig_lines) + "\n", encoding="utf-8")

            def _edit_row0(lines, fn):
                obj = json.loads(lines[0])
                fn(obj)
                lines[0] = json.dumps(obj)

            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("model_id", "other")), "MODEL_ID_MISMATCH")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("text", "other")), "UNEXPECTED_ROW_SCOPE")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("variant", "BOGUS")), "UNEXPECTED_ROW_SCOPE")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("bridge", True)), "UNEXPECTED_BRIDGE_ROW")
            _run_tampered_jsonl(lambda ls: ls.append(ls[0]), "DUPLICATE_ROW_KEY")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("other_tensors_that_differ", "bad")), "INVALID_OTHER_TENSORS")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("other_tensors_that_differ", [123])), "INVALID_OTHER_TENSORS")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("model_sha256", "0" * 64)), "PROVENANCE_CONFLICT")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o.__setitem__("weight_err_db_by_kind", [])), "INVALID_BY_KIND")
            _run_tampered_jsonl(lambda ls: _edit_row0(ls, lambda o: o["weight_err_db_by_kind"].pop("attn_q")), "INVALID_BY_KIND")
            _run_tampered_jsonl(lambda ls: ls.pop(), "UNEXPECTED_ROW_COUNT")
            # Tamper ~rand row imatrix_sha256 independently
            rand_idx = next(i for i, ln in enumerate(orig_lines) if '"hZBG128:b560s16~rand"' in ln)
            def _bad_rand(ls):
                obj = json.loads(ls[rand_idx])
                obj["imatrix_sha256"] = "bad"
                ls[rand_idx] = json.dumps(obj)
            _run_tampered_jsonl(_bad_rand, "INVALID_SHA256")
            # Tamper real-imatrix row independently
            imat_idx = next(i for i, ln in enumerate(orig_lines) if '"hZBG128:b560s16"' in ln)
            def _bad_imat(ls):
                obj = json.loads(ls[imat_idx])
                obj["imatrix_sha256"] = "bad"
                ls[imat_idx] = json.dumps(obj)
            _run_tampered_jsonl(_bad_imat, "INVALID_SHA256")
            # Blank line in JSONL skipped cleanly
            j270.write_text("\n" + "\n".join(orig_lines) + "\n", encoding="utf-8")
            rows_ok, _ = oracle.load_pinned_jsonl_dataset(pdir, verify_pin_hashes=False)
            self.assertEqual(len(rows_ok), 102)
            j270.write_text("\n".join(orig_lines) + "\n", encoding="utf-8")

            # Pillar 5 run.py invalid UTF-8, syntax error, and false-positive distractor AST
            run_p = pdir / "run.py"
            run_p.write_bytes(b"\xff\xfe")
            self._assert_err("INVALID_UTF8", oracle.certify_pillar5_run_py_audit, pdir, verify_pin_hashes=False)
            run_p.write_text("def broken(:\n", encoding="utf-8")
            self._assert_err("RUN_PY_SYNTAX_ERROR", oracle.certify_pillar5_run_py_audit, pdir, verify_pin_hashes=False)
            run_p.write_text(
                'import re\nWEIGHTED_BALL_RE = re.compile(r"x")\n'
                'def run_sweep(a):\n'
                '    return a.endswith("hZBG") or a.startswith("hZBG", 0) or a.startswith(a) or a.startswith(123) or a.startswith("") or a.startswith("NOPE") or len("hZBG")\n'
                'def quantize_tensor_type_flags():\n    return ["x"]\n',
                encoding="utf-8",
            )
            syn = oracle.certify_pillar5_run_py_audit(pdir, verify_pin_hashes=False)
            self.assertEqual(syn["run_bug_01_hardcoded_hzbg_lines"], [])
            self.assertEqual(syn["run_bug_01_missed_count"], 7)
            self.assertEqual(syn["run_bug_02_hardcoded_f16_lines"], [])
            # S30-DS-03: Mutate run.py prefix to "h" & "i" -> missed count drops dynamically to 0!
            run_p.write_text(
                'import re\nWEIGHTED_BALL_RE = re.compile(r"x")\n'
                'def run_sweep(a):\n    return a.startswith("h") or a.startswith("i")\n'
                'def quantize_tensor_type_flags():\n    return ["x"]\n',
                encoding="utf-8",
            )
            syn2 = oracle.certify_pillar5_run_py_audit(pdir, verify_pin_hashes=False)
            self.assertEqual(syn2["run_bug_01_ast_extracted_prefixes"], ["h", "i"])
            self.assertEqual(syn2["run_bug_01_missed_count"], 0)

        # validate_certificate_payload guards (incl. top-level payload schema & metadata under full_replay=False!)
        cert = oracle.build_standalone_certificate()
        self._assert_err("INVALID_BOOL", oracle.validate_certificate_payload, cert, full_replay=1)
        self._assert_err("INVALID_CERT_TYPE", oracle.validate_certificate_payload, [])
        self._assert_err("INVALID_CERT_KEYS", oracle.validate_certificate_payload, dict(cert, extra=1))
        self._assert_err("SCHEMA_MISMATCH", oracle.validate_certificate_payload, dict(cert, schema="bad"))
        self._assert_err("TRACK_ID_MISMATCH", oracle.validate_certificate_payload, dict(cert, track_id="bad"))
        self._assert_err("INVALID_PAYLOAD_TYPE", oracle.validate_certificate_payload, dict(cert, payload=[]))
        self._assert_err("INVALID_PAYLOAD_KEYS", oracle.validate_certificate_payload, _resign_cert(dict(cert, payload={})), full_replay=False)
        for bad_meta_k, bad_meta_v in (
            ("oracle_class", "approx"),
            ("novelty_tier", "N0"),
            ("pinned_source_commit", "deadbeef"),
            ("zero_rng", False),
        ):
            m_bad = copy.deepcopy(cert)
            m_bad["payload"][bad_meta_k] = bad_meta_v
            self._assert_err("INVALID_PAYLOAD_META", oracle.validate_certificate_payload, _resign_cert(m_bad), full_replay=False)
        m_pbad = copy.deepcopy(cert)
        m_pbad["payload"]["pillar1_provenance_and_prereg"] = []
        self._assert_err("INVALID_PAYLOAD_PILLAR", oracle.validate_certificate_payload, _resign_cert(m_pbad), full_replay=False)
        self._assert_err("PAYLOAD_SHA_MISMATCH", oracle.validate_certificate_payload, dict(cert, payload_sha256="0" * 64))
        self._assert_err("CERT_SHA_MISMATCH", oracle.validate_certificate_payload, dict(cert, certificate_sha256="0" * 64))
        mod = copy.deepcopy(cert)
        mod["payload"]["pillar1_provenance_and_prereg"]["p7_e4b_bridge_status"] = "OTHER"
        resigned = _resign_cert(mod)
        self.assertIs(oracle.validate_certificate_payload(resigned, full_replay=False)["valid"], True)
        self._assert_err("REPLAY_MISMATCH", oracle.validate_certificate_payload, resigned, full_replay=True)


class TestSecondaryAlgebraicAuditTampering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_payload = oracle.build_standalone_certificate()["payload"]

    def _tamper(self, expected_code: str, mutator) -> None:
        p = copy.deepcopy(self.base_payload)
        mutator(p)
        with self.assertRaises(oracle.GemmaM1Error) as ctx:
            audit.secondary_algebraic_audit(p)
        self.assertEqual(ctx.exception.code, expected_code)

    def test_all_five_pillar_tamper_paths(self) -> None:
        # Schema-safe fail-closed checks (no raw KeyError/TypeError on hollow payloads!)
        with self.assertRaises(oracle.GemmaM1Error) as ctx_t:
            audit.secondary_algebraic_audit("not_dict")
        self.assertEqual(ctx_t.exception.code, "AUDIT_PAYLOAD_TYPE")
        with self.assertRaises(oracle.GemmaM1Error) as ctx_e:
            audit.secondary_algebraic_audit({})
        self.assertEqual(ctx_e.exception.code, "AUDIT_P1_SCHEMA")
        self._tamper("AUDIT_P1_SCHEMA", lambda p: p.__setitem__("pillar1_provenance_and_prereg", []))

        # Pillar 1 (incl. S30-VER-04 z_sq tamper & P10 rho^2 tamper!)
        self._tamper("AUDIT_P1_ROWS", lambda p: p["pillar1_provenance_and_prereg"].__setitem__("total_rows_validated", 101))
        self._tamper("INVALID_INT", lambda p: p["pillar1_provenance_and_prereg"].__setitem__("p10_inversion_le_0_2_passes_of_6", True))
        self._tamper("AUDIT_P1_SHA", lambda p: p["pillar1_provenance_and_prereg"]["pinned_jsonl_sha256"].__setitem__("gemma-3-4b-it.jsonl", "0" * 64))
        self._tamper("AUDIT_P1_4B_WIKI", lambda p: p["pillar1_provenance_and_prereg"]["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"].__setitem__("verdict_wiki", "FAIL"))
        self._tamper("AUDIT_P1_4B_WIKI", lambda p: p["pillar1_provenance_and_prereg"]["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["wikitext"].__setitem__("z_sq", "1/1000"))
        self._tamper("AUDIT_P1_4B_WIKI", lambda p: p["pillar1_provenance_and_prereg"]["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["wikitext"].__setitem__("favors_cand_ge_2se", False))
        self._tamper("AUDIT_P1_4B_WIKI", lambda p: p["pillar1_provenance_and_prereg"]["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["wikitext"].__setitem__("favors_base_ge_2se", True))
        def _flip_p1_to_base_win(p):
            w = p["pillar1_provenance_and_prereg"]["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"]["wikitext"]
            w["gain_kl"] = "-1"
            w["z_sq"] = "4"
            w["favors_cand_ge_2se"] = False
            w["favors_base_ge_2se"] = True
        self._tamper("AUDIT_P1_4B_WIKI", _flip_p1_to_base_win)
        self._tamper("AUDIT_P1_4B_PY", lambda p: p["pillar1_provenance_and_prereg"]["p1_hzbg_vs_q4km_imat"]["gemma-3-4b-it"].__setitem__("verdict_python", "FAIL"))
        self._tamper("AUDIT_P1_P2_REV", lambda p: p["pillar1_provenance_and_prereg"]["p2_plain_zbg_vs_q4km"]["gemma-3-4b-it"].__setitem__("verdict_wiki", "HOLD"))
        def _flip_p2_no_win(p):
            w = p["pillar1_provenance_and_prereg"]["p2_plain_zbg_vs_q4km"]["gemma-3-4b-it"]["wikitext_q4km_beats_zbg"]
            w["z_sq"] = "1"
            w["favors_base_ge_2se"] = False
        self._tamper("AUDIT_P1_P2_REV", _flip_p2_no_win)
        self._tamper("AUDIT_P1_P3", lambda p: p["pillar1_provenance_and_prereg"]["p3_hzbg_vs_zbg"]["gemma-3-4b-it"].__setitem__("verdict", "FAIL"))
        self._tamper("AUDIT_P1_P4", lambda p: p["pillar1_provenance_and_prereg"]["p4_hzbg_k10_vs_hzbg"]["gemma-3-4b-it"].__setitem__("verdict", "HOLD"))
        def _flip_p4_cand_win(p):
            w = p["pillar1_provenance_and_prereg"]["p4_hzbg_k10_vs_hzbg"]["gemma-3-4b-it"]["wikitext"]
            w["gain_kl"] = "1"
            w["z_sq"] = "4"
            w["favors_cand_ge_2se"] = True
            w["favors_base_ge_2se"] = False
        self._tamper("AUDIT_P1_P4", _flip_p4_cand_win)
        self._tamper("AUDIT_P1_P6", lambda p: p["pillar1_provenance_and_prereg"]["p6_zbg_k10_vs_zbg"]["gemma-3-4b-it"].__setitem__("verdict", "HOLD"))
        self._tamper("AUDIT_P1_P8", lambda p: p["pillar1_provenance_and_prereg"]["p8_5bit_improvement_and_rank1"]["gemma-3-4b-it"].__setitem__("verdict", "FAIL"))
        self._tamper("AUDIT_P1_P9", lambda p: p["pillar1_provenance_and_prereg"]["p9_4bit_rise_and_beats_q4_0"]["gemma-3-4b-it"].__setitem__("verdict", "FAIL"))
        for k in ("p5_wiki_monotone_scale_decay", "p5_4b_python_collapse_below_10pct", "p10_spearman_holds_all_6", "bf16_4b_token_embd_asymmetry_certified"):
            self._tamper("AUDIT_P1_FLAG", lambda p, key=k: p["pillar1_provenance_and_prereg"].__setitem__(key, False))
        self._tamper("AUDIT_P1_P10_INV", lambda p: p["pillar1_provenance_and_prereg"].__setitem__("p10_inversion_le_0_2_passes_of_6", 0))
        self._tamper("AUDIT_P1_P10_RHO", lambda p: p["pillar1_provenance_and_prereg"]["p10_top1_ranking"]["gemma-3-4b-it"]["python"].__setitem__("spearman_rho_sq", "4/5"))
        self._tamper("AUDIT_P1_P10_INV", lambda p: p["pillar1_provenance_and_prereg"]["p10_top1_ranking"]["gemma-3-270m-it"]["wikitext"].__setitem__("max_inversion_pp", "1"))
        # Test ALL 3 sub-conditions of r_w[0] <= r_w[1] or r_w[1] <= r_w[2] or r_w[2] < 7/10 + Unicode crash guard!
        self._tamper("INVALID_RATIONAL", lambda p: p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-270m-it"]["wikitext"].__setitem__("retention_ratio", "²"))
        self._tamper("AUDIT_P1_P5_DECAY", lambda p: p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-270m-it"]["wikitext"].__setitem__("retention_ratio", "4/5"))
        self._tamper("AUDIT_P1_P5_DECAY", lambda p: p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-1b-it"]["wikitext"].__setitem__("retention_ratio", "3/4"))
        self._tamper("AUDIT_P1_P5_DECAY", lambda p: p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-4b-it"]["wikitext"].__setitem__("retention_ratio", "3/5"))
        # Boundary equality checks for <= vs <
        self._tamper("AUDIT_P1_P5_DECAY", lambda p: p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-270m-it"]["wikitext"].__setitem__("retention_ratio", p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-1b-it"]["wikitext"]["retention_ratio"]))
        self._tamper("AUDIT_P1_P5_DECAY", lambda p: p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-1b-it"]["wikitext"].__setitem__("retention_ratio", p["pillar1_provenance_and_prereg"]["p5_rand_retention"]["gemma-3-4b-it"]["wikitext"]["retention_ratio"]))

        # Pillar 2
        self._tamper("AUDIT_P2_COUNT", lambda p: p["pillar2_rate_distortion_convexity"].__setitem__("triples_count", 11))
        self._tamper("AUDIT_P2_SCHEMA", lambda p: p["pillar2_rate_distortion_convexity"].__setitem__("triples", {}))
        self._tamper("AUDIT_P2_SCHEMA", lambda p: p["pillar2_rate_distortion_convexity"].pop("triples"))
        self._tamper("AUDIT_P2_COUNT", lambda p: p["pillar2_rate_distortion_convexity"]["triples"].pop())
        self._tamper("AUDIT_P2_SCHEMA", lambda p: p["pillar2_rate_distortion_convexity"]["triples"].__setitem__(0, "bad"))
        self._tamper("AUDIT_P2_TRIPLE", lambda p: p["pillar2_rate_distortion_convexity"]["triples"][0].__setitem__("strictly_convex", False))
        for k in ("all_12_strictly_convex", "all_12_geo_beats_linear", "all_6_wiki_geo_within_2_5_pct", "all_3_zbg_slopes_match_6_02"):
            self._tamper("AUDIT_P2_FLAG", lambda p, key=k: p["pillar2_rate_distortion_convexity"].__setitem__(key, False))
        # Test BOTH sub-conditions of b_lo > b_hi or b_hi >= b_lin + exact width >= 1e-12!
        self._tamper("AUDIT_P2_JENSEN", lambda p: p["pillar2_rate_distortion_convexity"]["gemma3_4b_iso_kl_bracket"]["wikitext"].__setitem__("b_log_lo", "87/20"))
        self._tamper("AUDIT_P2_JENSEN", lambda p: p["pillar2_rate_distortion_convexity"]["gemma3_4b_iso_kl_bracket"]["wikitext"].__setitem__("b_log_hi", p["pillar2_rate_distortion_convexity"]["gemma3_4b_iso_kl_bracket"]["wikitext"]["b_lin_upper_bound"]))
        def _wide_bracket(p):
            iso = p["pillar2_rate_distortion_convexity"]["gemma3_4b_iso_kl_bracket"]["wikitext"]
            iso["b_log_lo"] = "433/100"
            iso["b_log_hi"] = "217/50"
        self._tamper("AUDIT_P2_WIDTH", _wide_bracket)
        self._tamper("AUDIT_P2_WIDTH", lambda p: p["pillar2_rate_distortion_convexity"]["gemma3_4b_iso_kl_bracket"]["wikitext"].__setitem__("b_log_width_lt_1e_12", False))

        # Exact inclusive boundary checks in secondary_algebraic_audit (kills < vs <= and > vs >= mutants!)
        p_bnd = copy.deepcopy(self.base_payload)
        p_bnd["pillar1_provenance_and_prereg"]["p10_top1_ranking"]["gemma-3-4b-it"]["python"]["spearman_rho_sq"] = "81/100"
        p_bnd["pillar4_e8_h8_anisotropy"]["empirical_paradox"]["max_abs_ebg_vs_hbg_db_across_21_pairs"] = "1/500"
        p_bnd["pillar4_e8_h8_anisotropy"]["empirical_paradox"]["min_zbg_minus_ebg_db_across_21_pairs"] = "13/20"
        self.assertIs(audit.secondary_algebraic_audit(p_bnd)["secondary_audit_passed"], True)
        def _exact_1e12_bracket(p):
            iso = p["pillar2_rate_distortion_convexity"]["gemma3_4b_iso_kl_bracket"]["wikitext"]
            iso["b_log_lo"] = "433/100"
            iso["b_log_hi"] = "4330000000001/1000000000000"
        self._tamper("AUDIT_P2_WIDTH", _exact_1e12_bracket)

        # Pillar 3 (incl. S30-VER-04 per-kind fallback count tamper!)
        for k in ("fallback_270m_1b_exact", "normal_4b_zero_fallback_exact", "q4_0_negative_control_exact"):
            self._tamper("AUDIT_P3_FLAG", lambda p, key=k: p["pillar3_q4km_fallback_decoder"].__setitem__(key, False))
        self._tamper("AUDIT_P3_COUNTS", lambda p: p["pillar3_q4km_fallback_decoder"]["decoded_models"]["gemma-3-270m-it"]["file:Q4_K_M"].__setitem__("q4_k_kind_count", 99))
        self._tamper("AUDIT_P3_COUNTS", lambda p: p["pillar3_q4km_fallback_decoder"]["decoded_models"]["gemma-3-4b-it"]["file:Q4_K_M"].__setitem__("q6_k_kind_count", 0))

        # Pillar 4
        for k in ("mse_identical_within_0_002_db", "e8_packing_gain_ge_0_65_db_all_21", "hbg_lower_ppl_wiki_all_3_models", "kl_flips_270m_hbg_to_4b_ebg"):
            self._tamper("AUDIT_P4_EMP", lambda p, key=k: p["pillar4_e8_h8_anisotropy"]["empirical_paradox"].__setitem__(key, False))
        self._tamper("AUDIT_P4_DIFF", lambda p: p["pillar4_e8_h8_anisotropy"]["empirical_paradox"].__setitem__("max_abs_ebg_vs_hbg_db_across_21_pairs", "1/100"))
        self._tamper("AUDIT_P4_GAIN", lambda p: p["pillar4_e8_h8_anisotropy"]["empirical_paradox"].__setitem__("min_zbg_minus_ebg_db_across_21_pairs", "1/2"))
        self._tamper("AUDIT_P4_ROOTS", lambda p: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__("e8_minimal_roots_count", 239))
        for k in (
            "h8_symmetric_and_sq_8I",
            "e8_all_240_roots_norm2_and_fixed",
            "qh_not_automorphism_of_e8_proved",
            "generic_isotropic_trace_invariance_exact",
            "witness_identical_l2_error",
            "exact_8x_opposite_sign_kl_flip",
            "isometric_qh_e1_unique_nearest_and_lt_1_5",
            "isometric_qh_root_unique_nearest_and_gt_1_3",
            "isometric_qh_single_vector_opposite_flip",
        ):
            self._tamper("AUDIT_P4_OP", lambda p, key=k: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__(key, False))
        self._tamper("AUDIT_P4_TRACE", lambda p: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__("witness_tr_sigma_e_h", "1/2"))
        def _zero_traces(p):
            op = p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"]
            op["witness_tr_sigma_e"] = "0"
            op["witness_tr_sigma_e_h"] = "0"
        self._tamper("AUDIT_P4_TRACE", _zero_traces)
        self._tamper("AUDIT_P4_FLIP_RATIO", lambda p: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__("h_act_1_axis_kl_e8", "1/2"))
        self._tamper("AUDIT_P4_FLIP_RATIO", lambda p: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__("h_act_2_dc_kl_h8", "1/2"))
        self._tamper("AUDIT_P4_QSQRT2", lambda p: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__("isometric_qh_e1_d2_qh_qsqrt2", ["1", "0"]))
        self._tamper("AUDIT_P4_QSQRT2", lambda p: p["pillar4_e8_h8_anisotropy"]["exact_e8_h8_operator_theorem"].__setitem__("isometric_qh_root_d2_qh_qsqrt2", ["1", "0"]))
        self._tamper("AUDIT_P4_PRED", lambda p: p["pillar4_e8_h8_anisotropy"]["pred_m1_01_hebg128_steelman"].__setitem__("pred_savings_strictly_above_12pct", False))

        # Pillar 5
        self._tamper("AUDIT_P5_SHA", lambda p: p["pillar5_run_py_ast_audit"].__setitem__("run_py_sha256", "0" * 64))
        self._tamper("AUDIT_P5_LINES", lambda p: p["pillar5_run_py_ast_audit"].__setitem__("run_bug_01_hardcoded_hzbg_lines", [967]))
        self._tamper("AUDIT_P5_LINES", lambda p: p["pillar5_run_py_ast_audit"].__setitem__("run_bug_02_hardcoded_f16_lines", [734]))
        self._tamper("AUDIT_P5_MISSED", lambda p: p["pillar5_run_py_ast_audit"].__setitem__("run_bug_01_missed_count", 4))
        self._tamper("AUDIT_P5_MISSED", lambda p: p["pillar5_run_py_ast_audit"]["run_bug_01_arms_missed_by_buggy"].pop())
        for k in (
            "weighted_ball_re_supports_all_6",
            "run_bug_01_detected_exact_lines",
            "run_bug_01_fixed_predicate_verified",
            "run_bug_02_detected_exact_line",
            "run_bug_02_fixed_selector_verified",
        ):
            self._tamper("AUDIT_P5_FLAG", lambda p, key=k: p["pillar5_run_py_ast_audit"].__setitem__(key, False))


class TestZoneBHardeningAndCLI(unittest.TestCase):
    def _assert_err(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.GemmaM1Error) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_dir_fd_path_symlink_hardlink_fifo_and_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._assert_err("INVALID_PATH_TYPE", audit.require_safe_path, str(root / "a.json"))
            self._assert_err("INVALID_PATH_TYPE", audit.require_safe_path, _PathSubclass(str(root / "a.json")))
            self._assert_err("INVALID_PATH", audit.require_safe_path, Path("bad\x00path"))
            self._assert_err("INVALID_UTF8_SURROGATE", audit.require_safe_path, Path("bad\ud800.json"))
            self.assertEqual(audit.require_safe_path(Path("/")), Path("/"))
            self._assert_err("NON_REGULAR_FILE", audit.read_regular_file_bytes, Path("/"))
            self._assert_err("NON_REGULAR_FILE", audit.write_regular_file_bytes, Path("/"), b"x")
            self._assert_err("INVALID_WRITE_DATA", audit.write_regular_file_bytes, root / "x.json", "not_bytes")
            self._assert_err("PATH_STAT_ERROR", audit._open_dir_chain, "/nonexistent_s30_anchor_xyz", [])

            self._assert_err("PATH_NOT_FOUND", audit.require_safe_path, root / "no_dir" / "a.json")
            reg_file = root / "plain.txt"
            reg_file.write_bytes(b"hello")
            self._assert_err("SYMLINK_FORBIDDEN", audit.require_safe_path, reg_file / "child.json")

            real_dir = root / "real"
            real_dir.mkdir()
            sym_dir = root / "sym_dir"
            sym_dir.symlink_to(real_dir)
            self._assert_err("SYMLINK_FORBIDDEN", audit.require_safe_path, sym_dir / "a.json")
            self._assert_err("SYMLINK_FORBIDDEN", audit.require_safe_path, sym_dir / ".." / "plain.txt")
            self._assert_err("SYMLINK_FORBIDDEN", audit.read_regular_file_bytes, sym_dir / "a.json")

            no_exec = root / "no_exec"
            no_exec.mkdir()
            sub_dir = no_exec / "inner"
            sub_dir.mkdir()
            os.chmod(no_exec, 0o000)
            try:
                if os.geteuid() != 0:
                    self._assert_err("PATH_STAT_ERROR", audit.require_safe_path, sub_dir / "a.json")
            finally:
                os.chmod(no_exec, 0o700)

            sym_leaf = root / "sym_leaf.json"
            sym_leaf.symlink_to(reg_file)
            self._assert_err("SYMLINK_FORBIDDEN", audit.require_safe_path, sym_leaf)
            self._assert_err("SYMLINK_FORBIDDEN", audit.read_regular_file_bytes, sym_leaf)
            self._assert_err("SYMLINK_FORBIDDEN", audit.write_regular_file_bytes, sym_leaf, b"x")
            self.assertEqual(audit.require_safe_path(root / "fresh.json"), root / "fresh.json")
            self.assertEqual(audit.require_safe_path(reg_file), reg_file)
            with mock.patch.object(audit.os, "lstat", side_effect=PermissionError("denied")):
                self._assert_err("PATH_STAT_ERROR", audit.require_safe_path, reg_file)
                self._assert_err("FILE_WRITE_ERROR", audit.write_regular_file_bytes, reg_file, b"x")

            self._assert_err("FILE_READ_ERROR", audit.read_regular_file_bytes, root / "missing.json")
            self._assert_err("NON_REGULAR_FILE", audit.read_regular_file_bytes, real_dir)
            self._assert_err("NON_REGULAR_FILE", audit.write_regular_file_bytes, real_dir, b"x")
            self._assert_err("NON_REGULAR_FILE", audit.read_regular_file_bytes, Path("/dev/null"))
            fifo_p = root / "pipe.fifo"
            os.mkfifo(fifo_p)
            self._assert_err("NON_REGULAR_FILE", audit.read_regular_file_bytes, fifo_p)
            self._assert_err("NON_REGULAR_FILE", audit.write_regular_file_bytes, fifo_p, b"x")

            hl_a = root / "hl_a.json"
            hl_b = root / "hl_b.json"
            hl_a.write_bytes(b"original")
            os.link(hl_a, hl_b)
            self._assert_err("HARDLINK_FORBIDDEN", audit.write_regular_file_bytes, hl_a, b"overwrite")
            self.assertEqual(hl_b.read_bytes(), b"original")

            big_ok = root / "exact_cap.bin"
            big_ok.write_bytes(b"a" * oracle.MAX_JSON_BYTES)
            self.assertEqual(len(audit.read_regular_file_bytes(big_ok)), oracle.MAX_JSON_BYTES)
            big_bad = root / "too_big.bin"
            big_bad.write_bytes(b"a" * (oracle.MAX_JSON_BYTES + 1))
            self._assert_err("FILE_TOO_LARGE", audit.read_regular_file_bytes, big_bad)
            self._assert_err("INT_TOO_LARGE", audit.read_regular_file_bytes, reg_file, max_bytes=1 << 100)
            self._assert_err("INT_TOO_SMALL", audit.read_regular_file_bytes, reg_file, max_bytes=0)
            with mock.patch.object(audit.os, "read", side_effect=OSError("io fail")):
                self._assert_err("FILE_READ_ERROR", audit.read_regular_file_bytes, reg_file)

            target = root / "out.bin"
            audit.write_regular_file_bytes(target, b"first")
            self.assertEqual(target.read_bytes(), b"first")
            real_write = os.write
            with mock.patch.object(audit.os, "write", side_effect=lambda fd, buf: real_write(fd, buf[:2])):
                audit.write_regular_file_bytes(target, b"second_payload")
            self.assertEqual(target.read_bytes(), b"second_payload")

            coll0 = root / f".{target.name}.s30tmp.{os.getpid()}.0"
            coll0.write_bytes(b"keep")
            audit.write_regular_file_bytes(target, b"after_retry")
            self.assertEqual(target.read_bytes(), b"after_retry")
            self.assertEqual(coll0.read_bytes(), b"keep")
            for idx in range(1, 16):
                (root / f".{target.name}.s30tmp.{os.getpid()}.{idx}").write_bytes(b"c")
            self._assert_err("FILE_WRITE_ERROR", audit.write_regular_file_bytes, target, b"fail")
            for idx in range(16):
                (root / f".{target.name}.s30tmp.{os.getpid()}.{idx}").unlink()

            ro_dir = root / "ro"
            ro_dir.mkdir()
            os.chmod(ro_dir, 0o500)
            try:
                if os.geteuid() != 0:
                    self._assert_err("FILE_WRITE_ERROR", audit.write_regular_file_bytes, ro_dir / "x.json", b"x")
            finally:
                os.chmod(ro_dir, 0o700)

            with mock.patch.object(audit.os, "replace", side_effect=OSError("replace fail")):
                self._assert_err("FILE_WRITE_ERROR", audit.write_regular_file_bytes, target, b"bad")
            self.assertEqual([p.name for p in root.glob(f".{target.name}.s30tmp.*")], [])
            with (
                mock.patch.object(audit.os, "replace", side_effect=OSError("replace fail")),
                mock.patch.object(audit.os, "unlink", side_effect=OSError("unlink fail")),
            ):
                self._assert_err("FILE_WRITE_ERROR", audit.write_regular_file_bytes, target, b"bad2")
            for leftover in root.glob(f".{target.name}.s30tmp.*"):
                leftover.unlink()

    def test_relative_path_pwd_and_deleted_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real_sub = root / "sub"
            real_sub.mkdir()
            sym_sub = root / "sym_sub"
            sym_sub.symlink_to(real_sub)

            old_cwd = os.getcwd()
            os.chdir(real_sub)
            try:
                with mock.patch.dict(os.environ, {"PWD": str(real_sub)}):
                    self.assertEqual(audit.require_safe_path(Path("local.json")), Path("local.json"))
                with mock.patch.dict(os.environ, {"PWD": str(root / "does_not_exist")}):
                    self.assertEqual(audit.require_safe_path(Path("local.json")), Path("local.json"))
                with mock.patch.dict(os.environ, {"PWD": "relative/pwd"}):
                    self.assertEqual(audit.require_safe_path(Path("local.json")), Path("local.json"))
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(audit.require_safe_path(Path("local.json")), Path("local.json"))
                with mock.patch.dict(os.environ, {"PWD": str(sym_sub)}):
                    self._assert_err("SYMLINK_FORBIDDEN", audit.require_safe_path, Path("local.json"))
                with mock.patch.dict(os.environ, {"PWD": "/tmp/\udcff"}):
                    self._assert_err("INVALID_UTF8_SURROGATE", audit.require_safe_path, Path("local.json"))
                with mock.patch.object(audit.Path, "cwd", side_effect=FileNotFoundError("no cwd")):
                    self._assert_err("CWD_UNAVAILABLE", audit.require_safe_path, Path("local.json"))
            finally:
                os.chdir(old_cwd)

            gone = root / "gone"
            gone.mkdir()
            script = (
                "import os, sys; "
                f"os.chdir({str(gone)!r}); "
                f"os.rmdir({str(gone)!r}); "
                f"sys.path.insert(0, {_THIS_DIR!r}); "
                "import gemma_m1_tomography_audit as a; "
                "raise SystemExit(a.main(['--check', 'cert.json', '--json']))"
            )
            proc = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(json.loads(proc.stderr)["error_code"], "CWD_UNAVAILABLE")

    def test_wire_format_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cert_path = root / "cert.json"
            res_w = audit.write_certificate_file(cert_path)
            self.assertIs(res_w["written"], True)
            res_c = audit.check_certificate_file(cert_path)
            self.assertIs(res_c["valid"], True)
            self.assertEqual(res_c["file_sha256"], res_w["file_sha256"])

            raw = cert_path.read_bytes()
            crlf_p = root / "crlf.json"
            crlf_p.write_bytes(raw[:-1] + b"\r\n")
            self._assert_err("CRLF_FORBIDDEN", audit.check_certificate_file, crlf_p)
            nolf_p = root / "nolf.json"
            nolf_p.write_bytes(raw[:-1])
            self._assert_err("MISSING_TRAILING_LF", audit.check_certificate_file, nolf_p)
            bad_utf8 = root / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe\n")
            self._assert_err("INVALID_UTF8", audit.check_certificate_file, bad_utf8)
            non_canon = root / "non_canon.json"
            non_canon.write_bytes(b" " + raw)
            self._assert_err("NON_CANONICAL_BYTES", audit.check_certificate_file, non_canon)

            self._assert_err("INVALID_ARGV", audit._parse_cli_argv, "not_a_list")
            self._assert_err("INVALID_STR", audit._parse_cli_argv, [123])
            self._assert_err("INVALID_UTF8_SURROGATE", audit._parse_cli_argv, ["--check", "a\ud800.json"])
            self.assertEqual(audit._parse_cli_argv([]), ("standalone", None, False))
            self.assertEqual(audit._parse_cli_argv(["--standalone", "--json"]), ("standalone", None, True))
            self.assertEqual(audit._parse_cli_argv(["--json", "--check", "a.json"]), ("check", Path("a.json"), True))
            self.assertEqual(audit._parse_cli_argv(["--write", "b.json"]), ("write", Path("b.json"), False))
            self._assert_err("DUPLICATE_FLAG", audit._parse_cli_argv, ["--json", "--json"])
            self._assert_err("CONFLICTING_MODE", audit._parse_cli_argv, ["--standalone", "--standalone"])
            self._assert_err("CONFLICTING_MODE", audit._parse_cli_argv, ["--standalone", "--check", "a.json"])
            self._assert_err("CONFLICTING_MODE", audit._parse_cli_argv, ["--check", "a.json", "--write", "b.json"])
            self._assert_err("MISSING_PATH_ARG", audit._parse_cli_argv, ["--check"])
            self._assert_err("MISSING_PATH_ARG", audit._parse_cli_argv, ["--write"])
            self._assert_err("INVALID_PATH_ARG", audit._parse_cli_argv, ["--check", ""])
            self._assert_err("INVALID_PATH_ARG", audit._parse_cli_argv, ["--write", ""])
            self._assert_err("INVALID_PATH_ARG", audit._parse_cli_argv, ["--check", "--json"])
            self._assert_err("INVALID_PATH_ARG", audit._parse_cli_argv, ["--write", "--json"])
            self._assert_err("UNKNOWN_FLAG", audit._parse_cli_argv, ["--write=--json"])
            self._assert_err("UNKNOWN_FLAG", audit._parse_cli_argv, ["--check=--json"])
            self._assert_err("UNKNOWN_FLAG", audit._parse_cli_argv, ["--bogus"])

            old_cwd = os.getcwd()
            os.chdir(root)
            try:
                for argv_bad, exp_code in (
                    (["--write", "--json"], "INVALID_PATH_ARG"),
                    (["--write=--json"], "UNKNOWN_FLAG"),
                    (["--check=--json"], "UNKNOWN_FLAG"),
                ):
                    out_s, err_s = io.StringIO(), io.StringIO()
                    with mock.patch.object(sys, "stdout", out_s), mock.patch.object(sys, "stderr", err_s):
                        rc = audit.main(argv_bad)
                    self.assertEqual(rc, 2)
                    self.assertEqual(out_s.getvalue(), "")
                    self.assertEqual(json.loads(err_s.getvalue())["error_code"], exp_code)
                    self.assertFalse((root / "--json").exists())

                for argv_ok in (
                    [],
                    ["--standalone", "--json"],
                    ["--write", "cli_cert.json", "--json"],
                    ["--check", "cli_cert.json", "--json"],
                    ["--check", "cli_cert.json"],
                ):
                    out_s, err_s = io.StringIO(), io.StringIO()
                    with mock.patch.object(sys, "stdout", out_s), mock.patch.object(sys, "stderr", err_s):
                        rc = audit.main(argv_ok)
                    self.assertEqual(rc, 0)
                    self.assertEqual(err_s.getvalue(), "")
                    if "--json" in argv_ok:
                        self.assertIsInstance(json.loads(out_s.getvalue()), dict)
                    else:
                        self.assertTrue(out_s.getvalue().startswith("OK "))
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
