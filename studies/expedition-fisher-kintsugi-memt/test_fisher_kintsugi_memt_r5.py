#!/usr/bin/env python3
"""Track S42 R5 Test Suite: 4 Mathematical Pillars, Exported-Helper Soundness,
100% Mutation/BoolOp Killers, >= 25 Re-Signed Non-Replay Tamper Kills & R2-Grade Zone-B Security.

Pure Python 3 standard library: zero AST floats, zero bare asserts, zero `.isdigit()`, zero mathematical RNG.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fisher_kintsugi_memt_oracle as oracle
import fisher_kintsugi_memt_audit as audit

Err = oracle.FisherKintsugiMemtError


class _SubInt(int):
    pass


class _SubStr(str):
    pass


class _SubFrac(Fraction):
    pass


class _SubPath(type(Path("."))):
    pass


class TestTrackS42FisherKintsugiMemtR5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.build_standalone_certificate()
        cls.payload = cls.cert["payload"]

    # -----------------------------------------------------------------------
    # 1. Four Exact Mathematical Pillars
    # -----------------------------------------------------------------------

    def test_01_pillar1_fisher_waterfilling_amgm(self) -> None:
        self.assertEqual(self.payload["novelty_tier"], "N1")
        self.assertEqual(self.payload["oracle_class"], "exact")
        self.assertEqual(len(self.payload["prior_art_cs13_anchors"]), 5)
        self.assertIn("professorpalmer/hard-proof", self.payload["prior_art_cs13_anchors"])
        self.assertIs(self.payload["measured_live_gemma4_lamq_weights"], False)
        self.assertIs(self.payload["measured_live_gbench_serving_endpoint"], False)
        self.assertIs(self.payload["measured_live_memt_554b_weights"], False)
        self.assertIs(self.payload["zero_trust_path_floats"], True)
        self.assertIs(self.payload["zero_bare_asserts"], True)
        self.assertIs(self.payload["zero_mathematical_rng"], True)

        p1 = self.payload["pillar1_fisher_waterfilling_amgm"]
        self.assertIs(p1["uniform_ratio_equals_one"], True)
        self.assertIs(p1["mild_ratio_exceeds_two"], True)
        self.assertIs(p1["heavy_ratio_exceeds_twelve"], True)
        self.assertIs(p1["strict_hierarchy_certified"], True)

        unif = p1["uniform_control"]
        self.assertEqual(unif["improvement_ratio"], "1")
        self.assertEqual(unif["extra_average_bpw"], "0")
        self.assertIs(unif["kkt_equalization_exact"], True)
        self.assertIs(unif["am_gm_identity_exact"], True)

        mild = p1["mild_skew"]
        self.assertEqual(mild["improvement_ratio"], "389/128")
        self.assertEqual(mild["improvement_ratio_4dp"], "3.0390")
        self.assertEqual(mild["extra_average_bpw"], "0")

        heavy = p1["heavy_outlier_gemma4_scale"]
        self.assertEqual(heavy["improvement_ratio"], "263/16")
        self.assertEqual(heavy["improvement_ratio_4dp"], "16.4375")
        self.assertEqual(heavy["b_star"], ["15/2", "7/2", "7/2", "7/2", "7/2", "7/2", "7/2", "7/2"])
        self.assertEqual(heavy["extra_average_bpw"], "0")
        self.assertEqual(heavy["k_uniform_total"], "263/512")
        self.assertEqual(heavy["k_fisher_total"], "1/32")
        self.assertIs(heavy["all_perturbations_strictly_worse"], True)
        self.assertEqual(
            [w["excess_kl"] for w in heavy["perturbation_witnesses"]],
            ["1/512", "9/1024", "11/1024"],
        )

    def test_02_pillar2_kintsugi_margin_vs_kl(self) -> None:
        p2 = self.payload["pillar2_kintsugi_margin_vs_kl"]
        self.assertIs(p2["cs20_average_kl_blindness_certified"], True)

        fw = p2["quadratic_fisher_witness"]
        self.assertIs(fw["fisher_shift_invariant_nullspace"], True)
        self.assertEqual(fw["fisher_matrix_rank"], 3)
        self.assertIs(fw["fisher_3x3_block_pd"], True)
        self.assertIs(fw["uniform_lower_avg_kl_yet_breaks_task"], True)
        self.assertEqual(fw["uniform_a"]["average_fisher_kl"], "11/512")
        self.assertEqual(fw["uniform_a"]["critical_token_margin"], "-1/2")
        self.assertIs(fw["uniform_a"]["task_pass"], False)
        self.assertEqual(fw["kintsugi_b"]["average_fisher_kl"], "1499/8192")
        self.assertEqual(fw["kintsugi_b"]["critical_token_margin"], "1/4")
        self.assertIs(fw["kintsugi_b"]["task_pass"], True)
        self.assertEqual(fw["kintsugi_over_uniform_kl_ratio"], "1499/176")

        dw = p2["exact_dyadic_shannon_witness"]
        self.assertEqual(dw["uniform_a_avg_kl_bits"], "1/16")
        self.assertEqual(dw["uniform_a_critical_prob_margin"], "-1/4")
        self.assertIs(dw["uniform_a_task_pass"], False)
        self.assertEqual(dw["kintsugi_b_avg_kl_bits"], "21/64")
        self.assertEqual(dw["kintsugi_b_routine_prob_margin"], "1/8")
        self.assertEqual(dw["kintsugi_b_critical_prob_margin"], "1/4")
        self.assertIs(dw["kintsugi_b_task_pass"], True)
        self.assertEqual(dw["kintsugi_over_uniform_kl_ratio"], "21/4")
        self.assertIs(dw["gibbs_strict_positivity_certified"], True)
        self.assertIs(dw["uniform_lower_avg_kl_yet_breaks_task"], True)

    def test_03_pillar3_gbench_p99_order_statistics(self) -> None:
        p3 = self.payload["pillar3_gbench_p99_order_statistics"]
        self.assertEqual(p3["sharp_99pct_threshold_n"], 459)
        self.assertEqual(p3["gbench_slo_cohort_threshold_n"], 1099)
        self.assertIs(p3["sharp_boundary_458_fails_459_passes"], True)
        self.assertIs(p3["cs18_uncertified_floor_enforced"], True)

        self.assertIs(p3["cohort_n100"]["pass_99pct_upper_tail"], False)
        self.assertEqual(p3["cohort_n100"]["cs18_gbench_slo_status"], "UNCERTIFIED")
        self.assertIs(p3["cohort_n458"]["pass_99pct_upper_tail"], False)
        self.assertEqual(p3["cohort_n458"]["reciprocal_miss_prob_floor"], 99)
        self.assertEqual(p3["cohort_n458"]["cs18_upper_tail_status"], "UNCERTIFIED")
        self.assertIs(p3["cohort_n459"]["pass_99pct_upper_tail"], True)
        self.assertEqual(p3["cohort_n459"]["reciprocal_miss_prob_floor"], 100)
        self.assertEqual(p3["cohort_n459"]["cs18_upper_tail_status"], "CERTIFIED_UPPER_TAIL")
        self.assertEqual(p3["cohort_n459"]["cs18_gbench_slo_status"], "UNCERTIFIED")

        c1099 = p3["cohort_n1099"]
        self.assertIs(c1099["pass_99pct_upper_tail"], True)
        self.assertIs(c1099["miss_prob_below_1_over_60000"], True)
        self.assertIs(c1099["den_exceeds_max_json_int_bits"], True)
        self.assertEqual(c1099["den_bit_length"], 7302)
        self.assertEqual(c1099["reciprocal_miss_prob_floor"], 62650)
        self.assertEqual(c1099["cs18_gbench_slo_status"], "CERTIFIED")
        br = p3["n1099_two_sided_order_stat_bracket"]
        self.assertEqual(br["order_stat_lower_rank"], 1081)
        self.assertEqual(br["order_stat_upper_rank"], 1096)
        self.assertIs(br["coverage_exceeds_95pct"], True)

    def test_04_pillar4_memt_gdn2_rank_and_channel2b(self) -> None:
        p4 = self.payload["pillar4_memt_gdn2_rank_and_channel2b"]
        cap = p4["capacity_obstruction"]
        self.assertIs(cap["hadamard_keys_orthonormal"], True)
        self.assertIs(cap["transpose_duality_l4_exact"], True)
        self.assertIs(cap["transpose_duality_l5_exact"], True)
        self.assertEqual(cap["rank_s4"], 4)
        self.assertIs(cap["l4_zero_error_exact"], True)
        self.assertEqual(cap["rank_key_matrix_k5"], 4)
        self.assertEqual(cap["rank_s5"], 4)
        self.assertIs(cap["nullspace_k5_c_zero_exact"], True)
        self.assertEqual(cap["universal_cauchy_schwarz_lb"], "106/5")
        self.assertEqual(cap["l5_total_err_sq"], "212/5")
        self.assertEqual(cap["delta_s_frob_norm_sq"], "212/5")
        self.assertIs(cap["parseval_error_identity_exact"], True)
        self.assertIs(cap["capacity_obstruction_certified"], True)

        pois = p4["channel_2b_poisoning_witness"]
        self.assertIs(pois["state_hashes_distinct"], True)
        self.assertEqual(pois["delta_s_exact_rank"], 1)
        self.assertEqual(pois["clean_margin"], "2")
        self.assertEqual(pois["poisoned_margin"], "-83/25")
        self.assertEqual(pois["margin_after_benign_orth_turn"], "-83/25")
        self.assertIs(pois["benign_readout_sign_flipped"], True)
        self.assertIs(pois["poison_persists_after_benign_turn"], True)

        air = p4["four_channel_airlock"]
        self.assertEqual(air["clean_record"]["airlock_4ch_decision"], "ALLOW")
        self.assertIs(air["legacy_3ch_blind_to_channel2b"], True)
        self.assertIs(air["all_four_negative_controls_denied"], True)
        self.assertIs(air["airlock_extension_certified"], True)

    # -----------------------------------------------------------------------
    # 2. Exported Primitives, Domain Helpers & Canonical JSON Guards
    # -----------------------------------------------------------------------

    def test_05_primitive_parsers_formatters_and_int_bounds(self) -> None:
        self.assertEqual(oracle.require_exact_int(5, "x", min_val=5, max_val=5), 5)
        for bad in (True, False, int(5) / int(2), "5", None, _SubInt(5)):
            with self.assertRaises(Err):
                oracle.require_exact_int(bad, "x")
        with self.assertRaises(Err):
            oracle.require_exact_int(4, "x", min_val=5)
        with self.assertRaises(Err):
            oracle.require_exact_int(6, "x", max_val=5)

        self.assertIs(oracle.require_exact_bool(True, "b"), True)
        with self.assertRaises(Err):
            oracle.require_exact_bool(1, "b")
        self.assertEqual(oracle.require_exact_str("", "s", non_empty=False), "")
        with self.assertRaises(Err):
            oracle.require_exact_str("", "s", non_empty=True)
        with self.assertRaises(Err):
            oracle.require_exact_str(_SubStr("a"), "s")
        with self.assertRaises(Err):
            oracle.require_exact_str("bad\ud800sur", "s")
        self.assertEqual(oracle.require_sha256_hex("a" * 64, "h"), "a" * 64)
        for bad_hex in ("a" * 63, "A" * 64, "g" * 64, "\u00e9" * 64):
            with self.assertRaises(Err):
                oracle.require_sha256_hex(bad_hex, "h")

        self.assertEqual(oracle.require_exact_fraction(Fraction(0, 1), "q", non_negative=True), Fraction(0, 1))
        self.assertEqual(oracle.require_exact_fraction(Fraction(1, 2), "q", positive=True), Fraction(1, 2))
        for bad in (1, True, int(1) / int(2), "1/2", None, _SubFrac(1, 2)):
            with self.assertRaises(Err):
                oracle.require_exact_fraction(bad, "q")
        with self.assertRaises(Err):
            oracle.require_exact_fraction(Fraction(0, 1), "q", positive=True)
        with self.assertRaises(Err):
            oracle.require_exact_fraction(Fraction(-1, 3), "q", non_negative=True)

        self.assertEqual(oracle.parse_exact_int_ascii("0", "n"), 0)
        self.assertEqual(oracle.parse_exact_int_ascii("-42", "n"), -42)
        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        too_big = 1 << oracle.MAX_INT_BITS
        self.assertEqual(len(str(max_ok)), oracle.MAX_INT_DIGITS)
        self.assertEqual(oracle.parse_exact_int_ascii(str(max_ok), "n"), max_ok)
        self.assertEqual(oracle.parse_exact_int_ascii(str(-max_ok), "n"), -max_ok)
        self.assertEqual(oracle.parse_exact_rational(oracle.format_exact_rational(Fraction(max_ok, 1)), "r"), Fraction(max_ok, 1))
        self.assertEqual(oracle.parse_exact_rational(oracle.format_exact_rational(Fraction(-max_ok, max_ok - 1)), "r"), Fraction(-max_ok, max_ok - 1))
        self.assertEqual(
            oracle.parse_strict_json_text(oracle.canonical_json_bytes({"k": max_ok}).decode("ascii"), require_object=True),
            {"k": max_ok},
        )
        for bad in ("", "-", "-0", "00", "007", "-01", "+5", " 5", "1_0", "\u00b2", "\u0661", str(too_big), "9" * oracle.MAX_INT_DIGITS, "9" * (oracle.MAX_INT_DIGITS + 1)):
            with self.assertRaises(Err):
                oracle.parse_exact_int_ascii(bad, "n")

        self.assertEqual(oracle.parse_exact_rational("7", "r"), Fraction(7, 1))
        self.assertEqual(oracle.parse_exact_rational("-3/4", "r"), Fraction(-3, 4))
        for bad in ("1/2/3", "3/1", "3/0", "3/-2", "2/4", "1/\u0662"):
            with self.assertRaises(Err):
                oracle.parse_exact_rational(bad, "r")
        self.assertEqual(oracle.format_exact_rational(Fraction(5, 1)), "5")
        self.assertEqual(oracle.format_exact_rational(Fraction(-5, 3)), "-5/3")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-7, 4), 2), "-1.75")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(7, 4), 2), "1.75")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(7, 4), 0), "1")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-1, 1000), 2), "0.00")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-1, 1000), 0), "0")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(1, 3), 18), "0.333333333333333333")
        with self.assertRaises(Err):
            oracle.format_decimal_trunc(Fraction(1, 2), 19)

        self.assertEqual(oracle.format_exact_rational(Fraction(max_ok, 1)), str(max_ok))
        self.assertEqual(oracle.format_decimal_trunc(Fraction(max_ok, 1), 0), str(max_ok))
        for bad_q in (Fraction(too_big, 1), Fraction(1, too_big), Fraction(10 ** 4500, 1), Fraction(1, 10 ** 4500)):
            with self.assertRaises(Err) as cm_r:
                oracle.format_exact_rational(bad_q)
            self.assertEqual(cm_r.exception.code, "INT_TOO_LARGE")
            with self.assertRaises(Err) as cm_d:
                oracle.format_decimal_trunc(bad_q, 2)
            self.assertEqual(cm_d.exception.code, "INT_TOO_LARGE")

    def test_06_json_canonicalization_cycles_and_strict_parser(self) -> None:
        raw = oracle.canonical_json_bytes({"b": [1, True, None, "x"], "a": 0})
        self.assertEqual(raw, b'{"a":0,"b":[1,true,null,"x"]}')
        with self.assertRaises(Err):
            oracle.canonical_json_bytes({"f": Fraction(1, 2)})
        with self.assertRaises(Err):
            oracle.canonical_json_bytes({"f": int(3) / int(2)})
        cyc: dict[str, object] = {}
        cyc["self"] = cyc
        with self.assertRaises(Err) as cm_cyc:
            oracle.canonical_json_bytes(cyc)
        self.assertEqual(cm_cyc.exception.code, "JSON_TOO_DEEP")

        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        self.assertIn(str(max_ok).encode("ascii"), oracle.canonical_json_bytes({"ok": max_ok}))
        for huge in (1 << oracle.MAX_INT_BITS, -(1 << oracle.MAX_INT_BITS), 10 ** 4300):
            with self.assertRaises(Err) as cm_hi:
                oracle.canonical_json_bytes({"big": huge})
            self.assertEqual(cm_hi.exception.code, "INT_TOO_LARGE")
        with self.assertRaises(Err) as cm_hj:
            oracle.parse_strict_json_text('{"big":' + "9" * 4500 + "}", require_object=True)
        self.assertEqual(cm_hj.exception.code, "INT_TOKEN_TOO_LONG")

        nested_ok: object = 0
        for _ in range(oracle.MAX_JSON_DEPTH):
            nested_ok = [nested_ok]
        oracle.canonical_json_bytes(nested_ok)
        with self.assertRaises(Err):
            oracle.canonical_json_bytes([nested_ok])

        exact_payload = {"s": "a" * (oracle.MAX_JSON_BYTES - 8)}
        exact_raw = oracle.canonical_json_bytes(exact_payload)
        self.assertEqual(len(exact_raw), oracle.MAX_JSON_BYTES)
        self.assertEqual(oracle.parse_strict_json_text(exact_raw.decode("ascii"), require_object=True), exact_payload)
        with self.assertRaises(Err) as cm_j1:
            oracle.canonical_json_bytes({"s": "a" * oracle.MAX_JSON_BYTES})
        self.assertEqual(cm_j1.exception.code, "JSON_TOO_LARGE")
        # Multi-byte UTF-8 size cap check
        with self.assertRaises(Err) as cm_j2:
            oracle.parse_strict_json_text('{"s":"' + "\u00e9" * (oracle.MAX_JSON_BYTES // 2) + '"}')
        self.assertEqual(cm_j2.exception.code, "JSON_TOO_LARGE")
        for bad_json in ('{"a":1.5}', '{"a":NaN}', '{"a":1,"a":2}', "[1,2]", "{bad"):
            with self.assertRaises(Err):
                oracle.parse_strict_json_text(bad_json, require_object=True)
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])

    def test_07_linear_algebra_and_domain_helper_boundaries(self) -> None:
        # Matrix & vector helpers (bareiss_det_exact, certify_sylvester_pd_exact, exact_matrix_rank_q, dot, mat_vec)
        swap_m = [[Fraction(0, 1), Fraction(1, 1)], [Fraction(1, 1), Fraction(0, 1)]]
        self.assertEqual(oracle.bareiss_det_exact(swap_m), Fraction(-1, 1))
        self.assertEqual(oracle.exact_matrix_rank_q(swap_m), 2)
        sing_m = [[Fraction(1, 1), Fraction(1, 1)], [Fraction(1, 1), Fraction(1, 1)]]
        self.assertEqual(oracle.bareiss_det_exact(sing_m), Fraction(0, 1))
        self.assertIs(oracle.certify_sylvester_pd_exact(sing_m), False)
        self.assertEqual(oracle.exact_matrix_rank_q(sing_m), 1)
        zero_col_m = [[Fraction(0, 1), Fraction(1, 1)], [Fraction(0, 1), Fraction(2, 1)]]
        self.assertEqual(oracle.exact_matrix_rank_q(zero_col_m), 1)
        wide_m = [[Fraction(1, 1), Fraction(0, 1), Fraction(2, 1)]]
        self.assertEqual(oracle.exact_matrix_rank_q(wide_m), 1)
        id16 = [[Fraction(1, 1) if i == j else Fraction(0, 1) for j in range(16)] for i in range(16)]
        self.assertEqual(oracle.bareiss_det_exact(id16), Fraction(1, 1))
        id17 = [[Fraction(1, 1) if i == j else Fraction(0, 1) for j in range(17)] for i in range(17)]
        for bad_mat, exp_code in (
            ("not_mat", "INVALID_MATRIX"),
            (123, "INVALID_MATRIX"),
            ([], "INVALID_MATRIX"),
            ([[Fraction(1, 1)]] * 17, "INVALID_MATRIX"),
            (id17, "INVALID_MATRIX"),
            (["a"], "INVALID_MATRIX"),
            ([[]], "INVALID_MATRIX"),
            ([[Fraction(1, 1)] * 17], "INVALID_MATRIX"),
            ([[Fraction(1, 1)], "x"], "RAGGED_MATRIX"),
            ([[Fraction(1, 1)], [Fraction(1, 1), Fraction(2, 1)]], "RAGGED_MATRIX"),
            ([[Fraction(1, 1), Fraction(2, 1)]], "NON_SQUARE_MATRIX"),
        ):
            with self.assertRaises(Err) as cm_m:
                oracle.bareiss_det_exact(bad_mat)
            self.assertEqual(cm_m.exception.code, exp_code)
        with self.assertRaises(Err) as cm_r17:
            oracle.exact_matrix_rank_q([[Fraction(1, 1)]] * 17)
        self.assertEqual(cm_r17.exception.code, "INVALID_MATRIX")
        with self.assertRaises(Err) as cm_c17:
            oracle.exact_matrix_rank_q([[Fraction(1, 1)] * 17])
        self.assertEqual(cm_c17.exception.code, "INVALID_MATRIX")
        with self.assertRaises(Err) as cm_ns:
            oracle.certify_sylvester_pd_exact([[Fraction(1, 1), Fraction(2, 1)], [Fraction(0, 1), Fraction(1, 1)]])
        self.assertEqual(cm_ns.exception.code, "NON_SYMMETRIC_MATRIX")

        self.assertEqual(oracle.dot_exact([Fraction(1, 2), Fraction(3, 1)], [Fraction(4, 1), Fraction(1, 3)]), Fraction(3, 1))
        self.assertEqual(oracle.mat_vec_exact([[Fraction(1, 1), Fraction(2, 1)]], [Fraction(3, 1), Fraction(4, 1)]), [Fraction(11, 1)])
        for u_bad, v_bad, exp_c in (
            ("ab", [Fraction(1, 1), Fraction(2, 1)], "INVALID_VECTOR"),
            ([Fraction(1, 1), Fraction(2, 1)], "ab", "INVALID_VECTOR"),
            ([], [], "INVALID_VECTOR"),
            ([Fraction(1, 1), Fraction(2, 1)], [Fraction(3, 1)], "VECTOR_LENGTH_MISMATCH"),
            ([1], [Fraction(1, 1)], "INVALID_FRACTION"),
        ):
            with self.assertRaises(Err) as cm_d:
                oracle.dot_exact(u_bad, v_bad)
            self.assertEqual(cm_d.exception.code, exp_c)
        for v_bad in ("12", [Fraction(1, 1)]):
            with self.assertRaises(Err) as cm_mv:
                oracle.mat_vec_exact([[Fraction(1, 1), Fraction(2, 1)]], v_bad)
            self.assertEqual(cm_mv.exception.code, "MATRIX_VECTOR_MISMATCH")

        # Pillar 1 helper boundary killers
        self.assertEqual(oracle.fisher_waterfilling_analysis([0] * 16, Fraction(2, 1), Fraction(4, 1))["row_count_m"], 16)
        for bad_args, exp_c in (
            (("bad", Fraction(2, 1), Fraction(4, 1), ()), "INVALID_EXPONENTS"),
            (([], Fraction(2, 1), Fraction(4, 1), ()), "INVALID_EXPONENTS"),
            (([0] * 17, Fraction(2, 1), Fraction(4, 1), ()), "INVALID_EXPONENTS"),
            (([1, 0], Fraction(2, 1), Fraction(4, 1), ()), "NON_INTEGER_MEAN_EXP"),
            (([0, 0], Fraction(1, 3), Fraction(4, 1), ()), "NON_INTEGER_RATE_EXPONENT"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), "bad"), "INVALID_PERTURBATIONS"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), ["bad"]), "INVALID_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), ["ab"]), "INVALID_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [123]), "INVALID_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p",)]), "INVALID_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p", "ab")]), "INVALID_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p", [Fraction(1, 1)])]), "INVALID_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p", [Fraction(1, 1), Fraction(0, 1)])]), "PERTURBATION_NOT_ZERO_SUM"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p", [Fraction(0, 1), Fraction(0, 1)])]), "TRIVIAL_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p", [Fraction(4, 1), Fraction(-4, 1)])]), "INFEASIBLE_PERTURBATION"),
            (([0, 0], Fraction(2, 1), Fraction(4, 1), [("p", [Fraction(1, 3), Fraction(-1, 3)])]), "NON_INTEGER_PERT_EXPONENT"),
        ):
            with self.assertRaises(Err) as cm_fw:
                oracle.fisher_waterfilling_analysis(bad_args[0], bad_args[1], bad_args[2], bad_args[3])
            self.assertEqual(cm_fw.exception.code, exp_c)

        # Pillar 2 helper boundary killers
        p16 = [Fraction(1, 16)] * 16
        self.assertEqual(oracle.quadratic_fisher_kl_exact(p16, [Fraction(0, 1)] * 16), Fraction(0, 1))
        self.assertEqual(oracle.exact_dyadic_shannon_kl_bits([Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 2)]), (Fraction(0, 1), [0, 0]))
        for bad_p, exp_c in (
            ("ab", "INVALID_DISTRIBUTION"),
            ([Fraction(1, 1)], "INVALID_DISTRIBUTION"),
            ([Fraction(1, 17)] * 17, "INVALID_DISTRIBUTION"),
            ([Fraction(1, 2), Fraction(1, 3)], "DISTRIBUTION_NOT_UNIT_SUM"),
        ):
            with self.assertRaises(Err) as cm_p:
                oracle.quadratic_fisher_kl_exact(bad_p, [Fraction(0, 1), Fraction(0, 1)])
            self.assertEqual(cm_p.exception.code, exp_c)
        for bad_d in ("ab", [Fraction(0, 1)]):
            with self.assertRaises(Err) as cm_qd:
                oracle.quadratic_fisher_kl_exact([Fraction(1, 2), Fraction(1, 2)], bad_d)
            self.assertEqual(cm_qd.exception.code, "INVALID_DELTA")
        with self.assertRaises(Err) as cm_dm:
            oracle.exact_dyadic_shannon_kl_bits([Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)])
        self.assertEqual(cm_dm.exception.code, "DISTRIBUTION_LENGTH_MISMATCH")
        for p_bad, q_bad in (
            ([Fraction(3, 8), Fraction(1, 4), Fraction(3, 8)], [Fraction(1, 8), Fraction(1, 2), Fraction(3, 8)]),
            ([Fraction(1, 2), Fraction(1, 8), Fraction(3, 8)], [Fraction(3, 8), Fraction(1, 4), Fraction(3, 8)]),
            ([Fraction(1, 8), Fraction(1, 2), Fraction(3, 8)], [Fraction(3, 8), Fraction(1, 4), Fraction(3, 8)]),
            ([Fraction(3, 8), Fraction(1, 4), Fraction(3, 8)], [Fraction(1, 2), Fraction(1, 8), Fraction(3, 8)]),
        ):
            with self.assertRaises(Err) as cm_nd:
                oracle.exact_dyadic_shannon_kl_bits(p_bad, q_bad)
            self.assertEqual(cm_nd.exception.code, "NON_DYADIC_RATIO")

        # Pillar 3 helper boundary killers (incl. n=1 exact divisibility & n=1098 below gbench threshold)
        c1 = oracle.gbench_p99_cohort_cert(1)
        self.assertEqual(c1["p_zero_exceedances_ppm_lo"], 990000)
        self.assertEqual(c1["p_zero_exceedances_ppm_hi"], 990000)
        self.assertEqual(c1["reciprocal_miss_prob_floor"], 1)
        self.assertEqual(c1["reciprocal_miss_prob_ceil"], 2)
        c1098 = oracle.gbench_p99_cohort_cert(1098)
        self.assertIs(c1098["miss_prob_below_1_over_60000"], True)
        self.assertEqual(c1098["cs18_gbench_slo_status"], "UNCERTIFIED")
        c2000 = oracle.gbench_p99_cohort_cert(2000)
        self.assertIs(c2000["miss_prob_below_1_over_60000"], True)
        for bad_n in (0, 2001):
            with self.assertRaises(Err):
                oracle.gbench_p99_cohort_cert(bad_n)

        # Pillar 4 helper boundary killers (gdn2_delta_step_exact & evaluate_four_channel_airlock)
        id2 = [[Fraction(1, 1), Fraction(0, 1)], [Fraction(0, 1), Fraction(1, 1)]]
        for bad_k, bad_v, exp_c in (
            ("ab", [Fraction(1, 1), Fraction(0, 1)], "INVALID_KEY"),
            ([Fraction(1, 1)], [Fraction(1, 1), Fraction(0, 1)], "INVALID_KEY"),
            ([Fraction(1, 1), Fraction(0, 1)], "ab", "INVALID_VALUE"),
            ([Fraction(1, 1), Fraction(0, 1)], [Fraction(1, 1)], "INVALID_VALUE"),
        ):
            with self.assertRaises(Err) as cm_g:
                oracle.gdn2_delta_step_exact(id2, bad_k, bad_v)
            self.assertEqual(cm_g.exception.code, exp_c)
        with self.assertRaises(Err) as cm_a1:
            oracle.evaluate_four_channel_airlock("not_dict")
        self.assertEqual(cm_a1.exception.code, "INVALID_AIRLOCK_RECORD")
        with self.assertRaises(Err) as cm_a2:
            oracle.evaluate_four_channel_airlock({})
        self.assertEqual(cm_a2.exception.code, "INVALID_AIRLOCK_KEYS")
        # Trigger every individual OR branch inside evaluate_four_channel_airlock!
        base_air = {
            "token_embd_hash_match": True,
            "block_dpi_ratio": "1",
            "max_block_dpi_ratio": "4",
            "kv_cross_model_reuse": False,
            "kv_residual_evicted_mentions": 0,
            "latent_state_hash_match": True,
            "latent_delta_norm_sq": "0",
            "max_latent_delta_norm_sq": "1",
            "latent_cross_session_carryover": False,
            "corpus_provenance_verified": True,
            "cot_mode_transfer_gap": "0",
            "max_cot_mode_transfer_gap": "1/20",
        }
        for field, val, exp_v in (
            ("block_dpi_ratio", "5", "CHANNEL_1_WEIGHT_INTEGRITY"),
            ("kv_residual_evicted_mentions", 1, "CHANNEL_2A_KV_PERSISTENCE"),
            ("corpus_provenance_verified", False, "CHANNEL_3_DATA_PROVENANCE"),
            ("latent_state_hash_match", False, "CHANNEL_2B_LATENT_MEMORY_POISONING"),
            ("latent_delta_norm_sq", "2", "CHANNEL_2B_LATENT_MEMORY_POISONING"),
            ("latent_cross_session_carryover", True, "CHANNEL_2B_LATENT_MEMORY_POISONING"),
        ):
            rec = dict(base_air)
            rec[field] = val
            res = oracle.evaluate_four_channel_airlock(rec)
            self.assertEqual(res["airlock_4ch_decision"], "DENY")
            self.assertIn(exp_v, res["violations_4ch"])

    # -----------------------------------------------------------------------
    # 3. Certificate Validation, Non-Replay Schema & >=25 Secondary Audit Kills
    # -----------------------------------------------------------------------

    def test_08_certificate_validation_nonreplay_and_secondary_audit_killers(self) -> None:
        res_ok = audit.verify_certificate_dict(self.cert, full_replay=True)
        self.assertIs(res_ok["valid"], True)
        self.assertIs(res_ok["secondary_audit"]["secondary_audit_passed"], True)
        res_nr = audit.verify_certificate_dict(self.cert, full_replay=False)
        self.assertIs(res_nr["full_replay"], False)

        for bad_flag in ("yes", 1, 0, None):
            with self.assertRaises(Err) as cm_bf:
                oracle.validate_certificate_payload(self.cert, full_replay=bad_flag)  # type: ignore[arg-type]
            self.assertEqual(cm_bf.exception.code, "INVALID_BOOL")

        with self.assertRaises(Err) as cm_c0:
            oracle.validate_certificate_payload("not_dict")
        self.assertEqual(cm_c0.exception.code, "CERT_NOT_OBJECT")
        c_extra = copy.deepcopy(self.cert)
        c_extra["extra"] = 1
        with self.assertRaises(Err) as cm_ck:
            oracle.validate_certificate_payload(c_extra)
        self.assertEqual(cm_ck.exception.code, "INVALID_CERT_KEYS")
        for field, bad_val, exp_code in (
            ("schema", "wrong", "INVALID_SCHEMA"),
            ("track_id", "wrong", "INVALID_TRACK_ID"),
            ("payload", [], "PAYLOAD_NOT_OBJECT"),
            ("payload_sha256", "0" * 64, "PAYLOAD_SHA_MISMATCH"),
            ("certificate_sha256", "0" * 64, "CERT_SHA_MISMATCH"),
        ):
            c_bad = copy.deepcopy(self.cert)
            c_bad[field] = bad_val
            with self.assertRaises(Err) as cm_f:
                oracle.validate_certificate_payload(c_bad, full_replay=False)
            self.assertEqual(cm_f.exception.code, exp_code)

        def _resign(payload_mut: dict[str, object]) -> dict[str, object]:
            p_sha = oracle.hashlib.sha256(oracle.canonical_json_bytes(payload_mut)).hexdigest()
            core = {"schema": oracle.SCHEMA_VERSION, "track_id": oracle.TRACK_ID, "payload_sha256": p_sha, "payload": payload_mut}
            c_sha = oracle.hashlib.sha256(oracle.canonical_json_bytes(core)).hexdigest()
            return {"schema": oracle.SCHEMA_VERSION, "track_id": oracle.TRACK_ID, "payload_sha256": p_sha, "certificate_sha256": c_sha, "payload": payload_mut}

        # Re-signed tampered payload passes non-replay schema if top-level valid, MUST FAIL full_replay=True!
        p_tamp_replay = copy.deepcopy(self.payload)
        p_tamp_replay["pillar1_fisher_waterfilling_amgm"]["uniform_control"]["row_count_m"] = 9
        with self.assertRaises(Err) as cm_rep:
            oracle.validate_certificate_payload(_resign(p_tamp_replay), full_replay=True)
        self.assertEqual(cm_rep.exception.code, "REPLAY_MISMATCH")

        # Non-replay top-level schema killers
        nonreplay_mutators = [
            lambda p: p.clear(),
            lambda p: p.__setitem__("novelty_tier", "N0"),
            lambda p: p.__setitem__("oracle_class", "approx"),
            lambda p: p.__setitem__("prior_art_cs13_anchors", ["only_one"]),
            lambda p: p.__setitem__("prior_art_cs13_anchors", [1, 2, 3, 4, 5]),
            lambda p: p.__setitem__("measured_live_gemma4_lamq_weights", True),
            lambda p: p.__setitem__("zero_mathematical_rng", False),
            lambda p: p.__setitem__("pillar1_fisher_waterfilling_amgm", []),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"].__setitem__("kkt_equalization_exact", 1),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"].__setitem__("mean_exponent", False),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"]["sensitivity_exponents"].__setitem__(0, False),
        ]
        for fn in nonreplay_mutators:
            pm = copy.deepcopy(self.payload)
            fn(pm)
            with self.assertRaises(Err):
                oracle.validate_certificate_payload(_resign(pm), full_replay=False)

        # Direct tuple-vs-list killer on _validate_payload_nonreplay_schema (kills type(anchors) is not list)
        pm_tup = copy.deepcopy(self.payload)
        pm_tup["prior_art_cs13_anchors"] = tuple(pm_tup["prior_art_cs13_anchors"])
        with self.assertRaises(Err):
            oracle._validate_payload_nonreplay_schema(pm_tup)

        # Direct secondary_audit_payload & helper killers
        with self.assertRaises(Err):
            audit.secondary_audit_payload("not_dict")
        with self.assertRaises(Err):
            audit.secondary_audit_payload({})
        with self.assertRaises(Err):
            audit._require_list_exact_len(("a", "b"), "x", 2, "ERR_LIST")
        with self.assertRaises(Err):
            audit._require_list_exact_len(["a"], "x", 2, "ERR_LIST")

        # >= 30 re-signed full_replay=False tamper kills covering every pillar, nested object, list & flag!
        resigned_tamper_mutators = [
            lambda p: p.__setitem__("prior_art_cs13_anchors", ["a", "b", "c", "d", "e"]),
            lambda p: p.__setitem__("measured_live_gbench_serving_endpoint", True),
            lambda p: p.__setitem__("zero_bare_asserts", False),
            lambda p: p["pillar1_fisher_waterfilling_amgm"].__setitem__("extra", 1),
            lambda p: p["pillar1_fisher_waterfilling_amgm"].__setitem__("heavy_ratio_exceeds_twelve", False),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"].__setitem__("extra", 1),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"].__setitem__("kkt_equalization_exact", False),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["mild_skew"].__setitem__("improvement_ratio", "2"),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"].__setitem__("b_star", ["4"] * 8),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"].__setitem__("perturbation_witnesses", []),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"]["perturbation_witnesses"].__setitem__(0, []),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"]["perturbation_witnesses"][0].__setitem__("extra", 1),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"]["perturbation_witnesses"][0].__setitem__("strictly_worse_than_fisher", False),
            lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"]["perturbation_witnesses"][0].__setitem__("excess_kl", "0"),
            lambda p: p["pillar2_kintsugi_margin_vs_kl"].__setitem__("cs20_average_kl_blindness_certified", False),
            lambda p: p["pillar2_kintsugi_margin_vs_kl"]["quadratic_fisher_witness"].__setitem__("fisher_matrix_rank", 4),
            lambda p: p["pillar2_kintsugi_margin_vs_kl"]["quadratic_fisher_witness"]["uniform_a"].__setitem__("task_pass", True),
            lambda p: p["pillar2_kintsugi_margin_vs_kl"]["quadratic_fisher_witness"]["kintsugi_b"].__setitem__("average_fisher_kl", "1/1000"),
            lambda p: p["pillar2_kintsugi_margin_vs_kl"]["exact_dyadic_shannon_witness"].__setitem__("uniform_a_task_pass", True),
            lambda p: p["pillar2_kintsugi_margin_vs_kl"]["exact_dyadic_shannon_witness"].__setitem__("kintsugi_b_avg_kl_bits", "1/64"),
            lambda p: p["pillar3_gbench_p99_order_statistics"].__setitem__("sharp_99pct_threshold_n", 458),
            lambda p: p["pillar3_gbench_p99_order_statistics"].__setitem__("cs18_uncertified_floor_enforced", False),
            lambda p: p["pillar3_gbench_p99_order_statistics"]["cohort_n458"].__setitem__("pass_99pct_upper_tail", True),
            lambda p: p["pillar3_gbench_p99_order_statistics"]["cohort_n1099"].__setitem__("reciprocal_miss_prob_floor", 59999),
            lambda p: p["pillar3_gbench_p99_order_statistics"]["n1099_two_sided_order_stat_bracket"].__setitem__("coverage_ppm_lo", 900000),
            lambda p: p["pillar3_gbench_p99_order_statistics"]["n1099_two_sided_order_stat_bracket"].__setitem__("order_stat_upper_rank", 1095),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["capacity_obstruction"].__setitem__("l4_zero_error_exact", False),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["capacity_obstruction"].__setitem__("universal_cauchy_schwarz_lb", "0"),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"].__setitem__("poisoned_margin", "1"),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"].__setitem__("clean_state_sha256", p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"]["poisoned_state_sha256"]),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"].update({
                "clean_state_sha256": p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"]["poisoned_state_sha256"],
                "poisoned_state_sha256": p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"]["clean_state_sha256"],
            }),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"].__setitem__("clean_state_sha256", "a" * 64),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["channel_2b_poisoning_witness"].__setitem__("poisoned_state_sha256", "b" * 64),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["four_channel_airlock"]["clean_record"].__setitem__("airlock_4ch_decision", "DENY"),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["four_channel_airlock"]["negative_controls"].__setitem__(0, []),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["four_channel_airlock"]["negative_controls"][0].__setitem__("extra", 1),
            lambda p: p["pillar4_memt_gdn2_rank_and_channel2b"]["four_channel_airlock"]["negative_controls"][2].__setitem__("legacy_3ch_decision", "DENY"),
        ]
        self.assertGreaterEqual(len(resigned_tamper_mutators), 30)
        for fn in resigned_tamper_mutators:
            pm = copy.deepcopy(self.payload)
            fn(pm)
            with self.assertRaises(Err):
                audit.verify_certificate_dict(_resign(pm), full_replay=False)

        # Direct non-dict / extra-key killer on secondary_audit_payload for sub-objects & top-level
        for fn_direct in (
            lambda p: p.__setitem__("extra_top", 1),
            lambda p: p.__setitem__("pillar1_fisher_waterfilling_amgm", []),
            lambda p: p["pillar1_fisher_waterfilling_amgm"].__setitem__("uniform_control", []),
        ):
            pm = copy.deepcopy(self.payload)
            fn_direct(pm)
            with self.assertRaises(Err):
                audit.secondary_audit_payload(pm)

    # -----------------------------------------------------------------------
    # 4. R2-Grade Zone-B File I/O Security, Concurrency & CLI Traps
    # -----------------------------------------------------------------------

    def test_09_zone_b_io_hardening_streaming_hardlinks_and_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cert_path = tdp / "cert.json"

            wres = audit.write_certificate_file(cert_path)
            self.assertIs(wres["written"], True)
            self.assertEqual(stat.S_IMODE(cert_path.stat().st_mode), 0o600)
            wres2 = audit.write_certificate_file(cert_path)
            self.assertIs(wres2["written"], True)
            self.assertEqual(audit.require_safe_path(cert_path), cert_path)
            self.assertEqual(audit.require_safe_path(tdp / "nonexistent.json"), tdp / "nonexistent.json")
            self.assertEqual(audit.require_safe_path(Path("/")), Path("/"))
            cres = audit.check_certificate_file(cert_path)
            self.assertIs(cres["valid"], True)
            self.assertIs(cres["secondary_audit"]["secondary_audit_passed"], True)

            old_cwd = os.getcwd()
            old_pwd = os.environ.get("PWD")
            try:
                os.chdir(td)
                os.environ["PWD"] = td
                self.assertEqual(audit.require_safe_path(Path("cert.json")), Path("cert.json"))
                sym_dir = tdp / "symdir"
                os.symlink(tdp, sym_dir)
                os.environ["PWD"] = str(sym_dir)
                with self.assertRaises(Err):
                    audit.require_safe_path(Path("cert.json"))
                os.environ["PWD"] = str(tdp / "no_such_dir")
                self.assertEqual(audit.require_safe_path(Path("cert.json")), Path("cert.json"))
            finally:
                os.chdir(old_cwd)
                if old_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = old_pwd

            with self.assertRaises(Err):
                audit.require_safe_path("cert.json")
            with self.assertRaises(Err):
                audit.require_safe_path(_SubPath(cert_path))
            with self.assertRaises(Err):
                audit.require_safe_path(Path(f"{td}/a\x00b"))
            for bad_data in ("not_bytes", bytearray(b"x")):
                with self.assertRaises(Err) as cm_wd:
                    audit.write_regular_file_bytes(tdp / "missing_dir" / "out.json", bad_data)  # type: ignore[arg-type]
                self.assertEqual(cm_wd.exception.code, "INVALID_WRITE_DATA")
            with self.assertRaises(Err):
                audit.read_regular_bytes_capped(Path("/"))
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(Path("/"), b"{}")
            with self.assertRaises(Err):
                audit.read_regular_bytes_capped(tdp)
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(tdp, b"{}")

            leaf_sym = tdp / "leaf_sym.json"
            os.symlink(cert_path, leaf_sym)
            for fn_call in (
                lambda: audit.require_safe_path(leaf_sym),
                lambda: audit.read_regular_bytes_capped(leaf_sym),
                lambda: audit.write_regular_file_bytes(leaf_sym, b"{}\n"),
            ):
                with self.assertRaises(Err) as cm_s:
                    fn_call()
                self.assertEqual(cm_s.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(Err) as cm_noent:
                audit.read_regular_bytes_capped(tdp / "missing.json")
            self.assertEqual(cm_noent.exception.code, "FILE_READ_ERROR")
            inter_sym = tdp / "inter_sym"
            os.symlink(tdp, inter_sym)
            with self.assertRaises(Err) as cm_is:
                audit.read_regular_bytes_capped(inter_sym / "cert.json")
            self.assertEqual(cm_is.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(Err) as cm_md:
                audit.read_regular_bytes_capped(tdp / "no_dir" / "cert.json")
            self.assertEqual(cm_md.exception.code, "PATH_NOT_FOUND")

            hl_target = tdp / "hl.json"
            os.link(cert_path, hl_target)
            with self.assertRaises(Err) as cm_hl:
                audit.write_regular_file_bytes(hl_target, b"{}\n")
            self.assertEqual(cm_hl.exception.code, "HARDLINK_FORBIDDEN")
            os.unlink(hl_target)

            fifo_p = tdp / "p.fifo"
            os.mkfifo(fifo_p)
            with self.assertRaises(Err):
                audit.read_regular_bytes_capped(fifo_p)
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(fifo_p, b"{}\n")

            stream_p = tdp / "stream140k.bin"
            blob140 = b"z" * 140_000
            stream_p.write_bytes(blob140)
            self.assertEqual(audit.read_regular_bytes_capped(stream_p, 140_000), blob140)
            with self.assertRaises(Err) as cm_sz:
                audit.read_regular_bytes_capped(stream_p, 139_999)
            self.assertEqual(cm_sz.exception.code, "FILE_TOO_LARGE")
            with self.assertRaises(Err) as cm_cap:
                audit.read_regular_bytes_capped(stream_p, 1 << 100)
            self.assertEqual(cm_cap.exception.code, "INT_ABOVE_MAX")
            with self.assertRaises(Err):
                audit.read_regular_bytes_capped(stream_p, 0)

            for name, raw_b, exp_code in (
                ("crlf.json", cert_path.read_bytes().replace(b"\n", b"\r\n"), "CRLF_FORBIDDEN"),
                ("nolf.json", cert_path.read_bytes().rstrip(b"\n"), "MISSING_TRAILING_LF"),
                ("badutf.json", b"\xff\xfe\n", "INVALID_UTF8"),
                ("noncan.json", b" " + cert_path.read_bytes(), "NON_CANONICAL_BYTES"),
            ):
                p = tdp / name
                p.write_bytes(raw_b)
                with self.assertRaises(Err) as cm_c:
                    audit.check_certificate_file(p)
                self.assertEqual(cm_c.exception.code, exp_code)

            tn1 = audit._next_tmp_name("col.json")
            tn2 = audit._next_tmp_name("col.json")
            self.assertNotEqual(tn1, tn2)
            seq1 = oracle.parse_exact_int_ascii(tn1.split(".")[-2], "seq1")
            seq2 = oracle.parse_exact_int_ascii(tn2.split(".")[-2], "seq2")
            self.assertEqual(seq2, seq1 + 1)

            col_target = tdp / "col.json"
            (tdp / ".col.json.s42tmp.fixed0").write_bytes(b"occupy")
            with mock.patch.object(audit, "_next_tmp_name", side_effect=[".col.json.s42tmp.fixed0", ".col.json.s42tmp.fixed1"]):
                audit.write_regular_file_bytes(col_target, b"ok\n")
            self.assertEqual(col_target.read_bytes(), b"ok\n")

            (tdp / ".col2.json.s42tmp.occ").write_bytes(b"x")
            with mock.patch.object(audit, "_next_tmp_name", return_value=".col2.json.s42tmp.occ"):
                with self.assertRaises(Err) as cm_col2:
                    audit.write_regular_file_bytes(tdp / "col2.json", b"ok\n")
            self.assertEqual(cm_col2.exception.code, "TEMP_RETRY_EXHAUSTED")

            hl_tmp_target = tdp / "hl_tmp.json"
            real_fstat = os.fstat

            def _fake_fstat(fd: int) -> os.stat_result:
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, 2, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))

            with mock.patch.object(audit.os, "fstat", side_effect=_fake_fstat):
                with self.assertRaises(Err) as cm_ht:
                    audit.write_regular_file_bytes(hl_tmp_target, b"ok\n")
            self.assertEqual(cm_ht.exception.code, "HARDLINK_FORBIDDEN")
            self.assertEqual(list(tdp.glob(".hl_tmp.json.s42tmp.*")), [])

            conc_target = tdp / "conc.json"
            conc_errs: list[str] = []
            conc_lock = threading.Lock()
            conc_ok = 0

            def _w(tid: int) -> None:
                nonlocal conc_ok
                data_b = f'{{"tid":{tid}}}\n'.encode("ascii")
                for _ in range(2):
                    try:
                        audit.write_regular_file_bytes(conc_target, data_b)
                        with conc_lock:
                            conc_ok += 1
                    except Exception as exc:  # pragma: no cover
                        with conc_lock:
                            conc_errs.append(f"{tid}:{type(exc).__name__}:{exc}")

            ths = [threading.Thread(target=_w, args=(i,)) for i in range(16)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            self.assertEqual(conc_errs, [])
            self.assertEqual(conc_ok, 32)
            self.assertEqual(list(tdp.glob(".conc.json.s42tmp.*")), [])

            if os.geteuid() != 0:
                unr = tdp / "unr.json"
                unr.write_bytes(b"{}\n")
                os.chmod(unr, 0o000)
                try:
                    with self.assertRaises(Err):
                        audit.read_regular_bytes_capped(unr)
                finally:
                    os.chmod(unr, 0o600)
                unr_d = tdp / "unr_d"
                unr_d.mkdir()
                os.chmod(unr_d, 0o000)
                try:
                    with self.assertRaises(Err):
                        audit.read_regular_bytes_capped(unr_d / "x.json")
                    with self.assertRaises(Err):
                        audit.write_regular_file_bytes(unr_d / "x.json", b"{}\n")
                finally:
                    os.chmod(unr_d, 0o700)

    def test_10_cli_parsing_deleted_cwd_and_flag_traps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_path = tdp / "cli_cert.json"

            for argv_ok in (
                [],
                ["--standalone"],
                ["--standalone", "--json"],
                ["--write", str(out_path), "--json"],
                ["--check", str(out_path)],
                ["--check", str(out_path), "--json"],
            ):
                buf_out = io.StringIO()
                buf_err = io.StringIO()
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    sys.stdout, sys.stderr = buf_out, buf_err
                    rc = audit.main(argv_ok)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                self.assertEqual(rc, 0)
                self.assertEqual(buf_err.getvalue(), "")

            bad_argvs: list[object] = [
                ["--write", "--json"],
                ["--write=--json"],
                ["--check=--json"],
                ["--check", "--json"],
                ["--json", "--json"],
                ["--standalone", "--standalone"],
                ["--standalone", "--check", str(out_path)],
                ["--write"],
                ["--check"],
                ["--write", ""],
                ["--stand"],
                ["--js"],
                ["--unknown"],
                ["--write", "bad\ud800.json"],
                "not_a_list",
            ]
            old_cwd = os.getcwd()
            os.chdir(td)
            try:
                for bad in bad_argvs:
                    buf_out = io.StringIO()
                    buf_err = io.StringIO()
                    old_out, old_err = sys.stdout, sys.stderr
                    try:
                        sys.stdout, sys.stderr = buf_out, buf_err
                        rc = audit.main(bad)  # type: ignore[arg-type]
                    finally:
                        sys.stdout, sys.stderr = old_out, old_err
                    self.assertEqual(rc, 2)
                    err_obj = json.loads(buf_err.getvalue())
                    self.assertIs(err_obj["valid"], False)
                    if bad == "not_a_list":
                        self.assertEqual(err_obj["error_code"], "INVALID_ARGV")
                    if bad == ["--write", ""]:
                        self.assertEqual(err_obj["error_code"], "INVALID_PATH_ARG")
                self.assertFalse((tdp / "--json").exists())
            finally:
                os.chdir(old_cwd)

            del_dir = tdp / "vanishing"
            del_dir.mkdir()
            script = (
                "import os, sys; "
                f"sys.path.insert(0, {str(_THIS_DIR)!r}); "
                "import fisher_kintsugi_memt_audit as a; "
                f"os.chdir({str(del_dir)!r}); os.rmdir({str(del_dir)!r}); "
                "raise SystemExit(a.main(['--check', 'rel.json']))"
            )
            proc = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("CWD_UNAVAILABLE", proc.stderr)

            kallsyms = Path("/proc/kallsyms")
            if kallsyms.exists():
                with self.assertRaises(Err) as cm_proc:
                    audit.read_regular_bytes_capped(kallsyms)
                self.assertEqual(cm_proc.exception.code, "FILE_TOO_LARGE")

    def test_11_disk_certificate_matches_standalone(self) -> None:
        disk_cert = _THIS_DIR / "fisher_kintsugi_memt_certificate.json"
        if disk_cert.exists():
            res = audit.check_certificate_file(disk_cert)
            self.assertIs(res["valid"], True)
            self.assertEqual(res["payload_sha256"], self.cert["payload_sha256"])
            self.assertEqual(res["certificate_sha256"], self.cert["certificate_sha256"])

    def test_12_zero_denominator_secondary_audit_typed_rejection(self) -> None:
        Err = oracle.FisherKintsugiMemtError

        def _resign_local(payload_mut: dict[str, object]) -> dict[str, object]:
            p_sha = oracle.hashlib.sha256(oracle.canonical_json_bytes(payload_mut)).hexdigest()
            core = {"schema": oracle.SCHEMA_VERSION, "track_id": oracle.TRACK_ID, "payload_sha256": p_sha, "payload": payload_mut}
            c_sha = oracle.hashlib.sha256(oracle.canonical_json_bytes(core)).hexdigest()
            return {"schema": oracle.SCHEMA_VERSION, "track_id": oracle.TRACK_ID, "payload_sha256": p_sha, "certificate_sha256": c_sha, "payload": payload_mut}

        cases: list[tuple[str, object, str]] = [
            ("p1_heavy_kf", lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"].__setitem__("k_fisher_total", "0"), "AUDIT_P1_SPEC_VAL"),
            ("p1_heavy_gm", lambda p: p["pillar1_fisher_waterfilling_amgm"]["heavy_outlier_gemma4_scale"].__setitem__("gm_weights", "0"), "AUDIT_P1_SPEC_VAL"),
            ("p1_mild_kf", lambda p: p["pillar1_fisher_waterfilling_amgm"]["mild_skew"].__setitem__("k_fisher_total", "0"), "AUDIT_P1_SPEC_VAL"),
            ("p1_mild_gm", lambda p: p["pillar1_fisher_waterfilling_amgm"]["mild_skew"].__setitem__("gm_weights", "0"), "AUDIT_P1_SPEC_VAL"),
            ("p1_uni_kf", lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"].__setitem__("k_fisher_total", "0"), "AUDIT_P1_SPEC_VAL"),
            ("p1_uni_gm", lambda p: p["pillar1_fisher_waterfilling_amgm"]["uniform_control"].__setitem__("gm_weights", "0"), "AUDIT_P1_SPEC_VAL"),
            ("p2_qa_avg", lambda p: p["pillar2_kintsugi_margin_vs_kl"]["quadratic_fisher_witness"]["uniform_a"].__setitem__("average_fisher_kl", "0"), "AUDIT_P2_FISHER_VAL"),
            ("p2_da_avg", lambda p: p["pillar2_kintsugi_margin_vs_kl"]["exact_dyadic_shannon_witness"].__setitem__("uniform_a_avg_kl_bits", "0"), "AUDIT_P2_DYADIC_VAL"),
        ]
        for label, mutator, expected_code in cases:
            with self.subTest(case=label):
                pm = copy.deepcopy(self.payload)
                mutator(pm)  # type: ignore[operator]
                bad = _resign_local(pm)
                with self.assertRaises(Err) as cm1:
                    audit.verify_certificate_dict(bad, full_replay=False)
                self.assertEqual(cm1.exception.code, expected_code)
                with self.assertRaises(Err) as cm2:
                    audit.secondary_audit_payload(bad["payload"])
                self.assertEqual(cm2.exception.code, expected_code)


if __name__ == "__main__":
    unittest.main()
