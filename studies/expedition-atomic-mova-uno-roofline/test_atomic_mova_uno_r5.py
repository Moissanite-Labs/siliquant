#!/usr/bin/env python3
"""Track S45 R5 Test Suite: 4 Mathematical/Systems Pillars, Exported-Helper Soundness,
100% Mutation/BoolOp Killers, 42 Re-Signed Secondary-Audit Tampers & R2-Grade Zone-B Security.

Pure Python 3 standard library: zero AST floats, zero bare asserts, zero mathematical RNG.
"""
from __future__ import annotations

import ast
import copy
from fractions import Fraction
import hashlib
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

import atomic_mova_uno_oracle as oracle
import atomic_mova_uno_audit as audit

Err = oracle.AtomicMovaUnoError


class _SubInt(int):
    pass


class _SubStr(str):
    pass


class _SubFrac(Fraction):
    pass


class _SubPath(type(Path("."))):
    pass


class _SubList(list):
    pass


def _resign_cert(cert: dict[str, object]) -> dict[str, object]:
    """Recompute payload_sha256 and certificate_sha256 after tampering payload (tests full_replay=False)."""
    payload = cert["payload"]
    p_sha = hashlib.sha256(oracle.canonical_json_bytes(payload)).hexdigest()
    cert["payload_sha256"] = p_sha
    pre = {
        "schema": cert["schema"],
        "track_id": cert["track_id"],
        "payload_sha256": p_sha,
        "payload": payload,
    }
    cert["certificate_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(pre)).hexdigest()
    return cert


class TestTrackS45AtomicMovaUnoR5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.build_standalone_certificate()
        cls.payload = cls.cert["payload"]

    # -----------------------------------------------------------------------
    # 0. AST Purity Gate (Zero Floats, Zero Bare Asserts, Zero min/max/islink/isdigit)
    # -----------------------------------------------------------------------

    def test_00_ast_purity_zero_floats_asserts_min_max(self) -> None:
        for fname in (
            "atomic_mova_uno_oracle.py",
            "atomic_mova_uno_audit.py",
            "test_atomic_mova_uno_r5.py",
        ):
            src = (_THIS_DIR / fname).read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert, f"bare assert in {fname}")
                if isinstance(node, ast.Constant):
                    self.assertNotIsInstance(node.value, float, f"float literal in {fname}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("float", "min", "max"), f"forbidden {node.func.id}() in {fname}")
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ("islink", "isdigit"), f"forbidden .{node.attr} in {fname}")

    # -----------------------------------------------------------------------
    # 1. Four Mathematical & Systems Pillars
    # -----------------------------------------------------------------------

    def test_01_pillar1_atomic_anchors_and_zipf_ssd_roofline(self) -> None:
        self.assertEqual(self.payload["novelty_tier"], "N1")
        self.assertEqual(self.payload["published_benchmarks_novelty"], "N0")
        self.assertEqual(self.payload["oracle_class"], "exact")
        self.assertIs(self.payload["measured_live_m4_max_hardware_run"], False)
        self.assertIs(self.payload["measured_live_k2_horizon_checkpoint_eval"], False)
        self.assertIs(self.payload["zero_trust_path_floats"], True)
        self.assertIs(self.payload["zero_bare_asserts"], True)
        self.assertIs(self.payload["zero_mathematical_rng"], True)

        p1 = self.payload["pillar1_atomic_ssd_roofline"]
        anc = p1["m4_max_inkling2_mini_anchors_n0"]
        self.assertEqual(anc["atomic_vs_stock_tps_ratio"], "10953/3311")
        self.assertEqual(anc["atomic_vs_stock_tps_4dp"], "3.3080")
        self.assertIs(anc["atomic_exceeds_33_over_10"], True)
        self.assertEqual(anc["atomic_vs_stock_ttft_ratio"], "709/957")
        self.assertIs(anc["atomic_ttft_below_3_over_4"], True)
        self.assertEqual(anc["strata_vs_atomic_tps_ratio"], "3764/3651")
        self.assertEqual(anc["strata_vs_atomic_ttft_speedup"], "709/421")

        rf = p1["zipf_ssd_roofline_n1"]
        self.assertEqual(rf["h_16"], "2436559/720720")
        self.assertEqual((rf["ssd_dominance_boundary_lo_gib"], rf["ssd_dominance_boundary_hi_gib"]), (9, 10))
        self.assertIs(rf["latency_strictly_decreasing_to_18gib"], True)
        self.assertIs(rf["full_residency_plateau_18_to_20gib"], True)
        self.assertIs(rf["resize_20_to_8_to_20_reversible"], True)
        self.assertIs(rf["resize_20gib_ssd_vanishes"], True)
        self.assertIs(rf["resize_8gib_ssd_dominated"], True)
        self.assertEqual(len(rf["sweep"]), 9)
        self.assertIs(rf["sweep"][3]["ssd_miss_dominated"], True)   # 9 GiB
        self.assertIs(rf["sweep"][4]["ssd_miss_dominated"], False)  # 10 GiB
        self.assertIs(rf["sweep"][8]["ssd_latency_vanishes"], True)  # 20 GiB

    def test_02_pillar2_budget_crossover_theorem(self) -> None:
        p2 = self.payload["pillar2_budget_crossover"]
        ds = p2["external_dspark_spec"]
        self.assertEqual((ds["lookahead_k"], ds["draft_memory_gib"]), (5, 4))
        self.assertEqual(ds["renewal_speedup"], "3843/1250")
        self.assertEqual(ds["renewal_speedup_4dp"], "3.0744")
        self.assertIs(ds["exceeds_3x"], True)
        self.assertIs(ds["geometric_closed_form_matches"], True)

        uno = p2["zero_draft_uno_spec"]
        self.assertEqual((uno["draft_memory_gib"], uno["lossless_speedup"]), (0, "11/5"))

        self.assertEqual(p2["exact_crossover_b_star_gib"], "3625767011/304898880")
        self.assertEqual(p2["exact_crossover_b_star_4dp"], "11.8917")
        self.assertIs(p2["crossover_residual_zero"], True)
        self.assertIs(p2["crossover_strictly_between_11_and_12"], True)
        self.assertIs(p2["uno_strictly_wins_all_below_b_star"], True)
        self.assertIs(p2["dspark_strictly_wins_all_above_b_star"], True)
        pts = p2["operating_points"]
        self.assertEqual([pt["winner"] for pt in pts[:4]], ["zero_draft_uno"] * 4)
        self.assertEqual([pt["winner"] for pt in pts[4:]], ["external_dspark"] * 4)

    def test_03_pillar3_mova_fisher_and_kintsugi_hazard(self) -> None:
        p3 = self.payload["pillar3_mova_fisher_kintsugi"]
        ma = p3["k2_horizon_mova_anchors_n0"]
        self.assertEqual((ma["mova_total_params_b"], ma["mova_active_params_b"], ma["dense_baseline_params_b"]), (36, 4, 32))
        self.assertEqual(ma["reported_matched_quality_active_fraction"], "1/7")

        sp = p3["softplus_gating_probes"]
        self.assertEqual(sp["sigma_derivatives"], ["1/5", "1/3", "1/2", "2/3", "4/5"])
        self.assertIs(sp["strictly_in_open_unit_interval"], True)
        self.assertIs(sp["strictly_increasing"], True)
        self.assertIs(sp["zero_dead_region_on_negative_z"], True)

        self.assertEqual(p3["dense_v_waterfilling"]["gain_ratio"], "5/4")
        self.assertEqual(p3["mova_sparse_v_waterfilling"]["gain_ratio"], "263/16")
        self.assertEqual(p3["mova_sparse_v_waterfilling"]["gain_ratio_4dp"], "16.4375")
        self.assertIs(p3["mova_exceeds_12x_kl_reduction"], True)
        self.assertEqual(p3["mova_vs_dense_leverage_multiplier"], "263/20")
        self.assertEqual(p3["convexity_perturbation_excess_kl"], "1/512")
        self.assertIs(p3["convexity_both_perturbations_strictly_worse"], True)

        kn = p3["kintsugi_outlier_expert_task_break"]
        self.assertEqual((kn["uniform_4bpw_post_margin"], kn["fisher_waterfilling_post_margin"]), ("-1/2", "7/8"))
        self.assertIs(kn["uniform_4bpw_task_broken"], True)
        self.assertIs(kn["fisher_waterfilling_task_preserved"], True)

    def test_04_pillar4_scope_ledger_and_64_token_parity_gate(self) -> None:
        p4 = self.payload["pillar4_scope_and_token_parity"]
        sc = p4["honest_scope_ledger"]
        self.assertEqual((sc["n0_source_count"], sc["n1_theorem_count"]), (3, 3))
        self.assertIs(sc["strict_n0_n1_separation_enforced"], True)
        self.assertIs(p4["gate_accepts_exact_64_of_64"], True)
        self.assertIs(p4["gate_rejects_single_token_drift"], True)
        self.assertEqual(p4["lossless_64_token_parity_positive"]["matched_tokens"], 64)
        self.assertIsNone(p4["lossless_64_token_parity_positive"]["first_mismatch_index"])
        self.assertEqual(p4["lossless_64_token_drift_negative"]["matched_tokens"], 63)
        self.assertEqual(p4["lossless_64_token_drift_negative"]["first_mismatch_index"], 63)

    # -----------------------------------------------------------------------
    # 2. Exported Oracle Primitives & Domain Helper Soundness
    # -----------------------------------------------------------------------

    def test_05_oracle_primitives_boundaries_and_rejections(self) -> None:
        self.assertEqual(oracle.require_exact_int(5, "x", min_val=5, max_val=5), 5)
        for bad in (True, _SubInt(5), "5"):
            with self.assertRaises(Err) as cm:
                oracle.require_exact_int(bad, "x")
            self.assertEqual(cm.exception.code, "INVALID_INT")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_int(4, "x", min_val=5)
        self.assertEqual(cm.exception.code, "INT_BELOW_MIN")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_int(6, "x", max_val=5)
        self.assertEqual(cm.exception.code, "INT_ABOVE_MAX")

        self.assertIs(oracle.require_exact_bool(False, "b"), False)
        with self.assertRaises(Err) as cm:
            oracle.require_exact_bool(1, "b")
        self.assertEqual(cm.exception.code, "INVALID_BOOL")

        self.assertEqual(oracle.require_exact_str("", "s", non_empty=False), "")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_str(_SubStr("a"), "s")
        self.assertEqual(cm.exception.code, "INVALID_STR")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_str("", "s")
        self.assertEqual(cm.exception.code, "EMPTY_STR")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_str("bad\ud800", "s")
        self.assertEqual(cm.exception.code, "INVALID_UTF8_STR")

        self.assertEqual(oracle.require_exact_fraction(Fraction(0, 1), "q", non_negative=True), Fraction(0, 1))
        self.assertEqual(oracle.require_exact_fraction(Fraction(1, 2), "q", positive=True), Fraction(1, 2))
        with self.assertRaises(Err) as cm:
            oracle.require_exact_fraction(_SubFrac(1, 2), "q")
        self.assertEqual(cm.exception.code, "INVALID_FRACTION")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_fraction(Fraction(0, 1), "q", positive=True)
        self.assertEqual(cm.exception.code, "FRACTION_NOT_POSITIVE")
        with self.assertRaises(Err) as cm:
            oracle.require_exact_fraction(Fraction(-1, 2), "q", non_negative=True)
        self.assertEqual(cm.exception.code, "FRACTION_NEGATIVE")

        good_hex = "a" * 64
        self.assertEqual(oracle.require_sha256_hex(good_hex, "h"), good_hex)
        for bad_hex in ("a" * 63, "a" * 65, "A" * 64, "g" * 64):
            with self.assertRaises(Err) as cm:
                oracle.require_sha256_hex(bad_hex, "h")
            self.assertEqual(cm.exception.code, "INVALID_SHA256")

        # Binding MAX_INT_BITS = 3321 (|n| <= 2^3321 - 1, exact 1000 digits) + pre-parse MAX_INT_DIGITS = 1000
        max_bit_int = (1 << oracle.MAX_INT_BITS) - 1
        max_bit_str = str(max_bit_int)
        self.assertEqual(len(max_bit_str), oracle.MAX_INT_DIGITS)
        self.assertEqual(oracle.parse_exact_int_ascii(max_bit_str, "n"), max_bit_int)
        self.assertEqual(oracle.parse_exact_int_ascii("-" + max_bit_str, "n"), -max_bit_int)
        self.assertEqual(oracle.parse_exact_int_ascii("0", "n"), 0)
        for tok, code in (
            ("-", "INVALID_INT_TOKEN"),
            ("9" * (oracle.MAX_INT_DIGITS + 1), "INT_TOKEN_TOO_LONG"),
            ("01", "LEADING_ZERO_INT"),
            ("1\u0662", "NON_ASCII_DIGIT"),
            ("-0", "NEGATIVE_ZERO_INT"),
            (str(1 << oracle.MAX_INT_BITS), "INT_TOO_LARGE"),
            ("9" * oracle.MAX_INT_DIGITS, "INT_TOO_LARGE"),
        ):
            with self.assertRaises(Err) as cm:
                oracle.parse_exact_int_ascii(tok, "n")
            self.assertEqual(cm.exception.code, code)

        self.assertEqual(oracle.parse_exact_rational("-3/4", "r"), Fraction(-3, 4))
        self.assertEqual(oracle.parse_exact_rational("7", "r"), Fraction(7, 1))
        for bad_r, code in (
            ("1/2/3", "INVALID_RATIONAL"),
            ("3/1", "NON_CANONICAL_DEN"),
            ("3/0", "NON_CANONICAL_DEN"),
            ("2/4", "UNREDUCED_RATIONAL"),
        ):
            with self.assertRaises(Err) as cm:
                oracle.parse_exact_rational(bad_r, "r")
            self.assertEqual(cm.exception.code, code)

        self.assertEqual(oracle.format_exact_rational(Fraction(5, 1)), "5")
        self.assertEqual(oracle.format_exact_rational(Fraction(-5, 3)), "-5/3")
        with self.assertRaises(Err):
            oracle.format_exact_rational(Fraction(1 << oracle.MAX_INT_BITS, 1))
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-1, 1000), 2), "0.00")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-1, 1000), 0), "0")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(-7, 2), 2), "-3.50")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(7, 2), 0), "3")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(1, 3), 18), "0.333333333333333333")
        with self.assertRaises(Err):
            oracle.format_decimal_trunc(Fraction(1, 3), 19)

        ok_nested: object = 1
        for _ in range(oracle.MAX_JSON_DEPTH):
            ok_nested = [ok_nested]
        oracle.validate_json_tree_iterative(ok_nested)
        with self.assertRaises(Err) as cm:
            oracle.validate_json_tree_iterative([ok_nested])
        self.assertEqual(cm.exception.code, "JSON_TOO_DEEP")
        oracle.validate_json_tree_iterative({"ok": [True, None, max_bit_int, "s"]})
        with self.assertRaises(Err) as cm:
            oracle.validate_json_tree_iterative({"bad": (1, 2)})
        self.assertEqual(cm.exception.code, "FORBIDDEN_JSON_TYPE")

        big_s = "a" * (oracle.MAX_JSON_BYTES - 2)
        self.assertEqual(len(oracle.canonical_json_bytes(big_s)), oracle.MAX_JSON_BYTES)
        with self.assertRaises(Err) as cm:
            oracle.canonical_json_bytes("a" * (oracle.MAX_JSON_BYTES - 1))
        self.assertEqual(cm.exception.code, "JSON_TOO_LARGE")

        big_json_ok = '{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}'
        self.assertEqual(len(big_json_ok.encode("utf-8")), oracle.MAX_JSON_BYTES)
        self.assertEqual(type(oracle.parse_strict_json_text(big_json_ok)), dict)
        with self.assertRaises(Err) as cm:
            oracle.parse_strict_json_text(big_json_ok + " ")
        self.assertEqual(cm.exception.code, "JSON_TOO_LARGE")
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])
        for bad_j, code in (
            ('{"x":1.5}', "FLOAT_FORBIDDEN"),
            ('{"x":1e2}', "FLOAT_FORBIDDEN"),
            ('{"x":NaN}', "NON_FINITE_FORBIDDEN"),
            ('{"a":1,"a":2}', "DUPLICATE_JSON_KEY"),
            ('{"a":', "INVALID_JSON"),
            ("[1,2]", "JSON_ROOT_NOT_OBJECT"),
            ('{"n":' + "9" * 4300 + "}", "INT_TOKEN_TOO_LONG"),
        ):
            with self.assertRaises(Err) as cm:
                oracle.parse_strict_json_text(bad_j)
            self.assertEqual(cm.exception.code, code)

    def test_06_domain_helpers_boundaries_and_rejections(self) -> None:
        self.assertEqual(oracle.harmonic_q(0), Fraction(0, 1))
        self.assertEqual(oracle.harmonic_q(1), Fraction(1, 1))
        self.assertEqual(oracle.zipf_cache_hit_rate(0, 16), Fraction(0, 1))
        self.assertEqual(oracle.zipf_cache_hit_rate(16, 16), Fraction(1, 1))
        with self.assertRaises(Err):
            oracle.zipf_cache_hit_rate(17, 16)

        # ssd_streaming_token_latency_ms below base_gib (avail < 0 -> c_cached = 0) and exact plateau
        r_low = oracle.ssd_streaming_token_latency_ms(0)
        self.assertEqual((r_low["cached_experts"], r_low["t_tok_ms"]), (0, "49"))

        # speculative_renewal_speedup exact boundaries (alpha < 1 passes, alpha == 1 fails)
        self.assertEqual(oracle.speculative_renewal_speedup(1, Fraction(1, 2), Fraction(0, 1)), Fraction(3, 2))
        with self.assertRaises(Err) as cm:
            oracle.speculative_renewal_speedup(1, Fraction(1, 1), Fraction(0, 1))
        self.assertEqual(cm.exception.code, "INVALID_ACCEPTANCE")

        # piecewise_linear_latency_ms exact branches (<=2, in (2,18), >=18)
        self.assertEqual(oracle.piecewise_linear_latency_ms(Fraction(0, 1)), Fraction(49, 1))
        self.assertEqual(oracle.piecewise_linear_latency_ms(Fraction(2, 1)), Fraction(49, 1))
        self.assertEqual(oracle.piecewise_linear_latency_ms(Fraction(18, 1)), Fraction(9, 1))
        self.assertEqual(oracle.piecewise_linear_latency_ms(Fraction(25, 1)), Fraction(9, 1))

        # softplus_derivative_at_exp_q
        self.assertEqual(oracle.softplus_derivative_at_exp_q(Fraction(1, 1)), Fraction(1, 2))
        with self.assertRaises(Err):
            oracle.softplus_derivative_at_exp_q(Fraction(0, 1))

        # fisher_waterfilling_kl rejections (type, empty, non-divisible mean)
        for bad_exps, code in (
            ([], "INVALID_EXPONENTS"),
            (_SubList([0, 0]), "INVALID_EXPONENTS"),
            ("ab", "INVALID_EXPONENTS"),
            ([1, 0], "NON_INTEGER_MEAN_EXP"),
        ):
            with self.assertRaises(Err) as cm:
                oracle.fisher_waterfilling_kl(bad_exps, 4)
            self.assertEqual(cm.exception.code, code)

        # verify_greedy_token_parity (multi-mismatch branch + type/length/vocab bounds)
        p_multi = oracle.verify_greedy_token_parity([1, 2, 3, 4], [9, 8, 3, 4], expected_len=4)
        self.assertEqual((p_multi["matched_tokens"], p_multi["first_mismatch_index"], p_multi["exact_match"]), (2, 0, False))
        for bad_ref, bad_cand in (
            (_SubList([0, 1, 2, 3]), [0, 1, 2, 3]),
            ([0, 1, 2], [0, 1, 2, 3]),
            ([0, 1, 2, 3], _SubList([0, 1, 2, 3])),
            ([0, 1, 2, 3], [0, 1, 2]),
        ):
            with self.assertRaises(Err) as cm:
                oracle.verify_greedy_token_parity(bad_ref, bad_cand, expected_len=4)
            self.assertEqual(cm.exception.code, "INVALID_TOKEN_SEQ")
        with self.assertRaises(Err):
            oracle.verify_greedy_token_parity([128_000, 0, 0, 0], [0, 0, 0, 0], expected_len=4)

    # -----------------------------------------------------------------------
    # 3. Certificate Envelope & 45 Re-Signed Secondary-Audit Tampers (full_replay=False)
    # -----------------------------------------------------------------------

    def test_07_certificate_envelope_and_45_resigned_secondary_audit_kills(self) -> None:
        res_full = audit.verify_certificate_dict(copy.deepcopy(self.cert), full_replay=True)
        self.assertIs(res_full["valid"], True)
        self.assertIs(res_full["full_replay_matched"], True)
        self.assertIs(res_full["secondary_audit"]["secondary_audit_passed"], True)

        res_sec = audit.verify_certificate_dict(copy.deepcopy(self.cert), full_replay=False)
        self.assertIs(res_sec["valid"], True)
        self.assertIs(res_sec["full_replay_matched"], False)

        # Top-level envelope rejections
        with self.assertRaises(Err) as cm:
            oracle.validate_certificate_payload([])
        self.assertEqual(cm.exception.code, "INVALID_CERT_TYPE")
        c = copy.deepcopy(self.cert); c["extra"] = 1
        with self.assertRaises(Err) as cm:
            oracle.validate_certificate_payload(c)
        self.assertEqual(cm.exception.code, "INVALID_CERT_KEYS")
        c = copy.deepcopy(self.cert); c["schema"] = "wrong"
        with self.assertRaises(Err) as cm:
            oracle.validate_certificate_payload(c)
        self.assertEqual(cm.exception.code, "SCHEMA_MISMATCH")
        c = copy.deepcopy(self.cert); c["track_id"] = "wrong"
        with self.assertRaises(Err) as cm:
            oracle.validate_certificate_payload(c)
        self.assertEqual(cm.exception.code, "TRACK_ID_MISMATCH")
        for bad_pl in ([], {"only": 1}):
            c = copy.deepcopy(self.cert); c["payload"] = bad_pl
            with self.assertRaises(Err) as cm:
                oracle.validate_certificate_payload(c)
            self.assertEqual(cm.exception.code, "INVALID_PAYLOAD_KEYS")
        c = copy.deepcopy(self.cert); c["payload_sha256"] = "0" * 64
        with self.assertRaises(Err) as cm:
            oracle.validate_certificate_payload(c)
        self.assertEqual(cm.exception.code, "PAYLOAD_SHA256_MISMATCH")
        c = copy.deepcopy(self.cert); c["certificate_sha256"] = "0" * 64
        with self.assertRaises(Err) as cm:
            oracle.validate_certificate_payload(c)
        self.assertEqual(cm.exception.code, "CERTIFICATE_SHA256_MISMATCH")

        # Direct secondary_audit_payload & internal list/vector guards
        for bad_pl in ([], {"only": 1}):
            with self.assertRaises(Err) as cm:
                audit.secondary_audit_payload(bad_pl)
            self.assertEqual(cm.exception.code, "AUDIT_INVALID_PAYLOAD")
        with self.assertRaises(Err) as cm:
            audit._require_list_len({"k": (1, 2)}, "k", 2, "BAD_LIST")
        self.assertEqual(cm.exception.code, "BAD_LIST")
        with self.assertRaises(Err) as cm:
            audit._parse_q_vec(["1"], 2, "v", "BAD_QLEN")
        self.assertEqual(cm.exception.code, "BAD_QLEN")
        with self.assertRaises(Err) as cm:
            audit._parse_q_vec(("1", "2"), 2, "v", "BAD_QTYPE")
        self.assertEqual(cm.exception.code, "BAD_QTYPE")
        with self.assertRaises(Err) as cm:
            audit._parse_int_vec([1], 2, "v", "BAD_ILEN")
        self.assertEqual(cm.exception.code, "BAD_ILEN")
        with self.assertRaises(Err) as cm:
            audit._parse_int_vec((1, 2), 2, "v", "BAD_ITYPE")
        self.assertEqual(cm.exception.code, "BAD_ITYPE")

        # 45 re-signed semantic & nested-key tampers killed under BOTH full_replay=True & full_replay=False!
        tamper_count = 0

        def _tamper(mutator, expected_code: str) -> None:
            nonlocal tamper_count
            tamper_count += 1
            tc = copy.deepcopy(self.cert)
            mutator(tc["payload"])
            _resign_cert(tc)
            with self.assertRaises(Err) as cm_t:
                audit.verify_certificate_dict(tc, full_replay=True)
            self.assertEqual(cm_t.exception.code, "FULL_REPLAY_MISMATCH")
            with self.assertRaises(Err) as cm_f:
                audit.verify_certificate_dict(tc, full_replay=False)
            self.assertEqual(cm_f.exception.code, expected_code)

        # Scope metadata & top-level pillar schemas (6)
        _tamper(lambda p: p.__setitem__("novelty_tier", "N2"), "AUDIT_SCOPE_META")
        _tamper(lambda p: p.__setitem__("published_benchmarks_novelty", "N1"), "AUDIT_SCOPE_META")
        _tamper(lambda p: p.__setitem__("measured_live_m4_max_hardware_run", True), "AUDIT_MEASURED_FLAG")
        _tamper(lambda p: p.__setitem__("zero_trust_path_floats", False), "AUDIT_ZERO_FLAG")
        _tamper(lambda p: p.__setitem__("pillar1_atomic_ssd_roofline", []), "AUDIT_P1")
        _tamper(lambda p: p.__setitem__("pillar1_atomic_ssd_roofline", {"x": 1}), "AUDIT_P1")

        # Pillar 1 tampers (10)
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["m4_max_inkling2_mini_anchors_n0"].__setitem__("extra", 1), "AUDIT_P1_ANC")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["m4_max_inkling2_mini_anchors_n0"].__setitem__("atomic_exceeds_33_over_10", False), "AUDIT_P1_ANC_FLAG")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["m4_max_inkling2_mini_anchors_n0"].__setitem__("atomic_vs_stock_tps_4dp", "3.3081"), "AUDIT_P1_ANC_VAL")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"].__setitem__("extra", 1), "AUDIT_P1_RF")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"].__setitem__("resize_20_to_8_to_20_reversible", False), "AUDIT_P1_RF_FLAG")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"].__setitem__("ssd_dominance_boundary_lo_gib", 8), "AUDIT_P1_RF_VAL")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"].__setitem__("sweep", []), "AUDIT_P1_SW")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"]["sweep"].__setitem__(0, []), "AUDIT_P1_SW")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"]["sweep"][0].__setitem__("extra", 1), "AUDIT_P1_SW")
        _tamper(lambda p: p["pillar1_atomic_ssd_roofline"]["zipf_ssd_roofline_n1"]["sweep"][2].__setitem__("tps_2dp", "50.00"), "AUDIT_P1_SW_VAL")

        # Pillar 2 tampers (10)
        _tamper(lambda p: p["pillar2_budget_crossover"]["external_dspark_spec"].__setitem__("extra", 1), "AUDIT_P2_DS")
        _tamper(lambda p: p["pillar2_budget_crossover"]["external_dspark_spec"].__setitem__("exceeds_3x", False), "AUDIT_P2_DS_FLAG")
        _tamper(lambda p: p["pillar2_budget_crossover"]["external_dspark_spec"].__setitem__("renewal_speedup_4dp", "3.0745"), "AUDIT_P2_DS_VAL")
        _tamper(lambda p: p["pillar2_budget_crossover"]["zero_draft_uno_spec"].__setitem__("lossless_speedup", "2"), "AUDIT_P2_UNO_VAL")
        _tamper(lambda p: p["pillar2_budget_crossover"].__setitem__("crossover_residual_zero", False), "AUDIT_P2_FLAG")
        _tamper(lambda p: p["pillar2_budget_crossover"].__setitem__("exact_crossover_b_star_4dp", "11.8918"), "AUDIT_P2_BSTAR")
        _tamper(lambda p: p["pillar2_budget_crossover"].__setitem__("operating_points", []), "AUDIT_P2_PTS")
        _tamper(lambda p: p["pillar2_budget_crossover"]["operating_points"].__setitem__(0, []), "AUDIT_P2_PTS")
        _tamper(lambda p: p["pillar2_budget_crossover"]["operating_points"][0].__setitem__("extra", 1), "AUDIT_P2_PTS")
        _tamper(lambda p: p["pillar2_budget_crossover"]["operating_points"][1].__setitem__("winner", "external_dspark"), "AUDIT_P2_PT_VAL")

        # Pillar 3 tampers (11)
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["k2_horizon_mova_anchors_n0"].__setitem__("reported_matched_quality_active_fraction", "1/8"), "AUDIT_P3_ANC_VAL")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["softplus_gating_probes"].__setitem__("strictly_increasing", False), "AUDIT_P3_SP_FLAG")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["softplus_gating_probes"].__setitem__("sigma_derivatives", ["1/5", "1/3", "1/2", "2/3", "3/4"]), "AUDIT_P3_SP_VAL")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["dense_v_waterfilling"].__setitem__("am_gm_identity_exact", False), "AUDIT_P3_DW_VAL")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["dense_v_waterfilling"].__setitem__("gain_ratio_4dp", "1.2501"), "AUDIT_P3_DW_VAL")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["mova_sparse_v_waterfilling"].__setitem__("extra", 1), "AUDIT_P3_MW")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["mova_sparse_v_waterfilling"].__setitem__("gain_ratio_4dp", "16.4376"), "AUDIT_P3_MW_VAL")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"].__setitem__("mova_exceeds_12x_kl_reduction", False), "AUDIT_P3_FLAG")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"].__setitem__("convexity_perturbation_excess_kl", "1/256"), "AUDIT_P3_VAL")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["kintsugi_outlier_expert_task_break"].__setitem__("uniform_4bpw_task_broken", False), "AUDIT_P3_KN_FLAG")
        _tamper(lambda p: p["pillar3_mova_fisher_kintsugi"]["kintsugi_outlier_expert_task_break"].__setitem__("fisher_waterfilling_post_margin", "1"), "AUDIT_P3_KN_VAL")

        # Pillar 4 tampers (8)
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["honest_scope_ledger"].__setitem__("strict_n0_n1_separation_enforced", False), "AUDIT_P4_SC_FLAG")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["honest_scope_ledger"].__setitem__("n0_source_count", 2), "AUDIT_P4_SC_VAL")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["honest_scope_ledger"]["n0_published_sources"].__setitem__(0, "forged"), "AUDIT_P4_SC_VAL")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"].__setitem__("gate_rejects_single_token_drift", False), "AUDIT_P4_FLAG")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["lossless_64_token_parity_positive"].__setitem__("extra", 1), "AUDIT_P4_POS")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["lossless_64_token_parity_positive"].__setitem__("matched_tokens", 63), "AUDIT_P4_POS_VAL")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["lossless_64_token_drift_negative"].__setitem__("extra", 1), "AUDIT_P4_NEG")
        _tamper(lambda p: p["pillar4_scope_and_token_parity"]["lossless_64_token_drift_negative"].__setitem__("first_mismatch_index", 62), "AUDIT_P4_NEG_VAL")

        self.assertEqual(tamper_count, 45)

    # -----------------------------------------------------------------------
    # 4. R2-Grade Zone-B File I/O, Symlinks, Hardlinks, Streaming & Concurrency
    # -----------------------------------------------------------------------

    def test_08_zone_b_io_symlinks_hardlinks_streaming_and_16t(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s45_zb_") as td:
            tdp = Path(td)
            cert_file = tdp / "cert.json"

            w_res = audit.write_certificate_file(cert_file)
            self.assertIs(w_res["written"], True)
            self.assertEqual(stat.S_IMODE(cert_file.stat().st_mode), 0o600)
            c_res = audit.check_certificate_file(cert_file)
            self.assertIs(c_res["valid"], True)
            audit.write_certificate_file(cert_file)

            raw_ok = cert_file.read_bytes()
            for fname, bad_b, code in (
                ("crlf.json", raw_ok.replace(b"\n", b"\r\n"), "CRLF_FORBIDDEN"),
                ("nolf.json", raw_ok[:-1], "MISSING_TRAILING_LF"),
                ("badutf8.json", b"\xff\xfe\n", "INVALID_UTF8"),
                ("noncanon.json", b" " + raw_ok, "NON_CANONICAL_BYTES"),
            ):
                p = tdp / fname
                p.write_bytes(bad_b)
                with self.assertRaises(Err) as cm:
                    audit.check_certificate_file(p)
                self.assertEqual(cm.exception.code, code)

            for bad_p, code in (
                (_SubPath(cert_file), "INVALID_PATH_TYPE"),
                (str(cert_file), "INVALID_PATH_TYPE"),
                (Path(f"{td}/a\x00b.json"), "INVALID_PATH"),
                (Path(f"{td}/a\ud800b.json"), "INVALID_UTF8_STR"),
            ):
                with self.assertRaises(Err) as cm:
                    audit.require_safe_path(bad_p)
                self.assertEqual(cm.exception.code, code)
            self.assertEqual(audit.require_safe_path(Path("/")), Path("/"))
            self.assertEqual(audit.require_safe_path(tdp / "nonexistent.json"), tdp / "nonexistent.json")
            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(Path("/"))
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(Err) as cm:
                audit.write_regular_file_bytes(Path("/"), b"x")
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(Err) as cm:
                audit.write_regular_file_bytes(tdp / "nb.json", "not_bytes")  # type: ignore[arg-type]
            self.assertEqual(cm.exception.code, "INVALID_WRITE_DATA")

            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(tdp / "no_dir" / "x.json")
            self.assertEqual(cm.exception.code, "PATH_NOT_FOUND")
            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(tdp / "missing.json")
            self.assertEqual(cm.exception.code, "FILE_READ_ERROR")
            sub_dir = tdp / "adir"
            sub_dir.mkdir()
            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(sub_dir)
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(Err) as cm:
                audit.write_regular_file_bytes(sub_dir, b"x")
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
            if hasattr(os, "mkfifo"):
                fifo = tdp / "pipe.json"
                os.mkfifo(fifo)
                with self.assertRaises(Err) as cm:
                    audit.read_regular_bytes_capped(fifo)
                self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
                with self.assertRaises(Err) as cm:
                    audit.write_regular_file_bytes(fifo, b"x")
                self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")

            sym_leaf = tdp / "sym_leaf.json"
            os.symlink(cert_file, sym_leaf)
            for fn in (
                lambda: audit.require_safe_path(sym_leaf),
                lambda: audit.read_regular_bytes_capped(sym_leaf),
                lambda: audit.write_regular_file_bytes(sym_leaf, b"x"),
            ):
                with self.assertRaises(Err) as cm:
                    fn()
                self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            real_sub = tdp / "real_sub" / "deep"
            real_sub.mkdir(parents=True)
            (real_sub / "ok.json").write_bytes(raw_ok)
            sym_parent = tdp / "sym_parent"
            os.symlink(tdp / "real_sub", sym_parent)
            for p_bad in (
                sym_parent / "deep" / "ok.json",
                sym_parent / ".." / "real_sub" / "deep" / "ok.json",
            ):
                with self.assertRaises(Err) as cm:
                    audit.read_regular_bytes_capped(p_bad)
                self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            orig_cwd = os.getcwd()
            orig_pwd = os.environ.get("PWD")
            try:
                os.chdir(real_sub)
                os.environ["PWD"] = str(sym_parent / "deep")
                with self.assertRaises(Err) as cm:
                    audit.read_regular_bytes_capped(Path("ok.json"))
                self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")
                os.environ["PWD"] = str(tdp / "stale_missing_dir")
                self.assertEqual(audit.read_regular_bytes_capped(Path("ok.json")), raw_ok)
                os.environ["PWD"] = "relative/pwd"
                self.assertEqual(audit.read_regular_bytes_capped(Path("ok.json")), raw_ok)
                os.environ["PWD"] = str(real_sub)
                self.assertEqual(audit.read_regular_bytes_capped(Path("ok.json")), raw_ok)
                os.environ.pop("PWD", None)
                self.assertEqual(audit.read_regular_bytes_capped(Path("ok.json")), raw_ok)
            finally:
                os.chdir(orig_cwd)
                if orig_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = orig_pwd

            hl_alias = tdp / "hl_alias.json"
            os.link(cert_file, hl_alias)
            with self.assertRaises(Err) as cm:
                audit.write_regular_file_bytes(hl_alias, b"tampered\n")
            self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")
            self.assertEqual(cert_file.read_bytes(), raw_ok)
            os.unlink(hl_alias)

            stream_p = tdp / "stream140k.bin"
            blob140 = b"x" * 140_000
            stream_p.write_bytes(blob140)
            self.assertEqual(audit.read_regular_bytes_capped(stream_p, max_bytes=140_000), blob140)
            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(stream_p, max_bytes=139_999)
            self.assertEqual(cm.exception.code, "FILE_TOO_LARGE")
            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(stream_p, max_bytes=0)
            self.assertEqual(cm.exception.code, "INT_BELOW_MIN")
            with self.assertRaises(Err) as cm:
                audit.read_regular_bytes_capped(stream_p, max_bytes=audit.MAX_ALLOWED_CAP_BYTES + 1)
            self.assertEqual(cm.exception.code, "INT_ABOVE_MAX")

            tn1 = audit._next_tmp_name("col.json")
            tn2 = audit._next_tmp_name("col.json")
            self.assertNotEqual(tn1, tn2)
            self.assertEqual(
                oracle.parse_exact_int_ascii(tn2.split(".")[-2], "s2"),
                oracle.parse_exact_int_ascii(tn1.split(".")[-2], "s1") + 1,
            )
            col_target = tdp / "col.json"
            (tdp / ".col.json.s45tmp.fixed0").write_bytes(b"occ")
            with mock.patch.object(audit, "_next_tmp_name", side_effect=[".col.json.s45tmp.fixed0", ".col.json.s45tmp.fixed1"]):
                audit.write_regular_file_bytes(col_target, b"ok\n")
            self.assertEqual(col_target.read_bytes(), b"ok\n")

            (tdp / ".col2.json.s45tmp.occ").write_bytes(b"occ")
            with mock.patch.object(audit, "_next_tmp_name", return_value=".col2.json.s45tmp.occ"):
                with self.assertRaises(Err) as cm:
                    audit.write_regular_file_bytes(tdp / "col2.json", b"ok\n")
            self.assertEqual(cm.exception.code, "TEMP_RETRY_EXHAUSTED")

            real_fstat = os.fstat

            def _fake_fstat(fd: int) -> os.stat_result:
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, 2, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))

            with mock.patch.object(audit.os, "fstat", side_effect=_fake_fstat):
                with self.assertRaises(Err) as cm:
                    audit.write_regular_file_bytes(tdp / "hltmp.json", b"ok\n")
            self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")
            self.assertEqual(list(tdp.glob(".hltmp.json.s45tmp.*")), [])

            conc_target = tdp / "conc.json"
            conc_errs: list[str] = []
            conc_lock = threading.Lock()
            conc_ok = 0

            def _cw(tid: int) -> None:
                nonlocal conc_ok
                data_b = f'{{"tid":{tid}}}\n'.encode("ascii")
                try:
                    audit.write_regular_file_bytes(conc_target, data_b)
                    with conc_lock:
                        conc_ok += 1
                except Exception as exc:  # pragma: no cover
                    with conc_lock:
                        conc_errs.append(f"{tid}:{exc}")

            ths = [threading.Thread(target=_cw, args=(i,)) for i in range(16)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            self.assertEqual(conc_errs, [])
            self.assertEqual(conc_ok, 16)
            self.assertEqual(list(tdp.glob(".conc.json.s45tmp.*")), [])

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

            kallsyms = Path("/proc/kallsyms")
            if kallsyms.exists():
                with self.assertRaises(Err) as cm:
                    audit.read_regular_bytes_capped(kallsyms)
                self.assertEqual(cm.exception.code, "FILE_TOO_LARGE")

    # -----------------------------------------------------------------------
    # 5. Strict CLI Parsing, Abbreviation/Flag Traps, Deleted CWD & Disk Cert
    # -----------------------------------------------------------------------

    def test_09_cli_parsing_flag_traps_deleted_cwd_and_disk_cert(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s45_cli_") as td:
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
                buf_out, buf_err = io.StringIO(), io.StringIO()
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    sys.stdout, sys.stderr = buf_out, buf_err
                    rc = audit.main(argv_ok)
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                self.assertEqual(rc, 0)
                self.assertEqual(buf_err.getvalue(), "")

            bad_argvs: list[tuple[object, str]] = [
                (["--write", "--json"], "INVALID_PATH_ARG"),
                (["--check", "--json"], "INVALID_PATH_ARG"),
                (["--write", ""], "INVALID_PATH_ARG"),
                (["--write=--json"], "UNKNOWN_FLAG"),
                (["--check=--json"], "UNKNOWN_FLAG"),
                (["--st"], "UNKNOWN_FLAG"),
                (["--wr", str(out_path)], "UNKNOWN_FLAG"),
                (["--ch", str(out_path)], "UNKNOWN_FLAG"),
                (["--js"], "UNKNOWN_FLAG"),
                (["--json", "--json"], "DUPLICATE_FLAG"),
                (["--standalone", "--standalone"], "CONFLICTING_MODE"),
                (["--standalone", "--check", str(out_path)], "CONFLICTING_MODE"),
                (["--write"], "MISSING_PATH_ARG"),
                (["--check"], "MISSING_PATH_ARG"),
                (["--write", "bad\ud800.json"], "INVALID_UTF8_STR"),
                ("not_a_list", "INVALID_ARGV"),
            ]
            for bad, exp_code in bad_argvs:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    sys.stdout, sys.stderr = buf_out, buf_err
                    rc = audit.main(bad)  # type: ignore[arg-type]
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
                self.assertEqual(rc, 2)
                err_obj = json.loads(buf_err.getvalue())
                self.assertIs(err_obj["valid"], False)
                self.assertEqual(err_obj["error_code"], exp_code)
            self.assertFalse(Path("--json").exists())

            del_dir = tdp / "vanishing"
            del_dir.mkdir()
            script = (
                "import os, sys; "
                f"sys.path.insert(0, {str(_THIS_DIR)!r}); "
                "import atomic_mova_uno_audit as a; "
                f"os.chdir({str(del_dir)!r}); os.rmdir({str(del_dir)!r}); "
                "raise SystemExit(a.main(['--check', 'rel.json']))"
            )
            proc = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("CWD_UNAVAILABLE", proc.stderr)

        disk_cert = _THIS_DIR / "atomic_mova_uno_certificate.json"
        if disk_cert.exists():
            res = audit.check_certificate_file(disk_cert)
            self.assertIs(res["valid"], True)
            self.assertEqual(res["payload_sha256"], self.cert["payload_sha256"])
            self.assertEqual(res["certificate_sha256"], self.cert["certificate_sha256"])


if __name__ == "__main__":
    unittest.main()
