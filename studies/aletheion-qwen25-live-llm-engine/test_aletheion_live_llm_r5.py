#!/usr/bin/env python3
"""Pure-Z/Q R5 Verification Test Suite for Live Qwen2.5-0.5B A_16 LLM Engine.

Strict invariants:
  - Zero bare assert statements, zero AST floats, zero .isdigit(), zero null-device writes.
  - Covers exact oracle derivation, fail-closed audit, atomic dir_fd I/O,
    Darwin prefix normalization, $PWD symlink/inode checks, hardlink/FIFO rejection,
    live receipt SHA/value pinning, and full branch/BoolOp mutation killing.
"""

from __future__ import annotations

import ast
import copy
from decimal import Decimal
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
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aletheion_live_llm_audit as audit
import aletheion_live_llm_oracle as oracle


def _resign(cert: dict) -> dict:
    c = copy.deepcopy(cert)
    p_sha = hashlib.sha256(audit.canonical_json_bytes(c["payload"])).hexdigest()
    c["payload_sha256"] = p_sha
    body = {
        "schema": c["schema"],
        "track_id": c["track_id"],
        "payload_sha256": p_sha,
        "payload": c["payload"],
    }
    c["certificate_sha256"] = hashlib.sha256(audit.canonical_json_bytes(body)).hexdigest()
    return c


class TestTrackS58LiveLLMEngineR5(unittest.TestCase):
    def test_01_ast_zero_float_and_zero_bare_asserts(self) -> None:
        for fname in (
            "aletheion_live_llm_oracle.py",
            "aletheion_live_llm_audit.py",
            "test_aletheion_live_llm_r5.py",
        ):
            src = (HERE / fname).read_text(encoding="utf-8")
            tree = ast.parse(src, filename=fname)
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert, msg=f"bare assert in {fname}")
                if isinstance(node, ast.Constant):
                    self.assertIsNot(type(node.value), float, msg=f"float literal in {fname}")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, "float", msg=f"float() call in {fname}")
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, "isdigit", msg=f".isdigit() in {fname}")

    def test_02_oracle_exact_inventory_and_a16_code(self) -> None:
        cert = oracle.build_certificate()
        self.assertEqual(cert["schema"], oracle.CERT_SCHEMA)
        self.assertEqual(cert["track_id"], oracle.CERT_TRACK_ID)
        exact = cert["payload"]["exact"]
        inv = exact["qwen25_05b_inventory"]
        self.assertEqual(inv["linear_matrix_count"], 168)
        self.assertEqual(inv["block_weight_count"], 357_826_560)
        self.assertEqual(inv["groups_32_count"], 11_182_080)
        self.assertEqual(inv["subblocks_16_count"], 22_364_160)
        self.assertEqual(inv["cold_18b_bytes"], 201_277_440)
        self.assertEqual(inv["bf16_linear_bytes"], 715_653_120)
        self.assertEqual(inv["tied_embed_bf16_bytes"], 272_269_312)
        self.assertEqual(inv["e2e_decode_weight_read_bytes"], 473_546_752)
        self.assertEqual(inv["nonblock_bf16_tensor_count"], 122)
        self.assertEqual(inv["bits_per_weight_q"], "9/2")
        self.assertEqual(inv["compression_ratio_bf16_over_a16_q"], "32/9")
        self.assertIs(inv["all_dims_aligned_32_128"], True)
        self.assertEqual(inv["shape_counts"], {"896x896": 48, "128x896": 48, "4864x896": 48, "896x4864": 24})

        code = exact["a16_extended_hamming_exact"]
        self.assertEqual(code["exhaustive_vectors_checked"], 65536)
        self.assertEqual(code["codeword_count"], 2048)
        self.assertEqual(code["min_nonzero_distance"], 4)
        self.assertEqual(code["cuda_avx512_mask_mismatches"], 0)
        self.assertEqual(code["swar_u4_to_s8_exact_matches"], 15)
        self.assertEqual(code["swar_u4_to_s8_domain_size"], 15)
        self.assertEqual(
            code["weight_enumerator"],
            {"0": 1, "4": 140, "6": 448, "8": 870, "10": 448, "12": 140, "16": 1},
        )
        self.assertEqual(oracle.frac_str(Fraction(6, 1)), "6")
        self.assertEqual(oracle.frac_str(Fraction(9, 2)), "9/2")
        with self.assertRaises(ValueError):
            oracle.dec_to_frac("")
        with self.assertRaises(ValueError):
            oracle.dec_to_frac(123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            oracle.frac_str(123)  # type: ignore[arg-type]

        # Verify oracle alignment branch when shape misaligned
        with mock.patch.object(oracle, "PROJ_SHAPES", (("bad", 30, 128),)):
            bad_inv = oracle.build_exact_algebraic_section()["qwen25_05b_inventory"]
            self.assertIs(bad_inv["all_dims_aligned_32_128"], False)
        with mock.patch.object(oracle, "PROJ_SHAPES", (("bad", 32, 64),)):
            bad_inv2 = oracle.build_exact_algebraic_section()["qwen25_05b_inventory"]
            self.assertIs(bad_inv2["all_dims_aligned_32_128"], False)

        # Verify each of the 4 Pareto conjuncts independently (isolating a vs c1)
        for key, repl in (
            ("bb_b1_tok_s", "65.0"),
            ("bb_c1_tok_s", "9999.0"),
            ("bb_b1_combined_mj", "1500.0"),
            ("bb_c1_cpu_pkg_mj", "1.0"),
            ("bb_b1_combined_mj", "1350.0"),
            ("e2e_b1_tok_s", "50.0"),
            ("e2e_c1_tok_s", "9999.0"),
            ("e2e_b1_combined_mj", "2500.0"),
            ("e2e_c1_cpu_pkg_mj", "1.0"),
            ("e2e_b1_combined_mj", "1900.0"),
        ):
            patched = dict(oracle.RAW_SAMPLED_DECIMALS)
            patched[key] = repl
            with mock.patch.object(oracle, "RAW_SAMPLED_DECIMALS", patched):
                s = oracle.build_sampled_empirical_section()
                if key.startswith("bb_"):
                    self.assertIs(s["decode_backbone_24layers_q"]["b1_pareto_dominates_a_and_c1"], False)
                else:
                    self.assertIs(s["decode_full_e2e_with_bf16_lm_head_q"]["b1_pareto_dominates_a_and_c1"], False)
        # Also flip ONLY b1_combined_mj < a_combined_mj while b1 < c1 stays True
        for key, repl in (("bb_a_combined_mj", "100.0"), ("e2e_a_combined_mj", "100.0")):
            patched = dict(oracle.RAW_SAMPLED_DECIMALS)
            patched[key] = repl
            with mock.patch.object(oracle, "RAW_SAMPLED_DECIMALS", patched):
                s = oracle.build_sampled_empirical_section()
                if key.startswith("bb_"):
                    self.assertIs(s["decode_backbone_24layers_q"]["b1_pareto_dominates_a_and_c1"], False)
                else:
                    self.assertIs(s["decode_full_e2e_with_bf16_lm_head_q"]["b1_pareto_dominates_a_and_c1"], False)

    def test_03_json_tree_and_parser_hardening(self) -> None:
        # Strict JSON parse checks + exact upper boundaries (kills boundary_cmp mutants!)
        self.assertEqual(audit.parse_strict_json('{"a":1,"b":true,"c":null,"d":[0,-5]}'), {"a": 1, "b": True, "c": None, "d": [0, -5]})
        max_valid_int = (1 << audit.MAX_INT_BITS) - 1  # 2048 bits, exact 617 digits == MAX_INT_DIGITS!
        self.assertEqual(len(str(max_valid_int)), audit.MAX_INT_DIGITS)
        self.assertEqual(audit.parse_strict_json('{"m":' + str(max_valid_int) + "}"), {"m": max_valid_int})
        self.assertEqual(audit.parse_exact_rational(str(max_valid_int), "max_r"), Fraction(max_valid_int, 1))
        # Exact MAX_JSON_BYTES boundary passes
        self.assertEqual(audit.parse_strict_json(" " * (audit.MAX_JSON_BYTES - 2) + "{}"), {})
        # Exact MAX_JSON_DEPTH boundary passes (depth == 64)
        exact_depth: list = []
        cur_ok = exact_depth
        for _ in range(audit.MAX_JSON_DEPTH):
            nxt_ok: list = []
            cur_ok.append(nxt_ok)
            cur_ok = nxt_ok
        audit._validate_json_tree(exact_depth)

        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json(123)  # type: ignore[arg-type]
        self.assertEqual(cm.exception.code, "INVALID_JSON_TEXT")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json("\ud800")
        self.assertEqual(cm.exception.code, "INVALID_UNICODE_STR")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json(" " * (audit.MAX_JSON_BYTES + 1))
        self.assertEqual(cm.exception.code, "JSON_TOO_LARGE")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json('{"a":1,"a":2}')
        self.assertEqual(cm.exception.code, "DUPLICATE_JSON_KEY")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json('{"a":1.5}')
        self.assertEqual(cm.exception.code, "FLOAT_FORBIDDEN")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json('{"a":NaN}')
        self.assertEqual(cm.exception.code, "NONFINITE_CONSTANT")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json('{"a":' + "9" * (audit.MAX_INT_DIGITS + 1) + "}")
        self.assertEqual(cm.exception.code, "INT_DIGIT_LIMIT")
        # 1 << 2048 has 617 digits (== MAX_INT_DIGITS) and 2049 bits (> MAX_INT_BITS): kills M013 & M014!
        over_bits = 1 << audit.MAX_INT_BITS
        self.assertEqual(len(str(over_bits)), audit.MAX_INT_DIGITS)
        with self.assertRaises(audit.AuditError) as cm:
            audit._parse_capped_int_token(str(over_bits))
        self.assertEqual(cm.exception.code, "INT_BIT_LIMIT")
        with self.assertRaises(audit.AuditError) as cm:
            audit.parse_strict_json("{bad")
        self.assertEqual(cm.exception.code, "JSON_DECODE_ERROR")

        # Tree validation checks
        deep: list = []
        cur = deep
        for _ in range(audit.MAX_JSON_DEPTH + 2):
            nxt: list = []
            cur.append(nxt)
            cur = nxt
        for bad, code in (
            (deep, "JSON_DEPTH_EXCEEDED"),
            ({1: "x"}, "NON_STR_KEY"),
            ({"\ud800": "x"}, "INVALID_UNICODE_STR"),
            ({"k": "\ud800"}, "INVALID_UNICODE_STR"),
            ({"k": 1 << 2049}, "INT_BIT_LIMIT"),
            ({"k": (1, 2)}, "FORBIDDEN_JSON_TYPE"),
            ({"k": json.loads("1.5")}, "FORBIDDEN_JSON_TYPE"),
        ):
            with self.assertRaises(audit.AuditError) as cm:
                audit._validate_json_tree(bad)
            self.assertEqual(cm.exception.code, code)

        # Digest & rational helpers
        for bad, code in (
            (123, "INVALID_DIGEST_TYPE"),
            ("abc", "INVALID_DIGEST_LEN"),
            ("G" * 64, "INVALID_DIGEST_HEX"),
        ):
            with self.assertRaises(audit.AuditError) as cm:
                audit.require_sha256_hex(bad, "d")
            self.assertEqual(cm.exception.code, code)

        self.assertEqual(audit.parse_exact_rational("-5", "r"), Fraction(-5, 1))
        self.assertEqual(audit.parse_exact_rational("9/2", "r"), Fraction(9, 2))
        for bad, code in (
            (5, "INVALID_RATIONAL"),
            ("", "INVALID_RATIONAL"),
            ("1/2/3", "INVALID_RATIONAL"),
            ("1/a", "INVALID_RATIONAL"),
            ("1/-", "INVALID_RATIONAL"),
            ("9" * (audit.MAX_INT_DIGITS + 1), "INT_DIGIT_LIMIT"),
            ("3/1", "NON_CANONICAL_DEN"),
            ("3/0", "NON_CANONICAL_DEN"),
            ("2/4", "NON_CANONICAL_RATIONAL"),
        ):
            with self.assertRaises(audit.AuditError) as cm:
                audit.parse_exact_rational(bad, "r")
            self.assertEqual(cm.exception.code, code)

        for fn, bad, code in (
            (audit._req_dict, [], "INVALID_DICT"),
            (audit._req_int, True, "INVALID_INT"),
            (audit._req_bool, 1, "INVALID_BOOL"),
            (audit._req_str, 1, "INVALID_STR"),
        ):
            with self.assertRaises(audit.AuditError) as cm:
                fn(bad, "x")
            self.assertEqual(cm.exception.code, code)

    def test_04_secondary_audit_exhaustive_rejection(self) -> None:
        base_cert = oracle.build_certificate()
        res = audit.verify_certificate_object(base_cert, full_replay=True)
        self.assertIs(res["valid"], True)
        self.assertIs(res["full_replay"], True)
        self.assertEqual(audit.secondary_audit_payload(base_cert["payload"]), {"secondary_audit_passed": True})

        # Top-level cert tamperings
        bad_keys = copy.deepcopy(base_cert)
        bad_keys["extra"] = 1
        with self.assertRaises(audit.AuditError) as cm:
            audit.verify_certificate_object(bad_keys, full_replay=False)
        self.assertEqual(cm.exception.code, "CERT_KEYS_MISMATCH")

        for field, val, code in (
            ("schema", "wrong", "SCHEMA_MISMATCH"),
            ("track_id", "wrong", "TRACK_ID_MISMATCH"),
            ("payload_sha256", "0" * 64, "PAYLOAD_SHA_MISMATCH"),
            ("certificate_sha256", "0" * 64, "CERT_SHA_MISMATCH"),
        ):
            c = copy.deepcopy(base_cert)
            c[field] = val
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_certificate_object(c, full_replay=False)
            self.assertEqual(cm.exception.code, code)

        # Extra anchor citation / extra key rejected by canonical payload pin even at full_replay=False!
        c_rep = copy.deepcopy(base_cert)
        c_rep["payload"]["anchor"]["cs13_citations"].append({"id": "EXTRA", "tier": "N0", "claim": "extra"})
        c_rep = _resign(c_rep)
        with self.assertRaises(audit.AuditError) as cm:
            audit.verify_certificate_object(c_rep, full_replay=False)
        self.assertEqual(cm.exception.code, "PAYLOAD_PIN_MISMATCH")

        # Full replay mismatch tested cleanly via mocked fresh oracle divergence
        with mock.patch.object(oracle, "build_certificate", return_value=c_rep):
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_certificate_object(base_cert, full_replay=True)
            self.assertEqual(cm.exception.code, "REPLAY_MISMATCH")

        # Positive rational guard (0 & negative -> NONPOSITIVE_VALUE; 0 ZeroDivisionError!)
        for bad_pos in ("0", "-1", "-3/2"):
            with self.assertRaises(audit.AuditError) as cm:
                audit._parse_positive_rational(bad_pos, "pos")
            self.assertEqual(cm.exception.code, "NONPOSITIVE_VALUE")
        self.assertEqual(audit._parse_positive_rational("1/2", "pos"), Fraction(1, 2))

        # Every single secondary_audit_payload rejection condition!
        def _fail_payload(mut_fn, expected_code: str) -> None:
            p = copy.deepcopy(base_cert["payload"])
            mut_fn(p)
            with self.assertRaises(audit.AuditError, msg=expected_code) as cm:
                audit.secondary_audit_payload(p)
            self.assertEqual(cm.exception.code, expected_code)

        # Negative VRAM & inverted Pareto dominance (with consistent fake ratios!) rejected!
        _fail_payload(lambda p: p["sampled"]["ring0_mlos_receipt"].__setitem__("linux_kmd_baseline_vram_mib", -1), "MLOS_RECEIPT_MISMATCH")
        for sec, b1_k, a_k, c1_k, b1_m, a_m, c1_m in (
            ("decode_backbone_24layers_q", "arm_b1_sm86_tok_s_q", "arm_a_bf16_tok_s_q", "arm_c1_avx512_8t_tok_s_q",
             "arm_b1_sm86_combined_mj_q", "arm_a_bf16_combined_mj_q", "arm_c1_avx512_cpu_pkg_mj_q"),
            ("decode_full_e2e_with_bf16_lm_head_q", "arm_b1_sm86_tok_s_q", "arm_a_bf16_tok_s_q", "arm_c1_avx512_8t_tok_s_q",
             "arm_b1_sm86_combined_mj_q", "arm_a_bf16_combined_mj_q", "arm_c1_avx512_cpu_pkg_mj_q"),
        ):
            # Test each of the 4 Pareto dominance inequalities independently with consistent ratios!
            for b1v, av, c1v, b1mv, amv, c1mv in (
                ("50", "100", "25", "100", "200", "300"),    # fails b1 > a only
                ("50", "25", "100", "100", "200", "300"),    # fails b1 > c1 only
                ("100", "50", "25", "200", "100", "300"),    # fails b1_mj < a_mj only
                ("100", "50", "25", "200", "300", "100"),    # fails b1_mj < c1_mj only
            ):
                def _inv_pareto(p, s=sec, bk=b1_k, ak=a_k, ck=c1_k, bm=b1_m, am=a_m, cmk=c1_m,
                                b1_=b1v, a_=av, c1_=c1v, b1m_=b1mv, am_=amv, c1m_=c1mv):
                    d = p["sampled"][s]
                    d[bk], d[ak], d[ck] = b1_, a_, c1_
                    d[bm], d[am], d[cmk] = b1m_, am_, c1m_
                    d["speedup_b1_vs_a_q"] = oracle.frac_str(Fraction(int(b1_), int(a_)))
                    d["speedup_b1_vs_c1_q"] = oracle.frac_str(Fraction(int(b1_), int(c1_)))
                    d["energy_ratio_a_over_b1_q"] = oracle.frac_str(Fraction(int(am_), int(b1m_)))
                    d["energy_ratio_c1_pkg_over_b1_q"] = oracle.frac_str(Fraction(int(c1m_), int(b1m_)))
                _fail_payload(_inv_pareto, "PARETO_MISMATCH")
            # Zero denominators rejected cleanly with NONPOSITIVE_VALUE (0 ZeroDivisionError!)
            for div_k in (a_k, c1_k, b1_m):
                _fail_payload(lambda p, s=sec, dk=div_k: p["sampled"][s].__setitem__(dk, "0"), "NONPOSITIVE_VALUE")
        _fail_payload(lambda p: p["sampled"]["prefill_b32_all168_q"].__setitem__("arm_a3_bf16_tok_s_q", "0"), "NONPOSITIVE_VALUE")
        _fail_payload(lambda p: p["sampled"]["arm_d_cold_18b_and_parity_q"].__setitem__("ttft_a16_sm86_ms_q", "0"), "NONPOSITIVE_VALUE")
        _fail_payload(lambda p: p["sampled"]["arm_d_cold_18b_and_parity_q"].__setitem__("gen_128_bf16_decode_tok_s_q", "0"), "NONPOSITIVE_VALUE")
        # Unchecked extra key in nested dict caught by canonical pin
        _fail_payload(lambda p: p["sampled"]["isa_receipts"].__setitem__("extra_nested", 1), "PAYLOAD_PIN_MISMATCH")

        _fail_payload(lambda p: p.pop("anchor"), "PAYLOAD_KEYS_MISMATCH")
        _fail_payload(lambda p: p["exact"].__setitem__("provenance_tier", "sampled"), "TIER_MISMATCH")
        _fail_payload(lambda p: p["sampled"].__setitem__("provenance_tier", "exact"), "TIER_MISMATCH")
        _fail_payload(lambda p: p["sampled"].__setitem__("measured_live_model_weights", False), "LIVE_WEIGHT_FLAG")
        _fail_payload(lambda p: p["anchor"].__setitem__("provenance_tier", "exact"), "TIER_MISMATCH")

        # Independent subcondition tests for inventory checks
        _fail_payload(
            lambda p: p["exact"]["qwen25_05b_inventory"].update(
                {"block_weight_count": 32, "groups_32_count": 1, "subblocks_16_count": 2, "cold_18b_bytes": 18, "bf16_linear_bytes": 64}
            ),
            "INVENTORY_MISMATCH",
        )
        _fail_payload(
            lambda p: p["exact"]["qwen25_05b_inventory"].update({"groups_32_count": 1, "cold_18b_bytes": 18}),
            "INVENTORY_MISMATCH",
        )
        for k, v in (
            ("linear_matrix_count", 167),
            ("subblocks_16_count", 1),
            ("cold_18b_bytes", 1),
            ("bf16_linear_bytes", 1),
            ("nonblock_bf16_tensor_count", 121),
            ("all_dims_aligned_32_128", False),
            ("shape_counts", {"896x896": 1}),
        ):
            _fail_payload(lambda p, kk=k, vv=v: p["exact"]["qwen25_05b_inventory"].__setitem__(kk, vv), "INVENTORY_MISMATCH")

        _fail_payload(lambda p: p["exact"]["qwen25_05b_inventory"].__setitem__("bits_per_weight_q", "4"), "BPW_MISMATCH")
        _fail_payload(lambda p: p["exact"]["qwen25_05b_inventory"].__setitem__("compression_ratio_bf16_over_a16_q", "4"), "COMP_MISMATCH")

        for k, v in (
            ("exhaustive_vectors_checked", 100),
            ("codeword_count", 1024),
            ("min_nonzero_distance", 2),
            ("cuda_avx512_mask_mismatches", 1),
            ("swar_u4_to_s8_exact_matches", 14),
            ("weight_enumerator", {"0": True}),
        ):
            _fail_payload(lambda p, kk=k, vv=v: p["exact"]["a16_extended_hamming_exact"].__setitem__(kk, vv), "CODE_MISMATCH")

        _fail_payload(lambda p: p["sampled"]["pinned_sha256"].pop("config_json"), "SHA_KEYS_MISMATCH")
        _fail_payload(lambda p: p["sampled"]["pinned_sha256"].__setitem__("config_json", "0" * 64), "PINNED_SHA_MISMATCH")

        for k, v in (
            ("sm86_kernel_count", 15),
            ("sm86_compiler_warnings", 1),
            ("sm86_total_stack_bytes", 1),
            ("sm86_total_spill_stores_bytes", 1),
            ("sm86_total_spill_loads_bytes", 1),
            ("sm86_total_sass_ldl", 1),
            ("sm86_total_sass_stl", 1),
            ("sm86_zero_spill_verified", False),
            ("avx512_compiler_warnings", 1),
            ("avx512_vpermb_count", 0),
            ("avx512_vpopcntq_count", 0),
            ("avx512_vpdpbusd_count", 0),
            ("avx512_isa_verified", False),
        ):
            _fail_payload(lambda p, kk=k, vv=v: p["sampled"]["isa_receipts"].__setitem__(kk, vv), "ISA_RECEIPT_MISMATCH")

        for k, v in (
            ("determinism_zone", "A"),
            ("clean_hardware_run", False),
            ("exit_code", 1),
            ("anomaly_count", 1),
            ("graphics_process_count", 1),
            ("compute_process_count", 1),
            ("linux_kmd_baseline_vram_mib", 5),
            ("thermal_throttled", True),
            ("slowdown_active", True),
        ):
            _fail_payload(lambda p, kk=k, vv=v: p["sampled"]["ring0_mlos_receipt"].__setitem__(kk, vv), "MLOS_RECEIPT_MISMATCH")

        for sec in ("decode_backbone_24layers_q", "decode_full_e2e_with_bf16_lm_head_q"):
            for rk in ("speedup_b1_vs_a_q", "speedup_b1_vs_c1_q", "energy_ratio_a_over_b1_q", "energy_ratio_c1_pkg_over_b1_q"):
                _fail_payload(lambda p, s=sec, r=rk: p["sampled"][s].__setitem__(r, "1"), "RATIO_MISMATCH")
            _fail_payload(lambda p, s=sec: p["sampled"][s].__setitem__("b1_pareto_dominates_a_and_c1", False), "PARETO_MISMATCH")

        _fail_payload(lambda p: p["sampled"]["prefill_b32_all168_q"].__setitem__("speedup_b3_vs_a3_q", "1"), "RATIO_MISMATCH")
        _fail_payload(lambda p: p["sampled"]["arm_d_cold_18b_and_parity_q"].__setitem__("ttft_speedup_a16_vs_bf16_q", "1"), "RATIO_MISMATCH")
        _fail_payload(lambda p: p["sampled"]["arm_d_cold_18b_and_parity_q"].__setitem__("gen_128_speedup_a16_vs_bf16_q", "1"), "RATIO_MISMATCH")

        # Isolate tf_t != 128 while rate == Fraction(tf_m, tf_t) holds
        _fail_payload(
            lambda p: p["sampled"]["arm_d_cold_18b_and_parity_q"].update(
                {"teacher_forced_128_total": 127, "teacher_forced_128_top1_rate_q": "105/127"}
            ),
            "PARITY_MISMATCH",
        )
        for k, v in (
            ("teacher_forced_128_top1_rate_q", "1/2"),
            ("total_live_syndrome_errors", 1),
            ("arm_c_vs_gpu_a16_greedy_16tok_matches", 15),
            ("arm_c_all_windows_gpu_idle_p8", False),
        ):
            _fail_payload(lambda p, kk=k, vv=v: p["sampled"]["arm_d_cold_18b_and_parity_q"].__setitem__(kk, vv), "PARITY_MISMATCH")

        _fail_payload(
            lambda p: p["anchor"].__setitem__(
                "cs13_citations",
                [{"id": "1", "tier": "N1", "claim": "a"}, {"id": "2", "tier": "N0", "claim": "b"}],
            ),
            "ANCHOR_MISMATCH",
        )
        _fail_payload(lambda p: p["anchor"].__setitem__("cs13_citations", "12345"), "ANCHOR_MISMATCH")
        _fail_payload(
            lambda p: p["anchor"].__setitem__(
                "cs13_citations",
                [{"id": str(i), "tier": "N1", "claim": "c"} for i in range(4)],
            ),
            "ANCHOR_MISMATCH",
        )

    def test_05_hardened_io_symlinks_hardlinks_fifo_pwd_darwin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s58_r5_") as td:
            root = Path(td)
            cert_path = root / "nested" / "cert.json"
            cert = oracle.build_certificate()
            raw = audit.canonical_json_bytes(cert)
            audit.write_regular_bytes_atomic_0600(cert_path, raw)
            # Overwriting an existing regular file (st_nlink == 1) succeeds (kills M066!)
            audit.write_regular_bytes_atomic_0600(cert_path, raw)
            self.assertEqual(stat.S_IMODE(os.stat(cert_path).st_mode), 0o600)
            chk = audit.check_certificate_file(cert_path, full_replay=True)
            self.assertIs(chk["valid"], True)
            self.assertEqual(chk["file_sha256"], hashlib.sha256(raw).hexdigest())

            # _is_symlink_at returns strict False on missing path (kills M042!)
            rfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                self.assertIs(audit._is_symlink_at(rfd, "does_not_exist"), False)
            finally:
                os.close(rfd)

            # Path validation
            with self.assertRaises(audit.AuditError) as cm:
                audit._validate_path_obj("/tmp/x")
            self.assertEqual(cm.exception.code, "INVALID_PATH_TYPE")
            with self.assertRaises(audit.AuditError) as cm:
                audit._validate_path_obj(Path("/tmp/\ud800"))
            self.assertEqual(cm.exception.code, "INVALID_UNICODE_STR")
            with self.assertRaises(audit.AuditError) as cm:
                audit._validate_path_obj(Path("a\x00b"))
            self.assertEqual(cm.exception.code, "INVALID_PATH")
            with self.assertRaises(audit.AuditError) as cm:
                audit._open_parent_dir_fd(Path("/"))
            self.assertEqual(cm.exception.code, "INVALID_PATH")
            with self.assertRaises(audit.AuditError) as cm:
                audit._open_parent_dir_fd(Path("."))
            self.assertEqual(cm.exception.code, "INVALID_PATH")
            # Explicit leaf == ".." kills M047!
            with self.assertRaises(audit.AuditError) as cm:
                audit._open_parent_dir_fd(root / "nested" / "..")
            self.assertEqual(cm.exception.code, "INVALID_PATH")
            with self.assertRaises(audit.AuditError) as cm:
                audit._open_parent_dir_fd(root / ".." / "x")
            self.assertEqual(cm.exception.code, "PARENT_TRAVERSAL_FORBIDDEN")

            # Missing file & missing ancestor (and assert read does NOT create missing dir -> kills M050!)
            missing_dir = root / "no_such_dir"
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_bytes_safe(missing_dir / "a.json")
            self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")
            self.assertFalse(missing_dir.exists())
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_bytes_safe(root / "missing.json")
            self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")

            # Symlink target & symlink ancestor (read & write)
            sym_file = root / "sym.json"
            sym_file.symlink_to(cert_path)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_bytes_safe(sym_file)
            self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as cm:
                audit.write_regular_bytes_atomic_0600(sym_file, raw)
            self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            sym_dir = root / "sym_dir"
            sym_dir.symlink_to(cert_path.parent)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_bytes_safe(sym_dir / "cert.json")
            self.assertEqual(cm.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")

            # Hardlink rejection (read & write)
            hl_file = root / "hardlink.json"
            os.link(cert_path, hl_file)
            try:
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_bytes_safe(hl_file)
                self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(hl_file, raw)
                self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")
            finally:
                hl_file.unlink()

            # Non-regular FIFO (scratch fifo, no system device writes) & directory target
            fifo_path = root / "test.fifo"
            os.mkfifo(fifo_path, 0o600)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_bytes_safe(fifo_path)
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(audit.AuditError) as cm:
                audit.write_regular_bytes_atomic_0600(fifo_path, raw)
            self.assertEqual(cm.exception.code, "NON_REGULAR_TARGET")

            # Read-only 0o500 directory fails closed with typed ATOMIC_WRITE_ERROR (direct & nested mkdir!)
            ro_dir = root / "ro_dir"
            ro_dir.mkdir(mode=0o700)
            os.chmod(ro_dir, 0o500)
            try:
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(ro_dir / "out.json", raw)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_ERROR")
                self.assertIn("cannot create temp file", cm.exception.message)
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(ro_dir / "sub" / "out.json", raw)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_ERROR")
                self.assertIn("cannot create ancestor", cm.exception.message)
                self.assertEqual(list(ro_dir.iterdir()), [])
            finally:
                os.chmod(ro_dir, 0o700)

            # Bounded EEXIST retry exhaustion + wrapped lstat/replace/write OSError -> ATOMIC_WRITE_ERROR
            real_open = os.open
            eexist_hits = 0
            def _eexist_on_excl(p, flags, *args, **kwargs):
                nonlocal eexist_hits
                if flags & os.O_EXCL:
                    eexist_hits += 1
                    raise FileExistsError("dup")
                return real_open(p, flags, *args, **kwargs)
            with mock.patch("os.open", side_effect=_eexist_on_excl):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(root / "eexist.json", raw)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_ERROR")
            self.assertEqual(eexist_hits, audit._MAX_TMP_RETRIES)
            with mock.patch("os.lstat", side_effect=PermissionError("denied")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(root / "lstat_err.json", raw)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_ERROR")
            with mock.patch("os.replace", side_effect=OSError("io fail")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(root / "rep_err.json", raw)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_ERROR")
            with mock.patch("os.write", side_effect=OSError("disk full")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(root / "wr_err.json", raw)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_ERROR")

            # Exact boundary & over-size regular file + streaming cap (kills M060, M061, M062!)
            big_path = root / "big.json"
            big_path.write_bytes(b"a" * 32)
            self.assertEqual(len(audit.read_regular_bytes_safe(big_path, max_bytes=32)), 32)
            with mock.patch("os.read", side_effect=RuntimeError("should fast-reject before os.read")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_bytes_safe(big_path, max_bytes=16)
                self.assertEqual(cm.exception.code, "FILE_TOO_LARGE")

            # Streaming read exact boundary & overflow when fstat reports st_size=0 (procfs simulation)
            real_fstat = os.fstat
            def _zero_size_fstat(fd: int):
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid, 0, st.st_atime, st.st_mtime, st.st_ctime))
            with mock.patch("os.fstat", side_effect=_zero_size_fstat):
                self.assertEqual(len(audit.read_regular_bytes_safe(big_path, max_bytes=32)), 32)
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_bytes_safe(big_path, max_bytes=16)
                self.assertEqual(cm.exception.code, "FILE_TOO_LARGE")

            # CRLF, invalid UTF-8, and non-canonical bytes rejection
            crlf_p = root / "crlf.json"
            crlf_p.write_bytes(raw.replace(b"\n", b"\r\n"))
            with self.assertRaises(audit.AuditError) as cm:
                audit.check_certificate_file(crlf_p)
            self.assertEqual(cm.exception.code, "CRLF_FORBIDDEN")

            bad_u8 = root / "badu8.json"
            bad_u8.write_bytes(b"\xff\xfe")
            with self.assertRaises(audit.AuditError) as cm:
                audit.check_certificate_file(bad_u8)
            self.assertEqual(cm.exception.code, "INVALID_UTF8")

            pretty_p = root / "pretty.json"
            pretty_p.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(audit.AuditError) as cm:
                audit.check_certificate_file(pretty_p)
            self.assertEqual(cm.exception.code, "NON_CANONICAL_BYTES")

            # Short-write & temp hardlink atomic write cleanup
            with mock.patch("os.write", return_value=0):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(root / "sw.json", b"abc")
                self.assertEqual(cm.exception.code, "SHORT_WRITE")
            self.assertFalse((root / "sw.json").exists())

            def _nlink2_fstat(fd: int):
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, 2, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
            with mock.patch("os.fstat", side_effect=_nlink2_fstat):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_regular_bytes_atomic_0600(root / "hl_tmp.json", b"abc")
                self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")

            # $PWD symlink & stale $PWD inode check
            real_sub = root / "real_sub"
            real_sub.mkdir()
            sym_sub = root / "sym_sub"
            sym_sub.symlink_to(real_sub)
            old_cwd = os.getcwd()
            try:
                os.chdir(real_sub)
                with mock.patch.dict(os.environ, {"PWD": str(sym_sub)}):
                    with self.assertRaises(audit.AuditError) as cm:
                        audit._check_pwd_no_symlinks()
                    self.assertEqual(cm.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
                with mock.patch.dict(os.environ, {"PWD": str(root)}):
                    # Stale mismatched $PWD passes cleanly
                    audit._check_pwd_no_symlinks()
                with mock.patch.object(audit.os, "environ", {"PWD": "/tmp/\ud800"}):
                    with self.assertRaises(audit.AuditError) as cm:
                        audit._check_pwd_no_symlinks()
                    self.assertEqual(cm.exception.code, "INVALID_UNICODE_STR")
                with mock.patch.object(audit.os, "environ", {}):
                    audit._check_pwd_no_symlinks()
                # Multi-segment relative PWD matching cwd inode must return early (not walk / -> kills line 265!)
                rel_dir = real_sub / "rel_a"
                rel_dir.mkdir()
                rel_sym = rel_dir / "rel_b"
                rel_sym.symlink_to(real_sub)
                try:
                    with mock.patch.object(audit.os, "environ", {"PWD": "rel_a/rel_b"}):
                        audit._check_pwd_no_symlinks()
                finally:
                    rel_sym.unlink()
                    rel_dir.rmdir()
                with mock.patch.dict(os.environ, {"PWD": "/nonexistent_s58_pwd_check"}):
                    audit._check_pwd_no_symlinks()
            finally:
                os.chdir(old_cwd)

            # Non-darwin absolute path leaves /var unchanged even with mocked root symlink (kills line 247!)
            fake_st = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 0, 0, 0, 0, 0, 0))
            with mock.patch.object(audit.sys, "platform", "linux"):
                with mock.patch("os.lstat", return_value=fake_st), mock.patch("os.readlink", return_value="private/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/folders/x")), Path("/var/folders/x"))

            # Mocked Darwin /var|/tmp|/etc prefix normalization (kills all line 247/250/254 BoolOp mutants!)
            with mock.patch.object(audit.sys, "platform", "darwin"):
                with mock.patch("os.lstat", return_value=fake_st), mock.patch("os.readlink", side_effect=lambda p: f"private/{Path(p).name}"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/folders/x")), Path("/private/var/folders/x"))
                    # Relative path with parts[1] == "var" must NOT be normalized
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("x/var/y")), Path("x/var/y"))
                    # Root "/" has len(parts) == 1
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/")), Path("/"))
                    # Absolute path not in _DARWIN_SYSTEM_LINKS (even if readlink matches private/opt -> kills line 250!)
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/opt/x")), Path("/opt/x"))
                # Root-owned non-symlink (S_IFDIR) not normalized (kills line 254 SUBCOND_DEL 0!)
                dir_root_st = os.stat_result((stat.S_IFDIR | 0o755, 1, 1, 1, 0, 0, 0, 0, 0, 0))
                with mock.patch("os.lstat", return_value=dir_root_st), mock.patch("os.readlink", return_value="private/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/folders/x")), Path("/var/folders/x"))
                # Non-root symlink not normalized
                nonroot_st = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 501, 0, 0, 0, 0, 0))
                with mock.patch("os.lstat", return_value=nonroot_st), mock.patch("os.readlink", return_value="private/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/folders/x")), Path("/var/folders/x"))
                # Non-matching readlink target not normalized
                with mock.patch("os.lstat", return_value=fake_st), mock.patch("os.readlink", return_value="other/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/folders/x")), Path("/var/folders/x"))
                # OSError fallback
                with mock.patch("os.lstat", side_effect=OSError("nope")):
                    self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/folders/x")), Path("/var/folders/x"))

    def test_06_live_receipts_verification_and_cli_modes(self) -> None:
        # Verify real live Track S58 receipts on disk
        vr = audit.verify_receipts_dir(HERE)
        self.assertIs(vr["valid"], True)
        self.assertEqual(vr["verified_files"], 6)
        self.assertEqual(vr["mlos_receipt_sha256"], oracle.PINNED_SHA256["mlos_receipt_inner_sha256"])

        # Tampered receipts rejection
        with tempfile.TemporaryDirectory(prefix="s58_rcpt_") as td:
            rdir = Path(td)
            for fname in (
                "qwen25_sm86_a16_engine.cu",
                "qwen25_avx512_a16_engine.c",
                "sm86_engine_sass_inspection.json",
                "avx512_engine_isa_inspection.json",
                "qwen25_live_engine_empirical_results.json",
                "mlos_zone_b_receipt.json",
            ):
                (rdir / fname).write_bytes((HERE / fname).read_bytes())
            self.assertIs(audit.verify_receipts_dir(rdir)["valid"], True)
            (rdir / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_dir(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_SHA_MISMATCH")

        # Receipt parser error branches
        with self.assertRaises(audit.AuditError) as cm:
            audit._parse_json_with_decimals(b"\xff", "x")
        self.assertEqual(cm.exception.code, "INVALID_UTF8")
        with self.assertRaises(audit.AuditError) as cm:
            audit._parse_json_with_decimals(b"{bad", "x")
        self.assertEqual(cm.exception.code, "RECEIPT_PARSE_ERROR")
        with self.assertRaises(audit.AuditError) as cm:
            audit._parse_json_with_decimals(b'{"a":NaN}', "x")
        self.assertEqual(cm.exception.code, "NONFINITE_CONSTANT")

        # Value mismatch branches inside verify_receipts_dir (Decimal-parsed to isolate each guard!)
        orig_emp = json.loads((HERE / "qwen25_live_engine_empirical_results.json").read_text(encoding="utf-8"), parse_float=Decimal)
        orig_mlos = json.loads((HERE / "mlos_zone_b_receipt.json").read_text(encoding="utf-8"), parse_float=Decimal)
        orig_sm86 = json.loads((HERE / "sm86_engine_sass_inspection.json").read_text(encoding="utf-8"), parse_float=Decimal)
        orig_avx = json.loads((HERE / "avx512_engine_isa_inspection.json").read_text(encoding="utf-8"), parse_float=Decimal)
        with mock.patch("hmac.compare_digest", return_value=True):
            for idx, mut in (
                (0, lambda o: o.__setitem__("measured_live_model_weights", False)),
                (1, lambda o: o.__setitem__("clean_hardware_run", False)),
                (1, lambda o: o.__setitem__("anomalies", ["bad"])),
                (2, lambda o: o.__setitem__("zero_spill_verified", False)),
                (3, lambda o: o.__setitem__("avx512_vbmi_vnni_vpopcntdq_verified", False)),
                (0, lambda o: o["arm_b_sm86_ptx_a16_gpu"]["b1_pure_dp4a"]["backbone_24layers"].__setitem__("gpu_event_tok_s", Decimal("1"))),
            ):
                objs = [copy.deepcopy(orig_emp), copy.deepcopy(orig_mlos), copy.deepcopy(orig_sm86), copy.deepcopy(orig_avx)]
                mut(objs[idx])
                with mock.patch.object(audit, "_parse_json_with_decimals", side_effect=objs):
                    with self.assertRaises(audit.AuditError) as cm:
                        audit.verify_receipts_dir(HERE)
                    self.assertEqual(cm.exception.code, "RECEIPT_VALUE_MISMATCH")

        # In-process CLI modes
        for cli_args in ([], ["--standalone", "--json"], ["--json"]):
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                rc = audit.main(cli_args)
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(buf.getvalue())["track_id"], oracle.CERT_TRACK_ID)

        with tempfile.TemporaryDirectory(prefix="s58_cli_") as td:
            cp = Path(td) / "c.json"
            for cli_args, exp_rc in (
                (["--write", str(cp)], 0),
                (["--check", str(cp)], 0),
                (["--check", str(cp), "--json"], 0),
                (["--verify-receipts", str(HERE)], 0),
                (["--write", str(cp), "extra"], 1),
                (["--write", "--json"], 1),
                (["--write", ""], 1),
                (["--check", "--json"], 1),
                (["--check", str(cp), "--bogus"], 1),
                (["--check", str(cp), "--json", "extra"], 1),
                (["--bogus", str(cp), "--json"], 1),
                (["--verify-receipts", str(HERE), "extra"], 1),
                (["--verify-receipts", "-bad"], 1),
                (["--bogus", str(HERE)], 1),
                (["--check"], 1),
                (["--bogus"], 1),
            ):
                buf = io.StringIO()
                with mock.patch("sys.stdout", buf):
                    rc = audit.main(cli_args)
                self.assertEqual(rc, exp_rc, msg=f"cli_args={cli_args}")
                parsed = json.loads(buf.getvalue())
                self.assertIs(parsed["valid"], exp_rc == 0)
            self.assertFalse((Path.cwd() / "--json").exists())

            # Programmatic bad argv types (non-iterable / non-str element) return clean INVALID_CLI_ARGS
            for bad_argv in (123, ["--check", 123]):
                buf = io.StringIO()
                with mock.patch("sys.stdout", buf):
                    rc = audit.main(bad_argv)
                self.assertEqual(rc, 1)
                self.assertEqual(json.loads(buf.getvalue())["error_code"], "INVALID_CLI_ARGS")
            # Tuple argv works identically to list
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf):
                rc = audit.main(("--check", str(cp)))
            self.assertEqual(rc, 0)
            self.assertIs(json.loads(buf.getvalue())["valid"], True)

            # Unexpected error branch
            buf = io.StringIO()
            with mock.patch("sys.stdout", buf), mock.patch.object(oracle, "build_certificate", side_effect=RuntimeError("boom")):
                rc = audit.main(["--standalone", "--json"])
            self.assertEqual(rc, 1)
            self.assertEqual(json.loads(buf.getvalue())["error_code"], "UNEXPECTED_ERROR")

        # Subprocess -I -O -B -W error smoke check on frozen on-disk cert if present
        cert_disk = HERE / "aletheion_live_llm_certificate.json"
        if cert_disk.is_file():
            proc = subprocess.run(
                [sys.executable, "-I", "-O", "-B", "-W", "error", str(HERE / "aletheion_live_llm_audit.py"), "--check", str(cert_disk)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIs(json.loads(proc.stdout)["valid"], True)


if __name__ == "__main__":
    unittest.main()
