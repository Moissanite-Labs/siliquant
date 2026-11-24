"""Verification & Mutation-Coverage Suite for RTX 3080 Siliquant A_16 Study.

Zero floats, zero float() calls, zero bare asserts, scratch os.mkfifo only,
zero secret* variable names.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from typing import Any
import sys
from unittest import mock

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import rtx3080_siliquant_a16_audit as audit
import rtx3080_siliquant_a16_oracle as oracle


def _resign(cert: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    out = copy.deepcopy(body)
    out["certificate_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(body)).hexdigest()
    return out


class _IntSub(int):
    pass


class _StrSub(str):
    pass


class TestOracleExactAndLiveReceipts(unittest.TestCase):
    def test_frac_helpers_and_syndromes(self) -> None:
        self.assertEqual(oracle.frac_pair(Fraction(9, 2)), [9, 2])
        with self.assertRaises(TypeError):
            oracle.frac_pair(9)  # type: ignore[arg-type]
        self.assertEqual(oracle._frac_from_token("356.5679"), Fraction(3565679, 10000))
        with self.assertRaises(ValueError):
            oracle._frac_from_token("")
        with self.assertRaises(ValueError):
            oracle._frac_from_token(123)  # type: ignore[arg-type]

        for j in range(16):
            self.assertEqual(
                oracle._syndrome_from_masks(1 << j, oracle.A16_PARITY_MASKS),
                1 | (j << 1),
            )
        for j in range(8):
            self.assertEqual(
                oracle._syndrome_from_masks(1 << j, oracle.A8_PARITY_MASKS),
                1 | (j << 1),
            )
        self.assertEqual(oracle._syndrome_from_masks(0, oracle.A16_PARITY_MASKS), 0)
        self.assertEqual(oracle._syndrome_from_masks(0xFFFF, oracle.A16_PARITY_MASKS), 0)

    def test_certificate_exact_values_and_digest(self) -> None:
        cert = oracle.build_certificate()
        self.assertEqual(cert["schema_version"], oracle.SCHEMA_VERSION)
        self.assertEqual(cert["oracle_module"], oracle.ORACLE_MODULE_NAME)
        self.assertEqual(
            cert["epistemic_taxonomy"],
            {"exact_pillar": "exact", "gpu_empirical_pillar": "sampled", "reference_baselines": "anchor"},
        )
        ham = cert["pillar_1_exact_hamming"]
        self.assertEqual(ham["a16_codeword_count"], 2048)
        self.assertEqual(ham["a16_min_distance"], 4)
        self.assertIs(ham["a16_all_even_weight"], True)
        self.assertEqual(ham["a16_kissing_number"], 2272)
        self.assertIs(ham["a16_single_flip_bijection"], True)
        self.assertEqual(ham["a8_codeword_count"], 16)
        self.assertEqual(ham["a8_min_distance"], 4)
        self.assertEqual(ham["a8_kissing_number"], 240)
        self.assertIs(ham["a8_single_flip_bijection"], True)
        self.assertEqual(
            ham["a16_weight_enumerator"],
            {"0": 1, "4": 140, "6": 448, "8": 870, "10": 448, "12": 140, "16": 1},
        )

        wire = cert["pillar_2_exact_wire_and_accumulator"]
        self.assertEqual(wire["int8_input_peak"], 39)
        self.assertEqual(wire["block_h2_unscaled_peak"], 78)
        self.assertIs(wire["block_h2_int8_safe"], True)
        self.assertEqual(wire["wht4_unscaled_peak"], 156)
        self.assertEqual(wire["wht8_unscaled_peak"], 312)
        self.assertEqual(wire["wht16_unscaled_peak"], 624)
        self.assertIs(wire["wht8_overflows_int8"], True)
        self.assertIs(wire["wht16_overflows_int8"], True)
        self.assertEqual(wire["f64_exact_integer_limit"], 1 << 53)
        self.assertIs(wire["linf_w64_exact_safe"], True)
        self.assertIs(wire["linf_w65_exceeds_f64"], True)
        self.assertEqual(wire["hampack_18b_bound"], 1207)
        self.assertEqual(wire["hampack_18b_max_coord"], 34)
        self.assertEqual(wire["hampack_18b_max_l1"], 196)
        self.assertIs(wire["hampack_18b_f64_safe"], True)
        self.assertEqual(wire["hampack_18b_count_bit_length"], 136)
        self.assertIs(wire["hampack_18b_count_fits_136b"], True)
        self.assertEqual(wire["hammingindex_72b_bound"], 5213)
        self.assertEqual(wire["hammingindex_72b_max_coord"], 72)
        self.assertIs(wire["hammingindex_72b_max_coord_gt_64"], True)
        self.assertIs(wire["hammingindex_72b_max_coord_fits_int8"], True)
        self.assertEqual(wire["hammingindex_72b_max_l1"], 816)
        self.assertIs(wire["hammingindex_72b_l2_ball_f64_safe"], True)
        self.assertEqual(wire["hammingindex_72b_count_bit_length"], 560)
        self.assertIs(wire["hammingindex_72b_count_fits_560b"], True)
        self.assertEqual(wire["bpw_18b_exact"], [9, 2])
        self.assertEqual(wire["bpw_72b_exact"], [9, 2])
        self.assertIs(wire["bpw_exact_match"], True)
        self.assertEqual(wire["weight_bytes_4096x4096"], 9437184)
        self.assertEqual(wire["weight_bytes_4096x14336"], 33030144)
        self.assertEqual(wire["rtx3080_theoretical_peak_bw_gb_s"], [48008, 125])
        self.assertEqual(wire["rtx3080_peak_4_5_bpw_gweights_s"], [768128, 1125])
        self.assertEqual(wire["rtx3080_l2_cache_bytes"], 4194304)
        self.assertEqual(wire["hampack_18b_table_bytes"], 30615552)
        self.assertEqual(wire["hammingindex_72b_compressed_table_bytes"], 24380664)

        samp = cert["pillar_3_sampled_rtx3080_and_anchors"]
        self.assertEqual(samp["oracle_class"], "sampled")
        self.assertIs(samp["measured_live_model_weights"], False)
        self.assertTrue(all(v is True for v in samp["sampled_gates"].values()))
        self.assertEqual(
            cert["verdict"],
            {"exact_invariants_pass": True, "sampled_gates_pass": True, "all_passed": True},
        )
        self.assertEqual(len(cert["certificate_sha256"]), 64)
        self.assertEqual(
            cert["certificate_sha256"],
            _resign(cert)["certificate_sha256"],
        )

        # Exercise every single clause in evaluate_exact_verdict
        self.assertIs(oracle.evaluate_exact_verdict(ham, wire), True)
        for hk, bad_v in (
            ("a16_codeword_count", 0),
            ("a16_min_distance", 2),
            ("a16_all_even_weight", False),
            ("a16_kissing_number", 0),
            ("a16_single_flip_bijection", False),
            ("a8_codeword_count", 0),
            ("a8_min_distance", 2),
            ("a8_kissing_number", 0),
            ("a8_single_flip_bijection", False),
        ):
            h_bad = copy.deepcopy(ham)
            h_bad[hk] = bad_v
            self.assertIs(oracle.evaluate_exact_verdict(h_bad, wire), False)
        for wk in (
            "block_h2_int8_safe",
            "wht8_overflows_int8",
            "wht16_overflows_int8",
            "linf_w64_exact_safe",
            "linf_w65_exceeds_f64",
            "hampack_18b_f64_safe",
            "hampack_18b_count_fits_136b",
            "hammingindex_72b_max_coord_gt_64",
            "hammingindex_72b_max_coord_fits_int8",
            "hammingindex_72b_l2_ball_f64_safe",
            "hammingindex_72b_count_fits_560b",
            "bpw_exact_match",
        ):
            w_bad = copy.deepcopy(wire)
            w_bad[wk] = False
            self.assertIs(oracle.evaluate_exact_verdict(ham, w_bad), False)

        # Exercise build_certificate verdict BoolOp when exact or sampled fails
        with mock.patch.object(oracle, "evaluate_exact_verdict", return_value=False):
            c_ex = oracle.build_certificate()
            self.assertIs(c_ex["verdict"]["exact_invariants_pass"], False)
            self.assertIs(c_ex["verdict"]["all_passed"], False)
        bad_samp = copy.deepcopy(samp)
        bad_samp["sampled_gates"]["ac_bf16_exceeds_battery"] = False
        with mock.patch.object(oracle, "_build_sampled_and_anchor_ratios", return_value=bad_samp):
            c_sp = oracle.build_certificate()
            self.assertIs(c_sp["verdict"]["sampled_gates_pass"], False)
            self.assertIs(c_sp["verdict"]["all_passed"], False)

        # Exercise individual BoolOp clauses inside _build_sampled_and_anchor_ratios
        for sec, k_flip, bad_tok, gate_name in (
            ("portal59_regime1_mean_matched_kl", "A16_axis", "0.9", "regime1_coherent_axis_beats_hadamard"),
            ("portal59_regime1_mean_matched_kl", "E8_axis", "0.9", "regime1_coherent_axis_beats_hadamard"),
            ("portal59_regime2_mean_matched_kl", "A16_H16", "2.0", "regime2_spikes_hadamard_beats_axis"),
            ("portal59_regime2_mean_matched_kl", "E8_H8", "2.0", "regime2_spikes_hadamard_beats_axis"),
        ):
            toks = copy.deepcopy(oracle.FROZEN_SAMPLED_TOKENS)
            toks[sec][k_flip] = bad_tok
            res_s = oracle._build_sampled_and_anchor_ratios(toks)
            self.assertIs(res_s["sampled_gates"][gate_name], False)

    def test_verify_live_receipts_pass_and_exhaustive_tampers(self) -> None:
        emp_raw = (STUDY_DIR / "rtx3080_siliquant_a16_empirical_results.json").read_bytes()
        mlos_raw = (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()
        cu_raw = (STUDY_DIR / "rtx3080_siliquant_a16_kernels.cu").read_bytes()
        emp_base = json.loads(emp_raw.decode("utf-8"), parse_float=Fraction)
        mlos_base = json.loads(mlos_raw.decode("utf-8"), parse_float=Fraction)
        emp_sha = hashlib.sha256(emp_raw).hexdigest()
        mlos_sha = hashlib.sha256(mlos_raw).hexdigest()
        cu_sha = hashlib.sha256(cu_raw).hexdigest()

        res = oracle.verify_live_receipts(emp_base, mlos_base, emp_sha, mlos_sha, cu_sha)
        self.assertIs(res["all_verified"], True)
        self.assertEqual(len(res["checks"]), 13)

        def check_fails(
            emp_mod: dict[str, Any] | None = None,
            mlos_mod: dict[str, Any] | None = None,
            e_sha: str = emp_sha,
            m_sha: str = mlos_sha,
            c_sha: str = cu_sha,
            target_key: str = "",
        ) -> None:
            r = oracle.verify_live_receipts(
                emp_base if emp_mod is None else emp_mod,
                mlos_base if mlos_mod is None else mlos_mod,
                e_sha,
                m_sha,
                c_sha,
            )
            self.assertIs(r["all_verified"], False, f"expected fail for {target_key}")
            if target_key:
                self.assertIs(r["checks"][target_key], False, f"expected {target_key} False")

        # Digest tampers
        check_fails(c_sha="0" * 64, target_key="cuda_sha256_match")
        e = copy.deepcopy(emp_base)
        e["cuda_source_sha256"] = "0" * 64
        check_fails(emp_mod=e, target_key="cuda_sha256_match")
        check_fails(e_sha="0" * 64, target_key="empirical_sha256_match")
        check_fails(m_sha="0" * 64, target_key="mlos_file_sha256_match")

        m = copy.deepcopy(mlos_base)
        m["receipt_sha256"] = "0" * 64
        check_fails(mlos_mod=m, target_key="mlos_receipt_sha256_match")

        # mlos_clean_zone_b tampers
        for field, bad_val in (
            ("clean_hardware_run", False),
            ("anomalies", ["BAD"]),
            ("determinism_zone", "A"),
        ):
            m = copy.deepcopy(mlos_base)
            m[field] = bad_val
            check_fails(mlos_mod=m, target_key="mlos_clean_zone_b")
        m = copy.deepcopy(mlos_base)
        m["execution"]["exit_code"] = 1
        check_fails(mlos_mod=m, target_key="mlos_clean_zone_b")

        # mlos_zero_desktop_processes tampers + boundary 4 MiB pass
        m_ok4 = copy.deepcopy(mlos_base)
        m_ok4["end_gpu_telemetry"]["linux_kmd_baseline_vram_mib"] = 4
        r_ok4 = oracle.verify_live_receipts(emp_base, m_ok4, emp_sha, mlos_sha, cu_sha)
        self.assertIs(r_ok4["checks"]["mlos_zero_desktop_processes"], True)
        for field, bad_val in (
            ("gpu_processes_queried", False),
            ("graphics_processes", [{"pid": 1}]),
            ("linux_kmd_baseline_vram_mib", True),
            ("linux_kmd_baseline_vram_mib", "2"),
            ("linux_kmd_baseline_vram_mib", 5),
        ):
            m = copy.deepcopy(mlos_base)
            m["end_gpu_telemetry"][field] = bad_val
            check_fails(mlos_mod=m, target_key="mlos_zero_desktop_processes")

        # Epistemic tags & ptxas
        for field, bad_val in (("oracle_class", "exact"), ("measured_live_model_weights", True)):
            e = copy.deepcopy(emp_base)
            e[field] = bad_val
            check_fails(emp_mod=e, target_key="epistemic_honesty_tags")

        for field, bad_val in (
            ("kernels_compiled", 23),
            ("max_spill_loads_bytes", 4),
            ("max_spill_stores_bytes", 4),
        ):
            e = copy.deepcopy(emp_base)
            e["bare_metal_sm86_cuda"]["ptxas_verification"][field] = bad_val
            check_fails(emp_mod=e, target_key="ptxas_24_kernels_zero_spill")

        # Direct GEMV 14 arms
        e = copy.deepcopy(emp_base)
        del e["bare_metal_sm86_cuda"]["fused_gemv_4_5_bpw"]["4096x4096"]["arms"]["A16_axis"]
        check_fails(emp_mod=e, target_key="direct_gemv_14_arms_verified")

        e = copy.deepcopy(emp_base)
        e["bare_metal_sm86_cuda"]["fused_gemv_4_5_bpw"]["4096x14336"]["arms"]["A16_axis"]["syndrome_errors"] = 1
        check_fails(emp_mod=e, target_key="direct_gemv_14_arms_verified")

        e = copy.deepcopy(emp_base)
        e["bare_metal_sm86_cuda"]["fused_gemv_4_5_bpw"]["4096x14336"]["arms"]["A16_axis"]["rel_l2_err_fp64"] = Fraction(1, 1000000)
        check_fails(emp_mod=e, target_key="direct_gemv_14_arms_verified")

        # PR #77 enumerative 18B & 72B tampers
        for meta_key, field, bad_val in (
            ("hampack_18B_meta", "bound", 1206),
            ("hampack_18B_meta", "count_limbs_hex", ["0x0", "0x0", "0x0"]),
            ("hampack_18B_meta", "pool_bit_mismatches", 1),
            ("hampack_18B_meta", "pool_syndrome_errors", 1),
            ("hammingindex_72B_meta", "bound", 5212),
            ("hammingindex_72B_meta", "count_limbs_hex", ["0x0"] * 9),
            ("hammingindex_72B_meta", "pool_bit_mismatches", 1),
            ("hammingindex_72B_meta", "pool_syndrome_errors", 1),
        ):
            e = copy.deepcopy(emp_base)
            e["bare_metal_sm86_cuda"]["siliquant_pr77_enumerative_wire_formats"][meta_key][field] = bad_val
            check_fails(emp_mod=e, target_key="pr77_enumerative_18b_72b_verified")

        for shape_k in ("4096x4096", "4096x14336"):
            for arm_k in ("hampack_18B_fused_gemv", "hammingindex_72B_fused_gemv"):
                e = copy.deepcopy(emp_base)
                e["bare_metal_sm86_cuda"]["siliquant_pr77_enumerative_wire_formats"]["shapes"][shape_k][arm_k]["syndrome_errors"] = 1
                check_fails(emp_mod=e, target_key="pr77_enumerative_18b_72b_verified")
                e = copy.deepcopy(emp_base)
                e["bare_metal_sm86_cuda"]["siliquant_pr77_enumerative_wire_formats"]["shapes"][shape_k][arm_k]["rel_l2_err_fp64"] = Fraction(1, 1000000)
                check_fails(emp_mod=e, target_key="pr77_enumerative_18b_72b_verified")
            for sa_k in ("hampack_18B_standalone", "hammingindex_72B_standalone"):
                e = copy.deepcopy(emp_base)
                e["bare_metal_sm86_cuda"]["siliquant_pr77_enumerative_wire_formats"]["shapes"][shape_k][sa_k]["syndrome_errors"] = 1
                check_fails(emp_mod=e, target_key="pr77_enumerative_18b_72b_verified")

        # Anisotropic Regime 1 tampers
        e = copy.deepcopy(emp_base)
        e["portal59_empirical_shootout"]["total_a16_syndrome_errors"] = 1
        check_fails(emp_mod=e, target_key="portal59_regime1_5_of_5_seeds")

        e = copy.deepcopy(emp_base)
        e["portal59_empirical_shootout"]["regime1_coherent_subspace"]["per_seed"].pop()
        check_fails(emp_mod=e, target_key="portal59_regime1_5_of_5_seeds")

        for fam in ("A16_family", "E8_family"):
            e = copy.deepcopy(emp_base)
            e["portal59_empirical_shootout"]["regime1_coherent_subspace"]["per_seed"][0]["matched_mse_max_db_diff"][fam] = Fraction(1, 1000000)
            check_fails(emp_mod=e, target_key="portal59_regime1_5_of_5_seeds")

        for k_lo, k_hi in (("A16_axis", "A16_H16"), ("E8_axis", "E8_H8")):
            e = copy.deepcopy(emp_base)
            row = e["portal59_empirical_shootout"]["regime1_coherent_subspace"]["per_seed"][0]["held_out_kl_quad_matched_mse"]
            row[k_lo] = row[k_hi]
            check_fails(emp_mod=e, target_key="portal59_regime1_5_of_5_seeds")

        # Anisotropic Regime 2 tampers
        e = copy.deepcopy(emp_base)
        e["portal59_empirical_shootout"]["regime2_axis_aligned_spikes"]["per_seed"].pop()
        check_fails(emp_mod=e, target_key="portal59_regime2_5_of_5_seeds")

        for fam in ("A16_family", "E8_family"):
            e = copy.deepcopy(emp_base)
            e["portal59_empirical_shootout"]["regime2_axis_aligned_spikes"]["per_seed"][0]["matched_mse_max_db_diff"][fam] = Fraction(1, 1000000)
            check_fails(emp_mod=e, target_key="portal59_regime2_5_of_5_seeds")

        for k_lo, k_hi in (("A16_H16", "A16_axis"), ("E8_H8", "E8_axis")):
            e = copy.deepcopy(emp_base)
            row = e["portal59_empirical_shootout"]["regime2_axis_aligned_spikes"]["per_seed"][0]["held_out_kl_quad_matched_mse"]
            row[k_lo] = row[k_hi]
            check_fails(emp_mod=e, target_key="portal59_regime2_5_of_5_seeds")

        # Calibration Experiment 3B tampers
        e = copy.deepcopy(emp_base)
        e["portal59_empirical_shootout"]["experiment3b_calibration_shootout"]["per_seed"].pop()
        check_fails(emp_mod=e, target_key="portal59_calib3b_5_of_5_seeds")

        for k_lo, k_hi in (
            ("hA16_real", "hA16_rand"),
            ("hA16_rand", "plain"),
            ("plain", "hA16_colproxy"),
        ):
            e = copy.deepcopy(emp_base)
            row = e["portal59_empirical_shootout"]["experiment3b_calibration_shootout"]["per_seed"][0]["A16_held_out_kl_quad"]
            row[k_lo] = row[k_hi]
            check_fails(emp_mod=e, target_key="portal59_calib3b_5_of_5_seeds")

        for field, bad_val in (
            ("p5_rand_retention_a16", Fraction(7, 10)),
            ("colproxy_retention_a16", Fraction(0, 1)),
            ("spearman_colproxy_vs_true_hessian", Fraction(0, 1)),
            ("spearman_real_vs_true_hessian", Fraction(0, 1)),
        ):
            e = copy.deepcopy(emp_base)
            e["portal59_empirical_shootout"]["experiment3b_calibration_shootout"]["per_seed"][0][field] = bad_val
            check_fails(emp_mod=e, target_key="portal59_calib3b_5_of_5_seeds")


class TestAuditSchemaSecondaryAndZoneB(unittest.TestCase):
    def test_strict_parser_tree_bijection_and_hardening(self) -> None:
        self.assertTrue(audit._is_hex64("0123456789abcdef" * 4))
        self.assertFalse(audit._is_hex64("A" * 64))
        self.assertFalse(audit._is_hex64("a" * 63))
        self.assertFalse(audit._is_hex64(123))

        # parse_strict_certificate_bytes rejections (including depth > MAX_JSON_DEPTH!)
        deep_nested = ('{"a":' * (audit.MAX_JSON_DEPTH + 1) + "0" + "}" * (audit.MAX_JSON_DEPTH + 1) + "\n").encode("utf-8")
        deep_2000 = ('{"a":' * 2000 + "0" + "}" * 2000 + "\n").encode("utf-8")
        for bad_raw, exp_code in (
            (b"", "NON_CANONICAL_FORMAT"),
            (b"{}", "NON_CANONICAL_FORMAT"),
            (b"{}\r\n", "NON_CANONICAL_FORMAT"),
            (b"\xff\n", "INVALID_UTF8"),
            (b'{"a":1.0}\n', "FLOAT_FORBIDDEN"),
            (b'{"a":NaN}\n', "CONSTANT_FORBIDDEN"),
            (b'{"a":1,"a":2}\n', "DUPLICATE_KEY"),
            (b'{"a":\n', "JSON_PARSE_ERROR"),
            (b"[1,2]\n", "INVALID_ROOT"),
            (b'{ "a": 1 }\n', "NON_CANONICAL_BYTES"),
            (deep_nested, "DEPTH_EXCEEDED"),
        ):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.parse_strict_certificate_bytes(bad_raw)
            self.assertEqual(ctx.exception.code, exp_code)

        # Depth-2000 nested JSON cleanly fails closed (DEPTH_EXCEEDED or JSON_PARSE_ERROR, never raw RecursionError)
        with self.assertRaises(audit.AuditError) as ctx:
            audit.parse_strict_certificate_bytes(deep_2000)
        self.assertIn(ctx.exception.code, ("DEPTH_EXCEEDED", "JSON_PARSE_ERROR"))

        # Exact depth & int bit boundaries
        d_ok: Any = 1
        for _ in range(audit.MAX_JSON_DEPTH):
            d_ok = [d_ok]
        audit._validate_json_tree_iter(d_ok)
        with self.assertRaises(audit.AuditError) as ctx:
            audit._validate_json_tree_iter([d_ok])
        self.assertEqual(ctx.exception.code, "DEPTH_EXCEEDED")

        audit._validate_json_tree_iter({"a": (1 << audit.MAX_INT_BITS) - 1})
        for bad_obj, exp_code in (
            ({1: "a"}, "INVALID_KEY_TYPE"),
            ({_StrSub("a"): 1}, "INVALID_KEY_TYPE"),
            ({"a": _IntSub(1)}, "INVALID_VALUE_TYPE"),
            ({"a": None}, "INVALID_VALUE_TYPE"),
            ({"a": 1 << audit.MAX_INT_BITS}, "INT_TOO_LARGE"),
        ):
            with self.assertRaises(audit.AuditError) as ctx:
                audit._validate_json_tree_iter(bad_obj)
            self.assertEqual(ctx.exception.code, exp_code)

        # Non-dict root rejection on secondary_audit_payload
        for non_dict in ([], "str", 123, True, None):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.secondary_audit_payload(non_dict)
            self.assertEqual(ctx.exception.code, "INVALID_ROOT")

        # Valid baseline passes both full_replay=False and full_replay=True
        base = oracle.build_certificate()
        self.assertIs(audit.verify_certificate_payload(base, full_replay=False)["valid"], True)
        self.assertIs(audit.verify_certificate_payload(base, full_replay=True)["valid"], True)

        # Bad digests
        c = copy.deepcopy(base)
        del c["certificate_sha256"]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.secondary_audit_payload(c)
        self.assertEqual(ctx.exception.code, "INVALID_DIGEST")

        c = copy.deepcopy(base)
        c["certificate_sha256"] = "not-hex64"
        with self.assertRaises(audit.AuditError) as ctx:
            audit.secondary_audit_payload(c)
        self.assertEqual(ctx.exception.code, "INVALID_DIGEST")

        c = copy.deepcopy(base)
        c["certificate_sha256"] = "0" * 64
        with self.assertRaises(audit.AuditError) as ctx:
            audit.secondary_audit_payload(c)
        self.assertEqual(ctx.exception.code, "DIGEST_MISMATCH")

        # Bool-in-int tamper: bool-in-int tamper on a16_weight_enumerator ("0": True, "16": True)
        c = copy.deepcopy(base)
        c["pillar_1_exact_hamming"]["a16_weight_enumerator"]["0"] = True
        c["pillar_1_exact_hamming"]["a16_weight_enumerator"]["16"] = True
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_TYPE_MISMATCH")

        # Forged ratio tamper: hollow forged ratios [41,1], [11,1], [999999,1]
        c = copy.deepcopy(base)
        c["pillar_2_exact_wire_and_accumulator"]["hampack_18b_f64_margin"] = [41, 1]
        c["pillar_2_exact_wire_and_accumulator"]["hammingindex_72b_f64_margin"] = [11, 1]
        c["pillar_3_sampled_rtx3080_and_anchors"]["preenc_over_hammingindex_72b_gemv_speedup"] = [999999, 1]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "INVARIANT_MISMATCH")

        # Keyset & list-length mismatches
        c = copy.deepcopy(base)
        c["extra"] = 1
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_KEY_MISMATCH")

        c = copy.deepcopy(base)
        c["pillar_2_exact_wire_and_accumulator"]["bpw_18b_exact"] = [9, 2, 1]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_LEN_MISMATCH")

        # full_replay=True REPLAY_MISMATCH branch
        fake_diff = copy.deepcopy(base)
        fake_diff["schema_version"] = "other"
        with mock.patch.object(oracle, "build_certificate", return_value=fake_diff):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_certificate_payload(base, full_replay=True)
            self.assertEqual(ctx.exception.code, "REPLAY_MISMATCH")

    def test_zone_b_io_pwd_symlink_atomic_write_and_cli(self) -> None:
        cert = oracle.build_certificate()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cert_path = root / "cert.json"

            # Reject non-dict & non-JSON/NaN payloads before write_certificate_atomic
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(cert_path, [])  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, "INVALID_ROOT")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(cert_path, {"x": None})
            self.assertEqual(ctx.exception.code, "INVALID_VALUE_TYPE")
            nan_val = json.loads("NaN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(cert_path, {"x": nan_val})
            self.assertEqual(ctx.exception.code, "INVALID_VALUE_TYPE")
            with self.assertRaises(ValueError):
                audit._strict_canonical_json_bytes({"x": nan_val})

            # Atomic 0o600 write & overwrite
            sha1 = audit.write_certificate_atomic(cert_path, cert)
            self.assertEqual(stat.S_IMODE(os.lstat(cert_path).st_mode), 0o600)
            res = audit.verify_certificate_file(cert_path, full_replay=True)
            self.assertEqual(res["file_sha256"], sha1)
            self.assertEqual(audit.write_certificate_atomic(cert_path, cert), sha1)

            # Logical $PWD symlink check on relative path
            real_sub = root / "real_sub"
            real_sub.mkdir()
            sub_cert = real_sub / "cert.json"
            audit.write_certificate_atomic(sub_cert, cert)
            sym_sub = root / "sym_sub"
            os.symlink(real_sub, sym_sub)

            prev_cwd = os.getcwd()
            try:
                os.chdir(real_sub)
                with mock.patch.dict(os.environ, {"PWD": str(real_sub)}):
                    self.assertEqual(audit.verify_certificate_file(Path("./cert.json"))["file_sha256"], sha1)
                    dfd = audit._open_dir_nofollow(Path("."))
                    os.close(dfd)
                with mock.patch.dict(os.environ, {"PWD": str(sym_sub)}):
                    with self.assertRaises(audit.AuditError) as ctx:
                        audit.verify_certificate_file(Path("cert.json"))
                    self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
            finally:
                os.chdir(prev_cwd)

            # Empty path, root '/', '..', missing dir/file, regular-file-as-dir, anchor fail, ELOOP
            with self.assertRaises(audit.AuditError) as ctx:
                audit._open_parent_dir_nofollow(Path(""))
            self.assertEqual(ctx.exception.code, "EMPTY_PATH")
            with self.assertRaises(audit.AuditError) as ctx:
                audit._open_parent_dir_nofollow(Path("/"))
            self.assertEqual(ctx.exception.code, "INVALID_BASENAME")
            with self.assertRaises(audit.AuditError) as ctx:
                audit._open_parent_dir_nofollow(root / "..")
            self.assertEqual(ctx.exception.code, "INVALID_BASENAME")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(root / "no_such_dir" / "x.json")
            self.assertEqual(ctx.exception.code, "DIR_OPEN_FAILED")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(cert_path / "child.json")
            self.assertEqual(ctx.exception.code, "DIR_OPEN_FAILED")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(root / "missing.json")
            self.assertEqual(ctx.exception.code, "FILE_OPEN_FAILED")

            real_open = os.open

            def fake_anchor_fail(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if p == "/":
                    raise OSError(errno.EACCES, "denied")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_siliquant_a16_audit.os.open", side_effect=fake_anchor_fail):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("/",), True)
                self.assertEqual(ctx.exception.code, "DIR_OPEN_FAILED")

            def fake_eloop(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if p == "eloop_comp":
                    raise OSError(errno.ELOOP, "too many levels of symbolic links")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_siliquant_a16_audit.os.open", side_effect=fake_eloop):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("eloop_comp",), False)
                self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            # chmod 000, symlink, hardlink, scratch os.mkfifo non-regular probe
            unreadable = root / "no_perm.json"
            unreadable.write_bytes(b"{}\n")
            os.chmod(unreadable, 0o000)
            try:
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(unreadable)
                self.assertEqual(ctx.exception.code, "FILE_OPEN_FAILED")
            finally:
                os.chmod(unreadable, 0o600)

            sym_file = root / "sym_cert.json"
            os.symlink(cert_path, sym_file)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(sym_file)
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(sym_file, cert)
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(sym_sub / "cert.json")
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(sym_sub)
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            hl_src = root / "hl_src.json"
            hl_dst = root / "hl_dst.json"
            audit.write_certificate_atomic(hl_src, cert)
            os.link(hl_src, hl_dst)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(hl_dst)
            self.assertEqual(ctx.exception.code, "HARDLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(hl_dst, cert)
            self.assertEqual(ctx.exception.code, "HARDLINK_FORBIDDEN")

            fifo_path = root / "scratch.fifo"
            os.mkfifo(fifo_path, 0o600)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(fifo_path)
            self.assertEqual(ctx.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(fifo_path, cert)
            self.assertEqual(ctx.exception.code, "NON_REGULAR_FILE")

            # Exact size boundary, stat fast-reject before os.read, and streaming cap
            exact_f = root / "exact4.json"
            exact_f.write_bytes(b"1234")
            self.assertEqual(audit.read_regular_file_bytes(exact_f, max_bytes=4), b"1234")

            big = root / "big.json"
            big.write_bytes(b"a" * (audit.MAX_CERT_BYTES + 1))
            with mock.patch("rtx3080_siliquant_a16_audit.os.read") as m_read:
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(big)
                self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")
                m_read.assert_not_called()

            small = root / "small.json"
            small.write_bytes(b"1234567890")
            with mock.patch("rtx3080_siliquant_a16_audit.os.fstat") as m_fstat:
                fake_st = os.stat(small)
                m_fstat.return_value = mock.Mock(st_mode=fake_st.st_mode, st_nlink=1, st_size=0)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(small, max_bytes=4)
                self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")

            # Bounded EEXIST retry exhaustion, short write, temp tampered nlink/mode, OSError replace
            def fake_eexist(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if isinstance(p, str) and ".tmp." in p:
                    raise FileExistsError("exists")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_siliquant_a16_audit.os.open", side_effect=fake_eexist):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "retry.json", cert)
                self.assertEqual(ctx.exception.code, "ATOMIC_WRITE_FAILED")
                self.assertIn("exhausted", ctx.exception.message)

            with mock.patch("rtx3080_siliquant_a16_audit.os.write", return_value=0):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "short.json", cert)
                self.assertEqual(ctx.exception.code, "SHORT_WRITE")
            self.assertEqual([p.name for p in root.glob(".short.json.tmp.*")], [])

            with mock.patch("rtx3080_siliquant_a16_audit.os.fstat") as m_fs2:
                m_fs2.return_value = mock.Mock(st_mode=stat.S_IFREG | 0o600, st_nlink=2)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "tamper1.json", cert)
                self.assertEqual(ctx.exception.code, "TEMP_TAMPERED")
            with mock.patch("rtx3080_siliquant_a16_audit.os.fstat") as m_fs3:
                m_fs3.return_value = mock.Mock(st_mode=stat.S_IFIFO | 0o600, st_nlink=1)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "tamper2.json", cert)
                self.assertEqual(ctx.exception.code, "TEMP_TAMPERED")

            with mock.patch("rtx3080_siliquant_a16_audit.os.replace", side_effect=OSError(errno.EIO, "io")):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "oserr.json", cert)
                self.assertEqual(ctx.exception.code, "ATOMIC_WRITE_FAILED")
            self.assertEqual([p.name for p in root.glob(".oserr.json.tmp.*")], [])

            # verify_receipts_directory: live pass + malformed/empty/{}/tamper (fail-closed schema check)
            self.assertIs(audit.verify_receipts_directory(STUDY_DIR)["all_verified"], True)

            bad_rdir = root / "bad_receipts"
            bad_rdir.mkdir()
            (bad_rdir / "rtx3080_siliquant_a16_empirical_results.json").write_bytes(b"[]\n")
            (bad_rdir / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            (bad_rdir / "rtx3080_siliquant_a16_kernels.cu").write_bytes(b"// cu\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_rdir)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            (bad_rdir / "rtx3080_siliquant_a16_empirical_results.json").write_bytes(b"{}\n")
            (bad_rdir / "mlos_zone_b_receipt.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_rdir)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            # Empty {} receipts return RECEIPT_SCHEMA_ERROR (closes raw KeyError!)
            (bad_rdir / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_rdir)
            self.assertEqual(ctx.exception.code, "RECEIPT_SCHEMA_ERROR")

            (bad_rdir / "rtx3080_siliquant_a16_empirical_results.json").write_bytes(b"{bad\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_rdir)
            self.assertEqual(ctx.exception.code, "RECEIPT_PARSE_ERROR")

            (bad_rdir / "rtx3080_siliquant_a16_empirical_results.json").write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_rdir)
            self.assertEqual(ctx.exception.code, "DUPLICATE_KEY")

            emp_ok = (STUDY_DIR / "rtx3080_siliquant_a16_empirical_results.json").read_bytes()
            mlos_ok = (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()
            (bad_rdir / "rtx3080_siliquant_a16_empirical_results.json").write_bytes(emp_ok)
            (bad_rdir / "mlos_zone_b_receipt.json").write_bytes(mlos_ok)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_rdir)
            self.assertEqual(ctx.exception.code, "RECEIPT_VERIFICATION_FAILED")

            # CLI tests (including NUL & UTF-8 surrogate paths)
            cli_cert = root / "cli_cert.json"
            for argv, expected_rc, expected_err in (
                ([], 0, None),
                (["--standalone", "--json"], 0, None),
                (["--write", str(cli_cert)], 0, None),
                (["--check", str(cli_cert)], 0, None),
                (["--verify-receipts", str(STUDY_DIR)], 0, None),
                (["--check", "-flag"], 2, "INVALID_CLI_PATH"),
                (["--check", ""], 2, "INVALID_CLI_PATH"),
                (["--check", "a\x00b"], 2, "INVALID_CLI_PATH"),
                (["--check", "bad\ud800.json"], 2, "INVALID_CLI_PATH"),
                (["--check"], 2, "USAGE_ERROR"),
                (["--check", str(cli_cert), "extra"], 2, "USAGE_ERROR"),
                (["--write"], 2, "USAGE_ERROR"),
                (["--write", str(cli_cert), "extra"], 2, "USAGE_ERROR"),
                (["--verify-receipts"], 2, "USAGE_ERROR"),
                (["--verify-receipts", str(STUDY_DIR), "extra"], 2, "USAGE_ERROR"),
                (["--bogus", "x"], 2, "USAGE_ERROR"),
                (["--bogus"], 2, "USAGE_ERROR"),
            ):
                out_buf = io.BytesIO()
                err_buf = io.BytesIO()
                out_wrap = io.TextIOWrapper(out_buf, encoding="utf-8")
                err_wrap = io.TextIOWrapper(err_buf, encoding="utf-8")
                with redirect_stdout(out_wrap), redirect_stderr(err_wrap):
                    rc = audit.main(argv)
                out_wrap.flush()
                err_wrap.flush()
                self.assertEqual(rc, expected_rc, f"CLI {argv} returned {rc}")
                if expected_rc == 0:
                    parsed = json.loads(out_buf.getvalue().decode("utf-8"))
                    self.assertIsInstance(parsed, dict)
                else:
                    parsed_err = json.loads(err_buf.getvalue().decode("utf-8"))
                    self.assertIs(parsed_err["valid"], False)
                    self.assertEqual(parsed_err["error_code"], expected_err)

    def test_disk_certificate_matches_standalone(self) -> None:
        disk_cert = STUDY_DIR / "rtx3080_siliquant_a16_certificate.json"
        res = audit.verify_certificate_file(disk_cert, full_replay=True)
        self.assertIs(res["valid"], True)
        self.assertEqual(stat.S_IMODE(os.lstat(disk_cert).st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
