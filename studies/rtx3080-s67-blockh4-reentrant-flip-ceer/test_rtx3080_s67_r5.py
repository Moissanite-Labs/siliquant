"""R5 & Zone-B Verification and Full-Mutation-Killing Suite for Track S67.

Zero floats, zero float() calls, zero bare asserts, scratch os.mkfifo only.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import rtx3080_s67_audit as audit
import rtx3080_s67_oracle as oracle


def _resign(cert: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    out = copy.deepcopy(body)
    out["certificate_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(body)).hexdigest()
    return out


class _IntSub(int):
    pass


class _StrSub(str):
    pass


class TestS67OracleAndReceipts(unittest.TestCase):
    def test_helpers_exact_verdict_and_gates(self) -> None:
        self.assertEqual(oracle.frac_pair(Fraction(9, 2)), [9, 2])
        with self.assertRaises(TypeError):
            oracle.frac_pair(9)  # type: ignore[arg-type]
        self.assertEqual(oracle._frac_tok("0.045183018"), Fraction(45183018, 1000000000))
        with self.assertRaises(ValueError):
            oracle._frac_tok("")
        with self.assertRaises(ValueError):
            oracle._frac_tok(123)  # type: ignore[arg-type]

        cert = oracle.build_certificate()
        self.assertEqual(cert["schema_version"], oracle.SCHEMA_VERSION)
        self.assertEqual(cert["oracle_module"], oracle.ORACLE_MODULE_NAME)
        p1 = cert["pillar_1_exact_4model_and_s59_block_h"]
        self.assertEqual(p1["models_audited"], 4)
        self.assertEqual(p1["total_arms_evaluated"], 29)
        self.assertEqual(p1["qwen_3scale_diagnosed_layers"], 616)
        self.assertEqual(p1["combined_4model_matrices"], 840)
        self.assertEqual(p1["combined_4model_weights"], 4757127168)
        self.assertEqual(p1["a16_rho_axis_by_b"], {"b1": 1, "b2": 2, "b4": 1, "b8": 2, "b16": 4})
        self.assertIs(p1["block_h4_restores_rho_axis_1_and_dyadic"], True)

        p2 = cert["pillar_2_sampled_live_ceer_cycle2"]
        self.assertEqual(p2["oracle_class"], "sampled")
        self.assertIs(p2["measured_live_model_weights"], True)
        self.assertTrue(all(v is True for v in p2["sampled_gates"].values()))
        self.assertEqual(
            cert["verdict"],
            {"exact_invariants_pass": True, "sampled_gates_pass": True, "all_passed": True},
        )

        # Boundary negations for block_h4_restores_rho_axis_1_and_dyadic BoolOp
        with mock.patch.object(oracle, "_a16_syndrome", return_value=1):
            self.assertIs(oracle._build_exact_model_and_s59_invariants()["block_h4_restores_rho_axis_1_and_dyadic"], False)
        with mock.patch.object(oracle.math, "isqrt", return_value=1):
            self.assertIs(oracle._build_exact_model_and_s59_invariants()["block_h4_restores_rho_axis_1_and_dyadic"], False)

        # Flip every clause of evaluate_exact_verdict
        for k, bad in (
            ("models_audited", 3),
            ("total_arms_evaluated", 28),
            ("smollm2_matrices", 0),
            ("smollm2_weights", 0),
            ("qwen05b_matrices", 0),
            ("qwen05b_weights", 0),
            ("qwen15b_matrices", 0),
            ("qwen15b_weights", 0),
            ("qwen3b_matrices", 0),
            ("qwen3b_weights", 0),
            ("qwen_3scale_diagnosed_layers", 0),
            ("combined_4model_matrices", 0),
            ("combined_4model_weights", 0),
            ("a16_rho_axis_by_b", {"b1": 1}),
            ("block_h4_restores_rho_axis_1_and_dyadic", False),
        ):
            p1_b = copy.deepcopy(p1)
            p1_b[k] = bad
            self.assertIs(oracle.evaluate_exact_verdict(p1_b), False)

        # Flip both BoolOp clauses in build_certificate
        with mock.patch.object(oracle, "evaluate_exact_verdict", return_value=False):
            c_e = oracle.build_certificate()
            self.assertIs(c_e["verdict"]["all_passed"], False)
        p2_bad = copy.deepcopy(p2)
        p2_bad["sampled_gates"]["p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"] = False
        with mock.patch.object(oracle, "_build_s67_sampled_ratios", return_value=p2_bad):
            c_s = oracle.build_certificate()
            self.assertIs(c_s["verdict"]["all_passed"], False)

        # Flip EVERY SINGLE subcondition of EVERY sampled gate
        gate_flips: tuple[tuple[tuple[str, ...], Any, str], ...] = (
            (("smollm2", "wt_kl", "A16_BlockH4"), "0.09", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen05b", "wt_kl", "A16_BlockH8"), "0.05", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen05b", "wt_kl", "A16_H16"), "0.05", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen15b", "wt_kl", "A16_axis"), "0.04", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen15b", "wt_kl", "A16_H16"), "0.04", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen3b", "wt_kl", "A16_BlockH8"), "0.04", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen3b", "wt_kl", "A16_BlockH2"), "0.04", "p_s67_1_block_h4_beats_block_h8_on_smollm2_05b_3b_and_h16_on_05b_15b"),
            (("qwen05b", "py_kl", "A16_axis"), "0.04", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen05b", "py_kl", "A16_BlockH2"), "0.04", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen15b", "py_kl", "A16_axis"), "0.02", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen3b", "py_kl", "E8_H8"), "0.0356", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen3b", "py_kl", "A16_axis"), "0.03", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen3b", "py_kl", "A16_H16"), "0.03", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen3b", "py_kl", "A16_BlockH4_matched"), "0.04", "p_s67_1_block_h4_wins_python_05b_and_crushes_e8_h8_and_axis_on_3b_python"),
            (("qwen05b", "z0_pct", "A16_BlockH4"), "39", "p_s67_1_dyadic_zero_retention_signature_live_in_weights"),
            (("qwen05b", "z0_pct", "A16_BlockH2"), "11", "p_s67_1_dyadic_zero_retention_signature_live_in_weights"),
            (("qwen05b", "z0_pct", "A16_BlockH8"), "3", "p_s67_1_dyadic_zero_retention_signature_live_in_weights"),
            (("qwen3b", "z0_pct", "A16_BlockH4"), "29", "p_s67_1_dyadic_zero_retention_signature_live_in_weights"),
            (("qwen3b", "z0_pct", "A16_BlockH2"), "11", "p_s67_1_dyadic_zero_retention_signature_live_in_weights"),
            (("qwen3b", "z0_pct", "A16_BlockH8"), "3", "p_s67_1_dyadic_zero_retention_signature_live_in_weights"),
            (("qwen05b", "wt_kl", "A16_axis"), "0.06", "p_s67_1_honest_steelman_nuance_axis_keeps_05b_and_3b_wiki"),
            (("qwen3b", "wt_kl", "A16_axis"), "0.05", "p_s67_1_honest_steelman_nuance_axis_keeps_05b_and_3b_wiki"),
            (("qwen3b", "wt_kl", "E8_H8"), "0.05", "p_s67_1_honest_steelman_nuance_axis_keeps_05b_and_3b_wiki"),
            (("qwen05b", "geom", "mean_col_outlier_ratio"), "2.0", "p_s67_2_zero_cal_weight_order_params_uniquely_min_at_15b"),
            (("qwen3b", "geom", "mean_col_outlier_ratio"), "2.0", "p_s67_2_zero_cal_weight_order_params_uniquely_min_at_15b"),
            (("qwen05b", "geom", "mean_small_weight_frac_05"), "0.04", "p_s67_2_zero_cal_weight_order_params_uniquely_min_at_15b"),
            (("qwen3b", "geom", "mean_small_weight_frac_05"), "0.04", "p_s67_2_zero_cal_weight_order_params_uniquely_min_at_15b"),
            (("qwen05b", "geom", "down_proj_kurtosis"), "1.0", "p_s67_2_zero_cal_weight_order_params_uniquely_min_at_15b"),
            (("qwen3b", "geom", "down_proj_kurtosis"), "1.0", "p_s67_2_zero_cal_weight_order_params_uniquely_min_at_15b"),
            (("qwen15b", "geom", "down_proj_quad_err_share"), "0.69", "p_s67_2_down_proj_distortion_spike_at_15b_and_gate_diag_explosion_at_3b"),
            (("qwen05b", "geom", "down_proj_quad_err_share"), "0.31", "p_s67_2_down_proj_distortion_spike_at_15b_and_gate_diag_explosion_at_3b"),
            (("qwen3b", "geom", "down_proj_quad_err_share"), "0.46", "p_s67_2_down_proj_distortion_spike_at_15b_and_gate_diag_explosion_at_3b"),
            (("qwen3b", "geom", "gate_proj_rand_diag_conc"), "100.0", "p_s67_2_down_proj_distortion_spike_at_15b_and_gate_diag_explosion_at_3b"),
            (("qwen05b", "geom", "bh4_beats_axis_count"), 84, "p_s67_2_block_h4_beats_axis_majority_of_616_layers"),
            (("qwen15b", "geom", "bh4_beats_axis_count"), 98, "p_s67_2_block_h4_beats_axis_majority_of_616_layers"),
            (("qwen3b", "geom", "bh4_beats_axis_count"), 126, "p_s67_2_block_h4_beats_axis_majority_of_616_layers"),
            (("smollm2", "e05_db", "A16_zero_lam20"), "2.5", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("smollm2", "e05_db", "A16_axis"), "2.1", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("qwen3b", "e05_db", "A16_zero_lam20"), "1.8", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("qwen3b", "e05_db", "A16_axis"), "1.4", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("smollm2", "wt_kl", "A16_zero_lam20"), "0.065", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("smollm2", "wt_kl", "A16_axis"), "0.0647", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("smollm2", "py_kl", "A16_zero_lam05"), "0.06", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("qwen3b", "wt_kl", "A16_zero_lam05"), "0.04", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
            (("qwen3b", "wt_kl", "A16_zero_lam20"), "0.04", "p_s67_3_proposal4_opposite_sign_smollm2_helps_qwen3b_hurts"),
        )
        for path_t, bad_v, gname in gate_flips:
            toks = copy.deepcopy(oracle.FROZEN_S67_TOKENS)
            cur: Any = toks
            for k in path_t[:-1]:
                cur = cur[k]
            cur[path_t[-1]] = bad_v
            out = oracle._build_s67_sampled_ratios(toks)
            self.assertIs(out["sampled_gates"][gname], False, f"flip {path_t} did not fail {gname}")

    def test_verify_live_receipts_pass_and_all_tampers(self) -> None:
        prereg_raw = (STUDY_DIR / "s67_preregistration.json").read_bytes()
        emp_raw = (STUDY_DIR / "rtx3080_s67_empirical_results.json").read_bytes()
        mlos_raw = (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()
        prereg_base = json.loads(prereg_raw.decode("utf-8"))
        emp_base = json.loads(emp_raw.decode("utf-8"), parse_float=Fraction)
        mlos_base = json.loads(mlos_raw.decode("utf-8"), parse_float=Fraction)
        p_sha = hashlib.sha256(prereg_raw).hexdigest()
        e_sha = hashlib.sha256(emp_raw).hexdigest()
        m_sha = hashlib.sha256(mlos_raw).hexdigest()

        res = oracle.verify_live_receipts(prereg_base, emp_base, mlos_base, p_sha, e_sha, m_sha)
        self.assertIs(res["all_verified"], True)
        self.assertEqual(len(res["checks"]), 11)

        def assert_fails(
            pr_m: dict[str, Any] | None = None,
            emp_m: dict[str, Any] | None = None,
            mlos_m: dict[str, Any] | None = None,
            ps: str = p_sha,
            es: str = e_sha,
            ms: str = m_sha,
            ck: str = "",
        ) -> None:
            r = oracle.verify_live_receipts(
                prereg_base if pr_m is None else pr_m,
                emp_base if emp_m is None else emp_m,
                mlos_base if mlos_m is None else mlos_m,
                ps,
                es,
                ms,
            )
            self.assertIs(r["all_verified"], False, f"expected fail for {ck}")
            if ck:
                self.assertIs(r["checks"][ck], False, f"expected {ck} False")

        # digests_match (all 8 clauses)
        assert_fails(ps="0" * 64, ck="digests_match")
        pr = copy.deepcopy(prereg_base)
        pr["preregistration_sha256"] = "0" * 64
        assert_fails(pr_m=pr, ck="digests_match")
        pr = copy.deepcopy(prereg_base)
        pr["epistemic_protocol"] = "tampered"
        assert_fails(pr_m=pr, ck="digests_match")
        e = copy.deepcopy(emp_base)
        e["preregistration_file_sha256"] = "0" * 64
        assert_fails(emp_m=e, ck="digests_match")
        e = copy.deepcopy(emp_base)
        e["preregistration_sha256"] = "0" * 64
        assert_fails(emp_m=e, ck="digests_match")
        assert_fails(es="0" * 64, ck="digests_match")
        assert_fails(ms="0" * 64, ck="digests_match")
        m = copy.deepcopy(mlos_base)
        m["receipt_sha256"] = "0" * 64
        assert_fails(mlos_m=m, ck="digests_match")

        # prereg_strictly_precedes_execution
        pr = copy.deepcopy(prereg_base)
        pr["created_utc"] = 123
        assert_fails(pr_m=pr, ck="prereg_strictly_precedes_execution")
        m = copy.deepcopy(mlos_base)
        m["execution"]["started_utc"] = 123
        assert_fails(mlos_m=m, ck="prereg_strictly_precedes_execution")
        pr = copy.deepcopy(prereg_base)
        pr["created_utc"] = mlos_base["execution"]["started_utc"]
        assert_fails(pr_m=pr, ck="prereg_strictly_precedes_execution")

        # mlos_clean_zone_b & mlos_zero_desktop_processes
        for fld, bv in (("clean_hardware_run", False), ("anomalies", ["X"]), ("determinism_zone", "A")):
            m = copy.deepcopy(mlos_base)
            m[fld] = bv
            assert_fails(mlos_m=m, ck="mlos_clean_zone_b")
        m = copy.deepcopy(mlos_base)
        m["execution"]["exit_code"] = 1
        assert_fails(mlos_m=m, ck="mlos_clean_zone_b")

        m_b4 = copy.deepcopy(mlos_base)
        m_b4["end_gpu_telemetry"]["linux_kmd_baseline_vram_mib"] = 4
        self.assertIs(
            oracle.verify_live_receipts(prereg_base, emp_base, m_b4, p_sha, e_sha, m_sha)["checks"]["mlos_zero_desktop_processes"],
            True,
        )
        for fld, bv in (
            ("gpu_processes_queried", False),
            ("graphics_processes", [{"pid": 1}]),
            ("linux_kmd_baseline_vram_mib", True),
            ("linux_kmd_baseline_vram_mib", 5),
        ):
            m = copy.deepcopy(mlos_base)
            m["end_gpu_telemetry"][fld] = bv
            assert_fails(mlos_m=m, ck="mlos_zero_desktop_processes")

        # real_weight_honesty_tags & corpora_pins_match
        for fld, bv in (("oracle_class", "exact"), ("measured_live_model_weights", False), ("ac_online", False)):
            e = copy.deepcopy(emp_base)
            e[fld] = bv
            assert_fails(emp_m=e, ck="real_weight_honesty_tags")

        for cf in ("wikitext2_revision", "wikitext2_test_sha256", "python_stdlib_sha256"):
            e = copy.deepcopy(emp_base)
            e["corpora"][cf] = "0" * 40
            assert_fails(emp_m=e, ck="corpora_pins_match")

        # all_4_model_pins_match
        for mn in ("HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
            for fld in ("revision", "safetensors_sha256"):
                e = copy.deepcopy(emp_base)
                e["model_pins"][mn][fld] = "0" * 40
                assert_fails(emp_m=e, ck="all_4_model_pins_match")
        e = copy.deepcopy(emp_base)
        e["model_pins"]["Qwen/Qwen2.5-3B"]["revision"] = "0" * 40
        assert_fails(emp_m=e, ck="all_4_model_pins_match")
        e = copy.deepcopy(emp_base)
        e["model_pins"]["Qwen/Qwen2.5-3B"]["safetensors_shards"] = e["model_pins"]["Qwen/Qwen2.5-3B"]["safetensors_shards"][:1]
        assert_fails(emp_m=e, ck="all_4_model_pins_match")
        for si in (0, 1):
            e = copy.deepcopy(emp_base)
            e["model_pins"]["Qwen/Qwen2.5-3B"]["safetensors_shards"][si]["sha256"] = "0" * 64
            assert_fails(emp_m=e, ck="all_4_model_pins_match")

        # all_29_real_arms_clean
        e = copy.deepcopy(emp_base)
        del e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_BlockH2"]
        assert_fails(emp_m=e, ck="all_29_real_arms_clean")
        for af, bv in (
            ("syndrome_errors", 1),
            ("non_block_tensors_differ", 1),
            ("max_matched_mse_db_diff", Fraction(1, 100000)),
        ):
            e = copy.deepcopy(emp_base)
            e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_axis"][af] = bv
            assert_fails(emp_m=e, ck="all_29_real_arms_clean")

        # p_s67_1_block_h4_python_3b_win_and_parity
        for mn, corp, lo, hi in (
            ("Qwen/Qwen2.5-3B", "python", "A16_BlockH4", "E8_H8"),
            ("Qwen/Qwen2.5-3B", "python", "A16_BlockH4", "A16_axis"),
            ("Qwen/Qwen2.5-0.5B", "python", "A16_BlockH4", "A16_axis"),
            ("Qwen/Qwen2.5-1.5B", "wikitext2", "A16_BlockH4", "A16_axis"),
            ("Qwen/Qwen2.5-3B", "wikitext2", "A16_BlockH4", "A16_BlockH8"),
        ):
            e = copy.deepcopy(emp_base)
            e["models"][mn]["arms"][lo][corp]["mean_kl"] = e["models"][mn]["arms"][hi][corp]["mean_kl"]
            assert_fails(emp_m=e, ck="p_s67_1_block_h4_python_3b_win_and_parity")
        # S67-VER-04: isolate A16_BlockH4 < A16_axis on 3B Python while keeping A16_BlockH4 < E8_H8 true
        e = copy.deepcopy(emp_base)
        e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_axis"]["python"]["mean_kl"] = Fraction(0)
        assert_fails(emp_m=e, ck="p_s67_1_block_h4_python_3b_win_and_parity")

        # p_s67_2_616_layer_order_params_separate_15b
        e = copy.deepcopy(emp_base)
        e["models"]["Qwen/Qwen2.5-3B"]["structural_geometry"]["total_layers"] = 250
        assert_fails(emp_m=e, ck="p_s67_2_616_layer_order_params_separate_15b")
        for mn, fld, bv in (
            ("Qwen/Qwen2.5-0.5B", "mean_col_outlier_ratio", Fraction(2, 1)),
            ("Qwen/Qwen2.5-3B", "mean_col_outlier_ratio", Fraction(2, 1)),
            ("Qwen/Qwen2.5-0.5B", "mean_small_weight_frac_05", Fraction(1, 100)),
            ("Qwen/Qwen2.5-3B", "mean_small_weight_frac_05", Fraction(1, 100)),
        ):
            e = copy.deepcopy(emp_base)
            e["models"][mn]["structural_geometry"][fld] = bv
            assert_fails(emp_m=e, ck="p_s67_2_616_layer_order_params_separate_15b")

        # p_s67_3_proposal4_opposite_sign_verified
        e = copy.deepcopy(emp_base)
        e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_zero_lam20"]["small_weight_e05_db"] = Fraction(3, 1)
        assert_fails(emp_m=e, ck="p_s67_3_proposal4_opposite_sign_verified")
        e = copy.deepcopy(emp_base)
        e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_zero_lam05"]["small_weight_e05_db"] = Fraction(3, 1)
        assert_fails(emp_m=e, ck="p_s67_3_proposal4_opposite_sign_verified")
        e = copy.deepcopy(emp_base)
        e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_zero_lam20"]["wikitext2"]["mean_kl"] = Fraction(1, 10)
        assert_fails(emp_m=e, ck="p_s67_3_proposal4_opposite_sign_verified")
        e = copy.deepcopy(emp_base)
        e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_zero_lam20"]["small_weight_e05_db"] = Fraction(3, 1)
        assert_fails(emp_m=e, ck="p_s67_3_proposal4_opposite_sign_verified")
        e = copy.deepcopy(emp_base)
        e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_zero_lam20"]["wikitext2"]["mean_kl"] = Fraction(1, 100)
        assert_fails(emp_m=e, ck="p_s67_3_proposal4_opposite_sign_verified")


class TestS67AuditHardening(unittest.TestCase):
    def test_strict_parser_tree_bijection_and_vectors(self) -> None:
        self.assertTrue(audit._is_hex64("0123456789abcdef" * 4))
        self.assertFalse(audit._is_hex64("A" * 64))
        self.assertFalse(audit._is_hex64("a" * 63))
        self.assertFalse(audit._is_hex64(1))

        deep_json = ('{"a":' * 70 + "0" + "}" * 70 + "\n").encode("utf-8")
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
            (deep_json, "DEPTH_EXCEEDED"),
        ):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.parse_strict_certificate_bytes(bad_raw)
            self.assertEqual(ctx.exception.code, exp_code)

        ok_d: Any = 1
        for _ in range(audit.MAX_JSON_DEPTH):
            ok_d = [ok_d]
        audit._validate_json_tree_iter(ok_d)
        audit._validate_json_tree_iter({"i": (1 << audit.MAX_INT_BITS) - 1, "b": True, "s": "x"})

        for bad_obj, exp_code in (
            ({1: "a"}, "INVALID_KEY_TYPE"),
            ({_StrSub("a"): 1}, "INVALID_KEY_TYPE"),
            ({"\ud800": 1}, "INVALID_UTF8_SURROGATE"),
            ({"a": "\ud800"}, "INVALID_UTF8_SURROGATE"),
            ({"a": _IntSub(1)}, "INVALID_VALUE_TYPE"),
            ({"a": None}, "INVALID_VALUE_TYPE"),
            ({"a": 1 << audit.MAX_INT_BITS}, "INT_TOO_LARGE"),
        ):
            with self.assertRaises(audit.AuditError) as ctx:
                audit._validate_json_tree_iter(bad_obj)
            self.assertEqual(ctx.exception.code, exp_code)

        for non_dict in ([], "str", 123, True, None):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.secondary_audit_payload(non_dict)
            self.assertEqual(ctx.exception.code, "INVALID_ROOT")

        base = oracle.build_certificate()
        self.assertIs(audit.verify_certificate_payload(base, full_replay=False)["valid"], True)
        self.assertIs(audit.verify_certificate_payload(base, full_replay=True)["valid"], True)

        # Exact bool check on full_replay BEFORE any disk I/O (S63-AUD-02 closure + M060 killer!)
        for bad_flag in (1, 0, "false", None):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_certificate_payload(base, full_replay=bad_flag)  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, "INVALID_FLAG_TYPE")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_certificate_file(STUDY_DIR / "nonexistent_flag_check.json", full_replay=bad_flag)  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, "INVALID_FLAG_TYPE")

        c = copy.deepcopy(base)
        del c["certificate_sha256"]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.secondary_audit_payload(c)
        self.assertEqual(ctx.exception.code, "INVALID_DIGEST")

        c = copy.deepcopy(base)
        c["certificate_sha256"] = "not-valid-hex64"
        with self.assertRaises(audit.AuditError) as ctx:
            audit.secondary_audit_payload(c)
        self.assertEqual(ctx.exception.code, "INVALID_DIGEST")

        c = copy.deepcopy(base)
        c["certificate_sha256"] = "0" * 64
        with self.assertRaises(audit.AuditError) as ctx:
            audit.secondary_audit_payload(c)
        self.assertEqual(ctx.exception.code, "DIGEST_MISMATCH")

        c = copy.deepcopy(base)
        c["pillar_1_exact_4model_and_s59_block_h"]["models_audited"] = True
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_TYPE_MISMATCH")

        c = copy.deepcopy(base)
        c["extra"] = 1
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_KEY_MISMATCH")

        c = copy.deepcopy(base)
        c["pillar_2_sampled_live_ceer_cycle2"]["qwen3b_bh4_over_e8_h8_py_kl"] = [1, 1, 0]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_LEN_MISMATCH")

        c = copy.deepcopy(base)
        c["pillar_2_sampled_live_ceer_cycle2"]["qwen3b_bh4_over_e8_h8_py_kl"] = [999999, 1]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "INVARIANT_MISMATCH")

        fake_diff = copy.deepcopy(base)
        fake_diff["schema_version"] = "other"
        with mock.patch.object(oracle, "build_certificate", return_value=fake_diff):
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_certificate_payload(base, full_replay=True)
            self.assertEqual(ctx.exception.code, "REPLAY_MISMATCH")

    def test_zone_b_io_pwd_darwin_atomic_write_and_cli(self) -> None:
        cert = oracle.build_certificate()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cert_path = root / "cert.json"

            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(cert_path, [])  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, "INVALID_ROOT")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(cert_path, {"x": None})
            self.assertEqual(ctx.exception.code, "INVALID_VALUE_TYPE")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(cert_path, {"x": "\ud800"})
            self.assertEqual(ctx.exception.code, "INVALID_UTF8_SURROGATE")

            sha1 = audit.write_certificate_atomic(cert_path, cert)
            self.assertEqual(stat.S_IMODE(os.lstat(cert_path).st_mode), 0o600)
            res = audit.verify_certificate_file(cert_path, full_replay=True)
            self.assertEqual(res["file_sha256"], sha1)
            self.assertEqual(audit.write_certificate_atomic(cert_path, cert), sha1)

            real_sub = root / "real_sub"
            real_sub.mkdir()
            sub_cert = real_sub / "cert.json"
            audit.write_certificate_atomic(sub_cert, cert)
            sym_sub = root / "sym_sub"
            os.symlink(real_sub, sym_sub)
            sym_other = root / "sym_other"
            os.symlink(root, sym_other)

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
                for benign_pwd in (str(sym_other), "/nonexistent_s67_pwd_check"):
                    with mock.patch.dict(os.environ, {"PWD": benign_pwd}):
                        self.assertEqual(audit.verify_certificate_file(Path("cert.json"))["file_sha256"], sha1)
                with mock.patch.object(audit.os, "environ", {"PWD": "/bad\x00pwd"}):
                    self.assertEqual(audit.verify_certificate_file(Path("cert.json"))["file_sha256"], sha1)
            finally:
                os.chdir(prev_cwd)

            class _FakeSt:
                def __init__(self, st_mode: int, st_uid: int) -> None:
                    self.st_mode = st_mode
                    self.st_uid = st_uid

            root_lnk = _FakeSt(stat.S_IFLNK | 0o777, 0)
            user_lnk = _FakeSt(stat.S_IFLNK | 0o777, 501)
            root_dir = _FakeSt(stat.S_IFDIR | 0o755, 0)

            def _fake_readlink(p: str) -> str:
                if p == "/var":
                    return "private/var"
                if p == "/tmp":
                    return "/private/tmp"
                if p == "/etc":
                    return "private/etc"
                raise OSError("unexpected readlink")

            with mock.patch.object(audit.sys, "platform", "linux"):
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))

            with (
                mock.patch.object(audit.sys, "platform", "darwin"),
                mock.patch.object(audit.os, "lstat", return_value=root_lnk),
                mock.patch.object(audit.os, "readlink", side_effect=_fake_readlink),
            ):
                self.assertEqual(
                    audit._normalize_darwin_system_prefix(("/", "var", "folders", "xx")),
                    ("/", "private", "var", "folders", "xx"),
                )
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "tmp")), ("/", "private", "tmp"))
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "etc", "x")), ("/", "private", "etc", "x"))
                self.assertEqual(audit._normalize_darwin_system_prefix(("/",)), ("/",))
                self.assertEqual(audit._normalize_darwin_system_prefix(("rel", "var")), ("rel", "var"))
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "tmp", "..", "etc")), ("/", "tmp", "..", "etc"))
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "usr", "bin")), ("/", "usr", "bin"))

            with (
                mock.patch.object(audit.sys, "platform", "darwin"),
                mock.patch.object(audit.os, "lstat", return_value=user_lnk),
                mock.patch.object(audit.os, "readlink", side_effect=_fake_readlink),
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))

            with (
                mock.patch.object(audit.sys, "platform", "darwin"),
                mock.patch.object(audit.os, "lstat", return_value=root_dir),
                mock.patch.object(audit.os, "readlink", side_effect=_fake_readlink),
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))

            with (
                mock.patch.object(audit.sys, "platform", "darwin"),
                mock.patch.object(audit.os, "lstat", side_effect=OSError("no alias")),
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))

            with (
                mock.patch.object(audit.sys, "platform", "darwin"),
                mock.patch.object(audit.os, "lstat", return_value=root_lnk),
                mock.patch.object(audit.os, "readlink", return_value="private/evil"),
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))

            with self.assertRaises(audit.AuditError) as ctx:
                audit._open_parent_dir_nofollow(Path(""))
            self.assertEqual(ctx.exception.code, "EMPTY_PATH")
            with self.assertRaises(audit.AuditError) as ctx:
                audit._open_parent_dir_nofollow(Path("/"))
            self.assertEqual(ctx.exception.code, "INVALID_BASENAME")
            with self.assertRaises(audit.AuditError) as ctx:
                audit._open_parent_dir_nofollow(root / "..")
            self.assertEqual(ctx.exception.code, "INVALID_BASENAME")
            # S67-VER-03: direct Python API rejection of lone-surrogate and NUL paths across all 3 entrypoints
            for bad_api_path in (Path("/tmp/\ud800.json"), Path("/tmp/a\x00b.json")):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(bad_api_path)
                self.assertEqual(ctx.exception.code, "INVALID_PATH")
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(bad_api_path, cert)
                self.assertEqual(ctx.exception.code, "INVALID_PATH")
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.verify_receipts_directory(bad_api_path)
                self.assertEqual(ctx.exception.code, "INVALID_PATH")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(root / "no_dir" / "a.json")
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

            with mock.patch("rtx3080_s67_audit.os.open", side_effect=fake_anchor_fail):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("/",), True)
                self.assertEqual(ctx.exception.code, "DIR_OPEN_FAILED")

            def fake_eloop(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if p == "eloop_x":
                    raise OSError(errno.ELOOP, "loop")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_s67_audit.os.open", side_effect=fake_eloop):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("eloop_x",), False)
                self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            def fake_enotdir(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if p == "enotdir_lnk":
                    raise OSError(errno.ENOTDIR, "not a dir")
                return real_open(p, flg, *a, **kw)

            with (
                mock.patch("rtx3080_s67_audit.os.open", side_effect=fake_enotdir),
                mock.patch("rtx3080_s67_audit.os.lstat", return_value=root_lnk),
            ):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("enotdir_lnk",), False)
                self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            unr = root / "unr.json"
            unr.write_bytes(b"{}\n")
            os.chmod(unr, 0o000)
            try:
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(unr)
                self.assertEqual(ctx.exception.code, "FILE_OPEN_FAILED")
            finally:
                os.chmod(unr, 0o600)

            sym_f = root / "sym.json"
            os.symlink(cert_path, sym_f)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(sym_f)
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(sym_f, cert)
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(sym_sub / "cert.json")
            self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            hl1 = root / "hl1.json"
            hl2 = root / "hl2.json"
            audit.write_certificate_atomic(hl1, cert)
            os.link(hl1, hl2)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(hl2)
            self.assertEqual(ctx.exception.code, "HARDLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(hl2, cert)
            self.assertEqual(ctx.exception.code, "HARDLINK_FORBIDDEN")

            fifo = root / "scratch.fifo"
            os.mkfifo(fifo, 0o600)
            with self.assertRaises(audit.AuditError) as ctx:
                audit.read_regular_file_bytes(fifo)
            self.assertEqual(ctx.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.write_certificate_atomic(fifo, cert)
            self.assertEqual(ctx.exception.code, "NON_REGULAR_FILE")

            ex4 = root / "ex4.json"
            ex4.write_bytes(b"1234")
            self.assertEqual(audit.read_regular_file_bytes(ex4, max_bytes=4), b"1234")

            big = root / "big.json"
            big.write_bytes(b"a" * (audit.MAX_CERT_BYTES + 1))
            with mock.patch("rtx3080_s67_audit.os.read") as m_read:
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(big)
                self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")
                m_read.assert_not_called()

            smf = root / "smf.json"
            smf.write_bytes(b"1234567890")
            with mock.patch("rtx3080_s67_audit.os.fstat") as m_fs:
                fst = os.stat(smf)
                m_fs.return_value = mock.Mock(st_mode=fst.st_mode, st_nlink=1, st_size=0)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(smf, max_bytes=4)
                self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")

            # S67-VER-02: wrapped OSError on os.fstat and os.read inside _read_regular_bytes_in_dir
            for sc_name in ("fstat", "read"):
                with mock.patch(f"rtx3080_s67_audit.os.{sc_name}", side_effect=OSError(errno.EIO, "EIO")):
                    with self.assertRaises(audit.AuditError) as ctx:
                        audit.read_regular_file_bytes(ex4)
                    self.assertEqual(ctx.exception.code, "FILE_READ_FAILED")

            def fake_eexist(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if isinstance(p, str) and ".tmp." in p:
                    raise FileExistsError("exists")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_s67_audit.os.open", side_effect=fake_eexist):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "retry.json", cert)
                self.assertEqual(ctx.exception.code, "ATOMIC_WRITE_FAILED")
                self.assertIn("exhausted", ctx.exception.message)

            with mock.patch("rtx3080_s67_audit.os.write", return_value=0):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "short.json", cert)
                self.assertEqual(ctx.exception.code, "SHORT_WRITE")

            with mock.patch("rtx3080_s67_audit.os.fstat") as m_fs2:
                m_fs2.return_value = mock.Mock(st_mode=stat.S_IFREG | 0o600, st_nlink=2)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "tamper1.json", cert)
                self.assertEqual(ctx.exception.code, "TEMP_TAMPERED")
            with mock.patch("rtx3080_s67_audit.os.fstat") as m_fs3:
                m_fs3.return_value = mock.Mock(st_mode=stat.S_IFIFO | 0o600, st_nlink=1)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "tamper2.json", cert)
                self.assertEqual(ctx.exception.code, "TEMP_TAMPERED")

            with mock.patch("rtx3080_s67_audit.os.replace", side_effect=OSError(errno.EIO, "io")):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "oserr.json", cert)
                self.assertEqual(ctx.exception.code, "ATOMIC_WRITE_FAILED")

            # verify_receipts_directory: live pass + malformed/empty/{}/tamper
            self.assertIs(audit.verify_receipts_directory(STUDY_DIR)["all_verified"], True)

            bad_r = root / "bad_r"
            bad_r.mkdir()
            (bad_r / "s67_preregistration.json").write_bytes(b"[]\n")
            (bad_r / "rtx3080_s67_empirical_results.json").write_bytes(b"{}\n")
            (bad_r / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            (bad_r / "s67_preregistration.json").write_bytes(b"{}\n")
            (bad_r / "rtx3080_s67_empirical_results.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            (bad_r / "rtx3080_s67_empirical_results.json").write_bytes(b"{}\n")
            (bad_r / "mlos_zone_b_receipt.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            (bad_r / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_SCHEMA_ERROR")

            (bad_r / "rtx3080_s67_empirical_results.json").write_bytes(b"{bad\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_PARSE_ERROR")

            (bad_r / "rtx3080_s67_empirical_results.json").write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "DUPLICATE_KEY")

            # Valid structure but mismatched prereg digest -> RECEIPT_VERIFICATION_FAILED
            pr_tamper = json.loads((STUDY_DIR / "s67_preregistration.json").read_text())
            pr_tamper["preregistration_sha256"] = "0" * 64
            (bad_r / "s67_preregistration.json").write_bytes(oracle.canonical_json_bytes(pr_tamper))
            (bad_r / "rtx3080_s67_empirical_results.json").write_bytes(
                (STUDY_DIR / "rtx3080_s67_empirical_results.json").read_bytes()
            )
            (bad_r / "mlos_zone_b_receipt.json").write_bytes(
                (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()
            )
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_VERIFICATION_FAILED")

            # S67-VER-01: ZeroDivisionError ("1/0") in receipt JSON cleanly caught on API and live CLI
            emp_zd = json.loads((STUDY_DIR / "rtx3080_s67_empirical_results.json").read_text())
            emp_zd["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_axis"]["max_matched_mse_db_diff"] = "1/0"
            (bad_r / "s67_preregistration.json").write_bytes((STUDY_DIR / "s67_preregistration.json").read_bytes())
            (bad_r / "rtx3080_s67_empirical_results.json").write_text(json.dumps(emp_zd) + "\n", encoding="utf-8")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_SCHEMA_ERROR")
            zd_ob, zd_eb = io.BytesIO(), io.BytesIO()
            zd_ow = io.TextIOWrapper(zd_ob, encoding="utf-8")
            zd_ew = io.TextIOWrapper(zd_eb, encoding="utf-8")
            with redirect_stdout(zd_ow), redirect_stderr(zd_ew):
                zd_rc = audit.main(["--verify-receipts", str(bad_r)])
            zd_ow.flush()
            zd_ew.flush()
            self.assertEqual(zd_rc, 2)
            self.assertEqual(json.loads(zd_eb.getvalue().decode("utf-8"))["error_code"], "RECEIPT_SCHEMA_ERROR")

            # CLI tests (incl. non-list, ["--check", 123], ["--check", _StrSub], NUL, surrogate, flag, arity)
            cli_c = root / "cli_c.json"
            for argv, exp_rc, exp_err in (
                ([], 0, None),
                (["--standalone", "--json"], 0, None),
                (["--write", str(cli_c)], 0, None),
                (["--check", str(cli_c)], 0, None),
                (["--verify-receipts", str(STUDY_DIR)], 0, None),
                (123, 2, "USAGE_ERROR"),
                (["--check", 123], 2, "USAGE_ERROR"),
                (["--check", _StrSub(str(cli_c))], 2, "USAGE_ERROR"),
                (["--check", "-flag"], 2, "INVALID_CLI_PATH"),
                (["--check", ""], 2, "INVALID_CLI_PATH"),
                (["--check", "a\x00b"], 2, "INVALID_CLI_PATH"),
                (["--check", "bad\ud800.json"], 2, "INVALID_CLI_PATH"),
                (["--check"], 2, "USAGE_ERROR"),
                (["--check", str(cli_c), "x"], 2, "USAGE_ERROR"),
                (["--write"], 2, "USAGE_ERROR"),
                (["--write", str(cli_c), "x"], 2, "USAGE_ERROR"),
                (["--verify-receipts"], 2, "USAGE_ERROR"),
                (["--verify-receipts", str(STUDY_DIR), "x"], 2, "USAGE_ERROR"),
                (["--bogus", "x"], 2, "USAGE_ERROR"),
                (["--bogus"], 2, "USAGE_ERROR"),
            ):
                ob, eb = io.BytesIO(), io.BytesIO()
                ow = io.TextIOWrapper(ob, encoding="utf-8")
                ew = io.TextIOWrapper(eb, encoding="utf-8")
                with redirect_stdout(ow), redirect_stderr(ew):
                    rc = audit.main(argv)  # type: ignore[arg-type]
                ow.flush()
                ew.flush()
                self.assertEqual(rc, exp_rc, f"CLI {argv} -> {rc}")
                if exp_rc == 0:
                    self.assertIsInstance(json.loads(ob.getvalue().decode("utf-8")), dict)
                else:
                    pe = json.loads(eb.getvalue().decode("utf-8"))
                    self.assertIs(pe["valid"], False)
                    self.assertEqual(pe["error_code"], exp_err)

    def test_disk_certificate_matches_standalone(self) -> None:
        disk_cert = STUDY_DIR / "rtx3080_s67_certificate.json"
        res = audit.verify_certificate_file(disk_cert, full_replay=True)
        self.assertIs(res["valid"], True)
        self.assertEqual(stat.S_IMODE(os.lstat(disk_cert).st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
