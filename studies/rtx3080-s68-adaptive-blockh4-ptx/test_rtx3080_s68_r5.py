"""Comprehensive R5 Unit & Adversarial Verification Suite for Adaptive Block-H4 Router & sm_86 PTX Study.

Covers 100% of oracle & audit mutation surfaces, whole-file DFS AST BoolOp clauses,
ZeroDivisionError, EIO on fstat/read, surrogate/NUL paths on direct API & CLI,
ENOSPC on atomic writer stages, O_NONBLOCK FIFO, empty-list schema rejection,
strict bool flags, bool-in-int receipt rejection, inner receipt hash verification,
inode-matched $PWD, and Darwin prefix normalization.
Zero floats, zero bare asserts.
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
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import rtx3080_s68_audit as audit
import rtx3080_s68_oracle as oracle


class _IntSub(int):
    pass


class _StrSub(str):
    pass


class _DictSub(dict):
    pass


class _ListSub(list):
    pass


def _resign(cert: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    out = dict(body)
    out["certificate_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(body)).hexdigest()
    return out


class TestS68OracleAndReceipts(unittest.TestCase):
    def test_helpers_exact_verdict_and_gates(self) -> None:
        self.assertEqual(oracle.q_pair(Fraction(9, 2)), [9, 2])
        self.assertEqual(oracle._to_q(Fraction(3, 5)), Fraction(3, 5))
        self.assertEqual(oracle._to_q(7), Fraction(7, 1))
        self.assertEqual(oracle._to_q("0.25"), Fraction(1, 4))
        with self.assertRaises(TypeError):
            oracle._to_q(None)
        with self.assertRaises(TypeError):
            oracle._to_q(True)

        self.assertTrue(oracle._is_int_eq(0, 0))
        self.assertFalse(oracle._is_int_eq(False, 0))
        self.assertFalse(oracle._is_int_eq(1, 0))
        # Reject floats inside canonical_json_bytes & test exact rational serializer
        rt_float = json.loads("1.5")
        with self.assertRaises(TypeError):
            oracle.canonical_json_bytes({"x": rt_float})
        with self.assertRaises(TypeError):
            oracle.canonical_json_bytes([rt_float])
        self.assertEqual(
            oracle._exact_compact_json_bytes(
                {"a": [None, True, False, 0, Fraction(4, 1), Fraction(-5, 2), Fraction(125, 1000)]}
            ),
            b'{"a":[null,true,false,0,4,-2.5,0.125]}',
        )
        with self.assertRaises(ValueError):
            oracle._exact_compact_json_bytes({"bad": Fraction(1, 3)})
        with self.assertRaises(TypeError):
            oracle._exact_compact_json_bytes({"bad": rt_float})

        # ISO-8601 timestamp parser boundary tests (kills every BoolOp clause!)
        self.assertEqual(oracle._parse_iso_utc_seconds("1970-01-01T00:00:00Z"), 0)
        self.assertLess(
            oracle._parse_iso_utc_seconds("2026-11-24T16:29:04Z"),
            oracle._parse_iso_utc_seconds("2026-11-24T16:34:06Z"),
        )
        for bad_ts in (
            12345,
            "2026-11-24T16:29:04",
            "2026-11-24T16:29:04extraZ",
            "2026-11-24T16:29:04X",
            "2026/11-24T16:29:04Z",
            "2026-11/24T16:29:04Z",
            "2026-11-24 16:29:04Z",
            "2026-11-24T16.29:04Z",
            "2026-11-24T16:29.04Z",
            "2026-00-24T16:29:04Z",
            "2026-13-24T16:29:04Z",
            "2026-11-00T16:29:04Z",
            "2026-11-32T16:29:04Z",
            "2026-11-24T24:29:04Z",
            "2026-11-24T16:60:04Z",
            "2026-11-24T16:29:60Z",
        ):
            with self.assertRaises(ValueError, msg=f"bad_ts={bad_ts!r}"):
                oracle._parse_iso_utc_seconds(bad_ts)  # type: ignore[arg-type]

        cert = oracle.build_certificate()
        self.assertTrue(cert["verdict"]["all_passed"])
        self.assertTrue(cert["verdict"]["exact_invariants_pass"])
        self.assertTrue(cert["verdict"]["sampled_gates_pass"])
        p1 = cert["pillar_1_exact_router_and_h4_arithmetic"]
        self.assertEqual(p1["combined_4model_matrices"], 840)
        self.assertEqual(p1["combined_4model_weights"], 4757127168)
        self.assertEqual(p1["total_arms_evaluated"], 46)
        self.assertEqual(p1["block_h4_max_butterfly_abs"], 28)
        self.assertTrue(p1["block_h4_lossless_int8_dp4a"])
        self.assertTrue(p1["qwen3b_1bit_below_1e7_bpw"])

        p2 = cert["pillar_2_sampled_live_ceer_cycle3"]
        self.assertTrue(p2["measured_live_model_weights"])
        self.assertEqual(p2["oracle_class"], "sampled")
        self.assertEqual(p2["four_model_zero_1bit_oracle_agreement"], [639, 840])
        self.assertEqual(p2["qwen3b_zero_2bit_oracle_agreement"], [214, 252])
        self.assertTrue(all(p2["sampled_gates"].values()))

        # Kill every single BoolOp sub-condition in _build_sampled_pillar & build_certificate!
        def _run_tampered_tokens(mut_fn: Any) -> dict[str, Any]:
            saved = copy.deepcopy(oracle.FROZEN_S68_TOKENS)
            try:
                mut_fn(oracle.FROZEN_S68_TOKENS)
                return oracle.build_certificate()
            finally:
                oracle.FROZEN_S68_TOKENS.clear()
                oracle.FROZEN_S68_TOKENS.update(saved)

        token_tampers = [
            lambda t: t["smollm2"].__setitem__("agree_1bit", 150),
            lambda t: t["qwen05b"].__setitem__("agree_1bit", 100),
            lambda t: t["qwen15b"].__setitem__("agree_1bit", 120),
            lambda t: t["qwen3b"].__setitem__("agree_1bit", 160),
            lambda t: t["smollm2"]["wt_kl"].__setitem__("A16_AdaptZeroCal_2bit", "0.062000000"),
            lambda t: t["smollm2"]["wt_kl"].__setitem__("A16_AdaptZeroCal_1bit", "0.062900000"),
            lambda t: t["smollm2"]["wt_kl"].__setitem__("A16_AdaptWeightOnly_1bit", "0.066000000"),
            lambda t: t["smollm2"]["wt_kl"].__setitem__("A16_BlockH4", "0.064000000"),
            lambda t: t["smollm2"]["py_kl"].__setitem__("A16_AdaptWeightOnly_1bit", "0.053000000"),
            lambda t: t["smollm2"]["py_kl"].__setitem__("A16_AdaptZeroCal_1bit", "0.057000000"),
            lambda t: t["smollm2"]["wt_kl"].__setitem__("Oracle_1bit", "0.055000000"),
            lambda t: t["qwen05b"]["py_kl"].__setitem__("E8_H8", "0.042000000"),
            lambda t: t["qwen05b"]["py_kl"].__setitem__("A16_BlockH4", "0.042000000"),
            lambda t: t["qwen3b"]["wt_kl"].__setitem__("A16_BlockH4", "0.045000000"),
            lambda t: t["qwen3b"]["wt_kl"].__setitem__("Oracle_1bit", "0.047000000"),
            lambda t: t["qwen3b"]["wt_kl"].__setitem__("A16_AdaptZeroCal_1bit", "0.050000000"),
            lambda t: t["qwen05b"]["wt_kl"].__setitem__("hA16_BlockH2_real", "0.029000000"),
            lambda t: t["qwen05b"]["wt_kl"].__setitem__("hA16_BlockH4_real", "0.027500000"),
            lambda t: t["qwen15b"]["wt_kl"].__setitem__("hA16_BlockH2_real", "0.023500000"),
            lambda t: t["qwen15b"]["wt_kl"].__setitem__("hA16_BlockH4_real", "0.024000000"),
            lambda t: t["qwen3b"]["wt_kl"].__setitem__("hA16_BlockH2_real", "0.024600000"),
            lambda t: t["qwen3b"]["wt_kl"].__setitem__("hA16_BlockH4_real", "0.024300000"),
            lambda t: t["qwen05b"]["py_kl"].__setitem__("hA16_BlockH4_real", "0.024500000"),
            lambda t: t["qwen15b"]["wt_ppl"].__setitem__("hA16_BlockH4_real", "12.900000"),
            lambda t: t["qwen3b"]["py_kl"].__setitem__("hA16_BlockH4_real", "0.023500000"),
            lambda t: t["qwen3b"]["py_ppl"].__setitem__("hA16_BlockH4_real", "3.180000"),
            lambda t: t["qwen3b"]["py_kl"].__setitem__("hA16_real", "0.022600000"),
            lambda t: t["cuda_ptx"].__setitem__("max_stack_bytes", 8),
            lambda t: t["cuda_ptx"].__setitem__("max_spill_stores_bytes", 4),
            lambda t: t["cuda_ptx"].__setitem__("max_spill_loads_bytes", 4),
            lambda t: t["cuda_ptx"].__setitem__("ptx_ldl_count", 1),
            lambda t: t["cuda_ptx"].__setitem__("ptx_stl_count", 1),
            lambda t: t["cuda_ptx"].__setitem__("stacked2_bh4_gws", "569.9"),
            lambda t: t["cuda_ptx"].__setitem__("stacked2_axis_gws", "650.0"),
            lambda t: t["cuda_ptx"].__setitem__("single_bh4_gws", "570.0"),
        ]
        for idx, fn in enumerate(token_tampers):
            bad_c = _run_tampered_tokens(fn)
            self.assertFalse(bad_c["verdict"]["sampled_gates_pass"], f"token tamper #{idx} not caught")
            self.assertFalse(bad_c["verdict"]["all_passed"], f"token tamper #{idx} passed all_passed")

        # Kill exact_ok BoolOp clauses in build_certificate
        for patch_k, patch_v in (
            ("combined_4model_matrices", 839),
            ("combined_4model_weights", 1),
            ("total_arms_evaluated", 45),
            ("block_h4_lossless_int8_dp4a", False),
            ("qwen3b_1bit_below_1e7_bpw", False),
        ):
            orig_p1 = oracle._build_exact_pillar()
            orig_p1[patch_k] = patch_v
            with mock.patch.object(oracle, "_build_exact_pillar", return_value=orig_p1):
                c_bad = oracle.build_certificate()
                self.assertFalse(c_bad["verdict"]["exact_invariants_pass"], f"exact patch {patch_k}")
                self.assertFalse(c_bad["verdict"]["all_passed"], f"exact all_passed {patch_k}")

    def test_verify_live_receipts_pass_and_all_tampers(self) -> None:
        res = audit.verify_receipts_directory(STUDY_DIR)
        self.assertTrue(res["all_verified"])
        self.assertTrue(all(res["checks"].values()))

        prereg_raw = (STUDY_DIR / "s68_preregistration.json").read_bytes()
        cuda_raw = (STUDY_DIR / "sm86_a16_blockh4_kernel.cu").read_bytes()
        emp_raw = (STUDY_DIR / "rtx3080_s68_empirical_results.json").read_bytes()
        mlos_raw = (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()

        prereg_base = json.loads(prereg_raw.decode("utf-8"))
        emp_base = json.loads(emp_raw.decode("utf-8"), parse_float=Fraction)
        mlos_base = json.loads(mlos_raw.decode("utf-8"), parse_float=Fraction)
        p_sha = hashlib.sha256(prereg_raw).hexdigest()
        c_sha = hashlib.sha256(cuda_raw).hexdigest()
        e_sha = hashlib.sha256(emp_raw).hexdigest()
        m_sha = hashlib.sha256(mlos_raw).hexdigest()

        def _check(p: Any, e: Any, m: Any, ps: str = p_sha, cs: str = c_sha, es: str = e_sha, ms: str = m_sha) -> dict[str, Any]:
            return oracle.verify_live_receipts(p, e, m, ps, cs, es, ms)

        def _resign_prereg_inner(p: dict[str, Any], _e: Any, _m: Any) -> None:
            p["epistemic_protocol"] = "resigned_different_prereg"
            body = {k: v for k, v in p.items() if k != "preregistration_sha256"}
            p["preregistration_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(body)).hexdigest()

        def _resign_mlos_inner(_p: Any, _e: Any, m: dict[str, Any]) -> None:
            m["profile_id"] = "resigned_different_profile"
            body = {k: v for k, v in m.items() if k != "receipt_sha256"}
            m["receipt_sha256"] = hashlib.sha256(oracle._exact_compact_json_bytes(body)).hexdigest()

        # 1. Every single digest check clause (incl. re-signed inner body vs pinned digest!)
        for idx, (ps, cs, es, ms, mut) in enumerate((
            ("0" * 64, c_sha, e_sha, m_sha, None),
            (p_sha, "0" * 64, e_sha, m_sha, None),
            (p_sha, c_sha, "0" * 64, m_sha, None),
            (p_sha, c_sha, e_sha, "0" * 64, None),
            (p_sha, c_sha, e_sha, m_sha, _resign_prereg_inner),
            (p_sha, c_sha, e_sha, m_sha, lambda p, e, m: p.__setitem__("epistemic_protocol", "tampered_inner_body")),
            (p_sha, c_sha, e_sha, m_sha, lambda p, e, m: e.__setitem__("cuda_source_sha256", "0" * 64)),
            (p_sha, c_sha, e_sha, m_sha, lambda p, e, m: e.__setitem__("preregistration_file_sha256", "0" * 64)),
            (p_sha, c_sha, e_sha, m_sha, lambda p, e, m: e.__setitem__("preregistration_sha256", "0" * 64)),
            (p_sha, c_sha, e_sha, m_sha, _resign_mlos_inner),
            (p_sha, c_sha, e_sha, m_sha, lambda p, e, m: m.__setitem__("profile_id", "tampered_mlos_inner")),
        )):
            p, e, m = copy.deepcopy(prereg_base), copy.deepcopy(emp_base), copy.deepcopy(mlos_base)
            if mut is not None:
                mut(p, e, m)
            r = _check(p, e, m, ps, cs, es, ms)
            self.assertFalse(r["checks"]["digests_match"], f"digest tamper #{idx}")
            self.assertFalse(r["all_verified"])

        # 2. Causal timestamp order clauses
        for mut in (
            lambda p, e, m: p.__setitem__("created_utc", "2026-11-24T17:00:00Z"),
            lambda p, e, m: m["execution"].__setitem__("ended_utc", "2026-11-24T16:34:00Z"),
        ):
            p, e, m = copy.deepcopy(prereg_base), copy.deepcopy(emp_base), copy.deepcopy(mlos_base)
            mut(p, e, m)
            self.assertFalse(_check(p, e, m)["checks"]["prereg_strictly_precedes_execution"])

        # 3. MLOS Zone-B & zero desktop clauses
        for check_key, mut in (
            ("mlos_clean_zone_b", lambda p, e, m: m.__setitem__("determinism_zone", "A")),
            ("mlos_clean_zone_b", lambda p, e, m: m.__setitem__("clean_hardware_run", False)),
            ("mlos_clean_zone_b", lambda p, e, m: m.__setitem__("anomalies", ["thermal"])),
            ("mlos_clean_zone_b", lambda p, e, m: m["execution"].__setitem__("exit_code", 1)),
            ("mlos_zero_desktop_processes", lambda p, e, m: m["start_gpu_telemetry"].__setitem__("graphics_processes", [{"pid": 1}])),
            ("mlos_zero_desktop_processes", lambda p, e, m: m["end_gpu_telemetry"].__setitem__("graphics_processes", [{"pid": 1}])),
            ("mlos_zero_desktop_processes", lambda p, e, m: m["start_gpu_telemetry"].__setitem__("compute_processes", [{"pid": 1}])),
            ("mlos_zero_desktop_processes", lambda p, e, m: m["end_gpu_telemetry"].__setitem__("compute_processes", [{"pid": 1}])),
            ("real_weight_honesty_tags", lambda p, e, m: e.__setitem__("measured_live_model_weights", False)),
            ("real_weight_honesty_tags", lambda p, e, m: e.__setitem__("oracle_class", "exact")),
            ("real_weight_honesty_tags", lambda p, e, m: e.__setitem__("ac_online", False)),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["HuggingFaceTB/SmolLM2-360M"].__setitem__("revision", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["HuggingFaceTB/SmolLM2-360M"].__setitem__("safetensors_sha256", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-0.5B"].__setitem__("revision", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-0.5B"].__setitem__("safetensors_sha256", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-1.5B"].__setitem__("revision", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-1.5B"].__setitem__("safetensors_sha256", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-3B"].__setitem__("revision", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-3B"].__setitem__("safetensors_shards", [])),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-3B"]["safetensors_shards"][0].__setitem__("sha256", "bad")),
            ("all_4_model_pins_match", lambda p, e, m: e["model_pins"]["Qwen/Qwen2.5-3B"]["safetensors_shards"][1].__setitem__("sha256", "bad")),
            ("corpora_pins_match", lambda p, e, m: e["corpora"].__setitem__("wikitext2_revision", "bad")),
            ("corpora_pins_match", lambda p, e, m: e["corpora"].__setitem__("wikitext2_val_sha256", "bad")),
            ("corpora_pins_match", lambda p, e, m: e["corpora"].__setitem__("wikitext2_test_sha256", "bad")),
            ("corpora_pins_match", lambda p, e, m: e["corpora"].__setitem__("python_stdlib_sha256", "bad")),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"].__setitem__("block_linear_matrices", 251)),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"].__setitem__("total_block_weights", 1)),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"].pop("A16_BlockH8")),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_BlockH4"].__setitem__("syndrome_errors", 1)),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_BlockH4"].__setitem__("non_block_tensors_differ", 1)),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_AdaptZeroCal_1bit"].__setitem__("oracle_agreement_count", 167)),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["Qwen/Qwen2.5-0.5B"]["arms"]["A16_AdaptZeroCal_1bit"].__setitem__("oracle_agreement_count", 132)),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["Qwen/Qwen2.5-1.5B"]["arms"]["A16_AdaptZeroCal_1bit"].__setitem__("oracle_agreement_count", 147)),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_AdaptZeroCal_1bit"].__setitem__("oracle_agreement_count", 189)),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_AdaptZeroCal_1bit"]["wikitext2"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_AdaptZeroCal_1bit"]["python"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["Qwen/Qwen2.5-0.5B"]["arms"]["A16_AdaptZeroCal_2bit"]["python"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_1_agreement_smollm2_win_and_qwen3b_oracle_nuance", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["Oracle_1bit"]["wikitext2"].__setitem__("mean_kl", Fraction(1, 100))),
            ("p_s68_2_calibrated_floors_and_h2_vs_h4_scaling", lambda p, e, m: e["models"]["Qwen/Qwen2.5-0.5B"]["arms"]["hA16_BlockH2_real"]["wikitext2"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_2_calibrated_floors_and_h2_vs_h4_scaling", lambda p, e, m: e["models"]["Qwen/Qwen2.5-0.5B"]["arms"]["hA16_BlockH4_real"]["python"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_2_calibrated_floors_and_h2_vs_h4_scaling", lambda p, e, m: e["models"]["Qwen/Qwen2.5-1.5B"]["arms"]["hA16_BlockH2_real"]["wikitext2"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_2_calibrated_floors_and_h2_vs_h4_scaling", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["hA16_BlockH2_real"]["wikitext2"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_2_calibrated_floors_and_h2_vs_h4_scaling", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["hA16_BlockH4_real"]["python"].__setitem__("mean_kl", Fraction(1, 10))),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["ptxas_verification"].__setitem__("max_stack_bytes", 8)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["ptxas_verification"].__setitem__("max_spill_stores_bytes", 4)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["ptxas_verification"].__setitem__("max_spill_loads_bytes", 4)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["ptxas_verification"].__setitem__("ptx_ldl_count", 1)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["ptxas_verification"].__setitem__("ptx_stl_count", 1)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_single_2048x11008"]["gemv_4_5bpw_a16_block_h4_dp4a"].__setitem__("syndrome_errors", 1)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_single_2048x11008"]["gemv_4_5bpw_a16_block_h4_dp4a"].__setitem__("int64_bit_exact_mismatches", 1)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_stacked2_4096x11008"]["gemv_4_5bpw_a16_block_h4_dp4a"].__setitem__("syndrome_errors", 1)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_stacked2_4096x11008"]["gemv_4_5bpw_a16_block_h4_dp4a"].__setitem__("int64_bit_exact_mismatches", 1)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_stacked2_4096x11008"]["gemv_4_5bpw_a16_block_h4_dp4a"].__setitem__("gweights_per_s", Fraction(568, 1))),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_stacked2_4096x11008"]["gemv_4_5bpw_a16_axis_dp4a"].__setitem__("gweights_per_s", Fraction(700, 1))),
            ("mlos_clean_zone_b", lambda p, e, m: m["execution"].__setitem__("exit_code", False)),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_BlockH4"].__setitem__("syndrome_errors", False)),
            ("all_46_real_arms_clean", lambda p, e, m: e["models"]["Qwen/Qwen2.5-3B"]["arms"]["A16_BlockH4"].__setitem__("non_block_tensors_differ", False)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["ptxas_verification"].__setitem__("max_stack_bytes", False)),
            ("p_s68_3_ptx_zero_spill_bit_exact_and_592gws", lambda p, e, m: e["cuda_sm86_ptx_blockh4_bench"]["shapes"]["down_proj_stacked2_4096x11008"]["gemv_4_5bpw_a16_block_h4_dp4a"].__setitem__("int64_bit_exact_mismatches", False)),
            ("cuda_export_zero_syndrome_errors", lambda p, e, m: e["cuda_export_meta"]["down_proj_single_2048x11008"].__setitem__("export_a16_axis_syndrome_errors", False)),
            ("cuda_export_zero_syndrome_errors", lambda p, e, m: e["cuda_export_meta"]["down_proj_single_2048x11008"].__setitem__("export_a16_axis_syndrome_errors", 1)),
            ("cuda_export_zero_syndrome_errors", lambda p, e, m: e["cuda_export_meta"]["down_proj_single_2048x11008"].__setitem__("export_a16_block_h4_syndrome_errors", 1)),
            ("cuda_export_zero_syndrome_errors", lambda p, e, m: e["cuda_export_meta"]["down_proj_stacked2_4096x11008"].__setitem__("export_a16_axis_syndrome_errors", 1)),
            ("cuda_export_zero_syndrome_errors", lambda p, e, m: e["cuda_export_meta"]["down_proj_stacked2_4096x11008"].__setitem__("export_a16_block_h4_syndrome_errors", 1)),
        ):
            p, e, m = copy.deepcopy(prereg_base), copy.deepcopy(emp_base), copy.deepcopy(mlos_base)
            mut(p, e, m)
            r = _check(p, e, m)
            self.assertFalse(r["checks"][check_key], f"check_key={check_key} did not trip")
            self.assertFalse(r["all_verified"])

        # Special test to kill arms_clean `and total_arms == 46` independently
        p, e, m = copy.deepcopy(prereg_base), copy.deepcopy(emp_base), copy.deepcopy(mlos_base)
        with mock.patch.dict(oracle.MODEL_DIMENSIONS, {"HuggingFaceTB/SmolLM2-360M": (224, 314572800, 8)}):
            e["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"].pop("Oracle_2bit")
            r = _check(p, e, m)
            self.assertFalse(r["checks"]["all_46_real_arms_clean"])


class TestS68AuditHardening(unittest.TestCase):
    def test_disk_certificate_matches_standalone(self) -> None:
        cert_path = STUDY_DIR / "rtx3080_s68_certificate.json"
        st = os.lstat(cert_path)
        self.assertIn(stat.S_IMODE(st.st_mode), (0o600, 0o644))
        raw_disk = cert_path.read_bytes()
        expected = oracle.build_certificate()
        self.assertEqual(raw_disk, oracle.canonical_json_bytes(expected))
        res_f = audit.verify_certificate_file(cert_path, full_replay=True)
        res_n = audit.verify_certificate_file(cert_path, full_replay=False)
        self.assertTrue(res_f["valid"])
        self.assertFalse(res_n["full_replay"])
        self.assertEqual(res_f["certificate_sha256"], expected["certificate_sha256"])

        # Strict bool guard BEFORE disk I/O (kills M060 even on nonexistent path!)
        for bad_flag in (1, 0, "yes", None):
            with self.assertRaises(audit.AuditError) as cm1:
                audit.verify_certificate_file(STUDY_DIR / "nonexistent_flag_check.json", full_replay=bad_flag)  # type: ignore[arg-type]
            self.assertEqual(cm1.exception.code, "INVALID_FLAG_TYPE")
            with self.assertRaises(audit.AuditError) as cm2:
                audit.verify_certificate_payload(expected, full_replay=bad_flag)  # type: ignore[arg-type]
            self.assertEqual(cm2.exception.code, "INVALID_FLAG_TYPE")

    def test_strict_parser_tree_bijection_and_vectors(self) -> None:
        cert = oracle.build_certificate()
        raw = oracle.canonical_json_bytes(cert)
        self.assertEqual(audit.parse_strict_certificate_bytes(raw), cert)

        # Format / UTF-8 / float / constant / dup-key / non-canonical / root type rejections
        for bad_raw, code in (
            (raw[:-1], "NON_CANONICAL_FORMAT"),
            (raw.replace(b"\n", b"\r\n"), "NON_CANONICAL_FORMAT"),
            (b"\xff\xfe\n", "INVALID_UTF8"),
            (b'{"x":1.5}\n', "FLOAT_FORBIDDEN"),
            (b'{"x":NaN}\n', "CONSTANT_FORBIDDEN"),
            (b'{"a":1,"a":2}\n', "DUPLICATE_KEY"),
            (b"{bad\n", "JSON_PARSE_ERROR"),
            (b"[1,2]\n", "INVALID_ROOT"),
            (b'{ "a": 1 }\n', "NON_CANONICAL_BYTES"),
        ):
            with self.assertRaises(audit.AuditError) as cm:
                audit.parse_strict_certificate_bytes(bad_raw)
            self.assertEqual(cm.exception.code, code, f"bad_raw={bad_raw!r}")

        # Exact boundary checks for MAX_JSON_DEPTH (M004) and MAX_INT_BITS (M008)
        exact_depth: Any = 1
        for _ in range(audit.MAX_JSON_DEPTH):
            exact_depth = {"k": exact_depth}
        audit._validate_json_tree_iter(exact_depth)
        audit._validate_json_tree_iter({"a": (1 << audit.MAX_INT_BITS) - 1})

        # Depth, key type, surrogate, big int, forbidden value type
        deep: Any = 1
        for _ in range(audit.MAX_JSON_DEPTH + 1):
            deep = {"k": deep}
        with self.assertRaises(audit.AuditError) as cm:
            audit._validate_json_tree_iter(deep)
        self.assertEqual(cm.exception.code, "DEPTH_EXCEEDED")

        for bad_obj, code in (
            ({1: "v"}, "INVALID_KEY_TYPE"),
            ({_StrSub("a"): "v"}, "INVALID_KEY_TYPE"),
            ({"\ud800": "v"}, "INVALID_UTF8_SURROGATE"),
            ({"a": "\ud800"}, "INVALID_UTF8_SURROGATE"),
            ({"a": 1 << (audit.MAX_INT_BITS + 1)}, "INT_TOO_LARGE"),
            ({"a": None}, "INVALID_VALUE_TYPE"),
            ({"a": _IntSub(1)}, "INVALID_VALUE_TYPE"),
            ({"a": _StrSub("x")}, "INVALID_VALUE_TYPE"),
            ({"a": _DictSub()}, "INVALID_VALUE_TYPE"),
            ({"a": _ListSub()}, "INVALID_VALUE_TYPE"),
        ):
            with self.assertRaises(audit.AuditError) as cm:
                audit._validate_json_tree_iter(bad_obj)
            self.assertEqual(cm.exception.code, code)

        # Secondary audit & bijective tree checks (incl. S64-VER-03 empty list & length mismatch!)
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload([])
        self.assertEqual(cm.exception.code, "INVALID_ROOT")

        for bad_sha in ("deadbeef", "G" * 64, 123, "A" * 64):
            c_bad = dict(cert)
            c_bad["certificate_sha256"] = bad_sha
            with self.assertRaises(audit.AuditError) as cm:
                audit.secondary_audit_payload(c_bad)
            self.assertEqual(cm.exception.code, "INVALID_DIGEST")

        c_no_sha = {k: v for k, v in cert.items() if k != "certificate_sha256"}
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload(c_no_sha)
        self.assertEqual(cm.exception.code, "INVALID_DIGEST")

        c_wrong_sha = dict(cert)
        c_wrong_sha["certificate_sha256"] = ("0" if cert["certificate_sha256"][0] != "0" else "1") + cert["certificate_sha256"][1:]
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload(c_wrong_sha)
        self.assertEqual(cm.exception.code, "DIGEST_MISMATCH")

        # Re-signed structural tampers (type mismatch, key mismatch, empty list, leaf mismatch)
        c_empty_list = copy.deepcopy(cert)
        c_empty_list["pillar_1_exact_router_and_h4_arithmetic"]["bpw_exact"] = []
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload(_resign(c_empty_list))
        self.assertEqual(cm.exception.code, "SCHEMA_LEN_MISMATCH")

        c_type = copy.deepcopy(cert)
        c_type["pillar_1_exact_router_and_h4_arithmetic"]["models_audited"] = True
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload(_resign(c_type))
        self.assertEqual(cm.exception.code, "SCHEMA_TYPE_MISMATCH")

        c_key = copy.deepcopy(cert)
        c_key["pillar_1_exact_router_and_h4_arithmetic"]["extra_key"] = 1
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload(_resign(c_key))
        self.assertEqual(cm.exception.code, "SCHEMA_KEY_MISMATCH")

        c_leaf = copy.deepcopy(cert)
        c_leaf["pillar_1_exact_router_and_h4_arithmetic"]["models_audited"] = 99
        with self.assertRaises(audit.AuditError) as cm:
            audit.secondary_audit_payload(_resign(c_leaf))
        self.assertEqual(cm.exception.code, "INVARIANT_MISMATCH")

        # Full replay divergence check
        with mock.patch.object(oracle, "build_certificate", return_value={"different": True}):
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_certificate_payload(cert, full_replay=True)
            self.assertEqual(cm.exception.code, "REPLAY_MISMATCH")

    def test_zone_b_io_pwd_darwin_atomic_write_and_cli(self) -> None:
        cert = oracle.build_certificate()
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)

            # Direct API surrogate / NUL / non-Path / empty / basename checks (S67-VER-03!)
            for bad_p in (Path("/tmp/\ud800.json"), Path("/tmp/a\x00b.json"), "not_a_path"):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.verify_certificate_file(bad_p)  # type: ignore[arg-type]
                self.assertEqual(cm.exception.code, "INVALID_PATH")
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_certificate_atomic(bad_p, cert)  # type: ignore[arg-type]
                self.assertEqual(cm.exception.code, "INVALID_PATH")
                with self.assertRaises(audit.AuditError) as cm:
                    audit.verify_receipts_directory(bad_p)  # type: ignore[arg-type]
                self.assertEqual(cm.exception.code, "INVALID_PATH")

            with self.assertRaises(audit.AuditError) as cm:
                audit._open_parent_dir_nofollow(Path(""))
            self.assertEqual(cm.exception.code, "EMPTY_PATH")
            for bad_bn in (Path("/"), Path(".."), Path("a/..")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit._open_parent_dir_nofollow(bad_bn)
                self.assertEqual(cm.exception.code, "INVALID_BASENAME")

            # Atomic write & overwrite + mode 0o600
            out_p = td_p / "cert.json"
            sha1 = audit.write_certificate_atomic(out_p, cert)
            self.assertEqual(stat.S_IMODE(os.lstat(out_p).st_mode), 0o600)
            self.assertEqual(sha1, audit.write_certificate_atomic(out_p, cert))
            with self.assertRaises(audit.AuditError) as cm:
                audit.write_certificate_atomic(out_p, [])  # type: ignore[arg-type]
            self.assertEqual(cm.exception.code, "INVALID_ROOT")

            # Symlink target & symlink ancestor & hardlink & FIFO (S62B-VER-01!) & size cap
            sym_t = td_p / "sym.json"
            os.symlink(out_p, sym_t)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_file_bytes(sym_t)
            self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as cm:
                audit.write_certificate_atomic(sym_t, cert)
            self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            sub_d = td_p / "real_dir"
            sub_d.mkdir()
            sym_d = td_p / "sym_dir"
            os.symlink(sub_d, sym_d)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_file_bytes(sym_d / "x.json")
            self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            # Simulate both ELOOP (M023) and ENOTDIR + lstat symlink branches
            real_open = os.open
            def _open_eloop(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                if path == "sym_dir":
                    raise OSError(errno.ELOOP, "Too many levels of symbolic links")
                return real_open(path, flags, *args, **kwargs)
            with mock.patch("os.open", side_effect=_open_eloop):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(sym_d / "x.json")
                self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            def _open_enotdir(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                if path == "sym_dir":
                    raise OSError(errno.ENOTDIR, "Not a directory")
                return real_open(path, flags, *args, **kwargs)
            with mock.patch("os.open", side_effect=_open_enotdir):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(sym_d / "x.json")
                self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")

            # Non-symlink file used as directory component -> DIR_OPEN_FAILED
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_file_bytes(out_p / "child.json")
            self.assertEqual(cm.exception.code, "DIR_OPEN_FAILED")

            # Anchor open failure
            def _open_anchor_fail(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                raise OSError(errno.EACCES, "denied")
            with mock.patch("os.open", side_effect=_open_anchor_fail):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(out_p)
                self.assertEqual(cm.exception.code, "DIR_OPEN_FAILED")

            # Hardlink rejection on read & write
            hl = td_p / "hl.json"
            os.link(out_p, hl)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_file_bytes(hl)
            self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")
            with self.assertRaises(audit.AuditError) as cm:
                audit.write_certificate_atomic(hl, cert)
            self.assertEqual(cm.exception.code, "HARDLINK_FORBIDDEN")
            os.unlink(hl)

            # FIFO O_NONBLOCK read & write rejection (S62B-VER-01)
            fifo_p = td_p / "pipe.fifo"
            os.mkfifo(fifo_p)
            with self.assertRaises(audit.AuditError) as cm:
                audit.read_regular_file_bytes(fifo_p)
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(audit.AuditError) as cm:
                audit.write_certificate_atomic(fifo_p, cert)
            self.assertEqual(cm.exception.code, "NON_REGULAR_FILE")

            # Exact boundary pass (kills M035 & M038) + st_size fast-reject before os.read (kills M036)
            exact_sz = out_p.stat().st_size
            self.assertEqual(len(audit.read_regular_file_bytes(out_p, max_bytes=exact_sz)), exact_sz)
            with mock.patch("os.read", side_effect=RuntimeError("must reject on st_size before os.read")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(out_p, max_bytes=16)
                self.assertEqual(cm.exception.code, "FILE_TOO_LARGE")
            real_fstat = os.fstat
            def _fake_small_stat(fd: int) -> Any:
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid, 4, st.st_atime, st.st_mtime, st.st_ctime))
            with mock.patch("os.fstat", side_effect=_fake_small_stat):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(out_p, max_bytes=16)
                self.assertEqual(cm.exception.code, "FILE_TOO_LARGE")

            # EIO on os.fstat & os.read (S67-VER-02!)
            with mock.patch("os.fstat", side_effect=OSError(errno.EIO, "fstat io error")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(out_p)
                self.assertEqual(cm.exception.code, "FILE_READ_FAILED")
            with mock.patch("os.read", side_effect=OSError(errno.EIO, "read io error")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.read_regular_file_bytes(out_p)
                self.assertEqual(cm.exception.code, "FILE_READ_FAILED")

            # Atomic write fault injection: EEXIST exhaustion (kills M045), short write, tampered tmp, ENOSPC (S66-VER-02!)
            def _open_eexist(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                if flags & os.O_CREAT:
                    raise FileExistsError()
                return real_open(path, flags, *args, **kwargs)
            with mock.patch("os.open", side_effect=_open_eexist), mock.patch("os.fchmod", side_effect=RuntimeError("must fail before fchmod")):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_certificate_atomic(td_p / "w1.json", cert)
                self.assertEqual(cm.exception.code, "ATOMIC_WRITE_FAILED")
                self.assertIn("exhausted", cm.exception.message)

            with mock.patch("os.write", return_value=0):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_certificate_atomic(td_p / "w2.json", cert)
                self.assertEqual(cm.exception.code, "SHORT_WRITE")

            def _fake_nlink2(fd: int) -> Any:
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, 2, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
            with mock.patch("os.fstat", side_effect=_fake_nlink2):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_certificate_atomic(td_p / "w3.json", cert)
                self.assertEqual(cm.exception.code, "TEMP_TAMPERED")

            def _fake_tmp_fifo(fd: int) -> Any:
                st = real_fstat(fd)
                return os.stat_result((stat.S_IFIFO | 0o600, st.st_ino, st.st_dev, 1, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
            with mock.patch("os.fstat", side_effect=_fake_tmp_fifo):
                with self.assertRaises(audit.AuditError) as cm:
                    audit.write_certificate_atomic(td_p / "w3b.json", cert)
                self.assertEqual(cm.exception.code, "TEMP_TAMPERED")

            for fn_name in ("os.fchmod", "os.write", "os.fsync", "os.replace"):
                with mock.patch(fn_name, side_effect=OSError(errno.ENOSPC, "no space")):
                    with self.assertRaises(audit.AuditError) as cm:
                        audit.write_certificate_atomic(td_p / "w4.json", cert)
                    self.assertEqual(cm.exception.code, "ATOMIC_WRITE_FAILED")
            self.assertEqual([p.name for p in td_p.glob(".*.tmp.*")], [])

            # Relative path + logical $PWD symlink ancestor check + non-matching $PWD
            orig_cwd = os.getcwd()
            try:
                os.chdir(sub_d)
                (sub_d / "local.json").write_bytes(oracle.canonical_json_bytes(cert))
                with mock.patch.dict(os.environ, {"PWD": str(sym_d)}):
                    with self.assertRaises(audit.AuditError) as cm:
                        audit.read_regular_file_bytes(Path("local.json"))
                    self.assertEqual(cm.exception.code, "SYMLINK_FORBIDDEN")
                with mock.patch.dict(os.environ, {"PWD": str(sub_d)}):
                    self.assertEqual(audit.read_regular_file_bytes(Path("./local.json")), oracle.canonical_json_bytes(cert))
                with mock.patch.dict(os.environ, {"PWD": "/nonexistent_pwd_12345"}):
                    self.assertEqual(audit.read_regular_file_bytes(Path("local.json")), oracle.canonical_json_bytes(cert))
            finally:
                os.chdir(orig_cwd)

            # Darwin prefix normalization helper tests (all lstat/readlink branches mocked!)
            with mock.patch.object(sys, "platform", "darwin"):
                root_lnk = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 0, 0, 11, 0, 0, 0))
                nonroot_lnk = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 1000, 1000, 11, 0, 0, 0))
                root_dir = os.stat_result((stat.S_IFDIR | 0o755, 1, 1, 2, 0, 0, 0, 0, 0, 0))
                with mock.patch("os.lstat", return_value=root_lnk), mock.patch("os.readlink", return_value="private/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/",)), ("/",))
                    self.assertEqual(audit._normalize_darwin_system_prefix(("rel", "var")), ("rel", "var"))
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "..", "x")), ("/", "var", "..", "x"))
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "usr", "local")), ("/", "usr", "local"))
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "folders")), ("/", "private", "var", "folders"))
                with mock.patch("os.lstat", return_value=nonroot_lnk), mock.patch("os.readlink", return_value="private/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))
                with mock.patch("os.lstat", return_value=root_dir), mock.patch("os.readlink", return_value="private/var"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))
                with mock.patch("os.lstat", return_value=root_lnk), mock.patch("os.readlink", return_value="/evil"):
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))
                with mock.patch("os.lstat", side_effect=OSError(errno.ENOENT, "missing")):
                    self.assertEqual(audit._normalize_darwin_system_prefix(("/", "var", "x")), ("/", "var", "x"))

            # Receipt directory error branches (incl. S67-VER-01 ZeroDivisionError!)
            rdir = td_p / "rec_test"
            rdir.mkdir()
            for fn in ("s68_preregistration.json", "sm86_a16_blockh4_kernel.cu", "rtx3080_s68_empirical_results.json", "mlos_zone_b_receipt.json"):
                (rdir / fn).write_bytes((STUDY_DIR / fn).read_bytes())

            # Bad JSON / float in prereg / non-dict root / ZeroDivisionError ("1/0") / failed check
            (rdir / "s68_preregistration.json").write_bytes(b'{"a":1.5}\n')
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "FLOAT_FORBIDDEN")

            (rdir / "s68_preregistration.json").write_bytes(b"{bad")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_PARSE_ERROR")

            (rdir / "s68_preregistration.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_ROOT_INVALID")

            (rdir / "s68_preregistration.json").write_bytes((STUDY_DIR / "s68_preregistration.json").read_bytes())
            (rdir / "rtx3080_s68_empirical_results.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_ROOT_INVALID")

            (rdir / "rtx3080_s68_empirical_results.json").write_bytes((STUDY_DIR / "rtx3080_s68_empirical_results.json").read_bytes())
            (rdir / "mlos_zone_b_receipt.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_ROOT_INVALID")

            # S67-VER-01 ZeroDivisionError via "1/0" token in empirical JSON (API & CLI!)
            (rdir / "mlos_zone_b_receipt.json").write_bytes((STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes())
            emp_zdiv = json.loads((STUDY_DIR / "rtx3080_s68_empirical_results.json").read_text(encoding="utf-8"))
            emp_zdiv["models"]["HuggingFaceTB/SmolLM2-360M"]["arms"]["A16_AdaptZeroCal_1bit"]["wikitext2"]["mean_kl"] = "1/0"
            (rdir / "rtx3080_s68_empirical_results.json").write_text(json.dumps(emp_zdiv), encoding="utf-8")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_SCHEMA_ERROR")

            err_buf = io.BytesIO()
            with mock.patch.object(sys, "stderr", mock.Mock(buffer=err_buf)):
                self.assertEqual(audit.main(["--verify-receipts", str(rdir)]), 2)
            self.assertEqual(json.loads(err_buf.getvalue().decode("utf-8"))["error_code"], "RECEIPT_SCHEMA_ERROR")

            # Valid schema but digest mismatch -> RECEIPT_VERIFICATION_FAILED
            emp_bad = json.loads((STUDY_DIR / "rtx3080_s68_empirical_results.json").read_text(encoding="utf-8"))
            emp_bad["ac_online"] = False
            (rdir / "rtx3080_s68_empirical_results.json").write_text(json.dumps(emp_bad), encoding="utf-8")
            with self.assertRaises(audit.AuditError) as cm:
                audit.verify_receipts_directory(rdir)
            self.assertEqual(cm.exception.code, "RECEIPT_VERIFICATION_FAILED")

            # CLI tests (all modes + non-list/non-str/subclass argv + bad paths)
            def _run_cli(args: Any) -> tuple[int, bytes, bytes]:
                ob, eb = io.BytesIO(), io.BytesIO()
                with mock.patch.object(sys, "stdout", mock.Mock(buffer=ob)), mock.patch.object(sys, "stderr", mock.Mock(buffer=eb)):
                    rc = audit.main(args)
                return rc, ob.getvalue(), eb.getvalue()

            rc, ob, _ = _run_cli([])
            self.assertEqual(rc, 0)
            self.assertEqual(ob, oracle.canonical_json_bytes(cert))
            rc, ob, _ = _run_cli(["--standalone", "--json"])
            self.assertEqual(rc, 0)
            self.assertEqual(ob, oracle.canonical_json_bytes(cert))
            cli_out = td_p / "cli_cert.json"
            rc, ob, _ = _run_cli(["--write", str(cli_out)])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(ob.decode("utf-8"))["written"])
            rc, ob, _ = _run_cli(["--check", str(cli_out)])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(ob.decode("utf-8"))["valid"])
            rc, ob, _ = _run_cli(["--verify-receipts", str(STUDY_DIR)])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(ob.decode("utf-8"))["all_verified"])

            with mock.patch.object(sys, "argv", ["audit.py", "--standalone", "--json"]):
                rc, ob, _ = _run_cli(None)
                self.assertEqual(rc, 0)

            for bad_argv, exp_code in (
                (123, "USAGE_ERROR"),
                (["--check", 123], "USAGE_ERROR"),
                (["--check", _StrSub(str(cli_out))], "USAGE_ERROR"),
                (["--bogus"], "USAGE_ERROR"),
                (["--bogus", str(STUDY_DIR)], "USAGE_ERROR"),
                (["--check", str(cli_out), "extra"], "USAGE_ERROR"),
                (["--write", str(cli_out), "extra"], "USAGE_ERROR"),
                (["--verify-receipts", str(STUDY_DIR), "extra"], "USAGE_ERROR"),
                (["--check", ""], "INVALID_CLI_PATH"),
                (["--check", "-flag"], "INVALID_CLI_PATH"),
                (["--check", "a\x00b"], "INVALID_PATH"),
                (["--check", "/tmp/\ud800.json"], "INVALID_PATH"),
            ):
                rc, _, eb = _run_cli(bad_argv)
                self.assertEqual(rc, 2, f"bad_argv={bad_argv!r}")
                parsed_err = json.loads(eb.decode("utf-8"))
                self.assertFalse(parsed_err["valid"])
                self.assertEqual(parsed_err["error_code"], exp_code, f"bad_argv={bad_argv!r}")


if __name__ == "__main__":
    unittest.main()
