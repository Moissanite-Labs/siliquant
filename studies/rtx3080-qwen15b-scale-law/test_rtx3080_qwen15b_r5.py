"""R5 & Zone-B Verification and Full-Mutation-Killing Suite for Track S60.

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

import rtx3080_qwen15b_audit as audit
import rtx3080_qwen15b_oracle as oracle


def _resign(cert: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    out = copy.deepcopy(body)
    out["certificate_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(body)).hexdigest()
    return out


class _IntSub(int):
    pass


class _StrSub(str):
    pass


class TestS60OracleAndReceipts(unittest.TestCase):
    def test_helpers_exact_verdict_and_gates(self) -> None:
        self.assertEqual(oracle.frac_pair(Fraction(9, 2)), [9, 2])
        with self.assertRaises(TypeError):
            oracle.frac_pair(9)  # type: ignore[arg-type]
        self.assertEqual(oracle._frac_tok("12.617995"), Fraction(12617995, 1000000))
        with self.assertRaises(ValueError):
            oracle._frac_tok("")
        with self.assertRaises(ValueError):
            oracle._frac_tok(123)  # type: ignore[arg-type]

        cert = oracle.build_certificate()
        self.assertEqual(cert["schema_version"], oracle.SCHEMA_VERSION)
        self.assertEqual(cert["oracle_module"], oracle.ORACLE_MODULE_NAME)
        p1 = cert["pillar_1_exact_model_and_wire"]
        self.assertEqual(p1["qwen15b_layers"], 28)
        self.assertEqual(p1["qwen15b_block_matrices"], 196)
        self.assertEqual(p1["qwen15b_weights_per_layer"], 46792704)
        self.assertEqual(p1["qwen15b_total_block_weights"], 1310195712)
        self.assertIs(p1["qwen15b_exact_128_tiling"], True)
        self.assertEqual(p1["down1_weights"], 13762560)
        self.assertEqual(p1["down1_4_5bpw_bytes"], 7741440)
        self.assertEqual(p1["down4_weights"], 55050240)
        self.assertEqual(p1["down4_4_5bpw_bytes"], 30965760)
        self.assertEqual(p1["hampack_18b_max_coord"], 34)
        self.assertEqual(p1["hammingindex_72b_max_coord"], 72)
        self.assertIs(p1["both_balls_fit_lossless_int8"], True)
        self.assertIs(p1["ball_det_scale_128_valid"], True)
        self.assertIs(p1["ball_det_scale_32_valid"], True)

        p2 = cert["pillar_2_sampled_live_qwen15b"]
        self.assertEqual(p2["oracle_class"], "sampled")
        self.assertIs(p2["measured_live_model_weights"], True)
        self.assertTrue(all(v is True for v in p2["sampled_gates"].values()))
        self.assertEqual(
            cert["verdict"],
            {"exact_invariants_pass": True, "sampled_gates_pass": True, "all_passed": True},
        )

        # Boundary negations for exact tiling and int8 fit
        self.assertIs(oracle._build_exact_model_and_format_invariants(q15_dims=(28, 1537, 256, 8960))["qwen15b_exact_128_tiling"], False)
        self.assertIs(oracle._build_exact_model_and_format_invariants(q15_dims=(28, 1536, 257, 8960))["qwen15b_exact_128_tiling"], False)
        self.assertIs(oracle._build_exact_model_and_format_invariants(q15_dims=(28, 1536, 256, 8961))["qwen15b_exact_128_tiling"], False)
        self.assertIs(oracle._build_exact_model_and_format_invariants(bound_18b=128 * 128)["both_balls_fit_lossless_int8"], False)
        self.assertIs(oracle._build_exact_model_and_format_invariants(bound_72b=128 * 128)["both_balls_fit_lossless_int8"], False)

        # Flip every clause of evaluate_exact_verdict
        for k, bad in (
            ("qwen15b_layers", 0),
            ("qwen15b_block_matrices", 0),
            ("qwen15b_weights_per_layer", 0),
            ("qwen15b_total_block_weights", 0),
            ("qwen15b_exact_128_tiling", False),
            ("down1_weights", 0),
            ("down1_4_5bpw_bytes", 0),
            ("down4_weights", 0),
            ("down4_4_5bpw_bytes", 0),
            ("both_balls_fit_lossless_int8", False),
            ("ball_det_scale_128_valid", False),
            ("ball_det_scale_32_valid", False),
        ):
            p1_b = copy.deepcopy(p1)
            p1_b[k] = bad
            self.assertIs(oracle.evaluate_exact_verdict(p1_b), False)

        # Flip every BoolOp clause in build_certificate
        with mock.patch.object(oracle, "evaluate_exact_verdict", return_value=False):
            c_e = oracle.build_certificate()
            self.assertIs(c_e["verdict"]["all_passed"], False)
        p2_bad = copy.deepcopy(p2)
        p2_bad["sampled_gates"]["cal_to_eval_regime_accuracy_above_70pct"] = False
        with mock.patch.object(oracle, "_build_qwen15b_sampled_ratios", return_value=p2_bad):
            c_s = oracle.build_certificate()
            self.assertIs(c_s["verdict"]["all_passed"], False)

        # Flip every single subcondition of every sampled gate
        gate_flips: tuple[tuple[tuple[str, ...], Any, str], ...] = (
            (("wt_kl", "A16_H16"), "0.09", "scale_crossover_a16_h16_beats_a16_axis_both_texts_and_matched"),
            (("wt_kl", "A16_H16_matched"), "0.09", "scale_crossover_a16_h16_beats_a16_axis_both_texts_and_matched"),
            (("py_kl", "A16_H16"), "0.09", "scale_crossover_a16_h16_beats_a16_axis_both_texts_and_matched"),
            (("py_kl", "A16_H16_matched"), "0.09", "scale_crossover_a16_h16_beats_a16_axis_both_texts_and_matched"),
            (("wt_kl", "A16_H16"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("wt_kl", "A16_axis"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("wt_kl", "E8_H8"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("wt_kl", "E8_axis"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("wt_kl", "Z_plain"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("wt_kl", "A16_BlockH2_matched"), "0.09", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("py_kl", "A16_H16"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("py_kl", "A16_axis"), "0.01", "a16_block_h2_best_uniform_uncal_both_texts_and_matched"),
            (("wt_kl", "E8_axis"), "0.09", "e8_hadamard_domain_split_at_1_5b"),
            (("py_kl", "E8_H8"), "0.09", "e8_hadamard_domain_split_at_1_5b"),
            (("anisotropy_counts", "block_h2_beats_h16"), 98, "block_h2_beats_h16_majority_of_196_layers"),
            (("anisotropy_counts", "regime1_axis_beats_h16"), 100, "block_h2_beats_h16_majority_of_196_layers"),
            (("wt_kl", "A16_hybrid_cal_BlockH2"), "0.09", "leakage_free_hybrid_beats_axis_and_h16_wins_python_while_bh2_keeps_wiki"),
            (("wt_kl", "A16_hybrid_cal_H16"), "0.09", "leakage_free_hybrid_beats_axis_and_h16_wins_python_while_bh2_keeps_wiki"),
            (("wt_kl", "A16_hybrid_cal_best3"), "0.09", "leakage_free_hybrid_beats_axis_and_h16_wins_python_while_bh2_keeps_wiki"),
            (("py_kl", "A16_BlockH2"), "0.01", "leakage_free_hybrid_beats_axis_and_h16_wins_python_while_bh2_keeps_wiki"),
            (("py_kl", "A16_H16"), "0.01", "leakage_free_hybrid_beats_axis_and_h16_wins_python_while_bh2_keeps_wiki"),
            (("wt_kl", "A16_BlockH2"), "0.049", "leakage_free_hybrid_beats_axis_and_h16_wins_python_while_bh2_keeps_wiki"),
            (("anisotropy_counts", "cal_to_eval_regime_agree"), 137, "cal_to_eval_regime_accuracy_above_70pct"),
            (("anchor_s49b_qwen05b_wt_p5_retention",), "0.90", "p5_wikitext_decay_emerges_0_5b_to_1_5b_while_python_stays_high"),
            (("wt_p5_rand_retention",), "0.96", "p5_wikitext_decay_emerges_0_5b_to_1_5b_while_python_stays_high"),
            (("wt_p5_rand_retention",), "0.84", "p5_wikitext_decay_emerges_0_5b_to_1_5b_while_python_stays_high"),
            (("py_p5_rand_retention",), "0.94", "p5_wikitext_decay_emerges_0_5b_to_1_5b_while_python_stays_high"),
            (("wt_kl", "hA16_real"), "0.04", "real_fisher_cuts_kl_over_50pct_and_beats_colproxy_both_texts"),
            (("py_kl", "hA16_real"), "0.025", "real_fisher_cuts_kl_over_50pct_and_beats_colproxy_both_texts"),
            (("wt_colproxy_retention",), "0.36", "real_fisher_cuts_kl_over_50pct_and_beats_colproxy_both_texts"),
            (("py_colproxy_retention",), "0.36", "real_fisher_cuts_kl_over_50pct_and_beats_colproxy_both_texts"),
            (("spearman_real",), "0.40", "real_fisher_cuts_kl_over_50pct_and_beats_colproxy_both_texts"),
            (("spearman_colproxy",), "0.60", "real_fisher_cuts_kl_over_50pct_and_beats_colproxy_both_texts"),
            (("wt_kl", "A16_axis_K10"), "0.09", "k10_mild_improvement_both_texts"),
            (("py_kl", "A16_axis_K10"), "0.09", "k10_mild_improvement_both_texts"),
            (("cuda_stacked4_6144x8960", "gemv_tc_18b_pct_triad"), "99", "stacked4_lossless_transcoded_int8_exceeds_100pct_triad"),
            (("cuda_stacked4_6144x8960", "gemv_tc_72b_pct_triad"), "99", "stacked4_lossless_transcoded_int8_exceeds_100pct_triad"),
            (("cuda_stacked4_6144x8960", "gemv_tc_72b_gweights_per_s"), "300", "stacked4_lossless_transcoded_int8_exceeds_100pct_triad"),
            (("cuda_stacked4_6144x8960", "gemv_direct_a16_pct_triad"), "96", "stacked4_direct_4_5bpw_high_triad_and_beats_single_grid"),
            (("cuda_stacked4_6144x8960", "gemv_direct_bh2_pct_triad"), "93", "stacked4_direct_4_5bpw_high_triad_and_beats_single_grid"),
            (("cuda_single_1536x8960", "gemv_direct_a16_gweights_per_s"), "700", "stacked4_direct_4_5bpw_high_triad_and_beats_single_grid"),
        )
        for path_t, bad_v, gname in gate_flips:
            toks = copy.deepcopy(oracle.FROZEN_QWEN15B_TOKENS)
            cur: Any = toks
            for k in path_t[:-1]:
                cur = cur[k]
            cur[path_t[-1]] = bad_v
            out = oracle._build_qwen15b_sampled_ratios(toks)
            self.assertIs(out["sampled_gates"][gname], False, f"flip {path_t} did not fail {gname}")

    def test_verify_live_receipts_pass_and_all_tampers(self) -> None:
        emp_raw = (STUDY_DIR / "rtx3080_qwen15b_empirical_results.json").read_bytes()
        mlos_raw = (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()
        cu_raw = (STUDY_DIR / "rtx3080_qwen15b_kernels.cu").read_bytes()
        emp_base = json.loads(emp_raw.decode("utf-8"), parse_float=Fraction)
        mlos_base = json.loads(mlos_raw.decode("utf-8"), parse_float=Fraction)
        e_sha = hashlib.sha256(emp_raw).hexdigest()
        m_sha = hashlib.sha256(mlos_raw).hexdigest()
        c_sha = hashlib.sha256(cu_raw).hexdigest()

        res = oracle.verify_live_receipts(emp_base, mlos_base, e_sha, m_sha, c_sha)
        self.assertIs(res["all_verified"], True)
        self.assertEqual(len(res["checks"]), 12)

        def assert_fails(
            emp_m: dict[str, Any] | None = None,
            mlos_m: dict[str, Any] | None = None,
            es: str = e_sha,
            ms: str = m_sha,
            cs: str = c_sha,
            ck: str = "",
        ) -> None:
            r = oracle.verify_live_receipts(
                emp_base if emp_m is None else emp_m,
                mlos_base if mlos_m is None else mlos_m,
                es,
                ms,
                cs,
            )
            self.assertIs(r["all_verified"], False, f"expected fail for {ck}")
            if ck:
                self.assertIs(r["checks"][ck], False, f"expected {ck} False")

        # digests_match
        assert_fails(cs="0" * 64, ck="digests_match")
        e = copy.deepcopy(emp_base)
        e["cuda_source_sha256"] = "0" * 64
        assert_fails(emp_m=e, ck="digests_match")
        assert_fails(es="0" * 64, ck="digests_match")
        assert_fails(ms="0" * 64, ck="digests_match")
        m = copy.deepcopy(mlos_base)
        m["receipt_sha256"] = "0" * 64
        assert_fails(mlos_m=m, ck="digests_match")

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
            oracle.verify_live_receipts(emp_base, m_b4, e_sha, m_sha, c_sha)["checks"]["mlos_zero_desktop_processes"],
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

        # real_weight_honesty_tags, corpora_pins_match, model_pins_match
        for fld, bv in (("oracle_class", "exact"), ("measured_live_model_weights", False), ("ac_online", False)):
            e = copy.deepcopy(emp_base)
            e[fld] = bv
            assert_fails(emp_m=e, ck="real_weight_honesty_tags")

        for cf in ("wikitext2_revision", "wikitext2_val_sha256", "wikitext2_test_sha256", "python_stdlib_sha256"):
            e = copy.deepcopy(emp_base)
            e["corpora"][cf] = "0" * 40
            assert_fails(emp_m=e, ck="corpora_pins_match")

        for mf, bv in (
            ("repo_id", "other"),
            ("revision", "0" * 40),
            ("model_safetensors_sha256", "0" * 64),
            ("tokenizer_json_sha256", "0" * 64),
            ("block_linear_matrices", 168),
            ("total_block_weights", 1),
        ):
            e = copy.deepcopy(emp_base)
            e["model"][mf] = bv
            assert_fails(emp_m=e, ck="model_pins_match")

        # all_16_real_arms_clean
        e = copy.deepcopy(emp_base)
        del e["model"]["arms"]["Z_plain"]
        assert_fails(emp_m=e, ck="all_16_real_arms_clean")
        for af, bv in (
            ("syndrome_errors", 1),
            ("non_block_tensors_differ", 1),
            ("max_matched_mse_db_diff", Fraction(1, 100000)),
        ):
            e = copy.deepcopy(emp_base)
            e["model"]["arms"]["A16_axis"][af] = bv
            assert_fails(emp_m=e, ck="all_16_real_arms_clean")

        # scale_crossover_block_h2_winner
        for corpus, lo, hi in (
            ("wikitext2", "A16_BlockH2", "A16_H16"),
            ("wikitext2", "A16_H16", "A16_axis"),
            ("wikitext2", "A16_BlockH2_matched", "A16_H16_matched"),
            ("wikitext2", "A16_H16_matched", "A16_axis"),
            ("python", "A16_BlockH2", "A16_axis"),
        ):
            e = copy.deepcopy(emp_base)
            e["model"]["arms"][lo][corpus]["mean_kl"] = e["model"]["arms"][hi][corpus]["mean_kl"]
            assert_fails(emp_m=e, ck="scale_crossover_block_h2_winner")

        # p5_decay_and_fisher_dominance
        e = copy.deepcopy(emp_base)
        e["model"]["arms"]["hA16_real"]["wikitext2"]["mean_kl"] = e["model"]["arms"]["hA16_rand"]["wikitext2"]["mean_kl"]
        assert_fails(emp_m=e, ck="p5_decay_and_fisher_dominance")
        e = copy.deepcopy(emp_base)
        e["model"]["arms"]["hA16_rand"]["wikitext2"]["mean_kl"] = e["model"]["arms"]["hA16_colproxy"]["wikitext2"]["mean_kl"]
        assert_fails(emp_m=e, ck="p5_decay_and_fisher_dominance")
        e = copy.deepcopy(emp_base)
        e["model"]["calibration_summary"]["wikitext2_p5_rand_retention"] = Fraction(95, 100)
        assert_fails(emp_m=e, ck="p5_decay_and_fisher_dominance")
        e = copy.deepcopy(emp_base)
        e["model"]["calibration_summary"]["wikitext2_colproxy_retention"] = Fraction(35, 100)
        assert_fails(emp_m=e, ck="p5_decay_and_fisher_dominance")

        # hybrid_routing_and_196_layer_census
        e = copy.deepcopy(emp_base)
        e["model"]["per_layer_anisotropy_diagnostic"]["total_layers_scored"] = 195
        assert_fails(emp_m=e, ck="hybrid_routing_and_196_layer_census")
        e = copy.deepcopy(emp_base)
        e["model"]["per_layer_anisotropy_diagnostic"]["regime1_axis_beats_h16_count"] = 100
        assert_fails(emp_m=e, ck="hybrid_routing_and_196_layer_census")
        e = copy.deepcopy(emp_base)
        e["model"]["hybrid_selector_routing"]["cal_predicted_regime1_axis"] = 93
        assert_fails(emp_m=e, ck="hybrid_routing_and_196_layer_census")
        e = copy.deepcopy(emp_base)
        e["model"]["hybrid_selector_routing"]["cal_to_eval_regime_agree_count"] = 137
        assert_fails(emp_m=e, ck="hybrid_routing_and_196_layer_census")
        e = copy.deepcopy(emp_base)
        e["model"]["arms"]["A16_hybrid_cal_H16"]["python"]["mean_kl"] = e["model"]["arms"]["A16_BlockH2"]["python"]["mean_kl"]
        assert_fails(emp_m=e, ck="hybrid_routing_and_196_layer_census")

        # cuda_10_kernels_zero_spill & cuda_both_shapes_exact
        for pf, bv in (
            ("kernels_compiled", 9),
            ("max_stack_bytes", 4),
            ("max_spill_loads_bytes", 4),
            ("max_spill_stores_bytes", 4),
        ):
            e = copy.deepcopy(emp_base)
            e["cuda_real_weight_transcode_and_serving"]["ptxas_verification"][pf] = bv
            assert_fails(emp_m=e, ck="cuda_10_kernels_zero_spill")

        for sname in ("down_proj_single_1536x8960", "down_proj_stacked4_6144x8960"):
            e = copy.deepcopy(emp_base)
            e["cuda_real_weight_transcode_and_serving"]["shapes"][sname]["weights"] = 1
            assert_fails(emp_m=e, ck="cuda_both_shapes_exact")

        for ck_pair in (
            ("transcode_18B_to_int8_soa", "bit_mismatches", 1),
            ("transcode_18B_to_int8_soa", "syndrome_errors", 1),
            ("transcode_72B_to_int8_soa", "bit_mismatches", 1),
            ("transcode_72B_to_int8_soa", "syndrome_errors", 1),
            ("gemv_transcoded_18B_lossless_int8_dp4a", "syndrome_errors", 1),
            ("gemv_transcoded_18B_lossless_int8_dp4a", "rel_l2_err_fp64", Fraction(1, 1000000)),
            ("gemv_transcoded_72B_lossless_int8_dp4a", "syndrome_errors", 1),
            ("gemv_transcoded_72B_lossless_int8_dp4a", "rel_l2_err_fp64", Fraction(1, 1000000)),
            ("gemv_direct_4_5bpw_a16_preencoded_dp4a", "syndrome_errors", 1),
            ("gemv_direct_4_5bpw_a16_preencoded_dp4a", "rel_l2_err_fp64", Fraction(1, 1000000)),
            ("gemv_direct_4_5bpw_block_h2_dp4a", "rel_l2_err_fp64", Fraction(1, 1000000)),
        ):
            e = copy.deepcopy(emp_base)
            e["cuda_real_weight_transcode_and_serving"]["shapes"]["down_proj_stacked4_6144x8960"][ck_pair[0]][ck_pair[1]] = ck_pair[2]
            assert_fails(emp_m=e, ck="cuda_both_shapes_exact")


class TestS60AuditHardening(unittest.TestCase):
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

        # Exact boundaries on _validate_json_tree_iter including UTF-16 surrogates
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
        c["pillar_1_exact_model_and_wire"]["qwen15b_block_matrices"] = True
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_TYPE_MISMATCH")

        c = copy.deepcopy(base)
        c["extra"] = 1
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_KEY_MISMATCH")

        c = copy.deepcopy(base)
        c["pillar_2_sampled_live_qwen15b"]["stacked4_direct_a16_pct_triad"] = [97, 1, 0]
        with self.assertRaises(audit.AuditError) as ctx:
            audit.verify_certificate_payload(_resign(c), full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_LEN_MISMATCH")

        c = copy.deepcopy(base)
        c["pillar_2_sampled_live_qwen15b"]["stacked4_tc_72b_over_onfly_speedup"] = [999999, 1]
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

            # Inode-matched logical $PWD check + stale/unrelated $PWD tolerance
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
                for benign_pwd in (str(sym_other), "/nonexistent_s60_pwd_check"):
                    with mock.patch.dict(os.environ, {"PWD": benign_pwd}):
                        self.assertEqual(audit.verify_certificate_file(Path("cert.json"))["file_sha256"], sha1)
                with mock.patch.object(audit.os, "environ", {"PWD": "/bad\x00pwd"}):
                    self.assertEqual(audit.verify_certificate_file(Path("cert.json"))["file_sha256"], sha1)
            finally:
                os.chdir(prev_cwd)

            # macOS Darwin verified root-owned /var, /tmp, /etc -> /private/* prefix normalization
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

            # Empty path, root '/', '..', missing dir/file, regular-file-as-dir, ELOOP, ENOTDIR+symlink, anchor fail
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

            with mock.patch("rtx3080_qwen15b_audit.os.open", side_effect=fake_anchor_fail):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("/",), True)
                self.assertEqual(ctx.exception.code, "DIR_OPEN_FAILED")

            def fake_eloop(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if p == "eloop_x":
                    raise OSError(errno.ELOOP, "loop")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_qwen15b_audit.os.open", side_effect=fake_eloop):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("eloop_x",), False)
                self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            def fake_enotdir(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if p == "enotdir_lnk":
                    raise OSError(errno.ENOTDIR, "not a dir")
                return real_open(p, flg, *a, **kw)

            with (
                mock.patch("rtx3080_qwen15b_audit.os.open", side_effect=fake_enotdir),
                mock.patch("rtx3080_qwen15b_audit.os.lstat", return_value=root_lnk),
            ):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit._walk_dir_nofollow(("enotdir_lnk",), False)
                self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")

            # chmod 000, symlink, hardlink, scratch os.mkfifo non-regular probe
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

            # Exact size boundary, stat fast-reject before os.read, and streaming cap
            ex4 = root / "ex4.json"
            ex4.write_bytes(b"1234")
            self.assertEqual(audit.read_regular_file_bytes(ex4, max_bytes=4), b"1234")

            big = root / "big.json"
            big.write_bytes(b"a" * (audit.MAX_CERT_BYTES + 1))
            with mock.patch("rtx3080_qwen15b_audit.os.read") as m_read:
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(big)
                self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")
                m_read.assert_not_called()

            smf = root / "smf.json"
            smf.write_bytes(b"1234567890")
            with mock.patch("rtx3080_qwen15b_audit.os.fstat") as m_fs:
                fst = os.stat(smf)
                m_fs.return_value = mock.Mock(st_mode=fst.st_mode, st_nlink=1, st_size=0)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.read_regular_file_bytes(smf, max_bytes=4)
                self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")

            # Atomic write edge guards: EEXIST retry exhaustion, short write, temp tampered nlink/mode, OSError replace
            def fake_eexist(p: Any, flg: int, *a: Any, **kw: Any) -> int:
                if isinstance(p, str) and ".tmp." in p:
                    raise FileExistsError("exists")
                return real_open(p, flg, *a, **kw)

            with mock.patch("rtx3080_qwen15b_audit.os.open", side_effect=fake_eexist):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "retry.json", cert)
                self.assertEqual(ctx.exception.code, "ATOMIC_WRITE_FAILED")
                self.assertIn("exhausted", ctx.exception.message)

            with mock.patch("rtx3080_qwen15b_audit.os.write", return_value=0):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "short.json", cert)
                self.assertEqual(ctx.exception.code, "SHORT_WRITE")

            with mock.patch("rtx3080_qwen15b_audit.os.fstat") as m_fs2:
                m_fs2.return_value = mock.Mock(st_mode=stat.S_IFREG | 0o600, st_nlink=2)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "tamper1.json", cert)
                self.assertEqual(ctx.exception.code, "TEMP_TAMPERED")
            with mock.patch("rtx3080_qwen15b_audit.os.fstat") as m_fs3:
                m_fs3.return_value = mock.Mock(st_mode=stat.S_IFIFO | 0o600, st_nlink=1)
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "tamper2.json", cert)
                self.assertEqual(ctx.exception.code, "TEMP_TAMPERED")

            with mock.patch("rtx3080_qwen15b_audit.os.replace", side_effect=OSError(errno.EIO, "io")):
                with self.assertRaises(audit.AuditError) as ctx:
                    audit.write_certificate_atomic(root / "oserr.json", cert)
                self.assertEqual(ctx.exception.code, "ATOMIC_WRITE_FAILED")

            # verify_receipts_directory: live pass + malformed/empty/{}/tamper
            self.assertIs(audit.verify_receipts_directory(STUDY_DIR)["all_verified"], True)

            bad_r = root / "bad_r"
            bad_r.mkdir()
            (bad_r / "rtx3080_qwen15b_empirical_results.json").write_bytes(b"[]\n")
            (bad_r / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            (bad_r / "rtx3080_qwen15b_kernels.cu").write_bytes(b"//\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            (bad_r / "rtx3080_qwen15b_empirical_results.json").write_bytes(b"{}\n")
            (bad_r / "mlos_zone_b_receipt.json").write_bytes(b"[]\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_ROOT_INVALID")

            (bad_r / "mlos_zone_b_receipt.json").write_bytes(b"{}\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_SCHEMA_ERROR")

            (bad_r / "rtx3080_qwen15b_empirical_results.json").write_bytes(b"{bad\n")
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_PARSE_ERROR")

            (bad_r / "rtx3080_qwen15b_empirical_results.json").write_bytes(b'{"a":1,"a":2}\n')
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "DUPLICATE_KEY")

            (bad_r / "rtx3080_qwen15b_empirical_results.json").write_bytes(
                (STUDY_DIR / "rtx3080_qwen15b_empirical_results.json").read_bytes()
            )
            (bad_r / "mlos_zone_b_receipt.json").write_bytes(
                (STUDY_DIR / "mlos_zone_b_receipt.json").read_bytes()
            )
            with self.assertRaises(audit.AuditError) as ctx:
                audit.verify_receipts_directory(bad_r)
            self.assertEqual(ctx.exception.code, "RECEIPT_VERIFICATION_FAILED")

            # CLI tests (including NUL, UTF-8 surrogate, flag-like, arity errors)
            cli_c = root / "cli_c.json"
            for argv, exp_rc, exp_err in (
                ([], 0, None),
                (["--standalone", "--json"], 0, None),
                (["--write", str(cli_c)], 0, None),
                (["--check", str(cli_c)], 0, None),
                (["--verify-receipts", str(STUDY_DIR)], 0, None),
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
                    rc = audit.main(argv)
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
        disk_cert = STUDY_DIR / "rtx3080_qwen15b_certificate.json"
        res = audit.verify_certificate_file(disk_cert, full_replay=True)
        self.assertIs(res["valid"], True)
        self.assertEqual(stat.S_IMODE(os.lstat(disk_cert).st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
