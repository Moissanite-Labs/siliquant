#!/usr/bin/env python3
"""
Track S20b (`AUDIT-119B`): R5 Adversarial, Zone-B & Mutation-Killing Test Suite
(`test_e8had_anisotropy_r5.py`).

Pure stdlib, zero bare `assert`s (`-O`-safe), zero floats on trust paths,
4-mode `-I` parity verified.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import e8had_anisotropy_audit as audit  # noqa: E402
import e8had_anisotropy_oracle as oracle  # noqa: E402


class _IntSub(int):
    pass


class _DictSub(dict):
    pass


class _StrSub(str):
    pass


class _TupleSub(tuple):
    pass


class _ListSub(list):
    pass


class TestTrackS20bE8HadAnisotropyR5(unittest.TestCase):
    cert: dict[str, object]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.generate_certificate()

    def test_01_root_tomography_and_block_h2_exact_identity(self) -> None:
        p = self.cert["payload"]
        assert_isinstance = self.assertIsInstance
        assert_isinstance(p, dict)
        rt = p["root_tomography_and_block_h2_identity"]
        self.assertEqual(rt["e8_root_count"], 240)
        self.assertEqual(rt["e8_root_hamming_weights"], {"2": 112, "8": 128})
        self.assertEqual(rt["e8_had_root_hamming_weights"], {"1": 16, "4": 224})
        self.assertEqual(rt["weight8_roots_eliminated"], 128)
        self.assertEqual(rt["extended_hamming_wt4_supports"], 14)
        self.assertEqual(rt["wt4_covered_pairs"], 28)
        self.assertEqual(rt["wt4_pair_cover_multiplicities"], [3])
        self.assertTrue(rt["block_h2_equals_wht8_roots_exact"])
        self.assertTrue(rt["intertwiner_half_h4_kron_i2_orthogonal"])
        self.assertTrue(rt["intertwiner_preserves_all_240_e8_roots"])
        self.assertEqual(rt["cvp_exact_distance_matches_bh2_vs_wht8"], 35)
        self.assertEqual(rt["cvp_test_points_total"], 35)
        self.assertEqual(rt["wht8_butterfly_ops"], 24)
        self.assertEqual(rt["block_h2_butterfly_ops"], 8)
        self.assertEqual(rt["butterfly_op_reduction_factor"], 3)

        # Direct verify_orthogonal_matrix negative, positive, and Zone-B S20B-BUG-01 controls
        i8 = tuple(tuple(1 if i == j else 0 for j in range(8)) for i in range(8))
        self.assertTrue(oracle.verify_orthogonal_matrix(i8, 1))
        self.assertTrue(oracle.verify_orthogonal_matrix([list(r) for r in i8], 1))
        self.assertFalse(oracle.verify_orthogonal_matrix(i8[:7], 1))
        bad_row_len = (i8[0][:7],) + i8[1:]
        self.assertFalse(oracle.verify_orthogonal_matrix(bad_row_len, 1))
        self.assertFalse(oracle.verify_orthogonal_matrix(i8, 2))
        bad_offdiag = ((1, 1, 0, 0, 0, 0, 0, 0),) + i8[1:]
        self.assertFalse(oracle.verify_orthogonal_matrix(bad_offdiag, 1))
        bool_i8 = tuple(tuple(True if i == j else False for j in range(8)) for i in range(8))
        huge_i8 = ((10**5000, 0, 0, 0, 0, 0, 0, 0),) + i8[1:]
        for bad_mat, bad_sc in (
            (i8, 0),
            (i8, True),
            (None, 1),
            (_ListSub(i8), 1),
            ((None,) * 8, 1),
            ((_TupleSub(i8[0]),) + i8[1:], 1),
            (bool_i8, 1),
            (huge_i8, 1),
        ):
            with self.assertRaises(oracle.AnisotropyError):
                oracle.verify_orthogonal_matrix(bad_mat, bad_sc)  # type: ignore[arg-type]

    def test_02_k_sparse_anisotropy_neighbor_bleed_and_cvp_guards(self) -> None:
        p = self.cert["payload"]
        ks = p["k_sparse_anisotropy_law"]
        self.assertEqual(ks["k1_axes"]["e8"], {"ray_count": 8, "max_sq_err": "1", "mean_sq_err": "1/3", "max_voronoi_gauge_sq": "1"})
        self.assertEqual(ks["k1_axes"]["e8_had"], {"ray_count": 8, "max_sq_err": "1/2", "mean_sq_err": "1/6", "max_voronoi_gauge_sq": "2"})
        self.assertEqual(ks["k2_pairs"]["e8"], {"ray_count": 56, "max_sq_err": "1/2", "mean_sq_err": "1/6", "max_voronoi_gauge_sq": "2"})
        self.assertEqual(ks["k2_pairs"]["e8_had"], {"ray_count": 56, "max_sq_err": "1", "mean_sq_err": "1/3", "max_voronoi_gauge_sq": "1"})
        self.assertEqual(ks["k3_triples"]["e8"], {"ray_count": 224, "max_sq_err": "1", "mean_sq_err": "1/2", "max_voronoi_gauge_sq": "4/3"})
        self.assertEqual(ks["k3_triples"]["e8_had"], {"ray_count": 224, "max_sq_err": "2/3", "mean_sq_err": "1/3", "max_voronoi_gauge_sq": "3/2"})
        self.assertEqual(ks["k4_quads_plus"]["e8"]["mean_sq_err"], "1/3")
        self.assertEqual(ks["k4_quads_plus"]["e8_had"]["mean_sq_err"], "13/30")
        self.assertEqual(ks["k8_all_ones"]["e8"]["mean_sq_err"], "1/6")
        self.assertEqual(ks["k8_all_ones"]["e8_had"]["mean_sq_err"], "1/3")

        nb = p["neighbor_bleed_and_voronoi_duality"]
        self.assertEqual(nb["e8_bleed_threshold_eps"], "0")
        self.assertEqual(nb["e8_had_bleed_threshold_eps_sq"], "1/2")
        self.assertEqual(nb["e8_axis_voronoi_gauge_sq"], "1")
        self.assertEqual(nb["e8_had_axis_voronoi_gauge_sq"], "2")
        self.assertEqual(len(nb["bleed_probes"]), 4)
        self.assertEqual(nb["bleed_probes"][0]["e8_amplification_vs_eps2"], "9801")
        self.assertTrue(all(row["e8_had_stayed_on_axis"] for row in nb["bleed_probes"]))

        # Independent cross-check of voronoi_gauge_sq_unit_e8 against explicit 240 roots
        roots2 = oracle.generate_e8_roots_doubled()
        for vec in (
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0),
            (3, 2, 1, 1, 1, 1, 1, 1),
            (3, 2, 1, 1, 1, 1, 1, -1),
            (1, 1, 1, 1, 1, 1, 1, -1),
            (-1, -1, -1, -1, -1, -1, -1, -1),
        ):
            norm2 = sum(c * c for c in vec)
            brute = max(Fraction((sum(vec[i] * r2[i] for i in range(8))) ** 2, 4 * norm2) for r2 in roots2)
            self.assertEqual(oracle.voronoi_gauge_sq_unit_e8(vec), brute)

        # Direct CVP & orbit invariance & error guards
        z_neg, sd_neg = oracle.closest_d8_scaled((-6, 0, 0, 0, 0, 0, 0, 0), 10)
        self.assertEqual(z_neg, (0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(sd_neg, 36)
        z_pos, sd_pos = oracle.closest_d8_scaled((6, 0, 0, 0, 0, 0, 0, 0), 10)
        self.assertEqual(z_pos, (0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(sd_pos, 36)
        # Positive residual flip on index 1 (beats index 0): (11/10, 4/10, 0^6) -> rounds (1,0), flips idx 1 to +1!
        z_flip_idx, _ = oracle.closest_d8_scaled((11, 4, 0, 0, 0, 0, 0, 0), 10)
        self.assertEqual(z_flip_idx, (1, 1, 0, 0, 0, 0, 0, 0))
        # Negative residual flip on index 1 (beats index 0): (11/10, -4/10, 0^6) -> rounds (1,0), flips idx 1 to -1!
        z_flip_neg_idx, _ = oracle.closest_d8_scaled((11, -4, 0, 0, 0, 0, 0, 0), 10)
        self.assertEqual(z_flip_neg_idx, (1, -1, 0, 0, 0, 0, 0, 0))
        # Exact Steiner-quadruple forced-jump checks
        st = nb["low_bitrate_shell_gap_and_steiner_jump"]
        self.assertEqual(st["z8_min_nonzero_shell_sq"], 1)
        self.assertEqual(st["e8_and_e8_had_min_nonzero_shell_sq"], 2)
        self.assertEqual(st["e8_half_integer_jump_sq"], "1/4")
        self.assertEqual(st["e8_had_steiner_quadruple_jump_sq"], "1/2")
        self.assertEqual(st["steiner_over_half_int_jump_sq_ratio"], 2)
        self.assertEqual(len(st["steiner_probes"]), 3)
        self.assertEqual(st["steiner_probes"][0]["e8_had_snapped_4th_coord_s2"], 1)
        self.assertEqual(st["steiner_probes"][0]["e8_had_4th_coord_sq_err"], "9801/20000")
        self.assertEqual(st["steiner_probes"][0]["e8_had_amplification_vs_eps2"], "9801")
        # E8 tie/half-integer coset exact check
        p2_half, _, _ = oracle.closest_e8_doubled((1, 1, 1, 1, 1, 1, 1, 1), 2)
        self.assertEqual(p2_half, (1, 1, 1, 1, 1, 1, 1, 1))
        # Direct uncached ray_quantization_stats_exact invariance under signed permutation
        self.assertEqual(
            oracle.ray_quantization_stats_exact((1, 1, 1, 0, 0, 0, 0, 0)),
            oracle.ray_quantization_stats_exact((0, -1, 0, 1, 0, 0, -1, 0)),
        )
        # Odd-all-nonzero orbit key distinction
        k_even = oracle._e8_orbit_key((1, 1, 1, 1, 1, 1, 1, 1))
        k_odd = oracle._e8_orbit_key((1, 1, 1, 1, 1, 1, 1, -1))
        self.assertNotEqual(k_even, k_odd)
        self.assertEqual(oracle._e8_orbit_key((0, 1, 0, 0, 0, 0, 0, 0)), ((0, 0, 0, 0, 0, 0, 0, 1), 0))
        # Dimension / container / bool / zero-vector Zone-B guards (S20B-BUG-01)
        self.assertEqual(oracle.closest_d8_scaled([0] * 8, 1), ((0,) * 8, 0))
        self.assertEqual(oracle.closest_e8_doubled([0] * 8, 1), ((0,) * 8, 0, 1))
        for bad_call in (
            lambda: oracle.closest_d8_scaled(None, 1),  # type: ignore[arg-type]
            lambda: oracle.closest_d8_scaled(_TupleSub((0,) * 8), 1),
            lambda: oracle.closest_d8_scaled((1, 2, 3), 1),
            lambda: oracle.closest_d8_scaled((0,) * 8, 0),
            lambda: oracle.closest_d8_scaled((True,) + (0,) * 7, 1),
            lambda: oracle.closest_e8_doubled(None, 1),  # type: ignore[arg-type]
            lambda: oracle.closest_e8_doubled(_ListSub([0] * 8), 1),
            lambda: oracle.closest_e8_doubled((1, 2), 1),
            lambda: oracle.closest_e8_doubled((1, 0, 0, 0, 0, 0, 0, 0), 0),
            lambda: oracle.closest_e8_doubled((1, 0, 0, 0, 0, 0, 0, 0), True),
            lambda: oracle.closest_e8_doubled((True,) + (0,) * 7, 1),
            lambda: oracle.closest_e8_doubled((10**5000,) + (0,) * 7, 1),
            lambda: oracle.ray_quantization_stats_exact(None),  # type: ignore[arg-type]
            lambda: oracle.ray_quantization_stats_exact((True,) + (0,) * 7),
            lambda: oracle.ray_quantization_stats_exact((1, 2, 3)),
            lambda: oracle.ray_quantization_stats_exact((0,) * 8),
            lambda: oracle.voronoi_gauge_sq_unit_e8(None),  # type: ignore[arg-type]
            lambda: oracle.voronoi_gauge_sq_unit_e8((True,) + (0,) * 7),
            lambda: oracle.voronoi_gauge_sq_unit_e8((1, 2, 3)),
            lambda: oracle.voronoi_gauge_sq_unit_e8((0,) * 8),
        ):
            with self.assertRaises(oracle.AnisotropyError):
                bad_call()

    def test_03_coset_benchmark_optimality_and_empirical_decomposition(self) -> None:
        p = self.cert["payload"]
        cosets = {c["family"]: c for c in p["orthogonal_coset_benchmark"]}
        self.assertEqual(len(cosets), 8)
        for c in cosets.values():
            self.assertTrue(c["orthogonal_verified"])
        self.assertEqual(cosets["wht8_e8_had"]["k1_all8"]["mean_sq_err"], "1/6")
        self.assertEqual(cosets["block_h2_fast"]["k1_all8"]["mean_sq_err"], "1/6")
        self.assertEqual(cosets["block_h2_fast"]["butterfly_ops_per_8block"], 8)
        self.assertEqual(cosets["circulant_cw_8_4"]["k2_all56"]["mean_sq_err"], "5/14")
        self.assertEqual(cosets["paley_w_8_7"]["k1_all8"]["mean_sq_err"], "47/105")
        self.assertEqual(cosets["paley_w_8_7"]["k2_all56"]["max_sq_err"], "8/9")

        opt = p["global_o8_optimality_witness"]
        self.assertEqual(opt["shapes_tested_norm2_le_16"], 46)
        self.assertEqual(opt["global_min_max_sq_err"], "1/2")
        self.assertEqual(opt["global_min_mean_sq_err"], "1/6")
        self.assertEqual(opt["root_norm2_shape_count"], 3)
        self.assertEqual(opt["min_max_shape_count"], 3)
        self.assertEqual(opt["min_mean_shape_count"], 3)
        self.assertEqual(opt["two_point_identity_verified_points"], 26)
        self.assertEqual(opt["two_point_identity_total_points"], 26)
        self.assertTrue(opt["uniqueness_via_doubly_even_self_dual_8_4_4_code"])

        emp = p["empirical_decomposition"]
        lk = emp["llama_k10_collapse"]
        self.assertEqual(lk["raw_drop_u6"], 4123)
        self.assertEqual(lk["raw_rel_reduction"], "4123/40161")
        self.assertEqual(lk["k10_drop_u6"], 174)
        self.assertEqual(lk["k10_rel_reduction"], "87/12689")
        self.assertEqual(lk["collapse_factor"], "52316747/3494007")
        self.assertEqual(lk["share_of_total_hbg_gain_from_loud_columns"], "3949/4123")
        self.assertEqual(lk["loud_damage_reduction_had_vs_e8"], "3949/14783")
        self.assertEqual(lk["loud_damage_reduction_had_vs_z8"], "6751/17585")

        vg = emp["voronoi_gauge_reversal"]
        self.assertTrue(vg["qwen25_h11_falsified"])
        self.assertEqual(vg["qwen25_vh8z_minus_ve8z_u6"], 3052)
        self.assertEqual(vg["qwen25_attn_k_penalty_mdB"], 187)
        self.assertEqual(vg["qwen25_attn_v_penalty_mdB"], 176)
        self.assertEqual(vg["smollm2_vh8z_rel_improvement"], "4875/41576")

        sb = emp["smollm2_3p0_low_bitrate_crossover"]
        self.assertEqual(sb["ebg_minus_zbg_u6"], 244546)
        self.assertEqual(sb["ebg_rel_penalty_over_zbg"], "244546/419643")
        self.assertEqual(sb["hbg_minus_zbg_u6"], 210259)
        self.assertEqual(sb["hbg_rel_penalty_over_zbg"], "30037/59949")
        self.assertEqual(sb["hbg_rel_reduction_vs_ebg"], "34287/664189")
        self.assertEqual(sb["ebg_werr_advantage_mdB"], 654)
        self.assertEqual(sb["hbg_small005_err_mdB"], 9606)
        self.assertEqual(sb["hbg_small020_err_mdB"], 2025)
        self.assertEqual(sb["steiner_jump_model_005_energy_ratio_hbg_over_ebg"], "1118/3189")
        self.assertEqual(sb["steiner_jump_model_020_energy_ratio_hbg_over_ebg"], "4448/4525")
        self.assertEqual(sb["qwen25_3p0_zbg_landed_kl_u6"], 215147)

    def test_04_gate8_evaluator_boundaries_and_clause_kills(self) -> None:
        # Pending baseline
        empty_eval = oracle.evaluate_gate8_rows({})
        self.assertEqual(
            empty_eval["verdicts"],
            {"G8-P1": "PENDING", "G8-P2": "PENDING", "G8-P3": "PENDING", "G8-P4": "PENDING"},
        )

        # Clean PASS bundle (strict interior; landed Qwen2.5 ZBG = 215147, SE = 3372)
        good_rows: dict[str, object] = {
            "qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2800},
            "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2800, "z005_bp": 9300, "err005_mdB": 8500},
            "llama_ve8z_ffn_down_mdB": -22600,
            "llama_vh8z_ffn_down_mdB": -22400,
            "llama_ve8z_k10_ffn_down_mdB": -22600,
            "llama_vh8z_k10_ffn_down_mdB": -22580,
            "qwen25_4p5_ebg_k10": {"kl_u6": 18000, "se_u6": 400},
            "qwen25_4p5_hbg_k10": {"kl_u6": 17800, "se_u6": 400},
            "block_h2_kl_diff_u6": 20,
            "wht8_se_u6": 500,
            "block_h2_werr_diff_mdB": 0,
        }
        good_eval = oracle.evaluate_gate8_rows(good_rows)
        self.assertEqual(
            good_eval["verdicts"],
            {"G8-P1": "PASS", "G8-P2": "PASS", "G8-P3": "PASS", "G8-P4": "PASS"},
        )

        # G8-P1 exact boundary PASS (z005 == 8800, err005 == 6000) & individual clause FAILs
        b1_edge = {
            "qwen25_3p0_ebg": {"kl_u6": 197935, "se_u6": 2000},  # red_ez >= 8% (17212/215147 = 0.080001), z2 > 9
            "qwen25_3p0_hbg": {"kl_u6": 193632, "se_u6": 2000, "z005_bp": 8800, "err005_mdB": 6000},  # red_hz >= 10%, z2 > 16
        }
        self.assertEqual(oracle.evaluate_gate8_rows(b1_edge)["verdicts"]["G8-P1"], "PASS")
        p1_fails = [
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 193633, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},  # red_hz < 10%
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 190000, "se_u6": 6000, "z005_bp": 9300, "err005_mdB": 8500}},  # z2_hz < 16
            {"qwen25_3p0_ebg": {"kl_u6": 197936, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},  # red_ez < 8%
            {"qwen25_3p0_ebg": {"kl_u6": 195000, "se_u6": 6500}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},  # z2_ez < 9
            {"qwen25_3p0_ebg": {"kl_u6": 190000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 161500, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},  # gap_he == 15/100 (not < 15%)
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 8799, "err005_mdB": 8500}},  # z005 < 8800
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 5999}},  # err005 < 6000
        ]
        for bad_p1 in p1_fails:
            self.assertEqual(oracle.evaluate_gate8_rows(bad_p1)["verdicts"]["G8-P1"], "FAIL")

        # G8-P2 exact boundary PASS (pen_raw == 150, pen_k10 == 49) & single-clause FAILs
        self.assertEqual(
            oracle.evaluate_gate8_rows(
                {"llama_ve8z_ffn_down_mdB": -22600, "llama_vh8z_ffn_down_mdB": -22450, "llama_ve8z_k10_ffn_down_mdB": -22600, "llama_vh8z_k10_ffn_down_mdB": -22551}
            )["verdicts"]["G8-P2"],
            "PASS",
        )
        self.assertEqual(
            oracle.evaluate_gate8_rows(
                {"llama_ve8z_ffn_down_mdB": -22600, "llama_vh8z_ffn_down_mdB": -22451, "llama_ve8z_k10_ffn_down_mdB": -22600, "llama_vh8z_k10_ffn_down_mdB": -22590}
            )["verdicts"]["G8-P2"],
            "FAIL",
        )
        self.assertEqual(
            oracle.evaluate_gate8_rows(
                {"llama_ve8z_ffn_down_mdB": -22600, "llama_vh8z_ffn_down_mdB": -22400, "llama_ve8z_k10_ffn_down_mdB": -22600, "llama_vh8z_k10_ffn_down_mdB": -22550}
            )["verdicts"]["G8-P2"],
            "FAIL",
        )

        # G8-P3 exact boundary PASS (rel_gap == 2/100, z2 < 4) & single-clause FAILs
        self.assertEqual(
            oracle.evaluate_gate8_rows(
                {"qwen25_4p5_ebg_k10": {"kl_u6": 10000, "se_u6": 300}, "qwen25_4p5_hbg_k10": {"kl_u6": 9800, "se_u6": 300}}
            )["verdicts"]["G8-P3"],
            "PASS",
        )
        self.assertEqual(
            oracle.evaluate_gate8_rows(
                {"qwen25_4p5_ebg_k10": {"kl_u6": 10000, "se_u6": 300}, "qwen25_4p5_hbg_k10": {"kl_u6": 9799, "se_u6": 300}}
            )["verdicts"]["G8-P3"],
            "FAIL",
        )
        self.assertEqual(
            oracle.evaluate_gate8_rows(
                {"qwen25_4p5_ebg_k10": {"kl_u6": 10000, "se_u6": 60}, "qwen25_4p5_hbg_k10": {"kl_u6": 9800, "se_u6": 80}}
            )["verdicts"]["G8-P3"],
            "FAIL",
        )  # z2 == 4 (not < 4!)

        # G8-P4 exact boundary PASS (10*kd == se, wd == 1) & single-clause FAILs
        self.assertEqual(
            oracle.evaluate_gate8_rows({"block_h2_kl_diff_u6": 50, "wht8_se_u6": 500, "block_h2_werr_diff_mdB": 1})["verdicts"]["G8-P4"],
            "PASS",
        )
        self.assertEqual(
            oracle.evaluate_gate8_rows({"block_h2_kl_diff_u6": 51, "wht8_se_u6": 500, "block_h2_werr_diff_mdB": 1})["verdicts"]["G8-P4"],
            "FAIL",
        )
        self.assertEqual(
            oracle.evaluate_gate8_rows({"block_h2_kl_diff_u6": 10, "wht8_se_u6": 500, "block_h2_werr_diff_mdB": 2})["verdicts"]["G8-P4"],
            "FAIL",
        )

        # Valid partial Gate-8 submissions MUST stay PENDING without crashing
        for partial_ok, want_pending in (
            ({"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2800}}, "G8-P1"),
            ({"qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2800, "z005_bp": 9300, "err005_mdB": 8500}}, "G8-P1"),
            ({"llama_ve8z_ffn_down_mdB": -22600}, "G8-P2"),
            ({"qwen25_4p5_ebg_k10": {"kl_u6": 18000, "se_u6": 400}}, "G8-P3"),
            ({"qwen25_4p5_hbg_k10": {"kl_u6": 17800, "se_u6": 400}}, "G8-P3"),
            ({"block_h2_kl_diff_u6": 10}, "G8-P4"),
            ({"block_h2_kl_diff_u6": 10, "wht8_se_u6": 500}, "G8-P4"),
        ):
            self.assertEqual(oracle.evaluate_gate8_rows(partial_ok)["verdicts"][want_pending], "PENDING")

        # Bad Gate-8 input containers / keys / types / malformed partial rows (S20B-BUG-02)
        for bad_in in (
            [],
            _DictSub(),
            {1: {}},
            {chr(0xD800): {}},
            {"qwen25_3p0_ebg": []},
            {"qwen25_3p0_hbg": []},
            {"qwen25_3p0_ebg": {"kl_u6": True, "se_u6": 100}},
            {"qwen25_3p0_ebg": {"kl_u6": 10**5000, "se_u6": -5}},
            {"qwen25_3p0_ebg": {"kl_u6": 100, "se_u6": 0}},
            {"qwen25_3p0_ebg": {"kl_u6": float("nan")}},
            {"qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2800, "z005_bp": 10001, "err005_mdB": 8500}},
            {"qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2800, "z005_bp": 9300, "err005_mdB": True}},
            {"qwen25_4p5_ebg_k10": []},
            {"qwen25_4p5_hbg_k10": []},
            {"qwen25_4p5_ebg_k10": {"kl_u6": 0, "se_u6": 400}},
            {"llama_ve8z_ffn_down_mdB": True},
            {"llama_vh8z_ffn_down_mdB": 1},
            {"llama_ve8z_k10_ffn_down_mdB": -100001},
            {"block_h2_kl_diff_u6": True},
            {"wht8_se_u6": 0},
            {"block_h2_werr_diff_mdB": "bad"},
            {"unknown_float": 1.5},
        ):
            with self.assertRaises(oracle.AnisotropyError):
                oracle.evaluate_gate8_rows(bad_in)  # type: ignore[arg-type]

    def test_05_zone_b_strict_io_and_bounds(self) -> None:
        max_2048 = (1 << oracle.MAX_INT_BITS) - 1
        over_2048 = 1 << oracle.MAX_INT_BITS
        self.assertEqual(oracle.require_int(max_2048, "m"), max_2048)
        self.assertEqual(oracle.require_int(5, "b", lo=5, hi=5), 5)
        for bad_call in (
            lambda: oracle.require_int(True, "b"),
            lambda: oracle.require_int(_IntSub(1), "s"),
            lambda: oracle.require_int(over_2048, "o"),
            lambda: oracle.require_int(10**5000, "h"),
            lambda: oracle.require_int(4, "lo", lo=5),
            lambda: oracle.require_int(6, "hi", hi=5),
            lambda: oracle.require_int(1, "huge_lo", lo=10**5000),
            lambda: oracle.require_int(1, "huge_hi", hi=-(10**5000)),
        ):
            with self.assertRaises(oracle.AnisotropyError):
                bad_call()

        good_hex = "0123456789abcdef" * 4
        other_hex = "fedcba9876543210" * 4
        self.assertTrue(oracle.constant_time_hex64_equal(good_hex, good_hex))
        self.assertFalse(oracle.constant_time_hex64_equal(good_hex, other_hex))
        for bad_hex in (123, "a" * 63, "a" * 65, "A" * 64, "\u00e9" * 64, chr(0xD800) * 64):
            with self.assertRaises(oracle.AnisotropyError):
                oracle.constant_time_hex64_equal(bad_hex, good_hex)

        self.assertEqual(oracle.fmt_q(Fraction(3, 1)), "3")
        self.assertEqual(oracle.fmt_q(Fraction(1, 6)), "1/6")
        self.assertEqual(oracle.fmt_q(Fraction(max_2048, max_2048 - 1)), f"{max_2048}/{max_2048 - 1}")
        for bad_q in (
            lambda: oracle.fmt_q(1),  # type: ignore[arg-type]
            lambda: oracle.fmt_q(Fraction(over_2048, 1)),
            lambda: oracle.fmt_q(Fraction(1, over_2048)),
            lambda: oracle.fmt_q(Fraction(10**5000, 1)),
        ):
            with self.assertRaises(oracle.AnisotropyError):
                bad_q()

        # validate_json_value depth & strict types
        d64: object = 0
        for _ in range(64):
            d64 = [d64]
        oracle.validate_json_value(d64, max_depth=64)
        oracle.validate_json_value({"a": True, "b": None, "c": max_2048, "d": -max_2048, "s": "ok"})
        d65 = [d64]
        deep1500: object = 0
        for _ in range(1500):
            deep1500 = [deep1500]
        for bad_val in (
            d65,
            deep1500,
            {1: "a"},
            {chr(0xD800): "a"},
            {"s": chr(0xD800)},
            {"n": over_2048},
            {"n": 10**5000},
            {"n": _IntSub(1)},
            {"o": object()},
            _DictSub(a=1),
        ):
            with self.assertRaises(oracle.AnisotropyError):
                oracle.validate_json_value(bad_val)
        with self.assertRaises(oracle.AnisotropyError) as ctx_flt:
            oracle.validate_json_value({"x": 1.5})
        self.assertEqual(ctx_flt.exception.code, "FLOAT_FORBIDDEN")

        # Direct _parse_capped_int_token exact boundaries (kills both MAX_INT_DIGITS and MAX_INT_BITS boundaries!)
        self.assertEqual(oracle._parse_capped_int_token("-42"), -42)
        self.assertEqual(oracle._parse_capped_int_token(str(max_2048)), max_2048)
        with self.assertRaises(oracle.AnisotropyError) as ctx_d620:
            oracle._parse_capped_int_token("1" + ("0" * (oracle.MAX_INT_DIGITS - 1)))
        self.assertEqual(ctx_d620.exception.code, "INT_BIT_LIMIT")
        with self.assertRaises(oracle.AnisotropyError) as ctx_d621:
            oracle._parse_capped_int_token("9" * (oracle.MAX_INT_DIGITS + 1))
        self.assertEqual(ctx_d621.exception.code, "INT_DIGIT_LIMIT")
        with self.assertRaises(oracle.AnisotropyError) as ctx_b2049:
            oracle._parse_capped_int_token(str(over_2048))
        self.assertEqual(ctx_b2049.exception.code, "INT_BIT_LIMIT")

        # parse_strict_json_text boundaries & Zone-B attacks
        self.assertEqual(oracle.parse_strict_json_text('{"k":1}'), {"k": 1})
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])
        exact_max_str = '{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}'
        self.assertEqual(len(exact_max_str.encode("utf-8")), oracle.MAX_JSON_BYTES)
        self.assertEqual(len(oracle.parse_strict_json_text(exact_max_str)["a"]), oracle.MAX_JSON_BYTES - 8)
        with self.assertRaises(oracle.AnisotropyError) as ctx_empty:
            oracle.parse_strict_json_text("   ")
        self.assertEqual(ctx_empty.exception.code, "EMPTY_JSON_INPUT")
        for bad_txt in (
            123,
            chr(0xD800),
            exact_max_str + " ",
            "{bad",
            '{"x":1.5}',
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"a":1,"a":2}',
            '{"n":' + ("9" * 5000) + "}",
            '{"s":"\\ud800"}',
            "[" * 100 + "0" + "]" * 100,
            "[1,2]",
        ):
            with self.assertRaises(oracle.AnisotropyError):
                oracle.parse_strict_json_text(bad_txt)  # type: ignore[arg-type]

    def test_06_auditor_file_verification_tamper_and_cli(self) -> None:
        cert_path = _HERE / "e8had_anisotropy_certificate.json"
        res = audit.verify_certificate_file(cert_path)
        self.assertTrue(res["valid"])
        self.assertEqual(res["payload_sha256"], self.cert["payload_sha256"])
        self.assertEqual(res["certificate_sha256"], self.cert["certificate_sha256"])

        # verify_certificate_dict tampering & structural errors (assert exact error codes!)
        good_copy = json.loads(oracle.canonical_json_bytes(self.cert).decode("ascii"))
        bad_schema_type = dict(good_copy, schema=123)
        bad_schema_sub = dict(good_copy, schema=_StrSub(good_copy["schema"]))
        bad_schema_val = dict(good_copy, schema="wrong.schema")
        bad_payload_type = dict(good_copy, payload=[1, 2])
        bad_keys = {k: v for k, v in good_copy.items() if k != "schema"}
        bad_psha = dict(good_copy, payload_sha256="0" * 64)
        bad_csha = dict(good_copy, certificate_sha256="0" * 64)
        # Self-consistent forged payload (passes internal hashes, MUST fail recompute check!)
        forged_payload = {"forged": 1}
        fp_sha = oracle.canonical_sha256(forged_payload)
        forged_body = {"schema": good_copy["schema"], "payload_sha256": fp_sha, "payload": forged_payload}
        forged_cert = {
            "schema": good_copy["schema"],
            "payload_sha256": fp_sha,
            "certificate_sha256": oracle.canonical_sha256(forged_body),
            "payload": forged_payload,
        }
        for bad_cert, want_code in (
            ([], "INVALID_CERT_TYPE"),
            (bad_keys, "INVALID_CERT_KEYS"),
            (bad_schema_type, "INVALID_SCHEMA"),
            (bad_schema_sub, "INVALID_SCHEMA"),
            (bad_schema_val, "INVALID_SCHEMA"),
            (bad_payload_type, "INVALID_PAYLOAD_TYPE"),
            (bad_psha, "PAYLOAD_SHA_MISMATCH"),
            (bad_csha, "CERT_SHA_MISMATCH"),
            (forged_cert, "RECOMPUTE_PAYLOAD_MISMATCH"),
        ):
            with self.assertRaises(oracle.AnisotropyError) as ctx_c:
                audit.verify_certificate_dict(bad_cert)
            self.assertEqual(ctx_c.exception.code, want_code)

        # File & symlink & boundary guards
        with self.assertRaises(oracle.AnisotropyError):
            audit.require_no_symlink_path("not_a_path")  # type: ignore[arg-type]
        with self.assertRaises(oracle.AnisotropyError):
            audit.require_no_symlink_path(Path("bad" + chr(0xD800)))
        with self.assertRaises(oracle.AnisotropyError):
            audit.require_no_symlink_path(Path("bad\x00path"))

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Symlink target & symlink ancestor
            real_dir = tmp / "real"
            real_dir.mkdir()
            real_file = real_dir / "c.json"
            real_file.write_bytes(oracle.canonical_json_bytes(self.cert))
            # Verify canonical file WITHOUT trailing newline passes cleanly
            self.assertTrue(audit.verify_certificate_file(real_file)["valid"])

            sym_file = tmp / "sym.json"
            sym_file.symlink_to(real_file)
            with self.assertRaises(oracle.AnisotropyError):
                audit.verify_certificate_file(sym_file)

            sym_dir = tmp / "symdir"
            sym_dir.symlink_to(real_dir)
            with self.assertRaises(oracle.AnisotropyError):
                audit.verify_certificate_file(sym_dir / "c.json")

            # Missing file
            with self.assertRaises(oracle.AnisotropyError):
                audit.verify_certificate_file(tmp / "missing.json")

            # Exact MAX_JSON_BYTES file (passes st_size check, fails schema/canonical) vs MAX_JSON_BYTES + 1 (fails FILE_TOO_LARGE)
            exact_sz = tmp / "exact_max.json"
            exact_sz.write_bytes(('{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}').encode("ascii"))
            with self.assertRaises(oracle.AnisotropyError) as ctx_exact:
                audit.verify_certificate_file(exact_sz)
            self.assertEqual(ctx_exact.exception.code, "INVALID_CERT_KEYS")

            over_sz = tmp / "over_max.json"
            over_sz.write_bytes(b" " * (oracle.MAX_JSON_BYTES + 1))
            with self.assertRaises(oracle.AnisotropyError) as ctx_over:
                audit.verify_certificate_file(over_sz)
            self.assertEqual(ctx_over.exception.code, "FILE_TOO_LARGE")

            # CR/CRLF, bad UTF-8, non-canonical pretty JSON
            cr_file = tmp / "cr.json"
            cr_file.write_bytes(oracle.canonical_json_bytes(self.cert) + b"\r\n")
            with self.assertRaises(oracle.AnisotropyError) as ctx_cr:
                audit.verify_certificate_file(cr_file)
            self.assertEqual(ctx_cr.exception.code, "CR_FORBIDDEN")

            bad_utf8 = tmp / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe")
            with self.assertRaises(oracle.AnisotropyError):
                audit.verify_certificate_file(bad_utf8)

            pretty = tmp / "pretty.json"
            pretty.write_text(json.dumps(self.cert, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(oracle.AnisotropyError):
                audit.verify_certificate_file(pretty)

        # CLI main() paths for oracle & auditor
        buf_o_json, buf_o_txt = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_o_json):
            self.assertEqual(oracle.main(["--standalone", "--json"]), 0)
        self.assertEqual(json.loads(buf_o_json.getvalue())["certificate_sha256"], self.cert["certificate_sha256"])
        with redirect_stdout(buf_o_txt):
            self.assertEqual(oracle.main(["--standalone"]), 0)
        self.assertIn("OK TRACK-S20B-E8HAD-ANISOTROPY", buf_o_txt.getvalue())

        buf_a_json, buf_a_txt, buf_a_err = io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(buf_a_json):
            self.assertEqual(audit.main(["--check", str(cert_path), "--json"]), 0)
        self.assertTrue(json.loads(buf_a_json.getvalue())["valid"])
        with redirect_stdout(buf_a_txt):
            self.assertEqual(audit.main(["--standalone"]), 0)
        self.assertIn("OK TRACK-S20B-AUDIT", buf_a_txt.getvalue())
        with redirect_stderr(buf_a_err):
            self.assertEqual(audit.main(["--check", str(_HERE / "nonexistent_cert.json")]), 1)
        self.assertIn("FAIL FILE_NOT_FOUND", buf_a_err.getvalue())


if __name__ == "__main__":
    unittest.main()
