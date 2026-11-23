"""Assertion-free, -O-safe R5 verification suite for Track S6 (AUDIT-103):
Moissanite-Labs/siliquant Force-Multiplier Audit & Exact Operator Bridges.

Pins every oracle guard, branchless bitmask E8 optimality vs independent
brute-force oracle, Issue #16 zero-codeword vs Code.centre shift penalty,
TPU v5e Part-E roofline bounds, homeostatic scaling + Kirchhoff R_eff vs
persist/ MST separation, al-Biruni cancellation & al-Kashi contraction,
Pillar-5 full re-derivation, and a live temporary-git-repo known-answer test.
"""

import copy
from fractions import Fraction
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import siliquant_bridge_oracle as o
import siliquant_bridge_audit as a


class TestPrimitivesAndE8Branchless(unittest.TestCase):
    def _assert_err(self, code, fn, *args, **kwargs):
        with self.assertRaises(o.SiliquantBridgeError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_primitives_and_label_guard(self):
        o.verify_no_floats({"a": [1, Fraction(1, 2), ("s", True, None)], "s": {1}, "fs": frozenset([2])})
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, 0.25)
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, 1 + 2j)
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, {1.5: "x"})
        self._assert_err("FLOAT_FORBIDDEN", o.verify_no_floats, [1, (2.0,)])

        self.assertEqual(o.require_non_empty_str(" ~ok\u0080 ", "f"), "~ok\u0080")
        self._assert_err("INVALID_TYPE", o.require_non_empty_str, 7, "f")
        self._assert_err("CONTROL_CHAR", o.require_non_empty_str, "a\x1fb", "f")
        self._assert_err("CONTROL_CHAR", o.require_non_empty_str, "a\x7fb", "f")
        self._assert_err("CONTROL_CHAR", o.require_non_empty_str, "a\u200bb", "f")
        self._assert_err("EMPTY_STRING", o.require_non_empty_str, "   ", "f")

        self.assertIs(o.require_exact_bool(True, "b"), True)
        self.assertIs(o.require_exact_bool(False, "b"), False)
        self._assert_err("INVALID_BOOL", o.require_exact_bool, 1, "b")

        self.assertEqual(o.require_exact_int(5, "x", 5, 5), 5)
        self._assert_err("INVALID_INT", o.require_exact_int, True, "x")
        self._assert_err("RANGE_ERROR", o.require_exact_int, 4, "x", min_value=5)
        self._assert_err("RANGE_ERROR", o.require_exact_int, 6, "x", max_value=5)

        self.assertEqual(o.require_exact_fraction(3, "q", Fraction(3, 1), Fraction(3, 1)), Fraction(3, 1))
        self.assertEqual(o.require_exact_fraction([2, 4], "q"), Fraction(1, 2))
        self.assertEqual(o.require_exact_fraction(Fraction(1, 2), "q", Fraction(1, 2), Fraction(1, 2)), Fraction(1, 2))
        self._assert_err("INVALID_FRACTION", o.require_exact_fraction, False, "q")
        self._assert_err("INVALID_FRACTION", o.require_exact_fraction, "1/2", "q")
        self._assert_err("INVALID_FRACTION", o.require_exact_fraction, "ab", "q")
        self._assert_err("INVALID_FRACTION", o.require_exact_fraction, [1, 2, 3], "q")
        self._assert_err("RANGE_ERROR", o.require_exact_fraction, [1, 0], "q")
        self._assert_err("RANGE_ERROR", o.require_exact_fraction, Fraction(1, 3), "q", min_value=Fraction(1, 2))
        self._assert_err("RANGE_ERROR", o.require_exact_fraction, Fraction(2, 3), "q", max_value=Fraction(1, 2))

        self.assertEqual(o.frac_str(Fraction(4, 6)), "2/3")
        self.assertEqual(o.format_exact_decimal(Fraction(3, 2), 0), "2")
        self.assertEqual(o.format_exact_decimal(Fraction(-3, 2), 0), "-2")
        self.assertEqual(o.format_exact_decimal(Fraction(-1, 3), 0), "0")
        self.assertEqual(o.format_exact_decimal(Fraction(1, 20), 1), "0.1")
        self.assertEqual(o.format_exact_decimal(Fraction(49, 1000), 1), "0.0")
        self.assertEqual(o.format_exact_decimal(Fraction(-1, 200), 2), "-0.01")
        self.assertEqual(o.format_exact_decimal(Fraction(-1, 10000), 2), "0.00")
        self.assertEqual(o.canonical_json_bytes({"b": 1, "a": 2}), b'{"a":2,"b":1}')
        self.assertEqual(o.canonical_sha256({"x": 1}), hashlib.sha256(b'{"x":1}').hexdigest())

        g = o.enforce_law5_label_hygiene(["Branchless_E8", "Roofline-v5e"])
        self.assertTrue(g["label_filter_ok"])
        self.assertEqual(g["guard_scope"], "label_filter_only_not_mathematical_proof")
        self._assert_err("INVALID_LABELS", o.enforce_law5_label_hygiene, [])
        self._assert_err("INVALID_LABELS", o.enforce_law5_label_hygiene, "abc")
        self._assert_err("EMPTY_STRING", o.enforce_law5_label_hygiene, ["---"])
        self._assert_err("DUPLICATE_LABEL", o.enforce_law5_label_hygiene, ["a-b", "A_B"])
        for bad in ("\uff4d\uff49\uff52\uff41\uff43\uff4c\uff45", "quantum_consciousness", "magical", "numerology", "telepathy"):
            self._assert_err("METAPHOR_FORBIDDEN", o.enforce_law5_label_hygiene, [bad])

    def test_pillar1_branchless_e8_and_issue16_zero_codeword(self):
        # Input validation
        self._assert_err("INVALID_VEC8", o.branchless_d8_decode_int, [0] * 7)
        self._assert_err("INVALID_VEC8", o.branchless_e8_decode_int, "bad")
        self._assert_err("RANGE_ERROR", o.branchless_e8_decode_int, [0] * 8, scale_shift=1)

        # Known-answer E8 checks: zero, half-integer coset, odd-parity D8 tie, non-member check,
        # AND SQ103-BYPASS-01 exact boundary extremes +-(1<<24) without shifted_in underflow!
        y0, d0 = o.branchless_e8_decode_int([0] * 8, 4)
        self.assertEqual(y0, [0] * 8)
        self.assertEqual(d0, 0)
        for extreme in (
            [-(1 << 24), 0, 0, 0, 0, 0, 0, 0],
            [1 << 24, 0, 0, 0, 0, 0, 0, 0],
            [-(1 << 24), 8, 8, 8, 8, 8, 8, 8],
        ):
            ye, de = o.branchless_e8_decode_int(extreme, 4)
            self.assertEqual(de, o.brute_force_e8_min_dist_sq(extreme, 4))
            self.assertTrue(o.is_valid_e8_point_scaled(ye, 4))
        self.assertTrue(o.is_valid_e8_point_scaled([0] * 8, 4))
        self.assertTrue(o.is_valid_e8_point_scaled([8] * 8, 4))       # (1/2)^8 in D8 + 1/2
        self.assertFalse(o.is_valid_e8_point_scaled([16, 0, 0, 0, 0, 0, 0, 0], 4))  # odd sum integer -> NOT E8!
        self.assertFalse(o.is_valid_e8_point_scaled([1, 0, 0, 0, 0, 0, 0, 0], 4))   # non-lattice point

        p1 = o.audit_branchless_e8_and_zero_codeword_exact()
        self.assertTrue(p1["passed"])
        self.assertEqual(p1["exact_brute_force_matches"], 48)
        self.assertEqual(p1["centre_zero_block_sq_err_exact"], "25/64")
        self.assertEqual(p1["shifted_63_zero_blocks_sq_err_exact"], "1575/64")

        # Must-fail / conjunct isolation on Pillar 1:
        # 1. Zero centre shift (centre_shift_scaled=[0]*8) -> centre_shift_harms_zeros False!
        p1_noc = o.audit_branchless_e8_and_zero_codeword_exact(centre_shift_scaled=[0] * 8)
        self.assertFalse(p1_noc["centre_shift_harms_zero_blocks"])
        self.assertFalse(p1_noc["passed"])
        # 2. Corrupt brute-force comparison -> all_branchless_optimal False!
        with mock.patch.object(o, "brute_force_e8_min_dist_sq", return_value=-1):
            p1_bad = o.audit_branchless_e8_and_zero_codeword_exact()
            self.assertFalse(p1_bad["all_branchless_optimal"])
            self.assertFalse(p1_bad["passed"])
        # 3. Corrupt membership check -> all_branchless_optimal False!
        with mock.patch.object(o, "is_valid_e8_point_scaled", return_value=False):
            p1_badm = o.audit_branchless_e8_and_zero_codeword_exact()
            self.assertFalse(p1_badm["all_branchless_optimal"])
            self.assertFalse(p1_badm["passed"])
        # 4. Isolate EVERY conjunct of zero_preserved & centre_shift_harms_zeros & passed!
        orig_dec = o.branchless_e8_decode_int
        def _bad_zero_vec(v, s=4):
            if v == [0] * 8:
                return [16, 16, 0, 0, 0, 0, 0, 0], 0
            return orig_dec(v, s)
        with mock.patch.object(o, "branchless_e8_decode_int", side_effect=_bad_zero_vec):
            p1_bzv = o.audit_branchless_e8_and_zero_codeword_exact()
            self.assertFalse(p1_bzv["zero_vector_preserved"])
            self.assertTrue(p1_bzv["centre_shift_harms_zero_blocks"])
            self.assertFalse(p1_bzv["passed"])
        def _bad_zero_dist(v, s=4):
            if v == [0] * 8:
                return [0] * 8, 1
            return orig_dec(v, s)
        with mock.patch.object(o, "branchless_e8_decode_int", side_effect=_bad_zero_dist):
            p1_bzd = o.audit_branchless_e8_and_zero_codeword_exact()
            self.assertFalse(p1_bzd["zero_vector_preserved"])
            self.assertFalse(p1_bzd["centre_shift_harms_zero_blocks"])
            self.assertFalse(p1_bzd["passed"])


class TestPillars2To5AndLiveDiagnostics(unittest.TestCase):
    def _assert_err(self, code, fn, *args, **kwargs):
        with self.assertRaises(o.SiliquantBridgeError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_pillar2_tpu_v5e_roofline(self):
        p2 = o.audit_tpu_v5e_roofline_exact()
        self.assertTrue(p2["passed"])
        self.assertIs(p2["measured_kernel_tok_s_present"], False)
        self.assertEqual(p2["format_ratios"]["ZBG128_b560s16"]["hbm_io_reduction_vs_fp16_exact"], "23/32")
        self.assertEqual(p2["format_ratios"]["ZBG128_b560s16"]["bandwidth_speedup_vs_q4km_exact"], "16/15")
        self.assertEqual(p2["format_ratios"]["E8_Had"]["hbm_io_reduction_vs_fp16_exact"], "1153/1600")
        self.assertEqual(p2["per_model_ceilings"]["Gemma-4-E4B"]["ZBG128_b560s16"]["decode_tok_s_ceiling_dec6"], "364.000000")

        # Guards
        self._assert_err("RANGE_ERROR", o.audit_tpu_v5e_roofline_exact, hbm_bytes_per_sec=0)
        self._assert_err("INVALID_FORMATS", o.audit_tpu_v5e_roofline_exact, formats={"FP16": 16})
        self._assert_err("INVALID_MODELS", o.audit_tpu_v5e_roofline_exact, models={})
        bad_f = dict(o.FORMAT_BPW, INT4=Fraction(0, 1))
        self._assert_err("RANGE_ERROR", o.audit_tpu_v5e_roofline_exact, formats=bad_f)
        bad_f2 = dict(o.FORMAT_BPW, INT4=Fraction(33, 1))
        self._assert_err("RANGE_ERROR", o.audit_tpu_v5e_roofline_exact, formats=bad_f2)
        # CLAUDE-SQ-01: reject whitespace-padded format/model keys that could override required entries!
        self._assert_err("INVALID_FORMATS", o.audit_tpu_v5e_roofline_exact, formats={**o.FORMAT_BPW, "Q4_K_M ": Fraction(5, 1)})
        self._assert_err("INVALID_FORMATS", o.audit_tpu_v5e_roofline_exact, formats={**o.FORMAT_BPW, " FP16": Fraction(1, 1)})
        self._assert_err("INVALID_MODELS", o.audit_tpu_v5e_roofline_exact, models={**o.MODEL_PARAMS, " Qwen2.5-1.5B": 1})

        # Pin EVERY conjunct in Pillar 2 individually (strictly isolating BW vs >70%!)
        f_zbg = dict(o.FORMAT_BPW, Q4_K_M=Fraction(9, 2))  # ZBG=4.5 == Q4_K_M fails BW, E8=4.47 < 4.5 passes, both >70%!
        r_zbg = o.audit_tpu_v5e_roofline_exact(formats=f_zbg)
        self.assertFalse(r_zbg["zbg_and_e8_beat_q4km_bandwidth"])
        self.assertTrue(r_zbg["hbm_reduction_exceeds_70pct"])
        self.assertFalse(r_zbg["passed"])
        f_e8 = dict(o.FORMAT_BPW, Q4_K_M=Fraction(447, 100), ZBG128_b560s16=Fraction(4, 1))  # ZBG<Q4, E8==Q4, both >70%!
        r_e8 = o.audit_tpu_v5e_roofline_exact(formats=f_e8)
        self.assertFalse(r_e8["zbg_and_e8_beat_q4km_bandwidth"])
        self.assertTrue(r_e8["hbm_reduction_exceeds_70pct"])
        self.assertFalse(r_e8["passed"])
        # Raise Q4_K_M to 6 so ZBG=5 < Q4_K_M passes bandwidth check, while 1 - 5/16 = 11/16 < 7/10 fails > 70%!
        f_70z = dict(o.FORMAT_BPW, Q4_K_M=Fraction(6, 1), ZBG128_b560s16=Fraction(5, 1))
        self.assertFalse(o.audit_tpu_v5e_roofline_exact(formats=f_70z)["passed"])
        f_70e = dict(o.FORMAT_BPW, Q4_K_M=Fraction(6, 1), E8_Had=Fraction(5, 1))
        self.assertFalse(o.audit_tpu_v5e_roofline_exact(formats=f_70e)["passed"])

    def test_pillar3_nature_to_siliquant_bridges(self):
        p3 = o.audit_nature_to_siliquant_bridges_exact()
        self.assertTrue(p3["passed"])
        self.assertEqual(p3["homeostatic_mult_ratio_max_err_exact"], "0/1")
        self.assertEqual(p3["proportional_clipping_droop_exact"], "1/16")
        self.assertTrue(p3["mst_blind_to_parallel_redundancy"])
        self.assertEqual(p3["reff_tree_exact"], "3/5")
        self.assertEqual(p3["reff_redundant_exact"], "61/327")
        self.assertEqual(p3["reff_redundant_after_cut_exact"], "73/336")

        # Guards on bipartite MST (incl. SQ103-BUG-04 zero-weight absent edge exclusion!), R_eff, solver & Pillar 3 params
        for fn in (o.bipartite_mst_weights_exact, o.bipartite_effective_resistance_exact):
            self._assert_err("INVALID_MATRIX", fn, [])
            self._assert_err("INVALID_MATRIX", fn, [[]])
            self._assert_err("INVALID_MATRIX", fn, [[1, 2], [3]])
        self.assertEqual(o.bipartite_mst_weights_exact([[0, 0], [0, 0]]), [])
        w_tree_cut = [[5, 5, 0], [0, 0, 5], [0, 0, 5]]
        self.assertEqual(o.bipartite_mst_weights_exact(w_tree_cut), [Fraction(5, 1)] * 4)
        # GPT6-SQ-02: R_eff on connected endpoints ignores unrelated isolated/disconnected components!
        self.assertEqual(o.bipartite_effective_resistance_exact([[1, 0], [0, 0]], row_u=0, col_v=0), Fraction(1, 1))
        self.assertEqual(o.bipartite_effective_resistance_exact([[1, 0], [0, 2]], row_u=0, col_v=0), Fraction(1, 1))
        with self.assertRaises(o.SiliquantBridgeError) as ctx_disc:
            o.bipartite_effective_resistance_exact([[1, 0], [0, 2]], 0, 1)
        self.assertEqual(ctx_disc.exception.code, "SINGULAR_MATRIX")
        self.assertIn("disconnected", str(ctx_disc.exception))
        # ZB4-SQ-02: _solve_linear_exact coerces int inputs to exact Fraction (never returns float!) and rejects float/bool
        sol_int = o._solve_linear_exact([[2, 0], [0, 2]], [1, 1])
        self.assertEqual(sol_int, [Fraction(1, 2), Fraction(1, 2)])
        self.assertTrue(all(type(v) is Fraction for v in sol_int))
        self._assert_err("FLOAT_FORBIDDEN", o._solve_linear_exact, [[2.0, 0], [0, 2]], [1, 1])
        self._assert_err("INVALID_FRACTION", o._solve_linear_exact, [[True, 0], [0, 2]], [1, 1])
        self._assert_err("SINGULAR_MATRIX", o._solve_linear_exact, [[1, 1], [1, 1]], [1, 1])
        for bad_kw in (
            {"homeo_gain": 0}, {"add_shift": 0}, {"step_u": 0}, {"kp_prop": 0}, {"ki_int": 0},
            {"w_base_override": [1, 2]}, {"w_base_override": [0, 1, 2, 3]},
            {"w_base_override": [Fraction(-1, 2), 1, 2, 3]},
        ):
            self._assert_err("INVALID_PARAMS", o.audit_nature_to_siliquant_bridges_exact, **bad_kw)

        # Pin EVERY single conjunct in Pillar 3 individually!
        # 3a-1: shift_ratio_max_err == 0 on constant w_base (mult_ratio_max_err == 0 still True)
        p3_const = o.audit_nature_to_siliquant_bridges_exact(w_base_override=[4, 4, 4, 4])
        self.assertFalse(p3_const["homeostatic_invariance_exact"])
        self.assertFalse(p3_const["passed"])
        # 3a-1: mult_ratio_max_err != 0 (shift_ratio_max_err > 0 still True)
        with mock.patch.object(o, "_homeostatic_scale_exact", side_effect=lambda w, g: [g * x + (1 if i == 1 else 0) for i, x in enumerate(w)]):
            p3_bm = o.audit_nature_to_siliquant_bridges_exact()
            self.assertFalse(p3_bm["homeostatic_invariance_exact"])
            self.assertFalse(p3_bm["passed"])
        # 3a-2: non-deadbeat integral gain ki_int=1 -> err_integral_deadbeat != 0 (prop_droop > 0 still True)
        p3_nondb = o.audit_nature_to_siliquant_bridges_exact(ki_int=Fraction(1, 1))
        self.assertFalse(p3_nondb["integral_beats_prop_droop"])
        self.assertFalse(p3_nondb["passed"])
        # 3a-2: prop_droop == 0 (err_integral_deadbeat == 0 still True)
        with mock.patch.object(o, "_proportional_droop_exact", return_value=Fraction(0, 1)):
            p3_zdr = o.audit_nature_to_siliquant_bridges_exact()
            self.assertFalse(p3_zdr["integral_beats_prop_droop"])
            self.assertFalse(p3_zdr["passed"])
        # 3b: MST, R_eff, bridge-cut conjuncts
        with mock.patch.object(o, "bipartite_mst_weights_exact", side_effect=[[Fraction(5)], [Fraction(4)]]):
            self.assertFalse(o.audit_nature_to_siliquant_bridges_exact()["passed"])
        with mock.patch.object(o, "bipartite_effective_resistance_exact", side_effect=[Fraction(1, 2), Fraction(1, 2), o.SiliquantBridgeError("SINGULAR_MATRIX", "disc"), Fraction(1, 1)]):
            self.assertFalse(o.audit_nature_to_siliquant_bridges_exact()["passed"])
        with mock.patch.object(o, "bipartite_effective_resistance_exact", side_effect=[Fraction(3, 5), Fraction(1, 5), Fraction(1, 1), Fraction(2, 5)]):
            self.assertFalse(o.audit_nature_to_siliquant_bridges_exact()["passed"])
        with mock.patch.object(o, "bipartite_effective_resistance_exact", side_effect=[Fraction(3, 5), Fraction(1, 5), o.SiliquantBridgeError("SINGULAR_MATRIX", "disc"), Fraction(1, 5)]):
            self.assertFalse(o.audit_nature_to_siliquant_bridges_exact()["passed"])

    def test_pillar4_bayt_al_hikma_bounds(self):
        p4 = o.audit_bayt_al_hikma_quant_bounds_exact()
        self.assertTrue(p4["passed"])
        self.assertEqual(p4["kappa_cancelling_exact"], "21/1")
        self.assertEqual(p4["amplification_ratio_exact"], "21/1")
        self.assertEqual(p4["kashi_contracted_err_exact"], "1/128")
        self.assertEqual(p4["kashi_divergent_err_exact"], "6561/128")

        for bad_kw in (
            {"perturb_eps": 0}, {"contraction_lip": 0}, {"contraction_lip": 1}, {"divergent_lip": 1},
            {"w_cancel_override": [1, 2]}, {"delta_cancel_override": [1, 2]},
            {"w_cancel_override": [10, -10, 0]},
        ):
            self._assert_err("INVALID_PARAMS", o.audit_bayt_al_hikma_quant_bounds_exact, **bad_kw)
        for bad_kw in ({"steps": 1}, {"steps": 41}):
            self._assert_err("RANGE_ERROR", o.audit_bayt_al_hikma_quant_bounds_exact, **bad_kw)

        # Pin EVERY single conjunct in Pillar 4 individually!
        # 4a-1: amplification_ratio != kappa_cancel / kappa_align (kappa_cancel > kappa_align still True)
        p4_amp = o.audit_bayt_al_hikma_quant_bounds_exact(delta_cancel_override=[Fraction(1, 4), Fraction(-1, 4), Fraction(1, 4)])
        self.assertFalse(p4_amp["biruni_amplification_exact"])
        self.assertFalse(p4_amp["passed"])
        # 4a-2: kappa_cancel == kappa_align (amplification_ratio == kappa_cancel/kappa_align == 1 still True)
        p4_kap = o.audit_bayt_al_hikma_quant_bounds_exact(w_cancel_override=[10, 10, 1])
        self.assertFalse(p4_kap["biruni_amplification_exact"])
        self.assertFalse(p4_kap["passed"])
        # 4b-1: final_err_c != expected_err_c (final_err_c < init_err still True, divergent still True)
        with mock.patch.object(o, "_kashi_iterate_exact", side_effect=[Fraction(2, 1), Fraction(10, 1)]):
            p4_kc1 = o.audit_bayt_al_hikma_quant_bounds_exact()
            self.assertFalse(p4_kc1["kashi_contraction_exact"])
            self.assertTrue(p4_kc1["kashi_divergence_control"])
            self.assertFalse(p4_kc1["passed"])
        # 4b-2: final_err_c == expected_err_c == 0 (not < init_err == 0), divergent still > 0!
        with mock.patch.object(o, "_kashi_iterate_exact", side_effect=[Fraction(1, 1), Fraction(2, 1)]):
            p4_kc2 = o.audit_bayt_al_hikma_quant_bounds_exact(init_s=Fraction(1, 1))
            self.assertFalse(p4_kc2["kashi_contraction_exact"])
            self.assertTrue(p4_kc2["kashi_divergence_control"])
            self.assertFalse(p4_kc2["passed"])
        # 4b-3: divergent fails (final_err_d <= init_err) while contraction passes!
        with mock.patch.object(o, "_kashi_iterate_exact", side_effect=[Fraction(1, 1) + Fraction(1, 128), Fraction(2, 1)]):
            p4_kd = o.audit_bayt_al_hikma_quant_bounds_exact()
            self.assertTrue(p4_kd["kashi_contraction_exact"])
            self.assertFalse(p4_kd["kashi_divergence_control"])
            self.assertFalse(p4_kd["passed"])

    def test_pillar5_repo_diag_certification_and_live_git_worktree(self):
        cert = a.run_audit(live=False)
        self.assertTrue(cert["all_gates_passed"])
        self.assertEqual(cert["verdict"], "AUDITED_SILIQUANT_FORCE_MULTIPLIER_BRIDGES")
        self.assertEqual(cert["mode"], "standalone_fixture_replay")
        self.assertIs(cert["live_repo_recounted"], False)
        body_no_sha = {k: v for k, v in cert.items() if k != "certificate_sha256"}
        self.assertEqual(cert["certificate_sha256"], o.canonical_sha256(body_no_sha))

        rd = cert["live_siliquant_repo_diagnostics"]
        p1 = cert["pillar1_branchless_e8_and_zero_codeword"]
        p2 = cert["pillar2_tpu_v5e_roofline_part_e"]
        p3 = cert["pillar3_nature_to_siliquant_bridges"]
        p4 = cert["pillar4_bayt_al_hikma_quant_bounds"]
        axes = dict(cert["five_axis_audit_checks"])

        # Repo diagnostics guards + single-flag conjunct pinning across all 8 booleans!
        self._assert_err("INVALID_DIAG", o.audit_siliquant_repo_diagnostics_exact, "bad")
        self._assert_err("INVALID_DIAG", o.audit_siliquant_repo_diagnostics_exact, {})
        fx = copy.deepcopy(a.STANDALONE_REPO_DIAG_FIXTURE)
        bad_src = copy.deepcopy(fx); bad_src["diag_source"] = "forged"
        self._assert_err("INVALID_DIAG", o.audit_siliquant_repo_diagnostics_exact, bad_src)
        for bad_sha in ("not_a_sha", "0" * 40, "f" * 40, "  " + fx["main_commit"] + "  "):
            bad_c = copy.deepcopy(fx); bad_c["main_commit"] = bad_sha
            self._assert_err("INVALID_COMMIT", o.audit_siliquant_repo_diagnostics_exact, bad_c)
        dup_c = copy.deepcopy(fx); dup_c["pr12_head_commit"] = dup_c["main_commit"]
        self._assert_err("DUPLICATE_COMMIT", o.audit_siliquant_repo_diagnostics_exact, dup_c)
        # GPT6-SQ-04: explicit lattice execution flag reflected in live_executed_test_suites
        fx_lat = copy.deepcopy(fx); fx_lat["pr12_lattice_tests_executed_in_live_check"] = True
        self.assertEqual(o.audit_siliquant_repo_diagnostics_exact(fx_lat)["live_executed_test_suites"], ["shape", "perm", "persist", "lattice"])
        self.assertEqual(rd["live_executed_test_suites"], ["shape", "perm", "persist"])
        self.assertIs(rd["pr12_lattice_tests_executed_in_live_check"], False)
        # GPT6-SQ-03: provenance binding rejects stamping live_repo_recounted=True on standalone fixture (and vice versa)!
        self._assert_err("PROVENANCE_MISMATCH", a.certify_from_repo_diag, "live_repo_recount", True, fx)
        fx_live_src = copy.deepcopy(fx); fx_live_src["diag_source"] = "live_git_inspection"
        self._assert_err("PROVENANCE_MISMATCH", a.certify_from_repo_diag, "standalone_fixture_replay", False, fx_live_src)
        for flag_k in (
            "pr12_shape_tests_passed", "pr12_perm_tests_passed", "pr12_persist_tests_passed",
            "pr12_design_out_has_smollm2_fix",
            "pr12_number_trace_exact_match", "pr12_claim_ledger_exact_match",
            "pr12_plan_before_data_exit_zero",
        ):
            bf = copy.deepcopy(fx); bf[flag_k] = False
            self.assertFalse(o.audit_siliquant_repo_diagnostics_exact(bf)["passed"])

        # Honest REJECTED_GATE_FAILED when repo_diag fails and axes honestly reflect it
        bf = copy.deepcopy(fx); bf["pr12_design_out_has_smollm2_fix"] = False
        rd_fail = o.audit_siliquant_repo_diagnostics_exact(bf)
        axes_fail = dict(axes, pr12_live_reproducibility=False)
        rej = o.certify_siliquant_bridge_audit_exact(
            "standalone_fixture_replay", False, a.OPERATOR_LABELS, rd_fail, p1, p2, p3, p4, axes_fail
        )
        self.assertFalse(rej["all_gates_passed"])
        self.assertEqual(rej["verdict"], "REJECTED_GATE_FAILED")

        # All Pillar 5 fail-closed guards
        self._assert_err("INVALID_MODE", o.certify_siliquant_bridge_audit_exact, "bad", False, a.OPERATOR_LABELS, rd, p1, p2, p3, p4, axes)
        self._assert_err("MODE_MISMATCH", o.certify_siliquant_bridge_audit_exact, "standalone_fixture_replay", True, a.OPERATOR_LABELS, rd, p1, p2, p3, p4, axes)
        self._assert_err("INVALID_AXES", o.certify_siliquant_bridge_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, rd, p1, p2, p3, p4, {"x": True})
        self._assert_err("INVALID_PILLAR", o.certify_siliquant_bridge_audit_exact, "standalone_fixture_replay", False, a.OPERATOR_LABELS, "bad", p1, p2, p3, p4, axes)
        for idx_p, bad_tuple in enumerate([
            (dict(rd, passed=False), p1, p2, p3, p4),
            (dict(rd, passed=1), p1, p2, p3, p4),                                      # ZB4-SQ-01 int 1 != True
            (rd, dict(p1, passed=False), p2, p3, p4),
            (rd, dict(p1, vectors_tested=Fraction(48, 1)), p2, p3, p4),                # ZB4-SQ-01 Fraction != int
            (rd, p1, dict(p2, passed=False), p3, p4),
            (rd, p1, dict(p2, measured_kernel_tok_s_present=0), p3, p4),               # ZB4-SQ-01 int 0 != False
            (rd, p1, p2, dict(p3, passed=False), p4),
            (rd, p1, p2, p3, dict(p4, passed=False)),
        ]):
            self._assert_err(
                "PILLAR_CONTRADICTION",
                o.certify_siliquant_bridge_audit_exact,
                "standalone_fixture_replay", False, a.OPERATOR_LABELS,
                bad_tuple[0], bad_tuple[1], bad_tuple[2], bad_tuple[3], bad_tuple[4], axes,
            )
        # Pin every branch & BoolOp(And) conjunct of _strict_equal directly
        self.assertFalse(o._strict_equal({"a": 1}, {"b": 1}))
        self.assertFalse(o._strict_equal({"a": 1}, {"a": 1, "b": 2}))
        self.assertFalse(o._strict_equal([1], [1, 2]))
        self.assertFalse(o._strict_equal([1], [True]))
        self.assertFalse(o._strict_equal(1, 2))
        self.assertTrue(o._strict_equal({"k": [1, True, "s"]}, {"k": [1, True, "s"]}))
        self._assert_err(
            "AXIS_CONTRADICTION",
            o.certify_siliquant_bridge_audit_exact,
            "standalone_fixture_replay", False, a.OPERATOR_LABELS,
            rd, p1, p2, p3, p4, dict(axes, tpu_v5e_part_e_roofline=False),
        )

        # Known-answer live git-repo test exercising compute_live_siliquant_diagnostics() end-to-end!
        with tempfile.TemporaryDirectory() as td:
            # Missing repo guard
            self._assert_err("REPO_MISSING", a.compute_live_siliquant_diagnostics, td)
            # Create minimal synthetic git repo with all 5 origin/* refs and passing checks!
            subprocess.run(["git", "init", "-q", td], check=True)
            subprocess.run(["git", "-C", td, "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", td, "config", "user.name", "Test"], check=True)
            for sub in ("shape", "perm", "persist", "lattice", "checks"):
                os.makedirs(os.path.join(td, sub))
            for sub in ("shape", "perm", "persist", "lattice"):
                with open(os.path.join(td, sub, "test_ok.py"), "w", encoding="utf-8") as fh:
                    fh.write("import unittest\nclass T(unittest.TestCase):\n    def test_x(self): self.assertEqual(1, 1)\n")
            with open(os.path.join(td, "shape", "design.py"), "w", encoding="utf-8") as fh:
                fh.write("print('design_ok')\n")
            with open(os.path.join(td, "shape", "design.out"), "w", encoding="utf-8") as fh:
                fh.write(
                    "EBG128:b560s16 | 4.4999 | -25.60 | 0.0200, 0.0195 | 0.0453, 0.0609 | 560 | 8017953 | 561.26 | - | -22.63 | +0.55\n"
                    "ZBG128:b560s16 | 4.4998 | -24.94 | 0.0232, 0.0227 | 0.0527, 0.0560 | 560 | 436149 | 30.53 | 8.7 | -19.35 | +0.82\n"
                    "EBG128:b504s8 | 3.9997 | -22.84 | 0.0377, 0.0371 | 0.0855, 0.1183 | 504 | 4369817 | 275.30 | 10.0 | -19.39 | +0.85\n"
                )
            for chk in ("number_trace", "claim_ledger", "plan_before_data"):
                with open(os.path.join(td, "checks", f"{chk}.py"), "w", encoding="utf-8") as fh:
                    fh.write(f"print('{chk}_ok')\n")
                with open(os.path.join(td, "checks", f"{chk}.out"), "w", encoding="utf-8") as fh:
                    fh.write(f"{chk}_ok\n")
            subprocess.run(["git", "-C", td, "add", "."], check=True)
            subprocess.run(["git", "-C", td, "commit", "-q", "-m", "pr12"], check=True)
            pr12_sha = subprocess.run(["git", "-C", td, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
            # Missing ref guard before creating refs
            self._assert_err("GIT_REF_MISSING", a.compute_live_siliquant_diagnostics, td)
            tree_sha = subprocess.check_output(
                ["git", "-C", td, "rev-parse", "HEAD^{tree}"], text=True
            ).strip()
            base_sha = subprocess.check_output(
                ["git", "-C", td, "commit-tree", tree_sha, "-m", "base snapshot"], text=True
            ).strip()
            main_sha = subprocess.check_output(
                ["git", "-C", td, "commit-tree", tree_sha, "-p", base_sha, "-p", pr12_sha,
                 "-m", "Merge pull request #12 (reviewed snapshot)"],
                text=True,
            ).strip()
            refs = (
                "refs/remotes/origin/main",
                "refs/remotes/origin/expedition/leech-l1-hadamard-symbiosis",
                "refs/remotes/origin/feat/magnitude-autotune-quant-bridge",
                "refs/remotes/origin/feat/round8-inkling-loom-speculative-tomography",
            )
            subprocess.run(["git", "-C", td, "update-ref", refs[0], main_sha], check=True)
            ref_shas = [pr12_sha, main_sha]
            for idx_r, ref in enumerate(refs[1:], start=2):
                subprocess.run(["git", "-C", td, "commit", "--allow-empty", "-q", "-m", f"ref{idx_r}"], check=True)
                rsha = subprocess.run(["git", "-C", td, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
                subprocess.run(["git", "-C", td, "update-ref", ref, rsha], check=True)
                ref_shas.append(rsha)
            live_c = a.run_audit(live=True, siliquant_repo=td)
            self.assertTrue(live_c["all_gates_passed"])
            self.assertIs(live_c["live_repo_recounted"], True)
            live_d = live_c["live_siliquant_repo_diagnostics"]
            self.assertEqual(live_d["main_commit"], ref_shas[1])
            self.assertEqual(live_d["pr12_head_commit"], pr12_sha)
            self.assertTrue(live_d["pr12_design_out_has_smollm2_fix"])
            self.assertTrue(live_d["pr12_plan_before_data_exit_zero"])

            # Clone & checkout failure guards (CLAUDE-SQ-05 shared-clone read-only isolation)
            orig_run = subprocess.run
            def _fail_clone(cmd, *args, **kwargs):
                if isinstance(cmd, list) and "clone" in cmd:
                    return subprocess.CompletedProcess(cmd, 1, "", "clone_boom")
                return orig_run(cmd, *args, **kwargs)
            with mock.patch.object(subprocess, "run", side_effect=_fail_clone):
                self._assert_err("CLONE_FAILED", a.compute_live_siliquant_diagnostics, td)
            def _fail_co(cmd, *args, **kwargs):
                if isinstance(cmd, list) and "checkout" in cmd:
                    return subprocess.CompletedProcess(cmd, 1, "", "co_boom")
                return orig_run(cmd, *args, **kwargs)
            with mock.patch.object(subprocess, "run", side_effect=_fail_co):
                self._assert_err("CHECKOUT_FAILED", a.compute_live_siliquant_diagnostics, td)

            # Pin EVERY conjunct & guard in siliquant_bridge_audit.py (SQ103-BYPASS-03 + GPT6-SQ-01 + CLAUDE-SQ-02!)
            # 1. _design_out_has_smollm2_fix: exact 11-column rows + uniqueness + exact decimal & KL token matching
            full_design = (
                "EBG128:b560s16 | 4.4999 | -25.60 | 0.0200, 0.0195 | 0.0453, 0.0609 | 560 | 8017953 | 561.26 | - | -22.63 | +0.55\n"
                "ZBG128:b560s16 | 4.4998 | -24.94 | 0.0232, 0.0227 | 0.0527, 0.0560 | 560 | 436149 | 30.53 | 8.7 | -19.35 | +0.82\n"
                "EBG128:b504s8 | 3.9997 | -22.84 | 0.0377, 0.0371 | 0.0855, 0.1183 | 504 | 4369817 | 275.30 | 10.0 | -19.39 | +0.85\n"
            )
            self.assertTrue(a._design_out_has_smollm2_fix(full_design))
            swapped_design = (
                "EBG128:b560s16 | 4.4999 | -25.60 | 0.0200, 0.0195 | 0.0453, 0.1183 | 560 | 8017953 | 561.26 | - | -22.63 | +0.55\n"
                "ZBG128:b560s16 | 4.4998 | -24.94 | 0.0232, 0.0227 | 0.0527, 0.0609 | 560 | 436149 | 30.53 | 8.7 | -19.35 | +0.82\n"
                "EBG128:b504s8 | 3.9997 | -22.84 | 0.0377, 0.0371 | 0.0855, 0.0560 | 504 | 4369817 | 275.30 | 10.0 | -19.39 | +0.85\n"
            )
            self.assertFalse(a._design_out_has_smollm2_fix(swapped_design))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.0609", "10.06090")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.0560", "10.05600")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.1183", "10.11830")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.0453, 0.0609", "0.0609, -")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.0453, 0.0609", "9, 0.0609")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.0453, 0.0609", "0.0453, 0.0609, 0.0")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("4.4999", "STALE")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("| +0.55", "| +0.55 | EXTRA")))
            self.assertFalse(a._design_out_has_smollm2_fix(full_design.replace("0.0200, 0.0195 | 0.0453, 0.0609", "0.0453, 0.0609")))
            dup_row = full_design + "EBG128:b560s16 | 4.4999 | -25.60 | 0.0200, 0.0195 | 0.0453, 0.0609 | 560 | 8017953 | 561.26 | - | -22.63 | +0.55\n"
            self.assertFalse(a._design_out_has_smollm2_fix(dup_row))
            dup_stale = full_design + "EBG128:b560s16 | 4.4999 | -25.60 | 0.0200, 0.0195 | 0.0453, - | 560 | 8017953 | 561.26 | - | -22.63 | +0.55\n"
            self.assertFalse(a._design_out_has_smollm2_fix(dup_stale))
            wrong_lbl = full_design.replace("EBG128:b560s16", "OTHER:b560s16")
            self.assertFalse(a._design_out_has_smollm2_fix(wrong_lbl))
            wrong_kl = full_design.replace("0.0609", "0.9999")
            self.assertFalse(a._design_out_has_smollm2_fix(wrong_kl))
            # 2. _verify_pr12_worktree: isolate nt_ok vs nt_out==expected AND cl_ok vs cl_out==expected!
            with mock.patch.object(a, "_run_cmd_ok", side_effect=[(True, ""), (True, ""), (True, ""), (False, "number_trace_ok\n"), (True, "claim_ledger_ok\n"), (True, "")]):
                c1 = a._verify_pr12_worktree(td)
                self.assertFalse(c1["pr12_number_trace_exact_match"])
                self.assertTrue(c1["pr12_claim_ledger_exact_match"])
            with mock.patch.object(a, "_run_cmd_ok", side_effect=[(True, ""), (True, ""), (True, ""), (True, "WRONG\n"), (True, "claim_ledger_ok\n"), (True, "")]):
                c2 = a._verify_pr12_worktree(td)
                self.assertFalse(c2["pr12_number_trace_exact_match"])
                self.assertTrue(c2["pr12_claim_ledger_exact_match"])
            with mock.patch.object(a, "_run_cmd_ok", side_effect=[(True, ""), (True, ""), (True, ""), (True, "number_trace_ok\n"), (False, "claim_ledger_ok\n"), (True, "")]):
                c3 = a._verify_pr12_worktree(td)
                self.assertTrue(c3["pr12_number_trace_exact_match"])
                self.assertFalse(c3["pr12_claim_ledger_exact_match"])
            with mock.patch.object(a, "_run_cmd_ok", side_effect=[(True, ""), (True, ""), (True, ""), (True, "number_trace_ok\n"), (True, "WRONG\n"), (True, "")]):
                c4 = a._verify_pr12_worktree(td)
                self.assertTrue(c4["pr12_number_trace_exact_match"])
                self.assertFalse(c4["pr12_claim_ledger_exact_match"])
            # ZB4-SQ-03: empty committed checks/*.out must fail even if script prints "" and exits 0!
            nt_p = os.path.join(td, "checks", "number_trace.out")
            cl_p = os.path.join(td, "checks", "claim_ledger.out")
            with open(nt_p, "w", encoding="utf-8") as fh:
                fh.write("   \n")
            with mock.patch.object(a, "_run_cmd_ok", side_effect=[(True, ""), (True, ""), (True, ""), (True, "   \n"), (True, "claim_ledger_ok\n"), (True, "")]):
                c5 = a._verify_pr12_worktree(td)
                self.assertFalse(c5["pr12_number_trace_exact_match"])
                self.assertTrue(c5["pr12_claim_ledger_exact_match"])
            with open(nt_p, "w", encoding="utf-8") as fh:
                fh.write("number_trace_ok\n")
            with open(cl_p, "w", encoding="utf-8") as fh:
                fh.write("")
            with mock.patch.object(a, "_run_cmd_ok", side_effect=[(True, ""), (True, ""), (True, ""), (True, "number_trace_ok\n"), (True, ""), (True, "")]):
                c6 = a._verify_pr12_worktree(td)
                self.assertTrue(c6["pr12_number_trace_exact_match"])
                self.assertFalse(c6["pr12_claim_ledger_exact_match"])
            with open(cl_p, "w", encoding="utf-8") as fh:
                fh.write("claim_ledger_ok\n")
            # 3. run_audit branch conjuncts (not live and siliquant_repo is None)
            live_fx = copy.deepcopy(a.STANDALONE_REPO_DIAG_FIXTURE)
            live_fx["diag_source"] = "live_git_inspection"
            with mock.patch.object(a, "compute_live_siliquant_diagnostics", return_value=copy.deepcopy(live_fx)) as m_cl:
                c_repo = a.run_audit(live=False, siliquant_repo=td)
                self.assertIs(c_repo["live_repo_recounted"], True)
                m_cl.assert_called_once_with(td)
            with mock.patch.object(a, "compute_live_siliquant_diagnostics", return_value=copy.deepcopy(live_fx)) as m_cl2:
                c_live = a.run_audit(live=True, siliquant_repo=None)
                self.assertIs(c_live["live_repo_recounted"], True)
                m_cl2.assert_called_once_with(a.default_siliquant_repo())

        # CLI & default_siliquant_repo checks
        self.assertTrue(a.default_siliquant_repo().endswith("siliquant"))
        with tempfile.TemporaryDirectory() as td:
            out_p = os.path.join(td, "cert.json")
            buf = io.StringIO()
            with mock.patch.object(sys, "stdout", buf):
                rc = a.main(["--standalone", "--output", out_p])
            self.assertEqual(rc, 0)
            with open(out_p, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), buf.getvalue())
        with mock.patch.object(a, "run_audit", return_value={"all_gates_passed": False}), mock.patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(a.main(["--standalone"]), 1)
        with mock.patch.object(a, "run_audit", return_value={"all_gates_passed": True}) as m_ra, mock.patch.object(sys, "stdout", io.StringIO()):
            self.assertEqual(a.main(["--live"]), 0)
            m_ra.assert_called_once_with(live=True, siliquant_repo=None)


def run_all():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=0, failfast=True).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("ALL_AUDIT103_SILIQUANT_BRIDGE_TESTS_PASSED")


if __name__ == "__main__":
    run_all()
