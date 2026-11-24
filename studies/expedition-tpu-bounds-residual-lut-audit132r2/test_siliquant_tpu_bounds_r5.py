#!/usr/bin/env python3
"""Track S32 R5 Test Suite: 5 Pillars, Exported-Helper Soundness, Mutation/BoolOp Killers & Zone-B Security.

Pure Python 3 standard library: zero floats, zero bare asserts, zero RNG.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import siliquant_tpu_bounds_oracle as oracle
import siliquant_tpu_bounds_audit as audit

Err = oracle.SiliquantTpuBoundsError


class _SubInt(int):
    pass


class _SubStr(str):
    pass


class _SubFrac(Fraction):
    pass


class _SubPath(type(Path("."))):
    pass


class TestTrackS32SiliquantTpuBoundsR5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.build_standalone_certificate()
        cls.payload = cls.cert["payload"]

    # -----------------------------------------------------------------------
    # 1. Five Mathematical & Hardware Pillars
    # -----------------------------------------------------------------------

    def test_01_pillar1_precast_overflow_and_accumulator(self) -> None:
        self.assertEqual(self.payload["novelty_tier"], "N1")
        self.assertEqual(self.payload["oracle_class"], "exact")
        self.assertIs(self.payload["measured_pallas_tok_s"], False)
        self.assertIs(self.payload["measured_live_model_kl_residual_lut"], False)
        self.assertIs(self.payload["measured_live_model_safety_benchmarks"], False)
        self.assertIs(self.payload["zero_trust_path_floats"], True)
        self.assertIs(self.payload["zero_bare_asserts"], True)
        self.assertIs(self.payload["zero_rng"], True)

        p1 = self.payload["pillar1_precast_overflow_separation"]
        self.assertIs(p1["h8_all_rows_preserve_mod4"], True)
        self.assertIs(p1["int8_separation_block_h2_vs_wht8_136b_certified"], True)
        self.assertIs(p1["bf16_half_integers_neg128_to_pos128_all_513_exact"], True)
        self.assertIs(p1["bf16_boundary_257_inexact_verified"], True)

        expected = {
            "EBG32:b88s8": (48, 360605761, 29, 19, 52, 26, True, True, 7039, 14078, 108, 13716, True, False),
            "EBG32:b136s8": (391, 1525056448801, 41, 55, 156, 78, False, True, 20090, 40180, 316, 40132, False, True),
            "1block_E8_3.0bpw": (18, 7659601, 23, 12, 32, 16, True, True, 4310, 8621, 66, 8382, True, False),
            "1block_E8_4.5bpw": (151, 34179445441, 35, 34, 96, 48, True, True, 12484, 24969, 196, 24892, True, False),
        }
        for name, (mm, w1, bw, mz, mw, mb, ws, bs, cs8, cs32, l1, d32, s16, o16) in expected.items():
            rec = p1["regimes"][name]
            self.assertEqual(rec["m_max"], mm)
            self.assertEqual(rec["w1_count"], w1)
            self.assertEqual(rec["w1_bit_width"], bw)
            self.assertEqual(rec["max_abs_z"], mz)
            self.assertIs(rec["max_abs_z_tight_certified"], True)
            self.assertIs(rec["max_abs_z_int8_safe"], True)
            self.assertEqual(rec["wht8_max"], mw)
            self.assertIs(rec["wht8_tight_certified"], True)
            self.assertIs(rec["wht8_int8_safe"], ws)
            self.assertEqual(rec["block_h2_max"], mb)
            self.assertIs(rec["block_h2_tight_certified"], True)
            self.assertIs(rec["block_h2_int8_safe"], bs)
            self.assertEqual(rec["cs_dot_8w_int8_sym127"], cs8)
            self.assertEqual(rec["cs_dot_32w_int8_sym127"], cs32)
            self.assertEqual(rec["exact_l1_32w_attainable"], l1)
            self.assertEqual(rec["exact_dot_32w_int8_sym127"], d32)
            self.assertIs(rec["exact_dot_32w_witness_verified"], True)
            self.assertIs(rec["dot_32w_int16_safe_sym127"], s16)
            self.assertIs(rec["dot_32w_int16_overflow_attainable"], o16)

        acc = p1["accumulator_width_k16384"]
        self.assertEqual(acc["hidden_dim_k"], 16384)
        self.assertEqual(acc["superblocks_32w"], 512)
        self.assertEqual(acc["peak_136b_cs_sym127"], 20572160)
        self.assertEqual(acc["peak_136b_attainable_sym127"], 20547584)
        self.assertEqual(acc["int32_headroom_cs_2dp"], "104.38")
        self.assertEqual(acc["int32_headroom_attainable_2dp"], "104.51")
        self.assertEqual(acc["int32_headroom_asym128_2dp"], "103.69")
        self.assertIs(acc["headroom_exceeds_100x_all_3"], True)

    def test_02_pillar2_wcet_vmem_and_tpu_roofline(self) -> None:
        p2 = self.payload["pillar2_wcet_vmem_tpu_roofline"]
        self.assertEqual(p2["tpu_v5e_vmem_bytes"], 134217728)
        vr = p2["vmem_residency"]
        self.assertEqual(vr["tree_88b"]["vmem_fraction"], "2745/67108864")
        self.assertEqual(vr["tree_136b"]["vmem_fraction"], "28425/67108864")
        self.assertEqual(vr["guide_a_88b"]["vmem_fraction"], "2921/4194304")
        self.assertEqual(vr["guide_a_136b"]["vmem_fraction"], "6263/524288")
        self.assertEqual(vr["flat_lut_12b_32kib"]["vmem_fraction"], "1/4096")
        self.assertEqual(vr["flat_lut_16b_512kib"]["vmem_fraction"], "1/256")
        self.assertIs(p2["guide_a_136b_leaves_gt_98_8_pct_free"], True)

        wb = p2["wcet_bounds"]
        self.assertEqual(wb["ebg32_b88s8"]["fixed_select_steps"], 6)
        self.assertIs(wb["ebg32_b88s8"]["fits_uint32_vpu"], True)
        self.assertEqual(wb["ebg32_b136s8"]["fixed_select_steps"], 9)
        self.assertIs(wb["ebg32_b136s8"]["fits_uint32_vpu"], False)
        self.assertIs(wb["ebg32_b136s8"]["fits_uint64_emulated"], True)
        self.assertEqual(wb["residual_2stage_flat_lut_e8"]["fixed_vmem_gathers"], 2)
        self.assertEqual(wb["residual_2stage_flat_lut_e8"]["fixed_vector_fmas"], 1)
        self.assertIs(wb["residual_2stage_flat_lut_e8"]["o1_wcet_certified"], True)

        roof = p2["tpu_v5e_three_way_roofline"]
        self.assertEqual(roof["b_bf16_ridge_4_50"], "24625/364")
        self.assertEqual(roof["b_bf16_ridge_4_80"], "19700/273")
        self.assertEqual(roof["b_int8_ridge_4_50"], "24625/182")
        self.assertEqual(roof["b_int8_ridge_4_80"], "39400/273")
        self.assertEqual(roof["b_int8_ridge_4_50_2dp"], "135.30")
        self.assertEqual(roof["b_int8_ridge_4_80_2dp"], "144.32")
        self.assertEqual(roof["r_decode_breakeven_gwps"], "1365")
        self.assertEqual(roof["r_hbm_sat_4_50_gwps"], "1456")
        self.assertEqual(roof["batch_73_bf16_bandwidth_gain"], "0")
        self.assertEqual(roof["batch_73_int8_bandwidth_gain"], "1/15")
        self.assertIs(roof["int8_doubles_bandwidth_regime"], True)
        self.assertIs(roof["batch_73_bf16_zero_gain_certified"], True)
        self.assertIs(roof["batch_73_int8_still_bandwidth_bound"], True)

    def test_03_pillar3_residual_flat_lut_e8_codec(self) -> None:
        p3 = self.payload["pillar3_residual_flat_lut_e8_codec"]
        shells = p3["eisenstein_e4_shells_m0_to_8"]
        self.assertEqual(len(shells), 9)
        self.assertEqual([s["r_e8"] for s in shells], [1, 240, 2160, 6720, 17520, 30240, 60480, 82560, 140400])
        self.assertEqual(shells[2]["cumulative"], 2401)
        self.assertEqual(shells[5]["cumulative"], 56881)
        self.assertEqual(shells[8]["cumulative"], 340321)

        l12 = p3["lut_12bit_32kib"]
        self.assertEqual(l12["enumerated_points"], 2401)
        self.assertEqual(l12["slack_slots"], 1695)
        self.assertEqual(l12["bytes_int8"], 32768)
        self.assertEqual(l12["max_abs_z"], 4)
        self.assertIs(l12["fits_signed_int4"], True)

        l16 = p3["lut_16bit_512kib"]
        self.assertEqual(l16["points"], 56881)
        self.assertEqual(l16["slack_slots"], 8655)
        self.assertEqual(l16["bytes_int8"], 524288)
        self.assertEqual(l16["max_abs_z"], 6)
        self.assertIs(l16["max_abs_z_tight_certified"], True)

        nc = p3["unconditional_noclip_contraction_theorem"]
        self.assertIs(nc["m2_noclip_verified"], True)
        self.assertEqual(nc["m2_noclip_worst_contraction"], "25/49")
        self.assertIs(nc["m5_noclip_sf_half_verified"], True)
        self.assertEqual(nc["m5_noclip_sf_half_contraction"], "1/4")
        self.assertIs(nc["m5_noclip_sf_5_12_verified"], True)
        self.assertEqual(nc["m5_noclip_sf_5_12_contraction"], "25/144")

        self.assertEqual(len(p3["codec_distortion_suite"]), 6)
        self.assertIs(p3["all_encode_decode_zero_mismatch"], True)
        self.assertIs(p3["all_6_opt_ratios_le_1_4"], True)
        self.assertIs(p3["aggregate_fixed_sf_1_3_lt_1_4"], True)
        self.assertIs(p3["aggregate_opt_ratio_le_1_8"], True)

    def test_04_pillar4_spectral_lipschitz_and_anisotropy(self) -> None:
        p4 = self.payload["pillar4_spectral_lipschitz_enclosure"]
        for key in (
            "h8_orthogonal_8I",
            "block_h2_orthogonal_2I",
            "single_stage_zero_unpack_residual_zero",
            "single_stage_bareiss_spectral_pd",
            "residual_2stage_exact_cross_identity",
            "residual_2stage_cauchy_schwarz_holds",
            "residual_2stage_direct_le_ceiling",
            "residual_2stage_bareiss_spectral_pd",
        ):
            self.assertIs(p4[key], True)
        an = p4["anisotropy_operator_separation"]
        self.assertIs(an["isotropic_trace_invariance_exact"], True)
        self.assertIs(an["block_h2_exact_2x_flip"], True)
        self.assertIs(an["h8_exact_8x_flip"], True)
        self.assertEqual(an["axis_kl_e8"], "1/4")
        self.assertEqual(an["axis_kl_block_h2"], "1/8")
        self.assertEqual(an["axis_kl_h8"], "1/32")
        self.assertEqual(an["pair_diag_kl_block_h2"], "1/4")
        self.assertEqual(an["full_dc_kl_h8"], "1/4")

    def test_05_pillar5_safety_refusal_margin(self) -> None:
        p5 = self.payload["pillar5_safety_refusal_margin_preservation"]
        self.assertEqual(p5["baseline_margin"], "272/25")
        self.assertEqual(p5["scalar_margin"], "-32")
        self.assertEqual(p5["residual_e8_margin"], "10")
        self.assertEqual(p5["mse_reduction_rho"], "19/6424")
        for key in (
            "scalar_breaks_refusal_boundary",
            "residual_e8_preserves_refusal_boundary",
            "sqrt_rho_ceiling_tightening_exact",
            "observed_drift_sq_within_worst_ceiling",
            "rank1_adversary_tightness_exact",
            "isotropic_variance_identity_exact",
            "dw_bareiss_spectral_pd",
            "token_embd_immutable_row_hash_required",
        ):
            self.assertIs(p5[key], True)
        self.assertIs(p5["token_embd_quantized_by_block_lut"], False)

    # -----------------------------------------------------------------------
    # 2. Exported-Helper Soundness, Boundary & BoolOp Killers
    # -----------------------------------------------------------------------

    def test_06_primitive_parsers_ascii_and_exact_type_guards(self) -> None:
        # require_exact_int exact boundaries & bool/subclass rejection
        self.assertEqual(oracle.require_exact_int(5, "x", min_val=5, max_val=5), 5)
        for bad in (True, False, int(5) / int(1), "5", None, _SubInt(5)):
            with self.assertRaises(Err):
                oracle.require_exact_int(bad, "x")
        with self.assertRaises(Err):
            oracle.require_exact_int(4, "x", min_val=5)
        with self.assertRaises(Err):
            oracle.require_exact_int(6, "x", max_val=5)

        # require_exact_bool & require_exact_str & lone surrogate rejection
        self.assertIs(oracle.require_exact_bool(True, "b"), True)
        with self.assertRaises(Err):
            oracle.require_exact_bool(1, "b")
        self.assertEqual(oracle.require_exact_str("", "s", non_empty=False), "")
        with self.assertRaises(Err):
            oracle.require_exact_str("", "s", non_empty=True)
        with self.assertRaises(Err):
            oracle.require_exact_str(_SubStr("a"), "s")
        with self.assertRaises(Err):
            oracle.require_exact_str("bad\ud800surrogate", "s")

        # require_exact_fraction rejects raw int, bool, float, subclass, and enforces sign bounds
        self.assertEqual(oracle.require_exact_fraction(Fraction(0, 1), "q", non_negative=True), Fraction(0, 1))
        self.assertEqual(oracle.require_exact_fraction(Fraction(1, 2), "q", positive=True), Fraction(1, 2))
        for bad in (1, 10 ** 17, True, int(1) / int(2), "1/2", None, _SubFrac(1, 2)):
            with self.assertRaises(Err):
                oracle.require_exact_fraction(bad, "q")
        with self.assertRaises(Err):
            oracle.require_exact_fraction(Fraction(0, 1), "q", positive=True)
        with self.assertRaises(Err):
            oracle.require_exact_fraction(Fraction(-1, 3), "q", non_negative=True)

        # parse_exact_int_ascii: ASCII-only digits, boundary MAX_INT_DIGITS, Unicode rejection
        self.assertEqual(oracle.parse_exact_int_ascii("0", "n"), 0)
        self.assertEqual(oracle.parse_exact_int_ascii("-42", "n"), -42)
        self.assertEqual(oracle.parse_exact_int_ascii("9" * oracle.MAX_INT_DIGITS, "n"), int("9" * oracle.MAX_INT_DIGITS))
        for bad in ("", "-", "-0", "00", "01", "+5", " 5", "1_0", "\u00b2", "\u0661\u0662", "\u2460", "9" * (oracle.MAX_INT_DIGITS + 1)):
            with self.assertRaises(Err):
                oracle.parse_exact_int_ascii(bad, "n")

        # parse_exact_rational & formatters
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
        self.assertEqual(oracle.format_decimal_trunc(Fraction(0, 1), 2), "0.00")
        self.assertEqual(oracle.format_decimal_trunc(Fraction(1, 3), 18), "0.333333333333333333")
        with self.assertRaises(Err):
            oracle.format_decimal_trunc(Fraction(1, 2), 19)

        # Bounded-integer cap (MAX_INT_BITS == 3322 boundary; no raw ValueError on 10**4500)
        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        too_big = 1 << oracle.MAX_INT_BITS
        self.assertEqual(oracle.format_exact_rational(Fraction(max_ok, 1)), str(max_ok))
        self.assertEqual(oracle.format_decimal_trunc(Fraction(max_ok, 1), 0), str(max_ok))
        for bad_q in (
            Fraction(too_big, 1),
            Fraction(1, too_big),
            Fraction(10 ** 4500, 1),
            Fraction(1, 10 ** 4500),
        ):
            with self.assertRaises(Err) as cm_r:
                oracle.format_exact_rational(bad_q)
            self.assertEqual(cm_r.exception.code, "INT_TOO_LARGE")
            with self.assertRaises(Err) as cm_d:
                oracle.format_decimal_trunc(bad_q, 2)
            self.assertEqual(cm_d.exception.code, "INT_TOO_LARGE")

    def test_07_json_canonicalization_and_strict_parser(self) -> None:
        raw = oracle.canonical_json_bytes({"b": [1, True, None, "x"], "a": 0})
        self.assertEqual(raw, b'{"a":0,"b":[1,true,null,"x"]}')
        with self.assertRaises(Err):
            oracle.canonical_json_bytes({"f": Fraction(1, 2)})
        # Bounded int boundary in canonical_json_bytes and parse_strict_json_text (no raw ValueError!)
        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        self.assertIn(str(max_ok).encode("ascii"), oracle.canonical_json_bytes({"ok": max_ok}))
        for huge in (1 << oracle.MAX_INT_BITS, -(1 << oracle.MAX_INT_BITS), 10 ** 4300):
            with self.assertRaises(Err) as cm_hi:
                oracle.canonical_json_bytes({"big": huge})
            self.assertEqual(cm_hi.exception.code, "INT_TOO_LARGE")
        for huge_json in ('{"big":' + "9" * 4200 + "}", '{"big":' + "9" * 4500 + "}"):
            with self.assertRaises(Err) as cm_hj:
                oracle.parse_strict_json_text(huge_json, require_object=True)
            self.assertEqual(cm_hj.exception.code, "INT_TOKEN_TOO_LONG")
        # Depth boundary: 32 nested lists OK, 33 rejected
        nested_ok: object = 0
        for _ in range(oracle.MAX_JSON_DEPTH):
            nested_ok = [nested_ok]
        oracle.canonical_json_bytes(nested_ok)
        with self.assertRaises(Err):
            oracle.canonical_json_bytes([nested_ok])
        # Size boundary: exact MAX_JSON_BYTES OK, MAX_JSON_BYTES+1 rejected
        exact_payload = {"s": "a" * (oracle.MAX_JSON_BYTES - 8)}
        exact_raw = oracle.canonical_json_bytes(exact_payload)
        self.assertEqual(len(exact_raw), oracle.MAX_JSON_BYTES)
        self.assertEqual(oracle.parse_strict_json_text(exact_raw.decode("ascii"), require_object=True), exact_payload)
        with self.assertRaises(Err) as cm_j1:
            oracle.canonical_json_bytes({"s": "a" * oracle.MAX_JSON_BYTES})
        self.assertEqual(cm_j1.exception.code, "JSON_TOO_LARGE")
        with self.assertRaises(Err) as cm_j2:
            oracle.parse_strict_json_text('{"s":"' + "a" * oracle.MAX_JSON_BYTES + '"}')
        self.assertEqual(cm_j2.exception.code, "JSON_TOO_LARGE")
        # Strict JSON parser rejections: float, NaN, duplicate keys, non-object, malformed
        for bad_json in ('{"a":1.5}', '{"a":NaN}', '{"a":1,"a":2}', "[1,2]", "{bad"):
            with self.assertRaises(Err):
                oracle.parse_strict_json_text(bad_json, require_object=True)
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])

    def test_08_lattice_codec_bf16_and_bareiss_helpers(self) -> None:
        # is_in_doubled_e8 & doubled_e8_shell (including sum == 0 mod 4 with mixed parity!)
        self.assertIs(oracle.is_in_doubled_e8((2, 2, 0, 0, 0, 0, 0, 0)), True)
        self.assertIs(oracle.is_in_doubled_e8((2, 1, 1, 0, 0, 0, 0, 0)), False)
        self.assertIs(oracle.is_in_doubled_e8((2, 0, 0, 0, 0, 0, 0, 0)), False)
        self.assertEqual(oracle.doubled_e8_shell((2, 2, 0, 0, 0, 0, 0, 0)), 1)
        with self.assertRaises(Err) as cm_2e8:
            oracle.doubled_e8_shell((2, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(cm_2e8.exception.code, "NOT_IN_2E8")
        with self.assertRaises(Err) as cm_vlen:
            oracle.is_in_doubled_e8((0, 0, 0, 0))
        self.assertEqual(cm_vlen.exception.code, "INVALID_VEC8")
        with self.assertRaises(Err) as cm_vtype:
            oracle.is_in_doubled_e8("00000000")
        self.assertEqual(cm_vtype.exception.code, "INVALID_VEC8")

        # BF16 exactness boundaries (0, powers of 2, exp -126..127, overflow 2^129)
        self.assertIs(oracle.is_exact_bf16_half_integer(0), True)
        self.assertIs(oracle.is_exact_bf16_half_integer(-256), True)
        self.assertIs(oracle.is_exact_bf16_half_integer(255), True)
        self.assertIs(oracle.is_exact_bf16_half_integer(257), False)
        self.assertIs(oracle.is_exact_bf16_half_integer(1 << 128), True)   # v = 2^127 exact max BF16 power
        self.assertIs(oracle.is_exact_bf16_half_integer(1 << 129), False)  # v = 2^128 overflows BF16 exp > 127

        # sigma3, r_e8, w1, ceil_log2, ceil_isqrt_rational
        self.assertEqual(oracle.sigma3_exact(1), 1)
        self.assertEqual(oracle.sigma3_exact(4), 73)  # perfect square divisor check (d == q = 2)
        self.assertEqual(oracle.r_e8_shell_count(0), 1)
        self.assertEqual(oracle.r_e8_shell_count(2), 2160)
        self.assertEqual(oracle.w1_ball_count(0), 1)
        self.assertEqual(oracle.ceil_log2_int(1), 0)
        self.assertEqual(oracle.ceil_log2_int(49), 6)
        self.assertEqual(oracle.ceil_log2_int(64), 6)
        self.assertEqual(oracle.ceil_log2_int(65), 7)
        self.assertEqual(oracle.ceil_isqrt_rational(Fraction(0, 1)), Fraction(0, 1))
        self.assertEqual(oracle.ceil_isqrt_rational(Fraction(9, 4)), Fraction(3, 2))
        s2 = oracle.ceil_isqrt_rational(Fraction(2, 1))
        self.assertGreaterEqual(s2 * s2, Fraction(2, 1))
        self.assertLess((s2 - Fraction(1, 10 ** 9)) ** 2, Fraction(2, 1))

        # enumerate_e8_ball_doubled exact boundaries m=0, m=1 (241 pts)
        self.assertEqual(oracle.enumerate_e8_ball_doubled(0), ((0, 0, 0, 0, 0, 0, 0, 0),))
        self.assertEqual(len(oracle.enumerate_e8_ball_doubled(1)), 241)
        with self.assertRaises(Err):
            oracle.enumerate_e8_ball_doubled(4)

        # nearest_e8_exact round-half-even & negative flip & D8+1/2 coset win
        # Test exact half-integer tie (even vs odd) and D8 parity fix downward
        v_neg = (Fraction(-3, 5), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1),
                 Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))
        self.assertEqual(oracle.nearest_e8_exact(v_neg), (Fraction(0, 1),) * 8)
        v_half = (Fraction(1, 2),) * 8
        self.assertEqual(oracle.nearest_e8_exact(v_half), v_half)
        # Exercise tie-rounding to even (1/2 -> 0, 3/2 -> 2) and downward D8 parity flip
        v_tie = (Fraction(3, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2),
                 Fraction(-7, 10), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))
        res_tie = oracle.nearest_e8_exact(v_tie)
        self.assertIs(oracle.is_in_doubled_e8([int(2 * c) for c in res_tie]), True)
        # Exercise outward ball search (point outside m <= 2 ball -> scans LUT)
        lut_m2 = oracle._get_lut_m2()
        far_x = (Fraction(5, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1),
                 Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1))
        idx_far = oracle.nearest_e8_in_ball_doubled(far_x, 2, lut_m2)
        self.assertEqual(lut_m2[idx_far], (4, 0, 0, 0, 0, 0, 0, 0))
        # Valid public encode_2stage_residual_e8 + decode_2stage_residual_e8 round-trip
        enc_c, enc_f = oracle.encode_2stage_residual_e8(far_x, Fraction(1, 1), Fraction(1, 2), lut_m2, m_max=2)
        self.assertEqual(
            oracle.decode_2stage_residual_e8(enc_c, enc_f, Fraction(1, 1), Fraction(1, 2), lut_m2),
            (Fraction(3, 1),) + (Fraction(0, 1),) * 7,
        )
        # LUT validation errors (assert exact INVALID_LUT on BOTH empty tuple AND non-tuple list!)
        with self.assertRaises(Err) as cm_l1:
            oracle.nearest_e8_in_ball_doubled(v_half, 2, ())
        self.assertEqual(cm_l1.exception.code, "INVALID_LUT")
        with self.assertRaises(Err) as cm_l2:
            oracle.nearest_e8_in_ball_doubled(v_half, 2, [(1, 1, 1, 1, 1, 1, 1, 1)])
        self.assertEqual(cm_l2.exception.code, "INVALID_LUT")
        with self.assertRaises(Err) as cm_l3:
            oracle.nearest_e8_in_ball_doubled(v_half, 2, ((0, 0, 0, 0, 0, 0, 0, 0),))
        self.assertEqual(cm_l3.exception.code, "LUT_MISSING_POINT")
        with self.assertRaises(Err) as cm_l4:
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), ())
        self.assertEqual(cm_l4.exception.code, "INVALID_LUT")
        with self.assertRaises(Err) as cm_l5:
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), [(0, 0, 0, 0, 0, 0, 0, 0)])
        self.assertEqual(cm_l5.exception.code, "INVALID_LUT")
        with self.assertRaises(Err):
            oracle.decode_2stage_residual_e8(2401, 0, Fraction(1, 1), Fraction(1, 2), lut_m2)

        # Custom LUT vector validation (S32-VER-02 / S32-DS-02): non-2E8 and out-of-ball rejected in both encode & decode
        bad_non_e8_lut = ((100, 1, 0, 0, 0, 0, 0, 0),)
        bad_oob_m2_lut = ((4, 4, 0, 0, 0, 0, 0, 0),)  # shell 4 > 2
        bad_oob_m5_lut = ((10, 10, 0, 0, 0, 0, 0, 0),)  # shell 25 > 5
        shell1_lut = ((2, 2, 0, 0, 0, 0, 0, 0),)  # exact shell 1
        shell5_lut = ((6, 2, 0, 0, 0, 0, 0, 0),)  # exact shell 5 (36+4 = 40 = 8*5)
        with self.assertRaises(Err) as cm_nv1:
            oracle.nearest_e8_in_ball_doubled(far_x, 2, bad_non_e8_lut)
        self.assertEqual(cm_nv1.exception.code, "INVALID_LUT_VECTOR")
        with self.assertRaises(Err) as cm_ob1:
            oracle.nearest_e8_in_ball_doubled(far_x, 2, bad_oob_m2_lut)
        self.assertEqual(cm_ob1.exception.code, "LUT_VECTOR_OUT_OF_BALL")
        with self.assertRaises(Err) as cm_ob2:
            oracle.nearest_e8_in_ball_doubled(far_x, 2, bad_oob_m5_lut)
        self.assertEqual(cm_ob2.exception.code, "LUT_VECTOR_OUT_OF_BALL")
        with self.assertRaises(Err) as cm_enc_bad:
            oracle.encode_2stage_residual_e8(far_x, Fraction(1, 1), Fraction(1, 2), bad_non_e8_lut)
        self.assertEqual(cm_enc_bad.exception.code, "INVALID_LUT_VECTOR")
        with self.assertRaises(Err) as cm_dnv:
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), bad_non_e8_lut)
        self.assertEqual(cm_dnv.exception.code, "INVALID_LUT_VECTOR")
        with self.assertRaises(Err) as cm_dob:
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), bad_oob_m5_lut)
        self.assertEqual(cm_dob.exception.code, "LUT_VECTOR_OUT_OF_BALL")
        # Also validate fine index entry when coarse index is valid
        mixed_bad_lut = ((0, 0, 0, 0, 0, 0, 0, 0), (100, 1, 0, 0, 0, 0, 0, 0))
        with self.assertRaises(Err) as cm_dfine:
            oracle.decode_2stage_residual_e8(0, 1, Fraction(1, 1), Fraction(1, 2), mixed_bad_lut)
        self.assertEqual(cm_dfine.exception.code, "INVALID_LUT_VECTOR")
        # Exact max_shell boundaries (shell == max_shell OK; shell > max_shell rejected)
        self.assertEqual(
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), shell1_lut, max_shell=1),
            (Fraction(3, 2), Fraction(3, 2)) + (Fraction(0, 1),) * 6,
        )
        with self.assertRaises(Err) as cm_ms0:
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), shell1_lut, max_shell=0)
        self.assertEqual(cm_ms0.exception.code, "LUT_VECTOR_OUT_OF_BALL")
        self.assertEqual(
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), shell5_lut, max_shell=5),
            (Fraction(9, 2), Fraction(3, 2)) + (Fraction(0, 1),) * 6,
        )
        with self.assertRaises(Err):
            oracle.decode_2stage_residual_e8(0, 0, Fraction(1, 1), Fraction(1, 2), shell5_lut, max_shell=6)

        with self.assertRaises(Err) as cm_qlen:
            oracle.nearest_e8_exact((Fraction(0, 1),) * 7)
        self.assertEqual(cm_qlen.exception.code, "INVALID_Q_VEC8")
        with self.assertRaises(Err) as cm_qtype:
            oracle.nearest_e8_exact("00000000")
        self.assertEqual(cm_qtype.exception.code, "INVALID_Q_VEC8")

        # bareiss_det_exact & certify_sylvester_pd_exact (singular, row swap, non-PD, non-symmetric, jagged, str)
        swap_m = [[Fraction(0, 1), Fraction(1, 1)], [Fraction(1, 1), Fraction(0, 1)]]
        self.assertEqual(oracle.bareiss_det_exact(swap_m), Fraction(-1, 1))
        sing_m = [[Fraction(1, 1), Fraction(1, 1)], [Fraction(1, 1), Fraction(1, 1)]]
        self.assertEqual(oracle.bareiss_det_exact(sing_m), Fraction(0, 1))
        self.assertIs(oracle.certify_sylvester_pd_exact(sing_m), False)
        self.assertIs(oracle.certify_sylvester_pd_exact([[Fraction(2, 1)]]), True)
        with self.assertRaises(Err) as cm_m1:
            oracle.bareiss_det_exact([])
        self.assertEqual(cm_m1.exception.code, "INVALID_MATRIX")
        with self.assertRaises(Err) as cm_m2:
            oracle.bareiss_det_exact("bad")
        self.assertEqual(cm_m2.exception.code, "INVALID_MATRIX")
        with self.assertRaises(Err) as cm_m3:
            oracle.bareiss_det_exact([[Fraction(1, 1), Fraction(2, 1)]])
        self.assertEqual(cm_m3.exception.code, "NON_SQUARE_MATRIX")
        with self.assertRaises(Err) as cm_m4:
            oracle.bareiss_det_exact(["a"])
        self.assertEqual(cm_m4.exception.code, "NON_SQUARE_MATRIX")
        with self.assertRaises(Err) as cm_m5:
            oracle.certify_sylvester_pd_exact([[Fraction(1, 1), Fraction(2, 1)], [Fraction(0, 1), Fraction(1, 1)]])
        self.assertEqual(cm_m5.exception.code, "NON_SYMMETRIC_MATRIX")

    def test_09_certificate_validation_and_secondary_audit_killers(self) -> None:
        # Valid full replay and non-replay + strict bool check on full_replay (S32-VER-03)
        res_ok = oracle.validate_certificate_payload(self.cert, full_replay=True)
        self.assertIs(res_ok["valid"], True)
        res_nr = oracle.validate_certificate_payload(self.cert, full_replay=False)
        self.assertIs(res_nr["full_replay"], False)
        for bad_flag in ("yes", 1, 0, None):
            with self.assertRaises(Err) as cm_rf:
                oracle.validate_certificate_payload(self.cert, full_replay=bad_flag)  # type: ignore[arg-type]
            self.assertEqual(cm_rf.exception.code, "INVALID_BOOL")

        # Unbounded int inside cert payload rejected cleanly (no raw ValueError!)
        c_big = copy.deepcopy(self.cert)
        c_big["payload"]["pillar2_wcet_vmem_tpu_roofline"]["tpu_v5e_vmem_bytes"] = 10 ** 4300
        with self.assertRaises(Err) as cm_cbig:
            oracle.validate_certificate_payload(c_big, full_replay=False)
        self.assertEqual(cm_cbig.exception.code, "INT_TOO_LARGE")

        # Top-level structure tamper tests (assert exact error codes!)
        with self.assertRaises(Err) as cm_c0:
            oracle.validate_certificate_payload("not_dict")
        self.assertEqual(cm_c0.exception.code, "CERT_NOT_OBJECT")
        c_extra = copy.deepcopy(self.cert)
        c_extra["extra"] = 1
        with self.assertRaises(Err) as cm_ck:
            oracle.validate_certificate_payload(c_extra)
        self.assertEqual(cm_ck.exception.code, "INVALID_CERT_KEYS")
        for field, bad_val, expected_code in (
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
            self.assertEqual(cm_f.exception.code, expected_code)

        # Exact inclusive boundary tests: aggregate_opt_ratio == 1/8, block_h2_max == 127, cs_dot_32w == 32767 MUST PASS!
        p_eq_1_8 = copy.deepcopy(self.payload)
        p_eq_1_8["pillar3_residual_flat_lut_e8_codec"]["aggregate_opt_ratio"] = "1/8"
        self.assertIs(audit.secondary_algebraic_audit(p_eq_1_8)["secondary_audit_passed"], True)
        p_eq_127 = copy.deepcopy(self.payload)
        p_eq_127["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"]["block_h2_max"] = oracle.INT8_MAX
        self.assertIs(audit.secondary_algebraic_audit(p_eq_127)["secondary_audit_passed"], True)
        p_eq_32767 = copy.deepcopy(self.payload)
        p_eq_32767["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"]["cs_dot_32w_int8_sym127"] = oracle.INT16_MAX
        self.assertIs(audit.secondary_algebraic_audit(p_eq_32767)["secondary_audit_passed"], True)

        # Re-signed tampered payload passes SHA checks with full_replay=False, fails full_replay=True AND secondary_algebraic_audit!
        def _resign(payload_mut: dict[str, object]) -> dict[str, object]:
            p_sha = oracle.hashlib.sha256(oracle.canonical_json_bytes(payload_mut)).hexdigest()
            core = {"schema": oracle.SCHEMA_VERSION, "track_id": oracle.TRACK_ID, "payload_sha256": p_sha, "payload": payload_mut}
            c_sha = oracle.hashlib.sha256(oracle.canonical_json_bytes(core)).hexdigest()
            return {"schema": oracle.SCHEMA_VERSION, "track_id": oracle.TRACK_ID, "payload_sha256": p_sha, "certificate_sha256": c_sha, "payload": payload_mut}

        p_tamper = copy.deepcopy(self.payload)
        p_tamper["measured_pallas_tok_s"] = True
        with self.assertRaises(Err):
            oracle.validate_certificate_payload(_resign(p_tamper), full_replay=True)

        # Hollow payload rejected cleanly even under full_replay=False & secondary_algebraic_audit (no raw KeyError!)
        with self.assertRaises(Err):
            oracle.validate_certificate_payload(_resign({}), full_replay=False)
        with self.assertRaises(Err):
            audit.secondary_algebraic_audit("not_dict")
        with self.assertRaises(Err):
            audit.secondary_algebraic_audit({})

        # Secondary audit branch killers (every single numeric/flag/type branch in secondary_algebraic_audit!)
        mutators = [
            lambda p: p.__setitem__("measured_pallas_tok_s", True),
            lambda p: p.__setitem__("zero_rng", False),
            lambda p: p.pop("pillar1_precast_overflow_separation"),
            lambda p: p.__setitem__("pillar1_precast_overflow_separation", []),
            lambda p: p["pillar1_precast_overflow_separation"].__setitem__("h8_all_rows_preserve_mod4", False),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"].__setitem__("block_h2_max", 128),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"].__setitem__("block_h2_max", "78"),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"].__setitem__("wht8_max", 127),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"].__setitem__("wht8_max", "156"),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("cs_dot_32w_int8_sym127", 32768),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("cs_dot_32w_int8_sym127", "14078"),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"].__setitem__("exact_dot_32w_int8_sym127", 32767),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b136s8"].__setitem__("exact_dot_32w_int8_sym127", "40132"),
            lambda p: p["pillar1_precast_overflow_separation"]["accumulator_width_k16384"].__setitem__("headroom_exceeds_100x_all_3", False),
            lambda p: p["pillar1_precast_overflow_separation"]["accumulator_width_k16384"].__setitem__("int32_headroom_cs", "100"),
            lambda p: p["pillar2_wcet_vmem_tpu_roofline"].__setitem__("tpu_v5e_vmem_bytes", 1),
            lambda p: p["pillar2_wcet_vmem_tpu_roofline"].__setitem__("guide_a_136b_leaves_gt_98_8_pct_free", False),
            lambda p: p["pillar2_wcet_vmem_tpu_roofline"]["tpu_v5e_three_way_roofline"].__setitem__("b_int8_ridge_4_50", "1"),
            lambda p: p["pillar2_wcet_vmem_tpu_roofline"]["tpu_v5e_three_way_roofline"].__setitem__("r_decode_breakeven_gwps", "1364"),
            lambda p: p["pillar2_wcet_vmem_tpu_roofline"]["tpu_v5e_three_way_roofline"].__setitem__("batch_73_bf16_zero_gain_certified", False),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"]["lut_12bit_32kib"].__setitem__("enumerated_points", 2400),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"]["lut_16bit_512kib"].__setitem__("points", 56880),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"].__setitem__("all_6_opt_ratios_le_1_4", False),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"].__setitem__("aggregate_opt_ratio", "1/4"),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"].__setitem__("codec_distortion_suite", {}),
            lambda p: p["pillar4_spectral_lipschitz_enclosure"].__setitem__("h8_orthogonal_8I", False),
            lambda p: p["pillar4_spectral_lipschitz_enclosure"]["anisotropy_operator_separation"].__setitem__("h8_exact_8x_flip", False),
            lambda p: p["pillar5_safety_refusal_margin_preservation"].__setitem__("scalar_breaks_refusal_boundary", False),
            lambda p: p["pillar5_safety_refusal_margin_preservation"].__setitem__("token_embd_quantized_by_block_lut", True),
            lambda p: p["pillar5_safety_refusal_margin_preservation"].__setitem__("mse_reduction_rho", "1/100"),
            # R2 secondary_algebraic_audit strict type, length & BoolOp/SWAP killers (S32-VER-03)
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"].__setitem__("extra_regime", {}),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"].pop("1block_E8_3.0bpw"),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("m_max", True),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"]["max_abs_z_witness"].__setitem__(1, True),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("max_abs_z_witness", None),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("max_abs_z_witness", (0,) * 8),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("max_abs_z_witness", [0] * 7),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("exact_dot_32w_witness_blocks", None),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("exact_dot_32w_witness_blocks", tuple([0] * 8 for _ in range(4))),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"].__setitem__("exact_dot_32w_witness_blocks", [[0] * 8] * 3),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"]["exact_dot_32w_witness_blocks"].__setitem__(0, None),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"]["exact_dot_32w_witness_blocks"].__setitem__(0, (0,) * 8),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"]["exact_dot_32w_witness_blocks"].__setitem__(0, [0] * 7),
            lambda p: p["pillar1_precast_overflow_separation"]["regimes"]["EBG32:b88s8"]["exact_dot_32w_witness_blocks"][0].__setitem__(0, True),
            lambda p: p["pillar1_precast_overflow_separation"]["accumulator_width_k16384"].__setitem__("hidden_dim_k", True),
            lambda p: p["pillar2_wcet_vmem_tpu_roofline"]["tpu_v5e_three_way_roofline"].__setitem__("hbm_bandwidth_gbs", True),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"].__setitem__("codec_distortion_suite", None),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"].__setitem__("codec_distortion_suite", tuple(p["pillar3_residual_flat_lut_e8_codec"]["codec_distortion_suite"])),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"].__setitem__("codec_distortion_suite", p["pillar3_residual_flat_lut_e8_codec"]["codec_distortion_suite"][:5]),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"]["codec_distortion_suite"].__setitem__(0, []),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"]["codec_distortion_suite"][0].__setitem__("idx_c_opt", True),
            lambda p: p["pillar3_residual_flat_lut_e8_codec"]["codec_distortion_suite"][0].__setitem__("idx_f_opt", True),
            lambda p: p["pillar5_safety_refusal_margin_preservation"].__setitem__("matrix_dim_n", 7),
            lambda p: p["pillar5_safety_refusal_margin_preservation"].__setitem__("matrix_dim_d", 7),
            lambda p: p["pillar5_safety_refusal_margin_preservation"].__setitem__("matrix_dim_n", True),
        ]
        for fn in mutators:
            pm = copy.deepcopy(self.payload)
            fn(pm)
            with self.assertRaises(Err):
                audit.secondary_algebraic_audit(pm)

        # Short-circuit order killer for matrix_dim_n != 8 or matrix_dim_d != 8 (kills BoolOp SWAP!)
        pm_dim_order = copy.deepcopy(self.payload)
        pm_dim_order["pillar5_safety_refusal_margin_preservation"]["matrix_dim_n"] = 7
        pm_dim_order["pillar5_safety_refusal_margin_preservation"]["matrix_dim_d"] = True
        with self.assertRaises(Err) as cm_dim:
            audit.secondary_algebraic_audit(pm_dim_order)
        self.assertEqual(cm_dim.exception.code, "AUDIT_P5_DIM")

    # -----------------------------------------------------------------------
    # 3. Zone-B File Security, CLI Parsing & Adversarial Path Hardening
    # -----------------------------------------------------------------------

    def test_10_zone_b_io_hardening_symlinks_hardlinks_and_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cert_path = tdp / "cert.json"

            # Valid write + overwrite existing single-link file (st_nlink == 1) + check + require_safe_path
            wres = audit.write_certificate_file(cert_path)
            self.assertIs(wres["written"], True)
            wres2 = audit.write_certificate_file(cert_path)
            self.assertIs(wres2["written"], True)
            self.assertEqual(audit.require_safe_path(cert_path), cert_path)
            self.assertEqual(audit.require_safe_path(tdp / "nonexistent.json"), tdp / "nonexistent.json")
            self.assertEqual(audit.require_safe_path(Path("/")), Path("/"))
            cres = audit.check_certificate_file(cert_path)
            self.assertIs(cres["valid"], True)
            self.assertIs(cres["secondary_audit"]["secondary_audit_passed"], True)

            # Relative path + PWD symlink check
            old_cwd = os.getcwd()
            old_pwd = os.environ.get("PWD")
            try:
                os.chdir(td)
                os.environ["PWD"] = td
                self.assertEqual(audit.require_safe_path(Path("cert.json")), Path("cert.json"))
                # PWD pointing through a symlink must be rejected!
                sym_dir = tdp / "symdir"
                os.symlink(tdp, sym_dir)
                os.environ["PWD"] = str(sym_dir)
                with self.assertRaises(Err):
                    audit.require_safe_path(Path("cert.json"))
                # Non-existent PWD ignored cleanly (only PATH_NOT_FOUND tolerated)
                os.environ["PWD"] = str(tdp / "no_such_dir")
                self.assertEqual(audit.require_safe_path(Path("cert.json")), Path("cert.json"))
            finally:
                os.chdir(old_cwd)
                if old_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = old_pwd

            # Type/NUL/directory/symlink/hardlink/FIFO/oversize/non-canonical/CRLF/UTF-8 rejections
            with self.assertRaises(Err):
                audit.require_safe_path("cert.json")
            with self.assertRaises(Err):
                audit.require_safe_path(_SubPath(cert_path))
            with self.assertRaises(Err):
                audit.require_safe_path(Path(f"{td}/a\x00b"))
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(cert_path, "not_bytes")  # type: ignore[arg-type]
            with self.assertRaises(Err):
                audit.read_regular_file_bytes(Path("/"))
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(Path("/"), b"{}")
            with self.assertRaises(Err):
                audit.read_regular_file_bytes(tdp)
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(tdp, b"{}")

            # Symlink at leaf & intermediate symlink (assert exact SYMLINK_FORBIDDEN & FILE_READ_ERROR!)
            leaf_sym = tdp / "leaf_sym.json"
            os.symlink(cert_path, leaf_sym)
            with self.assertRaises(Err) as cm_s1:
                audit.require_safe_path(leaf_sym)
            self.assertEqual(cm_s1.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(Err) as cm_s2:
                audit.read_regular_file_bytes(leaf_sym)
            self.assertEqual(cm_s2.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(Err) as cm_s3:
                audit.write_regular_file_bytes(leaf_sym, b"{}\n")
            self.assertEqual(cm_s3.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(Err) as cm_noent:
                audit.read_regular_file_bytes(tdp / "does_not_exist.json")
            self.assertEqual(cm_noent.exception.code, "FILE_READ_ERROR")
            inter_sym = tdp / "inter_sym"
            os.symlink(tdp, inter_sym)
            with self.assertRaises(Err) as cm_s4:
                audit.read_regular_file_bytes(inter_sym / "cert.json")
            self.assertEqual(cm_s4.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(Err) as cm_md:
                audit.read_regular_file_bytes(tdp / "missing_dir" / "cert.json")
            self.assertEqual(cm_md.exception.code, "PATH_NOT_FOUND")

            # Hardlink target refusal on write
            hl_target = tdp / "hl.json"
            os.link(cert_path, hl_target)
            with self.assertRaises(Err) as cm_hl:
                audit.write_regular_file_bytes(hl_target, b"{}\n")
            self.assertEqual(cm_hl.exception.code, "HARDLINK_FORBIDDEN")
            os.unlink(hl_target)

            # FIFO non-regular refusal
            fifo_p = tdp / "p.fifo"
            os.mkfifo(fifo_p)
            with self.assertRaises(Err):
                audit.read_regular_file_bytes(fifo_p)
            with self.assertRaises(Err):
                audit.write_regular_file_bytes(fifo_p, b"{}\n")

            # Exact MAX_JSON_BYTES boundary (exact OK, +1 rejected)
            exact_p = tdp / "exact_max.bin"
            exact_p.write_bytes(b"a" * oracle.MAX_JSON_BYTES)
            self.assertEqual(len(audit.read_regular_file_bytes(exact_p)), oracle.MAX_JSON_BYTES)
            over_p = tdp / "over_max.bin"
            over_p.write_bytes(b"a" * (oracle.MAX_JSON_BYTES + 1))
            with self.assertRaises(Err) as cm_sz:
                audit.read_regular_file_bytes(over_p)
            self.assertEqual(cm_sz.exception.code, "FILE_TOO_LARGE")

            # CRLF, missing trailing LF, invalid UTF-8, non-canonical JSON spacing (assert exact codes!)
            crlf_p = tdp / "crlf.json"
            crlf_p.write_bytes(cert_path.read_bytes().replace(b"\n", b"\r\n"))
            with self.assertRaises(Err) as cm_cr:
                audit.check_certificate_file(crlf_p)
            self.assertEqual(cm_cr.exception.code, "CRLF_FORBIDDEN")
            nolf_p = tdp / "nolf.json"
            nolf_p.write_bytes(cert_path.read_bytes().rstrip(b"\n"))
            with self.assertRaises(Err) as cm_lf:
                audit.check_certificate_file(nolf_p)
            self.assertEqual(cm_lf.exception.code, "MISSING_TRAILING_LF")
            badutf_p = tdp / "badutf.json"
            badutf_p.write_bytes(b"\xff\xfe\n")
            with self.assertRaises(Err) as cm_u8:
                audit.check_certificate_file(badutf_p)
            self.assertEqual(cm_u8.exception.code, "INVALID_UTF8")
            noncan_p = tdp / "noncan.json"
            noncan_p.write_bytes(b" " + cert_path.read_bytes())
            with self.assertRaises(Err) as cm_nc:
                audit.check_certificate_file(noncan_p)
            self.assertEqual(cm_nc.exception.code, "NON_CANONICAL_BYTES")

            # Monotonic counter & uniqueness of _next_tmp_name (S32-VER-04 / S32-DS-04)
            tn1 = audit._next_tmp_name("col.json")
            tn2 = audit._next_tmp_name("col.json")
            self.assertNotEqual(tn1, tn2)
            seq1 = oracle.parse_exact_int_ascii(tn1.split(".")[-2], "seq1")
            seq2 = oracle.parse_exact_int_ascii(tn2.split(".")[-2], "seq2")
            self.assertEqual(seq2, seq1 + 1)

            # Temp file collision retry (1 pre-existing tmp -> succeeds on attempt 2) & exhaustion (all 16)
            col_target = tdp / "col.json"
            first_tmp = tdp / ".col.json.s32tmp.fixed0"
            first_tmp.write_bytes(b"occupy")
            with mock.patch.object(
                audit,
                "_next_tmp_name",
                side_effect=[".col.json.s32tmp.fixed0", ".col.json.s32tmp.fixed1"],
            ):
                audit.write_regular_file_bytes(col_target, b"ok\n")
            self.assertEqual(col_target.read_bytes(), b"ok\n")

            col2_tmp = tdp / ".col2.json.s32tmp.occupied"
            col2_tmp.write_bytes(b"x")
            with mock.patch.object(audit, "_next_tmp_name", return_value=".col2.json.s32tmp.occupied"):
                with self.assertRaises(Err) as cm_col2:
                    audit.write_regular_file_bytes(tdp / "col2.json", b"ok\n")
            self.assertEqual(cm_col2.exception.code, "FILE_WRITE_ERROR")

            # Temp file hardlink guard (st_nlink != 1 -> HARDLINK_FORBIDDEN + temp cleanup)
            hl_tmp_target = tdp / "hl_tmp.json"
            real_fstat = os.fstat

            def _fake_fstat(fd: int) -> os.stat_result:
                st = real_fstat(fd)
                return os.stat_result((st.st_mode, st.st_ino, st.st_dev, 2, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))

            with mock.patch.object(audit.os, "fstat", side_effect=_fake_fstat):
                with self.assertRaises(Err) as cm_hltmp:
                    audit.write_regular_file_bytes(hl_tmp_target, b"ok\n")
            self.assertEqual(cm_hltmp.exception.code, "HARDLINK_FORBIDDEN")
            self.assertEqual(list(tdp.glob(".hl_tmp.json.s32tmp.*")), [])

            # Live 24-thread concurrent same-target write stress (48/48 OK in unit suite; 24Tx20=480 in stress gate)
            conc_target = tdp / "concurrent.json"
            conc_errors: list[str] = []
            conc_lock = threading.Lock()
            conc_ok = 0

            def _writer(tid: int) -> None:
                nonlocal conc_ok
                payload_bytes = f'{{"tid":{tid}}}\n'.encode("ascii")
                for _ in range(2):
                    try:
                        audit.write_regular_file_bytes(conc_target, payload_bytes)
                        with conc_lock:
                            conc_ok += 1
                    except Exception as exc:  # pragma: no cover - must never trigger
                        with conc_lock:
                            conc_errors.append(f"t{tid}: {type(exc).__name__}: {exc}")

            threads = [threading.Thread(target=_writer, args=(t,)) for t in range(24)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(conc_errors, [])
            self.assertEqual(conc_ok, 48)
            self.assertEqual(list(tdp.glob(".concurrent.json.s32tmp.*")), [])
            parsed_final = oracle.parse_strict_json_text(conc_target.read_bytes().decode("ascii"), require_object=True)
            self.assertIn("tid", parsed_final)

            # chmod 000 unreadable file & directory
            if os.geteuid() != 0:
                unr = tdp / "unr.json"
                unr.write_bytes(b"{}\n")
                os.chmod(unr, 0o000)
                try:
                    with self.assertRaises(Err):
                        audit.read_regular_file_bytes(unr)
                finally:
                    os.chmod(unr, 0o600)
                unr_d = tdp / "unr_d"
                unr_d.mkdir()
                os.chmod(unr_d, 0o000)
                try:
                    with self.assertRaises(Err):
                        audit.read_regular_file_bytes(unr_d / "x.json")
                    with self.assertRaises(Err):
                        audit.write_regular_file_bytes(unr_d / "x.json", b"{}\n")
                finally:
                    os.chmod(unr_d, 0o700)

    def test_11_cli_parsing_deleted_cwd_and_flag_traps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            out_path = tdp / "cli_cert.json"

            # Direct main() tests: zero-arg, --standalone --json, --write, --check, --check --json
            for argv_ok in ([], ["--standalone"], ["--standalone", "--json"], ["--write", str(out_path), "--json"], ["--check", str(out_path)], ["--check", str(out_path), "--json"]):
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

            # Strict CLI traps: --write --json, --write=--json, --check=--json, duplicate, conflicting, missing, empty, unknown, non-list, lone surrogate
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
                ["--unknown"],
                ["--write", "bad\ud800.json"],
                "not_a_list",
            ]
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
            # Ensure `--write --json` created ZERO files named `--json` in CWD
            self.assertFalse(Path("--json").exists())

            # Deleted-CWD fail-closed check (in isolated subprocess)
            del_dir = tdp / "vanishing"
            del_dir.mkdir()
            script = (
                "import os, sys; "
                f"sys.path.insert(0, {str(_THIS_DIR)!r}); "
                "import siliquant_tpu_bounds_audit as a; "
                f"os.chdir({str(del_dir)!r}); os.rmdir({str(del_dir)!r}); "
                "raise SystemExit(a.main(['--check', 'rel.json']))"
            )
            proc = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("CWD_UNAVAILABLE", proc.stderr)

            # /proc/kallsyms capped read -> FILE_TOO_LARGE (when present on Linux)
            kallsyms = Path("/proc/kallsyms")
            if kallsyms.exists():
                with self.assertRaises(Err) as cm_proc:
                    audit.read_regular_file_bytes(kallsyms)
                self.assertEqual(cm_proc.exception.code, "FILE_TOO_LARGE")

    def test_12_disk_certificate_matches_standalone(self) -> None:
        disk_cert = _THIS_DIR / "siliquant_tpu_bounds_certificate.json"
        if disk_cert.exists():
            res = audit.check_certificate_file(disk_cert)
            self.assertIs(res["valid"], True)
            self.assertEqual(res["payload_sha256"], self.cert["payload_sha256"])
            self.assertEqual(res["certificate_sha256"], self.cert["certificate_sha256"])


if __name__ == "__main__":
    unittest.main()
