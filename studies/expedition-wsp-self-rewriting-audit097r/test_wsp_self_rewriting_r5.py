#!/usr/bin/env python3
"""Assertion-free R5 mutation-killing, conjunct-killing & known-answer test suite for AUDIT-097.

Covers both wsp_self_rewriting_oracle.py and wsp_self_rewriting_audit.py under
python3 -I -O -B -W error with zero bare asserts and zero floats.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wsp_self_rewriting_audit import (
    CAR_RAMP_ACTOR_MAP,
    DEFAULT_GOVERNANCE_POLICY,
    DEFAULT_STAGE2_BOUNDS,
    EXP8_AUDITED_SPECS,
    _decode_ieee_tensor_scaled_f64,
    _evaluate_car_ramp_live_checkpoints,
    _evaluate_exp8_live_screens,
    _evaluate_exp8_live_tensors,
    _normalize_screen_rows,
    load_jsonl_rows,
    main as audit_main,
    parse_npy_bytes,
    run_arxiv_governance_audit,
    run_live_audit,
    run_siliquant_double_bind_witness,
    run_standalone_audit,
    run_synthetic_rank1_pruning_theorems,
    sha256_file,
    validate_audit_certificate,
)
from wsp_self_rewriting_oracle import (
    OracleError,
    RationalBound,
    bf16_u16_to_fraction,
    bf16_u16_to_scaled_int,
    canonical_sha256,
    certify_group_quantization_survival_exact,
    certify_paired_dual_screen,
    certify_rank1_vs_pruning_exact,
    certify_self_rewriting_session,
    contains_float_recursive,
    exact_mcnemar_two_sided_fraction,
    f32_u32_to_bf16_u16_rne,
    f32_u32_to_fraction,
    f32_u32_to_scaled_int,
    f64_u64_to_fraction,
    f64_u64_to_scaled_int,
    format_exact_decimal,
    format_exact_percent,
    frac_to_dict,
    make_fixed_radix_certificate,
    quantize_sym_rne_exact,
    verify_fixed_radix_stage2_division_free,
)


def _encode_npy(descr: str, shape: tuple[int, ...], body: bytes, *, version: int = 1) -> bytes:
    hdr = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}".encode("latin1")
    if version == 1:
        pad = (16 - ((10 + len(hdr) + 1) % 16)) % 16
        hdr_full = hdr + b" " * pad + b"\n"
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr_full)) + hdr_full + body
    pad = (16 - ((12 + len(hdr) + 1) % 16)) % 16
    hdr_full = hdr + b" " * pad + b"\n"
    return b"\x93NUMPY" + bytes([version, 0]) + struct.pack("<I", len(hdr_full)) + hdr_full + body


def _build_synthetic_intake(intake: Path) -> None:
    (intake / "sources").mkdir(parents=True)
    (intake / "raw").mkdir(parents=True)
    exp8 = intake / "repos" / "weight-edit-minimization" / "data" / "exp8"
    specs = exp8 / "specs"
    screens = intake / "repos" / "weight-edit-minimization" / "data" / "screens"
    car = intake / "repos" / "weight-edit-minimization" / "data" / "car-ramp"
    for d in (specs, screens, car):
        d.mkdir(parents=True)

    (intake / "sources" / "arxiv_2611_21885.pdf").write_bytes(b"%PDF-1.4 mock")
    (intake / "raw" / "openresearch_EdsyLoFUfY40.txt").write_text("mock", encoding="utf-8")

    # 2x2 Exp 8 tensors: base = 0.0, L15 delta = [[1, 1], [1, 1]], L19 delta = [[1, 1], [0, 0]]
    zero_f32_2x2 = struct.pack("<4I", 0, 0, 0, 0)
    (exp8 / "base_l15_down_proj_f32.npy").write_bytes(_encode_npy("<f4", (2, 2), zero_f32_2x2))
    (exp8 / "base_l19_down_proj_f32.npy").write_bytes(_encode_npy("<f4", (2, 2), zero_f32_2x2))

    f1 = 0x3F800000
    with zipfile.ZipFile(exp8 / "factors_layer15.npz", "w") as zf:
        zf.writestr("u.npy", _encode_npy("<f4", (2,), struct.pack("<2I", f1, f1)))
        zf.writestr("v.npy", _encode_npy("<f4", (2,), struct.pack("<2I", f1, f1)))
    with zipfile.ZipFile(exp8 / "factors_layer19.npz", "w") as zf:
        zf.writestr("u.npy", _encode_npy("<f4", (2,), struct.pack("<2I", f1, 0)))
        zf.writestr("v.npy", _encode_npy("<f4", (2,), struct.pack("<2I", f1, f1)))

    bf1 = 0x3F80
    with zipfile.ZipFile(exp8 / "edit.npz", "w") as zf:
        zf.writestr("t0_idx.npy", _encode_npy("<i4", (4,), struct.pack("<4i", 0, 1, 2, 3)))
        zf.writestr("t0_val.npy", _encode_npy("<u2", (4,), struct.pack("<4H", bf1, bf1, bf1, bf1)))
        zf.writestr("t1_idx.npy", _encode_npy("<i4", (2,), struct.pack("<2i", 0, 1)))
        zf.writestr("t1_val.npy", _encode_npy("<u2", (2,), struct.pack("<2H", bf1, bf1)))

    (exp8 / "frozen_edit_audit.json").write_text(
        json.dumps({"15": {"changed_values": 4}, "19": {"changed_values": 4}}),
        encoding="utf-8",
    )
    (exp8 / "final_choice.json").write_text("{}", encoding="utf-8")
    (exp8 / "results_table.json").write_text("[]", encoding="utf-8")
    (exp8 / "candidate_entries.json").write_text("{}", encoding="utf-8")
    (exp8 / "validate_D4k_math500.jsonl").write_text('{"id":"m0","correct":[1,0,1,0]}\n', encoding="utf-8")

    for sname in EXP8_AUDITED_SPECS:
        if sname.startswith("D"):
            body = {"scale": 1, "entries15": [[0, 0], [1, 1]], "entries19": [[0, 0]]}
        elif sname == "ER768_L15":
            body = {"scale": 1, "rows15": [0], "cols15": [0], "cols19": []}
        elif sname.startswith("ER"):
            body = {"scale": 1, "rows15": [0], "cols19": []}
        elif sname == "B15_full":
            body = {"scale": 1, "cols19": []}
        else:
            body = {"scale": 1, "cols15": [0, 1], "cols19": [0, 1]}
        (specs / f"{sname}.json").write_text(json.dumps(body), encoding="utf-8")

    arm_names = (
        "A64_topk64", "G000_scale", "A48_topk48", "B15_full", "B15_full_s125", "B19_full",
        "ER128_L15", "ER384_L15", "ER768_L15",
        "D1k_entries", "D2k_entries", "D4k_entries", "D8k_entries", "D16k_entries", "D32k_entries",
    )
    row_txt = '{"id":"s0","correct":[1],"finish":["stop"]}\n'
    for arm in arm_names:
        (screens / f"screen_{arm}.jsonl").write_text(row_txt, encoding="utf-8")

    # Base PPO in <f4, edited checkpoints in <f8 (40.0 = 0x4044000000000000)
    with zipfile.ZipFile(car / "base_ppo_final_seed0.npz", "w") as zf:
        for _, flax_key in CAR_RAMP_ACTOR_MAP:
            zf.writestr(flax_key, _encode_npy("<f4", (1,), struct.pack("<I", 0)))
    for fname in ("edited_actor.npz", "joint_free.npz", "greedy12_astra.npz", "greedy11_drop02.npz", "c2_14entry.npz", "c2_min_10entry.npz"):
        with zipfile.ZipFile(car / fname, "w") as zf:
            for actor_key, _ in CAR_RAMP_ACTOR_MAP:
                val_bits = 0x4044000000000000 if actor_key == "actor_head.W" else 0
                zf.writestr(f"{actor_key}.npy", _encode_npy("<f8", (1,), struct.pack("<Q", val_bits)))


class TestOraclePrimitives(unittest.TestCase):
    def test_rational_bound_formatters_and_float_detector(self) -> None:
        rb = RationalBound.from_dict({"num": 3, "den": 4})
        self.assertEqual(rb.to_dict(), {"num": 3, "den": 4})
        self.assertEqual(frac_to_dict(Fraction(6, 8)), {"num": 3, "den": 4})
        self.assertEqual(format_exact_decimal(Fraction(-3, 4), 2), "-0.75")
        self.assertEqual(format_exact_decimal(Fraction(7, 2), 0), "4")
        self.assertEqual(format_exact_percent(Fraction(1, 8), 2), "12.50%")

        self.assertFalse(contains_float_recursive({"a": [1, "2", (True, None), {3}, frozenset([4])]}))
        smuggled = json.loads('{"x": [1, {"y": 3.14159}]}')
        self.assertTrue(contains_float_recursive(smuggled))
        smuggled_key = {json.loads("1.5"): 1}
        self.assertTrue(contains_float_recursive(smuggled_key))
        self.assertTrue(contains_float_recursive(complex(1, 2)))
        self.assertTrue(contains_float_recursive({json.loads("1.5")}))
        self.assertTrue(contains_float_recursive(frozenset([json.loads("1.5")])))

        with self.assertRaisesRegex(OracleError, "mapping"):
            RationalBound.from_dict(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "requires 'num' and 'den'"):
            RationalBound.from_dict({"num": 1})
        with self.assertRaisesRegex(OracleError, "requires 'num' and 'den'"):
            RationalBound.from_dict({"den": 1})
        with self.assertRaisesRegex(OracleError, "only 'num' and 'den'"):
            RationalBound.from_dict({"num": 1, "den": 2, "extra": 3})
        for bad in ({"num": True, "den": 1}, {"num": -1, "den": 1}, {"num": 1, "den": 0}):
            with self.assertRaises(OracleError):
                RationalBound.from_dict(bad)  # type: ignore[arg-type]
        with self.assertRaises(OracleError):
            frac_to_dict(1)  # type: ignore[arg-type]
        with self.assertRaises(OracleError):
            format_exact_decimal(1, 2)  # type: ignore[arg-type]
        with self.assertRaises(OracleError):
            format_exact_decimal(Fraction(1, 2), -1)

    def test_exact_ieee_decoding_and_rne(self) -> None:
        self.assertEqual(f32_u32_to_fraction(0x3F800000), Fraction(1, 1))
        self.assertEqual(f32_u32_to_fraction(0xC0000000), Fraction(-2, 1))
        self.assertEqual(f32_u32_to_fraction(0x00000000), Fraction(0, 1))
        self.assertEqual(f32_u32_to_fraction(0x00000001), Fraction(1, 1 << 149))
        self.assertEqual(f32_u32_to_scaled_int(0x80000001), -1)

        self.assertEqual(bf16_u16_to_fraction(0x3F80), Fraction(1, 1))
        self.assertEqual(bf16_u16_to_fraction(0xBFC0), Fraction(-3, 2))
        self.assertEqual(bf16_u16_to_fraction(0x0000), Fraction(0, 1))
        self.assertEqual(bf16_u16_to_fraction(0x0001), Fraction(1, 1 << 133))
        self.assertEqual(bf16_u16_to_scaled_int(0x8001), -1)

        self.assertEqual(f32_u32_to_bf16_u16_rne(0x3F808000), 0x3F80)
        self.assertEqual(f32_u32_to_bf16_u16_rne(0x3F818000), 0x3F82)

        for bad_f32 in (-1, 1 << 32, 0x7F800000, 0xFFFFFFFF, True):
            with self.assertRaises(OracleError):
                f32_u32_to_scaled_int(bad_f32)  # type: ignore[arg-type]
            with self.assertRaises(OracleError):
                f32_u32_to_bf16_u16_rne(bad_f32)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "overflows"):
            f32_u32_to_bf16_u16_rne(0x7F7FFFFF)
        with self.assertRaisesRegex(OracleError, "overflows"):
            f32_u32_to_bf16_u16_rne(0xFF7FFFFF)

        for bad_bf16 in (-1, 1 << 16, 0x7F80, 0xFFFF, False):
            with self.assertRaises(OracleError):
                bf16_u16_to_scaled_int(bad_bf16)  # type: ignore[arg-type]

        self.assertEqual(f64_u64_to_fraction(0x3FF0000000000000), Fraction(1, 1))
        self.assertEqual(f64_u64_to_fraction(0xC044000000000000), Fraction(-40, 1))
        self.assertEqual(f64_u64_to_fraction(0), Fraction(0, 1))
        self.assertEqual(f64_u64_to_fraction(1), Fraction(1, 1 << 1074))
        self.assertEqual(f64_u64_to_scaled_int(0x8000000000000001), -1)
        for bad_f64 in (-1, 1 << 64, 0x7FF0000000000000, 0xFFF0000000000001, True):
            with self.assertRaises(OracleError):
                f64_u64_to_scaled_int(bad_f64)  # type: ignore[arg-type]


class TestRank1SiliquantAndStage2Oracle(unittest.TestCase):
    def test_pruning_dichotomy_and_all_mask_regimes(self) -> None:
        res = run_synthetic_rank1_pruning_theorems()
        self.assertTrue(res["theorem_holds"])
        self.assertEqual(res["full_rank1"]["regime"], "full_rank1")
        self.assertEqual(res["column_pruned"]["regime"], "column_mask")
        self.assertEqual(res["row_pruned"]["regime"], "row_mask")
        self.assertEqual(res["entry_pruned"]["regime"], "entry_scattered")
        self.assertTrue(res["row_pruned"]["closed_form_row_leakage_holds"])
        self.assertTrue(res["entry_pruned"]["pythagorean_exact"])
        self.assertTrue(res["entry_pruned"]["gram_defect_nonnegative"])
        self.assertFalse(res["entry_pruned"]["is_exact_rank_le_1"])
        self.assertTrue(res["entry_pruned"]["safety_orthogonality_breached"])

        u = [Fraction(1, 1), Fraction(2, 1)]
        v = [Fraction(3, 1), Fraction(4, 1)]
        s = [[Fraction(2, 1), Fraction(-1, 1)]]
        empty = certify_rank1_vs_pruning_exact(u, v, [[0, 0], [0, 0]], s)
        self.assertEqual(empty["regime"], "empty_mask")
        self.assertEqual(empty["leakage_fraction"], {"num": 0, "den": 1})
        sub = certify_rank1_vs_pruning_exact(u, v, [[1, 0], [0, 0]], s)
        self.assertEqual(sub["regime"], "cartesian_subblock")
        self.assertTrue(sub["is_exact_rank_le_1"])

        non_orth = certify_rank1_vs_pruning_exact(u, v, [[1, 1], [1, 1]], [[Fraction(1, 1), Fraction(0, 1)]])
        self.assertGreater(non_orth["base_max_safety_dot_sq"]["num"], 0)
        self.assertFalse(non_orth["safety_orthogonality_breached"])

        # Each conjunct tested independently (note len("ab") == rows == 2 & len("ab") == cols == 2)
        for bad_u in (123, [], "ab", b"ab"):
            with self.assertRaisesRegex(OracleError, "non-empty sequence"):
                certify_rank1_vs_pruning_exact(bad_u, v, [[1, 1], [1, 1]], s)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "must be Fraction"):
            certify_rank1_vs_pruning_exact([1, 2], v, [[1, 1], [1, 1]], s)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "must be non-zero"):
            certify_rank1_vs_pruning_exact([Fraction(0, 1), Fraction(0, 1)], v, [[1, 1], [1, 1]], s)

        for bad_mask in (123, "ab", b"ab", [[1, 1]]):
            with self.assertRaisesRegex(OracleError, "mask must have 2 rows"):
                certify_rank1_vs_pruning_exact(u, v, bad_mask, s)  # type: ignore[arg-type]
        for bad_row in ([123, [1, 1]], ["ab", [1, 1]], [b"ab", [1, 1]], [[1], [1, 1]]):
            with self.assertRaisesRegex(OracleError, "mask row 0 must have 2 columns"):
                certify_rank1_vs_pruning_exact(u, v, bad_row, s)  # type: ignore[arg-type]
        for bad_cell in ([[2, 0], [0, 1]], [[True, 0], [0, 1]]):
            with self.assertRaisesRegex(OracleError, "exact int 0 or 1"):
                certify_rank1_vs_pruning_exact(u, v, bad_cell, s)  # type: ignore[arg-type]
        for bad_s in (123, "ab", b"ab"):
            with self.assertRaisesRegex(OracleError, "safety_vectors must be a sequence"):
                certify_rank1_vs_pruning_exact(u, v, [[1, 1], [1, 1]], bad_s)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "length"):
            certify_rank1_vs_pruning_exact(u, v, [[1, 1], [1, 1]], [[Fraction(1, 1)]])

    def test_siliquant_quantization_survival_and_double_bind(self) -> None:
        # Exact tie-to-even, strictly-above-half, and saturation clamp boundaries
        self.assertEqual(quantize_sym_rne_exact(0, 0, 7), 0)
        self.assertEqual(quantize_sym_rne_exact(5, 70, 7), 0)   # 0.5 -> even 0
        self.assertEqual(quantize_sym_rne_exact(6, 70, 7), 1)   # 0.6 -> 1 (kills twice_rem > max_abs)
        self.assertEqual(quantize_sym_rne_exact(2, 10, 7), 1)   # 1.4 -> 1 (odd q, twice_rem=8 < 10 -> kills dropping tie check)
        self.assertEqual(quantize_sym_rne_exact(15, 70, 7), 2)  # 1.5 -> even 2
        self.assertEqual(quantize_sym_rne_exact(-15, 70, 7), -2)
        self.assertEqual(quantize_sym_rne_exact(100, 70, 7), 7)  # saturated clamp
        self.assertEqual(quantize_sym_rne_exact(-100, 70, 7), -7)

        with self.assertRaisesRegex(OracleError, "val_int must be exact int"):
            quantize_sym_rne_exact(True, 10, 7)  # type: ignore[arg-type]
        with self.assertRaises(OracleError):
            quantize_sym_rne_exact(1, -1, 7)
        with self.assertRaises(OracleError):
            quantize_sym_rne_exact(1, 10, 0)

        db = run_siliquant_double_bind_witness()
        self.assertTrue(db["double_bind_holds"])
        self.assertEqual(db["regime_a_outlier_erasure_g32"]["erased_edits"], 1)
        self.assertEqual(db["regime_a_outlier_erasure_g32"]["below_three_quarter_ulp"], 1)
        self.assertEqual(db["regime_a_outlier_rotated_recovery_g32"]["survived_code_flips"], 1)
        self.assertEqual(db["regime_b_sparse_spike_collateral_zeroing_g32"]["collateral_unedited_zeroed"], 31)

        # All-zero group edge & fail-closed validation on every conjunct
        z_res = certify_group_quantization_survival_exact([0, 0], [0, 0], qmax=7)
        self.assertEqual(z_res["edited_count"], 0)
        self.assertEqual(z_res["scale_distortion_ratio"], {"num": 0, "den": 1})

        for bad_bg in (123, [], "ab", b"ab"):
            with self.assertRaisesRegex(OracleError, "base_group must be a non-empty"):
                certify_group_quantization_survival_exact(bad_bg, [0, 0], qmax=7)  # type: ignore[arg-type]
        for bad_eg in (123, "ab", b"ab", [0]):
            with self.assertRaisesRegex(OracleError, "edited_group must match"):
                certify_group_quantization_survival_exact([0, 0], bad_eg, qmax=7)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "group entry 0 must be exact int"):
            certify_group_quantization_survival_exact(["x", 0], [0, 0], qmax=7)  # type: ignore[list-item]
        with self.assertRaisesRegex(OracleError, "group entry 1 must be exact int"):
            certify_group_quantization_survival_exact([0, 0], [0, "x"], qmax=7)  # type: ignore[list-item]

    def test_stage2_fixed_radix_certificate_pass_reject_and_guards(self) -> None:
        base_sha = hashlib.sha256(b"base").hexdigest()
        ad_sha = hashlib.sha256(b"adapter").hexdigest()
        cert = make_fixed_radix_certificate(
            adapter_id="ok_adapter",
            adapter_sha256=ad_sha,
            base_weights_sha256=base_sha,
            radix_scale=1 << 16,
            frobenius_sq=Fraction(3, 1),
            spectral_norm_sq=Fraction(2, 1),
            leakage_energy=Fraction(1, 2000),
            max_safety_dot_sq=Fraction(1, 20000),
        )
        ver = verify_fixed_radix_stage2_division_free(
            cert,
            DEFAULT_STAGE2_BOUNDS,
            expected_base_weights_sha256=base_sha,
        )
        self.assertTrue(ver["passed"])
        self.assertEqual(ver["violations"], [])

        # Pure-orthogonal boundary where leak == frob AND safe == spec kills > -> >= on all 3 consistency inequalities!
        exact_eq_cert = make_fixed_radix_certificate(
            adapter_id="exact_eq",
            adapter_sha256=ad_sha,
            base_weights_sha256=base_sha,
            radix_scale=10000,
            frobenius_sq=Fraction(1, 10000),
            spectral_norm_sq=Fraction(1, 10000),
            leakage_energy=Fraction(1, 10000),
            max_safety_dot_sq=Fraction(1, 10000),
        )
        self.assertTrue(verify_fixed_radix_stage2_division_free(exact_eq_cert, DEFAULT_STAGE2_BOUNDS)["passed"])

        # Scale pushes frobenius over bound
        ver_scaled = verify_fixed_radix_stage2_division_free(
            cert,
            DEFAULT_STAGE2_BOUNDS,
            scale_num=2,
            scale_den=1,
            expected_base_weights_sha256=base_sha,
        )
        self.assertFalse(ver_scaled["passed"])
        self.assertIn("frobenius_sq", ver_scaled["violations"])

        # Exact upper bound hit
        boundary_exact = make_fixed_radix_certificate(
            adapter_id="boundary_exact",
            adapter_sha256=ad_sha,
            base_weights_sha256=base_sha,
            radix_scale=10000,
            frobenius_sq=Fraction(10, 1),
            spectral_norm_sq=Fraction(10, 1),
            leakage_energy=Fraction(1, 1000),
            max_safety_dot_sq=Fraction(1, 10000),
        )
        self.assertTrue(verify_fixed_radix_stage2_division_free(boundary_exact, DEFAULT_STAGE2_BOUNDS)["passed"])

        # Forged cert_sha256 & wrong base_sha & non-str sha256
        forged = dict(cert, frobenius_sq_scaled=cert["frobenius_sq_scaled"] + 1)
        with self.assertRaisesRegex(OracleError, "forged"):
            verify_fixed_radix_stage2_division_free(forged, DEFAULT_STAGE2_BOUNDS)
        with self.assertRaisesRegex(OracleError, "base_weights_sha256 mismatch"):
            verify_fixed_radix_stage2_division_free(cert, DEFAULT_STAGE2_BOUNDS, expected_base_weights_sha256=hashlib.sha256(b"other").hexdigest())
        with self.assertRaisesRegex(OracleError, "sha256"):
            verify_fixed_radix_stage2_division_free(cert, DEFAULT_STAGE2_BOUNDS, expected_base_weights_sha256=123)  # type: ignore[arg-type]

        # make_fixed_radix_certificate guards (non-str adapter_id, blank adapter_id, int not Fraction, negative, spec>frob, leak>frob, safe>spec)
        for bad_aid in (123, " "):
            with self.assertRaisesRegex(OracleError, "adapter_id"):
                make_fixed_radix_certificate(
                    adapter_id=bad_aid,  # type: ignore[arg-type]
                    adapter_sha256=ad_sha,
                    base_weights_sha256=base_sha,
                    radix_scale=16,
                    frobenius_sq=Fraction(1, 1),
                    spectral_norm_sq=Fraction(1, 1),
                    leakage_energy=Fraction(0, 1),
                    max_safety_dot_sq=Fraction(0, 1),
                )
        with self.assertRaisesRegex(OracleError, "non-negative Fraction"):
            make_fixed_radix_certificate(
                adapter_id="int_not_frac",
                adapter_sha256=ad_sha,
                base_weights_sha256=base_sha,
                radix_scale=16,
                frobenius_sq=1,  # type: ignore[arg-type]
                spectral_norm_sq=Fraction(1, 1),
                leakage_energy=Fraction(0, 1),
                max_safety_dot_sq=Fraction(0, 1),
            )
        with self.assertRaisesRegex(OracleError, "non-negative Fraction"):
            make_fixed_radix_certificate(
                adapter_id="neg_all",
                adapter_sha256=ad_sha,
                base_weights_sha256=base_sha,
                radix_scale=16,
                frobenius_sq=Fraction(-1, 1),
                spectral_norm_sq=Fraction(-2, 1),
                leakage_energy=Fraction(-2, 1),
                max_safety_dot_sq=Fraction(-2, 1),
            )
        with self.assertRaisesRegex(OracleError, "spectral_norm_sq_scaled"):
            make_fixed_radix_certificate(
                adapter_id="bad_spec",
                adapter_sha256=ad_sha,
                base_weights_sha256=base_sha,
                radix_scale=16,
                frobenius_sq=Fraction(1, 1),
                spectral_norm_sq=Fraction(2, 1),
                leakage_energy=Fraction(0, 1),
                max_safety_dot_sq=Fraction(0, 1),
            )
        with self.assertRaisesRegex(OracleError, "leakage_energy_scaled"):
            make_fixed_radix_certificate(
                adapter_id="bad_leak",
                adapter_sha256=ad_sha,
                base_weights_sha256=base_sha,
                radix_scale=16,
                frobenius_sq=Fraction(1, 1),
                spectral_norm_sq=Fraction(1, 1),
                leakage_energy=Fraction(2, 1),
                max_safety_dot_sq=Fraction(0, 1),
            )
        with self.assertRaisesRegex(OracleError, "max_safety_dot_sq_scaled"):
            make_fixed_radix_certificate(
                adapter_id="bad_safe",
                adapter_sha256=ad_sha,
                base_weights_sha256=base_sha,
                radix_scale=16,
                frobenius_sq=Fraction(2, 1),
                spectral_norm_sq=Fraction(1, 1),
                leakage_energy=Fraction(0, 1),
                max_safety_dot_sq=Fraction(2, 1),
            )

        # Tampered-with-valid-digest checks on verify_fixed_radix_stage2_division_free
        for bad_aid in (123, "   "):
            b = {
                "adapter_id": bad_aid,
                "adapter_sha256": ad_sha,
                "base_weights_sha256": base_sha,
                "radix_scale": 16,
                "frobenius_sq_scaled": 10,
                "spectral_norm_sq_scaled": 10,
                "leakage_energy_scaled": 0,
                "max_safety_dot_sq_scaled": 0,
            }
            b["cert_sha256"] = canonical_sha256(b)
            with self.assertRaisesRegex(OracleError, "adapter_id"):
                verify_fixed_radix_stage2_division_free(b, DEFAULT_STAGE2_BOUNDS)

        for field_name, val, msg in (
            ("spectral_norm_sq_scaled", 11, "spectral > frobenius"),
            ("leakage_energy_scaled", 11, "leakage > frobenius"),
            ("max_safety_dot_sq_scaled", 11, "safety_dot_sq > spectral"),
        ):
            b = {
                "adapter_id": "bad_ineq",
                "adapter_sha256": ad_sha,
                "base_weights_sha256": base_sha,
                "radix_scale": 16,
                "frobenius_sq_scaled": 10,
                "spectral_norm_sq_scaled": 10,
                "leakage_energy_scaled": 0,
                "max_safety_dot_sq_scaled": 0,
            }
            b[field_name] = val
            b["cert_sha256"] = canonical_sha256(b)
            with self.assertRaisesRegex(OracleError, msg):
                verify_fixed_radix_stage2_division_free(b, DEFAULT_STAGE2_BOUNDS)

        with self.assertRaisesRegex(OracleError, "cert must be a mapping"):
            verify_fixed_radix_stage2_division_free(123, DEFAULT_STAGE2_BOUNDS)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "bounds must be a mapping"):
            verify_fixed_radix_stage2_division_free(cert, 123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "missing certificate field"):
            verify_fixed_radix_stage2_division_free({k: v for k, v in cert.items() if k != "radix_scale"}, DEFAULT_STAGE2_BOUNDS)
        with self.assertRaisesRegex(OracleError, "unexpected extra field in certificate"):
            verify_fixed_radix_stage2_division_free(dict(cert, smuggled=1), DEFAULT_STAGE2_BOUNDS)
        with self.assertRaisesRegex(OracleError, "missing bounds field"):
            verify_fixed_radix_stage2_division_free(cert, {"max_frobenius_sq": {"num": 1, "den": 1}})
        with self.assertRaisesRegex(OracleError, "unexpected extra field in bounds"):
            verify_fixed_radix_stage2_division_free(cert, dict(DEFAULT_STAGE2_BOUNDS, extra={"num": 1, "den": 1}))


class TestPairedMcNemarAndGovernance(unittest.TestCase):
    def test_mcnemar_and_dual_screen_gate(self) -> None:
        self.assertEqual(exact_mcnemar_two_sided_fraction(0, 0), Fraction(1, 1))
        self.assertEqual(exact_mcnemar_two_sided_fraction(1, 1), Fraction(1, 1))
        self.assertEqual(exact_mcnemar_two_sided_fraction(10, 0), Fraction(2, 1024))
        self.assertLessEqual(exact_mcnemar_two_sided_fraction(43, 21), Fraction(1, 100))

        # Exact 4-cell contingency matrix check kills EVERY And conjunct in the 4-way partition!
        b4 = [
            {"id": "p0", "correct": True, "finish_reason": "length"},
            {"id": "p1", "correct": False, "finish_reason": "stop"},
            {"id": "p2", "correct": True, "finish_reason": "length"},
            {"id": "p3", "correct": False, "finish_reason": "stop"},
        ]
        c4 = [
            {"id": "p0", "correct": True, "finish_reason": "length"},
            {"id": "p1", "correct": True, "finish_reason": "length"},
            {"id": "p2", "correct": False, "finish_reason": "stop"},
            {"id": "p3", "correct": False, "finish_reason": "stop"},
        ]
        r4 = certify_paired_dual_screen(b4, c4, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        self.assertEqual(r4["acc_contingency"], {"both_ok": 1, "cand_only": 1, "base_only": 1, "neither": 1})
        self.assertEqual(r4["cut_contingency"], {"both_cut": 1, "cand_more_cut": 1, "cand_less_cut": 1, "neither_cut": 1})
        self.assertTrue(r4["conservation_exact"])
        self.assertTrue(r4["passed_dual_gate"])

        base = [{"id": f"q{i}", "correct": i < 12, "finish_reason": "stop" if i < 14 else "length"} for i in range(16)]
        worse_acc = [{"id": f"q{i}", "correct": i < 2, "finish_reason": "stop" if i < 14 else "length"} for i in range(16)]
        worse_cut = [{"id": f"q{i}", "correct": i < 12, "finish_reason": "length" if (i < 12 or i >= 14) else "stop"} for i in range(16)]
        better = [{"id": f"q{i}", "correct": i < 15, "finish_reason": "stop"} for i in range(16)]

        r_acc = certify_paired_dual_screen(base, worse_acc, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        self.assertFalse(r_acc["passed_dual_gate"])
        self.assertIn("paired_accuracy_regression", r_acc["failed_gates"])

        r_cut = certify_paired_dual_screen(base, worse_cut, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        self.assertFalse(r_cut["passed_dual_gate"])
        self.assertIn("paired_cut_rate_regression", r_cut["failed_gates"])

        r_ok = certify_paired_dual_screen(base, better, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        self.assertTrue(r_ok["passed_dual_gate"])
        self.assertEqual(r_ok["candidate_finished"], 16)
        self.assertEqual(r_ok["candidate_finished_correct"], 15)

        # Significant improvement in BOTH accuracy and cut rate must PASS (kills direction conjuncts!)
        weak_base = [{"id": f"p{i}", "correct": i < 2, "finish_reason": "length" if i < 14 else "stop"} for i in range(16)]
        sig_better = [{"id": f"p{i}", "correct": i < 14, "finish_reason": "stop"} for i in range(16)]
        r_sig_better = certify_paired_dual_screen(weak_base, sig_better, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        self.assertTrue(r_sig_better["passed_dual_gate"])

        # Non-significant 1-item dip in BOTH accuracy and cut rate must PASS (kills p-value conjuncts!)
        noisy_worse = [{"id": f"q{i}", "correct": i < 11, "finish_reason": "stop" if i < 13 else "length"} for i in range(16)]
        r_noisy = certify_paired_dual_screen(base, noisy_worse, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        self.assertTrue(r_noisy["passed_dual_gate"])

        for bad_p in (Fraction(0, 1), Fraction(2, 1), 1):
            with self.assertRaises(OracleError):
                certify_paired_dual_screen(base, better, max_cut_regression_p=bad_p, min_acc_drop_p=Fraction(1, 20))  # type: ignore[arg-type]
            with self.assertRaises(OracleError):
                certify_paired_dual_screen(base, better, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=bad_p)  # type: ignore[arg-type]

        for bad_seq in (123, [], "ab", b"ab"):
            with self.assertRaisesRegex(OracleError, "non-empty sequence"):
                certify_paired_dual_screen(bad_seq, bad_seq, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "must be a mapping"):
            certify_paired_dual_screen([123], [123], max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))  # type: ignore[arg-type]
        for bad_row in (
            {"correct": True, "finish_reason": "stop"},
            {"id": "q0", "finish_reason": "stop"},
            {"id": "q0", "correct": True},
        ):
            with self.assertRaisesRegex(OracleError, "missing required keys"):
                certify_paired_dual_screen([bad_row], [bad_row], max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        for bad_pid in (123, " "):
            r = {"id": bad_pid, "correct": True, "finish_reason": "stop"}
            with self.assertRaisesRegex(OracleError, "non-empty str"):
                certify_paired_dual_screen([r], [r], max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        dup = [{"id": "q0", "correct": True, "finish_reason": "stop"}, {"id": "q0", "correct": True, "finish_reason": "stop"}]
        with self.assertRaisesRegex(OracleError, "duplicate problem id"):
            certify_paired_dual_screen(dup, dup, max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        with self.assertRaisesRegex(OracleError, "exact bool"):
            certify_paired_dual_screen([{"id": "q0", "correct": 1, "finish_reason": "stop"}], [{"id": "q0", "correct": 1, "finish_reason": "stop"}], max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        class _SubStr(str):
            pass

        for bad_fr in (123, "other", _SubStr("stop")):
            r = {"id": "q0", "correct": True, "finish_reason": bad_fr}
            with self.assertRaisesRegex(OracleError, "finish_reason"):
                certify_paired_dual_screen([r], [r], max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))
        with self.assertRaisesRegex(OracleError, "do not match"):
            certify_paired_dual_screen(base, [{"id": "other", "correct": True, "finish_reason": "stop"}], max_cut_regression_p=Fraction(1, 100), min_acc_drop_p=Fraction(1, 20))

    def test_self_rewriting_governance_gate_all_8_guards(self) -> None:
        base_sha = hashlib.sha256(b"base-27b").hexdigest()
        parent_sha = hashlib.sha256(b"parent-step1").hexdigest()
        good = {
            "session_id": "compliant",
            "sandbox_confined": True,
            "external_verifier_decides_submission": True,
            "base_weights_sha256": base_sha,
            "parent_weights_sha256": parent_sha,
            "session_index": 2,
            "cumulative_edit_claimed": True,
            "paired_heldout_control_present": True,
            "heldout_problem_count": 200,
            "practice_overlap_count": 0,
            "collateral_ablation_checked": True,
            "unnecessary_tensor_count": 0,
            "safety_canary_passed": 4,
            "safety_canary_total": 4,
            "unrelated_kl_shift": {"num": 1, "den": 20},
            "has_recurrent_state_banks": True,
            "state_banks_flushed_and_confirmed": True,
        }
        v_good = certify_self_rewriting_session(good, DEFAULT_GOVERNANCE_POLICY)
        self.assertTrue(v_good["accepted"])
        self.assertEqual(v_good["failed_gates"], [])

        # Non-cumulative step-2 session starting from base is accepted (kills cumulative_claimed conjunct!)
        non_cum = dict(good, cumulative_edit_claimed=False, parent_weights_sha256=base_sha)
        self.assertTrue(certify_self_rewriting_session(non_cum, DEFAULT_GOVERNANCE_POLICY)["accepted"])

        trips = [
            ({"sandbox_confined": False}, "G1_sandbox_confinement"),
            ({"external_verifier_decides_submission": False}, "G2_external_immutable_verifier"),
            ({"parent_weights_sha256": base_sha}, "G3_cumulative_lineage_reset_mismatch"),
            ({"paired_heldout_control_present": False}, "G4_paired_clean_heldout"),
            ({"heldout_problem_count": 199}, "G4_paired_clean_heldout"),
            ({"practice_overlap_count": 1}, "G4_paired_clean_heldout"),
            ({"collateral_ablation_checked": False}, "G5_collateral_ablation_bound"),
            ({"unnecessary_tensor_count": 1}, "G5_collateral_ablation_bound"),
            ({"safety_canary_passed": 3}, "G6_safety_canary_floor"),
            ({"safety_canary_passed": 0, "safety_canary_total": 0}, "G6_safety_canary_floor"),
            ({"unrelated_kl_shift": {"num": 51, "den": 1000}}, "G7_unrelated_kl_latch"),
            ({"state_banks_flushed_and_confirmed": False}, "G8_hybrid_state_bank_flush"),
        ]
        for patch, expected_gate in trips:
            mutated = dict(good)
            mutated.update(patch)
            ver = certify_self_rewriting_session(mutated, DEFAULT_GOVERNANCE_POLICY)
            self.assertFalse(ver["accepted"], msg=f"expected {expected_gate} to fail for {patch}")
            self.assertIn(expected_gate, ver["failed_gates"])

        step1 = dict(good, session_index=1, parent_weights_sha256=base_sha, has_recurrent_state_banks=False, state_banks_flushed_and_confirmed=False)
        self.assertTrue(certify_self_rewriting_session(step1, DEFAULT_GOVERNANCE_POLICY)["accepted"])

        # Non-shadowed type/schema/conservation rejection checks
        with self.assertRaisesRegex(OracleError, "session must be a mapping"):
            certify_self_rewriting_session(123, DEFAULT_GOVERNANCE_POLICY)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "policy must be a mapping"):
            certify_self_rewriting_session(good, 123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(OracleError, "exact bool"):
            certify_self_rewriting_session(dict(good, sandbox_confined=1), DEFAULT_GOVERNANCE_POLICY)
        with self.assertRaisesRegex(OracleError, "missing session field"):
            certify_self_rewriting_session({k: v for k, v in good.items() if k != "session_id"}, DEFAULT_GOVERNANCE_POLICY)
        with self.assertRaisesRegex(OracleError, "unexpected extra field in session"):
            certify_self_rewriting_session(dict(good, extra=1), DEFAULT_GOVERNANCE_POLICY)
        with self.assertRaisesRegex(OracleError, "missing policy field"):
            certify_self_rewriting_session(good, {k: v for k, v in DEFAULT_GOVERNANCE_POLICY.items() if k != "max_unrelated_kl_shift"})
        with self.assertRaisesRegex(OracleError, "unexpected extra field in policy"):
            certify_self_rewriting_session(good, dict(DEFAULT_GOVERNANCE_POLICY, extra=1))
        for bad_sid in (123, " "):
            with self.assertRaisesRegex(OracleError, "session_id"):
                certify_self_rewriting_session(dict(good, session_id=bad_sid), DEFAULT_GOVERNANCE_POLICY)
        with self.assertRaisesRegex(OracleError, "practice_overlap_count cannot exceed"):
            certify_self_rewriting_session(dict(good, heldout_problem_count=10, practice_overlap_count=11), DEFAULT_GOVERNANCE_POLICY)
        eq_ov = certify_self_rewriting_session(dict(good, heldout_problem_count=10, practice_overlap_count=10), DEFAULT_GOVERNANCE_POLICY)
        self.assertFalse(eq_ov["accepted"])
        self.assertIn("G4_paired_clean_heldout", eq_ov["failed_gates"])
        with self.assertRaisesRegex(OracleError, "cannot exceed safety_canary_total"):
            certify_self_rewriting_session(dict(good, safety_canary_passed=5, safety_canary_total=4), DEFAULT_GOVERNANCE_POLICY)
        with self.assertRaisesRegex(OracleError, "cannot exceed 1"):
            certify_self_rewriting_session(good, dict(DEFAULT_GOVERNANCE_POLICY, min_safety_canary_rate={"num": 2, "den": 1}))
        with self.assertRaisesRegex(OracleError, "sha256"):
            certify_self_rewriting_session(dict(good, base_weights_sha256="not-a-sha"), DEFAULT_GOVERNANCE_POLICY)

        gov_all = run_arxiv_governance_audit(base_sha)
        self.assertFalse(gov_all["verdicts"]["arxiv_2611_21885_proto1_27b_self_edit"]["accepted"])
        self.assertFalse(gov_all["verdicts"]["arxiv_2611_21885_proto1_27b_damaged_repair"]["accepted"])
        self.assertFalse(gov_all["verdicts"]["arxiv_2611_21885_proto2_session8_dt_bias"]["accepted"])
        self.assertTrue(gov_all["verdicts"]["warp_drive_aletheia_v2_compliant_session"]["accepted"])


class TestAuditRunnerValidatorAndParsers(unittest.TestCase):
    def test_npy_and_jsonl_and_sha256_parsers(self) -> None:
        raw_v1 = _encode_npy("<u2", (2,), struct.pack("<2H", 0x3F80, 0x4000), version=1)
        self.assertEqual(parse_npy_bytes(raw_v1)[:2], ("<u2", (2,)))

        raw_empty = _encode_npy("<u2", (0,), b"", version=1)
        self.assertEqual(parse_npy_bytes(raw_empty), ("<u2", (0,), b""))

        raw_v2 = _encode_npy("<f4", (1, 2), struct.pack("<2I", 0x3F800000, 0x40000000), version=2)
        self.assertEqual(parse_npy_bytes(raw_v2)[:2], ("<f4", (1, 2)))

        for bad_magic in (123, b"\x93NUMPY\x01\x00\x00", b"BADMAGIC123456"):
            with self.assertRaisesRegex(OracleError, "invalid .npy magic"):
                parse_npy_bytes(bad_magic)  # type: ignore[arg-type]
        for bad_v2 in (b"\x93NUMPY\x02\x00\x00\x00", b"\x93NUMPY\x02\x00\x00\x00\x00\x00"):
            with self.assertRaisesRegex(OracleError, "truncated .npy v2/v3"):
                parse_npy_bytes(bad_v2)
        with self.assertRaisesRegex(OracleError, "unsupported .npy major"):
            parse_npy_bytes(b"\x93NUMPY\x04\x00\x00\x00\x00\x00")
        with self.assertRaisesRegex(OracleError, "truncated .npy header payload"):
            parse_npy_bytes(b"\x93NUMPY\x01\x00\xff\x00abc")

        # Each conjunct of header dict check tested independently (list containing both keys kills isinstance(meta, dict)!)
        non_dict_hdr = b"['descr', 'shape']  \n"
        with self.assertRaisesRegex(OracleError, "malformed .npy header"):
            parse_npy_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(non_dict_hdr)) + non_dict_hdr)
        with self.assertRaisesRegex(OracleError, "malformed .npy header"):
            parse_npy_bytes(_encode_npy("<u2", (1,), b"\x00\x00").replace(b"'descr'", b"'other'"))
        with self.assertRaisesRegex(OracleError, "malformed .npy header"):
            parse_npy_bytes(_encode_npy("<u2", (1,), b"\x00\x00").replace(b"'shape'", b"'other'"))

        with self.assertRaisesRegex(OracleError, "unsupported IEEE dtype"):
            _decode_ieee_tensor_scaled_f64("<i4", b"\x00\x00\x00\x00", "bad")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            jf = p / "sample.jsonl"
            jf.write_text('{"id":"a","correct":[1],"finish":["stop"]}\n\n', encoding="utf-8")
            self.assertEqual(len(load_jsonl_rows(jf)), 1)
            digest, sz = sha256_file(jf)
            self.assertEqual(len(digest), 64)
            self.assertGreater(sz, 0)

            bad_jf = p / "bad.jsonl"
            bad_jf.write_text("[1,2,3]\n", encoding="utf-8")
            with self.assertRaisesRegex(OracleError, "not a JSON object"):
                load_jsonl_rows(bad_jf)
            with self.assertRaisesRegex(OracleError, "missing jsonl"):
                load_jsonl_rows(p / "missing.jsonl")
            with self.assertRaisesRegex(OracleError, "missing pinned"):
                sha256_file(p / "missing.bin")

    def test_certificate_validator_anti_bypass_and_cli(self) -> None:
        sa = run_standalone_audit()
        self.assertTrue(sa["overall_verified"])
        self.assertEqual(validate_audit_certificate(sa)["valid"], True)

        def _rehash(d: dict) -> dict:
            out = dict(d)
            out.pop("certificate_sha256", None)
            out["certificate_sha256"] = canonical_sha256(out)
            return out

        # Red-team probes matching AUDIT-098/099/100 verifier suite
        with self.assertRaisesRegex(OracleError, "mapping"):
            validate_audit_certificate(123)  # type: ignore[arg-type]
        float_smuggle = json.loads(json.dumps(sa))
        float_smuggle["stage2_bounds"]["max_frobenius_sq"]["num"] = json.loads("10.5")
        with self.assertRaisesRegex(OracleError, "floating-point"):
            validate_audit_certificate(_rehash(float_smuggle))
        with self.assertRaisesRegex(OracleError, "mode"):
            validate_audit_certificate({"schema_version": "x"})
        with self.assertRaisesRegex(OracleError, "mode"):
            validate_audit_certificate(_rehash(dict(sa, mode="invalid")))
        with self.assertRaisesRegex(OracleError, "key set mismatch"):
            validate_audit_certificate(dict(sa, forged_claim="N2_BREAKTHROUGH"))
        with self.assertRaisesRegex(OracleError, "schema_version"):
            validate_audit_certificate(_rehash(dict(sa, schema_version="wrong")))
        with self.assertRaisesRegex(OracleError, "audit_id"):
            validate_audit_certificate(_rehash(dict(sa, audit_id="AUDIT-999")))
        with self.assertRaisesRegex(OracleError, "novelty_class"):
            validate_audit_certificate(_rehash(dict(sa, novelty_class="N2")))
        for bad_live_flag in (True, 0):
            with self.assertRaisesRegex(OracleError, "live_intake_audited"):
                validate_audit_certificate(_rehash(dict(sa, live_intake_audited=bad_live_flag)))
        for bad_ver in (False, 1):
            with self.assertRaisesRegex(OracleError, "overall_verified"):
                validate_audit_certificate(_rehash(dict(sa, overall_verified=bad_ver)))
        for bad_ovf in (False, 1):
            with self.assertRaisesRegex(OracleError, "st_numpy_bf16_overflow_blocked"):
                validate_audit_certificate(_rehash(dict(sa, st_numpy_bf16_overflow_blocked=bad_ovf)))
        for bad_sha in (123, "NOT_A_SHA", "0" * 64):
            with self.assertRaisesRegex(OracleError, "non-degenerate 64-char lowercase"):
                validate_audit_certificate(dict(sa, certificate_sha256=bad_sha))
        with self.assertRaisesRegex(OracleError, "digest mismatch"):
            validate_audit_certificate(dict(sa, certificate_sha256="01" * 32))

        # WSP097-VAL-01..05 anti-bypass probes on standalone AND committed live certificate
        for bad_sb in (123, {}, dict(sa["stage2_bounds"], max_frobenius_sq={"num": 999999999, "den": 1}), dict(sa["stage2_bounds"], max_frobenius_sq={"num": 10, "den": True})):
            with self.assertRaisesRegex(OracleError, "stage2_bounds"):
                validate_audit_certificate(_rehash(dict(sa, stage2_bounds=bad_sb)))

        # WSP097-VAL-01: hollow sub-dicts & flipped booleans in exact_pruning_theorems & siliquant
        hollow_th = dict(sa["exact_pruning_theorems"], full_rank1={})
        for bad_th in (123, {"theorem_holds": True}, hollow_th):
            with self.assertRaisesRegex(OracleError, "exact_pruning_theorems"):
                validate_audit_certificate(_rehash(dict(sa, exact_pruning_theorems=bad_th)))
        hollow_db = dict(sa["siliquant_quantization_bridge"]["synthetic_double_bind"], regime_a_outlier_erasure_g32={})
        for bad_sq_keys in (123, dict(sa["siliquant_quantization_bridge"], forged_extra=1)):
            with self.assertRaisesRegex(OracleError, "siliquant_quantization_bridge key set"):
                validate_audit_certificate(_rehash(dict(sa, siliquant_quantization_bridge=bad_sq_keys)))
        for bad_db in (123, {"double_bind_holds": True}, hollow_db):
            with self.assertRaisesRegex(OracleError, "synthetic_double_bind"):
                validate_audit_certificate(_rehash(dict(sa, siliquant_quantization_bridge={"synthetic_double_bind": bad_db})))

        # WSP097-VAL-03 & VAL-05: flipped governance protocols, fake IDs, extra keys, list length mismatch
        gov_flip = json.loads(json.dumps(sa["arxiv_governance_audit"]))
        p1 = gov_flip["verdicts"]["arxiv_2611_21885_proto1_27b_self_edit"]
        p1["accepted"] = True
        p1["failed_gates"] = []
        p1["gate_count_failed"] = 0
        p1.pop("verdict_sha256")
        p1["verdict_sha256"] = canonical_sha256(p1)
        gov_fakes = json.loads(json.dumps(sa["arxiv_governance_audit"]))
        gov_fakes["verdicts"] = {f"fake_{i}": p1 for i in range(4)}
        gov_extra = dict(sa["arxiv_governance_audit"], extra_key=1)
        gov_short_list = json.loads(json.dumps(sa["arxiv_governance_audit"]))
        gov_short_list["verdicts"]["arxiv_2611_21885_proto1_27b_self_edit"]["failed_gates"] = ["G1_sandbox_confinement"]
        for bad_gov in (123, gov_flip, gov_fakes, gov_extra, gov_short_list):
            with self.assertRaisesRegex(OracleError, "arxiv_governance_audit"):
                validate_audit_certificate(_rehash(dict(sa, arxiv_governance_audit=bad_gov)))

        # WSP097-VAL-01: standalone hollow/inverted fixture verdicts
        for bad_s2f in (123, {"a64": {}, "d4k": {}, "c2_min": {}}):
            with self.assertRaisesRegex(OracleError, "stage2_fixture_verdicts"):
                validate_audit_certificate(_rehash(dict(sa, stage2_fixture_verdicts=bad_s2f)))
        for bad_pds in (123, {"passed_dual_gate": True}):
            with self.assertRaisesRegex(OracleError, "paired_dual_screen_fixture"):
                validate_audit_certificate(_rehash(dict(sa, paired_dual_screen_fixture=bad_pds)))

        # WSP097-VAL-02 & VAL-05: committed live certificate validation + EVERY re-signed live bypass blocked!
        from wsp_self_rewriting_audit import PINNED_INTAKE_FILES

        live_ok = json.loads((Path(__file__).resolve().parent / "wsp_self_rewriting_certificate.json").read_text(encoding="utf-8"))
        self.assertTrue(validate_audit_certificate(live_ok)["valid"])

        for bad_pa in ({}, 123):
            with self.assertRaisesRegex(OracleError, "pinned_artifacts"):
                validate_audit_certificate(_rehash(dict(live_ok, pinned_artifacts=bad_pa)))
        first_rel = PINNED_INTAKE_FILES[0]
        for bad_meta in (123, dict(live_ok["pinned_artifacts"][first_rel], extra=1)):
            pa_mut = dict(live_ok["pinned_artifacts"], **{first_rel: bad_meta})
            with self.assertRaisesRegex(OracleError, "invalid pinned_artifacts entry"):
                validate_audit_certificate(_rehash(dict(live_ok, pinned_artifacts=pa_mut)))
        pa_bad_sha = dict(live_ok["pinned_artifacts"], **{first_rel: {"sha256": "0" * 64, "bytes": 1}})
        with self.assertRaisesRegex(OracleError, "sha256"):
            validate_audit_certificate(_rehash(dict(live_ok, pinned_artifacts=pa_bad_sha)))
        pa_bad_sz = dict(live_ok["pinned_artifacts"], **{first_rel: {"sha256": "ab" * 32, "bytes": 0}})
        with self.assertRaisesRegex(OracleError, "positive"):
            validate_audit_certificate(_rehash(dict(live_ok, pinned_artifacts=pa_bad_sz)))

        def _mut_live(mutator) -> dict:
            c = json.loads(json.dumps(live_ok))
            mutator(c)
            return _rehash(c)

        # exp8_tensor_audit probes
        for fn in (
            lambda c: c.__setitem__("exp8_tensor_audit", 123),
            lambda c: c.__setitem__("exp8_tensor_audit", {"specs": {"x": 1}, "stage2_verdicts": {"y": 1}}),
            lambda c: c["exp8_tensor_audit"].__setitem__("actual_nonzero_bf16_entries", 999),
            lambda c: c["exp8_tensor_audit"].__setitem__("bf16_noop_collapse_count", 0),
            lambda c: c["exp8_tensor_audit"].__setitem__("specs", 123),
            lambda c: c["exp8_tensor_audit"].__setitem__("specs", {"A64_topk64": {}}),
            lambda c: c["exp8_tensor_audit"].__setitem__("stage2_verdicts", 123),
            lambda c: c["exp8_tensor_audit"]["stage2_verdicts"].__setitem__("extra", 1),
            lambda c: c["exp8_tensor_audit"]["stage2_verdicts"].__setitem__("A64_topk64", 123),
            lambda c: c["exp8_tensor_audit"]["stage2_verdicts"].__setitem__("A64_topk64", {}),
            lambda c: c["exp8_tensor_audit"]["stage2_verdicts"]["A64_topk64"].__setitem__("passed", False),
            lambda c: c["exp8_tensor_audit"]["stage2_verdicts"]["A64_topk64"].__setitem__("violations", ""),
            lambda c: c["exp8_tensor_audit"]["stage2_verdicts"]["A64_topk64"].__setitem__("violations", ["leakage_energy"]),
            lambda c: c["exp8_tensor_audit"].__setitem__("siliquant_live_exp8_int4_survival", {}),
            lambda c: (c["siliquant_quantization_bridge"].__setitem__("live_exp8_down_proj_int4_survival", 123), c["exp8_tensor_audit"].__setitem__("siliquant_live_exp8_int4_survival", 123)),
            lambda c: (c["siliquant_quantization_bridge"]["live_exp8_down_proj_int4_survival"].__setitem__("extra", 1), c["exp8_tensor_audit"]["siliquant_live_exp8_int4_survival"].__setitem__("extra", 1)),
            lambda c: (c["siliquant_quantization_bridge"]["live_exp8_down_proj_int4_survival"].__setitem__("layer_15", 123), c["exp8_tensor_audit"]["siliquant_live_exp8_int4_survival"].__setitem__("layer_15", 123)),
            lambda c: (c["siliquant_quantization_bridge"]["live_exp8_down_proj_int4_survival"]["layer_15"].__setitem__("extra", 1), c["exp8_tensor_audit"]["siliquant_live_exp8_int4_survival"]["layer_15"].__setitem__("extra", 1)),
            lambda c: (c["siliquant_quantization_bridge"]["live_exp8_down_proj_int4_survival"]["layer_15"]["G128"].__setitem__("erased_edits", 20000), c["exp8_tensor_audit"]["siliquant_live_exp8_int4_survival"]["layer_15"]["G128"].__setitem__("erased_edits", 20000)),
        ):
            with self.assertRaises(OracleError):
                validate_audit_certificate(_mut_live(fn))

        # exp8_screen_audit probes
        for fn in (
            lambda c: c.__setitem__("exp8_screen_audit", 123),
            lambda c: c["exp8_screen_audit"].__setitem__("extra", 1),
            lambda c: c.__setitem__("exp8_screen_audit", {"paired_vs_a64": {"x": 1}, "validate_d4k_math500": {"y": 1}}),
            lambda c: c["exp8_screen_audit"].__setitem__("paired_vs_a64", 123),
            lambda c: c["exp8_screen_audit"]["paired_vs_a64"].__setitem__("extra", 1),
            lambda c: c["exp8_screen_audit"]["paired_vs_a64"].__setitem__("D4k_entries", 123),
            lambda c: c["exp8_screen_audit"]["paired_vs_a64"]["D4k_entries"].__setitem__("passed_dual_gate", True),
            lambda c: c["exp8_screen_audit"].__setitem__("validate_d4k_math500", 123),
            lambda c: c["exp8_screen_audit"].__setitem__("validate_d4k_math500", {"total_correct": 1084}),
            lambda c: c["exp8_screen_audit"]["validate_d4k_math500"].__setitem__("total_correct", 1000),
            lambda c: c["exp8_screen_audit"]["validate_d4k_math500"].__setitem__("paired_control_missing", False),
        ):
            with self.assertRaises(OracleError):
                validate_audit_certificate(_mut_live(fn))

        # car_ramp_audit probes
        for fn in (
            lambda c: c.__setitem__("car_ramp_audit", 123),
            lambda c: c.__setitem__("car_ramp_audit", {"checkpoints": {"x": 1}, "stage2_verdicts": {"y": 1}}),
            lambda c: c["car_ramp_audit"].__setitem__("total_actor_params", 100),
            lambda c: c["car_ramp_audit"].__setitem__("checkpoints", 123),
            lambda c: c["car_ramp_audit"].__setitem__("checkpoints", {"edited_actor": {}}),
            lambda c: c["car_ramp_audit"].__setitem__("stage2_verdicts", 123),
            lambda c: c["car_ramp_audit"]["stage2_verdicts"].__setitem__("extra", 1),
            lambda c: c["car_ramp_audit"].__setitem__("siliquant_live_spike_collapse", {}),
            lambda c: (c["siliquant_quantization_bridge"].__setitem__("live_car_ramp_spike_collapse", 123), c["car_ramp_audit"].__setitem__("siliquant_live_spike_collapse", 123)),
            lambda c: (c["siliquant_quantization_bridge"]["live_car_ramp_spike_collapse"].__setitem__("extra", 1), c["car_ramp_audit"]["siliquant_live_spike_collapse"].__setitem__("extra", 1)),
            lambda c: (c["siliquant_quantization_bridge"]["live_car_ramp_spike_collapse"].__setitem__("c2_min_10entry_actor_head_W", 123), c["car_ramp_audit"]["siliquant_live_spike_collapse"].__setitem__("c2_min_10entry_actor_head_W", 123)),
            lambda c: (c["siliquant_quantization_bridge"]["live_car_ramp_spike_collapse"]["c2_min_10entry_actor_head_W"].__setitem__("extra", 1), c["car_ramp_audit"]["siliquant_live_spike_collapse"]["c2_min_10entry_actor_head_W"].__setitem__("extra", 1)),
            lambda c: (c["siliquant_quantization_bridge"]["live_car_ramp_spike_collapse"]["c2_min_10entry_actor_head_W"]["G32"].__setitem__("collateral_unedited_zeroed", 28), c["car_ramp_audit"]["siliquant_live_spike_collapse"]["c2_min_10entry_actor_head_W"]["G32"].__setitem__("collateral_unedited_zeroed", 28)),
        ):
            with self.assertRaises(OracleError):
                validate_audit_certificate(_mut_live(fn))

        # CLI standalone + --write-cert + --json
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "cert.json"
            buf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = buf
                rc = audit_main(["--standalone", "--json", "--write-cert", str(out_path)])
            finally:
                sys.stdout = old_stdout
            self.assertEqual(rc, 0)
            self.assertEqual(buf.getvalue(), out_path.read_text(encoding="utf-8"))

            buf_text = io.StringIO()
            try:
                sys.stdout = buf_text
                rc2 = audit_main(["--standalone"])
            finally:
                sys.stdout = old_stdout
            self.assertEqual(rc2, 0)
            self.assertIn("AUDIT-097", buf_text.getvalue())

            with self.assertRaisesRegex(OracleError, "intake directory does not exist"):
                run_live_audit(Path(td) / "no_such_intake")

    def test_known_answer_live_intake_and_fail_closed_guards(self) -> None:
        base_sha = hashlib.sha256(b"mock-base").hexdigest()
        with tempfile.TemporaryDirectory() as td:
            intake = Path(td)
            _build_synthetic_intake(intake)

            live_res = run_live_audit(intake)
            self.assertEqual(live_res["mode"], "live_intake_recount")
            self.assertTrue(live_res["live_intake_audited"])
            self.assertEqual(live_res["exp8_tensor_audit"]["actual_nonzero_bf16_entries"], 6)
            self.assertEqual(live_res["exp8_tensor_audit"]["bf16_noop_collapse_count"], 2)
            self.assertEqual(live_res["exp8_tensor_audit"]["specs"]["A64_topk64"]["total_frob_sq_dec"], "6.000000")
            self.assertEqual(live_res["exp8_tensor_audit"]["specs"]["A64_topk64"]["total_perp_sq_dec"], "0.000000")
            self.assertEqual(live_res["exp8_tensor_audit"]["specs"]["D4k_entries"]["total_perp_sq_dec"], "1.000000")
            self.assertEqual(live_res["exp8_tensor_audit"]["specs"]["ER384_L15"]["off_rank1_percent"], "50.000%")
            self.assertEqual(live_res["exp8_tensor_audit"]["specs"]["ER384_L15"]["total_active_entries"], 2)
            self.assertEqual(live_res["exp8_tensor_audit"]["specs"]["ER768_L15"]["total_active_entries"], 1)
            self.assertEqual(live_res["car_ramp_audit"]["total_actor_params"], 14)
            self.assertEqual(live_res["car_ramp_audit"]["checkpoints"]["c2_min_10entry"]["frob_sq_dec"], "1600.00")
            self.assertFalse(live_res["car_ramp_audit"]["stage2_verdicts"]["c2_min_10entry"]["passed"])

            buf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = buf
                rc_live = audit_main(["--live", "--intake-dir", str(intake)])
            finally:
                sys.stdout = old_stdout
            self.assertEqual(rc_live, 1)

            # Every conjunct of _normalize_screen_rows tested individually
            for missing_key in ({"correct": [1], "finish": ["stop"]}, {"id": "x", "finish": ["stop"]}, {"id": "x", "correct": [1]}):
                with self.assertRaisesRegex(OracleError, "missing required screen keys"):
                    _normalize_screen_rows([missing_key], "t")
            for bad_c in (123, [1, 0], [True], [2]):
                with self.assertRaisesRegex(OracleError, "1-element list of exact int"):
                    _normalize_screen_rows([{"id": "x", "correct": bad_c, "finish": ["stop"]}], "t")
            for bad_f in (123, ["stop", "stop"], [1]):
                with self.assertRaisesRegex(OracleError, "1-element list of str"):
                    _normalize_screen_rows([{"id": "x", "correct": [1], "finish": bad_f}], "t")

            repo = intake / "repos" / "weight-edit-minimization"
            exp8 = repo / "data" / "exp8"
            screens = repo / "data" / "screens"
            car = repo / "data" / "car-ramp"

            # Missing spec
            (exp8 / "specs" / "D1k_entries.json").unlink()
            with self.assertRaisesRegex(OracleError, "missing Exp 8 spec"):
                _evaluate_exp8_live_tensors(exp8, base_sha)

            # Non-finite bf16 weight in edit.npz (0x7F80 = +inf)
            with zipfile.ZipFile(exp8 / "edit.npz", "w") as zf:
                zf.writestr("t0_idx.npy", _encode_npy("<i4", (1,), struct.pack("<i", 0)))
                zf.writestr("t0_val.npy", _encode_npy("<u2", (1,), struct.pack("<H", 0x7F80)))
                zf.writestr("t1_idx.npy", _encode_npy("<i4", (1,), struct.pack("<i", 0)))
                zf.writestr("t1_val.npy", _encode_npy("<u2", (1,), struct.pack("<H", 0x3F80)))
            with self.assertRaisesRegex(OracleError, "non-finite bf16"):
                _evaluate_exp8_live_tensors(exp8, base_sha)

            # Bad edit.npz value dtype
            with zipfile.ZipFile(exp8 / "edit.npz", "w") as zf:
                zf.writestr("t0_idx.npy", _encode_npy("<i4", (1,), struct.pack("<i", 0)))
                zf.writestr("t0_val.npy", _encode_npy("<u4", (1,), struct.pack("<I", 0)))
            with self.assertRaisesRegex(OracleError, "expected <u2"):
                _evaluate_exp8_live_tensors(exp8, base_sha)

            # 1D base tensor rejected
            (exp8 / "base_l15_down_proj_f32.npy").write_bytes(_encode_npy("<f4", (4,), struct.pack("<4I", 0, 0, 0, 0)))
            with self.assertRaisesRegex(OracleError, "expected 2D base"):
                _evaluate_exp8_live_tensors(exp8, base_sha)

            # Missing factors_layer15.npz specifically (kills first conjunct of fac/base check!)
            (exp8 / "factors_layer15.npz").unlink()
            with self.assertRaisesRegex(OracleError, "missing Exp 8 layer"):
                _evaluate_exp8_live_tensors(exp8, base_sha)
            # Restore factors, delete base (kills second conjunct!)
            f1 = 0x3F800000
            with zipfile.ZipFile(exp8 / "factors_layer15.npz", "w") as zf:
                zf.writestr("u.npy", _encode_npy("<f4", (2,), struct.pack("<2I", f1, f1)))
                zf.writestr("v.npy", _encode_npy("<f4", (2,), struct.pack("<2I", f1, f1)))
            (exp8 / "base_l15_down_proj_f32.npy").unlink()
            with self.assertRaisesRegex(OracleError, "missing Exp 8 layer"):
                _evaluate_exp8_live_tensors(exp8, base_sha)

            # Missing edit.npz
            (exp8 / "edit.npz").unlink()
            with self.assertRaisesRegex(OracleError, "missing exp8/edit.npz"):
                _evaluate_exp8_live_tensors(exp8, base_sha)

            # validate_D4k_math500 every conjunct individually
            for line, rx in (
                ('{"correct":[1]}\n', "missing id/correct"),
                ('{"id":"m0"}\n', "missing id/correct"),
                ('{"id":1,"correct":[1]}\n', "missing id/correct"),
                ('{"id":"m0","correct":123}\n', "non-empty list"),
                ('{"id":"m0","correct":[]}\n', "non-empty list"),
                ('{"id":"m0","correct":[true]}\n', "exact int 0/1"),
                ('{"id":"m0","correct":[2]}\n', "exact int 0/1"),
            ):
                (exp8 / "validate_D4k_math500.jsonl").write_text(line, encoding="utf-8")
                with self.assertRaisesRegex(OracleError, rx):
                    _evaluate_exp8_live_screens(repo)

            # Empty screens dir
            for sf in screens.glob("screen_*.jsonl"):
                sf.unlink()
            with self.assertRaisesRegex(OracleError, "no screen_"):
                _evaluate_exp8_live_screens(repo)

            # Car-ramp shape mismatch, missing ckpt, missing base
            with zipfile.ZipFile(car / "edited_actor.npz", "w") as zf:
                for actor_key, _ in CAR_RAMP_ACTOR_MAP:
                    zf.writestr(f"{actor_key}.npy", _encode_npy("<f8", (2,), struct.pack("<2Q", 0, 0)))
            with self.assertRaisesRegex(OracleError, "shape mismatch"):
                _evaluate_car_ramp_live_checkpoints(car, base_sha)

            (car / "edited_actor.npz").unlink()
            with self.assertRaisesRegex(OracleError, "missing car-ramp checkpoint"):
                _evaluate_car_ramp_live_checkpoints(car, base_sha)

            (car / "base_ppo_final_seed0.npz").unlink()
            with self.assertRaisesRegex(OracleError, "missing car-ramp/base_ppo"):
                _evaluate_car_ramp_live_checkpoints(car, base_sha)

            # 32-element live quantization path + non-multiple 33-element + 128-element car-ramp spike path
            exp8_q = intake / "exp8_q"
            (exp8_q / "specs").mkdir(parents=True)
            f1 = 0x3F800000
            f_half = 0x3F000000  # 0.5
            f_two = 0x40000000  # 2.0
            f_four = 0x40800000  # 4.0
            f_twenty = 0x41A00000  # 20.0
            with zipfile.ZipFile(exp8_q / "factors_layer15.npz", "w") as zf:
                zf.writestr("u.npy", _encode_npy("<f4", (1,), struct.pack("<I", f1)))
                zf.writestr("v.npy", _encode_npy("<f4", (32,), struct.pack("<32I", *([f1] * 32))))
            with zipfile.ZipFile(exp8_q / "factors_layer19.npz", "w") as zf:
                zf.writestr("u.npy", _encode_npy("<f4", (1,), struct.pack("<I", f1)))
                zf.writestr("v.npy", _encode_npy("<f4", (33,), struct.pack("<33I", *([f1] * 33))))
            # L15 (1x32): base=[4.0, 0.0, 20.0, 2.0, ...], edit idx 0 -> 40.0 (mb=20 -> me=40):
            # idx 1=0 (qb=0->qe=0), idx 2=20 (qb=7->qe=4, flipped NOT zeroed!), idxs 3..31=2 (qb=1->qe=0, 29 zeroed!)
            base_l15_32 = [f_four, 0, f_twenty] + [f_two] * 29
            (exp8_q / "base_l15_down_proj_f32.npy").write_bytes(_encode_npy("<f4", (1, 32), struct.pack("<32I", *base_l15_32)))
            (exp8_q / "base_l19_down_proj_f32.npy").write_bytes(_encode_npy("<f4", (1, 33), struct.pack("<33I", *([0] * 33))))
            with zipfile.ZipFile(exp8_q / "edit.npz", "w") as zf:
                zf.writestr("t0_idx.npy", _encode_npy("<i4", (1,), struct.pack("<i", 0)))
                zf.writestr("t0_val.npy", _encode_npy("<u2", (1,), struct.pack("<H", 0x4220)))  # bf16 40.0
                zf.writestr("t1_idx.npy", _encode_npy("<i4", (1,), struct.pack("<i", 0)))
                zf.writestr("t1_val.npy", _encode_npy("<u2", (1,), struct.pack("<H", 0x3F80)))
            (exp8_q / "frozen_edit_audit.json").write_text(
                json.dumps({"15": {"changed_values": 1}, "19": {"changed_values": 1}}),
                encoding="utf-8",
            )
            for sname in EXP8_AUDITED_SPECS:
                (exp8_q / "specs" / f"{sname}.json").write_text(json.dumps({"scale": 1, "cols15": [0], "cols19": [0]}), encoding="utf-8")
            q_res = _evaluate_exp8_live_tensors(exp8_q, base_sha)
            g32_stats = q_res["siliquant_live_exp8_int4_survival"]["layer_15"]["G32"]
            self.assertEqual(g32_stats["collateral_unedited_flips"], 30)
            self.assertEqual(g32_stats["collateral_unedited_zeroed"], 29)
            self.assertEqual(q_res["siliquant_live_exp8_int4_survival"]["layer_19"], {})

            # 128-element car-ramp live spike collapse path (exercises label/actor_key/len>=128 guards!)
            car_128 = intake / "car_128"
            car_128.mkdir()
            b128 = [f_half] * 128
            with zipfile.ZipFile(car_128 / "base_ppo_final_seed0.npz", "w") as zf:
                for _, flax_key in CAR_RAMP_ACTOR_MAP:
                    zf.writestr(flax_key, _encode_npy("<f4", (128,), struct.pack("<128I", *b128)))
            for fname in ("edited_actor.npz", "joint_free.npz", "greedy12_astra.npz", "greedy11_drop02.npz", "c2_14entry.npz", "c2_min_10entry.npz"):
                with zipfile.ZipFile(car_128 / fname, "w") as zf:
                    for actor_key, _ in CAR_RAMP_ACTOR_MAP:
                        # Only actor_head.W gets 40.0 spike (31 zeroed); actor_head.b gets 6.0 spike (>5 thresh, 0 zeroed)
                        spike_bits = 0x4044000000000000 if actor_key == "actor_head.W" else 0x4018000000000000
                        vals_64 = [spike_bits] + [0x3FE0000000000000] * 127
                        zf.writestr(f"{actor_key}.npy", _encode_npy("<f8", (128,), struct.pack("<128Q", *vals_64)))
            c128_res = _evaluate_car_ramp_live_checkpoints(car_128, base_sha)
            self.assertEqual(set(c128_res["siliquant_live_spike_collapse"].keys()), {"c2_min_10entry_actor_head_W", "greedy12_astra_actor_head_W"})
            self.assertEqual(c128_res["siliquant_live_spike_collapse"]["c2_min_10entry_actor_head_W"]["G32"]["collateral_unedited_zeroed"], 31)

            # 32-element non-finite base weight inside touched quantization group guard
            exp8_nf = intake / "exp8_nf"
            exp8_nf.mkdir()
            with zipfile.ZipFile(exp8_nf / "factors_layer15.npz", "w") as zf:
                zf.writestr("u.npy", _encode_npy("<f4", (1,), struct.pack("<I", f1)))
                zf.writestr("v.npy", _encode_npy("<f4", (32,), struct.pack("<32I", *([f1] * 32))))
            base32 = [0] + [0x7F800000] + [0] * 30
            (exp8_nf / "base_l15_down_proj_f32.npy").write_bytes(_encode_npy("<f4", (1, 32), struct.pack("<32I", *base32)))
            with zipfile.ZipFile(exp8_nf / "edit.npz", "w") as zf:
                zf.writestr("t0_idx.npy", _encode_npy("<i4", (1,), struct.pack("<i", 0)))
                zf.writestr("t0_val.npy", _encode_npy("<u2", (1,), struct.pack("<H", 0x3F80)))
            with self.assertRaisesRegex(OracleError, "non-finite base weight"):
                _evaluate_exp8_live_tensors(exp8_nf, base_sha)


if __name__ == "__main__":
    unittest.main()
