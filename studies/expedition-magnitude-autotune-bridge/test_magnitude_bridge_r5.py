#!/usr/bin/env python3
"""Isolated R5 adversarial and boundary test suite for magnitude_autotune_bridge_audit.py."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parent))
import magnitude_autotune_bridge_audit as mod


def _has_any_float(obj: object) -> bool:
    if isinstance(obj, float):
        return True
    if isinstance(obj, dict):
        return any(_has_any_float(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_any_float(v) for v in obj)
    return False


def _valid_trust_config() -> dict[str, object]:
    sha = hashlib.sha256(b"checkpoint-pinned").hexdigest()
    return {
        "bind_host": "127.0.0.1",
        "auth_token": "0123456789abcdef",
        "verifier_mode": True,
        "temperature": Fraction(0, 1),
        "top_p": Fraction(1, 1),
        "explicit_seed": 42,
        "tuning_identity_sha256": sha,
        "cache_privacy": "owner",
        "cache_mode": 0o700,
        "cache_o_nofollow": True,
        "evidence_digest_hex": sha,
        "evidence_hash_verified": True,
        "jit_pointer_authenticated": True,
        "jit_library_symbols": ["expf", "sqrtf"],
        "checkpoint_sha256": sha,
        "expected_checkpoint_sha256": sha,
        "prefix_cache_reused": True,
        "request_tenant_id": "tenant-a",
        "cached_prefix_tenant_id": "tenant-a",
        "tainted_rows": 0,
        "active_banks": ["ConvWindow", "DeltaState"],
        "zeroed_banks": ["DeltaState", "ConvWindow"],
    }


def _make_fake_repo(root: Path, commit: str, *, symbolic_head: bool, corrupt_anchor: bool) -> None:
    git_dir = root / ".git"
    if symbolic_head:
        ref_dir = git_dir / "refs" / "heads"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (ref_dir / "main").write_text(f"{commit}\n", encoding="utf-8")
    else:
        git_dir.mkdir(parents=True, exist_ok=True)
        (git_dir / "HEAD").write_text(f"{commit}\n", encoding="utf-8")
    for idx, (_fid, rel_path, expected_line, needle) in enumerate(mod.ANCHOR_SPECS):
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            lines = target.read_text(encoding="utf-8").splitlines()
        else:
            lines = []
        while len(lines) < expected_line:
            lines.append("// filler")
        lines[expected_line - 1] = "corrupted" if (corrupt_anchor and idx == 0) else needle
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestVilleSupermartingale(unittest.TestCase):
    def test_exact_ratios_and_martingale_identity(self) -> None:
        self.assertEqual(mod.ville_mixture_ratio(0, 0), Fraction(1, 1))
        self.assertEqual(mod.ville_mixture_ratio(1, 0), Fraction(1, 2))
        self.assertEqual(mod.ville_mixture_ratio(1, 1), Fraction(3, 2))
        self.assertEqual(mod.ville_mixture_ratio(5, 3), Fraction(7, 10))
        self.assertEqual(mod.ville_mixture_ratio(6, 6), Fraction(127, 7))
        self.assertEqual(mod.ville_mixture_ratio(7, 7), Fraction(255, 8))
        self.assertEqual(mod.ville_mixture_ratio(8, 8), Fraction(511, 9))
        avg = (mod.ville_mixture_ratio(6, 3) + mod.ville_mixture_ratio(6, 4)) / Fraction(2, 1)
        self.assertEqual(avg, mod.ville_mixture_ratio(5, 3))

    def test_ville_invalid_bounds_raise(self) -> None:
        for bad in ((-1, 0), (3, -1), (3, 4), (True, 0), (1, True)):
            with self.assertRaises(ValueError):
                mod.ville_mixture_ratio(*bad)  # type: ignore[arg-type]

    def test_uncorrected_fwer_exact_and_guards(self) -> None:
        self.assertEqual(mod.uncorrected_fwer(1, Fraction(1, 20)), Fraction(1, 20))
        self.assertEqual(mod.uncorrected_fwer(2, Fraction(1, 10)), Fraction(19, 100))
        self.assertGreater(mod.uncorrected_fwer(20, Fraction(1, 20)), Fraction(64, 100))
        for bad in (
            (0, Fraction(1, 20)),
            (True, Fraction(1, 20)),
            (2, 0.05),
            (2, Fraction(0, 1)),
            (2, Fraction(1, 1)),
        ):
            with self.assertRaises(ValueError):
                mod.uncorrected_fwer(*bad)  # type: ignore[arg-type]

    def test_legacy_verdict_exact_boundaries(self) -> None:
        self.assertEqual(mod.magnitude_legacy_verdict(6, 0, Fraction(1, 20)), "undecided")
        self.assertEqual(mod.magnitude_legacy_verdict(7, 0, Fraction(1, 20)), "leads")
        self.assertEqual(mod.magnitude_legacy_verdict(0, 7, Fraction(1, 20)), "trails")
        self.assertEqual(mod.magnitude_legacy_verdict(1, 0, Fraction(2, 3)), "leads")
        self.assertEqual(mod.magnitude_legacy_verdict(0, 1, Fraction(2, 3)), "trails")
        for bad in (
            (1, 0, Fraction(0, 1)),
            (1, 0, Fraction(1, 1)),
            (1, 0, 0.05),
            (-1, 0, Fraction(1, 20)),
            (0, -1, Fraction(1, 20)),
            (True, 0, Fraction(1, 20)),
            (0, True, Fraction(1, 20)),
        ):
            with self.assertRaises(ValueError):
                mod.magnitude_legacy_verdict(*bad)  # type: ignore[arg-type]

    def test_hardened_gate_multiplicity_margin_and_anchor(self) -> None:
        res7 = mod.hardened_supermartingale_gate(
            7, 0, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), True
        )
        self.assertEqual(res7["verdict"], "undecided")
        self.assertFalse(res7["promote"])
        self.assertEqual(res7["threshold"], {"num": 40, "den": 1})

        res8 = mod.hardened_supermartingale_gate(
            8, 0, Fraction(1, 20), 1, Fraction(1, 50), Fraction(1, 50), True
        )
        self.assertEqual(res8["verdict"], "leads")
        self.assertTrue(res8["promote"])

        res_slow = mod.hardened_supermartingale_gate(
            8, 0, Fraction(1, 20), 1, Fraction(1, 51), Fraction(1, 50), True
        )
        self.assertEqual(res_slow["verdict"], "leads")
        self.assertFalse(res_slow["promote"])

        res_goodhart = mod.hardened_supermartingale_gate(
            8, 0, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), False
        )
        self.assertFalse(res_goodhart["promote"])

        res_trail = mod.hardened_supermartingale_gate(
            0, 8, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), True
        )
        self.assertEqual(res_trail["verdict"], "trails")
        self.assertFalse(res_trail["promote"])

        eq_lead = mod.hardened_supermartingale_gate(
            2, 0, Fraction(6, 7), 1, Fraction(1, 10), Fraction(1, 10), True
        )
        self.assertEqual(eq_lead["verdict"], "leads")
        self.assertTrue(eq_lead["promote"])
        eq_trail = mod.hardened_supermartingale_gate(
            0, 2, Fraction(6, 7), 1, Fraction(1, 10), Fraction(1, 10), True
        )
        self.assertEqual(eq_trail["verdict"], "trails")

        res_k2 = mod.hardened_supermartingale_gate(
            8, 0, Fraction(1, 20), 2, Fraction(1, 10), Fraction(1, 50), True
        )
        self.assertEqual(res_k2["verdict"], "undecided")
        self.assertEqual(res_k2["threshold"], {"num": 80, "den": 1})

        # Input guards (float, bool-as-int, non-bool anchor)
        for bad in (
            (1, 0, Fraction(0, 1), 1, Fraction(1, 10), Fraction(1, 50), True),
            (1, 0, Fraction(1, 1), 1, Fraction(1, 10), Fraction(1, 50), True),
            (1, 0, 0.05, 1, Fraction(1, 10), Fraction(1, 50), True),
            (1, 0, Fraction(1, 20), 0, Fraction(1, 10), Fraction(1, 50), True),
            (-1, 0, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), True),
            (0, -1, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), True),
            (1, 0, Fraction(1, 20), 1, Fraction(1, 10), Fraction(0, 1), True),
            (True, 0, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), True),
            (0, True, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), True),
            (1, 0, Fraction(1, 20), True, Fraction(1, 10), Fraction(1, 50), True),
            (8, 0, Fraction(1, 20), 1, True, Fraction(1, 50), True),
            (8, 0, Fraction(1, 20), 1, Fraction(1, 10), True, True),
            (8, 0, Fraction(1, 20), 1, 0.1, Fraction(1, 50), True),
            (8, 0, Fraction(1, 20), 1, Fraction(1, 10), 0.02, True),
            (8, 0, Fraction(1, 20), 1, Fraction(1, 10), Fraction(1, 50), "yes"),
        ):
            with self.assertRaises(ValueError):
                mod.hardened_supermartingale_gate(*bad)  # type: ignore[arg-type]


class TestRoundingAndAdmission(unittest.TestCase):
    def test_ieee_rounding_step_exact_rust_lattice_width(self) -> None:
        # Subnormal & zero
        self.assertEqual(mod.ieee_rounding_step(Fraction(0, 1), "F16"), Fraction(1, 2**24))
        self.assertEqual(mod.ieee_rounding_step(Fraction(0, 1), "BF16"), Fraction(1, 2**133))
        self.assertEqual(mod.ieee_rounding_step(Fraction(0, 1), "F32"), Fraction(1, 2**149))
        # Exact normal powers of two (e > min_normal_exp): w_T(2^e) = (3/4) * ULP per lattice.rs:448-462
        self.assertEqual(mod.ieee_rounding_step(Fraction(-1, 1), "BF16"), Fraction(3, 512))
        self.assertEqual(mod.ieee_rounding_step(Fraction(4096, 1), "BF16"), Fraction(24, 1))
        self.assertEqual(mod.ieee_rounding_step(Fraction(1, 2**13), "F16"), Fraction(3, 2**25))
        # Non-powers of two: w_T(v) = ULP
        self.assertEqual(mod.ieee_rounding_step(Fraction(3, 2), "BF16"), Fraction(1, 128))
        self.assertEqual(mod.ieee_rounding_step(Fraction(6144, 1), "BF16"), Fraction(32, 1))
        self.assertEqual(mod.ieee_rounding_step(Fraction(5, 11), "BF16"), Fraction(1, 512))
        # Exact min_normal_exp boundary (2^-14 in F16): gap_below == gap_above == 2^-24 (NO 3/4 factor!)
        self.assertEqual(mod.ieee_rounding_step(Fraction(1, 2**14), "F16"), Fraction(1, 2**24))
        self.assertEqual(mod.ieee_rounding_step(Fraction(1, 2**15), "F16"), Fraction(1, 2**24))
        # BF16 vs F32 ratio at 1.0 is still exact 65536
        self.assertEqual(
            mod.ieee_rounding_step(Fraction(1, 1), "BF16")
            / mod.ieee_rounding_step(Fraction(1, 1), "F32"),
            Fraction(65536, 1),
        )
        # Reject unknown format or raw float
        with self.assertRaises(ValueError):
            mod.ieee_rounding_step(Fraction(1, 1), "FP8")
        with self.assertRaises(ValueError):
            mod.ieee_rounding_step(1.0, "BF16")  # type: ignore[arg-type]

    def test_median_odd_and_even(self) -> None:
        self.assertEqual(
            mod._median_magnitude([Fraction(-5, 1), Fraction(1, 1), Fraction(3, 1)]),
            Fraction(3, 1),
        )
        self.assertEqual(
            mod._median_magnitude([Fraction(1, 1), Fraction(5, 1), Fraction(2, 1), Fraction(10, 1)]),
            Fraction(7, 2),
        )

    def test_magnitude_rms_admission_matches_rust_power_of_two_and_activation(self) -> None:
        # Power-of-two outlier 4096 is REJECTED under real Rust lattice.rs width (w=24 -> ratio^2=85.29 > 64)
        ref_4096 = [Fraction(4096, 1)] + [Fraction(1, 1)] * 4095
        cand_4096 = [Fraction(4096, 1)] + [Fraction(0, 1)] * 4095
        res_4096 = mod.magnitude_rms_admission(
            ref_4096, cand_4096, "BF16", "BF16", Fraction(8, 1), "activation"
        )
        self.assertFalse(res_4096["admitted"])
        self.assertEqual(res_4096["ratio_sq"], {"num": 1431306240, "den": 16781311})

        # Exact ratio_sq == alpha^2 equality boundary at non-power-of-two 3/2
        ref = [Fraction(3, 2)] * 3
        step = mod.ieee_rounding_step(Fraction(3, 2), "BF16")
        cand_sq = [Fraction(3, 2) + 2 * step, Fraction(3, 2), Fraction(3, 2)]
        res_exact = mod.magnitude_rms_admission(ref, cand_sq, "BF16", "BF16", Fraction(4, 1), "activation")
        self.assertTrue(res_exact["admitted"])
        self.assertEqual(res_exact["ratio_sq"], {"num": 16, "den": 1})
        self.assertEqual(res_exact["alpha_sq"], {"num": 16, "den": 1})

        # GoverningStep::Activation with activation=None falls back to subject_fmt (admission.rs:65)
        res_none = mod.magnitude_rms_admission(ref, cand_sq, "F32", None, Fraction(4, 1), "activation")
        self.assertEqual(res_none["step_fmt"], "F32")
        self.assertFalse(res_none["admitted"])

        # GoverningStep::Subject uses subject_fmt even when activation_fmt is provided
        res_subj = mod.magnitude_rms_admission(ref, cand_sq, "F32", "BF16", Fraction(4, 1), "subject")
        self.assertEqual(res_subj["step_fmt"], "F32")
        self.assertFalse(res_subj["admitted"])

        # Guards (incl. float alpha / float elements)
        for bad in (
            ([], [], "BF16", "BF16", Fraction(8, 1), "activation"),
            (ref, ref[:2], "BF16", "BF16", Fraction(8, 1), "activation"),
            ([1.5], [Fraction(3, 2)], "BF16", "BF16", Fraction(8, 1), "activation"),
            ([Fraction(3, 2)], [1.5], "BF16", "BF16", Fraction(8, 1), "activation"),
            (ref, ref, "BF16", "BF16", 8.0, "activation"),
            (ref, ref, "BF16", "BF16", Fraction(0, 1), "activation"),
            (ref, ref, "BF16", "BF16", Fraction(8, 1), "unknown"),
        ):
            with self.assertRaises(ValueError):
                mod.magnitude_rms_admission(*bad)  # type: ignore[arg-type]

    def test_hardened_dual_norm_closes_mag01_and_mag02(self) -> None:
        ref = [Fraction(3, 2)] * 3
        step = mod.ieee_rounding_step(Fraction(3, 2), "BF16")
        cand = [Fraction(3, 2) + 2 * step, Fraction(3, 2), Fraction(3, 2)]
        ok = mod.hardened_dual_norm_admission(
            ref, cand, "F32", "BF16", Fraction(4, 1), Fraction(2, 1), Fraction(4, 1), False
        )
        self.assertTrue(ok["admitted"])
        self.assertEqual(ok["step_fmt"], "BF16")
        self.assertEqual(ok["outlier_count"], 0)
        self.assertEqual(ok["bulk_count"], 3)

        # activation_fmt=None falls back to subject_fmt
        no_act = mod.hardened_dual_norm_admission(
            ref, cand, "F32", None, Fraction(4, 1), Fraction(2, 1), Fraction(4, 1), False
        )
        self.assertEqual(no_act["step_fmt"], "F32")
        self.assertFalse(no_act["admitted"])

        # Fails ONLY max_ulp_cap
        self.assertFalse(
            mod.hardened_dual_norm_admission(
                ref, cand, "F32", "BF16", Fraction(4, 1), Fraction(1, 1), Fraction(4, 1), False
            )["admitted"]
        )
        # Fails ONLY bulk_ratio_sq
        self.assertFalse(
            mod.hardened_dual_norm_admission(
                ref, cand, "F32", "BF16", Fraction(3, 1), Fraction(2, 1), Fraction(4, 1), False
            )["admitted"]
        )

        # Outlier partition boundary (exact cutoff stays in bulk) & outlier_ratio_sq gate
        ref_o = [Fraction(1, 1), Fraction(1, 1), Fraction(4, 1), Fraction(12, 1)]
        step_12 = mod.ieee_rounding_step(Fraction(12, 1), "BF16")
        cand_o = [Fraction(1, 1), Fraction(1, 1), Fraction(4, 1), Fraction(12, 1) + 2 * step_12]
        fail_out = mod.hardened_dual_norm_admission(
            ref_o, cand_o, "BF16", "BF16", Fraction(6, 1), Fraction(4, 1), Fraction(2, 1), False
        )
        self.assertEqual(fail_out["outlier_count"], 1)
        self.assertEqual(fail_out["bulk_count"], 3)
        self.assertFalse(fail_out["admitted"])

        ref_cut = [Fraction(1, 1), Fraction(1, 1), Fraction(4, 1)]
        cut_res = mod.hardened_dual_norm_admission(
            ref_cut, ref_cut, "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(4, 1), False
        )
        self.assertEqual(cut_res["outlier_count"], 0)
        self.assertEqual(cut_res["bulk_count"], 3)

        ref_o3 = [Fraction(1, 1)] * 5 + [Fraction(12, 1)] * 3
        cand_o3 = [Fraction(1, 1)] * 5 + [Fraction(12, 1) + 2 * step_12, Fraction(12, 1), Fraction(12, 1)]
        pass_out = mod.hardened_dual_norm_admission(
            ref_o3, cand_o3, "BF16", "BF16", Fraction(4, 1), Fraction(4, 1), Fraction(4, 1), False
        )
        self.assertEqual(pass_out["outlier_count"], 3)
        self.assertEqual(pass_out["outlier_ratio_sq"], {"num": 16, "den": 1})
        self.assertTrue(pass_out["admitted"])

        # Recurrent state locks to subject_fmt (F32)
        rec = mod.hardened_dual_norm_admission(
            ref, cand, "F32", "BF16", Fraction(8, 1), Fraction(16, 1), Fraction(4, 1), True
        )
        self.assertEqual(rec["step_fmt"], "F32")
        self.assertFalse(rec["admitted"])

        # Guards (incl. floats & non-bool is_recurrent_state)
        for bad in (
            ([], [], "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(4, 1), False),
            (ref, ref[:2], "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(4, 1), False),
            ([1.5], [Fraction(3, 2)], "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(4, 1), False),
            ([Fraction(3, 2)], [1.5], "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(4, 1), False),
            (ref, ref, "BF16", "BF16", 8.0, Fraction(4, 1), Fraction(4, 1), False),
            (ref, ref, "BF16", "BF16", Fraction(8, 1), 4.0, Fraction(4, 1), False),
            (ref, ref, "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), 4.0, False),
            (ref, ref, "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(4, 1), "no"),
            (ref, ref, "BF16", "BF16", Fraction(0, 1), Fraction(4, 1), Fraction(4, 1), False),
            (ref, ref, "BF16", "BF16", Fraction(8, 1), Fraction(0, 1), Fraction(4, 1), False),
            (ref, ref, "BF16", "BF16", Fraction(8, 1), Fraction(4, 1), Fraction(1, 1), False),
        ):
            with self.assertRaises(ValueError):
                mod.hardened_dual_norm_admission(*bad)  # type: ignore[arg-type]


class TestLineageAndTrustBoundary(unittest.TestCase):
    def test_signed_pearson_and_weight_sample_exact(self) -> None:
        xs = [Fraction(1, 1), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1)]
        ys_pos = [Fraction(3, 1) * x + Fraction(7, 1) for x in xs]
        ys_neg = [-x for x in xs]
        self.assertEqual(mod.pearson_signed_r_squared(xs, ys_pos), (1, Fraction(1, 1)))
        self.assertEqual(mod.pearson_signed_r_squared(xs, ys_neg), (-1, Fraction(1, 1)))
        self.assertTrue(mod.weight_sample_matches(xs, ys_pos))
        # Signed 1 - r rejects negative correlation (weight_lineage.rs:133)!
        self.assertFalse(mod.weight_sample_matches(xs, ys_neg))

        # Zero correlation: cov == 0 -> sign == 0 -> False
        xs_orth = [Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1)]
        ys_orth = [Fraction(1, 1), Fraction(-2, 1), Fraction(1, 1)]
        self.assertEqual(mod.pearson_signed_r_squared(xs_orth, ys_orth), (0, Fraction(0, 1)))
        self.assertFalse(mod.weight_sample_matches(xs_orth, ys_orth, Fraction(1, 2)))

        # Exact positive equality boundary r = 3/5 (r^2 = 9/25), max_divergence = 2/5
        xs_sq = [Fraction(-2, 1), Fraction(-1, 1), Fraction(1, 1), Fraction(2, 1)]
        ys_sq = [Fraction(-2, 1), Fraction(1, 1), Fraction(-1, 1), Fraction(2, 1)]
        self.assertEqual(mod.pearson_signed_r_squared(xs_sq, ys_sq), (1, Fraction(9, 25)))
        self.assertTrue(mod.weight_sample_matches(xs_sq, ys_sq, Fraction(2, 5)))
        self.assertFalse(mod.weight_sample_matches(xs_sq, ys_sq, Fraction(1, 3)))
        self.assertTrue(mod.weight_sample_matches(xs, ys_pos, Fraction(0, 1)))

        # Guards
        for bad_pair in (
            ([Fraction(1, 1)], [Fraction(1, 1)]),
            ([Fraction(1, 1), Fraction(2, 1)], [Fraction(1, 1)]),
            ([1.0, 2.0], [Fraction(1, 1), Fraction(2, 1)]),
            ([Fraction(1, 1), Fraction(2, 1)], [1.0, 2.0]),
            ([Fraction(1, 1), Fraction(1, 1)], [Fraction(1, 1), Fraction(2, 1)]),
            ([Fraction(1, 1), Fraction(2, 1)], [Fraction(5, 1), Fraction(5, 1)]),
        ):
            with self.assertRaises(ValueError):
                mod.pearson_signed_r_squared(*bad_pair)  # type: ignore[arg-type]
        for bad_div in (0.00025, Fraction(-1, 10), Fraction(1, 1)):
            with self.assertRaises(ValueError):
                mod.weight_sample_matches(xs, ys_pos, bad_div)  # type: ignore[arg-type]

    def test_trust_boundary_every_single_invariant_and_red_team_bypass(self) -> None:
        base = _valid_trust_config()
        self.assertEqual(mod.evaluate_trust_boundary(base), {"admitted": True, "reasons": []})

        # 1. Bind & auth token (whitespace, list, non-str host, short token)
        self.assertTrue(mod.evaluate_trust_boundary(dict(base, bind_host="0.0.0.0", auth_token="a" * 16))["admitted"])
        self.assertTrue(mod.evaluate_trust_boundary(dict(base, bind_host="::1", auth_token=""))["admitted"])
        for overrides in (
            {"bind_host": 127},
            {"bind_host": "0.0.0.0", "auth_token": "a" * 15},
            {"bind_host": "0.0.0.0", "auth_token": " " * 16},
            {"bind_host": "0.0.0.0", "auth_token": ["x"] * 16},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["BIND_UNAUTHENTICATED_NON_LOOPBACK"],
            )

        # 2. Sampling & verifier_mode (non-bool verifier_mode, float 0.0/1.0, bool False/True)
        cfg_non_verifier = dict(
            base,
            verifier_mode=False,
            temperature=Fraction(1, 1),
            top_p=Fraction(19, 20),
            explicit_seed=None,
            tuning_identity_sha256="none",
        )
        self.assertTrue(mod.evaluate_trust_boundary(cfg_non_verifier)["admitted"])
        self.assertEqual(
            mod.evaluate_trust_boundary(dict(base, verifier_mode="yes"))["reasons"],
            ["SAMPLING_NON_DETERMINISTIC", "TUNING_IDENTITY_UNPINNED"],
        )
        for overrides in (
            {"temperature": Fraction(1, 10)},
            {"temperature": False},
            {"temperature": 0.0},
            {"top_p": Fraction(19, 20)},
            {"top_p": True},
            {"top_p": 1.0},
            {"explicit_seed": None},
            {"explicit_seed": True},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["SAMPLING_NON_DETERMINISTIC"],
            )

        # 3. Tuning identity sha256
        for bad_sha in ("a" * 63, "g" * 64, 12345):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, tuning_identity_sha256=bad_sha))["reasons"],
                ["TUNING_IDENTITY_UNPINNED"],
            )

        # 4. Cache privacy / owner-only 0o077 mask / range [0, 0o777] / O_NOFOLLOW
        self.assertTrue(mod.evaluate_trust_boundary(dict(base, cache_mode=0o600))["admitted"])
        self.assertTrue(mod.evaluate_trust_boundary(dict(base, cache_mode=0))["admitted"])
        for overrides in (
            {"cache_privacy": "shared"},
            {"cache_mode": 0o755},
            {"cache_mode": 0o750},
            {"cache_mode": 0o704},
            {"cache_mode": 0o720},
            {"cache_mode": -0o700},
            {"cache_mode": 0o4700},
            {"cache_mode": False},
            {"cache_o_nofollow": False},
            {"cache_o_nofollow": "yes"},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["CACHE_PRIVACY_NOT_OWNER"],
            )

        # 5. Evidence digest & verification
        for overrides in (
            {"evidence_digest_hex": "0123456789abcdef"},
            {"evidence_hash_verified": False},
            {"evidence_hash_verified": "yes"},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["EVIDENCE_DIGEST_UNVERIFIED"],
            )

        # 6. JIT artifact auth & symbol allowlist
        for overrides in (
            {"jit_pointer_authenticated": False},
            {"jit_pointer_authenticated": "yes"},
            {"jit_library_symbols": "expf"},
            {"jit_library_symbols": [123]},
            {"jit_library_symbols": ["expf", "system"]},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["JIT_ARTIFACT_UNTRUSTED"],
            )

        # 7. Checkpoint SHA-256 pinning
        other_sha = hashlib.sha256(b"other").hexdigest()
        for overrides in (
            {"checkpoint_sha256": "short"},
            {"checkpoint_sha256": other_sha},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["CHECKPOINT_SHA256_MISMATCH"],
            )

        # 8. Prefix cache tenant isolation (non-bool reused, whitespace, bool tenant)
        self.assertTrue(
            mod.evaluate_trust_boundary(
                dict(base, prefix_cache_reused=False, request_tenant_id="", cached_prefix_tenant_id="tenant-a")
            )["admitted"]
        )
        for overrides in (
            {"prefix_cache_reused": "yes"},
            {"request_tenant_id": ""},
            {"request_tenant_id": "   "},
            {"request_tenant_id": True, "cached_prefix_tenant_id": True},
            {"cached_prefix_tenant_id": "   "},
            {"request_tenant_id": "tenant-b"},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["PREFIX_CACHE_TENANT_MISMATCH"],
            )

        # 9. State banks flush (empty banks, whitespace bank, non-list, bool tainted_rows)
        for overrides in (
            {"tainted_rows": 1},
            {"tainted_rows": False},
            {"active_banks": [], "zeroed_banks": []},
            {"active_banks": ["  "], "zeroed_banks": ["  "]},
            {"active_banks": "ConvWindow", "zeroed_banks": "ConvWindow"},
            {"zeroed_banks": "ConvWindow"},
            {"zeroed_banks": ["ConvWindow"]},
        ):
            self.assertEqual(
                mod.evaluate_trust_boundary(dict(base, **overrides))["reasons"],
                ["STATE_BANKS_NOT_FLUSHED"],
            )


class TestAnchorsSuiteAndCli(unittest.TestCase):
    def test_inspect_repo_symbolic_detached_mismatch_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_repo(root, mod.PINNED_COMMIT, symbolic_head=True, corrupt_anchor=False)
            ok = mod.inspect_magnitude_repo(root)
            self.assertTrue(ok["commit_matches"])
            self.assertTrue(ok["all_anchors_matched"])
            self.assertEqual(ok["anchors_verified"], len(mod.ANCHOR_SPECS))
            self.assertEqual(ok["anchors_total"], len(mod.ANCHOR_SPECS))

            _make_fake_repo(root, "0" * 40, symbolic_head=False, corrupt_anchor=False)
            wrong_sha = mod.inspect_magnitude_repo(root)
            self.assertFalse(wrong_sha["commit_matches"])
            self.assertFalse(wrong_sha["all_anchors_matched"])

            _make_fake_repo(root, mod.PINNED_COMMIT, symbolic_head=False, corrupt_anchor=True)
            corrupt = mod.inspect_magnitude_repo(root)
            self.assertTrue(corrupt["commit_matches"])
            self.assertFalse(corrupt["all_anchors_matched"])
            self.assertEqual(corrupt["anchors_verified"], len(mod.ANCHOR_SPECS) - 1)

            first_path = root / mod.ANCHOR_SPECS[0][1]
            first_path.write_text("short\n", encoding="utf-8")
            trunc = mod.inspect_magnitude_repo(root)
            self.assertFalse(trunc["all_anchors_matched"])

    def test_run_audit_suite_standalone_zero_float_and_proofs(self) -> None:
        res = mod.run_audit_suite(None)
        self.assertFalse(_has_any_float(res))
        self.assertEqual(res["anchors"]["anchor_mode"], "standalone")
        self.assertEqual(res["anchors"]["anchors_verified"], 0)
        self.assertEqual(res["anchors"]["anchors_total"], len(mod.ANCHOR_SPECS))
        self.assertFalse(res["anchors"]["all_anchors_matched"])

        proofs = res["proofs"]
        self.assertTrue(proofs["mag01a_localized_spike"]["legacy_admitted"])
        self.assertEqual(proofs["mag01a_localized_spike"]["legacy_ratio_sq"], {"num": 64827, "den": 1024})
        self.assertFalse(proofs["mag01a_localized_spike"]["hardened_admitted"])
        self.assertEqual(proofs["mag01a_localized_spike"]["hardened_max_ulp"], {"num": 147, "den": 1})

        self.assertEqual(proofs["mag01b_outlier_masking"]["outlier_value"], 6144)
        self.assertTrue(proofs["mag01b_outlier_masking"]["legacy_admitted"])
        self.assertEqual(
            proofs["mag01b_outlier_masking"]["legacy_ratio_sq"],
            {"num": 12881756160, "den": 268472311},
        )
        self.assertFalse(proofs["mag01b_outlier_masking"]["hardened_admitted"])
        self.assertEqual(proofs["mag01b_outlier_masking"]["hardened_outlier_count"], 1)
        self.assertEqual(
            proofs["mag01b_outlier_masking"]["hardened_bulk_ratio_sq"],
            {"num": 1048576, "den": 3},
        )

        self.assertEqual(
            proofs["mag02_recurrent_step_relaxation"]["bf16_over_f32_step_factor"],
            {"num": 65536, "den": 1},
        )
        self.assertTrue(proofs["mag02_recurrent_step_relaxation"]["legacy_admitted"])
        self.assertEqual(
            proofs["mag02_recurrent_step_relaxation"]["legacy_ratio_sq"],
            {"num": 4, "den": 3},
        )
        self.assertFalse(proofs["mag02_recurrent_step_relaxation"]["hardened_admitted"])
        self.assertEqual(proofs["mag02_recurrent_step_relaxation"]["hardened_step_fmt"], "F32")
        self.assertEqual(
            proofs["mag02_recurrent_step_relaxation"]["hardened_bulk_ratio_sq"],
            {"num": 17179869184, "den": 3},
        )

        self.assertTrue(proofs["mag03_ville_supermartingale"]["martingale_exact_identity"])
        self.assertEqual(proofs["mag03_ville_supermartingale"]["legacy_7_0_verdict"], "leads")
        self.assertEqual(proofs["mag03_ville_supermartingale"]["hardened_7_0_verdict"], "undecided")
        self.assertFalse(proofs["mag03_ville_supermartingale"]["hardened_8_0_anchor_fail_promote"])
        self.assertTrue(proofs["mag03_ville_supermartingale"]["hardened_8_0_pass_promote"])

        self.assertEqual(proofs["mag06_weight_sample_vs_sha256"]["pos_affine_sign"], 1)
        self.assertTrue(proofs["mag06_weight_sample_vs_sha256"]["weight_sample_pos_affine_matches"])
        self.assertFalse(proofs["mag06_weight_sample_vs_sha256"]["weight_sample_neg_affine_matches"])
        self.assertTrue(proofs["mag06_weight_sample_vs_sha256"]["sha256_catches_tamper"])

        self.assertFalse(proofs["mag04_05_06_08_trust_boundary"]["legacy_risky_admitted"])
        self.assertEqual(proofs["mag04_05_06_08_trust_boundary"]["legacy_risky_reason_count"], 9)
        self.assertTrue(proofs["mag04_05_06_08_trust_boundary"]["hardened_admitted"])

        body = dict(res)
        digest = body.pop("canonical_sha256")
        expected = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(digest, expected)

    def test_cli_main_and_default_repo_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_repo = tmp_path / "repo"
            _make_fake_repo(fake_repo, mod.PINNED_COMMIT, symbolic_head=True, corrupt_anchor=False)
            cert_path = tmp_path / "cert.json"

            buf = io.StringIO()
            with patch("sys.stdout", buf):
                rc = mod.main(["--repo", str(fake_repo), "--json", "--write-cert", str(cert_path)])
            self.assertEqual(rc, 0)
            parsed = json.loads(buf.getvalue())
            self.assertTrue(parsed["anchors"]["all_anchors_matched"])
            self.assertEqual(cert_path.read_text(encoding="utf-8"), buf.getvalue())

            buf_human = io.StringIO()
            with patch("sys.stdout", buf_human):
                rc2 = mod.main(["--standalone"])
            self.assertEqual(rc2, 0)
            self.assertIn("standalone", buf_human.getvalue())
            self.assertIn("sha256=", buf_human.getvalue())

            with patch.object(mod, "_default_repo_path", return_value=fake_repo):
                buf_auto = io.StringIO()
                with patch("sys.stdout", buf_auto):
                    rc3 = mod.main(["--json"])
                self.assertEqual(rc3, 0)
                self.assertEqual(json.loads(buf_auto.getvalue())["anchors"]["anchor_mode"], "live_readonly")

            with patch.object(mod, "__file__", str(tmp_path / "arm1" / "script.py")):
                self.assertIsNone(mod._default_repo_path())
            (tmp_path / "repos" / "magnitude" / ".git").mkdir(parents=True, exist_ok=True)
            (tmp_path / "repos" / "magnitude" / ".git" / "HEAD").write_text(
                f"{mod.PINNED_COMMIT}\n", encoding="utf-8"
            )
            with patch.object(mod, "__file__", str(tmp_path / "arm1" / "script.py")):
                self.assertEqual(mod._default_repo_path(), (tmp_path / "repos" / "magnitude").resolve())


if __name__ == "__main__":
    unittest.main()
