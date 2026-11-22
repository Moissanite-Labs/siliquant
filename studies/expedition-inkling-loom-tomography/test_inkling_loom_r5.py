#!/usr/bin/env python3
"""
Wave 2 Round 8 — Arm R8-3 (AUDIT-074) R1-R5 Verification & Mutation-Kill Suite.
Pure Python 3 stdlib, zero bare asserts, isolated-mode (-I) safe.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling_loom_tomography as mod


class TestInklingLoomTomographyR5(unittest.TestCase):
    # ------------------------------------------------------------------
    # Group 1: Core helpers, rounding, and strict type/domain guards
    # ------------------------------------------------------------------
    def test_helpers_and_guards(self) -> None:
        mod.require(True, "ok")
        with self.assertRaises(mod.LoomAuditError):
            mod.require(False, "fail")

        self.assertEqual(mod.require_int(7, "x"), 7)
        with self.assertRaises(mod.LoomAuditError):
            mod.require_int(True, "x")
        with self.assertRaises(mod.LoomAuditError):
            mod.require_int("7", "x")

        self.assertEqual(mod.require_frac(Fraction(3, 4), "f"), Fraction(3, 4))
        with self.assertRaises(mod.LoomAuditError):
            mod.require_frac(3, "f")

        valid_sha = "0" * 64
        self.assertTrue(mod.is_sha256_hex(valid_sha))
        self.assertFalse(mod.is_sha256_hex(123))
        self.assertFalse(mod.is_sha256_hex("0" * 63))
        self.assertFalse(mod.is_sha256_hex("0" * 63 + "A"))

        # Exact half-to-even rounding boundaries
        self.assertEqual(mod.round_half_even_scaled(Fraction(1, 2), 1), 0)  # even
        self.assertEqual(mod.round_half_even_scaled(Fraction(3, 2), 1), 2)  # odd -> round up
        self.assertEqual(mod.round_half_even_scaled(Fraction(3, 4), 1), 1)  # > half
        with self.assertRaises(mod.LoomAuditError):
            mod.round_half_even_scaled(Fraction(1, 2), 0)

        self.assertEqual(mod.format_fixed(Fraction(0, 1), 0), "0")
        self.assertEqual(mod.format_fixed(Fraction(-3, 2), 0), "-2")
        self.assertEqual(mod.format_fixed(Fraction(-1, 1000), 1), "0.0")
        self.assertEqual(mod.format_fixed(Fraction(-5, 4), 2), "-1.25")
        with self.assertRaisesRegex(mod.LoomAuditError, "decimals must be non-negative"):
            mod.format_fixed(Fraction(1, 2), -1)

    # ------------------------------------------------------------------
    # Group 2: Module 1 — Ring-0 Lockfile & Provenance Gate + Live Anchor
    # ------------------------------------------------------------------
    def test_lockfile_bundle_gate_refusals(self) -> None:
        sha_a = "a" * 64
        sha_b = "b" * 64
        good_lock = {"p.loom": sha_a}
        good_exec = {"p.loom": sha_a}
        good_prov = {
            "skills_commit": "abcdef1",
            "hrx_loom_reported": "0.2.0",
            "hrx_loom_sdk": "0.2.0",
            "kit_modified": False,
            "dirty_choices": [],
            "has_kernel_tests_evidence": True,
            "has_logits_evidence": True,
            "method_source_consistent": True,
        }

        res_ok = mod.verify_loom_lockfile_bundle(good_lock, good_exec, good_prov)
        self.assertTrue(res_ok["passed"])
        self.assertEqual(res_ok["verdict"], "PROMOTED_RING0_BOUND")
        self.assertEqual(res_ok["pinned_executables"], 1)

        # Input-shape guards & LOOM-BYPASS-03 (invalid required_executables keys)
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_loom_lockfile_bundle([], good_exec, good_prov)  # type: ignore[arg-type]
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_loom_lockfile_bundle(good_lock, {}, good_prov)
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_loom_lockfile_bundle(good_lock, good_exec, [])  # type: ignore[arg-type]
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_loom_lockfile_bundle(good_lock, {"p.loom": "not_sha"}, good_prov)
        for bad_req_key in ("", "   ", " p.loom ", "p\x00.loom", 123):
            with self.assertRaisesRegex(mod.LoomAuditError, "invalid required_executables key"):
                mod.verify_loom_lockfile_bundle(good_lock, {bad_req_key: sha_a}, good_prov)  # type: ignore[dict-item]

        # Every individual defect path + LOOM-BYPASS-01 (corrupted passive asset / bad key in lock_hashes)
        r_missing = mod.verify_loom_lockfile_bundle({}, good_exec, good_prov)
        self.assertIn("LOCK_MISSING_EXECUTABLE:p.loom", r_missing["defects"])
        self.assertFalse(r_missing["passed"])

        r_bad_sha = mod.verify_loom_lockfile_bundle({"p.loom": "zzz"}, good_exec, good_prov)
        self.assertIn("LOCK_INVALID_SHA256:p.loom", r_bad_sha["defects"])
        self.assertEqual(r_bad_sha["defects"].count("LOCK_INVALID_SHA256:p.loom"), 1)

        r_tampered_asset = mod.verify_loom_lockfile_bundle(
            {"p.loom": sha_a, "@checkpoint/Inkling-2-Mini-UD-Q4_K_XL.gguf": "TAMPERED-NOT-EVEN-HEX"},
            good_exec,
            good_prov,
        )
        self.assertFalse(r_tampered_asset["passed"])
        self.assertIn("LOCK_INVALID_SHA256:@checkpoint/Inkling-2-Mini-UD-Q4_K_XL.gguf", r_tampered_asset["defects"])

        r_bad_key = mod.verify_loom_lockfile_bundle(
            {"p.loom": sha_a, "   ": "not-a-sha"},
            good_exec,
            good_prov,
        )
        self.assertFalse(r_bad_key["passed"])
        self.assertIn("LOCK_INVALID_KEY", r_bad_key["defects"])

        r_mismatch = mod.verify_loom_lockfile_bundle({"p.loom": sha_b}, good_exec, good_prov)
        self.assertIn("LOCK_DIGEST_MISMATCH:p.loom", r_mismatch["defects"])

        # Skills commit boundary: length 6 fails, length 7 passes, 40 passes, 41 fails, non-hex/whitespace/case fails
        for bad_commit in ("abcdef", "a" * 41, "unknown", " UNKNOWN ", "abcdef1 ", "ABCDEF1", "abcdefg", 1234567):
            p_bad = dict(good_prov, skills_commit=bad_commit)
            self.assertIn("PROV_UNKNOWN_SKILLS_COMMIT", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_bad)["defects"])
        p_40 = dict(good_prov, skills_commit="0" * 40)
        self.assertTrue(mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_40)["passed"])

        # SDK split-brain vs unknown / placeholder / whitespace / case evasion (LOOM-BYPASS-02)
        for bad_sdk in ("unknown", " UNKNOWN ", " 0.2.0", "", None, "N/A", "na", "none", "null", "-", "tbd", "alpha-only"):
            p_unk_sdk = dict(good_prov, hrx_loom_reported=bad_sdk, hrx_loom_sdk=bad_sdk)
            res_bad_sdk = mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_unk_sdk)
            self.assertFalse(res_bad_sdk["passed"])
            self.assertIn("PROV_UNKNOWN_HRX_LOOM", res_bad_sdk["defects"])
        p_bad_sdk2 = dict(good_prov, hrx_loom_sdk="N/A")
        self.assertIn("PROV_UNKNOWN_HRX_LOOM", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_bad_sdk2)["defects"])
        p_split = dict(good_prov, hrx_loom_reported="0.1.9")
        self.assertIn("PROV_SDK_SPLIT_BRAIN", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_split)["defects"])

        # Dirty kit, dirty choices, missing gates, method contradiction
        p_dirty = dict(good_prov, kit_modified=True)
        self.assertIn("PROV_DIRTY_KIT", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_dirty)["defects"])
        p_choices = dict(good_prov, dirty_choices=["moe.decode.rows_per_wg"])
        self.assertIn("PROV_DIRTY_CHOICES", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_choices)["defects"])
        p_choices_bad = dict(good_prov, dirty_choices="bad")
        self.assertIn("PROV_DIRTY_CHOICES", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_choices_bad)["defects"])
        p_ktest = dict(good_prov, has_kernel_tests_evidence=False)
        self.assertIn("GATE_MISSING_KERNEL_TESTS", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_ktest)["defects"])
        p_logits = dict(good_prov, has_logits_evidence=False)
        self.assertIn("GATE_MISSING_LOGITS_EVIDENCE", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_logits)["defects"])
        p_method = dict(good_prov, method_source_consistent=False)
        self.assertIn("PROV_METHOD_SOURCE_CONTRADICTION", mod.verify_loom_lockfile_bundle(good_lock, good_exec, p_method)["defects"])

    def test_live_repo_mirror_negative_controls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Non-existent dir
            with self.assertRaisesRegex(mod.LoomAuditError, "repo_dir does not exist"):
                mod.audit_live_repo_mirror(root / "missing")
            # Empty dir -> missing first pinned file
            with self.assertRaisesRegex(mod.LoomAuditError, "missing pinned file"):
                mod.audit_live_repo_mirror(root)
            # Wrong byte size on model.json
            (root / "model.json").write_bytes(b"short")
            with self.assertRaisesRegex(mod.LoomAuditError, "byte size mismatch"):
                mod.audit_live_repo_mirror(root)
            # Exact byte size but wrong sha256 on model.json
            (root / "model.json").write_bytes(b"x" * 1659)
            with self.assertRaisesRegex(mod.LoomAuditError, "sha256 mismatch"):
                mod.audit_live_repo_mirror(root)

        # If local read-only mirror exists in the expedition workspace, live-verify it cleanly
        mirror = Path(__file__).resolve().parent.parent / "repos" / "inkling-loom"
        if mirror.is_dir():
            live = mod.audit_live_repo_mirror(mirror)
            self.assertEqual(live["mode"], "live_mirror_verified")
            self.assertEqual(live["verified_files_count"], 14)

        # Fast standalone synthetic 14-file mirror to exercise EVERY IR anchor guard cleanly
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "program").mkdir()
            synth_files: dict[str, bytes] = {
                "model.json": b"{}\n",
                "llm_config.json": b"{}\n",
                "weights.loom": b"func.def @w() {}\n",
                "head_weights.loom": (
                    b"// header\n"
                    b"global.rodata.def @root0 = bytes(\"6c6c6d5f707265705f71385f305f71345f30\")\n"
                    b"%nb0 = scalar.constant 14404176 : i64\n"
                ),
                "program/control.loom": (
                    b"func.decl import(\"input\") @require()\n"
                    b"%logits_flag = scalar.constant 1073741824 : i32\n"
                    b"%logits_mask = scalar.constant 1073741823 : i32\n"
                ),
                "program/llm.loom": (
                    b"target.decl @device\n"
                    b"kernel.def export(\"g_ink_sconv_replay_all\") @g_ink_sconv_replay_all() {\n"
                    b"  %c16_r = index.constant 16 : index\n"
                    b"  %in_r = index.cmp ult, %row, %c16_r : index\n"
                    b"  kernel.return\n}\n"
                    b"command.def public target(@device) @llm_advance() {\n"
                    b"  kernel.launch @g_logits_export\n}\n"
                    b"command.def public target(@device) @llm_verify() {\n"
                    b"  command.return\n}\n"
                    b"command.def public target(@device) @llm_dflash_warm() {\n"
                    b"  command.return\n}\n"
                ),
                "program/prepare.loom": (
                    b"config.def @llm.card = false : i1\n"
                    b"%block_tokens = scalar.constant 8 : i64\n"
                ),
                "program/service.loom": b"// comment\nfunc.def @s() {}\n",
                "program/storage.loom": (
                    b"// comment\n"
                    b"%recurrent_bytes = scalar.constant 174227456 : i64\n"
                    b"%carry_stride = scalar.constant 8404992 : i64\n"
                ),
                "README.md": b"# readme\n",
                "REPORT.md": b"# report\n",
                "info.json": b"{}\n",
                "model.json.lock": b"{}\n",
                ".gitattributes": b"*.bin filter=lfs\n",
            }
            for rel, content in synth_files.items():
                (tmp / rel).write_bytes(content)

            orig_exec = mod.PINNED_EXECUTABLE_AND_CONFIG_FILES
            orig_meta = mod.PINNED_METADATA_FILES

            def make_entries(orig: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
                out = []
                for e in orig:
                    raw = (tmp / str(e["path"])).read_bytes()
                    out.append(dict(e, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                return tuple(out)

            try:
                mod.PINNED_EXECUTABLE_AND_CONFIG_FILES = make_entries(orig_exec)
                mod.PINNED_METADATA_FILES = make_entries(orig_meta)
                res_s = mod.audit_live_repo_mirror(tmp)
                self.assertEqual(res_s["verified_files_count"], 14)

                def assert_tamper_fails(rel: str, old: bytes, new: bytes) -> None:
                    p = tmp / rel
                    orig_b = p.read_bytes()
                    new_b = orig_b.replace(old, new, 1)
                    self.assertNotEqual(orig_b, new_b)
                    p.write_bytes(new_b)
                    mod.PINNED_EXECUTABLE_AND_CONFIG_FILES = make_entries(orig_exec)
                    mod.PINNED_METADATA_FILES = make_entries(orig_meta)
                    try:
                        with self.assertRaises(mod.LoomAuditError):
                            mod.audit_live_repo_mirror(tmp)
                    finally:
                        p.write_bytes(orig_b)
                        mod.PINNED_EXECUTABLE_AND_CONFIG_FILES = make_entries(orig_exec)
                        mod.PINNED_METADATA_FILES = make_entries(orig_meta)

                assert_tamper_fails("program/llm.loom", b"target.decl @device", b"target.decl @device\n// 1073741823")
                assert_tamper_fails("program/llm.loom", b"kernel.launch @g_logits_export", b"kernel.launch @g_noop")
                assert_tamper_fails("program/prepare.loom", b"%block_tokens = scalar.constant 8 : i64", b"%block_tokens = scalar.constant 2 : i64")
                assert_tamper_fails("program/llm.loom", b"%c16_r = index.constant 16 : index", b"%c16_r = index.constant 32 : index")
                assert_tamper_fails("program/storage.loom", b"174227456", b"174227457")
                assert_tamper_fails("program/storage.loom", b"8404992", b"8404993")
                assert_tamper_fails("head_weights.loom", b"14404176", b"14404177")
                assert_tamper_fails("head_weights.loom", b"6c6c6d5f707265705f71385f305f71345f30", b"000000000000000000000000000000000000")
                assert_tamper_fails("program/control.loom", b"func.decl import(\"input\")", b"// comment\nfunc.decl import(\"input\")")
            finally:
                mod.PINNED_EXECUTABLE_AND_CONFIG_FILES = orig_exec
                mod.PINNED_METADATA_FILES = orig_meta

    # ------------------------------------------------------------------
    # Group 3: Module 2 — Theorems S1, S2 & LOOM-IR-01..05
    # ------------------------------------------------------------------
    def test_speculative_theorems_and_ir_bugs(self) -> None:
        # Simplex & CDF guards
        with self.assertRaisesRegex(mod.LoomAuditError, "must be non-empty sequence"):
            mod.validate_simplex((), "p")
        with self.assertRaisesRegex(mod.LoomAuditError, "must be non-negative"):
            mod.validate_simplex((Fraction(-1, 2), Fraction(3, 2)), "p")
        with self.assertRaisesRegex(mod.LoomAuditError, "must sum to exact 1"):
            mod.validate_simplex((Fraction(1, 2), Fraction(1, 3)), "p")

        self.assertEqual(mod.sample_cdf_exact((Fraction(1, 2), Fraction(1, 2)), Fraction(0, 1)), 0)
        self.assertEqual(mod.sample_cdf_exact((Fraction(1, 2), Fraction(1, 2)), Fraction(1, 2)), 1)
        with self.assertRaises(mod.LoomAuditError):
            mod.sample_cdf_exact((Fraction(1, 2), Fraction(1, 2)), Fraction(-1, 10))
        with self.assertRaises(mod.LoomAuditError):
            mod.sample_cdf_exact((Fraction(1, 2), Fraction(1, 2)), Fraction(1, 1))

        # Theorem S1 guards + verification
        p0 = (Fraction(1, 2), Fraction(1, 2))
        p1 = ((Fraction(1, 4), Fraction(3, 4)), (Fraction(3, 4), Fraction(1, 4)))
        s1 = mod.verify_counter_exact_match_losslessness(p0, p1, 0, (1, 2, 3, 4), 0)
        self.assertTrue(s1["all_bit_exact"])
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_counter_exact_match_losslessness(p0, (p1[0],), 0, (1,), 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_counter_exact_match_losslessness(p0, p1, -1, (1,), 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_counter_exact_match_losslessness(p0, p1, 2, (1,), 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_counter_exact_match_losslessness(p0, p1, 0, (1,), -1)
        with self.assertRaises(mod.LoomAuditError):
            mod.verify_counter_exact_match_losslessness(p0, p1, 0, (), 0)

        # Theorem S2 LKM vs Loom
        s2_eq = mod.verify_lkm_vs_loom_coupling(
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
        )
        self.assertEqual(s2_eq["alpha_lkm"], "1/1")
        self.assertEqual(s2_eq["alpha_loom_greedy_match"], "1/4")
        self.assertEqual(s2_eq["lkm_to_loom_ratio"], "4/1")

        # Edge case: draft argmax has p(d*) == 0 -> ratio INF
        s2_zero = mod.verify_lkm_vs_loom_coupling(
            (Fraction(0, 1), Fraction(1, 1)),
            (Fraction(3, 4), Fraction(1, 4)),
        )
        self.assertEqual(s2_zero["alpha_loom_greedy_match"], "0/1")
        self.assertEqual(s2_zero["lkm_to_loom_ratio"], "INF")
        self.assertEqual(s2_zero["alpha_lkm"], "1/4")

        with self.assertRaises(mod.LoomAuditError):
            mod.verify_lkm_vs_loom_coupling((Fraction(1, 1),), (Fraction(1, 2), Fraction(1, 2)))

        # LOOM-IR-01 bit-30 top_k corruption + fix + boundaries
        b5 = mod.simulate_loom_top_k_bit30(5, True, False)
        self.assertTrue(b5["corrupted"])
        self.assertEqual(b5["effective_k"], 64)
        self.assertEqual(b5["expected_k"], 5)
        f5 = mod.simulate_loom_top_k_bit30(5, True, True)
        self.assertFalse(f5["corrupted"])
        self.assertEqual(f5["effective_k"], 5)
        b0 = mod.simulate_loom_top_k_bit30(0, False, False)
        self.assertFalse(b0["corrupted"])
        self.assertEqual(b0["effective_k"], 64)
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_loom_top_k_bit30(5, 1, False)  # type: ignore[arg-type]
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_loom_top_k_bit30(5, True, 0)  # type: ignore[arg-type]

        # LOOM-IR-03 scry top_k=64 tail distortion + guards
        sc = mod.audit_scry_top_k64_tail_distortion(64, Fraction(1, 2), Fraction(9, 10), 200058)
        self.assertEqual(sc["kept_tokens_under_loom"], 58)
        self.assertEqual(sc["dropped_true_mass_tv"], "35/64")
        # Check exact boundary head_count=1, head_mass=1, top_p=1
        sc_b = mod.audit_scry_top_k64_tail_distortion(1, Fraction(1, 1), Fraction(1, 1), 2)
        self.assertEqual(sc_b["dropped_true_mass_tv"], "0/1")
        for bad_args in (
            (0, Fraction(1, 2), Fraction(1, 2), 100),
            (65, Fraction(1, 2), Fraction(1, 2), 100),
            (64, Fraction(1, 2), Fraction(1, 2), 64),
            (64, Fraction(0, 1), Fraction(1, 2), 100),
            (64, Fraction(2, 1), Fraction(1, 2), 100),
            (64, Fraction(1, 2), Fraction(0, 1), 100),
            (64, Fraction(1, 2), Fraction(2, 1), 100),
        ):
            with self.assertRaises(mod.LoomAuditError):
                mod.audit_scry_top_k64_tail_distortion(*bad_args)

        # LOOM-IR-05 shortconv replay isolation + row>=16 silent drop + fresh + boundaries
        h0 = (Fraction(1, 1), Fraction(2, 1), Fraction(3, 1))
        cap = (Fraction(10, 1), Fraction(20, 1), Fraction(30, 1), Fraction(40, 1))
        for row_ok in (0, 15):
            for cons in (0, 1, 2, 3, 4):
                for fr in (False, True):
                    r_ok = mod.simulate_shortconv_replay_all(h0, cap, cons, fr, row_id=row_ok)
                    self.assertTrue(r_ok["replay_executed"])
                    self.assertTrue(r_ok["matches_exact_ground_truth"])
        r16 = mod.simulate_shortconv_replay_all(h0, cap, 1, False, row_id=16)
        self.assertFalse(r16["replay_executed"])
        self.assertFalse(r16["matches_exact_ground_truth"])
        r16_fixed = mod.simulate_shortconv_replay_all(h0, cap, 1, False, row_id=16, enforce_row_cap_16=False)
        self.assertTrue(r16_fixed["matches_exact_ground_truth"])

        # Shortconv boundary guards
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all((Fraction(1, 1), Fraction(2, 1)), cap, 1, False, 0)  # type: ignore[arg-type]
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, (), 0, False, 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, tuple(Fraction(i, 1) for i in range(9)), 0, False, 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, cap, -1, False, 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, cap, 5, False, 0)
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, cap, 1, 0, 0)  # type: ignore[arg-type]
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, cap, 1, False, -1)
        with self.assertRaises(mod.LoomAuditError):
            mod.simulate_shortconv_replay_all(h0, cap, 1, False, 0, enforce_row_cap_16=1)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Group 4: Modules 3 & 4 — Q8_0->Q4_0, MoE, Markov, & Zero-Float Cert
    # ------------------------------------------------------------------
    def test_q4_0_moe_markov_and_zero_float_certificate(self) -> None:
        # Q4_0 block guards + zero block + asymmetric clipping
        zeros = tuple(Fraction(0, 1) for _ in range(32))
        qz = mod.quantize_q4_0_block_exact(zeros)
        self.assertEqual(qz["d"], "0/1")
        self.assertEqual(qz["max_abs_err"], "0/1")
        with self.assertRaises(mod.LoomAuditError):
            mod.quantize_q4_0_block_exact(zeros[:31])

        # MoE coupon-collector guards & boundaries
        self.assertEqual(mod.moe_expected_distinct_experts(128, 8, 1), Fraction(8, 1))
        self.assertEqual(mod.moe_expected_distinct_experts(128, 8, 2), Fraction(31, 2))
        self.assertEqual(mod.moe_expected_distinct_experts(4, 1, 1), Fraction(1, 1))
        self.assertEqual(mod.moe_expected_distinct_experts(4, 4, 1), Fraction(4, 1))
        with self.assertRaisesRegex(mod.LoomAuditError, "num_experts must be positive"):
            mod.moe_expected_distinct_experts(0, 1, 1)
        with self.assertRaisesRegex(mod.LoomAuditError, "num_experts must be positive"):
            mod.moe_expected_distinct_experts(-1, 1, 1)
        for bad in ((4, 0, 1), (4, 5, 1), (4, 1, 0)):
            with self.assertRaises(mod.LoomAuditError):
                mod.moe_expected_distinct_experts(*bad)

        # DSpark Markov contraction guards
        self.assertEqual(
            mod.dspark_markov_contraction_step(Fraction(1, 2), Fraction(1, 2), Fraction(0, 1)),
            Fraction(1, 4),
        )
        for bad in (
            (Fraction(-1, 1), Fraction(1, 2), Fraction(0, 1)),
            (Fraction(1, 1), Fraction(1, 2), Fraction(0, 1)),
            (Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)),
            (Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)),
            (Fraction(0, 1), Fraction(1, 2), Fraction(-1, 1)),
            (Fraction(0, 1), Fraction(1, 2), Fraction(1, 1)),
        ):
            with self.assertRaises(mod.LoomAuditError):
                mod.dspark_markov_contraction_step(*bad)

        # Full certificate build: ZERO floats strictly enforced + exact invariant checks
        cert = mod.build_certificate(repo_dir=None)
        self.assertEqual(cert["verdict"], "VERIFIED_EXACT_Q_ZERO_FLOAT")
        self.assertTrue(mod.is_sha256_hex(cert["canonical_sha256"]))
        self.assertEqual(cert["module_1_lockfile_and_provenance"]["published_ring0_gate"]["passed"], False)
        self.assertEqual(cert["module_1_lockfile_and_provenance"]["repaired_ring0_gate"]["passed"], True)

        m3 = cert["module_3_proposal_head_q8_to_q4"]
        self.assertEqual(m3["q8_blocks"], 14404176)
        self.assertEqual(m3["q8_bytes_unpadded"], 489741984)
        self.assertEqual(m3["q8_bytes_padded"], 492106752)
        self.assertEqual(m3["q4_bytes_prepared"], 259275168)
        self.assertEqual(m3["compression_ratio"], "9/17")
        self.assertEqual(m3["asymmetric_clipping_witness"]["negative_extreme_relative_loss"], "1/8")
        self.assertEqual(m3["argmax_flip_witness"]["exact_argmax"], "A")
        self.assertEqual(m3["argmax_flip_witness"]["q4_draft_argmax"], "B")

        m4 = cert["module_4_roofline_moe_and_geometry"]
        self.assertEqual(m4["metrics_verified_count"], 9)
        self.assertTrue(m4["all_medians_within_half_ulp"])
        self.assertTrue(m4["all_readme_margins_exact"])
        self.assertTrue(m4["all_readme_roofline_pcts_exact"])
        self.assertTrue(m4["all_tokens_per_pass_exact"])
        self.assertEqual(len(m4["ieee754_half_tie_rounding_diffs"]), 1)
        self.assertEqual(m4["ieee754_half_tie_rounding_diffs"][0]["metric"], "prefill_2k")
        geom = m4["looped_core_memory_geometry"]
        self.assertEqual(geom["logical_layers"], 48)
        self.assertEqual(geom["kv_planes"], 16)
        self.assertEqual(geom["sconv_total_bytes"], 3833856)
        self.assertEqual(geom["swa_ring_layer_bytes"], 4259840)
        self.assertEqual(geom["recurrent_slot_bytes"], 174227456)
        self.assertEqual(geom["sc_cap_bytes"], 163577856)
        self.assertEqual(geom["dspark_carry_stride"], 8404992)
        self.assertEqual(geom["dspark_tap_row_bytes"], 36864)
        self.assertEqual(geom["dspark_markov_contraction_example"], "1/8")

        raw_json = json.dumps(cert, indent=2, sort_keys=True)
        def reject_float(s: str) -> str:
            raise AssertionError(f"float found in JSON: {s}")
        json.loads(raw_json, parse_float=reject_float)

        # CLI smoke (human + --json + --write-cert)
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "cert.json"
            rc = mod.main(["--write-cert", str(out_p)])
            self.assertEqual(rc, 0)
            self.assertTrue(out_p.is_file())
            json.loads(out_p.read_text(encoding="utf-8"), parse_float=reject_float)


if __name__ == "__main__":
    unittest.main()
