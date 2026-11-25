#!/usr/bin/env python3
"""
Track S78: Comprehensive R5 + R2 Adversarial, Zone-B, Live-Lean Exploit,
Mutation & BoolOp Test Suite.

Strict hygiene:
  * Zero bare `assert`s (unittest assertions only).
  * Zero AST floats / float().
  * Zero hardcoded host-user paths.
  * Short-write mock uses one-shot `_zero_once` (guarantees timeout=0 under mutation testing).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest import mock

TRACK_DIR = Path(__file__).resolve().parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

import s78_engram_lora_siliquant_mdl as oracle
import verify_s78_engram_lora_mdl as verifier


def _resign(env: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(env)
    p_sha = hashlib.sha256(oracle.canonical_json_bytes(out["payload"])).hexdigest()
    out["payload_sha256"] = p_sha
    eb = {
        "schema_version": out["schema_version"],
        "track_id": out["track_id"],
        "payload_sha256": p_sha,
        "payload": out["payload"],
    }
    out["certificate_sha256"] = hashlib.sha256(oracle.canonical_json_bytes(eb)).hexdigest()
    return out


class TestOraclePureZQMath(unittest.TestCase):
    def test_01_frac_formatting_and_outward_rounding(self) -> None:
        self.assertEqual(oracle.frac_str(Fraction(7, 1)), "7")
        self.assertEqual(oracle.frac_str(Fraction(7, 3)), "7/3")
        self.assertEqual(oracle.frac_to_dec_str(Fraction(0, 1), 2), "0.00")
        self.assertEqual(oracle.frac_to_dec_str(Fraction(1, 200), 2), "0.01")
        self.assertEqual(oracle.frac_to_dec_str(Fraction(49, 10000), 2), "0.00")
        self.assertEqual(oracle.frac_to_dec_str(Fraction(-3, 2), 2), "-1.50")
        self.assertEqual(oracle.frac_to_dec_str(Fraction(-1, 10000), 2), "0.00")

        lo, hi = oracle.round_out_interval(Fraction(1, 3), Fraction(2, 3), scale=1000)
        self.assertLessEqual(lo, Fraction(1, 3))
        self.assertGreaterEqual(hi, Fraction(2, 3))
        self.assertEqual(lo, Fraction(333, 1000))
        self.assertEqual(hi, Fraction(667, 1000))

    def test_02_sqrt_ln2_pi_atan_sincos_exp_pow2_log2(self) -> None:
        with self.assertRaisesRegex(ValueError, "sqrt_bracket requires non-negative Fraction"):
            oracle.sqrt_bracket(Fraction(-1, 1))
        self.assertEqual(oracle.sqrt_bracket(Fraction(0, 1)), (Fraction(0, 1), Fraction(0, 1)))
        self.assertEqual(oracle.sqrt_bracket(Fraction(9, 4)), (Fraction(3, 2), Fraction(3, 2)))
        s2_lo, s2_hi = oracle.sqrt_bracket(Fraction(2, 1), digits=12)
        self.assertLess(s2_lo * s2_lo, 2)
        self.assertGreater(s2_hi * s2_hi, 2)

        ln2_lo, ln2_hi = oracle.ln2_bracket()
        self.assertLess(ln2_lo, ln2_hi)
        self.assertEqual(oracle.frac_to_dec_str(ln2_lo, 12), "0.693147180560")
        self.assertEqual(oracle.frac_to_dec_str(ln2_hi, 12), "0.693147180560")

        pi_lo, pi_hi = oracle.pi_bracket()
        self.assertLess(pi_lo, pi_hi)
        self.assertEqual(oracle.frac_to_dec_str(pi_lo, 12), "3.141592653590")
        self.assertEqual(oracle.frac_to_dec_str(pi_hi, 12), "3.141592653590")

        self.assertEqual(oracle._atan_small_bracket(Fraction(0, 1)), (Fraction(0, 1), Fraction(0, 1)))
        self.assertEqual(oracle._atan_pos_point_bracket(Fraction(0, 1)), (Fraction(0, 1), Fraction(0, 1)))
        at1_lo, at1_hi = oracle.atan_interval_bracket(Fraction(1, 1), Fraction(1, 1))
        self.assertLessEqual(at1_lo, pi_hi / 4)
        self.assertGreaterEqual(at1_hi, pi_lo / 4)

        sc0 = oracle._sin_cos_point_bracket(Fraction(0, 1))
        self.assertEqual(sc0, ((Fraction(0, 1), Fraction(0, 1)), (Fraction(1, 1), Fraction(1, 1))))
        (s_lo, s_hi), (c_lo, c_hi) = oracle._sin_cos_point_bracket(Fraction(2, 1))
        self.assertGreater(s_lo, Fraction(9, 10))
        self.assertLess(c_hi, Fraction(0, 1))

        self.assertEqual(oracle.exp_nonneg_bracket(Fraction(0, 1)), (Fraction(1, 1), Fraction(1, 1)))
        e1_lo, e1_hi = oracle.exp_nonneg_bracket(Fraction(1, 1))
        self.assertEqual(oracle.frac_to_dec_str(e1_lo, 8), "2.71828183")
        self.assertEqual(oracle.frac_to_dec_str(e1_hi, 8), "2.71828183")

        p0_lo, p0_hi = oracle.pow2_rational_bracket(Fraction(0, 1))
        self.assertEqual((p0_lo, p0_hi), (Fraction(1, 1), Fraction(1, 1)))
        p_pos_lo, p_pos_hi = oracle.pow2_rational_bracket(Fraction(5, 2))
        self.assertLess(p_pos_lo * p_pos_lo, 32)
        self.assertGreater(p_pos_hi * p_pos_hi, 32)
        p_neg_lo, p_neg_hi = oracle.pow2_rational_bracket(Fraction(-3, 2))
        self.assertLess(p_neg_lo * p_neg_lo, Fraction(1, 8))
        self.assertGreater(p_neg_hi * p_neg_hi, Fraction(1, 8))

        with self.assertRaisesRegex(ValueError, "ln_pos_bracket requires u > 0"):
            oracle.ln_pos_bracket(Fraction(0, 1))
        with self.assertRaisesRegex(ValueError, "ln_pos_bracket requires u > 0"):
            oracle.ln_pos_bracket(Fraction(-2, 1))
        l4_lo, l4_hi = oracle.log2_pos_bracket(Fraction(4, 1))
        self.assertEqual(oracle.frac_to_dec_str(l4_lo, 10), "2.0000000000")
        self.assertEqual(oracle.frac_to_dec_str(l4_hi, 10), "2.0000000000")
        lq_lo, lq_hi = oracle.log2_pos_bracket(Fraction(1, 4))
        self.assertEqual(oracle.frac_to_dec_str(lq_lo, 10), "-2.0000000000")
        self.assertEqual(oracle.frac_to_dec_str(lq_hi, 10), "-2.0000000000")
        l1_lo, l1_hi = oracle.log2_pos_bracket(Fraction(1, 1))
        self.assertEqual(oracle.frac_to_dec_str(l1_lo, 10), "0.0000000000")
        self.assertEqual(oracle.frac_to_dec_str(l1_hi, 10), "0.0000000000")

    def test_03_mp_engine_bounds_and_symmetry(self) -> None:
        with self.assertRaises(ValueError):
            oracle.mp_capacity_at_rank(512, 1024, 0)
        with self.assertRaises(ValueError):
            oracle.mp_capacity_at_rank(512, 1024, 513)
        a = oracle.mp_capacity_at_rank(512, 1024, 128)
        b = oracle.mp_capacity_at_rank(1024, 512, 128)
        self.assertEqual(a["mp_bpp_lo_dec6"], b["mp_bpp_lo_dec6"])
        self.assertEqual(a["mp_kb_decimal_dec3"], "23.937")
        (r_s1, _), (c_s1, _) = oracle.mp_rank_and_capacity_at_phi(64, 64, Fraction(1, 1))
        (r_s2, _), (c_s2, _) = oracle.mp_rank_and_capacity_at_phi(64, 64, Fraction(5, 2))
        self.assertLess(r_s1, r_s2)
        self.assertLess(c_s1, c_s2)
        full = oracle.mp_capacity_at_rank(64, 64, 64)
        self.assertEqual(full["d_eff"], 4096)
        self.assertEqual(full["kappa_lo_dec6"], "1.000000")
        self.assertEqual(full["kappa_hi_dec6"], "1.000000")
        shallow = oracle.mp_capacity_at_rank(64, 128, 16, bisect_steps=0)
        self.assertLess(shallow["_cap_lo"], shallow["_cap_hi"])

    def test_04_all_four_pillars_and_oracle_main(self) -> None:
        env = oracle.build_certificate_envelope()
        p = env["payload"]
        self.assertTrue(p["all_pillars_verified"])
        self.assertTrue(p["pillar_1_engram_audit"]["all_verified"])
        self.assertEqual(p["pillar_1_engram_audit"]["low_rank_edge_v_proj_512x1024"]["bpp_lo_dec6"], "1.401440")
        self.assertEqual(p["pillar_1_engram_audit"]["low_rank_edge_up_proj_1024x3584"]["bpp_lo_dec6"], "1.321133")
        self.assertEqual(p["pillar_1_engram_audit"]["fig3_v_proj_rank128"]["mp_bpp_lo_dec6"], "0.973987")
        self.assertEqual(p["pillar_1_engram_audit"]["full_rank_v_proj_rank512"]["mp_bpp_lo_dec6"], "0.480898")
        self.assertEqual(p["pillar_1_engram_audit"]["empirical_anchor_receipts"]["fig4_superadditive_multiplier_dec4"], "2.5308")
        self.assertTrue(p["pillar_1_engram_audit"]["empirical_anchor_receipts"]["fig7_exact_unit_conservation"])

        p2 = p["pillar_2_gauge_vs_mp_reconciliation"]
        self.assertTrue(p2["all_verified"])
        self.assertEqual(len(p2["rank_sweep"]), 9)
        self.assertTrue(p2["eta_alg_strictly_decreasing"])
        self.assertTrue(p2["mp_bpp_strictly_decreasing"])
        self.assertTrue(p2["kappa_strictly_decreasing"])
        self.assertTrue(p2["multiplicative_decomposition_certified"])
        self.assertTrue(p2["full_rank_wilks_kappa_exact_one"])
        self.assertTrue(p2["square_1024x1024_edge"]["saturates_exact_2x"])

        p3 = p["pillar_3_s34_mdl_and_cs15_cs16"]
        self.assertTrue(p3["all_verified"])
        self.assertTrue(p3["s34_offline_online_cancellation"]["all_residuals_zero"])
        self.assertTrue(p3["engram_noise_mdl_no_profit"]["all_strictly_negative_profit"])
        self.assertEqual(p3["cs15_cs16_obstruction"]["s31_cs16_matched_ratio_q"], "66/25")
        self.assertTrue(p3["cs15_cs16_obstruction"]["s31_cs16_sign_flipped"])
        self.assertTrue(p3["cs15_cs16_obstruction"]["capacity_sufficient_for_train_not_ood"])

        p4 = p["pillar_4_joint_bpw_rank_allocation_law"]
        self.assertTrue(p4["all_verified"])
        self.assertTrue(p4["all_weight_counts_exact"])
        self.assertTrue(p4["floor_monotone_in_T"])
        q3 = p4["frozen_model_allocations"]["Qwen/Qwen2.5-3B"]
        self.assertEqual(q3["rank_unlock_all7_b16"], 1065)
        self.assertEqual(q3["rank_unlock_all7_b64_adamw"], 266)
        self.assertEqual(q3["rank_unlock_mlp3_b16"], 1414)
        self.assertEqual(q3["rtx3080_6gib_bf16_rmax_b16"], 72)
        self.assertEqual(q3["rtx3080_6gib_quant_rmax_b16"], 1138)
        g4 = p4["frozen_model_allocations"]["google/gemma-3-4b-it"]
        self.assertEqual(g4["rank_unlock_all7_b16"], 1237)
        self.assertFalse(g4["rtx3080_6gib_bf16_fits"])
        self.assertTrue(g4["rtx3080_6gib_quant_fits"])
        self.assertEqual(g4["rtx3080_6gib_quant_rmax_b16"], 884)
        self.assertTrue(p4["gemma3_4b_convex_kl_ladder"]["empirical_strict_convexity_verified"])
        sched = {row["horizon_label"]: row for row in p4["gemma3_4b_lower_bitrate_floor_schedule"]}
        self.assertTrue(sched["serving_1B"]["siliquant_4p5_above_single_floor"])
        self.assertFalse(sched["serving_1B"]["siliquant_3p0_above_single_floor"])
        self.assertFalse(sched["serving_10B"]["siliquant_4p5_above_single_floor"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = oracle.main()
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["certificate_sha256"], env["certificate_sha256"])

        bad_env = copy.deepcopy(env)
        bad_env["payload"]["all_pillars_verified"] = False
        with mock.patch.object(oracle, "build_certificate_envelope", return_value=bad_env):
            self.assertEqual(oracle.main(), 1)


class TestVerifierZoneBAndLeanR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid_env = oracle.build_certificate_envelope()

    def test_05_require_ok_sha_parse_int_and_depth(self) -> None:
        verifier.require_ok(True, "NOOP")
        with self.assertRaisesRegex(ValueError, "BOOM"):
            verifier.require_ok(False, "BOOM")

        self.assertTrue(verifier._is_sha256_hex("a" * 64))
        self.assertFalse(verifier._is_sha256_hex("A" * 64))
        self.assertFalse(verifier._is_sha256_hex("a" * 63))
        self.assertFalse(verifier._is_sha256_hex("a" * 65))
        self.assertFalse(verifier._is_sha256_hex("a" * 64 + "\n"))
        self.assertFalse(verifier._is_sha256_hex(12345))

        # _parse_strict_int boundaries (ASCII-only, exact MAX_INT_DIGITS boundary, pre-parse 5000-digit block)
        self.assertEqual(verifier._parse_strict_int("0" * verifier.MAX_INT_DIGITS), 0)
        self.assertEqual(verifier._parse_strict_int("-" + ("0" * verifier.MAX_INT_DIGITS)), 0)
        with self.assertRaisesRegex(ValueError, "JSON_INVALID_INT_TOKEN"):
            verifier._parse_strict_int("12_3")
        with self.assertRaisesRegex(ValueError, "JSON_INVALID_INT_TOKEN"):
            verifier._parse_strict_int("\u0661\u0662")
        with self.assertRaisesRegex(ValueError, "JSON_INT_TOO_LARGE"):
            verifier._parse_strict_int("9" * (verifier.MAX_INT_DIGITS + 1))
        with self.assertRaisesRegex(ValueError, "JSON_INT_TOO_LARGE"):
            verifier._parse_strict_int("9" * 5000)

        # Exact boundary depth == MAX_JSON_DEPTH must PASS; MAX_JSON_DEPTH + 1 must FAIL
        at_limit: Any = 1
        for _ in range(verifier.MAX_JSON_DEPTH):
            at_limit = [at_limit]
        verifier._check_json_depth_and_types(at_limit)

        over_limit: Any = [at_limit]
        with self.assertRaisesRegex(ValueError, "JSON_MAX_DEPTH_EXCEEDED"):
            verifier._check_json_depth_and_types(over_limit)

        # Exact bit-length boundary (256 bits PASS, 257 bits FAIL)
        verifier._check_json_depth_and_types({"v": (1 << verifier.MAX_INT_BITS) - 1, "b": True, "n": None})
        with self.assertRaisesRegex(ValueError, "JSON_INT_TOO_LARGE"):
            verifier._check_json_depth_and_types({"v": 1 << verifier.MAX_INT_BITS})

        class EvilDict(dict):
            pass

        class LyingList(list):
            pass

        with self.assertRaisesRegex(ValueError, "JSON_DISALLOWED_TYPE"):
            verifier._check_json_depth_and_types(EvilDict({"a": 1}))
        with self.assertRaisesRegex(ValueError, "JSON_DISALLOWED_TYPE"):
            verifier._check_json_depth_and_types({"a": LyingList([1, 2])})

        verifier._check_json_depth_and_types({"ok": "\ud7ff\ue000"})
        with self.assertRaisesRegex(ValueError, "JSON_LONE_SURROGATE"):
            verifier._check_json_depth_and_types({"bad": "\ud800"})
        with self.assertRaisesRegex(ValueError, "JSON_LONE_SURROGATE"):
            verifier._check_json_depth_and_types({"bad": "\udfff"})
        with self.assertRaisesRegex(ValueError, "JSON_NUL_IN_STRING"):
            verifier._check_json_depth_and_types({"bad": "a\x00b"})

        with self.assertRaisesRegex(ValueError, "JSON_NON_STRING_KEY"):
            verifier._check_json_depth_and_types({1: "x"})
        with self.assertRaisesRegex(ValueError, "JSON_DISALLOWED_TYPE"):
            verifier._check_json_depth_and_types({"a": {1, 2}})

    def test_06_load_strict_json_hardened_io_fifo_hardlink_pwd_and_parse(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            good = d / "good.json"
            raw_good = oracle.canonical_json_bytes(self.valid_env)
            good.write_bytes(raw_good)
            loaded = verifier.load_strict_json_file(good)
            self.assertEqual(loaded["certificate_sha256"], self.valid_env["certificate_sha256"])

            # Exact MAX_JSON_BYTES boundary (kills <= -> < mutants!)
            with mock.patch.object(verifier, "MAX_JSON_BYTES", len(raw_good)):
                loaded_eq = verifier.load_strict_json_file(good)
                self.assertEqual(loaded_eq["certificate_sha256"], self.valid_env["certificate_sha256"])

            # Path guards: empty, non-str, NUL, root `/`, leaf `..`
            with self.assertRaisesRegex(ValueError, "EMPTY_PATH"):
                verifier._validate_path_str("")
            with self.assertRaisesRegex(ValueError, "EMPTY_PATH"):
                verifier._validate_path_str(123)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "NUL_IN_PATH"):
                verifier._validate_path_str("a\x00b")
            with self.assertRaisesRegex(ValueError, "INVALID_PATH"):
                verifier.load_strict_json_file(Path("a\x00b"))
            with self.assertRaisesRegex(ValueError, "INVALID_PATH"):
                verifier.load_strict_json_file(Path("/"))
            with self.assertRaisesRegex(ValueError, "INVALID_PATH"):
                verifier.load_strict_json_file(d / "..")

            # Missing file, missing parent, directory, symlink target, symlink ancestor
            d_fd = os.open(str(d), os.O_RDONLY | os.O_DIRECTORY)
            try:
                self.assertIs(verifier._is_symlink_at(d_fd, "missing.json"), False)
            finally:
                os.close(d_fd)
            with self.assertRaisesRegex(ValueError, "OPEN_FAILED"):
                verifier.load_strict_json_file(d / "missing.json")
            with self.assertRaisesRegex(ValueError, "INVALID_PARENT_DIR"):
                verifier.load_strict_json_file(d / "no_dir" / "missing.json")
            with self.assertRaisesRegex(ValueError, "NOT_REGULAR_FILE"):
                verifier.load_strict_json_file(d)
            sym = d / "sym.json"
            sym.symlink_to(good)
            with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                verifier.load_strict_json_file(sym)
            real_sub = d / "real_sub"
            real_sub.mkdir()
            (real_sub / "g.json").write_bytes(raw_good)
            sym_anc = d / "sym_anc"
            sym_anc.symlink_to(real_sub)
            with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                verifier.load_strict_json_file(sym_anc / "g.json")

            # S78-F2: Non-blocking FIFO rejection (must fail fast with NOT_REGULAR_FILE, never hang!)
            fifo_path = d / "pipe.fifo"
            os.mkfifo(fifo_path)
            t0 = time.monotonic_ns()
            with self.assertRaisesRegex(ValueError, "NOT_REGULAR_FILE"):
                verifier.load_strict_json_file(fifo_path)
            self.assertLess(time.monotonic_ns() - t0, 1_000_000_000)

            # S78-F2: Hardlink read rejection (st_nlink > 1 -> HARDLINK_FORBIDDEN)
            hl_read = d / "hl_read.json"
            os.link(good, hl_read)
            try:
                with self.assertRaisesRegex(ValueError, "HARDLINK_FORBIDDEN"):
                    verifier.load_strict_json_file(hl_read)
            finally:
                hl_read.unlink()

            # S78-F2: Logical $PWD symlink ancestor defense (cd -L sym_anc && --check g.json)
            proc_pwd = subprocess.run(
                [sys.executable, "-I", "-B", str(TRACK_DIR / "verify_s78_engram_lora_mdl.py"), "--check", "g.json"],
                cwd=str(sym_anc),
                env={"PATH": os.environ.get("PATH", ""), "PWD": str(sym_anc)},
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertEqual(proc_pwd.returncode, 1)
            self.assertIn("SYMLINK_IN_PATH", proc_pwd.stderr)

            # $PWD edge cases: unset, relative, NUL in PWD, missing stale PWD
            orig_cwd = os.getcwd()
            try:
                os.chdir(real_sub)
                with mock.patch.dict(os.environ, {"PWD": str(real_sub)}, clear=False):
                    self.assertEqual(verifier.load_strict_json_file(Path("g.json"))["certificate_sha256"], self.valid_env["certificate_sha256"])
                with mock.patch.dict(os.environ, {"PWD": "relative/pwd"}, clear=False):
                    self.assertEqual(verifier.load_strict_json_file(Path("g.json"))["certificate_sha256"], self.valid_env["certificate_sha256"])
                with mock.patch.dict(os.environ, {"PWD": str(d / "stale_nonexistent")}, clear=False):
                    self.assertEqual(verifier.load_strict_json_file(Path("g.json"))["certificate_sha256"], self.valid_env["certificate_sha256"])
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(verifier.load_strict_json_file(Path("g.json"))["certificate_sha256"], self.valid_env["certificate_sha256"])
            finally:
                os.chdir(orig_cwd)

            # Adversarial JSON parse cases (incl. pre-parse 5000-digit huge int & RecursionError)
            cases = [
                ("dup.json", b'{"a":1,"a":2}\n', "JSON_DUPLICATE_KEY"),
                ("flt.json", b'{"a":1.5}\n', "JSON_FLOAT_FORBIDDEN"),
                ("nan.json", b'{"a":NaN}\n', "JSON_CONSTANT_FORBIDDEN"),
                ("huge.json", (b'{"a":' + (b"9" * 5000) + b"}\n"), "JSON_INT_TOO_LARGE"),
                ("syn.json", b"{bad\n", "JSON_SYNTAX_ERROR"),
                ("arr.json", b"[1,2,3]\n", "JSON_ROOT_NOT_OBJECT"),
                ("nul.json", b'{"a":"x\x00y"}\n', "NUL_BYTE_IN_JSON"),
                ("crlf.json", b'{"a":1}\r\n', "CRLF_FORBIDDEN"),
                ("pretty.json", b'{\n  "a": 1\n}\n', "NON_CANONICAL_JSON_BYTES"),
                ("utf.json", b"\xff\xfe", "INVALID_UTF8"),
            ]
            for fname, raw, err_pat in cases:
                fp = d / fname
                fp.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, err_pat):
                    verifier.load_strict_json_file(fp)

            with mock.patch("json.loads", side_effect=RecursionError("deep")):
                with self.assertRaisesRegex(ValueError, "JSON_MAX_DEPTH_EXCEEDED"):
                    verifier.load_strict_json_file(good)

            # Oversized via fstat check AND via read-loop check
            fake_st = os.stat_result((stat.S_IFREG | 0o600, 0, 0, 1, 0, 0, verifier.MAX_JSON_BYTES + 1, 0, 0, 0))
            with mock.patch("os.fstat", return_value=fake_st):
                with self.assertRaisesRegex(ValueError, "FILE_TOO_LARGE"):
                    verifier.load_strict_json_file(good)

            orig_read = os.read
            calls = [0]

            def _big_read(fd: int, n: int) -> bytes:
                calls[0] += 1
                if calls[0] <= 2:
                    return b" " * (verifier.MAX_JSON_BYTES // 2 + 10)
                return orig_read(fd, n)

            with mock.patch("os.read", side_effect=_big_read):
                with self.assertRaisesRegex(ValueError, "FILE_TOO_LARGE"):
                    verifier.load_strict_json_file(good)

    def test_07_atomic_write_0600_toctou_and_short_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            out = d / "cert.json"
            payload = oracle.canonical_json_bytes(self.valid_env)
            verifier.atomic_write_0600(out, payload)
            self.assertEqual(stat.S_IMODE(os.lstat(out).st_mode), 0o600)
            self.assertEqual(out.read_bytes(), payload)

            # Overwrite existing regular file cleanly
            verifier.atomic_write_0600(out, payload)
            self.assertEqual(stat.S_IMODE(os.lstat(out).st_mode), 0o600)

            with self.assertRaisesRegex(ValueError, "INVALID_PATH"):
                verifier.atomic_write_0600(Path("a\x00b"), payload)
            with self.assertRaisesRegex(ValueError, "INVALID_PARENT_DIR"):
                verifier.atomic_write_0600(d / "nope" / "c.json", payload)
            sub_real = d / "real_dir"
            sub_real.mkdir()
            sub_sym = d / "sym_dir"
            sub_sym.symlink_to(sub_real)
            with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                verifier.atomic_write_0600(sub_sym / "c.json", payload)
            tgt_sym = d / "tgt_sym.json"
            tgt_sym.symlink_to(out)
            with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                verifier.atomic_write_0600(tgt_sym, payload)
            with self.assertRaisesRegex(ValueError, "TARGET_NOT_REGULAR"):
                verifier.atomic_write_0600(sub_real, payload)

            # Hardlinked target rejected (nlink > 1)
            hl_src = d / "hl_src.json"
            hl_src.write_bytes(payload)
            hl_dst = d / "hl_dst.json"
            os.link(hl_src, hl_dst)
            with self.assertRaisesRegex(ValueError, "HARDLINK_FORBIDDEN"):
                verifier.atomic_write_0600(hl_dst, payload)

            # Temp file hardlinked mid-write rejected (TMP_HARDLINK_DETECTED) + cleaned up
            fake_hl_st = os.stat_result((stat.S_IFREG | 0o600, 0, 0, 2, 0, 0, len(payload), 0, 0, 0))
            out_tmp_hl = d / "tmp_hl.json"
            with mock.patch("os.fstat", return_value=fake_hl_st):
                with self.assertRaisesRegex(ValueError, "TMP_HARDLINK_DETECTED"):
                    verifier.atomic_write_0600(out_tmp_hl, payload)
            self.assertEqual([p.name for p in d.iterdir() if p.name.endswith(".tmp")], [])

            # Pre-os.replace TOCTOU guard: target swapped to symlink during fsync!
            toctou_tgt = d / "toctou.json"
            real_fsync = os.fsync
            swapped = [False]

            def _swap_on_fsync(fd: int) -> None:
                real_fsync(fd)
                if not swapped[0]:
                    swapped[0] = True
                    toctou_tgt.symlink_to(out)

            with mock.patch("os.fsync", side_effect=_swap_on_fsync):
                with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                    verifier.atomic_write_0600(toctou_tgt, payload)
            self.assertEqual([p.name for p in d.iterdir() if p.name.endswith(".tmp")], [])

            # One-shot short-write mock (guarantees timeout=0 under mutation testing!)
            fired = [False]
            real_write = os.write

            def _zero_once(fd: int, data: bytes) -> int:
                if not fired[0]:
                    fired[0] = True
                    return 0
                return real_write(fd, data)

            out2 = d / "short.json"
            with mock.patch("os.write", side_effect=_zero_once):
                with self.assertRaisesRegex(OSError, "SHORT_WRITE"):
                    verifier.atomic_write_0600(out2, payload)
            self.assertFalse(out2.exists())
            self.assertEqual([p.name for p in d.iterdir() if p.name.endswith(".tmp")], [])

    def test_08_lean_r10_static_audit_live_exploits_and_kernel(self) -> None:
        # Clean quote-free, guillemet-free source
        clean_src = (
            "-- sorry in line comment\n"
            "/- /- admit \"quoted\" 'x' «bad» nested -/ -/\n"
            "theorem ok : 1 = 1 := by decide\n"
            "#print axioms ok\n"
        )
        res_clean = verifier.audit_lean_source_static(clean_src)
        self.assertTrue(res_clean["clean"], msg=str(res_clean))

        # Comprehensive R10+ static exploit matrix (including AUDIT-178 S78-F1 & AUDIT-176 bypasses!)
        bad_cases = [
            # Quote-parity inversion & comment-delimiter smuggling (AUDIT-178 S78-F1)
            ('def _a : String := s!"x"\n#eval 1\ndef _b : Char := \'"\'\n', "LEAN_QUOTE_OR_GUILLEMET_FORBIDDEN"),
            ('def _a : String := s!"x"\naxiom fake : False\ndef _b : Char := \'"\'\n', "LEAN_FORBIDDEN_KEYWORD"),
            ('def _s := s! "x"\ntheorem t : 1 = 1 := by native_decide\n', "LEAN_FORBIDDEN_KEYWORD"),
            ('def _open := "/-"\naxiom fake : False\ndef _close := "-/"\n', "LEAN_QUOTE_OR_GUILLEMET_FORBIDDEN"),
            ('def _r := r#"""#\n', "LEAN_QUOTE_OR_GUILLEMET_FORBIDDEN"),
            ("def _c : Char := 'a'\n", "LEAN_QUOTE_OR_GUILLEMET_FORBIDDEN"),
            ("set_option «debug».«skipKernelTC» true\n", "LEAN_SKIP_KERNEL_TC"),
            ("set_option «debug».«skipKernelTC» true\n", "LEAN_QUOTE_OR_GUILLEMET_FORBIDDEN"),
            # Forbidden keywords & meta-programming / IO / native escape hatches
            ("theorem t : False := by sorry", "LEAN_FORBIDDEN_KEYWORD"),
            ("theorem t : False := by admit", "LEAN_FORBIDDEN_KEYWORD"),
            ("axiom bogus : False", "LEAN_FORBIDDEN_KEYWORD"),
            ("unsafe def u := 1", "LEAN_FORBIDDEN_KEYWORD"),
            ("theorem t : 1 = 1 := by native_decide", "LEAN_FORBIDDEN_KEYWORD"),
            ("def x := sorryAx", "LEAN_FORBIDDEN_KEYWORD"),
            ("def x := Lean.ofReduceBool", "LEAN_FORBIDDEN_KEYWORD"),
            ("def x := Lean.ofReduceNat", "LEAN_FORBIDDEN_KEYWORD"),
            ("def x := Lean.trustCompiler", "LEAN_FORBIDDEN_KEYWORD"),
            ("def x := Lean.Meta.mkSorry", "LEAN_FORBIDDEN_KEYWORD"),
            ("elab bad : term => pure ()", "LEAN_FORBIDDEN_KEYWORD"),
            ("macro_rules | bad => 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("syntax bad : term", "LEAN_FORBIDDEN_KEYWORD"),
            ("declare_syntax_cat mycat", "LEAN_FORBIDDEN_KEYWORD"),
            ("notation myop => 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("prefix myop => 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("infix myop => 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("initialize x : Nat := 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("builtin_initialize y : Nat := 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("opaque hidden : Nat", "LEAN_FORBIDDEN_KEYWORD"),
            ("def f : Nat := dbg_trace 1; 0", "LEAN_FORBIDDEN_KEYWORD"),
            ("run_cmd pure ()", "LEAN_FORBIDDEN_KEYWORD"),
            ("run_meta pure ()", "LEAN_FORBIDDEN_KEYWORD"),
            # Partial, decide +native variants (incl. revert & DecideConfig), debug.*, #print spoofs, @[csimp], imports
            ("partial def p (n : Nat) : Nat := p n", "LEAN_PARTIAL_DECL"),
            ("theorem t : True := by decide +native", "LEAN_DECIDE_NATIVE"),
            ("theorem t : True := by decide (revert := true) +native", "LEAN_DECIDE_NATIVE"),
            ("theorem t : True := by decide -revert +native", "LEAN_DECIDE_NATIVE"),
            ("theorem t : True := by decide +revert +native", "LEAN_DECIDE_NATIVE"),
            ("theorem t : True := by decide (config := { native := true })", "LEAN_DECIDE_NATIVE"),
            ("theorem t : True := by decide (config := { native := !false })", "LEAN_DECIDE_NATIVE"),
            ("def cfg : DecideConfig := {}", "LEAN_DECIDE_NATIVE"),
            ("set_option debug.skipKernelTC true", "LEAN_SKIP_KERNEL_TC"),
            ("#eval 1 + 1", "LEAN_EVAL_OR_PRINT_SPOOF"),
            ("#eval! 1 + 1", "LEAN_EVAL_OR_PRINT_SPOOF"),
            ("#reduce 1 + 1", "LEAN_EVAL_OR_PRINT_SPOOF"),
            ("#print Nat.add", "LEAN_EVAL_OR_PRINT_SPOOF"),
            ("#print axioms example", "LEAN_EVAL_OR_PRINT_SPOOF"),
            ("#exit", "LEAN_EVAL_OR_PRINT_SPOOF"),
            ("@[extern foo] def f : Nat := 0", "LEAN_ATTR_TRUST_ESCAPE"),
            ("attribute [implemented_by g] f", "LEAN_ATTR_TRUST_ESCAPE"),
            ("@[csimp] theorem bad : 1 = 1 := rfl", "LEAN_ATTR_TRUST_ESCAPE"),
            ("import Mathlib.Data.Nat.Basic", "LEAN_NON_ZERO_IMPORTS"),
            ("import Lean", "LEAN_NON_ZERO_IMPORTS"),
            ("/- unterminated comment", "LEAN_UNTERMINATED_BLOCK_COMMENT"),
        ]
        for snippet, code in bad_cases:
            r = verifier.audit_lean_source_static(snippet)
            self.assertFalse(r["clean"], msg=f"Expected failure on: {snippet}")
            self.assertTrue(any(f.startswith(code) for f in r["findings"]), msg=f"{snippet} -> {r}")

        # Live Lean 4.36.0 kernel check on real frozen LoraCapacityJointMdl.lean
        lean_file = TRACK_DIR / "LoraCapacityJointMdl.lean"
        live = verifier.verify_lean_proof(lean_file)
        self.assertTrue(live["static_clean"])
        self.assertTrue(live["sha256_pinned"])
        self.assertTrue(live["zero_axioms_all"])
        self.assertEqual(live["theorems_verified"], len(verifier.EXPECTED_LEAN_THEOREMS))
        self.assertEqual(live["imports"], [])

        # Live E2E exploit regression tests (EVEN with enforce_sha256=False!) + SHA-256 trivialization block!
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # 1. Exact AUDIT-178 S78-F1 full 17x #eval + s!"x" + def _b : Char := '"' forgery!
            eval_lines = [
                'def _a : String := s!"x"',
                *[f'#eval IO.println "\'{thm}\' does not depend on any axioms"' for thm in verifier.EXPECTED_LEAN_THEOREMS],
                "def _b : Char := '\"'",
            ]
            forged_eval = d / "forged_eval.lean"
            forged_eval.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LEAN_STATIC_AUDIT_FAILED"):
                verifier.verify_lean_proof(forged_eval, enforce_sha256=False)

            # 2. Exact AUDIT-178 S78-F1 decide (revert := true) +native & decide -revert +native bypasses
            for idx, tactic in enumerate((
                "decide (revert := true) +native",
                "decide -revert +native",
                "decide +revert +native",
            )):
                bad_nat = d / f"bad_nat_{idx}.lean"
                bad_nat.write_text(f"theorem t : 1 = 1 := by {tactic}\n#print axioms t\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "LEAN_STATIC_AUDIT_FAILED"):
                    verifier.verify_lean_proof(bad_nat, enforce_sha256=False)

            # 3. Trivialized theorem substitution (`theorem ... : True := trivial`) — passes static & live Lean,
            #    BLOCKED by default EXPECTED_LEAN_SHA256 pin!
            triv_lines = ["namespace TrackS78LoraJointMdl"]
            for thm in verifier.EXPECTED_LEAN_THEOREMS:
                short = thm.split(".", 1)[1]
                triv_lines.append(f"theorem {short} : True := trivial")
            triv_lines.append("end TrackS78LoraJointMdl")
            for thm in verifier.EXPECTED_LEAN_THEOREMS:
                triv_lines.append(f"#print axioms {thm}")
            triv_lean = d / "trivialized.lean"
            triv_lean.write_text("\n".join(triv_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LEAN_SHA256_MISMATCH"):
                verifier.verify_lean_proof(triv_lean)
            # And non-bool enforce_sha256 rejected
            with self.assertRaisesRegex(ValueError, "INVALID_ENFORCE_SHA256_FLAG"):
                verifier.verify_lean_proof(lean_file, enforce_sha256=0)  # type: ignore[arg-type]

            # Directory, symlink, invalid UTF-8 rejected
            with self.assertRaisesRegex(ValueError, "NOT_REGULAR_FILE"):
                verifier.verify_lean_proof(d)
            sym = d / "sym.lean"
            sym.symlink_to(lean_file)
            with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                verifier.verify_lean_proof(sym)
            bad_utf = d / "bad_utf.lean"
            bad_utf.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(ValueError, "LEAN_INVALID_UTF8"):
                verifier.verify_lean_proof(bad_utf)

        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "LEAN_BINARY_NOT_FOUND"):
                verifier.verify_lean_proof(lean_file)

        cp_fail = subprocess.CompletedProcess(["lean"], 1, stdout="", stderr="err")
        with mock.patch("subprocess.run", return_value=cp_fail):
            with self.assertRaisesRegex(ValueError, "LEAN_COMPILE_FAILED"):
                verifier.verify_lean_proof(lean_file)

        cp_stderr = subprocess.CompletedProcess(["lean"], 0, stdout="", stderr="warning: unused")
        with mock.patch("subprocess.run", return_value=cp_stderr):
            with self.assertRaisesRegex(ValueError, "LEAN_NONEMPTY_STDERR"):
                verifier.verify_lean_proof(lean_file)

        cp_warn = subprocess.CompletedProcess(["lean"], 0, stdout="warning: x\n", stderr="")
        with mock.patch("subprocess.run", return_value=cp_warn):
            with self.assertRaisesRegex(ValueError, "LEAN_UNEXPECTED_STDOUT_LINE"):
                verifier.verify_lean_proof(lean_file)

        cp_short = subprocess.CompletedProcess(
            ["lean"],
            0,
            stdout="'TrackS78LoraJointMdl.gauge_quotient_eff_dof' does not depend on any axioms\n",
            stderr="",
        )
        with mock.patch("subprocess.run", return_value=cp_short):
            with self.assertRaisesRegex(ValueError, "LEAN_THEOREM_LIST_MISMATCH"):
                verifier.verify_lean_proof(lean_file)

        wrapped_lines = []
        for idx, thm in enumerate(verifier.EXPECTED_LEAN_THEOREMS):
            if idx == 0:
                wrapped_lines.append(f"'{thm}'\n  does not depend on any axioms")
            else:
                wrapped_lines.append(f"'{thm}' does not depend on any axioms")
        cp_wrap = subprocess.CompletedProcess(["lean"], 0, stdout="\n".join(wrapped_lines) + "\n", stderr="")
        with mock.patch("subprocess.run", return_value=cp_wrap):
            res_wrap = verifier.verify_lean_proof(lean_file)
            self.assertEqual(res_wrap["theorems_verified"], len(verifier.EXPECTED_LEAN_THEOREMS))

        cp_lead_ws = subprocess.CompletedProcess(["lean"], 0, stdout="  stray indented line\n", stderr="")
        with mock.patch("subprocess.run", return_value=cp_lead_ws):
            with self.assertRaisesRegex(ValueError, "LEAN_UNEXPECTED_STDOUT_LINE"):
                verifier.verify_lean_proof(lean_file)

    def test_09_verify_certificate_dict_fr1_fr0_and_secondary_audit_tamper_matrix(self) -> None:
        # FR1 (full_replay=True) and genuine FR0 (full_replay=False, never calls build_certificate_envelope!)
        ok_fr1 = verifier.verify_certificate_dict(self.valid_env, full_replay=True)
        self.assertTrue(ok_fr1["all_pillars_verified"])
        self.assertTrue(ok_fr1["full_replay"])
        self.assertTrue(ok_fr1["secondary_audit_passed"])

        with mock.patch.object(verifier, "build_certificate_envelope", side_effect=RuntimeError("NO_REPLAY_IN_FR0")):
            ok_fr0 = verifier.verify_certificate_dict(self.valid_env, full_replay=False)
            self.assertTrue(ok_fr0["all_pillars_verified"])
            self.assertFalse(ok_fr0["full_replay"])
            self.assertTrue(ok_fr0["secondary_audit_passed"])

        # Strict bool check on full_replay
        with self.assertRaisesRegex(ValueError, "INVALID_FULL_REPLAY_FLAG"):
            verifier.verify_certificate_dict(self.valid_env, full_replay=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "INVALID_FULL_REPLAY_FLAG"):
            verifier.verify_certificate_dict(self.valid_env, full_replay=0)  # type: ignore[arg-type]

        # Envelope rejections (both via verify_certificate_dict and directly via secondary_audit_payload!)
        with self.assertRaisesRegex(ValueError, "CERT_NOT_DICT"):
            verifier.secondary_audit_payload([])

        extra = copy.deepcopy(self.valid_env)
        extra["extra"] = 1
        with self.assertRaisesRegex(ValueError, "CERT_TOP_KEYS_MISMATCH"):
            verifier.secondary_audit_payload(extra)

        bad_schema = copy.deepcopy(self.valid_env)
        bad_schema["schema_version"] = "wrong"
        with self.assertRaisesRegex(ValueError, "SCHEMA_VERSION_MISMATCH"):
            verifier.secondary_audit_payload(bad_schema)

        bad_track = copy.deepcopy(self.valid_env)
        bad_track["track_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "TRACK_ID_MISMATCH"):
            verifier.secondary_audit_payload(bad_track)

        bad_psha = copy.deepcopy(self.valid_env)
        bad_psha["payload_sha256"] = "not-hex"
        with self.assertRaisesRegex(ValueError, "INVALID_PAYLOAD_SHA256"):
            verifier.secondary_audit_payload(bad_psha)

        bad_csha = copy.deepcopy(self.valid_env)
        bad_csha["certificate_sha256"] = "not-hex"
        with self.assertRaisesRegex(ValueError, "INVALID_CERT_SHA256"):
            verifier.secondary_audit_payload(bad_csha)

        bad_pdict = copy.deepcopy(self.valid_env)
        bad_pdict["payload"] = []
        with self.assertRaisesRegex(ValueError, "PAYLOAD_NOT_DICT"):
            verifier.secondary_audit_payload(bad_pdict)

        tampered_p = copy.deepcopy(self.valid_env)
        tampered_p["payload"]["novelty_tier"] = "N99"
        with self.assertRaisesRegex(ValueError, "PAYLOAD_SHA256_TAMPERED"):
            verifier.secondary_audit_payload(tampered_p)

        tampered_c = copy.deepcopy(self.valid_env)
        tampered_c["certificate_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "CERTIFICATE_SHA256_TAMPERED"):
            verifier.secondary_audit_payload(tampered_c)

        # FR1-only oracle drift checks (payload passes secondary_audit_payload, caught by oracle replay!)
        drift_env = copy.deepcopy(self.valid_env)
        drift_env["payload"]["pillar_1_engram_audit"]["lora_parameterization"] = "DRIFTED"
        drift_env = _resign(drift_env)
        verifier.secondary_audit_payload(drift_env)  # Passes FR0/SEC (non-invariant text leaf)
        with self.assertRaisesRegex(ValueError, "ORACLE_PAYLOAD_DRIFT"):
            verifier.verify_certificate_dict(drift_env, full_replay=True)

        fake_recomp = copy.deepcopy(self.valid_env)
        fake_recomp["certificate_sha256"] = "f" * 64
        with mock.patch.object(verifier, "build_certificate_envelope", return_value=fake_recomp):
            with self.assertRaisesRegex(ValueError, "ORACLE_CERT_DRIFT"):
                verifier.verify_certificate_dict(self.valid_env, full_replay=True)

        # Exhaustive re-signed leaf-tamper matrix for EVERY independent guard in secondary_audit_payload (FR0/SEC)
        def _tamper(mutator: Any, err_code: str) -> None:
            e = copy.deepcopy(self.valid_env)
            mutator(e["payload"])
            forged = _resign(e)
            with self.assertRaisesRegex(ValueError, err_code):
                verifier.secondary_audit_payload(forged)

        _tamper(lambda p: p.pop("novelty_tier"), "PAYLOAD_KEYS_MISMATCH")
        _tamper(lambda p: p.__setitem__("track_id", "BAD"), "PAYLOAD_TRACK_MISMATCH")
        _tamper(lambda p: p.__setitem__("oracle_class", "approx"), "ORACLE_CLASS_MISMATCH")
        _tamper(lambda p: p.__setitem__("novelty_tier", "N0"), "NOVELTY_TIER_MISMATCH")
        _tamper(lambda p: p.__setitem__("all_pillars_verified", False), "PILLARS_NOT_VERIFIED")
        _tamper(lambda p: p.__setitem__("constants_enclosures", []), "CONSTANTS_NOT_DICT")
        _tamper(lambda p: p["constants_enclosures"].__setitem__("ln2_lo_dec12", "0.7"), "LN2_ENCLOSURE_MISMATCH")
        _tamper(lambda p: p["constants_enclosures"].__setitem__("pi_lo_dec12", "3.1"), "PI_ENCLOSURE_MISMATCH")
        _tamper(lambda p: p.__setitem__("epistemic_quarantine", []), "QUARANTINE_NOT_DICT")
        _tamper(lambda p: p["epistemic_quarantine"].__setitem__("analytic_extrapolations_quarantined", "x"), "QUARANTINE_LIST_INVALID")
        _tamper(lambda p: p["epistemic_quarantine"].__setitem__("analytic_extrapolations_quarantined", []), "QUARANTINE_LIST_INVALID")

        # Pillar 1 & gauge row checks (including strict int rejecting bool `1 == True`)
        _tamper(lambda p: p.__setitem__("pillar_1_engram_audit", []), "P1_NOT_DICT")
        _tamper(lambda p: p["pillar_1_engram_audit"].__setitem__("all_verified", False), "P1_NOT_VERIFIED")
        _tamper(lambda p: p["pillar_1_engram_audit"].__setitem__("fig3_v_proj_rank128", []), "P1_R128_NOT_DICT")
        _tamper(lambda p: p["pillar_1_engram_audit"].__setitem__("full_rank_v_proj_rank512", []), "P1_R512_NOT_DICT")
        _tamper(lambda p: p["pillar_1_engram_audit"].__setitem__("empirical_anchor_receipts", []), "P1_EMP_NOT_DICT")
        _tamper(lambda p: p["pillar_1_engram_audit"]["fig3_v_proj_rank128"].__setitem__("m", 513), "GAUGE_M_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["fig3_v_proj_rank128"].__setitem__("n", 1025), "GAUGE_N_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["fig3_v_proj_rank128"].__setitem__("rank", 129), "GAUGE_RANK_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["fig3_v_proj_rank128"].__setitem__("p_raw", 1), "GAUGE_PRAW_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["fig3_v_proj_rank128"].__setitem__("d_eff", 1), "GAUGE_QUOTIENT_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["fig3_v_proj_rank128"].__setitem__("mp_kb_decimal_dec3", "0"), "P1_FIG3_KB_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["full_rank_v_proj_rank512"].__setitem__("kappa_lo_dec6", "2"), "P1_FULL_RANK_KAPPA_MISMATCH")
        _tamper(lambda p: p["pillar_1_engram_audit"]["empirical_anchor_receipts"].__setitem__("fig7_exact_unit_conservation", False), "P1_FIG7_CONSERVATION_FAILED")

        # Pillar 2 (incl. ZB05 rank=1 -> True forgery!)
        _tamper(lambda p: p.__setitem__("pillar_2_gauge_vs_mp_reconciliation", []), "P2_NOT_DICT")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"].__setitem__("all_verified", False), "P2_NOT_VERIFIED")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"].__setitem__("rank_sweep", {}), "P2_SWEEP_NOT_LIST")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"].__setitem__("rank_sweep", []), "P2_SWEEP_LEN_MISMATCH")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"]["rank_sweep"].__setitem__(0, []), "GAUGE_ROW_NOT_DICT")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"]["rank_sweep"][0].__setitem__("rank", True), "GAUGE_INT_INVALID")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"].__setitem__("square_1024x1024_edge", []), "P2_SQUARE_NOT_DICT")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"]["square_1024x1024_edge"].__setitem__("saturates_exact_2x", False), "P2_SQUARE_SATURATION_FAILED")
        _tamper(lambda p: p["pillar_2_gauge_vs_mp_reconciliation"].__setitem__("full_rank_wilks_kappa_exact_one", False), "P2_WILKS_CLOSURE_FAILED")

        # Pillar 3 (each cancellation invariant independently!)
        _tamper(lambda p: p.__setitem__("pillar_3_s34_mdl_and_cs15_cs16", []), "P3_NOT_DICT")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"].__setitem__("all_verified", False), "P3_NOT_VERIFIED")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"].__setitem__("s34_offline_online_cancellation", []), "P3_CANC_NOT_DICT")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"].__setitem__("regimes", {}), "P3_REGIMES_NOT_LIST")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"].__setitem__("regimes", []), "P3_REGIMES_LEN_MISMATCH")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"]["regimes"].__setitem__(0, []), "P3_REGIME_NOT_DICT")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"]["regimes"][0].__setitem__("cancellation_residual", False), "P3_INT_INVALID")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"]["regimes"][0].__setitem__("L_offline_bits", 71), "P3_CANC_DIFF_MISMATCH")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"]["regimes"][0].__setitem__("b_theta_bits", 13), "P3_CANC_RESIDUAL_MISMATCH")

        def _mut_nonzero_res(p: dict[str, Any]) -> None:
            r0 = p["pillar_3_s34_mdl_and_cs15_cs16"]["s34_offline_online_cancellation"]["regimes"][0]
            r0["b_theta_bits"] = 11
            r0["cancellation_residual"] = 1

        _tamper(_mut_nonzero_res, "P3_CANC_NONZERO_RESIDUAL")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"].__setitem__("engram_noise_mdl_no_profit", []), "P3_NOISE_NOT_DICT")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["engram_noise_mdl_no_profit"].__setitem__("all_strictly_negative_profit", False), "P3_NOISE_PROFIT_FAILED")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"].__setitem__("cs15_cs16_obstruction", []), "P3_OBS_NOT_DICT")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["cs15_cs16_obstruction"].__setitem__("net_joint_bit_gain_bytes", 1), "P3_NET_GAIN_NONZERO")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["cs15_cs16_obstruction"].__setitem__("s31_cs16_matched_ratio_q", "1/1"), "P3_CS16_RATIO_MISMATCH")
        _tamper(lambda p: p["pillar_3_s34_mdl_and_cs15_cs16"]["cs15_cs16_obstruction"].__setitem__("capacity_sufficient_for_train_not_ood", False), "P3_CS15_FLAG_FAILED")

        # Pillar 4
        _tamper(lambda p: p.__setitem__("pillar_4_joint_bpw_rank_allocation_law", []), "P4_NOT_DICT")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"].__setitem__("all_verified", False), "P4_NOT_VERIFIED")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"].__setitem__("frozen_model_allocations", []), "P4_MODELS_NOT_DICT")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"].pop("Qwen/Qwen2.5-0.5B"), "P4_MODEL_KEYS_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"].__setitem__("Qwen/Qwen2.5-0.5B", []), "P4_MODEL_NOT_DICT")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"]["Qwen/Qwen2.5-0.5B"].__setitem__("weight_count_exact_match", False), "P4_WEIGHT_MATCH_FAILED")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"]["google/gemma-3-4b-it"].__setitem__("rtx3080_6gib_bf16_rmax_b16", False), "P4_INT_INVALID")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"]["Qwen/Qwen2.5-3B"].__setitem__("rank_unlock_all7_b16", 999), "P4_QWEN3B_UNLOCK_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"]["google/gemma-3-4b-it"].__setitem__("rtx3080_6gib_bf16_fits", True), "P4_GEMMA_BF16_FITS_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["frozen_model_allocations"]["google/gemma-3-4b-it"].__setitem__("rtx3080_6gib_quant_fits", False), "P4_GEMMA_QUANT_FITS_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"].__setitem__("gemma3_4b_convex_kl_ladder", []), "P4_LADDER_NOT_DICT")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["gemma3_4b_convex_kl_ladder"].__setitem__("empirical_strict_convexity_verified", False), "P4_CONVEXITY_FAILED")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["gemma3_4b_convex_kl_ladder"].__setitem__("extrapolated_3p0_bpw_quarantine", "LIVE"), "P4_QUARANTINE_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"].__setitem__("gemma3_4b_lower_bitrate_floor_schedule", {}), "P4_SCHED_NOT_LIST")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"].__setitem__("gemma3_4b_lower_bitrate_floor_schedule", []), "P4_SCHED_LEN_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["gemma3_4b_lower_bitrate_floor_schedule"].__setitem__(2, []), "P4_SCHED_ROW_NOT_DICT")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["gemma3_4b_lower_bitrate_floor_schedule"][2].__setitem__("horizon_label", "bad"), "P4_SCHED_1B_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["gemma3_4b_lower_bitrate_floor_schedule"][2].__setitem__("siliquant_4p5_above_single_floor", False), "P4_SCHED_4P5_MISMATCH")
        _tamper(lambda p: p["pillar_4_joint_bpw_rank_allocation_law"]["gemma3_4b_lower_bitrate_floor_schedule"][2].__setitem__("siliquant_3p0_above_single_floor", True), "P4_SCHED_3P0_MISMATCH")

        # Lean report checks in secondary_audit_payload (now raising ValueError!)
        n_thms = len(verifier.EXPECTED_LEAN_THEOREMS)
        lean_ok = {"static_clean": True, "zero_axioms_all": True, "theorems_verified": n_thms}
        s_ok = verifier.secondary_audit_payload(self.valid_env)
        self.assertTrue(s_ok["ok"])
        self.assertFalse(s_ok["lean_verified"])
        self.assertTrue(verifier.secondary_audit_payload(self.valid_env, lean_ok)["lean_verified"])
        with self.assertRaisesRegex(ValueError, "LEAN_REPORT_NOT_DICT"):
            verifier.secondary_audit_payload(self.valid_env, "bad")
        for bad_lean_rep in (
            {"static_clean": False, "zero_axioms_all": True, "theorems_verified": n_thms},
            {"static_clean": True, "zero_axioms_all": False, "theorems_verified": n_thms},
            {"static_clean": True, "zero_axioms_all": True, "theorems_verified": True},
            {"static_clean": True, "zero_axioms_all": True, "theorems_verified": n_thms - 1},
        ):
            with self.assertRaisesRegex(ValueError, "LEAN_REPORT_INCOMPLETE"):
                verifier.secondary_audit_payload(self.valid_env, bad_lean_rep)

    def test_10_pinned_sources_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            f1 = root / "sub" / "a.txt"
            f1.parent.mkdir()
            raw = b"pinned-receipt-bytes\n"
            f1.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            fake_pins = {"src1": {"relative_ref": "sub/a.txt", "sha256": sha, "bytes": len(raw)}}

            with mock.patch.object(verifier, "PINNED_SOURCES", fake_pins):
                res = verifier.verify_pinned_sources_on_disk(root)
                self.assertEqual(res["sources_checked"], 1)
                self.assertTrue(res["all_matched"])

                bad_len = {"src1": {"relative_ref": "sub/a.txt", "sha256": sha, "bytes": len(raw) + 1}}
                with mock.patch.object(verifier, "PINNED_SOURCES", bad_len):
                    with self.assertRaisesRegex(ValueError, "PINNED_SOURCE_BYTES_MISMATCH"):
                        verifier.verify_pinned_sources_on_disk(root)

                bad_sha = {"src1": {"relative_ref": "sub/a.txt", "sha256": "0" * 64, "bytes": len(raw)}}
                with mock.patch.object(verifier, "PINNED_SOURCES", bad_sha):
                    with self.assertRaisesRegex(ValueError, "PINNED_SOURCE_SHA_MISMATCH"):
                        verifier.verify_pinned_sources_on_disk(root)

                sym = root / "sub" / "sym.txt"
                sym.symlink_to(f1)
                sym_pin = {"src1": {"relative_ref": "sub/sym.txt", "sha256": sha, "bytes": len(raw)}}
                with mock.patch.object(verifier, "PINNED_SOURCES", sym_pin):
                    with self.assertRaisesRegex(ValueError, "SYMLINK_IN_PATH"):
                        verifier.verify_pinned_sources_on_disk(root)

                dir_sub = root / "sub" / "a_dir"
                dir_sub.mkdir()
                dir_pin = {"src1": {"relative_ref": "sub/a_dir", "sha256": sha, "bytes": len(raw)}}
                with mock.patch.object(verifier, "PINNED_SOURCES", dir_pin):
                    with self.assertRaisesRegex(ValueError, "NOT_REGULAR_FILE"):
                        verifier.verify_pinned_sources_on_disk(root)

                with self.assertRaisesRegex(ValueError, "EXPEDITION_ROOT_INVALID"):
                    verifier.verify_pinned_sources_on_disk(root / "no_such_dir")

        # CLI tests: --standalone --json, --write, --check, --json (default lean), --lean, --expedition-root
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            cert_path = d / "out_cert.json"

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = verifier.main(["--standalone", "--json"])
            self.assertEqual(rc, 0)
            j_sa = json.loads(out_buf.getvalue())
            self.assertTrue(j_sa["ok"])
            self.assertNotIn("lean", j_sa)

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = verifier.main(["--write", str(cert_path), "--json"])
            self.assertEqual(rc, 0)
            j_wr = json.loads(out_buf.getvalue())
            self.assertNotIn("lean", j_wr)
            self.assertTrue(cert_path.exists())
            self.assertEqual(stat.S_IMODE(os.lstat(cert_path).st_mode), 0o600)

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = verifier.main(["--check", str(cert_path), "--json"])
            self.assertEqual(rc, 0)
            j_ck = json.loads(out_buf.getvalue())
            self.assertNotIn("lean", j_ck)

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = verifier.main(["--check", str(cert_path)])
            self.assertEqual(rc, 0)
            self.assertIn("PASS", out_buf.getvalue())

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = verifier.main(["--json"])
            self.assertEqual(rc, 0)
            j_def = json.loads(out_buf.getvalue())
            self.assertIn("lean", j_def)
            self.assertTrue(j_def["lean"]["zero_axioms_all"])

            f1 = d / "a.txt"
            f1.write_bytes(b"ok")
            fake_pins = {"src1": {"relative_ref": "a.txt", "sha256": hashlib.sha256(b"ok").hexdigest(), "bytes": 2}}
            out_buf = io.StringIO()
            with mock.patch.object(verifier, "PINNED_SOURCES", fake_pins):
                with redirect_stdout(out_buf):
                    rc = verifier.main(
                        ["--standalone", "--lean", str(TRACK_DIR / "LoraCapacityJointMdl.lean"), "--expedition-root", str(d), "--json"]
                    )
            self.assertEqual(rc, 0)
            j = json.loads(out_buf.getvalue())
            self.assertTrue(j["lean"]["zero_axioms_all"])
            self.assertTrue(j["pinned_sources"]["all_matched"])

            bad_argvs = [
                ["--unknown"],
                ["--write"],
                ["--write", str(cert_path), "--write", str(cert_path)],
                ["--standalone", "--standalone"],
                ["--json", "--json"],
                ["--write", str(cert_path), "--check", str(cert_path)],
                ["--check", str(d / "missing.json"), "--json"],
                ["--check", str(d / "missing.json")],
            ]
            for argv in bad_argvs:
                ob, eb = io.StringIO(), io.StringIO()
                with redirect_stdout(ob), redirect_stderr(eb):
                    rc = verifier.main(argv)
                self.assertEqual(rc, 1, msg=f"Expected rc=1 for {argv}")


if __name__ == "__main__":
    unittest.main()
