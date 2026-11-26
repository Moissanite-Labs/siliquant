#!/usr/bin/env python3
"""Comprehensive R5 Verification & Hardening Test Suite for Track S79:
2DSD x MultipoleAttention x Dense-and-Sparse (SqueezeLLM / KVQuant / Portal 77 / S74 CSR)
x siliquant BlockH4 QuadSnake.

Zero bare `assert` statements, scratch `os.mkfifo` non-regular device probes, zero floats on
trust paths, zero mathematical RNG. Supports both `0o600` and `0o644` modes on committed
certificates while enforcing strict `0o600` on `--write` outputs.
"""
from __future__ import annotations

import copy
import errno
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import s79_2dsd_multipole_oracle as oracle  # noqa: E402
import verify_s79_2dsd_multipole as audit  # noqa: E402

_CERT_PATH = _THIS_DIR / "s79_2dsd_multipole_certificate.json"


class _FloatSubclass(float):
    pass


class _IntSubclass(int):
    pass


def _find_lean_bin() -> str | None:
    env_bin = os.environ.get("LEAN_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    found = shutil.which("lean")
    if found:
        return found
    home_local = Path.home() / ".local" / "bin" / "lean"
    if home_local.is_file():
        return str(home_local)
    return None


def _resign_cert(payload: dict, oracle_sha: str | None = None) -> dict:
    p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    o_sha = oracle_sha if oracle_sha is not None else oracle.compute_oracle_module_sha256()
    cert_pre = {
        "schema_version": oracle.SCHEMA_VERSION,
        "project_id": oracle.PROJECT_ID,
        "oracle_module_sha256": o_sha,
        "payload_sha256": p_sha,
        "payload": payload,
    }
    c_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(cert_pre))
    return {
        "schema_version": oracle.SCHEMA_VERSION,
        "project_id": oracle.PROJECT_ID,
        "oracle_module_sha256": o_sha,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
        "payload": payload,
    }


class TestOraclePrimitivesAndEngines(unittest.TestCase):
    def test_require_helpers_and_boundaries(self) -> None:
        oracle.require_ok(True, "OK", "ok")
        with self.assertRaises(oracle.S79Error) as ctx:
            oracle.require_ok(False, "FAIL_CODE", "fail msg")
        self.assertEqual(ctx.exception.code, "FAIL_CODE")

        self.assertEqual(oracle.require_exact_int(0, "x"), 0)
        self.assertEqual(oracle.require_exact_int(5, "x", min_val=5, max_val=5), 5)
        for bad in (True, False, "5", None, _IntSubclass(5)):
            with self.assertRaises(oracle.S79Error):
                oracle.require_exact_int(bad, "x")
        with self.assertRaises(oracle.S79Error) as c_lo:
            oracle.require_exact_int(4, "x", min_val=5)
        self.assertEqual(c_lo.exception.code, "INT_BELOW_MIN")
        with self.assertRaises(oracle.S79Error) as c_hi:
            oracle.require_exact_int(6, "x", max_val=5)
        self.assertEqual(c_hi.exception.code, "INT_ABOVE_MAX")
        with self.assertRaises(oracle.S79Error) as c_big:
            oracle.require_exact_int(1 << oracle.MAX_INT_BITS, "x")
        self.assertEqual(c_big.exception.code, "INT_TOO_LARGE")

        self.assertIs(oracle.require_exact_bool(True, "b"), True)
        self.assertIs(oracle.require_exact_bool(False, "b"), False)
        for bad in (1, 0, "true", None):
            with self.assertRaises(oracle.S79Error):
                oracle.require_exact_bool(bad, "b")

        self.assertEqual(oracle.require_exact_str("", "s", non_empty=False), "")
        self.assertEqual(oracle.require_exact_str("abc", "s"), "abc")
        with self.assertRaises(oracle.S79Error):
            oracle.require_exact_str(123, "s")
        with self.assertRaises(oracle.S79Error):
            oracle.require_exact_str("", "s", non_empty=True)
        with self.assertRaises(oracle.S79Error) as c_sur:
            oracle.require_exact_str("\ud800", "s")
        self.assertEqual(c_sur.exception.code, "INVALID_UTF8_STR")

        good_sha = "a" * 64
        self.assertEqual(oracle.require_sha256_hex(good_sha, "h"), good_sha)
        self.assertTrue(oracle.check_digest_eq(good_sha, good_sha))
        self.assertFalse(oracle.check_digest_eq(good_sha, "b" * 64))
        for bad_sha in ("a" * 63, "a" * 65, "A" * 64, "g" * 64):
            with self.assertRaises(oracle.S79Error):
                oracle.require_sha256_hex(bad_sha, "h")

        good_commit = "0" * 40
        self.assertTrue(oracle.check_commit_eq(good_commit, good_commit))
        self.assertFalse(oracle.check_commit_eq(good_commit, "f" * 40))
        with self.assertRaises(oracle.S79Error):
            oracle.check_commit_eq("0" * 39, good_commit)

    def test_ascii_int_and_rational_parsers(self) -> None:
        self.assertEqual(oracle.parse_exact_int_ascii("0", "z"), 0)
        self.assertEqual(oracle.parse_exact_int_ascii("-42", "n"), -42)
        self.assertEqual(oracle.parse_exact_int_ascii("10", "p"), 10)
        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        self.assertEqual(oracle.parse_exact_int_ascii(str(max_ok), "m"), max_ok)
        self.assertEqual(oracle.parse_exact_int_ascii("-" + str(max_ok), "mn"), -max_ok)

        bad_ints = [
            ("-", "INVALID_INT_TOKEN"),
            ("9" * 1001, "INT_TOKEN_TOO_LONG"),
            ("9" * 5001, "INT_TOKEN_TOO_LONG"),
            ("01", "LEADING_ZERO_INT"),
            ("-01", "LEADING_ZERO_INT"),
            ("-0", "NEGATIVE_ZERO_INT"),
            ("12\u00b2", "NON_ASCII_DIGIT"),
            ("+5", "NON_ASCII_DIGIT"),
            (str(1 << oracle.MAX_INT_BITS), "INT_TOO_LARGE"),
        ]
        for tok, code in bad_ints:
            with self.assertRaises(oracle.S79Error) as ctx:
                oracle.parse_exact_int_ascii(tok, "i")
            self.assertEqual(ctx.exception.code, code)

        self.assertEqual(oracle.fmt_q(Fraction(7, 1)), "7")
        self.assertEqual(oracle.fmt_q(Fraction(-5, 3)), "-5/3")
        with self.assertRaises(oracle.S79Error):
            oracle.fmt_q(3)  # type: ignore[arg-type]

        self.assertEqual(oracle.parse_exact_rational("5", "q"), Fraction(5, 1))
        self.assertEqual(oracle.parse_exact_rational("-3/2", "q"), Fraction(-3, 2))
        bad_qs = [
            ("1/2/3", "INVALID_RATIONAL"),
            ("1/0", "NON_CANONICAL_DEN"),
            ("3/1", "NON_CANONICAL_DEN"),
            ("1/-2", "NON_CANONICAL_DEN"),
            ("2/4", "UNREDUCED_RATIONAL"),
            ("0/2", "UNREDUCED_RATIONAL"),
        ]
        for tok, code in bad_qs:
            with self.assertRaises(oracle.S79Error) as ctx:
                oracle.parse_exact_rational(tok, "q")
            self.assertEqual(ctx.exception.code, code)

    def test_strict_json_parser_and_canonical_encoder(self) -> None:
        obj = {"b": [1, True, False, None, "ok"], "a": {"k": -7}}
        canon = oracle.canonical_json_bytes(obj)
        self.assertEqual(canon, b'{"a":{"k":-7},"b":[1,true,false,null,"ok"]}')
        self.assertEqual(oracle.parse_strict_json_text(canon.decode("ascii")), obj)

        # Escaped quotes, backslashes, and brackets inside string literals do not affect depth
        esc_text = r'{"s":"\"\\{[{{{{\"","ok":1}'
        self.assertEqual(oracle.parse_strict_json_text(esc_text)["ok"], 1)

        # Closing brace at depth 0 is handled without crashing pre-check (then fails JSON parse)
        with self.assertRaises(oracle.S79Error) as c_unbal:
            oracle.parse_strict_json_text("}{}")
        self.assertEqual(c_unbal.exception.code, "INVALID_JSON")

        # Exact max depth (32) passes; depth 33 fails in both tree check and text parser
        deep_ok: dict = {}
        cur = deep_ok
        for _ in range(oracle.MAX_JSON_DEPTH - 1):
            nxt: dict = {}
            cur["x"] = nxt
            cur = nxt
        cur["x"] = 1  # scalar leaf at exact depth == MAX_JSON_DEPTH (32)
        oracle.check_json_value_strict(deep_ok)
        self.assertEqual(oracle.parse_strict_json_text(oracle.canonical_json_bytes(deep_ok).decode("ascii")), deep_ok)
        cur["x"] = {"overflow": 1}
        with self.assertRaises(oracle.S79Error) as c_dep:
            oracle.check_json_value_strict(deep_ok)
        self.assertEqual(c_dep.exception.code, "MAX_DEPTH_EXCEEDED")
        with self.assertRaises(oracle.S79Error) as c_dep2:
            oracle.parse_strict_json_text('{"a":' * 33 + "1" + "}" * 33)
        self.assertEqual(c_dep2.exception.code, "MAX_DEPTH_EXCEEDED")
        # Direct pre-parse depth guard test (would fail before json.loads even runs)
        oracle._check_json_nesting_depth("{" * 32 + "}" * 32)
        with self.assertRaises(oracle.S79Error) as c_dep3:
            oracle._check_json_nesting_depth("{" * 33)
        self.assertEqual(c_dep3.exception.code, "MAX_DEPTH_EXCEEDED")

        # Forbidden types, floats, constants, duplicates, CRLF, non-object root, exact & over-cap byte limits
        for bad_val in (1.5, _FloatSubclass(1.5), (1, 2), {1}, {1: "a"}, {"k": "\ud800"}):
            with self.assertRaises(oracle.S79Error):
                oracle.check_json_value_strict(bad_val)
        with self.assertRaises(oracle.S79Error):
            oracle.sha256_bytes("not-bytes")  # type: ignore[arg-type]
        exact_obj = {"key": "exact_size"}
        exact_raw = oracle.canonical_json_bytes(exact_obj)
        with mock.patch.object(oracle, "MAX_JSON_BYTES", len(exact_raw)):
            self.assertEqual(oracle.canonical_json_bytes(exact_obj), exact_raw)
            self.assertEqual(oracle.parse_strict_json_text(exact_raw.decode("ascii")), exact_obj)
        with mock.patch.object(oracle, "MAX_JSON_BYTES", len(exact_raw) - 1):
            with self.assertRaises(oracle.S79Error) as c_cbig:
                oracle.canonical_json_bytes(exact_obj)
            self.assertEqual(c_cbig.exception.code, "JSON_TOO_LARGE")
            with self.assertRaises(oracle.S79Error) as c_pbig:
                oracle.parse_strict_json_text(exact_raw.decode("ascii"))
            self.assertEqual(c_pbig.exception.code, "JSON_TOO_LARGE")

        for bad_json, code in (
            ('{"a":1}\r\n', "CRLF_FORBIDDEN"),
            ('{"a":1.5}', "JSON_FLOAT_FORBIDDEN"),
            ('{"a":NaN}', "JSON_CONSTANT_FORBIDDEN"),
            ('{"a":Infinity}', "JSON_CONSTANT_FORBIDDEN"),
            ('{"a":1,"a":2}', "DUPLICATE_JSON_KEY"),
            ("[1,2,3]", "JSON_ROOT_NOT_OBJECT"),
            ('{"a":' + "9" * 5000 + "}", "INT_TOKEN_TOO_LONG"),
            ("{bad", "INVALID_JSON"),
        ):
            with self.assertRaises(oracle.S79Error) as ctx:
                oracle.parse_strict_json_text(bad_json)
            self.assertEqual(ctx.exception.code, code)

    def test_pillar1_cascade_and_kv_math(self) -> None:
        p0 = [60, 30, 10]
        q1 = [40, 40, 20]
        q2 = [10, 50, 40]
        q3 = [70, 20, 10]
        res = oracle.cascade_verify_exact(p0, [q1, q2, q3])
        self.assertEqual(res["total_mass"], 100)
        self.assertEqual(res["stage_cumulative_accepted"], [80, 90, 100])
        self.assertEqual(res["final_residual_mass"], 0)
        self.assertEqual(res["reconstructed_target"], p0)

        # Order invariance of marginal distribution across permutations
        c12 = oracle.cascade_verify_exact(p0, [q1, q2])
        c21 = oracle.cascade_verify_exact(p0, [q2, q1])
        self.assertEqual(c12["reconstructed_target"], p0)
        self.assertEqual(c21["reconstructed_target"], p0)

        # Negative controls for cascade_verify_exact
        for bad_t, bad_ps, code in (
            ("bad", [], "CASCADE_EMPTY_TARGET"),
            ([], [], "CASCADE_EMPTY_TARGET"),
            ([10], "bad", "CASCADE_INVALID_PROPOSALS"),
            ([0, 0], [], "CASCADE_ZERO_MASS"),
            ([10, 10], ["ab"], "CASCADE_PROPOSAL_LEN"),
            ([10, 10], [[10]], "CASCADE_PROPOSAL_LEN"),
            ([10, 10], [[15, 10]], "CASCADE_PROPOSAL_SUM"),
        ):
            with self.assertRaises(oracle.S79Error) as ctx:
                oracle.cascade_verify_exact(bad_t, bad_ps)  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, code)

        # Eq. 6 KV sharing compression ratios & strict monotonicity in L
        r2k = oracle.kv_sharing_compression_ratio(4, 64, 2048)
        r8k = oracle.kv_sharing_compression_ratio(4, 64, 8192)
        r32k = oracle.kv_sharing_compression_ratio(4, 64, 32768)
        self.assertEqual(r2k, Fraction(128, 35))
        self.assertEqual(r8k, Fraction(512, 131))
        self.assertEqual(r32k, Fraction(2048, 515))
        self.assertTrue(r2k < r8k < r32k < Fraction(4, 1))
        with self.assertRaises(oracle.S79Error):
            oracle.kv_sharing_compression_ratio(1, 64, 2048)
        with self.assertRaises(oracle.S79Error):
            oracle.kv_sharing_compression_ratio(4, 0, 2048)
        with self.assertRaises(oracle.S79Error):
            oracle.kv_sharing_compression_ratio(4, 64, 64)

    def test_pillar2_dense_sparse_crossover_and_reversal(self) -> None:
        sq = oracle.dense_sparse_crossover_exact(
            Fraction(4, 1), Fraction(45, 10000), 32, Fraction(5, 4), Fraction(1, 1), 1000, 400
        )
        self.assertEqual(sq["delta_b"], Fraction(18, 125))
        self.assertEqual(sq["b_eff"], Fraction(518, 125))
        self.assertEqual(sq["distortion_ratio_csr_over_dense"], Fraction(25, 28))
        self.assertTrue(sq["csr_strictly_better"])

        rev = oracle.dense_sparse_crossover_exact(
            Fraction(4, 1), Fraction(1, 32), 32, Fraction(4, 1), Fraction(1, 1), 1000, 2000
        )
        self.assertEqual(rev["distortion_ratio_csr_over_dense"], Fraction(4, 3))
        self.assertFalse(rev["csr_strictly_better"])

        # Boundary s_out = 0 allowed
        z0 = oracle.dense_sparse_crossover_exact(
            Fraction(4, 1), Fraction(1, 100), 32, Fraction(1, 1), Fraction(1, 1), 100, 0
        )
        self.assertEqual(z0["distortion_ratio_csr_over_dense"], Fraction(1, 1))

        # Domain error checks (every subcondition tested!)
        bad_cases = [
            (4, Fraction(1, 100), 32, Fraction(1, 1), Fraction(1, 1), 100, 10),
            (Fraction(0, 1), Fraction(1, 100), 32, Fraction(1, 1), Fraction(1, 1), 100, 10),
            (Fraction(4, 1), 1, 32, Fraction(1, 1), Fraction(1, 1), 100, 10),
            (Fraction(4, 1), Fraction(0, 1), 32, Fraction(1, 1), Fraction(1, 1), 100, 10),
            (Fraction(4, 1), Fraction(1, 100), 0, Fraction(1, 1), Fraction(1, 1), 100, 10),
            (Fraction(4, 1), Fraction(1, 100), 32, 1, Fraction(1, 1), 100, 10),
            (Fraction(4, 1), Fraction(1, 100), 32, Fraction(1, 2), Fraction(1, 1), 100, 10),
            (Fraction(4, 1), Fraction(1, 100), 32, Fraction(1, 1), 1, 100, 10),
            (Fraction(4, 1), Fraction(1, 100), 32, Fraction(1, 1), Fraction(1, 2), 100, 10),
            (Fraction(4, 1), Fraction(1, 100), 32, Fraction(1, 1), Fraction(1, 1), 0, 10),
            (Fraction(4, 1), Fraction(1, 100), 32, Fraction(1, 1), Fraction(1, 1), 100, -1),
        ]
        for args in bad_cases:
            with self.assertRaises(oracle.S79Error):
                oracle.dense_sparse_crossover_exact(*args)  # type: ignore[arg-type]

    def test_pillar3_multipole_exact_engine(self) -> None:
        mp = oracle.multipole_two_key_exact([3, 4, 0, 0], [4, 2, 1, 0], [2, 2, -1, 0], 3, 1, 5, 1, 128)
        self.assertEqual(mp["q_norm_sq"], 25)
        self.assertEqual(mp["k_diff_norm_sq"], 8)
        self.assertEqual(mp["cs_lhs"], 36)
        self.assertEqual(mp["cs_rhs"], 200)
        self.assertEqual(mp["centroid_linear_residual"], 0)
        self.assertEqual(mp["actual_logit_spread_sq"], Fraction(9, 128))
        self.assertEqual(mp["cs_logit_bound_sq"], Fraction(25, 64))
        self.assertEqual(mp["value_weight_cov_residual"], 8)

        # Negative controls for every vector-type / length / domain branch (asserting exact codes!)
        bad_mp = [
            (("a", [1], [1], 1, 1, 1, 1, 1), "MULTIPOLE_VEC_TYPE"),
            (([1], "a", [1], 1, 1, 1, 1, 1), "MULTIPOLE_VEC_TYPE"),
            (([1], [1], "a", 1, 1, 1, 1, 1), "MULTIPOLE_VEC_TYPE"),
            (([], [], [], 1, 1, 1, 1, 1), "MULTIPOLE_DIM"),
            (([1, 2], [1], [1, 2], 1, 1, 1, 1, 1), "MULTIPOLE_DIM"),
            (([1, 2], [1, 2], [1], 1, 1, 1, 1, 1), "MULTIPOLE_DIM"),
            (([1], [1], [1], -1, 1, 1, 1, 1), "INT_BELOW_MIN"),
            (([1], [1], [1], 1, -1, 1, 1, 1), "INT_BELOW_MIN"),
            (([1], [1], [1], 1, 1, 1, 1, 0), "INT_BELOW_MIN"),
        ]
        for args, code in bad_mp:
            with self.assertRaises(oracle.S79Error) as ctx:
                oracle.multipole_two_key_exact(*args)  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, code)

    def test_build_certificate_budget_guard(self) -> None:
        cert = oracle.build_certificate()
        cert_len = len(oracle.canonical_json_bytes(cert))
        self.assertLessEqual(cert_len, oracle.TARGET_MAX_CERT_BYTES)
        # Measure exact serialized length when target_max_cert_bytes is 4 digits (same digit width as cert_len)
        with mock.patch.object(oracle, "TARGET_MAX_CERT_BYTES", cert_len):
            exact_len = len(oracle.canonical_json_bytes(oracle.build_certificate()))
        with mock.patch.object(oracle, "TARGET_MAX_CERT_BYTES", exact_len):
            c_eq = oracle.build_certificate()
            self.assertEqual(len(oracle.canonical_json_bytes(c_eq)), exact_len)
        with mock.patch.object(oracle, "TARGET_MAX_CERT_BYTES", exact_len - 1):
            with self.assertRaises(oracle.S79Error) as ctx:
                oracle.build_certificate()
            self.assertEqual(ctx.exception.code, "CERT_BUDGET_EXCEEDED")


class TestLean4StaticAndLiveVerifier(unittest.TestCase):
    def test_r9_r10_stripper_and_scanner_exploits(self) -> None:
        # Clean code with comments, raw strings, char literals, primes, unary !r, safe decide
        clean_src = "\n".join([
            "-- comment with sorry and axiom",
            "/- nested /- sorry -/ block -/",
            'def s1 := "sorry admit unsafe"',
            'def s2 := r#"sorry native_decide"#',
            'def s3 := s!"hello \\" {1 + 1} world"',
            'def s4 := !r"sorry"',
            'def s5 := foo!r',
            "def c1 := 'a'",
            "def c2 := '\\n'",
            "def c3 := '\\x41'",
            "def c4 := '\\u0041'",
            "def x' := 1",
            "def y' := x' + 'b'.toNat",
            "def badc := '\\xZZ'",
            "def badu := '\\uZZZZ'",
            "def notchar := 'ab",
            "def g1 := «a b sorry»",
            "def g2 := «axioms»",
            "def ok1 : 1 = 1 := by decide -native (native := false) (config := ({native := false}))",
            "def ok2 : 1 = 1 := by decide (native := ((Bool.false))) (config := {native := false, foo := 1}) (bar := 2)",
            "def ok3 : 1 = 1 := by decide +other",
            "def ok4 : 1 = 1 := by decide",
            "def ok5 : Nat := decide +",
            "def ok6 : Nat := decide (unclosed",
            "def ok7 : Nat := decide (no_colon)",
            "def ok8 : DecideConfig := {native := false}",
            "def ok9 := @[]",
            "def ok10 := @[simp]",
            "def ok11 := @[unclosed",
            "def ok12 := !!",
            "def ok13 := foo!!",
            "def ok14 := a!b",
            'def s6 := f! "a\\nb"',
            'def s7 := "a\\\nb"',
            'def s8 := s!"x { {1} } y"',
            'def s9 := s!"x { "inner" } y"',
            'def s10 := r#"a\nb"#',
            'def s11 := r#"unquoted"#',
            "def s12 := r_not_raw",
            "def s13 := m! \"a {s!\"b\"} c\"",
            "#print axioms TwoDSDAndDenseSparseLaws.twodsd_ouro_dory_kv_and_speedup_witness",
        ])
        res = audit.audit_lean4_source(clean_src)
        self.assertTrue(res["clean"], f"unexpected findings: {res}")
        self.assertEqual(res["notices"], [])

        # Hard exploit rules (all 11 classes + metaprogramming + decide native + eval spoof + attr + unterminated)
        exploits = [
            ("theorem t : True := by sorry", "LEAN4_SORRY"),
            ("theorem t : True := by admit", "LEAN4_ADMIT"),
            ("axiom bogus : False", "LEAN4_AXIOM"),
            ("unsafe def f : Nat := 0", "LEAN4_UNSAFE"),
            ("theorem t : 1 = 1 := by native_decide", "LEAN4_NATIVE_DECIDE"),
            ("def f := sorryAx", "LEAN4_SORRY_AX"),
            ("def f := Lean.Meta.mkSorry", "LEAN4_MK_SORRY"),
            ("def f := Lean.ofReduceBool", "LEAN4_OF_REDUCE_BOOL"),
            ("def f := Lean.ofReduceNat", "LEAN4_OF_REDUCE_NAT"),
            ("def f := Lean.trustCompiler", "LEAN4_TRUST_COMPILER"),
            ('syntax "foo!" interpolatedStr : term', "LEAN4_METAPROGRAMMING"),
            ("macro \"m\" : term => `(0)", "LEAN4_METAPROGRAMMING"),
            ("macro_rules | `(x) => `(0)", "LEAN4_METAPROGRAMMING"),
            ("elab \"e\" : term => pure default", "LEAN4_METAPROGRAMMING"),
            ("elab_rules : term | `(x) => pure default", "LEAN4_METAPROGRAMMING"),
            ("declare_syntax_cat mycat", "LEAN4_METAPROGRAMMING"),
            ("import Mathlib", "LEAN4_FORBIDDEN_IMPORT"),
            ("partial def loop : Nat := loop", "LEAN4_PARTIAL"),
            ("set_option debug.skipKernelTC true", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option debug.byAsSorry true", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option debug.proofAsSorry true", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option debug.terminalTacticsAsSorry true", "LEAN4_SKIP_KERNEL_TC"),
            ("#eval 1 + 1", "LEAN4_EVAL_SPOOF"),
            ("#reduce 1 + 1", "LEAN4_EVAL_SPOOF"),
            ("#exit", "LEAN4_EVAL_SPOOF"),
            ("#print Nat", "LEAN4_EVAL_SPOOF"),
            ("#print axioms example", "LEAN4_EVAL_SPOOF"),
            ("run_cmd pure ()", "LEAN4_EVAL_SPOOF"),
            ("initialize x : Nat := pure 0", "LEAN4_EVAL_SPOOF"),
            ("builtin_initialize y : Nat := pure 0", "LEAN4_EVAL_SPOOF"),
            ("@[extern \"c_fn\"] def f : Nat := 0", "LEAN4_ATTR_TRUST_ESCAPE"),
            ("attribute [implemented_by f] g", "LEAN4_ATTR_TRUST_ESCAPE"),
            ("theorem t : 1 = 1 := by decide +native", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (native := true)", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := {native := true})", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := notRecord)", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := {{native := false}})", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := {c with native := false})", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := {native := false, extra})", "LEAN4_DECIDE_NATIVE"),
            ("def c : DecideConfig := {native := true}", "LEAN4_DECIDE_NATIVE"),
            ("def c := {native := true} : Decide.Config", "LEAN4_DECIDE_NATIVE"),
            ("def c := {native := true : DecideConfig}", "LEAN4_DECIDE_NATIVE"),
            ('def s := s!"{sorry}"', "LEAN4_SORRY"),
            ("def «sorry» := 1", "LEAN4_SORRY"),
            ("#print «axioms» Foo", "LEAN4_EVAL_SPOOF"),
            ("/- unclosed comment", "LEAN4_UNTERMINATED_COMMENT"),
            ('def s := "unclosed', "LEAN4_UNTERMINATED_STRING"),
            ('def s := s!"unclosed', "LEAN4_UNTERMINATED_STRING"),
            ('def s := r#"unclosed', "LEAN4_UNTERMINATED_STRING"),
            ('def s := s!"unclosed {1', "LEAN4_UNTERMINATED_STRING"),
            ("def g := «unclosed", "LEAN4_UNTERMINATED_IDENT"),
        ]
        for snippet, expected_code in exploits:
            r = audit.audit_lean4_source(snippet)
            self.assertFalse(r["clean"], f"expected failure for {snippet!r}")
            self.assertIn(expected_code, r["codes"], f"missing {expected_code} in {r} for {snippet!r}")

        # Notices (vacuous signature, exists implication, csimp)
        n_res = audit.audit_lean4_source(
            "theorem vac : True := trivial\n"
            "theorem ex (P Q : Nat → Prop) : ∃ x, P x → Q x := by\n"
            "@[csimp] theorem cs : 1 = 1 := rfl\n"
        )
        notice_codes = {x["code"] for x in n_res["notices"]}
        self.assertIn("LEAN4_VACUOUS_SIGNATURE_NOTICE", notice_codes)
        self.assertIn("LEAN4_EXISTS_IMPLICATION_NOTICE", notice_codes)
        self.assertIn("LEAN4_CSIMP_REWRITE_NOTICE", notice_codes)

        # Direct scanner helper unit probes killing all BoolOp & boundary mutants
        self.assertFalse(audit._is_lean4_ident_prev("abc", -1))
        self.assertTrue(audit._is_lean4_ident_prev("a", 0))
        self.assertTrue(audit._is_lean4_ident_prev("_", 0))
        self.assertFalse(audit._is_lean4_ident_prev(" ", 0))
        self.assertTrue(audit._is_lean4_ident_prev("a!", 1))
        self.assertTrue(audit._is_lean4_ident_prev("_!", 1))
        self.assertFalse(audit._is_lean4_ident_prev("!!", 1))
        self.assertFalse(audit._is_lean4_ident_prev(" !", 1))

        self.assertTrue(audit._is_lean4_native_rhs_false_span("(false)", 0, 7))
        self.assertFalse(audit._is_lean4_native_rhs_false_span("(falsex", 0, 7))
        self.assertFalse(audit._is_lean4_native_rhs_false_span("xfalse)", 0, 7))

        self.assertTrue(audit._is_lean4_safe_config_span("{native := false}", 0, 17, {}))
        self.assertFalse(audit._is_lean4_safe_config_span("{native := falsex", 0, 17, {}))
        self.assertFalse(audit._is_lean4_safe_config_span("xnative := false}", 0, 17, {}))
        self.assertFalse(audit._is_lean4_safe_config_span("{a{b}", 0, 5, {}))
        self.assertFalse(audit._is_lean4_safe_config_span("{a}b}", 0, 5, {}))

        self.assertEqual(len(audit._scan_lean4_decide_native(") decide +native")), 1)
        self.assertEqual(len(audit._iter_lean4_attr_blocks("] @[extern]")), 1)

        # Direct _match_lean4_char_literal_end probes (all 4 escape shapes + bounds)
        char_cases = [
            ("'a'", 3),
            ("'a", None),
            ("'\\'", None),
            ("'\n'", None),
            ("'ab", None),
            ("'ab'", None),
            ("'\\n'", 4),
            ("'\\n", None),
            ("'\\\n'", None),
            ("'\\ab'", None),
            ("'\\x41'", 6),
            ("'\\x41", None),
            ("'\\x41a", None),
            ("'\\xZZ'", None),
            ("'\\y41'", None),
            ("'\\u0041'", 8),
            ("'\\u0041", None),
            ("'\\u0041a", None),
            ("'\\uZZZZ'", None),
            ("'\\v0041'", None),
        ]
        for s_ch, exp_end in char_cases:
            self.assertEqual(audit._match_lean4_char_literal_end(s_ch, 0, len(s_ch)), exp_end, f"char={s_ch!r}")

        # Direct strip_lean4_with_report edge probes (comments, raw strings, interp, char literals)
        self.assertIn("sorry", audit.strip_lean4_with_report("/- / -/ sorry")[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report("/- - sorry -/")[0])
        self.assertIn("sorry", audit.strip_lean4_with_report("1 / 2 sorry")[0])
        self.assertIn("sorry", audit.strip_lean4_with_report("1 - 2 sorry")[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report("-- sorry")[0])
        self.assertIn("sorry", audit.strip_lean4_with_report('ar#"a" sorry "#')[0])
        self.assertTrue(audit.audit_lean4_source('def s := r#"a" sorry "#')["clean"])
        self.assertEqual(audit.strip_lean4_with_report("r#")[1], [])
        self.assertEqual(audit.strip_lean4_with_report("r")[1], [])
        self.assertNotIn("sorry", audit.strip_lean4_with_report('!"{sorry}"s')[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report('x!"{sorry}"')[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report('as!"{sorry}"')[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report(' s "{sorry}"')[0])
        self.assertIn("ab", audit.strip_lean4_with_report("ab'")[0])
        self.assertIn("sorry", audit.strip_lean4_with_report("'\"' sorry")[0])
        self.assertIn("sorry", audit.strip_lean4_with_report(" '\"' sorry")[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report("a'\"'sorry\"")[0])
        self.assertNotIn("sorry", audit.strip_lean4_with_report("_'\"'sorry\"")[0])

    def test_live_lean_verifier_and_failure_modes(self) -> None:
        lean_bin = _find_lean_bin()
        self.assertIsNotNone(lean_bin, "Lean 4.36.0 binary must be available for R5 live verification")

        # Live verification of the real Track S79 Lean module
        res_live = audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
        self.assertTrue(res_live["valid"])
        self.assertEqual(res_live["kernel_theorems_verified"], 18)
        self.assertEqual(res_live["banned_axioms_found"], 0)
        self.assertTrue(res_live["live_lean_checked"])

        # Static-only mode
        res_static = audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=False)
        self.assertTrue(res_static["valid"])
        self.assertFalse(res_static["live_lean_checked"])

        # Auto-discovery via LEAN_BIN and shutil.which
        fake_ok_stdout = "\n".join(
            f"'TwoDSDAndDenseSparseLaws.{tn}' "
            + ("does not depend on any axioms" if not axs else f"depends on axioms: [{', '.join(axs)}]")
            for tn, axs in oracle.LEAN_MODULE_SPECS[0]["theorems"]
        ) + "\n"
        fake_ok = subprocess.CompletedProcess(["lean"], 0, stdout=fake_ok_stdout, stderr="")
        with mock.patch.dict(os.environ, {"LEAN_BIN": str(lean_bin)}), mock.patch.object(
            subprocess, "run", return_value=fake_ok
        ):
            res_env = audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=True)
            self.assertTrue(res_env["valid"])
        with mock.patch.dict(os.environ, {"LEAN_BIN": ""}, clear=False), mock.patch.object(
            shutil, "which", return_value=None
        ):
            with self.assertRaises(oracle.S79Error) as ctx_nf:
                audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=True)
            self.assertEqual(ctx_nf.exception.code, "LEAN_NOT_FOUND")

        # Tampered project files in scratch dir
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for fname in ("lean-toolchain", "lakefile.toml", "TwoDSDAndDenseSparseLaws.lean"):
                shutil.copy2(_THIS_DIR / fname, root / fname)

            # Tampered lean-toolchain
            (root / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n", encoding="utf-8")
            with self.assertRaises(oracle.S79Error) as ctx:
                audit.verify_lean_modules_live(root, verify_live_lean=False)
            self.assertEqual(ctx.exception.code, "LEAN_TOOLCHAIN_SHA_MISMATCH")
            shutil.copy2(_THIS_DIR / "lean-toolchain", root / "lean-toolchain")

            # Tampered lakefile.toml
            (root / "lakefile.toml").write_text("name = 'tampered'\n", encoding="utf-8")
            with self.assertRaises(oracle.S79Error) as ctx:
                audit.verify_lean_modules_live(root, verify_live_lean=False)
            self.assertEqual(ctx.exception.code, "LAKEFILE_SHA_MISMATCH")
            shutil.copy2(_THIS_DIR / "lakefile.toml", root / "lakefile.toml")

            # Tampered .lean SHA
            (root / "TwoDSDAndDenseSparseLaws.lean").write_text("-- tampered\n", encoding="utf-8")
            with self.assertRaises(oracle.S79Error) as ctx:
                audit.verify_lean_modules_live(root, verify_live_lean=False)
            self.assertEqual(ctx.exception.code, "LEAN_SHA256_MISMATCH")

            # Invalid UTF-8 .lean (with patched spec SHA)
            bad_utf8 = b"\xff\xfe\xfd"
            (root / "TwoDSDAndDenseSparseLaws.lean").write_bytes(bad_utf8)
            patched_spec = copy.deepcopy(list(oracle.LEAN_MODULE_SPECS))
            patched_spec[0]["sha256"] = oracle.sha256_bytes(bad_utf8)
            with mock.patch.object(oracle, "LEAN_MODULE_SPECS", tuple(patched_spec)):
                with self.assertRaises(oracle.S79Error) as ctx:
                    audit.verify_lean_modules_live(root, verify_live_lean=False)
                self.assertEqual(ctx.exception.code, "LEAN_INVALID_UTF8")

            # Static audit fail (finding-only with 0 notices) and notice-only fail (with patched spec SHA)
            for bad_src in (b"axiom bad_ax : Nat\n", b"theorem t : True := trivial\n"):
                (root / "TwoDSDAndDenseSparseLaws.lean").write_bytes(bad_src)
                patched_spec[0]["sha256"] = oracle.sha256_bytes(bad_src)
                with mock.patch.object(oracle, "LEAN_MODULE_SPECS", tuple(patched_spec)):
                    with self.assertRaises(oracle.S79Error) as ctx:
                        audit.verify_lean_modules_live(root, verify_live_lean=False)
                    self.assertEqual(ctx.exception.code, "LEAN_STATIC_AUDIT_FAILED")

        # Subprocess failure modes (OSError, rc!=0, nonempty stderr)
        with mock.patch.object(subprocess, "run", side_effect=OSError("no exec")):
            with self.assertRaises(oracle.S79Error) as ctx:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(ctx.exception.code, "LEAN_EXEC_FAILED")

        fake_fail = subprocess.CompletedProcess(["lean"], 1, stdout="", stderr="")
        with mock.patch.object(subprocess, "run", return_value=fake_fail):
            with self.assertRaises(oracle.S79Error) as ctx:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(ctx.exception.code, "LEAN_KERNEL_FAILED")

        fake_err = subprocess.CompletedProcess(["lean"], 0, stdout="", stderr="warning: unused")
        with mock.patch.object(subprocess, "run", return_value=fake_err):
            with self.assertRaises(oracle.S79Error) as ctx:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(ctx.exception.code, "LEAN_STDERR_NONEMPTY")

        # _verify_lean_stdout_axioms direct edge cases
        thms = (("t1", ()), ("t2", ("propext",)))
        good_out = (
            "'NS.t1' does not depend on any axioms\n"
            "'NS.t2' depends on axioms: [propext]\n"
        )
        audit._verify_lean_stdout_axioms(good_out, "NS", thms)

        bad_outs = [
            (good_out + "'NS.t1' does not depend on any axioms\n", "LEAN_DUPLICATE_AXIOM_LINE"),
            ("'NS.t1' depends on axioms: [sorryAx]\n'NS.t2' depends on axioms: [propext]\n", "LEAN_BANNED_AXIOM"),
            ("'NS.t1' depends on axioms: [Classical.choice]\n'NS.t2' depends on axioms: [propext]\n", "LEAN_BANNED_AXIOM"),
            (good_out + "warning: stray line\n", "LEAN_UNEXPECTED_STDOUT"),
            ("'NS.t1' does not depend on any axioms\n", "LEAN_THEOREM_SET_MISMATCH"),
            ("'NS.t1' depends on axioms: [propext]\n'NS.t2' depends on axioms: [propext]\n", "LEAN_AXIOM_MISMATCH"),
        ]
        for out_txt, code in bad_outs:
            with self.assertRaises(oracle.S79Error) as ctx:
                audit._verify_lean_stdout_axioms(out_txt, "NS", thms)
            self.assertEqual(ctx.exception.code, code)


class TestHardenedFilesystemAndDarwin(unittest.TestCase):
    def test_path_coercion_and_darwin_normalization(self) -> None:
        self.assertEqual(audit._coerce_exact_path("abc", "p"), Path("abc"))
        self.assertEqual(audit._coerce_exact_path(Path("abc"), "p"), Path("abc"))
        for bad, code in (
            ("a\x00b", "INVALID_PATH"),
            ("dir/", "INVALID_FILENAME"),
            (Path("a\x00b"), "INVALID_PATH"),
            (123, "INVALID_PATH_TYPE"),
        ):
            with self.assertRaises(oracle.S79Error) as ctx:
                audit._coerce_exact_path(bad, "p")
            self.assertEqual(ctx.exception.code, code)

        # Darwin branches: every subcondition tested with mocked valid root-owned symlink underneath
        class _FakeStat:
            def __init__(self, mode: int, uid: int) -> None:
                self.st_mode = mode
                self.st_uid = uid

        with mock.patch.object(os, "lstat", return_value=_FakeStat(stat.S_IFLNK, 0)), mock.patch.object(
            os, "readlink", return_value="private/tmp"
        ):
            with mock.patch.object(sys, "platform", "linux"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a")), Path("/tmp/a"))
            with mock.patch.object(sys, "platform", "darwin"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("rel/a")), Path("rel/a"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("x/tmp/a")), Path("x/tmp/a"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp"), allow_exact=False), Path("/tmp"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/../a")), Path("/tmp/../a"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/usr/a")), Path("/usr/a"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a/b")), Path("/private/tmp/a/b"))
                self.assertEqual(
                    audit._normalize_darwin_system_prefix(Path("/tmp"), allow_exact=True), Path("/private/tmp")
                )

        with mock.patch.object(sys, "platform", "darwin"):
            with mock.patch.object(os, "lstat", return_value=_FakeStat(stat.S_IFDIR, 0)), mock.patch.object(
                os, "readlink", return_value="private/tmp"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a")), Path("/tmp/a"))
            with mock.patch.object(os, "lstat", return_value=_FakeStat(stat.S_IFLNK, 501)), mock.patch.object(
                os, "readlink", return_value="private/tmp"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a")), Path("/tmp/a"))
            with mock.patch.object(os, "lstat", return_value=_FakeStat(stat.S_IFLNK, 0)), mock.patch.object(
                os, "readlink", return_value="/evil"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a")), Path("/tmp/a"))
            with mock.patch.object(os, "lstat", side_effect=OSError("nope")):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a")), Path("/tmp/a"))

    def test_dir_fd_traversal_symlinks_fifo_hardlinks_and_enospc(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real_dir = root / "real"
            real_dir.mkdir()
            reg = real_dir / "file.txt"
            reg.write_bytes(b"hello")

            # Read regular file + exact cap boundary + over cap
            self.assertEqual(audit.read_regular_bytes_capped(reg, max_bytes=5), b"hello")
            with self.assertRaises(oracle.S79Error) as c_big:
                audit.read_regular_bytes_capped(reg, max_bytes=4)
            self.assertEqual(c_big.exception.code, "FILE_TOO_LARGE")

            # Symlink target & symlink ancestor rejected on both read & write
            sym_file = real_dir / "sym_file"
            os.symlink(reg, sym_file)
            sym_dir = root / "sym_dir"
            os.symlink(real_dir, sym_dir)

            for bad_p in (sym_file, sym_dir / "file.txt"):
                with self.assertRaises(oracle.S79Error) as ctx:
                    audit.read_regular_bytes_capped(bad_p)
                self.assertEqual(ctx.exception.code, "SYMLINK_FORBIDDEN")
                with self.assertRaises(oracle.S79Error) as ctx2:
                    audit.write_bytes_atomic_0600(bad_p, b"x")
                self.assertEqual(ctx2.exception.code, "SYMLINK_FORBIDDEN")

            # Hardlink target rejected BEFORE creating temp file
            hl = real_dir / "hardlink.txt"
            os.link(reg, hl)
            with self.assertRaises(oracle.S79Error) as c_hlr:
                audit.read_regular_bytes_capped(hl)
            self.assertEqual(c_hlr.exception.code, "HARDLINK_FORBIDDEN")
            with mock.patch.object(audit, "_next_temp_name", side_effect=RuntimeError("must reject before temp")):
                with self.assertRaises(oracle.S79Error) as c_hlw:
                    audit.write_bytes_atomic_0600(hl, b"x")
                self.assertEqual(c_hlw.exception.code, "HARDLINK_FORBIDDEN")
            hl.unlink()

            # FIFO (non-regular device) rejected BEFORE creating temp file without hanging
            fifo = real_dir / "pipe.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(oracle.S79Error) as c_fr:
                audit.read_regular_bytes_capped(fifo)
            self.assertEqual(c_fr.exception.code, "NON_REGULAR_FILE")
            with mock.patch.object(audit, "_next_temp_name", side_effect=RuntimeError("must reject before temp")):
                with self.assertRaises(oracle.S79Error) as c_fw:
                    audit.write_bytes_atomic_0600(fifo, b"x")
                self.assertEqual(c_fw.exception.code, "NON_REGULAR_FILE")

            # Parent traversal '..' & non-directory ancestor & invalid leaf names
            with self.assertRaises(oracle.S79Error) as c_dotdot:
                audit.read_regular_bytes_capped(str(real_dir) + "/../real/file.txt")
            self.assertEqual(c_dotdot.exception.code, "PARENT_TRAVERSAL_FORBIDDEN")
            with self.assertRaises(oracle.S79Error) as c_nd:
                audit.read_regular_bytes_capped(reg / "child")
            self.assertEqual(c_nd.exception.code, "NOT_A_DIRECTORY")
            with self.assertRaises(oracle.S79Error) as c_inv:
                audit.read_regular_bytes_capped(".")
            self.assertEqual(c_inv.exception.code, "INVALID_FILENAME")
            with self.assertRaises(oracle.S79Error) as c_miss:
                audit.read_regular_bytes_capped(root / "missing_dir" / "x")
            self.assertEqual(c_miss.exception.code, "DIR_OPEN_FAILED")
            with self.assertRaises(oracle.S79Error) as c_unr:
                audit.read_regular_bytes_capped(real_dir / "nope.txt")
            self.assertEqual(c_unr.exception.code, "FILE_UNREADABLE")

            # Relative path read & $PWD symlink / deleted-CWD / inode mismatch checks
            old_cwd = os.getcwd()
            try:
                os.chdir(real_dir)
                with mock.patch.dict(os.environ, {"PWD": str(real_dir)}):
                    self.assertEqual(audit.read_regular_bytes_capped("file.txt"), b"hello")
                with mock.patch.object(os.environ, "get", return_value=None):
                    audit._verify_pwd_not_symlink()
                with mock.patch.dict(os.environ, {"PWD": ""}):
                    audit._verify_pwd_not_symlink()
                with mock.patch.dict(os.environ, {"PWD": "."}):
                    audit._verify_pwd_not_symlink()
                with mock.patch.dict(os.environ, {"PWD": "/nonexistent_s79_dir_xyz"}):
                    self.assertEqual(audit.read_regular_bytes_capped("file.txt"), b"hello")
                with mock.patch.dict(os.environ, {"PWD": str(root)}):
                    self.assertEqual(audit.read_regular_bytes_capped("file.txt"), b"hello")
                with mock.patch.dict(os.environ, {"PWD": str(sym_dir)}):
                    with self.assertRaises(oracle.S79Error) as c_psym:
                        audit.read_regular_bytes_capped("file.txt")
                    self.assertEqual(c_psym.exception.code, "SYMLINK_FORBIDDEN")
                with mock.patch.object(os.environ, "get", return_value="/bad\x00pwd"):
                    with self.assertRaises(oracle.S79Error) as c_pnul:
                        audit.read_regular_bytes_capped("file.txt")
                    self.assertEqual(c_pnul.exception.code, "INVALID_PATH")
                with mock.patch.dict(os.environ, {"PWD": str(real_dir)}), mock.patch.object(
                    os, "stat", side_effect=OSError("deleted cwd")
                ):
                    with self.assertRaises(oracle.S79Error) as c_cwd1:
                        audit.read_regular_bytes_capped("file.txt")
                    self.assertEqual(c_cwd1.exception.code, "CWD_UNAVAILABLE")
                with mock.patch.dict(os.environ, {"PWD": ""}), mock.patch.object(
                    os, "getcwd", side_effect=OSError("no cwd")
                ):
                    with self.assertRaises(oracle.S79Error) as c_cwd2:
                        audit.read_regular_bytes_capped("file.txt")
                    self.assertEqual(c_cwd2.exception.code, "CWD_UNAVAILABLE")
                fake_st = mock.Mock(st_dev=-1, st_ino=-1)
                with mock.patch.dict(os.environ, {"PWD": str(real_dir)}), mock.patch.object(
                    os, "fstat", return_value=fake_st
                ):
                    with self.assertRaises(oracle.S79Error) as c_pmm:
                        audit._verify_pwd_not_symlink()
                    self.assertEqual(c_pmm.exception.code, "PWD_SYMLINK_MISMATCH")
            finally:
                os.chdir(old_cwd)

            # Atomic 0o600 write under permissive umask 0o022
            out_file = real_dir / "atomic.json"
            old_umask = os.umask(0o022)
            try:
                audit.write_bytes_atomic_0600(out_file, b'{"ok":true}\n')
            finally:
                os.umask(old_umask)
            self.assertEqual(stat.S_IMODE(os.lstat(out_file).st_mode), 0o600)
            # Overwrite existing regular file cleanly
            audit.write_bytes_atomic_0600(out_file, b'{"ok":1}\n')
            self.assertEqual(out_file.read_bytes(), b'{"ok":1}\n')

            # Error paths for write_bytes_atomic_0600 and read_regular_bytes_capped
            with self.assertRaises(oracle.S79Error) as c_nb:
                audit.write_bytes_atomic_0600(out_file, "str")  # type: ignore[arg-type]
            self.assertEqual(c_nb.exception.code, "INVALID_BYTES")

            with mock.patch.object(os, "open", side_effect=OSError("root fail")):
                with self.assertRaises(oracle.S79Error) as c_ro:
                    audit._open_dir_chain_nofollow(("tmp",))
                self.assertEqual(c_ro.exception.code, "DIR_OPEN_FAILED")

            # ENOTDIR on symlink component fallback (macOS/BSD semantics simulated)
            orig_open = os.open

            def _enotdir_on_sym(path, flags, *a, **kw):
                if path == "sym_dir":
                    raise OSError(errno.ENOTDIR, "not a dir")
                return orig_open(path, flags, *a, **kw)

            with mock.patch.object(os, "open", side_effect=_enotdir_on_sym):
                with self.assertRaises(oracle.S79Error) as c_esym:
                    audit.read_regular_bytes_capped(sym_dir / "file.txt")
                self.assertEqual(c_esym.exception.code, "SYMLINK_FORBIDDEN")
                with mock.patch.object(os, "lstat", side_effect=OSError("lstat fail")):
                    with self.assertRaises(oracle.S79Error) as c_end2:
                        audit.read_regular_bytes_capped(sym_dir / "file.txt")
                    self.assertEqual(c_end2.exception.code, "NOT_A_DIRECTORY")

            # Mocked fstat/read IO errors on read_regular_bytes_capped
            with mock.patch.object(os, "fstat", side_effect=OSError("fstat fail")):
                with self.assertRaises(oracle.S79Error) as c_fs:
                    audit.read_regular_bytes_capped(reg)
                self.assertEqual(c_fs.exception.code, "READ_IO_ERROR")
            with mock.patch.object(os, "read", side_effect=OSError("read fail")):
                with self.assertRaises(oracle.S79Error) as c_rd:
                    audit.read_regular_bytes_capped(reg)
                self.assertEqual(c_rd.exception.code, "READ_IO_ERROR")

            # Mocked write error paths: lstat OSError, FileExistsError exhaustion, ENOSPC, short write, temp tamper, race checks
            with mock.patch.object(os, "lstat", side_effect=OSError(errno.EACCES, "perm")):
                with self.assertRaises(oracle.S79Error) as c_ls:
                    audit.write_bytes_atomic_0600(out_file, b"x")
                self.assertEqual(c_ls.exception.code, "TARGET_UNSTATTABLE")

            def _exists_on_tmp(path, flags, *a, **kw):
                if isinstance(path, str) and path.startswith(".atomic.json.tmp."):
                    raise FileExistsError("exists")
                return orig_open(path, flags, *a, **kw)

            with mock.patch.object(os, "open", side_effect=_exists_on_tmp):
                with self.assertRaises(oracle.S79Error) as c_exh:
                    audit.write_bytes_atomic_0600(out_file, b"x")
                self.assertEqual(c_exh.exception.code, "TEMP_CREATE_FAILED")

            def _enospc_on_tmp(path, flags, *a, **kw):
                if isinstance(path, str) and path.startswith(".atomic.json.tmp."):
                    raise OSError(errno.ENOSPC, "no space")
                return orig_open(path, flags, *a, **kw)

            with mock.patch.object(os, "open", side_effect=_enospc_on_tmp):
                with self.assertRaises(oracle.S79Error) as c_sp1:
                    audit.write_bytes_atomic_0600(out_file, b"x")
                self.assertEqual(c_sp1.exception.code, "WRITE_IO_ERROR")

            with mock.patch.object(os, "write", side_effect=OSError(errno.ENOSPC, "no space")):
                with self.assertRaises(oracle.S79Error) as c_sp2:
                    audit.write_bytes_atomic_0600(out_file, b"x")
                self.assertEqual(c_sp2.exception.code, "WRITE_IO_ERROR")
            # Confirm zero leftover temp files after ENOSPC!
            leftovers = [p.name for p in real_dir.iterdir() if ".tmp." in p.name]
            self.assertEqual(leftovers, [])

            with mock.patch.object(os, "write", return_value=0):
                with self.assertRaises(oracle.S79Error) as c_sw:
                    audit.write_bytes_atomic_0600(out_file, b"x")
                self.assertEqual(c_sw.exception.code, "SHORT_WRITE")

            # Temp tamper: non-regular mode OR nlink != 1
            for bad_st in (
                mock.Mock(st_mode=stat.S_IFIFO | 0o600, st_nlink=1),
                mock.Mock(st_mode=stat.S_IFREG | 0o600, st_nlink=2),
            ):
                with mock.patch.object(os, "fstat", return_value=bad_st):
                    with self.assertRaises(oracle.S79Error) as c_tt:
                        audit.write_bytes_atomic_0600(out_file, b"x")
                    self.assertEqual(c_tt.exception.code, "TEMP_TAMPERED")

            # Race re-check (second lstat sees symlink, FIFO, or hardlink)
            orig_lstat = os.lstat
            for race_mode, race_nlink, race_code in (
                (stat.S_IFLNK | 0o777, 1, "SYMLINK_FORBIDDEN"),
                (stat.S_IFIFO | 0o600, 1, "NON_REGULAR_FILE"),
                (stat.S_IFREG | 0o600, 2, "HARDLINK_FORBIDDEN"),
            ):
                calls = [0]

                def _race_lstat(path, *a, **kw):
                    calls[0] += 1
                    if calls[0] == 2:
                        return mock.Mock(st_mode=race_mode, st_nlink=race_nlink)
                    return orig_lstat(path, *a, **kw)

                with mock.patch.object(os, "lstat", side_effect=_race_lstat):
                    with self.assertRaises(oracle.S79Error) as c_rc:
                        audit.write_bytes_atomic_0600(out_file, b"x")
                    self.assertEqual(c_rc.exception.code, race_code)


class TestSecondaryAuditAndParametricTamper(unittest.TestCase):
    def test_exhaustive_field_tamper_kills_secondary_audit(self) -> None:
        base_cert = oracle.build_certificate()
        self.assertTrue(audit.verify_certificate_dict(base_cert, full_replay=True)["valid"])
        self.assertTrue(audit.verify_certificate_dict(base_cert, full_replay=False)["valid"])

        base_payload = base_cert["payload"]

        # Envelope checks (full_replay=False + exact error codes to kill every guard independently!)
        with self.assertRaises(oracle.S79Error) as c_rf:
            audit.verify_certificate_dict(base_cert, full_replay=1)  # type: ignore[arg-type]
        self.assertEqual(c_rf.exception.code, "INVALID_BOOL")
        with self.assertRaises(oracle.S79Error):
            audit.verify_certificate_dict("not-dict")
        c_extra = copy.deepcopy(base_cert)
        c_extra["extra"] = 1
        with self.assertRaises(oracle.S79Error) as c_ex:
            audit.verify_certificate_dict(c_extra, full_replay=False)
        self.assertEqual(c_ex.exception.code, "KEYSET_MISMATCH")
        c_sch = copy.deepcopy(base_cert)
        c_sch["schema_version"] = "wrong"
        with self.assertRaises(oracle.S79Error) as c_se:
            audit.verify_certificate_dict(c_sch, full_replay=False)
        self.assertEqual(c_se.exception.code, "SCHEMA_MISMATCH")
        c_pid = copy.deepcopy(base_cert)
        c_pid["project_id"] = "wrong"
        with self.assertRaises(oracle.S79Error) as c_pe:
            audit.verify_certificate_dict(c_pid, full_replay=False)
        self.assertEqual(c_pe.exception.code, "PROJECT_ID_MISMATCH")
        c_psha = copy.deepcopy(base_cert)
        c_psha["payload_sha256"] = "0" * 64
        with self.assertRaises(oracle.S79Error) as c_pse:
            audit.verify_certificate_dict(c_psha, full_replay=False)
        self.assertEqual(c_pse.exception.code, "PAYLOAD_SHA256_MISMATCH")
        c_csha = copy.deepcopy(base_cert)
        c_csha["certificate_sha256"] = "0" * 64
        with self.assertRaises(oracle.S79Error) as c_cse:
            audit.verify_certificate_dict(c_csha, full_replay=False)
        self.assertEqual(c_cse.exception.code, "CERTIFICATE_SHA256_MISMATCH")
        c_osha = copy.deepcopy(base_cert)
        c_osha["oracle_module_sha256"] = "0" * 64
        with self.assertRaises(oracle.S79Error) as c_osraw:
            audit.verify_certificate_dict(c_osha, full_replay=False)
        self.assertEqual(c_osraw.exception.code, "CERTIFICATE_SHA256_MISMATCH")
        # Re-signed envelope with a wrong oracle pin: only the pre-replay oracle
        # pin check can catch it -> fail closed ORACLE_SHA256_MISMATCH.
        c_osha = _resign_cert(base_payload, oracle_sha="0" * 64)
        with self.assertRaises(oracle.S79Error) as c_ose:
            audit.verify_certificate_dict(c_osha, full_replay=True)
        self.assertEqual(c_ose.exception.code, "ORACLE_SHA256_MISMATCH")

        # Full replay mismatch (simulated with patched secondary_audit_payload)
        p_alt = copy.deepcopy(base_payload)
        p_alt["summary"]["target_max_cert_bytes"] = 9999
        c_alt = _resign_cert(p_alt)
        with mock.patch.object(audit, "secondary_audit_payload", return_value=None):
            with self.assertRaises(oracle.S79Error) as ctx_fr:
                audit.verify_certificate_dict(c_alt, full_replay=True)
            self.assertEqual(ctx_fr.exception.code, "FULL_REPLAY_MISMATCH")

        # Tamper top-level project_id
        p_bad_id = copy.deepcopy(base_payload)
        p_bad_id["project_id"] = "TAMPERED"
        with self.assertRaises(oracle.S79Error):
            audit.verify_certificate_dict(_resign_cert(p_bad_id), full_replay=False)

        # Tamper every leaf field in Pillars 1, 2, 3, Summary, and Pillar 4!
        def _tamper_value(val):
            if type(val) is bool:
                return not val
            if type(val) is int:
                return val + 1
            if type(val) is str:
                if len(val) == 64 and all(c in "0123456789abcdef" for c in val):
                    return ("0" if val[0] != "0" else "1") + val[1:]
                if len(val) == 40 and all(c in "0123456789abcdef" for c in val):
                    return ("0" if val[0] != "0" else "1") + val[1:]
                if "/" in val and all(c in "-0123456789/" for c in val):
                    q = oracle.parse_exact_rational(val, "v")
                    return oracle.fmt_q(q + Fraction(1, 1000003))
                if val.lstrip("-").isdigit():
                    return str(int(val) + 1)
                return val + "_tampered"
            if type(val) is list:
                if val and type(val[0]) is int:
                    return [val[0] + 1] + val[1:]
                if val and type(val[0]) is str:
                    return [val[0] + "_x"] + val[1:]
                return ["extra"]
            return val

        for sec_key in (
            "pillar1_2dsd_cascade_and_kv_laws",
            "pillar2_dense_sparse_vs_siliquant",
            "pillar3_multipole_error_and_barrier",
            "summary",
        ):
            for field, orig in base_payload[sec_key].items():
                p_mut = copy.deepcopy(base_payload)
                p_mut[sec_key][field] = _tamper_value(orig)
                with self.assertRaises(oracle.S79Error, msg=f"unrejected tamper {sec_key}.{field}"):
                    audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)

        # Direct tuple-vs-list checks on _require_exact_int_list and _require_exact_str_list
        with self.assertRaises(oracle.S79Error) as c_il:
            audit._require_exact_int_list((60, 30, 10), [60, 30, 10], "t_int")
        self.assertEqual(c_il.exception.code, "INT_LIST_MISMATCH")
        with self.assertRaises(oracle.S79Error) as c_sl:
            audit._require_exact_str_list(tuple(oracle.PRIOR_LITERATURE_N0), list(oracle.PRIOR_LITERATURE_N0), "t_str")
        self.assertEqual(c_sl.exception.code, "STR_LIST_MISMATCH")

        # Pillar 4 exhaustive field tampering
        p4_base = base_payload["pillar4_formal_inventory_and_scope"]
        for field, orig in p4_base.items():
            if field in ("modules", "primary_source_commits", "primary_source_sha256"):
                continue
            p_mut = copy.deepcopy(base_payload)
            p_mut["pillar4_formal_inventory_and_scope"][field] = _tamper_value(orig)
            with self.assertRaises(oracle.S79Error, msg=f"unrejected tamper p4.{field}"):
                audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)

        # Pillar 4 nested modules shape & fields (len-1 non-list "a" kills type(mods) is not list!)
        for bad_mods in ("a", []):
            p_mut = copy.deepcopy(base_payload)
            p_mut["pillar4_formal_inventory_and_scope"]["modules"] = bad_mods
            with self.assertRaises(oracle.S79Error) as c_ms:
                audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)
            self.assertEqual(c_ms.exception.code, "P4_MODULES_SHAPE")

        for mf in ("relpath", "namespace", "sha256", "theorem_count"):
            p_mut = copy.deepcopy(base_payload)
            p_mut["pillar4_formal_inventory_and_scope"]["modules"][0][mf] = _tamper_value(p4_base["modules"][0][mf])
            with self.assertRaises(oracle.S79Error, msg=f"unrejected tamper mod.{mf}"):
                audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)

        p_mut = copy.deepcopy(base_payload)
        p_mut["pillar4_formal_inventory_and_scope"]["modules"][0]["theorems"]["cascade_single_step_lossless_universal"] = [
            "propext"
        ]
        with self.assertRaises(oracle.S79Error):
            audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)

        # Pillar 4 primary commits and snapshot SHAs
        for ck, orig in p4_base["primary_source_commits"].items():
            p_mut = copy.deepcopy(base_payload)
            p_mut["pillar4_formal_inventory_and_scope"]["primary_source_commits"][ck] = _tamper_value(orig)
            with self.assertRaises(oracle.S79Error, msg=f"unrejected tamper commit.{ck}"):
                audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)

        for sk, orig in p4_base["primary_source_sha256"].items():
            p_mut = copy.deepcopy(base_payload)
            p_mut["pillar4_formal_inventory_and_scope"]["primary_source_sha256"][sk] = _tamper_value(orig)
            with self.assertRaises(oracle.S79Error, msg=f"unrejected tamper sha.{sk}"):
                audit.verify_certificate_dict(_resign_cert(p_mut), full_replay=False)


class TestCLISurfaceAndCommittedCert(unittest.TestCase):
    def _run_main(self, argv: list[str] | tuple[str, ...] | None) -> tuple[int, str, str]:
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with mock.patch.object(sys, "stdout", out_buf), mock.patch.object(sys, "stderr", err_buf):
            rc = audit.main(argv)
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def test_committed_certificate_and_ci_0644_parity(self) -> None:
        self.assertTrue(_CERT_PATH.is_file())
        committed = _CERT_PATH.read_bytes()

        # --standalone matches committed certificate byte-for-byte
        rc_s, out_s, err_s = self._run_main(["--standalone"])
        self.assertEqual(rc_s, 0)
        self.assertEqual(err_s, "")
        self.assertEqual(out_s.encode("ascii"), committed)

        # --check passes on committed cert (text & --json)
        rc_c, out_c, err_c = self._run_main(["--check", str(_CERT_PATH)])
        self.assertEqual(rc_c, 0)
        self.assertTrue(out_c.startswith("VALID "))
        self.assertEqual(err_c, "")

        rc_cj, out_cj, _ = self._run_main(["--check", str(_CERT_PATH), "--json"])
        self.assertEqual(rc_cj, 0)
        self.assertTrue(json.loads(out_cj)["valid"])

        # Simulated CI git checkout copy with 0o644 permissions passes --check, while --write enforces 0o600
        with tempfile.TemporaryDirectory() as td:
            ci_copy = Path(td) / "ci_cert.json"
            ci_copy.write_bytes(committed)
            os.chmod(ci_copy, 0o644)
            rc_ci, _, err_ci = self._run_main(["--check", str(ci_copy)])
            self.assertEqual(rc_ci, 0)
            self.assertEqual(err_ci, "")

            # Non-canonical JSON & invalid UTF-8 rejected by --check
            non_canon = Path(td) / "non_canon.json"
            non_canon.write_bytes(b" " + committed)
            rc_nc, _, err_nc = self._run_main(["--check", str(non_canon)])
            self.assertEqual(rc_nc, 2)
            self.assertIn("NON_CANONICAL_BYTES", err_nc)

            bad_u8 = Path(td) / "bad_u8.json"
            bad_u8.write_bytes(b"\xff\xfe")
            rc_u8, _, err_u8 = self._run_main(["--check", str(bad_u8)])
            self.assertEqual(rc_u8, 2)
            self.assertIn("INVALID_UTF8", err_u8)

            # --write text & --json enforce 0o600 under umask 0o022
            out_w1 = Path(td) / "w1.json"
            out_w2 = Path(td) / "w2.json"
            old_um = os.umask(0o022)
            try:
                rc_w1, out_wt, _ = self._run_main(["--write", str(out_w1)])
                rc_w2, out_wj, _ = self._run_main(["--write", str(out_w2), "--json"])
            finally:
                os.umask(old_um)
            self.assertEqual(rc_w1, 0)
            self.assertTrue(out_wt.startswith("WROTE "))
            self.assertEqual(stat.S_IMODE(os.lstat(out_w1).st_mode), 0o600)
            self.assertEqual(out_w1.read_bytes(), committed)
            self.assertEqual(rc_w2, 0)
            self.assertTrue(json.loads(out_wj)["written"])
            self.assertEqual(stat.S_IMODE(os.lstat(out_w2).st_mode), 0o600)

    def test_planted_oracle_edit_fails_closed(self) -> None:
        # M2 control: the certificate pins sha256(s79_2dsd_multipole_oracle.py); the
        # verifier must check that pin BEFORE full replay and fail closed (rc=2,
        # ORACLE_SHA256_MISMATCH) when the oracle module on disk was edited.
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "bundle"
            bundle.mkdir()
            for name in (
                "s79_2dsd_multipole_oracle.py",
                "verify_s79_2dsd_multipole.py",
                "s79_2dsd_multipole_certificate.json",
            ):
                shutil.copy2(_THIS_DIR / name, bundle / name)
            verifier_copy = bundle / "verify_s79_2dsd_multipole.py"
            cert_copy = bundle / "s79_2dsd_multipole_certificate.json"
            oracle_copy = bundle / "s79_2dsd_multipole_oracle.py"

            def _run_check() -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-I", "-B", str(verifier_copy), "--check", str(cert_copy)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )

            # Positive control: unmodified copy of the bundle passes --check.
            proc_ok = _run_check()
            self.assertEqual(proc_ok.returncode, 0, msg=proc_ok.stderr)
            self.assertIn("VALID ", proc_ok.stdout)

            # Plant an oracle edit (constant bump) in the copied bundle.
            raw = oracle_copy.read_bytes()
            planted = raw.replace(b"MAX_INT_BITS = 3321", b"MAX_INT_BITS = 3322")
            self.assertNotEqual(planted, raw)
            oracle_copy.write_bytes(planted)

            # The planted edit must fail closed with a clear error code before replay.
            proc_bad = _run_check()
            self.assertEqual(proc_bad.returncode, 2)
            self.assertIn("ORACLE_SHA256_MISMATCH", proc_bad.stderr)
            self.assertIn("ERROR ORACLE_SHA256_MISMATCH", proc_bad.stderr)

    def test_self_check_verify_lean_and_cli_errors(self) -> None:
        # Default self_check (text & --json & argv=None)
        rc0, out0, _ = self._run_main([])
        self.assertEqual(rc0, 0)
        self.assertTrue(out0.startswith("OK "))

        rc0j, out0j, _ = self._run_main(["--json"])
        self.assertEqual(rc0j, 0)
        self.assertTrue(json.loads(out0j)["valid"])

        with mock.patch.object(sys, "argv", ["verify_s79_2dsd_multipole.py"]):
            rc_n, out_n, _ = self._run_main(None)
            self.assertEqual(rc_n, 0)
            self.assertTrue(out_n.startswith("OK "))

        # --verify-lean (text mocked for speed & live --json)
        lean_bin = _find_lean_bin()
        self.assertIsNotNone(lean_bin)
        fake_lean_res = {
            "valid": True,
            "modules_verified": ["TwoDSDAndDenseSparseLaws.lean"],
            "kernel_theorems_verified": 18,
            "live_lean_checked": True,
            "static_r9_r10_clean": True,
            "banned_axioms_found": 0,
        }
        with mock.patch.object(audit, "verify_lean_modules_live", return_value=fake_lean_res):
            rc_lt, out_lt, _ = self._run_main(["--verify-lean", str(_THIS_DIR)])
            self.assertEqual(rc_lt, 0)
            self.assertEqual(out_lt.strip(), "LEAN_VALID 1 18")

            rc_lj, out_lj, _ = self._run_main(["--verify-lean", str(_THIS_DIR), "--json"])
            self.assertEqual(rc_lj, 0)
            self.assertEqual(json.loads(out_lj)["kernel_theorems_verified"], 18)

        # CLI fail-closed rc=2 errors
        bad_argvs = [
            (["--standalone", "--standalone"], "CLI_CONFLICT"),
            (["--standalone", "--check", "x"], "CLI_CONFLICT"),
            (["--check", "x", "--write", "y"], "CLI_CONFLICT"),
            (["--json", "--json"], "CLI_DUPLICATE"),
            (["--check"], "CLI_MISSING_ARG"),
            (["--write"], "CLI_MISSING_ARG"),
            (["--verify-lean"], "CLI_MISSING_ARG"),
            (["--check", ""], "CLI_INVALID_PATH"),
            (["--check", "--json"], "CLI_INVALID_PATH"),
            (["--unknown"], "CLI_UNKNOWN_ARG"),
            ("bad-argv", "CLI_INVALID_ARGV"),
        ]
        for bad_argv, code in bad_argvs:
            rc, _, err = self._run_main(bad_argv)  # type: ignore[arg-type]
            self.assertEqual(rc, 2)
            self.assertIn(code, err)


if __name__ == "__main__":
    unittest.main()
