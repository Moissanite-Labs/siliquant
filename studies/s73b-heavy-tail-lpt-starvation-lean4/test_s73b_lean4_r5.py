#!/usr/bin/env python3
"""Comprehensive R5 Verification & Hardening Test Suite for Heavy-Tail LPT Starvation Theory.

Zero bare `assert` statements, scratch `os.mkfifo` non-regular device probes, zero floats on trust paths,
zero mathematical RNG. Supports both `0o600` and `0o644` modes on the committed fixture
certificate while enforcing strict `0o600` on `--write` outputs.
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

import s73b_lean4_audit as audit  # noqa: E402
import s73b_lean4_oracle as oracle  # noqa: E402

_CERT_PATH = _THIS_DIR / "s73b_certificate.json"


class _FloatSubclass(float):
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


def _resign_cert(payload: dict) -> dict:
    p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    cert_pre = {
        "schema_version": oracle.SCHEMA_VERSION,
        "project_id": oracle.PROJECT_ID,
        "payload_sha256": p_sha,
        "payload": payload,
    }
    c_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(cert_pre))
    return {
        "schema_version": oracle.SCHEMA_VERSION,
        "project_id": oracle.PROJECT_ID,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
        "payload": payload,
    }


class TestOraclePrimitivesAndMath(unittest.TestCase):
    def test_require_helpers_and_boundaries(self) -> None:
        oracle.require_ok(True, "OK", "ok")
        with self.assertRaises(oracle.S73bError) as ctx:
            oracle.require_ok(False, "FAIL_CODE", "fail msg")
        self.assertEqual(ctx.exception.code, "FAIL_CODE")

        self.assertEqual(oracle.require_exact_int(0, "x"), 0)
        self.assertEqual(oracle.require_exact_int(5, "x", min_val=5, max_val=5), 5)
        for bad in (True, False, "5", None):
            with self.assertRaises(oracle.S73bError):
                oracle.require_exact_int(bad, "x")
        with self.assertRaises(oracle.S73bError) as c_lo:
            oracle.require_exact_int(4, "x", min_val=5)
        self.assertEqual(c_lo.exception.code, "INT_BELOW_MIN")
        with self.assertRaises(oracle.S73bError) as c_hi:
            oracle.require_exact_int(6, "x", max_val=5)
        self.assertEqual(c_hi.exception.code, "INT_ABOVE_MAX")

        self.assertIs(oracle.require_exact_bool(True, "b"), True)
        self.assertIs(oracle.require_exact_bool(False, "b"), False)
        for bad in (1, 0, "true", None):
            with self.assertRaises(oracle.S73bError):
                oracle.require_exact_bool(bad, "b")

        self.assertEqual(oracle.require_exact_str("", "s", non_empty=False), "")
        self.assertEqual(oracle.require_exact_str("abc", "s"), "abc")
        with self.assertRaises(oracle.S73bError):
            oracle.require_exact_str(123, "s")
        with self.assertRaises(oracle.S73bError):
            oracle.require_exact_str("", "s", non_empty=True)
        with self.assertRaises(oracle.S73bError) as c_sur:
            oracle.require_exact_str("\ud800", "s")
        self.assertEqual(c_sur.exception.code, "INVALID_UTF8_STR")

        good_sha = "a" * 64
        self.assertEqual(oracle.require_sha256_hex(good_sha, "h"), good_sha)
        self.assertTrue(oracle.check_digest_eq(good_sha, good_sha))
        self.assertFalse(oracle.check_digest_eq(good_sha, "b" * 64))
        for bad_sha in ("a" * 63, "a" * 65, "A" * 64, "g" * 64):
            with self.assertRaises(oracle.S73bError):
                oracle.require_sha256_hex(bad_sha, "h")

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
            with self.assertRaises(oracle.S73bError) as ctx:
                oracle.parse_exact_int_ascii(tok, "i")
            self.assertEqual(ctx.exception.code, code, f"token={tok[:20]!r}")

        self.assertEqual(oracle.fmt_q(Fraction(7, 1)), "7")
        self.assertEqual(oracle.fmt_q(Fraction(-5, 3)), "-5/3")
        with self.assertRaises(oracle.S73bError):
            oracle.fmt_q(3)  # type: ignore[arg-type]

        self.assertEqual(oracle.parse_exact_rational("5", "q"), Fraction(5, 1))
        self.assertEqual(oracle.parse_exact_rational("-3/2", "q"), Fraction(-3, 2))
        bad_qs = [
            ("1/2/3", "INVALID_RATIONAL"),
            ("1/0", "ZERO_DENOMINATOR"),
            ("3/1", "NON_CANONICAL_DEN"),
            ("1/-2", "NON_CANONICAL_DEN"),
            ("2/4", "UNREDUCED_RATIONAL"),
            ("0/2", "UNREDUCED_RATIONAL"),
        ]
        for tok, code in bad_qs:
            with self.assertRaises(oracle.S73bError) as ctx:
                oracle.parse_exact_rational(tok, "q")
            self.assertEqual(ctx.exception.code, code, f"rational={tok!r}")

    def test_strict_json_parser_and_canonical_encoder(self) -> None:
        obj = {"b": [1, True, False, None, "ok"], "a": {"k": -7}}
        canon = oracle.canonical_json_bytes(obj)
        self.assertEqual(canon, b'{"a":{"k":-7},"b":[1,true,false,null,"ok"]}')
        self.assertEqual(oracle.parse_strict_json_text(canon.decode("ascii")), obj)
        self.assertEqual(len(oracle.sha256_bytes(canon)), 64)
        with self.assertRaises(oracle.S73bError):
            oracle.sha256_bytes("not-bytes")  # type: ignore[arg-type]

        ok_nested: dict = {}
        cur = ok_nested
        for _ in range(oracle.MAX_JSON_DEPTH):
            nxt: dict = {}
            cur["a"] = nxt
            cur = nxt
        oracle.check_json_value_strict(ok_nested)
        ok_raw_nested = '{"a":' * (oracle.MAX_JSON_DEPTH - 1) + '{}' + '}' * (oracle.MAX_JSON_DEPTH - 1)
        self.assertIsInstance(oracle.parse_strict_json_text(ok_raw_nested), dict)
        too_deep = {"a": ok_nested}
        with self.assertRaises(oracle.S73bError) as c_d:
            oracle.check_json_value_strict(too_deep)
        self.assertEqual(c_d.exception.code, "MAX_DEPTH_EXCEEDED")

        str_braces = '{"s":"' + ('{[\\"' * 40) + '"}'
        self.assertIn("s", oracle.parse_strict_json_text(str_braces))

        exact_16 = '{"k":"01234567"}'
        self.assertEqual(len(exact_16.encode("ascii")), 16)
        with mock.patch.object(oracle, "MAX_JSON_BYTES", 16):
            self.assertEqual(len(oracle.canonical_json_bytes({"k": "01234567"})), 16)
            self.assertEqual(oracle.parse_strict_json_text(exact_16), {"k": "01234567"})
            with self.assertRaises(oracle.S73bError) as c_big:
                oracle.canonical_json_bytes({"k": "012345678"})
            self.assertEqual(c_big.exception.code, "JSON_TOO_LARGE")
            with self.assertRaises(oracle.S73bError) as c_big2:
                oracle.parse_strict_json_text('{"k":"012345678"}')
            self.assertEqual(c_big2.exception.code, "JSON_TOO_LARGE")

        deep_raw = "[" * (oracle.MAX_JSON_DEPTH + 1) + "]" * (oracle.MAX_JSON_DEPTH + 1)
        bad_jsons = [
            ('{"x":1.5}', "JSON_FLOAT_FORBIDDEN"),
            ('{"x":NaN}', "JSON_CONSTANT_FORBIDDEN"),
            ('{"x":Infinity}', "JSON_CONSTANT_FORBIDDEN"),
            ('{"a":1,"a":2}', "DUPLICATE_JSON_KEY"),
            ('{"a":1}\r\n', "CRLF_FORBIDDEN"),
            (deep_raw, "MAX_DEPTH_EXCEEDED"),
            ('[1,2,3]', "JSON_ROOT_NOT_OBJECT"),
            ('{bad', "INVALID_JSON"),
            ('{"n":' + ("9" * 5001) + '}', "INT_TOKEN_TOO_LONG"),
        ]
        for raw, code in bad_jsons:
            with self.assertRaises(oracle.S73bError) as ctx:
                oracle.parse_strict_json_text(raw)
            self.assertEqual(ctx.exception.code, code)

        for fl_val in (Fraction(3, 2).__float__(), _FloatSubclass("1.5")):
            with self.assertRaises(oracle.S73bError) as c_fl:
                oracle.check_json_value_strict({"x": fl_val})
            self.assertEqual(c_fl.exception.code, "FORBIDDEN_JSON_TYPE")
            self.assertIn("floating-point value", c_fl.exception.message)

        with self.assertRaises(oracle.S73bError) as c_type:
            oracle.check_json_value_strict({"x": object()})
        self.assertEqual(c_type.exception.code, "FORBIDDEN_JSON_TYPE")
        self.assertIn("unsupported JSON type", c_type.exception.message)

        with self.assertRaises(oracle.S73bError) as c_big_int:
            oracle.check_json_value_strict({"x": 1 << oracle.MAX_INT_BITS})
        self.assertEqual(c_big_int.exception.code, "INT_TOO_LARGE")

    def test_scheduling_and_pareto_helpers(self) -> None:
        # Exercise every branch of cap_lpt_2ball (len(b0)<k, len(b1)>=k, l0<=l1, and cap-exhausted ball0!)
        self.assertEqual(oracle.cap_lpt_2ball([100, 64, 25, 16, 4, 1], 3), ([0, 4, 5], [1, 2, 3]))
        self.assertEqual(oracle.cap_lpt_2ball([10, 8, 5, 5, 3, 1], 3), ([0, 3, 5], [1, 2, 4]))
        self.assertEqual(oracle.cap_lpt_2ball([1, 10, 1, 1, 1, 1], 3), ([0, 2, 3], [1, 4, 5]))
        self.assertEqual(oracle.cap_lpt_2ball([10, 10], 1), ([0], [1]))
        with self.assertRaises(oracle.S73bError):
            oracle.cap_lpt_2ball([1, 2], 0)
        with self.assertRaises(oracle.S73bError) as c_len:
            oracle.cap_lpt_2ball([1, 2, 3], 2)
        self.assertEqual(c_len.exception.code, "LPT_LENGTH")
        with self.assertRaises(oracle.S73bError):
            oracle.cap_lpt_2ball([-1, 2], 1)

        self.assertEqual(oracle.snake_2ball_indices(3), ([0, 3, 4], [1, 2, 5]))
        with self.assertRaises(oracle.S73bError):
            oracle.snake_2ball_indices(0)

        # Direct branch coverage of pareto_corner_holds (kills every BoolOp OP_SWAP & SUBCOND_DEL)
        self.assertTrue(oracle.pareto_corner_holds(1, 2, 3, 4, 5, 6))
        self.assertFalse(oracle.pareto_corner_holds(2, 2, 3, 4, 5, 6))
        self.assertFalse(oracle.pareto_corner_holds(1, 3, 3, 4, 5, 6))
        self.assertFalse(oracle.pareto_corner_holds(1, 2, 3, 6, 5, 6))
        self.assertFalse(oracle.pareto_corner_holds(1, 2, 3, 4, 6, 6))

    def test_build_certificate_and_committed_fixture_parity(self) -> None:
        cert = oracle.build_certificate()
        res = audit.verify_certificate_dict(cert, full_replay=True)
        self.assertTrue(res["valid"])
        self.assertTrue(res["full_replay"])
        self.assertTrue(res["secondary_audit_passed"])
        self.assertEqual(cert["payload"]["summary"]["total_lean4_theorems"], 21)

        self.assertTrue(_CERT_PATH.is_file())
        mode = stat.S_IMODE(os.lstat(_CERT_PATH).st_mode)
        self.assertIn(mode, (0o600, 0o644))
        raw = audit.read_regular_bytes_capped(_CERT_PATH)
        canon = oracle.canonical_json_bytes(cert)
        self.assertEqual(raw, canon + b"\n")
        self.assertLessEqual(len(canon), oracle.TARGET_MAX_CERT_BYTES)

        exact_4digit_len = len(canon) - 1
        with mock.patch.object(oracle, "TARGET_MAX_CERT_BYTES", exact_4digit_len):
            self.assertEqual(len(oracle.canonical_json_bytes(oracle.build_certificate())), exact_4digit_len)
        with mock.patch.object(oracle, "TARGET_MAX_CERT_BYTES", exact_4digit_len - 1):
            with self.assertRaises(oracle.S73bError) as ctx:
                oracle.build_certificate()
            self.assertEqual(ctx.exception.code, "CERT_SIZE_EXCEEDED")


class TestExhaustiveSecondaryAuditAndReplayTampering(unittest.TestCase):
    def _mutate_leaf_value(self, val: object) -> object:
        if type(val) is bool:
            return not val
        if type(val) is int:
            return val + 1
        if type(val) is str:
            if len(val) == 64 and all(c in "0123456789abcdef" for c in val):
                return ("0" if val[0] != "0" else "1") + val[1:]
            if "/" in val:
                try:
                    q = oracle.parse_exact_rational(val, "mut")
                    return oracle.fmt_q(q + Fraction(1, 1))
                except oracle.S73bError:
                    return val + "_tampered"
            body = val[1:] if val.startswith("-") else val
            if body and all(c in "0123456789" for c in body):
                return str(int(val) + 1)
            return val + "_tampered"
        return val

    def _walk_and_tamper(self, base_payload: dict, path: tuple, node: object, count: list[int]) -> None:
        if type(node) is dict:
            for k in list(node.keys()):
                p_miss = copy.deepcopy(base_payload)
                cur = p_miss
                for step in path:
                    cur = cur[step]
                del cur[k]
                with self.assertRaises(oracle.S73bError, msg=f"missing key {path + (k,)}"):
                    audit.secondary_audit_payload(p_miss)
                count[0] += 1
            p_extra = copy.deepcopy(base_payload)
            cur = p_extra
            for step in path:
                cur = cur[step]
            cur["__unexpected_extra_key__"] = 1
            with self.assertRaises(oracle.S73bError, msg=f"extra key at {path}"):
                audit.secondary_audit_payload(p_extra)
            count[0] += 1
            for k, v in node.items():
                self._walk_and_tamper(base_payload, path + (k,), v, count)
        elif type(node) is list:
            p_len = copy.deepcopy(base_payload)
            cur = p_len
            for step in path[:-1]:
                cur = cur[step]
            cur[path[-1]] = cur[path[-1]][:-1] if len(node) > 0 else ["unexpected_extra"]
            with self.assertRaises(oracle.S73bError, msg=f"bad list length at {path}"):
                audit.secondary_audit_payload(p_len)
            count[0] += 1
            for idx, item in enumerate(node):
                self._walk_and_tamper(base_payload, path + (idx,), item, count)
        else:
            p_mut = copy.deepcopy(base_payload)
            cur = p_mut
            for step in path[:-1]:
                cur = cur[step]
            cur[path[-1]] = self._mutate_leaf_value(node)
            resigned = _resign_cert(p_mut)
            with self.assertRaises(oracle.S73bError, msg=f"FR0 bypass at {path}"):
                audit.verify_certificate_dict(resigned, full_replay=False)
            with self.assertRaises(oracle.S73bError, msg=f"SEC bypass at {path}"):
                audit.secondary_audit_payload(p_mut)
            count[0] += 1

    def test_exhaustive_resigned_leaf_and_schema_tamper_sweep(self) -> None:
        base = oracle.build_certificate()
        count = [0]
        self._walk_and_tamper(base["payload"], (), base["payload"], count)
        self.assertGreater(count[0], 180)

    def test_top_level_cert_and_full_replay_traps(self) -> None:
        cert = oracle.build_certificate()
        res0 = audit.verify_certificate_dict(cert, full_replay=False)
        self.assertFalse(res0["full_replay"])

        for bad in (1, 0, "true", None):
            with self.assertRaises(oracle.S73bError):
                audit.verify_certificate_dict(cert, full_replay=bad)  # type: ignore[arg-type]

        with self.assertRaises(oracle.S73bError):
            audit.verify_certificate_dict([cert])
        c_bad_schema = copy.deepcopy(cert)
        c_bad_schema["schema_version"] = "wrong.v1"
        with self.assertRaises(oracle.S73bError) as ctx:
            audit.verify_certificate_dict(c_bad_schema, full_replay=False)
        self.assertEqual(ctx.exception.code, "SCHEMA_MISMATCH")

        c_bad_proj = copy.deepcopy(cert)
        c_bad_proj["project_id"] = "WRONG"
        with self.assertRaises(oracle.S73bError) as ctx2:
            audit.verify_certificate_dict(c_bad_proj, full_replay=False)
        self.assertEqual(ctx2.exception.code, "PROJECT_ID_MISMATCH")

        c_bad_psha = copy.deepcopy(cert)
        c_bad_psha["payload_sha256"] = "0" * 64
        with self.assertRaises(oracle.S73bError) as ctx3:
            audit.verify_certificate_dict(c_bad_psha, full_replay=False)
        self.assertEqual(ctx3.exception.code, "PAYLOAD_SHA256_MISMATCH")

        c_bad_csha = copy.deepcopy(cert)
        c_bad_csha["certificate_sha256"] = "0" * 64
        with self.assertRaises(oracle.S73bError) as ctx4:
            audit.verify_certificate_dict(c_bad_csha, full_replay=False)
        self.assertEqual(ctx4.exception.code, "CERTIFICATE_SHA256_MISMATCH")

        fake_expected = copy.deepcopy(cert)
        fake_expected["certificate_sha256"] = "f" * 64
        with mock.patch.object(oracle, "build_certificate", return_value=fake_expected):
            with self.assertRaises(oracle.S73bError) as ctx5:
                audit.verify_certificate_dict(cert, full_replay=True)
            self.assertEqual(ctx5.exception.code, "FULL_REPLAY_MISMATCH")

        p_m = copy.deepcopy(cert["payload"])
        p_m["pillar4_formal_inventory_and_scope"]["modules"] = {"a": 1, "b": 2, "c": 3}
        with self.assertRaises(oracle.S73bError):
            audit.secondary_audit_payload(p_m)

        with self.assertRaises(oracle.S73bError):
            audit._require_exact_int_list((10, 8), [10, 8], "xs")
        with self.assertRaises(oracle.S73bError):
            audit._require_exact_str_list(("ok",), ["ok"], "strs")


class TestR9Lean4StaticScannerAndLiveKernel(unittest.TestCase):
    def test_r9_static_positive_controls(self) -> None:
        clean_samples = [
            "-- sorry axiom native_decide #eval initialize\ntheorem t : 1 = 1 := by rfl\n",
            "/- outer /- nested / - sorry -/ still / - comment -/\ntheorem t : 1 = 1 := by decide\n",
            'def s1 := "sorry \\" \\n axiom"\ndef s2 := r"sorry"\ndef s3 := r#"sorry"#\ndef s3b := r##"hello "# sorry"##\n',
            'def s4 (x : Nat) := s!"prefix \\n { { a := x } } sorry"\ndef s5 := m!"axiom"\ndef s6 := f!"unsafe"\n',
            'def s7 := s! "hello \\{sorry}"\ndef s8 := s!a "{sorry}"\ndef s9 := s! r"ok" "{sorry}"\n',
            'def s10 := s! «tok» "{sorry}"\ndef s11 := s! \'a\' "{sorry}"\n',
            'def s12 := s! "{ let _ := s! { a := 1 }; def b := "{sorry}" }"\ndef s13 := s! "{ s! }" ++ "{sorry}"\n',
            "def c1 : Char := '\"'\ndef c2 : Char := '\\''\ndef c3 : Char := '\\x27'\ndef c4 : Char := '\\u0027'\n",
            "def f (r : Nat) (x' : Nat) := r + x' + (1 / 2)\ntheorem t1 : 1 = 1 := by decide -native\n",
            "theorem t2 : 1 = 1 := by decide (native := false)\n",
            "theorem t3 : 1 = 1 := by decide (native := (Bool.false))\n",
            "theorem t4 : 1 = 1 := by decide (config := { native := false })\n",
            "theorem t5 : 1 = 1 := by decide (config := ({ }))\n",
            "theorem t6 : 1 = 1 := by decide (config := { native := false, zeta := true })\n",
            "def cfg : DecideConfig := { native := false }\n",
            "def dotOk := Foo.partial + Foo.axiom + Foo.unsafe + Foo.run_cmd + Foo.initialize + Foo.builtin_initialize\n",
            "theorem example_thm : 1 = 1 := by decide +kernel\n#print axioms example_thm\n",
            "def «cleanName» := 1\ndef «clean + op» := 2\n",
            'r"sorry"',
            's!"sorry"',
            "-- comment at eof without newline",
            "def u := ) + ] + (1 / 2)\ntheorem t8 : 1 = 1 := by decide (config := ({ }))\n",
        ]
        for idx, src in enumerate(clean_samples):
            res = audit.audit_lean4_source(src)
            self.assertTrue(res["clean"], f"FP on sample {idx}: {res}")
            self.assertEqual(res["notices"], [], f"Unexpected notice on sample {idx}: {res}")

        # Direct _is_lean4_ident_prev & strip_lean4_with_report boundary assertions
        self.assertFalse(audit._is_lean4_ident_prev("x", -1))
        self.assertTrue(audit._is_lean4_ident_prev("x", 0))
        self.assertTrue(audit._is_lean4_ident_prev("_", 0))
        self.assertTrue(audit._is_lean4_ident_prev(".", 0))
        self.assertTrue(audit._is_lean4_ident_prev("'", 0))
        self.assertFalse(audit._is_lean4_ident_prev("!", 0))
        self.assertFalse(audit._is_lean4_ident_prev("!!", 1))
        self.assertFalse(audit._is_lean4_ident_prev(" !", 1))
        self.assertTrue(audit._is_lean4_ident_prev("a!", 1))
        self.assertTrue(audit._is_lean4_ident_prev("a!!", 2))
        self.assertTrue(audit._is_lean4_ident_prev("_!", 1))
        self.assertTrue(audit._is_lean4_ident_prev("'?", 1))

        self.assertEqual(audit.strip_lean4_with_report("'a'")[0], "   ")
        self.assertEqual(audit.strip_lean4_with_report("'\\''")[0], "    ")
        self.assertEqual(audit.strip_lean4_with_report("'\\x27'")[0], "      ")
        self.assertEqual(audit.strip_lean4_with_report("'\\u0027'")[0], "        ")
        self.assertEqual(audit.strip_lean4_with_report("def «a b» := 1")[0], "def       := 1")
        self.assertEqual(audit.strip_lean4_with_report("def a := '")[0], "def a := '")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\")[0], "def a := '\\")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\'")[0], "def a := '\\'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\n'")[0], "def a := '\n'")
        self.assertEqual(audit.strip_lean4_with_report("def a := 'ab'")[0], "def a := 'ab'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\\n'")[0], "def a := '\\\n'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\ab'")[0], "def a := '\\ab'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\x2")[0], "def a := '\\x2")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\x27a")[0], "def a := '\\x27a")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\xZZ'")[0], "def a := '\\xZZ'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\y27'")[0], "def a := '\\y27'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\u002")[0], "def a := '\\u002")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\u0027a")[0], "def a := '\\u0027a")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\uZZZZ'")[0], "def a := '\\uZZZZ'")
        self.assertEqual(audit.strip_lean4_with_report("def a := '\\v0027'")[0], "def a := '\\v0027'")
        self.assertEqual(audit.strip_lean4_with_report("def a := r#")[0], "def a := r#")
        self.assertEqual(audit.strip_lean4_with_report('def a := bar"x"')[0], "def a := bar   ")
        self.assertEqual(audit.strip_lean4_with_report('def a := x.r"x"')[0], "def a := x.r   ")
        self.assertEqual(audit.strip_lean4_with_report('!"x"')[0], "!   ")
        self.assertEqual(audit.strip_lean4_with_report('def a := x!"x"')[0], "def a := x!   ")
        self.assertEqual(audit.strip_lean4_with_report('def a := as!"x"')[0], "def a := as!   ")
        self.assertEqual(audit.strip_lean4_with_report('def a := .s!"x"')[0], "def a := .s!   ")
        self.assertEqual(audit.strip_lean4_with_report('def a := s!x')[0], "def a := s!x")
        self.assertEqual(audit.strip_lean4_with_report('def a := s"x"')[0], "def a := s   ")

        self.assertTrue(audit.audit_lean4_source('def a := bar"{sorry}"\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('def a := x.r"{sorry}"\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('def a := x!"{sorry}"\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('def a := as!"{sorry}"\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('def a := .s!"{sorry}"\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('!"{sorry}"\n')["clean"])
        self.assertFalse(audit.audit_lean4_source('s!"{sorry}"\n')["clean"])
        self.assertFalse(audit.audit_lean4_source('def a := bar"\\""\nsorry\n')["clean"])
        self.assertFalse(audit.audit_lean4_source('def a := x.r"\\""\nsorry\n')["clean"])
        self.assertFalse(audit.audit_lean4_source('def a := foo!r"\\""\nsorry\n')["clean"])
        self.assertFalse(audit.audit_lean4_source('def a := foo?r"\\""\nsorry\n')["clean"])
        self.assertFalse(audit.audit_lean4_source('def a := r"\\" sorry\n')["clean"])
        self.assertFalse(audit.audit_lean4_source("def a := '\\'\"'\ntheorem t : False := by sorry\n")["clean"])
        self.assertFalse(audit.audit_lean4_source("def a := '\\\n' \" sorry\n")["clean"])
        self.assertFalse(audit.audit_lean4_source("def a := '\n' \" sorry\n")["clean"])
        self.assertFalse(audit.audit_lean4_source("def a := 'x\"'\ntheorem t : False := by sorry\n")["clean"])
        self.assertFalse(audit.audit_lean4_source("def a := '\\xZZ' \" sorry\n")["clean"])
        self.assertFalse(audit.audit_lean4_source("def a := '\\uZZZZ' \" sorry\n")["clean"])
        self.assertFalse(audit.audit_lean4_source('def s := s!"unclosed\n')["clean"])
        self.assertEqual(audit.audit_lean4_source('def s := s!"unclosed\n')["codes"], ["LEAN4_UNTERMINATED_STRING"])
        self.assertFalse(audit.audit_lean4_source("def u := ]\n@[extern \"c\"] opaque f : Nat\n")["clean"])
        self.assertFalse(audit._is_lean4_native_rhs_false_span("(false", 0, 6))
        self.assertFalse(audit._is_lean4_native_rhs_false_span("false)", 0, 6))
        self.assertFalse(audit._is_lean4_native_rhs_false_span("(falsex", 0, 7))
        self.assertFalse(audit._is_lean4_native_rhs_false_span("xfalse)", 0, 7))
        self.assertTrue(audit._is_lean4_native_rhs_false_span("(false)", 0, 7))
        self.assertFalse(audit.audit_lean4_source("def c := /- / -/ sorry\n")["clean"])
        self.assertTrue(audit.audit_lean4_source("def c := /- a / b - sorry -/ 1\n")["clean"])
        self.assertFalse(audit.audit_lean4_source("-- benign comment\ntheorem t : False := by sorry\n")["clean"])
        self.assertTrue(audit.audit_lean4_source('r"\\" -- x')["clean"])
        self.assertTrue(audit.audit_lean4_source('def s := r"\\" -- x')["clean"])
        self.assertTrue(audit.audit_lean4_source('def s := r#"a " sorry"#\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('def a := s""\n')["clean"])
        self.assertTrue(audit.audit_lean4_source('!"{sorry}" -- s')["clean"])
        self.assertFalse(audit.audit_lean4_source('s!"{sorry}" -- x')["clean"])
        self.assertFalse(audit.audit_lean4_source('def a := s!"{sorry}"\n')["clean"])

        # Notices (clean=True, non-empty notices)
        res_cs = audit.audit_lean4_source("@[csimp] theorem c : 1 = 1 := rfl\n")
        self.assertTrue(res_cs["clean"])
        self.assertEqual(res_cs["notices"][0]["code"], "LEAN4_CSIMP_REWRITE_NOTICE")
        res_cs2 = audit.audit_lean4_source("attribute [csimp] foo\n")
        self.assertTrue(res_cs2["clean"])
        self.assertEqual(res_cs2["notices"][0]["code"], "LEAN4_CSIMP_REWRITE_NOTICE")
        res_vac = audit.audit_lean4_source("theorem v : False → True := fun h => False.elim h\n")
        self.assertTrue(res_vac["clean"])
        self.assertEqual(res_vac["notices"][0]["code"], "LEAN4_VACUOUS_SIGNATURE_NOTICE")
        res_ex = audit.audit_lean4_source("theorem e : ∃ x : Nat, x = 1 → True := ⟨0, fun _ => trivial⟩\n")
        self.assertTrue(res_ex["clean"])
        self.assertEqual(res_ex["notices"][0]["code"], "LEAN4_EXISTS_IMPLICATION_NOTICE")

    def test_r9_static_adversarial_bypasses_all_caught(self) -> None:
        attacks: list[tuple[str, str]] = [
            ("theorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("theorem t : False := by admit\n", "LEAN4_ADMIT"),
            ("axiom bogus : False\n", "LEAN4_AXIOM"),
            ("unsafe def u : Nat := 0\n", "LEAN4_UNSAFE"),
            ("theorem t : 1 = 1 := by native_decide\n", "LEAN4_NATIVE_DECIDE"),
            ("def x := sorryAx False\n", "LEAN4_SORRY_AX"),
            ("def x := Lean.Meta.mkSorry\n", "LEAN4_MK_SORRY"),
            ("def x := Lean.ofReduceBool\n", "LEAN4_OF_REDUCE_BOOL"),
            ("def x := Lean.ofReduceNat\n", "LEAN4_OF_REDUCE_NAT"),
            ("def x := Lean.trustCompiler\n", "LEAN4_TRUST_COMPILER"),
            ("import Mathlib\n", "LEAN4_FORBIDDEN_IMPORT"),
            ("public\n import Init\n", "LEAN4_FORBIDDEN_IMPORT"),
            ("partial\n def loop (n : Nat) : Nat := loop n\n", "LEAN4_PARTIAL"),
            ("set_option\n debug . skipKernelTC true\n", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option debug.byAsSorry true\ntheorem fake : False := by trivial\n", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option debug.proofAsSorry true\ntheorem fake : False := by trivial\n", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option debug.terminalTacticsAsSorry true\ntheorem fake : False := by omega\n", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option «debug».«byAsSorry» true\ntheorem fake : False := by trivial\n", "LEAN4_SKIP_KERNEL_TC"),
            ("set_option «debug».«proofAsSorry» true\ntheorem fake : False := by trivial\n", "LEAN4_SKIP_KERNEL_TC"),
            ('@[extern "c_fn"] opaque f : Nat\n', "LEAN4_ATTR_TRUST_ESCAPE"),
            ("@[implemented_by g] def f := 0\n", "LEAN4_ATTR_TRUST_ESCAPE"),
            ('attribute [extern "c_fn"] f\n', "LEAN4_ATTR_TRUST_ESCAPE"),
            ("attribute [implemented_by g] f\n", "LEAN4_ATTR_TRUST_ESCAPE"),
            ("theorem t : True := trivial#eval 1\n", "LEAN4_EVAL_SPOOF"),
            ("#eval! 1\n", "LEAN4_EVAL_SPOOF"),
            ("#reduce 1 + 1\n", "LEAN4_EVAL_SPOOF"),
            ("#exit\n", "LEAN4_EVAL_SPOOF"),
            ("#print Nat\n", "LEAN4_EVAL_SPOOF"),
            ("#print axioms! Nat\n", "LEAN4_EVAL_SPOOF"),
            ("#print axioms theorem\n", "LEAN4_EVAL_SPOOF"),
            ("#print axioms example : True := trivial\n", "LEAN4_EVAL_SPOOF"),
            ("#print «axioms» Nat\n", "LEAN4_EVAL_SPOOF"),
            ("run_cmd pure ()\n", "LEAN4_EVAL_SPOOF"),
            ("run_meta pure ()\n", "LEAN4_EVAL_SPOOF"),
            ('initialize IO.println "\'Fake\' does not depend on any axioms"\n', "LEAN4_EVAL_SPOOF"),
            ('builtin_initialize IO.println "\'Fake\' does not depend on any axioms"\n', "LEAN4_EVAL_SPOOF"),
            ("«initialize» pure ()\n", "LEAN4_EVAL_SPOOF"),
            # R9 unary !r / !s! / spaced & commented s!/f!/m! escapes
            ('def a : Bool := !r"\\".isEmpty\ntheorem fake : False := by sorry\n-- "\n', "LEAN4_SORRY"),
            ('def a : Bool := !!r"\\".isEmpty\ntheorem fake : False := by sorry\n-- "\n', "LEAN4_SORRY"),
            ('def a : Bool := !r#"\\"#.isEmpty\ntheorem fake : False := by sorry\n-- "\n', "LEAN4_SORRY"),
            ('def a : Bool := !s!"{let _ := "x"; (by sorry : String)}".isEmpty\n', "LEAN4_SORRY"),
            ('def a : Bool := !f!"{let _ := "x"; (by sorry : String)}".pretty.isEmpty\n', "LEAN4_SORRY"),
            ('def a : String := s! "{(by sorry : String)}"\n', "LEAN4_SORRY"),
            ('def a : String := s!\n"{(by sorry : String)}"\n', "LEAN4_SORRY"),
            ('def a : String := s! /- c -/ "{(by sorry : String)}"\n', "LEAN4_SORRY"),
            ('def a : String := s! -- c\n"{(by sorry : String)}"\n', "LEAN4_SORRY"),
            ('def a : Std.Format := f! "{(by sorry : String)}"\n', "LEAN4_SORRY"),
            ('def a := m! "{sorry}"\n', "LEAN4_SORRY"),
            ('def a : String := s! "{ s! "}" ++ (by sorry : String) }"\n', "LEAN4_SORRY"),
            ('def a : String := s! "{let _ := (by set_option debug.skipKernelTC true in trivial : True); "ok"}"\n', "LEAN4_SKIP_KERNEL_TC"),
            ("def «--» := 1\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def «/-» := 1\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ('def «"» := 1\ntheorem t : False := by sorry\n', "LEAN4_SORRY"),
            ('def «s!"» := 1\ntheorem t : False := by sorry\n', "LEAN4_SORRY"),
            ('def «r#"» := 1\ntheorem t : False := by sorry\n', "LEAN4_SORRY"),
            ("def «a\nb» := 1\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def f (x' : Nat) := x' '\"' \ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def f (x!' : Nat) := x!' '\"'\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def f (x?' : Nat) := x?' '\"'\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def f := «x»' '\"'\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def c := '\\x27'\ndef s := '\"'\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def c := '\\u0027'\ndef s := '\"'\ntheorem t : False := by sorry\n", "LEAN4_SORRY"),
            ("def «unterminated\n", "LEAN4_UNTERMINATED_IDENT"),
            ("/- unclosed comment\n", "LEAN4_UNTERMINATED_COMMENT"),
            ('def s := "unclosed\n', "LEAN4_UNTERMINATED_STRING"),
            ('def s := r#"unclosed\n', "LEAN4_UNTERMINATED_STRING"),
            ('def s := s!"{ 1 + 2\n', "LEAN4_UNTERMINATED_STRING"),
            # Multi-flag & all Lean 4.15+ decide native escapes
            ("theorem t : 1 = 1 := by decide +native\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide -revert +native\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (revert := false) +native\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (revert := true) (native := true)\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide -zeta +native\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (native := true)\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (native := (true))\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide («native» := true)\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (other := 1) +native\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := cfg)\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { native := true })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { native := false, native := true })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { cfg with native := false })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (native :=   )\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config :=   )\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { native := false, 123 })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { a := { } })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { a := { })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { a := } })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { native := false junk })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { native := false)\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { a )\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := native := false })\n", "LEAN4_DECIDE_NATIVE"),
            ("theorem t : 1 = 1 := by decide (config := { native := false } })\n", "LEAN4_DECIDE_NATIVE"),
            ("def c : DecideConfig := { native := true }\n", "LEAN4_DECIDE_NATIVE"),
            ("def c := { native := true } : DecideConfig\n", "LEAN4_DECIDE_NATIVE"),
            ("def c := { native := true : DecideConfig }\n", "LEAN4_DECIDE_NATIVE"),
        ]
        for src, expected_code in attacks:
            res = audit.audit_lean4_source(src)
            self.assertFalse(res["clean"], f"Missed attack for {expected_code}: {src!r}")
            self.assertIn(expected_code, res["codes"], f"Wrong codes {res['codes']} for {src!r}")

        for edge in (
            "theorem a : 1 = 1 := by decide",
            "theorem b : 1 = 1 := by decide + 1",
            "theorem c : 1 = 1 := by decide (1",
            "theorem d : 1 = 1 := by decide (1)",
            "def x := 'a\n",
            "def y := '\\xZZ'\n",
            "def z := '\\uZZZZ'\n",
            "attribute [unclosed\n",
            "def r_id := r#1\n",
        ):
            res = audit.audit_lean4_source(edge)
            self.assertTrue(res["clean"], f"Unexpected finding on {edge!r}: {res}")

    def test_verify_lean_modules_live_and_failure_traps(self) -> None:
        st_res = audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=False)
        self.assertTrue(st_res["valid"])
        self.assertFalse(st_res["live_lean_checked"])
        self.assertEqual(st_res["kernel_theorems_verified"], 21)

        with self.assertRaises(oracle.S73bError):
            audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=1)  # type: ignore[arg-type]

        lean_bin = _find_lean_bin()
        self.assertIsNotNone(lean_bin, "Lean 4.36.0 binary must be available for live verification")
        live_res = audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
        self.assertTrue(live_res["valid"])
        self.assertTrue(live_res["live_lean_checked"])
        self.assertEqual(live_res["kernel_theorems_verified"], 21)
        self.assertEqual(live_res["banned_axioms_found"], 0)

        with mock.patch.dict(os.environ, {"LEAN_BIN": lean_bin}, clear=False):
            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--verify-lean", str(_THIS_DIR)]), 0)
                self.assertIn("LEAN_VALID 3 21", buf_out.getvalue())
            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--verify-lean", str(_THIS_DIR), "--json"]), 0)
                self.assertEqual(json.loads(buf_out.getvalue())["kernel_theorems_verified"], 21)

        with mock.patch.dict(os.environ, {"LEAN_BIN": ""}, clear=False):
            with mock.patch.object(shutil, "which", return_value=None):
                with self.assertRaises(oracle.S73bError) as c_nf:
                    audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=True)
                self.assertEqual(c_nf.exception.code, "LEAN_NOT_FOUND")

        with mock.patch.dict(os.environ, {"LEAN_BIN": lean_bin}, clear=False):
            env_res = audit.verify_lean_modules_live(_THIS_DIR, verify_live_lean=True)
            self.assertTrue(env_res["valid"])

        with mock.patch.object(subprocess, "run", side_effect=OSError("exec err")):
            with self.assertRaises(oracle.S73bError) as c_os:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(c_os.exception.code, "LEAN_EXEC_FAILED")
        with mock.patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired(["lean"], 120)):
            with self.assertRaises(oracle.S73bError) as c_to:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(c_to.exception.code, "LEAN_EXEC_FAILED")

        bad_rc = subprocess.CompletedProcess(["lean"], 1, stdout="", stderr="")
        with mock.patch.object(subprocess, "run", return_value=bad_rc):
            with self.assertRaises(oracle.S73bError) as c_rc:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(c_rc.exception.code, "LEAN_KERNEL_FAILED")
        bad_err = subprocess.CompletedProcess(["lean"], 0, stdout="", stderr="warning: bad")
        with mock.patch.object(subprocess, "run", return_value=bad_err):
            with self.assertRaises(oracle.S73bError) as c_se:
                audit.verify_lean_modules_live(_THIS_DIR, lean_bin=lean_bin, verify_live_lean=True)
            self.assertEqual(c_se.exception.code, "LEAN_STDERR_NONEMPTY")

        ns = "S73b.Test"
        exp = (("t1", ()), ("t2", ("Quot.sound", "propext")))
        good_out = (
            "'S73b.Test.t1' does not depend on any axioms\n"
            "'S73b.Test.t2' depends on axioms: [propext, Quot.sound]\n"
        )
        audit._verify_lean_stdout_axioms(good_out, ns, exp)
        for bad_out, code in (
            (good_out + "'S73b.Test.t1' does not depend on any axioms\n", "LEAN_DUPLICATE_AXIOM_LINE"),
            (
                "'S73b.Test.t1' depends on axioms: [Lean.ofReduceBool]\n"
                "'S73b.Test.t2' depends on axioms: [propext, Quot.sound]\n",
                "LEAN_BANNED_AXIOM",
            ),
            ("file.lean:1:0: warning: unused variable\n" + good_out, "LEAN_UNEXPECTED_STDOUT"),
            ("'S73b.Test.t1' does not depend on any axioms\n", "LEAN_THEOREM_SET_MISMATCH"),
            (
                "'S73b.Test.t1' depends on axioms: [propext]\n"
                "'S73b.Test.t2' depends on axioms: [propext, Quot.sound]\n",
                "LEAN_AXIOM_MISMATCH",
            ),
        ):
            with self.assertRaises(oracle.S73bError) as ctx:
                audit._verify_lean_stdout_axioms(bad_out, ns, exp)
            self.assertEqual(ctx.exception.code, code)

        with tempfile.TemporaryDirectory(prefix="s73b_live_neg_") as td:
            td_p = Path(td)
            warn_f = td_p / "Warn.lean"
            warn_f.write_text(
                "theorem warn_thm (unused_x : Int) : 1 = 1 := rfl\n#print axioms warn_thm\n",
                encoding="utf-8",
            )
            p_warn = subprocess.run([lean_bin, str(warn_f)], capture_output=True, text=True, check=False)
            self.assertEqual(p_warn.returncode, 0)
            self.assertIn("warning:", p_warn.stdout)
            with self.assertRaises(oracle.S73bError) as c_w:
                audit._verify_lean_stdout_axioms(p_warn.stdout, "", (("warn_thm", ()),))
            self.assertEqual(c_w.exception.code, "LEAN_UNEXPECTED_STDOUT")

            nd_f = td_p / "NativeDecide.lean"
            nd_f.write_text(
                "theorem nd_thm : 1 + 1 = 2 := by native_decide\n#print axioms nd_thm\n",
                encoding="utf-8",
            )
            p_nd = subprocess.run([lean_bin, str(nd_f)], capture_output=True, text=True, check=False)
            self.assertEqual(p_nd.returncode, 0)
            with self.assertRaises(oracle.S73bError) as c_nd:
                audit._verify_lean_stdout_axioms(p_nd.stdout, "", (("nd_thm", ()),))
            self.assertEqual(c_nd.exception.code, "LEAN_BANNED_AXIOM")

            ax_f = td_p / "CustomAx.lean"
            ax_f.write_text(
                "axiom customBogus : 1 = 1\ntheorem ax_thm : 1 = 1 := customBogus\n#print axioms ax_thm\n",
                encoding="utf-8",
            )
            p_ax = subprocess.run([lean_bin, str(ax_f)], capture_output=True, text=True, check=False)
            self.assertEqual(p_ax.returncode, 0)
            with self.assertRaises(oracle.S73bError) as c_ax:
                audit._verify_lean_stdout_axioms(p_ax.stdout, "", (("ax_thm", ()),))
            self.assertEqual(c_ax.exception.code, "LEAN_BANNED_AXIOM")

            for name in ("lean-toolchain", "lakefile.toml"):
                shutil.copy2(_THIS_DIR / name, td_p / name)
            shutil.copytree(_THIS_DIR / "S73bFormalTheory", td_p / "S73bFormalTheory")

            (td_p / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n", encoding="utf-8")
            with self.assertRaises(oracle.S73bError) as c_tc:
                audit.verify_lean_modules_live(td_p, verify_live_lean=False)
            self.assertEqual(c_tc.exception.code, "LEAN_TOOLCHAIN_SHA_MISMATCH")
            shutil.copy2(_THIS_DIR / "lean-toolchain", td_p / "lean-toolchain")

            (td_p / "lakefile.toml").write_text('name = "bad"\n', encoding="utf-8")
            with self.assertRaises(oracle.S73bError) as c_lf:
                audit.verify_lean_modules_live(td_p, verify_live_lean=False)
            self.assertEqual(c_lf.exception.code, "LAKEFILE_SHA_MISMATCH")
            shutil.copy2(_THIS_DIR / "lakefile.toml", td_p / "lakefile.toml")

            m0 = td_p / "S73bFormalTheory" / "HeavyTailLPTStarvation.lean"
            m0.write_text("-- tampered\n", encoding="utf-8")
            with self.assertRaises(oracle.S73bError) as c_ms:
                audit.verify_lean_modules_live(td_p, verify_live_lean=False)
            self.assertEqual(c_ms.exception.code, "LEAN_SHA256_MISMATCH")

            m0.write_bytes(b"\xff\xfe")
            with mock.patch.object(oracle, "check_digest_eq", return_value=True):
                with self.assertRaises(oracle.S73bError) as c_u8:
                    audit.verify_lean_modules_live(td_p, verify_live_lean=False)
                self.assertEqual(c_u8.exception.code, "LEAN_INVALID_UTF8")
                m0.write_text("theorem x : False := by sorry\n", encoding="utf-8")
                with self.assertRaises(oracle.S73bError) as c_sf:
                    audit.verify_lean_modules_live(td_p, verify_live_lean=False)
                self.assertEqual(c_sf.exception.code, "LEAN_STATIC_AUDIT_FAILED")
                m0.write_text("theorem v : False → True := fun h => False.elim h\n", encoding="utf-8")
                with self.assertRaises(oracle.S73bError) as c_sn:
                    audit.verify_lean_modules_live(td_p, verify_live_lean=False)
                self.assertEqual(c_sn.exception.code, "LEAN_STATIC_AUDIT_FAILED")


class TestFilesystemSecurityAndCLI(unittest.TestCase):
    def test_darwin_prefix_normalization_mocked(self) -> None:
        p_var = Path("/var/tmp/x")
        st_root_lnk = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        st_user_lnk = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 1000, 0, 0, 0, 0, 0))
        st_root_dir = os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 2, 0, 0, 0, 0, 0, 0))
        with mock.patch.object(sys, "platform", "linux"), mock.patch.object(
            os, "lstat", return_value=st_root_lnk
        ), mock.patch.object(os, "readlink", return_value="/private/var"):
            self.assertEqual(audit._normalize_darwin_system_prefix(p_var), p_var)

        with mock.patch.object(sys, "platform", "darwin"):
            with mock.patch.object(os, "lstat", return_value=st_root_lnk), mock.patch.object(
                os, "readlink", return_value="private/var"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("x/var/y")), Path("x/var/y"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var"), allow_exact=False), Path("/var"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/../etc/x")), Path("/var/../etc/x"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/usr/local/x")), Path("/usr/local/x"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/tmp/x")), Path("/private/var/tmp/x"))
                self.assertEqual(
                    audit._normalize_darwin_system_prefix(Path("/var"), allow_exact=True), Path("/private/var")
                )
            with mock.patch.object(os, "lstat", return_value=st_root_lnk), mock.patch.object(
                os, "readlink", return_value="/private/tmp"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a/b")), Path("/private/tmp/a/b"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("var/tmp/x")), Path("var/tmp/x"))
            with mock.patch.object(os, "lstat", return_value=st_user_lnk), mock.patch.object(
                os, "readlink", return_value="/private/var"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(p_var), p_var)
            with mock.patch.object(os, "lstat", return_value=st_root_dir), mock.patch.object(
                os, "readlink", return_value="/private/var"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(p_var), p_var)
            with mock.patch.object(os, "lstat", return_value=st_root_lnk), mock.patch.object(
                os, "readlink", return_value="/evil/var"
            ):
                self.assertEqual(audit._normalize_darwin_system_prefix(p_var), p_var)
            with mock.patch.object(os, "lstat", side_effect=OSError("no")):
                self.assertEqual(audit._normalize_darwin_system_prefix(p_var), p_var)

    def test_path_validation_fifo_symlink_hardlink_and_caps(self) -> None:
        for bad, code in (
            (123, "INVALID_PATH_TYPE"),
            ("a\x00b", "INVALID_PATH"),
            ("dir/", "INVALID_FILENAME"),
            ("\ud800", "INVALID_UTF8_STR"),
            (Path("a\x00b"), "INVALID_PATH"),
            (".", "INVALID_FILENAME"),
            ("..", "INVALID_FILENAME"),
            ("/", "INVALID_FILENAME"),
        ):
            with self.assertRaises(oracle.S73bError) as ctx:
                audit.read_regular_bytes_capped(bad)
            self.assertEqual(ctx.exception.code, code, f"bad={bad!r}")
        with self.assertRaises(oracle.S73bError):
            audit.write_bytes_atomic_0600("x", "not-bytes")  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory(prefix="s73b_fs_") as td:
            td_p = Path(td)
            big_data = b"A" * 70_000
            f_big = td_p / "big.bin"
            audit.write_bytes_atomic_0600(f_big, big_data)
            self.assertEqual(stat.S_IMODE(os.lstat(f_big).st_mode), 0o600)
            self.assertEqual(audit.read_regular_bytes_capped(f_big, max_bytes=70_000), big_data)
            with self.assertRaises(oracle.S73bError) as c_ts:
                audit.read_regular_bytes_capped(str(f_big) + "/")
            self.assertEqual(c_ts.exception.code, "INVALID_FILENAME")
            with self.assertRaises(oracle.S73bError) as c_cap:
                audit.read_regular_bytes_capped(f_big, max_bytes=69_999)
            self.assertEqual(c_cap.exception.code, "FILE_TOO_LARGE")

            audit.write_bytes_atomic_0600(str(f_big), b"small")
            self.assertEqual(audit.read_regular_bytes_capped(str(f_big)), b"small")

            with self.assertRaises(oracle.S73bError) as c_dotdot:
                audit.read_regular_bytes_capped(str(td_p / ".." / td_p.name / "big.bin"))
            self.assertEqual(c_dotdot.exception.code, "PARENT_TRAVERSAL_FORBIDDEN")

            with self.assertRaises(oracle.S73bError) as c_ndir:
                audit.read_regular_bytes_capped(str(f_big / "child"))
            self.assertEqual(c_ndir.exception.code, "NOT_A_DIRECTORY")

            with self.assertRaises(oracle.S73bError) as c_miss:
                audit.read_regular_bytes_capped(str(td_p / "nope" / "x"))
            self.assertEqual(c_miss.exception.code, "DIR_OPEN_FAILED")

            with self.assertRaises(oracle.S73bError) as c_mf:
                audit.read_regular_bytes_capped(str(td_p / "missing.json"))
            self.assertEqual(c_mf.exception.code, "FILE_UNREADABLE")

            fifo_p = td_p / "pipe.fifo"
            os.mkfifo(fifo_p, 0o600)
            with self.assertRaises(oracle.S73bError) as c_fr:
                audit.read_regular_bytes_capped(fifo_p)
            self.assertEqual(c_fr.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(oracle.S73bError) as c_fw:
                audit.write_bytes_atomic_0600(fifo_p, b"x")
            self.assertEqual(c_fw.exception.code, "NON_REGULAR_FILE")
            self.assertIn("refusing non-regular write target", c_fw.exception.message)

            sub_d = td_p / "sub"
            sub_d.mkdir()
            with self.assertRaises(oracle.S73bError):
                audit.read_regular_bytes_capped(sub_d)
            with self.assertRaises(oracle.S73bError):
                audit.write_bytes_atomic_0600(sub_d, b"x")

            sym_leaf = td_p / "sym.json"
            os.symlink(f_big, sym_leaf)
            with self.assertRaises(oracle.S73bError) as c_slr:
                audit.read_regular_bytes_capped(sym_leaf)
            self.assertEqual(c_slr.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(oracle.S73bError) as c_slw:
                audit.write_bytes_atomic_0600(sym_leaf, b"x")
            self.assertEqual(c_slw.exception.code, "SYMLINK_FORBIDDEN")

            sym_dir = td_p / "symdir"
            os.symlink(sub_d, sym_dir)
            with self.assertRaises(oracle.S73bError) as c_sdr:
                audit.read_regular_bytes_capped(sym_dir / "x")
            self.assertEqual(c_sdr.exception.code, "SYMLINK_FORBIDDEN")
            with self.assertRaises(oracle.S73bError) as c_sdw:
                audit.write_bytes_atomic_0600(sym_dir / "x", b"x")
            self.assertEqual(c_sdw.exception.code, "SYMLINK_FORBIDDEN")

            hl_target = td_p / "hl_a.bin"
            hl_alias = td_p / "hl_b.bin"
            audit.write_bytes_atomic_0600(hl_target, b"hl")
            os.link(hl_target, hl_alias)
            with self.assertRaises(oracle.S73bError) as c_hlr:
                audit.read_regular_bytes_capped(hl_target)
            self.assertEqual(c_hlr.exception.code, "HARDLINK_FORBIDDEN")
            with self.assertRaises(oracle.S73bError) as c_hlw:
                audit.write_bytes_atomic_0600(hl_target, b"new")
            self.assertEqual(c_hlw.exception.code, "HARDLINK_FORBIDDEN")
            self.assertIn("refusing hardlinked target", c_hlw.exception.message)

            ro_dir = td_p / "ro"
            ro_dir.mkdir(mode=0o700)
            os.chmod(ro_dir, 0o500)
            try:
                with self.assertRaises(oracle.S73bError) as c_ro:
                    audit.write_bytes_atomic_0600(ro_dir / "out.json", b"x")
                self.assertEqual(c_ro.exception.code, "WRITE_IO_ERROR")
            finally:
                os.chmod(ro_dir, 0o700)

            old_cwd = os.getcwd()
            try:
                os.chdir(sub_d)
                (sub_d / "rel.txt").write_bytes(b"rel")
                with mock.patch.dict(os.environ, {"PWD": str(sub_d)}, clear=False):
                    self.assertEqual(audit.read_regular_bytes_capped("./rel.txt"), b"rel")
                with mock.patch.object(os, "environ", {}):
                    self.assertEqual(audit.read_regular_bytes_capped("rel.txt"), b"rel")
                with mock.patch.object(os, "environ", {"PWD": "relative/pwd"}):
                    self.assertEqual(audit.read_regular_bytes_capped("rel.txt"), b"rel")
                with mock.patch.object(os, "environ", {"PWD": "."}):
                    self.assertEqual(audit.read_regular_bytes_capped("rel.txt"), b"rel")
                with mock.patch.dict(os.environ, {"PWD": str(td_p / "nonexistent_pwd")}, clear=False):
                    self.assertEqual(audit.read_regular_bytes_capped("rel.txt"), b"rel")
                with mock.patch.dict(os.environ, {"PWD": str(td_p)}, clear=False):
                    self.assertEqual(audit.read_regular_bytes_capped("rel.txt"), b"rel")
                with mock.patch.dict(os.environ, {"PWD": str(sym_dir)}, clear=False):
                    with self.assertRaises(oracle.S73bError) as c_pwd:
                        audit.read_regular_bytes_capped("rel.txt")
                    self.assertEqual(c_pwd.exception.code, "SYMLINK_FORBIDDEN")
                with mock.patch.object(os, "environ", {"PWD": "/tmp/a\x00b"}):
                    with self.assertRaises(oracle.S73bError) as c_pnul:
                        audit.read_regular_bytes_capped("rel.txt")
                    self.assertEqual(c_pnul.exception.code, "INVALID_PATH")
            finally:
                os.chdir(old_cwd)

    def test_mocked_toctou_and_io_fault_branches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s73b_mock_") as td:
            td_p = Path(td)
            f_ok = td_p / "ok.bin"
            audit.write_bytes_atomic_0600(f_ok, b"hello")

            with mock.patch.object(os, "open", side_effect=OSError(errno.EACCES, "denied")):
                with self.assertRaises(oracle.S73bError) as c_r:
                    audit._open_dir_chain_nofollow(("a",))
                self.assertEqual(c_r.exception.code, "DIR_OPEN_FAILED")

            real_open = os.open

            def open_enotdir(path, flags, *args, **kwargs):
                if path == "symcomp":
                    raise OSError(errno.ENOTDIR, "not dir")
                return real_open(path, flags, *args, **kwargs)

            st_sym = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 0, 0, 0, 0, 0, 0))
            with mock.patch.object(os, "open", side_effect=open_enotdir):
                with mock.patch.object(os, "lstat", return_value=st_sym):
                    with self.assertRaises(oracle.S73bError) as c_es:
                        audit._open_dir_chain_nofollow(("symcomp",))
                    self.assertEqual(c_es.exception.code, "SYMLINK_FORBIDDEN")
                with mock.patch.object(os, "lstat", side_effect=OSError("no")):
                    with self.assertRaises(oracle.S73bError) as c_en:
                        audit._open_dir_chain_nofollow(("symcomp",))
                    self.assertEqual(c_en.exception.code, "NOT_A_DIRECTORY")

            with mock.patch.dict(os.environ, {"PWD": str(td_p)}, clear=False):
                with mock.patch.object(os, "stat", side_effect=OSError("no dot")):
                    with self.assertRaises(oracle.S73bError) as c_dot:
                        audit._verify_pwd_not_symlink()
                    self.assertEqual(c_dot.exception.code, "CWD_UNAVAILABLE")
                st_same = os.stat(td_p)
                st_diff = os.stat_result((stat.S_IFDIR | 0o755, st_same.st_ino + 99, st_same.st_dev, 2, 0, 0, 0, 0, 0, 0))
                with mock.patch.object(os, "stat", return_value=st_same), mock.patch.object(
                    os, "fstat", return_value=st_diff
                ):
                    with self.assertRaises(oracle.S73bError) as c_pm:
                        audit._verify_pwd_not_symlink()
                    self.assertEqual(c_pm.exception.code, "PWD_SYMLINK_MISMATCH")
            with mock.patch.dict(os.environ, {"PWD": ""}, clear=False):
                with mock.patch.object(os, "getcwd", side_effect=OSError("deleted cwd")):
                    with self.assertRaises(oracle.S73bError) as c_gcw:
                        audit._open_parent_dir_fd(Path("rel.txt"))
                    self.assertEqual(c_gcw.exception.code, "CWD_UNAVAILABLE")

            def open_file_eloop(path, flags, *args, **kwargs):
                if path == "ok.bin":
                    raise OSError(errno.ELOOP, "loop")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=open_file_eloop):
                with self.assertRaises(oracle.S73bError) as c_el:
                    audit.read_regular_bytes_capped(f_ok)
                self.assertEqual(c_el.exception.code, "SYMLINK_FORBIDDEN")

            def open_file_eacces(path, flags, *args, **kwargs):
                if path == "ok.bin":
                    raise OSError(errno.EACCES, "denied")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=open_file_eacces):
                with self.assertRaises(oracle.S73bError) as c_ea:
                    audit.read_regular_bytes_capped(f_ok)
                self.assertEqual(c_ea.exception.code, "FILE_UNREADABLE")

            with mock.patch.object(os, "fstat", side_effect=OSError("fstat fail")):
                with self.assertRaises(oracle.S73bError) as c_fs:
                    audit.read_regular_bytes_capped(f_ok)
                self.assertEqual(c_fs.exception.code, "READ_IO_ERROR")

            st_real = os.lstat(f_ok)
            st_fifo = os.stat_result((stat.S_IFIFO | 0o600, st_real.st_ino, st_real.st_dev, 1, 0, 0, 0, 0, 0, 0))
            st_hl = os.stat_result((stat.S_IFREG | 0o600, st_real.st_ino, st_real.st_dev, 2, 0, 0, 5, 0, 0, 0))
            for fake_st, code in ((st_fifo, "NON_REGULAR_FILE"), (st_hl, "HARDLINK_FORBIDDEN")):
                with mock.patch.object(os, "fstat", return_value=fake_st):
                    with self.assertRaises(oracle.S73bError) as ctx:
                        audit.read_regular_bytes_capped(f_ok)
                    self.assertEqual(ctx.exception.code, code)

            with mock.patch.object(os, "read", side_effect=OSError("read fail")):
                with self.assertRaises(oracle.S73bError) as c_rd:
                    audit.read_regular_bytes_capped(f_ok)
                self.assertEqual(c_rd.exception.code, "READ_IO_ERROR")

            with mock.patch.object(os, "lstat", side_effect=OSError(errno.EIO, "io")):
                with self.assertRaises(oracle.S73bError) as c_ls:
                    audit.write_bytes_atomic_0600(td_p / "w.bin", b"x")
                self.assertEqual(c_ls.exception.code, "TARGET_UNSTATTABLE")

            calls = [0]

            def open_eexist_once(path, flags, *args, **kwargs):
                if isinstance(path, str) and path.startswith(".w1.bin.tmp."):
                    calls[0] += 1
                    if calls[0] == 1:
                        raise FileExistsError("exists")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=open_eexist_once):
                audit.write_bytes_atomic_0600(td_p / "w1.bin", b"ok")
            self.assertEqual(calls[0], 2)

            def open_eexist_always(path, flags, *args, **kwargs):
                if isinstance(path, str) and path.startswith(".w2.bin.tmp."):
                    raise FileExistsError("exists")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=open_eexist_always):
                with self.assertRaises(oracle.S73bError) as c_ex:
                    audit.write_bytes_atomic_0600(td_p / "w2.bin", b"ok")
                self.assertEqual(c_ex.exception.code, "TEMP_CREATE_FAILED")

            with mock.patch.object(os, "write", return_value=0):
                with self.assertRaises(oracle.S73bError) as c_sw:
                    audit.write_bytes_atomic_0600(td_p / "w3.bin", b"ok")
                self.assertEqual(c_sw.exception.code, "SHORT_WRITE")

            for bad_tmp in (st_fifo, st_hl):
                with mock.patch.object(os, "fstat", return_value=bad_tmp):
                    with self.assertRaises(oracle.S73bError) as c_tt:
                        audit.write_bytes_atomic_0600(td_p / "w4.bin", b"ok")
                    self.assertEqual(c_tt.exception.code, "TEMP_TAMPERED")

            for race_st, code in (
                (st_sym, "SYMLINK_FORBIDDEN"),
                (st_fifo, "NON_REGULAR_FILE"),
                (st_hl, "HARDLINK_FORBIDDEN"),
            ):
                lstat_seq = [FileNotFoundError(), race_st]

                def lstat_race(*args, **kwargs):
                    nxt = lstat_seq.pop(0)
                    if isinstance(nxt, BaseException):
                        raise nxt
                    return nxt

                with mock.patch.object(os, "lstat", side_effect=lstat_race):
                    with self.assertRaises(oracle.S73bError) as c_rc:
                        audit.write_bytes_atomic_0600(td_p / "w5.bin", b"ok")
                    self.assertEqual(c_rc.exception.code, code)

            with mock.patch.object(os, "replace", side_effect=OSError("rep fail")), mock.patch.object(
                os, "unlink", side_effect=OSError("unlink fail")
            ):
                with self.assertRaises(oracle.S73bError) as c_rep:
                    audit.write_bytes_atomic_0600(td_p / "w6.bin", b"ok")
                self.assertEqual(c_rep.exception.code, "WRITE_IO_ERROR")

    def test_cli_modes_and_error_traps(self) -> None:
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with mock.patch.object(sys, "stdout", buf_out), mock.patch.object(sys, "stderr", buf_err):
            self.assertEqual(audit.main([]), 0)
            self.assertIn("OK ", buf_out.getvalue())

        buf_out = io.StringIO()
        with mock.patch.object(sys, "stdout", buf_out):
            self.assertEqual(audit.main(["--json"]), 0)
            self.assertTrue(json.loads(buf_out.getvalue())["valid"])

        buf_out = io.StringIO()
        with mock.patch.object(sys, "stdout", buf_out):
            self.assertEqual(audit.main(["--standalone"]), 0)
            st_cert = oracle.parse_strict_json_text(buf_out.getvalue())
            self.assertTrue(audit.verify_certificate_dict(st_cert, full_replay=True)["valid"])

        buf_out = io.StringIO()
        with mock.patch.object(sys, "argv", ["s73b_lean4_audit.py"]), mock.patch.object(sys, "stdout", buf_out):
            self.assertEqual(audit.main(None), 0)

        with tempfile.TemporaryDirectory(prefix="s73b_cli_") as td:
            out_p = Path(td) / "cert.json"
            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--write", str(out_p)]), 0)
                self.assertIn("WROTE ", buf_out.getvalue())
            self.assertEqual(stat.S_IMODE(os.lstat(out_p).st_mode), 0o600)

            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--write", str(out_p), "--json"]), 0)
                self.assertTrue(json.loads(buf_out.getvalue())["written"])

            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--check", str(out_p)]), 0)
                self.assertIn("VALID ", buf_out.getvalue())
            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--check", str(out_p), "--json"]), 0)
                self.assertTrue(json.loads(buf_out.getvalue())["valid"])
            audit.write_bytes_atomic_0600(out_p, out_p.read_bytes().rstrip(b"\n"))
            buf_out = io.StringIO()
            with mock.patch.object(sys, "stdout", buf_out):
                self.assertEqual(audit.main(["--check", str(out_p)]), 0)

            non_canon = Path(td) / "non_canon.json"
            audit.write_bytes_atomic_0600(non_canon, b" " + out_p.read_bytes())
            buf_err = io.StringIO()
            with mock.patch.object(sys, "stderr", buf_err):
                self.assertEqual(audit.main(["--check", str(non_canon)]), 2)
                self.assertIn("NON_CANONICAL_BYTES", buf_err.getvalue())

            bad_utf8 = Path(td) / "bad_utf8.json"
            audit.write_bytes_atomic_0600(bad_utf8, b"\xff\xfe")
            buf_err = io.StringIO()
            with mock.patch.object(sys, "stderr", buf_err):
                self.assertEqual(audit.main(["--check", str(bad_utf8)]), 2)
                self.assertIn("INVALID_UTF8", buf_err.getvalue())

        bad_argvs: list[tuple[object, str]] = [
            (123, "CLI_INVALID_ARGV"),
            (["--unknown"], "CLI_UNKNOWN_ARG"),
            (["--unknown", "val"], "CLI_UNKNOWN_ARG"),
            (["--standalone", "--standalone"], "CLI_CONFLICT"),
            (["--standalone", "--write", "x"], "CLI_CONFLICT"),
            (["--json", "--json"], "CLI_DUPLICATE"),
            (["--write"], "CLI_MISSING_ARG"),
            (["--write", "--json"], "CLI_INVALID_PATH"),
            (["--check", ""], "CLI_INVALID_PATH"),
            (["--verify-lean", "-bad"], "CLI_INVALID_PATH"),
        ]
        for bad, code in bad_argvs:
            buf_err = io.StringIO()
            with mock.patch.object(sys, "stderr", buf_err):
                self.assertEqual(audit.main(bad), 2, f"bad argv={bad!r}")  # type: ignore[arg-type]
                self.assertIn(f"ERROR {code}:", buf_err.getvalue())


if __name__ == "__main__":
    failfast = "-f" in sys.argv or "--failfast" in sys.argv
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2, failfast=failfast).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    raise SystemExit(0)
