#!/usr/bin/env python3
"""Comprehensive R5 Test & Mutation-Killing Suite for the Opposite-Sign E_8 vs. A_16 Hadamard Law.

Covers:
  - Pure-Z/Q and Q(sqrt(2)) exact arithmetic, exact E_8 & A_16 closest-point decoders,
    Pillars 1-4 invariants, exported helper validation, and zero-overhead per-layer selector
  - Exhaustive recursive leaf/key/length tamper sweep under full_replay=False & direct secondary_audit_payload
  - Darwin /private/* prefix normalization (mocked), stale vs symlinked $PWD parity checks
  - Atomic dir_fd I/O: scratch os.mkfifo (zero null-device writes), spy os.open pre-open checks,
    chmod 500 OSError wrapping, TOCTOU swap, 16T/80T 0o600 bursts, >= 500 hardlink flips,
    Linux-guarded >= 550 renameat2 swaps
  - Strict manual CLI fail-closed tests (--write --json trap, main(123), canonical-byte check)
  - Live Lean 4.36.0 kernel + #print axioms verification (12/12 theorems, 0 banned axioms)
  - Live read-only exact-Fraction verification against empirical anchor 42ad90b1... (2 models, 392 layers)

100% pure Python 3 stdlib, zero bare asserts, zero null-device writes, zero mathematical RNG.
"""
from __future__ import annotations

import concurrent.futures
import copy
import ctypes
import errno
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import e8_vs_a16_hadamard_audit as audit  # noqa: E402
import e8_vs_a16_hadamard_oracle as oracle  # noqa: E402

_EMPIRICAL_ANCHOR_PATH = (
    _THIS_DIR.parent / "track-s49b-rtx3080-live-llm-a16-eval" / "rtx3080_live_llm_empirical_results.json"
)
_LEAN_COMPANION_PATH = _THIS_DIR / oracle.LEAN_COMPANION_FILENAME
_CERT_PATH = _THIS_DIR / "e8_vs_a16_hadamard_certificate.json"


class _SubInt(int):
    pass


class _SubStr(str):
    pass


class _SubDict(dict):
    pass


class _SubList(list):
    pass


class _SubPath(type(Path())):
    pass


def _resign_cert(payload: dict) -> dict:
    p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    cert_pre = {
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": p_sha,
        "payload": payload,
    }
    c_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(cert_pre))
    return {
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
        "payload": payload,
    }


def _collect_leaf_paths(obj, prefix=()):
    out = []
    if type(obj) is dict:
        for k, v in obj.items():
            out.extend(_collect_leaf_paths(v, prefix + (k,)))
    elif type(obj) is list:
        for i, v in enumerate(obj):
            out.extend(_collect_leaf_paths(v, prefix + (i,)))
    else:
        out.append((prefix, obj))
    return out


def _collect_container_paths(obj, prefix=()):
    dicts = []
    lists = []
    if type(obj) is dict:
        dicts.append(prefix)
        for k, v in obj.items():
            d2, l2 = _collect_container_paths(v, prefix + (k,))
            dicts.extend(d2)
            lists.extend(l2)
    elif type(obj) is list:
        lists.append(prefix)
        for i, v in enumerate(obj):
            d2, l2 = _collect_container_paths(v, prefix + (i,))
            dicts.extend(d2)
            lists.extend(l2)
    return dicts, lists


def _set_at_path(root, path, val):
    cur = root
    for seg in path[:-1]:
        cur = cur[seg]
    cur[path[-1]] = val


def _get_at_path(root, path):
    cur = root
    for seg in path:
        cur = cur[seg]
    return cur


class TestOraclePrimitivesAndQSqrt2(unittest.TestCase):
    def _assert_err(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.HadamardLawError) as cm:
            fn(*args, **kwargs)
        self.assertEqual(cm.exception.code, code)

    def test_exact_int_bool_str_sha_and_rationals(self) -> None:
        self.assertEqual(oracle.require_exact_int(0, "x"), 0)
        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        self.assertEqual(oracle.require_exact_int(max_ok, "x"), max_ok)
        self.assertEqual(oracle.require_exact_int(5, "x", min_val=5, max_val=5), 5)
        for bad in (True, False, "5", _SubInt(5)):
            self._assert_err("INVALID_INT", oracle.require_exact_int, bad, "x")
        self._assert_err("INT_BIT_LIMIT", oracle.require_exact_int, 1 << oracle.MAX_INT_BITS, "x")
        self._assert_err("INT_RANGE", oracle.require_exact_int, 4, "x", min_val=5)
        self._assert_err("INT_RANGE", oracle.require_exact_int, 6, "x", max_val=5)

        self.assertIs(oracle.require_exact_bool(True, "b"), True)
        self.assertIs(oracle.require_exact_bool(False, "b"), False)
        for bad_b in (1, 0, "true", None):
            self._assert_err("INVALID_BOOL", oracle.require_exact_bool, bad_b, "b")

        self.assertEqual(oracle.require_exact_str("ok", "s"), "ok")
        self._assert_err("INVALID_STR", oracle.require_exact_str, 123, "s")
        self._assert_err("INVALID_STR", oracle.require_exact_str, _SubStr("ok"), "s")
        self._assert_err("NUL_FORBIDDEN", oracle.require_exact_str, "a\x00b", "s")
        self._assert_err("INVALID_UTF8_SURROGATE", oracle.require_exact_str, "bad\ud800surr", "s")

        good_sha = "0" * 64
        self.assertEqual(oracle.require_sha256_hex(good_sha, "h"), good_sha)
        self.assertTrue(oracle.check_digest_eq(good_sha, good_sha))
        self.assertFalse(oracle.check_digest_eq(good_sha, "f" * 64))
        for bad_h in ("0" * 63, "0" * 65, "A" * 64, "g" * 64):
            self._assert_err("INVALID_SHA256", oracle.require_sha256_hex, bad_h, "h")
        self._assert_err("INVALID_BYTES", oracle.sha256_bytes, "not-bytes")

        self.assertEqual(oracle.format_exact_rational(Fraction(7, 1)), "7")
        self.assertEqual(oracle.format_exact_rational(Fraction(-3, 4)), "-3/4")
        self._assert_err("INVALID_FRACTION", oracle.format_exact_rational, 7)

        self.assertEqual(oracle.parse_exact_rational("0", "q"), Fraction(0))
        self.assertEqual(oracle.parse_exact_rational("-15/4", "q"), Fraction(-15, 4))
        # Exact MAX_INT_BITS boundary (3321 bits, 1000 digits) accepted; 3322 bits rejected
        self.assertEqual(oracle.parse_exact_rational(str(max_ok), "q"), Fraction(max_ok, 1))
        self._assert_err("INT_BIT_LIMIT", oracle.parse_exact_rational, str(1 << oracle.MAX_INT_BITS), "q")
        for bad_q in (
            "", "-", "-0", "01", "00", "+1", "1a", "1/2/3", "1/-2", "1/0", "1/1", "2/4", "-2/4", "0/2",
            "1" + "0" * 1000,
        ):
            self._assert_err("INVALID_RATIONAL", oracle.parse_exact_rational, bad_q, "q")

    def test_decimal_parser_and_strict_json(self) -> None:
        self.assertEqual(oracle.parse_decimal_str_as_fraction("0.071621153"), Fraction(71621153, 10 ** 9))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("-0.25"), Fraction(-1, 4))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("1.023e-06"), Fraction(1023, 10 ** 9))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("1.5E+2"), Fraction(150, 1))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("42"), Fraction(42, 1))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("1e0"), Fraction(1, 1))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("1e200"), Fraction(10 ** 200, 1))
        self.assertEqual(oracle.parse_decimal_str_as_fraction("1e-200"), Fraction(1, 10 ** 200))
        for bad_d in (
            "", "-", "1" * (oracle.MAX_INT_DIGITS + 1), "1e2e3", "1e", "1e+", "1e-", "1e12345", "1e00001",
            "1ea", "1e201", "1e-201", "1.2.3", ".5", "5.", "1a", ".e1", "e1",
        ):
            self._assert_err("INVALID_DECIMAL", oracle.parse_decimal_str_as_fraction, bad_d)

        max_ok = (1 << oracle.MAX_INT_BITS) - 1
        ok_str_json = json.dumps({"s": 'x\\"\\\\' + "9" * 1005, "n": max_ok, "flag": True, "empty": None})
        parsed = oracle.parse_strict_json_text(ok_str_json)
        self.assertEqual(parsed["n"], max_ok)
        self.assertIsNone(parsed["empty"])
        self._assert_err("INT_DIGIT_LIMIT", oracle._precheck_json_int_digits, "0" * 1001)
        self._assert_err("INT_DIGIT_LIMIT", oracle.parse_strict_json_text, '{"s":"ok\\\\","n":' + "1" * 1001 + "}")
        self._assert_err("FLOAT_FORBIDDEN", oracle.parse_strict_json_text, '{"a":1.5}')
        for bad_c in ('{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'):
            self._assert_err("NON_FINITE_FORBIDDEN", oracle.parse_strict_json_text, bad_c)
        self._assert_err("DUPLICATE_KEY", oracle.parse_strict_json_text, '{"a":1,"\\u0061":2}')
        self._assert_err("INVALID_JSON", oracle.parse_strict_json_text, "{bad")
        self._assert_err("JSON_DEPTH_LIMIT", oracle.parse_strict_json_text, "[" * 33 + "1" + "]" * 33)

        d32: object = max_ok
        for _ in range(oracle.MAX_JSON_DEPTH):
            d32 = [d32]
        oracle._check_json_value_strict(d32)
        self._assert_err("JSON_DEPTH_LIMIT", oracle._check_json_value_strict, [d32])
        self._assert_err("INT_BIT_LIMIT", oracle._check_json_value_strict, {"k": 1 << oracle.MAX_INT_BITS})
        self._assert_err("INVALID_STR", oracle._check_json_value_strict, {1: "badkey"})
        for bad_obj in ({"k": _SubInt(1)}, _SubDict({"k": 1}), _SubList([1]), (1, 2)):
            self._assert_err("INVALID_JSON_TYPE", oracle._check_json_value_strict, bad_obj)

    def test_qsqrt2_full_field_and_parser(self) -> None:
        self._assert_err("INVALID_QSQRT2_COEFF", oracle.QSqrt2, "bad", 0)
        self._assert_err("INVALID_QSQRT2_COEFF", oracle.QSqrt2, 0, "bad")
        self._assert_err("INVALID_QSQRT2", oracle.format_qsqrt2, 1)

        x = oracle.QSqrt2(Fraction(3, 2), Fraction(1, 4))
        y = oracle.QSqrt2(Fraction(1, 3), Fraction(-2, 5))
        self.assertEqual(x + y, oracle.QSqrt2(Fraction(11, 6), Fraction(-3, 20)))
        self.assertEqual(x - y, oracle.QSqrt2(Fraction(7, 6), Fraction(13, 20)))
        self.assertEqual(-x, oracle.QSqrt2(Fraction(-3, 2), Fraction(-1, 4)))
        prod = x * y
        self.assertEqual(prod / y, x)
        self.assertNotEqual(x, y)
        self.assertNotEqual(x, 1)
        self.assertNotEqual(oracle.QSqrt2(1, 2), oracle.QSqrt2(1, 3))
        self.assertNotEqual(oracle.QSqrt2(1, 2), oracle.QSqrt2(2, 2))
        self._assert_err("ZERO_DIVISION", lambda: x / oracle.Q0)

        sign_cases = [
            ((0, 0), 0), ((1, 0), 1), ((-1, 0), -1),
            ((0, 1), 1), ((1, 1), 1), ((-1, 1), 1), ((-2, 1), -1),
            ((0, -1), -1), ((-1, -1), -1), ((2, -1), 1), ((1, -1), -1),
        ]
        for (a, b), exp in sign_cases:
            self.assertEqual(oracle.QSqrt2(a, b).sign(), exp)

        self.assertTrue(oracle.QSqrt2(1, 0) < oracle.QSqrt2(0, 1))
        self.assertFalse(oracle.QSqrt2(0, 1) < oracle.QSqrt2(1, 0))
        self.assertTrue(oracle.QSqrt2(0, 1) <= oracle.QSqrt2(0, 1))
        self.assertFalse(oracle.QSqrt2(2, 0) <= oracle.QSqrt2(0, 1))

        self.assertEqual(oracle.QSqrt2(Fraction(3, 2), 0).floor(), 1)
        self.assertEqual(oracle.QSqrt2(Fraction(-3, 2), 0).floor(), -2)
        self.assertEqual(oracle.QSqrt2(0, 1).floor(), 1)
        self.assertEqual(oracle.QSqrt2(0, -1).floor(), -2)
        self.assertEqual(oracle.QSqrt2(2, -1).floor(), 0)
        self.assertEqual(oracle.QSqrt2(1, -1).floor(), -1)
        self.assertEqual(oracle.QSqrt2(0, 1).nearest_int(), 1)
        self.assertEqual(oracle.QSqrt2(0, Fraction(3, 4)).nearest_int(), 1)
        self.assertEqual(oracle.QSqrt2(0, -1).nearest_int(), -1)

        for qv, txt in [
            (oracle.QSqrt2(Fraction(1, 4), 0), "1/4"),
            (oracle.QSqrt2(0, 3), "3*sqrt2"),
            (oracle.QSqrt2(0, -3), "-3*sqrt2"),
            (oracle.QSqrt2(1, Fraction(3, 2)), "1 + 3/2*sqrt2"),
            (oracle.QSqrt2(Fraction(41, 4), -6), "41/4 - 6*sqrt2"),
        ]:
            self.assertEqual(oracle.format_qsqrt2(qv), txt)
            self.assertEqual(oracle.parse_qsqrt2(txt, "qv"), qv)
        for bad_qs in (
            "1 + 2*sqrt2 + 3*sqrt2",
            "1 + 2*sqrt2x",
            "1 + -2*sqrt2",
            "1 + 0*sqrt2",
            "1 - 2*sqrt2 - 3*sqrt2",
            "1 - 2*sqrt2x",
            "1 - -2*sqrt2",
            "1 - 0*sqrt2",
            "2*sqrt2x",
            "0*sqrt2",
        ):
            self._assert_err("INVALID_QSQRT2", oracle.parse_qsqrt2, bad_qs, "qv")

    def test_hadamard_decoders_helpers_and_selector(self) -> None:
        self._assert_err("INT_RANGE", oracle.hadamard_entry, -1, 0)
        self._assert_err("INVALID_BITVEC", oracle.hamming16_syndrome, (0,) * 15)
        self._assert_err("INVALID_BITVEC", oracle.hamming16_syndrome, "0" * 16)
        self._assert_err("INVALID_VECTOR", oracle.decode_e8_exact, [oracle.Q0] * 8)
        self._assert_err("INVALID_VECTOR_LEN", oracle.decode_e8_exact, (oracle.Q0,) * 7)
        self._assert_err("INVALID_VECTOR_ELEM", oracle.decode_e8_exact, (0,) * 8)
        self._assert_err("INVALID_VECTOR_LEN", oracle.decode_a16_exact, (oracle.Q0,) * 15)
        self._assert_err("INVALID_BLOCK_SIZE", oracle.block_hadamard_apply, (oracle.Q0,) * 16, 3)
        self._assert_err("INVALID_VECTOR_LEN", oracle.block_hadamard_apply, (), 2)
        self._assert_err("INVALID_VECTOR_LEN", oracle.block_hadamard_apply, (oracle.Q0,) * 3, 2)
        self._assert_err("INVALID_LATTICE", oracle.quantize_vector16, (oracle.Q0,) * 16, "D4", 1)
        self._assert_err("INVALID_BLOCK_SIZE", oracle.quantize_vector16, (oracle.Q0,) * 16, "E8", 16)

        self.assertEqual(len(oracle.C16_CODEWORDS), 2048)
        self.assertEqual(len(set(oracle.C16_CODEWORDS)), 2048)
        for cw in oracle.C16_CODEWORDS:
            bits = tuple((cw >> j) & 1 for j in range(16))
            self.assertTrue(oracle.in_code_c16(bits))
        self.assertFalse(oracle.in_code_c16((1,) + (0,) * 15))
        self.assertFalse(oracle.in_code_c16((1, 1) + (0,) * 14))
        self.assertTrue(oracle.in_code_c16((1, 1, 1, 1) + (0,) * 12))

        vec = tuple(oracle.QSqrt2(Fraction(j - 7, 4), Fraction(j % 3 - 1, 2)) for j in range(16))
        for b in (1, 2, 4, 8, 16):
            self.assertEqual(oracle.block_hadamard_apply(oracle.block_hadamard_apply(vec, b), b), vec)

        self.assertEqual(
            oracle._decode_d8_exact((oracle.QSqrt2(Fraction(7, 5), 0),) + (oracle.Q0,) * 7),
            (oracle.QSqrt2(2, 0),) + (oracle.Q0,) * 7,
        )
        self.assertEqual(
            oracle._decode_d8_exact((oracle.Q0, oracle.Q0, oracle.Q0, oracle.QSqrt2(Fraction(3, 5), 0)) + (oracle.Q0,) * 4),
            (oracle.Q0,) * 8,
        )
        # Mixed parity with sum == 4 (divisible by 4) specifically kills parity check guard in _in_doubled_e8
        self.assertFalse(oracle._in_doubled_e8((1, 1, 1, 1, 0, 0, 0, 0)))
        self.assertFalse(oracle._in_doubled_e8((2, 0, 0, 0, 0, 0, 0, 0)))
        self.assertTrue(oracle._in_doubled_e8((4, 0, 0, 0, 0, 0, 0, 0)))
        self.assertTrue(oracle._is_perfect_square_int(16))
        self.assertFalse(oracle._is_perfect_square_int(8))

        half_in = tuple(oracle.QSqrt2(Fraction(1, 2), 0) for _ in range(8))
        self.assertEqual(oracle.decode_e8_exact(half_in), half_in)
        self.assertEqual(
            oracle.decode_e8_exact((oracle.QSqrt2(Fraction(3, 2), 0),) + (oracle.Q0,) * 7),
            (oracle.QSqrt2(2, 0),) + (oracle.Q0,) * 7,
        )

        self.assertTrue(oracle._int_qsqrt2_lt(1, 0, 2, 0))
        self.assertFalse(oracle._int_qsqrt2_lt(2, 0, 1, 0))
        self.assertFalse(oracle._int_qsqrt2_lt(10, 2, 0, 1))
        self.assertFalse(oracle._int_qsqrt2_lt(0, 2, 0, 1))
        self.assertTrue(oracle._int_qsqrt2_lt(-2, 2, 0, 1))
        self.assertFalse(oracle._int_qsqrt2_lt(-1, 2, 0, 1))
        self.assertTrue(oracle._int_qsqrt2_lt(-5, 1, 0, 2))
        self.assertTrue(oracle._int_qsqrt2_lt(0, 1, 0, 2))
        self.assertTrue(oracle._int_qsqrt2_lt(1, 1, 0, 2))
        self.assertFalse(oracle._int_qsqrt2_lt(2, 1, 0, 2))

        self.assertEqual(
            oracle.decode_a16_exact((oracle.QSqrt2(Fraction(3, 2), 0),) + (oracle.Q0,) * 15),
            (oracle.QSqrt2(2, 0),) + (oracle.Q0,) * 15,
        )
        w4 = tuple(oracle.QSqrt2(Fraction(3, 4), 0) if j < 4 else oracle.Q0 for j in range(16))
        self.assertEqual(
            oracle.decode_a16_exact(w4),
            tuple(oracle.Q1 if j < 4 else oracle.Q0 for j in range(16)),
        )

        self.assertEqual(oracle.select_layer_arm("E8", "v_proj"), "E8_H8")
        self.assertEqual(oracle.select_layer_arm("E8", "k_proj", delta_kl_q=Fraction(-10)), "E8_H8")
        self.assertEqual(oracle.select_layer_arm("A16", "v_proj"), "A16_axis")
        self.assertEqual(oracle.select_layer_arm("A16", "k_proj"), "A16_candidate_probe")
        self.assertEqual(oracle.select_layer_arm("A16", "o_proj"), "A16_candidate_probe")
        self.assertEqual(oracle.select_layer_arm("A16", "k_proj", delta_kl_q=Fraction(-1), margin_q=Fraction(1)), "A16_axis")
        self.assertEqual(oracle.select_layer_arm("A16", "k_proj", delta_kl_q=Fraction(-2), margin_q=Fraction(1)), "A16_H16")
        self._assert_err("INVALID_LATTICE", oracle.select_layer_arm, "D4", "v_proj")
        self._assert_err("INVALID_PROJECTION", oracle.select_layer_arm, "A16", "bad_proj")
        self._assert_err("NUL_FORBIDDEN", oracle.select_layer_arm, "A16\x00", "v_proj")
        self._assert_err("INVALID_UTF8_SURROGATE", oracle.select_layer_arm, "A16", "v_proj\ud800")
        self._assert_err("INVALID_MARGIN", oracle.select_layer_arm, "A16", "v_proj", None, Fraction(-1))
        self._assert_err("INVALID_MARGIN", oracle.select_layer_arm, "A16", "v_proj", None, 0)
        self._assert_err("INVALID_DELTA_KL", oracle.select_layer_arm, "A16", "v_proj", 1, Fraction(0))


class TestCertificateTamperAndPillars(unittest.TestCase):
    def _assert_err(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.HadamardLawError) as cm:
            fn(*args, **kwargs)
        self.assertEqual(cm.exception.code, code)

    def test_certificate_builds_compact_and_passes_both_modes(self) -> None:
        cert = oracle.build_certificate()
        raw = oracle.canonical_json_bytes(cert)
        self.assertLess(len(raw), oracle.TARGET_MAX_CERT_BYTES)
        self.assertTrue(cert["payload"]["summary"]["all_pillars_verified"])
        res1 = audit.verify_certificate_dict(cert, full_replay=True)
        res0 = audit.verify_certificate_dict(cert, full_replay=False)
        self.assertTrue(res1["valid"] and res1["full_replay"] and res1["secondary_audit_passed"])
        self.assertTrue(res0["valid"] and not res0["full_replay"])

        # Strict full_replay exact bool check
        self._assert_err("INVALID_BOOL", audit.verify_certificate_dict, cert, full_replay=0)
        self._assert_err("INVALID_BOOL", audit.verify_certificate_dict, cert, full_replay=1)

        c_sch = copy.deepcopy(cert)
        c_sch["schema_version"] = "bad"
        self._assert_err("SCHEMA_MISMATCH", audit.verify_certificate_dict, c_sch, full_replay=False)
        c_trk = copy.deepcopy(cert)
        c_trk["track_id"] = "bad"
        self._assert_err("TRACK_ID_MISMATCH", audit.verify_certificate_dict, c_trk, full_replay=False)
        c_psha = copy.deepcopy(cert)
        c_psha["payload_sha256"] = "0" * 64
        self._assert_err("PAYLOAD_SHA256_MISMATCH", audit.verify_certificate_dict, c_psha, full_replay=False)
        c_csha = copy.deepcopy(cert)
        c_csha["certificate_sha256"] = "0" * 64
        self._assert_err("CERTIFICATE_SHA256_MISMATCH", audit.verify_certificate_dict, c_csha, full_replay=False)
        with mock.patch.object(oracle, "build_certificate", return_value={"certificate_sha256": "f" * 64}):
            self._assert_err("FULL_REPLAY_MISMATCH", audit.verify_certificate_dict, cert, full_replay=True)

    def test_exhaustive_leaf_and_container_tamper_under_fr0_and_direct(self) -> None:
        cert = oracle.build_certificate()
        base_payload = cert["payload"]
        leaves = _collect_leaf_paths(base_payload)
        self.assertGreater(len(leaves), 110)

        for path, orig in leaves:
            cands = []
            if type(orig) is bool:
                cands = [not orig, 1 if orig else 0]
            elif type(orig) is int:
                cands = [orig + 1, orig - 1, True]
            elif type(orig) is str:
                if len(orig) == 64 and all(ch in "0123456789abcdef" for ch in orig):
                    cands = ["f" * 64 if orig != "f" * 64 else "0" * 64, "bad_sha"]
                elif "*sqrt2" in orig:
                    cands = ["0", "1/4", "bad*sqrt2"]
                else:
                    alt_q = "1/3" if orig != "1/3" else "1/5"
                    cands = [alt_q, "tampered_value", "1/0"]
            for cand in cands:
                p_mut = copy.deepcopy(base_payload)
                _set_at_path(p_mut, path, cand)
                with self.assertRaises(oracle.HadamardLawError, msg=f"direct accepted {path}={cand!r}"):
                    audit.secondary_audit_payload(p_mut)
                c_mut = _resign_cert(p_mut)
                with self.assertRaises(oracle.HadamardLawError, msg=f"FR0 accepted {path}={cand!r}"):
                    audit.verify_certificate_dict(c_mut, full_replay=False)

        # Coordinated a16_axis_mse == a16_h16_mse != 5/2 kills a16_ax_mse != 5/2 subcondition
        p_mse = copy.deepcopy(base_payload)
        mw_c = p_mse["pillar2_exact_quadratic_kl_witnesses_and_regime_flip"]["matrix_witness_zero_mse_gap"]
        mw_c["a16_axis_mse"] = "1/3"
        mw_c["a16_h16_mse"] = "1/3"
        self._assert_err("P2_MATRIX_MISMATCH", audit.secondary_audit_payload, p_mse)

        # Coordinated per-model + combined count tamper kills individual P3_MODEL_MISMATCH subconditions
        for m_key, c_key in [
            ("total_layers", "combined_total_layers"),
            ("regime1_count", "combined_regime1_count"),
            ("regime2_count", "combined_regime2_count"),
            ("bh2_beats_h16_count", "combined_bh2_beats_h16_count"),
        ]:
            p_coord = copy.deepcopy(base_payload)
            p3_c = p_coord["pillar3_empirical_llm_bridge_and_selector"]
            p3_c["models"]["Qwen/Qwen2.5-0.5B"][m_key] += 1
            p3_c[c_key] += 1
            self._assert_err("P3_MODEL_MISMATCH", audit.secondary_audit_payload, p_coord)

        # Dict masquerading as list kills type(x) is not list independently of len / iteration equality
        for lp, exp_code in [
            (("pillar1_lattice_ray_anisotropy_and_hadamard", "a16_block_h_spectrum"), "P1_A16_SPECTRUM_SHAPE"),
            (("pillar1_lattice_ray_anisotropy_and_hadamard", "e8_block_h_spectrum"), "P1_E8_SPECTRUM_SHAPE"),
            (("pillar4_lean4_companion_and_cs13_scope", "lean_companion_theorems"), "P4_THEOREMS_MISMATCH"),
            (("pillar4_lean4_companion_and_cs13_scope", "prior_literature_n0"), "P4_LIT_MISMATCH"),
        ]:
            orig_list = _get_at_path(base_payload, lp)
            p_dict_as_list = copy.deepcopy(base_payload)
            if all(type(x) is str for x in orig_list):
                _set_at_path(p_dict_as_list, lp, {x: 0 for x in orig_list})
            else:
                _set_at_path(p_dict_as_list, lp, {f"k{i}": i for i in range(len(orig_list))})
            self._assert_err(exp_code, audit.secondary_audit_payload, p_dict_as_list)

        dicts, lists = _collect_container_paths(base_payload)
        for dp in dicts:
            p_ex = copy.deepcopy(base_payload)
            _get_at_path(p_ex, dp)["__extra__"] = 1
            self._assert_err("KEYSET_MISMATCH", audit.secondary_audit_payload, p_ex)
            self._assert_err("KEYSET_MISMATCH", audit.verify_certificate_dict, _resign_cert(p_ex), full_replay=False)
            if dp:
                p_wt = copy.deepcopy(base_payload)
                _set_at_path(p_wt, dp, [])
                self._assert_err("INVALID_OBJECT", audit.secondary_audit_payload, p_wt)

        for lp in lists:
            orig_len = len(_get_at_path(base_payload, lp))
            p_short = copy.deepcopy(base_payload)
            _get_at_path(p_short, lp).pop()
            with self.assertRaises(oracle.HadamardLawError):
                audit.secondary_audit_payload(p_short)
            p_long = copy.deepcopy(base_payload)
            lst_ref = _get_at_path(p_long, lp)
            lst_ref.append(copy.deepcopy(lst_ref[0]))
            with self.assertRaises(oracle.HadamardLawError):
                audit.secondary_audit_payload(p_long)
            # Same-length dict kills `type(x) is not list` independently of `len(x) != expected`
            p_wt = copy.deepcopy(base_payload)
            _set_at_path(p_wt, lp, {f"k{i}": f"v{i}" for i in range(orig_len)})
            with self.assertRaises(oracle.HadamardLawError):
                audit.secondary_audit_payload(p_wt)

        p_flt = copy.deepcopy(base_payload)
        p_flt["summary"]["target_max_cert_bytes"] = 10240.0
        self._assert_err("INVALID_JSON_TYPE", audit.secondary_audit_payload, p_flt)
        p_sub = copy.deepcopy(base_payload)
        p_sub["summary"]["target_max_cert_bytes"] = _SubInt(10240)
        self._assert_err("INVALID_JSON_TYPE", audit.secondary_audit_payload, p_sub)


class TestHardenedIOAndDarwinAndConcurrency(unittest.TestCase):
    def _assert_err(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.HadamardLawError) as cm:
            fn(*args, **kwargs)
        self.assertEqual(cm.exception.code, code)

    def test_path_coercion_and_darwin_normalization(self) -> None:
        self.assertEqual(audit._coerce_exact_path("a/b.json", "p"), Path("a/b.json"))
        self.assertEqual(audit._coerce_exact_path(Path("a/b.json"), "p"), Path("a/b.json"))
        self._assert_err("INVALID_PATH", audit._coerce_exact_path, "", "p")
        self._assert_err("NUL_FORBIDDEN", audit._coerce_exact_path, "a\x00b", "p")
        self._assert_err("INVALID_UTF8_SURROGATE", audit._coerce_exact_path, "a\ud800b", "p")
        self._assert_err("INVALID_FILENAME", audit._coerce_exact_path, "dir/", "p")
        self._assert_err("NUL_FORBIDDEN", audit._coerce_exact_path, Path("a\x00b"), "p")
        self._assert_err("INVALID_UTF8_SURROGATE", audit._coerce_exact_path, Path("a\ud800b"), "p")
        for bad_p in (123, _SubStr("a"), _SubPath("a")):
            self._assert_err("INVALID_PATH_TYPE", audit._coerce_exact_path, bad_p, "p")

        root_lnk = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        non_lnk = os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        user_lnk = os.stat_result((stat.S_IFLNK | 0o777, 0, 0, 1, 1000, 0, 0, 0, 0, 0))

        # Non-darwin must NOT rewrite even if /tmp were a root symlink (kills OP_SWAP & SUBCOND_DEL on platform check)
        with (
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(os, "lstat", return_value=root_lnk),
            mock.patch.object(os, "readlink", return_value="/private/tmp"),
        ):
            self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/a.json")), Path("/tmp/a.json"))

        # Mocked Darwin cases with readlink mocked so every single BoolOp subcondition is independently tested
        with mock.patch.object(sys, "platform", "darwin"):
            with mock.patch.object(os, "lstat", return_value=root_lnk), mock.patch.object(os, "readlink", return_value="private/tmp"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("x/tmp/y")), Path("x/tmp/y"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp"), allow_exact=False), Path("/tmp"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/../a.json")), Path("/tmp/../a.json"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/usr/a.json")), Path("/usr/a.json"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/x/y")), Path("/private/tmp/x/y"))
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp"), allow_exact=True), Path("/private/tmp"))
            with mock.patch.object(os, "lstat", return_value=root_lnk), mock.patch.object(os, "readlink", return_value="/private/var"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/var/x")), Path("/private/var/x"))
            with mock.patch.object(os, "lstat", return_value=non_lnk), mock.patch.object(os, "readlink", return_value="/private/tmp"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/x")), Path("/tmp/x"))
            with mock.patch.object(os, "lstat", return_value=user_lnk), mock.patch.object(os, "readlink", return_value="/private/tmp"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/x")), Path("/tmp/x"))
            with mock.patch.object(os, "lstat", return_value=root_lnk), mock.patch.object(os, "readlink", return_value="/evil"):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/x")), Path("/tmp/x"))
            with mock.patch.object(os, "lstat", side_effect=OSError("nope")):
                self.assertEqual(audit._normalize_darwin_system_prefix(Path("/tmp/x")), Path("/tmp/x"))

    def test_dir_chain_pwd_fifo_read_write_and_races(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s59_zb_") as td:
            base = Path(td)
            real_dir = base / "real"
            other_dir = base / "other"
            ro_dir = base / "ro_dir"
            real_dir.mkdir()
            other_dir.mkdir()
            ro_dir.mkdir()
            sym_dir = base / "symdir"
            os.symlink(real_dir, sym_dir)
            reg_file = real_dir / "data.bin"
            reg_file.write_bytes(b"hello")
            sym_file = real_dir / "symfile"
            os.symlink(reg_file, sym_file)
            fifo_path = real_dir / "scratch.fifo"
            os.mkfifo(fifo_path)

            with mock.patch.object(os, "open", side_effect=OSError(errno.EACCES, "denied")):
                self._assert_err("DIR_OPEN_FAILED", audit._open_dir_chain_nofollow, ("tmp",))

            fd_ok = audit._open_dir_chain_nofollow(tuple(p for p in real_dir.parts if p != "/") + ("", "."))
            os.close(fd_ok)
            self._assert_err("PARENT_TRAVERSAL_FORBIDDEN", audit._open_dir_chain_nofollow, ("tmp", ".."))
            self._assert_err("SYMLINK_FORBIDDEN", audit.read_regular_bytes_capped, sym_dir / "data.bin")
            self._assert_err("NOT_A_DIRECTORY", audit.read_regular_bytes_capped, reg_file / "child")
            self._assert_err("DIR_OPEN_FAILED", audit.read_regular_bytes_capped, base / "no_such_dir" / "x")

            real_open = os.open

            def _enotdir_open(path_arg, flags, *args, **kwargs):
                if path_arg == "ghost":
                    raise OSError(errno.ENOTDIR, "not a dir")
                return real_open(path_arg, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=_enotdir_open):
                self._assert_err(
                    "NOT_A_DIRECTORY",
                    audit._open_dir_chain_nofollow,
                    tuple(p for p in real_dir.parts if p != "/") + ("ghost",),
                )

            for bad_fn in (Path("/"), Path("."), Path("..")):
                self._assert_err("INVALID_FILENAME", audit._open_parent_dir_fd, bad_fn)

            # Relative path + $PWD tests: matching real, unset, relative, stale nonexistent, stale unrelated -> PASS;
            # matching symlinked $PWD, surrogate/NUL $PWD, race inode mismatch -> REJECT
            old_cwd = os.getcwd()
            old_pwd = os.environ.get("PWD")
            try:
                os.chdir(real_dir)
                os.environ["PWD"] = str(real_dir)
                self.assertEqual(audit.read_regular_bytes_capped("data.bin"), b"hello")
                os.environ.pop("PWD", None)
                self.assertEqual(audit.read_regular_bytes_capped("data.bin"), b"hello")
                os.environ["PWD"] = "."
                self.assertEqual(audit.read_regular_bytes_capped("data.bin"), b"hello")
                os.environ["PWD"] = "/nonexistent_stale_dir_99999"
                self.assertEqual(audit.read_regular_bytes_capped("data.bin"), b"hello")
                os.environ["PWD"] = str(other_dir)
                self.assertEqual(audit.read_regular_bytes_capped("data.bin"), b"hello")

                os.environ["PWD"] = str(sym_dir)
                self._assert_err("SYMLINK_FORBIDDEN", audit.read_regular_bytes_capped, "data.bin")
                with mock.patch.object(os, "environ", {"PWD": "/tmp/\ud800"}):
                    self._assert_err("INVALID_UTF8_SURROGATE", audit.read_regular_bytes_capped, "data.bin")
                with mock.patch.object(os, "environ", {"PWD": "/tmp/a\x00b"}):
                    self._assert_err("NUL_FORBIDDEN", audit.read_regular_bytes_capped, "data.bin")

                os.environ["PWD"] = str(real_dir)
                with mock.patch.object(os, "stat", side_effect=OSError("no dot")):
                    self._assert_err("CWD_UNAVAILABLE", audit._verify_pwd_not_symlink)
                fake_diff_dir = os.stat_result((stat.S_IFDIR | 0o755, 1111111, 2222222, 2, 0, 0, 0, 0, 0, 0))
                with mock.patch.object(os, "fstat", return_value=fake_diff_dir):
                    self._assert_err("PWD_SYMLINK_MISMATCH", audit._verify_pwd_not_symlink)
                with mock.patch.object(os, "getcwd", side_effect=OSError("no cwd")):
                    self._assert_err("CWD_UNAVAILABLE", audit._open_parent_dir_fd, Path("data.bin"))
            finally:
                os.chdir(old_cwd)
                if old_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = old_pwd

            # Read errors with exact code assertions
            self._assert_err("FILE_UNREADABLE", audit.read_regular_bytes_capped, real_dir / "missing.bin")
            self._assert_err("SYMLINK_FORBIDDEN", audit.read_regular_bytes_capped, sym_file)

            opened_names: list[str] = []

            def _spy_open(p_arg, flags, *args, **kwargs):
                opened_names.append(str(p_arg))
                return real_open(p_arg, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=_spy_open):
                self._assert_err("NON_REGULAR_FILE", audit.read_regular_bytes_capped, fifo_path)
            self.assertNotIn("scratch.fifo", opened_names)

            def _open_err_factory(err_no: int):
                def _fn(p_arg, flags, *args, **kwargs):
                    if p_arg == "data.bin":
                        raise OSError(err_no, "race")
                    return real_open(p_arg, flags, *args, **kwargs)
                return _fn

            with mock.patch.object(os, "open", side_effect=_open_err_factory(errno.ELOOP)):
                self._assert_err("SYMLINK_FORBIDDEN", audit.read_regular_bytes_capped, reg_file)
            with mock.patch.object(os, "open", side_effect=_open_err_factory(errno.EACCES)):
                self._assert_err("FILE_UNREADABLE", audit.read_regular_bytes_capped, reg_file)

            st_reg = os.lstat(reg_file)
            # Hardlink refused BEFORE os.open on read, AND fstat st_nlink > 1 race refused after open
            hl_r_a = real_dir / "hl_read_a.bin"
            hl_r_b = real_dir / "hl_read_b.bin"
            hl_r_a.write_bytes(b"hl")
            os.link(hl_r_a, hl_r_b)
            opened_hl: list[str] = []

            def _spy_hl_open(p_arg, flags, *args, **kwargs):
                opened_hl.append(str(p_arg))
                return real_open(p_arg, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=_spy_hl_open):
                self._assert_err("HARDLINK_FORBIDDEN", audit.read_regular_bytes_capped, hl_r_a)
            self.assertNotIn("hl_read_a.bin", opened_hl)

            fake_hl_same_ino = os.stat_result((stat.S_IFREG | 0o600, st_reg.st_ino, st_reg.st_dev, 2, 0, 0, 5, 0, 0, 0))
            with mock.patch.object(os, "fstat", return_value=fake_hl_same_ino):
                self._assert_err("HARDLINK_FORBIDDEN", audit.read_regular_bytes_capped, reg_file)

            # Same (st_dev, st_ino) as reg_file but non-regular mode specifically kills fstat S_ISREG check
            fake_fifo_same_ino = os.stat_result((stat.S_IFIFO | 0o600, st_reg.st_ino, st_reg.st_dev, 1, 0, 0, 0, 0, 0, 0))
            with mock.patch.object(os, "fstat", return_value=fake_fifo_same_ino):
                self._assert_err("NON_REGULAR_FILE", audit.read_regular_bytes_capped, reg_file)
            with mock.patch.object(os, "fstat", side_effect=OSError("fstat fail")):
                self._assert_err("READ_IO_ERROR", audit.read_regular_bytes_capped, reg_file)
            fake_swap_st = os.stat_result((stat.S_IFREG | 0o600, st_reg.st_ino + 999999, st_reg.st_dev, 1, 0, 0, 5, 0, 0, 0))
            with mock.patch.object(os, "fstat", return_value=fake_swap_st):
                self._assert_err("TOCTOU_SWAP", audit.read_regular_bytes_capped, reg_file)
            with mock.patch.object(os, "read", side_effect=OSError("read fail")):
                self._assert_err("READ_IO_ERROR", audit.read_regular_bytes_capped, reg_file)

            big_file = real_dir / "big.bin"
            big_bytes = b"A" * (audit._READ_CHUNK_BYTES + 123)
            big_file.write_bytes(big_bytes)
            self.assertEqual(audit.read_regular_bytes_capped(big_file, max_bytes=len(big_bytes)), big_bytes)
            self._assert_err("FILE_TOO_LARGE", audit.read_regular_bytes_capped, big_file, max_bytes=len(big_bytes) - 1)
            self._assert_err("INT_RANGE", audit.read_regular_bytes_capped, big_file, max_bytes=0)

            # Write errors: pre-open checks must abort BEFORE creating any temp file (verified with spy_open!)
            self._assert_err("INVALID_BYTES", audit.write_bytes_atomic_0600, real_dir / "out.bin", bytearray(b"x"))
            with mock.patch.object(os, "lstat", side_effect=OSError(errno.EACCES, "denied")):
                self._assert_err("TARGET_UNSTATTABLE", audit.write_bytes_atomic_0600, real_dir / "out.bin", b"x")

            hl_a = real_dir / "hl_a.bin"
            hl_b = real_dir / "hl_b.bin"
            hl_a.write_bytes(b"1")
            os.link(hl_a, hl_b)

            for bad_tgt, exp_code in [
                (sym_file, "SYMLINK_FORBIDDEN"),
                (fifo_path, "NON_REGULAR_FILE"),
                (hl_a, "HARDLINK_FORBIDDEN"),
            ]:
                created_tmp: list[str] = []

                def _spy_create(p_arg, flags, *args, _log=created_tmp, **kwargs):
                    if flags & os.O_CREAT:
                        _log.append(str(p_arg))
                    return real_open(p_arg, flags, *args, **kwargs)

                with mock.patch.object(os, "open", side_effect=_spy_create):
                    self._assert_err(exp_code, audit.write_bytes_atomic_0600, bad_tgt, b"x")
                self.assertEqual(created_tmp, [])

            # Read-only chmod 500 directory cleanly raises typed WRITE_IO_ERROR (never raw PermissionError)
            os.chmod(ro_dir, 0o500)
            try:
                self._assert_err("WRITE_IO_ERROR", audit.write_bytes_atomic_0600, ro_dir / "deny.json", b"x")
            finally:
                os.chmod(ro_dir, 0o700)

            # 31 EEXIST collisions succeed on 32nd attempt; 32 EEXIST collisions fail
            target_out = real_dir / "atomic.bin"
            seen_31 = {"n": 0}

            def _eexist_31(p_arg, flags, *args, **kwargs):
                if (flags & os.O_CREAT) and seen_31["n"] < 31:
                    seen_31["n"] += 1
                    raise FileExistsError("collision")
                return real_open(p_arg, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=_eexist_31):
                audit.write_bytes_atomic_0600(target_out, b"retry_ok")
            self.assertEqual(target_out.read_bytes(), b"retry_ok")
            self.assertEqual(stat.S_IMODE(target_out.stat().st_mode), 0o600)

            seen_32 = {"n": 0}

            def _eexist_32(p_arg, flags, *args, **kwargs):
                if (flags & os.O_CREAT) and seen_32["n"] < 32:
                    seen_32["n"] += 1
                    raise FileExistsError("always")
                return real_open(p_arg, flags, *args, **kwargs)

            with mock.patch.object(os, "open", side_effect=_eexist_32):
                self._assert_err("TEMP_CREATE_FAILED", audit.write_bytes_atomic_0600, target_out, b"fail")

            # Partial os.write succeeds; 0-byte os.write fails; os.write OSError fails cleanly
            real_write = os.write
            w_state = {"first": True}

            def _partial_write(fd, buf):
                if w_state["first"]:
                    w_state["first"] = False
                    return real_write(fd, buf[:2])
                return real_write(fd, buf)

            with mock.patch.object(os, "write", side_effect=_partial_write):
                audit.write_bytes_atomic_0600(target_out, b"abcdef")
            self.assertEqual(target_out.read_bytes(), b"abcdef")

            with mock.patch.object(os, "write", return_value=0):
                self._assert_err("SHORT_WRITE", audit.write_bytes_atomic_0600, target_out, b"abcdef")
            with mock.patch.object(os, "write", side_effect=OSError(errno.ENOSPC, "disk full")):
                self._assert_err("WRITE_IO_ERROR", audit.write_bytes_atomic_0600, target_out, b"abcdef")

            # Temp tampered (nlink == 2 or non-reg)
            for bad_st in [
                os.stat_result((stat.S_IFREG | 0o600, 1, 1, 2, 0, 0, 2, 0, 0, 0)),
                os.stat_result((stat.S_IFIFO | 0o600, 1, 1, 1, 0, 0, 2, 0, 0, 0)),
            ]:
                with mock.patch.object(os, "fstat", return_value=bad_st):
                    self._assert_err("TEMP_TAMPERED", audit.write_bytes_atomic_0600, target_out, b"ab")

            # Pre-replace lst2 races with exact error code assertions
            real_lstat = os.lstat
            for bad_st2, exp_code in [
                (os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 0, 0, 2, 0, 0, 0)), "SYMLINK_FORBIDDEN"),
                (os.stat_result((stat.S_IFIFO | 0o600, 1, 1, 1, 0, 0, 2, 0, 0, 0)), "NON_REGULAR_FILE"),
                (os.stat_result((stat.S_IFREG | 0o600, 1, 1, 2, 0, 0, 2, 0, 0, 0)), "HARDLINK_FORBIDDEN"),
            ]:
                cnt = {"n": 0}

                def _race_lstat(p_arg, *args, _st=bad_st2, _c=cnt, **kwargs):
                    _c["n"] += 1
                    if _c["n"] == 2:
                        return _st
                    return real_lstat(p_arg, *args, **kwargs)

                with mock.patch.object(os, "lstat", side_effect=_race_lstat):
                    self._assert_err(exp_code, audit.write_bytes_atomic_0600, target_out, b"ab")

            with mock.patch.object(os, "write", return_value=0), mock.patch.object(os, "unlink", side_effect=OSError("gone")):
                self._assert_err("SHORT_WRITE", audit.write_bytes_atomic_0600, target_out, b"ab")

    def test_concurrency_0600_hardlink_and_renameat2_stress(self) -> None:
        with tempfile.TemporaryDirectory(prefix="s59_conc_") as td:
            base = Path(td)
            t_out = base / "burst.json"
            payload_b = b'{"ok":true}\n'

            def _writer(idx: int) -> int:
                audit.write_bytes_atomic_0600(t_out, payload_b)
                return stat.S_IMODE(t_out.stat().st_mode)

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                modes = list(pool.map(_writer, range(80)))
            self.assertEqual(len(modes), 80)
            self.assertTrue(all(m == 0o600 for m in modes))

            hl_tgt = base / "hl_race.json"
            hl_alias = base / "hl_alias.json"
            hl_tgt.write_bytes(payload_b)
            stop_ev = threading.Event()
            flip_cnt = {"n": 0}

            def _flipper() -> None:
                for _ in range(520):
                    try:
                        os.link(hl_tgt, hl_alias)
                    except OSError:
                        pass
                    flip_cnt["n"] += 1
                    try:
                        os.unlink(hl_alias)
                    except OSError:
                        pass
                stop_ev.set()

            th = threading.Thread(target=_flipper)
            th.start()
            rejected = 0
            succeeded = 0
            while not stop_ev.is_set() or (rejected + succeeded) < 20:
                try:
                    audit.write_bytes_atomic_0600(hl_tgt, payload_b)
                    succeeded += 1
                except oracle.HadamardLawError:
                    rejected += 1
                if stop_ev.is_set() and (rejected + succeeded) >= 20:
                    break
            th.join()
            self.assertGreaterEqual(flip_cnt["n"], 500)
            self.assertGreaterEqual(rejected + succeeded, 20)

            if sys.platform == "linux" and platform.machine() == "x86_64":
                libc = ctypes.CDLL(None, use_errno=True)
                sys_renameat2 = 316
                rename_exchange = 2
                at_fdcwd = -100
                reg_a = base / "ex_reg.json"
                sym_b = base / "ex_sym.json"
                reg_a.write_bytes(payload_b)
                os.symlink(base / "nonexistent_target", sym_b)
                b_a = os.fsencode(str(reg_a))
                b_b = os.fsencode(str(sym_b))
                swaps = {"n": 0}
                ex_stop = threading.Event()

                def _swapper() -> None:
                    for _ in range(560):
                        rc = libc.syscall(sys_renameat2, at_fdcwd, b_a, at_fdcwd, b_b, rename_exchange)
                        if rc == 0:
                            swaps["n"] += 1
                    ex_stop.set()

                th2 = threading.Thread(target=_swapper)
                th2.start()
                reads = 0
                while not ex_stop.is_set():
                    try:
                        got = audit.read_regular_bytes_capped(reg_a)
                        self.assertEqual(got, payload_b)
                    except oracle.HadamardLawError:
                        pass
                    reads += 1
                th2.join()
                self.assertGreaterEqual(swaps["n"], 550)
                self.assertGreater(reads, 0)


class TestCLIEndToEnd(unittest.TestCase):
    def _run_main(self, argv) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            rc = audit.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_all_cli_modes_and_errors(self) -> None:
        rc, out, err = self._run_main([])
        self.assertEqual((rc, err), (0, ""))
        self.assertTrue(out.startswith("OK "))

        rc, out, err = self._run_main(("--json",))
        self.assertEqual((rc, err), (0, ""))
        self.assertTrue(json.loads(out)["valid"])

        rc, out_s, err = self._run_main(["--standalone"])
        self.assertEqual((rc, err), (0, ""))
        cert_s = oracle.parse_strict_json_text(out_s)
        self.assertTrue(cert_s["payload"]["summary"]["all_pillars_verified"])

        with tempfile.TemporaryDirectory(prefix="s59_cli_") as td:
            cpath = Path(td) / "cert.json"
            rc, out, err = self._run_main(["--write", str(cpath)])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(out.startswith("WROTE "))
            rc, out, err = self._run_main(["--write", str(cpath), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(json.loads(out)["written"])

            rc, out, err = self._run_main(["--check", str(cpath)])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(out.startswith("VALID "))
            rc, out, err = self._run_main(["--check", str(cpath), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(json.loads(out)["valid"])

            pretty = Path(td) / "pretty.json"
            pretty.write_text(json.dumps(cert_s, indent=2) + "\n", encoding="utf-8")
            rc, out, err = self._run_main(["--check", str(pretty)])
            self.assertEqual((rc, out), (2, ""))
            self.assertIn("NON_CANONICAL_BYTES", err)

            crlf = Path(td) / "crlf.json"
            crlf.write_bytes(oracle.canonical_json_bytes(cert_s) + b"\r\n")
            rc, out, err = self._run_main(["--check", str(crlf)])
            self.assertEqual((rc, out), (2, ""))
            self.assertIn("NON_CANONICAL_BYTES", err)

            bad_u8 = Path(td) / "bad_u8.json"
            bad_u8.write_bytes(b"\xff\xfe")
            rc, out, err = self._run_main(["--check", str(bad_u8)])
            self.assertEqual((rc, out), (2, ""))
            self.assertIn("INVALID_UTF8", err)

        fake_lean_res = {
            "valid": True,
            "lean_companion_sha256": oracle.LEAN_COMPANION_SHA256,
            "kernel_theorems_verified": 12,
            "banned_axioms_found": 0,
        }
        with mock.patch.object(audit, "verify_lean_companion_file", return_value=fake_lean_res):
            rc, out, err = self._run_main(["--verify-lean", str(_LEAN_COMPANION_PATH)])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(out.startswith("LEAN_VALID "))
            rc, out, err = self._run_main(["--verify-lean", str(_LEAN_COMPANION_PATH), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertEqual(json.loads(out)["kernel_theorems_verified"], 12)

        if _EMPIRICAL_ANCHOR_PATH.exists():
            rc, out, err = self._run_main(["--verify-anchor", str(_EMPIRICAL_ANCHOR_PATH)])
            self.assertEqual((rc, err), (0, ""))
            self.assertTrue(out.startswith("ANCHOR_VALID "))
            rc, out, err = self._run_main(["--verify-anchor", str(_EMPIRICAL_ANCHOR_PATH), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertEqual(json.loads(out)["layers_verified"], 392)

        if _CERT_PATH.exists():
            rc, out, err = self._run_main(["--check", str(_CERT_PATH), "--json"])
            self.assertEqual((rc, err), (0, ""))
            self.assertEqual(json.loads(out)["certificate_sha256"], cert_s["certificate_sha256"])

        # CLI argument errors with exact error-code checks (run in temp dir so mutated CLI_INVALID_PATH leaves no stray file)
        with tempfile.TemporaryDirectory(prefix="s59_cli_err_") as err_td:
            old_cwd = os.getcwd()
            old_pwd = os.environ.get("PWD")
            try:
                os.chdir(err_td)
                os.environ["PWD"] = err_td
                for bad_argv, exp_code in [
                    (123, "CLI_INVALID_ARGV"),
                    (["--standalone", "--standalone"], "CLI_CONFLICT"),
                    (["--json", "--json"], "CLI_DUPLICATE"),
                    (["--write", "/tmp/x", "--check", "/tmp/y"], "CLI_CONFLICT"),
                    (["--write"], "CLI_MISSING_ARG"),
                    (["--check"], "CLI_MISSING_ARG"),
                    (["--verify-lean"], "CLI_MISSING_ARG"),
                    (["--verify-anchor"], "CLI_MISSING_ARG"),
                    (["--write", "--json"], "CLI_INVALID_PATH"),
                    (["--write", ""], "CLI_INVALID_PATH"),
                    (["--unknown"], "CLI_UNKNOWN_ARG"),
                    (["--unknown", "extra"], "CLI_UNKNOWN_ARG"),
                ]:
                    rc, out, err = self._run_main(bad_argv)
                    self.assertEqual((rc, out), (2, ""))
                    self.assertIn(exp_code, err)
            finally:
                os.chdir(old_cwd)
                if old_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = old_pwd


class TestZLiveLeanAndEmpiricalAnchor(unittest.TestCase):
    def _assert_err(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.HadamardLawError) as cm:
            fn(*args, **kwargs)
        self.assertEqual(cm.exception.code, code)

    def test_live_lean_companion_and_failure_paths(self) -> None:
        res = audit.verify_lean_companion_file(_LEAN_COMPANION_PATH)
        self.assertTrue(res["valid"])
        self.assertEqual(res["kernel_theorems_verified"], 12)
        self.assertEqual(res["banned_axioms_found"], 0)

        with tempfile.TemporaryDirectory(prefix="s59_lean_t_") as td:
            bad_lean = Path(td) / "bad.lean"
            bad_lean.write_bytes(b"-- wrong\n")
            self._assert_err("LEAN_SHA256_MISMATCH", audit.verify_lean_companion_file, bad_lean)

            bad_utf8 = Path(td) / "bad_utf8.lean"
            bad_utf8.write_bytes(b"\xff\xfe")
            with mock.patch.object(oracle, "LEAN_COMPANION_SHA256", oracle.sha256_bytes(b"\xff\xfe")):
                self._assert_err("LEAN_INVALID_UTF8", audit.verify_lean_companion_file, bad_utf8)

        with mock.patch.object(shutil, "which", return_value=None):
            self._assert_err("LEAN_NOT_FOUND", audit.verify_lean_companion_file, _LEAN_COMPANION_PATH)

        # Missing / invalid lean_bin and TimeoutExpired cleanly wrapped into LEAN_EXEC_FAILED
        self._assert_err(
            "LEAN_EXEC_FAILED",
            audit.verify_lean_companion_file,
            _LEAN_COMPANION_PATH,
            lean_bin="/nonexistent_lean_binary_xyz_999",
        )
        with mock.patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired(["lean"], 120)):
            self._assert_err("LEAN_EXEC_FAILED", audit.verify_lean_companion_file, _LEAN_COMPANION_PATH, lean_bin="/fake/lean")

        all_thms_zero = "\n".join(
            f"'{oracle.LEAN_COMPANION_NAMESPACE}.{t}' does not depend on any axioms"
            for t in oracle.LEAN_COMPANION_THEOREMS
        )
        fake_zero_ok = subprocess.CompletedProcess(["lean"], 0, stdout=all_thms_zero + "\n", stderr="")
        with mock.patch.object(subprocess, "run", return_value=fake_zero_ok):
            r_z = audit.verify_lean_companion_file(_LEAN_COMPANION_PATH, lean_bin="/fake/lean")
            self.assertEqual(r_z["banned_axioms_found"], 0)

        fake_fail = subprocess.CompletedProcess(["lean"], 1, stdout=all_thms_zero, stderr="err")
        with mock.patch.object(subprocess, "run", return_value=fake_fail):
            self._assert_err("LEAN_KERNEL_FAILED", audit.verify_lean_companion_file, _LEAN_COMPANION_PATH, lean_bin="/fake/lean")

        for bad_ax_line in (
            f"'{oracle.LEAN_COMPANION_NAMESPACE}.h8_orthogonal_and_non_square' depends on axioms: [sorryAx]\n",
            f"'{oracle.LEAN_COMPANION_NAMESPACE}.h8_orthogonal_and_non_square' depends on axioms: [t._native.native_decide.ax_1_1]\n",
            f"'{oracle.LEAN_COMPANION_NAMESPACE}.h8_orthogonal_and_non_square' depends on axioms: [custom_ax]\n",
            f"'{oracle.LEAN_COMPANION_NAMESPACE}.h8_orthogonal_and_non_square' depends on axioms: []\n",
        ):
            fake_bad = subprocess.CompletedProcess(["lean"], 0, stdout=all_thms_zero + "\n" + bad_ax_line, stderr="")
            with mock.patch.object(subprocess, "run", return_value=fake_bad):
                self._assert_err("LEAN_BANNED_AXIOM", audit.verify_lean_companion_file, _LEAN_COMPANION_PATH, lean_bin="/fake/lean")

        # Live Lean 4.36.0 regression tests: native_decide, decide +native, and custom axiom MUST all be rejected
        lean_exe = shutil.which("lean")
        if lean_exe is not None:
            with tempfile.TemporaryDirectory(prefix="s59_lean_esc_") as esc_td:
                for fname, src_txt in (
                    ("esc_nd.lean", "namespace Aletheia.E8VsA16HadamardLaw\ntheorem t : 1 + 1 = 2 := by native_decide\nend Aletheia.E8VsA16HadamardLaw\n"),
                    ("esc_dn.lean", "namespace Aletheia.E8VsA16HadamardLaw\ntheorem t : 1 + 1 = 2 := by decide +native\nend Aletheia.E8VsA16HadamardLaw\n"),
                    ("esc_ax.lean", "namespace Aletheia.E8VsA16HadamardLaw\naxiom backdoor : False\ntheorem t : 1 + 1 = 2 := False.elim backdoor\nend Aletheia.E8VsA16HadamardLaw\n"),
                ):
                    esc_p = Path(esc_td) / fname
                    esc_b = src_txt.encode("utf-8")
                    esc_p.write_bytes(esc_b)
                    with (
                        mock.patch.object(oracle, "LEAN_COMPANION_SHA256", oracle.sha256_bytes(esc_b)),
                        mock.patch.object(oracle, "LEAN_COMPANION_THEOREMS", ("t",)),
                    ):
                        self._assert_err("LEAN_BANNED_AXIOM", audit.verify_lean_companion_file, esc_p, lean_bin=lean_exe)

        fake_missing = subprocess.CompletedProcess(["lean"], 0, stdout="only one line", stderr="")
        with mock.patch.object(subprocess, "run", return_value=fake_missing):
            self._assert_err("LEAN_MISSING_THEOREM", audit.verify_lean_companion_file, _LEAN_COMPANION_PATH, lean_bin="/fake/lean")

    def test_live_empirical_anchor_and_failure_paths(self) -> None:
        if _EMPIRICAL_ANCHOR_PATH.exists():
            res = audit.verify_empirical_anchor_file(_EMPIRICAL_ANCHOR_PATH)
            self.assertTrue(res["valid"])
            self.assertEqual(res["models_verified"], 2)
            self.assertEqual(res["layers_verified"], 392)

        with tempfile.TemporaryDirectory(prefix="s59_anc_t_") as td:
            p_wrong = Path(td) / "wrong.json"
            p_wrong.write_bytes(b"{}")
            self._assert_err("ANCHOR_SHA256_MISMATCH", audit.verify_empirical_anchor_file, p_wrong)

            p_badj = Path(td) / "bad.json"
            for bad_raw, exp_code in (
                (b"{bad", "ANCHOR_INVALID_JSON"),
                (b'{"n":' + b"9" * 5000 + b"}", "ANCHOR_INVALID_JSON"),
                (b"[]", "ANCHOR_SCHEMA_MISMATCH"),
                (b"{}", "ANCHOR_SCHEMA_MISMATCH"),
                (b'{"models":[]}', "ANCHOR_SCHEMA_MISMATCH"),
                (b'{"models":{}}', "ANCHOR_SCHEMA_MISMATCH"),
                (b'{"models":{"HuggingFaceTB/SmolLM2-360M":[]}}', "ANCHOR_SCHEMA_MISMATCH"),
            ):
                p_badj.write_bytes(bad_raw)
                with mock.patch.object(oracle, "EMPIRICAL_ANCHOR_SHA256", oracle.sha256_bytes(bad_raw)):
                    self._assert_err(exp_code, audit.verify_empirical_anchor_file, p_badj)

            synth = {"models": {}}
            for m_name, m_data in oracle.EMPIRICAL_MODELS_PINNED.items():
                synth["models"][m_name] = {
                    "arms": {k: {"wikitext2": {"mean_kl": v}} for k, v in m_data["wikitext2_kl"].items()},
                    "per_layer_anisotropy_diagnostic": {
                        "total_layers_scored": m_data["total_layers"],
                        "regime1_axis_beats_h16_count": m_data["regime1"],
                        "regime2_h16_beats_axis_count": m_data["regime2"],
                        "block_h2_beats_h16_count": m_data["bh2_beats_h16"],
                        "by_projection_kind": {
                            pk: {
                                "layers": pd["layers"],
                                "regime1_axis_beats_h16": pd["regime1"],
                                "regime2_h16_beats_axis": pd["regime2"],
                                "block_h2_beats_h16": pd["bh2_beats_h16"],
                                "mean_h16_over_axis_quad_err": pd["mean_h16_over_axis"],
                            }
                            for pk, pd in m_data["by_projection_kind"].items()
                        },
                    },
                }

            def _dump_synth(doc_obj: dict) -> bytes:
                raw_txt = json.dumps(doc_obj)
                for m_data in oracle.EMPIRICAL_MODELS_PINNED.values():
                    for dec in m_data["wikitext2_kl"].values():
                        raw_txt = raw_txt.replace(f'"{dec}"', dec)
                    for pd in m_data["by_projection_kind"].values():
                        raw_txt = raw_txt.replace(f'"{pd["mean_h16_over_axis"]}"', pd["mean_h16_over_axis"])
                return raw_txt.encode("utf-8")

            good_bytes = _dump_synth(synth)
            p_syn = Path(td) / "syn.json"
            p_syn.write_bytes(good_bytes)
            with mock.patch.object(oracle, "EMPIRICAL_ANCHOR_SHA256", oracle.sha256_bytes(good_bytes)):
                r_syn = audit.verify_empirical_anchor_file(p_syn)
                self.assertEqual(r_syn["layers_verified"], 392)

            for tamper_fn, exp_code in [
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"].__setitem__("total_layers_scored", 167), "ANCHOR_COUNT_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"].__setitem__("regime1_axis_beats_h16_count", 72), "ANCHOR_COUNT_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"].__setitem__("regime2_h16_beats_axis_count", 94), "ANCHOR_COUNT_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"].__setitem__("block_h2_beats_h16_count", 82), "ANCHOR_COUNT_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["arms"]["E8_H8"]["wikitext2"].__setitem__("mean_kl", "0.054841459"), "ANCHOR_KL_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"]["by_projection_kind"]["k_proj"].__setitem__("layers", 23), "ANCHOR_PROJ_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"]["by_projection_kind"]["k_proj"].__setitem__("regime1_axis_beats_h16", 8), "ANCHOR_PROJ_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"]["by_projection_kind"]["k_proj"].__setitem__("regime2_h16_beats_axis", 14), "ANCHOR_PROJ_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"]["by_projection_kind"]["k_proj"].__setitem__("block_h2_beats_h16", 9), "ANCHOR_PROJ_MISMATCH"),
                (lambda d: d["models"]["Qwen/Qwen2.5-0.5B"]["per_layer_anisotropy_diagnostic"]["by_projection_kind"]["k_proj"].__setitem__("mean_h16_over_axis_quad_err", "0.971996"), "ANCHOR_PROJ_MISMATCH"),
            ]:
                d_bad = copy.deepcopy(synth)
                tamper_fn(d_bad)
                b_bad = _dump_synth(d_bad)
                b_bad = b_bad.replace(b'"0.054841459"', b"0.054841459").replace(b'"0.971996"', b"0.971996")
                p_syn.write_bytes(b_bad)
                with mock.patch.object(oracle, "EMPIRICAL_ANCHOR_SHA256", oracle.sha256_bytes(b_bad)):
                    self._assert_err(exp_code, audit.verify_empirical_anchor_file, p_syn)


if __name__ == "__main__":
    unittest.main()
