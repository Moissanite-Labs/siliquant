#!/usr/bin/env python3
"""Assertion-free R5 mutation-killing, AST-BoolOp-killing, and Zone-B adversarial test suite
for AUDIT-115 (Track S15c: exact Sylvester-Hadamard E_8_Had isometry & adjoint GEMV).

Run with: python3 -I -O -B -W error test_e8had_isometry_r5.py -v
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from e8had_isometry_oracle import (
    E8HadError,
    build_rm13_hamming844_codewords,
    build_sylvester_h8,
    canonical_sha256,
    certify_dp_invariance_and_adjoint_gemv_exact,
    certify_e8_had_isometry_exact,
    certify_pr24_correspondence_and_roofline_exact,
    contains_forbidden_json_value,
    e8_theta_shell_count,
    enumerate_e8_had_scaled_shells,
    enumerate_e8_std_doubled_shells,
    format_exact_decimal,
    frac_to_dict,
    is_in_hamming844,
    is_valid_e8_had_scaled,
    is_valid_e8_std_doubled,
    map_e8_had_to_std_exact,
    map_e8_std_to_had_exact,
    matvec_h8_exact,
    require_exact_bool,
    require_exact_fraction,
    require_exact_int,
    require_git_sha40,
    require_sha256,
    require_vec8,
    sigma3_exact,
    wht8_butterfly_exact,
)
from e8had_isometry_audit import (
    main as audit_main,
    run_standalone_audit,
    validate_audit_certificate,
)


class TestPrimitivesAndIsometry(unittest.TestCase):
    def test_strict_validators_formatters_and_json_guard(self) -> None:
        self.assertEqual(require_exact_int(0, "x", allow_zero=True), 0)
        self.assertEqual(require_exact_int(1, "x", allow_zero=False), 1)
        self.assertEqual(require_exact_int(-5, "x", min_val=-5, max_val=5), -5)
        self.assertEqual(require_exact_int(5, "x", min_val=-5, max_val=5), 5)
        self.assertIs(require_exact_bool(True, "b"), True)
        self.assertEqual(require_exact_fraction(Fraction(3, 4), "f"), Fraction(3, 4))
        self.assertEqual(require_sha256("ab" * 32, "h"), "ab" * 32)
        self.assertEqual(require_git_sha40("ab" * 20, "g"), "ab" * 20)
        self.assertEqual(require_vec8([0] * 8, "v"), (0,) * 8)

        for bad_i in (True, False, "1", -1):
            with self.assertRaises(E8HadError):
                require_exact_int(bad_i, "x", allow_zero=True)
        with self.assertRaises(E8HadError):
            require_exact_int(0, "x", allow_zero=False)
        with self.assertRaises(E8HadError):
            require_exact_int(-6, "x", min_val=-5)
        with self.assertRaises(E8HadError):
            require_exact_int(6, "x", max_val=5)
        for bad_b in (0, 1, "true"):
            with self.assertRaises(E8HadError):
                require_exact_bool(bad_b, "b")
        with self.assertRaises(E8HadError):
            require_exact_fraction(1, "f")
        for bad_s in (123, "short", "A" * 64, "0" * 64):
            with self.assertRaisesRegex(E8HadError, "sha256"):
                require_sha256(bad_s, "h")
        for bad_g in (123, "short", "A" * 40, "0" * 40):
            with self.assertRaisesRegex(E8HadError, "git sha"):
                require_git_sha40(bad_g, "g")
        class _LyingSeq(Sequence):
            def __getitem__(self, idx: int) -> int:
                return 0
            def __len__(self) -> int:
                return 8
            def __iter__(self):
                return iter([0] * 9)

        class _ListSub(list):
            pass

        for bad_v in (123, "12345678", b"12345678", [0] * 7, [True] + [0] * 7, _LyingSeq(), _ListSub([0] * 8)):
            with self.assertRaises(E8HadError):
                require_vec8(bad_v, "v")

        self.assertEqual(frac_to_dict(Fraction(6, 8)), {"num": 3, "den": 4})
        self.assertEqual(format_exact_decimal(Fraction(-3, 4), 2), "-0.75")
        self.assertEqual(format_exact_decimal(Fraction(3, 4), 2), "0.75")
        self.assertEqual(format_exact_decimal(Fraction(7, 2), 0), "4")
        self.assertEqual(format_exact_decimal(Fraction(-1, 1000), 0), "0")
        self.assertEqual(format_exact_decimal(Fraction(-1, 10 ** 9), 6), "0.000000")
        self.assertEqual(format_exact_decimal(Fraction(0, 1), 0), "0")
        self.assertEqual(format_exact_decimal(Fraction(0, 1), 2), "0.00")
        self.assertEqual(format_exact_decimal(Fraction(1, 2), 18), "0.5" + ("0" * 17))
        with self.assertRaises(E8HadError):
            format_exact_decimal(Fraction(1, 3), 19)
        with self.assertRaises(E8HadError):
            format_exact_decimal(Fraction(1, 3), 100000)

        self.assertFalse(contains_forbidden_json_value({"a": [1, "x", (True, None)]}))
        self.assertFalse(contains_forbidden_json_value([1], max_depth=1))
        self.assertTrue(contains_forbidden_json_value(json.loads('{"x": 1.5}')))
        self.assertTrue(contains_forbidden_json_value({1: "int_key"}))
        self.assertTrue(contains_forbidden_json_value({1, 2}))
        self.assertTrue(contains_forbidden_json_value([json.loads("0.5")]))
        deep: list = [1]
        for _ in range(1500):
            deep = [deep]
        with self.assertRaisesRegex(E8HadError, "nesting depth"):
            contains_forbidden_json_value(deep)
        with self.assertRaises(E8HadError):
            contains_forbidden_json_value({}, max_depth=0)
        with self.assertRaisesRegex(E8HadError, "strict str-key"):
            canonical_sha256(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(E8HadError, "strict str-key"):
            canonical_sha256({1: "int_key"})  # type: ignore[dict-item]

    def test_wht8_hamming_lattice_maps_and_guards(self) -> None:
        h8 = build_sylvester_h8()
        self.assertEqual(len(h8), 8)
        v = (1, -2, 3, -4, 5, -6, 7, -8)
        bf, ops = wht8_butterfly_exact(v)
        self.assertEqual(ops, 24)
        self.assertEqual(bf, matvec_h8_exact(v))

        cws = build_rm13_hamming844_codewords()
        self.assertEqual(len(cws), 16)
        self.assertTrue(all(is_in_hamming844(c) for c in cws))
        self.assertFalse(is_in_hamming844([1, 0, 0, 0, 0, 0, 0, 0]))
        with self.assertRaisesRegex(E8HadError, "exact 0 or 1"):
            is_in_hamming844([2, 0, 0, 0, 0, 0, 0, 0])

        # Valid & invalid doubled E_8^std and Construction-A E_8^Had
        self.assertTrue(is_valid_e8_std_doubled([0] * 8))
        self.assertTrue(is_valid_e8_std_doubled([2, 2, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(is_valid_e8_std_doubled([1] * 8))
        self.assertFalse(is_valid_e8_std_doubled([1, 0, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(is_valid_e8_std_doubled([2, 0, 0, 0, 0, 0, 0, 0]))

        self.assertEqual(map_e8_std_to_had_exact([1] * 8), (2, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(map_e8_had_to_std_exact([2, 0, 0, 0, 0, 0, 0, 0]), (1,) * 8)
        self.assertTrue(is_valid_e8_had_scaled([2, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(is_valid_e8_had_scaled([1, 1, 1, 1, 0, 0, 0, 0]))
        self.assertFalse(is_valid_e8_had_scaled([1, 0, 0, 0, 0, 0, 0, 0]))

        with self.assertRaisesRegex(E8HadError, "not a valid doubled E_8"):
            map_e8_std_to_had_exact([2, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(E8HadError, "not a valid Construction-A"):
            map_e8_had_to_std_exact([1, 0, 0, 0, 0, 0, 0, 0])

        # Theta & shell enumerator boundary checks (m=0 and m=1)
        self.assertEqual(sigma3_exact(1), 1)
        self.assertEqual(sigma3_exact(4), 73)
        self.assertEqual(sigma3_exact(6), 252)
        self.assertEqual(e8_theta_shell_count(0), 1)
        self.assertEqual(e8_theta_shell_count(1), 240)
        self.assertEqual(len(enumerate_e8_std_doubled_shells(0)[0]), 1)
        self.assertEqual(len(enumerate_e8_had_scaled_shells(0)[0]), 1)
        with self.assertRaises(E8HadError):
            enumerate_e8_std_doubled_shells(5)
        with self.assertRaises(E8HadError):
            enumerate_e8_had_scaled_shells(5)

    def test_all_three_pillars_and_zone_b_certificate_anti_bypass(self) -> None:
        p1 = certify_e8_had_isometry_exact(m_max=3)
        self.assertTrue(p1["pillar_verified"])
        self.assertEqual(p1["total_vectors_verified"], 9121)
        self.assertEqual(p1["total_nonzero_vectors_verified"], 9120)
        self.assertEqual(p1["shell1_axis_vectors"], 16)
        self.assertEqual(p1["shell1_weight4_vectors"], 224)

        p2 = certify_dp_invariance_and_adjoint_gemv_exact()
        self.assertTrue(p2["pillar_verified"])
        self.assertEqual(p2["ebg32_b136s8_bits"], 136)
        self.assertEqual(p2["ebg32_b88s8_bits"], 88)

        p3 = certify_pr24_correspondence_and_roofline_exact()
        self.assertTrue(p3["pillar_verified"])

        cert = run_standalone_audit()
        self.assertTrue(cert["overall_verified"])
        self.assertTrue(validate_audit_certificate(cert)["valid"])

        def _rehash(d: dict) -> dict:
            out = dict(d)
            out.pop("certificate_sha256", None)
            out["certificate_sha256"] = canonical_sha256(out)
            return out

        with self.assertRaisesRegex(E8HadError, "mapping"):
            validate_audit_certificate(123)  # type: ignore[arg-type]
        flt = json.loads(json.dumps(cert))
        flt["pillar1_e8had_isometry_and_hamming"]["m_max_exhaustive"] = json.loads("3.5")
        with self.assertRaisesRegex(E8HadError, "forbidden"):
            validate_audit_certificate(flt)
        with self.assertRaisesRegex(E8HadError, "key set mismatch"):
            validate_audit_certificate(dict(cert, extra="x"))
        with self.assertRaisesRegex(E8HadError, "schema_version"):
            validate_audit_certificate(_rehash(dict(cert, schema_version="wrong")))
        with self.assertRaisesRegex(E8HadError, "audit_id"):
            validate_audit_certificate(_rehash(dict(cert, audit_id="AUDIT-999")))
        with self.assertRaisesRegex(E8HadError, "mode mismatch"):
            validate_audit_certificate(_rehash(dict(cert, mode="bad")))
        with self.assertRaisesRegex(E8HadError, "novelty_class"):
            validate_audit_certificate(_rehash(dict(cert, novelty_class="N2")))
        for bad_ver in (False, 1):
            with self.assertRaisesRegex(E8HadError, "overall_verified"):
                validate_audit_certificate(_rehash(dict(cert, overall_verified=bad_ver)))
        with self.assertRaisesRegex(E8HadError, "digest mismatch"):
            validate_audit_certificate(dict(cert, certificate_sha256="01" * 32))

        for pkey in (
            "pillar1_e8had_isometry_and_hamming",
            "pillar2_dp_invariance_and_adjoint_gemv",
            "pillar3_pr24_correspondence_and_roofline",
        ):
            bad_int_bool = dict(cert[pkey], pillar_verified=1)
            for bad_sub in (123, {"pillar_verified": True}, bad_int_bool):
                with self.assertRaisesRegex(E8HadError, pkey):
                    validate_audit_certificate(_rehash(dict(cert, **{pkey: bad_sub})))
        # List length & list element tampering inside pillar1
        p1_short = json.loads(json.dumps(cert["pillar1_e8had_isometry_and_hamming"]))
        p1_short["shells"] = []
        with self.assertRaisesRegex(E8HadError, "pillar1"):
            validate_audit_certificate(_rehash(dict(cert, pillar1_e8had_isometry_and_hamming=p1_short)))
        p1_elem = json.loads(json.dumps(cert["pillar1_e8had_isometry_and_hamming"]))
        p1_elem["shells"][0]["std_count"] = 99
        with self.assertRaisesRegex(E8HadError, "pillar1"):
            validate_audit_certificate(_rehash(dict(cert, pillar1_e8had_isometry_and_hamming=p1_elem)))

        # CLI tests (--standalone --json --write-cert, text mode, --check valid + IO/UTF-8/JSON/validation errors)
        def _run_cli(argv: list[str]) -> tuple[int, str, str]:
            bo, be = io.StringIO(), io.StringIO()
            old_o, old_e = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = bo, be
                rc = audit_main(argv)
            finally:
                sys.stdout, sys.stderr = old_o, old_e
            return rc, bo.getvalue(), be.getvalue()

        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "cert.json"
            rc, out, _ = _run_cli(["--standalone", "--json", "--write-cert", str(cp)])
            self.assertEqual(rc, 0)
            self.assertEqual(out, cp.read_text(encoding="utf-8"))

            rc, out, _ = _run_cli(["--standalone"])
            self.assertEqual(rc, 0)
            self.assertIn("AUDIT-115", out)

            rc, out, _ = _run_cli(["--check", str(cp)])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(out)["valid"])

            rc, _, err = _run_cli(["--check", str(Path(td) / "missing.json")])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(err)["error_code"], "CERT_IO_ERROR")

            bad_utf8 = Path(td) / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe\x00invalid")
            rc, _, err = _run_cli(["--check", str(bad_utf8)])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(err)["error_code"], "CERT_IO_ERROR")

            bad_json = Path(td) / "bad.json"
            for raw_bad in (
                "{broken",
                '{"big": ' + ("9" * 5000) + "}",
            ):
                bad_json.write_text(raw_bad, encoding="utf-8")
                rc, _, err = _run_cli(["--check", str(bad_json)])
                self.assertEqual(rc, 2)
                self.assertEqual(json.loads(err)["error_code"], "CERT_JSON_DECODE_ERROR")

            for payload, expected_sub in (
                ('{"a": 1.5}', "forbidden JSON float"),
                ('{"a": NaN}', "forbidden JSON float"),
                ('{"a": 1, "a": 2}', "duplicate JSON key"),
                ('{"deep": ' + ("[" * 100) + "1" + ("]" * 100) + "}", "nesting depth"),
                ('{"schema_version": "bad"}', "key set mismatch"),
            ):
                bad_json.write_text(payload, encoding="utf-8")
                rc, _, err = _run_cli(["--check", str(bad_json)])
                self.assertEqual(rc, 2)
                err_obj = json.loads(err)
                self.assertEqual(err_obj["error_code"], "CERT_VALIDATION_ERROR")
                self.assertIn(expected_sub, err_obj["error"])


if __name__ == "__main__":
    unittest.main()
