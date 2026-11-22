#!/usr/bin/env python3
"""Assertion-free R5 test suite for Track S15d (AUDIT-116) pure-Z/Q oracle and fail-closed CLI.

Zero bare asserts (100% -O immune), pure Python 3 stdlib, deterministic across -I/-O/-B/-W error.
Thoroughly exercises every public API, mathematical invariant, boundary value, and fail-closed
malformed-input guard (including bool-vs-int, bad types, bad ranges, tamper detection, and I/O).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from contextlib import redirect_stderr, redirect_stdout

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import e8_stage1_audit as audit
import e8_stage1_oracle as oracle


class TestE8Stage1OracleAndAudit(unittest.TestCase):
    def _expect_oracle_err(self, fn, *args) -> None:
        with self.assertRaises(oracle.Stage1OracleError):
            fn(*args)

    def _expect_audit_err(self, fn, *args) -> None:
        with self.assertRaises(audit.Stage1AuditError):
            fn(*args)

    def test_01_oracle_exact_primitives_and_fail_closed_guards(self) -> None:
        # format_exact_decimal exact values, rounding, negative & zero-place paths
        self.assertEqual(oracle.format_exact_decimal(Fraction(0, 1), 3), "0.000")
        self.assertEqual(oracle.format_exact_decimal(Fraction(-3, 2), 1), "-1.5")
        self.assertEqual(oracle.format_exact_decimal(Fraction(-1, 10000), 2), "0.00")
        self.assertEqual(oracle.format_exact_decimal(Fraction(5, 2), 0), "3")
        self.assertEqual(oracle.format_exact_decimal(Fraction(7, 4), 2), "1.75")
        self._expect_oracle_err(oracle.format_exact_decimal, 1.5, 2)
        self._expect_oracle_err(oracle.format_exact_decimal, Fraction(1, 2), True)
        self._expect_oracle_err(oracle.format_exact_decimal, Fraction(1, 2), -1)
        self._expect_oracle_err(oracle.format_exact_decimal, Fraction(1, 2), 13)

        # sigma3 & e8_shell_count exact values and strict int guards
        self.assertEqual(oracle.sigma3(1), 1)
        self.assertEqual(oracle.sigma3(4), 1 + 8 + 64)
        self.assertEqual(oracle.sigma3(6), 1 + 8 + 27 + 216)
        self.assertEqual(oracle.e8_shell_count(0), 1)
        self.assertEqual(oracle.e8_shell_count(1), 240)
        self.assertEqual(oracle.e8_shell_count(2), 2160)
        self.assertEqual(oracle.e8_shell_count(3), 6720)
        self._expect_oracle_err(oracle.sigma3, 0)
        self._expect_oracle_err(oracle.sigma3, True)
        self._expect_oracle_err(oracle.e8_shell_count, -1)
        self._expect_oracle_err(oracle.e8_shell_count, False)

        # is_valid_e8_doubled exactness & fail-closed guards
        self.assertTrue(oracle.is_valid_e8_doubled((0, 0, 0, 0, 0, 0, 0, 0)))
        self.assertTrue(oracle.is_valid_e8_doubled((2, -2, 0, 0, 0, 0, 0, 0)))
        self.assertTrue(oracle.is_valid_e8_doubled((1, 1, 1, 1, 1, 1, 1, 1)))
        self.assertFalse(oracle.is_valid_e8_doubled((2, 0, 0, 0, 0, 0, 0, 0)))
        self.assertFalse(oracle.is_valid_e8_doubled((1, 2, 0, 0, 0, 0, 0, 0)))
        self.assertFalse(oracle.is_valid_e8_doubled((1, 1, 2, 0, 0, 0, 0, 0)))
        self.assertFalse(oracle.is_valid_e8_doubled((1, 1, 1, 1, 1, 1, 1, -1)))
        self._expect_oracle_err(oracle.is_valid_e8_doubled, "12345678")
        self._expect_oracle_err(oracle.is_valid_e8_doubled, (0, 0, 0, 0, 0, 0, 0))
        self._expect_oracle_err(oracle.is_valid_e8_doubled, (True, 0, 0, 0, 0, 0, 0, 0))

        # build_inner_dp_counts boundary guards
        dp1 = oracle.build_inner_dp_counts(1)
        self.assertEqual(dp1["s1"], [1, 240])
        self._expect_oracle_err(oracle.build_inner_dp_counts, 0)
        self._expect_oracle_err(oracle.build_inner_dp_counts, 513)
        self._expect_oracle_err(oracle.build_inner_dp_counts, True)

        # build_topbit_bucket_guide exactness (both shift == 0 and shift > 0 + tail buckets)
        g_small = oracle.build_topbit_bucket_guide([1, 2, 1], 3)
        self.assertEqual(g_small["shift"], 0)
        self.assertEqual(g_small["guide_idx"][:4], [0, 1, 1, 2])
        self.assertEqual(g_small["guide_idx"][7], 2)
        self.assertEqual(g_small["max_span"], 0)

        g_large = oracle.build_topbit_bucket_guide([10, 30, 50, 20, 15], 2)
        self.assertGreater(g_large["shift"], 0)
        self.assertEqual(g_large["n_buckets"], 4)
        self.assertEqual(g_large["total"], 125)
        self.assertGreaterEqual(g_large["max_span"], 1)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, [], 2)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, "abc", 2)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, [1, 2], 0)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, [1, 2], 11)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, [0, 0], 2)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, [-1, 5], 2)
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, [True, 5], 2)

    def test_02_d8plus_leaders_and_unranking_guards(self) -> None:
        # Exact 22 partitions of 8 and 95,503 universal permutations
        self.assertEqual(oracle.count_integer_partitions_of_8_and_perms(), (22, 95503))

        # Exact shell-0 and shell-1 leaders
        l0 = oracle.generate_d8plus_shell_leaders(0)
        self.assertEqual(len(l0), 1)
        self.assertEqual(oracle.unrank_d8plus_leader_orbit(0, l0[0][1], l0[0][2], l0[0][3], 0), (0,) * 8)

        l1 = oracle.generate_d8plus_shell_leaders(1)
        self.assertEqual(len(l1), 2)
        self.assertEqual(sum(p << sb for _, _, p, sb in l1), 240)
        # Shells 4 and 5 exercise all-8-nonzero even leaders (idx == 8) & parity rejection
        for m_hi in (4, 5):
            lm = oracle.generate_d8plus_shell_leaders(m_hi)
            self.assertEqual(sum(p << sb for _, _, p, sb in lm), oracle.e8_shell_count(m_hi))
        # Verify both even and odd unranking paths
        v_ev = oracle.unrank_d8plus_leader_orbit(0, (2, 2, 0, 0, 0, 0, 0, 0), 28, 2, 3)
        self.assertTrue(oracle.is_valid_e8_doubled(v_ev))
        v_od0 = oracle.unrank_d8plus_leader_orbit(1, (1, 1, 1, 1, 1, 1, 1, 1), 1, 7, 0)
        v_od1 = oracle.unrank_d8plus_leader_orbit(1, (1, 1, 1, 1, 1, 1, 1, 1), 1, 7, 1)
        self.assertTrue(oracle.is_valid_e8_doubled(v_od0))
        self.assertTrue(oracle.is_valid_e8_doubled(v_od1))
        self.assertNotEqual(v_od0, v_od1)
        self.assertEqual(v_od1[7], -1)

        # Fail-closed guards on generate_d8plus_shell_leaders & unrank_d8plus_leader_orbit & count_all
        self._expect_oracle_err(oracle.generate_d8plus_shell_leaders, -1)
        self._expect_oracle_err(oracle.generate_d8plus_shell_leaders, 65)
        self._expect_oracle_err(oracle.generate_d8plus_shell_leaders, True)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 2, (0,) * 8, 1, 0, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (0,) * 7, 1, 0, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, "00000000", 1, 0, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (-2, 2, 0, 0, 0, 0, 0, 0), 28, 2, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (2, 2, 0, 0, 0, 0, 0, 0), 0, 2, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (2, 2, 0, 0, 0, 0, 0, 0), 40321, 2, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (2, 2, 0, 0, 0, 0, 0, 0), 28, 9, 0)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (2, 2, 0, 0, 0, 0, 0, 0), 28, 2, 112)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, (2, 2, 0, 0, 0, 0, 0, 0), 28, 2, -1)
        self._expect_oracle_err(oracle.count_all_d8plus_leaders_up_to, 0)
        self._expect_oracle_err(oracle.count_all_d8plus_leaders_up_to, 513)

    def test_03_full_certificate_and_fail_closed_zone_b_audit_cli(self) -> None:
        cert = audit.build_signed_certificate()
        self.assertTrue(cert["overall_verified"])

        p1 = cert["pillar1_exact_bit_widths"]
        self.assertTrue(p1["pillar_verified"])
        self.assertEqual(p1["ebg32_b88s8"]["max_s1"], 31456320)
        self.assertEqual(p1["ebg32_b88s8"]["max_s1_bits"], 25)
        self.assertEqual(p1["ebg32_b88s8"]["w1_ball_count"], 360605761)
        self.assertEqual(p1["ebg32_b88s8"]["w1_ball_bits"], 29)
        self.assertTrue(p1["ebg32_b88s8"]["w1_ball_fits_u32"])
        self.assertEqual(p1["ebg32_b136s8"]["max_s1"], 16749815040)
        self.assertEqual(p1["ebg32_b136s8"]["max_s1_bits"], 34)
        self.assertEqual(p1["ebg32_b136s8"]["w1_ball_count"], 1525056448801)
        self.assertEqual(p1["ebg32_b136s8"]["w1_ball_bits"], 41)
        self.assertTrue(p1["ebg32_b136s8"]["w1_ball_fits_u64"])
        self.assertEqual(p1["ebg32_b136s8"]["max_c4"], 14976)
        self.assertEqual(p1["ebg32_b136s8"]["max_c2"], 24)

        p2 = cert["pillar2_bucket_guide_and_lut_concentration"]
        self.assertTrue(p2["pillar_verified"])
        self.assertEqual(p2["flat_lut_m_le_2_bytes"], 19208)
        self.assertEqual(p2["flat_lut_m_le_3_bytes"], 72968)

        p3 = cert["pillar3_d8plus_leader_bijection_and_wall"]
        self.assertTrue(p3["pillar_verified"])
        self.assertEqual(p3["d8plus_m0_to_m3_vectors_verified"], 9121)
        self.assertEqual(p3["ebg32_b88s8_leaders"]["total_leaders"], 2000)
        self.assertEqual(p3["ebg32_b136s8_leaders"]["total_leaders"], 1033160)
        self.assertEqual(p3["leader_blowup_136b_over_88b"], "516.58")
        self.assertEqual(p3["universal_22_shape_perm_count"], 95503)
        self.assertEqual(p3["binomial_comb_mask_table_bytes"], 502)

        # Valid Zone-B check
        verdict = audit.validate_certificate_payload(cert)
        self.assertTrue(verdict["valid"])
        self.assertEqual(verdict["payload_sha256"], cert["payload_sha256"])

        # Fail-closed Zone-B rejections with exact message regexes (100% mutation kill + D1-D3 hardening)
        with self.assertRaisesRegex(audit.Stage1AuditError, "strict dict"):
            audit.canonical_json_bytes("not_a_dict")  # type: ignore[arg-type]
        with self.assertRaisesRegex(audit.Stage1AuditError, "strict dict"):
            audit.compute_payload_sha256("not_a_dict")  # type: ignore[arg-type]
        with self.assertRaisesRegex(audit.Stage1AuditError, "JSON object"):
            audit.validate_certificate_payload(["not_a_dict"])
        with self.assertRaisesRegex(audit.Stage1AuditError, "must be Path"):
            audit.validate_certificate_file("not_a_path")  # type: ignore[arg-type]
        with self.assertRaisesRegex(audit.Stage1AuditError, "strict str"):
            audit.validate_certificate_payload({1: "non_str_key"})
        with self.assertRaisesRegex(audit.Stage1AuditError, "forbidden JSON value"):
            audit.validate_certificate_payload({"f": 1.5})
        with self.assertRaisesRegex(audit.Stage1AuditError, "forbidden JSON value"):
            audit.validate_certificate_payload({"s": [1, (2,)]})

        # Exact depth boundary: depth 64 allowed, depth 65 rejected (kills > vs >= mutant!)
        d64: Any = 1
        for _ in range(64):
            d64 = [d64]
        audit._assert_safe_json_tree(d64)
        d65 = [d64]
        with self.assertRaisesRegex(audit.Stage1AuditError, "nesting depth"):
            audit.validate_certificate_payload({"deep": d65})

        # Exact integer bit-length boundary (S15D-BUG-02): 2048-bit allowed, 2049-bit & 10**5000 rejected
        max_ok_int = (1 << 2048) - 1
        self.assertIn(str(max_ok_int).encode("ascii"), audit.canonical_json_bytes({"big_ok": max_ok_int}))
        for bad_int in (1 << 2048, -(1 << 2048), 10**5000 - 1, 10**5000, -(10**5000)):
            with self.assertRaisesRegex(audit.Stage1AuditError, "integer bit length"):
                audit.canonical_json_bytes({"big": bad_int})
            with self.assertRaisesRegex(audit.Stage1AuditError, "integer bit length"):
                audit.validate_certificate_payload({"big": bad_int})

        # Lying Sequence / list subclass rejected by oracle (D3)
        class _ListSub(list):
            pass

        self._expect_oracle_err(oracle.is_valid_e8_doubled, _ListSub([0] * 8))
        self._expect_oracle_err(oracle.build_topbit_bucket_guide, _ListSub([1, 2]), 2)
        self._expect_oracle_err(oracle.unrank_d8plus_leader_orbit, 0, _ListSub([0] * 8), 1, 0, 0)

        for field, bad_val, rx in [
            ("schema_version", "wrong", "unexpected schema_version"),
            ("audit_id", "WRONG", "unexpected audit_id"),
            ("mode", "wrong", "unexpected mode"),
            ("novelty_class", "N0", "unexpected novelty_class"),
            ("overall_verified", False, "overall_verified must be True"),
            ("overall_verified", 1, "invalid type"),
            ("payload_sha256", "0" * 63, "64 lowercase hex ASCII"),
            ("payload_sha256", "0" * 65, "64 lowercase hex ASCII"),
            ("payload_sha256", "\u00e9" * 64, "64 lowercase hex ASCII"),
            ("payload_sha256", "A" * 64, "64 lowercase hex ASCII"),
            ("payload_sha256", "g" * 64, "64 lowercase hex ASCII"),
            ("payload_sha256", "0" * 64, "payload_sha256 mismatch"),
        ]:
            tampered = copy.deepcopy(cert)
            tampered[field] = bad_val
            with self.assertRaisesRegex(audit.Stage1AuditError, rx):
                audit.validate_certificate_payload(tampered)

        # Missing required key
        missing = copy.deepcopy(cert)
        del missing["pillar1_exact_bit_widths"]
        with self.assertRaisesRegex(audit.Stage1AuditError, "missing required key"):
            audit.validate_certificate_payload(missing)

        # Re-signed tampered payload (valid sha256, diverges from oracle; also nested bool-as-int forgery!)
        forged = copy.deepcopy(cert)
        forged["pillar1_exact_bit_widths"]["ebg32_b88s8"]["max_s1"] = 123
        forged["payload_sha256"] = audit.compute_payload_sha256(forged)
        with self.assertRaisesRegex(audit.Stage1AuditError, "diverges from pure-Z/Q oracle"):
            audit.validate_certificate_payload(forged)

        forged_bool = copy.deepcopy(cert)
        forged_bool["pillar1_exact_bit_widths"]["pillar_verified"] = 1
        forged_bool["payload_sha256"] = audit.compute_payload_sha256(forged_bool)
        with self.assertRaisesRegex(audit.Stage1AuditError, "diverges from pure-Z/Q oracle"):
            audit.validate_certificate_payload(forged_bool)

        # CLI standalone (--json & text) + --check (valid, missing file, bad UTF-8, bad JSON, floats, NaN, dup keys, 5000-digit int, deep nesting)
        out_json = io.StringIO()
        with redirect_stdout(out_json):
            rc_j = audit.main(["--standalone", "--json"])
        self.assertEqual(rc_j, 0)
        self.assertEqual(json.loads(out_json.getvalue()), cert)

        out_txt = io.StringIO()
        with redirect_stdout(out_txt):
            rc_t = audit.main(["--standalone"])
        self.assertEqual(rc_t, 0)
        self.assertIn("AUDIT-116 PURE-ZQ OK", out_txt.getvalue())

        with tempfile.TemporaryDirectory(prefix="s15d_r5_") as td:
            os.chmod(td, 0o700)
            p_ok = Path(td) / "ok.json"
            p_ok.write_text(out_json.getvalue(), encoding="utf-8")
            buf_ok = io.StringIO()
            with redirect_stdout(buf_ok):
                self.assertEqual(audit.main(["--check", str(p_ok)]), 0)
            self.assertTrue(json.loads(buf_ok.getvalue())["valid"])

            p_missing = Path(td) / "does_not_exist.json"
            err_m = io.StringIO()
            with redirect_stderr(err_m):
                self.assertEqual(audit.main(["--check", str(p_missing)]), 1)
            self.assertIn("CHECK FAILED", err_m.getvalue())

            p_bin = Path(td) / "bad_utf8.json"
            p_bin.write_bytes(b"\xff\xfe\xfd")
            with self.assertRaisesRegex(audit.Stage1AuditError, "cannot read"):
                audit.validate_certificate_file(p_bin)

            p_badj = Path(td) / "bad.json"
            bad_non_ascii_cert = copy.deepcopy(cert)
            bad_non_ascii_cert["payload_sha256"] = "\u00e9" * 64
            p_non_ascii = Path(td) / "non_ascii_sha.json"
            p_non_ascii.write_text(json.dumps(bad_non_ascii_cert, ensure_ascii=False), encoding="utf-8")
            err_na = io.StringIO()
            with redirect_stderr(err_na):
                self.assertEqual(audit.main(["--check", str(p_non_ascii)]), 1)
            self.assertIn("64 lowercase hex ASCII", err_na.getvalue())

            for bad_txt, rx in (
                ("{not json", "invalid JSON"),
                ('{"x": 1.5}', "forbidden JSON float"),
                ('{"x": NaN}', "forbidden JSON float"),
                ('{"a": 1, "a": 2}', "duplicate JSON key"),
                ('{"big": ' + ("9" * 5000) + "}", "invalid JSON"),
                ('{"big2049": ' + str(1 << 2048) + "}", "integer bit length"),
                ('{"d": ' + ("[" * 100) + "1" + ("]" * 100) + "}", "nesting depth"),
            ):
                p_badj.write_text(bad_txt, encoding="utf-8")
                with self.assertRaisesRegex(audit.Stage1AuditError, rx):
                    audit.validate_certificate_file(p_badj)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestE8Stage1OracleAndAudit)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
