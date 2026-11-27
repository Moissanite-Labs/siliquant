#!/usr/bin/env python3
"""Track S165 (EXP-Q47b) R5 Refusal, Boundary, Zone-B I/O, Lean R12/R13b,
and Exact Empirical Kill-Gate Test Suite.

Covers BOTH s165_john_siliquant_oracle.py AND verify_s165_john_siliquant.py
(zero shared code between them). Zero bare asserts, zero skips, zero floats.
"""
from __future__ import annotations

import ast
import copy
from decimal import Decimal
from fractions import Fraction
import hashlib
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

_HERE = Path(__file__).parent
BASE = _HERE if _HERE.is_absolute() else (Path(os.getcwd()) / _HERE)
sys.path.insert(0, str(BASE))

import s165_john_siliquant_oracle as oracle  # noqa: E402
import verify_s165_john_siliquant as verifier  # noqa: E402


def _capture_fds(fn, *args, **kwargs) -> tuple[int, str, str]:
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        s_out = os.dup(1)
        s_err = os.dup(2)
        try:
            os.dup2(out_f.fileno(), 1)
            os.dup2(err_f.fileno(), 2)
            rc = fn(*args, **kwargs)
        finally:
            os.dup2(s_out, 1)
            os.dup2(s_err, 2)
            os.close(s_out)
            os.close(s_err)
        out_f.seek(0)
        err_f.seek(0)
        return rc, out_f.read().decode("utf-8"), err_f.read().decode("utf-8")


def _copy_pkg(dst: Path) -> None:
    for fn in (
        oracle.LEAN_FILE_NAME,
        "lean-toolchain",
        "lakefile.toml",
        "lake-manifest.json",
        oracle.RECEIPT_FILE_NAME,
        oracle.EMPIRICAL_RESULTS_FILE_NAME,
        oracle.CERT_FILE_NAME,
    ):
        shutil.copyfile(BASE / fn, dst / fn)


def _good_lean_runner(_base: Path) -> dict[str, object]:
    src = (BASE / verifier.LEAN_SRC_NAME).read_text(encoding="utf-8")
    thms = verifier._PUB_THM_RE.findall(verifier.strip_lean_comments(src))
    lines = [f"'JohnQuantOperators.{t}' depends on axioms: [propext]" for t in thms]
    # Exact 7 zero-axiom theorems and 17 propext-only theorems matching live Lean 4.36.0
    zero_set = {
        "diameter_sq_gap_cube_over_ball_universal",
        "volume_ratio_cube_over_cross_growth_universal",
        "exact_spherical_abs_recurrence_d3_d4_d8_d16",
        "exact_mean_width_brackets_d3_d4_d8_d16",
        "exact_volume_gaps_d3_d4_d8_d16",
        "hadwiger_d4_zonotope_and_star_helly_obstruction",
        "polar_cap_bulk_contraction_and_crossover_universal",
    }
    for idx, t in enumerate(thms):
        if t in zero_set:
            lines[idx] = f"'JohnQuantOperators.{t}' does not depend on any axioms"
    return {
        "version_rc": 0,
        "version_output": "Lean (version 4.36.0, linux, release)\n",
        "build_rc": 0,
        "build_output": "Build completed successfully (4 jobs).\n",
        "axiom_rc": 0,
        "axiom_stderr": "",
        "axiom_stdout": "\n".join(lines) + "\n",
    }


def _load_receipt_copy() -> dict[str, object]:
    return json.loads((BASE / oracle.RECEIPT_FILE_NAME).read_text(encoding="utf-8"))


def _load_raw_copy() -> dict[str, object]:
    return json.loads((BASE / oracle.EMPIRICAL_RESULTS_FILE_NAME).read_text(encoding="utf-8"), parse_float=Decimal)


class TestS165R5(unittest.TestCase):
    cert: dict[str, object] = {}
    emp_sha: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.emp_sha = hashlib.sha256((BASE / oracle.EMPIRICAL_RESULTS_FILE_NAME).read_bytes()).hexdigest()
        rc_o, out_o, err_o = _capture_fds(oracle.main, ["--json"])
        if rc_o != 0 or err_o != "":
            raise RuntimeError(f"oracle baseline failed rc={rc_o} err={err_o}")
        rc_v, out_v, err_v = _capture_fds(verifier.main, ["--check", "--json"])
        if rc_v != 0 or err_v != "":
            raise RuntimeError(f"verifier baseline failed rc={rc_v} err={err_v}")
        if out_o != out_v:
            raise RuntimeError("oracle and zero-shared-code verifier JSON output diverged")
        cls.cert = json.loads(out_v)

    def _expect_ora(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(oracle.OracleError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def _expect_ver(self, code: str, fn, *args, **kwargs) -> None:
        with self.assertRaises(verifier.VerifyError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.code, code)

    def test_01_static_hygiene_and_zero_shared_code(self) -> None:
        ver_tree = ast.parse((BASE / "verify_s165_john_siliquant.py").read_text(encoding="utf-8"))
        ora_tree = ast.parse((BASE / "s165_john_siliquant_oracle.py").read_text(encoding="utf-8"))
        for node in ast.walk(ver_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn("oracle", alias.name)
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("oracle", node.module or "")
        for tree in (ver_tree, ora_tree):
            for node in ast.walk(tree):
                self.assertNotIsInstance(node, ast.Assert)
                if isinstance(node, ast.Constant):
                    self.assertNotIsInstance(node.value, float)
                if isinstance(node, ast.Name):
                    self.assertNotEqual(node.id, "float")

    def test_02_exact_certificate_invariants_and_honest_verdicts(self) -> None:
        self.assertEqual(self.cert["schema"], "s165.q47b.hadwiger_john_siliquant.v1")
        self.assertEqual(self.cert["energy_claim"], "UNCONTROLLED_LAPTOP_NOT_CLAIMED")
        self.assertEqual(self.cert["lean"]["public_theorems"], 24)
        p1 = self.cert["pillar1_john_geometry"]
        self.assertEqual(p1["mean_width_exact_pi_cleared"]["3"], "3/2")
        self.assertEqual(p1["mean_width_exact_pi_cleared"]["8_times_pi"], "256/35")
        self.assertEqual(p1["mean_width_exact_pi_cleared"]["16_times_pi"], "65536/6435")
        self.assertEqual(p1["o4_john_contacts"], 8)
        self.assertEqual(p1["o16_john_contacts"], 16)
        self.assertEqual(p1["d4_star_helly_14_cap_area"], "7/8")
        self.assertEqual(p1["d4_zonotope_v2_witness_sq"], [48, 36])

        p3 = self.cert["pillar3_helly_and_polar_bounds"]
        self.assertEqual(p3["polar_win_ratio"], "198/625")
        self.assertEqual(p3["polar_reversal_ratio"], "256/125")

        p4 = self.cert["pillar4_theta_and_empirical_gates"]
        self.assertEqual(p4["a16_g128_b30_theta_bounds"], {"352": 547, "357": 577, "368": 651})
        self.assertEqual(p4["a16_g128_b45_theta_bounds"], {"544": 4385, "549": 4629, "560": 5213})
        self.assertEqual(p4["e8_g128_theta_bounds"], {"368": 420, "560": 3380})

        eg = p4["empirical_gates"]
        self.assertEqual(eg["total_live_arms_verified"], 52)
        self.assertEqual(eg["kg1_a16_prongs_passed"], 1)
        self.assertEqual(eg["kg1_a16_prongs_total"], 8)
        self.assertEqual(eg["kg1_john_b1_verdict"], "PARTIAL")
        self.assertEqual(eg["kg2_helly_b2_verdict"], "FALSIFIED")
        self.assertEqual(eg["arm_b3_polar_cap_verdict"], "PARTIAL")
        self.assertEqual(eg["overall_empirical_verdict"], "KG1_PARTIAL_KG2_FALSIFIED_B3_PARTIAL")
        self.assertTrue(eg["b3b_payload_plus_scale_within_3bpw"])
        self.assertFalse(eg["b3b_with_1bit_selector_map_within_3bpw"])
        self.assertFalse(eg["b3a_with_16b_rowptr_within_3bpw"])

        # Parameterized mathematical helpers boundary tests (both oracle and verifier)
        self.assertEqual(oracle.check_john_gram([(1, 1), (1, -1)], 2, 2), 2)
        self._expect_ora("P1_GRAM_COUNT", oracle.check_john_gram, [(1, 1)], 2, 2)
        self._expect_ora("P1_VEC_DIM", oracle.check_john_gram, [(1, 1), (1,)], 2, 2)
        self._expect_ora("P1_VEC_NORM", oracle.check_john_gram, [(2, 0), (0, 1)], 2, 1)
        self._expect_ora("P1_GRAM_DIAG", oracle.check_john_gram, [(1, 1), (1, 1)], 2, 2)

        self._expect_ora("P2_EMPTY", oracle.check_folded_dot_invariance, ())
        self._expect_ora("P2_SCALE_POS", oracle.check_folded_dot_invariance, ((Fraction(1), Fraction(1), Fraction(1), Fraction(0)),))
        self._expect_ora("P2_CS_LEN", oracle.check_cauchy_schwarz_pair, (), (), ())
        self._expect_ora("P2_CS_LEN", oracle.check_cauchy_schwarz_pair, (Fraction(1),), (Fraction(1), Fraction(2)), (Fraction(1),))
        self._expect_ora("P2_CS_LEN", oracle.check_cauchy_schwarz_pair, (Fraction(1),), (Fraction(1),), (Fraction(1), Fraction(1)))
        self._expect_ora("P2_U_POS", oracle.check_cauchy_schwarz_pair, (Fraction(1),), (Fraction(1),), (Fraction(0),))

        self.assertEqual(oracle.check_chebyshev_interval(5, 5), (Fraction(5, 1), Fraction(0, 1), Fraction(5, 1)))
        self._expect_ora("P3_INT_ORDER", oracle.check_chebyshev_interval, 6, 5)
        self._expect_ora("P3_MBULK_POS", oracle.check_polar_crossover, 0, 5, 10, 5, 1, 1)
        self._expect_ora("P3_MCONTRACT", oracle.check_polar_crossover, 5, 5, 10, 5, 1, 1)
        self._expect_ora("P3_ECAP_POS", oracle.check_polar_crossover, 3, 5, 10, 0, 1, 1)
        self._expect_ora("P3_MDEN_POS", oracle.check_polar_crossover, 3, 5, 10, 5, 1, 0)

        self.assertIn("1", oracle.compute_a16_g128_theta_bounds(1, (1,)))
        self._expect_ora("P4_RMAX_POS", oracle.compute_a16_g128_theta_bounds, 0, (1,))
        self.assertIn("1", oracle.compute_e8_g128_theta_bounds(2, (1,)))
        self._expect_ora("P4_E8_RMAX", oracle.compute_e8_g128_theta_bounds, 1, (1,))

        self.assertEqual(verifier.verify_contact_gram([[1, 1], [1, -1]], 2, 2), 2)
        self._expect_ver("V_GRAM_ROWS", verifier.verify_contact_gram, [[1, 1]], 2, 2)
        self._expect_ver("V_GRAM_DIM", verifier.verify_contact_gram, [[1, 1], [1]], 2, 2)
        self._expect_ver("V_GRAM_NORM", verifier.verify_contact_gram, [[2, 0], [0, 1]], 2, 1)
        self._expect_ver("V_GRAM_ENTRY", verifier.verify_contact_gram, [[1, 1], [1, 1]], 2, 2)

        self._expect_ver("V_P2_EMPTY", verifier.verify_folded_channels, ())
        self._expect_ver("V_P2_SCALE", verifier.verify_folded_channels, ((Fraction(1), Fraction(1), Fraction(1), Fraction(0)),))
        self.assertEqual(verifier.verify_chebyshev_bounds(5, 5), (Fraction(5, 1), Fraction(0, 1), Fraction(5, 1)))
        self._expect_ver("V_P3_ORDER", verifier.verify_chebyshev_bounds, 6, 5)
        self._expect_ver("V_P3_MB", verifier.verify_polar_ratio, 0, 5, 10, 5, 1, 1)
        self._expect_ver("V_P3_MC", verifier.verify_polar_ratio, 5, 5, 10, 5, 1, 1)
        self._expect_ver("V_P3_EC", verifier.verify_polar_ratio, 3, 5, 10, 0, 1, 1)
        self._expect_ver("V_P3_MD", verifier.verify_polar_ratio, 3, 5, 10, 5, 1, 0)

        self.assertIn("352", verifier.independent_a16_theta_bounds(1, (352,)))
        self._expect_ver("V_P4_RMAX", verifier.independent_a16_theta_bounds, 0, (352,))
        self.assertIn("368", verifier.independent_e8_theta_bounds(2, (368,)))
        self._expect_ver("V_P4_E8_RMAX", verifier.independent_e8_theta_bounds, 1, (368,))

    def test_03_zone_b_io_and_json_scanner_oracle(self) -> None:
        self._expect_ora("ORA_CLI_PATH", oracle._validate_cli_path_arg, "")
        self._expect_ora("ORA_NUL_PATH", oracle._validate_cli_path_arg, "a\x00b")
        self._expect_ora("ORA_TRAVERSAL", oracle._validate_cli_path_arg, "..")
        self._expect_ora("ORA_TRAVERSAL", oracle._validate_cli_path_arg, ".")
        self._expect_ora("ORA_TRAVERSAL", oracle._validate_cli_path_arg, "a/../b")
        self._expect_ora("ORA_NUL_PATH", oracle._read_regular_bytes, Path("bad\x00name"))
        self._expect_ora("ORA_TRAVERSAL", oracle._read_regular_bytes, Path("/tmp/../etc/passwd"))
        self._expect_ora("ORA_ROOT_PATH", oracle._read_regular_bytes, Path("/"))
        self._expect_ora("ORA_FILE_OPEN", oracle._read_regular_bytes, Path("/nonexistent_s165_probe"))

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            self._expect_ora("ORA_PARENT_OPEN", oracle._read_regular_bytes, tp / "missing_dir" / "a.txt")
            self._expect_ora("ORA_NOT_REG", oracle._read_regular_bytes, tp)
            f8 = tp / "f8.bin"
            f9 = tp / "f9.bin"
            f8.write_bytes(b"12345678")
            f9.write_bytes(b"123456789")
            with mock.patch.object(oracle, "MAX_BYTES", 8):
                self.assertEqual(oracle._read_regular_bytes(f8), b"12345678")
                self._expect_ora("ORA_TOO_LARGE", oracle._read_regular_bytes, f9)

        self._expect_ora("ORA_BAD_UTF8", oracle._parse_strict_json_bytes, b"\xff\xfe\n")
        self._expect_ora("ORA_BOM", oracle._parse_strict_json_bytes, "\ufeff{}\n".encode("utf-8"))
        self._expect_ora("ORA_CRLF", oracle._parse_strict_json_bytes, b"{}\r\n")
        self._expect_ora("ORA_NO_FINAL_LF", oracle._parse_strict_json_bytes, b"{}")

        ok_depth = ('{"a":' + "[" * 31 + "1" + "]" * 31 + "}\n").encode("utf-8")
        bad_depth = ('{"a":' + "[" * 32 + "1" + "]" * 32 + "}\n").encode("utf-8")
        self.assertIsInstance(oracle._parse_strict_json_bytes(ok_depth), dict)
        self._expect_ora("ORA_JSON_DEPTH", oracle._parse_strict_json_bytes, bad_depth)
        self._expect_ora("ORA_JSON_UNBAL", oracle._parse_strict_json_bytes, b"{}}\n")

        ok_dig = ('{"n":' + "9" * 256 + "}\n").encode("utf-8")
        bad_dig = ('{"n":' + "9" * 257 + "}\n").encode("utf-8")
        self.assertIsInstance(oracle._parse_strict_json_bytes(ok_dig), dict)
        self._expect_ora("ORA_INT_DIGITS", oracle._parse_strict_json_bytes, bad_dig)

        self._expect_ora("ORA_FLOAT_TOKEN", oracle._parse_strict_json_bytes, b'{"x":1.5}\n')
        self._expect_ora("ORA_FLOAT_TOKEN", oracle._parse_strict_json_bytes, b'{"x":1e2}\n')
        dec_obj = oracle._parse_strict_json_bytes(b'{"x":1.5}\n', allow_decimal=True)
        self.assertEqual(dec_obj["x"], Decimal("1.5"))
        self._expect_ora("ORA_BAD_CONST", oracle._parse_strict_json_bytes, b'{"x":1e99999999999999999999}\n', allow_decimal=True)

        self._expect_ora("ORA_JSON_STATE", oracle._parse_strict_json_bytes, b'{"x":1\n')
        self._expect_ora("ORA_JSON_STATE", oracle._parse_strict_json_bytes, b'"unterm\n')
        self._expect_ora("ORA_DUP_KEY", oracle._parse_strict_json_bytes, b'{"a":1,"a":2}\n')
        self._expect_ora("ORA_BAD_CONST", oracle._parse_strict_json_bytes, b'{"a":NaN}\n')
        self._expect_ora("ORA_JSON_PARSE", oracle._parse_strict_json_bytes, b'{"a":}\n')
        self._expect_ora("ORA_NOT_DICT", oracle._parse_strict_json_bytes, b"[1,2]\n")

        self._expect_ora("K_INT", oracle._req_int, {}, "a", "K_INT")
        self._expect_ora("K_INT", oracle._req_int, {"a": True}, "a", "K_INT")
        self._expect_ora("K_STR", oracle._req_str, {}, "a", "K_STR")
        self._expect_ora("K_STR", oracle._req_str, {"a": 1}, "a", "K_STR")
        self._expect_ora("K_DICT", oracle._req_dict, {}, "a", "K_DICT")
        self._expect_ora("K_DICT", oracle._req_dict, {"a": []}, "a", "K_DICT")
        self._expect_ora("K_LIST", oracle._req_list, {}, "a", "K_LIST")
        self._expect_ora("K_LIST", oracle._req_list, {"a": {}}, "a", "K_LIST")
        self._expect_ora("SC_ERR", oracle._scale_num, "bad", 10, "SC_ERR")
        self._expect_ora("SC_ERR", oracle._scale_num, Decimal("Infinity"), 10, "SC_ERR")

    def test_04_zone_b_io_and_json_scanner_verifier(self) -> None:
        self._expect_ver("BAD_PATH_TYPE", verifier.validate_cli_path, 123)
        self._expect_ver("BAD_PATH_STR", verifier.validate_cli_path, "")
        self._expect_ver("BAD_PATH_STR", verifier.validate_cli_path, "a\x00b")
        self._expect_ver("BAD_PATH_TRAV", verifier.validate_cli_path, "..")
        self._expect_ver("BAD_PATH_TRAV", verifier.validate_cli_path, ".")
        self._expect_ver("BAD_PATH_TRAV", verifier.validate_cli_path, "a/../b")
        self._expect_ver("BAD_PATH_TYPE", verifier.read_strict_bytes, 123)
        self._expect_ver("BAD_PATH_STR", verifier.read_strict_bytes, "")
        self._expect_ver("BAD_PATH_STR", verifier.read_strict_bytes, "bad\x00path")
        self._expect_ver("BAD_PATH_TRAV", verifier.read_strict_bytes, "/tmp/../etc/passwd")
        self._expect_ver("ROOT_PATH", verifier.read_strict_bytes, Path("/"))
        self._expect_ver("FILE_OPEN", verifier.read_strict_bytes, Path("/nonexistent_s165_probe"))

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            self._expect_ver("PARENT_OPEN", verifier.read_strict_bytes, tp / "missing_dir" / "a.txt")
            self._expect_ver("NOT_REG_FILE", verifier.read_strict_bytes, tp)
            sym = tp / "link.txt"
            target = tp / "real.txt"
            target.write_bytes(b"hello\n")
            os.symlink(target, sym)
            self._expect_ver("FILE_OPEN", verifier.read_strict_bytes, sym)

            f8 = tp / "f8.bin"
            f9 = tp / "f9.bin"
            f8.write_bytes(b"12345678")
            f9.write_bytes(b"123456789")
            with mock.patch.object(verifier, "MAX_FILE_BYTES", 8):
                self.assertEqual(verifier.read_strict_bytes(f8), b"12345678")
                self._expect_ver("FILE_TOO_LARGE", verifier.read_strict_bytes, f9)

        self._expect_ver("BAD_UTF8", verifier.decode_strict_utf8_lf, b"\xff\xfe\n")
        self._expect_ver("BAD_BOM", verifier.decode_strict_utf8_lf, "\ufeff{}\n".encode("utf-8"))
        self._expect_ver("BAD_CRLF", verifier.decode_strict_utf8_lf, b"{}\r\n")
        self._expect_ver("BAD_FINAL_LF", verifier.decode_strict_utf8_lf, b"{}")

        ok_depth = '{"a":' + "[" * 31 + "1" + "]" * 31 + "}\n"
        bad_depth = '{"a":' + "[" * 32 + "1" + "]" * 32 + "}\n"
        self.assertIsInstance(verifier.parse_strict_json(ok_depth), dict)
        self._expect_ver("JSON_DEPTH", verifier.parse_strict_json, bad_depth)
        self._expect_ver("JSON_UNBAL", verifier.parse_strict_json, "{}}\n")

        ok_dig = '{"n":' + "9" * 256 + "}\n"
        bad_dig = '{"n":' + "9" * 257 + "}\n"
        self.assertIsInstance(verifier.parse_strict_json(ok_dig), dict)
        self._expect_ver("INT_DIGITS", verifier.parse_strict_json, bad_dig)
        self._expect_ver("FLOAT_TOKEN", verifier.parse_strict_json, '{"x":1.5}\n')
        self._expect_ver("FLOAT_TOKEN", verifier.parse_strict_json, '{"x":1E2}\n')
        self.assertEqual(verifier.parse_strict_json('{"x":1.5}\n', allow_decimal=True)["x"], Decimal("1.5"))
        self._expect_ver("BAD_CONST", verifier.parse_strict_json, '{"x":1e99999999999999999999}\n', allow_decimal=True)
        self._expect_ver("JSON_EOF", verifier.parse_strict_json, '{"x":1\n')
        self._expect_ver("JSON_EOF", verifier.parse_strict_json, '"ab\n')
        self._expect_ver("DUP_KEY", verifier.parse_strict_json, '{"a":1,"a":2}\n')
        self._expect_ver("BAD_CONST", verifier.parse_strict_json, '{"a":Infinity}\n')
        self._expect_ver("BAD_JSON", verifier.parse_strict_json, '{"a":}\n')
        self._expect_ver("NOT_OBJECT", verifier.parse_strict_json, "[1,2]\n")

        self._expect_ver("CERT_TYPE", verifier.verify_exact_tree, {"a": True}, {"a": 1})
        self._expect_ver("CERT_KEYS", verifier.verify_exact_tree, {}, {"a": 1})
        self._expect_ver("CERT_LEN", verifier.verify_exact_tree, [1], [1, 2])
        self._expect_ver("CERT_VAL", verifier.verify_exact_tree, {"a": 2}, {"a": 1})

        self._expect_ver("VK_I", verifier._need_int, {}, "a", "VK_I")
        self._expect_ver("VK_I", verifier._need_int, {"a": False}, "a", "VK_I")
        self._expect_ver("VK_S", verifier._need_str, {}, "a", "VK_S")
        self._expect_ver("VK_S", verifier._need_str, {"a": 1}, "a", "VK_S")
        self._expect_ver("VK_D", verifier._need_dict, {}, "a", "VK_D")
        self._expect_ver("VK_D", verifier._need_dict, {"a": []}, "a", "VK_D")
        self._expect_ver("VK_L", verifier._need_list, {}, "a", "VK_L")
        self._expect_ver("VK_L", verifier._need_list, {"a": {}}, "a", "VK_L")
        self._expect_ver("VQ_ERR", verifier._quant_dec, "bad", 10, "VQ_ERR")
        self._expect_ver("VQ_ERR", verifier._quant_dec, Decimal("NaN"), 10, "VQ_ERR")

    def test_05_lean_audit_and_live_runner_gates(self) -> None:
        cleaned = verifier.strip_lean_comments("a /- outer /- inner -/ still -/ b -- tail\nc\n")
        self.assertIn("a", cleaned)
        self.assertIn("b", cleaned)
        self.assertNotIn("inner", cleaned)
        s_slash_dash = verifier.strip_lean_comments("x / y - z\na /- p / q - r /- inner -/ still -/ b -- tail\nc")
        self.assertEqual(s_slash_dash[:9], "x / y - z")
        self.assertIn("a ", s_slash_dash)
        self.assertIn("b ", s_slash_dash)
        self.assertIn("\nc", s_slash_dash)
        self.assertNotIn("inner", s_slash_dash)
        self.assertNotIn("still", s_slash_dash)
        self.assertNotIn("p", s_slash_dash)
        self.assertNotIn("q", s_slash_dash)
        self.assertNotIn("r", s_slash_dash)
        self.assertEqual(verifier.strip_lean_comments("u -- eof"), "u       ")
        self._expect_ver("UNTERM_COMMENT", verifier.strip_lean_comments, "a /- open\n")

        good_lines = [f"'JohnQuantOperators.t{i}' depends on axioms: [propext]" for i in range(17)]
        for i in range(17, 24):
            good_lines.append(f"'JohnQuantOperators.t{i}' does not depend on any axioms")
        ax_res = verifier.check_axiom_output(good_lines, 24)
        self.assertEqual(ax_res, {"zero_axiom": 7, "propext_only": 17})
        self._expect_ver("AXIOM_COUNT", verifier.check_axiom_output, good_lines[:-1], 24)
        bad_fmt = list(good_lines)
        bad_fmt[0] = "garbage line"
        self._expect_ver("AXIOM_FORMAT", verifier.check_axiom_output, bad_fmt, 24)
        bad_br = list(good_lines)
        bad_br[0] = "'JohnQuantOperators.t0' depends on axioms: propext"
        self._expect_ver("AXIOM_BRACKETS", verifier.check_axiom_output, bad_br, 24)
        bad_br2 = list(good_lines)
        bad_br2[0] = "'JohnQuantOperators.t0' depends on axioms: [propext"
        self._expect_ver("AXIOM_BRACKETS", verifier.check_axiom_output, bad_br2, 24)
        bad_br3 = list(good_lines)
        bad_br3[0] = "'JohnQuantOperators.t0' depends on axioms: propext]"
        self._expect_ver("AXIOM_BRACKETS", verifier.check_axiom_output, bad_br3, 24)
        bad_ax = list(good_lines)
        bad_ax[0] = "'JohnQuantOperators.t0' depends on axioms: [Classical.choice]"
        self._expect_ver("FORBIDDEN_AXIOM", verifier.check_axiom_output, bad_ax, 24)

        # Non-UTF-8 bytes from live subprocess must fail closed with VerifyError LIVE_LEAN_UTF8
        self._expect_ver("LIVE_LEAN_UTF8", verifier._decode_proc_bytes, b"\xff\xfe")
        with mock.patch.object(
            subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["lean"], 0, stdout=b"\xff\xfe", stderr=b""),
        ):
            self._expect_ver("LIVE_LEAN_UTF8", verifier.default_live_lean_runner, BASE)

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            _copy_pkg(tp)
            res = verifier.verify_lean_bundle(tp, lean_runner=_good_lean_runner)
            self.assertEqual(res["public_theorems"], 24)

            for key, val, code in (
                ("version_rc", 1, "LIVE_LEAN_VER"),
                ("version_output", "Lean 4.33.1", "LIVE_LEAN_VER"),
                ("build_rc", 1, "LIVE_LAKE_RC"),
                ("build_output", "warning: unused", "LIVE_LAKE_WARN"),
                ("axiom_rc", 1, "LIVE_AX_RC"),
                ("axiom_stderr", "err", "LIVE_AX_RC"),
            ):
                def _bad_run(_b: Path, k=key, v=val) -> dict[str, object]:
                    d = _good_lean_runner(_b)
                    d[k] = v
                    return d
                self._expect_ver(code, verifier.verify_lean_bundle, tp, lean_runner=_bad_run)

            # Axiom split enforcement (LOW-2): faking 6/18 instead of 7/17 must fail closed
            def _wrong_split(_b: Path) -> dict[str, object]:
                d = _good_lean_runner(_b)
                lines = str(d["axiom_stdout"]).splitlines()
                lines[1] = lines[1].replace("does not depend on any axioms", "depends on axioms: [propext]")
                d["axiom_stdout"] = "\n".join(lines) + "\n"
                return d
            self._expect_ver("LIVE_AX_SPLIT_ZERO", verifier.verify_lean_bundle, tp, lean_runner=_wrong_split)

            with mock.patch.object(verifier, "EXPECTED_PROPEXT_ONLY_THMS", 99):
                self._expect_ver("LIVE_AX_SPLIT_PROP", verifier.verify_lean_bundle, tp, lean_runner=_good_lean_runner)

            (tp / ".lake").mkdir()
            self._expect_ver("LAKE_RESIDUE", verifier.verify_lean_bundle, tp)
            (tp / ".lake").rmdir()

            for fname, code, mut_fn in (
                ("lean-toolchain", "BAD_TOOLCHAIN", lambda _s: "leanprover/lean4:v4.33.1\n"),
                ("lakefile.toml", "LAKEFILE_REQ", lambda s: s + "[[require]]\nname = \"mathlib\"\n"),
                ("lake-manifest.json", "MANIFEST_PKGS", lambda _s: '{"packages": ["x"]}\n'),
            ):
                orig = (tp / fname).read_text(encoding="utf-8")
                (tp / fname).write_text(mut_fn(orig), encoding="utf-8")
                self._expect_ver(code, verifier.verify_lean_bundle, tp, check_pins=False)
                (tp / fname).write_text(orig, encoding="utf-8")

            for fname, pin_attr, code in (
                (verifier.LEAN_SRC_NAME, "PIN_LEAN", "PIN_LEAN"),
                ("lean-toolchain", "PIN_TOOLCHAIN", "PIN_TC"),
                ("lakefile.toml", "PIN_LAKEFILE", "PIN_LF"),
                ("lake-manifest.json", "PIN_MANIFEST", "PIN_MF"),
            ):
                with mock.patch.object(verifier, pin_attr, "0" * 64):
                    self._expect_ver(code, verifier.verify_lean_bundle, tp, check_pins=True)

            (tp / "Extra.lean").write_text("-- extra\n", encoding="utf-8")
            self._expect_ver("LEAN_FILE_SET", verifier.verify_lean_bundle, tp, check_pins=False)
            (tp / "Extra.lean").unlink()
            (tp / "Zzz.lean").write_text("-- extra z\n", encoding="utf-8")
            self._expect_ver("LEAN_FILE_SET", verifier.verify_lean_bundle, tp, check_pins=False)
            (tp / "Zzz.lean").unlink()
            (tp / verifier.LEAN_SRC_NAME).rename(tp / "Wrong.lean")
            self._expect_ver("LEAN_FILE_SET", verifier.verify_lean_bundle, tp, check_pins=False)
            (tp / "Wrong.lean").rename(tp / verifier.LEAN_SRC_NAME)

            orig_lean = (tp / verifier.LEAN_SRC_NAME).read_text(encoding="utf-8")
            for snippet in (
                "\nimport Std\n",
                '\ndef s : String := s!"{ "/-" }"\ntheorem fake : False := sorry\ndef t : String := s!"{ "-/" }"\n',
                '\ndef bs : String := !s!"{ "/-" }"\ntheorem fake2 : False := sorry\ndef bt : String := !s!"{ "-/" }"\n',
                "\nvariable (hFalse : False)\ninclude hFalse\ntheorem sneaky : False := hFalse\n",
                "\ntheorem st : False := by stop\n",
                "\ntheorem rx : False := _root_.sorryAx False true\n",
                "\ntheorem dn : 1 = 1 := by decide +native\n",
                "\nrun_cmd pure ()\n",
                "\nrun_elab pure ()\n",
                "\nrun_meta pure ()\n",
                "\ninstance : Inhabited Empty := ⟨sorry⟩\n",
                "\n@[default_instance] def d : Nat := 0\n",
                "\n#eval 1 + 1\n",
            ):
                (tp / verifier.LEAN_SRC_NAME).write_text(orig_lean + snippet, encoding="utf-8")
                self._expect_ver("LEAN_FORBIDDEN", verifier.verify_lean_bundle, tp, check_pins=False)

            (tp / verifier.LEAN_SRC_NAME).write_text(orig_lean + "\n#print Nat\n", encoding="utf-8")
            self._expect_ver("LEAN_RAW_PRINT", verifier.verify_lean_bundle, tp, check_pins=False)

            (tp / verifier.LEAN_SRC_NAME).write_text(
                orig_lean + "\ntheorem extra_t : 1 = 1 := rfl\n#print axioms JohnQuantOperators.extra_t\n",
                encoding="utf-8",
            )
            self._expect_ver("LEAN_THM_COUNT", verifier.verify_lean_bundle, tp, check_pins=False)

            dropped_print = "\n".join(orig_lean.splitlines()[:-1]) + "\n"
            (tp / verifier.LEAN_SRC_NAME).write_text(dropped_print, encoding="utf-8")
            self._expect_ver("LEAN_AUDIT_SET", verifier.verify_lean_bundle, tp, check_pins=False)

            dup_print = orig_lean + orig_lean.splitlines()[-1] + "\n"
            (tp / verifier.LEAN_SRC_NAME).write_text(dup_print, encoding="utf-8")
            self._expect_ver("LEAN_AUDIT_SET", verifier.verify_lean_bundle, tp, check_pins=False)

            swapped_print = "\n".join(orig_lean.splitlines()[:-1] + ["#print axioms JohnQuantOperators.wrong_thm"]) + "\n"
            (tp / verifier.LEAN_SRC_NAME).write_text(swapped_print, encoding="utf-8")
            self._expect_ver("LEAN_AUDIT_SET", verifier.verify_lean_bundle, tp, check_pins=False)

    def test_06_empirical_receipt_cross_verify_and_kill_gates(self) -> None:
        base_rec = _load_receipt_copy()
        base_raw = _load_raw_copy()
        p4 = self.cert["pillar4_theta_and_empirical_gates"]
        b30 = p4["a16_g128_b30_theta_bounds"]
        b45 = p4["a16_g128_b45_theta_bounds"]
        e8 = p4["e8_g128_theta_bounds"]
        m0 = "Qwen/Qwen2.5-0.5B"
        m1 = "Qwen/Qwen2.5-1.5B"
        a0 = "A2_b30_A16_BlockH4_hess"

        for mod, eval_fn, err_cls, pfx in (
            (oracle, oracle.evaluate_empirical_receipt, oracle.OracleError, "REC_"),
            (verifier, verifier.independent_empirical_replay, verifier.VerifyError, "V_REC_"),
        ):
            def _bad_both(mut_fn, expected_code: str) -> None:
                rec = copy.deepcopy(base_rec)
                raw = copy.deepcopy(base_raw)
                mut_fn(rec, raw)
                with self.assertRaises(err_cls) as ctx:
                    eval_fn(rec, raw, self.emp_sha, b30, b45, e8)
                self.assertEqual(ctx.exception.code, expected_code)

            # Top-level guards
            _bad_both(lambda r, _w: r.__setitem__("schema", "bad"), pfx + "SCHEMA_VAL")
            _bad_both(lambda r, _w: r.__setitem__("prereg_sha256", "0" * 64), pfx + "PREREG_PIN")
            _bad_both(lambda r, _w: r.__setitem__("empirical_results_sha256", "0" * 64), pfx + "EMP_PIN")
            _bad_both(lambda r, _w: r.__setitem__("energy_claim", "12.5 W"), pfx + "ENERGY_VAL")
            _bad_both(lambda r, _w: r.__setitem__("measured_live_model_weights", False), pfx + ("LIVE_WTS" if pfx == "REC_" else "LIVE"))
            _bad_both(lambda r, _w: r.__setitem__("gpu_vram_total_mib", 0), pfx + ("VRAM_CAP" if pfx == "REC_" else "VRAM_RNG"))
            _bad_both(lambda r, _w: r.__setitem__("gpu_vram_total_mib", 10241), pfx + ("VRAM_CAP" if pfx == "REC_" else "VRAM_RNG"))

            # Cross-verification against raw s165_empirical_results.json (MED-7)
            _bad_both(lambda _r, w: w.__setitem__("prereg_sha256", "0" * 64), "RAW_PREREG_PIN" if pfx == "REC_" else "V_RAW_PRE_PIN")
            _bad_both(lambda _r, w: w.__setitem__("energy_claim", "1 W"), "RAW_ENERGY_VAL" if pfx == "REC_" else "V_RAW_EN_VAL")
            _bad_both(lambda _r, w: w.__setitem__("measured_live_model_weights", False), "RAW_LIVE_WTS" if pfx == "REC_" else "V_RAW_LIVE")
            _bad_both(lambda _r, w: w["models"].pop(m1), "RAW_MODEL_SET" if pfx == "REC_" else "V_RAW_MSET")
            _bad_both(lambda r, _w: r["models"].pop(m1), pfx + ("MODEL_SET" if pfx == "REC_" else "MSET"))
            _bad_both(lambda _r, w: w["models"][m0].__setitem__("num_hidden_layers", 23), "XREF_LAYERS" if pfx == "REC_" else "V_XREF_LY")
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("num_hidden_layers", 23), w["models"][m0].__setitem__("num_hidden_layers", 23)), "XREF_SHAPE_L" if pfx == "REC_" else "V_XREF_EL")
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("num_hidden_layers", 25), w["models"][m0].__setitem__("num_hidden_layers", 25)), "XREF_SHAPE_L" if pfx == "REC_" else "V_XREF_EL")
            _bad_both(lambda _r, w: w["models"][m0].__setitem__("total_block_weights", 1), "XREF_WTS" if pfx == "REC_" else "V_XREF_WTS")
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("total_block_weights", 1), w["models"][m0].__setitem__("total_block_weights", 1)), "XREF_SHAPE_WTS" if pfx == "REC_" else "V_XREF_SWTS")
            _bad_both(lambda _r, w: w["models"][m0].__setitem__("group_size", 64), "XREF_GS" if pfx == "REC_" else "V_XREF_GS")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("group_size", 64), "CS08_GS" if pfx == "REC_" else "V_CS08_GS")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("group_size", 256), "CS08_GS" if pfx == "REC_" else "V_CS08_GS")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("lattice", "D4"), "CS08_LAT" if pfx == "REC_" else "V_CS08_LAT")
            _bad_both(lambda _r, w: w["models"][m0].__setitem__("peak_vram_mib", Decimal("1.0")), "XREF_PEAK" if pfx == "REC_" else "V_XREF_PK")
            _bad_both(lambda _r, w: w["models"][m0]["physical_zero_byte_fold_check"].__setitem__("extra_runtime_bytes", 4), "XREF_FOLD_B" if pfx == "REC_" else "V_XREF_FB")
            _bad_both(lambda _r, w: w["models"][m0]["physical_zero_byte_fold_check"].__setitem__("prequant_fold_mean_kl", Decimal("0.9")), "XREF_FOLD_KL" if pfx == "REC_" else "V_XREF_FKL")
            _bad_both(lambda _r, w: w["models"][m0]["arms"].pop("A0_b30_RTN_unwhitened"), "RAW_ARM_SET" if pfx == "REC_" else "V_RAW_ASET")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0]["layer_weighted_mse_rel"].pop(), "RAW_LAYER_LEN" if pfx == "REC_" else "V_RAW_LLEN")
            _bad_both(lambda _r, w: w["models"][m0]["arms"]["B1b_b30_JohnHess_BlockH4_Folded0B"]["layer_weighted_mse_rel"].pop(), "RAW_LAYER_LEN" if pfx == "REC_" else "V_RAW_LLEN")
            _bad_both(lambda _r, w: w["models"][m0]["arms"]["A3_b30_A16_BlockH4_QuadSnake_SN"]["layer_weighted_mse_rel"].pop(), "RAW_LAYER_LEN" if pfx == "REC_" else "V_RAW_LLEN")
            _bad_both(lambda _r, w: w["models"][m0]["arms"]["B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B"]["layer_weighted_mse_rel"].pop(), "RAW_LAYER_LEN" if pfx == "REC_" else "V_RAW_LLEN")
            _bad_both(lambda _r, w: w["models"][m0]["arms"]["B1b_b30_JohnHess_BlockH4_Folded0B"]["layer_weighted_mse_rel"].__setitem__(0, Decimal("0.0001")), "XREF_B1B_WINS" if pfx == "REC_" else "V_XREF_WB1B")
            _bad_both(lambda _r, w: w["models"][m0]["arms"]["B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B"]["layer_weighted_mse_rel"].__setitem__(0, Decimal("0.0001")), "XREF_B1C_WINS" if pfx == "REC_" else "V_XREF_WB1C")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("bpw_target", "bad"), "RAW_TGT" if pfx == "REC_" else "V_RAW_TGT")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("ball_bound", 999), "XREF_BB" if pfx == "REC_" else "V_XREF_BB")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("bpw_target", "4.5"), "XREF_TGT" if pfx == "REC_" else "V_XREF_TGT")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("effective_bpw", Decimal("2.5")), "XREF_EFF" if pfx == "REC_" else "V_XREF_EFF")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("syndrome_errors", 1), "XREF_SYN" if pfx == "REC_" else "V_XREF_SYN")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("out_of_ball_groups", 1), "XREF_OOB" if pfx == "REC_" else "V_XREF_OOB")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("protected_non_block_diffs", 1), "XREF_NBD" if pfx == "REC_" else "V_XREF_NBD")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("weighted_mse_rel", Decimal("0.99")), "XREF_WMSE" if pfx == "REC_" else "V_XREF_WM")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("mean_p999_abs_err", Decimal("0.99")), "XREF_P999" if pfx == "REC_" else "V_XREF_P9")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0].__setitem__("mean_p999_weighted_abs_err", Decimal("0.99")), "XREF_WP999" if pfx == "REC_" else "V_XREF_WP9")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0]["wikitext2"].__setitem__("ppl", Decimal("99.0")), "XREF_WPPL" if pfx == "REC_" else "V_XREF_WP")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0]["wikitext2"].__setitem__("mean_kl", Decimal("9.0")), "XREF_WKL" if pfx == "REC_" else "V_XREF_WK")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0]["python"].__setitem__("ppl", Decimal("99.0")), "XREF_PPPL" if pfx == "REC_" else "V_XREF_PP")
            _bad_both(lambda _r, w: w["models"][m0]["arms"][a0]["python"].__setitem__("mean_kl", Decimal("9.0")), "XREF_PKL" if pfx == "REC_" else "V_XREF_PK")

            # Synchronized receipt+raw mutations for boundary & CS-08 & kill-gate guards
            def _sync_arm(rec, raw, model, arm, r_key, r_val, w_key, w_val, sub=None):
                rec["models"][model]["arms"][arm][r_key] = r_val
                if sub is None:
                    raw["models"][model]["arms"][arm][w_key] = w_val
                else:
                    raw["models"][model]["arms"][arm][sub][w_key] = w_val

            _bad_both(lambda r, w: (r["models"][m0].__setitem__("peak_vram_kib", 0), w["models"][m0].__setitem__("peak_vram_mib", Decimal("0.0"))), pfx + "PEAK_CAP")
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("peak_vram_kib", 8192 * 1024), w["models"][m0].__setitem__("peak_vram_mib", Decimal("8192.0"))), pfx + "PEAK_CAP")
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("prequant_fold_extra_bytes", 1), w["models"][m0]["physical_zero_byte_fold_check"].__setitem__("extra_runtime_bytes", 1)), pfx + ("FOLD_ZERO" if pfx == "REC_" else "FBYTES_Z"))
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("prequant_fold_kl_nano", -1), w["models"][m0]["physical_zero_byte_fold_check"].__setitem__("prequant_fold_mean_kl", Decimal("-0.000000001"))), pfx + ("FOLD_KL_BOUND" if pfx == "REC_" else "FKL_CAP"))
            _bad_both(lambda r, w: (r["models"][m0].__setitem__("prequant_fold_kl_nano", 5000000), w["models"][m0]["physical_zero_byte_fold_check"].__setitem__("prequant_fold_mean_kl", Decimal("0.005"))), pfx + ("FOLD_KL_BOUND" if pfx == "REC_" else "FKL_CAP"))
            _bad_both(lambda r, _w: r["models"][m0]["arms"].pop(a0), pfx + ("ARM_SET" if pfx == "REC_" else "ASET"))
            _bad_both(lambda r, w: _sync_arm(r, w, m0, a0, "syndrome_errors", 1, "syndrome_errors", 1), pfx + ("SYN_ZERO" if pfx == "REC_" else "SYN_NZ"))
            _bad_both(lambda r, w: _sync_arm(r, w, m0, a0, "out_of_ball_groups", 1, "out_of_ball_groups", 1), pfx + ("OOB_ZERO" if pfx == "REC_" else "OOB_NZ"))
            _bad_both(lambda r, w: _sync_arm(r, w, m0, a0, "protected_non_block_diffs", 1, "protected_non_block_diffs", 1), pfx + ("NBD_ZERO" if pfx == "REC_" else "NBD_NZ"))

            # Non-circular CS-08 guards (MED-4)
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "A0_b30_RTN_unwhitened", "ball_bound", 1, "ball_bound", 1), "CS08_RTN_BB" if pfx == "REC_" else "V_CS08_RTN")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "A0_b30_RTN_unwhitened", "effective_bpw_micro", 3000001, "effective_bpw", Decimal("3.000001")), pfx + ("CS08_BPW" if pfx == "REC_" else "BPW"))
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "A1_b30_E8_H8_hess", "ball_bound", 421, "ball_bound", 421), "CS08_E8_BB" if pfx == "REC_" else "V_CS08_E8_BB")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "A1_b30_E8_H8_hess", "effective_bpw_micro", 2999999, "effective_bpw", Decimal("2.999999")), "CS08_E8_BPW" if pfx == "REC_" else "V_CS08_E8_BPW")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "ball_bound", 548, "ball_bound", 548), "CS08_CHEB_BB" if pfx == "REC_" else "V_CS08_CHEB_BB")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "effective_bpw_micro", 2999999, "effective_bpw", Decimal("2.999999")), "CS08_CHEB_BPW" if pfx == "REC_" else "V_CS08_CHEB_BPW")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "ball_bound", 578, "ball_bound", 578), "CS08_CSR_BB" if pfx == "REC_" else "V_CS08_CSR_BB")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "effective_bpw_micro", 2994000, "effective_bpw", Decimal("2.994")), "CS08_CSR_BPW" if pfx == "REC_" else "V_CS08_CSR_BPW")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "ball_bound", 547, "ball_bound", 547), "CS08_B3B_BB" if pfx == "REC_" else "V_CS08_B3B_BB")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "effective_bpw_micro", 2997000, "effective_bpw", Decimal("2.997")), "CS08_B3B_BPW" if pfx == "REC_" else "V_CS08_B3B_BPW")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, a0, "ball_bound", 652, "ball_bound", 652), "CS08_STD_BB" if pfx == "REC_" else "V_CS08_STD_BB")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, a0, "effective_bpw_micro", 2999999, "effective_bpw", Decimal("2.999999")), "CS08_STD_BPW" if pfx == "REC_" else "V_CS08_STD_BPW")

            # KG1 / KG2 / B3 verdict tamper guards (HIGH-1, MED-3)
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B", "weighted_mse_nano", 1000000, "weighted_mse_rel", Decimal("0.001")), "VERDICT_KG1_B1C" if pfx == "REC_" else "V_KG1_B1C")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B", "wikitext2_ppl_micro", 15000000, "ppl", Decimal("15.0"), sub="wikitext2"), "VERDICT_KG1_B1C" if pfx == "REC_" else "V_KG1_B1C")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B1b_b30_JohnHess_BlockH4_Folded0B", "weighted_mse_nano", 1000000, "weighted_mse_rel", Decimal("0.001")), "VERDICT_KG1_CNT" if pfx == "REC_" else "V_KG1_CNT")
            _bad_both(lambda r, w: _sync_arm(r, w, m1, "B1b_b30_JohnHess_BlockH4_Folded0B", "wikitext2_ppl_micro", 16000000, "ppl", Decimal("16.0"), sub="wikitext2"), "VERDICT_KG1_CNT" if pfx == "REC_" else "V_KG1_CNT")

            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B", "p999_abs_err_nano", 1000000, "mean_p999_abs_err", Decimal("0.001")), "VERDICT_KG2_P999" if pfx == "REC_" else "V_KG2_P999")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "p999_abs_err_nano", 1000000, "mean_p999_abs_err", Decimal("0.001")), "VERDICT_KG2_P999" if pfx == "REC_" else "V_KG2_P999")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B", "p999_abs_err_nano", 1000000, "mean_p999_abs_err", Decimal("0.001")), "VERDICT_KG2_P999" if pfx == "REC_" else "V_KG2_P999")
            _bad_both(
                lambda r, w: (
                    _sync_arm(r, w, m0, "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B", "wikitext2_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="wikitext2"),
                    _sync_arm(r, w, m0, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "wikitext2_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="wikitext2"),
                    _sync_arm(r, w, m1, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "wikitext2_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="wikitext2"),
                    _sync_arm(r, w, m1, "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B", "wikitext2_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="wikitext2"),
                ),
                "VERDICT_KG2_PPL" if pfx == "REC_" else "V_KG2_PPL",
            )

            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "wikitext2_ppl_micro", 99000000, "ppl", Decimal("99.0"), sub="wikitext2"), "VERDICT_B3" if pfx == "REC_" else "V_B3_PART")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "python_ppl_micro", 99000000, "ppl", Decimal("99.0"), sub="python"), "VERDICT_B3" if pfx == "REC_" else "V_B3_PART")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "python_ppl_micro", 5870000, "ppl", Decimal("5.87"), sub="python"), "VERDICT_B3" if pfx == "REC_" else "V_B3_PART")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, a0, "python_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="python"), "VERDICT_B3" if pfx == "REC_" else "V_B3_PART")
            _bad_both(lambda r, w: _sync_arm(r, w, m0, "A3_b30_A16_BlockH4_QuadSnake_SN", "python_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="python"), "VERDICT_B3" if pfx == "REC_" else "V_B3_PART")

            _bad_both(
                lambda r, w: (
                    w["models"][m0]["arms"]["B3a_b30_PolarCap_CSR025_BlockH4_Matched"].__setitem__("csr_entries", 961000),
                    _sync_arm(r, w, m0, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "effective_bpw_micro", 3000001, "effective_bpw", Decimal("3.000001")),
                ),
                pfx + ("CS08_BPW" if pfx == "REC_" else "BPW"),
            )
            _bad_both(
                lambda r, w: (
                    w["models"][m0]["arms"]["B3b_b30_PolarCap_BlockSplit_1over19_Matched"].__setitem__("polar_hi_groups", 200000),
                    _sync_arm(r, w, m0, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "effective_bpw_micro", 3027525, "effective_bpw", Decimal("3.027525")),
                ),
                pfx + ("CS08_BPW" if pfx == "REC_" else "BPW"),
            )

            # Exact upper/lower inclusive boundary PASS check (VRAM=10240, fold KL=0, bpw=3000000, CSR=960960, 1 KG2 falsify hit)
            rec_b = copy.deepcopy(base_rec)
            raw_b = copy.deepcopy(base_raw)
            rec_b["gpu_vram_total_mib"] = 10240
            rec_b["models"][m0]["prequant_fold_kl_nano"] = 0
            raw_b["models"][m0]["physical_zero_byte_fold_check"]["prequant_fold_mean_kl"] = Decimal("0")
            _sync_arm(rec_b, raw_b, m0, "A0_b30_RTN_unwhitened", "effective_bpw_micro", 3000000, "effective_bpw", Decimal("3.000000"))
            raw_b["models"][m0]["arms"]["B3a_b30_PolarCap_CSR025_BlockH4_Matched"]["csr_entries"] = 960960
            _sync_arm(rec_b, raw_b, m0, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "effective_bpw_micro", 3000000, "effective_bpw", Decimal("3.000000"))
            raw_b["models"][m0]["arms"]["B3b_b30_PolarCap_BlockSplit_1over19_Matched"]["polar_hi_groups"] = 151481
            _sync_arm(rec_b, raw_b, m0, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "effective_bpw_micro", 3000000, "effective_bpw", Decimal("3.000000"))
            for mm, aa in (
                (m0, "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B"),
                (m0, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched"),
                (m0, "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B"),
                (m1, "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B"),
                (m1, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched"),
            ):
                _sync_arm(rec_b, raw_b, mm, aa, "wikitext2_ppl_micro", 1000000, "ppl", Decimal("1.0"), sub="wikitext2")
            res_b = eval_fn(rec_b, raw_b, self.emp_sha, b30, b45, e8)
            self.assertEqual(res_b["kg2_ppl_degraded_without_p999_hits"], 1)
            self.assertEqual(res_b["overall_empirical_verdict"], "KG1_PARTIAL_KG2_FALSIFIED_B3_PARTIAL")

    def test_07_oracle_pins_zone_b_write_and_cli_gates(self) -> None:
        # Pin mismatches in oracle & verifier
        for pin_attr, code in (
            ("PINNED_LEAN_SHA256", "ORA_PIN_LEAN"),
            ("PINNED_TOOLCHAIN_SHA256", "ORA_PIN_TC"),
            ("PINNED_LAKEFILE_SHA256", "ORA_PIN_LF"),
            ("PINNED_MANIFEST_SHA256", "ORA_PIN_MF"),
            ("PINNED_RECEIPT_SHA256", "ORA_PIN_REC"),
            ("PINNED_EMPIRICAL_RESULTS_SHA256", "ORA_PIN_EMP"),
        ):
            with mock.patch.object(oracle, pin_attr, "0" * 64):
                self._expect_ora(code, oracle.build_certificate, BASE)

        with mock.patch.object(verifier, "PIN_RECEIPT", "0" * 64):
            self._expect_ver("PIN_RECEIPT", verifier.build_verified_payload, BASE)
        with mock.patch.object(verifier, "PIN_EMP_RAW", "0" * 64):
            self._expect_ver("PIN_EMP_RAW", verifier.build_verified_payload, BASE)

        # CLI refusal checks (no abbreviated flags, single-line JSON errors on bad args & traversal)
        for bad_args in (["--bogus"], ["--json=1"], ["--wr"], ["--js"]):
            rc, _out, err = _capture_fds(oracle.main, bad_args)
            self.assertEqual(rc, 2)
            self.assertEqual(err.count("\n"), 1)
            self.assertEqual(json.loads(err)["code"], "ORA_CLI_ARGS")

        for bad_args in (["--unknown"], ["--json=1"], ["--live"], ["--ch"]):
            rc, _out, err = _capture_fds(verifier.main, bad_args)
            self.assertEqual(rc, 2)
            self.assertEqual(err.count("\n"), 1)
            self.assertEqual(json.loads(err)["code"], "BAD_CLI_ARGS")

        rc_tr, _o_tr, err_tr = _capture_fds(verifier.main, ["--check", ".."])
        self.assertEqual(rc_tr, 2)
        self.assertEqual(json.loads(err_tr)["code"], "BAD_PATH_TRAV")

        rc_c, _o, err_c = _capture_fds(verifier.main, ["--check", "--standalone"])
        self.assertEqual(rc_c, 2)
        self.assertEqual(json.loads(err_c)["code"], "MODE_CONFLICT")

        rc, out, err = _capture_fds(oracle.main, [])
        self.assertEqual(rc, 0)
        self.assertIn("PASS ", out)
        self.assertEqual(err, "")

        # HIGH-2: Live Zone-B --write symlink-dir, symlink-file, hardlink, dir, FIFO, and atomic overwrite tests
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            real_dir = tp / "real_dir"
            real_dir.mkdir()
            victim = real_dir / "victim.json"
            victim.write_bytes(b"KEEP_INTACT\n")

            link_dir = tp / "link_dir"
            os.symlink(real_dir, link_dir)
            rc_ld, _o, err_ld = _capture_fds(oracle.main, ["--write", str(link_dir / "victim.json")])
            self.assertEqual(rc_ld, 2)
            self.assertEqual(json.loads(err_ld)["code"], "ORA_PARENT_OPEN")
            self.assertEqual(victim.read_bytes(), b"KEEP_INTACT\n")

            sym_file = tp / "sym_victim.json"
            os.symlink(victim, sym_file)
            self._expect_ora("ORA_WRITE_SYMLINK", oracle.write_cert_file, sym_file, self.cert)
            self.assertEqual(victim.read_bytes(), b"KEEP_INTACT\n")

            hardlink = tp / "hardlink.json"
            os.link(victim, hardlink)
            rc_hl, _o, err_hl = _capture_fds(oracle.main, ["--write", str(hardlink)])
            self.assertEqual(rc_hl, 2)
            self.assertEqual(json.loads(err_hl)["code"], "ORA_WRITE_HARDLINK")
            self.assertEqual(victim.read_bytes(), b"KEEP_INTACT\n")

            self._expect_ora("ORA_WRITE_NOT_REG", oracle.write_cert_file, real_dir, self.cert)

            fifo = tp / "pipe.json"
            os.mkfifo(fifo)
            self._expect_ora("ORA_WRITE_NOT_REG", oracle.write_cert_file, fifo, self.cert)

            # Low-level write_cert_file OS failure guards
            out_cert = tp / "cert.json"
            with mock.patch.object(os, "lstat", side_effect=PermissionError("denied")):
                self._expect_ora("ORA_WRITE_LSTAT", oracle.write_cert_file, out_cert, self.cert)
            orig_open = os.open

            def _open_excl_fail(path_arg, flags, mode=0o777, *, dir_fd=None):
                if flags & os.O_EXCL:
                    raise FileExistsError("exists")
                return orig_open(path_arg, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(os, "open", side_effect=_open_excl_fail):
                self._expect_ora("ORA_WRITE_TMP_OPEN", oracle.write_cert_file, out_cert, self.cert)
            with mock.patch.object(os, "write", return_value=0):
                self._expect_ora("ORA_WRITE_SHORT", oracle.write_cert_file, out_cert, self.cert)

            # Successful atomic write -> overwrite existing regular file -> verifier --check -> tamper check
            rc_w, _o_w, err_w = _capture_fds(oracle.main, ["--write", str(out_cert)])
            self.assertEqual(rc_w, 0)
            self.assertEqual(err_w, "")
            self.assertTrue(out_cert.exists())
            oracle.write_cert_file(out_cert, self.cert)

            rc_v, out_v, err_v = _capture_fds(verifier.main, ["--check", str(out_cert)])
            self.assertEqual(rc_v, 0)
            self.assertIn("PASS ", out_v)
            self.assertEqual(err_v, "")

            tampered = json.loads(out_cert.read_text(encoding="utf-8"))
            tampered["pillar1_john_geometry"] = {}
            out_cert.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rc_t, _out_t, err_t = _capture_fds(verifier.main, ["--check", str(out_cert)])
            self.assertEqual(rc_t, 2)
            self.assertIn("CERT_KEYS", err_t)

        rc_s, out_s, err_s = _capture_fds(verifier.main, ["--standalone"])
        self.assertEqual(rc_s, 0)
        self.assertIn("PASS ", out_s)
        self.assertEqual(err_s, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
