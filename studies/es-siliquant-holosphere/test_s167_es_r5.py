#!/usr/bin/env python3
"""Track S167 (EXP-Q51a/b/c) Strict R5 Refusal, Boundary, Zone-B I/O,
--write O_NOFOLLOW Symlink/Hardlink, UTF-8 Fail-Closed, bool/int Strict JSON,
Disk Results Reconciliation, Programmatic Verdict Branches, Lean R12/R13b
Hardening, CLI Strictness, and Benchmark Pure-Python Unit Test Suite.

Covers s167_es_oracle.py, verify_s167_es.py, and pure-Python helpers in
s167_es_live_bench.py. Zero bare asserts, zero skips, zero float literals.
"""
from __future__ import annotations

import ast
import copy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_HERE = Path(__file__).parent
BASE = _HERE if _HERE.is_absolute() else (Path(os.getcwd()) / _HERE)
sys.path.insert(0, str(BASE))

import s167_es_live_bench as bench  # noqa: E402
import s167_es_oracle as oracle  # noqa: E402
import verify_s167_es as verifier  # noqa: E402


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
        oracle.PREREG_FILE_NAME,
        oracle.BENCH_FILE_NAME,
        oracle.RESULTS_FILE_NAME,
        oracle.RECEIPT_FILE_NAME,
        oracle.CERT_FILE_NAME,
    ):
        shutil.copyfile(BASE / fn, dst / fn)


def _good_lean_runner(_pinned: dict[str, bytes]) -> dict[str, object]:
    src = (BASE / verifier.LEAN_SRC_NAME).read_text(encoding="utf-8")
    thms = verifier._PUB_THM_RE.findall(verifier.strip_lean_comments(src))
    lines: list[str] = []
    for idx, t in enumerate(thms):
        if t == "folded_scale_zero_byte_es_invariance_universal":
            lines.append(f"'EvolutionOperators.{t}' depends on axioms: [propext]")
        elif idx == 1:
            lines.append(f"'EvolutionOperators.{t}' depends on axioms: [propext,")
            lines.append(" Quot.sound]")
        else:
            lines.append(f"'EvolutionOperators.{t}' depends on axioms: [propext, Quot.sound]")
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


def _load_results_q_copy() -> dict[str, object]:
    return json.loads((BASE / oracle.RESULTS_FILE_NAME).read_text(encoding="utf-8"), parse_float=Fraction)


def _make_passing_receipt() -> dict[str, object]:
    """Build a synthetic valid receipt where EVERY EXP-Q51a & EXP-Q51b gate PASSES (CONFIRMED)."""
    rec = _load_receipt_copy()
    qa = rec["exp_q51a"]
    qa["peak_vram_05b_kib"] = 6815743  # exact < 6815744 boundary
    qa["peak_vram_15b_kib"] = 6815743  # exact < 6815744 boundary
    for row in qa["arms"]["C1_VsonicV_BF16_SharedSeed"]:
        row["test_exact_acc_micro"] = 50000
    for row in qa["arms"]["C2_QES_Unblocked_INT4_EF"]:
        row["test_exact_acc_micro"] = 60000
    for row in qa["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"]:
        row["test_exact_acc_micro"] = 80000  # exact +2.00 pp across all 7 seeds -> p_holm = 1/32 < 0.05!
        row["syndrome_errors"] = 0
        row["oob_groups"] = 0

    qb = rec["exp_q51b"]
    grid = qb["vram_o1_scaling_grid"]
    for idx, gk in enumerate(oracle.VRAM_GRID_KEYS):
        grid[gk]["es_peak_vram_kib"] = 523264 + (1023 if idx == 0 else 0)  # max=524287 < 524288, spread=1023 < 1024
        grid[gk]["es_max_su2_shell_err_nano"] = 9999  # exact < 10000 boundary
    base_ood = qb["base_ood_8x_acc_micro"]
    base_bpb = qb["base_wikitext2_bpb_micro"]
    for row in qb["arms"]["ArmC_HoloSphere_Cayley_ES"]:
        row["max_su2_shell_err_nano"] = 9999
        row["su2_shell_err_overflow"] = False
        row["ood_8x_exact_acc_micro"] = base_ood + 100000  # exact +10.00 pp boundary
        row["wikitext2_bpb_finite"] = True
        row["wikitext2_bpb_micro"] = base_bpb + 20000  # exact +0.02 BPB drift boundary
        row["a16_syndrome_errors"] = 0
    return rec


class TestS167R5(unittest.TestCase):
    cert: dict[str, object] = {}

    @classmethod
    def setUpClass(cls) -> None:
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
        ver_tree = ast.parse((BASE / "verify_s167_es.py").read_text(encoding="utf-8"))
        ora_tree = ast.parse((BASE / "s167_es_oracle.py").read_text(encoding="utf-8"))
        tst_tree = ast.parse((BASE / "test_s167_es_r5.py").read_text(encoding="utf-8"))
        bn_tree = ast.parse((BASE / "s167_es_live_bench.py").read_text(encoding="utf-8"))
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
        for node in ast.walk(tst_tree):
            self.assertNotIsInstance(node, ast.Assert)
            if isinstance(node, ast.Constant):
                self.assertNotIsInstance(node.value, float)
        for node in ast.walk(bn_tree):
            self.assertNotIsInstance(node, ast.Assert)

    def test_02_exact_mathematical_pillars_and_boundaries(self) -> None:
        self.assertEqual(self.cert["schema"], "s167.q51.es_siliquant_holosphere.v2")
        self.assertEqual(self.cert["energy_claim"], "UNCONTROLLED_LAPTOP_NOT_CLAIMED")
        self.assertEqual(self.cert["lean"]["public_theorems"], 21)

        p1 = self.cert["pillar1_antithetic_cancellation"]
        self.assertEqual(p1["vec4_antithetic_diff"], "201/8000")
        self.assertEqual(p1["vec4_antithetic_sum"], "472/35")
        self.assertEqual(p1["onesided_bias_excess_sq2"], "111392/1225")

        p2 = self.cert["pillar2_orthogonal_hadamard_h4"]
        self.assertEqual(p2["h4_parseval_sq"], 816)
        self.assertEqual(p2["h4_input_sq"], 204)
        self.assertEqual(p2["h4_exact_recovery_err_sq"], 0)
        choro = p2["choromanski_orth_over_iid_mse"]
        self.assertEqual(choro["d4_n4"], "2/5")
        self.assertEqual(choro["d16_n16"], "2/17")
        self.assertEqual(choro["d128_n16"], "38/43")
        self.assertEqual(choro["d128_n30"], "100/129")

        p3 = self.cert["pillar3_heat_kernel_goodhart"]
        self.assertEqual(p3["spike_over_plateau_ratio"]["r1"], "1/100")
        self.assertEqual(p3["spike_over_plateau_ratio"]["r4"], "1/100000000")
        self.assertEqual(p3["pascal_center_spike_masses_m0_to_m6"][-1], "231/1024")
        self.assertIs(p3["pascal_plateau_conservation_exact"], True)

        p4 = self.cert["pillar4_bf16_alkashi_cs08"]
        self.assertEqual(p4["bf16_ulp_at_1"], "1/128")
        self.assertEqual(p4["bf16_half_ulp_at_1"], "1/256")
        self.assertEqual(p4["vsonic_alpha_over_half_ulp"], "16/125")
        self.assertEqual(p4["bf16_canonical_g7_moves"], 0)
        self.assertEqual(p4["bf16_canonical_g8_moves"], 1)
        self.assertEqual(p4["alkashi_q64_16x8_weight_advance"], 2)
        self.assertEqual(p4["alkashi_q64_16x8_residual"], 0)
        self.assertEqual(p4["cs08_es_over_drgrpo_flops"], "1/3")
        self.assertEqual(p4["cs08_es_over_grpo_flops"], "1/4")
        self.assertEqual(p4["cs08_hbm_ratio_n16"], "32/3")
        self.assertEqual(p4["cs08_hbm_ratio_n30"], "20")
        self.assertEqual(p4["folded_scale_dot_exact"], "211/112")
        self.assertEqual(p4["folded_scale_extra_bytes"], 0)

        p5 = self.cert["pillar5_cayley_su2_so3_manifold"]
        self.assertEqual(p5["cayley_su2_denom"], 2529)
        self.assertEqual(p5["cayley_su2_norm_sq"], "1")
        self.assertEqual(p5["cayley_so3_det"], "1")
        self.assertEqual(p5["cayley_so3_m00"], "2489/2529")
        self.assertEqual(p5["cayley_so3_m01"], "-224/2529")
        self.assertEqual(p5["euclidean_additive_shell_err"], "1/100")

        # Parameterized oracle & verifier math helper boundary tests
        self.assertEqual(oracle.check_orth_rows_gram([(1, 0), (0, 1)], 2, 1), 2)
        self._expect_ora("P2_GRAM_DIM", oracle.check_orth_rows_gram, [(1, 0)], 2, 1)
        self._expect_ora("P2_H4_GRAM", oracle.check_orth_rows_gram, [(1, 1), (1, 1)], 2, 2)
        self.assertEqual(verifier.verify_orth_gram([[1, 0], [0, 1]], 2, 1), 2)
        self._expect_ver("V_P2_DIM", verifier.verify_orth_gram, [[1, 0]], 2, 1)
        self._expect_ver("V_P2_ORTH", verifier.verify_orth_gram, [[1, 1], [1, 1]], 2, 2)

        self.assertEqual(oracle.run_alkashi_centered_ef(1, 0, -1, [2]), (0, 1 - 0) if False else (1, -1))
        self.assertEqual(oracle.run_alkashi_centered_ef(4, 1, 3, [5]), (2, 0))
        self._expect_ora("P4_H_POS", oracle.run_alkashi_centered_ef, 0, 0, 0, [1])
        self._expect_ora("P4_E0_LO", oracle.run_alkashi_centered_ef, 4, 0, -5, [1])
        self._expect_ora("P4_E0_HI", oracle.run_alkashi_centered_ef, 4, 0, 4, [1])
        self.assertEqual(oracle.bf16_grid_round_int(2000, 2896), 2000)
        self.assertEqual(oracle.bf16_grid_round_int(2000, 3000), 4000)
        self.assertEqual(oracle.bf16_grid_round_int(2000, 1000), 2000)
        self.assertEqual(oracle.bf16_grid_round_int(2000, 999), 0)
        self.assertEqual(oracle.verify_reciprocal_fold_dot(((Fraction(2), Fraction(3), Fraction(5), Fraction(1, 5)),)), Fraction(6))
        self._expect_ora("P4_FOLD_INV", oracle.verify_reciprocal_fold_dot, ((Fraction(2), Fraction(3), Fraction(5), Fraction(1, 4)),))
        self.assertEqual(verifier.check_folded_scale_invariance(((Fraction(2), Fraction(3), Fraction(5), Fraction(1, 5)),)), Fraction(6))
        self._expect_ver("V_P4_FOLD", verifier.check_folded_scale_invariance, ((Fraction(2), Fraction(3), Fraction(5), Fraction(1, 4)),))

        self.assertEqual(oracle._exact_perm_p_fraction([5]), Fraction(1, 1))
        self._expect_ora("ORA_PERM_EMPTY", oracle._exact_perm_p_fraction, [])
        self.assertEqual(verifier._perm_p_exact([5]), Fraction(1, 1))
        self._expect_ver("V_PERM_N", verifier._perm_p_exact, [])

        # Holm-Bonferroni pair tests (both p0 <= p1 and p0 > p1 and cap at 1)
        for hfn in (oracle._holm_adjust_pair, verifier._holm_2):
            self.assertEqual(hfn(Fraction(1, 64), Fraction(1, 4)), (Fraction(1, 32), Fraction(1, 4)))
            self.assertEqual(hfn(Fraction(1, 4), Fraction(1, 64)), (Fraction(1, 4), Fraction(1, 32)))
            self.assertEqual(hfn(Fraction(1, 8), Fraction(3, 16)), (Fraction(1, 4), Fraction(1, 4)))
            self.assertEqual(hfn(Fraction(3, 4), Fraction(3, 4)), (Fraction(1, 1), Fraction(1, 1)))

        swap_m = [
            [Fraction(0), Fraction(1), Fraction(0)],
            [Fraction(1), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(2)],
        ]
        inv_s = verifier._mat3_inv_exact(swap_m)
        self.assertEqual(inv_s[0][1], Fraction(1))
        self.assertEqual(inv_s[2][2], Fraction(1, 2))
        sing_m = [
            [Fraction(0), Fraction(1), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(1)],
            [Fraction(0), Fraction(0), Fraction(0)],
        ]
        self._expect_ver("V_MAT_SING", verifier._mat3_inv_exact, sing_m)

    def test_03_zone_b_io_write_nofollow_and_json_oracle(self) -> None:
        self._expect_ora("ORA_PATH_TYPE", oracle._split_clean_parts, 123)
        self._expect_ora("ORA_EMPTY_PATH", oracle._split_clean_parts, "")
        self._expect_ora("ORA_NUL_PATH", oracle._read_regular_bytes, Path("bad\x00name"))
        self._expect_ora("ORA_ROOT_PATH", oracle._read_regular_bytes, Path("/"))
        self._expect_ora("ORA_DOTDOT_PATH", oracle._read_regular_bytes, Path("/tmp/../etc/passwd"))
        self._expect_ora("ORA_FILE_OPEN", oracle._read_regular_bytes, Path("/nonexistent_s167_probe"))

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            self._expect_ora("ORA_PARENT_OPEN", oracle._read_regular_bytes, tp / "missing_dir" / "a.txt")
            self._expect_ora("ORA_NOT_REG", oracle._read_regular_bytes, tp)

            real_f = tp / "real.txt"
            real_f.write_bytes(b"hello\n")
            hard_f = tp / "hard.txt"
            os.link(real_f, hard_f)
            self._expect_ora("ORA_HARDLINK", oracle._read_regular_bytes, real_f)
            os.unlink(hard_f)
            self.assertEqual(oracle._read_regular_bytes(real_f), b"hello\n")

            sym_f = tp / "sym.txt"
            os.symlink(real_f, sym_f)
            self._expect_ora("ORA_FILE_OPEN", oracle._read_regular_bytes, sym_f)

            victim = tp / "victim.json"
            victim.write_bytes(b'{"victim":1}\n')
            sub = tp / "sub"
            sub.mkdir()
            sym_dir = tp / "sym_dir"
            os.symlink(sub, sym_dir)

            rc_trav, _o, err_trav = _capture_fds(oracle.main, ["--write", str(sym_dir) + "/../victim.json"])
            self.assertEqual(rc_trav, 2)
            self.assertIn("ORA_DOTDOT_PATH", err_trav)
            self.assertEqual(victim.read_bytes(), b'{"victim":1}\n')

            rc_sd, _o, err_sd = _capture_fds(oracle.main, ["--write", str(sym_dir / "out.json")])
            self.assertEqual(rc_sd, 2)
            self.assertIn("ORA_PARENT_OPEN", err_sd)

            sym_leaf = tp / "sym_leaf.json"
            os.symlink(victim, sym_leaf)
            rc_sl, _o, err_sl = _capture_fds(oracle.main, ["--write", str(sym_leaf)])
            self.assertEqual(rc_sl, 2)
            self.assertIn("ORA_WRITE_NOT_REG", err_sl)
            self.assertEqual(victim.read_bytes(), b'{"victim":1}\n')

            hard_v = tp / "hard_v.json"
            os.link(victim, hard_v)
            rc_hl, _o, err_hl = _capture_fds(oracle.main, ["--write", str(victim)])
            self.assertEqual(rc_hl, 2)
            self.assertIn("ORA_WRITE_HARDLINK", err_hl)
            os.unlink(hard_v)

            out_reg = tp / "ok.json"
            out_reg.write_bytes(b"{}\n")
            rc_ow, _o, err_ow = _capture_fds(oracle.main, ["--write", str(out_reg)])
            self.assertEqual(rc_ow, 0)
            self.assertEqual(err_ow, "")

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
        rat_obj = oracle._parse_strict_json_bytes(b'{"x":1.5,"y":1e-2}\n', allow_rationals=True)
        self.assertEqual(rat_obj["x"], Fraction(3, 2))
        self.assertEqual(rat_obj["y"], Fraction(1, 100))
        self.assertIs(type(rat_obj["x"]), Fraction)

        self._expect_ora("ORA_JSON_DEPTH_END", oracle._parse_strict_json_bytes, b'{"x":1\n')
        self._expect_ora("ORA_JSON_STR_END", oracle._parse_strict_json_bytes, b'"unterm\n')
        self._expect_ora("ORA_DUP_KEY", oracle._parse_strict_json_bytes, b'{"a":1,"a":2}\n')
        self._expect_ora("ORA_BAD_CONST", oracle._parse_strict_json_bytes, b'{"a":NaN}\n')
        self._expect_ora("ORA_JSON_PARSE", oracle._parse_strict_json_bytes, b'{"a":}\n')
        self._expect_ora("ORA_NOT_DICT", oracle._parse_strict_json_bytes, b"[1,2]\n")
        self.assertEqual(oracle._parse_strict_json_bytes(b'{"k":"a\\"b"}\n'), {"k": 'a"b'})

        self._expect_ora("K_INT", oracle._req_int, {}, "a", "K_INT")
        self._expect_ora("K_INT", oracle._req_int, {"a": True}, "a", "K_INT")
        self._expect_ora("K_BOOL", oracle._req_bool, {}, "a", "K_BOOL")
        self._expect_ora("K_BOOL", oracle._req_bool, {"a": 1}, "a", "K_BOOL")
        self._expect_ora("K_STR", oracle._req_str, {}, "a", "K_STR")
        self._expect_ora("K_STR", oracle._req_str, {"a": 1}, "a", "K_STR")
        self._expect_ora("K_DICT", oracle._req_dict, {}, "a", "K_DICT")
        self._expect_ora("K_DICT", oracle._req_dict, {"a": []}, "a", "K_DICT")
        self._expect_ora("K_LIST", oracle._req_list, {}, "a", "K_LIST")
        self._expect_ora("K_LIST", oracle._req_list, {"a": {}}, "a", "K_LIST")
        self.assertEqual(oracle._req_rat({"a": 5}, "a", "K_RAT"), Fraction(5, 1))
        self.assertEqual(oracle._req_rat({"a": Fraction(3, 2)}, "a", "K_RAT"), Fraction(3, 2))
        self._expect_ora("K_RAT", oracle._req_rat, {}, "a", "K_RAT")
        self._expect_ora("K_RAT", oracle._req_rat, {"a": True}, "a", "K_RAT")

    def test_04_zone_b_io_write_nofollow_bool_int_and_utf8_verifier(self) -> None:
        self._expect_ver("BAD_PATH_TYPE", verifier.read_strict_bytes, 123)
        self._expect_ver("BAD_PATH_EMPTY", verifier.read_strict_bytes, "")
        self._expect_ver("BAD_PATH_NUL", verifier.read_strict_bytes, "bad\x00path")
        self._expect_ver("ROOT_PATH", verifier.read_strict_bytes, Path("/"))
        self._expect_ver("DOTDOT_PATH", verifier.read_strict_bytes, "/tmp/../etc/hosts")
        self._expect_ver("FILE_OPEN", verifier.read_strict_bytes, Path("/nonexistent_s167_probe"))

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            self._expect_ver("PARENT_OPEN", verifier.read_strict_bytes, tp / "missing_dir" / "a.txt")
            self._expect_ver("NOT_REG_FILE", verifier.read_strict_bytes, tp)

            target = tp / "real.txt"
            target.write_bytes(b"hello\n")
            hard_t = tp / "hard.txt"
            os.link(target, hard_t)
            self._expect_ver("HARDLINK_READ", verifier.read_strict_bytes, target)
            os.unlink(hard_t)

            sym = tp / "link.txt"
            os.symlink(target, sym)
            self._expect_ver("FILE_OPEN", verifier.read_strict_bytes, sym)

            victim = tp / "victim.json"
            victim.write_bytes(b'{"victim":true}\n')
            sub = tp / "sub"
            sub.mkdir()
            sym_dir = tp / "sym_dir"
            os.symlink(sub, sym_dir)

            rc1, _o, err1 = _capture_fds(verifier.main, ["--write", str(sym_dir) + "/../victim.json"])
            self.assertEqual(rc1, 2)
            self.assertIn("DOTDOT_PATH", err1)
            self.assertEqual(victim.read_bytes(), b'{"victim":true}\n')

            rc2, _o, err2 = _capture_fds(verifier.main, ["--write", str(sym_dir / "out.json")])
            self.assertEqual(rc2, 2)
            self.assertIn("PARENT_OPEN", err2)

            sym_leaf = tp / "sym_leaf.json"
            os.symlink(victim, sym_leaf)
            rc3, _o, err3 = _capture_fds(verifier.main, ["--write", str(sym_leaf)])
            self.assertEqual(rc3, 2)
            self.assertIn("WRITE_NOT_REG", err3)
            self.assertEqual(victim.read_bytes(), b'{"victim":true}\n')

            hard_v = tp / "hard_v.json"
            os.link(victim, hard_v)
            rc4, _o, err4 = _capture_fds(verifier.main, ["--write", str(victim)])
            self.assertEqual(rc4, 2)
            self.assertIn("WRITE_HARDLINK", err4)
            os.unlink(hard_v)

            out_c = tp / "written_cert.json"
            rc_w, out_w, err_w = _capture_fds(verifier.main, ["--write", str(out_c)])
            self.assertEqual(rc_w, 0)
            self.assertIn("PASS ", out_w)
            self.assertEqual(err_w, "")

            rc_ok, _o, err_ok = _capture_fds(verifier.main, ["--check", str(out_c)])
            self.assertEqual(rc_ok, 0)
            self.assertEqual(err_ok, "")

            spoof_int_bool = json.loads(out_c.read_text(encoding="utf-8"))
            spoof_int_bool["lean"]["lean_files"] = True
            spoof_p = tp / "spoof_bool_int.json"
            spoof_p.write_text(json.dumps(spoof_int_bool, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rc_sp, _o, err_sp = _capture_fds(verifier.main, ["--check", str(spoof_p)])
            self.assertEqual(rc_sp, 2)
            self.assertIn("CERT_TYPE", err_sp)

            spoof_bool_int = json.loads(out_c.read_text(encoding="utf-8"))
            spoof_bool_int["pillar6_empirical_live_gates"]["gates"]["exp_q51a"]["zero_syndrome_errors"] = 1
            spoof_p2 = tp / "spoof_int_bool.json"
            spoof_p2.write_text(json.dumps(spoof_bool_int, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rc_sp2, _o, err_sp2 = _capture_fds(verifier.main, ["--check", str(spoof_p2)])
            self.assertEqual(rc_sp2, 2)
            self.assertIn("CERT_TYPE", err_sp2)

            bad_utf8 = tp / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe\xfd\n")
            rc_u8, _o, err_u8 = _capture_fds(verifier.main, ["--check", str(bad_utf8)])
            self.assertEqual(rc_u8, 2)
            self.assertIn("BAD_UTF8", err_u8)

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
        rat_v = verifier.parse_strict_json('{"x":1.5,"y":1E-2}\n', allow_rationals=True)
        self.assertEqual(rat_v["x"], Fraction(3, 2))
        self.assertEqual(rat_v["y"], Fraction(1, 100))

        self._expect_ver("JSON_EOF_DEPTH", verifier.parse_strict_json, '{"x":1\n')
        self._expect_ver("JSON_EOF_STR", verifier.parse_strict_json, '"unterm\n')
        self._expect_ver("DUP_KEY", verifier.parse_strict_json, '{"a":1,"a":2}\n')
        self._expect_ver("BAD_CONST", verifier.parse_strict_json, '{"a":Infinity}\n')
        self._expect_ver("BAD_JSON", verifier.parse_strict_json, '{"a":}\n')
        self._expect_ver("NOT_OBJECT", verifier.parse_strict_json, "[1,2]\n")
        self.assertEqual(verifier.parse_strict_json('{"k":"x\\"y"}\n'), {"k": 'x"y'})

        self._expect_ver("CERT_TYPE", verifier.verify_exact_tree, True, 1)
        self._expect_ver("CERT_KEYS", verifier.verify_exact_tree, {"a": 1}, {"a": 1, "b": 2})
        self._expect_ver("CERT_LEN", verifier.verify_exact_tree, [1], [1, 2])
        self._expect_ver("CERT_VAL", verifier.verify_exact_tree, {"a": 1}, {"a": 2})

        self._expect_ver("N_I", verifier._need_int, {}, "a", "N_I")
        self._expect_ver("N_I", verifier._need_int, {"a": True}, "a", "N_I")
        self._expect_ver("N_B", verifier._need_bool, {}, "a", "N_B")
        self._expect_ver("N_B", verifier._need_bool, {"a": 1}, "a", "N_B")
        self._expect_ver("N_S", verifier._need_str, {}, "a", "N_S")
        self._expect_ver("N_S", verifier._need_str, {"a": 1}, "a", "N_S")
        self._expect_ver("N_D", verifier._need_dict, {}, "a", "N_D")
        self._expect_ver("N_D", verifier._need_dict, {"a": []}, "a", "N_D")
        self._expect_ver("N_L", verifier._need_list, {}, "a", "N_L")
        self._expect_ver("N_L", verifier._need_list, {"a": {}}, "a", "N_L")
        self.assertEqual(verifier._need_rat({"a": 5}, "a", "N_R"), Fraction(5))
        self.assertEqual(verifier._need_rat({"a": Fraction(3, 2)}, "a", "N_R"), Fraction(3, 2))
        self._expect_ver("N_R", verifier._need_rat, {}, "a", "N_R")
        self._expect_ver("N_R", verifier._need_rat, {"a": True}, "a", "N_R")

    def test_05_lean_static_r12_r13b_bypasses_and_live_runner_guards(self) -> None:
        s_clean = verifier.strip_lean_comments("a / b - c /- outer /- inner -/ \n x -/ d -- eof_comment")
        self.assertIn("a / b - c", s_clean)
        self.assertIn("d", s_clean)
        self.assertNotIn("inner", s_clean)
        self.assertNotIn("eof_comment", s_clean)
        self._expect_ver("UNTERM_COMMENT", verifier.strip_lean_comments, "/- unclosed")

        # Axiom line parser including exact 3-bucket split (zero, propext-only, propext+Quot.sound)
        thms = {"t1", "t2", "t3"}
        ax_res = verifier.check_axiom_output(
            [
                "",
                "'EvolutionOperators.t1' does not depend on any axioms",
                "'EvolutionOperators.t2' depends on axioms: [propext]",
                "'EvolutionOperators.t3' depends on axioms: [propext,",
                " Quot.sound]",
            ],
            thms,
        )
        self.assertEqual(ax_res, {"zero_axiom": 1, "propext_only": 1, "propext_quot": 1})
        self._expect_ver("AXIOM_ORPHAN_CONT", verifier.check_axiom_output, ["  orphan"], thms)
        self._expect_ver("AXIOM_COUNT", verifier.check_axiom_output, ["'EvolutionOperators.t1' does not depend on any axioms"], thms)
        self._expect_ver("AXIOM_FORMAT", verifier.check_axiom_output, ["bad1", "bad2", "bad3"], thms)
        self._expect_ver(
            "AXIOM_DUP_THM",
            verifier.check_axiom_output,
            [
                "'EvolutionOperators.t1' does not depend on any axioms",
                "'EvolutionOperators.t1' does not depend on any axioms",
                "'EvolutionOperators.t2' does not depend on any axioms",
            ],
            thms,
        )
        self._expect_ver(
            "FORBIDDEN_AXIOM",
            verifier.check_axiom_output,
            [
                "'EvolutionOperators.t1' does not depend on any axioms",
                "'EvolutionOperators.t2' depends on axioms: [Classical.choice]",
                "'EvolutionOperators.t3' does not depend on any axioms",
            ],
            thms,
        )
        self._expect_ver(
            "AXIOM_THM_SET",
            verifier.check_axiom_output,
            [
                "'EvolutionOperators.t1' does not depend on any axioms",
                "'EvolutionOperators.t2' does not depend on any axioms",
                "'EvolutionOperators.t4' does not depend on any axioms",
            ],
            thms,
        )

        # verify_lean_bundle with mocked live runner PASS
        info = verifier.verify_lean_bundle(BASE, run_live_lean=True, lean_runner=_good_lean_runner, check_pins=True)
        self.assertEqual(info["public_theorems"], 21)

        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            _copy_pkg(tp)
            (tp / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n", encoding="utf-8")
            self._expect_ver("BAD_TOOLCHAIN", verifier.verify_lean_bundle, tp, check_pins=False)
            shutil.copyfile(BASE / "lean-toolchain", tp / "lean-toolchain")

            (tp / "lakefile.toml").write_text('name = "X"\n[[require]]\nname = "mathlib"\n', encoding="utf-8")
            self._expect_ver("LAKEFILE_REQ", verifier.verify_lean_bundle, tp, check_pins=False)
            shutil.copyfile(BASE / "lakefile.toml", tp / "lakefile.toml")

            (tp / "lake-manifest.json").write_text('{"packages": ["mathlib"]}\n', encoding="utf-8")
            self._expect_ver("MANIFEST_PKGS", verifier.verify_lean_bundle, tp, check_pins=False)
            shutil.copyfile(BASE / "lake-manifest.json", tp / "lake-manifest.json")

            (tp / "Extra.lean").write_text("def x : Nat := 1\n", encoding="utf-8")
            self._expect_ver("LEAN_FILE_COUNT", verifier.verify_lean_bundle, tp, check_pins=False)
            os.unlink(tp / "Extra.lean")

            os.rename(tp / verifier.LEAN_SRC_NAME, tp / "WrongName.lean")
            self._expect_ver("LEAN_FILE_NAME", verifier.verify_lean_bundle, tp, check_pins=False)
            os.rename(tp / "WrongName.lean", tp / verifier.LEAN_SRC_NAME)

            (tp / ".lake").mkdir()
            self._expect_ver("LAKE_RESIDUE", verifier.verify_lean_bundle, tp, check_pins=False)
            os.rmdir(tp / ".lake")

            orig_lean = (tp / verifier.LEAN_SRC_NAME).read_text(encoding="utf-8")
            # [MED-10] Regression tests: sorry, by stop skip, variable+include, @[default_instance] instance, attribute!
            for bad_snippet in (
                "\ntheorem bad : True := sorry\n",
                "\ntheorem bad : False := by stop skip\n",
                "\nvariable (h : False)\ninclude h\ntheorem bad : False := h\n",
                "\n@[default_instance] instance : Inhabited Nat := ⟨0⟩\n",
                "\nattribute [simp] Nat.add_zero\n",
            ):
                (tp / verifier.LEAN_SRC_NAME).write_text(orig_lean + bad_snippet, encoding="utf-8")
                self._expect_ver("LEAN_FORBIDDEN", verifier.verify_lean_bundle, tp, check_pins=False)

            (tp / verifier.LEAN_SRC_NAME).write_text(orig_lean + "\n#print Nat\n", encoding="utf-8")
            self._expect_ver("LEAN_RAW_PRINT", verifier.verify_lean_bundle, tp, check_pins=False)

            (tp / verifier.LEAN_SRC_NAME).write_text(orig_lean + "\ntheorem extra_t : True := trivial\n", encoding="utf-8")
            self._expect_ver("LEAN_THM_COUNT", verifier.verify_lean_bundle, tp, check_pins=False)

            cut = orig_lean.rsplit("#print axioms", 1)[0]
            (tp / verifier.LEAN_SRC_NAME).write_text(cut, encoding="utf-8")
            self._expect_ver("LEAN_PRINT_COUNT", verifier.verify_lean_bundle, tp, check_pins=False)

            swapped = orig_lean.replace(
                "#print axioms EvolutionOperators.bf16_canonical_es_ulp_stall_witness",
                "#print axioms EvolutionOperators.nonexistent_thm_xyz",
            )
            (tp / verifier.LEAN_SRC_NAME).write_text(swapped, encoding="utf-8")
            self._expect_ver("LEAN_AUDIT_SET", verifier.verify_lean_bundle, tp, check_pins=False)
            (tp / verifier.LEAN_SRC_NAME).write_text(orig_lean, encoding="utf-8")

        for attr, code in (
            ("PIN_LEAN", "PIN_LEAN"),
            ("PIN_TOOLCHAIN", "PIN_TC"),
            ("PIN_LAKEFILE", "PIN_LF"),
            ("PIN_MANIFEST", "PIN_MF"),
        ):
            with mock.patch.object(verifier, attr, "0" * 64):
                self._expect_ver(code, verifier.verify_lean_bundle, BASE, check_pins=True)

        def _run_mut(patch_dict: dict[str, object], code: str) -> None:
            def _r(pf: dict[str, bytes]) -> dict[str, object]:
                base_r = _good_lean_runner(pf)
                base_r.update(patch_dict)
                return base_r

            self._expect_ver(code, verifier.verify_lean_bundle, BASE, run_live_lean=True, lean_runner=_r)

        _run_mut({"version_rc": 1}, "LIVE_LEAN_VRC")
        _run_mut({"version_output": "Lean (version 4.36.0-rc1, linux)\n"}, "LIVE_LEAN_VER")
        _run_mut({"version_output": "Lean (version 4.36.01, linux)\n"}, "LIVE_LEAN_VER")
        _run_mut({"build_rc": 1}, "LIVE_LAKE_RC")
        _run_mut({"build_output": "warning: unused variable\n"}, "LIVE_LAKE_WARN")
        _run_mut({"axiom_rc": 1}, "LIVE_AX_RC")
        _run_mut({"axiom_stderr": "error"}, "LIVE_AX_STDERR")

        # Axiom split mismatch must fail closed with LIVE_AX_SPLIT ([LOW-3])
        def _bad_split_runner(pf: dict[str, bytes]) -> dict[str, object]:
            base_r = _good_lean_runner(pf)
            base_r["axiom_stdout"] = str(base_r["axiom_stdout"]).replace(
                "depends on axioms: [propext]\n", "does not depend on any axioms\n", 1
            )
            return base_r

        self._expect_ver("LIVE_AX_SPLIT", verifier.verify_lean_bundle, BASE, run_live_lean=True, lean_runner=_bad_split_runner)

        # [MED-10] Non-UTF-8 lean subprocess output MUST return structured RC=2 LIVE_LEAN_UTF8 (never traceback!)
        self._expect_ver("LIVE_LEAN_UTF8", verifier._decode_subproc_utf8, b"\xff\xfe")
        bad_cp = subprocess.CompletedProcess(["lean", "--version"], 0, stdout=b"\xff\xfe", stderr=b"")
        with mock.patch.object(subprocess, "run", return_value=bad_cp):
            rc_nf, _o, err_nf = _capture_fds(verifier.main, ["--check", "--live-lean"])
            self.assertEqual(rc_nf, 2)
            parsed_err = json.loads(err_nf)
            self.assertEqual(parsed_err["code"], "LIVE_LEAN_UTF8")

    def test_06_programmatic_verdicts_boundaries_and_results_reconciliation(self) -> None:
        base_rec = _load_receipt_copy()
        pass_rec = _make_passing_receipt()
        base_res = _load_results_q_copy()

        # 1) Disk tamper of s167_empirical_results.json MUST fail closed with RC=2 in BOTH oracle & verifier ([MED-2])!
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            _copy_pkg(tp)
            (tp / oracle.RESULTS_FILE_NAME).write_text('{"tampered": true}\n', encoding="utf-8")
            self._expect_ora("ORA_PIN_RES", oracle.build_certificate, tp)
            self._expect_ver("PIN_EMP_RAW", verifier.build_verified_payload, tp)

        # 2) Direct reconciliation failure checks in oracle (reconcile_results_with_receipt)
        def _bad_rec_res(mut_res, mut_rec, code: str) -> None:
            rs = copy.deepcopy(base_res)
            rc = copy.deepcopy(base_rec)
            if mut_res is not None:
                mut_res(rs)
            if mut_rec is not None:
                mut_rec(rc)
            self._expect_ora(code, oracle.reconcile_results_with_receipt, rs, rc)

        oracle.reconcile_results_with_receipt(base_res, base_rec)
        _bad_rec_res(lambda s: s.__setitem__("schema", "bad"), None, "RES_SCHEMA_VAL")
        _bad_rec_res(lambda s: s.__setitem__("prereg_sha256", "0" * 64), None, "RES_PREREG_MATCH")
        _bad_rec_res(lambda s: s.__setitem__("bench_script_sha256", "0" * 64), None, "RES_BENCH_MATCH")
        _bad_rec_res(lambda s: s.__setitem__("energy_claim", "CLAIMED"), None, "RES_ENERGY_VAL")
        _bad_rec_res(lambda s: s["exp_q51a"]["summary"].__setitem__("peak_vram_05b_mib", Fraction(1)), None, "RES_PK05_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["summary"].__setitem__("peak_vram_15b_mib", Fraction(1)), None, "RES_PK15_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["summary"].__setitem__("mean_bf16_ulp_stalled_frac", Fraction(0)), None, "RES_STALL_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["qwen15b_full_28layer_probe"].__setitem__("layers_quantized", 1), None, "RES_15LYR_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["qwen15b_full_28layer_probe"].__setitem__("total_a16_blocks", 1), None, "RES_15BLK_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["qwen15b_full_28layer_probe"].__setitem__("syndrome_errors", 1), None, "RES_15SYN_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["qwen15b_full_28layer_probe"].__setitem__("oob_groups", 1), None, "RES_15OOB_MATCH")
        _bad_rec_res(lambda s: s["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"].pop(), None, "RES_QA_ARMLEN")
        _bad_rec_res(lambda s: s["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"].__setitem__(0, 1), None, "RES_QA_ROW")
        _bad_rec_res(None, lambda c: c["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"].__setitem__(0, 1), "RES_QA_ROW")
        _bad_rec_res(lambda s: s["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"][0].__setitem__("seed", 9999), None, "RES_SEED_M")
        _bad_rec_res(lambda s: s["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"][0].__setitem__("test_exact_acc", Fraction(0)), None, "RES_TACC_M")
        _bad_rec_res(lambda s: s["exp_q51a"]["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"][0].__setitem__("syndrome_errors", 1), None, "RES_TSYN_M")
        _bad_rec_res(lambda s: s["exp_q51a"]["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"][0].__setitem__("oob_groups", 1), None, "RES_TOOB_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["baseline_base"].__setitem__("ood_8x_exact_acc", Fraction(0)), None, "RES_BOOD_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["baseline_base"].__setitem__("wikitext2_bpb", Fraction(0)), None, "RES_BBPB_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["vram_o1_scaling_grid"]["T512_R1"].__setitem__("es_peak_vram_mib", Fraction(1)), None, "RES_GEK_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["vram_o1_scaling_grid"]["T512_R1"].__setitem__("es_max_su2_shell_err", Fraction(1)), None, "RES_GSH_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["vram_o1_scaling_grid"]["T512_R1"].__setitem__("bptt_oom", True), None, "RES_GOOM_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"].pop(), None, "RES_QB_ARMLEN")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"].__setitem__(0, 1), None, "RES_QB_ROW")
        _bad_rec_res(None, lambda c: c["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"].__setitem__(0, 1), "RES_QB_ROW")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("ood_8x_exact_acc", Fraction(1)), None, "RES_OOD_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("wikitext2_bpb_finite", False), None, "RES_FIN_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("wikitext2_bpb", Fraction(1)), None, "RES_BPB_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("a16_syndrome_errors", 1), None, "RES_BSYN_M")
        _bad_rec_res(lambda s: s["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("a16_codeword_flips", 0), None, "RES_BFLP_M")

        # 3) Verifier independent receipt recomputation guards (including finite overflow cap branch!)
        rec_recomp = verifier.recompute_receipt_from_results(
            base_res, verifier.PIN_EMP_RAW, verifier.PIN_PREREG, verifier.PIN_BENCH
        )
        verifier.verify_exact_tree(base_rec, rec_recomp)
        rs_cap = copy.deepcopy(base_res)
        rs_cap["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"][0]["su2_shell_err_nonfinite"] = False
        rs_cap["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"][0]["max_finite_su2_shell_err"] = Fraction(10 ** 6, 1)  # exact SHELL_CAP_NANO boundary!
        rec_cap = verifier.recompute_receipt_from_results(rs_cap, verifier.PIN_EMP_RAW, verifier.PIN_PREREG, verifier.PIN_BENCH)
        self.assertIs(rec_cap["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"][0]["su2_shell_err_overflow"], True)
        self.assertEqual(rec_cap["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"][0]["max_su2_shell_err_nano"], verifier.SHELL_CAP_NANO)
        rs_cap["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"][0]["max_finite_su2_shell_err"] = Fraction(10 ** 6, 1) - Fraction(1, 10 ** 9)
        rec_nocap = verifier.recompute_receipt_from_results(rs_cap, verifier.PIN_EMP_RAW, verifier.PIN_PREREG, verifier.PIN_BENCH)
        self.assertIs(rec_nocap["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"][0]["su2_shell_err_overflow"], False)

        for mut, code in (
            (lambda s: s.__setitem__("schema", "bad"), "V_RES_SCH_V"),
            (lambda s: s.__setitem__("prereg_sha256", "0" * 64), "V_RES_PR_M"),
            (lambda s: s.__setitem__("bench_script_sha256", "0" * 64), "V_RES_BN_M"),
            (lambda s: s["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"].__setitem__(0, 1), "V_RES_AROW"),
            (lambda s: s["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"].__setitem__(0, 1), "V_RES_BROW"),
        ):
            bad_rs = copy.deepcopy(base_res)
            mut(bad_rs)
            self._expect_ver(code, verifier.recompute_receipt_from_results, bad_rs, verifier.PIN_EMP_RAW, verifier.PIN_PREREG, verifier.PIN_BENCH)

        # 4) Structural guards + ALL programmatic verdict branches (CONFIRMED / PARTIAL / FALSIFIED) in BOTH modules!
        for mod_kind, fn in (
            ("ora", oracle.evaluate_empirical_receipt),
            ("ver", verifier.independent_empirical_replay),
        ):
            def _bad(mutator, code_ora: str, code_ver: str) -> None:
                c = copy.deepcopy(base_rec)
                mutator(c)
                if mod_kind == "ora":
                    self._expect_ora(code_ora, fn, c)
                else:
                    self._expect_ver(code_ver, fn, c)

            _bad(lambda r: r.__setitem__("schema", "bad"), "REC_SCHEMA_VAL", "V_REC_SCHEMA_VAL")
            _bad(lambda r: r.__setitem__("prereg_sha256", "0" * 64), "REC_PREREG_PIN", "V_REC_PREREG_PIN")
            _bad(lambda r: r.__setitem__("bench_script_sha256", "0" * 64), "REC_BENCH_PIN", "V_REC_BENCH_PIN")
            _bad(lambda r: r.__setitem__("empirical_results_sha256", "0" * 64), "REC_RAW_PIN", "V_REC_RAW_PIN")
            _bad(lambda r: r.__setitem__("energy_claim", "CLAIMED"), "REC_ENERGY_VAL", "V_REC_ENERGY_VAL")
            _bad(lambda r: r.__setitem__("measured_live_model_weights", False), "REC_LIVE_VAL", "V_REC_LIVE_VAL")
            _bad(lambda r: r.__setitem__("gpu_vram_total_mib", 0), "REC_VRAM_POS", "V_REC_VPOS")
            _bad(lambda r: r.__setitem__("gpu_vram_total_mib", 10241), "REC_VRAM_CAP", "V_REC_VCAP")
            _bad(lambda r: r["exp_q51a"].__setitem__("linear_layers_05b", 100), "REC_QA_LYR_VAL", "V_QA_LYR_VAL")
            _bad(lambda r: r["exp_q51a"].__setitem__("total_a16_blocks_05b", 100), "REC_QA_BLK_VAL", "V_QA_BLK_VAL")
            _bad(lambda r: r["exp_q51a"].__setitem__("m15_layers_quantized", 1), "REC_QA_15LYR_VAL", "V_QA_15LYR_VAL")
            _bad(lambda r: r["exp_q51a"].__setitem__("m15_linear_layers_quantized", 7), "REC_QA_15LIN_VAL", "V_QA_15LIN_VAL")
            _bad(lambda r: r["exp_q51a"].__setitem__("m15_total_a16_blocks", 10), "REC_QA_15BLK_VAL", "V_QA_15BLK_VAL")
            _bad(lambda r: r["exp_q51a"].__setitem__("n_seeds", 2), "REC_QA_N_MIN", "V_QA_N_MIN")
            _bad(lambda r: r["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"].pop(), "REC_QA_C1_N", "V_QA_C1_N")
            _bad(lambda r: r["exp_q51a"]["arms"]["C2_QES_Unblocked_INT4_EF"].pop(), "REC_QA_C2_N", "V_QA_C2_N")
            _bad(lambda r: r["exp_q51a"]["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"].pop(), "REC_QA_T1_N", "V_QA_T1_N")
            _bad(lambda r: r["exp_q51a"]["arms"]["C1_VsonicV_BF16_SharedSeed"].__setitem__(0, 1), "REC_QA_C1_D", "V_QA_D1")
            _bad(lambda r: r["exp_q51a"]["arms"]["C2_QES_Unblocked_INT4_EF"].__setitem__(0, 1), "REC_QA_C2_D", "V_QA_D2")
            _bad(lambda r: r["exp_q51a"]["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"].__setitem__(0, 1), "REC_QA_T1_D", "V_QA_DT")
            _bad(lambda r: r["exp_q51b"].__setitem__("total_a16_layers", 10), "REC_QB_LYR_VAL", "V_QB_LYR_VAL")
            _bad(lambda r: r["exp_q51b"].__setitem__("total_a16_blocks", 10), "REC_QB_BLK_VAL", "V_QB_BLK_VAL")
            _bad(lambda r: r["exp_q51b"]["vram_o1_scaling_grid"].pop("T512_R1"), "REC_QB_GKEYS", "V_QB_GKEYS")
            _bad(lambda r: r["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"].pop(), "REC_QB_ARMA_N", "V_QB_LA_N")
            _bad(lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"].pop(), "REC_QB_ARMC_N", "V_QB_LC_N")
            _bad(lambda r: r["exp_q51b"]["arms"]["ArmA_Euclidean_Additive_ES"].__setitem__(0, 1), "REC_QB_RA_D", "V_QB_RA")
            _bad(lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"].__setitem__(0, 1), "REC_QB_RC_D", "V_QB_RC")

            # Exact CONFIRMED boundary on both Q51a & Q51b (+ vram_tot == 10240 boundary + ckpt oom counter)
            c_conf = copy.deepcopy(pass_rec)
            c_conf["gpu_vram_total_mib"] = 10240
            c_conf["exp_q51b"]["vram_o1_scaling_grid"]["T512_R1"]["bptt_ckpt_oom"] = True
            res_conf = fn(c_conf)
            self.assertEqual(res_conf["exp_q51a"]["verdict"], "CONFIRMED")
            self.assertEqual(res_conf["exp_q51b"]["verdict"], "CONFIRMED")
            self.assertEqual(res_conf["exp_q51b"]["bptt_ckpt_oom_cells"], 1)
            self.assertEqual(res_conf["overall_empirical_verdict"], "Q51a_CONFIRMED_Q51b_CONFIRMED_Q51c_CONFIRMED")

            # Q51a individual sub-gate ablation -> PARTIAL vs FALSIFIED (kills every BoolOp & boundary!)
            c_p = copy.deepcopy(pass_rec)
            c_p["exp_q51a"]["n_seeds"] = 3
            for arm_k in ("C1_VsonicV_BF16_SharedSeed", "C2_QES_Unblocked_INT4_EF", "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"):
                c_p["exp_q51a"]["arms"][arm_k] = c_p["exp_q51a"]["arms"][arm_k][:3]
            r_p = fn(c_p)
            self.assertIs(r_p["exp_q51a"]["p_holm_lt_005"], False)
            self.assertEqual(r_p["exp_q51a"]["verdict"], "PARTIAL")

            for mut_a in (
                lambda r: r["exp_q51a"].__setitem__("peak_vram_05b_kib", 6815744),
                lambda r: r["exp_q51a"].__setitem__("peak_vram_15b_kib", 6815744),
                lambda r: r["exp_q51a"].__setitem__("a16_null_syndrome_errors", 1),
                lambda r: r["exp_q51a"].__setitem__("m15_probe_oob_groups", 1),
                lambda r: r["exp_q51a"]["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"][0].__setitem__("syndrome_errors", 1),
                lambda r: r["exp_q51a"]["arms"]["T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"][0].__setitem__("test_exact_acc_micro", 79999),
            ):
                cf = copy.deepcopy(pass_rec)
                mut_a(cf)
                rf = fn(cf)
                self.assertEqual(rf["exp_q51a"]["verdict"], "FALSIFIED")

            # Q51b individual sub-gate ablation -> PARTIAL vs FALSIFIED (kills every BoolOp & boundary!)
            for mut_b_part in (
                lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("ood_8x_exact_acc_micro", r["exp_q51b"]["base_ood_8x_acc_micro"] + 99999),
                lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("wikitext2_bpb_micro", r["exp_q51b"]["base_wikitext2_bpb_micro"] + 20001),
                lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("wikitext2_bpb_finite", False),
            ):
                cp = copy.deepcopy(pass_rec)
                mut_b_part(cp)
                rp = fn(cp)
                self.assertEqual(rp["exp_q51b"]["verdict"], "PARTIAL")

            for mut_b_fail in (
                lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("su2_shell_err_overflow", True),
                lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("max_su2_shell_err_nano", 10000),
                lambda r: r["exp_q51b"]["vram_o1_scaling_grid"]["T512_R1"].__setitem__("es_peak_vram_kib", 524288),
                lambda r: r["exp_q51b"]["vram_o1_scaling_grid"]["T512_R1"].__setitem__("es_peak_vram_kib", 523264 + 1024),
                lambda r: r["exp_q51b"]["arms"]["ArmC_HoloSphere_Cayley_ES"][0].__setitem__("a16_syndrome_errors", 1),
            ):
                cf = copy.deepcopy(pass_rec)
                mut_b_fail(cf)
                rf = fn(cf)
                self.assertEqual(rf["exp_q51b"]["verdict"], "FALSIFIED")

    def test_07_pin_guards_cli_strictness_and_bench_unit_tests(self) -> None:
        for attr, code in (
            ("PINNED_LEAN_SHA256", "ORA_PIN_LEAN"),
            ("PINNED_TOOLCHAIN_SHA256", "ORA_PIN_TC"),
            ("PINNED_LAKEFILE_SHA256", "ORA_PIN_LF"),
            ("PINNED_MANIFEST_SHA256", "ORA_PIN_MF"),
            ("PINNED_PREREG_SHA256", "ORA_PIN_PREREG"),
            ("PINNED_BENCH_SHA256", "ORA_PIN_BENCH"),
            ("PINNED_EMPIRICAL_RESULTS_SHA256", "ORA_PIN_RES"),
            ("PINNED_RECEIPT_SHA256", "ORA_PIN_REC"),
        ):
            with mock.patch.object(oracle, attr, "0" * 64):
                self._expect_ora(code, oracle.build_certificate, BASE)

        for attr, code in (
            ("PIN_PREREG", "PIN_PREREG"),
            ("PIN_BENCH", "PIN_BENCH"),
            ("PIN_EMP_RAW", "PIN_EMP_RAW"),
            ("PIN_RECEIPT", "PIN_RECEIPT"),
        ):
            with mock.patch.object(verifier, attr, "0" * 64):
                self._expect_ver(code, verifier.build_verified_payload, BASE)

        # Oracle CLI modes, --help, allow_abbrev=False, and --json=1 single-line JSON ([LOW-5])
        rc, out, err = _capture_fds(oracle.main, [])
        self.assertEqual(rc, 0)
        self.assertIn("PASS ", out)
        self.assertEqual(err, "")

        rc_h, out_h, err_h = _capture_fds(oracle.main, ["--help"])
        self.assertEqual(rc_h, 0)
        self.assertIn("usage:", out_h)
        self.assertEqual(err_h, "")

        for bad_args in (["--bogus"], ["--wr"], ["--json=1"]):
            rc_u, _o, err_u = _capture_fds(oracle.main, bad_args)
            self.assertEqual(rc_u, 2)
            self.assertEqual(len(err_u.splitlines()), 1)
            self.assertEqual(json.loads(err_u)["code"], "ORA_CLI_ARGS")

        with mock.patch.object(oracle, "run_cli", side_effect=OSError("boom")):
            rc_os, _o, err_os = _capture_fds(oracle.main, [])
            self.assertEqual(rc_os, 2)
            self.assertIn("ORA_OS_ERROR", err_os)

        # Verifier CLI modes, --help, allow_abbrev=False (--ch refused!), and --json=1 single-line JSON ([LOW-5])
        rc_s, out_s, err_s = _capture_fds(verifier.main, ["--standalone"])
        self.assertEqual(rc_s, 0)
        self.assertIn("PASS ", out_s)
        self.assertEqual(err_s, "")

        rc_vh, out_vh, err_vh = _capture_fds(verifier.main, ["-h"])
        self.assertEqual(rc_vh, 0)
        self.assertIn("usage:", out_vh)
        self.assertEqual(err_vh, "")

        for bad_args in (["--unknown"], ["--ch"], ["--json=1"]):
            rc_vb, _o, err_vb = _capture_fds(verifier.main, bad_args)
            self.assertEqual(rc_vb, 2)
            self.assertEqual(len(err_vb.splitlines()), 1)
            self.assertEqual(json.loads(err_vb)["code"], "BAD_CLI_ARGS")

        rc_mc, _o, err_mc = _capture_fds(verifier.main, ["--check", "--standalone"])
        self.assertEqual(rc_mc, 2)
        self.assertIn("MODE_CONFLICT", err_mc)

        rc_mc2, _o, err_mc2 = _capture_fds(verifier.main, ["--write", "--standalone"])
        self.assertEqual(rc_mc2, 2)
        self.assertIn("MODE_CONFLICT", err_mc2)

        with mock.patch.object(verifier, "run_cli", side_effect=subprocess.SubprocessError("sub")):
            rc_se, _o, err_se = _capture_fds(verifier.main, [])
            self.assertEqual(rc_se, 2)
            self.assertIn("SUBPROCESS_ERROR", err_se)

        with mock.patch.object(verifier, "run_cli", side_effect=OSError("io")):
            rc_oe, _o, err_oe = _capture_fds(verifier.main, [])
            self.assertEqual(rc_oe, 2)
            self.assertIn("OS_ERROR", err_oe)

        # [LOW-9] Pure-Python unit tests for s167_es_live_bench.py (runs cleanly under python3 -I with zero torch!)
        r_ok, is_ok = bench.verify_countdown_output("3 * 5 - 2 = 13\nignored", [3, 5, 2], 13)
        self.assertIs(is_ok, True)
        self.assertEqual(int(r_ok), 1)
        r_bad, is_bad = bench.verify_countdown_output("3 / 5", [3, 5, 2], 13)
        self.assertIs(is_bad, False)
        self.assertEqual(int(r_bad), 0)
        r_over, is_over = bench.verify_countdown_output("3 + 3", [3, 5, 2], 6)
        self.assertIs(is_over, False)
        self.assertEqual(int(round(r_over * 10)), 1)
        r_empty, is_empty = bench.verify_countdown_output("", [3, 5, 2], 13)
        self.assertIs(is_empty, False)
        self.assertEqual(int(r_empty), 0)

        split = bench.build_countdown_split(4, seed=167001, exclude_keys=set())
        self.assertEqual(len(split), 4)
        self.assertEqual(bench._eval_s3_word("aa"), 0)
        self.assertEqual(bench._eval_s3_word("bbb"), 0)
        self.assertEqual(bench._eval_a5_word_action0("aaa"), 0)
        self.assertEqual(bench._eval_a5_word_action0("bbbbb"), 0)
        g_split = bench.build_group_word_split(2, 2, min_len=2, max_len=4, seed=167201, exclude_words=set())
        self.assertEqual(len(g_split), 4)

        # Live roundtrip of bench.build_pure_int_receipt on real disk s167_empirical_results.json!
        raw_disk_bytes = (BASE / oracle.RESULTS_FILE_NAME).read_bytes()
        raw_disk_obj = json.loads(raw_disk_bytes.decode("utf-8"))
        rebuilt_rec = bench.build_pure_int_receipt(
            raw_disk_obj,
            hashlib.sha256(raw_disk_bytes).hexdigest(),
            oracle.PINNED_PREREG_SHA256,
            oracle.PINNED_BENCH_SHA256,
        )
        verifier.verify_exact_tree(rebuilt_rec, _load_receipt_copy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
