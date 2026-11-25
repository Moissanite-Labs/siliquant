#!/usr/bin/env python3
"""
Track S20b Post-Freeze Gate-8 Live Evaluation (`AUDIT-119B-LIVE`):
R5 Adversarial, Zone-B & Mutation-Killing Test Suite (`test_gate8_live_eval_r5.py`).

Pure stdlib, zero bare `assert`s (`-O`-safe), zero floats on trust paths,
4-mode `-I` parity verified.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import gate8_live_eval_audit as audit  # noqa: E402
import gate8_live_eval_oracle as oracle  # noqa: E402


class _IntSub(int):
    pass


class _DictSub(dict):
    pass


class _StrSub(str):
    pass


class _EvilPath(type(Path())):
    pass


def _make_mock_results_jsonl(rows_map: dict[str, dict[str, int]]) -> str:
    variants = (
        ("qwen25_3p0_zbg", "ZBG128:b368s16"),
        ("qwen25_3p0_ebg", "EBG128:b368s16"),
        ("qwen25_3p0_hbg", "HBG128:b368s16"),
    )
    lines = ['{"model":"other","variant":"ignore"}']
    for key, var in variants:
        r = rows_map[key]
        kl = Decimal(r["kl_u6"]) / Decimal(1_000_000)
        se = Decimal(r["se_u6"]) / Decimal(1_000_000)
        we = Decimal(r["werr_mdB"]) / Decimal(1_000)
        z05 = Decimal(r["z005_bp"]) / Decimal(100)
        e05 = Decimal(r["err005_mdB"]) / Decimal(1_000)
        z20 = Decimal(r["z020_bp"]) / Decimal(100)
        e20 = Decimal(r["err020_mdB"]) / Decimal(1_000)
        lines.append(
            f'{{"model":"qwen2.5-1.5b-instruct-fp16.gguf","variant":"{var}",'
            f'"kl_mean":{kl},"kl_mean_se":{se},"weight_err_db_pooled":{we},'
            f'"small_weights":{{"0.05":{{"all":{{"stored_zero_percent":{z05},"err_over_size_db":{e05}}}}},'
            f'"0.2":{{"all":{{"stored_zero_percent":{z20},"err_over_size_db":{e20}}}}}}}}}'
        )
    return "\n".join(lines) + "\n"


class TestTrackS20bGate8LiveEvalR5(unittest.TestCase):
    cert: dict[str, object]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.generate_certificate()

    def test_01_g8p1_exact_verdicts_clauses_margins_and_timeline(self) -> None:
        p = self.cert["payload"]
        self.assertIsInstance(p, dict)
        ge = p["gate8_evaluation"]
        self.assertEqual(
            ge["verdicts"],
            {"G8-P1": "PASS", "G8-P2": "PENDING", "G8-P3": "PENDING", "G8-P4": "PENDING"},
        )
        self.assertEqual(ge["clauses_passed"], 9)
        self.assertEqual(ge["clauses_total"], 9)
        self.assertTrue(all(ge["clauses"].values()))
        self.assertEqual(
            ge["details"]["G8-P1"],
            {
                "red_hbg_vs_zbg": "36621/215147",
                "red_ebg_vs_zbg": "45809/215147",
                "abs_gap_hbg_ebg": "4594/84669",
                "z2_hbg_vs_zbg": "1341097641/21005200",
                "z2_ebg_vs_zbg": "2098464481/20949409",
            },
        )
        self.assertEqual(ge["margins"]["z005_margin_bp"], 593)
        self.assertEqual(ge["margins"]["err005_margin_mdB"], 3093)

        sec = p["qwen25_3p0_secondary_diagnostics"]
        self.assertEqual(sec["hbg_minus_ebg_kl_u6"], 9188)
        self.assertEqual(sec["z2_hbg_vs_ebg"], "84419344/19213841")
        self.assertEqual(sec["ebg_werr_advantage_over_zbg_mdB"], 658)
        self.assertEqual(sec["hbg_werr_advantage_over_zbg_mdB"], 656)
        self.assertEqual(sec["hbg_minus_ebg_err005_mdB"], -4591)
        self.assertEqual(sec["hbg_minus_ebg_err020_mdB"], -102)
        self.assertEqual(sec["steiner_jump_model_005_energy_ratio_hbg_over_ebg"], "1214/3223")
        self.assertEqual(sec["steiner_jump_model_020_energy_ratio_hbg_over_ebg"], "2498/2353")

        ts = p["law4_timestamp_audit"]
        self.assertEqual(ts["exact_deltas_s"]["ebg_commit_minus_g8p1_oracle_finalized_s"], 72)
        self.assertEqual(ts["exact_deltas_s"]["hbg_commit_minus_g8p1_oracle_finalized_s"], 660)
        self.assertEqual(ts["exact_deltas_s"]["ebg_commit_minus_report_mtime_s"], -146)
        self.assertEqual(ts["exact_deltas_s"]["hbg_commit_minus_report_mtime_s"], 442)
        self.assertEqual(ts["exact_deltas_s"]["board_1155_minus_hbg_commit_s"], 806)
        self.assertEqual(
            ts["locally_authored_before_upstream_commit"],
            {
                "g8p1_oracle_before_0d843d95_ebg": True,
                "g8p1_oracle_before_be63add8_hbg": True,
                "report_mtime_before_0d843d95_ebg": False,
                "report_mtime_before_be63add8_hbg": True,
            },
        )
        self.assertEqual(
            ts["externally_cryptographically_precommitted_via_board_1155"],
            {"G8-P1": False, "G8-P2": True, "G8-P3": True, "G8-P4": True},
        )

    def test_02_g8p1_scorer_boundaries_and_zone_b_guards(self) -> None:
        # Exact boundary PASS and individual clause FAILs (kills every comparison mutant!)
        edge_pass = {
            "qwen25_3p0_ebg": {"kl_u6": 197935, "se_u6": 2000},
            "qwen25_3p0_hbg": {"kl_u6": 193632, "se_u6": 2000, "z005_bp": 8800, "err005_mdB": 6000},
        }
        self.assertEqual(oracle.score_g8_p1_standalone(edge_pass)["verdicts"]["G8-P1"], "PASS")
        for bad_clause in (
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 215147, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 193633, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 190000, "se_u6": 6000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 215147, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 197936, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 195000, "se_u6": 6500}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 190000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 161500, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 8799, "err005_mdB": 8500}},
            {"qwen25_3p0_ebg": {"kl_u6": 185000, "se_u6": 2000}, "qwen25_3p0_hbg": {"kl_u6": 180000, "se_u6": 2000, "z005_bp": 9300, "err005_mdB": 5999}},
        ):
            self.assertEqual(oracle.score_g8_p1_standalone(bad_clause)["verdicts"]["G8-P1"], "FAIL")

        # Strict type & missing-key guards
        with self.assertRaises(oracle.Gate8EvalError) as ctx_rt:
            oracle.score_g8_p1_standalone([])  # type: ignore[arg-type]
        self.assertEqual(ctx_rt.exception.code, "INVALID_ROWS_TYPE")
        for bad_rows in (
            _DictSub(),
            {},
            {"qwen25_3p0_ebg": {"kl_u6": 1, "se_u6": 1}},
            {"qwen25_3p0_hbg": {"kl_u6": 1, "se_u6": 1, "z005_bp": 9000, "err005_mdB": 7000}},
            {"qwen25_3p0_ebg": [], "qwen25_3p0_hbg": {}},
            {"qwen25_3p0_ebg": {}, "qwen25_3p0_hbg": []},
            {"qwen25_3p0_ebg": {"kl_u6": True, "se_u6": 10}, "qwen25_3p0_hbg": {"kl_u6": 10, "se_u6": 10, "z005_bp": 9000, "err005_mdB": 7000}},
            {"qwen25_3p0_ebg": {"kl_u6": float("nan"), "se_u6": 10}, "qwen25_3p0_hbg": {"kl_u6": 10, "se_u6": 10, "z005_bp": 9000, "err005_mdB": 7000}},
        ):
            with self.assertRaises(oracle.Gate8EvalError):
                oracle.score_g8_p1_standalone(bad_rows)  # type: ignore[arg-type]

        # _round_dec_to_int positive & negative checks
        self.assertEqual(oracle._round_dec_to_int(2, 1000, "i"), 2000)
        self.assertEqual(oracle._round_dec_to_int(Decimal("0.178526"), 1_000_000, "d"), 178526)
        for bad_d in (True, "1.0", Decimal("NaN"), Decimal("Infinity"), Decimal("1e1000")):
            with self.assertRaises(oracle.Gate8EvalError):
                oracle._round_dec_to_int(bad_d, 1000, "bad")

    def test_03_live_s20b_loader_and_results_jsonl_parser(self) -> None:
        # 1. If real frozen sibling Track S20b exists, verify live read-only execution with 0 pycache!
        real_s20b = _HERE.parent / "track-s20b-e8had-anisotropy-gate8"
        if real_s20b.is_dir():
            live_res = oracle.verify_live_s20b_scorer(real_s20b)
            self.assertTrue(live_res["live_s20b_verified"])
            self.assertFalse(live_res["pycache_created_in_s20b"])
            self.assertEqual(live_res["s20b_live_verdicts"]["G8-P1"], "PASS")

        # 2. Deterministic isolated synthetic Track S20b & results.jsonl fixtures (kills 100% of mutants even in tempdir!)
        standalone = oracle.score_g8_p1_standalone(oracle.LIVE_SCORER_INPUT_ROWS)
        good_stub = (
            "def evaluate_gate8_rows(rows):\n"
            "    if not rows:\n"
            "        return {'verdicts': {'G8-P1': 'PENDING', 'G8-P2': 'PENDING', 'G8-P3': 'PENDING', 'G8-P4': 'PENDING'}, 'details': {}}\n"
            f"    return {repr({'verdicts': standalone['verdicts'], 'details': standalone['details']})}\n"
        )
        bad_verdict_stub = (
            "def evaluate_gate8_rows(rows):\n"
            "    return {'verdicts': {'G8-P1': 'FAIL', 'G8-P2': 'PENDING', 'G8-P3': 'PENDING', 'G8-P4': 'PENDING'}, 'details': {}}\n"
        )
        bad_base_stub = (
            "def evaluate_gate8_rows(rows):\n"
            "    if not rows:\n"
            "        return {'verdicts': {'G8-P1': 'PASS', 'G8-P2': 'PENDING', 'G8-P3': 'PENDING', 'G8-P4': 'PENDING'}, 'details': {}}\n"
            f"    return {repr({'verdicts': standalone['verdicts'], 'details': standalone['details']})}\n"
        )
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            s_dir = tmp / "s20b"
            s_dir.mkdir()
            o_file = s_dir / "e8had_anisotropy_oracle.py"
            c_file = s_dir / "e8had_anisotropy_certificate.json"
            o_file.write_text(good_stub, encoding="utf-8")
            c_file.write_text('{"ok":1}\n', encoding="utf-8")
            o_sha = hashlib.sha256(o_file.read_bytes()).hexdigest()
            c_sha = hashlib.sha256(c_file.read_bytes()).hexdigest()

            res_ok = oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha=o_sha, expected_cert_sha=c_sha)
            self.assertTrue(res_ok["live_s20b_verified"])
            self.assertFalse(res_ok["pycache_created_in_s20b"])

            # Missing dir, missing oracle, missing cert, SHA mismatches
            with self.assertRaises(oracle.Gate8EvalError) as c_ndir:
                oracle.verify_live_s20b_scorer(tmp / "nope", expected_oracle_sha=o_sha, expected_cert_sha=c_sha)
            self.assertEqual(c_ndir.exception.code, "S20B_DIR_NOT_FOUND")
            with self.assertRaises(oracle.Gate8EvalError) as c_osha:
                oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha="0" * 64, expected_cert_sha=c_sha)
            self.assertEqual(c_osha.exception.code, "S20B_ORACLE_SHA_MISMATCH")
            with self.assertRaises(oracle.Gate8EvalError) as c_csha:
                oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha=o_sha, expected_cert_sha="0" * 64)
            self.assertEqual(c_csha.exception.code, "S20B_CERT_SHA_MISMATCH")

            # Divergent scorer & broken baseline
            o_file.write_text(bad_verdict_stub, encoding="utf-8")
            o_sha2 = hashlib.sha256(o_file.read_bytes()).hexdigest()
            with self.assertRaises(oracle.Gate8EvalError) as c_div:
                oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha=o_sha2, expected_cert_sha=c_sha)
            self.assertEqual(c_div.exception.code, "LIVE_SCORER_PARITY_MISMATCH")

            o_file.write_text(bad_base_stub, encoding="utf-8")
            o_sha3 = hashlib.sha256(o_file.read_bytes()).hexdigest()
            with self.assertRaises(oracle.Gate8EvalError) as c_base:
                oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha=o_sha3, expected_cert_sha=c_sha)
            self.assertEqual(c_base.exception.code, "BASELINE_PENDING_MISMATCH")

            # Missing files inside s20b_dir
            only_o = tmp / "only_o"
            only_o.mkdir()
            (only_o / "e8had_anisotropy_oracle.py").write_text(good_stub, encoding="utf-8")
            with self.assertRaises(oracle.Gate8EvalError) as c_m1:
                oracle.verify_live_s20b_scorer(only_o, expected_oracle_sha=o_sha, expected_cert_sha=c_sha)
            self.assertEqual(c_m1.exception.code, "S20B_FILES_MISSING")
            only_c = tmp / "only_c"
            only_c.mkdir()
            (only_c / "e8had_anisotropy_certificate.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(oracle.Gate8EvalError) as c_m2:
                oracle.verify_live_s20b_scorer(only_c, expected_oracle_sha=o_sha, expected_cert_sha=c_sha)
            self.assertEqual(c_m2.exception.code, "S20B_FILES_MISSING")

            # parse_results_jsonl_file valid & adversarial error cases
            good_jsonl = tmp / "results.jsonl"
            good_jsonl.write_text(_make_mock_results_jsonl(oracle.QWEN25_3P0_ROWS_U6), encoding="utf-8")
            parsed = oracle.parse_results_jsonl_file(good_jsonl)
            self.assertTrue(parsed["matches_pinned_rows"])
            self.assertFalse(parsed["matches_pinned_sha256"])
            self.assertEqual(parsed["total_rows"], 4)

            with self.assertRaises(oracle.Gate8EvalError) as c_nf:
                oracle.parse_results_jsonl_file(tmp / "missing.jsonl")
            self.assertEqual(c_nf.exception.code, "FILE_NOT_FOUND")

            big_jsonl = tmp / "big.jsonl"
            big_jsonl.write_bytes(b" " * (oracle.MAX_JSON_BYTES + 1))
            with self.assertRaises(oracle.Gate8EvalError) as c_big:
                oracle.parse_results_jsonl_file(big_jsonl)
            self.assertEqual(c_big.exception.code, "FILE_TOO_LARGE")

            exact_jsonl = tmp / "exact_lim.jsonl"
            exact_jsonl.write_bytes(('{"m":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}').encode("ascii"))
            self.assertEqual(oracle.parse_results_jsonl_file(exact_jsonl)["total_rows"], 1)

            bad_u8 = tmp / "bad_u8.jsonl"
            bad_u8.write_bytes(b"\xff\xfe")
            with self.assertRaises(oracle.Gate8EvalError) as c_u8:
                oracle.parse_results_jsonl_file(bad_u8)
            self.assertEqual(c_u8.exception.code, "INVALID_UTF8")

            empty_j = tmp / "empty.jsonl"
            empty_j.write_text("   \n", encoding="utf-8")
            with self.assertRaises(oracle.Gate8EvalError) as c_emp:
                oracle.parse_results_jsonl_file(empty_j)
            self.assertEqual(c_emp.exception.code, "EMPTY_JSONL")

            for name, body, code in (
                ("nan.jsonl", '{"x": NaN}\n', "JSON_CONSTANT_FORBIDDEN"),
                ("bad.jsonl", "{bad\n", "JSONL_SYNTAX_ERROR"),
                ("arr.jsonl", "[1,2]\n", "JSONL_ROW_NOT_OBJECT"),
                ("surr_val.jsonl", '{"a":"\\ud800"}\n', "INVALID_UTF8"),
                ("surr_key.jsonl", '{"\\ud800":"a"}\n', "INVALID_UTF8"),
                ("surr_list.jsonl", '{"a":["\\ud800"]}\n', "INVALID_UTF8"),
                ("depth65.jsonl", '{"a":' + ("[" * 64) + "0" + ("]" * 64) + "}\n", "MAX_DEPTH_EXCEEDED"),
                ("nosw.jsonl", '{"model":"qwen2.5-1.5b-instruct-fp16.gguf","variant":"EBG128:b368s16"}\n', "MISSING_SMALL_WEIGHTS"),
                ("nosw05.jsonl", '{"model":"qwen2.5-1.5b-instruct-fp16.gguf","variant":"EBG128:b368s16","small_weights":{"0.05":1,"0.2":{}}}\n', "MISSING_SMALL_WEIGHTS"),
                ("nosw02.jsonl", '{"model":"qwen2.5-1.5b-instruct-fp16.gguf","variant":"EBG128:b368s16","small_weights":{"0.05":{},"0.2":1}}\n', "MISSING_SMALL_WEIGHTS"),
                ("noall05.jsonl", '{"model":"qwen2.5-1.5b-instruct-fp16.gguf","variant":"EBG128:b368s16","small_weights":{"0.05":{"all":1},"0.2":{"all":{}}}}\n', "MISSING_SMALL_WEIGHTS_ALL"),
                ("noall02.jsonl", '{"model":"qwen2.5-1.5b-instruct-fp16.gguf","variant":"EBG128:b368s16","small_weights":{"0.05":{"all":{}},"0.2":{"all":1}}}\n', "MISSING_SMALL_WEIGHTS_ALL"),
            ):
                fp = tmp / name
                fp.write_text(body, encoding="utf-8")
                with self.assertRaises(oracle.Gate8EvalError) as ctx_j:
                    oracle.parse_results_jsonl_file(fp)
                self.assertEqual(ctx_j.exception.code, code)

            d64_jsonl = tmp / "depth64.jsonl"
            d64_jsonl.write_text('{"a":' + ("[" * 63) + "0" + ("]" * 63) + "}\n", encoding="utf-8")
            self.assertEqual(oracle.parse_results_jsonl_file(d64_jsonl)["total_rows"], 1)

            with self.assertRaises(oracle.Gate8EvalError) as ctx_epj:
                oracle.parse_results_jsonl_file(_EvilPath(good_jsonl))
            self.assertEqual(ctx_epj.exception.code, "INVALID_PATH_TYPE")

            # Broken importlib spec guards (kills spec is None or spec.loader is None)
            o_file.write_text(good_stub, encoding="utf-8")
            with mock.patch.object(oracle.importlib.util, "spec_from_file_location", return_value=None):
                with self.assertRaises(oracle.Gate8EvalError) as c_sp1:
                    oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha=o_sha, expected_cert_sha=c_sha)
                self.assertEqual(c_sp1.exception.code, "S20B_IMPORT_SPEC_FAILED")
            fake_spec = types.SimpleNamespace(loader=None)
            with mock.patch.object(oracle.importlib.util, "spec_from_file_location", return_value=fake_spec):
                with self.assertRaises(oracle.Gate8EvalError) as c_sp2:
                    oracle.verify_live_s20b_scorer(s_dir, expected_oracle_sha=o_sha, expected_cert_sha=c_sha)
                self.assertEqual(c_sp2.exception.code, "S20B_IMPORT_SPEC_FAILED")

            # CLI --live-s20b & --results-jsonl paths (isolated tempdir safe for mutate/BoolOp)
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc = oracle.main(["--live-s20b", str(tmp / "nope")])
            self.assertEqual(rc, 1)
            self.assertIn("FAIL S20B_DIR_NOT_FOUND", err_buf.getvalue())

            with mock.patch.object(oracle, "verify_live_s20b_scorer", return_value={"live_s20b_verified": True}):
                out_j, out_t = io.StringIO(), io.StringIO()
                with redirect_stdout(out_j):
                    self.assertEqual(oracle.main(["--live-s20b", str(s_dir), "--json"]), 0)
                self.assertTrue(json.loads(out_j.getvalue())["s20b_parity"]["live_s20b_verified"])
                with redirect_stdout(out_t):
                    self.assertEqual(oracle.main(["--live-s20b", str(s_dir)]), 0)
                self.assertIn("OK TRACK-S20B-GATE8-LIVE-PARITY", out_t.getvalue())
                with mock.patch.object(
                    oracle, "parse_results_jsonl_file", return_value={"matches_pinned_sha256": True, "matches_pinned_rows": True}
                ):
                    out_rj = io.StringIO()
                    with redirect_stdout(out_rj):
                        self.assertEqual(oracle.main(["--live-s20b", str(s_dir), "--results-jsonl", str(good_jsonl), "--json"]), 0)
                    self.assertTrue(json.loads(out_rj.getvalue())["results_jsonl_parity"]["matches_pinned_sha256"])
                for bad_pair in (
                    {"matches_pinned_sha256": False, "matches_pinned_rows": True},
                    {"matches_pinned_sha256": True, "matches_pinned_rows": False},
                ):
                    with mock.patch.object(oracle, "parse_results_jsonl_file", return_value=bad_pair):
                        err_rj = io.StringIO()
                        with redirect_stderr(err_rj):
                            self.assertEqual(oracle.main(["--live-s20b", str(s_dir), "--results-jsonl", str(good_jsonl)]), 1)
                        self.assertIn("FAIL RESULTS_JSONL_MISMATCH", err_rj.getvalue())

    def test_04_zone_b_strict_io_auditor_and_cli(self) -> None:
        max_2048 = (1 << oracle.MAX_INT_BITS) - 1
        over_2048 = 1 << oracle.MAX_INT_BITS
        self.assertEqual(oracle.require_int(max_2048, "m"), max_2048)
        self.assertEqual(oracle.require_int(5, "b", lo=5, hi=5), 5)
        for bad_call in (
            lambda: oracle.require_int(True, "b"),
            lambda: oracle.require_int(_IntSub(1), "s"),
            lambda: oracle.require_int(over_2048, "o"),
            lambda: oracle.require_int(10**5000, "h"),
            lambda: oracle.require_int(4, "lo", lo=5),
            lambda: oracle.require_int(6, "hi", hi=5),
            lambda: oracle.require_int(5, "b_lo", lo=True),
            lambda: oracle.require_int(5, "s_lo", lo=_IntSub(0)),
            lambda: oracle.require_int(5, "b_hi", hi=True),
            lambda: oracle.require_int(5, "s_hi", hi=_IntSub(10)),
            lambda: oracle.require_int(1, "huge_lo", lo=10**5000),
            lambda: oracle.require_int(1, "huge_hi", hi=-(10**5000)),
        ):
            with self.assertRaises(oracle.Gate8EvalError):
                bad_call()

        good_hex = "0123456789abcdef" * 4
        other_hex = "fedcba9876543210" * 4
        self.assertTrue(oracle.constant_time_hex64_equal(good_hex, good_hex))
        self.assertFalse(oracle.constant_time_hex64_equal(good_hex, other_hex))
        for bad_hex in (123, "a" * 63, "a" * 65, "A" * 64, "\u00e9" * 64, chr(0xD800) * 64):
            with self.assertRaises(oracle.Gate8EvalError):
                oracle.constant_time_hex64_equal(bad_hex, good_hex)

        self.assertEqual(oracle.fmt_q(Fraction(3, 1)), "3")
        self.assertEqual(oracle.fmt_q(Fraction(1, 6)), "1/6")
        self.assertEqual(oracle.fmt_q(Fraction(max_2048, max_2048 - 1)), f"{max_2048}/{max_2048 - 1}")
        for bad_q in (
            lambda: oracle.fmt_q(1),  # type: ignore[arg-type]
            lambda: oracle.fmt_q(Fraction(over_2048, 1)),
            lambda: oracle.fmt_q(Fraction(1, over_2048)),
            lambda: oracle.fmt_q(Fraction(10**5000, 1)),
        ):
            with self.assertRaises(oracle.Gate8EvalError):
                bad_q()

        d64: object = 0
        for _ in range(64):
            d64 = [d64]
        oracle.validate_json_value(d64, max_depth=64)
        oracle.validate_json_value({"a": True, "b": None, "c": max_2048, "d": -max_2048, "s": "ok"})
        d65 = [d64]
        deep1500: object = 0
        for _ in range(1500):
            deep1500 = [deep1500]
        for bad_val in (
            d65,
            deep1500,
            {1: "a"},
            {chr(0xD800): "a"},
            {"s": chr(0xD800)},
            {"n": over_2048},
            {"n": 10**5000},
            {"n": _IntSub(1)},
            {"o": object()},
            _DictSub(a=1),
        ):
            with self.assertRaises(oracle.Gate8EvalError):
                oracle.validate_json_value(bad_val)
        with self.assertRaises(oracle.Gate8EvalError) as ctx_flt:
            oracle.validate_json_value({"x": 1.5})
        self.assertEqual(ctx_flt.exception.code, "FLOAT_FORBIDDEN")

        self.assertEqual(oracle._parse_capped_int_token("-42"), -42)
        self.assertEqual(oracle._parse_capped_int_token(str(max_2048)), max_2048)
        with self.assertRaises(oracle.Gate8EvalError) as ctx_d620:
            oracle._parse_capped_int_token("1" + ("0" * (oracle.MAX_INT_DIGITS - 1)))
        self.assertEqual(ctx_d620.exception.code, "INT_BIT_LIMIT")
        with self.assertRaises(oracle.Gate8EvalError) as ctx_d621:
            oracle._parse_capped_int_token("9" * (oracle.MAX_INT_DIGITS + 1))
        self.assertEqual(ctx_d621.exception.code, "INT_DIGIT_LIMIT")
        with self.assertRaises(oracle.Gate8EvalError) as ctx_b2049:
            oracle._parse_capped_int_token(str(over_2048))
        self.assertEqual(ctx_b2049.exception.code, "INT_BIT_LIMIT")

        self.assertEqual(oracle.parse_strict_json_text('{"k":1}'), {"k": 1})
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])
        for bad_ro in (0, 1, "yes", None):
            with self.assertRaises(oracle.Gate8EvalError) as ctx_ro:
                oracle.parse_strict_json_text("[1]", require_object=bad_ro)  # type: ignore[arg-type]
            self.assertEqual(ctx_ro.exception.code, "INVALID_BOOL_TYPE")
        exact_max_str = '{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}'
        self.assertEqual(len(oracle.parse_strict_json_text(exact_max_str)["a"]), oracle.MAX_JSON_BYTES - 8)
        with self.assertRaises(oracle.Gate8EvalError) as ctx_empty:
            oracle.parse_strict_json_text("   ")
        self.assertEqual(ctx_empty.exception.code, "EMPTY_JSON_INPUT")
        for bad_txt in (
            123,
            chr(0xD800),
            exact_max_str + " ",
            "{bad",
            '{"x":1.5}',
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"a":1,"a":2}',
            '{"n":' + ("9" * 5000) + "}",
            '{"s":"\\ud800"}',
            "[" * 100 + "0" + "]" * 100,
            "[1,2]",
        ):
            with self.assertRaises(oracle.Gate8EvalError):
                oracle.parse_strict_json_text(bad_txt)  # type: ignore[arg-type]

        # Auditor certificate verification, tamper, symlink & CLI checks
        cert_path = _HERE / "gate8_live_eval_certificate.json"
        res = audit.verify_certificate_file(cert_path)
        self.assertTrue(res["valid"])
        self.assertEqual(res["payload_sha256"], self.cert["payload_sha256"])
        self.assertEqual(res["certificate_sha256"], self.cert["certificate_sha256"])

        good_copy = json.loads(oracle.canonical_json_bytes(self.cert).decode("ascii"))
        forged_payload = {"forged": 1}
        fp_sha = oracle.canonical_sha256(forged_payload)
        forged_body = {"schema": good_copy["schema"], "payload_sha256": fp_sha, "payload": forged_payload}
        forged_cert = {
            "schema": good_copy["schema"],
            "payload_sha256": fp_sha,
            "certificate_sha256": oracle.canonical_sha256(forged_body),
            "payload": forged_payload,
        }
        for bad_cert, want_code in (
            ([], "INVALID_CERT_TYPE"),
            ({k: v for k, v in good_copy.items() if k != "schema"}, "INVALID_CERT_KEYS"),
            (dict(good_copy, schema=123), "INVALID_SCHEMA"),
            (dict(good_copy, schema=_StrSub(good_copy["schema"])), "INVALID_SCHEMA"),
            (dict(good_copy, schema="wrong.schema"), "INVALID_SCHEMA"),
            (dict(good_copy, payload=[1, 2]), "INVALID_PAYLOAD_TYPE"),
            (dict(good_copy, payload_sha256="0" * 64), "PAYLOAD_SHA_MISMATCH"),
            (dict(good_copy, certificate_sha256="0" * 64), "CERT_SHA_MISMATCH"),
            (forged_cert, "RECOMPUTE_PAYLOAD_MISMATCH"),
        ):
            with self.assertRaises(oracle.Gate8EvalError) as ctx_c:
                audit.verify_certificate_dict(bad_cert)
            self.assertEqual(ctx_c.exception.code, want_code)

        for bad_p in ("not_a_path", _EvilPath("c.json"), Path("bad" + chr(0xD800)), Path("bad\x00path")):
            with self.assertRaises(oracle.Gate8EvalError):
                oracle.require_no_symlink_path(bad_p)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            real_dir = tmp / "real"
            real_dir.mkdir()
            real_sub = real_dir / "sub"
            real_sub.mkdir()
            other_dir = tmp / "other"
            other_dir.mkdir()
            real_file = real_dir / "c.json"
            real_file.write_bytes(oracle.canonical_json_bytes(self.cert))
            sub_file = real_sub / "c.json"
            sub_file.write_bytes(oracle.canonical_json_bytes(self.cert))
            self.assertTrue(audit.verify_certificate_file(real_file)["valid"])

            with self.assertRaises(oracle.Gate8EvalError) as ctx_ep:
                audit.verify_certificate_file(_EvilPath(real_file))
            self.assertEqual(ctx_ep.exception.code, "INVALID_PATH_TYPE")

            sym_file = tmp / "sym.json"
            sym_file.symlink_to(real_file)
            with self.assertRaises(oracle.Gate8EvalError):
                audit.verify_certificate_file(sym_file)

            sym_dir = tmp / "symdir"
            sym_dir.symlink_to(real_dir)
            with self.assertRaises(oracle.Gate8EvalError):
                audit.verify_certificate_file(sym_dir / "c.json")

            sym_other = tmp / "symother"
            sym_other.symlink_to(other_dir)
            orig_cwd = os.getcwd()
            try:
                os.chdir(real_dir)
                with mock.patch.dict(os.environ, {"PWD": str(real_dir)}):
                    self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("PWD", None)
                    self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                with mock.patch.dict(os.environ, {"PWD": str(sym_other)}):
                    self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                with mock.patch.dict(os.environ, {"PWD": str(sym_dir)}):
                    with self.assertRaises(oracle.Gate8EvalError) as ctx_pwd1:
                        audit.verify_certificate_file(Path("c.json"))
                    self.assertEqual(ctx_pwd1.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
                    err_pwd = io.StringIO()
                    with redirect_stderr(err_pwd):
                        self.assertEqual(audit.main(["--check", "c.json"]), 1)
                    self.assertIn("FAIL SYMLINK_ANCESTOR_FORBIDDEN", err_pwd.getvalue())
                os.chdir(real_sub)
                with mock.patch.dict(os.environ, {"PWD": str(sym_dir / "sub")}):
                    with self.assertRaises(oracle.Gate8EvalError) as ctx_pwd2:
                        audit.verify_certificate_file(Path("c.json"))
                    self.assertEqual(ctx_pwd2.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
            finally:
                os.chdir(orig_cwd)

            with self.assertRaises(oracle.Gate8EvalError):
                audit.verify_certificate_file(tmp / "missing.json")

            exact_sz = tmp / "exact_max.json"
            exact_sz.write_bytes(('{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}').encode("ascii"))
            with self.assertRaises(oracle.Gate8EvalError) as ctx_ex:
                audit.verify_certificate_file(exact_sz)
            self.assertEqual(ctx_ex.exception.code, "INVALID_CERT_KEYS")

            over_sz = tmp / "over_max.json"
            over_sz.write_bytes(b" " * (oracle.MAX_JSON_BYTES + 1))
            with self.assertRaises(oracle.Gate8EvalError) as ctx_ov:
                audit.verify_certificate_file(over_sz)
            self.assertEqual(ctx_ov.exception.code, "FILE_TOO_LARGE")

            cr_file = tmp / "cr.json"
            cr_file.write_bytes(oracle.canonical_json_bytes(self.cert) + b"\r\n")
            with self.assertRaises(oracle.Gate8EvalError) as ctx_cr:
                audit.verify_certificate_file(cr_file)
            self.assertEqual(ctx_cr.exception.code, "CR_FORBIDDEN")

            bad_utf8 = tmp / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe")
            with self.assertRaises(oracle.Gate8EvalError):
                audit.verify_certificate_file(bad_utf8)

            pretty = tmp / "pretty.json"
            pretty.write_text(json.dumps(self.cert, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(oracle.Gate8EvalError):
                audit.verify_certificate_file(pretty)

        buf_oj, buf_ot = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_oj):
            self.assertEqual(oracle.main(["--standalone", "--json"]), 0)
        self.assertEqual(json.loads(buf_oj.getvalue())["certificate_sha256"], self.cert["certificate_sha256"])
        with redirect_stdout(buf_ot):
            self.assertEqual(oracle.main(("--standalone",)), 0)
        self.assertIn("OK TRACK-S20B-GATE8-LIVE-EVAL", buf_ot.getvalue())

        self.assertIsNone(oracle.validate_cli_argv(None))
        for bad_argv, want_code in (
            ([123], "INVALID_ARGV"),
            (123, "INVALID_ARGV"),
            ([_StrSub("--standalone")], "INVALID_ARGV"),
            ([chr(0xD800)], "INVALID_UTF8"),
        ):
            err_o, err_a = io.StringIO(), io.StringIO()
            with redirect_stderr(err_o):
                self.assertEqual(oracle.main(bad_argv), 1)  # type: ignore[arg-type]
            self.assertIn(f"FAIL {want_code}", err_o.getvalue())
            with redirect_stderr(err_a):
                self.assertEqual(audit.main(bad_argv), 1)  # type: ignore[arg-type]
            self.assertIn(f"FAIL {want_code}", err_a.getvalue())

        buf_aj, buf_at, buf_ae = io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(buf_aj):
            self.assertEqual(audit.main(["--check", str(cert_path), "--json"]), 0)
        self.assertTrue(json.loads(buf_aj.getvalue())["valid"])
        with redirect_stdout(buf_at):
            self.assertEqual(audit.main(["--standalone"]), 0)
        self.assertIn("OK TRACK-S20B-GATE8-LIVE-AUDIT", buf_at.getvalue())
        with redirect_stderr(buf_ae):
            self.assertEqual(audit.main(["--check", str(_HERE / "nonexistent_cert.json")]), 1)
        self.assertIn("FAIL FILE_NOT_FOUND", buf_ae.getvalue())


if __name__ == "__main__":
    unittest.main()
