#!/usr/bin/env python3
"""
Track S21a (`AUDIT-121A`): R5 Adversarial, Zone-B & Mutation-Killing Test Suite
(`test_conformal_coxeter_r5.py`).

Pure stdlib, zero bare `assert`s (`-O`-safe), zero floats on trust paths,
4-mode `-I` parity verified.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conformal_coxeter_audit as audit  # noqa: E402
import conformal_coxeter_oracle as oracle  # noqa: E402


class _IntSub(int):
    pass


class _DictSub(dict):
    pass


class _StrSub(str):
    pass


class _TupleSub(tuple):
    pass


class _ListSub(list):
    pass


class _EvilPath(type(Path())):
    pass


class TestTrackS21aConformalCoxeterR5(unittest.TestCase):
    cert: dict[str, object]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.generate_certificate()

    def test_01_theta_budgets_shells_and_angular_coxeter_comparison(self) -> None:
        p = self.cert["payload"]
        self.assertIsInstance(p, dict)
        tb = p["theta_ball_budgets"]
        self.assertEqual(tb["e8_3p0_bpw"]["max_shell_m"], 18)
        self.assertEqual(tb["e8_3p0_bpw"]["max_radius_sq"], 36)
        self.assertEqual(tb["e8_3p0_bpw"]["used_points"], 7659601)
        self.assertEqual(tb["e8_3p0_bpw"]["unused_slots"], 729007)
        self.assertEqual(tb["e8_4p5_bpw"]["max_shell_m"], 151)
        self.assertEqual(tb["e8_4p5_bpw"]["used_points"], 34179445441)
        self.assertEqual(tb["e8_4p5_bpw"]["unused_slots"], 180292927)
        self.assertEqual(tb["z8_3p0_bpw"]["max_radius_sq"], 37)
        self.assertEqual(tb["z8_3p0_bpw"]["used_points"], 8069761)
        self.assertEqual(tb["z8_4p5_bpw"]["max_radius_sq"], 302)
        self.assertEqual(tb["z8_4p5_bpw"]["used_points"], 33949336273)

        sh = tb["e8_had_shells_m1_m2_m3"]
        self.assertEqual(sh["m1"]["count"], 240)
        self.assertEqual(sh["m1"]["hamming_weights"], {"1": 16, "4": 224})
        self.assertEqual(sh["m2"]["count"], 2160)
        self.assertEqual(sh["m2"]["hamming_weights"], {"2": 112, "5": 1792, "8": 256})
        self.assertEqual(sh["m3"]["count"], 6720)
        self.assertEqual(sh["m3"]["hamming_weights"], {"3": 448, "4": 896, "6": 5376})

        ang = p["s7_angular_coxeter_comparison"]
        self.assertTrue(ang["block_h2_equals_wht8_shell1_exact"])
        fams = ang["families"]
        self.assertEqual(fams["z8_shell1"]["size"], 16)
        self.assertEqual(fams["z8_shell1"]["sin2_by_direction"]["k1_axis"], "0")
        self.assertEqual(fams["z8_shell1"]["sin2_by_direction"]["k8_all_ones"], "7/8")
        self.assertEqual(fams["d8_shell1"]["size"], 112)
        self.assertEqual(fams["d8_shell1"]["sin2_by_direction"]["k2_pair"], "0")
        self.assertEqual(fams["e8_shell1"]["size"], 240)
        self.assertEqual(fams["e8_shell1"]["sin2_by_direction"]["k1_axis"], "1/2")
        self.assertEqual(fams["e8_shell1"]["sin2_by_direction"]["k8_all_ones"], "0")
        self.assertEqual(fams["e8_had_wht8_shell1"]["sin2_by_direction"]["k1_axis"], "0")
        self.assertEqual(fams["e8_had_wht8_shell1"]["sin2_by_direction"]["k3_triple"], "1/4")
        self.assertEqual(fams["e8_had_wht8_shell1"]["sin2_by_direction"]["k4_hamming_quad"], "0")
        self.assertEqual(fams["e8_had_wht8_shell1"]["sin2_by_direction"]["k4_non_hamming_quad"], "7/16")
        self.assertEqual(fams["block_h2_shell1"], fams["e8_had_wht8_shell1"])
        self.assertEqual(ang["e8_had_shell1_union_shell2"]["sin2_by_direction"]["k1_axis"], "0")
        self.assertEqual(ang["e8_had_shell1_union_shell2"]["sin2_by_direction"]["k2_pair"], "0")
        self.assertEqual(ang["e8_had_shell1_union_shell2"]["sin2_by_direction"]["k8_all_ones"], "0")
        self.assertEqual(ang["e8_had_shell1_union_shell2"]["sin2_by_direction"]["k4_non_hamming_quad"], "7/32")
        self.assertEqual(ang["e8_had_shell1_union_shell2_union_shell3"]["sin2_by_direction"]["k3_triple"], "0")
        self.assertEqual(ang["e8_had_shell1_union_shell2_union_shell3"]["sin2_by_direction"]["k4_non_hamming_quad"], "7/32")

        # Direct boundary & guard checks on theta helpers
        self.assertEqual(oracle.exact_sigma3(1), 1)
        self.assertEqual(oracle.exact_sigma3(4), 1 + 8 + 64)
        self.assertEqual(oracle.exact_e8_ball_budget(1)["used_points"], 1)
        self.assertEqual(oracle.exact_e8_ball_budget(241)["max_shell_m"], 1)
        self.assertEqual(oracle.exact_z8_ball_budget(17, k_limit=2)["max_radius_sq"], 1)
        with self.assertRaises(oracle.ConformalError) as ctx_zlim:
            oracle.exact_z8_ball_budget(1000, k_limit=1)
        self.assertEqual(ctx_zlim.exception.code, "Z8_LIMIT_TOO_SMALL")
        for bad_theta in (
            lambda: oracle.exact_sigma3(0),
            lambda: oracle.exact_sigma3(True),
            lambda: oracle.exact_sigma3(100_000_001),
            lambda: oracle.exact_e8_ball_budget(0),
            lambda: oracle.exact_e8_ball_budget(True),
            lambda: oracle.exact_e8_ball_budget((1 << 64) + 1),
            lambda: oracle.exact_z8_ball_budget(0),
            lambda: oracle.exact_z8_ball_budget((1 << 64) + 1),
        ):
            with self.assertRaises(oracle.ConformalError):
                bad_theta()

    def test_02_conformal_ladders_five_ensembles_and_obstruction(self) -> None:
        p = self.cert["payload"]
        ladders = {row["rho"]: row for row in p["conformal_radial_ladders"]}
        self.assertEqual(ladders["2"]["max_rel_sq_err"], "1/9")
        self.assertEqual(ladders["2"]["uniform_rel_mse"], "1/28")
        self.assertEqual(ladders["9/8"]["max_rel_sq_err"], "1/289")
        self.assertEqual(ladders["9/8"]["uniform_rel_mse"], "1/868")
        self.assertEqual(ladders["17/16"]["uniform_rel_mse"], "1/3268")

        wing = p["graded_conformal_wing_budget"]
        self.assertEqual(wing["inner_cavity_points"], 11520)
        self.assertEqual(wing["outer_heavy_tail_points"], 48000)
        self.assertEqual(wing["total_wing_points"], 59520)
        self.assertTrue(wing["fits_3p0_unused_slots_zero_extra_bits"])
        self.assertTrue(wing["fits_4p5_unused_slots_zero_extra_bits"])
        self.assertEqual(wing["coupled_128d_cardinal_ratio_3p0"], "59520/7659601")

        ens = p["five_ensemble_distortion_law"]
        self.assertEqual(ens["e1_inner_cavity"]["flat_hbg_total_err"], "15/8")
        self.assertEqual(ens["e1_inner_cavity"]["graded_wing_total_err"], "0")
        e2 = ens["e2_bulk_typical_shell"]
        self.assertEqual(e2["total_signal_energy"], "359/8")
        self.assertEqual(e2["flat_hbg_total_err"], "15/8")
        self.assertEqual(e2["graded_wing_total_err"], "15/8")
        self.assertTrue(e2["graded_zero_bulk_loss"])
        self.assertEqual(e2["uniform_conformal_m3_24shells_total_err"], "3682629051/536870912")
        self.assertEqual(e2["uniform_over_flat_bulk_penalty_ratio"], "1227543017/335544320")
        self.assertEqual(e2["uniform_vs_flat_scale_inflation_crossover_norm_sq"], "11047887153/83886080")

        e3 = ens["e3_heavy_tail_outliers_3p0"]
        self.assertEqual(len(e3), 4)
        self.assertTrue(all(row["wing_beats_flat_clip_lb"] for row in e3))
        self.assertEqual(e3[3]["norm_sq"], "256")

        e4 = ens["e4_inner_steiner_triple"]
        self.assertEqual(e4["flat_hbg_snapped_4th_coord_s2"], 1)
        self.assertEqual(e4["flat_hbg_sq_err"], "9801/20000")
        self.assertEqual(e4["graded_wing_sq_err"], "1/20000")
        self.assertEqual(e4["exact_reduction_factor"], "9801")

        e5 = ens["e5_bulk_steiner_obstruction"]
        self.assertEqual(e5["flat_hbg_snapped_4th_coord_s2"], 1)
        self.assertEqual(e5["flat_hbg_sq_err"], "9801/20000")
        self.assertEqual(e5["graded_wing_sq_err"], "9801/20000")
        self.assertTrue(e5["obstruction_persists_in_bulk"])
        self.assertEqual(
            e5["min_continuous_radial_shell_proj_err_y_e5"],
            {"m1": "20001/20000", "m2": "30001/20000", "m3": "35937/80000"},
        )
        self.assertEqual(
            e5["min_continuous_radial_shell_proj_err_y0_unperturbed"],
            {"m1": "1", "m2": "3/2", "m3": "11/24"},
        )
        self.assertEqual(e5["z8_separable_bleed_err"], "0")
        self.assertEqual(e5["z8_total_sq_err"], "1/20000")
        self.assertEqual(e5["shell3_continuous_over_z8_total_ratio"], "35937/4")

        # Direct CVP, angular, conformal-shell & ladder Zone-B guards (kills all branches & BoolOps)
        self.assertEqual(oracle._closest_d8_int((6, 0, 0, 0, 0, 0, 0, 0), 10), ((0,) * 8, 36))
        self.assertEqual(oracle._closest_d8_int((-6, 0, 0, 0, 0, 0, 0, 0), 10), ((0,) * 8, 36))
        self.assertEqual(oracle._closest_d8_int((11, 4, 0, 0, 0, 0, 0, 0), 10)[0], (1, 1, 0, 0, 0, 0, 0, 0))
        self.assertEqual(oracle._closest_d8_int((11, -4, 0, 0, 0, 0, 0, 0), 10)[0], (1, -1, 0, 0, 0, 0, 0, 0))
        y_unit = (Fraction(1),) + (Fraction(0),) * 7
        q_h, d_h = oracle.closest_e8_had_s2_exact([Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)])  # type: ignore[arg-type]
        self.assertEqual(q_h, (1, 1, 1, 1, 0, 0, 0, 0))
        self.assertEqual(d_h, Fraction(0))
        self.assertEqual(oracle.angular_sin2_exact([1, 0, 0, 0, 0, 0, 0, 0], [(1, 0, 0, 0, 0, 0, 0, 0)], 1), Fraction(0))  # type: ignore[arg-type]
        self.assertEqual(oracle.angular_sin2_exact((1, 1, 0, 0, 0, 0, 0, 0), ((1, 0, 0, 0, 0, 0, 0, 0),), 1), Fraction(1, 2))
        self.assertEqual(
            oracle.best_conformal_shell_err_exact(
                y_unit,
                [(Fraction(1, 2), 1), (Fraction(1, 4), 1)],
                {1: ((2, 0, 0, 0, 0, 0, 0, 0),)},
            ),
            Fraction(0),
        )
        unit_code = ((1, 0, 0, 0, 0, 0, 0, 0),)
        ok_shells = {1: ((2, 0, 0, 0, 0, 0, 0, 0),)}
        for bad_call in (
            lambda: oracle.closest_e8_had_s2_exact(None),  # type: ignore[arg-type]
            lambda: oracle.closest_e8_had_s2_exact(_TupleSub(y_unit)),
            lambda: oracle.closest_e8_had_s2_exact((Fraction(1),)),
            lambda: oracle.closest_e8_had_s2_exact((1,) * 8),  # type: ignore[arg-type]
            lambda: oracle.closest_e8_had_s2_exact((Fraction(10**5000, 1),) + (Fraction(0),) * 7),
            lambda: oracle.best_conformal_shell_err_exact(None, [], ok_shells),  # type: ignore[arg-type]
            lambda: oracle.best_conformal_shell_err_exact((Fraction(1),), [], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact((1,) * 8, [], ok_shells),  # type: ignore[arg-type]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, None, ok_shells),  # type: ignore[arg-type]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, _ListSub(), ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [], None),  # type: ignore[arg-type]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [], _DictSub()),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [None], ok_shells),  # type: ignore[list-item]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1),)], ok_shells),  # type: ignore[list-item]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(1, 1)], ok_shells),  # type: ignore[list-item]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(0), 1)], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(-1), 1)], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(10**5000, 1), 1)], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 0)], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), True)], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 100_000_001)], {100_000_001: ((2, 0, 0, 0, 0, 0, 0, 0),)}),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 1 << 2000)], {1 << 2000: ((2, 0, 0, 0, 0, 0, 0, 0),)}),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 99)], ok_shells),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 1)], {1: ()}),
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 1)], {1: None}),  # type: ignore[dict-item]
            lambda: oracle.best_conformal_shell_err_exact(y_unit, [(Fraction(1), 1)], {1: ((True,) + (0,) * 7,)}),
            lambda: oracle.best_conformal_shell_err_exact((Fraction(2),) + (Fraction(0),) * 7, [(Fraction(1), 1)], {1: ((100, 0, 0, 0, 0, 0, 0, 0),)}),
            lambda: oracle.angular_sin2_exact(None, unit_code, 1),  # type: ignore[arg-type]
            lambda: oracle.angular_sin2_exact(_ListSub([1, 0, 0, 0, 0, 0, 0, 0]), unit_code, 1),
            lambda: oracle.angular_sin2_exact((1, 2), unit_code, 1),
            lambda: oracle.angular_sin2_exact((True,) + (0,) * 7, unit_code, 1),
            lambda: oracle.angular_sin2_exact((0,) * 8, unit_code, 1),
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), unit_code, 0),
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), unit_code, True),
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), None, 1),  # type: ignore[arg-type]
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), (), 1),
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), (None,), 1),  # type: ignore[arg-type]
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), ((True,) + (0,) * 7,), 1),
            lambda: oracle.angular_sin2_exact((1, 0, 0, 0, 0, 0, 0, 0), ((2, 0, 0, 0, 0, 0, 0, 0),), 1),
            lambda: oracle.log_radial_ladder_metrics(2),  # type: ignore[arg-type]
            lambda: oracle.log_radial_ladder_metrics(Fraction(1, 1)),
            lambda: oracle.log_radial_ladder_metrics(Fraction(1, 2)),
            lambda: oracle.log_radial_ladder_metrics(Fraction(10**5000, 1)),
        ):
            with self.assertRaises(oracle.ConformalError):
                bad_call()

    def test_03_zone_b_strict_io_and_bounds(self) -> None:
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
            lambda: oracle.require_int(1, "huge_lo", lo=10**5000),
            lambda: oracle.require_int(1, "huge_hi", hi=-(10**5000)),
            lambda: oracle.require_int(5, "bool_lo", lo=True),
            lambda: oracle.require_int(5, "frac_lo", lo=Fraction(10, 1)),  # type: ignore[arg-type]
            lambda: oracle.require_int(5, "str_lo", lo="x"),  # type: ignore[arg-type]
            lambda: oracle.require_int(5, "bool_hi", hi=True),
            lambda: oracle.require_int(5, "frac_hi", hi=Fraction(1, 1)),  # type: ignore[arg-type]
        ):
            with self.assertRaises(oracle.ConformalError):
                bad_call()

        self.assertIsNone(oracle.validate_cli_argv(None))
        self.assertEqual(oracle.validate_cli_argv(("--standalone",)), ["--standalone"])
        for bad_argv in (123, [1], [None], [_StrSub("--json")], [chr(0xD800)]):
            with self.assertRaises(oracle.ConformalError):
                oracle.validate_cli_argv(bad_argv)

        good_hex = "0123456789abcdef" * 4
        other_hex = "fedcba9876543210" * 4
        self.assertTrue(oracle.constant_time_hex64_equal(good_hex, good_hex))
        self.assertFalse(oracle.constant_time_hex64_equal(good_hex, other_hex))
        for bad_hex in (123, "a" * 63, "a" * 65, "A" * 64, "\u00e9" * 64, chr(0xD800) * 64):
            with self.assertRaises(oracle.ConformalError):
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
            with self.assertRaises(oracle.ConformalError):
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
            with self.assertRaises(oracle.ConformalError):
                oracle.validate_json_value(bad_val)
        with self.assertRaises(oracle.ConformalError) as ctx_flt:
            oracle.validate_json_value({"x": 1.5})
        self.assertEqual(ctx_flt.exception.code, "FLOAT_FORBIDDEN")

        self.assertEqual(oracle._parse_capped_int_token("-42"), -42)
        self.assertEqual(oracle._parse_capped_int_token(str(max_2048)), max_2048)
        with self.assertRaises(oracle.ConformalError) as ctx_d620:
            oracle._parse_capped_int_token("1" + ("0" * (oracle.MAX_INT_DIGITS - 1)))
        self.assertEqual(ctx_d620.exception.code, "INT_BIT_LIMIT")
        with self.assertRaises(oracle.ConformalError) as ctx_d621:
            oracle._parse_capped_int_token("9" * (oracle.MAX_INT_DIGITS + 1))
        self.assertEqual(ctx_d621.exception.code, "INT_DIGIT_LIMIT")
        with self.assertRaises(oracle.ConformalError) as ctx_b2049:
            oracle._parse_capped_int_token(str(over_2048))
        self.assertEqual(ctx_b2049.exception.code, "INT_BIT_LIMIT")

        self.assertEqual(oracle.parse_strict_json_text('{"k":1}'), {"k": 1})
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])
        exact_max_str = '{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}'
        self.assertEqual(len(exact_max_str.encode("utf-8")), oracle.MAX_JSON_BYTES)
        self.assertEqual(len(oracle.parse_strict_json_text(exact_max_str)["a"]), oracle.MAX_JSON_BYTES - 8)
        with self.assertRaises(oracle.ConformalError) as ctx_empty:
            oracle.parse_strict_json_text("   ")
        self.assertEqual(ctx_empty.exception.code, "EMPTY_JSON_INPUT")
        with self.assertRaises(oracle.ConformalError) as ctx_rb:
            oracle.parse_strict_json_text('{"k":1}', require_object=0)  # type: ignore[arg-type]
        self.assertEqual(ctx_rb.exception.code, "INVALID_BOOL_TYPE")
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
            with self.assertRaises(oracle.ConformalError):
                oracle.parse_strict_json_text(bad_txt)  # type: ignore[arg-type]

    def test_04_auditor_file_verification_tamper_and_cli(self) -> None:
        cert_path = _HERE / "conformal_coxeter_certificate.json"
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
            with self.assertRaises(oracle.ConformalError) as ctx_c:
                audit.verify_certificate_dict(bad_cert)
            self.assertEqual(ctx_c.exception.code, want_code)

        for bad_p in ("not_a_path", _EvilPath(str(cert_path)), Path("bad" + chr(0xD800)), Path("bad\x00path")):
            with self.assertRaises(oracle.ConformalError):
                audit.require_no_symlink_path(bad_p)  # type: ignore[arg-type]
        with self.assertRaises(oracle.ConformalError) as ctx_ep:
            audit.verify_certificate_file(_EvilPath(str(cert_path)))
        self.assertEqual(ctx_ep.exception.code, "INVALID_PATH_TYPE")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            real_dir = tmp / "real"
            real_dir.mkdir()
            real_file = real_dir / "c.json"
            real_file.write_bytes(oracle.canonical_json_bytes(self.cert))
            self.assertTrue(audit.verify_certificate_file(real_file)["valid"])

            sym_file = tmp / "sym.json"
            sym_file.symlink_to(real_file)
            with self.assertRaises(oracle.ConformalError):
                audit.verify_certificate_file(sym_file)

            sym_dir = tmp / "symdir"
            sym_dir.symlink_to(real_dir)
            with self.assertRaises(oracle.ConformalError):
                audit.verify_certificate_file(sym_dir / "c.json")

            # Relative-path symlinked PWD guards (S21A-DS-02 / D7)
            sub_dir = real_dir / "sub"
            sub_dir.mkdir()
            (sub_dir / "c.json").write_bytes(oracle.canonical_json_bytes(self.cert))
            other_dir = tmp / "other"
            other_dir.mkdir()
            (other_dir / "c.json").write_bytes(oracle.canonical_json_bytes(self.cert))
            old_cwd = os.getcwd()
            old_pwd = os.environ.get("PWD")
            try:
                os.chdir(real_dir)
                # 1. Clean PWD matching real_dir -> relative path passes
                os.environ["PWD"] = str(real_dir)
                self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                # 2. Unset PWD -> relative path passes cleanly
                os.environ.pop("PWD", None)
                self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                # 3. Symlinked PWD matching cwd -> rejected with SYMLINK_ANCESTOR_FORBIDDEN
                os.environ["PWD"] = str(sym_dir)
                with self.assertRaises(oracle.ConformalError) as ctx_pwd1:
                    audit.verify_certificate_file(Path("c.json"))
                self.assertEqual(ctx_pwd1.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
                # 4. Child dir under symlinked ancestor in PWD -> rejected
                os.chdir(sub_dir)
                os.environ["PWD"] = str(sym_dir / "sub")
                with self.assertRaises(oracle.ConformalError) as ctx_pwd2:
                    audit.verify_certificate_file(Path("c.json"))
                self.assertEqual(ctx_pwd2.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
                # 5. Stale symlink PWD pointing elsewhere -> passes
                os.chdir(other_dir)
                os.environ["PWD"] = str(sym_dir)
                self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
            finally:
                os.chdir(old_cwd)
                if old_pwd is None:
                    os.environ.pop("PWD", None)
                else:
                    os.environ["PWD"] = old_pwd

            with self.assertRaises(oracle.ConformalError):
                audit.verify_certificate_file(tmp / "missing.json")

            exact_sz = tmp / "exact_max.json"
            exact_sz.write_bytes(('{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}').encode("ascii"))
            with self.assertRaises(oracle.ConformalError) as ctx_ex:
                audit.verify_certificate_file(exact_sz)
            self.assertEqual(ctx_ex.exception.code, "INVALID_CERT_KEYS")

            over_sz = tmp / "over_max.json"
            over_sz.write_bytes(b" " * (oracle.MAX_JSON_BYTES + 1))
            with self.assertRaises(oracle.ConformalError) as ctx_ov:
                audit.verify_certificate_file(over_sz)
            self.assertEqual(ctx_ov.exception.code, "FILE_TOO_LARGE")

            cr_file = tmp / "cr.json"
            cr_file.write_bytes(oracle.canonical_json_bytes(self.cert) + b"\r\n")
            with self.assertRaises(oracle.ConformalError) as ctx_cr:
                audit.verify_certificate_file(cr_file)
            self.assertEqual(ctx_cr.exception.code, "CR_FORBIDDEN")

            bad_utf8 = tmp / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe")
            with self.assertRaises(oracle.ConformalError):
                audit.verify_certificate_file(bad_utf8)

            pretty = tmp / "pretty.json"
            pretty.write_text(json.dumps(self.cert, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(oracle.ConformalError):
                audit.verify_certificate_file(pretty)

        buf_oj, buf_ot, buf_oe = io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(buf_oj):
            self.assertEqual(oracle.main(["--standalone", "--json"]), 0)
        self.assertEqual(json.loads(buf_oj.getvalue())["certificate_sha256"], self.cert["certificate_sha256"])
        with redirect_stdout(buf_ot):
            self.assertEqual(oracle.main(["--standalone"]), 0)
        self.assertIn("OK TRACK-S21A-CONFORMAL-COXETER", buf_ot.getvalue())
        with redirect_stderr(buf_oe):
            self.assertEqual(oracle.main([1]), 1)  # type: ignore[list-item]
        self.assertIn("FAIL INVALID_ARGV", buf_oe.getvalue())

        buf_aj, buf_at, buf_ae, buf_ae2 = io.StringIO(), io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(buf_aj):
            self.assertEqual(audit.main(["--check", str(cert_path), "--json"]), 0)
        self.assertTrue(json.loads(buf_aj.getvalue())["valid"])
        with redirect_stdout(buf_at):
            self.assertEqual(audit.main(["--standalone"]), 0)
        self.assertIn("OK TRACK-S21A-AUDIT", buf_at.getvalue())
        with redirect_stderr(buf_ae):
            self.assertEqual(audit.main(["--check", str(_HERE / "nonexistent_cert.json")]), 1)
        self.assertIn("FAIL FILE_NOT_FOUND", buf_ae.getvalue())
        with redirect_stderr(buf_ae2):
            self.assertEqual(audit.main([1]), 1)  # type: ignore[list-item]
        self.assertIn("FAIL INVALID_ARGV", buf_ae2.getvalue())


if __name__ == "__main__":
    unittest.main()
