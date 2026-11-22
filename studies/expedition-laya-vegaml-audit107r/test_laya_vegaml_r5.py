#!/usr/bin/env python3
"""Assertion-free R5 mutation-killing, AST-conjunct-killing, and Zone-B adversarial test suite
for AUDIT-107 (Track S10: Laya & VegaML pure-Q operator bridges).

Run with: python3 -I -O -B -W error test_laya_vegaml_r5.py -v
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from laya_vegaml_oracle import (
    EVIDENCE_INCOMPARABLE,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_MISSING,
    EVIDENCE_PRESENT,
    EVIDENCE_UNKNOWN,
    LayaVegamlError,
    _f64_dyadic_rne_fraction,
    canonical_sha256,
    certify_empty_conformal_non_trigger_exact,
    certify_epistemic_lattice_laws_exact,
    certify_guardrail_781_inversion_and_cap_exact,
    certify_ieee754_conformal_ceil_overshoot_exact,
    certify_insample_coupling_counterexample_exact,
    certify_quantized_reader_port_recovery_exact,
    conformal_lac_qhat_exact,
    contains_float_recursive,
    epistemic_meet_exact,
    evaluate_evidence_promotion_exact,
    format_exact_decimal,
    format_exact_percent,
    frac_to_dict,
    inspect_calibration_fail_closed_exact,
    parse_exact_decimal_to_fraction,
    require_exact_bool,
    require_exact_fraction,
    require_exact_int,
    require_git_sha40,
    require_sha256,
    route_answer_fail_closed_exact,
    route_laya_vega_default_exact,
    solve_linear_system_exact,
    solve_ridge_port_exact,
    wilson_lower_meets_target_exact,
)
from laya_vegaml_audit import (
    EXPECTED_UNMERGED_PRS,
    PINNED_ARTIFACTS_EXPECTED,
    PINNED_INTAKE_FILES,
    PINNED_REPO_HEADS_EXPECTED,
    PINNED_UNMERGED_PRS_EXPECTED,
    _audit_unmerged_pr_snapshot,
    main as audit_main,
    read_git_head_readonly,
    run_live_audit,
    run_standalone_audit,
    sha256_file,
    validate_audit_certificate,
)


def _build_synthetic_intake(root: Path, *, packed_ref: bool = False) -> None:
    laya_sha = "1234567890abcdef1234567890abcdef12345678"
    vega_sha = "abcdef1234567890abcdef1234567890abcdef12"
    (root / "laya" / ".git" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (root / "laya" / "laya").mkdir(parents=True, exist_ok=True)
    (root / "laya" / "docs").mkdir(parents=True, exist_ok=True)
    (root / "vegaml" / ".git" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (root / "vegaml" / "vegaml").mkdir(parents=True, exist_ok=True)

    if packed_ref:
        (root / "laya" / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (root / "laya" / ".git" / "packed-refs").write_text(
            f"# refs/heads/main\n^deadbeef\n{laya_sha} refs/heads/main\n",
            encoding="utf-8",
        )
        (root / "vegaml" / ".git" / "HEAD").write_text(vega_sha + "\n", encoding="utf-8")
    else:
        (root / "laya" / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (root / "laya" / ".git" / "refs" / "heads" / "main").write_text(laya_sha + "\n", encoding="utf-8")
        (root / "vegaml" / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (root / "vegaml" / ".git" / "refs" / "heads" / "main").write_text(vega_sha + "\n", encoding="utf-8")

    (root / "laya" / "laya" / "evidence.py").write_text(
        'PRESENT="PRESENT"\nUNKNOWN="UNKNOWN"\nINSUFFICIENT="INSUFFICIENT"\n'
        'def inspect_calibration(cfg):\n'
        '    e = cfg["training"]["laya_train_calibration"]["choice"]\n'
        '    if isinstance(e.get("items"), bool) or not isinstance(e.get("items"), int):\n'
        '        return {"state": INSUFFICIENT, "types": {"choice": {"state": UNKNOWN}}}\n'
        '    return {"state": PRESENT, "types": {"choice": {"state": PRESENT}}}\n',
        encoding="utf-8",
    )
    (root / "laya" / "laya" / "vega.py").write_text(
        'class _OtherAgent:\n'
        '    pass\n'
        'class VegaAgent:\n'
        '    def _other(self):\n'
        '        return {}\n'
        '    def _route(self, raw, folded, min_conf):\n'
        '        return {"abstain": folded.get("abstain")}\n',
        encoding="utf-8",
    )
    (root / "laya" / "laya" / "confidence.py").write_text("# mock confidence\n", encoding="utf-8")
    (root / "laya" / "laya" / "router.py").write_text("# mock router\n", encoding="utf-8")
    (root / "laya" / "docs" / "guardrail-limitations.md").write_text(
        "| id | shape | jailbreak | prompt_injection | harm_severity |\n"
        "| c011 | meeting | 0.000 | 0.182 | 0.82 |\n"
        "| c012 | faq | 0.005 | 0.301 | 0.78 |\n"
        "| c042 | report | 0.000 | 0.000 | 0.51 |\n"
        "**Inversion.** c038 is a benign passage.\nIt scores 0.619 and is blocked.\n",
        encoding="utf-8",
    )
    (root / "vegaml" / "vegaml" / "calibrate.py").write_text(
        'def _helper():\n'
        '    return 0\n'
        'def _conformal_qhat(scores, alpha):\n'
        '    return 1.0 - alpha\n'
        'def calibrate(rows, alpha):\n'
        '    t = _fit_temperature(rows)\n'
        '    sc = [(p, y) for p, y in rows]\n'
        '    return _conformal_qhat(sc, alpha)\n',
        encoding="utf-8",
    )
    (root / "vegaml" / "vegaml" / "band.py").write_text(
        'class _OtherBand:\n'
        '    pass\n'
        'class Band:\n'
        '    def _other(self):\n'
        '        return {}\n'
        '    def route(self, answer, qid):\n'
        '        return {"route": "unbanded", "review": False}\n',
        encoding="utf-8",
    )
    (root / "vegaml" / "vegaml" / "router.py").write_text("# mock\n", encoding="utf-8")
    (root / "vegaml" / "vegaml" / "_attach.py").write_text("# mock\n", encoding="utf-8")
    (root / "vegaml" / "vegaml" / "_universal.py").write_text("# mock\n", encoding="utf-8")
    (root / "vegaml" / "vegaml" / "_common.py").write_text("# mock\n", encoding="utf-8")

    prs = [
        {
            "repo": repo,
            "number": num,
            "state": "closed" if num == 1502 else "open",
            "merged": False,
            "head_sha": f"{num:04x}" * 10,
            "title": f"Mock PR {num}",
        }
        for repo, num in EXPECTED_UNMERGED_PRS
    ]
    (root / "unmerged_prs_snapshot.json").write_text(json.dumps({"prs": prs}), encoding="utf-8")


class TestOraclePrimitivesAndPillar1(unittest.TestCase):
    def test_strict_validators_decimals_and_float_detector(self) -> None:
        self.assertEqual(require_exact_int(0, "x", allow_zero=True), 0)
        self.assertEqual(require_exact_int(1, "x", allow_zero=False), 1)
        self.assertIs(require_exact_bool(True, "b"), True)
        self.assertEqual(require_exact_fraction(Fraction(3, 4), "f"), Fraction(3, 4))
        self.assertEqual(require_sha256("ab" * 32, "s"), "ab" * 32)
        self.assertEqual(require_git_sha40("ab" * 20, "g"), "ab" * 20)

        for bad_i in (True, False, "1", -1):
            with self.assertRaises(LayaVegamlError):
                require_exact_int(bad_i, "x", allow_zero=True)
        with self.assertRaises(LayaVegamlError):
            require_exact_int(0, "x", allow_zero=False)
        for bad_b in (0, 1, "true"):
            with self.assertRaises(LayaVegamlError):
                require_exact_bool(bad_b, "b")
        with self.assertRaises(LayaVegamlError):
            require_exact_fraction(1, "f")
        for bad_s in (123, "short", "A" * 64, "0" * 64):
            with self.assertRaises(LayaVegamlError):
                require_sha256(bad_s, "s")
        for bad_g in (123, "short", "A" * 40, "0" * 40):
            with self.assertRaises(LayaVegamlError):
                require_git_sha40(bad_g, "g")

        # Strict decimal parser
        self.assertEqual(parse_exact_decimal_to_fraction("0.182", "d"), Fraction(182, 1000))
        self.assertEqual(parse_exact_decimal_to_fraction("-42", "d"), Fraction(-42, 1))
        self.assertEqual(parse_exact_decimal_to_fraction("-0.25", "d"), Fraction(-1, 4))
        self.assertEqual(parse_exact_decimal_to_fraction("0", "d"), Fraction(0, 1))
        self.assertEqual(parse_exact_decimal_to_fraction("0.0", "d"), Fraction(0, 1))
        for bad_dec in (123, "+1.0", "1_0", " 1.0", "1e-3", "01.2", "1" * 65, "nan"):
            with self.assertRaisesRegex(LayaVegamlError, "strict decimal"):
                parse_exact_decimal_to_fraction(bad_dec, "d")
        for neg_zero in ("-0", "-0.0", "-0.000"):
            with self.assertRaisesRegex(LayaVegamlError, "negative zero"):
                parse_exact_decimal_to_fraction(neg_zero, "d")

        # Formatters
        self.assertEqual(frac_to_dict(Fraction(6, 8)), {"num": 3, "den": 4})
        self.assertEqual(format_exact_decimal(Fraction(-3, 4), 2), "-0.75")
        self.assertEqual(format_exact_decimal(Fraction(7, 2), 0), "4")
        self.assertEqual(format_exact_percent(Fraction(1, 8), 2), "12.50%")

        # Recursive float/complex/set/bytes/non-str-key detector
        self.assertFalse(contains_float_recursive({"a": [1, "x", (True, None)]}))
        self.assertTrue(contains_float_recursive(json.loads('{"x": 1.5}')))
        self.assertTrue(contains_float_recursive({1: "int_key"}))
        self.assertTrue(contains_float_recursive(complex(1, 2)))
        self.assertTrue(contains_float_recursive({1, 2}))
        self.assertTrue(contains_float_recursive(frozenset([1])))
        self.assertTrue(contains_float_recursive(b"bytes"))
        self.assertTrue(contains_float_recursive([json.loads("0.5")]))

        with self.assertRaises(LayaVegamlError):
            canonical_sha256(123)  # type: ignore[arg-type]
        with self.assertRaises(LayaVegamlError):
            canonical_sha256({"x": complex(1, 0)})

        # Exact dyadic RNE helper boundaries, positive range, shift<0, carry & tie-to-even
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(1, 2)), Fraction(1, 2))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(3, 2)), Fraction(3, 2))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(20, 1)), Fraction(20, 1))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(5, 3)), Fraction(7505999378950827, 1 << 52))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(3, 7)), Fraction(7720456504063707, 1 << 54))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction((1 << 54) + 3, 1)), Fraction((1 << 54) + 4, 1))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(2, 1) - Fraction(1, 1 << 54)), Fraction(2, 1))
        self.assertEqual(_f64_dyadic_rne_fraction(Fraction(1 << 1023, 1)), Fraction(1 << 1023, 1))
        # Exact tie at 2^-53: 1/2 + 2^-54 -> even q stays 1/2; 1/2 + 3*2^-54 -> odd q+1 rounds up to (2^52+2)/2^53
        tie_even = Fraction((1 << 53) + 1, 1 << 54)
        tie_odd = Fraction((1 << 53) + 3, 1 << 54)
        below_half = Fraction((1 << 54) + 5, 1 << 55)  # odd q=(1<<52)+1, rem=1/4 < 1/2 -> rounds down!
        above_half = Fraction((1 << 54) + 3, 1 << 55)  # even q=(1<<52), rem=3/4 > 1/2 -> rounds up!
        self.assertEqual(_f64_dyadic_rne_fraction(tie_even), Fraction(1, 2))
        self.assertEqual(_f64_dyadic_rne_fraction(tie_odd), Fraction((1 << 51) + 1, 1 << 52))
        self.assertEqual(_f64_dyadic_rne_fraction(below_half), Fraction((1 << 52) + 1, 1 << 53))
        self.assertEqual(_f64_dyadic_rne_fraction(above_half), Fraction((1 << 52) + 1, 1 << 53))
        for bad_d in (Fraction(0, 1), Fraction(-1, 2)):
            with self.assertRaisesRegex(LayaVegamlError, "strictly positive"):
                _f64_dyadic_rne_fraction(bad_d)
        for oob_d in (Fraction(1, 1 << 1023), Fraction(1 << 1024, 1)):
            with self.assertRaisesRegex(LayaVegamlError, "exponent range"):
                _f64_dyadic_rne_fraction(oob_d)
        with self.assertRaisesRegex(LayaVegamlError, "overflows"):
            _f64_dyadic_rne_fraction(Fraction((1 << 1024) - (1 << 970), 1))

    def test_pillar1_split_conformal_and_coupling_theorem(self) -> None:
        # Exact rank + unclamped > n clamp boundary (n=2, alpha=1/10 -> ceil(2.7)=3 -> clamped to 2)
        clamped = conformal_lac_qhat_exact([Fraction(1, 5), Fraction(4, 5)], Fraction(1, 10))
        self.assertEqual(clamped["rank"], 2)
        self.assertEqual(clamped["qhat"], {"num": 4, "den": 5})

        exact19 = conformal_lac_qhat_exact([Fraction(i, 20) for i in range(1, 20)], Fraction(1, 10))
        self.assertEqual(exact19["rank"], 18)
        self.assertEqual(exact19["qhat"], {"num": 9, "den": 10})
        self.assertTrue(exact19["meets_nominal_target"])

        for bad_scores in (123, [], "ab", b"ab"):
            with self.assertRaisesRegex(LayaVegamlError, "non-empty sequence"):
                conformal_lac_qhat_exact(bad_scores, Fraction(1, 10))  # type: ignore[arg-type]
        for bad_s in (Fraction(-1, 10), Fraction(11, 10)):
            with self.assertRaises(LayaVegamlError):
                conformal_lac_qhat_exact([bad_s], Fraction(1, 10))
        for bad_a in (Fraction(0, 1), Fraction(1, 1)):
            with self.assertRaises(LayaVegamlError):
                conformal_lac_qhat_exact([Fraction(1, 2)], bad_a)

        # True IEEE-754 binary64 overshoots (both factors and product rounded in binary64!)
        w149 = certify_ieee754_conformal_ceil_overshoot_exact(149, Fraction(18, 100))
        self.assertEqual(w149["exact_rank"], 123)
        self.assertEqual(w149["binary64_rank"], 124)
        self.assertTrue(w149["overshoot_witnessed"])

        w29 = certify_ieee754_conformal_ceil_overshoot_exact(29, Fraction(7, 10))
        self.assertEqual(w29["exact_rank"], 9)
        self.assertEqual(w29["binary64_rank"], 10)
        self.assertTrue(w29["overshoot_witnessed"])

        # LAYA107-BUG-01 regression: binary64 product rounds back to exact integer (zero false positives!)
        w19 = certify_ieee754_conformal_ceil_overshoot_exact(19, Fraction(1, 10))
        self.assertEqual(w19["exact_rank"], 18)
        self.assertEqual(w19["binary64_rank"], 18)
        self.assertFalse(w19["overshoot_witnessed"])

        w14 = certify_ieee754_conformal_ceil_overshoot_exact(14, Fraction(1, 5))
        self.assertEqual(w14["exact_rank"], 12)
        self.assertEqual(w14["binary64_rank"], 12)
        self.assertFalse(w14["overshoot_witnessed"])

        # Extreme alpha within half-ULP of 1 rounds a_f64 -> 1.0 -> 1.0 - a_f64 == 0
        w_near1 = certify_ieee754_conformal_ceil_overshoot_exact(10, Fraction((1 << 55) - 1, 1 << 55))
        self.assertEqual(w_near1["binary64_rank"], 0)
        self.assertFalse(w_near1["overshoot_witnessed"])

        coup = certify_insample_coupling_counterexample_exact()
        self.assertTrue(coup["theorem_holds"])
        self.assertEqual(coup["disjoint_exact_coverage"], {"num": 19, "den": 20})
        self.assertEqual(coup["coupled_exact_coverage"], {"num": 1, "den": 2})
        self.assertEqual(coup["zero_coverage_coupled_splits"], 10)


class TestPillar2AndPillar3(unittest.TestCase):
    def test_pillar2_empty_set_wilson_and_routers(self) -> None:
        r_nt = certify_empty_conformal_non_trigger_exact(2, Fraction(1, 2))
        self.assertFalse(r_nt["abstention_reachable"])
        self.assertEqual(r_nt["uniform_pset_size"], 2)
        self.assertTrue(r_nt["iff_verified"])

        r_tr = certify_empty_conformal_non_trigger_exact(2, Fraction(2, 5))
        self.assertTrue(r_tr["abstention_reachable"])
        self.assertEqual(r_tr["uniform_pset_size"], 0)
        self.assertTrue(r_tr["iff_verified"])

        with self.assertRaises(LayaVegamlError):
            certify_empty_conformal_non_trigger_exact(1, Fraction(1, 2))

        # Exact Wilson lower bound: exact-square (12/12) & non-square (10/12) + center-fail vs quad-fail!
        z95 = Fraction(16449, 10000)
        w_ok = wilson_lower_meets_target_exact(12, 12, z95, Fraction(816, 1000))
        self.assertTrue(w_ok["meets_target"])
        self.assertEqual(w_ok["lb_floor_6dp"], "0.816010")
        self.assertEqual(w_ok["lb_ceil_6dp"], "0.816011")

        # 12/12 at tau=0.817: center_ok=True (0.817 <= 0.908), quad_ok=False! (kills quad_ok conjunct!)
        w_quad_fail = wilson_lower_meets_target_exact(12, 12, z95, Fraction(817, 1000))
        self.assertTrue(w_quad_fail["center_ok"])
        self.assertFalse(w_quad_fail["quadratic_ok"])
        self.assertFalse(w_quad_fail["meets_target"])

        # 12/12 at tau=0.95: center_ok=False (0.95 > 0.908), quad_ok=False; test center_ok=False with quad_ok=True!
        # For h=6, n=12, z=1: center=0.5, upper root U ~ 0.635, so tau=0.80 has center_ok=False AND quad_ok=True!
        w_center_fail = wilson_lower_meets_target_exact(6, 12, Fraction(1, 1), Fraction(4, 5))
        self.assertFalse(w_center_fail["center_ok"])
        self.assertTrue(w_center_fail["quadratic_ok"])
        self.assertFalse(w_center_fail["meets_target"])

        # Non-perfect-square radicand (h=10, n=12, z=16449/10000)
        w_nonsq = wilson_lower_meets_target_exact(10, 12, z95, Fraction(1, 2))
        self.assertTrue(w_nonsq["meets_target"])
        self.assertLessEqual(w_nonsq["lb_floor_6dp"], w_nonsq["lb_ceil_6dp"])

        with self.assertRaises(LayaVegamlError):
            wilson_lower_meets_target_exact(13, 12, z95, Fraction(1, 2))
        with self.assertRaises(LayaVegamlError):
            wilson_lower_meets_target_exact(5, 12, Fraction(0, 1), Fraction(1, 2))

        # Fail-closed union router: exercise EVERY single trigger & fail-closed type guard
        good_ans = {"confidence": Fraction(9, 10), "unbound": False, "abstain": False, "prediction_set": ["a"]}
        self.assertEqual(route_answer_fail_closed_exact(good_ans, band_threshold=Fraction(4, 5), band_reachable=True)["route"], "auto")

        # Each individual trigger in isolation
        r_unconf = route_answer_fail_closed_exact(good_ans, band_threshold=None, band_reachable=True)
        self.assertEqual(r_unconf["triggers"], ["band_unconfigured"])
        r_unreach = route_answer_fail_closed_exact(good_ans, band_threshold=Fraction(4, 5), band_reachable=False)
        self.assertEqual(r_unreach["triggers"], ["band_unreachable"])
        r_noconf = route_answer_fail_closed_exact(dict(good_ans, confidence=None), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_noconf["triggers"], ["missing_confidence"])
        r_lowconf = route_answer_fail_closed_exact(dict(good_ans, confidence=Fraction(1, 2)), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_lowconf["triggers"], ["confidence_below_band"])
        r_m_unb = route_answer_fail_closed_exact(dict(good_ans, unbound=None), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_m_unb["triggers"], ["missing_unbound"])
        self.assertEqual(r_m_unb["route"], "review")
        r_ood = route_answer_fail_closed_exact(dict(good_ans, unbound=True), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_ood["triggers"], ["unbound_ood"])
        r_m_abs = route_answer_fail_closed_exact(dict(good_ans, abstain=None), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_m_abs["triggers"], ["missing_abstain"])
        self.assertEqual(r_m_abs["route"], "review")
        r_abs = route_answer_fail_closed_exact(dict(good_ans, abstain=True), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_abs["triggers"], ["conformal_abstain"])
        r_nopset = route_answer_fail_closed_exact(dict(good_ans, prediction_set=None), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_nopset["triggers"], ["missing_prediction_set"])
        r_empty = route_answer_fail_closed_exact(dict(good_ans, prediction_set=[]), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_empty["triggers"], ["empty_prediction_set"])
        r_ambig = route_answer_fail_closed_exact(dict(good_ans, prediction_set=["a", "b"]), band_threshold=Fraction(4, 5), band_reachable=True)
        self.assertEqual(r_ambig["triggers"], ["conformal_set_ambiguous"])

        with self.assertRaises(LayaVegamlError):
            route_answer_fail_closed_exact(123, band_threshold=Fraction(1, 2), band_reachable=True)  # type: ignore[arg-type]
        for bad_pset in (123, "ab", b"ab"):
            with self.assertRaises(LayaVegamlError):
                route_answer_fail_closed_exact(dict(good_ans, prediction_set=bad_pset), band_threshold=Fraction(1, 2), band_reachable=True)

        # Default laya/vega router gap & guard coverage
        self.assertEqual(route_laya_vega_default_exact(good_ans, band_threshold=Fraction(4, 5), band_reachable=True)["mode_used"], "band_only")
        self.assertEqual(route_laya_vega_default_exact(good_ans, band_threshold=None, band_reachable=True)["mode_used"], "conformal_fallback")
        self.assertEqual(route_laya_vega_default_exact(good_ans, band_threshold=Fraction(4, 5), band_reachable=False)["mode_used"], "conformal_fallback")
        self.assertEqual(route_laya_vega_default_exact(dict(good_ans, confidence=None), band_threshold=Fraction(4, 5), band_reachable=True)["mode_used"], "conformal_fallback")
        self.assertTrue(route_laya_vega_default_exact(dict(good_ans, abstain=True), band_threshold=None, band_reachable=False)["review"])
        self.assertTrue(route_laya_vega_default_exact(dict(good_ans, prediction_set=["a", "b"]), band_threshold=None, band_reachable=False)["review"])
        self.assertFalse(route_laya_vega_default_exact(dict(good_ans, prediction_set=None, abstain=False), band_threshold=None, band_reachable=False)["review"])
        self.assertFalse(route_laya_vega_default_exact(dict(good_ans, prediction_set=["a"], abstain=None), band_threshold=None, band_reachable=False)["review"])
        self.assertTrue(route_laya_vega_default_exact(dict(good_ans, prediction_set=None, abstain=None), band_threshold=None, band_reachable=False)["review"])
        with self.assertRaises(LayaVegamlError):
            route_laya_vega_default_exact(123, band_threshold=None, band_reachable=True)  # type: ignore[arg-type]
        for bad_pset in (123, "ab", b"ab"):
            with self.assertRaises(LayaVegamlError):
                route_laya_vega_default_exact(dict(good_ans, prediction_set=bad_pset), band_threshold=None, band_reachable=True)

    def test_pillar3_epistemic_lattice_and_calibration_inspector(self) -> None:
        laws = certify_epistemic_lattice_laws_exact()
        self.assertTrue(laws["all_laws_hold"])
        self.assertEqual(laws["triples_checked"], 125)

        for bad_st in (123, ["PRESENT"], "BOGUS"):
            with self.assertRaises(LayaVegamlError):
                epistemic_meet_exact(bad_st, EVIDENCE_PRESENT)
            with self.assertRaises(LayaVegamlError):
                epistemic_meet_exact(EVIDENCE_PRESENT, bad_st)

        # Fail-closed inspect_calibration covers EVERY branch
        with self.assertRaises(LayaVegamlError):
            inspect_calibration_fail_closed_exact(123)
        self.assertEqual(inspect_calibration_fail_closed_exact({})["state"], EVIDENCE_UNKNOWN)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {}})["state"], EVIDENCE_UNKNOWN)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {}}})["state"], EVIDENCE_MISSING)
        for bad_cal_k in (1, "", "  "):
            with self.assertRaisesRegex(LayaVegamlError, "non-empty str"):
                inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {bad_cal_k: {}}}})
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": 123}}})["state"], EVIDENCE_UNKNOWN)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": {"items": True, "issues": []}}}})["state"], EVIDENCE_UNKNOWN)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": {"items": -5, "issues": []}}}})["state"], EVIDENCE_UNKNOWN)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": {"items": 0, "issues": []}}}})["state"], EVIDENCE_MISSING)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": {"items": 10, "issues": "bad"}}}})["state"], EVIDENCE_UNKNOWN)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": {"items": 10, "issues": ["clamped"]}}}})["state"], EVIDENCE_INSUFFICIENT)
        self.assertEqual(inspect_calibration_fail_closed_exact({"training": {"laya_train_calibration": {"c": {"items": 10, "issues": []}}}})["state"], EVIDENCE_PRESENT)

        # Promotion gate guards
        sha = hashlib.sha256(b"ckpt").hexdigest()
        self.assertTrue(evaluate_evidence_promotion_exact({"cal": EVIDENCE_PRESENT}, checkpoint_sha256=sha)["can_promote_to_verified"])
        self.assertFalse(evaluate_evidence_promotion_exact({"cal": EVIDENCE_PRESENT}, checkpoint_sha256=None)["can_promote_to_verified"])
        for bad_f in (123, {}):
            with self.assertRaises(LayaVegamlError):
                evaluate_evidence_promotion_exact(bad_f)  # type: ignore[arg-type]
        for bad_k in (123, " "):
            with self.assertRaises(LayaVegamlError):
                evaluate_evidence_promotion_exact({bad_k: EVIDENCE_PRESENT})  # type: ignore[dict-item]


class TestPillar4AndAuditRunner(unittest.TestCase):
    def test_pillar4_linear_solver_port_recovery_and_guardrail781(self) -> None:
        # Pivot swap + exact solution + singular fail-closed
        sol = solve_linear_system_exact(
            [[Fraction(0), Fraction(2)], [Fraction(3), Fraction(1)]],
            [[Fraction(4)], [Fraction(5)]],
        )
        self.assertEqual(sol, [[Fraction(1)], [Fraction(2)]])

        for bad_m in (123, [], "ab", b"ab"):
            with self.assertRaisesRegex(LayaVegamlError, "2D sequence"):
                solve_linear_system_exact(bad_m, [[Fraction(1)]])  # type: ignore[arg-type]
        with self.assertRaisesRegex(LayaVegamlError, "exact Fraction"):
            solve_linear_system_exact([[1]], [[Fraction(1)]])  # type: ignore[list-item]
        for bad_row in ([[]], [[Fraction(1)], "bad"], [[Fraction(1)], b"b"], [[Fraction(1)], 123]):
            with self.assertRaisesRegex(LayaVegamlError, "non-empty sequence"):
                solve_linear_system_exact(bad_row, [[Fraction(1)]])  # type: ignore[arg-type]
        with self.assertRaisesRegex(LayaVegamlError, "rectangular"):
            solve_linear_system_exact([[Fraction(1), Fraction(2)], [Fraction(1)]], [[Fraction(1)], [Fraction(2)]])
        with self.assertRaisesRegex(LayaVegamlError, "square"):
            solve_linear_system_exact([[Fraction(1), Fraction(2)]], [[Fraction(1)]])
        with self.assertRaisesRegex(LayaVegamlError, "row count"):
            solve_linear_system_exact([[Fraction(1)]], [[Fraction(1)], [Fraction(2)]])
        with self.assertRaisesRegex(LayaVegamlError, "singular"):
            solve_linear_system_exact([[Fraction(1), Fraction(1)], [Fraction(2), Fraction(2)]], [[Fraction(1)], [Fraction(2)]])

        with self.assertRaisesRegex(LayaVegamlError, "identical row count"):
            solve_ridge_port_exact([[Fraction(1)]], [[Fraction(1)], [Fraction(2)]])
        with self.assertRaisesRegex(LayaVegamlError, "non-negative"):
            solve_ridge_port_exact([[Fraction(1)], [Fraction(2)]], [[Fraction(1)], [Fraction(2)]], lam=Fraction(-1, 10))

        # Port recovery guards (shape, zero-mean E, orthogonality X_c^T E == 0)
        x = [[Fraction(1)], [Fraction(-1)]]
        w0 = [[Fraction(2)]]
        b0 = [Fraction(0)]
        a = [[Fraction(3)]]
        c = [Fraction(1)]
        e0 = [[Fraction(0)], [Fraction(0)]]
        with self.assertRaisesRegex(LayaVegamlError, "w0 row count"):
            certify_quantized_reader_port_recovery_exact(x, [[Fraction(1)], [Fraction(2)]], b0, a, c, e0)
        for bad_b0 in (123, "a", b"a", [Fraction(0), Fraction(0)]):
            with self.assertRaisesRegex(LayaVegamlError, "b0 length"):
                certify_quantized_reader_port_recovery_exact(x, w0, bad_b0, a, c, e0)  # type: ignore[arg-type]
        for bad_a in ([[Fraction(1), Fraction(2)]], [[Fraction(1)], [Fraction(2)]]):
            with self.assertRaisesRegex(LayaVegamlError, "a_drift"):
                certify_quantized_reader_port_recovery_exact(x, w0, b0, bad_a, c, e0)
        for bad_c in (123, "a", b"a", [Fraction(1), Fraction(2)]):
            with self.assertRaisesRegex(LayaVegamlError, "c_shift length"):
                certify_quantized_reader_port_recovery_exact(x, w0, b0, a, bad_c, e0)  # type: ignore[arg-type]
        for bad_e in ([[Fraction(0)]], [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]):
            with self.assertRaisesRegex(LayaVegamlError, "e_orth shape"):
                certify_quantized_reader_port_recovery_exact(x, w0, b0, a, c, bad_e)
        with self.assertRaisesRegex(LayaVegamlError, "zero-mean"):
            certify_quantized_reader_port_recovery_exact(x, w0, b0, a, c, [[Fraction(1)], [Fraction(1)]])
        with self.assertRaisesRegex(LayaVegamlError, "orthogonal"):
            certify_quantized_reader_port_recovery_exact(x, w0, b0, a, c, [[Fraction(1)], [Fraction(-1)]])

        # Guardrail #781 theorem & fail-closed guards
        high_harm = certify_guardrail_781_inversion_and_cap_exact(
            [{"id": "hi", "prompt_injection": Fraction(1, 10), "harm_severity": Fraction(2, 1)}],
            benign_inversion_score=Fraction(1, 2),
        )
        self.assertFalse(high_harm["all_embedded_harm_capped_below_threshold"])
        self.assertFalse(high_harm["cases"][0]["cannot_reach_default_threshold"])

        high_pi = certify_guardrail_781_inversion_and_cap_exact(
            [{"id": "hi_pi", "prompt_injection": Fraction(6, 10), "harm_severity": Fraction(1, 2)}],
            benign_inversion_score=Fraction(1, 2),
        )
        self.assertFalse(high_pi["cases"][0]["cannot_reach_default_threshold"])
        self.assertFalse(high_pi["scalar_threshold_impossible"])

        for bad_cases in (123, [], "ab", b"ab"):
            with self.assertRaisesRegex(LayaVegamlError, "non-empty sequence"):
                certify_guardrail_781_inversion_and_cap_exact(bad_cases, benign_inversion_score=Fraction(1, 2))  # type: ignore[arg-type]
        for bad_entry in (
            123,
            {"prompt_injection": Fraction(0), "harm_severity": Fraction(0)},
            {"id": "c1", "harm_severity": Fraction(0)},
            {"id": "c1", "prompt_injection": Fraction(0)},
        ):
            with self.assertRaises(LayaVegamlError):
                certify_guardrail_781_inversion_and_cap_exact([bad_entry], benign_inversion_score=Fraction(1, 2))  # type: ignore[list-item]
        for bad_id in (123, " "):
            with self.assertRaises(LayaVegamlError):
                certify_guardrail_781_inversion_and_cap_exact(
                    [{"id": bad_id, "prompt_injection": Fraction(0), "harm_severity": Fraction(0)}],
                    benign_inversion_score=Fraction(1, 2),
                )
        for bad_hs in (Fraction(-1, 10), Fraction(4, 1)):
            with self.assertRaises(LayaVegamlError):
                certify_guardrail_781_inversion_and_cap_exact(
                    [{"id": "c1", "prompt_injection": Fraction(0), "harm_severity": bad_hs}],
                    benign_inversion_score=Fraction(1, 2),
                )

    def test_certificate_validator_anti_bypass_live_mock_and_cli(self) -> None:
        sa = run_standalone_audit()
        self.assertTrue(sa["overall_verified"])
        self.assertTrue(validate_audit_certificate(sa)["valid"])

        def _rehash(d: dict) -> dict:
            out = dict(d)
            out.pop("certificate_sha256", None)
            out["certificate_sha256"] = canonical_sha256(out)
            return out

        with self.assertRaisesRegex(LayaVegamlError, "mapping"):
            validate_audit_certificate(123)  # type: ignore[arg-type]
        flt = json.loads(json.dumps(sa))
        flt["pillar1_split_conformal"]["exact_lac_n19_a010"]["n"] = json.loads("19.5")
        with self.assertRaisesRegex(LayaVegamlError, "forbidden float"):
            validate_audit_certificate(flt)
        with self.assertRaisesRegex(LayaVegamlError, "mode"):
            validate_audit_certificate({"schema_version": "x"})
        with self.assertRaisesRegex(LayaVegamlError, "mode"):
            validate_audit_certificate(_rehash(dict(sa, mode="bad")))
        with self.assertRaisesRegex(LayaVegamlError, "key set mismatch"):
            validate_audit_certificate(dict(sa, forged="N2"))
        with self.assertRaisesRegex(LayaVegamlError, "schema_version"):
            validate_audit_certificate(_rehash(dict(sa, schema_version="wrong")))
        with self.assertRaisesRegex(LayaVegamlError, "audit_id"):
            validate_audit_certificate(_rehash(dict(sa, audit_id="AUDIT-999")))
        with self.assertRaisesRegex(LayaVegamlError, "novelty_class"):
            validate_audit_certificate(_rehash(dict(sa, novelty_class="N2")))
        for bad_live in (True, 0):
            with self.assertRaisesRegex(LayaVegamlError, "live_intake_audited"):
                validate_audit_certificate(_rehash(dict(sa, live_intake_audited=bad_live)))
        for bad_ver in (False, 1):
            with self.assertRaisesRegex(LayaVegamlError, "overall_verified"):
                validate_audit_certificate(_rehash(dict(sa, overall_verified=bad_ver)))
        with self.assertRaisesRegex(LayaVegamlError, "digest mismatch"):
            validate_audit_certificate(dict(sa, certificate_sha256="01" * 32))

        # Hollow 1-key shells, 1==True smuggling, and list length mismatches on all 4 pillars
        for pkey in (
            "pillar1_split_conformal",
            "pillar2_escalation_band_and_router",
            "pillar3_epistemic_evidence_lattice",
            "pillar4_siliquant_port_and_guardrail781",
        ):
            bad_int_bool = dict(sa[pkey], pillar_verified=1)
            for bad_sub in (123, {"pillar_verified": True}, bad_int_bool):
                with self.assertRaisesRegex(LayaVegamlError, pkey):
                    validate_audit_certificate(_rehash(dict(sa, **{pkey: bad_sub})))
        p4_short_cases = json.loads(json.dumps(sa["pillar4_siliquant_port_and_guardrail781"]))
        p4_short_cases["guardrail_781_inversion_and_cap"]["cases"] = []
        with self.assertRaisesRegex(LayaVegamlError, "pillar4"):
            validate_audit_certificate(_rehash(dict(sa, pillar4_siliquant_port_and_guardrail781=p4_short_cases)))

        # Live synthetic intake (both loose-ref & packed-ref/detached-HEAD) + CLI
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "intake"
            _build_synthetic_intake(root, packed_ref=False)
            live_cert = run_live_audit(root)
            self.assertTrue(live_cert["overall_verified"])
            self.assertTrue(validate_audit_certificate(live_cert, intake_dir=root)["valid"])

            # Tampered synthetic live cert rejected against intake_dir
            forged_syn_heads = dict(live_cert["pinned_repo_heads"], laya="deadbeef" * 5)
            with self.assertRaisesRegex(LayaVegamlError, "provenance mismatch against intake_dir"):
                validate_audit_certificate(_rehash(dict(live_cert, pinned_repo_heads=forged_syn_heads)), intake_dir=root)

            # Default offline pinned provenance verification (intake_dir=None; closes LAYA107-BUG-02!)
            pinned_live = _rehash(
                dict(
                    live_cert,
                    pinned_repo_heads=json.loads(json.dumps(PINNED_REPO_HEADS_EXPECTED)),
                    pinned_artifacts=json.loads(json.dumps(PINNED_ARTIFACTS_EXPECTED)),
                    unmerged_pr_gems=json.loads(json.dumps(PINNED_UNMERGED_PRS_EXPECTED)),
                )
            )
            self.assertTrue(validate_audit_certificate(pinned_live)["valid"])
            with self.assertRaisesRegex(LayaVegamlError, "pinned_repo_heads provenance mismatch"):
                validate_audit_certificate(
                    _rehash(dict(pinned_live, pinned_repo_heads=dict(PINNED_REPO_HEADS_EXPECTED, laya="deadbeef" * 5)))
                )
            forged_pa = json.loads(json.dumps(PINNED_ARTIFACTS_EXPECTED))
            forged_pa["laya/laya/evidence.py"] = {"sha256": "ab" * 32, "bytes": 1}
            with self.assertRaisesRegex(LayaVegamlError, "pinned_artifacts provenance mismatch"):
                validate_audit_certificate(_rehash(dict(pinned_live, pinned_artifacts=forged_pa)))
            forged_prs_sha = json.loads(json.dumps(PINNED_UNMERGED_PRS_EXPECTED))
            forged_prs_sha["prs"][0]["head_sha"] = "cd" * 20
            with self.assertRaisesRegex(LayaVegamlError, "unmerged_pr_gems provenance mismatch"):
                validate_audit_certificate(_rehash(dict(pinned_live, unmerged_pr_gems=forged_prs_sha)))

            # Live certificate validator anti-hollow & strict key-set probes
            for bad_h in (123, {"laya": "ab" * 20}):
                with self.assertRaisesRegex(LayaVegamlError, "pinned_repo_heads incomplete"):
                    validate_audit_certificate(_rehash(dict(live_cert, pinned_repo_heads=bad_h)), intake_dir=root)
            for bad_pa in (123, {}):
                with self.assertRaisesRegex(LayaVegamlError, "pinned_artifacts incomplete"):
                    validate_audit_certificate(_rehash(dict(live_cert, pinned_artifacts=bad_pa)), intake_dir=root)
            for bad_meta in (123, dict(live_cert["pinned_artifacts"][PINNED_INTAKE_FILES[0]], extra=1)):
                pa_bad_entry = dict(live_cert["pinned_artifacts"], **{PINNED_INTAKE_FILES[0]: bad_meta})
                with self.assertRaisesRegex(LayaVegamlError, "invalid pinned_artifacts entry"):
                    validate_audit_certificate(_rehash(dict(live_cert, pinned_artifacts=pa_bad_entry)), intake_dir=root)
            for bad_ast in (
                123,
                dict(live_cert["live_source_ast_audit"], calibrate_insample_coupled=False),
                dict(live_cert["live_source_ast_audit"], extra_key=1),
            ):
                with self.assertRaisesRegex(LayaVegamlError, "live_source_ast_audit"):
                    validate_audit_certificate(_rehash(dict(live_cert, live_source_ast_audit=bad_ast)), intake_dir=root)
            for bad_prs, rx in (
                (123, "key set mismatch"),
                (dict(live_cert["unmerged_pr_gems"], extra=1), "key set mismatch"),
                ({"pr_count": 7, "prs": live_cert["unmerged_pr_gems"]["prs"]}, "pr_count mismatch"),
            ):
                with self.assertRaisesRegex(LayaVegamlError, rx):
                    validate_audit_certificate(_rehash(dict(live_cert, unmerged_pr_gems=bad_prs)), intake_dir=root)
            dup_pr_list = list(live_cert["unmerged_pr_gems"]["prs"]) + [live_cert["unmerged_pr_gems"]["prs"][0]]
            for bad_pr_list in ("abcdefgh", [], dup_pr_list):
                with self.assertRaisesRegex(LayaVegamlError, "list incomplete"):
                    validate_audit_certificate(_rehash(dict(live_cert, unmerged_pr_gems={"pr_count": 8, "prs": bad_pr_list})), intake_dir=root)
            def _mut_pr0(fn) -> dict:
                c = json.loads(json.dumps(live_cert))
                fn(c["unmerged_pr_gems"]["prs"])
                return _rehash(c)
            for fn, rx in (
                (lambda L: L.__setitem__(0, 123), "key set mismatch"),
                (lambda L: L.__setitem__(0, dict(L[0], extra=1)), "key set mismatch"),
                (lambda L: L[0].__setitem__("merged", True), "merged=False"),
                (lambda L: L[0].__setitem__("repo", 123), "field type or value invalid"),
                (lambda L: L[0].__setitem__("repo", "  "), "field type or value invalid"),
                (lambda L: L[0].__setitem__("number", True), "field type or value invalid"),
                (lambda L: L[0].__setitem__("number", 0), "field type or value invalid"),
                (lambda L: L[0].__setitem__("state", 123), "field type or value invalid"),
                (lambda L: L[0].__setitem__("state", ["open"]), "field type or value invalid"),
                (lambda L: L[0].__setitem__("state", "MERGED_FORGED"), "field type or value invalid"),
                (lambda L: L[0].__setitem__("title", 123), "field type or value invalid"),
                (lambda L: L[0].__setitem__("title", "  "), "field type or value invalid"),
                (lambda L: L[0].__setitem__("number", 9999), "PR set mismatch"),
            ):
                with self.assertRaisesRegex(LayaVegamlError, rx):
                    validate_audit_certificate(_mut_pr0(fn), intake_dir=root)

            with self.assertRaisesRegex(LayaVegamlError, "missing unmerged_prs_snapshot"):
                _audit_unmerged_pr_snapshot(Path(td) / "no_snap_dir")

            # Packed-ref & detached-HEAD git reader paths + CLI --live + --standalone + --write-cert
            root_packed = Path(td) / "intake_packed"
            _build_synthetic_intake(root_packed, packed_ref=True)
            self.assertEqual(len(read_git_head_readonly(root_packed / "laya")), 40)
            self.assertEqual(len(read_git_head_readonly(root_packed / "vegaml")), 40)

            out_cert = Path(td) / "cert.json"
            buf = io.StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = buf
                rc_sa = audit_main(["--standalone", "--json", "--write-cert", str(out_cert)])
            finally:
                sys.stdout = old_stdout
            self.assertEqual(rc_sa, 0)
            self.assertEqual(buf.getvalue(), out_cert.read_text(encoding="utf-8"))

            buf_txt = io.StringIO()
            try:
                sys.stdout = buf_txt
                rc_live = audit_main(["--live", "--intake-dir", str(root)])
            finally:
                sys.stdout = old_stdout
            self.assertEqual(rc_live, 0)
            self.assertIn("AUDIT-107", buf_txt.getvalue())

            # Fail-closed live intake error guards
            with self.assertRaisesRegex(LayaVegamlError, "does not exist"):
                run_live_audit(Path(td) / "missing")
            with self.assertRaisesRegex(LayaVegamlError, "missing pinned artifact"):
                sha256_file(Path(td) / "no_file")
            (root_packed / "laya" / ".git" / "packed-refs").write_text("# empty\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "unresolvable git ref"):
                read_git_head_readonly(root_packed / "laya")
            (root_packed / "laya" / ".git" / "HEAD").unlink()
            with self.assertRaisesRegex(LayaVegamlError, "missing .git/HEAD"):
                read_git_head_readonly(root_packed / "laya")

            # Live AST & PR snapshot failure guards
            bad_root = Path(td) / "bad_intake"
            _build_synthetic_intake(bad_root)
            (bad_root / "vegaml" / "vegaml" / "calibrate.py").write_text("def _conformal_qhat(s, a): return 1.0 - a\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing calibrate"):
                run_live_audit(bad_root)
            (bad_root / "vegaml" / "vegaml" / "calibrate.py").write_text("def calibrate(r, a): return 1\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing calibrate"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            (bad_root / "vegaml" / "vegaml" / "band.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing Band class"):
                run_live_audit(bad_root)
            (bad_root / "vegaml" / "vegaml" / "band.py").write_text("class Band:\n    pass\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing Band.route"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            (bad_root / "laya" / "laya" / "vega.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing VegaAgent class"):
                run_live_audit(bad_root)
            (bad_root / "laya" / "laya" / "vega.py").write_text("class VegaAgent:\n    pass\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing VegaAgent._route"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            (bad_root / "laya" / "laya" / "evidence.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "missing inspect_calibration"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            (bad_root / "laya" / "docs" / "guardrail-limitations.md").write_text("empty\n", encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "failed to parse expected c011"):
                run_live_audit(bad_root)
            (bad_root / "laya" / "docs" / "guardrail-limitations.md").write_text(
                "| c011 | a | 0.0 | 0.1 | 0.8 |\n| c012 | a | 0.0 | 0.2 | 0.7 |\n| c042 | a | 0.0 | 0.0 | 0.5 |\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LayaVegamlError, "failed to parse c038"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            for bad_snap in ("[]", '{"prs": 123}'):
                (bad_root / "unmerged_prs_snapshot.json").write_text(bad_snap, encoding="utf-8")
                with self.assertRaisesRegex(LayaVegamlError, "malformed"):
                    run_live_audit(bad_root)
            (bad_root / "unmerged_prs_snapshot.json").write_text('{"prs": [123]}', encoding="utf-8")
            with self.assertRaisesRegex(LayaVegamlError, "must be a mapping"):
                run_live_audit(bad_root)
            valid_pr = {"repo": "x", "number": 1, "state": "open", "merged": False, "head_sha": "ab" * 20, "title": "t"}
            for bad_field, bad_val in (
                ("repo", 123),
                ("repo", "  "),
                ("number", True),
                ("number", 0),
                ("title", 123),
                ("title", "  "),
                ("state", 123),
                ("state", ["open"]),
                ("state", "MERGED_FORGED"),
                ("merged", 1),
            ):
                bad_pr = dict(valid_pr, **{bad_field: bad_val})
                (bad_root / "unmerged_prs_snapshot.json").write_text(json.dumps({"prs": [bad_pr]}), encoding="utf-8")
                with self.assertRaisesRegex(LayaVegamlError, "missing or malformed"):
                    run_live_audit(bad_root)
            (bad_root / "unmerged_prs_snapshot.json").write_text(
                json.dumps({"prs": [dict(valid_pr, merged=True)]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LayaVegamlError, "expected unmerged PR"):
                run_live_audit(bad_root)
            good_prs = [
                {"repo": r, "number": n, "state": "open", "merged": False, "head_sha": f"{n:04x}" * 10, "title": "t"}
                for r, n in EXPECTED_UNMERGED_PRS
            ]
            wrong_same_len = list(good_prs)
            wrong_same_len[0] = dict(wrong_same_len[0], number=9999)
            dup_extra_len = list(good_prs) + [good_prs[0]]
            for bad_pr_set in (wrong_same_len, dup_extra_len):
                (bad_root / "unmerged_prs_snapshot.json").write_text(json.dumps({"prs": bad_pr_set}), encoding="utf-8")
                with self.assertRaisesRegex(LayaVegamlError, "PR set mismatch"):
                    run_live_audit(bad_root)
            (bad_root / "unmerged_prs_snapshot.json").unlink()
            with self.assertRaisesRegex(LayaVegamlError, "missing pinned artifact"):
                run_live_audit(bad_root)


if __name__ == "__main__":
    unittest.main()
