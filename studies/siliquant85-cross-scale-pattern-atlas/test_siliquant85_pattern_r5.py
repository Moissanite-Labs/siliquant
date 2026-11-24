"""Exhaustive Aletheia R5 test suite for siliquant PR #85 Cross-Scale Pattern Atlas.

Covers pure-Z/Q oracle theorems, FR1/FR0/direct-secondary tamper rejection,
float/subclass/surrogate/depth/int-bits/duplicate-key rejection, atomic dir_fd
I/O (O_NOFOLLOW, FIFO via scratch mkfifo, hardlink st_nlink==2, EEXIST retry,
chmod 500 OSError wrapping, Darwin prefix normalization, $PWD dev/ino parity),
CLI fail-closed edges, and mock read-only portal cross-checking.

Zero bare asserts, zero floats, zero .isdigit(), zero RNG, zero unsafe paths.
"""
from __future__ import annotations

import copy
import errno
from fractions import Fraction
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import siliquant85_pattern_audit as audit
import siliquant85_pattern_oracle as oracle


class _IntSub(int):
    pass


class _StrSub(str):
    pass


class _DictSub(dict):
    pass


class _ListSub(list):
    pass


def _resign(payload: dict[str, object]) -> dict[str, object]:
    p_sha = audit.sha256_hex(audit.canonical_json_bytes(payload))
    env: dict[str, object] = {
        "schema_version": oracle.SCHEMA_VERSION,
        "payload_sha256": p_sha,
        "payload": payload,
    }
    env["certificate_sha256"] = audit.sha256_hex(audit.canonical_json_bytes(env))
    return env


def _fmt_micro(micro: int) -> str:
    whole, frac = divmod(micro, 1000000)
    return f"{whole}.{frac:06d}"


def _make_mock_portal(base_dir: str) -> None:
    """Create a synthetic portal matching PINNED_PORTAL_SHA256 structure for unit testing."""
    for rel_path in oracle.PINNED_PORTAL_SHA256:
        full = os.path.join(base_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(b"mock\n")


class TestSiliquant85PatternR5(unittest.TestCase):
    def _expect_audit_error(self, code: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except audit.AuditError as exc:
            self.assertEqual(exc.code, code)
            return
        self.fail(f"expected AuditError({code}) but call succeeded")

    def test_01_oracle_exact_and_sampled_invariants(self) -> None:
        cert = audit.build_certificate()
        verified_fr1 = audit.verify_certificate_object(cert, full_recompute=True)
        verified_fr0 = audit.verify_certificate_object(cert, full_recompute=False)
        self.assertEqual(verified_fr1["certificate_sha256"], verified_fr0["certificate_sha256"])

        p = cert["payload"]
        self.assertEqual(p["exact"]["provenance_tier"], "exact")
        self.assertEqual(p["sampled"]["provenance_tier"], "sampled")
        self.assertEqual(p["anchor"]["provenance_tier"], "anchor")
        self.assertFalse(p["sampled"]["measured_live_llm_in_this_track"])
        self.assertFalse(p["sampled"]["measured_live_gpu_in_this_track"])

        # Exact Hamming rates & DP table reconciliation
        hf = p["exact"]["hamming_family"]
        self.assertEqual([r["code_rate_q"] for r in hf], [[1, 2], [11, 16], [13, 16]])
        self.assertEqual([r["covering_sq_128"] for r in hf], [56, 44, 38])
        dp = p["exact"]["dp_table_reconciliation"]
        self.assertEqual(dp["a16_b1207_two_level"], 660776)
        self.assertEqual(dp["a16_b1207_flat"], 1275648)
        self.assertEqual(dp["a32_b1015_two_level"], 2147824)
        self.assertEqual(dp["a32_b1015_flat"], 2145792)
        self.assertEqual(dp["a8_b1567_flat"], 827904)
        self.assertEqual(dp["z32_b783_flat"], 25872)

        # Signed gap edge: equal KL returns sign 0, both-zero SE raises ValueError
        zero_gap = oracle.signed_gap_sq(100, 10, 100, 10)
        self.assertEqual(zero_gap, {"sign": 0, "gap_sq_q": [0, 1]})
        with self.assertRaises(ValueError):
            oracle.signed_gap_sq(100, 0, 100, 0)

        # exact_ranks guards: <2 or duplicate values raise ValueError
        with self.assertRaises(ValueError):
            oracle.exact_ranks([Fraction(1, 1)])
        with self.assertRaises(ValueError):
            oracle.exact_ranks([Fraction(1, 1), Fraction(1, 1)])

        # Oracle BoolOp helper coverage (every conjunct flipped independently)
        # Ranks [1,2,3,4] vs [2,1,4,3]: rho=3/5, tau=1/3; vs [1,3,2,4]: rho=4/5, tau=2/3
        # Ranks [1,2,3,4] vs [1,4,2,3]: rho=3/5 (same rho!), tau=1/3 vs [2,1,4,3] (same!)
        # For n=5: [1,2,3,4,5] vs [2,1,5,3,4] has d2=1+1+4+1+1=8 -> rho=3/5, disc=1+2=3 -> tau=2/5;
        # vs [1,4,2,3,5] has d2=0+4+1+1+0=6 -> rho=7/10, disc=2 -> tau=3/5;
        # vs [3,1,2,4,5] has d2=4+1+1=6 -> rho=7/10 (same rho!), disc=2 -> tau=3/5.
        # Same Spearman but different Kendall on n=5:
        # [1,2,3,4,5] vs [2,3,1,4,5]: d2=1+1+4=6 (rho=7/10), disc=2 (tau=3/5)
        # [1,2,3,4,5] vs [2,1,4,3,5]: d2=1+1+1+1=4 (rho=4/5), disc=2 (tau=3/5) -> SAME tau, DIFFERENT rho!
        self.assertTrue(oracle.check_stage_a_pair_equal([1, 2, 3, 4, 5], [2, 3, 1, 4, 5], [1, 2, 3, 4, 5], [2, 3, 1, 4, 5]))
        self.assertFalse(oracle.check_stage_a_pair_equal([1, 2, 3, 4, 5], [2, 3, 1, 4, 5], [1, 2, 3, 4, 5], [2, 1, 4, 3, 5]))
        # Same Spearman (d2=12 -> rho=2/5) but different Kendall (disc=4 vs disc=3):
        # [1,2,3,4,5] vs [3,4,1,2,5]: d2=4+4+4+4=16, disc=4
        # [1,2,3,4,5] vs [4,1,2,3,5]: d2=9+1+1+1=12 (rho=2/5), disc=3 (tau=2/5)
        # [1,2,3,4,5] vs [2,4,3,1,5]: d2=1+4+0+9=14
        # [1,2,3,4,5] vs [3,2,4,1,5]: d2=4+0+1+9=14
        # [1,2,3,4,5] vs [1,4,3,2,5]: d2=0+4+0+4=8 (rho=3/5), disc=3 (tau=2/5)
        # [1,2,3,4,5] vs [3,1,4,2,5]: d2=4+1+1+4=10 (rho=1/2), disc=3 (tau=2/5)
        # Same Spearman (d2=8 -> rho=3/5): [1,4,3,2,5] (disc=3) vs [2,1,5,3,4] (disc=3)
        # On n=6: [1,2,3,4,5,6] vs [1,4,3,2,5,6] d2=8, disc=3 vs [2,1,4,3,6,5] d2=6, disc=3
        # On n=6: d2=10: [2,1,3,6,4,5] d2=1+1+0+4+1+1=8, disc=3 vs [1,3,2,5,4,6] d2=4, disc=2
        # d2=10 with different disc: [1,2,5,4,3,6] (d2=8, disc=3) vs [2,1,4,3,5,6] (d2=4, disc=2)
        # [1,2,3,4,5,6] vs [2,3,1,5,4,6] (d2=8, disc=3) vs [3,1,2,4,6,5] (d2=8, disc=3)
        # [3,2,1,4,5,6] d2=8 (disc=3) vs [2,4,1,3,5,6] d2=1+4+4+1=10 (disc=3)
        # [4,1,2,3,5,6] d2=12 (disc=3) vs [2,1,4,5,3,6] d2=1+1+1+1+4=8 (disc=3)
        # [1,2,3,4,5,6] vs [3,4,2,1,5,6] d2=4+4+1+9=18 (disc=5) vs [4,2,3,1,5,6] d2=9+0+0+9=18 (disc=5)
        # [1,2,3,4,5,6] vs [2,3,5,1,4,6] d2=1+1+4+9+1=16 (disc=4) vs [3,4,1,2,5,6] d2=16 (disc=4)
        # [1,2,3,4,5,6] vs [4,2,1,3,5,6] d2=9+0+4+1=14 (disc=4) vs [2,5,1,3,4,6] d2=1+9+4+1+1=16
        # [1,2,3,4,5,6] vs [3,1,5,2,4,6] d2=4+1+4+4+1=14 (disc=4)
        # [1,2,3,4,5,6] vs [4,1,3,2,5,6] d2=9+1+0+4=14 (disc=4)
        # [1,2,3,4,5,6] vs [5,1,2,3,4,6] d2=16+1+1+1+1=20 (disc=4) vs [3,4,1,2,6,5] d2=18 (disc=5)
        # [1,2,3,4,5,6] vs [2,5,3,1,4,6] d2=1+9+0+9+1=20 (disc=5)!! SAME rho (d2=20), DIFFERENT tau (disc=4 vs disc=5)!
        self.assertFalse(
            oracle.check_stage_a_pair_equal(
                [1, 2, 3, 4, 5, 6], [5, 1, 2, 3, 4, 6],
                [1, 2, 3, 4, 5, 6], [2, 5, 3, 1, 4, 6],
            )
        )

        # is_llama_a32_anomaly conjunct tests
        base_db = {"A32BG": -25627, "HBG": -25585}
        base_e05 = {"A32BG": 15, "HBG": 348}
        base_kl = {"A32BG": 39540, "HBG": 36038}
        self.assertTrue(oracle.is_llama_a32_anomaly(base_db, base_e05, base_kl))
        self.assertFalse(oracle.is_llama_a32_anomaly({"A32BG": -25500, "HBG": -25585}, base_e05, base_kl))
        self.assertFalse(oracle.is_llama_a32_anomaly(base_db, {"A32BG": 400, "HBG": 348}, base_kl))
        self.assertFalse(oracle.is_llama_a32_anomaly(base_db, base_e05, {"A32BG": 35000, "HBG": 36038}))

        # is_real_ffn_gaussian conjunct tests
        base_pb = {"Laplace": 598, "Gaussian": -440, "SmolLM2_blk1_ffn_down": -372, "SmolLM2_blk5_ffn_down": -400}
        self.assertTrue(oracle.is_real_ffn_gaussian(base_pb))
        for k, bad_v in (("Laplace", -1), ("Gaussian", 1), ("SmolLM2_blk1_ffn_down", 1), ("SmolLM2_blk5_ffn_down", 1)):
            mut_pb = dict(base_pb)
            mut_pb[k] = bad_v
            self.assertFalse(oracle.is_real_ffn_gaussian(mut_pb))

        # Pattern 1-5 core numeric witnesses
        p1 = p["sampled"]["pattern_1_regime_triangle"]
        self.assertEqual(p1["models"]["SmolLM2-1.7B"]["z_minus_e_micro"], -4899)
        self.assertEqual(p1["models"]["Llama-3.2-1B"]["gap_a32_vs_hbg"]["sign"], -1)
        self.assertEqual(p1["models"]["SmolLM2-1.7B"]["gap_a32_vs_hbg"]["sign"], 1)
        p2 = p["sampled"]["pattern_2_bitrate_amplification"]
        self.assertEqual(p2["models"]["SmolLM2-1.7B"]["kl30_over_kl45_q"]["HBG"], [314951, 25761])
        p3 = p["sampled"]["pattern_3_colproxy_sign_inversion"]
        self.assertEqual(p3["noise_positive_cells"], 21)
        self.assertEqual(p3["noise_beats_sig_cells"], 20)
        self.assertEqual(p3["gam_attn_positive_count"], 9)
        self.assertEqual(p3["gam_attn_total_count"], 9)
        self.assertEqual(p3["gam_ffn_negative_count"], 6)
        self.assertEqual(p3["gam_ffn_total_count"], 6)
        self.assertEqual(p3["per_model"]["Qwen2.5-1.5B"]["sig_mean_q"], [-2123, 7000])
        self.assertEqual(p3["per_model"]["Llama-3.2-1B"]["sig_mean_q"], [1049, 3500])
        p4 = p["sampled"]["pattern_4_sphere_and_tax_decomposition"]
        self.assertEqual(p4["pball_p2_minus_p1_milli_db"]["Laplace"], 598)
        self.assertEqual(p4["pball_p2_minus_p1_milli_db"]["SmolLM2_blk1_ffn_down"], -372)
        self.assertEqual(p4["blk10_p15_bump_milli_db"], 45)
        p5 = p["sampled"]["pattern_5_stage_a_rank_matrix"]
        self.assertEqual(
            p5["rank_concordance"]["five_arm"]["pairwise"]["Qwen2.5-1.5B_vs_SmolLM2-1.7B"]["spearman_q"],
            [1, 2],
        )
        self.assertEqual(
            p5["rank_concordance"]["seven_arm"]["pairwise"]["Qwen2.5-1.5B_vs_Llama-3.2-1B"]["spearman_q"],
            [13, 14],
        )

    def test_02_require_ok_and_json_tree_guards(self) -> None:
        self._expect_audit_error("BOOL_GUARD", audit.require_ok, 1, "BOOL_GUARD", "int is not bool")
        self._expect_audit_error("BOOL_GUARD", audit.require_ok, False, "BOOL_GUARD", "false rejected")
        self._expect_audit_error("JSON_FORBIDDEN_TYPE", audit._validate_json_tree, {"x": None})
        self._expect_audit_error("JSON_FORBIDDEN_TYPE", audit._validate_json_tree, {"x": _IntSub(1)})
        self._expect_audit_error("JSON_FORBIDDEN_TYPE", audit._validate_json_tree, {"x": _StrSub("a")})
        self._expect_audit_error("JSON_FORBIDDEN_TYPE", audit._validate_json_tree, _DictSub({"a": 1}))
        self._expect_audit_error("JSON_FORBIDDEN_TYPE", audit._validate_json_tree, [_ListSub([1])])
        self._expect_audit_error("JSON_KEY_TYPE", audit._validate_json_tree, {1: "a"})
        self._expect_audit_error("JSON_BAD_UNICODE", audit._validate_json_tree, {"\ud800": "a"})
        self._expect_audit_error("JSON_BAD_UNICODE", audit._validate_json_tree, {"a": "\udfff"})
        self._expect_audit_error("JSON_INT_OVERFLOW", audit._validate_json_tree, {"x": 1 << 2049})

        deep: object = 1
        for _ in range(audit.MAX_JSON_DEPTH + 1):
            deep = [deep]
        self._expect_audit_error("JSON_DEPTH_EXCEEDED", audit._validate_json_tree, deep)

        # Boundary: exact MAX_JSON_DEPTH and exact MAX_INT_BITS succeed
        ok_deep: object = 1
        for _ in range(audit.MAX_JSON_DEPTH):
            ok_deep = [ok_deep]
        audit._validate_json_tree(ok_deep)
        audit._validate_json_tree({"x": (1 << audit.MAX_INT_BITS) - 1})

    def test_03_certificate_envelope_and_hex_guards(self) -> None:
        cert = audit.build_certificate()
        bad_schema = copy.deepcopy(cert)
        bad_schema["extra"] = 1
        self._expect_audit_error("CERT_SCHEMA", audit.verify_certificate_object, bad_schema, full_recompute=False)

        bad_ver = copy.deepcopy(cert)
        bad_ver["schema_version"] = "wrong"
        self._expect_audit_error("SCHEMA_MISMATCH", audit.verify_certificate_object, bad_ver, full_recompute=False)

        bad_psha = copy.deepcopy(cert)
        bad_psha["payload_sha256"] = "Z" * 64
        self._expect_audit_error("BAD_PAYLOAD_SHA", audit.verify_certificate_object, bad_psha, full_recompute=False)

        bad_psha_len = copy.deepcopy(cert)
        bad_psha_len["payload_sha256"] = "a" * 63
        self._expect_audit_error("BAD_PAYLOAD_SHA", audit.verify_certificate_object, bad_psha_len, full_recompute=False)

        bad_csha = copy.deepcopy(cert)
        bad_csha["certificate_sha256"] = "A" * 64
        self._expect_audit_error("BAD_CERT_SHA", audit.verify_certificate_object, bad_csha, full_recompute=False)

        bad_pnotdict = copy.deepcopy(cert)
        bad_pnotdict["payload"] = [1, 2]
        self._expect_audit_error("PAYLOAD_NOT_DICT", audit.verify_certificate_object, bad_pnotdict, full_recompute=False)

        mismatch_psha = copy.deepcopy(cert)
        mismatch_psha["payload_sha256"] = "0" * 64
        self._expect_audit_error("PAYLOAD_SHA_MISMATCH", audit.verify_certificate_object, mismatch_psha, full_recompute=False)

        mismatch_csha = copy.deepcopy(cert)
        mismatch_csha["certificate_sha256"] = "0" * 64
        self._expect_audit_error("CERT_SHA_MISMATCH", audit.verify_certificate_object, mismatch_csha, full_recompute=False)

        # Direct helper checks for _is_lower_hex64, _frac_from_pair, _require_dict, strict bool full_recompute
        self.assertFalse(audit._is_lower_hex64(123))
        self._expect_audit_error("PAIR_ERR", audit._frac_from_pair, (1, 2), "PAIR_ERR")
        self._expect_audit_error("PAIR_ERR", audit._frac_from_pair, [1, True], "PAIR_ERR")
        self._expect_audit_error("DICT_ERR", audit._require_dict, [1], "DICT_ERR")
        for bad_bool in (0, 1, "false", None):
            self._expect_audit_error(
                "INVALID_BOOL",
                audit.verify_certificate_object,
                cert,
                full_recompute=bad_bool,
            )

        # Verify FR1 full_recompute check triggers if build_certificate disagrees
        alt = copy.deepcopy(cert)
        alt["certificate_sha256"] = "1" * 64
        with mock.patch.object(audit, "build_certificate", return_value=alt):
            self._expect_audit_error("FULL_RECOMPUTE_MISMATCH", audit.verify_certificate_object, cert, full_recompute=True)

    def test_04_secondary_audit_all_semantic_branches_and_resign_fr0(self) -> None:
        base = oracle.build_payload()

        def mutate_and_expect(code: str, mutator) -> None:
            p = copy.deepcopy(base)
            mutator(p)
            # Test BOTH direct secondary_audit_payload AND re-signed FR0!
            self._expect_audit_error(code, audit.secondary_audit_payload, p)
            self._expect_audit_error(code, audit.verify_certificate_object, _resign(p), full_recompute=False)

        # Top-level schema & exact pillar branches
        mutate_and_expect("PAYLOAD_SCHEMA", lambda p: p.update({"extra": 1}))
        mutate_and_expect("EXACT_SCHEMA", lambda p: p["exact"].update({"extra": 1}))
        mutate_and_expect("EXACT_TIER", lambda p: p["exact"].update({"provenance_tier": "sampled"}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"].update({"rate_monotone_increasing": False}))
        mutate_and_expect("EXACT_HAMMING_TYPE", lambda p: p["exact"].update({"hamming_family": "bad"}))
        mutate_and_expect("EXACT_HAMMING_LEN", lambda p: p["exact"]["hamming_family"].pop())
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"extra": 1}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"m": 9}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"n": 9}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"k": 9}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"d_min": 3}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"code_rate_q": [1, 0]}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"code_rate_q": [1, 3]}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"parity_density_q": [1, 3]}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"det": 99}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"covering_sq_128": 55}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"code_rate_q": [True, 2]}))
        mutate_and_expect("EXACT_HAMMING_MISMATCH", lambda p: p["exact"]["hamming_family"][0].update({"code_rate_q": [1]}))

        # DP table branches (including zero denominator, bool-in-int, & S66-VER-01 A32 non-int guards)
        mutate_and_expect("EXACT_DP_BOUNDS", lambda p: p["exact"]["dp_table_reconciliation"].update({"a16_b1207_two_level": True}))
        mutate_and_expect("EXACT_DP_BOUNDS", lambda p: p["exact"]["dp_table_reconciliation"].update({"a16_b1207_two_level": 0}))
        mutate_and_expect("EXACT_DP_BOUNDS", lambda p: p["exact"]["dp_table_reconciliation"].update({"a16_b1207_flat": True}))
        mutate_and_expect("EXACT_DP_BOUNDS", lambda p: p["exact"]["dp_table_reconciliation"].update({"a16_b1207_flat": 0}))
        mutate_and_expect("EXACT_DP_TABLE_MISMATCH", lambda p: p["exact"]["dp_table_reconciliation"].update({"a16_flat_over_two_level_q": [1, 1]}))
        for bad_val in (True, 0, "bad", [], {}):
            mutate_and_expect("EXACT_DP_BOUNDS", lambda p, v=bad_val: p["exact"]["dp_table_reconciliation"].update({"a32_b1015_two_level": v}))
            mutate_and_expect("EXACT_DP_BOUNDS", lambda p, v=bad_val: p["exact"]["dp_table_reconciliation"].update({"a32_b1015_flat": v}))
        for bad_diff in (True, "bad"):
            mutate_and_expect("EXACT_DP_BOUNDS", lambda p, v=bad_diff: p["exact"]["dp_table_reconciliation"].update({"a32_two_level_minus_flat": v}))
        mutate_and_expect("EXACT_DP_TABLE_MISMATCH", lambda p: p["exact"]["dp_table_reconciliation"].update({"a32_two_level_minus_flat": 0}))
        with tempfile.TemporaryDirectory() as tmp_dp:
            for field_name, bad_val in (("a32_b1015_two_level", "bad"), ("a32_b1015_flat", {})):
                p_bad = copy.deepcopy(base)
                p_bad["exact"]["dp_table_reconciliation"][field_name] = bad_val
                c_bad = _resign(p_bad)
                bad_path = os.path.join(tmp_dp, f"{field_name}.json")
                with open(bad_path, "wb") as fh:
                    fh.write(audit.canonical_json_bytes(c_bad) + b"\n")
                err_buf = io.StringIO()
                with mock.patch.object(sys, "stderr", err_buf):
                    self.assertEqual(audit.main(["--check", bad_path]), 1)
                self.assertIn("FAIL EXACT_DP_BOUNDS", err_buf.getvalue())

        # Stage-A invariance branches
        mutate_and_expect("EXACT_STAGE_A_INV_FAIL", lambda p: p["exact"]["stage_a_invariance"].update({"seven_arm_scale_perm_exact_invariant": False}))
        mutate_and_expect("EXACT_STAGE_A_INV_FAIL", lambda p: p["exact"]["stage_a_invariance"].update({"five_arm_scale_perm_exact_invariant": False}))

        # Sampled provenance & number-trace branches
        mutate_and_expect("SAMPLED_SCHEMA", lambda p: p["sampled"].update({"extra": 1}))
        mutate_and_expect("SAMPLED_PROVENANCE", lambda p: p["sampled"].update({"provenance_tier": "exact"}))
        mutate_and_expect("SAMPLED_PROVENANCE", lambda p: p["sampled"].update({"measured_live_llm_in_this_track": True}))
        mutate_and_expect("SAMPLED_PROVENANCE", lambda p: p["sampled"].update({"measured_live_gpu_in_this_track": True}))
        mutate_and_expect("TRACE_AUDIT_MISMATCH", lambda p: p["sampled"]["number_trace_audit"].update({"pr85_hamming_not_found": 1}))
        mutate_and_expect("TRACE_AUDIT_MISMATCH", lambda p: p["sampled"]["number_trace_audit"].update({"pr85_packed_not_found": 1}))
        mutate_and_expect("TRACE_AUDIT_MISMATCH", lambda p: p["sampled"]["number_trace_audit"].update({"c1_holds": False}))

        # Pattern 1-5 branches
        mutate_and_expect("PATTERN1_REGIME_MISMATCH", lambda p: p["sampled"]["pattern_1_regime_triangle"].update({"a16_peaks_pooled_mse_all_three": False}))
        mutate_and_expect("PATTERN1_REGIME_MISMATCH", lambda p: p["sampled"]["pattern_1_regime_triangle"].update({"llama_a32_beats_hbg_on_both_d_and_e05_yet_loses_kl": False}))
        mutate_and_expect("PATTERN1_REGIME_MISMATCH", lambda p: p["sampled"]["pattern_1_regime_triangle"]["models"]["Llama-3.2-1B"]["gap_a32_vs_hbg"].update({"sign": 1}))
        mutate_and_expect("PATTERN1_REGIME_MISMATCH", lambda p: p["sampled"]["pattern_1_regime_triangle"]["models"]["Llama-3.2-1B"]["gap_a32_vs_hbg"].update({"gap_sq_q": [18, 1]}))
        mutate_and_expect("PATTERN1_REGIME_MISMATCH", lambda p: p["sampled"]["pattern_1_regime_triangle"]["models"]["SmolLM2-1.7B"]["gap_a32_vs_hbg"].update({"sign": -1}))
        mutate_and_expect("PATTERN1_REGIME_MISMATCH", lambda p: p["sampled"]["pattern_1_regime_triangle"]["models"]["SmolLM2-1.7B"]["gap_a32_vs_hbg"].update({"gap_sq_q": [17, 1]}))

        mutate_and_expect("PATTERN2_AMP_MISMATCH", lambda p: p["sampled"]["pattern_2_bitrate_amplification"].update({"smollm2_hamming_breaks_e8_catastrophe": False}))
        mutate_and_expect("PATTERN2_AMP_MISMATCH", lambda p: p["sampled"]["pattern_2_bitrate_amplification"].update({"smollm2_hbg_over_a32_amp_ratio_q": [3, 2]}))

        mutate_and_expect("PATTERN3_COLPROXY_MISMATCH", lambda p: p["sampled"]["pattern_3_colproxy_sign_inversion"].update({"total_cells": 20}))
        mutate_and_expect("PATTERN3_COLPROXY_MISMATCH", lambda p: p["sampled"]["pattern_3_colproxy_sign_inversion"].update({"noise_positive_cells": 20}))
        mutate_and_expect("PATTERN3_COLPROXY_MISMATCH", lambda p: p["sampled"]["pattern_3_colproxy_sign_inversion"].update({"gam_attn_positive_count": 8}))
        mutate_and_expect("PATTERN3_COLPROXY_MISMATCH", lambda p: p["sampled"]["pattern_3_colproxy_sign_inversion"].update({"gam_ffn_negative_count": 5}))

        mutate_and_expect("PATTERN4_TAX_SPEED_MISMATCH", lambda p: p["sampled"]["pattern_4_sphere_and_tax_decomposition"].update({"real_ffn_down_is_gaussian_not_laplace": False}))
        mutate_and_expect("PATTERN4_TAX_SPEED_MISMATCH", lambda p: p["sampled"]["pattern_4_sphere_and_tax_decomposition"]["silicon_engine_speed_bridge"].update({"avx512_enum_8t_over_pr85_max_q": [20, 1]}))
        mutate_and_expect("PATTERN4_TAX_SPEED_MISMATCH", lambda p: p["sampled"]["pattern_4_sphere_and_tax_decomposition"]["silicon_engine_speed_bridge"].update({"sm86_72b_transcoded_over_onfly_q": [700, 1]}))

        mutate_and_expect("PATTERN5_RANK_MISMATCH", lambda p: p["sampled"]["pattern_5_stage_a_rank_matrix"].update({"a16_minimax_top2_all_models_45": False}))
        mutate_and_expect("PATTERN5_RANK_MISMATCH", lambda p: p["sampled"]["pattern_5_stage_a_rank_matrix"].update({"a16_minimax_top2_all_models_30": False}))

        mutate_and_expect("ANCHOR_DIGEST_MISMATCH", lambda p: p["anchor"].update({"provenance_tier": "exact"}))
        mutate_and_expect("ANCHOR_DIGEST_MISMATCH", lambda p: p["anchor"].update({"pinned_file_count": 17}))
        mutate_and_expect("ANCHOR_DIGEST_MISMATCH", lambda p: p["anchor"]["pinned_portal_sha256"].pop("notes/colproxy.out"))
        mutate_and_expect("ANCHOR_DIGEST_MISMATCH", lambda p: p["anchor"]["pinned_portal_sha256"].update({"notes/colproxy.out": "bad"}))

        # Final canonical bijective catch on an arbitrary deep leaf not individually pinned above
        mutate_and_expect("PAYLOAD_CANONICAL_MISMATCH", lambda p: p["summary"].update({"pattern_count": 99}))
        mutate_and_expect("PAYLOAD_CANONICAL_MISMATCH", lambda p: p["sampled"]["pattern_4_sphere_and_tax_decomposition"].update({"blk10_p15_bump_milli_db": 46}))

    def test_05_strict_json_parser_rejections(self) -> None:
        self._expect_audit_error("JSON_BAD_UTF8", audit._strict_json_loads, b"\xff\xfe")
        self._expect_audit_error("JSON_FLOAT_FORBIDDEN", audit._strict_json_loads, b'{"a": 1.5}')
        self._expect_audit_error("JSON_CONST_FORBIDDEN", audit._strict_json_loads, b'{"a": NaN}')
        self._expect_audit_error("JSON_DUPLICATE_KEY", audit._strict_json_loads, b'{"a": 1, "a": 2}')
        self._expect_audit_error("JSON_PARSE_ERROR", audit._strict_json_loads, b"{broken")
        self._expect_audit_error("JSON_PARSE_ERROR", audit._strict_json_loads, b'{"x":' + b"9" * 5000 + b"}")

    def test_06_atomic_io_hardlink_fifo_symlink_eexist_chmod500_and_pwd(self) -> None:
        old_umask = os.umask(0o077)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cert = audit.build_certificate()
                out_path = os.path.join(tmp, "sub", "cert.json")
                file_sha = audit.write_certificate_atomic(out_path, cert)
                st = os.lstat(out_path)
                self.assertEqual(stat.S_IMODE(st.st_mode), 0o600)
                with open(out_path, "rb") as fh:
                    disk_bytes = fh.read()
                self.assertEqual(audit.sha256_hex(disk_bytes), file_sha)
                loaded = audit.read_certificate_file(out_path)
                self.assertEqual(loaded["certificate_sha256"], cert["certificate_sha256"])

                # Overwrite existing regular file cleanly
                audit.write_certificate_atomic(out_path, cert)

                # Hardlink rejection on BOTH read and write
                hl_path = os.path.join(tmp, "sub", "cert_link.json")
                os.link(out_path, hl_path)
                self._expect_audit_error("HARDLINK_REJECTED", audit.read_certificate_file, out_path)
                self._expect_audit_error("HARDLINK_REJECTED", audit.write_certificate_atomic, out_path, cert)
                os.unlink(hl_path)

                # Symlink rejection on read, write, and parent directory
                sym_file = os.path.join(tmp, "sym.json")
                os.symlink(out_path, sym_file)
                self._expect_audit_error("IO_ERROR", audit.read_certificate_file, sym_file)
                self._expect_audit_error("NOT_REGULAR_FILE", audit.write_certificate_atomic, sym_file, cert)
                sym_dir = os.path.join(tmp, "symdir")
                os.symlink(os.path.join(tmp, "sub"), sym_dir)
                self._expect_audit_error("IO_ERROR", audit.read_certificate_file, os.path.join(sym_dir, "cert.json"))

                # Scratch FIFO rejection via O_NONBLOCK + S_ISREG
                fifo_path = os.path.join(tmp, "pipe.fifo")
                os.mkfifo(fifo_path, 0o600)
                self._expect_audit_error("NOT_REGULAR_FILE", audit.read_certificate_file, fifo_path)
                self._expect_audit_error("NOT_REGULAR_FILE", audit.write_certificate_atomic, fifo_path, cert)

                # Oversized file rejection, exact max_bytes boundary, and non-canonical bytes (CRLF/pretty)
                big_path = os.path.join(tmp, "big.json")
                with open(big_path, "wb") as fh:
                    fh.write(b" " * (audit.MAX_JSON_BYTES + 1))
                self._expect_audit_error("FILE_SIZE_EXCEEDS_MAX", audit.read_certificate_file, big_path)

                exact5 = os.path.join(tmp, "exact5.bin")
                with open(exact5, "wb") as fh:
                    fh.write(b"12345")
                dparts, dleaf = audit._split_clean_path(exact5)
                dfd = audit._open_dir_chain(dparts, create_missing=False)
                try:
                    self.assertEqual(audit._read_regular_bytes_in_dir(dfd, dleaf, 5), b"12345")
                finally:
                    os.close(dfd)

                crlf_path = os.path.join(tmp, "crlf.json")
                with open(crlf_path, "wb") as fh:
                    fh.write(audit.canonical_json_bytes(cert) + b"\r\n")
                self._expect_audit_error("NON_CANONICAL_BYTES", audit.read_certificate_file, crlf_path)

                pretty_path = os.path.join(tmp, "pretty.json")
                with open(pretty_path, "wb") as fh:
                    fh.write((json.dumps(cert, indent=2) + "\n").encode("utf-8"))
                self._expect_audit_error("NON_CANONICAL_BYTES", audit.read_certificate_file, pretty_path)

                # Reading inside a missing directory must fail closed without creating it
                missing_dir = os.path.join(tmp, "no_such_dir_xyz")
                self._expect_audit_error("IO_ERROR", audit.read_certificate_file, os.path.join(missing_dir, "c.json"))
                self.assertFalse(os.path.exists(missing_dir))

                # EEXIST temp retry succeeds on collision and fails when retries exhausted
                eexist_path = os.path.join(tmp, "eexist.json")
                pid = os.getpid()
                coll = os.path.join(tmp, f".eexist.json.tmp.{pid}.0")
                with open(coll, "wb") as fh:
                    fh.write(b"x")
                audit.write_certificate_atomic(eexist_path, cert)
                self.assertTrue(os.path.isfile(eexist_path))
                for i in range(audit.MAX_TEMP_RETRIES):
                    with open(os.path.join(tmp, f".exhaust.json.tmp.{pid}.{i}"), "wb") as fh:
                        fh.write(b"x")
                self._expect_audit_error(
                    "TEMP_RETRY_EXHAUSTED",
                    audit.write_certificate_atomic,
                    os.path.join(tmp, "exhaust.json"),
                    cert,
                )

                # Read-only directory (chmod 500) OSError wrapped cleanly
                ro_dir = os.path.join(tmp, "ro")
                os.mkdir(ro_dir, 0o500)
                try:
                    self._expect_audit_error(
                        "IO_ERROR",
                        audit.write_certificate_atomic,
                        os.path.join(ro_dir, "newdir", "c.json"),
                        cert,
                    )
                    self._expect_audit_error(
                        "TEMP_OPEN_ERROR",
                        audit.write_certificate_atomic,
                        os.path.join(ro_dir, "c.json"),
                        cert,
                    )
                finally:
                    os.chmod(ro_dir, 0o700)

                # Relative path + $PWD matching vs stale $PWD
                prev_cwd = os.getcwd()
                prev_pwd = os.environ.get("PWD")
                try:
                    os.chdir(os.path.join(tmp, "sub"))
                    os.environ["PWD"] = os.path.join(tmp, "sub")
                    self.assertEqual(audit._safe_getcwd(), os.path.join(tmp, "sub"))
                    os.environ["PWD"] = tmp
                    self.assertEqual(audit._safe_getcwd(), os.getcwd())
                    os.environ["PWD"] = os.path.join(tmp, "nonexistent_dir_xyz")
                    self.assertEqual(audit._safe_getcwd(), os.getcwd())
                    rel_loaded = audit.read_certificate_file("cert.json")
                    self.assertEqual(rel_loaded["certificate_sha256"], cert["certificate_sha256"])
                finally:
                    os.chdir(prev_cwd)
                    if prev_pwd is None:
                        os.environ.pop("PWD", None)
                    else:
                        os.environ["PWD"] = prev_pwd
        finally:
            os.umask(old_umask)

    def test_07_path_guards_darwin_and_oserror_fault_injection(self) -> None:
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, 123)
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, "")
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, "/a/b\x00c")
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, "/a/\ud800.json")
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, "/")
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, "/a/../b")
        self._expect_audit_error("INVALID_PATH", audit._split_clean_path, "/a/./b")

        # S66-VER-03: lone-surrogate paths rejected cleanly on all file/portal/CLI entrypoints & $PWD
        bad_sur = "/tmp/\ud800.json"
        self._expect_audit_error("INVALID_PATH", audit.read_certificate_file, bad_sur)
        self._expect_audit_error("INVALID_PATH", audit.write_certificate_atomic, bad_sur, audit.build_certificate())
        self._expect_audit_error("INVALID_PATH", audit.verify_portal_readonly, "/tmp/\ud800")
        for cli_bad in (["--check", bad_sur], ["--write", bad_sur], ["--verify-portal", "/tmp/\ud800"]):
            err_sur = io.StringIO()
            with mock.patch.object(sys, "stderr", err_sur):
                self.assertEqual(audit.main(cli_bad), 1)
            self.assertIn("FAIL INVALID_PATH", err_sur.getvalue())
        with mock.patch.object(os, "environ", {"PWD": "/\ud800"}):
            self.assertEqual(audit._safe_getcwd(), os.getcwd())

        # Darwin /var, /tmp, /etc root-owned symlink normalization with readlink target verification
        fake_root_lnk = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 0, 0, 0, 0, 0, 0))
        fake_root_dir = os.stat_result((stat.S_IFDIR | 0o755, 1, 1, 1, 0, 0, 0, 0, 0, 0))
        fake_user_lnk = os.stat_result((stat.S_IFLNK | 0o777, 1, 1, 1, 1000, 1000, 0, 0, 0, 0))
        rel_rl = lambda pref: "private" + pref
        abs_rl = lambda pref: "/private" + pref
        evil_rl = lambda pref: "/evil" + pref

        def _raise_os(_: str):
            raise OSError(errno.ENOENT, "missing")

        self.assertEqual(audit._normalize_darwin_prefix("/var/x", "linux", lambda _: fake_root_lnk, rel_rl), "/var/x")
        self.assertEqual(audit._normalize_darwin_prefix("/var/x", "darwin", lambda _: fake_root_lnk, rel_rl), "/private/var/x")
        self.assertEqual(audit._normalize_darwin_prefix("/tmp", "darwin", lambda _: fake_root_lnk, abs_rl), "/private/tmp")
        self.assertEqual(audit._normalize_darwin_prefix("/var/x", "darwin", lambda _: fake_root_lnk, evil_rl), "/var/x")
        self.assertEqual(audit._normalize_darwin_prefix("/var/x", "darwin", lambda _: fake_root_lnk, _raise_os), "/var/x")
        self.assertEqual(audit._normalize_darwin_prefix("/var/x", "darwin", lambda _: fake_root_dir, rel_rl), "/var/x")
        self.assertEqual(audit._normalize_darwin_prefix("/var/x", "darwin", lambda _: fake_user_lnk, rel_rl), "/var/x")
        self.assertEqual(audit._normalize_darwin_prefix("/opt/x", "darwin", lambda _: fake_root_lnk, rel_rl), "/opt/x")
        self.assertEqual(audit._normalize_darwin_prefix("/etc/x", "darwin", _raise_os, rel_rl), "/etc/x")

        # $PWD different st_dev with identical st_ino must fall back to os.getcwd()
        real_cwd = os.getcwd()
        st_a = os.stat_result((stat.S_IFDIR | 0o755, 999, 10, 1, 0, 0, 0, 0, 0, 0))
        st_b = os.stat_result((stat.S_IFDIR | 0o755, 999, 20, 1, 0, 0, 0, 0, 0, 0))
        with mock.patch.dict(os.environ, {"PWD": "/fake_dev_cwd"}), mock.patch.object(os, "stat", side_effect=[st_a, st_b]):
            self.assertEqual(audit._safe_getcwd(), real_cwd)

        # Fault-injection on root open, read growth, temp fstat, write zero-byte, replace OSError, lstat OSError, and S66-VER-02 ENOSPC
        with mock.patch.object(os, "open", side_effect=OSError(errno.EACCES, "denied")):
            self._expect_audit_error("IO_ERROR", audit._open_dir_chain, ["tmp"], create_missing=False)

        with tempfile.TemporaryDirectory() as tmp:
            cert = audit.build_certificate()
            p = os.path.join(tmp, "f.json")
            audit.write_certificate_atomic(p, cert)
            with mock.patch.object(os, "read", side_effect=OSError(errno.EIO, "io")):
                self._expect_audit_error("IO_ERROR", audit.read_certificate_file, p)
            tiny = os.path.join(tmp, "tiny.bin")
            with open(tiny, "wb") as fh:
                fh.write(b"abc")
            with mock.patch.object(os, "read", side_effect=[b"123456", b""]):
                dir_parts, leaf = audit._split_clean_path(tiny)
                dfd = audit._open_dir_chain(dir_parts, create_missing=False)
                try:
                    self._expect_audit_error("STREAM_READ_EXCEEDS_MAX", audit._read_regular_bytes_in_dir, dfd, leaf, 5)
                finally:
                    os.close(dfd)
            fake_hl_st = os.stat_result((stat.S_IFREG | 0o600, 1, 1, 2, 0, 0, 0, 0, 0, 0))
            fake_fifo_st = os.stat_result((stat.S_IFIFO | 0o600, 1, 1, 1, 0, 0, 0, 0, 0, 0))
            with mock.patch.object(os, "fstat", return_value=fake_hl_st):
                self._expect_audit_error("HARDLINK_REJECTED", audit.write_certificate_atomic, os.path.join(tmp, "hl.json"), cert)
            with mock.patch.object(os, "fstat", return_value=fake_fifo_st):
                self._expect_audit_error("HARDLINK_REJECTED", audit.write_certificate_atomic, os.path.join(tmp, "ff.json"), cert)
            with mock.patch.object(os, "lstat", side_effect=OSError(errno.EACCES, "lstat")):
                self._expect_audit_error("IO_ERROR", audit.write_certificate_atomic, p, cert)
            with mock.patch.object(os, "write", return_value=0):
                self._expect_audit_error("IO_ERROR", audit.write_certificate_atomic, os.path.join(tmp, "z.json"), cert)
            with mock.patch.object(os, "replace", side_effect=OSError(errno.EXDEV, "xdev")):
                self._expect_audit_error("IO_ERROR", audit.write_certificate_atomic, os.path.join(tmp, "r.json"), cert)
            # S66-VER-02: OSError inside temp-file write block (fchmod/fstat/write/fsync) wrapped cleanly with zero leaked temp files
            enospc_dir = os.path.join(tmp, "enospc")
            os.mkdir(enospc_dir, 0o700)
            for sc in ("fchmod", "fstat", "write", "fsync"):
                t_path = os.path.join(enospc_dir, f"{sc}.json")
                err_io = io.StringIO()
                with mock.patch.object(os, sc, side_effect=OSError(errno.ENOSPC, "no space")), mock.patch.object(sys, "stderr", err_io):
                    self.assertEqual(audit.main(["--write", t_path]), 1)
                self.assertIn("FAIL IO_ERROR", err_io.getvalue())
            self.assertEqual(os.listdir(enospc_dir), [])

    def test_08_cli_modes_and_fail_closed_argv(self) -> None:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdout", buf_out), mock.patch.object(sys, "stderr", buf_err):
            self.assertEqual(audit.main([]), 0)
            self.assertEqual(audit.main(["--standalone"]), 0)
            self.assertEqual(audit.main(["--standalone", "--json"]), 0)
            # Fail-closed bad argv shapes & flag-as-path attacks
            self.assertEqual(audit.main(123), 1)
            self.assertEqual(audit.main([123]), 1)
            self.assertEqual(audit.main(["--write"]), 1)
            self.assertEqual(audit.main(["--write", _StrSub("/tmp/sub")]), 1)
            self.assertEqual(audit.main(["--write", "--json"]), 1)
            self.assertEqual(audit.main(["--write", ""]), 1)
            self.assertEqual(audit.main(["--check", "-bad"]), 1)
            self.assertEqual(audit.main(["--check", "/tmp/a", "--extra"]), 1)
            self.assertEqual(audit.main(["--write", "/tmp/a", "--json"]), 1)
            self.assertEqual(audit.main(["--unknown"]), 1)
            self.assertIn("INVALID_CLI_ARGS", buf_err.getvalue())
            # Also verify _parse_cli_args rejects str subclass and empty target path directly
            self._expect_audit_error("INVALID_CLI_ARGS", audit._parse_cli_args, ["--standalone", _StrSub("--json")])
            self._expect_audit_error("INVALID_CLI_ARGS", audit._parse_cli_args, ["--write", ""])

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cli_cert.json")
            out1, out2, out3 = io.StringIO(), io.StringIO(), io.StringIO()
            with mock.patch.object(sys, "stdout", out1):
                self.assertEqual(audit.main(["--write", p]), 0)
            with mock.patch.object(sys, "stdout", out2):
                self.assertEqual(audit.main(["--check", p]), 0)
            with mock.patch.object(sys, "stdout", out3):
                self.assertEqual(audit.main(["--check", p, "--json"]), 0)
            with open(p, "rb") as fh:
                cli_disk = fh.read()
            self.assertEqual(out3.getvalue().encode("utf-8"), cli_disk)

    def test_09_portal_readonly_checker_paths(self) -> None:
        self.assertEqual(audit._decimal_text_to_micro("0.018974"), 18974)
        self.assertEqual(audit._decimal_text_to_micro("0.0189746"), 18975)
        for bad_dec in ("NaN", "Infinity", "", "1/0", 123):
            self._expect_audit_error("PORTAL_DECIMAL_INVALID", audit._decimal_text_to_micro, bad_dec)

        with tempfile.TemporaryDirectory() as tmp:
            _make_mock_portal(tmp)
            # SHA mismatch triggers PORTAL_SHA_MISMATCH
            self._expect_audit_error("PORTAL_SHA_MISMATCH", audit.verify_portal_readonly, tmp)

            # Missing nested subdir triggers PORTAL_IO_ERROR
            empty_dir = os.path.join(tmp, "empty")
            os.mkdir(empty_dir)
            self._expect_audit_error("PORTAL_IO_ERROR", audit.verify_portal_readonly, empty_dir)

            # Mock valid portal with matching SHAs and exact 16 JSONL rows -> PASS + CLI --verify-portal
            good_dir = os.path.join(tmp, "good")
            os.mkdir(good_dir)
            hl_rows = []
            for mfile, mname in (
                ("qwen2.5-1.5b-instruct-fp16.gguf", "Qwen2.5-1.5B"),
                ("SmolLM2-1.7B-Instruct-f16.gguf", "SmolLM2-1.7B"),
                ("Llama-3.2-1B-Instruct-f16.gguf", "Llama-3.2-1B"),
            ):
                for arm in ("A16BG", "A32BG"):
                    hl_rows.append(json.dumps({"model": mfile, "variant": f"{arm}128:b560s16", "kl_mean": _fmt_micro(oracle.KL_45_MICRO[mname][arm])}))
                if mname != "Llama-3.2-1B":
                    for arm in ("A16BG", "A32BG"):
                        hl_rows.append(json.dumps({"model": mfile, "variant": f"{arm}128:b368s16", "kl_mean": _fmt_micro(oracle.KL_30_MICRO[mname][arm])}))
            hl_rows.append(json.dumps({"model": "other.gguf", "variant": "A16BG128:b560s16", "kl_mean": "0.100000"}))
            hl_rows.append(json.dumps({"model": "other.gguf", "variant": "A16BG128:b368s16", "kl_mean": "0.100000"}))
            hl_rows.append(json.dumps({"model": "qwen2.5-1.5b-instruct-fp16.gguf", "variant": "HBG128:b560s16", "kl_mean": "0.018875"}))
            hl_rows.append("")
            pk_rows = []
            for mfile, mname in (
                ("qwen2.5-1.5b-instruct-fp16.gguf", "Qwen2.5-1.5B"),
                ("SmolLM2-1.7B-Instruct-f16.gguf", "SmolLM2-1.7B"),
                ("Llama-3.2-1B-Instruct-f16.gguf", "Llama-3.2-1B"),
            ):
                pk_rows.append(json.dumps({"model": mfile, "variant": "PH16:b136s8", "kl_mean": _fmt_micro(oracle.PACKED_KL_MICRO[mname]["PH16"])}))
                pk_rows.append(json.dumps({"model": mfile, "variant": "PH16:b136s8+K10", "kl_mean": _fmt_micro(oracle.PACKED_KL_MICRO[mname]["PH16+K10"])}))
            pk_rows.append("")

            custom_hashes = {}
            for rel_path in oracle.PINNED_PORTAL_SHA256:
                full = os.path.join(good_dir, rel_path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                if rel_path.endswith("hamming-lattices-real-weights-2026-11-24/results.jsonl"):
                    raw = ("\n".join(hl_rows)).encode("utf-8")
                elif rel_path.endswith("packed-hamming-2026-11-24/results.jsonl"):
                    raw = ("\n".join(pk_rows)).encode("utf-8")
                else:
                    raw = b"ok\n"
                with open(full, "wb") as fh:
                    fh.write(raw)
                custom_hashes[rel_path] = hashlib.sha256(raw).hexdigest()

            with mock.patch.object(oracle, "PINNED_PORTAL_SHA256", custom_hashes):
                res = audit.verify_portal_readonly(good_dir)
                self.assertEqual(res["verified_files"], 18)
                self.assertEqual(res["checked_jsonl_rows"], 16)
                buf1, buf2 = io.StringIO(), io.StringIO()
                with mock.patch.object(sys, "stdout", buf1):
                    self.assertEqual(audit.main(["--verify-portal", good_dir]), 0)
                with mock.patch.object(sys, "stdout", buf2):
                    self.assertEqual(audit.main(["--verify-portal", good_dir, "--json"]), 0)
                self.assertIn("PASS portal", buf1.getvalue())
                self.assertIn('"portal_verified":true', buf2.getvalue())

                # Tamper one KL value and verify PORTAL_VALUE_MISMATCH on 4.5, 3.0, and packed
                for rel_target, bad_line in (
                    (
                        "experiments/hamming-lattices-real-weights-2026-11-24/results.jsonl",
                        json.dumps({"model": "qwen2.5-1.5b-instruct-fp16.gguf", "variant": "A16BG128:b560s16", "kl_mean": "0.999999"}),
                    ),
                    (
                        "experiments/hamming-lattices-real-weights-2026-11-24/results.jsonl",
                        json.dumps({"model": "qwen2.5-1.5b-instruct-fp16.gguf", "variant": "A16BG128:b368s16", "kl_mean": "0.999999"}),
                    ),
                    (
                        "experiments/packed-hamming-2026-11-24/results.jsonl",
                        json.dumps({"model": "qwen2.5-1.5b-instruct-fp16.gguf", "variant": "PH16:b136s8", "kl_mean": "0.999999"}),
                    ),
                ):
                    full = os.path.join(good_dir, rel_target)
                    with open(full, "rb") as fh:
                        orig_raw = fh.read()
                    tampered_raw = (bad_line + "\n").encode("utf-8")
                    with open(full, "wb") as fh:
                        fh.write(tampered_raw)
                    alt_hashes = dict(custom_hashes)
                    alt_hashes[rel_target] = hashlib.sha256(tampered_raw).hexdigest()
                    with mock.patch.object(oracle, "PINNED_PORTAL_SHA256", alt_hashes):
                        self._expect_audit_error("PORTAL_VALUE_MISMATCH", audit.verify_portal_readonly, good_dir)
                    with open(full, "wb") as fh:
                        fh.write(orig_raw)

                # Fewer than 16 rows triggers PORTAL_ROW_COUNT; malformed JSONL triggers PORTAL_JSONL_SCHEMA
                hl_full = os.path.join(good_dir, "experiments/hamming-lattices-real-weights-2026-11-24/results.jsonl")
                short_raw = b"\n"
                with open(hl_full, "wb") as fh:
                    fh.write(short_raw)
                alt_hashes = dict(custom_hashes)
                alt_hashes["experiments/hamming-lattices-real-weights-2026-11-24/results.jsonl"] = hashlib.sha256(short_raw).hexdigest()
                with mock.patch.object(oracle, "PINNED_PORTAL_SHA256", alt_hashes):
                    self._expect_audit_error("PORTAL_ROW_COUNT", audit.verify_portal_readonly, good_dir)

                bad_jsonl = b"{not-json}\n"
                with open(hl_full, "wb") as fh:
                    fh.write(bad_jsonl)
                alt_hashes["experiments/hamming-lattices-real-weights-2026-11-24/results.jsonl"] = hashlib.sha256(bad_jsonl).hexdigest()
                with mock.patch.object(oracle, "PINNED_PORTAL_SHA256", alt_hashes):
                    self._expect_audit_error("PORTAL_JSONL_SCHEMA", audit.verify_portal_readonly, good_dir)

                # Direct _parse_jsonl_rows schema checks (non-dict, missing fields, NaN constant, dup key) & unknown packed model
                for raw_bad in (
                    b'["model","variant","kl_mean"]\n',
                    b'{"variant":"v","kl_mean":"0.1"}\n',
                    b'{"model":"m","kl_mean":"0.1"}\n',
                    b'{"model":"m","variant":"v"}\n',
                    b'{"model":"m","variant":"v","kl_mean":NaN}\n',
                    b'{"model":"m","model":"m2","variant":"v","kl_mean":"0.1"}\n',
                ):
                    self._expect_audit_error("PORTAL_JSONL_SCHEMA", audit._parse_jsonl_rows, raw_bad)

                pk_full = os.path.join(good_dir, "experiments/packed-hamming-2026-11-24/results.jsonl")
                bad_pk = (json.dumps({"model": "unknown.gguf", "variant": "PH16:b136s8", "kl_mean": "0.1"}) + "\n").encode("utf-8")
                with open(hl_full, "wb") as fh:
                    fh.write(("\n".join(hl_rows)).encode("utf-8"))
                with open(pk_full, "wb") as fh:
                    fh.write(bad_pk)
                alt_hashes2 = dict(custom_hashes)
                alt_hashes2["experiments/packed-hamming-2026-11-24/results.jsonl"] = hashlib.sha256(bad_pk).hexdigest()
                with mock.patch.object(oracle, "PINNED_PORTAL_SHA256", alt_hashes2):
                    self._expect_audit_error("PORTAL_JSONL_SCHEMA", audit.verify_portal_readonly, good_dir)


if __name__ == "__main__":
    unittest.main()
