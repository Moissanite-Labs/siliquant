#!/usr/bin/env python3
"""Assertion-free R5 mutation-killing, AST-conjunct-killing, and Zone-B adversarial test suite
for AUDIT-112 (Track S15: siliquant#23/#25 pure-Z/Q binary-tree shell-convolution & E_8^4 codec).

Run with: python3 -I -O -B -W error test_ballpack_tree_r5.py -v
"""

from __future__ import annotations

from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ballpack_tree_oracle import (
    BallpackTreeError,
    _bytes_per_entry_for_bits,
    _hardware_type_for_bits,
    build_e8_tree_tables,
    build_exact_reciprocal,
    build_z_tree_tables,
    canonical_sha256,
    certify_e8_tree_bijection_exact,
    certify_shaping_and_bandwidth_ceiling_exact,
    certify_z_tree_bijection_exact,
    contains_forbidden_json_value,
    count_z_ball_serial_dp,
    deterministic_sample_ranks,
    divmod_exact_reciprocal,
    e8_theta_shell_count_exact,
    format_exact_decimal,
    format_exact_percent,
    frac_to_dict,
    is_valid_e8_doubled_block,
    rank_e8_block_doubled,
    rank_tree_e8_blocks,
    rank_tree_z,
    require_exact_bool,
    require_exact_fraction,
    require_exact_int,
    require_git_sha40,
    require_non_empty_str,
    require_sha256,
    sigma3_exact,
    unrank_e8_block_doubled,
    unrank_tree_e8_blocks,
    unrank_tree_z,
    verify_reciprocal_universal,
)
from ballpack_tree_audit import (
    PINNED_ARTIFACTS_EXPECTED,
    PINNED_COMMIT_SHA,
    PINNED_INTAKE_FILES,
    PINNED_SOURCE_AUDIT_EXPECTED,
    main as audit_main,
    run_live_audit,
    run_standalone_audit,
    sha256_file,
    validate_audit_certificate,
)


def _build_synthetic_intake(root: Path) -> None:
    (root / "shape").mkdir(parents=True, exist_ok=True)
    (root / "shape" / "ballindex.py").write_text(
        "class _Dummy:\n    pass\n"
        "class BallIndex:\n"
        '    """BallIndex mock."""\n'
        "    def _other(self):\n        return 0\n"
        "    def table_entries(self):\n"
        "        if self.lattice == 'Z8':\n"
        "            return (self.n + 1) * (self.budget + 1)\n"
        "        if self.lattice == 'E8':\n"
        "            return (self.n // 8) * 9 * 4 * (self.budget + 1) + (self.n // 8 + 1) * (self.budget + 1)\n"
        "    def unrank(self, k):\n"
        "        for i in range(self.n):\n            pass\n",
        encoding="utf-8",
    )
    (root / "shape" / "ballpack.c").write_text(
        "#define LIMBS 3\n"
        "static inline int unrank_one(int n) { for (int i = 0; i < n; ++i) {} return 0; }\n",
        encoding="utf-8",
    )
    (root / "shape" / "ballpack.py").write_text(
        "class _DummyFmt:\n    pass\n"
        "class Format:\n"
        "    def _other(self):\n        return 0\n"
        "    def __init__(self, n=32, bits=136, steps=16):\n"
        "        self.index = BallIndex('Z8', n, 783)\n",
        encoding="utf-8",
    )


class TestPrimitivesAndReciprocals(unittest.TestCase):
    def test_strict_validators_formatters_and_json_guard(self) -> None:
        self.assertEqual(require_exact_int(0, "x", allow_zero=True), 0)
        self.assertEqual(require_exact_int(1, "x", allow_zero=False), 1)
        self.assertEqual(require_exact_int(-5, "x", min_val=-5), -5)
        self.assertIs(require_exact_bool(True, "b"), True)
        self.assertEqual(require_exact_fraction(Fraction(3, 4), "f"), Fraction(3, 4))
        self.assertEqual(require_non_empty_str("ok", "s"), "ok")
        self.assertEqual(require_sha256("ab" * 32, "h"), "ab" * 32)
        self.assertEqual(require_git_sha40("ab" * 20, "g"), "ab" * 20)

        for bad_i in (True, False, "1", -1):
            with self.assertRaises(BallpackTreeError):
                require_exact_int(bad_i, "x", allow_zero=True)
        with self.assertRaises(BallpackTreeError):
            require_exact_int(0, "x", allow_zero=False)
        with self.assertRaises(BallpackTreeError):
            require_exact_int(-6, "x", min_val=-5)
        for bad_b in (0, 1, "true"):
            with self.assertRaises(BallpackTreeError):
                require_exact_bool(bad_b, "b")
        with self.assertRaises(BallpackTreeError):
            require_exact_fraction(1, "f")
        for bad_str in (123, "", " ok", "ok "):
            with self.assertRaisesRegex(BallpackTreeError, "non-empty trimmed str"):
                require_non_empty_str(bad_str, "s")
        for bad_s in (123, "short", "A" * 64, "0" * 64):
            with self.assertRaisesRegex(BallpackTreeError, "sha256"):
                require_sha256(bad_s, "h")
        for bad_g in (123, "short", "A" * 40, "0" * 40):
            with self.assertRaisesRegex(BallpackTreeError, "git sha"):
                require_git_sha40(bad_g, "g")

        self.assertEqual(frac_to_dict(Fraction(6, 8)), {"num": 3, "den": 4})
        self.assertEqual(format_exact_decimal(Fraction(-3, 4), 2), "-0.75")
        self.assertEqual(format_exact_decimal(Fraction(3, 4), 2), "0.75")
        self.assertEqual(format_exact_decimal(Fraction(7, 2), 0), "4")
        self.assertEqual(format_exact_decimal(Fraction(-1, 1000), 0), "0")
        self.assertEqual(format_exact_decimal(Fraction(-1, 10 ** 9), 6), "0.000000")
        self.assertEqual(format_exact_decimal(Fraction(0, 1), 0), "0")
        self.assertEqual(format_exact_decimal(Fraction(0, 1), 2), "0.00")
        self.assertEqual(format_exact_percent(Fraction(1, 8), 2), "12.50%")

        self.assertEqual(_hardware_type_for_bits(16), "uint16")
        self.assertEqual(_hardware_type_for_bits(32), "uint32")
        self.assertEqual(_hardware_type_for_bits(64), "uint64")
        self.assertEqual(_hardware_type_for_bits(128), "uint128")
        self.assertEqual(_hardware_type_for_bits(136), "uint192")
        self.assertEqual(_bytes_per_entry_for_bits(16), 2)
        self.assertEqual(_bytes_per_entry_for_bits(32), 4)
        self.assertEqual(_bytes_per_entry_for_bits(64), 8)
        self.assertEqual(_bytes_per_entry_for_bits(128), 16)
        self.assertEqual(_bytes_per_entry_for_bits(136), 24)

        self.assertFalse(contains_forbidden_json_value({"a": [1, "x", (True, None)]}))
        self.assertTrue(contains_forbidden_json_value(json.loads('{"x": 1.5}')))
        self.assertTrue(contains_forbidden_json_value({1: "int_key"}))
        self.assertTrue(contains_forbidden_json_value(complex(1, 2)))
        self.assertTrue(contains_forbidden_json_value({1, 2}))
        self.assertTrue(contains_forbidden_json_value(frozenset([1])))
        self.assertTrue(contains_forbidden_json_value(b"bytes"))
        self.assertTrue(contains_forbidden_json_value([json.loads("0.5")]))

        with self.assertRaisesRegex(BallpackTreeError, "strict str-key"):
            canonical_sha256(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(BallpackTreeError, "strict str-key"):
            canonical_sha256({1: "int_key"})  # type: ignore[dict-item]

        self.assertEqual(deterministic_sample_ranks(1, n_random=0), [0])
        self.assertEqual(deterministic_sample_ranks(2, n_random=0), [0, 1])

    def test_zero_division_barrett_granlund_montgomery_reciprocals(self) -> None:
        # Power of two (d=1, d=8) and non-power-of-two (d=7, d=1000003)
        for d, max_div in ((1, 0), (1, 100), (8, 999), (7, 500), (1000003, (1 << 64) - 1)):
            rec = build_exact_reciprocal(d, max_div)
            self.assertTrue(rec["universally_valid"])
            self.assertEqual(rec["is_pow2"], (d & (d - 1)) == 0)
            for x in (0, 1, min(d - 1, max_div), min(d, max_div), max_div // 2, max_div):
                if x <= max_div:
                    q, r = divmod_exact_reciprocal(x, d, max_div, rec["mult"], rec["shift"])
                    self.assertEqual((q, r), divmod(x, d))

        # Universal certificate negative controls: both conjuncts tested individually!
        # 1. mult * d < 2^shift (underestimates -> fails first conjunct!)
        self.assertFalse(verify_reciprocal_universal(7, 10, 1, 3))  # 7 < 8
        # 2. mult * d >= 2^shift, YET (mult * d - 2^shift) * max_dividend >= 2^shift (fails second conjunct!)
        self.assertFalse(verify_reciprocal_universal(7, 8, 2, 3))  # 14 >= 8, (14-8)*8 = 48 >= 8!
        with self.assertRaisesRegex(BallpackTreeError, "exceeds certified"):
            divmod_exact_reciprocal(11, 7, 10, 15, 7)
        with self.assertRaisesRegex(BallpackTreeError, "fail universal"):
            divmod_exact_reciprocal(5, 7, 8, 2, 3)


class TestPillars1And2Bijections(unittest.TestCase):
    def test_pillar1_z_tree_bijection_and_guards(self) -> None:
        # Exhaustive Z^4 (B=4, W=89) & sampled Z^8 (B=3)
        z4 = certify_z_tree_bijection_exact(4, 4, exhaustive=True)
        self.assertEqual(z4["codebook_count"], 89)
        self.assertTrue(z4["count_matches_ballindex_dp"])
        self.assertTrue(z4["all_roundtrips_exact"])
        self.assertTrue(z4["exhaustive_bijection_distinct"])
        self.assertTrue(z4["all_reciprocals_universally_valid"])

        # Z^2 base case (bound=0 and bound=2)
        z2_0 = certify_z_tree_bijection_exact(2, 0, exhaustive=True)
        self.assertEqual(z2_0["codebook_count"], 1)
        self.assertEqual(z2_0["tree_split_depth"], 0)
        z8_s = certify_z_tree_bijection_exact(8, 2, exhaustive=False, n_random=6)
        self.assertTrue(z8_s["all_roundtrips_exact"])

        # Z^16 sampled exercises non-root Ball->Shell+Ball AND Shell->Shell+Shell
        z16 = certify_z_tree_bijection_exact(16, 4, exhaustive=False, n_random=8)
        self.assertTrue(z16["all_roundtrips_exact"])
        self.assertTrue(z16["count_matches_ballindex_dp"])

        # Guards
        for bad_n in (True, "4", 0, 1, 3, 6):
            with self.assertRaisesRegex(BallpackTreeError, "power of two"):
                build_z_tree_tables(bad_n, 4)  # type: ignore[arg-type]
        tbl = build_z_tree_tables(4, 4)
        for bad_pt in (123, "abcd", b"abcd"):
            with self.assertRaisesRegex(BallpackTreeError, "sequence of exact ints"):
                rank_tree_z(bad_pt, tbl)  # type: ignore[arg-type]
        with self.assertRaisesRegex(BallpackTreeError, "length must equal 4"):
            rank_tree_z([0, 0], tbl)
        with self.assertRaisesRegex(BallpackTreeError, "exceeds ball bound"):
            rank_tree_z([3, 0, 0, 0], tbl)
        with self.assertRaisesRegex(BallpackTreeError, "exceeds codebook"):
            unrank_tree_z(tbl["count"], tbl)
        self.assertEqual(count_z_ball_serial_dp(4, 4), 89)

    def test_pillar2_e8_theta_and_tree_codec_and_guards(self) -> None:
        self.assertEqual(sigma3_exact(1), 1)
        self.assertEqual(sigma3_exact(4), 1 + 8 + 64)  # perfect square divisor d==q
        self.assertEqual(sigma3_exact(6), 1 + 8 + 27 + 216)  # non-square divisor d!=q
        self.assertEqual(e8_theta_shell_count_exact(0), 1)
        self.assertEqual(e8_theta_shell_count_exact(1), 240)
        self.assertEqual(e8_theta_shell_count_exact(2), 2160)

        # is_valid_e8_doubled_block guards
        self.assertTrue(is_valid_e8_doubled_block([0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(is_valid_e8_doubled_block([2, 2, 0, 0, 0, 0, 0, 0]))
        self.assertTrue(is_valid_e8_doubled_block([1, 1, 1, 1, 1, 1, 1, 1]))
        self.assertFalse(is_valid_e8_doubled_block(123))  # type: ignore[arg-type]
        self.assertFalse(is_valid_e8_doubled_block("12345678"))  # type: ignore[arg-type]
        self.assertFalse(is_valid_e8_doubled_block(b"\x00" * 8))  # type: ignore[arg-type]
        self.assertFalse(is_valid_e8_doubled_block([0] * 7))
        self.assertFalse(is_valid_e8_doubled_block([False] + [0] * 7))
        self.assertFalse(is_valid_e8_doubled_block([True] + [0] * 7))
        self.assertFalse(is_valid_e8_doubled_block([1, 0, 0, 0, 0, 0, 0, 0]))  # mixed parity
        self.assertFalse(is_valid_e8_doubled_block([2, 0, 0, 0, 0, 0, 0, 0]))  # odd sum/2
        self.assertFalse(is_valid_e8_doubled_block([-1, 1, 1, 1, 1, 1, 1, 1]))  # odd sum/2

        # E_8^1 (m=0 and m=1 exhaustive) + E_8^2 (m=2) + E_8^4 (m=2 sampled -> hits non-root Ball & Shell!)
        e1_0 = certify_e8_tree_bijection_exact(1, 0, exhaustive=True)
        self.assertEqual(e1_0["codebook_count"], 1)
        e1_1 = certify_e8_tree_bijection_exact(1, 1, exhaustive=True)
        self.assertEqual(e1_1["codebook_count"], 241)
        self.assertTrue(e1_1["theta_series_all_shells_exact"])
        self.assertTrue(e1_1["all_roundtrips_exact"])
        self.assertTrue(e1_1["exhaustive_bijection_distinct"])
        e4_2 = certify_e8_tree_bijection_exact(4, 2, exhaustive=False, n_random=16)
        self.assertTrue(e4_2["all_roundtrips_exact"])
        self.assertTrue(e4_2["theta_series_all_shells_exact"])

        # Direct single-block shell/ball rank & unrank + S15-BUG-03 / S15-BUG-04 guards
        tbl1 = build_e8_tree_tables(1, 2)
        for is_ball in (True, False):
            cap = sum(tbl1["single"]["shells_e8"][:3]) if is_ball else tbl1["single"]["shells_e8"][2]
            for rk in (0, cap - 1):
                u_pt = unrank_e8_block_doubled(rk, tbl1, is_ball=is_ball, m_bound=2)
                self.assertEqual(rank_e8_block_doubled(u_pt, tbl1, is_ball=is_ball, m_bound=2), rk)
            with self.assertRaisesRegex(BallpackTreeError, "rank exceeds shell/ball count"):
                unrank_e8_block_doubled(cap, tbl1, is_ball=is_ball, m_bound=2)
        for bad_rk in (True, False, -1, 999999):
            with self.assertRaises(BallpackTreeError):
                unrank_e8_block_doubled(bad_rk, tbl1, is_ball=True, m_bound=2)  # type: ignore[arg-type]
        for bad_ib in (0, 1, "true"):
            with self.assertRaises(BallpackTreeError):
                unrank_e8_block_doubled(0, tbl1, is_ball=bad_ib, m_bound=2)  # type: ignore[arg-type]
            with self.assertRaises(BallpackTreeError):
                rank_e8_block_doubled([0] * 8, tbl1, is_ball=bad_ib, m_bound=2)  # type: ignore[arg-type]
        for bad_mb in (True, -1, 3):
            with self.assertRaises(BallpackTreeError):
                unrank_e8_block_doubled(0, tbl1, is_ball=True, m_bound=bad_mb)  # type: ignore[arg-type]
            with self.assertRaises(BallpackTreeError):
                rank_e8_block_doubled([0] * 8, tbl1, is_ball=True, m_bound=bad_mb)  # type: ignore[arg-type]
        with self.assertRaisesRegex(BallpackTreeError, "not a valid doubled E_8"):
            rank_e8_block_doubled([1, 0, 0, 0, 0, 0, 0, 0], tbl1, is_ball=True, m_bound=2)
        with self.assertRaisesRegex(BallpackTreeError, "does not match requested"):
            rank_e8_block_doubled([4, 4, 4, 0, 0, 0, 0, 0], tbl1, is_ball=True, m_bound=2)
        with self.assertRaisesRegex(BallpackTreeError, "does not match requested"):
            rank_e8_block_doubled([0] * 8, tbl1, is_ball=False, m_bound=2)

        for bad_nb in (0, 3):
            with self.assertRaises(BallpackTreeError):
                build_e8_tree_tables(bad_nb, 1)
        for bad_u in (123, "a" * 8, b"a" * 8):
            with self.assertRaisesRegex(BallpackTreeError, "sequence of exact ints"):
                rank_tree_e8_blocks(bad_u, tbl1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(BallpackTreeError, "length must equal 8"):
            rank_tree_e8_blocks([0] * 4, tbl1)
        with self.assertRaisesRegex(BallpackTreeError, "block 0 is not a valid doubled E_8"):
            rank_tree_e8_blocks([1, 0, 0, 0, 0, 0, 0, 0], tbl1)
        with self.assertRaisesRegex(BallpackTreeError, "exceeds E_8 ball bound"):
            rank_tree_e8_blocks([4, 4, 4, 0, 0, 0, 0, 0], tbl1)
        with self.assertRaisesRegex(BallpackTreeError, "exceeds E_8 codebook"):
            unrank_tree_e8_blocks(tbl1["count"], tbl1)


class TestPillar3AndAuditRunner(unittest.TestCase):
    def test_pillar3_guards_and_audit_certificate_anti_bypass_and_cli(self) -> None:
        with self.assertRaisesRegex(BallpackTreeError, "Z\\^32 table dict"):
            certify_shaping_and_bandwidth_ceiling_exact(123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(BallpackTreeError, "Z\\^32 table dict"):
            certify_shaping_and_bandwidth_ceiling_exact({"n": 16})

        sa = run_standalone_audit()
        self.assertTrue(sa["overall_verified"])
        self.assertTrue(validate_audit_certificate(sa)["valid"])

        def _rehash(d: dict) -> dict:
            out = dict(d)
            out.pop("certificate_sha256", None)
            out["certificate_sha256"] = canonical_sha256(out)
            return out

        with self.assertRaisesRegex(BallpackTreeError, "mapping"):
            validate_audit_certificate(123)  # type: ignore[arg-type]
        flt = json.loads(json.dumps(sa))
        flt["pillar1_z32_tree_bijection"]["z32_b783_136bit"]["bound"] = json.loads("783.5")
        with self.assertRaisesRegex(BallpackTreeError, "forbidden"):
            validate_audit_certificate(flt)
        with self.assertRaisesRegex(BallpackTreeError, "mode invalid"):
            validate_audit_certificate({"schema_version": "x"})
        with self.assertRaisesRegex(BallpackTreeError, "mode invalid"):
            validate_audit_certificate(_rehash(dict(sa, mode="bad")))
        with self.assertRaisesRegex(BallpackTreeError, "key set mismatch"):
            validate_audit_certificate(dict(sa, forged="N2"))
        with self.assertRaisesRegex(BallpackTreeError, "schema_version"):
            validate_audit_certificate(_rehash(dict(sa, schema_version="wrong")))
        with self.assertRaisesRegex(BallpackTreeError, "audit_id"):
            validate_audit_certificate(_rehash(dict(sa, audit_id="AUDIT-999")))
        with self.assertRaisesRegex(BallpackTreeError, "novelty_class"):
            validate_audit_certificate(_rehash(dict(sa, novelty_class="N2")))
        for bad_live in (True, 0):
            with self.assertRaisesRegex(BallpackTreeError, "live_intake_audited"):
                validate_audit_certificate(_rehash(dict(sa, live_intake_audited=bad_live)))
        for bad_ver in (False, 1):
            with self.assertRaisesRegex(BallpackTreeError, "overall_verified"):
                validate_audit_certificate(_rehash(dict(sa, overall_verified=bad_ver)))
        with self.assertRaisesRegex(BallpackTreeError, "digest mismatch"):
            validate_audit_certificate(dict(sa, certificate_sha256="01" * 32))

        for pkey in (
            "pillar1_z32_tree_bijection",
            "pillar2_e8_4_tree_codec",
            "pillar3_shaping_and_roofline",
        ):
            bad_int_bool = dict(sa[pkey], pillar_verified=1)
            for bad_sub in (123, {"pillar_verified": True}, bad_int_bool):
                with self.assertRaisesRegex(BallpackTreeError, pkey):
                    validate_audit_certificate(_rehash(dict(sa, **{pkey: bad_sub})))
        p3_short = json.loads(json.dumps(sa["pillar3_shaping_and_roofline"]))
        p3_short["shaping_and_bandwidth"]["subblock_shaping_comparison"] = []
        with self.assertRaisesRegex(BallpackTreeError, "pillar3"):
            validate_audit_certificate(_rehash(dict(sa, pillar3_shaping_and_roofline=p3_short)))
        p3_elem = json.loads(json.dumps(sa["pillar3_shaping_and_roofline"]))
        p3_elem["shaping_and_bandwidth"]["subblock_shaping_comparison"][0]["sub_bound"] = 999
        with self.assertRaisesRegex(BallpackTreeError, "pillar3"):
            validate_audit_certificate(_rehash(dict(sa, pillar3_shaping_and_roofline=p3_elem)))

        # Synthetic live intake + S15-BUG-05 fail-closed pinned provenance + CLI (--standalone, --live, --check)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "intake"
            _build_synthetic_intake(root)
            # Unpinned / tampered intake MUST fail closed (overall_verified=False)!
            unpinned_cert = run_live_audit(root)
            self.assertFalse(unpinned_cert["overall_verified"])
            with self.assertRaisesRegex(BallpackTreeError, "overall_verified"):
                validate_audit_certificate(unpinned_cert, intake_dir=root)
            forged_true_unpinned = _rehash(dict(unpinned_cert, overall_verified=True))
            with self.assertRaisesRegex(BallpackTreeError, "pinned_artifacts provenance mismatch"):
                validate_audit_certificate(forged_true_unpinned, intake_dir=root)

            # Tampered table_entries() body (e.g., return 0) must ALSO invalidate live_source_audit!
            tampered_root = Path(td) / "tampered_intake"
            _build_synthetic_intake(tampered_root)
            (tampered_root / "shape" / "ballindex.py").write_text(
                "class BallIndex:\n"
                "    def table_entries(self):\n        return ('Z8', 'E8', 0)\n"
                "    def unrank(self, k):\n        for i in range(self.n):\n            pass\n",
                encoding="utf-8",
            )
            tampered_cert = run_live_audit(tampered_root)
            self.assertFalse(tampered_cert["overall_verified"])
            self.assertEqual(tampered_cert["live_source_audit"]["ballindex_z32_table_entries"], 0)
            self.assertEqual(tampered_cert["live_source_audit"]["ballindex_e8_136_table_entries"], 0)
            self.assertEqual(tampered_cert["live_source_audit"]["ballindex_e8_88_table_entries"], 0)

            # Test missing second half of table_entries formula (kills second conjunct in bi_tbl_formula_ok!)
            (tampered_root / "shape" / "ballindex.py").write_text(
                "class BallIndex:\n"
                "    def table_entries(self):\n        return ('Z8', 'E8', (self.n + 1) * (self.budget + 1))\n"
                "    def unrank(self, k):\n        for i in range(self.n):\n            pass\n",
                encoding="utf-8",
            )
            tampered_cert2 = run_live_audit(tampered_root)
            self.assertEqual(tampered_cert2["live_source_audit"]["ballindex_z32_table_entries"], 0)

            # Test missing first half of table_entries formula (kills first conjunct in bi_tbl_formula_ok!)
            (tampered_root / "shape" / "ballindex.py").write_text(
                "class BallIndex:\n"
                "    def table_entries(self):\n"
                "        return ('Z8', 'E8', self.n // 8 * 9 * 4 * (self.budget + 1))\n"
                "    def unrank(self, k):\n        for i in range(self.n):\n            pass\n",
                encoding="utf-8",
            )
            tampered_cert3 = run_live_audit(tampered_root)
            self.assertEqual(tampered_cert3["live_source_audit"]["ballindex_z32_table_entries"], 0)

            # Even if pinned SHAs match tampered_root, broken live_source_audit MUST set overall_verified=False!
            tamp_expected = {
                rel: {"sha256": sha256_file(tampered_root / rel)[0], "bytes": sha256_file(tampered_root / rel)[1]}
                for rel in PINNED_INTAKE_FILES
            }
            with mock.patch.dict(PINNED_ARTIFACTS_EXPECTED, tamp_expected, clear=True):
                self.assertFalse(run_live_audit(tampered_root)["overall_verified"])

            # Offline pinned provenance (intake_dir=None) + mismatched intake_dir rejection
            pinned_live = _rehash(
                dict(
                    unpinned_cert,
                    overall_verified=True,
                    pinned_commit_sha=PINNED_COMMIT_SHA,
                    pinned_artifacts=json.loads(json.dumps(PINNED_ARTIFACTS_EXPECTED)),
                    live_source_audit=json.loads(json.dumps(PINNED_SOURCE_AUDIT_EXPECTED)),
                )
            )
            self.assertTrue(validate_audit_certificate(pinned_live)["valid"])
            with self.assertRaisesRegex(BallpackTreeError, "provenance mismatch against intake_dir"):
                validate_audit_certificate(pinned_live, intake_dir=root)
            with self.assertRaisesRegex(BallpackTreeError, "pinned_commit_sha provenance mismatch"):
                validate_audit_certificate(_rehash(dict(pinned_live, pinned_commit_sha="deadbeef" * 5)))
            forged_pa = json.loads(json.dumps(PINNED_ARTIFACTS_EXPECTED))
            forged_pa["shape/ballpack.c"] = {"sha256": "ab" * 32, "bytes": 1}
            with self.assertRaisesRegex(BallpackTreeError, "pinned_artifacts provenance mismatch"):
                validate_audit_certificate(_rehash(dict(pinned_live, pinned_artifacts=forged_pa)))

            # Structural live validator guards
            for bad_pa in (123, {}):
                with self.assertRaisesRegex(BallpackTreeError, "pinned_artifacts incomplete"):
                    validate_audit_certificate(_rehash(dict(pinned_live, pinned_artifacts=bad_pa)))
            for bad_meta in (123, dict(pinned_live["pinned_artifacts"][PINNED_INTAKE_FILES[0]], extra=1)):
                pa_bad = dict(pinned_live["pinned_artifacts"], **{PINNED_INTAKE_FILES[0]: bad_meta})
                with self.assertRaisesRegex(BallpackTreeError, "invalid pinned_artifacts entry"):
                    validate_audit_certificate(_rehash(dict(pinned_live, pinned_artifacts=pa_bad)))
            with self.assertRaisesRegex(BallpackTreeError, "live_source_audit"):
                validate_audit_certificate(
                    _rehash(dict(pinned_live, live_source_audit=dict(PINNED_SOURCE_AUDIT_EXPECTED, ballpack_c_limbs=2))),
                )

            # Matched pinned intake_dir positive path (kills inverted _type_strict_eq(cert, expected_live)!)
            syn_expected = {
                rel: {"sha256": sha256_file(root / rel)[0], "bytes": sha256_file(root / rel)[1]}
                for rel in PINNED_INTAKE_FILES
            }
            with mock.patch.dict(PINNED_ARTIFACTS_EXPECTED, syn_expected, clear=True):
                live_cert = run_live_audit(root)
                self.assertTrue(live_cert["overall_verified"])
                self.assertTrue(validate_audit_certificate(live_cert, intake_dir=root)["valid"])

            # CLI --standalone --json --write-cert, --standalone text, --live, and --check (valid + error paths)
            sa_cert_path = Path(td) / "sa_cert.json"
            pinned_cert_path = Path(td) / "pinned_cert.json"
            pinned_cert_path.write_text(json.dumps(pinned_live, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            def _run_cli(argv: list[str]) -> tuple[int, str, str]:
                bo, be = io.StringIO(), io.StringIO()
                old_o, old_e = sys.stdout, sys.stderr
                try:
                    sys.stdout, sys.stderr = bo, be
                    rc = audit_main(argv)
                finally:
                    sys.stdout, sys.stderr = old_o, old_e
                return rc, bo.getvalue(), be.getvalue()

            rc, out, _ = _run_cli(["--standalone", "--json", "--write-cert", str(sa_cert_path)])
            self.assertEqual(rc, 0)
            self.assertEqual(out, sa_cert_path.read_text(encoding="utf-8"))

            # Unpinned synthetic --live returns exit 1; pinned returns exit 0!
            rc, out, _ = _run_cli(["--live", "--intake-dir", str(root)])
            self.assertEqual(rc, 1)
            self.assertIn("verified=False", out)
            with mock.patch.dict(PINNED_ARTIFACTS_EXPECTED, syn_expected, clear=True):
                rc, out, _ = _run_cli(["--live", "--intake-dir", str(root)])
                self.assertEqual(rc, 0)
                self.assertIn("verified=True", out)

            rc, out, _ = _run_cli(["--check", str(pinned_cert_path)])
            self.assertEqual(rc, 0)
            self.assertTrue(json.loads(out)["valid"])

            rc, _, err = _run_cli(["--check", str(Path(td) / "nonexistent.json")])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(err)["error_code"], "CERT_IO_ERROR")

            # S15-BUG-01: non-UTF-8 bytes in --check must return exit 2 + JSON CERT_IO_ERROR (no traceback!)
            bad_utf8_p = Path(td) / "non_utf8.json"
            bad_utf8_p.write_bytes(b"\xff\xfe\x00invalid")
            rc, _, err = _run_cli(["--check", str(bad_utf8_p)])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(err)["error_code"], "CERT_IO_ERROR")

            bad_json_p = Path(td) / "bad.json"
            bad_json_p.write_text("{broken", encoding="utf-8")
            rc, _, err = _run_cli(["--check", str(bad_json_p)])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(err)["error_code"], "CERT_JSON_DECODE_ERROR")

            for bad_payload, expected_sub in (
                ('{"a": 1.5}', "forbidden JSON float"),
                ('{"a": NaN}', "forbidden JSON float"),
                ('{"a": 1, "a": 2}', "duplicate JSON key"),
                ('{"mode": "bad"}', "mode invalid"),
            ):
                bad_json_p.write_text(bad_payload, encoding="utf-8")
                rc, _, err = _run_cli(["--check", str(bad_json_p)])
                self.assertEqual(rc, 2)
                err_obj = json.loads(err)
                self.assertEqual(err_obj["error_code"], "CERT_VALIDATION_ERROR")
                self.assertIn(expected_sub, err_obj["error"])

            # Fail-closed live intake parser guards
            with self.assertRaisesRegex(BallpackTreeError, "does not exist"):
                run_live_audit(Path(td) / "no_dir")
            with self.assertRaisesRegex(BallpackTreeError, "missing pinned artifact"):
                sha256_file(Path(td) / "no_file")

            bad_root = Path(td) / "bad_intake"
            _build_synthetic_intake(bad_root)
            (bad_root / "shape" / "ballindex.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(BallpackTreeError, "missing BallIndex class"):
                run_live_audit(bad_root)
            (bad_root / "shape" / "ballindex.py").write_text("class BallIndex:\n    def table_entries(self): pass\n", encoding="utf-8")
            with self.assertRaisesRegex(BallpackTreeError, "missing unrank or table_entries"):
                run_live_audit(bad_root)
            (bad_root / "shape" / "ballindex.py").write_text("class BallIndex:\n    def unrank(self, k): pass\n", encoding="utf-8")
            with self.assertRaisesRegex(BallpackTreeError, "missing unrank or table_entries"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            (bad_root / "shape" / "ballpack.c").write_text("int x = 0;\n", encoding="utf-8")
            with self.assertRaisesRegex(BallpackTreeError, "missing #define LIMBS"):
                run_live_audit(bad_root)
            _build_synthetic_intake(bad_root)
            (bad_root / "shape" / "ballpack.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(BallpackTreeError, "missing Format class"):
                run_live_audit(bad_root)
            (bad_root / "shape" / "ballpack.py").write_text("class Format:\n    pass\n", encoding="utf-8")
            with self.assertRaisesRegex(BallpackTreeError, "missing Format.__init__"):
                run_live_audit(bad_root)


if __name__ == "__main__":
    unittest.main()
