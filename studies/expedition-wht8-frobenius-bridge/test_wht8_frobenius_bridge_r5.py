#!/usr/bin/env python3
"""
Track S21f (`AUDIT-121F`): R5 Adversarial, Mathematical, Cryptographic, Zone-B,
and Mutation-Killing Test Suite (`test_wht8_frobenius_bridge_r5.py`).

Pure stdlib, zero bare `assert`s (`-O`-safe), zero floats on trust paths,
4-mode `-I` parity verified.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import ctypes
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import wht8_frobenius_bridge_audit as audit  # noqa: E402
import wht8_frobenius_bridge_oracle as oracle  # noqa: E402

# Runtime float constructed without AST float literal or float() call
_RUNTIME_FLOAT = json.loads("1.5")


class _IntSub(int):
    pass


class _DictSub(dict):
    pass


class _ListSub(list):
    pass


class _StrSub(str):
    pass


class _EvilPath(type(Path())):
    pass


class TestTrackS21fWht8FrobeniusBridgeR5(unittest.TestCase):
    cert: dict[str, object]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cert = oracle.generate_certificate()

    def test_01_theorems_isometry_eisenstein_and_exact_witnesses(self) -> None:
        p = self.cert["payload"]
        self.assertIsInstance(p, dict)
        t1 = p["theorem1_wht8_isometry"]
        self.assertEqual(
            t1,
            {
                "h8_symmetric": True,
                "h8_gram_equals_8_i8": True,
                "butterfly_24op_matches_h8_all_basis": True,
                "exact_norm_scale_factor": 8,
            },
        )
        # Verify apply_h8_butterfly norm identity ||H_8 z||^2 == 8 ||z||^2 on all 240 E_8 roots
        e8_t = oracle.build_e8_shell_tables(4)
        e8_4_t = oracle.build_e8_4_shell_tables(4)
        self.assertEqual(e8_t["shell_counts"], (1, 240, 2160, 6720, 17520))
        self.assertTrue(e8_t["eisenstein_e4_exact"])
        self.assertTrue(e8_4_t["e4_squared_equals_e8_exact"])
        for idx in range(1, 241):
            z = oracle.unrank_e8(idx, e8_t)
            self.assertEqual(oracle.rank_e8(z, e8_t), idx)
            rot = oracle.apply_h8_butterfly(z)
            self.assertEqual(sum(x * x for x in rot), 8 * sum(x * x for x in z))
            self.assertEqual(sum(x * x for x in z), 8)
        self.assertEqual(oracle.unrank_e8(0, e8_t), (0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(oracle.rank_e8((0, 0, 0, 0, 0, 0, 0, 0), e8_t), 0)

        # Invalid apply_h8_butterfly inputs (including _ListSub of length 8 killing SUBCOND_DEL)
        for bad_v in (123, (1, 2), _ListSub([0] * 8), [True] + [0] * 7):
            with self.assertRaises(oracle.Wht8BridgeError):
                oracle.apply_h8_butterfly(bad_v)  # type: ignore[arg-type]

        # E_8^4 roundtrips across m=0, m=1, m=4 including zero blocks and boundary offsets
        z0 = (0, 0, 0, 0, 0, 0, 0, 0)
        z1 = (2, 2, 0, 0, 0, 0, 0, 0)
        z1h = (1, 1, 1, 1, 1, 1, 1, 1)
        z2 = (2, 2, 2, 2, 0, 0, 0, 0)
        for quad in ((z0, z0, z0, z0), (z0, z1, z0, z0), (z1, z1h, z0, z2), (z2, z0, z1, z1h)):
            k4 = oracle.rank_e8_4(quad, e8_t, e8_4_t)
            self.assertEqual(oracle.unrank_e8_4(k4, e8_t, e8_4_t), quad)
            m_got, _ = oracle.shell_from_index(k4, e8_4_t["cum_counts"])
            self.assertEqual(m_got, sum(sum(c * c for c in blk) // 8 for blk in quad))

        # Exact boundary m == max_shell passes validate_doubled_e8 (kills m > max_shell -> >=)
        _, _, m_exact = oracle.validate_doubled_e8((4, 4, 0, 0, 0, 0, 0, 0), 4)
        self.assertEqual(m_exact, 4)

        # Validation rejections for validate_doubled_e8, rank_e8_4, shell_from_index
        for bad_z, code in (
            (123, "INVALID_E8_VEC"),
            (_ListSub([0] * 8), "INVALID_E8_VEC"),
            ((0, 0), "INVALID_E8_VEC"),
            ((1, 2, 0, 0, 0, 0, 0, 0), "INVALID_E8_COSET"),
            ((2, 0, 0, 0, 0, 0, 0, 0), "INVALID_E8_PARITY"),
            ((1, 1, 1, 1, 1, 1, 1, -1), "INVALID_E8_PARITY"),
            ((6, 2, 0, 0, 0, 0, 0, 0), "E8_SHELL_EXCEEDED"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_z:
                oracle.validate_doubled_e8(bad_z, 4)
            self.assertEqual(ctx_z.exception.code, code)

        for bad_q4, code in (
            (123, "INVALID_E8_4_BLOCK"),
            (_ListSub([z0, z0, z0, z0]), "INVALID_E8_4_BLOCK"),
            ((z0, z0, z0), "INVALID_E8_4_BLOCK"),
            ((z2, z2, z1, z0), "E8_4_SHELL_EXCEEDED"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_q4:
                oracle.rank_e8_4(bad_q4, e8_t, e8_4_t)  # type: ignore[arg-type]
            self.assertEqual(ctx_q4.exception.code, code)

        with self.assertRaises(oracle.Wht8BridgeError) as ctx_oob:
            oracle.shell_from_index(e8_t["total_points"], e8_t["cum_counts"])
        self.assertEqual(ctx_oob.exception.code, "SHELL_INDEX_OUT_OF_RANGE")

        # Low-level helper guards: shell_from_index, validate_doubled_e8, rank_e8/unrank_e8/rank_e8_4/unrank_e8_4
        for bad_cum, exp_c in (
            ((), "INVALID_SHELL_TABLE"),
            ("123", "INVALID_SHELL_TABLE"),
            (_ListSub([1]), "INVALID_SHELL_TABLE"),
            ((True,), "INVALID_INT_TYPE"),
            ((0,), "INT_BELOW_MIN"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_sc:
                oracle.shell_from_index(0, bad_cum)  # type: ignore[arg-type]
            self.assertEqual(ctx_sc.exception.code, exp_c)
        for bad_ms, exp_c in ((True, "INVALID_INT_TYPE"), (-1, "INT_BELOW_MIN"), (oracle.MAX_CERT_SHELL + 1, "INT_ABOVE_MAX")):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ms:
                oracle.validate_doubled_e8(z0, bad_ms)  # type: ignore[arg-type]
            self.assertEqual(ctx_ms.exception.code, exp_c)
        for bad_e8t in ([], {}, _DictSub(e8_t)):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_rt:
                oracle.rank_e8(z0, bad_e8t)  # type: ignore[arg-type]
            self.assertEqual(ctx_rt.exception.code, "INVALID_E8_TABLES")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ut:
                oracle.unrank_e8(0, bad_e8t)  # type: ignore[arg-type]
            self.assertEqual(ctx_ut.exception.code, "INVALID_E8_TABLES")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_r4t:
                oracle.rank_e8_4((z0, z0, z0, z0), bad_e8t, e8_4_t)  # type: ignore[arg-type]
            self.assertEqual(ctx_r4t.exception.code, "INVALID_E8_TABLES")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_u4t:
                oracle.unrank_e8_4(0, bad_e8t, e8_4_t)  # type: ignore[arg-type]
            self.assertEqual(ctx_u4t.exception.code, "INVALID_E8_TABLES")
        for bad_e84t in ([], {}, _DictSub(e8_4_t)):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_r4b:
                oracle.rank_e8_4((z0, z0, z0, z0), e8_t, bad_e84t)  # type: ignore[arg-type]
            self.assertEqual(ctx_r4b.exception.code, "INVALID_E8_4_TABLES")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_u4b:
                oracle.unrank_e8_4(0, e8_t, bad_e84t)  # type: ignore[arg-type]
            self.assertEqual(ctx_u4b.exception.code, "INVALID_E8_4_TABLES")
        bad_mcap4 = dict(e8_4_t, max_shell=True)
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_mc4:
            oracle.rank_e8_4((z0, z0, z0, z0), e8_t, bad_mcap4)
        self.assertEqual(ctx_mc4.exception.code, "INVALID_INT_TYPE")

        # Certificate theorem payload exact numbers
        t2 = p["theorem2_eisenstein_shell_identities"]
        self.assertEqual(t2["e8_4_88bit_max_shell"], 48)
        self.assertEqual(t2["e8_4_88bit_bsearch_max_cmp"], 6)
        self.assertEqual(t2["e8_4_136bit_max_shell"], 391)
        self.assertEqual(t2["e8_4_136bit_bsearch_max_cmp"], 9)
        self.assertEqual(t2["op_count_per_32w_superblock"]["speedup_ratio_88bit"], "191/9")
        self.assertEqual(t2["op_count_per_32w_superblock"]["speedup_ratio_136bit"], "191/12")

        t34 = p["theorem3_4_zero_unpack_and_tier2_exact_parity"]
        self.assertTrue(t34["rank1_honest_exact_match"])
        self.assertTrue(t34["rank2_exact_match"])
        self.assertTrue(t34["e8_4_superblock_exact_match"])
        self.assertTrue(t34["e8_4_superblock_roundtrip_exact"])
        self.assertTrue(t34["rank2_tight_tier2"]["gram_diag_matches_tier1"])
        self.assertEqual(t34["rank1_honest_tier1"]["tier_used"], "TIER1_ZERO_UNPACK_EXACT_RANK1")
        self.assertEqual(t34["rank2_fast_accept_tier1"]["tier_used"], "TIER1_ZERO_UNPACK_FAST_ACCEPT")
        self.assertEqual(t34["rank2_tight_tier2"]["tier_used"], "TIER2_WHT8_BYPASSED_GRAM_BAREISS")
        self.assertEqual(t34["rank2_tight_tier2"]["wht8_butterflies_run"], 0)
        self.assertEqual(t34["e8_4_superblock_tier1"]["coordinates_unranked"], 0)

        t5 = p["theorem5_cross_org_attack_closures"]
        for key in (
            "wd40_05_radix_scale_exp2_relabel_32_to_48",
            "warp13_b_times_8_scale_forgery",
            "warp13_alpha_32_metadata_inflation",
            "shell_index_bump_m1_to_m3",
            "rank2_bareiss_1ulp_spectral_understatement",
        ):
            self.assertTrue(t5[key]["blocked"], key)

    def test_02_bareiss_psd_and_bitstream_stage2_edge_cases(self) -> None:
        # Bareiss exact PSD & indefinite edge cases (including pivoting, rank deficiency, zero-diag, prev_piv > 1)
        self.assertTrue(oracle.bareiss_is_psd([]))
        self.assertTrue(oracle.bareiss_is_psd([[0]]))
        self.assertTrue(oracle.bareiss_is_psd([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        self.assertFalse(oracle.bareiss_is_psd([[-1]]))
        self.assertTrue(oracle.bareiss_is_psd([[0, 0], [0, 1]]))
        self.assertFalse(oracle.bareiss_is_psd([[0, 1], [1, 0]]))
        self.assertTrue(oracle.bareiss_is_psd([[1, 1, 1], [1, 1, 1], [1, 1, 2]]))
        self.assertFalse(oracle.bareiss_is_psd([[1, 1, 2], [1, 1, 1], [2, 1, 2]]))
        self.assertFalse(oracle.bareiss_is_psd([[1, 1, 2], [1, 1, 1], [2, 1, 4]]))
        self.assertTrue(oracle.bareiss_is_psd([[2, 1, 1], [1, 2, 1], [1, 1, 2]]))
        for bad_m, code in (
            ("not_mat", "INVALID_MATRIX"),
            ((), "INVALID_MATRIX"),
            ([[1, 2]], "INVALID_MATRIX"),
            ([(1,)], "INVALID_MATRIX"),
            (["bad"], "INVALID_MATRIX"),
            ([[1, 2], [3, 1]], "NON_SYMMETRIC_MATRIX"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_m:
                oracle.bareiss_is_psd(bad_m)  # type: ignore[arg-type]
            self.assertEqual(ctx_m.exception.code, code)

        # pack_e8had_adapter validation boundaries (incl. S21F-BUG-03 b_scales=[1]/a_scales=[1]/[None])
        for bad_kw, code in (
            (dict(b_scales=[[1]], b_indices=[[0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1, superblock_e8_4=1), "INVALID_FLAG"),
            (dict(b_scales=[], b_indices=[], a_scales=[], a_indices=[], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_RANK"),
            (dict(b_scales=([1],), b_indices=[[0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_RANK"),
            (dict(b_scales=[[1]], b_indices=(), a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_PACK_INPUT"),
            (dict(b_scales=[[1]], b_indices=[[0]], a_scales=(), a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_PACK_INPUT"),
            (dict(b_scales=[[1]], b_indices=[[0]], a_scales=[[1]], a_indices=(), scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_PACK_INPUT"),
            (dict(b_scales=[[1]], b_indices=[[0], [0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "RANK_MISMATCH"),
            (dict(b_scales=[[1]], b_indices=[[0]], a_scales=[[1], [1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "RANK_MISMATCH"),
            (dict(b_scales=[[1]], b_indices=[[0]], a_scales=[[1]], a_indices=[[0], [0]], scale_exp2=0, max_shell=2, alpha_num=1), "RANK_MISMATCH"),
            (dict(b_scales=[1], b_indices=[[0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_ROW"),
            (dict(b_scales=[[1]], b_indices=[[0]], a_scales=[1], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_ROW"),
            (dict(b_scales=[None], b_indices=[[0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_ROW"),
            (dict(b_scales=[(1,)], b_indices=[[0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_ROW"),
            (dict(b_scales=[[1], (1,)], b_indices=[[0], [0]], a_scales=[[1], [1]], a_indices=[[0], [0]], scale_exp2=0, max_shell=2, alpha_num=2), "INVALID_ROW"),
            (dict(b_scales=[[1]], b_indices=[(0,)], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "INVALID_ROW"),
            (dict(b_scales=[[1], [1, 1]], b_indices=[[0], [0]], a_scales=[[1], [1]], a_indices=[[0], [0]], scale_exp2=0, max_shell=2, alpha_num=1), "BLOCK_COUNT_MISMATCH"),
            (dict(b_scales=[[1]], b_indices=[[0, 0]], a_scales=[[1]], a_indices=[[0]], scale_exp2=0, max_shell=2, alpha_num=1), "BLOCK_COUNT_MISMATCH"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_pk:
                oracle.pack_e8had_adapter(**bad_kw)  # type: ignore[arg-type]
            self.assertEqual(ctx_pk.exception.code, code)

        # parse_e8had_bitstream fail-closed header/length/alpha/flag checks
        with self.assertRaises(oracle.Wht8BridgeError) as c_bt:
            oracle.parse_e8had_bitstream("bytes")  # type: ignore[arg-type]
        self.assertEqual(c_bt.exception.code, "INVALID_BITSTREAM_TYPE")
        with self.assertRaises(oracle.Wht8BridgeError) as c_tr:
            oracle.parse_e8had_bitstream(b"SQE8")
        self.assertEqual(c_tr.exception.code, "TRUNCATED_HEADER")
        with self.assertRaises(oracle.Wht8BridgeError) as c_big:
            oracle.parse_e8had_bitstream(b"\x00" * (oracle.MAX_BITSTREAM_BYTES + 1))
        self.assertEqual(c_big.exception.code, "BITSTREAM_TOO_LARGE")
        with self.assertRaises(oracle.Wht8BridgeError) as c_exact_lim:
            oracle.parse_e8had_bitstream(b"BADMAGIC" + b"\x00" * (oracle.MAX_BITSTREAM_BYTES - 8))
        self.assertEqual(c_exact_lim.exception.code, "INVALID_MAGIC")

        def _hdr(magic=oracle.BITSTREAM_MAGIC, ver=1, flags=0, rank=1, do=1, di=1, qs=0, ms=2, ib=2, sb=1, an=1, ad=1, extra=b""):
            return struct.pack(oracle._HEADER_FMT, magic, ver, flags, rank, do, di, qs, ms, ib, sb, an, ad) + extra

        for raw_bad, code in (
            (_hdr(magic=b"BADMAGIC"), "INVALID_MAGIC"),
            (_hdr(ver=2), "INVALID_VERSION"),
            (_hdr(flags=2), "INVALID_FLAGS"),
            (_hdr(rank=0), "INVALID_RANK"),
            (_hdr(rank=oracle.MAX_RANK + 1), "INVALID_RANK"),
            (_hdr(do=0), "INVALID_BLOCK_COUNT"),
            (_hdr(do=oracle.MAX_BLOCKS_PER_VEC + 1), "INVALID_BLOCK_COUNT"),
            (_hdr(di=0), "INVALID_BLOCK_COUNT"),
            (_hdr(di=oracle.MAX_BLOCKS_PER_VEC + 1), "INVALID_BLOCK_COUNT"),
            (_hdr(qs=oracle.MAX_SCALE_EXP2 + 1), "INVALID_SCALE_EXP2"),
            (_hdr(ms=0), "INVALID_MAX_SHELL"),
            (_hdr(ms=oracle.MAX_CERT_SHELL + 1), "INVALID_MAX_SHELL"),
            (_hdr(ib=0), "INVALID_RECORD_WIDTH"),
            (_hdr(ib=17), "INVALID_RECORD_WIDTH"),
            (_hdr(sb=0), "INVALID_RECORD_WIDTH"),
            (_hdr(sb=9), "INVALID_RECORD_WIDTH"),
            (_hdr(an=1, ad=0), "NON_UNIT_ALPHA_SCALE"),
            (_hdr(an=0, ad=0), "NON_UNIT_ALPHA_SCALE"),
            (_hdr(an=2, ad=1), "NON_UNIT_ALPHA_SCALE"),
            (_hdr(extra=b"\x00" * 5), "BITSTREAM_LENGTH_MISMATCH"),
            (_hdr(extra=b"\x00" * 7), "BITSTREAM_LENGTH_MISMATCH"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_h:
                oracle.parse_e8had_bitstream(raw_bad)
            self.assertEqual(ctx_h.exception.code, code)

        # Boundary checks: exact MAX_RANK, MAX_BLOCKS_PER_VEC, MAX_SCALE_EXP2, MAX_CERT_SHELL, ib=16, sb=8, exact size
        valid_edge = _hdr(rank=oracle.MAX_RANK, do=1, di=1, qs=oracle.MAX_SCALE_EXP2, ms=oracle.MAX_CERT_SHELL, ib=16, sb=8, an=oracle.MAX_RANK, ad=1, extra=b"\x00" * (oracle.MAX_RANK * 2 * 24))
        self.assertEqual(oracle.parse_e8had_bitstream(valid_edge)["rank"], oracle.MAX_RANK)
        valid_edge2 = _hdr(rank=1, do=oracle.MAX_BLOCKS_PER_VEC, di=oracle.MAX_BLOCKS_PER_VEC, qs=0, ms=1, ib=1, sb=1, an=1, ad=1, extra=b"\x00" * (2 * oracle.MAX_BLOCKS_PER_VEC * 2))
        self.assertEqual(oracle.parse_e8had_bitstream(valid_edge2)["d_out_blocks"], oracle.MAX_BLOCKS_PER_VEC)

        # Stage-2 verifier branches (cert validation, Tier 1 exact rank-1 spectral fail, Tier 2 frob fail, E_8^4 Tier 2)
        e8_t = oracle.build_e8_shell_tables(4)
        e8_4_t = oracle.build_e8_4_shell_tables(4)
        z1 = (2, 2, 0, 0, 0, 0, 0, 0)
        z0 = (0, 0, 0, 0, 0, 0, 0, 0)
        k1 = oracle.rank_e8(z1, e8_t)
        raw_r1 = oracle.pack_e8had_adapter(
            b_scales=[[1]], b_indices=[[k1]], a_scales=[[1]], a_indices=[[k1]],
            scale_exp2=0, max_shell=4, alpha_num=1,
        )
        # ||B||^2=2, ||A||^2=2 -> ||BA||_F^2 = 4, sigma_max^2 = 4
        sha_r1 = hashlib.sha256(raw_r1).hexdigest()
        base_cert = {
            "adapter_sha256": sha_r1,
            "radix_scale_exp2": 0,
            "frobenius_sq_scaled": 4,
            "spectral_norm_sq_scaled": 4,
            "max_frobenius_sq_scaled": 4,
            "max_spectral_norm_sq_scaled": 4,
        }
        self.assertTrue(oracle.verify_packed_e8had_stage2(raw_r1, base_cert)["admitted"])
        # Force Tier 2 on E_8^4 super-block AND rank-1!
        k4 = oracle.rank_e8_4((z1, z0, z0, z0), e8_t, e8_4_t)
        raw_sb = oracle.pack_e8had_adapter(
            b_scales=[[1]], b_indices=[[k4]], a_scales=[[1]], a_indices=[[k4]],
            scale_exp2=0, max_shell=4, alpha_num=1, superblock_e8_4=True,
        )
        cert_sb = dict(base_cert, adapter_sha256=hashlib.sha256(raw_sb).hexdigest())
        res_sb_t2 = oracle.verify_packed_e8had_stage2(raw_sb, cert_sb, force_exact_tier2=True)
        self.assertEqual(res_sb_t2["tier_used"], "TIER2_WHT8_BYPASSED_GRAM_BAREISS")
        self.assertEqual(res_sb_t2["coordinates_unranked"], 64)
        self.assertTrue(res_sb_t2["gram_diag_matches_tier1"])

        # Rank-1 spectral fail (frob=4 ok, spec=3 < 4)
        with self.assertRaises(oracle.Wht8BridgeError) as c_sp1:
            oracle.verify_packed_e8had_stage2(raw_r1, dict(base_cert, spectral_norm_sq_scaled=3))
        self.assertEqual(c_sp1.exception.code, "SPECTRAL_EXCEEDS_CERTIFIED")
        # Tier-2 frob fail
        with self.assertRaises(oracle.Wht8BridgeError) as c_f2:
            oracle.verify_packed_e8had_stage2(raw_r1, dict(base_cert, frobenius_sq_scaled=3, spectral_norm_sq_scaled=3), force_exact_tier2=True)
        self.assertEqual(c_f2.exception.code, "FROBENIUS_EXCEEDS_CERTIFIED")

        # Rank-2 cs_rank < cs_prod path (kills cs_prod <= cs_rank mutant!)
        # B diags = [2, 0], A diags = [0, 2] -> cs_prod = 4, cs_rank = 2*(0+0) = 0 -> zero_unpack_ub = 0!
        raw_disj = oracle.pack_e8had_adapter(
            b_scales=[[1], [0]], b_indices=[[k1], [0]], a_scales=[[0], [1]], a_indices=[[0], [k1]],
            scale_exp2=0, max_shell=4, alpha_num=2,
        )
        cert_disj = {
            "adapter_sha256": hashlib.sha256(raw_disj).hexdigest(),
            "radix_scale_exp2": 0,
            "frobenius_sq_scaled": 0,
            "spectral_norm_sq_scaled": 0,
            "max_frobenius_sq_scaled": 0,
            "max_spectral_norm_sq_scaled": 0,
        }
        res_disj = oracle.verify_packed_e8had_stage2(raw_disj, cert_disj)
        self.assertEqual(res_disj["tier_used"], "TIER1_ZERO_UNPACK_FAST_ACCEPT")
        self.assertEqual(res_disj["zero_unpack_ub_q"], "0")
        # Rank-2 fast-accept rejected when only frob passes but spec fails (kills BoolOp and->or on fast-accept!)
        raw_r2_eq = oracle.pack_e8had_adapter(
            b_scales=[[1], [1]], b_indices=[[k1], [k1]], a_scales=[[1], [1]], a_indices=[[k1], [k1]],
            scale_exp2=0, max_shell=4, alpha_num=2,
        )
        # B=[[v, v]], A=[[v],[v]]^T -> BA = 2 v v^T -> ||BA||_F^2 = 16, sigma_max^2 = 16, ub = 16
        cert_r2_spec_tight = {
            "adapter_sha256": hashlib.sha256(raw_r2_eq).hexdigest(),
            "radix_scale_exp2": 0,
            "frobenius_sq_scaled": 16,
            "spectral_norm_sq_scaled": 15,
            "max_frobenius_sq_scaled": 20,
            "max_spectral_norm_sq_scaled": 20,
        }
        with self.assertRaises(oracle.Wht8BridgeError) as c_r2s:
            oracle.verify_packed_e8had_stage2(raw_r2_eq, cert_r2_spec_tight)
        self.assertEqual(c_r2s.exception.code, "SPECTRAL_EXCEEDS_CERTIFIED")

        # Stage-2 cert bad inputs
        with self.assertRaises(oracle.Wht8BridgeError) as c_fb:
            oracle.verify_packed_e8had_stage2(raw_r1, base_cert, force_exact_tier2=1)  # type: ignore[arg-type]
        self.assertEqual(c_fb.exception.code, "INVALID_BOOL_TYPE")
        for bad_c, code in (
            ([], "INVALID_STAGE2_CERT"),
            ({}, "INVALID_STAGE2_KEYS"),
            (dict(base_cert, spectral_norm_sq_scaled=5, max_spectral_norm_sq_scaled=5), "SPECTRAL_EXCEEDS_FROBENIUS"),
            (dict(base_cert, frobenius_sq_scaled=5, max_frobenius_sq_scaled=4), "FROBENIUS_EXCEEDS_POLICY"),
            (dict(base_cert, spectral_norm_sq_scaled=4, max_spectral_norm_sq_scaled=3), "SPECTRAL_EXCEEDS_POLICY"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_sc:
                oracle.verify_packed_e8had_stage2(raw_r1, bad_c)  # type: ignore[arg-type]
            self.assertEqual(ctx_sc.exception.code, code)

    def test_03_zone_b_strict_io_and_helpers(self) -> None:
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
            with self.assertRaises(oracle.Wht8BridgeError):
                bad_call()

        good_hex = "0123456789abcdef" * 4
        other_hex = "fedcba9876543210" * 4
        self.assertTrue(oracle.constant_time_hex64_equal(good_hex, good_hex))
        self.assertFalse(oracle.constant_time_hex64_equal(good_hex, other_hex))
        for bad_hex in (123, "a" * 63, "a" * 65, "A" * 64, "\u00e9" * 64, chr(0xD800) * 64):
            with self.assertRaises(oracle.Wht8BridgeError):
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
            with self.assertRaises(oracle.Wht8BridgeError):
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
        cyc_d: dict = {}
        cyc_d["self"] = cyc_d
        cyc_l: list = []
        cyc_l.append(cyc_l)
        for cyc_obj in (cyc_d, cyc_l):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_cyc:
                oracle.validate_json_value(cyc_obj)
            self.assertEqual(ctx_cyc.exception.code, "JSON_CYCLE")
        shared = {"x": 1}
        oracle.validate_json_value({"a": shared, "b": shared})
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
            with self.assertRaises(oracle.Wht8BridgeError):
                oracle.validate_json_value(bad_val)
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_4300:
            oracle.canonical_json_bytes({"x": 10**4300})
        self.assertEqual(ctx_4300.exception.code, "INT_BIT_LIMIT")
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_flt:
            oracle.validate_json_value({"x": _RUNTIME_FLOAT})
        self.assertEqual(ctx_flt.exception.code, "FLOAT_FORBIDDEN")

        self.assertEqual(oracle._parse_capped_int_token("-42"), -42)
        self.assertEqual(oracle._parse_capped_int_token(str(max_2048)), max_2048)
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_d620:
            oracle._parse_capped_int_token("1" + ("0" * (oracle.MAX_INT_DIGITS - 1)))
        self.assertEqual(ctx_d620.exception.code, "INT_BIT_LIMIT")
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_d621:
            oracle._parse_capped_int_token("9" * (oracle.MAX_INT_DIGITS + 1))
        self.assertEqual(ctx_d621.exception.code, "INT_DIGIT_LIMIT")
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_b2049:
            oracle._parse_capped_int_token(str(over_2048))
        self.assertEqual(ctx_b2049.exception.code, "INT_BIT_LIMIT")

        self.assertEqual(oracle.parse_strict_json_text('{"k":1}'), {"k": 1})
        self.assertEqual(oracle.parse_strict_json_text("[1,2]", require_object=False), [1, 2])
        for bad_ro in (0, 1, "yes", None):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ro:
                oracle.parse_strict_json_text("[1]", require_object=bad_ro)  # type: ignore[arg-type]
            self.assertEqual(ctx_ro.exception.code, "INVALID_BOOL_TYPE")
        exact_max_str = '{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}'
        self.assertEqual(len(oracle.parse_strict_json_text(exact_max_str)["a"]), oracle.MAX_JSON_BYTES - 8)
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_empty:
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
            with self.assertRaises(oracle.Wht8BridgeError):
                oracle.parse_strict_json_text(bad_txt)  # type: ignore[arg-type]

    def _resign(self, cert: dict) -> dict:
        c = copy.deepcopy(cert)
        p_sha = oracle.canonical_sha256(c["payload"])
        c["payload_sha256"] = p_sha
        body = {"schema": c["schema"], "payload_sha256": p_sha, "payload": c["payload"]}
        c["certificate_sha256"] = oracle.canonical_sha256(body)
        return c

    def test_04_auditor_symlink_pwd_secondary_and_cli_gates(self) -> None:
        cert_path = _HERE / "wht8_frobenius_bridge_certificate.json"
        res = audit.verify_certificate_file(cert_path)
        self.assertTrue(res["valid"])
        self.assertTrue(res["secondary_audit"]["secondary_audit_passed"])
        self.assertEqual(res["payload_sha256"], self.cert["payload_sha256"])
        self.assertEqual(res["certificate_sha256"], self.cert["certificate_sha256"])

        good_copy = json.loads(oracle.canonical_json_bytes(self.cert).decode("ascii"))
        self.assertTrue(audit.verify_certificate_dict(good_copy, full_replay=False)["valid"])
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_frb:
            audit.verify_certificate_dict(good_copy, full_replay=1)  # type: ignore[arg-type]
        self.assertEqual(ctx_frb.exception.code, "INVALID_BOOL_TYPE")

        forged_hollow = self._resign(dict(good_copy, payload={"forged": 1}))
        bad_meta = copy.deepcopy(good_copy)
        bad_meta["payload"]["track"] = "WRONG"
        bad_thm = copy.deepcopy(good_copy)
        bad_thm["payload"]["theorem1_wht8_isometry"] = []
        bool_int = copy.deepcopy(good_copy)
        bool_int["payload"]["theorem1_wht8_isometry"]["exact_norm_scale_factor"] = True
        bool_list = copy.deepcopy(good_copy)
        bool_list["payload"]["theorem2_eisenstein_shell_identities"]["e8_shell_counts_m0_to_m8"][0] = True

        for bad_cert, want_code, fr in (
            ([], "INVALID_CERT_TYPE", False),
            ({k: v for k, v in good_copy.items() if k != "schema"}, "INVALID_CERT_KEYS", False),
            (dict(good_copy, schema=123), "INVALID_SCHEMA", False),
            (dict(good_copy, schema=_StrSub(good_copy["schema"])), "UNSUPPORTED_JSON_TYPE", False),
            (dict(good_copy, schema="wrong.schema"), "INVALID_SCHEMA", False),
            (dict(good_copy, payload=[1, 2]), "INVALID_PAYLOAD_TYPE", False),
            (dict(good_copy, payload_sha256="0" * 64), "PAYLOAD_SHA_MISMATCH", False),
            (dict(good_copy, certificate_sha256="0" * 64), "CERT_SHA_MISMATCH", False),
            (forged_hollow, "INVALID_PAYLOAD_KEYS", False),
            (self._resign(bad_meta), "INVALID_PAYLOAD_META", False),
            (self._resign(bad_thm), "INVALID_THEOREM_TYPE", False),
            (self._resign(bool_int), "BOOL_IN_NON_BOOL_FIELD", False),
            (self._resign(bool_list), "BOOL_IN_LIST", False),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_c:
                audit.verify_certificate_dict(bad_cert, full_replay=fr)
            self.assertEqual(ctx_c.exception.code, want_code)

        # Verify full_replay=True detects oracle recomputation divergence even when secondary audit passes
        with mock.patch.object(audit, "generate_certificate", return_value=dict(good_copy, payload_sha256="0" * 64)):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_rep:
                audit.verify_certificate_dict(good_copy, full_replay=True)
            self.assertEqual(ctx_rep.exception.code, "RECOMPUTE_PAYLOAD_MISMATCH")

        # Exhaustive secondary_audit_payload tamper matrix (schema-safe + bool-for-int!)
        base_p = good_copy["payload"]
        self.assertTrue(audit.secondary_audit_payload(base_p)["secondary_audit_passed"])
        for bad_sp, want_code in (
            ([], "INVALID_PAYLOAD_TYPE"),
            ({}, "AUDIT_SCHEMA_ERROR"),
            ({"theorem1_wht8_isometry": None}, "AUDIT_SCHEMA_ERROR"),
        ):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_sp:
                audit.secondary_audit_payload(bad_sp)
            self.assertEqual(ctx_sp.exception.code, want_code)

        def _sec_tamper(code: str, fn) -> None:
            p = copy.deepcopy(base_p)
            fn(p)
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_t:
                audit.secondary_audit_payload(p)
            self.assertEqual(ctx_t.exception.code, code)

        _sec_tamper("AUDIT_T1_ISOMETRY", lambda p: p["theorem1_wht8_isometry"].__setitem__("h8_symmetric", False))
        _sec_tamper("INVALID_INT_TYPE", lambda p: p["theorem1_wht8_isometry"].__setitem__("exact_norm_scale_factor", True))
        _sec_tamper("AUDIT_T2_EISENSTEIN", lambda p: p["theorem2_eisenstein_shell_identities"].__setitem__("e8_4_88bit_max_shell", 47))
        _sec_tamper("AUDIT_T2_EISENSTEIN", lambda p: p["theorem2_eisenstein_shell_identities"].__setitem__("e8_4_88bit_codebook_size", "0"))
        _sec_tamper("INVALID_INT_TYPE", lambda p: p["theorem2_eisenstein_shell_identities"]["e8_shell_counts_m0_to_m8"].__setitem__(0, True))
        _sec_tamper("AUDIT_T2_OPS", lambda p: p["theorem2_eisenstein_shell_identities"]["op_count_per_32w_superblock"].__setitem__("zero_unpack_88bit_ops", 8))
        _sec_tamper("AUDIT_T34_RANK1", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank1_honest_tier1"].__setitem__("coordinates_unranked", 1))
        _sec_tamper("AUDIT_T34_RANK1", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank1_honest_tier1"].__setitem__("admitted", False))
        _sec_tamper("INVALID_DIGEST_LEN", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank1_honest_tier1"].__setitem__("adapter_sha256", "not_hex"))
        _sec_tamper("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank2_tight_tier2"].__setitem__("wht8_butterflies_run", 1))
        _sec_tamper("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank2_tight_tier2"].__setitem__("adapter_sha256", "0" * 64))
        _sec_tamper("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"].__setitem__("rank2_exact_singular_values_sq_q", []))
        _sec_tamper("AUDIT_T34_SUPERBLOCK", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["e8_4_superblock_tier1"].__setitem__("exact_ba_frob_sq_q", "2"))
        _sec_tamper("AUDIT_T34_SUPERBLOCK", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["e8_4_superblock_tier1"].__setitem__("admitted", False))
        _sec_tamper("AUDIT_T5_ATTACKS", lambda p: p["theorem5_cross_org_attack_closures"]["shell_index_bump_m1_to_m3"].__setitem__("blocked", False))
        _sec_tamper("AUDIT_T5_ATTACKS", lambda p: p["theorem5_cross_org_attack_closures"]["wd40_05_radix_scale_exp2_relabel_32_to_48"].__setitem__("rejection_code", "FAKE"))

        # Re-signed full_replay=False contradiction, pinned adapter_sha256 & extra-nested-key tampers MUST fail closed
        inflated_sha = "376ebb08" + ("0" * 56)
        for want_c, mutate_fn in (
            ("AUDIT_T1_ISOMETRY", lambda p: p["theorem1_wht8_isometry"].__setitem__("extra_key", 1)),
            ("AUDIT_T2_EISENSTEIN", lambda p: p["theorem2_eisenstein_shell_identities"].__setitem__("extra_key", 1)),
            ("AUDIT_T2_EISENSTEIN", lambda p: p["theorem2_eisenstein_shell_identities"].__setitem__("e8_4_136bit_codebook_size", "0")),
            ("AUDIT_T2_OPS", lambda p: p["theorem2_eisenstein_shell_identities"]["op_count_per_32w_superblock"].__setitem__("extra_key", 1)),
            ("AUDIT_T34_RANK1", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"].__setitem__("extra_key", 1)),
            ("AUDIT_T34_RANK1", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank1_honest_tier1"].__setitem__("extra_key", 1)),
            ("AUDIT_T34_RANK1", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank1_honest_tier1"].__setitem__("admitted", False)),
            ("AUDIT_T34_RANK1", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank1_honest_tier1"].__setitem__("adapter_sha256", inflated_sha)),
            ("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"].__setitem__("rank2_exact_singular_values_sq_q", [])),
            ("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank2_tight_tier2"].__setitem__("extra_key", 1)),
            ("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank2_tight_tier2"].__setitem__("adapter_sha256", inflated_sha)),
            ("AUDIT_T34_RANK2", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["rank2_fast_accept_tier1"].__setitem__("adapter_sha256", inflated_sha)),
            ("AUDIT_T34_SUPERBLOCK", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["e8_4_superblock_tier1"].__setitem__("extra_key", 1)),
            ("AUDIT_T34_SUPERBLOCK", lambda p: p["theorem3_4_zero_unpack_and_tier2_exact_parity"]["e8_4_superblock_tier1"].__setitem__("adapter_sha256", inflated_sha)),
            ("AUDIT_T5_ATTACKS", lambda p: p["theorem5_cross_org_attack_closures"].__setitem__("extra_key", 1)),
            ("AUDIT_T5_ATTACKS", lambda p: p["theorem5_cross_org_attack_closures"]["wd40_05_radix_scale_exp2_relabel_32_to_48"].__setitem__("extra_key", 1)),
            ("AUDIT_T5_ATTACKS", lambda p: p["theorem5_cross_org_attack_closures"]["rank2_bareiss_1ulp_spectral_understatement"].__setitem__("rejection_code", "FAKE")),
        ):
            tc = copy.deepcopy(good_copy)
            mutate_fn(tc["payload"])
            tc["payload_sha256"] = oracle.canonical_sha256(tc["payload"])
            tc["certificate_sha256"] = oracle.canonical_sha256(
                {"schema": tc["schema"], "payload_sha256": tc["payload_sha256"], "payload": tc["payload"]}
            )
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_fr:
                audit.verify_certificate_dict(tc, full_replay=False)
            self.assertEqual(ctx_fr.exception.code, want_c)

        for bad_p in ("not_a_path", _EvilPath("c.json"), Path("bad" + chr(0xD800)), Path("bad\x00path")):
            with self.assertRaises(oracle.Wht8BridgeError):
                audit.require_no_symlink_path(bad_p)  # type: ignore[arg-type]
        for dir_p in (Path("/"), Path("."), Path("..")):
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_dp:
                audit.require_no_symlink_path(dir_p)
            self.assertEqual(ctx_dp.exception.code, "NON_REGULAR_FILE")

        with self.assertRaises(oracle.Wht8BridgeError) as ctx_mb:
            audit.read_regular_bytes_capped(cert_path, max_bytes=1 << 100)
        self.assertEqual(ctx_mb.exception.code, "INT_ABOVE_MAX")
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_mb2:
            audit.read_regular_bytes_capped(cert_path, max_bytes=oracle.HARD_MAX_BYTES_CAP + 1)
        self.assertEqual(ctx_mb2.exception.code, "INT_ABOVE_MAX")
        with self.assertRaises(oracle.Wht8BridgeError) as ctx_nb:
            audit.write_regular_file_bytes(Path("/tmp/unused_s21f"), "not_bytes")  # type: ignore[arg-type]
        self.assertEqual(ctx_nb.exception.code, "INVALID_WRITE_DATA")

        # S21F-BUG-01 live /proc/kallsyms st_size==0 streaming cap check (when present)
        kallsyms = Path("/proc/kallsyms")
        if kallsyms.exists():
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ks:
                audit.read_regular_bytes_capped(kallsyms, max_bytes=1024)
            self.assertIn(ctx_ks.exception.code, ("FILE_TOO_LARGE", "FILE_READ_ERROR", "NON_REGULAR_FILE"))

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            real_dir = tmp / "real"
            real_dir.mkdir(mode=0o700)
            real_sub = real_dir / "sub"
            real_sub.mkdir(mode=0o700)
            other_dir = tmp / "other"
            other_dir.mkdir(mode=0o700)
            real_file = real_dir / "c.json"
            real_file.write_bytes(oracle.canonical_json_bytes(self.cert))
            sub_file = real_sub / "c.json"
            sub_file.write_bytes(oracle.canonical_json_bytes(self.cert))
            self.assertTrue(audit.verify_certificate_file(real_file)["valid"])
            audit.require_no_symlink_path(real_file)
            audit.require_no_symlink_path(real_dir / "nonexistent.json")

            # S21F-BUG-02 chmod 000 fail-closed check (no PermissionError traceback!)
            if os.geteuid() != 0:
                unreadable = real_dir / "chmod000.json"
                unreadable.write_bytes(oracle.canonical_json_bytes(self.cert) + b"\n")
                os.chmod(unreadable, 0o000)
                try:
                    with self.assertRaises(oracle.Wht8BridgeError) as ctx_perm:
                        audit.verify_certificate_file(unreadable)
                    self.assertEqual(ctx_perm.exception.code, "FILE_READ_ERROR")
                    err_p = io.StringIO()
                    with redirect_stderr(err_p):
                        self.assertEqual(audit.main(["--check", str(unreadable)]), 1)
                    self.assertIn("FAIL FILE_READ_ERROR", err_p.getvalue())
                finally:
                    os.chmod(unreadable, 0o600)

            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ep:
                audit.verify_certificate_file(_EvilPath(real_file))
            self.assertEqual(ctx_ep.exception.code, "INVALID_PATH_TYPE")

            sym_file = tmp / "sym.json"
            sym_file.symlink_to(real_file)
            for fn_s in (audit.require_no_symlink_path, audit.verify_certificate_file, lambda p: audit.write_regular_file_bytes(p, b"x")):
                with self.assertRaises(oracle.Wht8BridgeError) as ctx_sf:
                    fn_s(sym_file)
                self.assertEqual(ctx_sf.exception.code, "SYMLINK_FORBIDDEN")

            sym_dir = tmp / "symdir"
            sym_dir.symlink_to(real_dir)
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_sd:
                audit.verify_certificate_file(sym_dir / "c.json")
            self.assertEqual(ctx_sd.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_dotdot:
                audit.verify_certificate_file(sym_dir / ".." / "real" / "c.json")
            self.assertEqual(ctx_dotdot.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ndir:
                audit.require_no_symlink_path(real_file / "child")
            self.assertEqual(ctx_ndir.exception.code, "PATH_STAT_ERROR")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_mdir:
                audit.require_no_symlink_path(tmp / "no_such_dir" / "c.json")
            self.assertEqual(ctx_mdir.exception.code, "FILE_NOT_FOUND")

            fifo_p = real_dir / "pipe.fifo"
            os.mkfifo(fifo_p)
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_fifo_r:
                audit.read_regular_bytes_capped(fifo_p)
            self.assertEqual(ctx_fifo_r.exception.code, "NON_REGULAR_FILE")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_fifo_w:
                audit.write_regular_file_bytes(fifo_p, b"x")
            self.assertEqual(ctx_fifo_w.exception.code, "NON_REGULAR_FILE")

            hl_a = real_dir / "hl_a.json"
            hl_b = real_dir / "hl_b.json"
            hl_a.write_bytes(b"KEEP\n")
            os.link(hl_a, hl_b)
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_hl:
                audit.write_regular_file_bytes(hl_a, b"CLOBBER\n")
            self.assertEqual(ctx_hl.exception.code, "HARDLINK_FORBIDDEN")
            self.assertEqual(hl_b.read_bytes(), b"KEEP\n")

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
                with mock.patch.dict(os.environ, {"PWD": "../" + real_dir.name}):
                    self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                with mock.patch.dict(os.environ, {"PWD": str(sym_other)}):
                    self.assertTrue(audit.verify_certificate_file(Path("c.json"))["valid"])
                with mock.patch.dict(os.environ, {"PWD": str(sym_dir)}):
                    with self.assertRaises(oracle.Wht8BridgeError) as ctx_pwd1:
                        audit.verify_certificate_file(Path("c.json"))
                    self.assertEqual(ctx_pwd1.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
                    err_pwd = io.StringIO()
                    with redirect_stderr(err_pwd):
                        self.assertEqual(audit.main(["--check", "c.json"]), 1)
                    self.assertIn("FAIL SYMLINK_ANCESTOR_FORBIDDEN", err_pwd.getvalue())
                os.chdir(real_sub)
                with mock.patch.dict(os.environ, {"PWD": str(sym_dir / "sub")}):
                    with self.assertRaises(oracle.Wht8BridgeError) as ctx_pwd2:
                        audit.verify_certificate_file(Path("c.json"))
                    self.assertEqual(ctx_pwd2.exception.code, "SYMLINK_ANCESTOR_FORBIDDEN")
            finally:
                os.chdir(orig_cwd)

            with self.assertRaises(oracle.Wht8BridgeError) as ctx_miss:
                audit.verify_certificate_file(tmp / "missing.json")
            self.assertEqual(ctx_miss.exception.code, "FILE_NOT_FOUND")

            exact_sz = tmp / "exact_max.json"
            exact_sz.write_bytes(('{"a":"' + ("x" * (oracle.MAX_JSON_BYTES - 8)) + '"}').encode("ascii"))
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ex:
                audit.verify_certificate_file(exact_sz)
            self.assertEqual(ctx_ex.exception.code, "INVALID_CERT_KEYS")

            over_sz = tmp / "over_max.json"
            over_sz.write_bytes(b" " * (oracle.MAX_JSON_BYTES + 1))
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_ov:
                audit.verify_certificate_file(over_sz)
            self.assertEqual(ctx_ov.exception.code, "FILE_TOO_LARGE")

            cr_file = tmp / "cr.json"
            cr_file.write_bytes(oracle.canonical_json_bytes(self.cert) + b"\r\n")
            with self.assertRaises(oracle.Wht8BridgeError) as ctx_cr:
                audit.verify_certificate_file(cr_file)
            self.assertEqual(ctx_cr.exception.code, "CR_FORBIDDEN")

            bad_utf8 = tmp / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe")
            with self.assertRaises(oracle.Wht8BridgeError):
                audit.verify_certificate_file(bad_utf8)

            pretty = tmp / "pretty.json"
            pretty.write_text(json.dumps(self.cert, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(oracle.Wht8BridgeError):
                audit.verify_certificate_file(pretty)

            # CLI write/check roundtrip including '=' syntax & 0o600 mode
            wr_p = real_dir / "written.json"
            buf_w = io.StringIO()
            with redirect_stdout(buf_w):
                self.assertEqual(audit.main([f"--write={wr_p}", "--json"]), 0)
            self.assertTrue(json.loads(buf_w.getvalue())["written"])
            self.assertEqual(stat.S_IMODE(os.lstat(wr_p).st_mode), 0o600)
            buf_c = io.StringIO()
            with redirect_stdout(buf_c):
                self.assertEqual(audit.main([f"--check={wr_p}", "--json"]), 0)
            self.assertTrue(json.loads(buf_c.getvalue())["valid"])

        buf_oj, buf_ot = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_oj):
            self.assertEqual(oracle.main(["--standalone", "--json"]), 0)
        self.assertEqual(json.loads(buf_oj.getvalue())["certificate_sha256"], self.cert["certificate_sha256"])
        with redirect_stdout(buf_ot):
            self.assertEqual(oracle.main(("--standalone",)), 0)
        self.assertIn("OK TRACK-S21F-WHT8-FROBENIUS", buf_ot.getvalue())
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx_abbr:
                oracle.main(["--st"])
            self.assertEqual(ctx_abbr.exception.code, 2)

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

        for bad_cli, want_code in (
            (["--json", "--json"], "DUPLICATE_FLAG"),
            (["--standalone", "--standalone"], "CONFLICTING_MODE"),
            (["--standalone", "--check", "a.json"], "CONFLICTING_MODE"),
            (["--standalone", "--check=a.json"], "CONFLICTING_MODE"),
            (["--check", "a.json", "--write", "b.json"], "CONFLICTING_MODE"),
            (["--check"], "MISSING_PATH_ARG"),
            (["--write"], "MISSING_PATH_ARG"),
            (["--check", ""], "INVALID_PATH_ARG"),
            (["--check", "--json"], "INVALID_PATH_ARG"),
            (["--write=--json"], "INVALID_PATH_ARG"),
            (["--check="], "INVALID_PATH_ARG"),
            (["--unknown"], "UNKNOWN_FLAG"),
        ):
            err_b = io.StringIO()
            with redirect_stderr(err_b):
                self.assertEqual(audit.main(bad_cli), 1)
            self.assertIn(f"FAIL {want_code}", err_b.getvalue())

        buf_aj, buf_at, buf_ae = io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(buf_aj):
            self.assertEqual(audit.main(["--check", str(cert_path), "--json"]), 0)
        self.assertTrue(json.loads(buf_aj.getvalue())["valid"])
        with redirect_stdout(buf_at):
            self.assertEqual(audit.main(["--standalone"]), 0)
        self.assertIn("OK TRACK-S21F-WHT8-FROBENIUS-AUDIT", buf_at.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main([]), 0)
        with redirect_stderr(buf_ae):
            self.assertEqual(audit.main(["--check", str(_HERE / "nonexistent_cert.json")]), 1)
        self.assertIn("FAIL FILE_NOT_FOUND", buf_ae.getvalue())


class TestZ98KernelRacesAndConcurrencyStress(unittest.TestCase):
    """Deleted CWD, renameat2 TOCTOU, hardlink race, 16T/80T concurrent writers, EEXIST/fstat."""

    def test_deleted_cwd_concurrency_renameat2_and_hardlink_races(self) -> None:
        # 1. Deleted CWD fail-closed
        with tempfile.TemporaryDirectory(prefix="s21f_cwd_") as tmp:
            doomed = Path(tmp) / "doomed"
            doomed.mkdir(mode=0o700)
            old_cwd = os.getcwd()
            try:
                os.chdir(doomed)
                os.rmdir(doomed)
                with self.assertRaises(oracle.Wht8BridgeError) as ctx_cwd:
                    audit.require_no_symlink_path(Path("rel.json"))
                self.assertEqual(ctx_cwd.exception.code, "CWD_UNAVAILABLE")
                proc = subprocess.run(
                    [sys.executable, "-I", "-B", str(_HERE / "wht8_frobenius_bridge_audit.py"), "--check", "rel.json"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 1)
                self.assertIn("FAIL CWD_UNAVAILABLE", proc.stderr)
            finally:
                os.chdir(old_cwd)

        # 2. Live >= 500 concurrent renameat2(RENAME_EXCHANGE) swaps across 550 R/W cycles
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            AT_FDCWD = -100
            RENAME_EXCHANGE = 2
            with tempfile.TemporaryDirectory(prefix="s21f_toctou_") as tmp:
                td = Path(tmp)
                secret_dir = td / "secret"
                secret_dir.mkdir(mode=0o700)
                secret_file = secret_dir / "data.json"
                secret_file.write_bytes(b"SECRET_DO_NOT_LEAK_OR_CLOBBER\n")
                safe_dir = td / "slot_a"
                safe_dir.mkdir(mode=0o700)
                (safe_dir / "data.json").write_bytes(b"SAFE_PAYLOAD\n")
                sym_slot = td / "slot_b"
                sym_slot.symlink_to(secret_dir)

                stop_ev = threading.Event()
                swaps = [0]
                p_a = os.fsencode(str(safe_dir))
                p_b = os.fsencode(str(sym_slot))

                def _swapper() -> None:
                    while not stop_ev.is_set() or swaps[0] < 600:
                        if renameat2(AT_FDCWD, p_a, AT_FDCWD, p_b, RENAME_EXCHANGE) == 0:
                            swaps[0] += 1
                        if stop_ev.is_set() and swaps[0] >= 600:
                            break

                t = threading.Thread(target=_swapper)
                t.start()
                try:
                    target = safe_dir / "data.json"
                    for i in range(550):
                        try:
                            got = audit.read_regular_bytes_capped(target)
                            self.assertNotIn(b"SECRET", got)
                        except oracle.Wht8BridgeError as exc:
                            self.assertIn(exc.code, ("SYMLINK_ANCESTOR_FORBIDDEN", "SYMLINK_FORBIDDEN", "PATH_STAT_ERROR", "FILE_READ_ERROR", "FILE_NOT_FOUND", "FILE_IO_ERROR"))
                        try:
                            audit.write_regular_file_bytes(target, f"SAFE_WRITE_{i}\n".encode("ascii"))
                        except oracle.Wht8BridgeError as exc:
                            self.assertIn(exc.code, ("SYMLINK_ANCESTOR_FORBIDDEN", "SYMLINK_FORBIDDEN", "PATH_STAT_ERROR", "FILE_WRITE_ERROR", "FILE_NOT_FOUND", "FILE_IO_ERROR"))
                finally:
                    stop_ev.set()
                    t.join(timeout=5)

                self.assertGreaterEqual(swaps[0], 500)
                self.assertEqual(secret_file.read_bytes(), b"SECRET_DO_NOT_LEAK_OR_CLOBBER\n")
                self.assertEqual([p.name for p in safe_dir.glob(".s21f_wht8_tmp_*")], [])

        # 3. Live >= 420 hardlink race + 16T distinct + 16T same-target + 80T burst + EEXIST/fstat guards
        with tempfile.TemporaryDirectory(prefix="s21f_conc_") as tmp:
            td = Path(tmp)
            victim = td / "victim.json"
            victim.write_bytes(b"VICTIM_INODE_IMMUTABLE\n")
            target = td / "target.json"
            stop_hl = threading.Event()

            def _hl_racer() -> None:
                while not stop_hl.is_set():
                    try:
                        os.unlink(target)
                    except OSError:
                        pass
                    try:
                        os.link(victim, target)
                    except OSError:
                        pass

            th = threading.Thread(target=_hl_racer)
            th.start()
            try:
                for i in range(420):
                    try:
                        audit.write_regular_file_bytes(target, f"NEW_INODE_{i}\n".encode("ascii"))
                    except oracle.Wht8BridgeError as exc:
                        self.assertIn(exc.code, ("HARDLINK_FORBIDDEN", "FILE_WRITE_ERROR"))
            finally:
                stop_hl.set()
                th.join(timeout=5)
            self.assertEqual(victim.read_bytes(), b"VICTIM_INODE_IMMUTABLE\n")
            self.assertEqual([p.name for p in td.glob(".s21f_wht8_tmp_*")], [])

            mt_errors: list[str] = []
            mt_lock = threading.Lock()
            d_dir = td / "mt_distinct"
            d_dir.mkdir(mode=0o700)

            def _dw(tid: int) -> None:
                p = d_dir / f"w_{tid}.json"
                for k in range(40):
                    try:
                        audit.write_regular_file_bytes(p, f"D{tid}_{k}\n".encode("ascii"))
                    except Exception as exc:
                        with mt_lock:
                            mt_errors.append(f"d_{tid}_{k}:{exc}")

            ts = [threading.Thread(target=_dw, args=(t,)) for t in range(16)]
            for x in ts:
                x.start()
            for x in ts:
                x.join(timeout=15)
            self.assertEqual(mt_errors, [])
            for tid in range(16):
                p = d_dir / f"w_{tid}.json"
                self.assertEqual(p.read_bytes(), f"D{tid}_39\n".encode("ascii"))
                self.assertEqual(stat.S_IMODE(os.lstat(p).st_mode), 0o600)
            self.assertEqual([p.name for p in d_dir.glob(".s21f_wht8_tmp_*")], [])

            s_dir = td / "mt_same"
            s_dir.mkdir(mode=0o700)
            shared = s_dir / "shared.json"
            valid_same = {f"S{tid}_{k}\n".encode("ascii") for tid in range(16) for k in range(40)}

            def _sw(tid: int) -> None:
                for k in range(40):
                    try:
                        audit.write_regular_file_bytes(shared, f"S{tid}_{k}\n".encode("ascii"))
                    except Exception as exc:
                        with mt_lock:
                            mt_errors.append(f"s_{tid}_{k}:{exc}")

            ts = [threading.Thread(target=_sw, args=(t,)) for t in range(16)]
            for x in ts:
                x.start()
            for x in ts:
                x.join(timeout=15)
            self.assertEqual(mt_errors, [])
            self.assertIn(shared.read_bytes(), valid_same)
            self.assertEqual([p.name for p in s_dir.glob(".s21f_wht8_tmp_*")], [])

            b_dir = td / "mt_burst80"
            b_dir.mkdir(mode=0o700)
            barrier = threading.Barrier(80)

            def _bw(tid: int) -> None:
                barrier.wait(timeout=10)
                try:
                    audit.write_regular_file_bytes(b_dir / f"b_{tid}.json", f"B{tid}\n".encode("ascii"))
                except Exception as exc:
                    with mt_lock:
                        mt_errors.append(f"b_{tid}:{exc}")

            ts = [threading.Thread(target=_bw, args=(t,)) for t in range(80)]
            for x in ts:
                x.start()
            for x in ts:
                x.join(timeout=15)
            self.assertEqual(mt_errors, [])
            self.assertEqual(len(list(b_dir.glob("b_*.json"))), 80)
            self.assertEqual([p.name for p in b_dir.glob(".s21f_wht8_tmp_*")], [])

            orig_open = audit.os.open
            orig_fstat = audit.os.fstat
            try:
                calls = {"n": 0}
                def _collide_once(p_arg, flags, *a, **kw):
                    if isinstance(p_arg, str) and p_arg.startswith(".s21f_wht8_tmp_") and calls["n"] == 0:
                        calls["n"] += 1
                        raise FileExistsError("simulated")
                    return orig_open(p_arg, flags, *a, **kw)
                audit.os.open = _collide_once  # type: ignore[assignment]
                audit.write_regular_file_bytes(s_dir / "retry_ok.json", b"OK\n")
                self.assertEqual((s_dir / "retry_ok.json").read_bytes(), b"OK\n")

                def _always_collide(p_arg, flags, *a, **kw):
                    if isinstance(p_arg, str) and p_arg.startswith(".s21f_wht8_tmp_"):
                        raise FileExistsError("always")
                    return orig_open(p_arg, flags, *a, **kw)
                audit.os.open = _always_collide  # type: ignore[assignment]
                with self.assertRaises(oracle.Wht8BridgeError) as ctx_exh:
                    audit.write_regular_file_bytes(s_dir / "retry_fail.json", b"NO\n")
                self.assertEqual(ctx_exh.exception.code, "TEMP_RETRY_EXHAUSTED")

                audit.os.open = orig_open  # type: ignore[assignment]
                class _FakeStat:
                    st_nlink = 2
                audit.os.fstat = lambda _fd: _FakeStat()  # type: ignore[assignment]
                with self.assertRaises(oracle.Wht8BridgeError) as ctx_thl:
                    audit.write_regular_file_bytes(s_dir / "tmp_hl.json", b"NO\n")
                self.assertEqual(ctx_thl.exception.code, "HARDLINK_FORBIDDEN")
            finally:
                audit.os.open = orig_open
                audit.os.fstat = orig_fstat
            self.assertEqual([p.name for p in s_dir.glob(".s21f_wht8_tmp_*")], [])


if __name__ == "__main__":
    unittest.main()
