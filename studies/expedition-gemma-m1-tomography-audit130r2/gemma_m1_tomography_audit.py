#!/usr/bin/env python3
"""Track S30: Zone-B Hardened CLI, Full Pure-Q Replay & Secondary Algebraic Cross-Check.

Pure Python 3 standard library: zero floats, zero bare asserts, zero RNG.
Uses atomic per-component `dir_fd` (`openat(..., O_DIRECTORY | O_NOFOLLOW)`) traversal,
`O_NONBLOCK` + `S_ISREG` checks, capped chunked reads, hardlink refusal (`st_nlink > 1`),
and atomic `O_CREAT | O_EXCL | O_NOFOLLOW` + `os.replace(..., src_dir_fd, dst_dir_fd)` writes.
"""
from __future__ import annotations

import errno
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gemma_m1_tomography_oracle import (
    GemmaM1Error,
    MAX_JSON_BYTES,
    MODELS,
    PINNED_FILE_SHA256,
    SCHEMA_VERSION,
    TEXTS,
    TRACK_ID,
    build_standalone_certificate,
    canonical_json_bytes,
    parse_exact_rational,
    parse_strict_json_text,
    require_exact_bool,
    require_exact_int,
    require_exact_str,
    validate_certificate_payload,
)

_EXACT_PATH_TYPE = type(Path("."))
_READ_CHUNK_BYTES = 65536
_MAX_READ_CAP_BYTES = 64 * 1024 * 1024
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


# ---------------------------------------------------------------------------
# Race-Free Zone-B File I/O (Per-Component dir_fd/openat + Atomic os.replace)
# ---------------------------------------------------------------------------

def _open_dir_chain(anchor: str, parts: Sequence[str]) -> int:
    """Walk directory components atomically via `openat(..., O_DIRECTORY | O_NOFOLLOW)`."""
    try:
        dfd = os.open(anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise GemmaM1Error("PATH_STAT_ERROR", f"cannot open anchor dir: {type(exc).__name__}") from exc

    for part in parts:
        try:
            nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
        except OSError as exc:
            os.close(dfd)
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise GemmaM1Error("SYMLINK_FORBIDDEN", "symlink or non-dir forbidden in directory path") from exc
            if exc.errno == errno.ENOENT:
                raise GemmaM1Error("PATH_NOT_FOUND", "directory component not found") from exc
            raise GemmaM1Error("PATH_STAT_ERROR", f"cannot traverse dir: {type(exc).__name__}") from exc
        os.close(dfd)
        dfd = nxt
    return dfd


def _validate_path_and_split(path: Any) -> tuple[str, list[str]]:
    if type(path) is not _EXACT_PATH_TYPE:
        raise GemmaM1Error("INVALID_PATH_TYPE", "path must be exact pathlib.Path")
    raw_str = require_exact_str(str(path), "path")
    if "\x00" in raw_str:
        raise GemmaM1Error("INVALID_PATH", "path contains NUL byte")

    if not path.is_absolute():
        try:
            Path.cwd()
        except OSError as exc:
            raise GemmaM1Error("CWD_UNAVAILABLE", f"cannot resolve cwd: {type(exc).__name__}") from exc
        pwd_env = require_exact_str(os.environ.get("PWD", ""), "PWD", non_empty=False)
        if pwd_env.startswith("/"):
            pwd_path = Path(pwd_env)
            pwd_parts = [p for p in pwd_path.parts[1:] if p not in ("", ".")]
            try:
                pwd_fd = _open_dir_chain(pwd_path.anchor, pwd_parts)
            except GemmaM1Error as exc:
                if exc.code != "PATH_NOT_FOUND":
                    raise
            else:
                os.close(pwd_fd)
        rel_parts = [p for p in path.parts if p not in ("", ".")]
        return ".", rel_parts
    abs_parts = [p for p in path.parts[1:] if p not in ("", ".")]
    return path.anchor, abs_parts


def require_safe_path(path: Any) -> Path:
    """Validate path type, CWD/PWD, and atomic `openat` directory chain with zero symlinks."""
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        return path
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            st = os.lstat(parts[-1], dir_fd=dfd)
        except FileNotFoundError:
            return path
        except OSError as exc:
            raise GemmaM1Error("PATH_STAT_ERROR", f"cannot lstat leaf: {type(exc).__name__}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise GemmaM1Error("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
    finally:
        os.close(dfd)
    return path


def _open_leaf_for_read(path: Path) -> int:
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise GemmaM1Error("NON_REGULAR_FILE", "target path refers to a directory")
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise GemmaM1Error("SYMLINK_FORBIDDEN", "symlink forbidden at leaf") from exc
            raise GemmaM1Error("FILE_READ_ERROR", f"cannot open file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


def read_regular_file_bytes(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    cap = require_exact_int(max_bytes, "max_bytes", min_val=1, max_val=_MAX_READ_CAP_BYTES)
    fd = _open_leaf_for_read(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise GemmaM1Error("NON_REGULAR_FILE", "target is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            to_read = min(_READ_CHUNK_BYTES, cap + 1 - total)
            chunk = os.read(fd, to_read)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise GemmaM1Error("FILE_TOO_LARGE", f"file exceeds {cap} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    except GemmaM1Error:
        raise
    except OSError as exc:
        raise GemmaM1Error("FILE_READ_ERROR", f"failed reading file: {type(exc).__name__}") from exc
    finally:
        os.close(fd)


def write_regular_file_bytes(path: Path, data: bytes) -> None:
    if type(data) is not bytes:
        raise GemmaM1Error("INVALID_WRITE_DATA", "data must be exact bytes")
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise GemmaM1Error("NON_REGULAR_FILE", "target path refers to a directory")
    leaf = parts[-1]
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise GemmaM1Error("FILE_WRITE_ERROR", f"cannot lstat target: {type(exc).__name__}") from exc
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise GemmaM1Error("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
            if not stat.S_ISREG(st.st_mode):
                raise GemmaM1Error("NON_REGULAR_FILE", "cannot write to non-regular file")
            if st.st_nlink > 1:
                raise GemmaM1Error("HARDLINK_FORBIDDEN", "cannot write through hardlinked file")

        tmp_fd: int | None = None
        tmp_name = ""
        for attempt in range(16):
            cand = f".{leaf}.s30tmp.{os.getpid()}.{attempt}"
            try:
                tmp_fd = os.open(
                    cand,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=dfd,
                )
                tmp_name = cand
                break
            except FileExistsError:
                continue
            except OSError as exc:
                raise GemmaM1Error("FILE_WRITE_ERROR", f"cannot create temp file: {type(exc).__name__}") from exc
        if tmp_fd is None:
            raise GemmaM1Error("FILE_WRITE_ERROR", "temp file collision limit exceeded")

        try:
            try:
                view = memoryview(data)
                written = 0
                while written < len(data):
                    written += os.write(tmp_fd, view[written:])
                os.fsync(tmp_fd)
            finally:
                os.close(tmp_fd)
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            os.fsync(dfd)
        except OSError as exc:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
            raise GemmaM1Error("FILE_WRITE_ERROR", f"failed writing file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------
# Secondary Algebraic Invariant Cross-Check Over All Five Pillars
# ---------------------------------------------------------------------------

def _require_dict(mapping: dict[str, Any], key: str, err_code: str) -> dict[str, Any]:
    if key not in mapping or type(mapping[key]) is not dict:
        raise GemmaM1Error(err_code, f"missing or invalid object key: {key}")
    return mapping[key]


def _require_list(mapping: dict[str, Any], key: str, err_code: str) -> list[Any]:
    if key not in mapping or type(mapping[key]) is not list:
        raise GemmaM1Error(err_code, f"missing or invalid list key: {key}")
    return mapping[key]


def _require_true(mapping: dict[str, Any], key: str, err_code: str) -> None:
    if not require_exact_bool(mapping.get(key), key):
        raise GemmaM1Error(err_code, f"required invariant {key} is false")


def _check_pair_stat(pair_obj: dict[str, Any], *, expect_cand_win: bool, expect_base_win: bool, err_code: str) -> None:
    gain = parse_exact_rational(pair_obj.get("gain_kl"), "gain_kl")
    z_sq = parse_exact_rational(pair_obj.get("z_sq"), "z_sq")
    fav_c = require_exact_bool(pair_obj.get("favors_cand_ge_2se"), "favors_cand_ge_2se")
    fav_b = require_exact_bool(pair_obj.get("favors_base_ge_2se"), "favors_base_ge_2se")
    if fav_c is not all((gain > 0, z_sq >= 4)) or fav_b is not all((gain < 0, z_sq >= 4)):
        raise GemmaM1Error(err_code, "z_sq / gain_kl inconsistent with 2-SE flags")
    if expect_cand_win and not fav_c:
        raise GemmaM1Error(err_code, "expected candidate to win by >= 2 SE")
    if expect_base_win and not fav_b:
        raise GemmaM1Error(err_code, "expected baseline to win by >= 2 SE")
    if not expect_cand_win and fav_c:
        raise GemmaM1Error(err_code, "candidate unexpectedly won by >= 2 SE")


def secondary_algebraic_audit(payload: Any) -> dict[str, Any]:
    """Schema-safe secondary algebraic invariant cross-check over all five Track S30 pillars."""
    if type(payload) is not dict:
        raise GemmaM1Error("AUDIT_PAYLOAD_TYPE", "payload must be exact dict")

    # Pillar 1: Provenance & PR #47 P1..P10 invariants (including z_sq >= 4 & Spearman rho^2 >= 81/100)
    p1 = _require_dict(payload, "pillar1_provenance_and_prereg", "AUDIT_P1_SCHEMA")
    if require_exact_int(p1.get("total_rows_validated"), "total_rows_validated") != 102:
        raise GemmaM1Error("AUDIT_P1_ROWS", "expected exactly 102 validated JSONL rows")
    pin_map = _require_dict(p1, "pinned_jsonl_sha256", "AUDIT_P1_SCHEMA")
    for m in MODELS:
        fname = f"{m}.jsonl"
        if pin_map.get(fname) != PINNED_FILE_SHA256[fname]:
            raise GemmaM1Error("AUDIT_P1_SHA", f"pinned JSONL SHA mismatch for {fname}")

    p1_cmp = _require_dict(_require_dict(p1, "p1_hzbg_vs_q4km_imat", "AUDIT_P1_SCHEMA"), "gemma-3-4b-it", "AUDIT_P1_SCHEMA")
    if p1_cmp.get("verdict_wiki") != "HOLD":
        raise GemmaM1Error("AUDIT_P1_4B_WIKI", "4B P1 Wiki must HOLD")
    if p1_cmp.get("verdict_python") != "HOLD":
        raise GemmaM1Error("AUDIT_P1_4B_PY", "4B P1 Python must HOLD")
    _check_pair_stat(_require_dict(p1_cmp, "wikitext", "AUDIT_P1_SCHEMA"), expect_cand_win=True, expect_base_win=False, err_code="AUDIT_P1_4B_WIKI")
    _check_pair_stat(_require_dict(p1_cmp, "python", "AUDIT_P1_SCHEMA"), expect_cand_win=True, expect_base_win=False, err_code="AUDIT_P1_4B_PY")

    p2_4b = _require_dict(_require_dict(p1, "p2_plain_zbg_vs_q4km", "AUDIT_P1_SCHEMA"), "gemma-3-4b-it", "AUDIT_P1_SCHEMA")
    if p2_4b.get("verdict_wiki") != "FAIL_REVERSED":
        raise GemmaM1Error("AUDIT_P1_P2_REV", "4B P2 must reverse")
    _check_pair_stat(_require_dict(p2_4b, "wikitext_q4km_beats_zbg", "AUDIT_P1_SCHEMA"), expect_cand_win=False, expect_base_win=True, err_code="AUDIT_P1_P2_REV")

    p3_all = _require_dict(p1, "p3_hzbg_vs_zbg", "AUDIT_P1_SCHEMA")
    p4_all = _require_dict(p1, "p4_hzbg_k10_vs_hzbg", "AUDIT_P1_SCHEMA")
    p6_all = _require_dict(p1, "p6_zbg_k10_vs_zbg", "AUDIT_P1_SCHEMA")
    p8_all = _require_dict(p1, "p8_5bit_improvement_and_rank1", "AUDIT_P1_SCHEMA")
    p9_all = _require_dict(p1, "p9_4bit_rise_and_beats_q4_0", "AUDIT_P1_SCHEMA")
    for m in MODELS:
        p3_m = _require_dict(p3_all, m, "AUDIT_P1_SCHEMA")
        if p3_m.get("verdict") != "HOLD":
            raise GemmaM1Error("AUDIT_P1_P3", f"P3 must HOLD on {m}")
        for t in TEXTS:
            _check_pair_stat(_require_dict(p3_m, t, "AUDIT_P1_SCHEMA"), expect_cand_win=True, expect_base_win=False, err_code="AUDIT_P1_P3")
        p4_m = _require_dict(p4_all, m, "AUDIT_P1_SCHEMA")
        if p4_m.get("verdict") != "FAIL":
            raise GemmaM1Error("AUDIT_P1_P4", f"P4 must FAIL on {m}")
        for t in TEXTS:
            _check_pair_stat(_require_dict(p4_m, t, "AUDIT_P1_SCHEMA"), expect_cand_win=False, expect_base_win=False, err_code="AUDIT_P1_P4")
        p6_m = _require_dict(p6_all, m, "AUDIT_P1_SCHEMA")
        if p6_m.get("verdict") != "FAIL":
            raise GemmaM1Error("AUDIT_P1_P6", f"P6 must FAIL on {m}")
        _check_pair_stat(_require_dict(p6_m, "wikitext", "AUDIT_P1_SCHEMA"), expect_cand_win=False, expect_base_win=False, err_code="AUDIT_P1_P6")
        p8_m = _require_dict(p8_all, m, "AUDIT_P1_SCHEMA")
        if p8_m.get("verdict") != "HOLD":
            raise GemmaM1Error("AUDIT_P1_P8", f"P8 must HOLD on {m}")
        for t in TEXTS:
            _check_pair_stat(_require_dict(p8_m, t, "AUDIT_P1_SCHEMA"), expect_cand_win=True, expect_base_win=False, err_code="AUDIT_P1_P8")
        p9_m = _require_dict(p9_all, m, "AUDIT_P1_SCHEMA")
        if p9_m.get("verdict") != "HOLD":
            raise GemmaM1Error("AUDIT_P1_P9", f"P9 must HOLD on {m}")
        for t in TEXTS:
            _check_pair_stat(_require_dict(p9_m, t, "AUDIT_P1_SCHEMA"), expect_cand_win=True, expect_base_win=False, err_code="AUDIT_P1_P9")

    for key in (
        "p5_wiki_monotone_scale_decay",
        "p5_4b_python_collapse_below_10pct",
        "p10_spearman_holds_all_6",
        "bf16_4b_token_embd_asymmetry_certified",
    ):
        _require_true(p1, key, "AUDIT_P1_FLAG")
    if require_exact_int(p1.get("p10_inversion_le_0_2_passes_of_6"), "p10_inv_cnt") != 1:
        raise GemmaM1Error("AUDIT_P1_P10_INV", "expected exactly 1/6 cells passing <=0.2 pp inversion")
    p10_all = _require_dict(p1, "p10_top1_ranking", "AUDIT_P1_SCHEMA")
    inv_pass_check = 0
    for m in MODELS:
        p10_m = _require_dict(p10_all, m, "AUDIT_P1_SCHEMA")
        for t in TEXTS:
            cell = _require_dict(p10_m, t, "AUDIT_P1_SCHEMA")
            if parse_exact_rational(cell.get("spearman_rho_sq"), "spearman_rho_sq") < Fraction(81, 100):
                raise GemmaM1Error("AUDIT_P1_P10_RHO", f"Spearman rho^2 < 81/100 on {m} {t}")
            if parse_exact_rational(cell.get("max_inversion_pp"), "max_inv") <= Fraction(1, 5):
                inv_pass_check += 1
    if inv_pass_check != 1:
        raise GemmaM1Error("AUDIT_P1_P10_INV", "recounted inversion passes != 1")

    p5_all = _require_dict(p1, "p5_rand_retention", "AUDIT_P1_SCHEMA")
    r_w = [
        parse_exact_rational(_require_dict(_require_dict(p5_all, m, "AUDIT_P1_SCHEMA"), "wikitext", "AUDIT_P1_SCHEMA").get("retention_ratio"), "p5_ret")
        for m in MODELS
    ]
    if r_w[0] <= r_w[1] or r_w[1] <= r_w[2] or r_w[2] < Fraction(7, 10):
        raise GemmaM1Error("AUDIT_P1_P5_DECAY", "P5 Wiki retention must decay monotonically >= 7/10")

    # Pillar 2: Rate-distortion convexity & Jensen iso-KL bracket
    p2 = _require_dict(payload, "pillar2_rate_distortion_convexity", "AUDIT_P2_SCHEMA")
    if require_exact_int(p2.get("triples_count"), "triples_count") != 12:
        raise GemmaM1Error("AUDIT_P2_COUNT", "expected 12 measured triples")
    triples = _require_list(p2, "triples", "AUDIT_P2_SCHEMA")
    if len(triples) != 12:
        raise GemmaM1Error("AUDIT_P2_COUNT", "triples list length != 12")
    for tr in triples:
        if type(tr) is not dict:
            raise GemmaM1Error("AUDIT_P2_SCHEMA", "triple item must be dict")
        _require_true(tr, "strictly_convex", "AUDIT_P2_TRIPLE")
        _require_true(tr, "geo_beats_linear_strictly", "AUDIT_P2_TRIPLE")
    for key in (
        "all_12_strictly_convex",
        "all_12_geo_beats_linear",
        "all_6_wiki_geo_within_2_5_pct",
        "all_3_zbg_slopes_match_6_02",
    ):
        _require_true(p2, key, "AUDIT_P2_FLAG")
    iso_all = _require_dict(p2, "gemma3_4b_iso_kl_bracket", "AUDIT_P2_SCHEMA")
    for t in TEXTS:
        iso = _require_dict(iso_all, t, "AUDIT_P2_SCHEMA")
        b_lo = parse_exact_rational(iso.get("b_log_lo"), "b_log_lo")
        b_hi = parse_exact_rational(iso.get("b_log_hi"), "b_log_hi")
        b_lin = parse_exact_rational(iso.get("b_lin_upper_bound"), "b_lin")
        if b_lo > b_hi or b_hi >= b_lin:
            raise GemmaM1Error("AUDIT_P2_JENSEN", f"invalid Jensen iso-KL ordering on 4B {t}")
        if (b_hi - b_lo) >= Fraction(1, 10 ** 12):
            raise GemmaM1Error("AUDIT_P2_WIDTH", f"iso-KL bracket width >= 1e-12 on 4B {t}")
        _require_true(iso, "b_log_width_lt_1e_12", "AUDIT_P2_WIDTH")

    # Pillar 3: Q4_K_M fallback decoder (re-validate per-kind fallback counts across all 3 models!)
    p3 = _require_dict(payload, "pillar3_q4km_fallback_decoder", "AUDIT_P3_SCHEMA")
    for key in ("fallback_270m_1b_exact", "normal_4b_zero_fallback_exact", "q4_0_negative_control_exact"):
        _require_true(p3, key, "AUDIT_P3_FLAG")
    dec_models = _require_dict(p3, "decoded_models", "AUDIT_P3_SCHEMA")
    expected_counts = {
        "gemma-3-270m-it": {"file:Q4_K_M": (1, 4, 1, 1), "file:Q4_K_M-imat": (1, 4, 1, 1), "file:Q4_0": (0, 0, 0, 0)},
        "gemma-3-1b-it": {"file:Q4_K_M": (1, 4, 1, 1), "file:Q4_K_M-imat": (1, 4, 1, 1), "file:Q4_0": (0, 0, 0, 0)},
        "gemma-3-4b-it": {"file:Q4_K_M": (0, 0, 2, 5), "file:Q4_K_M-imat": (0, 0, 2, 5), "file:Q4_0": (0, 0, 0, 0)},
    }
    for m, arms_exp in expected_counts.items():
        m_obj = _require_dict(dec_models, m, "AUDIT_P3_SCHEMA")
        for arm, (e8, e5, e6, e4) in arms_exp.items():
            a_obj = _require_dict(m_obj, arm, "AUDIT_P3_SCHEMA")
            actual = (
                require_exact_int(a_obj.get("q8_0_kind_count"), "q8_0_kind_count"),
                require_exact_int(a_obj.get("q5_kind_count"), "q5_kind_count"),
                require_exact_int(a_obj.get("q6_k_kind_count"), "q6_k_kind_count"),
                require_exact_int(a_obj.get("q4_k_kind_count"), "q4_k_kind_count"),
            )
            if actual != (e8, e5, e6, e4):
                raise GemmaM1Error("AUDIT_P3_COUNTS", f"unexpected fallback counts for {m} {arm}")

    # Pillar 4: E8-vs-H8*E8 anisotropy & PRED-M1-01
    p4 = _require_dict(payload, "pillar4_e8_h8_anisotropy", "AUDIT_P4_SCHEMA")
    emp = _require_dict(p4, "empirical_paradox", "AUDIT_P4_SCHEMA")
    for key in (
        "mse_identical_within_0_002_db",
        "e8_packing_gain_ge_0_65_db_all_21",
        "hbg_lower_ppl_wiki_all_3_models",
        "kl_flips_270m_hbg_to_4b_ebg",
    ):
        _require_true(emp, key, "AUDIT_P4_EMP")
    if parse_exact_rational(emp.get("max_abs_ebg_vs_hbg_db_across_21_pairs"), "max_diff") > Fraction(2, 1000):
        raise GemmaM1Error("AUDIT_P4_DIFF", "max |EBG-HBG| exceeds 0.002 dB")
    if parse_exact_rational(emp.get("min_zbg_minus_ebg_db_across_21_pairs"), "min_gain") < Fraction(65, 100):
        raise GemmaM1Error("AUDIT_P4_GAIN", "min(ZBG-EBG) below 0.65 dB")

    op = _require_dict(p4, "exact_e8_h8_operator_theorem", "AUDIT_P4_SCHEMA")
    if require_exact_int(op.get("e8_minimal_roots_count"), "e8_minimal_roots_count") != 240:
        raise GemmaM1Error("AUDIT_P4_ROOTS", "E8 must have exactly 240 minimal roots")
    for key in (
        "h8_symmetric_and_sq_8I",
        "e8_all_240_roots_norm2_and_fixed",
        "qh_not_automorphism_of_e8_proved",
        "generic_isotropic_trace_invariance_exact",
        "witness_identical_l2_error",
        "exact_8x_opposite_sign_kl_flip",
        "isometric_qh_e1_unique_nearest_and_lt_1_5",
        "isometric_qh_root_unique_nearest_and_gt_1_3",
        "isometric_qh_single_vector_opposite_flip",
    ):
        _require_true(op, key, "AUDIT_P4_OP")
    tr_e = parse_exact_rational(op.get("witness_tr_sigma_e"), "tr_e")
    tr_h = parse_exact_rational(op.get("witness_tr_sigma_e_h"), "tr_h")
    if tr_e != tr_h or tr_e <= 0:
        raise GemmaM1Error("AUDIT_P4_TRACE", "witness traces must be equal and > 0")
    h1_e = parse_exact_rational(op.get("h_act_1_axis_kl_e8"), "h1_e")
    h1_h = parse_exact_rational(op.get("h_act_1_axis_kl_h8"), "h1_h")
    h2_e = parse_exact_rational(op.get("h_act_2_dc_kl_e8"), "h2_e")
    h2_h = parse_exact_rational(op.get("h_act_2_dc_kl_h8"), "h2_h")
    if h1_e != 8 * h1_h or h2_h != 8 * h2_e:
        raise GemmaM1Error("AUDIT_P4_FLIP_RATIO", "expected exact 8x opposite-sign KL flip")
    if _require_list(op, "isometric_qh_e1_d2_qh_qsqrt2", "AUDIT_P4_SCHEMA") != ["3", "-2"]:
        raise GemmaM1Error("AUDIT_P4_QSQRT2", "expected isometric Q_H e1 distance 3 - 2*sqrt(2)")
    if _require_list(op, "isometric_qh_root_d2_qh_qsqrt2", "AUDIT_P4_SCHEMA") != ["6", "-4"]:
        raise GemmaM1Error("AUDIT_P4_QSQRT2", "expected isometric Q_H root distance 6 - 4*sqrt(2)")
    _require_true(_require_dict(p4, "pred_m1_01_hebg128_steelman", "AUDIT_P4_SCHEMA"), "pred_savings_strictly_above_12pct", "AUDIT_P4_PRED")

    # Pillar 5: run.py AST audit & verified patches
    p5 = _require_dict(payload, "pillar5_run_py_ast_audit", "AUDIT_P5_SCHEMA")
    if p5.get("run_py_sha256") != PINNED_FILE_SHA256["run.py"]:
        raise GemmaM1Error("AUDIT_P5_SHA", "pinned run.py SHA-256 mismatch")
    if _require_list(p5, "run_bug_01_hardcoded_hzbg_lines", "AUDIT_P5_SCHEMA") != [967, 1035]:
        raise GemmaM1Error("AUDIT_P5_LINES", "RUN-BUG-01 lines must equal [967, 1035]")
    if _require_list(p5, "run_bug_02_hardcoded_f16_lines", "AUDIT_P5_SCHEMA") != [733]:
        raise GemmaM1Error("AUDIT_P5_LINES", "RUN-BUG-02 line must equal [733]")
    missed_arms = _require_list(p5, "run_bug_01_arms_missed_by_buggy", "AUDIT_P5_SCHEMA")
    if require_exact_int(p5.get("run_bug_01_missed_count"), "run_bug_01_missed_count") != 5 or len(missed_arms) != 5:
        raise GemmaM1Error("AUDIT_P5_MISSED", "RUN-BUG-01 must report exactly 5 missed arms")
    for key in (
        "weighted_ball_re_supports_all_6",
        "run_bug_01_detected_exact_lines",
        "run_bug_01_fixed_predicate_verified",
        "run_bug_02_detected_exact_line",
        "run_bug_02_fixed_selector_verified",
    ):
        _require_true(p5, key, "AUDIT_P5_FLAG")

    return {
        "secondary_audit_passed": True,
        "models_checked": len(MODELS),
        "triples_checked": p2["triples_count"],
        "e8_roots_checked": op["e8_minimal_roots_count"],
    }


def check_certificate_file(path: Path) -> dict[str, Any]:
    raw = read_regular_file_bytes(path)
    if b"\r" in raw:
        raise GemmaM1Error("CRLF_FORBIDDEN", "CRLF line endings forbidden in certificate")
    if not raw.endswith(b"\n"):
        raise GemmaM1Error("MISSING_TRAILING_LF", "certificate must end with a single LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GemmaM1Error("INVALID_UTF8", "certificate is not valid UTF-8") from exc
    cert = parse_strict_json_text(text, require_object=True)
    canonical_raw = canonical_json_bytes(cert)
    if raw != canonical_raw + b"\n":
        raise GemmaM1Error("NON_CANONICAL_BYTES", "certificate bytes are not canonical ASCII JSON + LF")
    res = validate_certificate_payload(cert, full_replay=True)
    sec = secondary_algebraic_audit(cert["payload"])
    res["canonical_sha256"] = hashlib.sha256(canonical_raw).hexdigest()
    res["file_sha256"] = hashlib.sha256(raw).hexdigest()
    res["secondary_audit"] = sec
    return res


def write_certificate_file(path: Path) -> dict[str, Any]:
    cert = build_standalone_certificate()
    canonical_raw = canonical_json_bytes(cert)
    raw = canonical_raw + b"\n"
    write_regular_file_bytes(path, raw)
    return {
        "written": True,
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": cert["payload_sha256"],
        "certificate_sha256": cert["certificate_sha256"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Strict Fail-Closed Manual CLI Parsing
# ---------------------------------------------------------------------------

def _parse_cli_argv(argv: Sequence[str]) -> tuple[str, Path | None, bool]:
    if type(argv) not in (list, tuple):
        raise GemmaM1Error("INVALID_ARGV", "argv must be exact list/tuple of strict str")
    for item in argv:
        require_exact_str(item, "argv_item", non_empty=False)

    mode: str | None = None
    target: Path | None = None
    as_json = False
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--json":
            if as_json:
                raise GemmaM1Error("DUPLICATE_FLAG", "duplicate --json flag")
            as_json = True
            i += 1
        elif tok == "--standalone":
            if mode is not None:
                raise GemmaM1Error("CONFLICTING_MODE", "multiple mode flags forbidden")
            mode = "standalone"
            i += 1
        elif tok in ("--check", "--write"):
            if mode is not None:
                raise GemmaM1Error("CONFLICTING_MODE", "multiple mode flags forbidden")
            if i + 1 >= n:
                raise GemmaM1Error("MISSING_PATH_ARG", f"{tok} requires a path argument")
            val = argv[i + 1]
            if not val or val.startswith("-"):
                raise GemmaM1Error("INVALID_PATH_ARG", f"{tok} path cannot be empty or flag-like")
            mode = "check" if tok == "--check" else "write"
            target = Path(val)
            i += 2
        else:
            raise GemmaM1Error("UNKNOWN_FLAG", f"unrecognized CLI argument: {tok}")

    if mode is None:
        mode = "standalone"
    return mode, target, as_json


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv: Any = sys.argv[1:] if argv is None else argv
    try:
        mode, target, as_json = _parse_cli_argv(raw_argv)
        if mode == "standalone":
            out: dict[str, Any] = build_standalone_certificate()
        elif mode == "write":
            out = write_certificate_file(target)  # type: ignore[arg-type]
        else:
            out = check_certificate_file(target)  # type: ignore[arg-type]
    except GemmaM1Error as exc:
        err = {"valid": False, "error_code": exc.code, "error_message": exc.message}
        sys.stderr.write(canonical_json_bytes(err).decode("ascii") + "\n")
        return 2

    if as_json:
        sys.stdout.write(canonical_json_bytes(out).decode("ascii") + "\n")
    else:
        sys.stdout.write(
            f"OK {SCHEMA_VERSION} payload_sha256={out['payload_sha256']} "
            f"certificate_sha256={out['certificate_sha256']}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
