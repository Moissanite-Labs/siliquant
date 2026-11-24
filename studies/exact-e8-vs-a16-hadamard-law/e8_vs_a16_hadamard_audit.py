#!/usr/bin/env python3
"""Hardened Standalone CLI, Full Pure-Z/Q Replay, Field-Complete Secondary Auditor,
Live Lean 4.36.0 Companion Verifier, and Read-Only Empirical Anchor Checker
for the Opposite-Sign E_8 vs. A_16 Hadamard Law.

100% pure Python 3 stdlib, zero IEEE-754 floats, zero bare asserts, zero `.isdigit()`,
zero `os.path.islink`, zero mathematical RNG (`secrets.token_hex` used strictly for
collision-free atomic temp filenames).
"""
from __future__ import annotations

import errno
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from typing import Any

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import e8_vs_a16_hadamard_oracle as oracle  # noqa: E402

MAX_JSON_BYTES = oracle.MAX_JSON_BYTES
MAX_ALLOWED_CAP_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 65_536
_EXACT_PATH_TYPES = (Path, type(Path()))
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TEMP_NAME_LOCK = threading.Lock()
_TEMP_COUNTER = 0
_DARWIN_SYSTEM_ROOT_ALIASES: tuple[tuple[str, str], ...] = (
    ("/var", "/private/var"),
    ("/tmp", "/private/tmp"),
    ("/etc", "/private/etc"),
)
_ALLOWED_LEAN_AXIOMS: frozenset[str] = frozenset({"propext", "Classical.choice", "Quot.sound"})
_AXIOM_BLOCK_RE = re.compile(
    r"'([^']+)'\s+(?:(does not depend on any axioms)|depends on axioms:\s*\[([^\]]*)\])",
    flags=re.DOTALL,
)

_CERT_KEYS = frozenset({"schema_version", "track_id", "payload_sha256", "certificate_sha256", "payload"})
_PAYLOAD_KEYS = frozenset({
    "track_id",
    "pillar1_lattice_ray_anisotropy_and_hadamard",
    "pillar2_exact_quadratic_kl_witnesses_and_regime_flip",
    "pillar3_empirical_llm_bridge_and_selector",
    "pillar4_lean4_companion_and_cs13_scope",
    "summary",
})
_P1_KEYS = frozenset({
    "h8_orthogonal",
    "h8_is_exact_rational",
    "h16_orthogonal",
    "h16_is_exact_rational",
    "block_h2_is_exact_rational",
    "c16_codeword_count",
    "c16_min_weight",
    "a16_kissing_number",
    "e8_root_count",
    "h16_rows_in_c16_both_polarities",
    "e8_hadamard_rows_are_shortest_roots",
    "a16_block_h_spectrum",
    "e8_block_h_spectrum",
    "opposite_sign_contrast_factor",
    "a16_block_h4_restores_rho_1",
    "a16_block_h4_is_lattice_isometry",
    "coherent_dc_concentration_factors",
    "block_h2_int8_bounds",
})
_P1_A16_ROW_KEYS = frozenset({"block_size", "indicator_in_c16", "shortest_axis_norm_sq", "rho_axis"})
_P1_E8_ROW_KEYS = frozenset({"block_size", "shortest_axis_norm_sq", "rho_axis"})
_P1_DC_KEYS = frozenset({"h16", "h8", "block_h2"})
_P1_INT8_KEYS = frozenset({
    "max_abs_block_h2_for_39",
    "int8_max",
    "ball_72b_max_abs",
    "ball_72b_r2",
    "ball_18b_max_abs",
    "ball_18b_r2",
    "all_fit_int8",
})
_P2_KEYS = frozenset({
    "single_spike_3_2_e0",
    "matrix_witness_zero_mse_gap",
    "regime_flip_criterion",
})
_P2_SPIKE_KEYS = frozenset({
    "a16_axis_err_sq",
    "a16_block_h2_err_sq",
    "a16_h16_err_sq",
    "a16_block_h4_err_sq",
    "e8_axis_err_sq",
    "e8_h8_err_sq",
    "a16_strict_ordering_holds",
    "e8_opposite_sign_holds",
})
_P2_MAT_KEYS = frozenset({
    "a16_axis_mse",
    "a16_h16_mse",
    "a16_block_h4_mse",
    "a16_zero_mse_gap",
    "a16_axis_nat_kl",
    "a16_block_h2_nat_kl",
    "a16_h16_nat_kl",
    "a16_block_h4_nat_kl",
    "a16_axis_matched_kl",
    "a16_block_h2_matched_kl",
    "a16_h16_matched_kl",
    "e8_axis_nat_kl",
    "e8_h8_nat_kl",
    "e8_axis_matched_kl",
    "e8_h8_matched_kl",
    "a16_natural_ordering_holds",
    "a16_matched_ordering_holds",
    "a16_matched_midpoint_exact",
    "e8_natural_opposite_sign_holds",
    "e8_matched_opposite_sign_holds",
    "a16_block_h4_beats_axis_on_witness",
})
_P2_REG_KEYS = frozenset({
    "trace_delta_sigma",
    "h1_axis_spike_delta_kl",
    "h2_coherent_dc_delta_kl",
    "sign_reversal_verified",
})
_P3_KEYS = frozenset({
    "anchor_filename",
    "anchor_sha256",
    "models",
    "combined_total_layers",
    "combined_regime1_count",
    "combined_regime2_count",
    "combined_bh2_beats_h16_count",
    "k_proj_both_models_h16_lt_98_100",
    "by_projection_kind_combined",
    "selector_rule_checks_passed",
})
_P3_MODEL_KEYS = frozenset({
    "total_layers",
    "regime1_count",
    "regime2_count",
    "bh2_beats_h16_count",
    "wikitext2_kl_q",
    "e8_h8_lt_e8_axis",
    "a16_axis_lt_bh2_lt_h16",
    "matched_preserves_a16_gap",
})
_P3_KL_KEYS = frozenset({"E8_H8", "E8_axis", "A16_axis", "A16_BlockH2", "A16_H16", "A16_H16_matched"})
_P3_PROJ_KEYS = frozenset({"layers", "regime1", "regime2", "bh2_beats_h16", "selector_default_arm"})
_P4_KEYS = frozenset({
    "lean_companion_filename",
    "lean_companion_sha256",
    "lean_companion_theorem_count",
    "lean_companion_theorems",
    "novelty_tier",
    "oracle_class",
    "max_int_bits_exact_cap",
    "max_int_digits_preparse_ceiling",
    "measured_live_rtx3080_in_this_track",
    "measured_live_block_h4_llm_in_this_track",
    "zero_trust_path_floats",
    "zero_bare_asserts",
    "zero_mathematical_rng",
    "prior_literature_n0",
})
_SUMMARY_KEYS = frozenset({"all_pillars_verified", "target_max_cert_bytes"})


# ---------------------------------------------------------------------------
# Darwin Alias Normalization, Atomic dir_fd Traversal & Durable 0o600 Write
# ---------------------------------------------------------------------------

def _coerce_exact_path(path: Any, name: str) -> Path:
    if type(path) is str:
        s = oracle.require_exact_str(path, name)
        if not s:
            raise oracle.HadamardLawError("INVALID_PATH", f"{name} must be non-empty")
        if s.endswith("/"):
            raise oracle.HadamardLawError("INVALID_FILENAME", f"trailing slash refused: {s!r}")
        return Path(s)
    if type(path) in _EXACT_PATH_TYPES:
        oracle.require_exact_str(str(path), name)
        return path
    raise oracle.HadamardLawError("INVALID_PATH_TYPE", f"{name} must be str or Path")


def _normalize_darwin_system_prefix(path: Path, *, allow_exact: bool = False) -> Path:
    """Map root-owned macOS /var, /tmp, /etc aliases to /private/* without resolving user symlinks."""
    if sys.platform != "darwin" or not path.is_absolute():
        return path
    parts = path.parts
    min_parts = 2 if allow_exact else 3
    if len(parts) < min_parts or ".." in parts:
        return path
    top = "/" + parts[1]
    for alias, canonical in _DARWIN_SYSTEM_ROOT_ALIASES:
        if top != alias:
            continue
        try:
            st = os.lstat(alias)
            if not stat.S_ISLNK(st.st_mode) or st.st_uid != 0:
                return path
            target = os.readlink(alias)
            if target not in (canonical, canonical.lstrip("/")):
                return path
        except OSError:
            return path
        return Path(canonical, *parts[2:])
    return path


def _open_dir_chain_nofollow(parts: tuple[str, ...]) -> int:
    try:
        cur_fd = os.open("/", _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise oracle.HadamardLawError("DIR_OPEN_FAILED", f"cannot open root directory: {exc}") from exc
    for comp in parts:
        if comp in ("", "."):
            continue
        if comp == "..":
            os.close(cur_fd)
            raise oracle.HadamardLawError("PARENT_TRAVERSAL_FORBIDDEN", "path must not contain '..'")
        try:
            next_fd = os.open(comp, _DIR_OPEN_FLAGS, dir_fd=cur_fd)
        except OSError as exc:
            is_sym = exc.errno == errno.ELOOP
            if exc.errno == errno.ENOTDIR:
                try:
                    is_sym = stat.S_ISLNK(os.lstat(comp, dir_fd=cur_fd).st_mode)
                except OSError:
                    is_sym = False
            os.close(cur_fd)
            if is_sym:
                raise oracle.HadamardLawError("SYMLINK_FORBIDDEN", f"symlink component rejected: {comp!r}") from exc
            if exc.errno == errno.ENOTDIR:
                raise oracle.HadamardLawError("NOT_A_DIRECTORY", f"component is not a directory: {comp!r}") from exc
            raise oracle.HadamardLawError("DIR_OPEN_FAILED", f"cannot traverse {comp!r}: {exc}") from exc
        os.close(cur_fd)
        cur_fd = next_fd
    return cur_fd


def _verify_pwd_not_symlink() -> None:
    pwd = os.environ.get("PWD")
    if not pwd or not pwd.startswith("/"):
        return
    oracle.require_exact_str(pwd, "PWD")
    try:
        st_dot = os.stat(".")
    except OSError as exc:
        raise oracle.HadamardLawError("CWD_UNAVAILABLE", f"cannot stat current directory: {exc}") from exc
    try:
        st_logical = os.stat(pwd)
    except OSError:
        return
    if (st_logical.st_dev, st_logical.st_ino) != (st_dot.st_dev, st_dot.st_ino):
        return
    norm_pwd = _normalize_darwin_system_prefix(Path(pwd), allow_exact=True)
    parts = tuple(p for p in norm_pwd.parts if p != "/")
    dfd = _open_dir_chain_nofollow(parts)
    try:
        st_pwd = os.fstat(dfd)
    finally:
        os.close(dfd)
    if (st_pwd.st_dev, st_pwd.st_ino) != (st_dot.st_dev, st_dot.st_ino):
        raise oracle.HadamardLawError("PWD_SYMLINK_MISMATCH", "$PWD changed during verification")


def _open_parent_dir_fd(path_obj: Path) -> tuple[int, str]:
    if path_obj.name in ("", ".", ".."):
        raise oracle.HadamardLawError("INVALID_FILENAME", f"invalid target filename: {str(path_obj)!r}")
    if path_obj.is_absolute():
        norm = _normalize_darwin_system_prefix(path_obj)
        parts = tuple(p for p in norm.parts if p != "/")
        return (_open_dir_chain_nofollow(parts[:-1]), parts[-1])
    _verify_pwd_not_symlink()
    try:
        cwd = os.getcwd()
    except OSError as exc:
        raise oracle.HadamardLawError("CWD_UNAVAILABLE", f"current working directory unavailable: {exc}") from exc
    norm_cwd = _normalize_darwin_system_prefix(Path(cwd), allow_exact=True)
    full_parts = tuple(p for p in norm_cwd.parts if p != "/") + path_obj.parts
    return (_open_dir_chain_nofollow(full_parts[:-1]), full_parts[-1])


def read_regular_bytes_capped(path: Any, *, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    """Read regular non-symlink file via atomic dir_fd + O_NOFOLLOW + O_NONBLOCK with streaming cap."""
    p_obj = _coerce_exact_path(path, "read_path")
    cap = oracle.require_exact_int(max_bytes, "max_bytes", min_val=1, max_val=MAX_ALLOWED_CAP_BYTES)
    dfd, fname = _open_parent_dir_fd(p_obj)
    try:
        try:
            lst = os.lstat(fname, dir_fd=dfd)
        except OSError as exc:
            raise oracle.HadamardLawError("FILE_UNREADABLE", f"cannot lstat {fname!r}: {exc}") from exc
        if stat.S_ISLNK(lst.st_mode):
            raise oracle.HadamardLawError("SYMLINK_FORBIDDEN", f"symlink target refused: {fname!r}")
        if not stat.S_ISREG(lst.st_mode):
            raise oracle.HadamardLawError("NON_REGULAR_FILE", f"non-regular file refused: {fname!r}")
        if lst.st_nlink > 1:
            raise oracle.HadamardLawError("HARDLINK_FORBIDDEN", f"hardlinked file refused: {fname!r}")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(fname, flags, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise oracle.HadamardLawError("SYMLINK_FORBIDDEN", f"symlink race refused: {fname!r}") from exc
            raise oracle.HadamardLawError("FILE_UNREADABLE", f"cannot open {fname!r}: {exc}") from exc
        try:
            try:
                fst = os.fstat(fd)
            except OSError as exc:
                raise oracle.HadamardLawError("READ_IO_ERROR", f"fstat failed: {exc}") from exc
            if not stat.S_ISREG(fst.st_mode):
                raise oracle.HadamardLawError("NON_REGULAR_FILE", f"opened target not regular: {fname!r}")
            if fst.st_nlink > 1:
                raise oracle.HadamardLawError("HARDLINK_FORBIDDEN", f"opened target hardlinked: {fname!r}")
            if (fst.st_dev, fst.st_ino) != (lst.st_dev, lst.st_ino):
                raise oracle.HadamardLawError("TOCTOU_SWAP", f"file identity changed during open: {fname!r}")
            chunks: list[bytes] = []
            total = 0
            while True:
                try:
                    buf = os.read(fd, _READ_CHUNK_BYTES)
                except OSError as exc:
                    raise oracle.HadamardLawError("READ_IO_ERROR", f"read failed: {exc}") from exc
                if not buf:
                    break
                total += len(buf)
                if total > cap:
                    raise oracle.HadamardLawError("FILE_TOO_LARGE", f"file exceeds {cap} byte cap")
                chunks.append(buf)
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)


def _next_temp_name(base_name: str) -> str:
    global _TEMP_COUNTER
    with _TEMP_NAME_LOCK:
        _TEMP_COUNTER += 1
        seq = _TEMP_COUNTER
    return f".{base_name}.tmp.{os.getpid()}.{seq}.{secrets.token_hex(8)}"


def write_bytes_atomic_0600(path: Any, data: bytes) -> None:
    """Durable 0o600 atomic write via dir_fd, refusing symlinks and multi-link targets."""
    if type(data) is not bytes:
        raise oracle.HadamardLawError("INVALID_BYTES", "data must be exact bytes")
    p_obj = _coerce_exact_path(path, "write_path")
    dfd, fname = _open_parent_dir_fd(p_obj)
    tmp_name: str | None = None
    tmp_fd: int | None = None
    try:
        try:
            lst = os.lstat(fname, dir_fd=dfd)
        except FileNotFoundError:
            lst = None
        except OSError as exc:
            raise oracle.HadamardLawError("TARGET_UNSTATTABLE", f"cannot lstat target {fname!r}: {exc}") from exc
        if lst is not None:
            if stat.S_ISLNK(lst.st_mode):
                raise oracle.HadamardLawError("SYMLINK_FORBIDDEN", f"refusing symlink write target: {fname!r}")
            if not stat.S_ISREG(lst.st_mode):
                raise oracle.HadamardLawError("NON_REGULAR_FILE", f"refusing non-regular write target: {fname!r}")
            if lst.st_nlink > 1:
                raise oracle.HadamardLawError("HARDLINK_FORBIDDEN", f"refusing hardlinked target: {fname!r}")

        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        for _ in range(32):
            cand = _next_temp_name(fname)
            try:
                tmp_fd = os.open(cand, create_flags, 0o600, dir_fd=dfd)
                tmp_name = cand
                break
            except FileExistsError:
                continue
            except OSError as exc:
                raise oracle.HadamardLawError("WRITE_IO_ERROR", f"cannot create temp file: {exc}") from exc
        if tmp_fd is None:
            raise oracle.HadamardLawError("TEMP_CREATE_FAILED", "exhausted temp filename retries")

        try:
            view = memoryview(data)
            written = 0
            while written < len(data):
                n = os.write(tmp_fd, view[written:])
                if n <= 0:
                    raise oracle.HadamardLawError("SHORT_WRITE", "os.write returned non-positive count")
                written += n
            os.fchmod(tmp_fd, 0o600)
            os.fsync(tmp_fd)
            st_tmp = os.fstat(tmp_fd)
            if not stat.S_ISREG(st_tmp.st_mode) or st_tmp.st_nlink != 1:
                raise oracle.HadamardLawError("TEMP_TAMPERED", "temp file mode or link count altered")
            os.close(tmp_fd)
            tmp_fd = None

            try:
                lst2 = os.lstat(fname, dir_fd=dfd)
            except FileNotFoundError:
                lst2 = None
            if lst2 is not None:
                if stat.S_ISLNK(lst2.st_mode):
                    raise oracle.HadamardLawError("SYMLINK_FORBIDDEN", f"symlink appeared at target: {fname!r}")
                if not stat.S_ISREG(lst2.st_mode):
                    raise oracle.HadamardLawError("NON_REGULAR_FILE", f"non-regular target appeared: {fname!r}")
                if lst2.st_nlink > 1:
                    raise oracle.HadamardLawError("HARDLINK_FORBIDDEN", f"hardlink appeared at target: {fname!r}")

            os.replace(tmp_name, fname, src_dir_fd=dfd, dst_dir_fd=dfd)
            tmp_name = None
            os.fsync(dfd)
        except oracle.HadamardLawError:
            raise
        except OSError as exc:
            raise oracle.HadamardLawError("WRITE_IO_ERROR", f"atomic write failed: {exc}") from exc
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


# ---------------------------------------------------------------------------
# Field-Complete Strict-Keyset Secondary Auditor & Certificate Verifier
# ---------------------------------------------------------------------------

def _require_dict_exact_keys(obj: Any, expected: frozenset[str], name: str) -> dict[str, Any]:
    if type(obj) is not dict:
        raise oracle.HadamardLawError("INVALID_OBJECT", f"{name} must be a dict")
    if frozenset(obj.keys()) != expected:
        raise oracle.HadamardLawError("KEYSET_MISMATCH", f"{name} keys do not match schema")
    return obj


def _audit_pillar1(p1_raw: Any) -> None:
    p1 = _require_dict_exact_keys(p1_raw, _P1_KEYS, "pillar1")
    if (
        oracle.require_exact_bool(p1["h8_orthogonal"], "h8_orth") is not True
        or oracle.require_exact_bool(p1["h8_is_exact_rational"], "h8_rat") is not False
        or oracle.require_exact_bool(p1["h16_orthogonal"], "h16_orth") is not True
        or oracle.require_exact_bool(p1["h16_is_exact_rational"], "h16_rat") is not True
        or oracle.require_exact_bool(p1["block_h2_is_exact_rational"], "bh2_rat") is not False
        or oracle.require_exact_int(p1["c16_codeword_count"], "c16_cnt") != 2048
        or oracle.require_exact_int(p1["c16_min_weight"], "c16_dmin") != 4
        or oracle.require_exact_int(p1["a16_kissing_number"], "a16_kiss") != 2272
        or oracle.require_exact_int(p1["e8_root_count"], "e8_roots") != 240
        or oracle.require_exact_int(p1["h16_rows_in_c16_both_polarities"], "h16_c16") != 16
        or oracle.require_exact_int(p1["e8_hadamard_rows_are_shortest_roots"], "e8_had") != 8
        or oracle.require_exact_int(p1["opposite_sign_contrast_factor"], "contrast") != 8
        or oracle.require_exact_bool(p1["a16_block_h4_restores_rho_1"], "bh4_rho1") is not True
        or oracle.require_exact_bool(p1["a16_block_h4_is_lattice_isometry"], "bh4_iso") is not False
    ):
        raise oracle.HadamardLawError("P1_SCALAR_MISMATCH", "Pillar 1 scalar invariants mismatch")

    a16_sp = p1["a16_block_h_spectrum"]
    exp_a16 = ((1, False, 4, 1), (2, False, 8, 2), (4, True, 4, 1), (8, True, 8, 2), (16, True, 16, 4))
    if type(a16_sp) is not list or len(a16_sp) != len(exp_a16):
        raise oracle.HadamardLawError("P1_A16_SPECTRUM_SHAPE", "a16_block_h_spectrum shape mismatch")
    for row, (b, inc, nsq, rho) in zip(a16_sp, exp_a16):
        r = _require_dict_exact_keys(row, _P1_A16_ROW_KEYS, "a16_spectrum_row")
        if (
            oracle.require_exact_int(r["block_size"], "b") != b
            or oracle.require_exact_bool(r["indicator_in_c16"], "inc") is not inc
            or oracle.require_exact_int(r["shortest_axis_norm_sq"], "nsq") != nsq
            or oracle.require_exact_int(r["rho_axis"], "rho") != rho
        ):
            raise oracle.HadamardLawError("P1_A16_SPECTRUM_VAL", "a16_block_h_spectrum entry mismatch")

    e8_sp = p1["e8_block_h_spectrum"]
    exp_e8 = ((1, 4, 2), (2, 2, 1), (4, 4, 2), (8, 2, 1))
    if type(e8_sp) is not list or len(e8_sp) != len(exp_e8):
        raise oracle.HadamardLawError("P1_E8_SPECTRUM_SHAPE", "e8_block_h_spectrum shape mismatch")
    for row, (b, nsq, rho) in zip(e8_sp, exp_e8):
        r = _require_dict_exact_keys(row, _P1_E8_ROW_KEYS, "e8_spectrum_row")
        if (
            oracle.require_exact_int(r["block_size"], "b") != b
            or oracle.require_exact_int(r["shortest_axis_norm_sq"], "nsq") != nsq
            or oracle.require_exact_int(r["rho_axis"], "rho") != rho
        ):
            raise oracle.HadamardLawError("P1_E8_SPECTRUM_VAL", "e8_block_h_spectrum entry mismatch")

    dc = _require_dict_exact_keys(p1["coherent_dc_concentration_factors"], _P1_DC_KEYS, "dc_factors")
    if (
        oracle.require_exact_int(dc["h16"], "dc_h16") != 16
        or oracle.require_exact_int(dc["h8"], "dc_h8") != 8
        or oracle.require_exact_int(dc["block_h2"], "dc_bh2") != 2
    ):
        raise oracle.HadamardLawError("P1_DC_MISMATCH", "coherent DC concentration factors mismatch")

    i8 = _require_dict_exact_keys(p1["block_h2_int8_bounds"], _P1_INT8_KEYS, "int8_bounds")
    m39 = oracle.require_exact_int(i8["max_abs_block_h2_for_39"], "m39")
    imax = oracle.require_exact_int(i8["int8_max"], "imax")
    m72 = oracle.require_exact_int(i8["ball_72b_max_abs"], "m72")
    r72 = oracle.require_exact_int(i8["ball_72b_r2"], "r72")
    m18 = oracle.require_exact_int(i8["ball_18b_max_abs"], "m18")
    r18 = oracle.require_exact_int(i8["ball_18b_r2"], "r18")
    if (
        m39 != 78
        or imax != 127
        or m72 != 72
        or r72 != 5213
        or m18 != 34
        or r18 != 1207
        or oracle.require_exact_bool(i8["all_fit_int8"], "all_fit_int8") is not True
    ):
        raise oracle.HadamardLawError("P1_INT8_MISMATCH", "Block-H2 INT8 peak bounds mismatch")


def _audit_pillar2(p2_raw: Any) -> None:
    p2 = _require_dict_exact_keys(p2_raw, _P2_KEYS, "pillar2")
    sp = _require_dict_exact_keys(p2["single_spike_3_2_e0"], _P2_SPIKE_KEYS, "single_spike")
    sp_ax = oracle.parse_qsqrt2(sp["a16_axis_err_sq"], "sp_ax")
    sp_bh2 = oracle.parse_qsqrt2(sp["a16_block_h2_err_sq"], "sp_bh2")
    sp_h16 = oracle.parse_qsqrt2(sp["a16_h16_err_sq"], "sp_h16")
    sp_bh4 = oracle.parse_qsqrt2(sp["a16_block_h4_err_sq"], "sp_bh4")
    sp_e8_ax = oracle.parse_qsqrt2(sp["e8_axis_err_sq"], "sp_e8_ax")
    sp_e8_h8 = oracle.parse_qsqrt2(sp["e8_h8_err_sq"], "sp_e8_h8")
    if (
        sp_ax != oracle.QSqrt2(Fraction(1, 4), 0)
        or sp_bh2 != oracle.QSqrt2(Fraction(41, 4), -6)
        or sp_h16 != oracle.QSqrt2(Fraction(9, 4), 0)
        or sp_bh4 != oracle.QSqrt2(Fraction(1, 4), 0)
        or sp_e8_ax != oracle.QSqrt2(Fraction(1, 4), 0)
        or sp_e8_h8 != oracle.QSqrt2(Fraction(17, 4), -3)
        or oracle.require_exact_bool(sp["a16_strict_ordering_holds"], "sp_a16_ord") is not True
        or oracle.require_exact_bool(sp["e8_opposite_sign_holds"], "sp_e8_opp") is not True
    ):
        raise oracle.HadamardLawError("P2_SPIKE_MISMATCH", "single-spike exact witness mismatch")

    mw = _require_dict_exact_keys(p2["matrix_witness_zero_mse_gap"], _P2_MAT_KEYS, "matrix_witness")
    a16_ax_mse = oracle.parse_qsqrt2(mw["a16_axis_mse"], "a16_ax_mse")
    a16_h16_mse = oracle.parse_qsqrt2(mw["a16_h16_mse"], "a16_h16_mse")
    a16_bh4_mse = oracle.parse_qsqrt2(mw["a16_block_h4_mse"], "a16_bh4_mse")
    a16_ax_n = oracle.parse_qsqrt2(mw["a16_axis_nat_kl"], "a16_ax_n")
    a16_bh2_n = oracle.parse_qsqrt2(mw["a16_block_h2_nat_kl"], "a16_bh2_n")
    a16_h16_n = oracle.parse_qsqrt2(mw["a16_h16_nat_kl"], "a16_h16_n")
    a16_bh4_n = oracle.parse_qsqrt2(mw["a16_block_h4_nat_kl"], "a16_bh4_n")
    a16_ax_m = oracle.parse_qsqrt2(mw["a16_axis_matched_kl"], "a16_ax_m")
    a16_bh2_m = oracle.parse_qsqrt2(mw["a16_block_h2_matched_kl"], "a16_bh2_m")
    a16_h16_m = oracle.parse_qsqrt2(mw["a16_h16_matched_kl"], "a16_h16_m")
    e8_ax_n = oracle.parse_qsqrt2(mw["e8_axis_nat_kl"], "e8_ax_n")
    e8_h8_n = oracle.parse_qsqrt2(mw["e8_h8_nat_kl"], "e8_h8_n")
    e8_ax_m = oracle.parse_qsqrt2(mw["e8_axis_matched_kl"], "e8_ax_m")
    e8_h8_m = oracle.parse_qsqrt2(mw["e8_h8_matched_kl"], "e8_h8_m")
    exp_e8_h8_m = oracle.QSqrt2(Fraction(3745, 64), Fraction(-315, 8)) / oracle.QSqrt2(Fraction(29, 2), -9)
    if (
        a16_ax_mse != oracle.QSqrt2(Fraction(5, 2), 0)
        or a16_h16_mse != a16_ax_mse
        or a16_bh4_mse != oracle.QSqrt2(Fraction(1, 2), 0)
        or a16_ax_n != oracle.QSqrt2(Fraction(385, 64), 0)
        or a16_bh2_n != oracle.QSqrt2(Fraction(7585, 64), Fraction(-555, 8))
        or a16_h16_n != oracle.QSqrt2(Fraction(1465, 64), 0)
        or a16_bh4_n != oracle.QSqrt2(Fraction(185, 64), 0)
        or a16_ax_m != oracle.QSqrt2(Fraction(77, 32), 0)
        or a16_bh2_m != oracle.QSqrt2(Fraction(185, 32), 0)
        or a16_h16_m != oracle.QSqrt2(Fraction(293, 32), 0)
        or e8_ax_n != oracle.QSqrt2(Fraction(185, 64), 0)
        or e8_h8_n != oracle.QSqrt2(Fraction(3745, 64), Fraction(-315, 8))
        or e8_ax_m != oracle.QSqrt2(Fraction(185, 32), 0)
        or e8_h8_m != exp_e8_h8_m
        or oracle.require_exact_bool(mw["a16_zero_mse_gap"], "a16_zero_mse") is not True
        or oracle.require_exact_bool(mw["a16_natural_ordering_holds"], "a16_nat_ord") is not True
        or oracle.require_exact_bool(mw["a16_matched_ordering_holds"], "a16_mat_ord") is not True
        or oracle.require_exact_bool(mw["a16_matched_midpoint_exact"], "a16_mid") is not True
        or oracle.require_exact_bool(mw["e8_natural_opposite_sign_holds"], "e8_nat_opp") is not True
        or oracle.require_exact_bool(mw["e8_matched_opposite_sign_holds"], "e8_mat_opp") is not True
        or oracle.require_exact_bool(mw["a16_block_h4_beats_axis_on_witness"], "bh4_wins") is not True
    ):
        raise oracle.HadamardLawError("P2_MATRIX_MISMATCH", "matrix witness exact values mismatch")

    rf = _require_dict_exact_keys(p2["regime_flip_criterion"], _P2_REG_KEYS, "regime_flip")
    tr_ds = oracle.parse_exact_rational(rf["trace_delta_sigma"], "tr_ds")
    dl1 = oracle.parse_exact_rational(rf["h1_axis_spike_delta_kl"], "dl1")
    dl2 = oracle.parse_exact_rational(rf["h2_coherent_dc_delta_kl"], "dl2")
    if (
        tr_ds != 0
        or dl1 != Fraction(135, 8)
        or dl2 != Fraction(-30, 1)
        or oracle.require_exact_bool(rf["sign_reversal_verified"], "sign_rev") is not True
    ):
        raise oracle.HadamardLawError("P2_REGIME_MISMATCH", "regime-flip exact criterion mismatch")


def _audit_pillar3(p3_raw: Any) -> None:
    p3 = _require_dict_exact_keys(p3_raw, _P3_KEYS, "pillar3")
    if p3["anchor_filename"] != oracle.EMPIRICAL_ANCHOR_FILENAME:
        raise oracle.HadamardLawError("P3_ANCHOR_NAME", "anchor_filename mismatch")
    if not oracle.check_digest_eq(
        oracle.require_sha256_hex(p3["anchor_sha256"], "anchor_sha256"),
        oracle.EMPIRICAL_ANCHOR_SHA256,
    ):
        raise oracle.HadamardLawError("P3_ANCHOR_SHA", "anchor_sha256 mismatch")

    models = _require_dict_exact_keys(p3["models"], frozenset(oracle.EMPIRICAL_MODELS_PINNED.keys()), "models")
    tot_l = 0
    tot_r1 = 0
    tot_r2 = 0
    tot_bh2 = 0
    for m_name, exp_m in oracle.EMPIRICAL_MODELS_PINNED.items():
        m = _require_dict_exact_keys(models[m_name], _P3_MODEL_KEYS, m_name)
        ml = oracle.require_exact_int(m["total_layers"], "total_layers")
        mr1 = oracle.require_exact_int(m["regime1_count"], "regime1_count")
        mr2 = oracle.require_exact_int(m["regime2_count"], "regime2_count")
        mbh = oracle.require_exact_int(m["bh2_beats_h16_count"], "bh2_beats_h16_count")
        if (
            ml != exp_m["total_layers"]
            or mr1 != exp_m["regime1"]
            or mr2 != exp_m["regime2"]
            or mbh != exp_m["bh2_beats_h16"]
            or oracle.require_exact_bool(m["e8_h8_lt_e8_axis"], "e8_ord") is not True
            or oracle.require_exact_bool(m["a16_axis_lt_bh2_lt_h16"], "a16_ord") is not True
            or oracle.require_exact_bool(m["matched_preserves_a16_gap"], "mat_gap") is not True
        ):
            raise oracle.HadamardLawError("P3_MODEL_MISMATCH", f"model {m_name} counts mismatch")
        kl_map = _require_dict_exact_keys(m["wikitext2_kl_q"], _P3_KL_KEYS, "wikitext2_kl_q")
        for arm_k, dec_str in exp_m["wikitext2_kl"].items():
            if oracle.parse_exact_rational(kl_map[arm_k], arm_k) != oracle.parse_decimal_str_as_fraction(dec_str):
                raise oracle.HadamardLawError("P3_KL_MISMATCH", f"{m_name} {arm_k} KL rational mismatch")
        tot_l += ml
        tot_r1 += mr1
        tot_r2 += mr2
        tot_bh2 += mbh

    if (
        oracle.require_exact_int(p3["combined_total_layers"], "comb_l") != tot_l
        or oracle.require_exact_int(p3["combined_regime1_count"], "comb_r1") != tot_r1
        or oracle.require_exact_int(p3["combined_regime2_count"], "comb_r2") != tot_r2
        or oracle.require_exact_int(p3["combined_bh2_beats_h16_count"], "comb_bh2") != tot_bh2
        or oracle.require_exact_bool(p3["k_proj_both_models_h16_lt_98_100"], "k_lt_98") is not True
        or oracle.require_exact_bool(p3["selector_rule_checks_passed"], "sel_ok") is not True
    ):
        raise oracle.HadamardLawError("P3_COMBINED_MISMATCH", "combined empirical totals mismatch")

    proj_map = _require_dict_exact_keys(
        p3["by_projection_kind_combined"], frozenset(oracle.PROJECTION_KINDS), "by_projection_kind_combined"
    )
    for pk in oracle.PROJECTION_KINDS:
        sm = oracle.EMPIRICAL_MODELS_PINNED["HuggingFaceTB/SmolLM2-360M"]["by_projection_kind"][pk]
        qw = oracle.EMPIRICAL_MODELS_PINNED["Qwen/Qwen2.5-0.5B"]["by_projection_kind"][pk]
        pr = _require_dict_exact_keys(proj_map[pk], _P3_PROJ_KEYS, pk)
        if (
            oracle.require_exact_int(pr["layers"], "p_layers") != sm["layers"] + qw["layers"]
            or oracle.require_exact_int(pr["regime1"], "p_r1") != sm["regime1"] + qw["regime1"]
            or oracle.require_exact_int(pr["regime2"], "p_r2") != sm["regime2"] + qw["regime2"]
            or oracle.require_exact_int(pr["bh2_beats_h16"], "p_bh2") != sm["bh2_beats_h16"] + qw["bh2_beats_h16"]
            or oracle.require_exact_str(pr["selector_default_arm"], "p_sel") != oracle.select_layer_arm("A16", pk)
        ):
            raise oracle.HadamardLawError("P3_PROJ_MISMATCH", f"projection {pk} summary mismatch")


def _audit_pillar4(p4_raw: Any) -> None:
    p4 = _require_dict_exact_keys(p4_raw, _P4_KEYS, "pillar4")
    if p4["lean_companion_filename"] != oracle.LEAN_COMPANION_FILENAME:
        raise oracle.HadamardLawError("P4_LEAN_NAME", "lean_companion_filename mismatch")
    if not oracle.check_digest_eq(
        oracle.require_sha256_hex(p4["lean_companion_sha256"], "lean_sha"),
        oracle.LEAN_COMPANION_SHA256,
    ):
        raise oracle.HadamardLawError("P4_LEAN_SHA", "lean_companion_sha256 mismatch")
    thms = p4["lean_companion_theorems"]
    if type(thms) is not list or [oracle.require_exact_str(t, "thm") for t in thms] != list(oracle.LEAN_COMPANION_THEOREMS):
        raise oracle.HadamardLawError("P4_THEOREMS_MISMATCH", "lean_companion_theorems mismatch")
    if (
        oracle.require_exact_int(p4["lean_companion_theorem_count"], "thm_cnt") != len(oracle.LEAN_COMPANION_THEOREMS)
        or p4["novelty_tier"] != "N1"
        or p4["oracle_class"] != "exact"
        or oracle.require_exact_int(p4["max_int_bits_exact_cap"], "max_bits") != oracle.MAX_INT_BITS
        or oracle.require_exact_int(p4["max_int_digits_preparse_ceiling"], "max_digits") != oracle.MAX_INT_DIGITS
        or oracle.require_exact_bool(p4["measured_live_rtx3080_in_this_track"], "live_gpu") is not False
        or oracle.require_exact_bool(p4["measured_live_block_h4_llm_in_this_track"], "live_bh4") is not False
        or oracle.require_exact_bool(p4["zero_trust_path_floats"], "zero_floats") is not True
        or oracle.require_exact_bool(p4["zero_bare_asserts"], "zero_asserts") is not True
        or oracle.require_exact_bool(p4["zero_mathematical_rng"], "zero_rng") is not True
    ):
        raise oracle.HadamardLawError("P4_SCOPE_MISMATCH", "Pillar 4 CS-13 scope mismatch")
    lit = p4["prior_literature_n0"]
    if type(lit) is not list or [oracle.require_exact_str(x, "lit") for x in lit] != list(oracle.PRIOR_LITERATURE_N0):
        raise oracle.HadamardLawError("P4_LIT_MISMATCH", "prior_literature_n0 mismatch")


def secondary_audit_payload(payload: dict[str, Any]) -> None:
    """Field-complete strict-keyset algebraic cross-check (zero cached summary trust)."""
    oracle._check_json_value_strict(payload)
    p = _require_dict_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    if p["track_id"] != oracle.TRACK_ID:
        raise oracle.HadamardLawError("TRACK_ID_MISMATCH", "payload track_id mismatch")
    _audit_pillar1(p["pillar1_lattice_ray_anisotropy_and_hadamard"])
    _audit_pillar2(p["pillar2_exact_quadratic_kl_witnesses_and_regime_flip"])
    _audit_pillar3(p["pillar3_empirical_llm_bridge_and_selector"])
    _audit_pillar4(p["pillar4_lean4_companion_and_cs13_scope"])
    summ = _require_dict_exact_keys(p["summary"], _SUMMARY_KEYS, "summary")
    if (
        oracle.require_exact_bool(summ["all_pillars_verified"], "all_pillars_verified") is not True
        or oracle.require_exact_int(summ["target_max_cert_bytes"], "target_max_cert_bytes") != oracle.TARGET_MAX_CERT_BYTES
    ):
        raise oracle.HadamardLawError("SUMMARY_MISMATCH", "summary fields mismatch")


def verify_certificate_dict(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Verify certificate schema, constant-time digests, secondary audit, and optional full replay."""
    replay_flag = oracle.require_exact_bool(full_replay, "full_replay")
    c = _require_dict_exact_keys(cert, _CERT_KEYS, "certificate")
    if c["schema_version"] != oracle.SCHEMA_VERSION:
        raise oracle.HadamardLawError("SCHEMA_MISMATCH", "unexpected schema_version")
    if c["track_id"] != oracle.TRACK_ID:
        raise oracle.HadamardLawError("TRACK_ID_MISMATCH", "unexpected track_id")
    decl_p_sha = oracle.require_sha256_hex(c["payload_sha256"], "payload_sha256")
    decl_c_sha = oracle.require_sha256_hex(c["certificate_sha256"], "certificate_sha256")
    payload = _require_dict_exact_keys(c["payload"], _PAYLOAD_KEYS, "payload")

    actual_p_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(payload))
    if not oracle.check_digest_eq(decl_p_sha, actual_p_sha):
        raise oracle.HadamardLawError("PAYLOAD_SHA256_MISMATCH", "payload digest mismatch")
    cert_pre = {
        "schema_version": c["schema_version"],
        "track_id": c["track_id"],
        "payload_sha256": decl_p_sha,
        "payload": payload,
    }
    actual_c_sha = oracle.sha256_bytes(oracle.canonical_json_bytes(cert_pre))
    if not oracle.check_digest_eq(decl_c_sha, actual_c_sha):
        raise oracle.HadamardLawError("CERTIFICATE_SHA256_MISMATCH", "certificate digest mismatch")

    secondary_audit_payload(payload)

    if replay_flag:
        expected = oracle.build_certificate()
        if not oracle.check_digest_eq(decl_c_sha, expected["certificate_sha256"]):
            raise oracle.HadamardLawError("FULL_REPLAY_MISMATCH", "certificate disagrees with deterministic replay")

    return {
        "valid": True,
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": actual_p_sha,
        "certificate_sha256": actual_c_sha,
        "full_replay": replay_flag,
        "secondary_audit_passed": True,
    }


def verify_lean_companion_file(path: Any, *, lean_bin: str | None = None) -> dict[str, Any]:
    """Verify E8VsA16HadamardLaw.lean SHA-256 and live Lean 4 kernel + #print axioms."""
    raw = read_regular_bytes_capped(path, max_bytes=MAX_JSON_BYTES)
    file_sha = oracle.sha256_bytes(raw)
    if not oracle.check_digest_eq(file_sha, oracle.LEAN_COMPANION_SHA256):
        raise oracle.HadamardLawError("LEAN_SHA256_MISMATCH", f"Lean companion SHA-256 mismatch: {file_sha}")
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise oracle.HadamardLawError("LEAN_INVALID_UTF8", "Lean companion is not valid UTF-8") from exc

    if lean_bin is not None:
        resolved_lean = oracle.require_exact_str(lean_bin, "lean_bin")
    else:
        resolved_lean = shutil.which("lean")
    if resolved_lean is None:
        raise oracle.HadamardLawError("LEAN_NOT_FOUND", "lean binary not found for live kernel verification")

    with tempfile.TemporaryDirectory(prefix="e8a16_lean_ax_") as td:
        ax_path = Path(td) / "check_axioms.lean"
        ax_lines = [src, ""]
        for thm in oracle.LEAN_COMPANION_THEOREMS:
            ax_lines.append(f"#print axioms {oracle.LEAN_COMPANION_NAMESPACE}.{thm}")
        ax_path.write_text("\n".join(ax_lines) + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [resolved_lean, str(ax_path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise oracle.HadamardLawError("LEAN_EXEC_FAILED", f"cannot invoke lean: {exc}") from exc
        if proc.returncode != 0:
            raise oracle.HadamardLawError("LEAN_KERNEL_FAILED", f"lean exited {proc.returncode}")
        seen_theorems: set[str] = set()
        for m_ax in _AXIOM_BLOCK_RE.finditer(proc.stdout):
            full_name = m_ax.group(1)
            if m_ax.group(2) is None:
                for tok in m_ax.group(3).split(","):
                    ax = tok.strip()
                    if ax not in _ALLOWED_LEAN_AXIOMS:
                        raise oracle.HadamardLawError("LEAN_BANNED_AXIOM", f"disallowed axiom {ax!r} in {full_name}")
            seen_theorems.add(full_name)
        for thm in oracle.LEAN_COMPANION_THEOREMS:
            fq = f"{oracle.LEAN_COMPANION_NAMESPACE}.{thm}"
            if fq not in seen_theorems:
                raise oracle.HadamardLawError("LEAN_MISSING_THEOREM", f"theorem {fq!r} missing from #print axioms")
    return {
        "valid": True,
        "lean_companion_sha256": file_sha,
        "kernel_theorems_verified": len(oracle.LEAN_COMPANION_THEOREMS),
        "banned_axioms_found": 0,
    }


def verify_empirical_anchor_file(path: Any) -> dict[str, Any]:
    """Read-only SHA-256 and exact-Fraction parity check against live empirical results JSON."""
    raw = read_regular_bytes_capped(path, max_bytes=MAX_JSON_BYTES)
    file_sha = oracle.sha256_bytes(raw)
    if not oracle.check_digest_eq(file_sha, oracle.EMPIRICAL_ANCHOR_SHA256):
        raise oracle.HadamardLawError("ANCHOR_SHA256_MISMATCH", f"empirical anchor SHA-256 mismatch: {file_sha}")
    try:
        raw_txt = raw.decode("utf-8")
        oracle._precheck_json_int_digits(raw_txt)
        doc = json.loads(
            raw_txt,
            parse_float=oracle.parse_decimal_str_as_fraction,
            parse_constant=oracle._reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, oracle.HadamardLawError) as exc:
        raise oracle.HadamardLawError("ANCHOR_INVALID_JSON", f"cannot parse empirical anchor: {exc}") from exc
    if type(doc) is not dict or type(doc.get("models")) is not dict:
        raise oracle.HadamardLawError("ANCHOR_SCHEMA_MISMATCH", "anchor must be an object with a 'models' dict")
    models_doc = doc["models"]
    layers_checked = 0
    try:
        for m_name, exp_m in oracle.EMPIRICAL_MODELS_PINNED.items():
            md = models_doc[m_name]
            diag = md["per_layer_anisotropy_diagnostic"]
            if (
                diag["total_layers_scored"] != exp_m["total_layers"]
                or diag["regime1_axis_beats_h16_count"] != exp_m["regime1"]
                or diag["regime2_h16_beats_axis_count"] != exp_m["regime2"]
                or diag["block_h2_beats_h16_count"] != exp_m["bh2_beats_h16"]
            ):
                raise oracle.HadamardLawError("ANCHOR_COUNT_MISMATCH", f"anchor layer counts mismatch for {m_name}")
            for arm_k, dec_str in exp_m["wikitext2_kl"].items():
                if md["arms"][arm_k]["wikitext2"]["mean_kl"] != oracle.parse_decimal_str_as_fraction(dec_str):
                    raise oracle.HadamardLawError("ANCHOR_KL_MISMATCH", f"anchor KL mismatch for {m_name} {arm_k}")
            for pk, exp_p in exp_m["by_projection_kind"].items():
                pd = diag["by_projection_kind"][pk]
                if (
                    pd["layers"] != exp_p["layers"]
                    or pd["regime1_axis_beats_h16"] != exp_p["regime1"]
                    or pd["regime2_h16_beats_axis"] != exp_p["regime2"]
                    or pd["block_h2_beats_h16"] != exp_p["bh2_beats_h16"]
                    or pd["mean_h16_over_axis_quad_err"] != oracle.parse_decimal_str_as_fraction(exp_p["mean_h16_over_axis"])
                ):
                    raise oracle.HadamardLawError("ANCHOR_PROJ_MISMATCH", f"anchor projection mismatch for {m_name} {pk}")
            layers_checked += diag["total_layers_scored"]
    except oracle.HadamardLawError:
        raise
    except (KeyError, TypeError) as exc:
        raise oracle.HadamardLawError("ANCHOR_SCHEMA_MISMATCH", f"malformed anchor schema: {exc}") from exc
    return {
        "valid": True,
        "anchor_sha256": file_sha,
        "models_verified": len(oracle.EMPIRICAL_MODELS_PINNED),
        "layers_verified": layers_checked,
    }


# ---------------------------------------------------------------------------
# Strict Manual CLI Parser (Zero Abbreviation, Fail-Closed rc=2)
# ---------------------------------------------------------------------------

def _parse_cli_args(argv: list[str]) -> dict[str, Any]:
    mode = "self_check"
    target_path: str | None = None
    as_json = False
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg == "--standalone":
            if mode != "self_check":
                raise oracle.HadamardLawError("CLI_CONFLICT", "conflicting CLI mode flags")
            mode = "standalone"
            i += 1
        elif arg == "--json":
            if as_json:
                raise oracle.HadamardLawError("CLI_DUPLICATE", "duplicate --json flag")
            as_json = True
            i += 1
        elif arg in ("--write", "--check", "--verify-lean", "--verify-anchor"):
            if mode != "self_check":
                raise oracle.HadamardLawError("CLI_CONFLICT", "conflicting CLI mode flags")
            if i + 1 >= n:
                raise oracle.HadamardLawError("CLI_MISSING_ARG", f"{arg} requires a path argument")
            cand = argv[i + 1]
            if not cand or cand.startswith("-"):
                raise oracle.HadamardLawError("CLI_INVALID_PATH", f"{arg} path must not be empty or flag-like")
            mode = arg[2:].replace("-", "_")
            target_path = cand
            i += 2
        else:
            raise oracle.HadamardLawError("CLI_UNKNOWN_ARG", f"unknown argument {arg!r}")
    return {"mode": mode, "path": target_path, "json": as_json}


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    try:
        if argv is None:
            raw_argv = list(sys.argv[1:])
        else:
            if type(argv) not in (list, tuple):
                raise oracle.HadamardLawError("CLI_INVALID_ARGV", "argv must be list, tuple, or None")
            raw_argv = [oracle.require_exact_str(a, "argv") for a in argv]
        opts = _parse_cli_args(raw_argv)
        mode = opts["mode"]
        if mode == "self_check":
            cert = oracle.build_certificate()
            res = verify_certificate_dict(cert, full_replay=True)
            if opts["json"]:
                sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
            else:
                sys.stdout.write(f"OK {res['schema_version']} {res['certificate_sha256']}\n")
            return 0
        if mode == "standalone":
            cert = oracle.build_certificate()
            verify_certificate_dict(cert, full_replay=True)
            sys.stdout.write(oracle.canonical_json_bytes(cert).decode("ascii") + "\n")
            return 0
        if mode == "write":
            cert = oracle.build_certificate()
            verify_certificate_dict(cert, full_replay=True)
            out_bytes = oracle.canonical_json_bytes(cert) + b"\n"
            write_bytes_atomic_0600(opts["path"], out_bytes)
            if opts["json"]:
                sys.stdout.write(
                    oracle.canonical_json_bytes({"written": True, "certificate_sha256": cert["certificate_sha256"]}).decode("ascii")
                    + "\n"
                )
            else:
                sys.stdout.write(f"WROTE {cert['certificate_sha256']}\n")
            return 0
        if mode == "check":
            raw_bytes = read_regular_bytes_capped(opts["path"], max_bytes=MAX_JSON_BYTES)
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise oracle.HadamardLawError("INVALID_UTF8", "certificate file is not valid UTF-8") from exc
            parsed = oracle.parse_strict_json_text(raw_text)
            canon = oracle.canonical_json_bytes(parsed)
            if raw_bytes not in (canon, canon + b"\n"):
                raise oracle.HadamardLawError("NON_CANONICAL_BYTES", "certificate file is not canonical JSON")
            res = verify_certificate_dict(parsed, full_replay=True)
            if opts["json"]:
                sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
            else:
                sys.stdout.write(f"VALID {res['certificate_sha256']}\n")
            return 0
        if mode == "verify_lean":
            res = verify_lean_companion_file(opts["path"])
            if opts["json"]:
                sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
            else:
                sys.stdout.write(f"LEAN_VALID {res['lean_companion_sha256']} {res['kernel_theorems_verified']}\n")
            return 0
        res = verify_empirical_anchor_file(opts["path"])
        if opts["json"]:
            sys.stdout.write(oracle.canonical_json_bytes(res).decode("ascii") + "\n")
        else:
            sys.stdout.write(f"ANCHOR_VALID {res['anchor_sha256']} {res['layers_verified']}\n")
        return 0
    except oracle.HadamardLawError as exc:
        sys.stderr.write(f"ERROR {exc.code}: {exc.message}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
