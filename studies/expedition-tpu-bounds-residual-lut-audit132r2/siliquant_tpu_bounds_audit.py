#!/usr/bin/env python3
"""Track S32: Zone-B Hardened CLI, Full Pure-Q Replay & Secondary Algebraic Cross-Check.

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
import secrets
import stat
import sys
import threading
from typing import Any, Sequence

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from siliquant_tpu_bounds_oracle import (
    INT8_MAX,
    INT16_MAX,
    MAX_JSON_BYTES,
    SCHEMA_VERSION,
    TPU_V5E_VMEM_BYTES,
    TRACK_ID,
    SiliquantTpuBoundsError,
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
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TMP_LOCK = threading.Lock()
_TMP_COUNTER = 0


def _next_tmp_name(leaf: str) -> str:
    global _TMP_COUNTER
    with _TMP_LOCK:
        _TMP_COUNTER += 1
        seq = _TMP_COUNTER
    return f".{leaf}.s32tmp.{os.getpid()}.{threading.get_ident()}.{seq}.{secrets.token_hex(8)}"


# ---------------------------------------------------------------------------
# Race-Free Zone-B File I/O (Per-Component dir_fd/openat + Atomic os.replace)
# ---------------------------------------------------------------------------

def _open_dir_chain(anchor: str, parts: Sequence[str]) -> int:
    """Walk directory components atomically via `openat(..., O_DIRECTORY | O_NOFOLLOW)`."""
    try:
        dfd = os.open(anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise SiliquantTpuBoundsError("PATH_STAT_ERROR", f"cannot open anchor dir: {type(exc).__name__}") from exc

    for part in parts:
        try:
            nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
        except OSError as exc:
            os.close(dfd)
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise SiliquantTpuBoundsError("SYMLINK_FORBIDDEN", "symlink or non-dir forbidden in directory path") from exc
            if exc.errno == errno.ENOENT:
                raise SiliquantTpuBoundsError("PATH_NOT_FOUND", "directory component not found") from exc
            raise SiliquantTpuBoundsError("PATH_STAT_ERROR", f"cannot traverse dir: {type(exc).__name__}") from exc
        os.close(dfd)
        dfd = nxt
    return dfd


def _validate_path_and_split(path: Any) -> tuple[str, list[str]]:
    if type(path) is not _EXACT_PATH_TYPE:
        raise SiliquantTpuBoundsError("INVALID_PATH_TYPE", "path must be exact pathlib.Path")
    raw_str = require_exact_str(str(path), "path")
    if "\x00" in raw_str:
        raise SiliquantTpuBoundsError("INVALID_PATH", "path contains NUL byte")

    if not path.is_absolute():
        try:
            Path.cwd()
        except OSError as exc:
            raise SiliquantTpuBoundsError("CWD_UNAVAILABLE", f"cannot resolve cwd: {type(exc).__name__}") from exc
        pwd_env = require_exact_str(os.environ.get("PWD", ""), "PWD", non_empty=False)
        if pwd_env.startswith("/"):
            pwd_path = Path(pwd_env)
            pwd_parts = [p for p in pwd_path.parts[1:] if p not in ("", ".")]
            try:
                pwd_fd = _open_dir_chain(pwd_path.anchor, pwd_parts)
            except SiliquantTpuBoundsError as exc:
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
            raise SiliquantTpuBoundsError("PATH_STAT_ERROR", f"cannot lstat leaf: {type(exc).__name__}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise SiliquantTpuBoundsError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
    finally:
        os.close(dfd)
    return path


def _open_leaf_for_read(path: Path) -> int:
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise SiliquantTpuBoundsError("NON_REGULAR_FILE", "target path refers to a directory")
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SiliquantTpuBoundsError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf") from exc
            raise SiliquantTpuBoundsError("FILE_READ_ERROR", f"cannot open file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


def read_regular_file_bytes(path: Path) -> bytes:
    fd = _open_leaf_for_read(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SiliquantTpuBoundsError("NON_REGULAR_FILE", "target is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_JSON_BYTES:
                raise SiliquantTpuBoundsError("FILE_TOO_LARGE", f"file exceeds {MAX_JSON_BYTES} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    except SiliquantTpuBoundsError:
        raise
    except OSError as exc:
        raise SiliquantTpuBoundsError("FILE_READ_ERROR", f"failed reading file: {type(exc).__name__}") from exc
    finally:
        os.close(fd)


def write_regular_file_bytes(path: Path, data: bytes) -> None:
    if type(data) is not bytes:
        raise SiliquantTpuBoundsError("INVALID_WRITE_DATA", "data must be exact bytes")
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise SiliquantTpuBoundsError("NON_REGULAR_FILE", "target path refers to a directory")
    leaf = parts[-1]
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise SiliquantTpuBoundsError("FILE_WRITE_ERROR", f"cannot lstat target: {type(exc).__name__}") from exc
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise SiliquantTpuBoundsError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
            if not stat.S_ISREG(st.st_mode):
                raise SiliquantTpuBoundsError("NON_REGULAR_FILE", "cannot write to non-regular file")
            if st.st_nlink > 1:
                raise SiliquantTpuBoundsError("HARDLINK_FORBIDDEN", "cannot write through hardlinked file")

        tmp_fd: int | None = None
        tmp_name = ""
        for _ in range(16):
            cand = _next_tmp_name(leaf)
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
                raise SiliquantTpuBoundsError("FILE_WRITE_ERROR", f"cannot create temp file: {type(exc).__name__}") from exc
        if tmp_fd is None:
            raise SiliquantTpuBoundsError("FILE_WRITE_ERROR", "temp file collision limit exceeded")

        try:
            try:
                view = memoryview(data)
                written = 0
                while written < len(data):
                    written += os.write(tmp_fd, view[written:])
                os.fsync(tmp_fd)
                if os.fstat(tmp_fd).st_nlink != 1:
                    raise SiliquantTpuBoundsError("HARDLINK_FORBIDDEN", "temp file link count != 1")
            finally:
                os.close(tmp_fd)
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            os.fsync(dfd)
        except SiliquantTpuBoundsError:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
            raise SiliquantTpuBoundsError("FILE_WRITE_ERROR", f"failed writing file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------
# Secondary Algebraic Invariant Cross-Check Over All Five Pillars
# ---------------------------------------------------------------------------

def _require_dict(mapping: dict[str, Any], key: str, err_code: str) -> dict[str, Any]:
    if key not in mapping or type(mapping[key]) is not dict:
        raise SiliquantTpuBoundsError(err_code, f"missing or invalid object key: {key}")
    return mapping[key]


def _require_flag(mapping: dict[str, Any], key: str, expected: bool, err_code: str) -> None:
    if require_exact_bool(mapping.get(key), key) is not expected:
        raise SiliquantTpuBoundsError(err_code, f"invariant {key} must be {expected}")


def _require_int_list(mapping: dict[str, Any], key: str, exact_len: int, err_code: str) -> list[int]:
    val = mapping.get(key)
    if type(val) is not list or len(val) != exact_len:
        raise SiliquantTpuBoundsError(err_code, f"{key} must be list of length {exact_len}")
    return [require_exact_int(x, f"{key}[{i}]") for i, x in enumerate(val)]


def _audit_regime_types(rec: dict[str, Any]) -> None:
    for int_k in (
        "m_max",
        "w1_count",
        "w1_bit_width",
        "max_abs_z",
        "max_abs_z_witness_shell",
        "wht8_max",
        "wht8_witness_shell",
        "block_h2_max",
        "block_h2_witness_shell",
        "cs_dot_8w_int8_sym127",
        "cs_dot_32w_int8_sym127",
        "exact_l1_32w_attainable",
        "exact_dot_32w_int8_sym127",
        "exact_dot_32w_int8_asym128",
    ):
        require_exact_int(rec.get(int_k), int_k)
    for vec_k in ("max_abs_z_witness", "wht8_witness", "block_h2_witness"):
        _require_int_list(rec, vec_k, 8, "AUDIT_P1_WITNESS")
    blks = rec.get("exact_dot_32w_witness_blocks")
    if type(blks) is not list or len(blks) != 4:
        raise SiliquantTpuBoundsError("AUDIT_P1_WITNESS", "exact_dot_32w_witness_blocks must be 4 blocks")
    for b_idx, blk in enumerate(blks):
        if type(blk) is not list or len(blk) != 8:
            raise SiliquantTpuBoundsError("AUDIT_P1_WITNESS", f"witness block {b_idx} must have length 8")
        for c_idx, c in enumerate(blk):
            require_exact_int(c, f"witness_blocks[{b_idx}][{c_idx}]")


def secondary_algebraic_audit(payload: Any) -> dict[str, Any]:
    """Secondary algebraic invariant cross-check over all five Track S32 pillars."""
    if type(payload) is not dict:
        raise SiliquantTpuBoundsError("AUDIT_PAYLOAD_TYPE", "payload must be dict")
    # Law-4 epistemic honesty flags
    for key in (
        "measured_pallas_tok_s",
        "measured_live_model_kl_residual_lut",
        "measured_live_model_safety_benchmarks",
    ):
        _require_flag(payload, key, False, "AUDIT_LAW4_FLAG")
    for key in ("zero_trust_path_floats", "zero_bare_asserts", "zero_rng"):
        _require_flag(payload, key, True, "AUDIT_HYGIENE_FLAG")

    # Pillar 1: Pre-cast dynamic range, Block-H2 vs WHT_8 INT8 separation, BF16 & accumulator width
    p1 = _require_dict(payload, "pillar1_precast_overflow_separation", "AUDIT_P1_MISSING")
    for key in (
        "h8_all_rows_preserve_mod4",
        "int8_separation_block_h2_vs_wht8_136b_certified",
        "bf16_half_integers_neg128_to_pos128_all_513_exact",
        "bf16_boundary_257_inexact_verified",
    ):
        _require_flag(p1, key, True, "AUDIT_P1_FLAG")
    regimes = _require_dict(p1, "regimes", "AUDIT_P1_REGIMES")
    expected_regimes = {"EBG32:b88s8", "EBG32:b136s8", "1block_E8_3.0bpw", "1block_E8_4.5bpw"}
    if set(regimes.keys()) != expected_regimes:
        raise SiliquantTpuBoundsError("AUDIT_P1_REGIME_COUNT", "regimes must contain exactly the 4 certified regimes")
    for rname in expected_regimes:
        _audit_regime_types(_require_dict(regimes, rname, "AUDIT_P1_REGIMES"))
    r88 = regimes["EBG32:b88s8"]
    r136 = regimes["EBG32:b136s8"]
    if r136["block_h2_max"] > INT8_MAX:
        raise SiliquantTpuBoundsError("AUDIT_P1_BH2", "136b Block-H2 must stay <= 127")
    if r136["wht8_max"] <= INT8_MAX:
        raise SiliquantTpuBoundsError("AUDIT_P1_WHT8", "136b WHT_8 must exceed 127")
    if r88["cs_dot_32w_int8_sym127"] > INT16_MAX:
        raise SiliquantTpuBoundsError("AUDIT_P1_88_ACC", "88b 32-w dot must fit in int16")
    if r136["exact_dot_32w_int8_sym127"] <= INT16_MAX:
        raise SiliquantTpuBoundsError("AUDIT_P1_136_ACC", "136b 32-w dot must overflow int16")
    acc = _require_dict(p1, "accumulator_width_k16384", "AUDIT_P1_ACC")
    for int_k in ("hidden_dim_k", "superblocks_32w", "peak_136b_cs_sym127", "peak_136b_attainable_sym127", "peak_136b_attainable_asym128"):
        require_exact_int(acc.get(int_k), int_k)
    _require_flag(acc, "headroom_exceeds_100x_all_3", True, "AUDIT_P1_HEADROOM")
    if parse_exact_rational(acc.get("int32_headroom_cs"), "int32_headroom_cs") <= 100:
        raise SiliquantTpuBoundsError("AUDIT_P1_HEADROOM_VAL", "int32 headroom must exceed 100x")

    # Pillar 2: VMEM, static WCET & TPU v5e 3-way roofline
    p2 = _require_dict(payload, "pillar2_wcet_vmem_tpu_roofline", "AUDIT_P2_MISSING")
    if require_exact_int(p2.get("tpu_v5e_vmem_bytes"), "tpu_v5e_vmem_bytes") != TPU_V5E_VMEM_BYTES:
        raise SiliquantTpuBoundsError("AUDIT_P2_VMEM", "VMEM constant mismatch")
    _require_flag(p2, "guide_a_136b_leaves_gt_98_8_pct_free", True, "AUDIT_P2_FREE")
    roof = _require_dict(p2, "tpu_v5e_three_way_roofline", "AUDIT_P2_ROOF_OBJ")
    for int_k in ("hbm_bandwidth_gbs", "bf16_mxu_tflops", "int8_mxu_tops"):
        require_exact_int(roof.get(int_k), int_k)
    b_bf16_450 = parse_exact_rational(roof.get("b_bf16_ridge_4_50"), "b_bf16_450")
    b_int8_450 = parse_exact_rational(roof.get("b_int8_ridge_4_50"), "b_int8_450")
    if b_int8_450 != 2 * b_bf16_450:
        raise SiliquantTpuBoundsError("AUDIT_P2_RIDGE_2X", "INT8 ridge must equal 2x BF16 ridge")
    if parse_exact_rational(roof.get("r_decode_breakeven_gwps"), "r_decode") != Fraction(1365, 1):
        raise SiliquantTpuBoundsError("AUDIT_P2_RDECODE", "break-even decode rate must be 1365 G w/s")
    for key in (
        "int8_doubles_bandwidth_regime",
        "batch_73_bf16_zero_gain_certified",
        "batch_73_int8_still_bandwidth_bound",
    ):
        _require_flag(roof, key, True, "AUDIT_P2_ROOF")

    # Pillar 3: 2-Stage Residual Flat-LUT E_8 codec & contraction
    p3 = _require_dict(payload, "pillar3_residual_flat_lut_e8_codec", "AUDIT_P3_MISSING")
    l12 = _require_dict(p3, "lut_12bit_32kib", "AUDIT_P3_L12_OBJ")
    l16 = _require_dict(p3, "lut_16bit_512kib", "AUDIT_P3_L16_OBJ")
    if require_exact_int(l12.get("enumerated_points"), "enumerated_points") != 2401:
        raise SiliquantTpuBoundsError("AUDIT_P3_LUT12", "12-bit LUT must have 2401 points")
    if require_exact_int(l16.get("points"), "points") != 56881:
        raise SiliquantTpuBoundsError("AUDIT_P3_LUT16", "16-bit LUT must have 56881 points")
    for key in (
        "all_encode_decode_zero_mismatch",
        "all_6_opt_ratios_le_1_4",
        "aggregate_fixed_sf_1_3_lt_1_4",
        "aggregate_opt_ratio_le_1_8",
    ):
        _require_flag(p3, key, True, "AUDIT_P3_CODEC")
    if parse_exact_rational(p3.get("aggregate_opt_ratio"), "agg_opt") > Fraction(1, 8):
        raise SiliquantTpuBoundsError("AUDIT_P3_RATIO", "aggregate optimal ratio must be <= 1/8")
    suite = p3.get("codec_distortion_suite")
    if type(suite) is not list or len(suite) != 6:
        raise SiliquantTpuBoundsError("AUDIT_P3_SUITE", "codec_distortion_suite must be list of 6 vectors")
    for item in suite:
        if type(item) is not dict:
            raise SiliquantTpuBoundsError("AUDIT_P3_SUITE", "suite item must be dict")
        require_exact_int(item.get("idx_c_opt"), "idx_c_opt")
        require_exact_int(item.get("idx_f_opt"), "idx_f_opt")

    # Pillar 4: Spectral / Lipschitz tension-drift enclosure & Block-H2 anisotropy
    p4 = _require_dict(payload, "pillar4_spectral_lipschitz_enclosure", "AUDIT_P4_MISSING")
    for key in (
        "h8_orthogonal_8I",
        "block_h2_orthogonal_2I",
        "single_stage_zero_unpack_residual_zero",
        "single_stage_bareiss_spectral_pd",
        "residual_2stage_exact_cross_identity",
        "residual_2stage_cauchy_schwarz_holds",
        "residual_2stage_direct_le_ceiling",
        "residual_2stage_bareiss_spectral_pd",
    ):
        _require_flag(p4, key, True, "AUDIT_P4_SPEC")
    aniso = _require_dict(p4, "anisotropy_operator_separation", "AUDIT_P4_ANISO_OBJ")
    for key in ("isotropic_trace_invariance_exact", "block_h2_exact_2x_flip", "h8_exact_8x_flip"):
        _require_flag(aniso, key, True, "AUDIT_P4_ANISO")

    # Pillar 5: Safety/refusal boundary preservation
    p5 = _require_dict(payload, "pillar5_safety_refusal_margin_preservation", "AUDIT_P5_MISSING")
    if require_exact_int(p5.get("matrix_dim_n"), "matrix_dim_n") != 8 or require_exact_int(p5.get("matrix_dim_d"), "matrix_dim_d") != 8:
        raise SiliquantTpuBoundsError("AUDIT_P5_DIM", "safety witness matrix must be 8x8")
    for key in (
        "scalar_breaks_refusal_boundary",
        "residual_e8_preserves_refusal_boundary",
        "sqrt_rho_ceiling_tightening_exact",
        "observed_drift_sq_within_worst_ceiling",
        "rank1_adversary_tightness_exact",
        "isotropic_variance_identity_exact",
        "dw_bareiss_spectral_pd",
        "token_embd_immutable_row_hash_required",
    ):
        _require_flag(p5, key, True, "AUDIT_P5_SAFETY")
    _require_flag(p5, "token_embd_quantized_by_block_lut", False, "AUDIT_P5_EMBD")
    if parse_exact_rational(p5.get("mse_reduction_rho"), "rho") >= Fraction(1, 100):
        raise SiliquantTpuBoundsError("AUDIT_P5_RHO", "expected > 100x MSE reduction on witness")

    return {
        "secondary_audit_passed": True,
        "regimes_checked": len(regimes),
        "lut12_points_checked": l12["enumerated_points"],
        "suite_vectors_checked": len(suite),
    }


def check_certificate_file(path: Path) -> dict[str, Any]:
    raw = read_regular_file_bytes(path)
    if b"\r" in raw:
        raise SiliquantTpuBoundsError("CRLF_FORBIDDEN", "CRLF line endings forbidden in certificate")
    if not raw.endswith(b"\n"):
        raise SiliquantTpuBoundsError("MISSING_TRAILING_LF", "certificate must end with a single LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SiliquantTpuBoundsError("INVALID_UTF8", "certificate is not valid UTF-8") from exc
    cert = parse_strict_json_text(text, require_object=True)
    canonical_raw = canonical_json_bytes(cert)
    if raw != canonical_raw + b"\n":
        raise SiliquantTpuBoundsError("NON_CANONICAL_BYTES", "certificate bytes are not canonical ASCII JSON + LF")
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
        raise SiliquantTpuBoundsError("INVALID_ARGV", "argv must be exact list/tuple of strict str")
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
                raise SiliquantTpuBoundsError("DUPLICATE_FLAG", "duplicate --json flag")
            as_json = True
            i += 1
        elif tok == "--standalone":
            if mode is not None:
                raise SiliquantTpuBoundsError("CONFLICTING_MODE", "multiple mode flags forbidden")
            mode = "standalone"
            i += 1
        elif tok in ("--check", "--write"):
            if mode is not None:
                raise SiliquantTpuBoundsError("CONFLICTING_MODE", "multiple mode flags forbidden")
            if i + 1 >= n:
                raise SiliquantTpuBoundsError("MISSING_PATH_ARG", f"{tok} requires a path argument")
            val = argv[i + 1]
            if not val or val.startswith("-"):
                raise SiliquantTpuBoundsError("INVALID_PATH_ARG", f"{tok} path cannot be empty or flag-like")
            mode = "check" if tok == "--check" else "write"
            target = Path(val)
            i += 2
        else:
            raise SiliquantTpuBoundsError("UNKNOWN_FLAG", f"unrecognized CLI argument: {tok}")

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
    except SiliquantTpuBoundsError as exc:
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
