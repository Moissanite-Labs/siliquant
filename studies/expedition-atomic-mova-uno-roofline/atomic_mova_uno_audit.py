#!/usr/bin/env python3
"""Track S45: R2-Grade Zone-B Hardened CLI, Full Pure-Q Replay & Field-Complete Secondary Audit.

Pure Python 3 standard library: zero floats, zero bare asserts, zero mathematical RNG.
Zone-B hardening (baked in from line 1):
  - Concrete pathlib.Path check + NUL-byte and lone UTF-16 surrogate path refusal
  - Deleted-CWD OSError wrapping -> CWD_UNAVAILABLE (rc=2)
  - Atomic dir_fd PWD walk + per-component openat(O_RDONLY | O_DIRECTORY | O_NOFOLLOW) (zero os.path.islink)
  - Leaf O_RDONLY | O_NOFOLLOW | O_NONBLOCK + fstat S_ISREG + 64 KiB streaming capped reads (max_bytes <= 64 MiB)
  - Atomic write: type(data) is bytes BEFORE dir/temp open -> target lstat (st_nlink > 1 -> HARDLINK_FORBIDDEN)
    -> thread-locked monotonic + CSPRNG (secrets.token_hex(8)) 0o600 O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK
    temp file in same dir_fd -> bounded 32-attempt EEXIST retry -> os.fsync(tmp_fd)
    -> pre-replace os.fstat(tmp_fd).st_nlink == 1 -> os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd) -> os.fsync(dfd)
  - Strict manual CLI parser (no flag abbreviations, rejecting space- & equals-syntax flag-like/empty paths -> rc=2)
  - Full pure-Q replay + field-complete strict-nested-key secondary_audit_payload (S19c 100% on ALL 4 surfaces)
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

from atomic_mova_uno_oracle import (
    MAX_JSON_BYTES,
    SCHEMA_VERSION,
    TRACK_ID,
    _PAYLOAD_KEYS,
    AtomicMovaUnoError,
    build_standalone_certificate,
    canonical_json_bytes,
    format_decimal_trunc,
    harmonic_q,
    parse_exact_rational,
    parse_strict_json_text,
    piecewise_linear_latency_ms,
    require_exact_bool,
    require_exact_int,
    require_exact_str,
    require_sha256_hex,
    validate_certificate_payload,
)

MAX_ALLOWED_CAP_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 65_536
_EXACT_PATH_TYPE = type(Path("."))
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TMP_LOCK = threading.Lock()
_TMP_COUNTER = 0


def _next_tmp_name(leaf: str) -> str:
    global _TMP_COUNTER
    with _TMP_LOCK:
        _TMP_COUNTER += 1
        seq = _TMP_COUNTER
    return f".{leaf}.s45tmp.{os.getpid()}.{threading.get_ident()}.{seq}.{secrets.token_hex(8)}"


# ---------------------------------------------------------------------------
# Race-Free Zone-B File I/O (Per-Component dir_fd/openat + Atomic os.replace)
# ---------------------------------------------------------------------------

def _open_dir_chain(anchor: str, parts: Sequence[str]) -> int:
    """Walk directory components atomically via `openat(..., O_DIRECTORY | O_NOFOLLOW)`."""
    try:
        dfd = os.open(anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise AtomicMovaUnoError("PATH_STAT_ERROR", f"cannot open anchor dir: {type(exc).__name__}") from exc

    for part in parts:
        try:
            nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
        except OSError as exc:
            os.close(dfd)
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise AtomicMovaUnoError("SYMLINK_FORBIDDEN", "symlink or non-dir forbidden in path") from exc
            if exc.errno == errno.ENOENT:
                raise AtomicMovaUnoError("PATH_NOT_FOUND", "directory component not found") from exc
            raise AtomicMovaUnoError("PATH_STAT_ERROR", f"cannot traverse dir: {type(exc).__name__}") from exc
        os.close(dfd)
        dfd = nxt
    return dfd


def _check_pwd_no_symlinks() -> None:
    pwd_env = require_exact_str(os.environ.get("PWD", ""), "PWD", non_empty=False)
    if pwd_env.startswith("/"):
        pwd_path = Path(pwd_env)
        pwd_parts = [p for p in pwd_path.parts[1:] if p not in ("", ".")]
        try:
            pwd_fd = _open_dir_chain(pwd_path.anchor, pwd_parts)
        except AtomicMovaUnoError as exc:
            if exc.code != "PATH_NOT_FOUND":
                raise
        else:
            os.close(pwd_fd)


def _validate_path_and_split(path: Any) -> tuple[str, list[str]]:
    if type(path) is not _EXACT_PATH_TYPE:
        raise AtomicMovaUnoError("INVALID_PATH_TYPE", "path must be exact pathlib.Path")
    raw_str = require_exact_str(str(path), "path")
    if "\x00" in raw_str:
        raise AtomicMovaUnoError("INVALID_PATH", "path contains NUL byte")

    if not path.is_absolute():
        try:
            Path.cwd()
        except OSError as exc:
            raise AtomicMovaUnoError("CWD_UNAVAILABLE", f"cannot resolve cwd: {type(exc).__name__}") from exc
        _check_pwd_no_symlinks()
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
            raise AtomicMovaUnoError("PATH_STAT_ERROR", f"cannot lstat leaf: {type(exc).__name__}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise AtomicMovaUnoError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
    finally:
        os.close(dfd)
    return path


def _open_leaf_for_read(path: Path) -> int:
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise AtomicMovaUnoError("NON_REGULAR_FILE", "target path refers to a directory")
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise AtomicMovaUnoError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf") from exc
            raise AtomicMovaUnoError("FILE_READ_ERROR", f"cannot open file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


def read_regular_bytes_capped(path: Path, max_bytes: Any = MAX_JSON_BYTES) -> bytes:
    """Read regular file bytes via atomic dir_fd + O_NOFOLLOW|O_NONBLOCK in bounded 64 KiB streaming chunks."""
    cap = require_exact_int(max_bytes, "max_bytes", min_val=1, max_val=MAX_ALLOWED_CAP_BYTES)
    fd = _open_leaf_for_read(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise AtomicMovaUnoError("NON_REGULAR_FILE", "target is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            rem = cap + 1 - total
            to_read = _READ_CHUNK_BYTES if _READ_CHUNK_BYTES < rem else rem
            chunk = os.read(fd, to_read)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise AtomicMovaUnoError("FILE_TOO_LARGE", f"file exceeds {cap} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    except AtomicMovaUnoError:
        raise
    except OSError as exc:
        raise AtomicMovaUnoError("FILE_READ_ERROR", f"failed reading file: {type(exc).__name__}") from exc
    finally:
        os.close(fd)


def write_regular_file_bytes(path: Path, data: bytes) -> None:
    """Write regular file atomically via 0o600 CSPRNG temp file, fsync, pre-replace st_nlink==1 & os.replace."""
    if type(data) is not bytes:
        raise AtomicMovaUnoError("INVALID_WRITE_DATA", "data must be exact bytes")
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise AtomicMovaUnoError("NON_REGULAR_FILE", "target path refers to a directory")
    leaf = parts[-1]
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise AtomicMovaUnoError("FILE_WRITE_ERROR", f"cannot lstat target: {type(exc).__name__}") from exc
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise AtomicMovaUnoError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
            if not stat.S_ISREG(st.st_mode):
                raise AtomicMovaUnoError("NON_REGULAR_FILE", "cannot write to non-regular file")
            if st.st_nlink > 1:
                raise AtomicMovaUnoError("HARDLINK_FORBIDDEN", "cannot write through hardlinked file")

        tmp_fd: int | None = None
        tmp_name = ""
        for _ in range(32):
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
                raise AtomicMovaUnoError("FILE_WRITE_ERROR", f"cannot create temp file: {type(exc).__name__}") from exc
        if tmp_fd is None:
            raise AtomicMovaUnoError("TEMP_RETRY_EXHAUSTED", "temp file collision limit exceeded")

        try:
            try:
                view = memoryview(data)
                written = 0
                while written < len(data):
                    written += os.write(tmp_fd, view[written:])
                os.fsync(tmp_fd)
                if os.fstat(tmp_fd).st_nlink != 1:
                    raise AtomicMovaUnoError("HARDLINK_FORBIDDEN", "temp file link count != 1")
            finally:
                os.close(tmp_fd)
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            os.fsync(dfd)
        except AtomicMovaUnoError:
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
            raise AtomicMovaUnoError("FILE_WRITE_ERROR", f"failed writing file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------
# Field-Complete Strict-Nested-Key Secondary Audit Over All Four Pillars
# ---------------------------------------------------------------------------

_P1_KEYS = frozenset({"m4_max_inkling2_mini_anchors_n0", "zipf_ssd_roofline_n1"})
_ANCHOR_KEYS = frozenset({
    "hardware", "prefill_tokens", "greedy_parity_tokens", "atomic_memory_cap_gib", "atomic_kv_cap_mib",
    "stock_llama_cpp_tps", "stock_llama_cpp_ttft_s", "atomic_engine_tps", "atomic_engine_ttft_s",
    "strata_sq10_tps", "strata_sq10_ttft_s", "atomic_vs_stock_tps_ratio", "atomic_vs_stock_tps_4dp",
    "atomic_exceeds_33_over_10", "atomic_vs_stock_ttft_ratio", "atomic_ttft_below_3_over_4",
    "strata_vs_stock_tps_ratio", "strata_vs_atomic_tps_ratio", "strata_vs_atomic_ttft_speedup",
})
_ROOF_KEYS = frozenset({
    "total_experts", "base_resident_gib", "h_16", "t_res_ms", "w_miss_mb", "ssd_bw_mb_s",
    "t_ssd_full_miss_ms", "ssd_dominance_boundary_lo_gib", "ssd_dominance_boundary_hi_gib",
    "latency_strictly_decreasing_to_18gib", "full_residency_plateau_18_to_20gib",
    "resize_20_to_8_to_20_reversible", "resize_20gib_ssd_vanishes", "resize_8gib_ssd_dominated",
    "resize_8gib_vs_20gib_slowdown", "sweep",
})
_ROOF_ROW_KEYS = frozenset({
    "budget_gib", "cached_experts", "hit_rate", "hit_rate_4dp", "miss_rate",
    "t_res_ms", "t_ssd_ms", "t_ssd_ms_4dp", "t_tok_ms", "t_tok_ms_4dp",
    "tps", "tps_2dp", "ssd_miss_dominated", "ssd_latency_vanishes",
})

_P2_KEYS = frozenset({
    "external_dspark_spec", "zero_draft_uno_spec", "exact_crossover_b_star_gib",
    "exact_crossover_b_star_4dp", "crossover_residual_zero", "crossover_strictly_between_11_and_12",
    "uno_strictly_wins_all_below_b_star", "dspark_strictly_wins_all_above_b_star", "operating_points",
})
_DSPARK_KEYS = frozenset({
    "lookahead_k", "acceptance_alpha", "draft_step_cost_c", "draft_memory_gib",
    "renewal_speedup", "renewal_speedup_4dp", "exceeds_3x", "geometric_closed_form_matches",
})
_UNO_KEYS = frozenset({"draft_memory_gib", "lossless_speedup", "lossless_speedup_4dp"})
_OP_PT_KEYS = frozenset({"budget_gib", "dspark_tps", "dspark_tps_2dp", "uno_tps", "uno_tps_2dp", "winner"})

_P3_KEYS = frozenset({
    "k2_horizon_mova_anchors_n0", "softplus_gating_probes", "dense_v_waterfilling",
    "mova_sparse_v_waterfilling", "mova_exceeds_12x_kl_reduction", "mova_vs_dense_leverage_multiplier",
    "convexity_perturbation_excess_kl", "convexity_both_perturbations_strictly_worse",
    "kintsugi_outlier_expert_task_break",
})
_MOVA_ANC_KEYS = frozenset({
    "mova_total_params_b", "mova_active_params_b", "dense_baseline_params_b",
    "active_over_total_ratio", "active_over_dense_ratio", "reported_matched_quality_active_fraction",
})
_SP_KEYS = frozenset({
    "exp_z_grid", "sigma_derivatives", "strictly_in_open_unit_interval",
    "strictly_increasing", "zero_dead_region_on_negative_z",
})
_WF_KEYS = frozenset({
    "m_groups", "avg_bpw", "alpha", "exponents", "am_weight", "gm_weight",
    "k_uniform", "k_fisher", "k_equalized_per_group", "optimal_bits",
    "budget_conserved_exact", "gain_ratio", "gain_ratio_4dp", "am_gm_identity_exact",
})
_KINT_KEYS = frozenset({
    "clean_logit_margin", "uniform_4bpw_margin_shift", "uniform_4bpw_post_margin",
    "uniform_4bpw_task_broken", "fisher_waterfilling_margin_shift",
    "fisher_waterfilling_post_margin", "fisher_waterfilling_task_preserved",
})

_P4_KEYS = frozenset({
    "honest_scope_ledger", "lossless_64_token_parity_positive", "lossless_64_token_drift_negative",
    "gate_accepts_exact_64_of_64", "gate_rejects_single_token_drift",
})
_SCOPE_KEYS = frozenset({
    "n0_published_sources", "n1_certified_theorems", "n0_source_count",
    "n1_theorem_count", "strict_n0_n1_separation_enforced",
})
_PARITY_KEYS = frozenset({
    "total_tokens", "matched_tokens", "mismatched_tokens", "first_mismatch_index",
    "reference_seq_sha256", "candidate_seq_sha256", "exact_match",
})


def _require_dict_exact_keys(
    parent: dict[str, Any],
    key: str,
    expected_keys: frozenset[str],
    err_code: str,
) -> dict[str, Any]:
    val = parent.get(key)
    if type(val) is not dict:
        raise AtomicMovaUnoError(err_code, f"{key} must be a JSON object")
    if frozenset(val.keys()) != expected_keys:
        raise AtomicMovaUnoError(err_code, f"{key} keys mismatch")
    return val


def _require_flag(d: dict[str, Any], key: str, expected: bool, err_code: str) -> None:
    if require_exact_bool(d[key], key) is not expected:
        raise AtomicMovaUnoError(err_code, f"{key} must be {expected}")


def _require_list_len(parent: dict[str, Any], key: str, length: int, err_code: str) -> list[Any]:
    val = parent.get(key)
    if type(val) is not list or len(val) != length:
        raise AtomicMovaUnoError(err_code, f"{key} must be list of length {length}")
    return val


def _parse_q_vec(vec: Any, length: int, name: str, err_code: str) -> tuple[Fraction, ...]:
    if type(vec) is not list or len(vec) != length:
        raise AtomicMovaUnoError(err_code, f"{name} must be list of length {length}")
    return tuple(parse_exact_rational(item, f"{name}[{i}]") for i, item in enumerate(vec))


def _parse_int_vec(vec: Any, length: int, name: str, err_code: str) -> tuple[int, ...]:
    if type(vec) is not list or len(vec) != length:
        raise AtomicMovaUnoError(err_code, f"{name} must be list of length {length}")
    return tuple(require_exact_int(item, f"{name}[{i}]") for i, item in enumerate(vec))


def _audit_pillar1(p1: dict[str, Any]) -> None:
    anc = _require_dict_exact_keys(p1, "m4_max_inkling2_mini_anchors_n0", _ANCHOR_KEYS, "AUDIT_P1_ANC")
    for fk in ("atomic_exceeds_33_over_10", "atomic_ttft_below_3_over_4"):
        _require_flag(anc, fk, True, "AUDIT_P1_ANC_FLAG")
    tps_s = parse_exact_rational(anc["stock_llama_cpp_tps"], "tps_s")
    ttft_s = parse_exact_rational(anc["stock_llama_cpp_ttft_s"], "ttft_s")
    tps_a = parse_exact_rational(anc["atomic_engine_tps"], "tps_a")
    ttft_a = parse_exact_rational(anc["atomic_engine_ttft_s"], "ttft_a")
    tps_st = parse_exact_rational(anc["strata_sq10_tps"], "tps_st")
    ttft_st = parse_exact_rational(anc["strata_sq10_ttft_s"], "ttft_st")
    r_at = parse_exact_rational(anc["atomic_vs_stock_tps_ratio"], "r_at")
    if (
        require_exact_str(anc["hardware"], "hw"),
        require_exact_int(anc["prefill_tokens"], "pt"),
        require_exact_int(anc["greedy_parity_tokens"], "gp"),
        require_exact_int(anc["atomic_memory_cap_gib"], "mc"),
        require_exact_int(anc["atomic_kv_cap_mib"], "kc"),
        tps_s, ttft_s, tps_a, ttft_a, tps_st, ttft_st,
        r_at, require_exact_str(anc["atomic_vs_stock_tps_4dp"], "4dp"),
        parse_exact_rational(anc["atomic_vs_stock_ttft_ratio"], "r_ttft"),
        parse_exact_rational(anc["strata_vs_stock_tps_ratio"], "r_st"),
        parse_exact_rational(anc["strata_vs_atomic_tps_ratio"], "r_sa"),
        parse_exact_rational(anc["strata_vs_atomic_ttft_speedup"], "r_sattft"),
    ) != (
        "Apple M4 Max 48 GB (published Nov 23 2026)",
        4000, 64, 18, 100,
        Fraction(3311, 100), Fraction(957, 100), Fraction(10953, 100), Fraction(709, 100),
        Fraction(11292, 100), Fraction(421, 100),
        Fraction(10953, 3311), "3.3080",
        Fraction(709, 957), Fraction(11292, 3311), Fraction(3764, 3651), Fraction(709, 421),
    ):
        raise AtomicMovaUnoError("AUDIT_P1_ANC_VAL", "M4 Max anchor invariants mismatch")

    rf = _require_dict_exact_keys(p1, "zipf_ssd_roofline_n1", _ROOF_KEYS, "AUDIT_P1_RF")
    for fk in (
        "latency_strictly_decreasing_to_18gib", "full_residency_plateau_18_to_20gib",
        "resize_20_to_8_to_20_reversible", "resize_20gib_ssd_vanishes", "resize_8gib_ssd_dominated",
    ):
        _require_flag(rf, fk, True, "AUDIT_P1_RF_FLAG")
    h16 = harmonic_q(16)
    h6 = harmonic_q(6)
    t8 = Fraction(9, 1) + (Fraction(1, 1) - h6 / h16) * Fraction(40, 1)
    if (
        require_exact_int(rf["total_experts"], "te"),
        require_exact_int(rf["base_resident_gib"], "bg"),
        parse_exact_rational(rf["h_16"], "h16"),
        parse_exact_rational(rf["t_res_ms"], "tr"),
        parse_exact_rational(rf["w_miss_mb"], "wm"),
        parse_exact_rational(rf["ssd_bw_mb_s"], "bw"),
        parse_exact_rational(rf["t_ssd_full_miss_ms"], "tm"),
        require_exact_int(rf["ssd_dominance_boundary_lo_gib"], "blo"),
        require_exact_int(rf["ssd_dominance_boundary_hi_gib"], "bhi"),
        parse_exact_rational(rf["resize_8gib_vs_20gib_slowdown"], "sd"),
    ) != (16, 2, h16, Fraction(9, 1), Fraction(280, 1), Fraction(7000, 1), Fraction(40, 1), 9, 10, t8 / Fraction(9, 1)):
        raise AtomicMovaUnoError("AUDIT_P1_RF_VAL", "Zipf SSD roofline summary mismatch")

    sw = _require_list_len(rf, "sweep", 9, "AUDIT_P1_SW")
    exp_bs = (2, 6, 8, 9, 10, 12, 16, 18, 20)
    for idx, row in enumerate(sw):
        if type(row) is not dict or frozenset(row.keys()) != _ROOF_ROW_KEYS:
            raise AtomicMovaUnoError("AUDIT_P1_SW", f"sweep[{idx}] schema mismatch")
        eb = exp_bs[idx]
        ec = 16 if (eb - 2) > 16 else (eb - 2)
        eh = harmonic_q(ec) / h16
        em = Fraction(1, 1) - eh
        ets = em * Fraction(40, 1)
        ett = Fraction(9, 1) + ets
        etps = Fraction(1000, 1) / ett
        if (
            require_exact_int(row["budget_gib"], "b"),
            require_exact_int(row["cached_experts"], "c"),
            parse_exact_rational(row["hit_rate"], "h"),
            require_exact_str(row["hit_rate_4dp"], "h4"),
            parse_exact_rational(row["miss_rate"], "m"),
            parse_exact_rational(row["t_res_ms"], "tr"),
            parse_exact_rational(row["t_ssd_ms"], "ts"),
            require_exact_str(row["t_ssd_ms_4dp"], "ts4"),
            parse_exact_rational(row["t_tok_ms"], "tt"),
            require_exact_str(row["t_tok_ms_4dp"], "tt4"),
            parse_exact_rational(row["tps"], "tps"),
            require_exact_str(row["tps_2dp"], "tps2"),
            require_exact_bool(row["ssd_miss_dominated"], "sd"),
            require_exact_bool(row["ssd_latency_vanishes"], "sv"),
        ) != (
            eb, ec, eh, format_decimal_trunc(eh, 4), em, Fraction(9, 1),
            ets, format_decimal_trunc(ets, 4), ett, format_decimal_trunc(ett, 4),
            etps, format_decimal_trunc(etps, 2), ets > Fraction(9, 1), ets == 0,
        ):
            raise AtomicMovaUnoError("AUDIT_P1_SW_VAL", f"sweep[{idx}] mismatch")


def _audit_pillar2(p2: dict[str, Any]) -> None:
    ds = _require_dict_exact_keys(p2, "external_dspark_spec", _DSPARK_KEYS, "AUDIT_P2_DS")
    for fk in ("exceeds_3x", "geometric_closed_form_matches"):
        _require_flag(ds, fk, True, "AUDIT_P2_DS_FLAG")
    s_ext = Fraction(3843, 1250)
    if (
        require_exact_int(ds["lookahead_k"], "k"),
        parse_exact_rational(ds["acceptance_alpha"], "a"),
        parse_exact_rational(ds["draft_step_cost_c"], "c"),
        require_exact_int(ds["draft_memory_gib"], "dm"),
        parse_exact_rational(ds["renewal_speedup"], "se"),
        require_exact_str(ds["renewal_speedup_4dp"], "se4"),
    ) != (5, Fraction(4, 5), Fraction(1, 25), 4, s_ext, "3.0744"):
        raise AtomicMovaUnoError("AUDIT_P2_DS_VAL", "DSpark spec mismatch")

    uno = _require_dict_exact_keys(p2, "zero_draft_uno_spec", _UNO_KEYS, "AUDIT_P2_UNO")
    s_uno = Fraction(11, 5)
    if (
        require_exact_int(uno["draft_memory_gib"], "dm"),
        parse_exact_rational(uno["lossless_speedup"], "su"),
        require_exact_str(uno["lossless_speedup_4dp"], "su4"),
    ) != (0, s_uno, "2.2000"):
        raise AtomicMovaUnoError("AUDIT_P2_UNO_VAL", "Uno spec mismatch")

    for fk in (
        "crossover_residual_zero", "crossover_strictly_between_11_and_12",
        "uno_strictly_wins_all_below_b_star", "dspark_strictly_wins_all_above_b_star",
    ):
        _require_flag(p2, fk, True, "AUDIT_P2_FLAG")
    b_star = parse_exact_rational(p2["exact_crossover_b_star_gib"], "b_star")
    t_e_s = piecewise_linear_latency_ms(b_star - Fraction(4, 1))
    t_u_s = piecewise_linear_latency_ms(b_star)
    if (
        b_star,
        require_exact_str(p2["exact_crossover_b_star_4dp"], "b4"),
        s_ext * t_u_s == s_uno * t_e_s,
    ) != (Fraction(3625767011, 304898880), "11.8917", True):
        raise AtomicMovaUnoError("AUDIT_P2_BSTAR", "exact crossover B* mismatch")

    pts = _require_list_len(p2, "operating_points", 8, "AUDIT_P2_PTS")
    exp_bs = (6, 8, 10, 11, 12, 16, 20, 22)
    for idx, pt in enumerate(pts):
        if type(pt) is not dict or frozenset(pt.keys()) != _OP_PT_KEYS:
            raise AtomicMovaUnoError("AUDIT_P2_PTS", f"operating_points[{idx}] schema mismatch")
        eb = exp_bs[idx]
        etps_e = Fraction(1000, 1) * s_ext / piecewise_linear_latency_ms(Fraction(eb - 4, 1))
        etps_u = Fraction(1000, 1) * s_uno / piecewise_linear_latency_ms(Fraction(eb, 1))
        ewin = "external_dspark" if etps_e > etps_u else "zero_draft_uno"
        if (
            require_exact_int(pt["budget_gib"], "b"),
            parse_exact_rational(pt["dspark_tps"], "dt"),
            require_exact_str(pt["dspark_tps_2dp"], "dt2"),
            parse_exact_rational(pt["uno_tps"], "ut"),
            require_exact_str(pt["uno_tps_2dp"], "ut2"),
            require_exact_str(pt["winner"], "win"),
        ) != (eb, etps_e, format_decimal_trunc(etps_e, 2), etps_u, format_decimal_trunc(etps_u, 2), ewin):
            raise AtomicMovaUnoError("AUDIT_P2_PT_VAL", f"operating_points[{idx}] mismatch")


def _audit_wf_block(
    wf: dict[str, Any],
    exp_exps: tuple[int, ...],
    exp_am: Fraction,
    exp_bits: tuple[Fraction, ...],
    exp_4dp: str,
    err_code: str,
) -> None:
    for fk in ("budget_conserved_exact", "am_gm_identity_exact"):
        _require_flag(wf, fk, True, err_code)
    k_eq = Fraction(1, 256)
    k_fis = Fraction(1, 32)
    k_unif = exp_am * k_fis
    if (
        require_exact_int(wf["m_groups"], "m"),
        require_exact_int(wf["avg_bpw"], "b"),
        require_exact_int(wf["alpha"], "a"),
        _parse_int_vec(wf["exponents"], 8, "exps", err_code),
        parse_exact_rational(wf["am_weight"], "am"),
        parse_exact_rational(wf["gm_weight"], "gm"),
        parse_exact_rational(wf["k_uniform"], "ku"),
        parse_exact_rational(wf["k_fisher"], "kf"),
        parse_exact_rational(wf["k_equalized_per_group"], "ke"),
        _parse_q_vec(wf["optimal_bits"], 8, "ob", err_code),
        parse_exact_rational(wf["gain_ratio"], "gr"),
        require_exact_str(wf["gain_ratio_4dp"], "gr4"),
    ) != (8, 4, 2, exp_exps, exp_am, Fraction(1, 1), k_unif, k_fis, k_eq, exp_bits, exp_am, exp_4dp):
        raise AtomicMovaUnoError(err_code, "waterfilling block mismatch")


def _audit_pillar3(p3: dict[str, Any]) -> None:
    ma = _require_dict_exact_keys(p3, "k2_horizon_mova_anchors_n0", _MOVA_ANC_KEYS, "AUDIT_P3_ANC")
    if (
        require_exact_int(ma["mova_total_params_b"], "tp"),
        require_exact_int(ma["mova_active_params_b"], "ap"),
        require_exact_int(ma["dense_baseline_params_b"], "dp"),
        parse_exact_rational(ma["active_over_total_ratio"], "aot"),
        parse_exact_rational(ma["active_over_dense_ratio"], "aod"),
        parse_exact_rational(ma["reported_matched_quality_active_fraction"], "raf"),
    ) != (36, 4, 32, Fraction(1, 9), Fraction(1, 8), Fraction(1, 7)):
        raise AtomicMovaUnoError("AUDIT_P3_ANC_VAL", "MoVA anchors mismatch")

    sp = _require_dict_exact_keys(p3, "softplus_gating_probes", _SP_KEYS, "AUDIT_P3_SP")
    for fk in ("strictly_in_open_unit_interval", "strictly_increasing", "zero_dead_region_on_negative_z"):
        _require_flag(sp, fk, True, "AUDIT_P3_SP_FLAG")
    if (
        _parse_q_vec(sp["exp_z_grid"], 5, "ez", "AUDIT_P3_SP_VAL"),
        _parse_q_vec(sp["sigma_derivatives"], 5, "sd", "AUDIT_P3_SP_VAL"),
    ) != (
        (Fraction(1, 4), Fraction(1, 2), Fraction(1, 1), Fraction(2, 1), Fraction(4, 1)),
        (Fraction(1, 5), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(4, 5)),
    ):
        raise AtomicMovaUnoError("AUDIT_P3_SP_VAL", "softplus gating probes mismatch")

    dw = _require_dict_exact_keys(p3, "dense_v_waterfilling", _WF_KEYS, "AUDIT_P3_DW")
    _audit_wf_block(
        dw,
        (1, 1, 1, 1, -1, -1, -1, -1),
        Fraction(5, 4),
        (Fraction(9, 2), Fraction(9, 2), Fraction(9, 2), Fraction(9, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2)),
        "1.2500",
        "AUDIT_P3_DW_VAL",
    )
    mw = _require_dict_exact_keys(p3, "mova_sparse_v_waterfilling", _WF_KEYS, "AUDIT_P3_MW")
    _audit_wf_block(
        mw,
        (7, -1, -1, -1, -1, -1, -1, -1),
        Fraction(263, 16),
        (Fraction(15, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2), Fraction(7, 2)),
        "16.4375",
        "AUDIT_P3_MW_VAL",
    )

    for fk in ("mova_exceeds_12x_kl_reduction", "convexity_both_perturbations_strictly_worse"):
        _require_flag(p3, fk, True, "AUDIT_P3_FLAG")
    if (
        parse_exact_rational(p3["mova_vs_dense_leverage_multiplier"], "mul"),
        parse_exact_rational(p3["convexity_perturbation_excess_kl"], "ekl"),
    ) != (Fraction(263, 20), Fraction(1, 512)):
        raise AtomicMovaUnoError("AUDIT_P3_VAL", "MoVA leverage/convexity invariants mismatch")

    kn = _require_dict_exact_keys(p3, "kintsugi_outlier_expert_task_break", _KINT_KEYS, "AUDIT_P3_KN")
    for fk in ("uniform_4bpw_task_broken", "fisher_waterfilling_task_preserved"):
        _require_flag(kn, fk, True, "AUDIT_P3_KN_FLAG")
    if (
        parse_exact_rational(kn["clean_logit_margin"], "cm"),
        parse_exact_rational(kn["uniform_4bpw_margin_shift"], "us"),
        parse_exact_rational(kn["uniform_4bpw_post_margin"], "up"),
        parse_exact_rational(kn["fisher_waterfilling_margin_shift"], "fs"),
        parse_exact_rational(kn["fisher_waterfilling_post_margin"], "fp"),
    ) != (Fraction(1, 1), Fraction(-3, 2), Fraction(-1, 2), Fraction(-1, 8), Fraction(7, 8)):
        raise AtomicMovaUnoError("AUDIT_P3_KN_VAL", "Kintsugi outlier-expert task-break invariants mismatch")


def _audit_pillar4(p4: dict[str, Any]) -> None:
    sc = _require_dict_exact_keys(p4, "honest_scope_ledger", _SCOPE_KEYS, "AUDIT_P4_SC")
    _require_flag(sc, "strict_n0_n1_separation_enforced", True, "AUDIT_P4_SC_FLAG")
    n0s = _require_list_len(sc, "n0_published_sources", 3, "AUDIT_P4_SC_VAL")
    n1s = _require_list_len(sc, "n1_certified_theorems", 3, "AUDIT_P4_SC_VAL")
    if (
        [require_exact_str(x, "n0") for x in n0s],
        [require_exact_str(x, "n1") for x in n1s],
        require_exact_int(sc["n0_source_count"], "n0c"),
        require_exact_int(sc["n1_theorem_count"], "n1c"),
    ) != (
        [
            "AtomicBot-ai/atomic-engine (Nov 23 2026 M4 Max 48 GB Inkling 2 Mini)",
            "Strata SQ-10 (strata-inkling 0.1.65 M4 Max 48 GB)",
            "K2-Horizon (arXiv:2611.21003v1: MoVA 36B-A4B, Softplus Gating, Uno 2.2x)",
        ],
        [
            "Discrete Zipf-1 SSD expert-streaming roofline & 20->8->20 GiB phase transition",
            "Fixed-memory Budget-Crossover Theorem (External DSpark vs Zero-Draft Uno)",
            "Sparse MoVA Fisher water-filling concentration & outlier-expert task-break hazard",
        ],
        3,
        3,
    ):
        raise AtomicMovaUnoError("AUDIT_P4_SC_VAL", "scope ledger entries mismatch")

    for fk in ("gate_accepts_exact_64_of_64", "gate_rejects_single_token_drift"):
        _require_flag(p4, fk, True, "AUDIT_P4_FLAG")

    ref_seq = [(1009 * t + 37) % 128_000 for t in range(64)]
    drift_seq = list(ref_seq)
    drift_seq[63] = (drift_seq[63] + 1) % 128_000
    exp_ref_sha = hashlib.sha256(canonical_json_bytes(ref_seq)).hexdigest()
    exp_drift_sha = hashlib.sha256(canonical_json_bytes(drift_seq)).hexdigest()

    pos = _require_dict_exact_keys(p4, "lossless_64_token_parity_positive", _PARITY_KEYS, "AUDIT_P4_POS")
    if (
        require_exact_int(pos["total_tokens"], "tt"),
        require_exact_int(pos["matched_tokens"], "mt"),
        require_exact_int(pos["mismatched_tokens"], "mm"),
        pos["first_mismatch_index"],
        require_sha256_hex(pos["reference_seq_sha256"], "rs"),
        require_sha256_hex(pos["candidate_seq_sha256"], "cs"),
        require_exact_bool(pos["exact_match"], "em"),
    ) != (64, 64, 0, None, exp_ref_sha, exp_ref_sha, True):
        raise AtomicMovaUnoError("AUDIT_P4_POS_VAL", "positive 64/64 token parity mismatch")

    neg = _require_dict_exact_keys(p4, "lossless_64_token_drift_negative", _PARITY_KEYS, "AUDIT_P4_NEG")
    if (
        require_exact_int(neg["total_tokens"], "tt"),
        require_exact_int(neg["matched_tokens"], "mt"),
        require_exact_int(neg["mismatched_tokens"], "mm"),
        require_exact_int(neg["first_mismatch_index"], "fm"),
        require_sha256_hex(neg["reference_seq_sha256"], "rs"),
        require_sha256_hex(neg["candidate_seq_sha256"], "cs"),
        require_exact_bool(neg["exact_match"], "em"),
    ) != (64, 63, 1, 63, exp_ref_sha, exp_drift_sha, False):
        raise AtomicMovaUnoError("AUDIT_P4_NEG_VAL", "negative 63/64 token drift mismatch")


def secondary_audit_payload(payload: Any) -> dict[str, Any]:
    """Field-complete strict-nested-key independent algebraic audit enforced under both replay modes."""
    if type(payload) is not dict or frozenset(payload.keys()) != _PAYLOAD_KEYS:
        raise AtomicMovaUnoError("AUDIT_INVALID_PAYLOAD", "payload must be exact schema dict")
    if (
        require_exact_str(payload["novelty_tier"], "novelty_tier"),
        require_exact_str(payload["published_benchmarks_novelty"], "published_benchmarks_novelty"),
        require_exact_str(payload["oracle_class"], "oracle_class"),
    ) != ("N1", "N0", "exact"):
        raise AtomicMovaUnoError("AUDIT_SCOPE_META", "novelty/oracle_class metadata mismatch")
    for fk in ("measured_live_m4_max_hardware_run", "measured_live_k2_horizon_checkpoint_eval"):
        _require_flag(payload, fk, False, "AUDIT_MEASURED_FLAG")
    for fk in ("zero_trust_path_floats", "zero_bare_asserts", "zero_mathematical_rng"):
        _require_flag(payload, fk, True, "AUDIT_ZERO_FLAG")

    p1 = _require_dict_exact_keys(payload, "pillar1_atomic_ssd_roofline", _P1_KEYS, "AUDIT_P1")
    p2 = _require_dict_exact_keys(payload, "pillar2_budget_crossover", _P2_KEYS, "AUDIT_P2")
    p3 = _require_dict_exact_keys(payload, "pillar3_mova_fisher_kintsugi", _P3_KEYS, "AUDIT_P3")
    p4 = _require_dict_exact_keys(payload, "pillar4_scope_and_token_parity", _P4_KEYS, "AUDIT_P4")

    _audit_pillar1(p1)
    _audit_pillar2(p2)
    _audit_pillar3(p3)
    _audit_pillar4(p4)

    return {
        "secondary_audit_passed": True,
        "pillars_audited": 4,
        "roofline_budgets_audited": 9,
        "crossover_points_audited": 8,
        "token_parity_length": 64,
    }


def verify_certificate_dict(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Validate an in-memory certificate dict with digests, schema, secondary audit & optional full replay."""
    res = validate_certificate_payload(cert, full_replay=full_replay)
    sec = secondary_audit_payload(cert["payload"])
    res["secondary_audit"] = sec
    return res


def check_certificate_file(path: Path) -> dict[str, Any]:
    raw = read_regular_bytes_capped(path, MAX_JSON_BYTES)
    if b"\r" in raw:
        raise AtomicMovaUnoError("CRLF_FORBIDDEN", "CRLF line endings forbidden in certificate")
    if not raw.endswith(b"\n"):
        raise AtomicMovaUnoError("MISSING_TRAILING_LF", "certificate must end with a single LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AtomicMovaUnoError("INVALID_UTF8", "certificate is not valid UTF-8") from exc
    cert = parse_strict_json_text(text, require_object=True)
    canonical_raw = canonical_json_bytes(cert)
    if raw != canonical_raw + b"\n":
        raise AtomicMovaUnoError("NON_CANONICAL_BYTES", "certificate bytes are not canonical ASCII JSON + LF")
    res = verify_certificate_dict(cert, full_replay=True)
    res["canonical_sha256"] = hashlib.sha256(canonical_raw).hexdigest()
    res["file_sha256"] = hashlib.sha256(raw).hexdigest()
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
# Strict Fail-Closed Manual CLI Parsing (Zero Abbreviations, Zero Flag Traps)
# ---------------------------------------------------------------------------

def _parse_cli_argv(argv: Sequence[str]) -> tuple[str, Path | None, bool]:
    if type(argv) not in (list, tuple):
        raise AtomicMovaUnoError("INVALID_ARGV", "argv must be exact list/tuple of strict str")
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
                raise AtomicMovaUnoError("DUPLICATE_FLAG", "duplicate --json flag")
            as_json = True
            i += 1
        elif tok == "--standalone":
            if mode is not None:
                raise AtomicMovaUnoError("CONFLICTING_MODE", "multiple mode flags forbidden")
            mode = "standalone"
            i += 1
        elif tok in ("--check", "--write"):
            if mode is not None:
                raise AtomicMovaUnoError("CONFLICTING_MODE", "multiple mode flags forbidden")
            if i + 1 >= n:
                raise AtomicMovaUnoError("MISSING_PATH_ARG", f"{tok} requires a path argument")
            val = argv[i + 1]
            if not val or val.startswith("-"):
                raise AtomicMovaUnoError("INVALID_PATH_ARG", f"{tok} path cannot be empty or flag-like")
            mode = "check" if tok == "--check" else "write"
            target = Path(val)
            i += 2
        else:
            raise AtomicMovaUnoError("UNKNOWN_FLAG", f"unrecognized CLI argument: {tok}")

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
    except AtomicMovaUnoError as exc:
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
