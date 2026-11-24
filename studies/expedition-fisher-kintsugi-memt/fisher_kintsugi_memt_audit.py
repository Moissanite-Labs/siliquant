#!/usr/bin/env python3
"""Track S42: R2-Grade Zone-B Hardened CLI, Full Pure-Q Replay & Secondary Algebraic Cross-Check.

Pure Python 3 standard library: zero floats, zero bare asserts, zero `.isdigit()`, zero mathematical RNG.
Zone-B hardening (baked in from line 1):
  - Concrete pathlib.Path check + NUL-byte and lone UTF-16 surrogate path refusal
  - Deleted-CWD OSError wrapping -> CWD_UNAVAILABLE (rc=2)
  - Atomic dir_fd PWD walk + per-component openat(O_RDONLY | O_DIRECTORY | O_NOFOLLOW) (zero os.path.islink)
  - Leaf O_RDONLY | O_NOFOLLOW | O_NONBLOCK + fstat S_ISREG + 64 KiB streaming capped reads (max_bytes <= 64 MiB)
  - Atomic write: type(data) is bytes BEFORE dir/temp open -> target lstat (st_nlink > 1 -> HARDLINK_FORBIDDEN)
    -> thread-locked monotonic + CSPRNG (secrets.token_hex(8)) 0o600 O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK
    temp file in same dir_fd -> bounded 32-attempt EEXIST retry -> os.fsync(tmp_fd)
    -> pre-replace os.fstat(tmp_fd).st_nlink == 1 -> os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd) -> os.fsync(dfd)
  - Strict manual CLI parser (allow_abbrev=False semantics) rejecting space- and equals-syntax
    flag-like/empty/abbreviated paths (--write --json -> rc=2, 0 files created)
  - Full pure-Q replay + field-complete strict-keyset secondary_audit_payload (enforced under full_replay=False too)
"""
from __future__ import annotations

import errno
from fractions import Fraction
import hashlib
import hmac
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

from fisher_kintsugi_memt_oracle import (
    MAX_JSON_BYTES,
    SCHEMA_VERSION,
    TRACK_ID,
    _PAYLOAD_KEYS,
    FisherKintsugiMemtError,
    build_standalone_certificate,
    canonical_json_bytes,
    format_exact_rational,
    gdn2_delta_step_exact,
    parse_exact_rational,
    parse_strict_json_text,
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
    return f".{leaf}.s42tmp.{os.getpid()}.{threading.get_ident()}.{seq}.{secrets.token_hex(8)}"


# ---------------------------------------------------------------------------
# Race-Free Zone-B File I/O (Per-Component dir_fd/openat + Atomic os.replace)
# ---------------------------------------------------------------------------

def _open_dir_chain(anchor: str, parts: Sequence[str]) -> int:
    """Walk directory components atomically via `openat(..., O_DIRECTORY | O_NOFOLLOW)`."""
    try:
        dfd = os.open(anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise FisherKintsugiMemtError("PATH_STAT_ERROR", f"cannot open anchor dir: {type(exc).__name__}") from exc

    for part in parts:
        try:
            nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
        except OSError as exc:
            os.close(dfd)
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise FisherKintsugiMemtError("SYMLINK_FORBIDDEN", "symlink or non-dir forbidden in path") from exc
            if exc.errno == errno.ENOENT:
                raise FisherKintsugiMemtError("PATH_NOT_FOUND", "directory component not found") from exc
            raise FisherKintsugiMemtError("PATH_STAT_ERROR", f"cannot traverse dir: {type(exc).__name__}") from exc
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
        except FisherKintsugiMemtError as exc:
            if exc.code != "PATH_NOT_FOUND":
                raise
        else:
            os.close(pwd_fd)


def _validate_path_and_split(path: Any) -> tuple[str, list[str]]:
    if type(path) is not _EXACT_PATH_TYPE:
        raise FisherKintsugiMemtError("INVALID_PATH_TYPE", "path must be exact pathlib.Path")
    raw_str = require_exact_str(str(path), "path")
    if "\x00" in raw_str:
        raise FisherKintsugiMemtError("INVALID_PATH", "path contains NUL byte")

    if not path.is_absolute():
        try:
            Path.cwd()
        except OSError as exc:
            raise FisherKintsugiMemtError("CWD_UNAVAILABLE", f"cannot resolve cwd: {type(exc).__name__}") from exc
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
            raise FisherKintsugiMemtError("PATH_STAT_ERROR", f"cannot lstat leaf: {type(exc).__name__}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise FisherKintsugiMemtError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
    finally:
        os.close(dfd)
    return path


def _open_leaf_for_read(path: Path) -> int:
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise FisherKintsugiMemtError("NON_REGULAR_FILE", "target path refers to a directory")
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise FisherKintsugiMemtError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf") from exc
            raise FisherKintsugiMemtError("FILE_READ_ERROR", f"cannot open file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


def read_regular_bytes_capped(path: Path, max_bytes: Any = MAX_JSON_BYTES) -> bytes:
    """Read regular file bytes via atomic dir_fd + O_NOFOLLOW|O_NONBLOCK in bounded 64 KiB streaming chunks."""
    cap = require_exact_int(max_bytes, "max_bytes", min_val=1, max_val=MAX_ALLOWED_CAP_BYTES)
    fd = _open_leaf_for_read(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise FisherKintsugiMemtError("NON_REGULAR_FILE", "target is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            to_read = min(_READ_CHUNK_BYTES, cap + 1 - total)
            chunk = os.read(fd, to_read)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise FisherKintsugiMemtError("FILE_TOO_LARGE", f"file exceeds {cap} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    except FisherKintsugiMemtError:
        raise
    except OSError as exc:
        raise FisherKintsugiMemtError("FILE_READ_ERROR", f"failed reading file: {type(exc).__name__}") from exc
    finally:
        os.close(fd)


def write_regular_file_bytes(path: Path, data: bytes) -> None:
    """Write regular file atomically via 0o600 CSPRNG temp file, fsync, pre-replace st_nlink==1 & os.replace."""
    if type(data) is not bytes:
        raise FisherKintsugiMemtError("INVALID_WRITE_DATA", "data must be exact bytes")
    anchor, parts = _validate_path_and_split(path)
    if not parts:
        raise FisherKintsugiMemtError("NON_REGULAR_FILE", "target path refers to a directory")
    leaf = parts[-1]
    dfd = _open_dir_chain(anchor, parts[:-1])
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise FisherKintsugiMemtError("FILE_WRITE_ERROR", f"cannot lstat target: {type(exc).__name__}") from exc
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise FisherKintsugiMemtError("SYMLINK_FORBIDDEN", "symlink forbidden at leaf")
            if not stat.S_ISREG(st.st_mode):
                raise FisherKintsugiMemtError("NON_REGULAR_FILE", "cannot write to non-regular file")
            if st.st_nlink > 1:
                raise FisherKintsugiMemtError("HARDLINK_FORBIDDEN", "cannot write through hardlinked file")

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
                raise FisherKintsugiMemtError("FILE_WRITE_ERROR", f"cannot create temp file: {type(exc).__name__}") from exc
        if tmp_fd is None:
            raise FisherKintsugiMemtError("TEMP_RETRY_EXHAUSTED", "temp file collision limit exceeded")

        try:
            try:
                view = memoryview(data)
                written = 0
                while written < len(data):
                    written += os.write(tmp_fd, view[written:])
                os.fsync(tmp_fd)
                if os.fstat(tmp_fd).st_nlink != 1:
                    raise FisherKintsugiMemtError("HARDLINK_FORBIDDEN", "temp file link count != 1")
            finally:
                os.close(tmp_fd)
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            os.fsync(dfd)
        except FisherKintsugiMemtError:
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
            raise FisherKintsugiMemtError("FILE_WRITE_ERROR", f"failed writing file: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)


# ---------------------------------------------------------------------------
# Field-Complete Strict-Keyset Secondary Algebraic Cross-Check (All 4 Pillars)
# ---------------------------------------------------------------------------

_P1_KEYS = frozenset({
    "uniform_control",
    "mild_skew",
    "heavy_outlier_gemma4_scale",
    "uniform_ratio_equals_one",
    "mild_ratio_exceeds_two",
    "heavy_ratio_exceeds_twelve",
    "strict_hierarchy_certified",
})
_SPEC_KEYS = frozenset({
    "row_count_m",
    "alpha",
    "b_bar",
    "sensitivity_exponents",
    "mean_exponent",
    "am_weights",
    "gm_weights",
    "b_star",
    "extra_average_bpw",
    "all_b_star_positive",
    "k_equalized_per_row",
    "kkt_equalization_exact",
    "k_uniform_total",
    "k_fisher_total",
    "improvement_ratio",
    "improvement_ratio_4dp",
    "am_gm_identity_exact",
    "perturbation_witnesses",
    "all_perturbations_strictly_worse",
})
_PERT_KEYS = frozenset({
    "label",
    "delta",
    "k_perturbed_total",
    "excess_kl",
    "ratio_to_optimal",
    "strictly_worse_than_fisher",
})
_P2_KEYS = frozenset({
    "quadratic_fisher_witness",
    "exact_dyadic_shannon_witness",
    "cs20_average_kl_blindness_certified",
})
_FISHER_WIT_KEYS = frozenset({
    "prompt_tokens",
    "vocab_size",
    "fisher_shift_invariant_nullspace",
    "fisher_matrix_rank",
    "fisher_3x3_block_pd",
    "uniform_a",
    "kintsugi_b",
    "kintsugi_over_uniform_kl_ratio",
    "kintsugi_over_uniform_kl_ratio_4dp",
    "uniform_lower_avg_kl_yet_breaks_task",
})
_Q_EVAL_KEYS = frozenset({
    "per_token_fisher_kl",
    "average_fisher_kl",
    "per_token_margins",
    "per_token_top1",
    "critical_token_margin",
    "min_margin",
    "task_pass",
})
_DYADIC_WIT_KEYS = frozenset({
    "prompt_tokens",
    "uniform_a_routine_kl_bits",
    "uniform_a_critical_kl_bits",
    "uniform_a_critical_log2_exponents",
    "uniform_a_avg_kl_bits",
    "uniform_a_critical_prob_margin",
    "uniform_a_critical_top1",
    "uniform_a_task_pass",
    "kintsugi_b_routine_kl_bits",
    "kintsugi_b_routine_log2_exponents",
    "kintsugi_b_critical_kl_bits",
    "kintsugi_b_avg_kl_bits",
    "kintsugi_b_routine_prob_margin",
    "kintsugi_b_critical_prob_margin",
    "kintsugi_b_task_pass",
    "kintsugi_over_uniform_kl_ratio",
    "gibbs_strict_positivity_certified",
    "uniform_lower_avg_kl_yet_breaks_task",
})
_P3_KEYS = frozenset({
    "sharp_99pct_threshold_n",
    "gbench_slo_cohort_threshold_n",
    "cohort_n100",
    "cohort_n458",
    "cohort_n459",
    "cohort_n1099",
    "sharp_boundary_458_fails_459_passes",
    "n1099_two_sided_order_stat_bracket",
    "cs18_uncertified_floor_enforced",
})
_COHORT_KEYS = frozenset({
    "n",
    "den_bit_length",
    "den_decimal_digits",
    "den_exceeds_max_json_int_bits",
    "pass_99pct_upper_tail",
    "miss_prob_below_1_over_60000",
    "reciprocal_miss_prob_floor",
    "reciprocal_miss_prob_ceil",
    "p_zero_exceedances_ppm_lo",
    "p_zero_exceedances_ppm_hi",
    "p_le_1_exceedance_ppm_lo",
    "p_le_1_exceedance_ppm_hi",
    "p_le_2_exceedances_ppm_lo",
    "p_le_2_exceedances_ppm_hi",
    "p_ge_1_exceedance_ppm_lo",
    "p_ge_1_exceedance_ppm_hi",
    "p_ge_5_exceedances_ppm_lo",
    "p_ge_5_exceedances_ppm_hi",
    "cs18_upper_tail_status",
    "cs18_gbench_slo_status",
})
_BRACKET_KEYS = frozenset({
    "exceedance_range_min",
    "exceedance_range_max",
    "order_stat_lower_rank",
    "order_stat_upper_rank",
    "coverage_ppm_lo",
    "coverage_ppm_hi",
    "coverage_exceeds_95pct",
})
_P4_KEYS = frozenset({
    "capacity_obstruction",
    "channel_2b_poisoning_witness",
    "four_channel_airlock",
})
_CAP_KEYS = frozenset({
    "state_dim_d",
    "hadamard_keys_orthonormal",
    "transpose_duality_l4_exact",
    "transpose_duality_l5_exact",
    "rank_s4",
    "l4_per_key_err_sq",
    "l4_total_err_sq",
    "l4_zero_error_exact",
    "k5_unit_norm_exact",
    "rank_key_matrix_k5",
    "rank_s5",
    "nullspace_witness_c",
    "nullspace_k5_c_zero_exact",
    "inconsistency_vector_d",
    "universal_cauchy_schwarz_lb",
    "l5_per_key_err_sq",
    "l5_total_err_sq",
    "delta_s_frob_norm_sq",
    "parseval_error_identity_exact",
    "capacity_obstruction_certified",
})
_POIS_KEYS = frozenset({
    "frozen_weights_delta_w_zero",
    "benign_query_delta_p_data_zero",
    "zero_kv_carryover_delta_kv_zero",
    "clean_state_sha256",
    "poisoned_state_sha256",
    "state_hashes_distinct",
    "delta_s_exact_rank",
    "delta_s_frob_norm_sq",
    "clean_margin",
    "poisoned_margin",
    "margin_after_benign_orth_turn",
    "benign_readout_sign_flipped",
    "poison_persists_after_benign_turn",
})
_AIR_KEYS = frozenset({
    "clean_record",
    "negative_controls",
    "legacy_3ch_blind_to_channel2b",
    "all_four_negative_controls_denied",
    "airlock_extension_certified",
})
_AIR_RES_KEYS = frozenset({
    "legacy_3ch_decision",
    "airlock_4ch_decision",
    "violations_3ch",
    "violations_4ch",
})
_AIR_NEG_KEYS = _AIR_RES_KEYS | frozenset({"name"})

_EXPECTED_CS13_ANCHORS = [
    "ShinMegamiBoson/lookatmyquant",
    "professorpalmer/kintsugi-method",
    "professorpalmer/hard-proof",
    "google-deepmind/gbench",
    "MemTensor/MemT-1.0-Pro",
]


def _require_dict_exact_keys(
    mapping: dict[str, Any],
    key: str,
    expected_keys: frozenset[str],
    err_code: str,
) -> dict[str, Any]:
    val = mapping.get(key)
    if type(val) is not dict or frozenset(val.keys()) != expected_keys:
        raise FisherKintsugiMemtError(err_code, f"missing or invalid object key set: {key}")
    return val


def _require_flag(mapping: dict[str, Any], key: str, expected: bool, err_code: str) -> None:
    if require_exact_bool(mapping.get(key), key) is not expected:
        raise FisherKintsugiMemtError(err_code, f"invariant {key} must be {expected}")


def _require_list_exact_len(val: Any, name: str, expected_len: int, err_code: str) -> list[Any]:
    if type(val) is not list or len(val) != expected_len:
        raise FisherKintsugiMemtError(err_code, f"{name} must be list of length {expected_len}")
    return val


def _require_int_list(val: Any, name: str, expected_len: int, err_code: str) -> list[int]:
    items = _require_list_exact_len(val, name, expected_len, err_code)
    return [require_exact_int(item, f"{name}[{i}]") for i, item in enumerate(items)]


def _require_str_list(val: Any, name: str, expected_len: int, err_code: str) -> list[str]:
    items = _require_list_exact_len(val, name, expected_len, err_code)
    return [require_exact_str(item, f"{name}[{i}]") for i, item in enumerate(items)]


def _require_rational_list(val: Any, name: str, expected_len: int, err_code: str) -> list[Fraction]:
    items = _require_list_exact_len(val, name, expected_len, err_code)
    return [parse_exact_rational(item, f"{name}[{i}]") for i, item in enumerate(items)]


def _derive_gdn2_state_hashes() -> tuple[str, str]:
    """Re-derive exact canonical SHA-256 digests of GDN-2 states S4 (clean) and S5 (poisoned)."""
    half = Fraction(1, 2)
    keys4 = (
        [half, half, half, half],
        [half, -half, half, -half],
        [half, half, -half, -half],
        [half, -half, -half, half],
    )
    vals4 = (
        [Fraction(2, 1), Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(2, 1), Fraction(0, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(0, 1), Fraction(2, 1), Fraction(0, 1)],
        [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(2, 1)],
    )
    s4 = [[Fraction(0, 1)] * 4 for _ in range(4)]
    for kv, vv in zip(keys4, vals4):
        s4 = gdn2_delta_step_exact(s4, kv, vv)
    k5 = [Fraction(3, 5), Fraction(4, 5), Fraction(0, 1), Fraction(0, 1)]
    v5 = [Fraction(0, 1), Fraction(6, 1), Fraction(0, 1), Fraction(0, 1)]
    s5 = gdn2_delta_step_exact(s4, k5, v5)
    s4_sha = hashlib.sha256(
        canonical_json_bytes([[format_exact_rational(x) for x in row] for row in s4])
    ).hexdigest()
    s5_sha = hashlib.sha256(
        canonical_json_bytes([[format_exact_rational(x) for x in row] for row in s5])
    ).hexdigest()
    return s4_sha, s5_sha


def secondary_audit_payload(payload: Any) -> dict[str, Any]:
    """Fail-closed field-complete secondary algebraic invariant cross-check over all four Track S42 pillars."""
    if type(payload) is not dict or frozenset(payload.keys()) != _PAYLOAD_KEYS:
        raise FisherKintsugiMemtError("AUDIT_PAYLOAD_TYPE", "payload must be dict with exact Track S42 keys")
    if (
        payload["novelty_tier"],
        payload["oracle_class"],
        _require_str_list(payload["prior_art_cs13_anchors"], "anchors", 5, "AUDIT_META"),
    ) != ("N1", "exact", _EXPECTED_CS13_ANCHORS):
        raise FisherKintsugiMemtError("AUDIT_META", "unexpected metadata or CS-13 anchors")

    for key in (
        "measured_live_gemma4_lamq_weights",
        "measured_live_gbench_serving_endpoint",
        "measured_live_memt_554b_weights",
    ):
        _require_flag(payload, key, False, "AUDIT_LAW4_FLAG")
    for key in ("zero_trust_path_floats", "zero_bare_asserts", "zero_mathematical_rng"):
        _require_flag(payload, key, True, "AUDIT_HYGIENE_FLAG")

    # Pillar 1: Exact AM-GM Fisher Water-Filling
    p1 = _require_dict_exact_keys(payload, "pillar1_fisher_waterfilling_amgm", _P1_KEYS, "AUDIT_P1_MISSING")
    for fk in (
        "uniform_ratio_equals_one",
        "mild_ratio_exceeds_two",
        "heavy_ratio_exceeds_twelve",
        "strict_hierarchy_certified",
    ):
        _require_flag(p1, fk, True, "AUDIT_P1_FLAG")

    spec_defs = (
        (
            "uniform_control",
            [0] * 8,
            Fraction(1, 1),
            [Fraction(4, 1)] * 8,
            Fraction(1, 32),
            Fraction(1, 1),
            "1.0000",
            0,
        ),
        (
            "mild_skew",
            [4, 2, 0, 0, 0, 0, -2, -4],
            Fraction(389, 128),
            [Fraction(6, 1), Fraction(5, 1), Fraction(4, 1), Fraction(4, 1), Fraction(4, 1), Fraction(4, 1), Fraction(3, 1), Fraction(2, 1)],
            Fraction(389, 4096),
            Fraction(389, 128),
            "3.0390",
            0,
        ),
        (
            "heavy_outlier_gemma4_scale",
            [7, -1, -1, -1, -1, -1, -1, -1],
            Fraction(263, 16),
            [Fraction(15, 2)] + [Fraction(7, 2)] * 7,
            Fraction(263, 512),
            Fraction(263, 16),
            "16.4375",
            3,
        ),
    )
    for sk, exp_es, exp_am, exp_bstar, exp_ku, exp_r, exp_r4dp, exp_npert in spec_defs:
        sd = _require_dict_exact_keys(p1, sk, _SPEC_KEYS, "AUDIT_P1_SPEC")
        for fk in (
            "all_b_star_positive",
            "kkt_equalization_exact",
            "am_gm_identity_exact",
            "all_perturbations_strictly_worse",
        ):
            _require_flag(sd, fk, True, "AUDIT_P1_SPEC_FLAG")
        am_q = parse_exact_rational(sd["am_weights"], f"{sk}.am")
        gm_q = parse_exact_rational(sd["gm_weights"], f"{sk}.gm")
        ku_q = parse_exact_rational(sd["k_uniform_total"], f"{sk}.ku")
        kf_q = parse_exact_rational(sd["k_fisher_total"], f"{sk}.kf")
        rat_q = parse_exact_rational(sd["improvement_ratio"], f"{sk}.ratio")
        bstar_q = _require_rational_list(sd["b_star"], f"{sk}.b_star", 8, "AUDIT_P1_SPEC_VAL")
        if (
            require_exact_int(sd["row_count_m"], f"{sk}.m"),
            parse_exact_rational(sd["alpha"], f"{sk}.alpha"),
            parse_exact_rational(sd["b_bar"], f"{sk}.b_bar"),
            _require_int_list(sd["sensitivity_exponents"], f"{sk}.es", 8, "AUDIT_P1_SPEC_VAL"),
            require_exact_int(sd["mean_exponent"], f"{sk}.e_bar"),
            am_q,
            gm_q,
            bstar_q,
            sum(bstar_q, Fraction(0, 1)) / Fraction(8, 1),
            parse_exact_rational(sd["extra_average_bpw"], f"{sk}.extra_bpw"),
            parse_exact_rational(sd["k_equalized_per_row"], f"{sk}.k_eq"),
            ku_q,
            kf_q,
            rat_q,
            rat_q * kf_q == ku_q,
            rat_q * gm_q == am_q,
            require_exact_str(sd["improvement_ratio_4dp"], f"{sk}.4dp"),
        ) != (
            8,
            Fraction(2, 1),
            Fraction(4, 1),
            exp_es,
            0,
            exp_am,
            Fraction(1, 1),
            exp_bstar,
            Fraction(4, 1),
            Fraction(0, 1),
            Fraction(1, 256),
            exp_ku,
            Fraction(1, 32),
            exp_r,
            True,
            True,
            exp_r4dp,
        ):
            raise FisherKintsugiMemtError("AUDIT_P1_SPEC_VAL", f"{sk} water-filling invariants mismatch")

        perts = _require_list_exact_len(sd["perturbation_witnesses"], f"{sk}.perts", exp_npert, "AUDIT_P1_PERT")
        if exp_npert == 3:
            pert_exp = (
                ("pair_half_bit", [Fraction(1, 2), Fraction(-1, 2)] + [Fraction(0, 1)] * 6, Fraction(17, 512), Fraction(1, 512), Fraction(17, 16)),
                ("pair_one_bit", [Fraction(1, 1), Fraction(-1, 1)] + [Fraction(0, 1)] * 6, Fraction(41, 1024), Fraction(9, 1024), Fraction(41, 32)),
                ("multi_row_asymmetric", [Fraction(1, 1), Fraction(1, 2), Fraction(-1, 2), Fraction(-1, 1)] + [Fraction(0, 1)] * 4, Fraction(43, 1024), Fraction(11, 1024), Fraction(43, 32)),
            )
            for idx, item in enumerate(perts):
                if type(item) is not dict or frozenset(item.keys()) != _PERT_KEYS:
                    raise FisherKintsugiMemtError("AUDIT_P1_PERT", "perturbation item must have exact keys")
                _require_flag(item, "strictly_worse_than_fisher", True, "AUDIT_P1_PERT")
                d_list = _require_rational_list(item["delta"], "pert.delta", 8, "AUDIT_P1_PERT")
                kp = parse_exact_rational(item["k_perturbed_total"], "pert.kp")
                ex = parse_exact_rational(item["excess_kl"], "pert.ex")
                rt = parse_exact_rational(item["ratio_to_optimal"], "pert.rt")
                if (
                    require_exact_str(item["label"], "pert.label"),
                    d_list,
                    sum(d_list, Fraction(0, 1)),
                    kp,
                    ex,
                    rt,
                    ex == kp - Fraction(1, 32),
                    rt == kp * Fraction(32, 1),
                ) != (pert_exp[idx][0], pert_exp[idx][1], Fraction(0, 1), pert_exp[idx][2], pert_exp[idx][3], pert_exp[idx][4], True, True):
                    raise FisherKintsugiMemtError("AUDIT_P1_PERT_VAL", f"perturbation[{idx}] mismatch")

    # Pillar 2: Kintsugi Task-Break Logit-Margin Inversion vs. Smooth Average KL
    p2 = _require_dict_exact_keys(payload, "pillar2_kintsugi_margin_vs_kl", _P2_KEYS, "AUDIT_P2_MISSING")
    _require_flag(p2, "cs20_average_kl_blindness_certified", True, "AUDIT_P2_FLAG")

    fw = _require_dict_exact_keys(p2, "quadratic_fisher_witness", _FISHER_WIT_KEYS, "AUDIT_P2_FISHER")
    for fk in (
        "fisher_shift_invariant_nullspace",
        "fisher_3x3_block_pd",
        "uniform_lower_avg_kl_yet_breaks_task",
    ):
        _require_flag(fw, fk, True, "AUDIT_P2_FISHER_FLAG")
    qa = _require_dict_exact_keys(fw, "uniform_a", _Q_EVAL_KEYS, "AUDIT_P2_QA")
    qb = _require_dict_exact_keys(fw, "kintsugi_b", _Q_EVAL_KEYS, "AUDIT_P2_QB")
    _require_flag(qa, "task_pass", False, "AUDIT_P2_QA_FLAG")
    _require_flag(qb, "task_pass", True, "AUDIT_P2_QB_FLAG")
    qa_kls = _require_rational_list(qa["per_token_fisher_kl"], "qa.kls", 4, "AUDIT_P2_QA_VAL")
    qb_kls = _require_rational_list(qb["per_token_fisher_kl"], "qb.kls", 4, "AUDIT_P2_QB_VAL")
    qa_avg = parse_exact_rational(qa["average_fisher_kl"], "qa.avg")
    qb_avg = parse_exact_rational(qb["average_fisher_kl"], "qb.avg")
    fw_rat = parse_exact_rational(fw["kintsugi_over_uniform_kl_ratio"], "fw.rat")
    if (
        require_exact_int(fw["prompt_tokens"], "fw.tokens"),
        require_exact_int(fw["vocab_size"], "fw.vocab"),
        require_exact_int(fw["fisher_matrix_rank"], "fw.rank"),
        qa_kls,
        qa_avg,
        qa_avg == sum(qa_kls, Fraction(0, 1)) / Fraction(4, 1),
        _require_rational_list(qa["per_token_margins"], "qa.margins", 4, "AUDIT_P2_QA_VAL"),
        _require_int_list(qa["per_token_top1"], "qa.top1", 4, "AUDIT_P2_QA_VAL"),
        parse_exact_rational(qa["critical_token_margin"], "qa.cm"),
        parse_exact_rational(qa["min_margin"], "qa.mm"),
        qb_kls,
        qb_avg,
        qb_avg == sum(qb_kls, Fraction(0, 1)) / Fraction(4, 1),
        _require_rational_list(qb["per_token_margins"], "qb.margins", 4, "AUDIT_P2_QB_VAL"),
        _require_int_list(qb["per_token_top1"], "qb.top1", 4, "AUDIT_P2_QB_VAL"),
        parse_exact_rational(qb["critical_token_margin"], "qb.cm"),
        parse_exact_rational(qb["min_margin"], "qb.mm"),
        fw_rat,
        fw_rat * qa_avg == qb_avg,
        require_exact_str(fw["kintsugi_over_uniform_kl_ratio_4dp"], "fw.4dp"),
    ) != (
        4,
        4,
        3,
        [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1), Fraction(11, 128)],
        Fraction(11, 512),
        True,
        [Fraction(4, 1), Fraction(4, 1), Fraction(4, 1), Fraction(-1, 2)],
        [0, 0, 0, 1],
        Fraction(-1, 2),
        Fraction(-1, 2),
        [Fraction(31, 128), Fraction(31, 128), Fraction(31, 128), Fraction(11, 2048)],
        Fraction(1499, 8192),
        True,
        [Fraction(3, 1), Fraction(3, 1), Fraction(3, 1), Fraction(1, 4)],
        [0, 0, 0, 0],
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1499, 176),
        True,
        "8.5170",
    ):
        raise FisherKintsugiMemtError("AUDIT_P2_FISHER_VAL", "quadratic Fisher witness mismatch")

    dw = _require_dict_exact_keys(p2, "exact_dyadic_shannon_witness", _DYADIC_WIT_KEYS, "AUDIT_P2_DYADIC")
    _require_flag(dw, "uniform_a_task_pass", False, "AUDIT_P2_DYADIC_FLAG")
    for fk in ("kintsugi_b_task_pass", "gibbs_strict_positivity_certified", "uniform_lower_avg_kl_yet_breaks_task"):
        _require_flag(dw, fk, True, "AUDIT_P2_DYADIC_FLAG")
    da_avg = parse_exact_rational(dw["uniform_a_avg_kl_bits"], "dw.a_avg")
    db_avg = parse_exact_rational(dw["kintsugi_b_avg_kl_bits"], "dw.b_avg")
    dw_rat = parse_exact_rational(dw["kintsugi_over_uniform_kl_ratio"], "dw.rat")
    if (
        require_exact_int(dw["prompt_tokens"], "dw.tokens"),
        parse_exact_rational(dw["uniform_a_routine_kl_bits"], "dw.a_r"),
        parse_exact_rational(dw["uniform_a_critical_kl_bits"], "dw.a_c"),
        _require_int_list(dw["uniform_a_critical_log2_exponents"], "dw.a_ks", 3, "AUDIT_P2_DYADIC_VAL"),
        da_avg,
        parse_exact_rational(dw["uniform_a_critical_prob_margin"], "dw.a_m"),
        require_exact_int(dw["uniform_a_critical_top1"], "dw.a_t1"),
        parse_exact_rational(dw["kintsugi_b_routine_kl_bits"], "dw.b_r"),
        _require_int_list(dw["kintsugi_b_routine_log2_exponents"], "dw.b_ks", 4, "AUDIT_P2_DYADIC_VAL"),
        parse_exact_rational(dw["kintsugi_b_critical_kl_bits"], "dw.b_c"),
        db_avg,
        parse_exact_rational(dw["kintsugi_b_routine_prob_margin"], "dw.b_rm"),
        parse_exact_rational(dw["kintsugi_b_critical_prob_margin"], "dw.b_cm"),
        dw_rat,
        dw_rat * da_avg == db_avg,
    ) != (
        4,
        Fraction(0, 1),
        Fraction(1, 4),
        [1, -1, 0],
        Fraction(1, 16),
        Fraction(-1, 4),
        1,
        Fraction(7, 16),
        [1, -1, -2, -1],
        Fraction(0, 1),
        Fraction(21, 64),
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(21, 4),
        True,
    ):
        raise FisherKintsugiMemtError("AUDIT_P2_DYADIC_VAL", "exact dyadic Shannon KL witness mismatch")

    # Pillar 3: gbench P99 Order-Statistic Sample-Size Law
    p3 = _require_dict_exact_keys(payload, "pillar3_gbench_p99_order_statistics", _P3_KEYS, "AUDIT_P3_MISSING")
    for fk in ("sharp_boundary_458_fails_459_passes", "cs18_uncertified_floor_enforced"):
        _require_flag(p3, fk, True, "AUDIT_P3_FLAG")
    if (
        require_exact_int(p3["sharp_99pct_threshold_n"], "p3.n99"),
        require_exact_int(p3["gbench_slo_cohort_threshold_n"], "p3.n1099"),
    ) != (459, 1099):
        raise FisherKintsugiMemtError("AUDIT_P3_THRESH", "P99 sample-size thresholds mismatch")

    cohort_specs = (
        ("cohort_n100", 100, 665, 201, False, False, False, 2, 3, 366032, 366033, 735761, 735762, 920626, 920627, 633967, 633968, 3432, 3433, "UNCERTIFIED", "UNCERTIFIED"),
        ("cohort_n458", 458, 3043, 917, False, False, False, 99, 100, 10021, 10022, 56381, 56382, 163385, 163386, 989978, 989979, 483558, 483559, "UNCERTIFIED", "UNCERTIFIED"),
        ("cohort_n459", 459, 3050, 919, False, True, False, 100, 101, 9920, 9921, 55918, 55919, 162315, 162316, 990079, 990080, 485446, 485447, "CERTIFIED_UPPER_TAIL", "UNCERTIFIED"),
        ("cohort_n1099", 1099, 7302, 2199, True, True, True, 62650, 62651, 15, 16, 193, 194, 1175, 1176, 999984, 999985, 985150, 985151, "CERTIFIED_UPPER_TAIL", "CERTIFIED"),
    )
    for ck, *exp_vals in cohort_specs:
        cd = _require_dict_exact_keys(p3, ck, _COHORT_KEYS, "AUDIT_P3_COHORT")
        actual_vals = (
            require_exact_int(cd["n"], f"{ck}.n"),
            require_exact_int(cd["den_bit_length"], f"{ck}.bits"),
            require_exact_int(cd["den_decimal_digits"], f"{ck}.digs"),
            require_exact_bool(cd["den_exceeds_max_json_int_bits"], f"{ck}.exc"),
            require_exact_bool(cd["pass_99pct_upper_tail"], f"{ck}.p99"),
            require_exact_bool(cd["miss_prob_below_1_over_60000"], f"{ck}.p60k"),
            require_exact_int(cd["reciprocal_miss_prob_floor"], f"{ck}.inv_lo"),
            require_exact_int(cd["reciprocal_miss_prob_ceil"], f"{ck}.inv_hi"),
            require_exact_int(cd["p_zero_exceedances_ppm_lo"], f"{ck}.p0_lo"),
            require_exact_int(cd["p_zero_exceedances_ppm_hi"], f"{ck}.p0_hi"),
            require_exact_int(cd["p_le_1_exceedance_ppm_lo"], f"{ck}.p1_lo"),
            require_exact_int(cd["p_le_1_exceedance_ppm_hi"], f"{ck}.p1_hi"),
            require_exact_int(cd["p_le_2_exceedances_ppm_lo"], f"{ck}.p2_lo"),
            require_exact_int(cd["p_le_2_exceedances_ppm_hi"], f"{ck}.p2_hi"),
            require_exact_int(cd["p_ge_1_exceedance_ppm_lo"], f"{ck}.ge1_lo"),
            require_exact_int(cd["p_ge_1_exceedance_ppm_hi"], f"{ck}.ge1_hi"),
            require_exact_int(cd["p_ge_5_exceedances_ppm_lo"], f"{ck}.ge5_lo"),
            require_exact_int(cd["p_ge_5_exceedances_ppm_hi"], f"{ck}.ge5_hi"),
            require_exact_str(cd["cs18_upper_tail_status"], f"{ck}.u_stat"),
            require_exact_str(cd["cs18_gbench_slo_status"], f"{ck}.s_stat"),
        )
        if actual_vals != tuple(exp_vals):
            raise FisherKintsugiMemtError("AUDIT_P3_COHORT_VAL", f"{ck} order-statistic values mismatch")

    br = _require_dict_exact_keys(p3, "n1099_two_sided_order_stat_bracket", _BRACKET_KEYS, "AUDIT_P3_BRACKET")
    _require_flag(br, "coverage_exceeds_95pct", True, "AUDIT_P3_BRACKET_FLAG")
    if (
        require_exact_int(br["exceedance_range_min"], "br.emin"),
        require_exact_int(br["exceedance_range_max"], "br.emax"),
        require_exact_int(br["order_stat_lower_rank"], "br.rlo"),
        require_exact_int(br["order_stat_upper_rank"], "br.rhi"),
        require_exact_int(br["coverage_ppm_lo"], "br.clo"),
        require_exact_int(br["coverage_ppm_hi"], "br.chi"),
    ) != (4, 18, 1081, 1096, 978156, 978157):
        raise FisherKintsugiMemtError("AUDIT_P3_BRACKET_VAL", "n1099 two-sided order-statistic bracket mismatch")

    # Pillar 4: MemT GDN-2 Rank-Capacity Obstruction & Channel-2B Poisoning
    p4 = _require_dict_exact_keys(payload, "pillar4_memt_gdn2_rank_and_channel2b", _P4_KEYS, "AUDIT_P4_MISSING")
    cap = _require_dict_exact_keys(p4, "capacity_obstruction", _CAP_KEYS, "AUDIT_P4_CAP")
    for fk in (
        "hadamard_keys_orthonormal",
        "transpose_duality_l4_exact",
        "transpose_duality_l5_exact",
        "l4_zero_error_exact",
        "k5_unit_norm_exact",
        "nullspace_k5_c_zero_exact",
        "parseval_error_identity_exact",
        "capacity_obstruction_certified",
    ):
        _require_flag(cap, fk, True, "AUDIT_P4_CAP_FLAG")
    l5_errs = _require_rational_list(cap["l5_per_key_err_sq"], "cap.l5_errs", 5, "AUDIT_P4_CAP_VAL")
    l5_tot = parse_exact_rational(cap["l5_total_err_sq"], "cap.l5_tot")
    cs_lb = parse_exact_rational(cap["universal_cauchy_schwarz_lb"], "cap.cs_lb")
    frob_ds = parse_exact_rational(cap["delta_s_frob_norm_sq"], "cap.frob_ds")
    if (
        require_exact_int(cap["state_dim_d"], "cap.d"),
        require_exact_int(cap["rank_s4"], "cap.rs4"),
        _require_rational_list(cap["l4_per_key_err_sq"], "cap.l4_errs", 4, "AUDIT_P4_CAP_VAL"),
        parse_exact_rational(cap["l4_total_err_sq"], "cap.l4_tot"),
        require_exact_int(cap["rank_key_matrix_k5"], "cap.rk5"),
        require_exact_int(cap["rank_s5"], "cap.rs5"),
        _require_rational_list(cap["nullspace_witness_c"], "cap.c", 5, "AUDIT_P4_CAP_VAL"),
        _require_rational_list(cap["inconsistency_vector_d"], "cap.d_vec", 4, "AUDIT_P4_CAP_VAL"),
        cs_lb,
        l5_errs,
        l5_tot,
        frob_ds,
        l5_tot == sum(l5_errs, Fraction(0, 1)),
        l5_tot == frob_ds,
        l5_tot >= cs_lb,
    ) != (
        4,
        4,
        [Fraction(0, 1)] * 4,
        Fraction(0, 1),
        4,
        4,
        [Fraction(7, 10), Fraction(-1, 10), Fraction(7, 10), Fraction(-1, 10), Fraction(-1, 1)],
        [Fraction(7, 5), Fraction(-31, 5), Fraction(7, 5), Fraction(-1, 5)],
        Fraction(106, 5),
        [Fraction(2597, 125), Fraction(53, 125), Fraction(2597, 125), Fraction(53, 125), Fraction(0, 1)],
        Fraction(212, 5),
        Fraction(212, 5),
        True,
        True,
        True,
    ):
        raise FisherKintsugiMemtError("AUDIT_P4_CAP_VAL", "GDN-2 rank-capacity invariants mismatch")

    pois = _require_dict_exact_keys(p4, "channel_2b_poisoning_witness", _POIS_KEYS, "AUDIT_P4_POIS")
    for fk in (
        "frozen_weights_delta_w_zero",
        "benign_query_delta_p_data_zero",
        "zero_kv_carryover_delta_kv_zero",
        "state_hashes_distinct",
        "benign_readout_sign_flipped",
        "poison_persists_after_benign_turn",
    ):
        _require_flag(pois, fk, True, "AUDIT_P4_POIS_FLAG")
    exp_s4_sha, exp_s5_sha = _derive_gdn2_state_hashes()
    c_sha = require_sha256_hex(pois["clean_state_sha256"], "pois.c_sha")
    p_sha = require_sha256_hex(pois["poisoned_state_sha256"], "pois.p_sha")
    if (
        hmac.compare_digest(c_sha, exp_s4_sha),
        hmac.compare_digest(p_sha, exp_s5_sha),
        require_exact_int(pois["delta_s_exact_rank"], "pois.rank"),
        parse_exact_rational(pois["delta_s_frob_norm_sq"], "pois.frob"),
        parse_exact_rational(pois["clean_margin"], "pois.cm"),
        parse_exact_rational(pois["poisoned_margin"], "pois.pm"),
        parse_exact_rational(pois["margin_after_benign_orth_turn"], "pois.om"),
    ) != (True, True, 1, Fraction(212, 5), Fraction(2, 1), Fraction(-83, 25), Fraction(-83, 25)):
        raise FisherKintsugiMemtError("AUDIT_P4_POIS_VAL", "Channel-2B poisoning witness mismatch")

    air = _require_dict_exact_keys(p4, "four_channel_airlock", _AIR_KEYS, "AUDIT_P4_AIR")
    for fk in (
        "legacy_3ch_blind_to_channel2b",
        "all_four_negative_controls_denied",
        "airlock_extension_certified",
    ):
        _require_flag(air, fk, True, "AUDIT_P4_AIR_FLAG")
    clean_r = _require_dict_exact_keys(air, "clean_record", _AIR_RES_KEYS, "AUDIT_P4_AIR_CLEAN")
    if (
        require_exact_str(clean_r["legacy_3ch_decision"], "clean.3ch"),
        require_exact_str(clean_r["airlock_4ch_decision"], "clean.4ch"),
        _require_str_list(clean_r["violations_3ch"], "clean.v3", 0, "AUDIT_P4_AIR_CLEAN"),
        _require_str_list(clean_r["violations_4ch"], "clean.v4", 0, "AUDIT_P4_AIR_CLEAN"),
    ) != ("ALLOW", "ALLOW", [], []):
        raise FisherKintsugiMemtError("AUDIT_P4_AIR_CLEAN", "clean 4-channel airlock record mismatch")

    negs = _require_list_exact_len(air["negative_controls"], "air.negs", 4, "AUDIT_P4_AIR_NEG")
    neg_exp = (
        ("channel1_weight_tamper", "DENY", "DENY", ["CHANNEL_1_WEIGHT_INTEGRITY"], ["CHANNEL_1_WEIGHT_INTEGRITY"]),
        ("channel2a_kv_reuse", "DENY", "DENY", ["CHANNEL_2A_KV_PERSISTENCE"], ["CHANNEL_2A_KV_PERSISTENCE"]),
        ("channel2b_latent_poisoning", "ALLOW", "DENY", [], ["CHANNEL_2B_LATENT_MEMORY_POISONING"]),
        ("channel3_data_cot_drop", "DENY", "DENY", ["CHANNEL_3_DATA_PROVENANCE"], ["CHANNEL_3_DATA_PROVENANCE"]),
    )
    for idx, item in enumerate(negs):
        if type(item) is not dict or frozenset(item.keys()) != _AIR_NEG_KEYS:
            raise FisherKintsugiMemtError("AUDIT_P4_AIR_NEG", "negative control must have exact keys")
        exp_name, exp_3, exp_4, exp_v3, exp_v4 = neg_exp[idx]
        if (
            require_exact_str(item["name"], "neg.name"),
            require_exact_str(item["legacy_3ch_decision"], "neg.3ch"),
            require_exact_str(item["airlock_4ch_decision"], "neg.4ch"),
            _require_str_list(item["violations_3ch"], "neg.v3", len(exp_v3), "AUDIT_P4_AIR_NEG"),
            _require_str_list(item["violations_4ch"], "neg.v4", len(exp_v4), "AUDIT_P4_AIR_NEG"),
        ) != (exp_name, exp_3, exp_4, exp_v3, exp_v4):
            raise FisherKintsugiMemtError("AUDIT_P4_AIR_NEG", f"negative control[{idx}] mismatch")

    return {
        "secondary_audit_passed": True,
        "pillars_audited": 4,
        "fisher_spectra_audited": 3,
        "p99_cohorts_audited": 4,
        "airlock_negative_controls_audited": 4,
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
        raise FisherKintsugiMemtError("CRLF_FORBIDDEN", "CRLF line endings forbidden in certificate")
    if not raw.endswith(b"\n"):
        raise FisherKintsugiMemtError("MISSING_TRAILING_LF", "certificate must end with a single LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FisherKintsugiMemtError("INVALID_UTF8", "certificate is not valid UTF-8") from exc
    cert = parse_strict_json_text(text, require_object=True)
    canonical_raw = canonical_json_bytes(cert)
    if raw != canonical_raw + b"\n":
        raise FisherKintsugiMemtError("NON_CANONICAL_BYTES", "certificate bytes are not canonical ASCII JSON + LF")
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
# Strict Fail-Closed Manual CLI Parsing (allow_abbrev=False Semantics)
# ---------------------------------------------------------------------------

def _parse_cli_argv(argv: Sequence[str]) -> tuple[str, Path | None, bool]:
    if type(argv) not in (list, tuple):
        raise FisherKintsugiMemtError("INVALID_ARGV", "argv must be exact list/tuple of strict str")
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
                raise FisherKintsugiMemtError("DUPLICATE_FLAG", "duplicate --json flag")
            as_json = True
            i += 1
        elif tok == "--standalone":
            if mode is not None:
                raise FisherKintsugiMemtError("CONFLICTING_MODE", "multiple mode flags forbidden")
            mode = "standalone"
            i += 1
        elif tok in ("--check", "--write"):
            if mode is not None:
                raise FisherKintsugiMemtError("CONFLICTING_MODE", "multiple mode flags forbidden")
            if i + 1 >= n:
                raise FisherKintsugiMemtError("MISSING_PATH_ARG", f"{tok} requires a path argument")
            val = argv[i + 1]
            if not val or val.startswith("-"):
                raise FisherKintsugiMemtError("INVALID_PATH_ARG", f"{tok} path cannot be empty or flag-like")
            mode = "check" if tok == "--check" else "write"
            target = Path(val)
            i += 2
        else:
            raise FisherKintsugiMemtError("UNKNOWN_FLAG", f"unrecognized CLI argument: {tok}")

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
    except FisherKintsugiMemtError as exc:
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
