#!/usr/bin/env python3
"""Track S47 (Portal 52 / siliquant#77): Zone-B Hardened CLI, Full Pure-Z/Q Replay &
Cross-Multiplied Deep Secondary Auditor.

100% pure Python 3 stdlib, zero IEEE-754 floats, zero bare asserts, zero .isdigit(),
zero islink/is_symlink, zero mathematical RNG (secrets.token_hex used strictly for atomic
temp filename uniqueness).
"""
from __future__ import annotations

import errno
from fractions import Fraction
import functools
import hashlib
import hmac
import json
import math
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

import hamming_construction_a_oracle as oracle  # noqa: E402

MAX_JSON_BYTES = 1_048_576
MAX_ALLOWED_CAP_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 65_536
MAX_JSON_DEPTH = 64
MAX_JSON_INT_DIGITS = oracle.MAX_INT_DIGITS
_EXACT_PATH_TYPES = (Path, type(Path()))
_HEX_LOWER = frozenset("0123456789abcdef")
_ASCII_DIGITS = frozenset("0123456789")
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_TEMP_NAME_LOCK = threading.Lock()
_TEMP_COUNTER = 0

_CERT_KEYS = frozenset({"schema_version", "track_id", "payload_sha256", "certificate_sha256", "payload"})
_PAYLOAD_KEYS = frozenset({
    "track_id",
    "portal_ref",
    "novelty_class",
    "measured_live_model_weights",
    "prior_literature_n0",
    "pillar1_hamming_codes_and_lattices",
    "pillar2_syndrome_decoder_optimality",
    "pillar3_voronoi_zero_retention_and_accumulators",
    "pillar4_honest_scope_and_proxy_obstruction",
    "summary",
})
_P1_KEYS = frozenset({
    "lattices",
    "e8_hadamard_a8_bijection",
    "barnes_wall_lambda16_comparison",
    "strict_packing_gain_ordering_A8_lt_A16_lt_BW16_lt_A32",
    "all_verified",
})
_P1_LAT_KEYS = frozenset({
    "name",
    "m",
    "n",
    "k",
    "d_min",
    "code_rate_q",
    "codeword_count",
    "dual_codeword_count",
    "h_g_orthogonal_mod2",
    "exhaustive_weight_enum_checked",
    "exhaustive_matches_macwilliams",
    "a4_weight4_count",
    "macwilliams_a4_exact",
    "lambda1_sq_A_n",
    "kissing_number_tau",
    "covolume_A_n_q",
    "det_gram_A_n_q",
    "det_gram_matches_formula",
    "gamma_pow_n",
    "gamma_pow_32",
    "verified",
})
_P1_E8_KEYS = frozenset({
    "e8_root_count",
    "mapped_a8_min_vector_count",
    "all_images_integral",
    "all_images_norm_sq_4_in_L8",
    "all_images_mod2_in_C8",
    "bijection_exact_240_of_240",
})
_P1_BW_KEYS = frozenset({
    "a16_kissing",
    "barnes_wall_lambda16_kissing",
    "a16_sparser_than_lambda16",
    "a16_gamma_pow32",
    "barnes_wall_lambda16_gamma_pow32",
})
_P2_KEYS = frozenset({
    "c8_vectors_tested",
    "c8_exact_matches_16cw_bruteforce",
    "c8_exact_matches_rotated_e8",
    "c8_fast_wagner_certified_count",
    "c16_vectors_tested",
    "c16_exact_matches_2048cw_bruteforce",
    "c16_fast_wagner_certified_count",
    "c32_vectors_tested",
    "c32_valid_and_le_wagner_count",
    "naive_wagner_suboptimality_controls",
    "all_verified",
})
_P2_CTRL_KEYS = frozenset({
    "name",
    "n",
    "naive_wagner_sq_dist_q",
    "exact_bnb_sq_dist_q",
    "certified_optimal_sq_dist_q",
    "exhaustive_bruteforce_executed",
    "optimal_proof_method",
    "naive_wagner_certified_optimal",
    "exact_strictly_beats_naive_wagner",
    "exact_matches_certified_optimal",
})
_P3_KEYS = frozenset({
    "inverse_scale_enclosures_verified",
    "unit_covolume_dead_zones",
    "dead_zone_strictly_expanding",
    "discrete_laplace_zero_retention",
    "laplace_zero_retention_strictly_increasing",
    "midband_parity_spill_control",
    "integer_accumulator_bounds",
    "wht_int8_bounds",
    "f64_exact_decode_theorem",
    "all_verified",
})
_P3_DZ_KEYS = frozenset({"name", "n", "k", "code_rate_q", "r_pack_sq_pow8_q", "alpha_inf_pow16_q"})
_P3_ZR_KEYS = frozenset({
    "name", "n", "inv_scale_bracket_q", "zero_coords", "total_coords", "zero_retention_q", "zero_retention_dec6",
})
_P3_SPILL_KEYS = frozenset({"lattice", "nonzero_coords_after_decode", "weight4_parity_spill_triggered"})
_P3_ACC_KEYS = frozenset({"name", "width", "w_max", "act_abs_max", "max_abs_dot", "fits_int16", "fits_int32"})
_P3_WHT_KEYS = frozenset({"name", "n", "max_abs_wht_of_int8", "fits_int16"})
_P3_F64_KEYS = frozenset({
    "group_width",
    "f64_exact_int_limit_2p53",
    "f16_max_finite_scaled_2p24",
    "sharp_unconditional_w_max",
    "w64_max_scaled_sum",
    "w64_unconditionally_exact_in_f64",
    "w65_failure_witness_scaled_sum",
    "w65_exceeds_2p53_and_is_odd",
    "int8_w128_exact_when_act_le_32752",
})
_P4_KEYS = frozenset({
    "measured_live_model_weights",
    "siliquant_issue_ref",
    "data_free_column_proxy_obstruction",
    "all_verified",
})
_P4_REF_KEYS = frozenset({"repo", "issue_number", "survey_commit", "plan_commit"})
_P4_PROXY_KEYS = frozenset({
    "weight_col_rms_sq_q",
    "data_free_inv_rms_sq_proxy_q",
    "true_activation_hessian_diag_q",
    "proxy_col0_over_col1_ratio_q",
    "true_col1_over_col0_ratio_q",
    "total_ratio_distortion_q",
    "ranking_sign_inverted",
    "calibration_activations_required",
})
_SUMMARY_KEYS = frozenset({
    "pillar_count",
    "all_four_pillars_verified",
    "float_count_on_trust_path",
    "bare_assert_count",
    "rng_calls",
})

# Bijective set of ALL boolean leaf keys anywhere inside payload (exact bool <-> key membership;
# nullable boolean keys accept exact bool or None only — never int/str/float).
_BOOL_LEAF_KEYS = frozenset({
    "measured_live_model_weights",
    "all_verified",
    "h_g_orthogonal_mod2",
    "exhaustive_weight_enum_checked",
    "macwilliams_a4_exact",
    "det_gram_matches_formula",
    "verified",
    "all_images_integral",
    "all_images_norm_sq_4_in_L8",
    "all_images_mod2_in_C8",
    "bijection_exact_240_of_240",
    "a16_sparser_than_lambda16",
    "strict_packing_gain_ordering_A8_lt_A16_lt_BW16_lt_A32",
    "exhaustive_bruteforce_executed",
    "naive_wagner_certified_optimal",
    "exact_strictly_beats_naive_wagner",
    "exact_matches_certified_optimal",
    "inverse_scale_enclosures_verified",
    "dead_zone_strictly_expanding",
    "laplace_zero_retention_strictly_increasing",
    "weight4_parity_spill_triggered",
    "fits_int16",
    "fits_int32",
    "w64_unconditionally_exact_in_f64",
    "w65_exceeds_2p53_and_is_odd",
    "int8_w128_exact_when_act_le_32752",
    "ranking_sign_inverted",
    "calibration_activations_required",
    "all_four_pillars_verified",
})
_NULLABLE_BOOL_LEAF_KEYS = frozenset({"exhaustive_matches_macwilliams"})


# ============================================================================
# 1. Zone-B Path, CWD, Atomic dir_fd Traversal & Durable 0o600 Atomic Write
# ============================================================================

def _require_utf8_no_surrogate(text: Any, label: str) -> str:
    if type(text) is not str:
        raise oracle.HammingLatticeError("INVALID_STR", f"{label}: exact str required")
    oracle.require_ok("\x00" not in text, "NUL_BYTE_FORBIDDEN", f"{label}: NUL byte forbidden")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise oracle.HammingLatticeError("INVALID_UTF8_SURROGATE", f"{label}: lone UTF-16 surrogate forbidden") from exc
    return text


def _require_concrete_path(path: Any, label: str) -> Path:
    oracle.require_ok(type(path) in _EXACT_PATH_TYPES, "INVALID_PATH_TYPE", f"{label}: concrete pathlib.Path required")
    _require_utf8_no_surrogate(str(path), label)
    return path


def _safe_cwd(label: str) -> Path:
    try:
        return Path.cwd()
    except OSError as exc:
        raise oracle.HammingLatticeError("CWD_UNAVAILABLE", f"{label}: CWD unavailable ({type(exc).__name__})") from exc


def _check_pwd_no_symlinks(label: str) -> None:
    pwd = os.environ.get("PWD", "")
    if pwd.startswith("/"):
        _require_utf8_no_surrogate(pwd, f"{label} PWD")
        pwd_path = Path(pwd)
        try:
            dfd = os.open("/", _DIR_OPEN_FLAGS)
        except OSError as exc:
            raise oracle.HammingLatticeError("CWD_UNAVAILABLE", f"{label}: cannot open root for PWD") from exc
        try:
            for part in pwd_path.parts[1:]:
                if part == ".":
                    continue
                try:
                    next_fd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise oracle.HammingLatticeError(
                            "SYMLINK_ANCESTOR_FORBIDDEN", f"{label}: symlink PWD ancestor forbidden"
                        ) from exc
                    raise oracle.HammingLatticeError(
                        "CWD_UNAVAILABLE", f"{label}: PWD ancestor unavailable ({type(exc).__name__})"
                    ) from exc
                os.close(dfd)
                dfd = next_fd
        finally:
            os.close(dfd)


def _open_dir_chain(target: Path, label: str, *, create_parents: bool) -> tuple[int, str]:
    if target.is_absolute():
        parts = target.parts[1:]
    else:
        cwd = _safe_cwd(label)
        _check_pwd_no_symlinks(label)
        parts = cwd.parts[1:] + target.parts
    oracle.require_ok(
        len(parts) >= 1 and parts[-1] not in ("", ".", ".."),
        "NOT_A_REGULAR_FILE",
        f"{label}: path does not name a regular file leaf",
    )
    parent_parts = parts[:-1]
    leaf = parts[-1]
    err_code = "FILE_WRITE_ERROR" if create_parents else "FILE_READ_ERROR"
    try:
        dfd = os.open("/", _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise oracle.HammingLatticeError(err_code, f"{label}: cannot open root") from exc
    try:
        for part in parent_parts:
            if part == ".":
                continue
            try:
                next_fd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
            except OSError as exc:
                if create_parents and exc.errno == errno.ENOENT:
                    try:
                        os.mkdir(part, 0o755, dir_fd=dfd)
                    except FileExistsError:
                        pass
                    except OSError as mk_exc:
                        raise oracle.HammingLatticeError(
                            "FILE_WRITE_ERROR", f"{label}: mkdir failed ({type(mk_exc).__name__})"
                        ) from mk_exc
                    try:
                        next_fd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
                    except OSError as re_exc:
                        raise oracle.HammingLatticeError(
                            "FILE_WRITE_ERROR", f"{label}: cannot open created dir ({type(re_exc).__name__})"
                        ) from re_exc
                elif exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise oracle.HammingLatticeError(
                        "SYMLINK_FORBIDDEN", f"{label}: symlink or non-directory ancestor forbidden"
                    ) from exc
                else:
                    raise oracle.HammingLatticeError(
                        err_code, f"{label}: cannot traverse ancestor ({type(exc).__name__})"
                    ) from exc
            os.close(dfd)
            dfd = next_fd
        return dfd, leaf
    except BaseException:
        os.close(dfd)
        raise


def read_regular_bytes_capped(path: Path, label: str = "file", max_bytes: int = MAX_JSON_BYTES) -> bytes:
    target = _require_concrete_path(path, label)
    oracle.require_exact_int(max_bytes, "max_bytes", lo=1, hi=MAX_ALLOWED_CAP_BYTES)
    dfd, leaf = _open_dir_chain(target, label, create_parents=False)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        try:
            fd = os.open(leaf, flags, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise oracle.HammingLatticeError("SYMLINK_FORBIDDEN", f"{label}: symlink leaf forbidden") from exc
            raise oracle.HammingLatticeError(
                "FILE_READ_ERROR", f"{label}: cannot open ({type(exc).__name__})"
            ) from exc
    finally:
        os.close(dfd)

    chunks: list[bytes] = []
    total = 0
    try:
        st_f = os.fstat(fd)
        oracle.require_ok(stat.S_ISREG(st_f.st_mode), "NOT_A_REGULAR_FILE", f"{label}: not a regular file")
        while total <= max_bytes:
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except oracle.HammingLatticeError:
        raise
    except OSError as exc:
        raise oracle.HammingLatticeError("FILE_READ_ERROR", f"{label}: read failed ({type(exc).__name__})") from exc
    finally:
        os.close(fd)
    oracle.require_ok(total <= max_bytes, "FILE_TOO_LARGE", f"{label}: exceeds {max_bytes} bytes")
    return b"".join(chunks)


def write_regular_bytes_safe(path: Path, payload: bytes, label: str = "output") -> None:
    global _TEMP_COUNTER
    oracle.require_ok(type(payload) is bytes, "INVALID_PAYLOAD_BYTES", f"{label}: payload must be exact bytes")
    target = _require_concrete_path(path, label)
    dfd, leaf = _open_dir_chain(target, label, create_parents=True)
    tmp_name: str | None = None
    try:
        try:
            st_existing = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st_existing = None
        except OSError as exc:
            raise oracle.HammingLatticeError(
                "FILE_WRITE_ERROR", f"{label}: lstat failed ({type(exc).__name__})"
            ) from exc
        if st_existing is not None:
            oracle.require_ok(not stat.S_ISLNK(st_existing.st_mode), "SYMLINK_FORBIDDEN", f"{label}: symlink leaf forbidden")
            oracle.require_ok(stat.S_ISREG(st_existing.st_mode), "NOT_A_REGULAR_FILE", f"{label}: non-regular target")
            oracle.require_ok(st_existing.st_nlink <= 1, "HARDLINK_FORBIDDEN", f"{label}: hardlinked target forbidden")

        pid = os.getpid()
        for _ in range(32):
            with _TEMP_NAME_LOCK:
                _TEMP_COUNTER += 1
                seq_num = _TEMP_COUNTER
                cand = f".s47.{pid}.{seq_num}.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(
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
                raise oracle.HammingLatticeError(
                    "FILE_WRITE_ERROR", f"{label}: create temp failed ({type(exc).__name__})"
                ) from exc
        else:
            raise oracle.HammingLatticeError("FILE_WRITE_ERROR", f"{label}: exhausted temp file slots")

        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                written += os.write(fd, view[written:])
            os.fsync(fd)
            st_tmp = os.fstat(fd)
            oracle.require_ok(st_tmp.st_nlink == 1, "HARDLINK_FORBIDDEN", f"{label}: temp link count altered")
        except oracle.HammingLatticeError:
            raise
        except OSError as exc:
            raise oracle.HammingLatticeError(
                "FILE_WRITE_ERROR", f"{label}: write failed ({type(exc).__name__})"
            ) from exc
        finally:
            os.close(fd)

        try:
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            tmp_name = None
            os.fsync(dfd)
        except OSError as exc:
            raise oracle.HammingLatticeError(
                "FILE_WRITE_ERROR", f"{label}: atomic replace failed ({type(exc).__name__})"
            ) from exc
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


# ============================================================================
# 2. Strict ASCII Parsers, Binding 3,321-Bit Cap & Canonical JSON
# ============================================================================

def require_hex64_lower_ascii(val: Any, label: str = "sha256") -> str:
    s = _require_utf8_no_surrogate(val, label)
    oracle.require_ok(len(s) == 64, "INVALID_HEX64", f"{label}: expected 64-char lowercase ASCII hex")
    for ch in s:
        oracle.require_ok(ch in _HEX_LOWER, "INVALID_HEX64", f"{label}: non-hex-lowercase char")
    return s


def _parse_bounded_int(token: str) -> int:
    s = _require_utf8_no_surrogate(token, "json_int")
    oracle.require_ok(s != "-0", "INVALID_JSON_INT", "negative zero forbidden")
    digits = s[1:] if s.startswith("-") else s
    oracle.require_ok(len(digits) >= 1, "INVALID_JSON_INT", "JSON int has no digits")
    oracle.require_ok(len(digits) <= MAX_JSON_INT_DIGITS, "INT_DIGIT_LIMIT", "JSON int exceeds digit cap")
    oracle.require_ok(digits == "0" or not digits.startswith("0"), "INVALID_JSON_INT", "leading zero forbidden")
    for ch in digits:
        oracle.require_ok(ch in _ASCII_DIGITS, "INVALID_JSON_INT", "non-ASCII digit forbidden")
    val = int(s)
    oracle.require_ok(val.bit_length() <= oracle.MAX_INT_BITS, "INT_BIT_LIMIT", "JSON int exceeds 3,321-bit cap")
    return val


def parse_exact_rational(token: Any, label: str = "rational") -> Fraction:
    s = _require_utf8_no_surrogate(token, label)
    parts = s.split("/")
    oracle.require_ok(len(parts) in (1, 2), "INVALID_RATIONAL", f"{label}: expected 'n' or 'n/d'")
    num = _parse_bounded_int(parts[0])
    if len(parts) == 1:
        return oracle.require_exact_fraction(Fraction(num, 1), label)
    den = _parse_bounded_int(parts[1])
    oracle.require_ok(den >= 2, "INVALID_RATIONAL", f"{label}: denominator must be >= 2")
    oracle.require_ok(math.gcd(abs(num), den) == 1, "NON_CANONICAL_RATIONAL", f"{label}: not in lowest terms")
    return oracle.require_exact_fraction(Fraction(num, den), label)


def _validate_json_tree(root: Any, label: str = "json") -> None:
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        oracle.require_ok(depth <= MAX_JSON_DEPTH, "JSON_DEPTH_LIMIT", f"{label}: nesting exceeds {MAX_JSON_DEPTH}")
        t = type(node)
        if node is None or t is bool:
            continue
        if t is int:
            oracle.require_ok(node.bit_length() <= oracle.MAX_INT_BITS, "INT_BIT_LIMIT", f"{label}: int exceeds 3,321 bits")
            continue
        if t is str:
            _require_utf8_no_surrogate(node, label)
            continue
        if t is list:
            nxt = depth + 1
            for item in node:
                stack.append((item, nxt))
            continue
        if t is dict:
            nxt = depth + 1
            for k, v in node.items():
                _require_utf8_no_surrogate(k, f"{label}.key")
                stack.append((v, nxt))
            continue
        raise oracle.HammingLatticeError("FORBIDDEN_JSON_TYPE", f"{label}: unsupported type {t.__name__}")


def _reject_float(token: str) -> Any:
    raise oracle.HammingLatticeError("FLOAT_LITERAL_FORBIDDEN", f"float literal forbidden: {token}")


def _reject_constant(token: str) -> Any:
    raise oracle.HammingLatticeError("NONFINITE_LITERAL_FORBIDDEN", f"non-finite constant forbidden: {token}")


def _strict_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        _require_utf8_no_surrogate(k, "json_key")
        oracle.require_ok(k not in out, "DUPLICATE_JSON_KEY", f"duplicate key {k!r}")
        out[k] = v
    return out


def parse_strict_json_text(text: Any, *, require_object: bool = True, max_bytes: int = MAX_JSON_BYTES) -> Any:
    s = _require_utf8_no_surrogate(text, "json_text")
    oracle.require_exact_bool(require_object, "require_object")
    oracle.require_exact_int(max_bytes, "max_bytes", lo=1, hi=MAX_ALLOWED_CAP_BYTES)
    oracle.require_ok(len(s.encode("utf-8")) <= max_bytes, "JSON_TOO_LARGE", "JSON text exceeds byte cap")
    try:
        parsed = json.loads(
            s,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            parse_int=_parse_bounded_int,
            object_pairs_hook=_strict_pairs_hook,
        )
    except oracle.HammingLatticeError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise oracle.HammingLatticeError("JSON_PARSE_ERROR", f"invalid JSON ({type(exc).__name__})") from exc
    _validate_json_tree(parsed, "parsed_json")
    if require_object and type(parsed) is not dict:
        raise oracle.HammingLatticeError("JSON_ROOT_NOT_OBJECT", "JSON root must be an object")
    return parsed


def canonical_json_bytes(obj: Any) -> bytes:
    _validate_json_tree(obj, "canonical_json")
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


# ============================================================================
# 3. Bijective Bool-Leaf Guard & Cross-Multiplied Deep Secondary Auditor
# ============================================================================

def _req_dict(val: Any, label: str, expected_keys: frozenset[str]) -> dict[str, Any]:
    oracle.require_ok(type(val) is dict, "INVALID_SCHEMA_DICT", f"{label}: exact dict required")
    oracle.require_ok(frozenset(val.keys()) == expected_keys, "INVALID_SCHEMA_KEYS", f"{label}: key set mismatch")
    return val


def _req_list(val: Any, label: str, exact_len: int) -> list[Any]:
    oracle.require_ok(type(val) is list and len(val) == exact_len, "INVALID_SCHEMA_LIST", f"{label}: list len {exact_len} required")
    return val


def _req_int_f(obj: dict[str, Any], key: str, label: str) -> int:
    return oracle.require_exact_int(obj[key], f"{label}.{key}")


def _validate_bool_leaf_bijection(root: dict[str, Any]) -> None:
    """Enforce strict bijection: bool leaves accept exact bool (or None for nullable bool keys)."""
    stack: list[Any] = [root]
    while stack:
        node = stack.pop()
        if type(node) is dict:
            for k, v in node.items():
                if k in _BOOL_LEAF_KEYS:
                    oracle.require_exact_bool(v, f"bool_leaf.{k}")
                elif k in _NULLABLE_BOOL_LEAF_KEYS:
                    if v is not None:
                        oracle.require_exact_bool(v, f"nullable_bool_leaf.{k}")
                else:
                    oracle.require_ok(type(v) is not bool, "UNEXPECTED_BOOL_LEAF", f"{k}: bool forbidden on non-bool key")
                    stack.append(v)
        elif type(node) is list:
            for item in node:
                oracle.require_ok(type(item) is not bool, "UNEXPECTED_BOOL_ITEM", "bool forbidden inside list")
                stack.append(item)


@functools.lru_cache(maxsize=1)
def _reference_payload() -> dict[str, Any]:
    return oracle.build_payload()


def secondary_audit_payload(payload: Any) -> dict[str, Any]:
    """Fail-closed nested-keyset + JSON-tree + bijective-bool + cross-multiplied algebraic audit.

    ZERO division by any parsed rational token, and strict bounds before every shift/power/floor-div
    so injected '0', negative ints, or floats can never trigger raw exceptions.
    """
    pl = _req_dict(payload, "payload", _PAYLOAD_KEYS)
    _validate_json_tree(pl, "payload")
    _validate_bool_leaf_bijection(pl)

    p1 = _req_dict(pl["pillar1_hamming_codes_and_lattices"], "p1", _P1_KEYS)
    p1_lats = _req_list(p1["lattices"], "p1.lattices", 3)
    p1_e8 = _req_dict(p1["e8_hadamard_a8_bijection"], "p1.e8", _P1_E8_KEYS)
    p1_bw = _req_dict(p1["barnes_wall_lambda16_comparison"], "p1.bw", _P1_BW_KEYS)

    p2 = _req_dict(pl["pillar2_syndrome_decoder_optimality"], "p2", _P2_KEYS)
    p2_ctrls = _req_list(p2["naive_wagner_suboptimality_controls"], "p2.controls", 3)

    p3 = _req_dict(pl["pillar3_voronoi_zero_retention_and_accumulators"], "p3", _P3_KEYS)
    p3_dz = _req_list(p3["unit_covolume_dead_zones"], "p3.dead_zones", 3)
    p3_zr = _req_list(p3["discrete_laplace_zero_retention"], "p3.zero_retention", 3)
    p3_sp = _req_dict(p3["midband_parity_spill_control"], "p3.spill", _P3_SPILL_KEYS)
    p3_acc = _req_list(p3["integer_accumulator_bounds"], "p3.acc", 4)
    p3_wht = _req_list(p3["wht_int8_bounds"], "p3.wht", 3)
    p3_f64 = _req_dict(p3["f64_exact_decode_theorem"], "p3.f64", _P3_F64_KEYS)

    p4 = _req_dict(pl["pillar4_honest_scope_and_proxy_obstruction"], "p4", _P4_KEYS)
    _req_dict(p4["siliquant_issue_ref"], "p4.ref", _P4_REF_KEYS)
    p4_px = _req_dict(p4["data_free_column_proxy_obstruction"], "p4.proxy", _P4_PROXY_KEYS)

    summ = _req_dict(pl["summary"], "summary", _SUMMARY_KEYS)
    lit = _req_list(pl["prior_literature_n0"], "prior_literature_n0", 7)
    for idx, entry in enumerate(lit):
        _require_utf8_no_surrogate(entry, f"prior_literature_n0[{idx}]")

    # P1: Cross-multiplied lattice identities (ZERO division by parsed rationals; bounds before shifts!)
    p1_rows_ok = True
    for row in p1_lats:
        ld = _req_dict(row, "p1.lat", _P1_LAT_KEYS)
        m_v = _req_int_f(ld, "m", "p1.lat")
        n_v = _req_int_f(ld, "n", "p1.lat")
        k_v = _req_int_f(ld, "k", "p1.lat")
        _req_int_f(ld, "d_min", "p1.lat")
        _req_int_f(ld, "codeword_count", "p1.lat")
        _req_int_f(ld, "dual_codeword_count", "p1.lat")
        _req_int_f(ld, "lambda1_sq_A_n", "p1.lat")
        a4_v = _req_int_f(ld, "a4_weight4_count", "p1.lat")
        tau_v = _req_int_f(ld, "kissing_number_tau", "p1.lat")
        gp_n = _req_int_f(ld, "gamma_pow_n", "p1.lat")
        gp_32 = _req_int_f(ld, "gamma_pow_32", "p1.lat")
        rate_q = parse_exact_rational(ld["code_rate_q"], "p1.lat.rate")
        covol_q = parse_exact_rational(ld["covolume_A_n_q"], "p1.lat.covol")
        det_q = parse_exact_rational(ld["det_gram_A_n_q"], "p1.lat.det")
        if m_v not in (3, 4, 5):
            p1_rows_ok = False
            continue
        if not all((n_v == (1 << m_v), k_v == n_v - m_v - 1)):
            p1_rows_ok = False
            continue
        expected_ex_checked = m_v <= 4
        expected_ex_mw = True if expected_ex_checked else None
        if not all((
            rate_q.numerator * n_v == k_v * rate_q.denominator,
            24 * a4_v == n_v * (n_v - 1) * (n_v - 2),
            tau_v == 2 * n_v + 16 * a4_v,
            covol_q > 0,
            covol_q * covol_q == det_q,
            covol_q.numerator * (1 << k_v) == (1 << (n_v // 2)) * covol_q.denominator,
            det_q.numerator * (1 << (2 * k_v)) == (1 << n_v) * det_q.denominator,
            gp_n * det_q.numerator == (1 << n_v) * det_q.denominator,
            gp_32 == gp_n ** (32 // n_v),
            ld["exhaustive_weight_enum_checked"] is expected_ex_checked,
            ld["exhaustive_matches_macwilliams"] is expected_ex_mw,
            ld["macwilliams_a4_exact"],
            ld["verified"],
        )):
            p1_rows_ok = False

    # P2: Parse and check all 3 naive-Wagner suboptimality controls strictly via comparisons (no division)
    p2_rows_ok = True
    for row in p2_ctrls:
        cd = _req_dict(row, "p2.ctrl", _P2_CTRL_KEYS)
        n_c = _req_int_f(cd, "n", "p2.ctrl")
        d_wag = parse_exact_rational(cd["naive_wagner_sq_dist_q"], "p2.ctrl.wag")
        d_ex = parse_exact_rational(cd["exact_bnb_sq_dist_q"], "p2.ctrl.ex")
        d_opt = parse_exact_rational(cd["certified_optimal_sq_dist_q"], "p2.ctrl.opt")
        method = _require_utf8_no_surrogate(cd["optimal_proof_method"], "p2.ctrl.method")
        expected_bf = n_c in (8, 16)
        expected_method = "exhaustive_bruteforce" if expected_bf else "analytical_ge3_flip_lower_bound"
        if not all((
            n_c in (8, 16, 32),
            d_ex < d_wag,
            d_ex == d_opt,
            cd["exhaustive_bruteforce_executed"] is expected_bf,
            method == expected_method,
            not cd["naive_wagner_certified_optimal"],
            cd["exact_strictly_beats_naive_wagner"],
            cd["exact_matches_certified_optimal"],
        )):
            p2_rows_ok = False

    # P3: Cross-multiplied zero-retention, dead-zone, accumulator & f64 exact-decode bounds
    p3_rows_ok = True
    prev_rp = Fraction(0)
    prev_ai = Fraction(0)
    for row in p3_dz:
        dz = _req_dict(row, "p3.dz", _P3_DZ_KEYS)
        n_v = _req_int_f(dz, "n", "p3.dz")
        k_v = _req_int_f(dz, "k", "p3.dz")
        rate_q = parse_exact_rational(dz["code_rate_q"], "p3.dz.rate")
        rp_q = parse_exact_rational(dz["r_pack_sq_pow8_q"], "p3.dz.rp")
        ai_q = parse_exact_rational(dz["alpha_inf_pow16_q"], "p3.dz.ai")
        if not all((
            rate_q.numerator * n_v == k_v * rate_q.denominator,
            rp_q > prev_rp,
            ai_q > prev_ai,
        )):
            p3_rows_ok = False
        prev_rp = rp_q
        prev_ai = ai_q

    prev_zr = Fraction(-1)
    for row in p3_zr:
        zr = _req_dict(row, "p3.zr", _P3_ZR_KEYS)
        zc = _req_int_f(zr, "zero_coords", "p3.zr")
        tc = _req_int_f(zr, "total_coords", "p3.zr")
        parse_exact_rational(zr["inv_scale_bracket_q"], "p3.zr.inv")
        zq = parse_exact_rational(zr["zero_retention_q"], "p3.zr.q")
        if not all((tc > 0, zq.numerator * tc == zc * zq.denominator, zq > prev_zr)):
            p3_rows_ok = False
        prev_zr = zq

    for row in p3_acc:
        ad = _req_dict(row, "p3.acc", _P3_ACC_KEYS)
        w_v = _req_int_f(ad, "width", "p3.acc")
        wm = _req_int_f(ad, "w_max", "p3.acc")
        am = _req_int_f(ad, "act_abs_max", "p3.acc")
        md = _req_int_f(ad, "max_abs_dot", "p3.acc")
        if md != w_v * wm * am:
            p3_rows_ok = False

    for row in p3_wht:
        wd = _req_dict(row, "p3.wht", _P3_WHT_KEYS)
        n_v = _req_int_f(wd, "n", "p3.wht")
        if _req_int_f(wd, "max_abs_wht_of_int8", "p3.wht") != n_v * 128:
            p3_rows_ok = False

    if not all((
        _req_int_f(p3_sp, "nonzero_coords_after_decode", "p3.spill") == 4,
        _req_int_f(p3_f64, "w64_max_scaled_sum", "p3.f64") <= _req_int_f(p3_f64, "f64_exact_int_limit_2p53", "p3.f64"),
        _req_int_f(p3_f64, "w65_failure_witness_scaled_sum", "p3.f64") > _req_int_f(p3_f64, "f64_exact_int_limit_2p53", "p3.f64"),
    )):
        p3_rows_ok = False

    # P4: Cross-multiplied proxy inversion checks (ZERO division by parsed rationals!)
    w_rms = [parse_exact_rational(tok, "p4.w_rms") for tok in _req_list(p4_px["weight_col_rms_sq_q"], "p4.w_rms", 2)]
    px_v = [parse_exact_rational(tok, "p4.px") for tok in _req_list(p4_px["data_free_inv_rms_sq_proxy_q"], "p4.px", 2)]
    th_v = [parse_exact_rational(tok, "p4.th") for tok in _req_list(p4_px["true_activation_hessian_diag_q"], "p4.th", 2)]
    pr_q = parse_exact_rational(p4_px["proxy_col0_over_col1_ratio_q"], "p4.pr")
    tr_q = parse_exact_rational(p4_px["true_col1_over_col0_ratio_q"], "p4.tr")
    dt_q = parse_exact_rational(p4_px["total_ratio_distortion_q"], "p4.dt")
    p4_math_ok = all((
        w_rms[0] * px_v[0] == Fraction(1, 1),
        w_rms[1] * px_v[1] == Fraction(1, 1),
        px_v[0] == pr_q * px_v[1],
        th_v[1] == tr_q * th_v[0],
        dt_q == pr_q * tr_q,
        px_v[0] > px_v[1],
        th_v[0] < th_v[1],
        not p4["measured_live_model_weights"],
        not pl["measured_live_model_weights"],
    ))

    ref = _reference_payload()
    ok_all = all((
        pl["track_id"] == oracle.TRACK_ID,
        pl["portal_ref"] == "Portal 52",
        pl["novelty_class"] == "N1",
        p1_rows_ok,
        _req_int_f(p1_e8, "e8_root_count", "p1.e8") == 240,
        _req_int_f(p1_e8, "mapped_a8_min_vector_count", "p1.e8") == 240,
        _req_int_f(p1_bw, "a16_kissing", "p1.bw") < _req_int_f(p1_bw, "barnes_wall_lambda16_kissing", "p1.bw"),
        p2_rows_ok,
        p3_rows_ok,
        p4_math_ok,
        pl == ref,
        _req_int_f(summ, "pillar_count", "summary") == 4,
        summ["all_four_pillars_verified"],
        _req_int_f(summ, "float_count_on_trust_path", "summary") == 0,
        _req_int_f(summ, "bare_assert_count", "summary") == 0,
        _req_int_f(summ, "rng_calls", "summary") == 0,
    ))
    oracle.require_ok(ok_all, "SECONDARY_AUDIT_FAILED", "secondary cross-multiplied audit failed")
    return {"secondary_audit_passed": True, "pillars_audited": 4}


def build_standalone_certificate() -> dict[str, Any]:
    payload = oracle.build_payload()
    payload_sha = _sha256_canonical(payload)
    outer = {
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = _sha256_canonical(outer)
    return {
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }


def verify_certificate_dict(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    oracle.require_exact_bool(full_replay, "full_replay")
    oracle.require_ok(type(cert) is dict, "INVALID_CERT_TYPE", "certificate must be exact dict")
    _validate_json_tree(cert, "cert")
    oracle.require_ok(frozenset(cert.keys()) == _CERT_KEYS, "INVALID_CERT_KEYS", "certificate keys mismatch")
    oracle.require_ok(cert["schema_version"] == oracle.SCHEMA_VERSION, "INVALID_SCHEMA", "schema_version mismatch")
    oracle.require_ok(cert["track_id"] == oracle.TRACK_ID, "INVALID_TRACK_ID", "track_id mismatch")
    oracle.require_ok(type(cert["payload"]) is dict, "INVALID_PAYLOAD_TYPE", "payload must be exact dict")

    claimed_p = require_hex64_lower_ascii(cert["payload_sha256"], "payload_sha256")
    claimed_c = require_hex64_lower_ascii(cert["certificate_sha256"], "certificate_sha256")

    actual_p = _sha256_canonical(cert["payload"])
    oracle.require_ok(hmac.compare_digest(claimed_p, actual_p), "PAYLOAD_SHA256_MISMATCH", "payload_sha256 mismatch")

    outer = {
        "schema_version": cert["schema_version"],
        "track_id": cert["track_id"],
        "payload_sha256": claimed_p,
        "payload": cert["payload"],
    }
    actual_c = _sha256_canonical(outer)
    oracle.require_ok(hmac.compare_digest(claimed_c, actual_c), "CERT_SHA256_MISMATCH", "certificate_sha256 mismatch")

    sec = secondary_audit_payload(cert["payload"])
    if full_replay:
        ref = build_standalone_certificate()
        oracle.require_ok(hmac.compare_digest(actual_p, ref["payload_sha256"]), "REPLAY_MISMATCH", "replay mismatch")

    return {
        "valid": True,
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": actual_p,
        "certificate_sha256": actual_c,
        "full_replay": full_replay,
        "secondary_audit": sec,
    }


def check_certificate_file(path: Path, *, full_replay: bool = True) -> dict[str, Any]:
    raw = read_regular_bytes_capped(path, "certificate", MAX_JSON_BYTES)
    oracle.require_ok(b"\r" not in raw, "CRLF_FORBIDDEN", "CRLF line endings forbidden")
    oracle.require_ok(raw.endswith(b"\n"), "MISSING_TRAILING_LF", "certificate must end with LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise oracle.HammingLatticeError("INVALID_UTF8", "certificate is not valid UTF-8") from exc
    cert = parse_strict_json_text(text, require_object=True)
    canon = canonical_json_bytes(cert)
    oracle.require_ok(raw == canon + b"\n", "NON_CANONICAL_BYTES", "certificate is not canonical ASCII JSON + LF")
    res = verify_certificate_dict(cert, full_replay=full_replay)
    res["canonical_sha256"] = hashlib.sha256(canon).hexdigest()
    res["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return res


def write_certificate_file(path: Path) -> dict[str, Any]:
    cert = build_standalone_certificate()
    canon = canonical_json_bytes(cert)
    raw = canon + b"\n"
    write_regular_bytes_safe(path, raw, "certificate")
    return {
        "written": True,
        "schema_version": oracle.SCHEMA_VERSION,
        "track_id": oracle.TRACK_ID,
        "payload_sha256": cert["payload_sha256"],
        "certificate_sha256": cert["certificate_sha256"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


# ============================================================================
# 4. Strict Fail-Closed CLI (Space & Equals Flag-Like Rejection)
# ============================================================================

def _reject_bad_path_arg(tok: str, val: str) -> Path:
    oracle.require_ok(bool(val) and not val.startswith("-"), "INVALID_PATH_ARG", f"{tok} path empty or flag-like")
    return Path(val)


def _parse_cli_argv(argv: Sequence[str]) -> tuple[str, Path | None, bool]:
    oracle.require_ok(type(argv) in (list, tuple), "INVALID_ARGV", "argv must be list or tuple")
    for item in argv:
        _require_utf8_no_surrogate(item, "argv_item")

    mode: str | None = None
    target: Path | None = None
    as_json = False
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "--json":
            oracle.require_ok(not as_json, "DUPLICATE_FLAG", "duplicate --json flag")
            as_json = True
            i += 1
        elif tok == "--standalone":
            oracle.require_ok(mode is None, "CONFLICTING_MODE", "multiple mode flags forbidden")
            mode = "standalone"
            i += 1
        elif tok in ("--check", "--write"):
            oracle.require_ok(mode is None, "CONFLICTING_MODE", "multiple mode flags forbidden")
            oracle.require_ok(i + 1 < n, "MISSING_PATH_ARG", f"{tok} requires a path argument")
            target = _reject_bad_path_arg(tok, argv[i + 1])
            mode = "check" if tok == "--check" else "write"
            i += 2
        elif tok.startswith("--check=") or tok.startswith("--write="):
            oracle.require_ok(mode is None, "CONFLICTING_MODE", "multiple mode flags forbidden")
            flag_name, val = tok.split("=", 1)
            target = _reject_bad_path_arg(flag_name, val)
            mode = "check" if flag_name == "--check" else "write"
            i += 1
        else:
            raise oracle.HammingLatticeError("UNKNOWN_FLAG", f"unrecognized CLI argument: {tok}")

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
    except oracle.HammingLatticeError as exc:
        err = {"valid": False, "error_code": exc.code, "error_message": exc.detail}
        sys.stderr.write(canonical_json_bytes(err).decode("ascii") + "\n")
        return 2

    if as_json:
        sys.stdout.write(canonical_json_bytes(out).decode("ascii") + "\n")
    else:
        sys.stdout.write(
            f"OK {oracle.SCHEMA_VERSION} payload_sha256={out['payload_sha256']} "
            f"certificate_sha256={out['certificate_sha256']}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
