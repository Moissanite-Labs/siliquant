#!/usr/bin/env python3
"""
Track S21f (`AUDIT-121F` R2): Zone-B-Hardened Fail-Closed Certificate Auditor
(`wht8_frobenius_bridge_audit.py`).

Pure Python 3 standard library (`0` AST floats, `0` `float()` calls, `0` bare
`assert`s, `0` `.isdigit()`, `0` mathematical RNG — `secrets.token_hex(8)` used
solely for collision-free atomic temp file naming under concurrent writers).

Provides full pure-Z/Q deterministic replay and secondary algebraic cross-checking
with atomic per-component `dir_fd` (`openat` + `O_RDONLY|O_DIRECTORY|O_NOFOLLOW`)
traversal, `dir_fd` `$PWD` symlink checking, `65536`-byte streaming capped reads
(`O_RDONLY|O_NOFOLLOW|O_NONBLOCK` + `fstat` `S_ISREG` closing `/proc/kallsyms`
and `chmod 000`), locked + random + bounded `EEXIST` retry `0o600` atomic writes
with `os.fsync(tmp_fd)`, pre-replace `os.fstat(tmp_fd).st_nlink == 1` guard,
target `st_nlink <= 1` check, `os.replace(..., src_dir_fd=dfd, dst_dir_fd=dfd)`,
directory `os.fsync(dfd)`, deleted-CWD `CWD_UNAVAILABLE` handling, lone UTF-16
surrogate rejection, strict zero-abbreviation CLI parsing, and full structural +
bool-vs-int payload verification under both `full_replay=True` and `False`.
"""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from wht8_frobenius_bridge_oracle import (  # noqa: E402
    HARD_MAX_BYTES_CAP,
    MAX_JSON_BYTES,
    MAX_JSON_DEPTH,
    Wht8BridgeError,
    _require_valid_utf8_str,
    canonical_json_bytes,
    canonical_sha256,
    constant_time_hex64_equal,
    generate_certificate,
    parse_strict_json_text,
    require_hex64_ascii,
    require_int,
    validate_cli_argv,
    validate_json_value,
)

_EXPECTED_SCHEMA = "aletheia.track_s21f.zero_unpack_wht8_frobenius_bridge.v1"
_REQUIRED_TOP_KEYS = frozenset({"schema", "payload_sha256", "certificate_sha256", "payload"})
_THEOREM_KEYS = (
    "theorem1_wht8_isometry",
    "theorem2_eisenstein_shell_identities",
    "theorem3_4_zero_unpack_and_tier2_exact_parity",
    "theorem5_cross_org_attack_closures",
)
_PAYLOAD_KEYS = frozenset({"track", "audit_id", "oracle_class", "novelty_class", *_THEOREM_KEYS})
_ALLOWED_BOOL_KEYS = frozenset({
    "admitted",
    "blocked",
    "butterfly_24op_matches_h8_all_basis",
    "e4_squared_equals_e8_exact",
    "e8_4_superblock_exact_match",
    "e8_4_superblock_roundtrip_exact",
    "eisenstein_e4_exact",
    "gram_diag_matches_tier1",
    "h8_gram_equals_8_i8",
    "h8_symmetric",
    "rank1_honest_exact_match",
    "rank2_exact_match",
    "superblock_e8_4",
})

_EXACT_PATH_TYPE = type(Path("."))
_READ_CHUNK_BYTES = 65536
_TMP_LOCK = threading.Lock()
_TMP_COUNTER = 0
_TMP_MAX_ATTEMPTS = 32


# ---------------------------------------------------------------------------
# Zone-B Hardened Atomic dir_fd File I/O (TOCTOU, PWD, Hardlinks, Surrogates)
# ---------------------------------------------------------------------------

def _step_dir_fd_chain(start_fd: int, components: Sequence[str], label: str) -> int:
    """Atomically traverse directory components via openat(O_RDONLY|O_DIRECTORY|O_NOFOLLOW)."""
    dfd = start_fd
    for comp in components:
        try:
            nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dfd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                try:
                    is_sym = stat.S_ISLNK(os.lstat(comp, dir_fd=dfd).st_mode)
                except OSError:
                    is_sym = True
                os.close(dfd)
                if is_sym:
                    raise Wht8BridgeError("SYMLINK_ANCESTOR_FORBIDDEN", f"symlink ancestor forbidden at {label}") from exc
                raise Wht8BridgeError("PATH_STAT_ERROR", f"non-directory ancestor at {label}") from exc
            os.close(dfd)
            if exc.errno == errno.ENOENT:
                raise Wht8BridgeError("FILE_NOT_FOUND", f"ancestor not found at {label}") from exc
            raise Wht8BridgeError("FILE_IO_ERROR", f"cannot traverse {label}: {type(exc).__name__}") from exc
        os.close(dfd)
        dfd = nxt
    return dfd


def _check_pwd_no_symlinks() -> None:
    """Verify logical $PWD has no symlink components using atomic dir_fd traversal."""
    pwd_env = os.environ.get("PWD")
    if not pwd_env or not pwd_env.startswith("/"):
        return
    _require_valid_utf8_str(pwd_env, "PWD")
    try:
        if os.path.realpath(pwd_env) != os.path.realpath(os.getcwd()):
            return
    except OSError as exc:
        raise Wht8BridgeError("CWD_UNAVAILABLE", f"cannot resolve PWD/CWD: {type(exc).__name__}") from exc
    pwd_parts = [p for p in Path(pwd_env).parts[1:] if p not in ("", ".")]
    try:
        root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise Wht8BridgeError("CWD_UNAVAILABLE", f"cannot open root: {type(exc).__name__}") from exc
    end_fd = _step_dir_fd_chain(root_fd, pwd_parts, "pwd_ancestor")
    os.close(end_fd)


def _open_parent_dir_fd(path: Any) -> tuple[int, str]:
    """Validate exact Path and atomically pin parent dir_fd against TOCTOU swaps."""
    if type(path) is not _EXACT_PATH_TYPE:
        raise Wht8BridgeError("INVALID_PATH_TYPE", "path must be strict concrete pathlib.Path")
    raw_s = str(path)
    _require_valid_utf8_str(raw_s, "path")
    if "\x00" in raw_s:
        raise Wht8BridgeError("INVALID_PATH_NUL", "NUL byte forbidden in path")

    if path.is_absolute():
        raw_parts = [p for p in path.parts[1:] if p not in ("", ".")]
        try:
            start_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            raise Wht8BridgeError("FILE_IO_ERROR", f"cannot open root anchor: {type(exc).__name__}") from exc
    else:
        try:
            os.getcwd()
            start_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            raise Wht8BridgeError("CWD_UNAVAILABLE", f"current working directory unavailable: {type(exc).__name__}") from exc
        try:
            _check_pwd_no_symlinks()
        except Wht8BridgeError:
            os.close(start_fd)
            raise
        raw_parts = [p for p in path.parts if p not in ("", ".")]

    if not raw_parts or raw_parts[-1] == "..":
        os.close(start_fd)
        raise Wht8BridgeError("NON_REGULAR_FILE", "target path names a directory")

    dfd = _step_dir_fd_chain(start_fd, raw_parts[:-1], "path_component")
    return dfd, raw_parts[-1]


def require_no_symlink_path(path: Any) -> None:
    """Verify exact Path type, CWD/PWD safety, ancestor dir_fd chain, and non-symlink leaf."""
    dfd, leaf = _open_parent_dir_fd(path)
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise Wht8BridgeError("PATH_STAT_ERROR", f"cannot lstat target: {type(exc).__name__}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise Wht8BridgeError("SYMLINK_FORBIDDEN", "symlink target path forbidden")
    finally:
        os.close(dfd)


def read_regular_bytes_capped(path: Path, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    """Stream regular file in bounded 65536-byte chunks via dir_fd + O_NOFOLLOW + O_NONBLOCK."""
    cap = require_int(max_bytes, "max_bytes", lo=1, hi=HARD_MAX_BYTES_CAP)
    dfd, leaf = _open_parent_dir_fd(path)
    fd = -1
    try:
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise Wht8BridgeError("SYMLINK_FORBIDDEN", "symlink target path forbidden") from exc
            if exc.errno == errno.ENOENT:
                raise Wht8BridgeError("FILE_NOT_FOUND", "certificate file not found") from exc
            raise Wht8BridgeError("FILE_READ_ERROR", f"cannot open file: {type(exc).__name__}") from exc
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise Wht8BridgeError("NON_REGULAR_FILE", "target is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise Wht8BridgeError("FILE_TOO_LARGE", f"file exceeds {cap} bytes")
            chunks.append(chunk)
        return b"".join(chunks)
    except Wht8BridgeError:
        raise
    except OSError as exc:
        raise Wht8BridgeError("FILE_READ_ERROR", f"failed reading file: {type(exc).__name__}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(dfd)


def write_regular_file_bytes(path: Path, payload: bytes) -> None:
    """Thread-safe atomic 0o600 dir_fd write with EEXIST retry, fsync, and st_nlink guards."""
    global _TMP_COUNTER
    if type(payload) is not bytes:
        raise Wht8BridgeError("INVALID_WRITE_DATA", "payload must be strict bytes")
    dfd, leaf = _open_parent_dir_fd(path)
    tmp_fd = -1
    created_tmp = False
    tmp_name = ""
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        except OSError as exc:
            raise Wht8BridgeError("FILE_WRITE_ERROR", f"cannot stat target: {type(exc).__name__}") from exc
        if st is not None:
            if stat.S_ISLNK(st.st_mode):
                raise Wht8BridgeError("SYMLINK_FORBIDDEN", "cannot write to symlink target")
            if not stat.S_ISREG(st.st_mode):
                raise Wht8BridgeError("NON_REGULAR_FILE", "cannot write to non-regular file")
            if st.st_nlink > 1:
                raise Wht8BridgeError("HARDLINK_FORBIDDEN", "cannot write through hardlinked file")

        try:
            for _ in range(_TMP_MAX_ATTEMPTS):
                with _TMP_LOCK:
                    _TMP_COUNTER += 1
                    seq = _TMP_COUNTER
                cand = f".s21f_wht8_tmp_{os.getpid()}_{seq}_{secrets.token_hex(8)}"
                try:
                    tmp_fd = os.open(
                        cand,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                        0o600,
                        dir_fd=dfd,
                    )
                    tmp_name = cand
                    created_tmp = True
                    break
                except FileExistsError:
                    continue
            if not created_tmp:
                raise Wht8BridgeError("TEMP_RETRY_EXHAUSTED", "exhausted atomic temp file retries")
            view = memoryview(payload)
            written = 0
            while written < len(payload):
                written += os.write(tmp_fd, view[written:])
            os.fsync(tmp_fd)
            if os.fstat(tmp_fd).st_nlink != 1:
                raise Wht8BridgeError("HARDLINK_FORBIDDEN", "temp file hardlinked before replace")
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            created_tmp = False
            os.close(tmp_fd)
            tmp_fd = -1
            os.fsync(dfd)
        except Wht8BridgeError:
            raise
        except OSError as exc:
            raise Wht8BridgeError("FILE_WRITE_ERROR", f"failed atomic write: {type(exc).__name__}") from exc
    finally:
        if tmp_fd >= 0:
            os.close(tmp_fd)
        if created_tmp:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


# ---------------------------------------------------------------------------
# Structural Schema Gate & Secondary Algebraic Cross-Check
# ---------------------------------------------------------------------------

def _validate_payload_structure(payload: dict[str, Any]) -> None:
    """Enforce exact top-level schema and iterative bool-vs-int separation under full_replay=False/True."""
    if frozenset(payload.keys()) != _PAYLOAD_KEYS:
        raise Wht8BridgeError("INVALID_PAYLOAD_KEYS", "payload top-level keys mismatch")
    if (payload["track"], payload["audit_id"], payload["oracle_class"], payload["novelty_class"]) != (
        "S21f",
        "AUDIT-121F",
        "exact",
        "N1",
    ):
        raise Wht8BridgeError("INVALID_PAYLOAD_META", "payload metadata fields mismatch")
    for tkey in _THEOREM_KEYS:
        if type(payload[tkey]) is not dict:
            raise Wht8BridgeError("INVALID_THEOREM_TYPE", f"{tkey} must be strict dict")

    stack: list[Any] = [payload]
    while stack:
        cur = stack.pop()
        if type(cur) is dict:
            for k, v in cur.items():
                if type(v) is bool:
                    if k not in _ALLOWED_BOOL_KEYS:
                        raise Wht8BridgeError("BOOL_IN_NON_BOOL_FIELD", f"unexpected bool in field {k}")
                elif type(v) in (dict, list):
                    stack.append(v)
        else:
            for item in cur:
                if type(item) is bool:
                    raise Wht8BridgeError("BOOL_IN_LIST", "bool element forbidden inside list")
                if type(item) in (dict, list):
                    stack.append(item)


_EXPECTED_R1_ADAPTER_SHA = "6a7042740c3ecb4b0d6e69ceb66fbfdf2a1db4473f8ed4cccf2db72289abc047"
_EXPECTED_R2_ADAPTER_SHA = "5cbe46e423d1e9a46764ad2b5e239e293af8caba1c12e98a04c6e19f80bb3194"
_EXPECTED_SB_ADAPTER_SHA = "5ae432908f9025184ce4fa507f27ad06ae5508711e2c6ced493484633c7830fd"

_T1_KEYS = frozenset({
    "butterfly_24op_matches_h8_all_basis",
    "exact_norm_scale_factor",
    "h8_gram_equals_8_i8",
    "h8_symmetric",
})
_T2_KEYS = frozenset({
    "e4_squared_equals_e8_exact",
    "e8_4_136bit_bsearch_max_cmp",
    "e8_4_136bit_codebook_size",
    "e8_4_136bit_max_shell",
    "e8_4_88bit_bsearch_max_cmp",
    "e8_4_88bit_codebook_size",
    "e8_4_88bit_max_shell",
    "e8_shell_counts_m0_to_m8",
    "eisenstein_e4_exact",
    "max_shell_verified_dp",
    "op_count_per_32w_superblock",
})
_OPS_KEYS = frozenset({
    "full_unpack_wht8_frobenius_ops",
    "speedup_ratio_136bit",
    "speedup_ratio_88bit",
    "zero_unpack_136bit_ops",
    "zero_unpack_88bit_ops",
})
_T34_KEYS = frozenset({
    "e8_4_superblock_exact_match",
    "e8_4_superblock_full_wht8_ref_q",
    "e8_4_superblock_roundtrip_exact",
    "e8_4_superblock_tier1",
    "rank1_honest_exact_match",
    "rank1_honest_full_wht8_ref_q",
    "rank1_honest_tier1",
    "rank1_inflated_full_wht8_ref_q",
    "rank2_exact_match",
    "rank2_exact_singular_values_sq_q",
    "rank2_fast_accept_tier1",
    "rank2_full_wht8_ref_q",
    "rank2_tight_tier2",
})
_R1_SB_KEYS = frozenset({
    "a_frob_sq_q",
    "adapter_sha256",
    "admitted",
    "b_frob_sq_q",
    "coordinates_unranked",
    "exact_ba_frob_sq_q",
    "rank",
    "shell_lookups",
    "superblock_e8_4",
    "tier_used",
    "wht8_butterflies_run",
    "zero_unpack_ub_q",
})
_R2_TIER2_KEYS = _R1_SB_KEYS | frozenset({"gram_diag_matches_tier1"})
_R2_FAST_KEYS = _R1_SB_KEYS - frozenset({"exact_ba_frob_sq_q"})
_T5_KEYS = frozenset({
    "rank2_bareiss_1ulp_spectral_understatement",
    "shell_index_bump_m1_to_m3",
    "warp13_alpha_32_metadata_inflation",
    "warp13_b_times_8_scale_forgery",
    "wd40_05_radix_scale_exp2_relabel_32_to_48",
})
_T5_WD40_KEYS = frozenset({"blocked", "rejection_code", "relabelled_declared_q", "true_frobenius_sq_q"})
_T5_B8_KEYS = frozenset({"blocked", "resigned_sha_rejection_code", "stale_sha_rejection_code"})
_T5_SIMPLE_KEYS = frozenset({"blocked", "rejection_code"})
_T5_ULP_KEYS = frozenset({"blocked", "exact_spectral_sq_q", "rejection_code", "understated_bound_q"})


def secondary_audit_payload(payload: Any) -> dict[str, Any]:
    """Schema-safe, strict-key, pinned-digest secondary algebraic cross-check over all 5 Track S21f theorems."""
    if type(payload) is not dict:
        raise Wht8BridgeError("INVALID_PAYLOAD_TYPE", "payload must be strict dict")
    try:
        # Theorem 1: WHT_8 isometry
        t1 = payload["theorem1_wht8_isometry"]
        if (
            type(t1) is dict,
            frozenset(t1.keys()),
            t1["h8_symmetric"],
            t1["h8_gram_equals_8_i8"],
            t1["butterfly_24op_matches_h8_all_basis"],
            require_int(t1["exact_norm_scale_factor"], "t1.scale"),
        ) != (True, _T1_KEYS, True, True, True, 8):
            raise Wht8BridgeError("AUDIT_T1_ISOMETRY", "WHT_8 isometry fields mismatch")

        # Theorem 2: Eisenstein E_4 / E_8^4 shell tables & zero-unpack speedup
        t2 = payload["theorem2_eisenstein_shell_identities"]
        shells = [require_int(x, "e8_shell") for x in t2["e8_shell_counts_m0_to_m8"]]
        if (
            type(t2) is dict,
            frozenset(t2.keys()),
            require_int(t2["max_shell_verified_dp"], "t2.max_shell"),
            shells,
            t2["eisenstein_e4_exact"],
            t2["e4_squared_equals_e8_exact"],
            require_int(t2["e8_4_88bit_max_shell"], "t2.m4_88"),
            require_int(t2["e8_4_88bit_bsearch_max_cmp"], "t2.cmp_88"),
            t2["e8_4_88bit_codebook_size"],
            require_int(t2["e8_4_136bit_max_shell"], "t2.m4_136"),
            require_int(t2["e8_4_136bit_bsearch_max_cmp"], "t2.cmp_136"),
            t2["e8_4_136bit_codebook_size"],
        ) != (
            True,
            _T2_KEYS,
            8,
            [1, 240, 2160, 6720, 17520, 30240, 60480, 82560, 140400],
            True,
            True,
            48,
            6,
            "263197133594409826368003841",
            391,
            9,
            "85889107931845736051215402649708868585601",
        ):
            raise Wht8BridgeError("AUDIT_T2_EISENSTEIN", "Eisenstein shell identity fields mismatch")
        ops = t2["op_count_per_32w_superblock"]
        if (
            type(ops) is dict,
            frozenset(ops.keys()),
            require_int(ops["full_unpack_wht8_frobenius_ops"], "ops.full"),
            require_int(ops["zero_unpack_88bit_ops"], "ops.zu88"),
            require_int(ops["zero_unpack_136bit_ops"], "ops.zu136"),
            ops["speedup_ratio_88bit"],
            ops["speedup_ratio_136bit"],
        ) != (True, _OPS_KEYS, 191, 9, 12, "191/9", "191/12"):
            raise Wht8BridgeError("AUDIT_T2_OPS", "32-weight superblock op ratios mismatch")

        # Theorems 3 & 4: Zero-unpack Tier-1 and WHT_8-bypassed Tier-2 exact parity + pinned adapter_sha256
        t34 = payload["theorem3_4_zero_unpack_and_tier2_exact_parity"]
        r1 = t34["rank1_honest_tier1"]
        r2 = t34["rank2_tight_tier2"]
        r2f = t34["rank2_fast_accept_tier1"]
        sb = t34["e8_4_superblock_tier1"]
        if (
            type(t34) is dict,
            frozenset(t34.keys()),
            type(r1) is dict,
            frozenset(r1.keys()),
            r1["tier_used"],
            r1["admitted"],
            r1["superblock_e8_4"],
            require_int(r1["rank"], "r1.rank"),
            require_int(r1["shell_lookups"], "r1.lookups"),
            require_int(r1["coordinates_unranked"], "r1.unrank"),
            require_int(r1["wht8_butterflies_run"], "r1.wht8"),
            r1["a_frob_sq_q"],
            r1["b_frob_sq_q"],
            r1["zero_unpack_ub_q"],
            r1["exact_ba_frob_sq_q"],
            constant_time_hex64_equal(r1["adapter_sha256"], _EXPECTED_R1_ADAPTER_SHA, "r1.adapter_sha256"),
            t34["rank1_honest_full_wht8_ref_q"],
            t34["rank1_inflated_full_wht8_ref_q"],
            t34["rank1_honest_exact_match"],
        ) != (
            True,
            _T34_KEYS,
            True,
            _R1_SB_KEYS,
            "TIER1_ZERO_UNPACK_EXACT_RANK1",
            True,
            False,
            1,
            4,
            0,
            0,
            "1",
            "3/2",
            "3/2",
            "3/2",
            True,
            "3/2",
            "6",
            True,
        ):
            raise Wht8BridgeError("AUDIT_T34_RANK1", "Tier-1 rank-1 witness mismatch")
        if (
            type(r2) is dict,
            frozenset(r2.keys()),
            type(r2f) is dict,
            frozenset(r2f.keys()),
            r2["tier_used"],
            r2["admitted"],
            r2["superblock_e8_4"],
            require_int(r2["rank"], "r2.rank"),
            require_int(r2["shell_lookups"], "r2.lookups"),
            require_int(r2["coordinates_unranked"], "r2.unrank"),
            require_int(r2["wht8_butterflies_run"], "r2.wht8"),
            r2["a_frob_sq_q"],
            r2["b_frob_sq_q"],
            r2["zero_unpack_ub_q"],
            r2["exact_ba_frob_sq_q"],
            r2["gram_diag_matches_tier1"],
            t34["rank2_full_wht8_ref_q"],
            t34["rank2_exact_singular_values_sq_q"],
            t34["rank2_exact_match"],
            r2f["tier_used"],
            r2f["admitted"],
            r2f["superblock_e8_4"],
            require_int(r2f["rank"], "r2f.rank"),
            require_int(r2f["shell_lookups"], "r2f.lookups"),
            require_int(r2f["coordinates_unranked"], "r2f.unrank"),
            require_int(r2f["wht8_butterflies_run"], "r2f.wht8"),
            r2f["a_frob_sq_q"],
            r2f["b_frob_sq_q"],
            r2f["zero_unpack_ub_q"],
            constant_time_hex64_equal(r2["adapter_sha256"], _EXPECTED_R2_ADAPTER_SHA, "r2.adapter_sha256"),
            constant_time_hex64_equal(r2f["adapter_sha256"], _EXPECTED_R2_ADAPTER_SHA, "r2f.adapter_sha256"),
            constant_time_hex64_equal(r2["adapter_sha256"], r2f["adapter_sha256"], "r2.adapter_sha256"),
        ) != (
            True,
            _R2_TIER2_KEYS,
            True,
            _R2_FAST_KEYS,
            "TIER2_WHT8_BYPASSED_GRAM_BAREISS",
            True,
            False,
            2,
            4,
            32,
            0,
            "1",
            "4",
            "4",
            "2",
            True,
            "2",
            ["3/2", "1/2"],
            True,
            "TIER1_ZERO_UNPACK_FAST_ACCEPT",
            True,
            False,
            2,
            4,
            0,
            0,
            "1",
            "4",
            "4",
            True,
            True,
            True,
        ):
            raise Wht8BridgeError("AUDIT_T34_RANK2", "Tier-2 rank-2 witness mismatch")
        if (
            type(sb) is dict,
            frozenset(sb.keys()),
            sb["tier_used"],
            sb["admitted"],
            sb["superblock_e8_4"],
            require_int(sb["rank"], "sb.rank"),
            require_int(sb["shell_lookups"], "sb.lookups"),
            require_int(sb["coordinates_unranked"], "sb.unrank"),
            require_int(sb["wht8_butterflies_run"], "sb.wht8"),
            sb["a_frob_sq_q"],
            sb["b_frob_sq_q"],
            sb["zero_unpack_ub_q"],
            sb["exact_ba_frob_sq_q"],
            constant_time_hex64_equal(sb["adapter_sha256"], _EXPECTED_SB_ADAPTER_SHA, "sb.adapter_sha256"),
            t34["e8_4_superblock_full_wht8_ref_q"],
            t34["e8_4_superblock_exact_match"],
            t34["e8_4_superblock_roundtrip_exact"],
        ) != (
            True,
            _R1_SB_KEYS,
            "TIER1_ZERO_UNPACK_EXACT_RANK1",
            True,
            True,
            1,
            2,
            0,
            0,
            "3/2",
            "2",
            "3",
            "3",
            True,
            "3",
            True,
            True,
        ):
            raise Wht8BridgeError("AUDIT_T34_SUPERBLOCK", "E_8^4 superblock witness mismatch")

        # Theorem 5: All 5 cross-org attack closures (6 test variants)
        t5 = payload["theorem5_cross_org_attack_closures"]
        wd40 = t5["wd40_05_radix_scale_exp2_relabel_32_to_48"]
        b8 = t5["warp13_b_times_8_scale_forgery"]
        a32 = t5["warp13_alpha_32_metadata_inflation"]
        bump = t5["shell_index_bump_m1_to_m3"]
        ulp = t5["rank2_bareiss_1ulp_spectral_understatement"]
        if (
            type(t5) is dict,
            frozenset(t5.keys()),
            type(wd40) is dict,
            frozenset(wd40.keys()),
            type(b8) is dict,
            frozenset(b8.keys()),
            type(a32) is dict,
            frozenset(a32.keys()),
            type(bump) is dict,
            frozenset(bump.keys()),
            type(ulp) is dict,
            frozenset(ulp.keys()),
            wd40["blocked"],
            wd40["rejection_code"],
            wd40["relabelled_declared_q"],
            wd40["true_frobenius_sq_q"],
            b8["blocked"],
            b8["stale_sha_rejection_code"],
            b8["resigned_sha_rejection_code"],
            a32["blocked"],
            a32["rejection_code"],
            bump["blocked"],
            bump["rejection_code"],
            ulp["blocked"],
            ulp["rejection_code"],
            ulp["exact_spectral_sq_q"],
            ulp["understated_bound_q"],
        ) != (
            True,
            _T5_KEYS,
            True,
            _T5_WD40_KEYS,
            True,
            _T5_B8_KEYS,
            True,
            _T5_SIMPLE_KEYS,
            True,
            _T5_SIMPLE_KEYS,
            True,
            _T5_ULP_KEYS,
            True,
            "FROBENIUS_EXCEEDS_CERTIFIED",
            "3/32768",
            "6",
            True,
            "ADAPTER_SHA_MISMATCH",
            "FROBENIUS_EXCEEDS_CERTIFIED",
            True,
            "NON_UNIT_ALPHA_SCALE",
            True,
            "FROBENIUS_EXCEEDS_CERTIFIED",
            True,
            "SPECTRAL_EXCEEDS_CERTIFIED",
            "3/2",
            "6442450943/4294967296",
        ):
            raise Wht8BridgeError("AUDIT_T5_ATTACKS", "all 5 cross-org attack closures must be blocked")
    except Wht8BridgeError:
        raise
    except (KeyError, TypeError, AttributeError) as exc:
        raise Wht8BridgeError("AUDIT_SCHEMA_ERROR", f"malformed payload structure: {type(exc).__name__}") from exc

    return {
        "secondary_audit_passed": True,
        "theorems_checked": 5,
        "attack_closures_checked": 5,
    }


def verify_certificate_dict(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    if type(full_replay) is not bool:
        raise Wht8BridgeError("INVALID_BOOL_TYPE", "full_replay must be strict bool")
    validate_json_value(cert)
    if type(cert) is not dict:
        raise Wht8BridgeError("INVALID_CERT_TYPE", "certificate must be strict dict")
    if set(cert.keys()) != _REQUIRED_TOP_KEYS:
        raise Wht8BridgeError("INVALID_CERT_KEYS", "certificate top-level keys mismatch")
    if cert["schema"] != _EXPECTED_SCHEMA:
        raise Wht8BridgeError("INVALID_SCHEMA", "unexpected certificate schema")
    if type(cert["payload"]) is not dict:
        raise Wht8BridgeError("INVALID_PAYLOAD_TYPE", "payload must be strict dict")

    claimed_payload_sha = require_hex64_ascii(cert["payload_sha256"], "payload_sha256")
    claimed_cert_sha = require_hex64_ascii(cert["certificate_sha256"], "certificate_sha256")

    actual_payload_sha = canonical_sha256(cert["payload"])
    if not constant_time_hex64_equal(actual_payload_sha, claimed_payload_sha, "payload_sha256"):
        raise Wht8BridgeError("PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    cert_body = {
        "schema": cert["schema"],
        "payload_sha256": claimed_payload_sha,
        "payload": cert["payload"],
    }
    actual_cert_sha = canonical_sha256(cert_body)
    if not constant_time_hex64_equal(actual_cert_sha, claimed_cert_sha, "certificate_sha256"):
        raise Wht8BridgeError("CERT_SHA_MISMATCH", "certificate_sha256 mismatch")

    _validate_payload_structure(cert["payload"])
    sec = secondary_audit_payload(cert["payload"])

    if full_replay:
        expected = generate_certificate()
        if not constant_time_hex64_equal(actual_payload_sha, expected["payload_sha256"], "recompute_payload_sha256"):
            raise Wht8BridgeError("RECOMPUTE_PAYLOAD_MISMATCH", "certificate payload diverges from standalone oracle")

    return {
        "valid": True,
        "schema": _EXPECTED_SCHEMA,
        "payload_sha256": actual_payload_sha,
        "certificate_sha256": actual_cert_sha,
        "full_replay": full_replay,
        "secondary_audit": sec,
    }


def verify_certificate_file(path: Path) -> dict[str, Any]:
    raw = read_regular_bytes_capped(path, max_bytes=MAX_JSON_BYTES)
    if b"\r" in raw:
        raise Wht8BridgeError("CR_FORBIDDEN", "CR/CRLF line endings forbidden in certificate")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Wht8BridgeError("INVALID_UTF8", f"invalid UTF-8: {exc}") from exc
    parsed = parse_strict_json_text(text, require_object=True, max_depth=MAX_JSON_DEPTH)
    canonical_raw = canonical_json_bytes(parsed)
    if raw not in (canonical_raw, canonical_raw + b"\n"):
        raise Wht8BridgeError("NON_CANONICAL_BYTES", "certificate file is not canonical JSON")
    res = verify_certificate_dict(parsed, full_replay=True)
    res["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return res


def write_certificate_file(path: Path) -> dict[str, Any]:
    cert = generate_certificate()
    raw = canonical_json_bytes(cert) + b"\n"
    write_regular_file_bytes(path, raw)
    return {
        "written": True,
        "schema": _EXPECTED_SCHEMA,
        "payload_sha256": cert["payload_sha256"],
        "certificate_sha256": cert["certificate_sha256"],
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Strict Zero-Abbreviation Fail-Closed CLI Parsing
# ---------------------------------------------------------------------------

def _validate_cli_path_value(flag: str, val: str) -> Path:
    _require_valid_utf8_str(val, f"{flag}.path")
    if not val or val.startswith("-"):
        raise Wht8BridgeError("INVALID_PATH_ARG", f"{flag} path cannot be empty or flag-like")
    return Path(val)


def _parse_cli_argv(argv: Any) -> tuple[str, Path | None, bool]:
    checked = validate_cli_argv(argv)
    tokens = sys.argv[1:] if checked is None else checked
    mode: str | None = None
    target: Path | None = None
    as_json = False
    i = 0
    n = len(tokens)
    while i < n:
        tok = _require_valid_utf8_str(tokens[i], "argv")
        if tok == "--json":
            if as_json:
                raise Wht8BridgeError("DUPLICATE_FLAG", "duplicate --json flag")
            as_json = True
            i += 1
        elif tok == "--standalone":
            if mode is not None:
                raise Wht8BridgeError("CONFLICTING_MODE", "multiple mode flags forbidden")
            mode = "standalone"
            i += 1
        elif tok in ("--check", "--write"):
            if mode is not None:
                raise Wht8BridgeError("CONFLICTING_MODE", "multiple mode flags forbidden")
            if i + 1 >= n:
                raise Wht8BridgeError("MISSING_PATH_ARG", f"{tok} requires a path argument")
            mode = "check" if tok == "--check" else "write"
            target = _validate_cli_path_value(tok, tokens[i + 1])
            i += 2
        elif tok.startswith("--check=") or tok.startswith("--write="):
            if mode is not None:
                raise Wht8BridgeError("CONFLICTING_MODE", "multiple mode flags forbidden")
            flag_name, raw_val = tok.split("=", 1)
            mode = "check" if flag_name == "--check" else "write"
            target = _validate_cli_path_value(flag_name, raw_val)
            i += 1
        else:
            raise Wht8BridgeError("UNKNOWN_FLAG", f"unrecognized CLI argument: {tok}")

    if mode is None:
        mode = "standalone"
    return mode, target, as_json


def main(argv: Any = None) -> int:
    try:
        mode, target, as_json = _parse_cli_argv(argv)
        if mode == "check":
            res = verify_certificate_file(target)  # type: ignore[arg-type]
        elif mode == "write":
            res = write_certificate_file(target)  # type: ignore[arg-type]
        else:
            res = verify_certificate_dict(generate_certificate(), full_replay=True)
    except Wht8BridgeError as exc:
        sys.stderr.write(f"FAIL {exc.code}: {exc.message}\n")
        return 1
    if as_json:
        sys.stdout.write(canonical_json_bytes(res).decode("ascii") + "\n")
        return 0
    print(
        f"OK TRACK-S21F-WHT8-FROBENIUS-AUDIT payload_sha256={res['payload_sha256']} "
        f"certificate_sha256={res['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
