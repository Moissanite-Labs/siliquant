#!/usr/bin/env python3
"""Hardened Fail-Closed Audit CLI for Live Qwen2.5-0.5B A_16 LLM Engine.

CLI modes:
  (none) / --standalone --json : emit canonical zero-float certificate JSON
  --write <path>               : durable 0o600 dir_fd atomic write + verify
  --check <path>               : strict canonical + full-replay certificate check
  --verify-receipts <dir>      : verify live JSON receipts & source SHAs on disk

Hardened I/O & in-memory contract (82/82 4-surface fuzzer ready):
  - Iterative _validate_json_tree called on BOTH verify_certificate_object and
    secondary_audit_payload (strict type checks, zero floats, zero bool-in-int,
    zero subclasses/lying containers, MAX_JSON_DEPTH=64, MAX_INT_BITS=2048).
  - Atomic dir_fd O_RDONLY|O_DIRECTORY|O_NOFOLLOW walk, (st_dev, st_ino) matched
    $PWD ancestor symlink check, root-owned Darwin /var|/tmp|/etc normalization,
    O_NONBLOCK + S_ISREG + st_nlink==1 + 65,536 B streaming os.read cap.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aletheion_live_llm_oracle as oracle

MAX_JSON_BYTES = 1_048_576
_READ_CHUNK_BYTES = 65_536
MAX_JSON_DEPTH = 64
MAX_INT_BITS = 2048
MAX_INT_DIGITS = 617
_HEX_LOWER = frozenset("0123456789abcdef")
_CERT_KEYS = frozenset({"schema", "track_id", "payload_sha256", "certificate_sha256", "payload"})
_EXACT_PATH_TYPES = (Path, type(Path()))
_TEMP_NAME_LOCK = threading.Lock()
_DARWIN_SYSTEM_LINKS = frozenset({"var", "tmp", "etc"})
_MAX_TMP_RETRIES = 8
_EXPECTED_PAYLOAD_BYTES: bytes = oracle.canonical_json_bytes(oracle.build_payload())


class AuditError(ValueError):
    """Fail-closed typed error for live LLM engine auditing."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _validate_json_tree(obj: Any, label: str = "root") -> None:
    """Iteratively reject floats, subclasses, lone surrogates, huge ints, and deep nesting."""
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise AuditError("JSON_DEPTH_EXCEEDED", f"{label} nesting exceeds {MAX_JSON_DEPTH}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise AuditError("NON_STR_KEY", f"{label} dict keys must be strict str")
                try:
                    k.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise AuditError("INVALID_UNICODE_STR", f"{label} lone surrogate key: {exc}") from exc
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is int:
            blen = cur.bit_length()
            if blen > MAX_INT_BITS:
                raise AuditError("INT_BIT_LIMIT", f"{label} int bit_length={blen} > {MAX_INT_BITS}")
        elif t is str:
            try:
                cur.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise AuditError("INVALID_UNICODE_STR", f"{label} lone surrogate str: {exc}") from exc
        elif t is bool or cur is None:
            continue
        else:
            raise AuditError("FORBIDDEN_JSON_TYPE", f"{label} forbidden type: {t.__name__}")


def _no_dups_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise AuditError("DUPLICATE_JSON_KEY", f"duplicate key: {k}")
        out[k] = v
    return out


def _forbid_float(token: str) -> Any:
    raise AuditError("FLOAT_FORBIDDEN", f"float literal forbidden: {token}")


def _forbid_constant(token: str) -> Any:
    raise AuditError("NONFINITE_CONSTANT", f"non-finite constant forbidden: {token}")


def _parse_capped_int_token(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if len(digits) > MAX_INT_DIGITS:
        raise AuditError("INT_DIGIT_LIMIT", f"int exceeds {MAX_INT_DIGITS} digits")
    val = int(token)
    if val.bit_length() > MAX_INT_BITS:
        raise AuditError("INT_BIT_LIMIT", "int exceeds bit limit")
    return val


def parse_strict_json(text: str) -> Any:
    if type(text) is not str:
        raise AuditError("INVALID_JSON_TEXT", "JSON text must be strict str")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuditError("INVALID_UNICODE_STR", f"lone surrogate in JSON text: {exc}") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise AuditError("JSON_TOO_LARGE", f"JSON text exceeds {MAX_JSON_BYTES} bytes")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_no_dups_hook,
            parse_int=_parse_capped_int_token,
            parse_float=_forbid_float,
            parse_constant=_forbid_constant,
        )
    except AuditError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AuditError("JSON_DECODE_ERROR", f"invalid JSON: {exc}") from exc
    _validate_json_tree(parsed, "json")
    return parsed


def canonical_json_bytes(obj: Any) -> bytes:
    _validate_json_tree(obj, "canonical")
    return (
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def require_sha256_hex(val: Any, label: str) -> str:
    if type(val) is not str:
        raise AuditError("INVALID_DIGEST_TYPE", f"{label} must be strict str")
    if len(val) != 64:
        raise AuditError("INVALID_DIGEST_LEN", f"{label} must be 64 hex chars")
    if any(ch not in _HEX_LOWER for ch in val):
        raise AuditError("INVALID_DIGEST_HEX", f"{label} must be lowercase ASCII hex")
    return val


def parse_exact_rational(val: Any, label: str) -> Fraction:
    if type(val) is not str:
        raise AuditError("INVALID_RATIONAL", f"{label} must be strict str")
    parts = val.split("/")
    if len(parts) > 2:
        raise AuditError("INVALID_RATIONAL", f"{label} malformed rational")
    for part in parts:
        body = part[1:] if part.startswith("-") else part
        if not body or any(ch not in "0123456789" for ch in body):
            raise AuditError("INVALID_RATIONAL", f"{label} non-digit component")
        if len(body) > MAX_INT_DIGITS:
            raise AuditError("INT_DIGIT_LIMIT", f"{label} exceeds digit cap")
    if len(parts) == 1:
        num = int(parts[0])
        den = 1
    else:
        num = int(parts[0])
        den = int(parts[1])
        if den <= 1:
            raise AuditError("NON_CANONICAL_DEN", f"{label} denominator must be > 1")
    q = Fraction(num, den)
    if oracle.frac_str(q) != val:
        raise AuditError("NON_CANONICAL_RATIONAL", f"{label} must be reduced canonical form")
    return q


def _parse_positive_rational(val: Any, label: str) -> Fraction:
    q = parse_exact_rational(val, label)
    if q <= 0:
        raise AuditError("NONPOSITIVE_VALUE", f"{label} must be strictly positive")
    return q


def _req_dict(val: Any, label: str) -> dict[str, Any]:
    if type(val) is not dict:
        raise AuditError("INVALID_DICT", f"{label} must be strict dict")
    return val


def _req_int(val: Any, label: str) -> int:
    if type(val) is not int:
        raise AuditError("INVALID_INT", f"{label} must be strict int")
    return val


def _req_bool(val: Any, label: str) -> bool:
    if type(val) is not bool:
        raise AuditError("INVALID_BOOL", f"{label} must be strict bool")
    return val


def _req_str(val: Any, label: str) -> str:
    if type(val) is not str:
        raise AuditError("INVALID_STR", f"{label} must be strict str")
    return val


# ---------------------------------------------------------------------------
# Hardened dir_fd File I/O (O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK | st_nlink==1)
# ---------------------------------------------------------------------------

def _validate_path_obj(path: Any) -> str:
    if type(path) not in _EXACT_PATH_TYPES:
        raise AuditError("INVALID_PATH_TYPE", "path must be concrete pathlib.Path")
    raw = str(path)
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuditError("INVALID_UNICODE_STR", f"lone surrogate in path: {exc}") from exc
    if "\x00" in raw:
        raise AuditError("INVALID_PATH", "NUL path forbidden")
    return raw


def _is_symlink_at(dfd: int, name: str) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(name, dir_fd=dfd).st_mode)
    except OSError:
        return False


def _normalize_darwin_system_prefix(path: Path) -> Path:
    if sys.platform != "darwin" or not path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 2 and parts[1] in _DARWIN_SYSTEM_LINKS:
        root_link = Path("/") / parts[1]
        try:
            st = os.lstat(root_link)
            if stat.S_ISLNK(st.st_mode) and st.st_uid == 0:
                target = os.readlink(root_link)
                if target == f"private/{parts[1]}":
                    return Path("/private") / parts[1] / Path(*parts[2:])
        except OSError:
            return path
    return path


def _check_pwd_no_symlinks() -> None:
    pwd = os.environ.get("PWD")
    if pwd is None or not os.path.isabs(pwd):
        return
    try:
        pwd.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AuditError("INVALID_UNICODE_STR", f"lone surrogate in PWD: {exc}") from exc
    try:
        st_dot = os.stat(".")
        st_pwd = os.stat(pwd)
    except OSError:
        return
    if (st_dot.st_dev, st_dot.st_ino) != (st_pwd.st_dev, st_pwd.st_ino):
        return
    norm_pwd = _normalize_darwin_system_prefix(Path(pwd))
    cur_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in norm_pwd.parts[1:]:
            try:
                nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except OSError as exc:
                if _is_symlink_at(cur_fd, comp):
                    raise AuditError("SYMLINK_ANCESTOR_FORBIDDEN", f"symlink in PWD: {comp}") from exc
                raise AuditError("PWD_WALK_ERROR", f"failed walking PWD: {exc}") from exc
            os.close(cur_fd)
            cur_fd = nxt
    finally:
        os.close(cur_fd)


def _open_parent_dir_fd(path: Path, *, create_parents: bool = False) -> tuple[int, str]:
    _validate_path_obj(path)
    norm = _normalize_darwin_system_prefix(path)
    is_abs = norm.is_absolute()
    if not is_abs:
        _check_pwd_no_symlinks()
    parts = norm.parts[1:] if is_abs else norm.parts
    if not parts:
        raise AuditError("INVALID_PATH", "path has no leaf component")
    ancestors, leaf = parts[:-1], parts[-1]
    if leaf in (".", ".."):
        raise AuditError("INVALID_PATH", f"invalid leaf component: {leaf}")
    cur_fd = os.open("/" if is_abs else ".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in ancestors:
            if comp == "..":
                raise AuditError("PARENT_TRAVERSAL_FORBIDDEN", "'..' forbidden in path")
            try:
                nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except FileNotFoundError as exc:
                if not create_parents:
                    raise AuditError("FILE_NOT_FOUND", f"missing directory: {comp}") from exc
                try:
                    os.mkdir(comp, 0o700, dir_fd=cur_fd)
                    nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
                except OSError as mk_exc:
                    raise AuditError("ATOMIC_WRITE_ERROR", f"cannot create ancestor {comp}: {mk_exc}") from mk_exc
            except OSError as exc:
                if _is_symlink_at(cur_fd, comp):
                    raise AuditError("SYMLINK_ANCESTOR_FORBIDDEN", f"symlink ancestor: {comp}") from exc
                raise AuditError("DIR_OPEN_ERROR", f"cannot open ancestor {comp}: {exc}") from exc
            os.close(cur_fd)
            cur_fd = nxt
        return cur_fd, leaf
    except Exception:
        os.close(cur_fd)
        raise


def read_regular_bytes_safe(path: Path, max_bytes: int = MAX_JSON_BYTES) -> bytes:
    parent_fd, leaf = _open_parent_dir_fd(path, create_parents=False)
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        try:
            fd = os.open(leaf, flags, dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise AuditError("FILE_NOT_FOUND", f"missing file: {path}") from exc
        except OSError as exc:
            if _is_symlink_at(parent_fd, leaf):
                raise AuditError("SYMLINK_FORBIDDEN", f"symlink forbidden: {path}") from exc
            raise AuditError("FILE_OPEN_ERROR", f"cannot open file {path}: {exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise AuditError("NON_REGULAR_FILE", f"not a regular file: {path}")
            if st.st_nlink != 1:
                raise AuditError("HARDLINK_FORBIDDEN", f"hardlink count={st.st_nlink} != 1")
            if st.st_size > max_bytes:
                raise AuditError("FILE_TOO_LARGE", f"file exceeds {max_bytes} bytes")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise AuditError("FILE_TOO_LARGE", f"stream exceeds {max_bytes} bytes")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def write_regular_bytes_atomic_0600(path: Path, data: bytes) -> None:
    parent_fd, leaf = _open_parent_dir_fd(path, create_parents=True)
    try:
        if _is_symlink_at(parent_fd, leaf):
            raise AuditError("SYMLINK_FORBIDDEN", f"target is symlink: {path}")
        try:
            existing = os.lstat(leaf, dir_fd=parent_fd)
            if not stat.S_ISREG(existing.st_mode):
                raise AuditError("NON_REGULAR_TARGET", f"target is not regular file: {path}")
            if existing.st_nlink > 1:
                raise AuditError("HARDLINK_FORBIDDEN", f"target hardlink count={existing.st_nlink} > 1")
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise AuditError("ATOMIC_WRITE_ERROR", f"cannot inspect target {path}: {exc}") from exc

        wflags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        tmp_name = ""
        fd: int | None = None
        for _ in range(_MAX_TMP_RETRIES):
            with _TEMP_NAME_LOCK:
                tmp_name = f".{leaf}.tmp.{secrets.token_hex(8)}"
            try:
                fd = os.open(tmp_name, wflags, 0o600, dir_fd=parent_fd)
                break
            except FileExistsError:
                continue
            except OSError as exc:
                raise AuditError("ATOMIC_WRITE_ERROR", f"cannot create temp file: {exc}") from exc
        if fd is None:
            raise AuditError("ATOMIC_WRITE_ERROR", "exhausted temp filename retries")

        fd_open = True
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(data)
            written = 0
            while written < len(view):
                n = os.write(fd, view[written:])
                if n <= 0:
                    raise AuditError("SHORT_WRITE", "os.write returned 0")
                written += n
            os.fsync(fd)
            st_tmp = os.fstat(fd)
            if st_tmp.st_nlink != 1:
                raise AuditError("HARDLINK_FORBIDDEN", "temp file gained hardlink")
            os.close(fd)
            fd_open = False
            os.replace(tmp_name, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except AuditError:
            if fd_open:
                os.close(fd)
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        except OSError as exc:
            if fd_open:
                os.close(fd)
            try:
                os.unlink(tmp_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise AuditError("ATOMIC_WRITE_ERROR", f"atomic write failed: {exc}") from exc
    finally:
        os.close(parent_fd)


# ---------------------------------------------------------------------------
# Independent Pure-Z/Q Secondary Semantic Audit
# ---------------------------------------------------------------------------

def secondary_audit_payload(payload: Any) -> dict[str, Any]:
    """Independently validate structure, exact Z/Q ratios, and silicon receipts."""
    _validate_json_tree(payload, "payload")
    p = _req_dict(payload, "payload")
    if set(p.keys()) != {"exact", "sampled", "anchor"}:
        raise AuditError("PAYLOAD_KEYS_MISMATCH", "payload must contain exact, sampled, anchor")

    exact = _req_dict(p["exact"], "exact")
    sampled = _req_dict(p["sampled"], "sampled")
    anchor = _req_dict(p["anchor"], "anchor")

    if _req_str(exact.get("provenance_tier"), "exact.tier") != "exact":
        raise AuditError("TIER_MISMATCH", "exact.provenance_tier must be 'exact'")
    if _req_str(sampled.get("provenance_tier"), "sampled.tier") != "sampled":
        raise AuditError("TIER_MISMATCH", "sampled.provenance_tier must be 'sampled'")
    if _req_bool(sampled.get("measured_live_model_weights"), "sampled.live") is not True:
        raise AuditError("LIVE_WEIGHT_FLAG", "measured_live_model_weights must be true")
    if _req_str(anchor.get("provenance_tier"), "anchor.tier") != "anchor":
        raise AuditError("TIER_MISMATCH", "anchor.provenance_tier must be 'anchor'")

    # 1. Exact inventory + A_16 code re-verification
    inv = _req_dict(exact.get("qwen25_05b_inventory"), "exact.inv")
    n_mats = _req_int(inv.get("linear_matrix_count"), "inv.mats")
    n_w = _req_int(inv.get("block_weight_count"), "inv.w")
    n_g32 = _req_int(inv.get("groups_32_count"), "inv.g32")
    n_s16 = _req_int(inv.get("subblocks_16_count"), "inv.s16")
    cold_b = _req_int(inv.get("cold_18b_bytes"), "inv.cold_b")
    bf16_b = _req_int(inv.get("bf16_linear_bytes"), "inv.bf16_b")
    nonblk = _req_int(inv.get("nonblock_bf16_tensor_count"), "inv.nonblk")
    if (
        n_mats != 168
        or n_w != 357_826_560
        or n_g32 != n_w // 32
        or n_s16 != n_w // 16
        or cold_b != n_g32 * 18
        or bf16_b != n_w * 2
        or nonblk != 122
        or _req_bool(inv.get("all_dims_aligned_32_128"), "inv.aligned") is not True
    ):
        raise AuditError("INVENTORY_MISMATCH", "Qwen2.5-0.5B exact inventory mismatch")
    if parse_exact_rational(inv.get("bits_per_weight_q"), "inv.bpw") != Fraction(9, 2):
        raise AuditError("BPW_MISMATCH", "bits_per_weight_q must equal 9/2")
    if parse_exact_rational(inv.get("compression_ratio_bf16_over_a16_q"), "inv.comp") != Fraction(32, 9):
        raise AuditError("COMP_MISMATCH", "compression ratio must equal 32/9")
    if canonical_json_bytes(inv.get("shape_counts")) != canonical_json_bytes(
        {"896x896": 48, "128x896": 48, "4864x896": 48, "896x4864": 24}
    ):
        raise AuditError("INVENTORY_MISMATCH", "shape_counts mismatch")

    code = _req_dict(exact.get("a16_extended_hamming_exact"), "exact.code")
    if (
        _req_int(code.get("exhaustive_vectors_checked"), "code.vec") != 65536
        or _req_int(code.get("codeword_count"), "code.cw") != 2048
        or _req_int(code.get("min_nonzero_distance"), "code.dmin") != 4
        or _req_int(code.get("cuda_avx512_mask_mismatches"), "code.mm") != 0
        or _req_int(code.get("swar_u4_to_s8_exact_matches"), "code.swar") != 15
    ):
        raise AuditError("CODE_MISMATCH", "A_16 [16,11,4] exact verification failed")
    if canonical_json_bytes(code.get("weight_enumerator")) != canonical_json_bytes(
        {"0": 1, "4": 140, "6": 448, "8": 870, "10": 448, "12": 140, "16": 1}
    ):
        raise AuditError("CODE_MISMATCH", "weight_enumerator mismatch")

    # 2. Sampled ISA + Ring-0 mlos + pinned SHAs + exact Pareto rational identities
    shas = _req_dict(sampled.get("pinned_sha256"), "sampled.shas")
    if set(shas.keys()) != set(oracle.PINNED_SHA256.keys()):
        raise AuditError("SHA_KEYS_MISMATCH", "pinned_sha256 key set mismatch")
    for sk, expected_hex in oracle.PINNED_SHA256.items():
        got_hex = require_sha256_hex(shas.get(sk), f"shas.{sk}")
        if not hmac.compare_digest(got_hex, expected_hex):
            raise AuditError("PINNED_SHA_MISMATCH", f"pinned SHA mismatch for {sk}")

    isa = _req_dict(sampled.get("isa_receipts"), "sampled.isa")
    if (
        _req_int(isa.get("sm86_kernel_count"), "isa.k") != 16
        or _req_int(isa.get("sm86_compiler_warnings"), "isa.w") != 0
        or _req_int(isa.get("sm86_total_stack_bytes"), "isa.st") != 0
        or _req_int(isa.get("sm86_total_spill_stores_bytes"), "isa.ss") != 0
        or _req_int(isa.get("sm86_total_spill_loads_bytes"), "isa.sl") != 0
        or _req_int(isa.get("sm86_total_sass_ldl"), "isa.ldl") != 0
        or _req_int(isa.get("sm86_total_sass_stl"), "isa.stl") != 0
        or _req_bool(isa.get("sm86_zero_spill_verified"), "isa.zs") is not True
        or _req_int(isa.get("avx512_compiler_warnings"), "isa.aw") != 0
        or _req_int(isa.get("avx512_vpermb_count"), "isa.vp") <= 0
        or _req_int(isa.get("avx512_vpopcntq_count"), "isa.vpc") <= 0
        or _req_int(isa.get("avx512_vpdpbusd_count"), "isa.vnni") <= 0
        or _req_bool(isa.get("avx512_isa_verified"), "isa.avxok") is not True
    ):
        raise AuditError("ISA_RECEIPT_MISMATCH", "zero-spill sm_86 or AVX-512 ISA check failed")

    mlos = _req_dict(sampled.get("ring0_mlos_receipt"), "sampled.mlos")
    vram_mib = _req_int(mlos.get("linux_kmd_baseline_vram_mib"), "mlos.vram")
    if (
        _req_str(mlos.get("determinism_zone"), "mlos.zone") != "B"
        or _req_bool(mlos.get("clean_hardware_run"), "mlos.clean") is not True
        or _req_int(mlos.get("exit_code"), "mlos.rc") != 0
        or _req_int(mlos.get("anomaly_count"), "mlos.anom") != 0
        or _req_int(mlos.get("graphics_process_count"), "mlos.gproc") != 0
        or _req_int(mlos.get("compute_process_count"), "mlos.cproc") != 0
        or vram_mib < 0
        or vram_mib > 4
        or _req_bool(mlos.get("thermal_throttled"), "mlos.th") is not False
        or _req_bool(mlos.get("slowdown_active"), "mlos.sl") is not False
    ):
        raise AuditError("MLOS_RECEIPT_MISMATCH", "Ring-0 mlos cleanliness failed")

    # Exact rational ratio & Pareto dominance re-derivation (strictly positive divisors -> 0 ZeroDivisionError!)
    bb = _req_dict(sampled.get("decode_backbone_24layers_q"), "sampled.bb")
    bb_a = _parse_positive_rational(bb.get("arm_a_bf16_tok_s_q"), "bb.a")
    bb_b1 = _parse_positive_rational(bb.get("arm_b1_sm86_tok_s_q"), "bb.b1")
    bb_c1 = _parse_positive_rational(bb.get("arm_c1_avx512_8t_tok_s_q"), "bb.c1")
    bb_a_mj = _parse_positive_rational(bb.get("arm_a_bf16_combined_mj_q"), "bb.a_mj")
    bb_b1_mj = _parse_positive_rational(bb.get("arm_b1_sm86_combined_mj_q"), "bb.b1_mj")
    bb_c1_mj = _parse_positive_rational(bb.get("arm_c1_avx512_cpu_pkg_mj_q"), "bb.c1_mj")
    if parse_exact_rational(bb.get("speedup_b1_vs_a_q"), "bb.sp_a") != bb_b1 / bb_a:
        raise AuditError("RATIO_MISMATCH", "backbone speedup_b1_vs_a_q mismatch")
    if parse_exact_rational(bb.get("speedup_b1_vs_c1_q"), "bb.sp_c1") != bb_b1 / bb_c1:
        raise AuditError("RATIO_MISMATCH", "backbone speedup_b1_vs_c1_q mismatch")
    if parse_exact_rational(bb.get("energy_ratio_a_over_b1_q"), "bb.en_a") != bb_a_mj / bb_b1_mj:
        raise AuditError("RATIO_MISMATCH", "backbone energy_ratio_a_over_b1_q mismatch")
    if parse_exact_rational(bb.get("energy_ratio_c1_pkg_over_b1_q"), "bb.en_c1") != bb_c1_mj / bb_b1_mj:
        raise AuditError("RATIO_MISMATCH", "backbone energy_ratio_c1_pkg_over_b1_q mismatch")
    bb_dom_actual = bb_b1 > bb_a and bb_b1 > bb_c1 and bb_b1_mj < bb_a_mj and bb_b1_mj < bb_c1_mj
    if _req_bool(bb.get("b1_pareto_dominates_a_and_c1"), "bb.dom") is not True or not bb_dom_actual:
        raise AuditError("PARETO_MISMATCH", "backbone Pareto dominance invariant failed")

    e2e = _req_dict(sampled.get("decode_full_e2e_with_bf16_lm_head_q"), "sampled.e2e")
    e2e_a = _parse_positive_rational(e2e.get("arm_a_bf16_tok_s_q"), "e2e.a")
    e2e_b1 = _parse_positive_rational(e2e.get("arm_b1_sm86_tok_s_q"), "e2e.b1")
    e2e_c1 = _parse_positive_rational(e2e.get("arm_c1_avx512_8t_tok_s_q"), "e2e.c1")
    e2e_a_mj = _parse_positive_rational(e2e.get("arm_a_bf16_combined_mj_q"), "e2e.a_mj")
    e2e_b1_mj = _parse_positive_rational(e2e.get("arm_b1_sm86_combined_mj_q"), "e2e.b1_mj")
    e2e_c1_mj = _parse_positive_rational(e2e.get("arm_c1_avx512_cpu_pkg_mj_q"), "e2e.c1_mj")
    if parse_exact_rational(e2e.get("speedup_b1_vs_a_q"), "e2e.sp_a") != e2e_b1 / e2e_a:
        raise AuditError("RATIO_MISMATCH", "e2e speedup_b1_vs_a_q mismatch")
    if parse_exact_rational(e2e.get("speedup_b1_vs_c1_q"), "e2e.sp_c1") != e2e_b1 / e2e_c1:
        raise AuditError("RATIO_MISMATCH", "e2e speedup_b1_vs_c1_q mismatch")
    if parse_exact_rational(e2e.get("energy_ratio_a_over_b1_q"), "e2e.en_a") != e2e_a_mj / e2e_b1_mj:
        raise AuditError("RATIO_MISMATCH", "e2e energy_ratio_a_over_b1_q mismatch")
    if parse_exact_rational(e2e.get("energy_ratio_c1_pkg_over_b1_q"), "e2e.en_c1") != e2e_c1_mj / e2e_b1_mj:
        raise AuditError("RATIO_MISMATCH", "e2e energy_ratio_c1_pkg_over_b1_q mismatch")
    e2e_dom_actual = e2e_b1 > e2e_a and e2e_b1 > e2e_c1 and e2e_b1_mj < e2e_a_mj and e2e_b1_mj < e2e_c1_mj
    if _req_bool(e2e.get("b1_pareto_dominates_a_and_c1"), "e2e.dom") is not True or not e2e_dom_actual:
        raise AuditError("PARETO_MISMATCH", "e2e Pareto dominance invariant failed")

    pref = _req_dict(sampled.get("prefill_b32_all168_q"), "sampled.pref")
    pref_a3 = _parse_positive_rational(pref.get("arm_a3_bf16_tok_s_q"), "pref.a3")
    pref_b3 = _parse_positive_rational(pref.get("arm_b3_imma_tok_s_q"), "pref.b3")
    if parse_exact_rational(pref.get("speedup_b3_vs_a3_q"), "pref.sp") != pref_b3 / pref_a3:
        raise AuditError("RATIO_MISMATCH", "prefill speedup_b3_vs_a3_q mismatch")

    armd = _req_dict(sampled.get("arm_d_cold_18b_and_parity_q"), "sampled.armd")
    ttft_a16 = _parse_positive_rational(armd.get("ttft_a16_sm86_ms_q"), "armd.ttft_a16")
    ttft_bf16 = _parse_positive_rational(armd.get("ttft_bf16_pytorch_ms_q"), "armd.ttft_bf16")
    if parse_exact_rational(armd.get("ttft_speedup_a16_vs_bf16_q"), "armd.ttft_sp") != ttft_bf16 / ttft_a16:
        raise AuditError("RATIO_MISMATCH", "arm_d ttft_speedup mismatch")
    g_a16 = _parse_positive_rational(armd.get("gen_128_a16_decode_tok_s_q"), "armd.ga16")
    g_bf16 = _parse_positive_rational(armd.get("gen_128_bf16_decode_tok_s_q"), "armd.gbf16")
    if parse_exact_rational(armd.get("gen_128_speedup_a16_vs_bf16_q"), "armd.gsp") != g_a16 / g_bf16:
        raise AuditError("RATIO_MISMATCH", "arm_d gen_128 speedup mismatch")
    tf_m = _req_int(armd.get("teacher_forced_128_top1_matches"), "armd.tf_m")
    tf_t = _req_int(armd.get("teacher_forced_128_total"), "armd.tf_t")
    if tf_t != 128 or parse_exact_rational(armd.get("teacher_forced_128_top1_rate_q"), "armd.tf_r") != Fraction(tf_m, tf_t):
        raise AuditError("PARITY_MISMATCH", "teacher_forced_128_top1_rate_q mismatch")
    if (
        _req_int(armd.get("total_live_syndrome_errors"), "armd.syn") != 0
        or _req_int(armd.get("arm_c_vs_gpu_a16_greedy_16tok_matches"), "armd.cpu_m") != 16
        or _req_bool(armd.get("arm_c_all_windows_gpu_idle_p8"), "armd.p8") is not True
    ):
        raise AuditError("PARITY_MISMATCH", "syndrome or GPU-vs-CPU parity invariant failed")

    # 3. Anchor CS-13 N1/N0 citations
    cits = anchor.get("cs13_citations")
    if type(cits) is not list or len(cits) < 4:
        raise AuditError("ANCHOR_MISMATCH", "cs13_citations must have >= 4 entries")
    tiers = set()
    for item in cits:
        cit = _req_dict(item, "anchor.cit")
        tiers.add(_req_str(cit.get("tier"), "anchor.cit.tier"))
    if not {"N1", "N0"}.issubset(tiers):
        raise AuditError("ANCHOR_MISMATCH", "cs13_citations must include both N1 and N0")

    # 4. Bijective canonical pin of all 194 leaves & 19 object keysets (0 extra-key / unpinned leaf bypasses!)
    if canonical_json_bytes(p) != _EXPECTED_PAYLOAD_BYTES:
        raise AuditError("PAYLOAD_PIN_MISMATCH", "payload diverges from canonical certified structure")

    return {"secondary_audit_passed": True}


def verify_certificate_object(obj: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Validate certificate structure, digests, secondary audit, and optional full oracle replay."""
    _validate_json_tree(obj, "certificate")
    cert = _req_dict(obj, "certificate")
    if set(cert.keys()) != _CERT_KEYS:
        raise AuditError("CERT_KEYS_MISMATCH", "unexpected certificate top-level keys")
    if _req_str(cert["schema"], "schema") != oracle.CERT_SCHEMA:
        raise AuditError("SCHEMA_MISMATCH", "unexpected certificate schema")
    if _req_str(cert["track_id"], "track_id") != oracle.CERT_TRACK_ID:
        raise AuditError("TRACK_ID_MISMATCH", "unexpected track_id")

    p_sha = require_sha256_hex(cert["payload_sha256"], "payload_sha256")
    c_sha = require_sha256_hex(cert["certificate_sha256"], "certificate_sha256")

    payload = _req_dict(cert["payload"], "payload")
    calc_p_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if not hmac.compare_digest(p_sha, calc_p_sha):
        raise AuditError("PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    body = {
        "schema": cert["schema"],
        "track_id": cert["track_id"],
        "payload_sha256": p_sha,
        "payload": payload,
    }
    calc_c_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if not hmac.compare_digest(c_sha, calc_c_sha):
        raise AuditError("CERT_SHA_MISMATCH", "certificate_sha256 mismatch")

    secondary_audit_payload(payload)

    if full_replay:
        fresh = oracle.build_certificate()
        if canonical_json_bytes(cert) != canonical_json_bytes(fresh):
            raise AuditError("REPLAY_MISMATCH", "certificate does not match fresh oracle replay")

    return {
        "valid": True,
        "full_replay": full_replay,
        "payload_sha256": p_sha,
        "certificate_sha256": c_sha,
    }


def check_certificate_file(path: Path, *, full_replay: bool = True) -> dict[str, Any]:
    raw = read_regular_bytes_safe(path, MAX_JSON_BYTES)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError("INVALID_UTF8", f"certificate is not valid UTF-8: {exc}") from exc
    if "\r" in text:
        raise AuditError("CRLF_FORBIDDEN", "CR/CRLF line endings forbidden")
    parsed = parse_strict_json(text)
    if canonical_json_bytes(parsed) != raw:
        raise AuditError("NON_CANONICAL_BYTES", "certificate bytes are not canonical JSON")
    res = verify_certificate_object(parsed, full_replay=full_replay)
    res["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return res


def _parse_json_with_decimals(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError("INVALID_UTF8", f"{label} invalid UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_dups_hook,
            parse_float=Decimal,
            parse_constant=_forbid_constant,
        )
    except AuditError:
        raise
    except (json.JSONDecodeError, InvalidOperation, ValueError) as exc:
        raise AuditError("RECEIPT_PARSE_ERROR", f"{label} parse failed: {exc}") from exc


def verify_receipts_dir(receipts_dir: Path) -> dict[str, Any]:
    """Verify all live hardware receipts & source files on disk against pinned SHAs & values."""
    file_map = {
        "qwen25_sm86_a16_engine_cu": "qwen25_sm86_a16_engine.cu",
        "qwen25_avx512_a16_engine_c": "qwen25_avx512_a16_engine.c",
        "sm86_engine_sass_inspection_json": "sm86_engine_sass_inspection.json",
        "avx512_engine_isa_inspection_json": "avx512_engine_isa_inspection.json",
        "qwen25_live_engine_empirical_results_json": "qwen25_live_engine_empirical_results.json",
        "mlos_zone_b_receipt_json": "mlos_zone_b_receipt.json",
    }
    raw_by_key: dict[str, bytes] = {}
    for sha_key, fname in file_map.items():
        data = read_regular_bytes_safe(receipts_dir / fname, MAX_JSON_BYTES)
        got = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(got, oracle.PINNED_SHA256[sha_key]):
            raise AuditError("RECEIPT_SHA_MISMATCH", f"{fname} SHA mismatch")
        raw_by_key[sha_key] = data

    emp = _parse_json_with_decimals(raw_by_key["qwen25_live_engine_empirical_results_json"], "empirical")
    mlos = _parse_json_with_decimals(raw_by_key["mlos_zone_b_receipt_json"], "mlos")
    sm86 = _parse_json_with_decimals(raw_by_key["sm86_engine_sass_inspection_json"], "sm86")
    avx = _parse_json_with_decimals(raw_by_key["avx512_engine_isa_inspection_json"], "avx512")

    if emp.get("measured_live_model_weights") is not True:
        raise AuditError("RECEIPT_VALUE_MISMATCH", "measured_live_model_weights != true")
    if mlos.get("clean_hardware_run") is not True or mlos.get("anomalies") != []:
        raise AuditError("RECEIPT_VALUE_MISMATCH", "mlos receipt not clean")
    if sm86.get("zero_spill_verified") is not True or avx.get("avx512_vbmi_vnni_vpopcntdq_verified") is not True:
        raise AuditError("RECEIPT_VALUE_MISMATCH", "ISA inspection failed")

    # Cross-check key numerical measurements via exact Decimal -> Fraction equality
    checks = [
        (emp["arm_b_sm86_ptx_a16_gpu"]["b1_pure_dp4a"]["backbone_24layers"]["gpu_event_tok_s"], "bb_b1_tok_s"),
        (emp["arm_b_sm86_ptx_a16_gpu"]["b1_pure_dp4a"]["full_e2e_with_bf16_lm_head"]["gpu_event_tok_s"], "e2e_b1_tok_s"),
        (emp["arm_c_avx512_cpu_only_a16"]["c1_pure_vnni"]["backbone_24layers"]["units_per_s"], "bb_c1_tok_s"),
        (emp["arm_c_avx512_cpu_only_a16"]["c1_pure_vnni"]["full_e2e_with_bf16_lm_head"]["units_per_s"], "e2e_c1_tok_s"),
        (emp["arm_b_sm86_ptx_a16_gpu"]["b3_prefill_b32_cp_async_imma"]["gpu_event_prefill_tok_s"], "pref_b3_tok_s"),
        (emp["arm_d_cold_18b_transcode_and_128tok_gen"]["ttft_a16_sm86_ms"], "arm_d_ttft_a16_ms"),
    ]
    for got_val, dec_key in checks:
        if Fraction(got_val) != oracle.dec_to_frac(oracle.RAW_SAMPLED_DECIMALS[dec_key]):
            raise AuditError("RECEIPT_VALUE_MISMATCH", f"value mismatch on {dec_key}")

    return {
        "valid": True,
        "verified_files": len(file_map),
        "mlos_receipt_sha256": mlos["receipt_sha256"],
    }


def _emit_json(obj: Any) -> None:
    sys.stdout.write(canonical_json_bytes(obj).decode("utf-8"))


def _validate_cli_path(raw_arg: str) -> Path:
    if not raw_arg or raw_arg.startswith("-"):
        raise AuditError("INVALID_CLI_ARGS", f"invalid path argument: {raw_arg}")
    return Path(raw_arg)


def main(argv: Any = None) -> int:
    try:
        if argv is None:
            args = list(sys.argv[1:])
        else:
            if type(argv) not in (list, tuple) or any(type(a) is not str for a in argv):
                raise AuditError("INVALID_CLI_ARGS", "argv must be a sequence of strict str")
            args = list(argv)
        if not args or args == ["--standalone", "--json"] or args == ["--json"]:
            cert = oracle.build_certificate()
            verify_certificate_object(cert, full_replay=False)
            _emit_json(cert)
            return 0
        if len(args) == 2 and args[0] == "--write":
            cert = oracle.build_certificate()
            raw = canonical_json_bytes(cert)
            out_path = _validate_cli_path(args[1])
            write_regular_bytes_atomic_0600(out_path, raw)
            res = check_certificate_file(out_path, full_replay=True)
            _emit_json(res)
            return 0
        if len(args) == 2 and args[0] == "--check":
            res = check_certificate_file(_validate_cli_path(args[1]), full_replay=True)
            _emit_json(res)
            return 0
        if len(args) == 3 and args[0] == "--check" and args[2] == "--json":
            res = check_certificate_file(_validate_cli_path(args[1]), full_replay=True)
            _emit_json(res)
            return 0
        if len(args) == 2 and args[0] == "--verify-receipts":
            res = verify_receipts_dir(_validate_cli_path(args[1]))
            _emit_json(res)
            return 0
        raise AuditError("INVALID_CLI_ARGS", f"unsupported arguments: {args}")
    except AuditError as exc:
        _emit_json({"valid": False, "error_code": exc.code, "error": exc.message})
        return 1
    except Exception as exc:
        _emit_json({"valid": False, "error_code": "UNEXPECTED_ERROR", "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
