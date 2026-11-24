"""Fail-Closed Zone-B Verifier & CLI for Track S67 (CEER Cycle-2 Block-H4 & Re-Entrant Flip).

Hardened against adversarial 4-surface fuzzing vectors (CLI, full_replay=True/False,
secondary_audit_payload):
  - Iterative JSON tree validation (bounded depth=64, exact types, no bool-in-int, strict UTF-8)
  - Iterative bijective exact-type payload pin in secondary_audit_payload (0 hollow checks)
  - Exact-bool guard on full_replay flag
  - Constant-time hmac.compare_digest SHA-256 verification
  - NUL / UTF-8 surrogate path rejection and typed argv validation inside try
  - Inode-matched logical $PWD symlink-ancestor check + O_DIRECTORY|O_NOFOLLOW walk
    (with verified root-owned macOS /var, /tmp, /etc -> /private/* prefix normalization)
  - Atomic 0o600 write with allow_nan=False, st_nlink==1 guard, bounded EEXIST retry, dual fsync

Zero floats, zero float() calls, zero bare asserts.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import secrets as _rand_tokens
import stat
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

_HERE_DIR = str(Path(__file__).resolve().parent)
if _HERE_DIR not in sys.path:
    sys.path.insert(0, _HERE_DIR)

import rtx3080_s67_oracle as oracle

MAX_CERT_BYTES = 65536
MAX_RECEIPT_BYTES = 524288
MAX_JSON_DEPTH = 64
MAX_INT_BITS = 8000
MAX_TMP_RETRIES = 8
HEX_LOWER_CHARS = frozenset("0123456789abcdef")

_EXPECTED_CERT_CACHE: dict[str, Any] = oracle.build_certificate()
_EXPECTED_BODY_CACHE: dict[str, Any] = {
    k: v for k, v in _EXPECTED_CERT_CACHE.items() if k != "certificate_sha256"
}


class AuditError(ValueError):
    """Fail-closed Zone-B audit error carrying a structured error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _is_hex64(val: Any) -> bool:
    return type(val) is str and len(val) == 64 and all(ch in HEX_LOWER_CHARS for ch in val)


def _validate_json_tree_iter(root: Any) -> None:
    """Iteratively validate exact JSON types (no subclasses, no floats, valid UTF-8, bounded depth & ints)."""
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise AuditError("DEPTH_EXCEEDED", "JSON nesting depth exceeded")
        t = type(node)
        if t is dict:
            for k, v in node.items():
                if type(k) is not str:
                    raise AuditError("INVALID_KEY_TYPE", "dict keys must be exact str")
                try:
                    k.encode("utf-8", "strict")
                except UnicodeEncodeError:
                    raise AuditError("INVALID_UTF8_SURROGATE", "dict key contains unpaired UTF-16 surrogate") from None
                stack.append((v, depth + 1))
        elif t is list:
            for item in node:
                stack.append((item, depth + 1))
        elif t is int:
            if node.bit_length() > MAX_INT_BITS:
                raise AuditError("INT_TOO_LARGE", "integer magnitude exceeded bound")
        elif t is bool:
            continue
        elif t is str:
            try:
                node.encode("utf-8", "strict")
            except UnicodeEncodeError:
                raise AuditError("INVALID_UTF8_SURROGATE", "string contains unpaired UTF-16 surrogate") from None
        else:
            raise AuditError("INVALID_VALUE_TYPE", f"forbidden value type: {t.__name__}")


def _verify_exact_tree_iter(actual: Any, expected: Any) -> None:
    """Iteratively enforce exact-type bijective equality against canonical expected tree."""
    stack: list[tuple[Any, Any]] = [(actual, expected)]
    while stack:
        a, e = stack.pop()
        if type(a) is not type(e):
            raise AuditError("SCHEMA_TYPE_MISMATCH", f"expected {type(e).__name__}, got {type(a).__name__}")
        if type(e) is dict:
            if frozenset(a.keys()) != frozenset(e.keys()):
                raise AuditError("SCHEMA_KEY_MISMATCH", "object keyset mismatch")
            for k in e:
                stack.append((a[k], e[k]))
        elif type(e) is list:
            if len(a) != len(e):
                raise AuditError("SCHEMA_LEN_MISMATCH", "list length mismatch")
            for ai, ei in zip(a, e):
                stack.append((ai, ei))
        else:
            if a != e:
                raise AuditError("INVARIANT_MISMATCH", "payload leaf does not match certified invariant")


_DARWIN_SYSTEM_ROOT_ALIASES: dict[str, tuple[str, ...]] = {
    "var": ("private/var", "/private/var"),
    "tmp": ("private/tmp", "/private/tmp"),
    "etc": ("private/etc", "/private/etc"),
}


def _normalize_darwin_system_prefix(parts: tuple[str, ...]) -> tuple[str, ...]:
    """Map verified root-owned macOS /var, /tmp, /etc ancestor prefixes to /private/*."""
    if sys.platform != "darwin":
        return parts
    if len(parts) < 2 or parts[0] != "/" or ".." in parts:
        return parts
    head = parts[1]
    allowed_targets = _DARWIN_SYSTEM_ROOT_ALIASES.get(head)
    if allowed_targets is None:
        return parts
    alias = "/" + head
    try:
        st = os.lstat(alias)
        if not stat.S_ISLNK(st.st_mode) or st.st_uid != 0:
            return parts
        target = os.readlink(alias)
    except OSError:
        return parts
    if target not in allowed_targets:
        return parts
    return ("/", "private", *parts[1:])


def _walk_dir_nofollow(parts: tuple[str, ...], is_abs: bool) -> int:
    """Open a directory path walking every component with O_DIRECTORY | O_NOFOLLOW."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if not is_abs:
        logical_pwd = os.environ.get("PWD", "")
        if logical_pwd.startswith("/"):
            try:
                cwd_st = os.stat(".")
                pwd_st = os.stat(logical_pwd)
                same_cwd = (pwd_st.st_dev, pwd_st.st_ino) == (cwd_st.st_dev, cwd_st.st_ino)
            except (OSError, ValueError):
                same_cwd = False
            if same_cwd:
                pwd_fd = _walk_dir_nofollow(Path(logical_pwd).parts, True)
                os.close(pwd_fd)
    else:
        parts = _normalize_darwin_system_prefix(parts)
    try:
        cur_fd = os.open("/" if is_abs else ".", flags)
    except OSError as exc:
        raise AuditError("DIR_OPEN_FAILED", f"cannot open anchor directory: {exc.strerror}") from None
    walk_parts = parts[1:] if is_abs else parts

    try:
        for comp in walk_parts:
            if comp == ".":
                continue
            try:
                nxt_fd = os.open(comp, flags, dir_fd=cur_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise AuditError("SYMLINK_FORBIDDEN", f"symlink ancestor component: {comp}") from None
                if exc.errno == errno.ENOTDIR:
                    try:
                        if stat.S_ISLNK(os.lstat(comp, dir_fd=cur_fd).st_mode):
                            raise AuditError("SYMLINK_FORBIDDEN", f"symlink ancestor component: {comp}") from None
                    except OSError:
                        pass
                raise AuditError("DIR_OPEN_FAILED", f"cannot open directory component {comp}: {exc.strerror}") from None
            os.close(cur_fd)
            cur_fd = nxt_fd
    except Exception:
        os.close(cur_fd)
        raise
    return cur_fd


def _check_path_encoding(path: Path) -> None:
    raw = os.fspath(path)
    if "\x00" in raw:
        raise AuditError("INVALID_PATH", "path contains NUL byte")
    try:
        raw.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise AuditError("INVALID_PATH", "path contains invalid UTF-8 surrogates") from None


def _open_parent_dir_nofollow(path: Path) -> tuple[int, str]:
    _check_path_encoding(path)
    if not path.parts:
        raise AuditError("EMPTY_PATH", "empty file path")
    name = path.name
    if not name or name in (".", ".."):
        raise AuditError("INVALID_BASENAME", "path must reference a regular file basename")
    return _walk_dir_nofollow(path.parent.parts, path.is_absolute()), name


def _open_dir_nofollow(dir_path: Path) -> int:
    _check_path_encoding(dir_path)
    return _walk_dir_nofollow(dir_path.parts, dir_path.is_absolute())


def _read_regular_bytes_in_dir(dir_fd: int, name: str, max_bytes: int) -> bytes:
    open_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, open_flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise AuditError("SYMLINK_FORBIDDEN", f"symlink target rejected: {name}") from None
        raise AuditError("FILE_OPEN_FAILED", f"cannot open {name}: {exc.strerror}") from None

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise AuditError("NON_REGULAR_FILE", f"target is not a regular file: {name}")
        if st.st_nlink != 1:
            raise AuditError("HARDLINK_FORBIDDEN", f"hardlinked target rejected (nlink={st.st_nlink})")
        if st.st_size > max_bytes:
            raise AuditError("FILE_TOO_LARGE", f"file exceeds {max_bytes} byte cap")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 8192)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise AuditError("FILE_TOO_LARGE", f"stream exceeded {max_bytes} byte cap")
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise AuditError("FILE_READ_FAILED", f"cannot read {name}: {exc.strerror}") from None
    finally:
        os.close(fd)


def read_regular_file_bytes(path: Path, max_bytes: int = MAX_CERT_BYTES) -> bytes:
    dir_fd, name = _open_parent_dir_nofollow(path)
    try:
        return _read_regular_bytes_in_dir(dir_fd, name, max_bytes)
    finally:
        os.close(dir_fd)


def write_certificate_atomic(path: Path, payload: dict[str, Any]) -> str:
    """Validate payload tree and write canonical JSON atomically at 0o600 with dual fsync."""
    if type(payload) is not dict:
        raise AuditError("INVALID_ROOT", "certificate payload must be an exact dict")
    _validate_json_tree_iter(payload)
    raw = oracle.canonical_json_bytes(payload)
    dir_fd, name = _open_parent_dir_nofollow(path)
    tmp_name = ""
    tmp_created = False
    try:
        try:
            existing = os.lstat(name, dir_fd=dir_fd)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise AuditError("SYMLINK_FORBIDDEN", f"refusing to overwrite symlink: {name}")
            if not stat.S_ISREG(existing.st_mode):
                raise AuditError("NON_REGULAR_FILE", f"refusing to overwrite non-regular target: {name}")
            if existing.st_nlink != 1:
                raise AuditError("HARDLINK_FORBIDDEN", f"refusing to overwrite hardlinked file: {name}")

        fd = -1
        for _ in range(MAX_TMP_RETRIES):
            tmp_name = f".{name}.tmp.{_rand_tokens.token_hex(8)}"
            try:
                fd = os.open(
                    tmp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=dir_fd,
                )
            except FileExistsError:
                continue
            tmp_created = True
            break
        if not tmp_created:
            raise AuditError("ATOMIC_WRITE_FAILED", "exhausted unique temporary filename attempts")

        try:
            os.fchmod(fd, 0o600)
            view = memoryview(raw)
            written = 0
            while written < len(view):
                n = os.write(fd, view[written:])
                if n <= 0:
                    raise AuditError("SHORT_WRITE", "os.write returned non-positive count")
                written += n
            st_tmp = os.fstat(fd)
            if not stat.S_ISREG(st_tmp.st_mode) or st_tmp.st_nlink != 1:
                raise AuditError("TEMP_TAMPERED", "temporary file link count or type changed")
            os.fsync(fd)
        finally:
            os.close(fd)

        os.replace(tmp_name, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        tmp_created = False
        os.fsync(dir_fd)
        return hashlib.sha256(raw).hexdigest()
    except AuditError:
        if tmp_created:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except OSError:
                pass
        raise
    except OSError as exc:
        if tmp_created:
            try:
                os.unlink(tmp_name, dir_fd=dir_fd)
            except OSError:
                pass
        raise AuditError("ATOMIC_WRITE_FAILED", f"write failed: {exc.strerror}") from None
    finally:
        os.close(dir_fd)


def _reject_float(_token: str) -> Any:
    raise AuditError("FLOAT_FORBIDDEN", "floating-point JSON token rejected")


def _reject_constant(token: str) -> Any:
    raise AuditError("CONSTANT_FORBIDDEN", f"non-standard JSON constant rejected: {token}")


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in pairs:
        if key in out:
            raise AuditError("DUPLICATE_KEY", f"duplicate JSON key rejected: {key}")
        out[key] = val
    return out


def parse_strict_certificate_bytes(raw: bytes) -> dict[str, Any]:
    """Parse raw UTF-8 certificate bytes rejecting floats, duplicates, CRLF, and deep nesting."""
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise AuditError("NON_CANONICAL_FORMAT", "certificate must be LF-terminated without CR")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise AuditError("INVALID_UTF8", "certificate is not valid UTF-8") from None
    try:
        parsed = json.loads(
            text,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object_pairs,
        )
    except AuditError:
        raise
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AuditError("JSON_PARSE_ERROR", str(exc)) from None
    if type(parsed) is not dict:
        raise AuditError("INVALID_ROOT", "certificate root must be a JSON object")
    _validate_json_tree_iter(parsed)
    if oracle.canonical_json_bytes(parsed) != raw:
        raise AuditError("NON_CANONICAL_BYTES", "certificate bytes are not canonical JSON")
    return parsed


def secondary_audit_payload(payload: Any) -> None:
    """Fail-closed schema + constant-time digest + exact-type bijective audit without oracle replay."""
    if type(payload) is not dict:
        raise AuditError("INVALID_ROOT", "certificate payload must be an exact dict")
    _validate_json_tree_iter(payload)
    if "certificate_sha256" not in payload or not _is_hex64(payload["certificate_sha256"]):
        raise AuditError("INVALID_DIGEST", "certificate_sha256 must be 64 lowercase hex chars")
    body = {k: v for k, v in payload.items() if k != "certificate_sha256"}
    recomputed = hashlib.sha256(oracle.canonical_json_bytes(body)).hexdigest()
    if not hmac.compare_digest(recomputed, payload["certificate_sha256"]):
        raise AuditError("DIGEST_MISMATCH", "certificate_sha256 does not match payload")
    _verify_exact_tree_iter(body, _EXPECTED_BODY_CACHE)


def verify_certificate_payload(payload: Any, full_replay: bool = True) -> dict[str, Any]:
    """Verify certificate via strict secondary audit and optional full oracle replay."""
    if type(full_replay) is not bool:
        raise AuditError("INVALID_FLAG_TYPE", "full_replay must be an exact bool")
    secondary_audit_payload(payload)
    if full_replay:
        expected = oracle.build_certificate()
        if payload != expected:
            raise AuditError("REPLAY_MISMATCH", "payload diverges from deterministic oracle replay")
    return {
        "valid": True,
        "full_replay": full_replay,
        "certificate_sha256": payload["certificate_sha256"],
    }


def verify_certificate_file(path: Path, full_replay: bool = True) -> dict[str, Any]:
    if type(full_replay) is not bool:
        raise AuditError("INVALID_FLAG_TYPE", "full_replay must be an exact bool")
    raw = read_regular_file_bytes(path, max_bytes=MAX_CERT_BYTES)
    payload = parse_strict_certificate_bytes(raw)
    result = verify_certificate_payload(payload, full_replay=full_replay)
    result["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def verify_receipts_directory(dir_path: Path) -> dict[str, Any]:
    """Read preregistration, live 4-model empirical JSON, and mlos receipt via Zone-B dir_fd."""
    dir_fd = _open_dir_nofollow(dir_path)
    try:
        prereg_raw = _read_regular_bytes_in_dir(
            dir_fd, "s67_preregistration.json", MAX_CERT_BYTES
        )
        emp_raw = _read_regular_bytes_in_dir(
            dir_fd, "rtx3080_s67_empirical_results.json", MAX_RECEIPT_BYTES
        )
        mlos_raw = _read_regular_bytes_in_dir(
            dir_fd, "mlos_zone_b_receipt.json", MAX_CERT_BYTES
        )
    finally:
        os.close(dir_fd)

    try:
        prereg_obj = json.loads(
            prereg_raw.decode("utf-8"),
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object_pairs,
        )
        emp_obj = json.loads(
            emp_raw.decode("utf-8"),
            parse_float=Fraction,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object_pairs,
        )
        mlos_obj = json.loads(
            mlos_raw.decode("utf-8"),
            parse_float=Fraction,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object_pairs,
        )
    except AuditError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AuditError("RECEIPT_PARSE_ERROR", str(exc)) from None

    if type(prereg_obj) is not dict or type(emp_obj) is not dict or type(mlos_obj) is not dict:
        raise AuditError("RECEIPT_ROOT_INVALID", "receipt JSON roots must be objects")

    try:
        res = oracle.verify_live_receipts(
            prereg_obj,
            emp_obj,
            mlos_obj,
            hashlib.sha256(prereg_raw).hexdigest(),
            hashlib.sha256(emp_raw).hexdigest(),
            hashlib.sha256(mlos_raw).hexdigest(),
        )
    except (KeyError, TypeError, AttributeError, ValueError, ZeroDivisionError) as exc:
        raise AuditError("RECEIPT_SCHEMA_ERROR", f"malformed receipt structure: {exc}") from None

    if not res["all_verified"]:
        raise AuditError("RECEIPT_VERIFICATION_FAILED", json.dumps(res["checks"], sort_keys=True))
    return res


def _validate_path_arg(raw_arg: str) -> Path:
    if not raw_arg or raw_arg.startswith("-") or "\x00" in raw_arg:
        raise AuditError("INVALID_CLI_PATH", "path argument must not be empty, flag-like, or contain NUL")
    try:
        raw_arg.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise AuditError("INVALID_CLI_PATH", "path argument contains invalid UTF-8 surrogates") from None
    return Path(raw_arg)


def main(argv: list[str] | None = None) -> int:
    try:
        if argv is None:
            args = list(sys.argv[1:])
        else:
            if type(argv) is not list or any(type(item) is not str for item in argv):
                raise AuditError("USAGE_ERROR", "argv must be a list of strings")
            args = list(argv)
        if args == [] or args == ["--standalone", "--json"]:
            cert = oracle.build_certificate()
            sys.stdout.buffer.write(oracle.canonical_json_bytes(cert))
            return 0
        if len(args) == 2 and args[0] == "--check":
            res = verify_certificate_file(_validate_path_arg(args[1]), full_replay=True)
            sys.stdout.buffer.write(oracle.canonical_json_bytes(res))
            return 0
        if len(args) == 2 and args[0] == "--write":
            cert = oracle.build_certificate()
            file_sha = write_certificate_atomic(_validate_path_arg(args[1]), cert)
            out = {"written": True, "file_sha256": file_sha, "certificate_sha256": cert["certificate_sha256"]}
            sys.stdout.buffer.write(oracle.canonical_json_bytes(out))
            return 0
        if len(args) == 2 and args[0] == "--verify-receipts":
            res = verify_receipts_directory(_validate_path_arg(args[1]))
            sys.stdout.buffer.write(oracle.canonical_json_bytes(res))
            return 0
        raise AuditError(
            "USAGE_ERROR",
            "expected [--standalone --json] | [--check PATH] | [--write PATH] | [--verify-receipts DIR]",
        )
    except AuditError as exc:
        err = {"valid": False, "error_code": exc.code, "error": exc.message}
        sys.stderr.buffer.write(oracle.canonical_json_bytes(err))
        return 2


if __name__ == "__main__":
    sys.exit(main())
