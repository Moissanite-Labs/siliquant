"""Fail-Closed Certificate Verifier & CLI for RTX 3080 Siliquant A_16 CUDA Study.

Supports:
  --standalone --json       Emit canonical RTX 3080 A_16 study certificate to stdout
  --check <cert_path>       strict hardened fail-closed verification of a cert file
  --write <cert_path>       Atomic 0o600 strict write with allow_nan=False & dual fsync
  --verify-receipts <dir>   strict read & pure-Q verification of live GPU JSON + mlos receipt

Hardened against all 4-surface strict adversarial vectors:
  - Iterative JSON tree validation (bounded depth=64, exact types, no bool-in-int, no floats/NaN)
  - Iterative bijective exact-type payload pin in secondary_audit_payload (0 hollow ratios)
  - NUL and UTF-8 surrogate CLI path rejection
  - Logical $PWD symlink-ancestor check on relative paths + O_DIRECTORY|O_NOFOLLOW walk
  - Atomic 0o600 write with allow_nan=False, st_nlink==1 guard, bounded EEXIST retry, dual fsync
  - Fail-closed schema error trapping in verify_receipts_directory

Zero floats, zero float() calls, zero bare asserts, zero .isdigit(), zero islink.
"""

from __future__ import annotations

import errno
import hashlib
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

import rtx3080_siliquant_a16_oracle as oracle

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
    """Fail-closed strict audit error carrying a structured error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _is_hex64(val: Any) -> bool:
    return type(val) is str and len(val) == 64 and all(ch in HEX_LOWER_CHARS for ch in val)


def _strict_canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize payload to strict canonical UTF-8 JSON bytes (allow_nan=False, LF terminated)."""
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return (text + "\n").encode("utf-8")


def _validate_json_tree_iter(root: Any) -> None:
    """Iteratively validate exact JSON types (no subclasses, no floats, bounded depth & ints)."""
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
                stack.append((v, depth + 1))
        elif t is list:
            for item in node:
                stack.append((item, depth + 1))
        elif t is int:
            if node.bit_length() > MAX_INT_BITS:
                raise AuditError("INT_TOO_LARGE", "integer magnitude exceeded bound")
        elif t is bool or t is str:
            continue
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


def _walk_dir_nofollow(parts: tuple[str, ...], is_abs: bool) -> int:
    """Open a directory path by walking every component with O_DIRECTORY | O_NOFOLLOW."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if not is_abs:
        logical_pwd = os.environ.get("PWD", "")
        if logical_pwd.startswith("/"):
            pwd_fd = _walk_dir_nofollow(Path(logical_pwd).parts, True)
            os.close(pwd_fd)
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


def _open_parent_dir_nofollow(path: Path) -> tuple[int, str]:
    """Walk parent components with O_DIRECTORY | O_NOFOLLOW from root or cwd."""
    if not path.parts:
        raise AuditError("EMPTY_PATH", "empty file path")
    name = path.name
    if not name or name in (".", ".."):
        raise AuditError("INVALID_BASENAME", "path must reference a regular file basename")
    return _walk_dir_nofollow(path.parent.parts, path.is_absolute()), name


def _open_dir_nofollow(dir_path: Path) -> int:
    """Open an existing directory with O_DIRECTORY | O_NOFOLLOW across all ancestors."""
    return _walk_dir_nofollow(dir_path.parts, dir_path.is_absolute())


def _read_regular_bytes_in_dir(dir_fd: int, name: str, max_bytes: int) -> bytes:
    """Fail-closed strict regular-file read anchored on dir_fd with streaming size cap."""
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
    finally:
        os.close(fd)


def read_regular_file_bytes(path: Path, max_bytes: int = MAX_CERT_BYTES) -> bytes:
    """Read a regular file safely under strict dir_fd NOFOLLOW guards."""
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
    raw = _strict_canonical_json_bytes(payload)
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
                tmp_created = True
                break
            except FileExistsError:
                continue
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
    raise AuditError("FLOAT_FORBIDDEN", "floating-point JSON token rejected in certificate")


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
    if _strict_canonical_json_bytes(parsed) != raw:
        raise AuditError("NON_CANONICAL_BYTES", "certificate bytes are not canonical JSON")
    return parsed


def secondary_audit_payload(payload: Any) -> None:
    """Fail-closed schema + digest + exact-type bijective audit without oracle replay."""
    if type(payload) is not dict:
        raise AuditError("INVALID_ROOT", "certificate payload must be an exact dict")
    _validate_json_tree_iter(payload)
    if "certificate_sha256" not in payload or not _is_hex64(payload["certificate_sha256"]):
        raise AuditError("INVALID_DIGEST", "certificate_sha256 must be 64 lowercase hex chars")
    body = {k: v for k, v in payload.items() if k != "certificate_sha256"}
    recomputed = hashlib.sha256(_strict_canonical_json_bytes(body)).hexdigest()
    if recomputed != payload["certificate_sha256"]:
        raise AuditError("DIGEST_MISMATCH", "certificate_sha256 does not match payload")
    _verify_exact_tree_iter(body, _EXPECTED_BODY_CACHE)


def verify_certificate_payload(payload: Any, full_replay: bool = True) -> dict[str, Any]:
    """Verify certificate via strict secondary audit and optional full oracle replay."""
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
    """Read and verify a strict certificate file."""
    raw = read_regular_file_bytes(path, max_bytes=MAX_CERT_BYTES)
    payload = parse_strict_certificate_bytes(raw)
    result = verify_certificate_payload(payload, full_replay=full_replay)
    result["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


def verify_receipts_directory(dir_path: Path) -> dict[str, Any]:
    """Read live empirical JSON, mlos receipt, and CUDA source via strict dir_fd and verify in Q."""
    dir_fd = _open_dir_nofollow(dir_path)
    try:
        emp_raw = _read_regular_bytes_in_dir(
            dir_fd, "rtx3080_siliquant_a16_empirical_results.json", MAX_RECEIPT_BYTES
        )
        mlos_raw = _read_regular_bytes_in_dir(
            dir_fd, "mlos_zone_b_receipt.json", MAX_CERT_BYTES
        )
        cu_raw = _read_regular_bytes_in_dir(
            dir_fd, "rtx3080_siliquant_a16_kernels.cu", MAX_RECEIPT_BYTES
        )
    finally:
        os.close(dir_fd)

    try:
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

    if type(emp_obj) is not dict or type(mlos_obj) is not dict:
        raise AuditError("RECEIPT_ROOT_INVALID", "receipt JSON roots must be objects")

    try:
        res = oracle.verify_live_receipts(
            emp_obj,
            mlos_obj,
            hashlib.sha256(emp_raw).hexdigest(),
            hashlib.sha256(mlos_raw).hexdigest(),
            hashlib.sha256(cu_raw).hexdigest(),
        )
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
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
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args == [] or args == ["--standalone", "--json"]:
            cert = oracle.build_certificate()
            sys.stdout.buffer.write(_strict_canonical_json_bytes(cert))
            return 0
        if len(args) == 2 and args[0] == "--check":
            res = verify_certificate_file(_validate_path_arg(args[1]), full_replay=True)
            sys.stdout.buffer.write(_strict_canonical_json_bytes(res))
            return 0
        if len(args) == 2 and args[0] == "--write":
            cert = oracle.build_certificate()
            file_sha = write_certificate_atomic(_validate_path_arg(args[1]), cert)
            out = {"written": True, "file_sha256": file_sha, "certificate_sha256": cert["certificate_sha256"]}
            sys.stdout.buffer.write(_strict_canonical_json_bytes(out))
            return 0
        if len(args) == 2 and args[0] == "--verify-receipts":
            res = verify_receipts_directory(_validate_path_arg(args[1]))
            sys.stdout.buffer.write(_strict_canonical_json_bytes(res))
            return 0
        raise AuditError(
            "USAGE_ERROR",
            "expected [--standalone --json] | [--check PATH] | [--write PATH] | [--verify-receipts DIR]",
        )
    except AuditError as exc:
        err = {"valid": False, "error_code": exc.code, "error": exc.message}
        sys.stderr.buffer.write(_strict_canonical_json_bytes(err))
        return 2


if __name__ == "__main__":
    sys.exit(main())
