#!/usr/bin/env python3
"""Track S30 (Portal 26): Exact-Q M1 Max Siliquant Gemma 3 (270M/1B/4B) Tomography.

100% pure Python 3 standard library (exact Z/Q arithmetic only):
  - ZERO floating-point literals, float() casts, or float trust-path arithmetic
  - ZERO bare assert statements
  - ZERO RNG / non-deterministic inputs

Certifies five exact mathematical & engineering pillars grounded in the pinned
M1 Max `Moissanite-Labs/siliquant` `exp/gemma-ball-results @ dc10d4bbca` receipts:
  1. Exact-Q raw-receipt provenance (`102` JSONL rows across `270M/1B/4B`) &
     complete `PR #47` preregistered `P1..P10` scoring (plus `token_embd.weight`
     BF16->F16 mutation provenance).
  2. Exact log-convex rate-distortion theorem (`12/12` measured triples) &
     two-sided pure-Q Jensen + rational-log iso-KL enclosure (upgrading `PR #57`
     `analyze.py:217`).
  3. Exact per-kind `Q4_K_M` fallback decoder from `weight_err_db_by_kind`
     (`270M/1B` `Q8_0`/`Q5`/`Q6_K`/`Q4_K` fallbacks vs `4B` normal `4.8151 bpw`).
  4. Exact pure-Z/Q `E8`-vs-`H8*E8` noise-covariance anisotropy operator theorem
     (`EBG128` vs `HBG128`, Law-5 bridge to `Track S20b` `siliquant#42`) &
     Law-2 steelman `hEBG128:b560s16` prediction (`PRED-M1-01`).
  5. Exact static AST audit & verified patch specification for `pinned_inputs/run.py`
     (`RUN-BUG-01` real-imatrix dispatch & `RUN-BUG-02` BF16 `token_embd` preservation).
"""
from __future__ import annotations

import ast
from collections.abc import Sequence
import errno
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA_VERSION = "aletheia.gemma_m1_tomography.certificate.v1"
TRACK_ID = "TRACK-S30-SILIQUANT-GEMMA-M1-TOMOGRAPHY"

MAX_JSON_BYTES = 1_048_576
MAX_JSON_DEPTH = 64
MAX_INT_DIGITS = 1000
MAX_DEC_MANTISSA_DIGITS = 256
MAX_DEC_EXPONENT = 512

_LOWER_HEX = frozenset("0123456789abcdef")
_CERT_TOP_KEYS = frozenset({"schema", "track_id", "payload_sha256", "certificate_sha256", "payload"})
_PAYLOAD_TOP_KEYS = frozenset(
    {
        "oracle_class",
        "novelty_tier",
        "zero_trust_path_floats",
        "zero_rng",
        "zero_bare_asserts",
        "pinned_source_commit",
        "pillar1_provenance_and_prereg",
        "pillar2_rate_distortion_convexity",
        "pillar3_q4km_fallback_decoder",
        "pillar4_e8_h8_anisotropy",
        "pillar5_run_py_ast_audit",
    }
)
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_READ_CHUNK_BYTES = 65536

PINNED_INPUTS_DIR = Path(__file__).resolve().parent / "pinned_inputs"

MODELS: tuple[str, ...] = ("gemma-3-270m-it", "gemma-3-1b-it", "gemma-3-4b-it")
TEXTS: tuple[str, ...] = ("wikitext", "python")
KINDS: tuple[str, ...] = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")

PINNED_FILE_SHA256: dict[str, str] = {
    "gemma-3-270m-it.jsonl": "dafe50f93262be5628b97a83b915a0f65aa4eaa52f09b1b9bfd36381c4b34a83",
    "gemma-3-1b-it.jsonl": "040717b2d9c24d47e64fcc3b41156a375d651f180cd7b9f558ec603c60f7733f",
    "gemma-3-4b-it.jsonl": "5895d850aff01a154a3b57e1c47468c90121e34316fa6e850ae866f6ec4dc23e",
    "run.py": "afd7ef8ae34d7986cef9a10825518a61174a44feb325ee0c1684b8b6cded159c",
}

EXPECTED_VARIANTS: frozenset[str] = frozenset(
    {
        "file:Q4_0",
        "file:Q4_K_M",
        "file:Q4_K_M-imat",
        "ZBG128:b560s16",
        "hZBG128:b560s16",
        "hZBG128:b560s16~rand",
        "ZBG128:b560s16+K10",
        "hZBG128:b560s16+K10",
        "EBG128:b560s16",
        "VE8Z32:16",
        "HBG128:b560s16",
        "ZBG128:b496s16",
        "hZBG128:b496s16",
        "ZBG128:b624s16",
        "hZBG128:b624s16",
        "ZB128@R:16",
        "ZB128@R:32",
    }
)
REAL_IMAT_ARMS: frozenset[str] = frozenset(
    {
        "file:Q4_K_M-imat",
        "hZBG128:b560s16",
        "hZBG128:b560s16+K10",
        "hZBG128:b496s16",
        "hZBG128:b624s16",
    }
)
WEIGHTED_BALL_RE = re.compile(r"^([ih])(ZBG|EBG|HBG)(\d+):b(\d+)s(\d+)(?:~(rand))?$")
LOUD_BALL_RE = re.compile(r"^(h?ZBG)(\d+):b(\d+)s(\d+)\+K(\d+)$")
_DTYPE_TO_LLAMA_FLAG: dict[str, str] = {"BF16": "bf16", "F16": "f16", "F32": "f32"}


class GemmaM1Error(ValueError):
    """Typed fail-closed error for all Track S30 oracle and Zone-B checks."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Strict Zone-B Parsing, ASCII Exact Rationals/Decimals & Integrity Primitives
# ---------------------------------------------------------------------------

def _int_digit_len(value: int) -> int:
    mag = abs(value)
    if mag.bit_length() > MAX_INT_DIGITS * 4:
        return MAX_INT_DIGITS + 1
    return len(str(mag))


def _is_ascii_digits(text: str) -> bool:
    return bool(text) and all("0" <= ch <= "9" for ch in text)


def require_exact_int(value: Any, label: str, *, min_val: int | None = None, max_val: int | None = None) -> int:
    if type(value) is not int:
        raise GemmaM1Error("INVALID_INT", f"{label} must be exact int")
    if _int_digit_len(value) > MAX_INT_DIGITS:
        raise GemmaM1Error("INT_DIGITS_EXCEEDED", f"{label} exceeds {MAX_INT_DIGITS} digits")
    if min_val is not None and value < min_val:
        raise GemmaM1Error("INT_TOO_SMALL", f"{label} < {min_val}")
    if max_val is not None and value > max_val:
        raise GemmaM1Error("INT_TOO_LARGE", f"{label} > {max_val}")
    return value


def require_exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise GemmaM1Error("INVALID_BOOL", f"{label} must be exact bool")
    return value


def require_exact_str(value: Any, label: str, *, non_empty: bool = True) -> str:
    if type(value) is not str:
        raise GemmaM1Error("INVALID_STR", f"{label} must be exact str")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise GemmaM1Error("INVALID_UTF8_SURROGATE", f"{label} contains lone UTF-16 surrogate") from exc
    if non_empty and not value:
        raise GemmaM1Error("EMPTY_STR", f"{label} must be non-empty")
    return value


def require_exact_fraction(value: Any, label: str) -> Fraction:
    if type(value) is not Fraction:
        raise GemmaM1Error("INVALID_FRACTION", f"{label} must be exact Fraction")
    return value


def require_sha256_hex(value: Any, label: str) -> str:
    text = require_exact_str(value, label)
    if len(text) != 64 or not all(ch in _LOWER_HEX for ch in text):
        raise GemmaM1Error("INVALID_SHA256", f"{label} must be 64 lowercase hex chars")
    return text


def format_exact_rational(value: Fraction) -> str:
    q = require_exact_fraction(value, "rational")
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def _outward_floor_10(value: Fraction, digits: int) -> Fraction:
    scale = 10 ** digits
    return Fraction((value.numerator * scale) // value.denominator, scale)


def _outward_ceil_10(value: Fraction, digits: int) -> Fraction:
    scale = 10 ** digits
    return Fraction(-((-value.numerator * scale) // value.denominator), scale)


def format_decimal_trunc(value: Fraction, digits: int) -> str:
    """Exact pure-integer truncated decimal string (zero floats)."""
    q = require_exact_fraction(value, "value")
    d = require_exact_int(digits, "digits", min_val=0, max_val=24)
    sign = "-" if q < 0 else ""
    mag = abs(q)
    scale = 10 ** d
    scaled = (mag.numerator * scale) // mag.denominator
    if d == 0:
        return f"{sign}{scaled}"
    whole, frac = divmod(scaled, scale)
    return f"{sign}{whole}.{frac:0{d}d}"


def _parse_ascii_int_digits(digits: str) -> int:
    acc = 0
    for ch in digits:
        acc = acc * 10 + (ord(ch) - 48)
    return acc


def parse_exact_rational(value: Any, label: str = "rational") -> Fraction:
    """Parse canonical reduced exact rational string `p` or `p/q` (strict ASCII 0-9, `q > 1`, `gcd(|p|,q)=1`)."""
    text = require_exact_str(value, label)
    parts = text.split("/")
    if len(parts) == 1:
        num_s = parts[0]
        neg = num_s.startswith("-")
        num_body = num_s[1:] if neg else num_s
        if not _is_ascii_digits(num_body):
            raise GemmaM1Error("INVALID_RATIONAL", f"{label} contains non-ASCII-digit characters")
        if len(num_body) > MAX_INT_DIGITS:
            raise GemmaM1Error("INT_DIGITS_EXCEEDED", f"{label} exceeds {MAX_INT_DIGITS} digits")
        if len(num_body) > 1 and num_body[0] == "0":
            raise GemmaM1Error("NON_CANONICAL_RATIONAL", f"{label} has leading zero")
        if neg and num_body == "0":
            raise GemmaM1Error("NON_CANONICAL_RATIONAL", f"{label} cannot be -0")
        mag = _parse_ascii_int_digits(num_body)
        return Fraction(-mag if neg else mag, 1)
    if len(parts) != 2:
        raise GemmaM1Error("INVALID_RATIONAL", f"{label} has multiple '/' separators")
    num_s, den_s = parts[0], parts[1]
    neg = num_s.startswith("-")
    num_body = num_s[1:] if neg else num_s
    if not _is_ascii_digits(num_body) or not _is_ascii_digits(den_s):
        raise GemmaM1Error("INVALID_RATIONAL", f"{label} contains non-ASCII-digit characters")
    if len(num_body) > MAX_INT_DIGITS or len(den_s) > MAX_INT_DIGITS:
        raise GemmaM1Error("INT_DIGITS_EXCEEDED", f"{label} exceeds {MAX_INT_DIGITS} digits")
    den = _parse_ascii_int_digits(den_s)
    if den == 0:
        raise GemmaM1Error("ZERO_DENOMINATOR", f"{label} has zero denominator")
    if num_body[0] == "0" or den_s[0] == "0":
        raise GemmaM1Error("NON_CANONICAL_RATIONAL", f"{label} has leading zero or zero numerator in fraction")
    if den == 1:
        raise GemmaM1Error("NON_CANONICAL_RATIONAL", f"{label} denominator must be > 1")
    mag = _parse_ascii_int_digits(num_body)
    if math.gcd(mag, den) != 1:
        raise GemmaM1Error("NON_CANONICAL_RATIONAL", f"{label} must be reduced to lowest terms")
    return Fraction(-mag if neg else mag, den)


def parse_exact_decimal_token(token: Any, label: str = "decimal") -> Fraction:
    """Parse an ASCII decimal/scientific JSON number token directly to exact Fraction (zero floats)."""
    text = require_exact_str(token, label)
    neg = text.startswith("-")
    body = text[1:] if neg else text
    lower = body.lower()
    if "e" in lower:
        e_parts = lower.split("e")
        if len(e_parts) != 2:
            raise GemmaM1Error("INVALID_DECIMAL", f"{label} has multiple exponents")
        mant_s, exp_s = e_parts
        exp_neg = exp_s.startswith("-")
        exp_body = exp_s[1:] if exp_s[:1] in ("+", "-") else exp_s
        if not _is_ascii_digits(exp_body) or len(exp_body) > 5:
            raise GemmaM1Error("INVALID_DECIMAL", f"{label} has invalid exponent")
        exp_val = -int(exp_body) if exp_neg else int(exp_body)
        if abs(exp_val) > MAX_DEC_EXPONENT:
            raise GemmaM1Error("DECIMAL_EXPONENT_EXCEEDED", f"{label} exponent out of bounds")
    else:
        mant_s, exp_val = lower, 0

    if "." in mant_s:
        d_parts = mant_s.split(".")
        if len(d_parts) != 2:
            raise GemmaM1Error("INVALID_DECIMAL", f"{label} has multiple decimal points")
        int_s, frac_s = d_parts
        if not _is_ascii_digits(int_s) or not _is_ascii_digits(frac_s):
            raise GemmaM1Error("INVALID_DECIMAL", f"{label} has non-ASCII-digit mantissa")
        digits_s = int_s + frac_s
        exp_val -= len(frac_s)
    else:
        if not _is_ascii_digits(mant_s):
            raise GemmaM1Error("INVALID_DECIMAL", f"{label} has non-ASCII-digit integer")
        digits_s = mant_s

    if len(digits_s) > MAX_DEC_MANTISSA_DIGITS:
        raise GemmaM1Error("DECIMAL_MANTISSA_EXCEEDED", f"{label} mantissa exceeds limit")
    mag = int(digits_s)
    num = -mag if neg else mag
    if exp_val >= 0:
        return Fraction(num * (10 ** exp_val), 1)
    return Fraction(num, 10 ** (-exp_val))


def validate_json_tree_iterative(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    limit = require_exact_int(max_depth, "max_depth", min_val=1, max_val=256)
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > limit:
            raise GemmaM1Error("JSON_DEPTH_EXCEEDED", f"JSON nesting depth > {limit}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                require_exact_str(k, "dict_key", non_empty=False)
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is str:
            require_exact_str(cur, "str_value", non_empty=False)
        elif t is int:
            require_exact_int(cur, "int_value")
        elif t is bool or cur is None:
            continue
        else:
            raise GemmaM1Error("FORBIDDEN_JSON_TYPE", f"forbidden JSON value type: {t.__name__}")


def _strict_parse_int_token(token: Any) -> int:
    text = require_exact_str(token, "int_token")
    neg = text.startswith("-")
    body = text[1:] if neg else text
    if not _is_ascii_digits(body):
        raise GemmaM1Error("INVALID_INT_TOKEN", "integer token must contain ASCII 0-9 digits only")
    if len(body) > MAX_INT_DIGITS:
        raise GemmaM1Error("INT_DIGITS_EXCEEDED", f"integer token exceeds {MAX_INT_DIGITS} digits")
    if len(body) > 1 and body[0] == "0":
        raise GemmaM1Error("INVALID_INT_TOKEN", "integer token cannot have leading zero")
    if neg and body == "0":
        raise GemmaM1Error("INVALID_INT_TOKEN", "integer token cannot be -0")
    mag = _parse_ascii_int_digits(body)
    return -mag if neg else mag


def _reject_float_token(token: str) -> Any:
    raise GemmaM1Error("FLOAT_FORBIDDEN", f"floating-point JSON literal forbidden: {token}")


def _reject_constant_token(token: str) -> Any:
    raise GemmaM1Error("NONFINITE_JSON_FORBIDDEN", f"non-finite JSON token forbidden: {token}")


def _strict_object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in pairs:
        require_exact_str(key, "json_key", non_empty=False)
        if key in out:
            raise GemmaM1Error("DUPLICATE_JSON_KEY", f"duplicate JSON key: {key!r}")
        out[key] = val
    return out


def parse_strict_json_text(text: str, *, require_object: bool = True) -> Any:
    require_exact_str(text, "json_text")
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise GemmaM1Error("JSON_TOO_LARGE", f"JSON text exceeds {MAX_JSON_BYTES} bytes")
    try:
        parsed = json.loads(
            text,
            parse_float=_reject_float_token,
            parse_int=_strict_parse_int_token,
            parse_constant=_reject_constant_token,
            object_pairs_hook=_strict_object_pairs_hook,
        )
    except GemmaM1Error:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise GemmaM1Error("INVALID_JSON", f"JSON parse failed: {type(exc).__name__}") from exc
    if require_object and type(parsed) is not dict:
        raise GemmaM1Error("ROOT_NOT_OBJECT", "JSON root must be an object")
    validate_json_tree_iterative(parsed)
    return parsed


def _parse_raw_jsonl_line(line: str) -> dict[str, Any]:
    require_exact_str(line, "jsonl_line")
    try:
        parsed = json.loads(
            line,
            parse_float=parse_exact_decimal_token,
            parse_int=_strict_parse_int_token,
            parse_constant=_reject_constant_token,
            object_pairs_hook=_strict_object_pairs_hook,
        )
    except GemmaM1Error:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise GemmaM1Error("INVALID_JSONL", f"JSONL parse failed: {type(exc).__name__}") from exc
    if type(parsed) is not dict:
        raise GemmaM1Error("INVALID_JSONL", "JSONL row must be a JSON object")
    return parsed


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_tree_iterative(obj)
    try:
        rendered = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GemmaM1Error("CANONICAL_ENCODE_ERROR", f"cannot encode canonical JSON: {exc}") from exc
    return rendered.encode("ascii")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


# ---------------------------------------------------------------------------
# Race-Free Pinned-Input Reader (dir_fd/openat + Pin SHA-256)
# ---------------------------------------------------------------------------

def _read_pinned_file_bytes(pinned_dir: Path, filename: str, expected_sha256: str | None = None) -> bytes:
    if type(pinned_dir) is not type(Path(".")):
        raise GemmaM1Error("INVALID_PINNED_DIR", "pinned_dir must be exact pathlib.Path")
    require_exact_str(filename, "filename")
    if "/" in filename or "\x00" in filename or filename in (".", ".."):
        raise GemmaM1Error("INVALID_PINNED_FILENAME", "filename must be a bare leaf name without NUL")
    raw_dir = require_exact_str(str(pinned_dir), "pinned_dir")
    if "\x00" in raw_dir:
        raise GemmaM1Error("INVALID_PINNED_DIR", "pinned_dir contains NUL byte")
    checked_sha = require_sha256_hex(expected_sha256, "expected_sha256") if expected_sha256 is not None else None

    anchor = pinned_dir.anchor if pinned_dir.is_absolute() else "."
    parts = [p for p in (pinned_dir.parts[1:] if pinned_dir.is_absolute() else pinned_dir.parts) if p not in ("", ".")]
    try:
        dfd = os.open(anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise GemmaM1Error("PINNED_READ_ERROR", f"cannot open anchor: {type(exc).__name__}") from exc
    for part in parts:
        try:
            nxt = os.open(part, _DIR_OPEN_FLAGS, dir_fd=dfd)
        except OSError as exc:
            os.close(dfd)
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise GemmaM1Error("SYMLINK_FORBIDDEN", "symlink forbidden in pinned_dir") from exc
            raise GemmaM1Error("PINNED_READ_ERROR", f"cannot traverse pinned_dir: {type(exc).__name__}") from exc
        os.close(dfd)
        dfd = nxt

    try:
        try:
            fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise GemmaM1Error("SYMLINK_FORBIDDEN", f"symlink forbidden at pinned file {filename}") from exc
            raise GemmaM1Error("PINNED_READ_ERROR", f"cannot open pinned file {filename}: {type(exc).__name__}") from exc
    finally:
        os.close(dfd)

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise GemmaM1Error("NON_REGULAR_FILE", f"pinned file {filename} is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_JSON_BYTES:
                raise GemmaM1Error("FILE_TOO_LARGE", f"pinned file {filename} exceeds byte cap")
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(fd)

    if checked_sha is not None:
        actual_sha = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(actual_sha, checked_sha):
            raise GemmaM1Error("PINNED_INPUT_SHA_MISMATCH", f"pinned file {filename} SHA-256 mismatch")
    return raw


# ---------------------------------------------------------------------------
# Exact Pure-Q Elementary Enclosures & Linear Algebra / E8-H8 Primitives
# ---------------------------------------------------------------------------

def ln_bounds_exact(x: Fraction, terms: int = 16) -> tuple[Fraction, Fraction]:
    """Exact pure-Q two-sided rational interval [lo, hi] enclosing ln(x) for x > 0 (zero floats)."""
    q = require_exact_fraction(x, "x")
    if q <= 0:
        raise GemmaM1Error("LOG_DOMAIN_ERROR", "ln_bounds_exact requires x > 0")
    n = require_exact_int(terms, "terms", min_val=1, max_val=64)
    if q < 1:
        lo_inv, hi_inv = ln_bounds_exact(Fraction(1, 1) / q, n)
        return (-hi_inv, -lo_inv)
    u = (q - 1) / (q + 1)
    u2 = u * u
    p = u
    lo = Fraction(0, 1)
    for k in range(n):
        lo += Fraction(2, 2 * k + 1) * p
        p *= u2
    rem = Fraction(2, 2 * n + 1) * p / (Fraction(1, 1) - u2)
    return (lo, lo + rem)


def exp_neg_bounds_exact(x: Fraction, even_order: int = 12) -> tuple[Fraction, Fraction]:
    """Exact pure-Q two-sided rational interval [lo, hi] enclosing exp(-x) for 0 <= x <= 1."""
    q = require_exact_fraction(x, "x")
    if q < 0 or q > 1:
        raise GemmaM1Error("EXP_DOMAIN_ERROR", "exp_neg_bounds_exact requires 0 <= x <= 1")
    n = require_exact_int(even_order, "even_order", min_val=2, max_val=40)
    if n % 2 != 0:
        raise GemmaM1Error("EXP_ORDER_ODD", "even_order must be an even integer")
    total = Fraction(1, 1)
    term = Fraction(1, 1)
    lo = Fraction(0, 1)
    for k in range(1, n + 1):
        term = -term * q / Fraction(k, 1)
        total += term
        if k == n - 1:
            lo = total
    return (lo, total)


def exp_neg_interval_exact(x_lo: Fraction, x_hi: Fraction, even_order: int = 12) -> tuple[Fraction, Fraction]:
    """Enclose exp(-x) for x in [x_lo, x_hi] subset [0, 1] using monotone-decreasing property."""
    a = require_exact_fraction(x_lo, "x_lo")
    b = require_exact_fraction(x_hi, "x_hi")
    if a > b:
        raise GemmaM1Error("INVALID_INTERVAL", "x_lo must be <= x_hi")
    lo_from_hi, _ = exp_neg_bounds_exact(b, even_order)
    _, hi_from_lo = exp_neg_bounds_exact(a, even_order)
    return (lo_from_hi, hi_from_lo)


def _require_square_fraction_matrix(mat: Any, label: str, *, exact_dim: int | None = None) -> list[list[Fraction]]:
    if not isinstance(mat, Sequence) or isinstance(mat, (str, bytes)):
        raise GemmaM1Error("INVALID_MATRIX", f"{label} must be a sequence of rows")
    n = len(mat)
    if n == 0:
        raise GemmaM1Error("INVALID_MATRIX", f"{label} must be non-empty")
    if exact_dim is not None and n != exact_dim:
        raise GemmaM1Error("INVALID_MATRIX_DIM", f"{label} must be {exact_dim}x{exact_dim}")
    out: list[list[Fraction]] = []
    for r_idx, row in enumerate(mat):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise GemmaM1Error("INVALID_MATRIX", f"{label}[{r_idx}] must be a sequence")
        if len(row) != n:
            raise GemmaM1Error("INVALID_MATRIX_SHAPE", f"{label} must be square ({n}x{n})")
        out.append([require_exact_fraction(v, f"{label}[{r_idx}][{c_idx}]") for c_idx, v in enumerate(row)])
    return out


def _require_fraction_vec8(vec: Any, label: str) -> list[Fraction]:
    if not isinstance(vec, Sequence) or isinstance(vec, (str, bytes)):
        raise GemmaM1Error("INVALID_VEC8", f"{label} must be a sequence")
    if len(vec) != 8:
        raise GemmaM1Error("INVALID_VEC8", f"{label} must have length 8")
    return [require_exact_fraction(v, f"{label}[{i}]") for i, v in enumerate(vec)]


def mat_mul_exact(a: Sequence[Sequence[Fraction]], b: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    ma = _require_square_fraction_matrix(a, "a")
    mb = _require_square_fraction_matrix(b, "b", exact_dim=len(ma))
    n = len(ma)
    return [
        [sum((ma[i][k] * mb[k][j] for k in range(n)), Fraction(0, 1)) for j in range(n)]
        for i in range(n)
    ]


def mat_transpose_exact(a: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    ma = _require_square_fraction_matrix(a, "a")
    n = len(ma)
    return [[ma[j][i] for j in range(n)] for i in range(n)]


def mat_trace_exact(a: Sequence[Sequence[Fraction]]) -> Fraction:
    ma = _require_square_fraction_matrix(a, "a")
    return sum((ma[i][i] for i in range(len(ma))), Fraction(0, 1))


def mat_vec8_exact(mat: Sequence[Sequence[Fraction]], vec: Sequence[Fraction]) -> list[Fraction]:
    m = _require_square_fraction_matrix(mat, "mat", exact_dim=8)
    v = _require_fraction_vec8(vec, "vec")
    return [sum((m[i][j] * v[j] for j in range(8)), Fraction(0, 1)) for i in range(8)]


def hadamard8_unnorm_exact() -> list[list[Fraction]]:
    """Exact unnormalized order-8 Sylvester Hadamard matrix H_8 in {-1, +1}^{8x8} (H_8^2 = 8 I_8)."""
    h: list[list[int]] = [[1]]
    for _ in range(3):
        h = [row + row for row in h] + [row + [-x for x in row] for row in h]
    return [[Fraction(x, 1) for x in row] for row in h]


def nearest_d8_exact(vec: Sequence[Fraction]) -> list[Fraction]:
    """Exact pure-Q Conway-Sloane nearest-point decoder for D_8 = {z in Z^8 : sum(z) even}."""
    x = _require_fraction_vec8(vec, "vec")
    rounded = [(2 * v.numerator + v.denominator) // (2 * v.denominator) for v in x]
    if sum(rounded) % 2 != 0:
        best_i = max(range(8), key=lambda i: (abs(x[i] - rounded[i]), -i))
        rounded[best_i] += 1 if x[best_i] >= rounded[best_i] else -1
    return [Fraction(v, 1) for v in rounded]


def nearest_e8_exact(vec: Sequence[Fraction]) -> list[Fraction]:
    """Exact pure-Q Conway-Sloane nearest-point decoder for E_8 = D_8 U (D_8 + (1/2)^8)."""
    x = _require_fraction_vec8(vec, "vec")
    p0 = nearest_d8_exact(x)
    half = Fraction(1, 2)
    p1 = [v + half for v in nearest_d8_exact([u - half for u in x])]
    d0 = sum(((a - b) * (a - b) for a, b in zip(x, p0)), Fraction(0, 1))
    d1 = sum(((a - b) * (a - b) for a, b in zip(x, p1)), Fraction(0, 1))
    return p0 if d0 <= d1 else p1


def _e8_minimal_roots_exact() -> list[list[Fraction]]:
    roots: list[list[Fraction]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = [Fraction(0, 1)] * 8
                    v[i] = Fraction(si, 1)
                    v[j] = Fraction(sj, 1)
                    roots.append(v)
    for mask in range(256):
        if mask.bit_count() % 2 == 0:
            roots.append([Fraction(-1 if (mask >> i) & 1 else 1, 2) for i in range(8)])
    return roots


def classify_llama_quant_kind(w_err_db: Fraction) -> str:
    """Exact pure-Q band classifier for llama.cpp 6 dB/bit quantization ladder from weight_err_db_by_kind."""
    db_v = require_exact_fraction(w_err_db, "w_err_db")
    if db_v <= Fraction(-33, 1):
        return "Q8_0"
    if db_v <= Fraction(-278, 10):
        return "Q6_K"
    if db_v <= Fraction(-25, 1):
        return "Q5_0_or_Q5_K"
    if db_v <= Fraction(-215, 10):
        return "Q4_K"
    return "Q4_0_or_lower"


def needs_real_imatrix_fixed(arm: str) -> bool:
    """Repaired RUN-BUG-01 predicate matching all real-imatrix arms (hZBG/hEBG/hHBG/iZBG/iEBG/iHBG + Q4_K_M-imat)."""
    tok = require_exact_str(arm, "arm")
    if tok == "file:Q4_K_M-imat":
        return True
    m_w = WEIGHTED_BALL_RE.match(tok)
    if m_w is not None:
        return m_w.group(6) != "rand"
    m_l = LOUD_BALL_RE.match(tok)
    if m_l is not None:
        return m_l.group(1).startswith("h")
    return False


def quantize_tensor_type_flags_fixed(tensor_names: Sequence[str], token_embd_dtype: str) -> list[str]:
    """Repaired RUN-BUG-02 flag selector preserving source BF16/F16/F32 dtype of token_embd.weight."""
    if not isinstance(tensor_names, Sequence) or isinstance(tensor_names, (str, bytes)):
        raise GemmaM1Error("INVALID_TENSOR_NAMES", "tensor_names must be a sequence of str")
    names = {require_exact_str(n, "tensor_name") for n in tensor_names}
    dtype_key = require_exact_str(token_embd_dtype, "token_embd_dtype")
    if dtype_key not in _DTYPE_TO_LLAMA_FLAG:
        raise GemmaM1Error("UNSUPPORTED_EMBD_DTYPE", f"unsupported token_embd dtype: {dtype_key}")
    flags = ["--tensor-type", f"^token_embd\\.weight$={_DTYPE_TO_LLAMA_FLAG[dtype_key]}"]
    if any("per_layer_model_proj" in n for n in names):
        flags.extend(["--tensor-type", "per_layer_model_proj=f32"])
    if any("inp_gate" in n for n in names):
        flags.extend(["--tensor-type", "inp_gate=f32"])
    if any(".proj." in n for n in names):
        flags.extend(["--tensor-type", r"\.proj\.=f32"])
    return flags


# ---------------------------------------------------------------------------
# Pinned JSONL Dataset Loader & Strict Provenance Validator
# ---------------------------------------------------------------------------

def _as_fraction_num(
    value: Any,
    label: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> Fraction:
    t = type(value)
    if t is int:
        require_exact_int(value, label)
        q = Fraction(value, 1)
    elif t is Fraction:
        q = value
    else:
        raise GemmaM1Error("INVALID_NUMERIC_FIELD", f"{label} must be exact int or Fraction")
    if positive and q <= 0:
        raise GemmaM1Error("NON_POSITIVE_METRIC", f"{label} must be > 0")
    if non_negative and q < 0:
        raise GemmaM1Error("NEGATIVE_METRIC", f"{label} must be >= 0")
    return q


def load_pinned_jsonl_dataset(
    pinned_dir: Path | None = None,
    *,
    verify_pin_hashes: bool = True,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, str]]:
    """Load and fail-closed validate all 102 raw M1 Max JSONL rows over exact Q."""
    require_exact_bool(verify_pin_hashes, "verify_pin_hashes")
    base_dir = PINNED_INPUTS_DIR if pinned_dir is None else pinned_dir
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    provenance: dict[tuple[Any, ...], str] = {}

    for model in MODELS:
        fname = f"{model}.jsonl"
        exp_sha = PINNED_FILE_SHA256[fname] if verify_pin_hashes else None
        raw = _read_pinned_file_bytes(base_dir, fname, expected_sha256=exp_sha)
        file_hashes[fname] = hashlib.sha256(raw).hexdigest()
        try:
            text_body = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GemmaM1Error("INVALID_UTF8", f"{fname} is not valid UTF-8") from exc

        model_rows = 0
        for line in text_body.splitlines():
            if not line.strip():
                continue
            row = _parse_raw_jsonl_line(line)
            if row.get("model_id") != model:
                raise GemmaM1Error("MODEL_ID_MISMATCH", f"unexpected model_id in {fname}")
            text_name = require_exact_str(row.get("text"), "text")
            variant = require_exact_str(row.get("variant"), "variant")
            if text_name not in TEXTS or variant not in EXPECTED_VARIANTS:
                raise GemmaM1Error("UNEXPECTED_ROW_SCOPE", f"unexpected text/variant: {text_name}/{variant}")
            if require_exact_bool(row.get("bridge"), "bridge"):
                raise GemmaM1Error("UNEXPECTED_BRIDGE_ROW", "bridge row forbidden in Gemma 3 sweep")
            key = (model, text_name, variant)
            if key in rows:
                raise GemmaM1Error("DUPLICATE_ROW_KEY", f"duplicate row {key}")

            m_sha = require_sha256_hex(row.get("model_sha256"), "model_sha256")
            t_sha = require_sha256_hex(row.get("text_sha256"), "text_sha256")
            pins: list[tuple[tuple[Any, ...], str]] = [
                (("model", model), m_sha),
                (("text", model, text_name), t_sha),
            ]
            if variant in REAL_IMAT_ARMS or variant.endswith("~rand"):
                im_sha = require_sha256_hex(row.get("imatrix_sha256"), "imatrix_sha256")
                pins.append((("imatrix", model, variant.endswith("~rand")), im_sha))
            if variant.startswith("file:"):
                f_sha = require_sha256_hex(row.get("file_sha256"), "file_sha256")
                pins.append((("file", model, variant), f_sha))
                diff_list = row.get("other_tensors_that_differ")
                if type(diff_list) is not list or not all(type(x) is str for x in diff_list):
                    raise GemmaM1Error("INVALID_OTHER_TENSORS", "other_tensors_that_differ must be list[str]")

            for p_key, p_val in pins:
                prev = provenance.get(p_key)
                if prev is not None and not hmac.compare_digest(prev, p_val):
                    raise GemmaM1Error("PROVENANCE_CONFLICT", f"conflicting hash for {p_key}")
                provenance[p_key] = p_val

            norm: dict[str, Any] = {
                "model_id": model,
                "text": text_name,
                "variant": variant,
                "bits_per_weight": _as_fraction_num(row.get("bits_per_weight"), "bits_per_weight", positive=True),
                "kl_mean": _as_fraction_num(row.get("kl_mean"), "kl_mean", positive=True),
                "kl_mean_se": _as_fraction_num(row.get("kl_mean_se"), "kl_mean_se", positive=True),
                "ppl": _as_fraction_num(row.get("ppl"), "ppl", positive=True),
                "ppl_base": _as_fraction_num(row.get("ppl_base"), "ppl_base", positive=True),
                "same_top": _as_fraction_num(row.get("same_top"), "same_top", non_negative=True),
                "weight_err_db_pooled": _as_fraction_num(row.get("weight_err_db_pooled"), "weight_err_db_pooled"),
                "other_tensors_that_differ": list(row["other_tensors_that_differ"]) if "other_tensors_that_differ" in row else [],
            }
            by_kind_raw = row.get("weight_err_db_by_kind")
            if type(by_kind_raw) is not dict or set(by_kind_raw.keys()) != set(KINDS):
                raise GemmaM1Error("INVALID_BY_KIND", f"missing tensor kinds in {key}")
            norm["weight_err_db_by_kind"] = {
                k: _as_fraction_num(by_kind_raw[k], f"weight_err_db_by_kind.{k}") for k in KINDS
            }
            rows[key] = norm
            model_rows += 1
        if model_rows != 34:
            raise GemmaM1Error("UNEXPECTED_ROW_COUNT", f"{fname} expected 34 rows, got {model_rows}")

    return rows, file_hashes


# ---------------------------------------------------------------------------
# Pillar 1: Raw-Receipt Provenance & Complete PR #47 P1..P10 Scoring
# ---------------------------------------------------------------------------

def _pair_z_sq(cand: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate vs baseline: positive gain means cand has lower KL than base."""
    c_kl = require_exact_fraction(cand["kl_mean"], "cand.kl_mean")
    b_kl = require_exact_fraction(base["kl_mean"], "base.kl_mean")
    c_se = require_exact_fraction(cand["kl_mean_se"], "cand.kl_mean_se")
    b_se = require_exact_fraction(base["kl_mean_se"], "base.kl_mean_se")
    if c_kl <= 0 or b_kl <= 0:
        raise GemmaM1Error("NON_POSITIVE_KL", "kl_mean must be strictly positive")
    if c_se <= 0 or b_se <= 0:
        raise GemmaM1Error("NON_POSITIVE_SE", "kl_mean_se must be strictly positive")
    gain = b_kl - c_kl
    var = c_se * c_se + b_se * b_se
    z_sq = (gain * gain) / var
    # Exact pure-integer 2-decimal signed |z| string via math.isqrt (zero floats; never emits "-0.00")
    scaled_z = math.isqrt((z_sq.numerator * 10000) // z_sq.denominator)
    if scaled_z == 0:
        sgn = ""
    elif gain > 0:
        sgn = "+"
    else:
        sgn = "-"
    z_str = f"{sgn}{scaled_z // 100}.{scaled_z % 100:02d}"
    return {
        "cand_kl": format_exact_rational(c_kl),
        "base_kl": format_exact_rational(b_kl),
        "gain_kl": format_exact_rational(gain),
        "z_sq": format_exact_rational(z_sq),
        "z_signed_2dp": z_str,
        "favors_cand_ge_2se": all((gain > 0, z_sq >= 4)),
        "favors_base_ge_2se": all((gain < 0, z_sq >= 4)),
    }


def _exact_spearman_and_inversions(sub_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Exact Pearson-of-midranks Spearman rho^2 (tie-exact over Q) and top-1 inversion count."""
    if type(sub_rows) not in (list, tuple) or len(sub_rows) <= 1:
        raise GemmaM1Error("INSUFFICIENT_ROWS", "Spearman requires at least 2 rows")
    n = len(sub_rows)

    def _avg_ranks(vals: list[Fraction], *, descending: bool) -> list[Fraction]:
        order = sorted(range(n), key=lambda idx: vals[idx], reverse=descending)
        ranks = [Fraction(0, 1)] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = Fraction(i + j + 2, 2)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    r_kl = _avg_ranks([r["kl_mean"] for r in sub_rows], descending=False)
    r_top = _avg_ranks([r["same_top"] for r in sub_rows], descending=True)
    mu = Fraction(n + 1, 2)
    dx = [a - mu for a in r_kl]
    dy = [b - mu for b in r_top]
    cov = sum((a * b for a, b in zip(dx, dy)), Fraction(0, 1))
    vx = sum((a * a for a in dx), Fraction(0, 1))
    vy = sum((b * b for b in dy), Fraction(0, 1))
    if vx <= 0 or vy <= 0:
        raise GemmaM1Error("DEGENERATE_RANKS", "constant ranks have undefined Spearman correlation")

    rho_sq = (cov * cov) / (vx * vy)
    untied_v = Fraction(n * (n * n - 1), 12)
    has_ties = any((vx != untied_v, vy != untied_v))
    rho_exact_str = format_exact_rational(cov / vx) if vx == vy else "IRRATIONAL_TIED_SQRT"
    scaled_rho = math.isqrt((rho_sq.numerator * 100_000_000) // rho_sq.denominator)
    sgn = "-" if all((cov < 0, scaled_rho > 0)) else ""
    rho_4dp = f"{sgn}{scaled_rho // 10000}.{scaled_rho % 10000:04d}"
    rho_ge_9_10 = all((cov > 0, 100 * cov * cov >= 81 * vx * vy))

    max_inv = Fraction(0, 1)
    worst_pair = ["", ""]
    for i in range(n):
        for j in range(n):
            if sub_rows[i]["kl_mean"] < sub_rows[j]["kl_mean"]:
                gap = sub_rows[j]["same_top"] - sub_rows[i]["same_top"]
                if gap > max_inv:
                    max_inv = gap
                    worst_pair = [sub_rows[i]["variant"], sub_rows[j]["variant"]]

    return {
        "spearman_rho_sq": format_exact_rational(rho_sq),
        "spearman_rho_exact": rho_exact_str,
        "spearman_rho_4dp": rho_4dp,
        "spearman_has_ties": has_ties,
        "rho_ge_9_10": rho_ge_9_10,
        "max_inversion_pp": format_exact_rational(max_inv),
        "max_inversion_3dp": format_decimal_trunc(max_inv, 3),
        "worst_inversion_pair": worst_pair,
        "inversion_le_0_2_pp": max_inv <= Fraction(1, 5),
        "p10_full_hold": all((rho_ge_9_10, max_inv <= Fraction(1, 5))),
    }


def certify_pillar1_provenance_and_prereg(
    rows: dict[tuple[str, str, str], dict[str, Any]],
    file_hashes: dict[str, str],
) -> dict[str, Any]:
    """Certify raw-receipt provenance and exact-Q PR #47 P1..P10 verdicts across 270M/1B/4B."""
    p1: dict[str, Any] = {}
    p2: dict[str, Any] = {}
    p3: dict[str, Any] = {}
    p4: dict[str, Any] = {}
    p5: dict[str, Any] = {}
    p6: dict[str, Any] = {}
    p8: dict[str, Any] = {}
    p9: dict[str, Any] = {}
    p10: dict[str, Any] = {}
    embd_diff: dict[str, Any] = {}

    wiki_retentions: list[Fraction] = []
    for model in MODELS:
        # P1: hZBG128:b560s16 < file:Q4_K_M-imat by >= 2 SE (Wiki primary, Python secondary)
        p1_w = _pair_z_sq(rows[(model, "wikitext", "hZBG128:b560s16")], rows[(model, "wikitext", "file:Q4_K_M-imat")])
        p1_p = _pair_z_sq(rows[(model, "python", "hZBG128:b560s16")], rows[(model, "python", "file:Q4_K_M-imat")])
        p1[model] = {
            "wikitext": p1_w,
            "python": p1_p,
            "verdict_wiki": "HOLD" if p1_w["favors_cand_ge_2se"] else "FAIL",
            "verdict_python": "HOLD" if p1_p["favors_cand_ge_2se"] else "FAIL",
        }

        # P2: plain ZBG128:b560s16 > file:Q4_K_M (Llama-like vs Qwen/SmolLM2-like)
        p2_w = _pair_z_sq(rows[(model, "wikitext", "file:Q4_K_M")], rows[(model, "wikitext", "ZBG128:b560s16")])
        p2_p = _pair_z_sq(rows[(model, "python", "file:Q4_K_M")], rows[(model, "python", "ZBG128:b560s16")])
        p2[model] = {
            "wikitext_q4km_beats_zbg": p2_w,
            "python_q4km_beats_zbg": p2_p,
            "verdict_wiki": "HOLD" if p2_w["favors_cand_ge_2se"] else "FAIL_REVERSED",
        }

        # P3: hZBG < ZBG on BOTH headline texts by >= 2 SE
        p3_w = _pair_z_sq(rows[(model, "wikitext", "hZBG128:b560s16")], rows[(model, "wikitext", "ZBG128:b560s16")])
        p3_p = _pair_z_sq(rows[(model, "python", "hZBG128:b560s16")], rows[(model, "python", "ZBG128:b560s16")])
        p3[model] = {
            "wikitext": p3_w,
            "python": p3_p,
            "verdict": "HOLD" if all((p3_w["favors_cand_ge_2se"], p3_p["favors_cand_ge_2se"])) else "FAIL",
        }

        # P4: hZBG+K10 < hZBG on BOTH headline texts by >= 2 SE
        p4_w = _pair_z_sq(rows[(model, "wikitext", "hZBG128:b560s16+K10")], rows[(model, "wikitext", "hZBG128:b560s16")])
        p4_p = _pair_z_sq(rows[(model, "python", "hZBG128:b560s16+K10")], rows[(model, "python", "hZBG128:b560s16")])
        p4[model] = {
            "wikitext": p4_w,
            "python": p4_p,
            "verdict": "HOLD" if all((p4_w["favors_cand_ge_2se"], p4_p["favors_cand_ge_2se"])) else "FAIL",
        }

        # P5: data-free ~rand retains >= 70% of real-text matrix KL gain over plain ball
        p5_entry: dict[str, Any] = {}
        for t in TEXTS:
            zbg_kl = rows[(model, t, "ZBG128:b560s16")]["kl_mean"]
            hzbg_kl = rows[(model, t, "hZBG128:b560s16")]["kl_mean"]
            rand_kl = rows[(model, t, "hZBG128:b560s16~rand")]["kl_mean"]
            ret = (zbg_kl - rand_kl) / (zbg_kl - hzbg_kl)
            real_rel_cut = (zbg_kl - hzbg_kl) / zbg_kl
            if t == "wikitext":
                wiki_retentions.append(ret)
            p5_entry[t] = {
                "retention_ratio": format_exact_rational(ret),
                "retention_pct_2dp": format_decimal_trunc(ret * 100, 2),
                "real_imat_rel_kl_cut_pct_2dp": format_decimal_trunc(real_rel_cut * 100, 2),
                "ge_70_pct": ret >= Fraction(7, 10),
            }
        p5_entry["verdict_wiki"] = "HOLD" if p5_entry["wikitext"]["ge_70_pct"] else "FAIL"
        p5[model] = p5_entry

        # P6: ZBG+K10 < ZBG on Wiki by >= 2 SE OR >= 5% relative KL reduction
        p6_w = _pair_z_sq(rows[(model, "wikitext", "ZBG128:b560s16+K10")], rows[(model, "wikitext", "ZBG128:b560s16")])
        rel_cut_p6 = (
            rows[(model, "wikitext", "ZBG128:b560s16")]["kl_mean"]
            - rows[(model, "wikitext", "ZBG128:b560s16+K10")]["kl_mean"]
        ) / rows[(model, "wikitext", "ZBG128:b560s16")]["kl_mean"]
        p6_holds = any((p6_w["favors_cand_ge_2se"], rel_cut_p6 >= Fraction(1, 20)))
        p6[model] = {
            "wikitext": p6_w,
            "rel_kl_reduction": format_exact_rational(rel_cut_p6),
            "rel_kl_reduction_pct_2dp": format_decimal_trunc(rel_cut_p6 * 100, 2),
            "verdict": "HOLD" if p6_holds else "FAIL",
        }

        # P8: hZBG128:b624s16 improves on hZBG128:b560s16 by >= 2 SE and holds rank #1
        p8_w = _pair_z_sq(rows[(model, "wikitext", "hZBG128:b624s16")], rows[(model, "wikitext", "hZBG128:b560s16")])
        p8_p = _pair_z_sq(rows[(model, "python", "hZBG128:b624s16")], rows[(model, "python", "hZBG128:b560s16")])
        best_w = min((rows[(model, "wikitext", v)]["kl_mean"], v) for v in sorted(EXPECTED_VARIANTS))[1]
        best_p = min((rows[(model, "python", v)]["kl_mean"], v) for v in sorted(EXPECTED_VARIANTS))[1]
        p8_ok = all(
            (
                p8_w["favors_cand_ge_2se"],
                p8_p["favors_cand_ge_2se"],
                best_w == "hZBG128:b624s16",
                best_p == "hZBG128:b624s16",
            )
        )
        p8[model] = {
            "wikitext": p8_w,
            "python": p8_p,
            "rank1_wikitext": best_w,
            "rank1_python": best_p,
            "verdict": "HOLD" if p8_ok else "FAIL",
        }

        # P9: at ~4.005 bits, KL rises for all ~4-bit arms (ZBG, hZBG, ZB128@R:16, Q4_0) AND hZBG4 beats Q4_0 by >= 2 SE
        p9_w = _pair_z_sq(rows[(model, "wikitext", "hZBG128:b496s16")], rows[(model, "wikitext", "file:Q4_0")])
        p9_p = _pair_z_sq(rows[(model, "python", "hZBG128:b496s16")], rows[(model, "python", "file:Q4_0")])
        rises_all = all(
            all(
                (
                    rows[(model, t, "ZBG128:b496s16")]["kl_mean"] > rows[(model, t, "ZBG128:b560s16")]["kl_mean"],
                    rows[(model, t, "hZBG128:b496s16")]["kl_mean"] > rows[(model, t, "hZBG128:b560s16")]["kl_mean"],
                    rows[(model, t, "ZB128@R:16")]["kl_mean"] > rows[(model, t, "ZB128@R:32")]["kl_mean"],
                    rows[(model, t, "ZB128@R:16")]["kl_mean"] > rows[(model, t, "ZBG128:b560s16")]["kl_mean"],
                    rows[(model, t, "file:Q4_0")]["kl_mean"] > rows[(model, t, "file:Q4_K_M")]["kl_mean"],
                )
            )
            for t in TEXTS
        )
        p9[model] = {
            "wikitext": p9_w,
            "python": p9_p,
            "kl_rises_at_4bit_all_arms": rises_all,
            "verdict": "HOLD" if all((rises_all, p9_w["favors_cand_ge_2se"], p9_p["favors_cand_ge_2se"])) else "FAIL",
        }

        # P10: Top-1 ranking Spearman rho >= 0.90 AND max inversion <= 0.2 pp
        p10[model] = {
            t: _exact_spearman_and_inversions([rows[(model, t, v)] for v in sorted(EXPECTED_VARIANTS)])
            for t in TEXTS
        }

        # Stock arm other_tensors_that_differ provenance check
        stock_diffs = {
            arm: rows[(model, "wikitext", arm)]["other_tensors_that_differ"]
            for arm in ("file:Q4_0", "file:Q4_K_M", "file:Q4_K_M-imat")
        }
        embd_diff[model] = {
            "stock_arms": stock_diffs,
            "all_stock_clean": all(len(v) == 0 for v in stock_diffs.values()),
            "all_stock_mutate_token_embd": all(v == ["token_embd.weight"] for v in stock_diffs.values()),
        }

    p5_wiki_monotone_decay = all(a > b for a, b in zip(wiki_retentions, wiki_retentions[1:]))
    p10_rho_holds_all_6 = all(p10[m][t]["rho_ge_9_10"] for m in MODELS for t in TEXTS)
    p10_inv_passes_count = sum(1 for m in MODELS for t in TEXTS if p10[m][t]["inversion_le_0_2_pp"])

    return {
        "pinned_jsonl_sha256": file_hashes,
        "total_rows_validated": len(rows),
        "p1_hzbg_vs_q4km_imat": p1,
        "p2_plain_zbg_vs_q4km": p2,
        "p3_hzbg_vs_zbg": p3,
        "p4_hzbg_k10_vs_hzbg": p4,
        "p5_rand_retention": p5,
        "p5_wiki_monotone_scale_decay": p5_wiki_monotone_decay,
        "p5_4b_python_collapse_below_10pct": parse_exact_rational(p5["gemma-3-4b-it"]["python"]["retention_ratio"]) < Fraction(1, 10),
        "p6_zbg_k10_vs_zbg": p6,
        "p7_e4b_bridge_status": "NOT_APPLICABLE_GEMMA3",
        "p8_5bit_improvement_and_rank1": p8,
        "p9_4bit_rise_and_beats_q4_0": p9,
        "p10_top1_ranking": p10,
        "p10_spearman_holds_all_6": p10_rho_holds_all_6,
        "p10_inversion_le_0_2_passes_of_6": p10_inv_passes_count,
        "stock_other_tensors_provenance": embd_diff,
        "bf16_4b_token_embd_asymmetry_certified": all(
            (
                embd_diff["gemma-3-270m-it"]["all_stock_clean"],
                embd_diff["gemma-3-1b-it"]["all_stock_clean"],
                embd_diff["gemma-3-4b-it"]["all_stock_mutate_token_embd"],
            )
        ),
    }


# ---------------------------------------------------------------------------
# Pillar 2: Exact Log-Convex Rate-Distortion & Two-Sided Pure-Q Iso-KL Bracket
# ---------------------------------------------------------------------------

def certify_pillar2_rate_distortion_convexity(
    rows: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Certify 12/12 convexity, pure-Q geometric accuracy, -6.02 dB/bit ladder, and 4B Jensen iso-KL bracket."""
    triples: list[dict[str, Any]] = []
    for model in MODELS:
        for text in TEXTS:
            for fam, arm_triple in (
                ("ZBG", ("ZBG128:b496s16", "ZBG128:b560s16", "ZBG128:b624s16")),
                ("hZBG", ("hZBG128:b496s16", "hZBG128:b560s16", "hZBG128:b624s16")),
            ):
                r0, r1, r2 = (rows[(model, text, a)] for a in arm_triple)
                b0, b1, b2 = r0["bits_per_weight"], r1["bits_per_weight"], r2["bits_per_weight"]
                k0, k1, k2 = r0["kl_mean"], r1["kl_mean"], r2["kl_mean"]
                lam = (b1 - b0) / (b2 - b0)
                k_lin = k0 + lam * (k2 - k0)
                lin_over = (k_lin - k1) / k1

                # Exact pure-Q enclosure of log-linear midpoint:
                # k_geo = k0 * exp(-lam * ln(k0 / k2))
                ln_lo, ln_hi = ln_bounds_exact(k0 / k2, 16)
                arg_lo_half = (lam * ln_lo) / Fraction(2, 1)
                arg_hi_half = (lam * ln_hi) / Fraction(2, 1)
                eh_lo, eh_hi = exp_neg_interval_exact(arg_lo_half, arg_hi_half, 14)
                k_geo_lo = k0 * (eh_lo * eh_lo)
                k_geo_hi = k0 * (eh_hi * eh_hi)
                geo_err_lo = (k_geo_lo - k1) / k1
                geo_err_hi = (k_geo_hi - k1) / k1
                max_geo_err = max(abs(geo_err_lo), abs(geo_err_hi))

                triples.append(
                    {
                        "model": model,
                        "text": text,
                        "family": fam,
                        "strictly_convex": k1 < k_lin,
                        "linear_overestimate_pct_2dp": format_decimal_trunc(lin_over * 100, 2),
                        "linear_overestimate_exact": format_exact_rational(lin_over),
                        "geo_err_lo_pct_2dp": format_decimal_trunc(geo_err_lo * 100, 2),
                        "geo_err_hi_pct_2dp": format_decimal_trunc(geo_err_hi * 100, 2),
                        "geo_within_2_5_pct": max_geo_err <= Fraction(25, 1000),
                        "geo_beats_linear_strictly": max_geo_err < lin_over,
                    }
                )
    wiki_geo_within_2_5_pct = all(t["geo_within_2_5_pct"] for t in triples if t["text"] == "wikitext")

    # Exact -6.02 dB/bit ZBG128 weight-error ladder across all 3 models
    ln2_lo, ln2_hi = ln_bounds_exact(Fraction(2, 1), 16)
    ln10_lo, ln10_hi = ln_bounds_exact(Fraction(10, 1), 24)
    shannon_slope_lo = Fraction(20, 1) * (ln2_lo / ln10_hi)
    shannon_slope_hi = Fraction(20, 1) * (ln2_hi / ln10_lo)
    zbg_slopes: dict[str, Any] = {}
    for model in MODELS:
        d4 = rows[(model, "wikitext", "ZBG128:b496s16")]["weight_err_db_pooled"]
        d45 = rows[(model, "wikitext", "ZBG128:b560s16")]["weight_err_db_pooled"]
        d5 = rows[(model, "wikitext", "ZBG128:b624s16")]["weight_err_db_pooled"]
        b4 = rows[(model, "wikitext", "ZBG128:b496s16")]["bits_per_weight"]
        b5 = rows[(model, "wikitext", "ZBG128:b624s16")]["bits_per_weight"]
        emp_slope = (d4 - d5) / (b5 - b4)
        zbg_slopes[model] = {
            "db_4bit": format_decimal_trunc(d4, 2),
            "db_4_5bit": format_decimal_trunc(d45, 2),
            "db_5bit": format_decimal_trunc(d5, 2),
            "empirical_db_per_bit_4dp": format_decimal_trunc(emp_slope, 4),
            "within_0_05_of_6_02": abs(emp_slope - shannon_slope_lo) < Fraction(5, 100),
        }

    # Exact Two-Sided Jensen + Pure-Q Log-Linear Iso-KL Bracket on 4B (hZBG128 matching file:Q4_K_M-imat)
    iso_4b: dict[str, Any] = {}
    for text in TEXTS:
        r0 = rows[("gemma-3-4b-it", text, "hZBG128:b496s16")]
        r1 = rows[("gemma-3-4b-it", text, "hZBG128:b560s16")]
        rt = rows[("gemma-3-4b-it", text, "file:Q4_K_M-imat")]
        b0, k0 = r0["bits_per_weight"], r0["kl_mean"]
        b1, k1 = r1["bits_per_weight"], r1["kl_mean"]
        bt, kt = rt["bits_per_weight"], rt["kl_mean"]

        b_lin = b0 + (b1 - b0) * (k0 - kt) / (k0 - k1)
        sav_lin = (bt - b_lin) / bt

        num_lo, num_hi = ln_bounds_exact(k0 / kt, 16)
        den_lo, den_hi = ln_bounds_exact(k0 / k1, 16)
        b_log_lo = _outward_floor_10(b0 + (b1 - b0) * (num_lo / den_hi), 15)
        b_log_hi = _outward_ceil_10(b0 + (b1 - b0) * (num_hi / den_lo), 15)
        sav_log_lo = (bt - b_log_hi) / bt
        sav_log_hi = (bt - b_log_lo) / bt

        # Algebraic Jensen proof check: for strictly convex decreasing K(b),
        # K_lin(b_lin) == kt => K(b_lin) <= kt => true root b_true <= b_lin,
        # and log-linear K_log(b) is strictly convex => b_log < b_lin strictly!
        iso_4b[text] = {
            "target_bpw": format_exact_rational(bt),
            "target_kl": format_exact_rational(kt),
            "b_lin_upper_bound": format_exact_rational(b_lin),
            "b_lin_4dp": format_decimal_trunc(b_lin, 4),
            "savings_lin_pct_2dp": format_decimal_trunc(sav_lin * 100, 2),
            "b_log_lo": format_exact_rational(b_log_lo),
            "b_log_hi": format_exact_rational(b_log_hi),
            "b_log_width_lt_1e_12": (b_log_hi - b_log_lo) < Fraction(1, 10 ** 12),
            "b_log_4dp": format_decimal_trunc(b_log_lo, 4),
            "savings_log_lo_pct_2dp": format_decimal_trunc(sav_log_lo * 100, 2),
            "savings_log_hi_pct_2dp": format_decimal_trunc(sav_log_hi * 100, 2),
            "jensen_strict_gap_b_log_lt_b_lin": b_log_hi < b_lin,
        }

    return {
        "triples_count": len(triples),
        "all_12_strictly_convex": all(t["strictly_convex"] for t in triples),
        "all_12_geo_beats_linear": all(t["geo_beats_linear_strictly"] for t in triples),
        "all_6_wiki_geo_within_2_5_pct": wiki_geo_within_2_5_pct,
        "triples": triples,
        "shannon_20_log10_2_lo_4dp": format_decimal_trunc(shannon_slope_lo, 4),
        "shannon_20_log10_2_hi_4dp": format_decimal_trunc(shannon_slope_hi, 4),
        "zbg_weight_err_slopes": zbg_slopes,
        "all_3_zbg_slopes_match_6_02": all(v["within_0_05_of_6_02"] for v in zbg_slopes.values()),
        "gemma3_4b_iso_kl_bracket": iso_4b,
    }


# ---------------------------------------------------------------------------
# Pillar 3: Exact Per-Kind Q4_K_M Fallback Decoder
# ---------------------------------------------------------------------------

def certify_pillar3_q4km_fallback_decoder(
    rows: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Decode exact per-kind llama.cpp Q4_K_M assignments from weight_err_db_by_kind."""
    expected_fallback = {
        "attn_v": "Q8_0",
        "attn_q": "Q5_0_or_Q5_K",
        "attn_k": "Q5_0_or_Q5_K",
        "ffn_gate": "Q5_0_or_Q5_K",
        "ffn_up": "Q5_0_or_Q5_K",
        "ffn_down": "Q6_K",
        "attn_output": "Q4_K",
    }
    expected_normal_4b = {
        "attn_v": "Q6_K",
        "ffn_down": "Q6_K",
        "attn_q": "Q4_K",
        "attn_k": "Q4_K",
        "attn_output": "Q4_K",
        "ffn_gate": "Q4_K",
        "ffn_up": "Q4_K",
    }

    decoded: dict[str, Any] = {}
    for model in MODELS:
        per_arm: dict[str, Any] = {}
        for arm in ("file:Q4_0", "file:Q4_K_M", "file:Q4_K_M-imat"):
            row = rows[(model, "wikitext", arm)]
            kinds_map = {k: classify_llama_quant_kind(row["weight_err_db_by_kind"][k]) for k in KINDS}
            per_arm[arm] = {
                "bits_per_weight_4dp": format_decimal_trunc(row["bits_per_weight"], 4),
                "decoded_by_kind": kinds_map,
                "q4_k_kind_count": sum(1 for v in kinds_map.values() if v == "Q4_K"),
                "q8_0_kind_count": sum(1 for v in kinds_map.values() if v == "Q8_0"),
                "q5_kind_count": sum(1 for v in kinds_map.values() if v == "Q5_0_or_Q5_K"),
                "q6_k_kind_count": sum(1 for v in kinds_map.values() if v == "Q6_K"),
            }
        decoded[model] = per_arm

    fallback_270m_1b_exact = all(
        decoded[m][arm]["decoded_by_kind"] == expected_fallback
        for m in ("gemma-3-270m-it", "gemma-3-1b-it")
        for arm in ("file:Q4_K_M", "file:Q4_K_M-imat")
    )
    normal_4b_exact = all(
        decoded["gemma-3-4b-it"][arm]["decoded_by_kind"] == expected_normal_4b
        for arm in ("file:Q4_K_M", "file:Q4_K_M-imat")
    )
    q4_0_control_exact = all(
        all(v == "Q4_0_or_lower" for v in decoded[m]["file:Q4_0"]["decoded_by_kind"].values())
        for m in MODELS
    )

    return {
        "decoded_models": decoded,
        "fallback_270m_1b_exact": fallback_270m_1b_exact,
        "normal_4b_zero_fallback_exact": normal_4b_exact,
        "q4_0_negative_control_exact": q4_0_control_exact,
    }


# ---------------------------------------------------------------------------
# Pillar 4: Exact Pure-Q E8-vs-H8*E8 Anisotropy Operator Theorem & PRED-M1-01
# ---------------------------------------------------------------------------

def certify_pillar4_e8_h8_anisotropy(
    rows: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Certify empirical EBG/HBG paradox, exact E8/H8 covariance theorem, and PRED-M1-01."""
    # 1. Empirical M1 Max EBG128 vs HBG128 MSE-blindness & PPL/KL divergence
    ebg_hbg_diffs: list[Fraction] = []
    zbg_ebg_gains: list[Fraction] = []
    emp_models: dict[str, Any] = {}
    for model in MODELS:
        eb = rows[(model, "wikitext", "EBG128:b560s16")]
        hb = rows[(model, "wikitext", "HBG128:b560s16")]
        zb = rows[(model, "wikitext", "ZBG128:b560s16")]
        for k in KINDS:
            ebg_hbg_diffs.append(abs(eb["weight_err_db_by_kind"][k] - hb["weight_err_db_by_kind"][k]))
            zbg_ebg_gains.append(zb["weight_err_db_by_kind"][k] - eb["weight_err_db_by_kind"][k])
        emp_models[model] = {
            "ebg_ppl_wiki": format_exact_rational(eb["ppl"]),
            "hbg_ppl_wiki": format_exact_rational(hb["ppl"]),
            "base_ppl_wiki": format_exact_rational(eb["ppl_base"]),
            "hbg_lower_ppl_wiki": hb["ppl"] < eb["ppl"],
            "kl_ebg_vs_hbg_wiki": _pair_z_sq(eb, hb),
            "kl_ebg_vs_hbg_python": _pair_z_sq(
                rows[(model, "python", "EBG128:b560s16")],
                rows[(model, "python", "HBG128:b560s16")],
            ),
        }
    max_abs_ebg_hbg_db_diff = max(ebg_hbg_diffs)
    min_zbg_minus_ebg_db = min(zbg_ebg_gains)

    # 2. Exact Pure-Z/Q H_8 & E_8 Lattice / Noise-Covariance Anisotropy Operator Theorem
    h8 = hadamard8_unnorm_exact()
    h8_sq = mat_mul_exact(h8, h8)
    h8_symmetric = h8 == mat_transpose_exact(h8)
    h8_sq_equals_8I = all(h8_sq[i][j] == (Fraction(8, 1) if i == j else Fraction(0, 1)) for i in range(8) for j in range(8))

    roots = _e8_minimal_roots_exact()
    roots_norm2_exact = all(sum((c * c for c in r), Fraction(0, 1)) == 2 for r in roots)
    roots_fixed_exact = all(nearest_e8_exact(r) == r for r in roots)

    # Non-automorphism proof: r = (1,1,0^6) in E8 => (Q_H r)_0^2 = (1/8)(H8 r)_0^2 = 1/2
    # Any x in E8 has 2*x_0 in Z => 4*x_0^2 is a perfect square integer; 4*(1/2) = 2 is NOT a square!
    r_test = [Fraction(1, 1), Fraction(1, 1)] + [Fraction(0, 1)] * 6
    h8_r = mat_vec8_exact(h8, r_test)
    four_times_qh_r0_sq = Fraction(4, 8) * (h8_r[0] * h8_r[0])  # == 2
    qh_not_automorphism_e8 = all(
        (four_times_qh_r0_sq == 2, math.isqrt(int(four_times_qh_r0_sq)) ** 2 != int(four_times_qh_r0_sq))
    )

    def _conj_qh(sigma: list[list[Fraction]]) -> list[list[Fraction]]:
        prod = mat_mul_exact(mat_mul_exact(h8, sigma), h8)
        return [[prod[i][j] / Fraction(8, 1) for j in range(8)] for i in range(8)]

    # Isotropic trace invariance on a dense generic rational symmetric 8x8 matrix
    sigma_gen = [[Fraction((i + 1) * (j + 1) + (3 if i == j else 0), 7) for j in range(8)] for i in range(8)]
    sigma_gen_rot = _conj_qh(sigma_gen)
    generic_isotropic_trace_invariance = mat_trace_exact(sigma_gen_rot) == mat_trace_exact(sigma_gen)

    # Exact E8 Voronoi error witness w1 = (1/2, 0^7): nearest_e8(w1) == 0 => e = w1, ||e||^2 = 1/4
    w1 = [Fraction(1, 2)] + [Fraction(0, 1)] * 7
    e_w1 = [a - b for a, b in zip(w1, nearest_e8_exact(w1))]
    sigma_e = [[e_w1[i] * e_w1[j] for j in range(8)] for i in range(8)]
    sigma_e_h = _conj_qh(sigma_e)

    # Rank-1 coordinate-axis Fisher H_act^(1) = e_1 e_1^T vs rank-1 DC Fisher H_act^(2) = (1/8) 1 1^T
    h_act_1 = [[Fraction(1 if i == 0 and j == 0 else 0, 1) for j in range(8)] for i in range(8)]
    h_act_2 = [[Fraction(1, 8) for _ in range(8)] for _ in range(8)]

    kl_h1_e8 = mat_trace_exact(mat_mul_exact(h_act_1, sigma_e))
    kl_h1_h8 = mat_trace_exact(mat_mul_exact(h_act_1, sigma_e_h))
    kl_h2_e8 = mat_trace_exact(mat_mul_exact(h_act_2, sigma_e))
    kl_h2_h8 = mat_trace_exact(mat_mul_exact(h_act_2, sigma_e_h))

    # Exact isometric Q_H = (1/sqrt(8)) H_8 = (1/4)*sqrt(2)*H_8 single-vector witnesses over Q(sqrt(2))!
    # Represent a + b*sqrt(2) exactly as (a, b) in Fraction x Fraction (zero floats, isometric det(Q_H)=+-1).
    def _qsqrt2_dist_sq(qh_vec_b: list[Fraction], e8_pt: list[Fraction]) -> tuple[Fraction, Fraction]:
        # || (0, b_i) - (p_i, 0) ||^2 = sum(p_i^2 + 2*b_i^2) + sqrt(2) * sum(-2*p_i*b_i)
        rat_part = sum((p * p + 2 * b * b for p, b in zip(e8_pt, qh_vec_b)), Fraction(0, 1))
        sqrt2_part = sum((-2 * p * b for p, b in zip(e8_pt, qh_vec_b)), Fraction(0, 1))
        return (rat_part, sqrt2_part)

    # Witness A: E_8 deep hole e1 = (1, 0^7) -> d^2(e1, E8) = 1; Q_H e1 = (1/4)*sqrt(2)*(1^8), nearest p1 = (1/2)^8 in E8.
    e1_hole = [Fraction(1, 1)] + [Fraction(0, 1)] * 7
    e1_e8_err = [a - b for a, b in zip(e1_hole, nearest_e8_exact(e1_hole))]
    d2_e1_e8 = sum((c * c for c in e1_e8_err), Fraction(0, 1))
    qh_e1_b = [c / Fraction(4, 1) for c in mat_vec8_exact(h8, e1_hole)]
    p1_e8 = [Fraction(1, 2)] * 8
    d2_e1_qh_a, d2_e1_qh_b = _qsqrt2_dist_sq(qh_e1_b, p1_e8)  # (3, -2) = 3 - 2*sqrt(2)
    # Certify p1 in E8, and 3 - 2*sqrt(2) < 1/5 < 1/2 = R_pack(E8)^2 (hence p1 is UNIQUE nearest E8 point!):
    # 3 - 2*sqrt(2) < 1/5 <=> 2*sqrt(2) > 14/5 <=> 2 * (-b)^2 > (a - 1/5)^2
    qh_e1_lt_1_5 = all(
        (
            nearest_e8_exact(p1_e8) == p1_e8,
            d2_e1_e8 == Fraction(1, 1),
            d2_e1_qh_b < 0,
            d2_e1_qh_a > Fraction(1, 5),
            2 * d2_e1_qh_b * d2_e1_qh_b > (d2_e1_qh_a - Fraction(1, 5)) ** 2,
        )
    )

    # Witness B: E_8 root r = e1 + e2 = (1, 1, 0^6) -> d^2(r, E8) = 0; Q_H r = (1/2)*sqrt(2)*(1,0,1,0,1,0,1,0),
    # nearest p2 = (1,0,1,0,1,0,1,0) in E8. Exact d^2(Q_H r, p2) = 6 - 4*sqrt(2) in (1/3, 1/2 = R_pack(E8)^2)!
    r_vec = [Fraction(1, 1), Fraction(1, 1)] + [Fraction(0, 1)] * 6
    r_e8_err = [a - b for a, b in zip(r_vec, nearest_e8_exact(r_vec))]
    d2_r_e8 = sum((c * c for c in r_e8_err), Fraction(0, 1))
    qh_r_b = [c / Fraction(4, 1) for c in mat_vec8_exact(h8, r_vec)]
    p2_e8 = [Fraction(1 if i % 2 == 0 else 0, 1) for i in range(8)]
    d2_r_qh_a, d2_r_qh_b = _qsqrt2_dist_sq(qh_r_b, p2_e8)  # (6, -4) = 6 - 4*sqrt(2)
    # Certify p2 in E8, 6 - 4*sqrt(2) < 1/2 (unique nearest in E8!) AND 6 - 4*sqrt(2) > 1/3:
    # < 1/2 <=> 2 * b^2 > (a - 1/2)^2; > 1/3 <=> (a - 1/3)^2 > 2 * b^2
    qh_r_gt_1_3_and_unique = all(
        (
            nearest_e8_exact(p2_e8) == p2_e8,
            d2_r_e8 == Fraction(0, 1),
            d2_r_qh_b < 0,
            d2_r_qh_a > Fraction(1, 2),
            2 * d2_r_qh_b * d2_r_qh_b > (d2_r_qh_a - Fraction(1, 2)) ** 2,
            (d2_r_qh_a - Fraction(1, 3)) ** 2 > 2 * d2_r_qh_b * d2_r_qh_b,
        )
    )

    # 3. Law-2 Steelman: PRED-M1-01 pure-Q prediction for hEBG128:b560s16 on 4B
    # Theoretical -0.65 dB factor = 10^{-13/200} = exp(-(13/200) * ln(10))
    ln10_lo, ln10_hi = ln_bounds_exact(Fraction(10, 1), 24)
    fac_lo, fac_hi = exp_neg_interval_exact(Fraction(13, 200) * ln10_lo, Fraction(13, 200) * ln10_hi, 14)
    hzbg_4b_w = rows[("gemma-3-4b-it", "wikitext", "hZBG128:b560s16")]["kl_mean"]
    hzbg_4b_p = rows[("gemma-3-4b-it", "python", "hZBG128:b560s16")]["kl_mean"]
    emp_4b_ratio_w = (
        rows[("gemma-3-4b-it", "wikitext", "EBG128:b560s16")]["kl_mean"]
        / rows[("gemma-3-4b-it", "wikitext", "ZBG128:b560s16")]["kl_mean"]
    )
    pred_hebg_wiki_theo_lo = hzbg_4b_w * fac_lo
    pred_hebg_wiki_theo_hi = hzbg_4b_w * fac_hi
    pred_hebg_py_theo_lo = hzbg_4b_p * fac_lo
    pred_hebg_py_theo_hi = hzbg_4b_p * fac_hi
    pred_hebg_wiki_emp = hzbg_4b_w * emp_4b_ratio_w

    # Matched-KL bpw shift under empirical 4B Wiki log-slope alpha = ln(k0/k1) / (b1 - b0):
    r0_4b = rows[("gemma-3-4b-it", "wikitext", "hZBG128:b496s16")]
    r1_4b = rows[("gemma-3-4b-it", "wikitext", "hZBG128:b560s16")]
    bt_4b = rows[("gemma-3-4b-it", "wikitext", "file:Q4_K_M-imat")]["bits_per_weight"]
    kt_4b = rows[("gemma-3-4b-it", "wikitext", "file:Q4_K_M-imat")]["kl_mean"]
    b0_4b, k0_4b = r0_4b["bits_per_weight"], r0_4b["kl_mean"]
    b1_4b, k1_4b = r1_4b["bits_per_weight"], r1_4b["kl_mean"]
    den_lo, den_hi = ln_bounds_exact(k0_4b / k1_4b, 16)
    num_lo, num_hi = ln_bounds_exact(k0_4b / kt_4b, 16)
    b_hzbg_lo = b0_4b + (b1_4b - b0_4b) * (num_lo / den_hi)
    b_hzbg_hi = b0_4b + (b1_4b - b0_4b) * (num_hi / den_lo)
    # Extra bit saving from -0.65 dB (ln factor = (13/200)*ln(10))
    db_e8_lo = (b1_4b - b0_4b) * (Fraction(13, 200) * ln10_lo / den_hi)
    db_e8_hi = (b1_4b - b0_4b) * (Fraction(13, 200) * ln10_hi / den_lo)
    b_hebg_lo = b_hzbg_lo - db_e8_hi
    b_hebg_hi = b_hzbg_hi - db_e8_lo
    sav_hebg_lo = (bt_4b - b_hebg_hi) / bt_4b
    sav_hebg_hi = (bt_4b - b_hebg_lo) / bt_4b

    # Empirical 4B EBG/ZBG ratio transfer bit saving:
    ln_emp_lo, ln_emp_hi = ln_bounds_exact(Fraction(1, 1) / emp_4b_ratio_w, 16)
    db_emp_lo = (b1_4b - b0_4b) * (ln_emp_lo / den_hi)
    sav_hebg_emp_lo = (bt_4b - (b_hzbg_hi - db_emp_lo)) / bt_4b

    return {
        "empirical_paradox": {
            "max_abs_ebg_vs_hbg_db_across_21_pairs": format_exact_rational(max_abs_ebg_hbg_db_diff),
            "mse_identical_within_0_002_db": max_abs_ebg_hbg_db_diff <= Fraction(2, 1000),
            "min_zbg_minus_ebg_db_across_21_pairs": format_exact_rational(min_zbg_minus_ebg_db),
            "e8_packing_gain_ge_0_65_db_all_21": min_zbg_minus_ebg_db >= Fraction(65, 100),
            "hbg_lower_ppl_wiki_all_3_models": all(emp_models[m]["hbg_lower_ppl_wiki"] for m in MODELS),
            "kl_flips_270m_hbg_to_4b_ebg": all(
                (
                    parse_exact_rational(emp_models["gemma-3-270m-it"]["kl_ebg_vs_hbg_wiki"]["gain_kl"]) < 0,
                    parse_exact_rational(emp_models["gemma-3-4b-it"]["kl_ebg_vs_hbg_wiki"]["gain_kl"]) > 0,
                )
            ),
            "by_model": emp_models,
        },
        "exact_e8_h8_operator_theorem": {
            "h8_symmetric_and_sq_8I": all((h8_symmetric, h8_sq_equals_8I)),
            "e8_minimal_roots_count": len(roots),
            "e8_all_240_roots_norm2_and_fixed": all((len(roots) == 240, roots_norm2_exact, roots_fixed_exact)),
            "qh_not_automorphism_of_e8_proved": qh_not_automorphism_e8,
            "generic_isotropic_trace_invariance_exact": generic_isotropic_trace_invariance,
            "witness_tr_sigma_e": format_exact_rational(mat_trace_exact(sigma_e)),
            "witness_tr_sigma_e_h": format_exact_rational(mat_trace_exact(sigma_e_h)),
            "witness_identical_l2_error": mat_trace_exact(sigma_e) == mat_trace_exact(sigma_e_h),
            "h_act_1_axis_kl_e8": format_exact_rational(kl_h1_e8),
            "h_act_1_axis_kl_h8": format_exact_rational(kl_h1_h8),
            "h_act_2_dc_kl_e8": format_exact_rational(kl_h2_e8),
            "h_act_2_dc_kl_h8": format_exact_rational(kl_h2_h8),
            "exact_8x_opposite_sign_kl_flip": all(
                (kl_h1_e8 == 8 * kl_h1_h8, kl_h2_h8 == 8 * kl_h2_e8, kl_h1_h8 < kl_h1_e8, kl_h2_e8 < kl_h2_h8)
            ),
            "isometric_qh_e1_d2_e8": format_exact_rational(d2_e1_e8),
            "isometric_qh_e1_d2_qh_qsqrt2": [format_exact_rational(d2_e1_qh_a), format_exact_rational(d2_e1_qh_b)],
            "isometric_qh_e1_unique_nearest_and_lt_1_5": qh_e1_lt_1_5,
            "isometric_qh_root_d2_e8": format_exact_rational(d2_r_e8),
            "isometric_qh_root_d2_qh_qsqrt2": [format_exact_rational(d2_r_qh_a), format_exact_rational(d2_r_qh_b)],
            "isometric_qh_root_unique_nearest_and_gt_1_3": qh_r_gt_1_3_and_unique,
            "isometric_qh_single_vector_opposite_flip": all((qh_e1_lt_1_5, qh_r_gt_1_3_and_unique)),
        },
        "pred_m1_01_hebg128_steelman": {
            "kl_multiplier_10_neg_0_065_lo_6dp": format_decimal_trunc(fac_lo, 6),
            "kl_multiplier_10_neg_0_065_hi_6dp": format_decimal_trunc(fac_hi, 6),
            "pred_4b_wiki_kl_theo_lo_6dp": format_decimal_trunc(pred_hebg_wiki_theo_lo, 6),
            "pred_4b_wiki_kl_theo_hi_6dp": format_decimal_trunc(pred_hebg_wiki_theo_hi, 6),
            "pred_4b_py_kl_theo_lo_6dp": format_decimal_trunc(pred_hebg_py_theo_lo, 6),
            "pred_4b_py_kl_theo_hi_6dp": format_decimal_trunc(pred_hebg_py_theo_hi, 6),
            "pred_4b_wiki_kl_emp_transfer_6dp": format_decimal_trunc(pred_hebg_wiki_emp, 6),
            "pred_4b_wiki_matched_bpw_lo_4dp": format_decimal_trunc(b_hebg_lo, 4),
            "pred_4b_wiki_matched_bpw_hi_4dp": format_decimal_trunc(b_hebg_hi, 4),
            "pred_4b_wiki_savings_theo_lo_pct_2dp": format_decimal_trunc(sav_hebg_lo * 100, 2),
            "pred_4b_wiki_savings_theo_hi_pct_2dp": format_decimal_trunc(sav_hebg_hi * 100, 2),
            "pred_4b_wiki_savings_emp_transfer_pct_2dp": format_decimal_trunc(sav_hebg_emp_lo * 100, 2),
            "pred_savings_strictly_above_12pct": all((sav_hebg_lo > Fraction(12, 100), sav_hebg_emp_lo > Fraction(13, 100))),
        },
    }


# ---------------------------------------------------------------------------
# Pillar 5: Exact Static AST Audit & Verified Patch Spec for pinned_inputs/run.py
# ---------------------------------------------------------------------------

def _extract_weighted_startswith_prefix(node: ast.AST, weighted_arms: Sequence[str]) -> str | None:
    """Return literal string prefix if AST node is `<expr>.startswith(<str>)` matching a weighted ball arm."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr != "startswith":
        return None
    if len(node.args) != 1:
        return None
    arg0 = node.args[0]
    if not isinstance(arg0, ast.Constant):
        return None
    if type(arg0.value) is not str:
        return None
    pfx = arg0.value
    if not pfx:
        return None
    if any(arm.startswith(pfx) for arm in weighted_arms):
        return pfx
    return None


def certify_pillar5_run_py_audit(
    pinned_dir: Path | None = None,
    *,
    verify_pin_hashes: bool = True,
) -> dict[str, Any]:
    """Certify static AST findings RUN-BUG-01..02 on pinned_inputs/run.py and verify patched predicates."""
    require_exact_bool(verify_pin_hashes, "verify_pin_hashes")
    base_dir = PINNED_INPUTS_DIR if pinned_dir is None else pinned_dir
    exp_sha = PINNED_FILE_SHA256["run.py"] if verify_pin_hashes else None
    raw = _read_pinned_file_bytes(base_dir, "run.py", expected_sha256=exp_sha)
    run_sha = hashlib.sha256(raw).hexdigest()
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GemmaM1Error("INVALID_UTF8", "run.py is not valid UTF-8") from exc
    try:
        tree = ast.parse(src, "run.py")
    except SyntaxError as exc:
        raise GemmaM1Error("RUN_PY_SYNTAX_ERROR", f"run.py AST parse failed: {exc}") from exc

    positive_real_imat_arms = (
        "file:Q4_K_M-imat",
        "hZBG128:b560s16",
        "hZBG128:b560s16+K10",
        "hEBG128:b560s16",
        "hHBG128:b560s16",
        "iZBG128:b560s16",
        "iEBG128:b560s16",
        "iHBG128:b560s16",
    )
    weighted_only_arms = positive_real_imat_arms[1:]

    # Verify WEIGHTED_BALL_RE supports all 6 weighted prefixes [ih] x (ZBG|EBG|HBG)
    weighted_re_literal = ""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target0 = node.targets[0]
            if all((len(node.targets) == 1, isinstance(target0, ast.Name), getattr(target0, "id", None) == "WEIGHTED_BALL_RE")):
                val = node.value
                if isinstance(val, ast.Call):
                    if val.args:
                        arg0 = val.args[0]
                        if isinstance(arg0, ast.Constant):
                            if type(arg0.value) is str:
                                weighted_re_literal = arg0.value

    # Dynamically extract RUN-BUG-01 `.startswith(<prefix>)` literals in run_sweep & RUN-BUG-02 in quantize_tensor_type_flags
    startswith_sites: list[tuple[int, str]] = []
    embd_f16_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "run_sweep":
                for sub in ast.walk(node):
                    pfx = _extract_weighted_startswith_prefix(sub, weighted_only_arms)
                    if pfx is not None:
                        startswith_sites.append((sub.lineno, pfx))
            elif node.name == "quantize_tensor_type_flags":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant):
                        if sub.value == r"^token_embd\.weight$=f16":
                            embd_f16_lines.append(sub.lineno)

    startswith_sites.sort()
    hzbg_lines = [ln for ln, _ in startswith_sites]
    ast_prefixes = sorted({pfx for _, pfx in startswith_sites})
    embd_f16_lines.sort()

    # Certify RUN-BUG-01 dynamically from the AST-extracted prefix set (zero hardcoded buggy prefix!)
    def _ast_buggy_pred(arm: str) -> bool:
        if arm == "file:Q4_K_M-imat":
            return True
        return all((any(arm.startswith(p) for p in ast_prefixes), not arm.endswith("~rand")))

    negative_non_real_arms = (
        "file:Q4_0",
        "file:Q4_K_M",
        "ZBG128:b560s16",
        "ZBG128:b560s16+K10",
        "EBG128:b560s16",
        "HBG128:b560s16",
        "hZBG128:b560s16~rand",
        "hEBG128:b560s16~rand",
        "iHBG128:b560s16~rand",
    )
    missed_by_buggy = [a for a in positive_real_imat_arms if not _ast_buggy_pred(a)]
    fixed_all_positive_true = all(needs_real_imatrix_fixed(a) for a in positive_real_imat_arms)
    fixed_all_negative_false = all(not needs_real_imatrix_fixed(a) for a in negative_non_real_arms)

    # Certify RUN-BUG-02 fixed dtype-preserving flags for BF16, F16, F32 + E4B extra tensors
    bf16_flags = quantize_tensor_type_flags_fixed(["token_embd.weight"], "BF16")
    f16_flags = quantize_tensor_type_flags_fixed(["token_embd.weight"], "F16")
    f32_flags = quantize_tensor_type_flags_fixed(["token_embd.weight"], "F32")
    e4b_flags = quantize_tensor_type_flags_fixed(
        ["token_embd.weight", "per_layer_model_proj.weight", "blk.0.inp_gate.weight", "blk.0.proj.weight"],
        "BF16",
    )

    return {
        "run_py_sha256": run_sha,
        "weighted_ball_re_supports_all_6": weighted_re_literal == r"^([ih])(ZBG|EBG|HBG)(\d+):b(\d+)s(\d+)(?:~(rand))?$",
        "run_bug_01_ast_extracted_prefixes": ast_prefixes,
        "run_bug_01_hardcoded_hzbg_lines": hzbg_lines,
        "run_bug_01_detected_exact_lines": all((hzbg_lines == [967, 1035], ast_prefixes == ["hZBG"])),
        "run_bug_01_arms_missed_by_buggy": missed_by_buggy,
        "run_bug_01_missed_count": len(missed_by_buggy),
        "run_bug_01_fixed_predicate_verified": all((fixed_all_positive_true, fixed_all_negative_false, len(missed_by_buggy) == 5)),
        "run_bug_02_hardcoded_f16_lines": embd_f16_lines,
        "run_bug_02_detected_exact_line": embd_f16_lines == [733],
        "run_bug_02_fixed_selector_verified": all(
            (
                bf16_flags == ["--tensor-type", r"^token_embd\.weight$=bf16"],
                f16_flags == ["--tensor-type", r"^token_embd\.weight$=f16"],
                f32_flags == ["--tensor-type", r"^token_embd\.weight$=f32"],
                e4b_flags
                == [
                    "--tensor-type",
                    r"^token_embd\.weight$=bf16",
                    "--tensor-type",
                    "per_layer_model_proj=f32",
                    "--tensor-type",
                    "inp_gate=f32",
                    "--tensor-type",
                    r"\.proj\.=f32",
                ],
            )
        ),
    }


# ---------------------------------------------------------------------------
# Certificate Builder & Full-Replay Validator
# ---------------------------------------------------------------------------

def build_standalone_payload(pinned_dir: Path | None = None) -> dict[str, Any]:
    rows, file_hashes = load_pinned_jsonl_dataset(pinned_dir, verify_pin_hashes=(pinned_dir is None))
    return {
        "oracle_class": "exact",
        "novelty_tier": "N1",
        "zero_trust_path_floats": True,
        "zero_rng": True,
        "zero_bare_asserts": True,
        "pinned_source_commit": "dc10d4bbca",
        "pillar1_provenance_and_prereg": certify_pillar1_provenance_and_prereg(rows, file_hashes),
        "pillar2_rate_distortion_convexity": certify_pillar2_rate_distortion_convexity(rows),
        "pillar3_q4km_fallback_decoder": certify_pillar3_q4km_fallback_decoder(rows),
        "pillar4_e8_h8_anisotropy": certify_pillar4_e8_h8_anisotropy(rows),
        "pillar5_run_py_ast_audit": certify_pillar5_run_py_audit(pinned_dir, verify_pin_hashes=(pinned_dir is None)),
    }


def build_standalone_certificate(pinned_dir: Path | None = None) -> dict[str, Any]:
    payload = build_standalone_payload(pinned_dir)
    payload_sha = canonical_sha256(payload)
    envelope = {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = canonical_sha256(envelope)
    return {
        "schema": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }


def validate_certificate_payload(cert: Any, *, full_replay: bool = True) -> dict[str, Any]:
    """Fail-closed Zone-B validation of a Track S30 certificate object."""
    require_exact_bool(full_replay, "full_replay")
    if type(cert) is not dict:
        raise GemmaM1Error("INVALID_CERT_TYPE", "certificate must be exact dict")
    validate_json_tree_iterative(cert)
    if set(cert.keys()) != _CERT_TOP_KEYS:
        raise GemmaM1Error("INVALID_CERT_KEYS", "certificate top-level keys mismatch")

    schema = require_exact_str(cert["schema"], "schema")
    if schema != SCHEMA_VERSION:
        raise GemmaM1Error("SCHEMA_MISMATCH", f"unexpected schema: {schema!r}")
    track_id = require_exact_str(cert["track_id"], "track_id")
    if track_id != TRACK_ID:
        raise GemmaM1Error("TRACK_ID_MISMATCH", f"unexpected track_id: {track_id!r}")

    payload = cert["payload"]
    if type(payload) is not dict:
        raise GemmaM1Error("INVALID_PAYLOAD_TYPE", "payload must be exact dict")
    if set(payload.keys()) != _PAYLOAD_TOP_KEYS:
        raise GemmaM1Error("INVALID_PAYLOAD_KEYS", "payload top-level keys mismatch")
    if payload["oracle_class"] != "exact" or payload["novelty_tier"] != "N1":
        raise GemmaM1Error("INVALID_PAYLOAD_META", "payload oracle_class/novelty_tier mismatch")
    if payload["pinned_source_commit"] != "dc10d4bbca":
        raise GemmaM1Error("INVALID_PAYLOAD_META", "payload pinned_source_commit mismatch")
    for flag_k in ("zero_trust_path_floats", "zero_rng", "zero_bare_asserts"):
        if require_exact_bool(payload[flag_k], flag_k) is not True:
            raise GemmaM1Error("INVALID_PAYLOAD_META", f"payload {flag_k} must be True")
    for pillar_k in (
        "pillar1_provenance_and_prereg",
        "pillar2_rate_distortion_convexity",
        "pillar3_q4km_fallback_decoder",
        "pillar4_e8_h8_anisotropy",
        "pillar5_run_py_ast_audit",
    ):
        if type(payload[pillar_k]) is not dict:
            raise GemmaM1Error("INVALID_PAYLOAD_PILLAR", f"payload {pillar_k} must be exact dict")

    claimed_payload_sha = require_sha256_hex(cert["payload_sha256"], "payload_sha256")
    claimed_cert_sha = require_sha256_hex(cert["certificate_sha256"], "certificate_sha256")

    actual_payload_sha = canonical_sha256(payload)
    if not hmac.compare_digest(actual_payload_sha, claimed_payload_sha):
        raise GemmaM1Error("PAYLOAD_SHA_MISMATCH", "payload_sha256 does not match canonical payload")

    envelope = {
        "schema": schema,
        "track_id": track_id,
        "payload_sha256": claimed_payload_sha,
        "payload": payload,
    }
    actual_cert_sha = canonical_sha256(envelope)
    if not hmac.compare_digest(actual_cert_sha, claimed_cert_sha):
        raise GemmaM1Error("CERT_SHA_MISMATCH", "certificate_sha256 does not match canonical envelope")

    if full_replay:
        expected = build_standalone_certificate()
        if not hmac.compare_digest(actual_cert_sha, expected["certificate_sha256"]):
            raise GemmaM1Error("REPLAY_MISMATCH", "certificate diverges from live exact replay")

    return {
        "valid": True,
        "schema": schema,
        "track_id": track_id,
        "payload_sha256": actual_payload_sha,
        "certificate_sha256": actual_cert_sha,
        "full_replay": full_replay,
    }
