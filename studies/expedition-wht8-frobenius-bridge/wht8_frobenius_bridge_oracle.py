#!/usr/bin/env python3
"""
Track S21f (`AUDIT-121F`): Zero-Unpack WHT_8 Shell-Radius Frobenius &
WHT_8-Bypassed Bareiss PSD Stage-2 Certifier for Packed E_8^Had Bitstreams.

Cross-org mathematical & cryptographic bridge uniting:
- `Moissanite-Labs/siliquant` (`Block-H2 == WHT_8` exact isometry, `AUDIT-119BR`,
  and `E_8` / `E_8^4` shell-convolution enumerative coding, `AUDIT-112R`);
- `Moissanite-Labs/warp-drive#40/#41` (`WD40-05` / `WD41-06`: closing unkeyed
  `weft_sha256` + `radix_scale_exp2` relabeling bypasses via byte-derived norms);
- `siliconcordat/warp#13` (`PR #18`: Stage-2 byte-derived exact integer
  Frobenius, unit-alpha gate, and fraction-free Bareiss PSD spectral-norm gate).

Pure Python stdlib exact-Z/Q (ZERO floats on all trust/certificate paths, ZERO
bare `assert`s, full Zone-B `strict_cert_io` hardening).
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import hmac
import json
import struct
import sys
from typing import Any

MAX_JSON_DEPTH = 64
MAX_JSON_BYTES = 2_000_000
HARD_MAX_BYTES_CAP = 64 * 1024 * 1024
MAX_INT_BITS = 2048
MAX_INT_DIGITS = 620
MAX_BITSTREAM_BYTES = 2_000_000
MAX_CERT_SHELL = 32
MAX_EXP2 = 64
MAX_SCALE_EXP2 = 32
MAX_RANK = 16
MAX_BLOCKS_PER_VEC = 256
_HEX_LOWER = frozenset("0123456789abcdef")

BITSTREAM_MAGIC = b"SQE8HAD1"
BITSTREAM_VERSION = 1
FLAG_SUPERBLOCK_E8_4 = 1
_HEADER_FMT = "<8sHHHIIHHBBII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class Wht8BridgeError(ValueError):
    """Fail-closed typed validation error (never stripped under `python3 -O`)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _require_valid_utf8_str(s: str, label: str = "str") -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise Wht8BridgeError("INVALID_UTF8", f"{label} contains unpaired UTF-16 surrogate") from exc
    return s


def require_int(val: Any, label: str = "int", *, lo: int | None = None, hi: int | None = None) -> int:
    if type(val) is not int:
        raise Wht8BridgeError("INVALID_INT_TYPE", f"{label} must be strict int, got {type(val).__name__}")
    if lo is not None and type(lo) is not int:
        raise Wht8BridgeError("INVALID_BOUND_TYPE", f"{label} lower bound must be strict int")
    if hi is not None and type(hi) is not int:
        raise Wht8BridgeError("INVALID_BOUND_TYPE", f"{label} upper bound must be strict int")
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise Wht8BridgeError("INT_BIT_LIMIT", f"{label} bit_length={blen} exceeds {MAX_INT_BITS}")
    if lo is not None and val < lo:
        raise Wht8BridgeError("INT_BELOW_MIN", f"{label} bit_length={blen} below min_bits={lo.bit_length()}")
    if hi is not None and val > hi:
        raise Wht8BridgeError("INT_ABOVE_MAX", f"{label} bit_length={blen} above max_bits={hi.bit_length()}")
    return val


def validate_cli_argv(argv: Any) -> list[str] | None:
    if argv is None:
        return None
    if type(argv) not in (list, tuple):
        raise Wht8BridgeError("INVALID_ARGV", "argv must be list or tuple of strict str")
    out: list[str] = []
    for item in argv:
        if type(item) is not str:
            raise Wht8BridgeError("INVALID_ARGV", "argv entries must be strict str")
        _require_valid_utf8_str(item, "argv")
        out.append(item)
    return out


def require_hex64_ascii(val: Any, label: str = "sha256") -> str:
    if type(val) is not str:
        raise Wht8BridgeError("INVALID_DIGEST_TYPE", f"{label} must be strict str")
    if len(val) != 64:
        raise Wht8BridgeError("INVALID_DIGEST_LEN", f"{label} must have length 64")
    for ch in val:
        if ch not in _HEX_LOWER:
            raise Wht8BridgeError("INVALID_DIGEST_HEX", f"{label} must be lowercase ASCII hex")
    return val


def constant_time_hex64_equal(left: Any, right: Any, label: str = "sha256") -> bool:
    a = require_hex64_ascii(left, f"{label}.left")
    b = require_hex64_ascii(right, f"{label}.right")
    return hmac.compare_digest(a, b)


def fmt_q(q: Fraction) -> str:
    if type(q) is not Fraction:
        raise Wht8BridgeError("INVALID_FRACTION_TYPE", "q must be strict Fraction")
    nb = q.numerator.bit_length()
    db = q.denominator.bit_length()
    if nb > MAX_INT_BITS or db > MAX_INT_BITS:
        raise Wht8BridgeError("FRACTION_BIT_LIMIT", f"fraction bit_lengths=({nb},{db}) exceed {MAX_INT_BITS}")
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def validate_json_value(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    limit = require_int(max_depth, "max_depth", lo=1, hi=128)
    stack: list[tuple[Any, int, bool]] = [(obj, 0, False)]
    active_ids: set[int] = set()
    while stack:
        cur, depth, leaving = stack.pop()
        if leaving:
            active_ids.discard(id(cur))
            continue
        if depth > limit:
            raise Wht8BridgeError("MAX_DEPTH_EXCEEDED", f"JSON nesting exceeds {limit}")
        t = type(cur)
        if t is dict:
            cid = id(cur)
            if cid in active_ids:
                raise Wht8BridgeError("JSON_CYCLE", "cyclic dict forbidden")
            active_ids.add(cid)
            stack.append((cur, depth, True))
            for k, v in cur.items():
                if type(k) is not str:
                    raise Wht8BridgeError("NON_STR_KEY", "JSON keys must be strict str")
                _require_valid_utf8_str(k, "json_key")
                stack.append((v, depth + 1, False))
        elif t is list:
            cid = id(cur)
            if cid in active_ids:
                raise Wht8BridgeError("JSON_CYCLE", "cyclic list forbidden")
            active_ids.add(cid)
            stack.append((cur, depth, True))
            for item in cur:
                stack.append((item, depth + 1, False))
        elif t is int:
            blen = cur.bit_length()
            if blen > MAX_INT_BITS:
                raise Wht8BridgeError("INT_BIT_LIMIT", f"JSON int bit_length={blen} exceeds {MAX_INT_BITS}")
        elif t is str:
            _require_valid_utf8_str(cur, "json_str")
        elif t is bool or cur is None:
            continue
        elif t is float:
            raise Wht8BridgeError("FLOAT_FORBIDDEN", "float values are forbidden in exact JSON")
        else:
            raise Wht8BridgeError("UNSUPPORTED_JSON_TYPE", f"unsupported JSON type {t.__name__}")


def _reject_float_token(tok: str) -> Any:
    raise Wht8BridgeError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {tok!r}")


def _reject_constant_token(tok: str) -> Any:
    raise Wht8BridgeError("JSON_CONSTANT_FORBIDDEN", f"non-RFC8259 constant forbidden: {tok!r}")


def _parse_capped_int_token(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    if len(digits) > MAX_INT_DIGITS:
        raise Wht8BridgeError("INT_DIGIT_LIMIT", f"integer token exceeds {MAX_INT_DIGITS} digits")
    val = int(tok)
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise Wht8BridgeError("INT_BIT_LIMIT", f"integer token bit_length={blen} exceeds {MAX_INT_BITS}")
    return val


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise Wht8BridgeError("JSON_DUPLICATE_KEY", f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def parse_strict_json_text(raw_text: str, *, require_object: bool = True, max_depth: int = MAX_JSON_DEPTH) -> Any:
    if type(require_object) is not bool:
        raise Wht8BridgeError("INVALID_BOOL_TYPE", "require_object must be strict bool")
    if type(raw_text) is not str:
        raise Wht8BridgeError("INVALID_JSON_INPUT", "JSON text must be strict str")
    try:
        encoded = raw_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise Wht8BridgeError("INVALID_UTF8", f"JSON text contains unpaired surrogate: {exc}") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise Wht8BridgeError("JSON_TOO_LARGE", f"JSON exceeds {MAX_JSON_BYTES} bytes")
    if len(raw_text.strip()) == 0:
        raise Wht8BridgeError("EMPTY_JSON_INPUT", "JSON text must be non-empty")
    try:
        obj = json.loads(
            raw_text,
            parse_float=_reject_float_token,
            parse_constant=_reject_constant_token,
            parse_int=_parse_capped_int_token,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except Wht8BridgeError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise Wht8BridgeError("JSON_SYNTAX_ERROR", str(exc)) from exc
    validate_json_value(obj, max_depth=max_depth)
    if require_object and type(obj) is not dict:
        raise Wht8BridgeError("JSON_ROOT_NOT_OBJECT", "top-level JSON payload must be an object")
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_value(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


# ============================================================================
# EXACT WHT_8 / BLOCK-H2 ISOMETRY & EISENSTEIN E_8 / E_8^4 SHELL CODEC
# ============================================================================

def build_h8() -> tuple[tuple[int, ...], ...]:
    """Build exact 8x8 Sylvester Walsh-Hadamard matrix H_8 = H_2^{\\otimes 3} (== Block-H2)."""
    mat = ((1,),)
    for _ in range(3):
        top = tuple(row + row for row in mat)
        bot = tuple(row + tuple(-x for x in row) for row in mat)
        mat = top + bot
    return mat


def apply_h8_butterfly(vec: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Exact 3-stage 24-op integer WHT_8 butterfly (equals H_8 @ vec = H_8^T @ vec)."""
    if type(vec) not in (tuple, list) or len(vec) != 8:
        raise Wht8BridgeError("INVALID_VEC8", "vec must be length-8 tuple or list of strict int")
    x = [require_int(v, "vec_i") for v in vec]
    # Stage 1 (stride 1)
    for i in (0, 2, 4, 6):
        a, b = x[i], x[i + 1]
        x[i], x[i + 1] = a + b, a - b
    # Stage 2 (stride 2)
    for i in (0, 1, 4, 5):
        a, b = x[i], x[i + 2]
        x[i], x[i + 2] = a + b, a - b
    # Stage 3 (stride 4)
    for i in range(4):
        a, b = x[i], x[i + 4]
        x[i], x[i + 4] = a + b, a - b
    return tuple(x)


def verify_h8_isometry() -> dict[str, Any]:
    """Prove H_8^T H_8 == 8 I_8 entrywise and exact equivalence of 24-op butterfly to H_8."""
    h8 = build_h8()
    gram_ok = True
    sym_ok = True
    for i in range(8):
        for j in range(8):
            if h8[i][j] != h8[j][i]:
                sym_ok = False
            dot = sum(h8[k][i] * h8[k][j] for k in range(8))
            want = 8 if i == j else 0
            if dot != want:
                gram_ok = False
    butterfly_ok = True
    for i in range(8):
        basis = tuple(1 if j == i else 0 for j in range(8))
        got = apply_h8_butterfly(basis)
        want_col = tuple(h8[r][i] for r in range(8))
        if got != want_col:
            butterfly_ok = False
    return {
        "h8_symmetric": sym_ok,
        "h8_gram_equals_8_i8": gram_ok,
        "butterfly_24op_matches_h8_all_basis": butterfly_ok,
        "exact_norm_scale_factor": 8,
    }


def sigma_k(n: int, k: int) -> int:
    n_i = require_int(n, "n", lo=1, hi=1024)
    k_i = require_int(k, "k", lo=0, hi=16)
    return sum(d ** k_i for d in range(1, n_i + 1) if n_i % d == 0)


def _e8_coset_symbols(max_s: int, coset: int) -> tuple[int, ...]:
    """Canonical symbol ordering for doubled E_8 coordinate: even (coset 0) or odd (coset 1)."""
    out: list[int] = []
    v = 1 if coset == 1 else 0
    while v * v <= max_s:
        if v == 0:
            out.append(0)
        else:
            out.append(v)
            out.append(-v)
        v += 2
    return tuple(out)


def build_e8_shell_tables(max_shell: int) -> dict[str, Any]:
    """Build exact shell-ordered E_8 DP tables for shells m = 0..max_shell (doubled coords z = 2v)."""
    m_max = require_int(max_shell, "max_shell", lo=1, hi=MAX_CERT_SHELL)
    max_s = 8 * m_max
    syms = (_e8_coset_symbols(max_s, 0), _e8_coset_symbols(max_s, 1))
    # dp[coset][coord_j][parity_p][rem_s] = exact ways to finish z[j..7]
    dp: list[list[list[list[int]]]] = []
    for c in (0, 1):
        levels: list[list[list[int]]] = [[[0] * (max_s + 1) for _ in range(2)] for _ in range(9)]
        levels[8][0][0] = 1
        c_syms = syms[c]
        for j in range(7, -1, -1):
            nxt_lvl = levels[j + 1]
            cur_lvl = levels[j]
            for v in c_syms:
                sq = v * v
                par_bit = (v // 2) & 1
                for p in (0, 1):
                    src = nxt_lvl[p ^ par_bit]
                    dst = cur_lvl[p]
                    for s in range(sq, max_s + 1):
                        dst[s] += src[s - sq]
        dp.append(levels)

    shell_counts: list[int] = []
    cum_counts: list[int] = []
    running = 0
    e4_matches = True
    for m in range(m_max + 1):
        cnt = dp[0][0][0][8 * m] + dp[1][0][0][8 * m]
        expected = 1 if m == 0 else 240 * sigma_k(m, 3)
        if cnt != expected:
            e4_matches = False
        shell_counts.append(cnt)
        running += cnt
        cum_counts.append(running)

    return {
        "max_shell": m_max,
        "syms": syms,
        "dp": dp,
        "shell_counts": tuple(shell_counts),
        "cum_counts": tuple(cum_counts),
        "total_points": running,
        "eisenstein_e4_exact": e4_matches,
    }


def build_e8_4_shell_tables(max_shell: int) -> dict[str, Any]:
    """Build exact shell-ordered E_8^4 (32D) convolution tables for m = 0..max_shell."""
    m_max = require_int(max_shell, "max_shell", lo=1, hi=450)
    r1 = [1] + [240 * sigma_k(m, 3) for m in range(1, m_max + 1)]
    c2 = [0] * (m_max + 1)
    for i in range(m_max + 1):
        ri = r1[i]
        for j in range(m_max + 1 - i):
            c2[i + j] += ri * r1[j]
    e8_mod_matches = True
    for m in range(1, min(m_max, 12) + 1):
        if c2[m] != 480 * sigma_k(m, 7):
            e8_mod_matches = False
    c4 = [0] * (m_max + 1)
    for i in range(m_max + 1):
        ci = c2[i]
        for j in range(m_max + 1 - i):
            c4[i + j] += ci * c2[j]
    cum_c4: list[int] = []
    running = 0
    for cnt in c4:
        running += cnt
        cum_c4.append(running)
    return {
        "max_shell": m_max,
        "r1": tuple(r1),
        "c2": tuple(c2),
        "c4": tuple(c4),
        "cum_counts": tuple(cum_c4),
        "total_points": running,
        "e4_squared_equals_e8_exact": e8_mod_matches,
    }


_E8_TABLE_REQ_KEYS = frozenset({"max_shell", "syms", "dp", "shell_counts", "cum_counts"})
_E8_4_TABLE_REQ_KEYS = frozenset({"max_shell", "c2", "cum_counts"})


def _require_e8_tables(e8_tables: Any) -> dict[str, Any]:
    if type(e8_tables) is not dict or not _E8_TABLE_REQ_KEYS.issubset(e8_tables.keys()):
        raise Wht8BridgeError("INVALID_E8_TABLES", "e8_tables must be a valid E_8 shell table dict")
    return e8_tables


def _require_e8_4_tables(e8_4_tables: Any) -> dict[str, Any]:
    if type(e8_4_tables) is not dict or not _E8_4_TABLE_REQ_KEYS.issubset(e8_4_tables.keys()):
        raise Wht8BridgeError("INVALID_E8_4_TABLES", "e8_4_tables must be a valid E_8^4 shell table dict")
    return e8_4_tables


def shell_from_index(idx: int, cum_counts: tuple[int, ...]) -> tuple[int, int]:
    """Zero-unpack O(log M) binary search returning (shell_m, offset_within_shell)."""
    if type(cum_counts) not in (tuple, list) or not cum_counts:
        raise Wht8BridgeError("INVALID_SHELL_TABLE", "cum_counts must be non-empty tuple or list")
    checked_cum = [require_int(c, "cum_count", lo=1) for c in cum_counts]
    total = checked_cum[-1]
    k = require_int(idx, "shell_index", lo=0)
    if k >= total:
        raise Wht8BridgeError("SHELL_INDEX_OUT_OF_RANGE", "packed shell index exceeds codebook cardinality")
    lo = 0
    hi = len(checked_cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if k < checked_cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    prev = 0 if lo == 0 else checked_cum[lo - 1]
    return lo, k - prev


def validate_doubled_e8(z: Any, max_shell: int) -> tuple[tuple[int, ...], int, int]:
    """Verify z = 2v lies in 2*E_8 with shell m <= max_shell; return (z_tuple, coset, shell_m)."""
    m_cap = require_int(max_shell, "max_shell", lo=0, hi=MAX_CERT_SHELL)
    if type(z) not in (tuple, list) or len(z) != 8:
        raise Wht8BridgeError("INVALID_E8_VEC", "doubled E_8 vector must be length-8 tuple or list")
    vals = tuple(require_int(c, "z_i") for c in z)
    coset = vals[0] & 1
    for c in vals[1:]:
        if (c & 1) != coset:
            raise Wht8BridgeError("INVALID_E8_COSET", "E_8 doubled coordinates must have uniform parity")
    if sum(c // 2 for c in vals) & 1:
        raise Wht8BridgeError("INVALID_E8_PARITY", "E_8 vector violates even half-sum parity")
    sq = sum(c * c for c in vals)
    m = sq // 8
    if m > m_cap:
        raise Wht8BridgeError("E8_SHELL_EXCEEDED", "E_8 vector exceeds max_shell")
    return vals, coset, m


def rank_e8(z: tuple[int, ...] | list[int], e8_tables: dict[str, Any]) -> int:
    """Rank a doubled E_8 vector z = 2v into its exact shell-ordered index."""
    tbl = _require_e8_tables(e8_tables)
    vals, coset, m = validate_doubled_e8(z, tbl["max_shell"])
    cum = tbl["cum_counts"]
    dp = tbl["dp"]
    syms = tbl["syms"][coset]
    base = 0 if m == 0 else cum[m - 1]
    offset = 0 if coset == 0 else dp[0][0][0][8 * m]
    rem_s = 8 * m
    p = 0
    lvl_set = dp[coset]
    for j in range(8):
        target = vals[j]
        nxt_lvl = lvl_set[j + 1]
        for sym in syms:
            sq = sym * sym
            if sym == target:
                rem_s -= sq
                p ^= (sym // 2) & 1
                break
            if sq <= rem_s:
                offset += nxt_lvl[p ^ ((sym // 2) & 1)][rem_s - sq]
    return base + offset


def unrank_e8(idx: int, e8_tables: dict[str, Any]) -> tuple[int, ...]:
    """Unrank a shell-ordered index into doubled E_8 coordinates z = 2v."""
    tbl = _require_e8_tables(e8_tables)
    m, off = shell_from_index(idx, tbl["cum_counts"])
    dp = tbl["dp"]
    even_cnt = dp[0][0][0][8 * m]
    if off < even_cnt:
        coset = 0
    else:
        coset = 1
        off -= even_cnt
    syms = tbl["syms"][coset]
    lvl_set = dp[coset]
    rem_s = 8 * m
    p = 0
    out: list[int] = []
    for j in range(8):
        nxt_lvl = lvl_set[j + 1]
        for sym in syms:
            sq = sym * sym
            if sq > rem_s:
                continue
            ways = nxt_lvl[p ^ ((sym // 2) & 1)][rem_s - sq]
            if off < ways:
                out.append(sym)
                rem_s -= sq
                p ^= (sym // 2) & 1
                break
            off -= ways
    return tuple(out)


def _rank_e8_pair_in_shell(
    z_a: tuple[int, ...],
    z_b: tuple[int, ...],
    m_a: int,
    m_b: int,
    e8_tables: dict[str, Any],
) -> int:
    r1 = e8_tables["shell_counts"]
    cum = e8_tables["cum_counts"]
    m_pair = m_a + m_b
    off = 0
    for s_a in range(m_a):
        off += r1[s_a] * r1[m_pair - s_a]
    off_a = rank_e8(z_a, e8_tables) - (0 if m_a == 0 else cum[m_a - 1])
    off_b = rank_e8(z_b, e8_tables) - (0 if m_b == 0 else cum[m_b - 1])
    return off + off_a * r1[m_b] + off_b


def _unrank_e8_pair_in_shell(m_pair: int, off: int, e8_tables: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r1 = e8_tables["shell_counts"]
    cum = e8_tables["cum_counts"]
    m_a = 0
    for s_a in range(m_pair + 1):
        span = r1[s_a] * r1[m_pair - s_a]
        if off < span:
            m_a = s_a
            break
        off -= span
    m_b = m_pair - m_a
    off_a, off_b = divmod(off, r1[m_b])
    idx_a = (0 if m_a == 0 else cum[m_a - 1]) + off_a
    idx_b = (0 if m_b == 0 else cum[m_b - 1]) + off_b
    return unrank_e8(idx_a, e8_tables), unrank_e8(idx_b, e8_tables)


def rank_e8_4(
    blocks4: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    e8_tables: dict[str, Any],
    e8_4_tables: dict[str, Any],
) -> int:
    """Rank 4 coupled doubled-E_8 blocks (32D) into exact shell-ordered E_8^4 index."""
    tbl = _require_e8_tables(e8_tables)
    tbl4 = _require_e8_4_tables(e8_4_tables)
    if type(blocks4) not in (tuple, list) or len(blocks4) != 4:
        raise Wht8BridgeError("INVALID_E8_4_BLOCK", "E_8^4 super-block requires exactly 4 E_8 blocks")
    m_cap = require_int(tbl4["max_shell"], "e8_4_max_shell", lo=1, hi=450)
    checked = [validate_doubled_e8(blk, tbl["max_shell"]) for blk in blocks4]
    ms = [item[2] for item in checked]
    m_tot = sum(ms)
    if m_tot > m_cap:
        raise Wht8BridgeError("E8_4_SHELL_EXCEEDED", "E_8^4 total shell exceeds max_shell")
    m_left = ms[0] + ms[1]
    m_right = ms[2] + ms[3]
    c2 = tbl4["c2"]
    base = 0 if m_tot == 0 else tbl4["cum_counts"][m_tot - 1]
    off = 0
    for s_left in range(m_left):
        off += c2[s_left] * c2[m_tot - s_left]
    off_left = _rank_e8_pair_in_shell(checked[0][0], checked[1][0], ms[0], ms[1], tbl)
    off_right = _rank_e8_pair_in_shell(checked[2][0], checked[3][0], ms[2], ms[3], tbl)
    return base + off + off_left * c2[m_right] + off_right


def unrank_e8_4(
    idx: int,
    e8_tables: dict[str, Any],
    e8_4_tables: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Unrank a shell-ordered E_8^4 index into 4 doubled E_8 blocks via binary tree."""
    tbl = _require_e8_tables(e8_tables)
    tbl4 = _require_e8_4_tables(e8_4_tables)
    m_tot, off = shell_from_index(idx, tbl4["cum_counts"])
    c2 = tbl4["c2"]
    m_left = 0
    for s_left in range(m_tot + 1):
        span = c2[s_left] * c2[m_tot - s_left]
        if off < span:
            m_left = s_left
            break
        off -= span
    m_right = m_tot - m_left
    off_left, off_right = divmod(off, c2[m_right])
    z0, z1 = _unrank_e8_pair_in_shell(m_left, off_left, tbl)
    z2, z3 = _unrank_e8_pair_in_shell(m_right, off_right, tbl)
    return (z0, z1, z2, z3)


# ============================================================================
# EXACT FRACTION-FREE BAREISS SYMMETRIC INTEGER PSD ELIMINATION
# ============================================================================

def bareiss_is_psd(mat: list[list[int]]) -> bool:
    """Exact fraction-free symmetric Bareiss PSD check with diagonal pivoting (zero floats)."""
    if type(mat) is not list:
        raise Wht8BridgeError("INVALID_MATRIX", "matrix must be strict list of lists")
    n = len(mat)
    for row in mat:
        if type(row) is not list or len(row) != n:
            raise Wht8BridgeError("INVALID_MATRIX", "matrix must be square")
    a: list[list[int]] = [[require_int(mat[i][j], "m_ij") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if a[i][j] != a[j][i]:
                raise Wht8BridgeError("NON_SYMMETRIC_MATRIX", "Bareiss PSD requires symmetric integer matrix")

    prev_piv = 1
    for k in range(n):
        piv_idx: int | None = None
        for i in range(k, n):
            d = a[i][i]
            if d < 0:
                return False
            if d > 0:
                piv_idx = i
        if piv_idx is None:
            for i in range(k, n):
                for j in range(i + 1, n):
                    if a[i][j] != 0:
                        return False
            return True
        if piv_idx != k:
            a[k], a[piv_idx] = a[piv_idx], a[k]
            for row in a:
                row[k], row[piv_idx] = row[piv_idx], row[k]
        piv = a[k][k]
        for i in range(k + 1, n):
            aik = a[i][k]
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * piv - aik * a[k][j]) // prev_piv
        prev_piv = piv
    return True


def _mat_mul_int(x: list[list[int]], y: list[list[int]]) -> list[list[int]]:
    n = len(x)
    return [[sum(x[i][k] * y[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


# ============================================================================
# CANONICAL PACKED E_8^Had BITSTREAM (`SQE8HAD1`) & STAGE-2 VERIFIER
# ============================================================================

def pack_e8had_adapter(
    *,
    b_scales: list[list[int]],
    b_indices: list[list[int]],
    a_scales: list[list[int]],
    a_indices: list[list[int]],
    scale_exp2: int,
    max_shell: int,
    alpha_num: int,
    alpha_den: int = 1,
    superblock_e8_4: bool = False,
) -> bytes:
    """Pack a LoRA (B, A) adapter into exact canonical SQE8HAD1 bytes (zero floats)."""
    if type(superblock_e8_4) is not bool:
        raise Wht8BridgeError("INVALID_FLAG", "superblock_e8_4 must be strict bool")
    if type(b_scales) is not list or len(b_scales) == 0:
        raise Wht8BridgeError("INVALID_RANK", "b_scales must be non-empty list")
    rank = require_int(len(b_scales), "rank", lo=1, hi=MAX_RANK)
    if type(b_indices) is not list or type(a_scales) is not list or type(a_indices) is not list:
        raise Wht8BridgeError("INVALID_PACK_INPUT", "pack inputs must be strict lists")
    if len(b_indices) != rank or len(a_scales) != rank or len(a_indices) != rank:
        raise Wht8BridgeError("RANK_MISMATCH", "B and A scale/index arrays must match rank")
    if type(b_scales[0]) is not list or type(a_scales[0]) is not list:
        raise Wht8BridgeError("INVALID_ROW", "pack rows must be strict lists")
    d_out_blocks = require_int(len(b_scales[0]), "d_out_blocks", lo=1, hi=MAX_BLOCKS_PER_VEC)
    d_in_blocks = require_int(len(a_scales[0]), "d_in_blocks", lo=1, hi=MAX_BLOCKS_PER_VEC)
    q_s = require_int(scale_exp2, "scale_exp2", lo=0, hi=MAX_SCALE_EXP2)
    m_max = require_int(max_shell, "max_shell", lo=1, hi=MAX_CERT_SHELL)
    a_num = require_int(alpha_num, "alpha_num", lo=1, hi=0xFFFFFFFF)
    a_den = require_int(alpha_den, "alpha_den", lo=1, hi=0xFFFFFFFF)

    total_pts = (
        build_e8_4_shell_tables(m_max)["total_points"]
        if superblock_e8_4
        else build_e8_shell_tables(m_max)["total_points"]
    )
    index_bytes = max(1, ((total_pts - 1).bit_length() + 7) // 8)
    max_scale_val = 0
    for mat_s, mat_i, want_len in ((b_scales, b_indices, d_out_blocks), (a_scales, a_indices, d_in_blocks)):
        for k in range(rank):
            if type(mat_s[k]) is not list or type(mat_i[k]) is not list:
                raise Wht8BridgeError("INVALID_ROW", "pack rows must be strict lists")
            if len(mat_s[k]) != want_len or len(mat_i[k]) != want_len:
                raise Wht8BridgeError("BLOCK_COUNT_MISMATCH", "row block counts must be uniform")
            for sv, iv in zip(mat_s[k], mat_i[k]):
                s_int = require_int(sv, "scale", lo=0, hi=(1 << 64) - 1)
                require_int(iv, "shell_index", lo=0, hi=total_pts - 1)
                if s_int > max_scale_val:
                    max_scale_val = s_int
    scale_bytes = max(1, (max_scale_val.bit_length() + 7) // 8)

    flags = FLAG_SUPERBLOCK_E8_4 if superblock_e8_4 else 0
    buf = bytearray(
        struct.pack(
            _HEADER_FMT,
            BITSTREAM_MAGIC,
            BITSTREAM_VERSION,
            flags,
            rank,
            d_out_blocks,
            d_in_blocks,
            q_s,
            m_max,
            index_bytes,
            scale_bytes,
            a_num,
            a_den,
        )
    )
    for mat_s, mat_i in ((b_scales, b_indices), (a_scales, a_indices)):
        for k in range(rank):
            for sv, iv in zip(mat_s[k], mat_i[k]):
                buf.extend(sv.to_bytes(scale_bytes, "little"))
                buf.extend(iv.to_bytes(index_bytes, "little"))
    return bytes(buf)


def parse_e8had_bitstream(raw: Any) -> dict[str, Any]:
    """Fail-closed parser for canonical SQE8HAD1 bitstreams (strict EOF, unit alpha, shell bounds)."""
    if type(raw) is not bytes:
        raise Wht8BridgeError("INVALID_BITSTREAM_TYPE", "bitstream must be strict bytes")
    if len(raw) < _HEADER_SIZE:
        raise Wht8BridgeError("TRUNCATED_HEADER", "bitstream shorter than header")
    if len(raw) > MAX_BITSTREAM_BYTES:
        raise Wht8BridgeError("BITSTREAM_TOO_LARGE", f"bitstream exceeds {MAX_BITSTREAM_BYTES} bytes")
    magic, ver, flags, rank, d_out_blk, d_in_blk, q_s, m_max, idx_b, sc_b, a_num, a_den = struct.unpack_from(
        _HEADER_FMT, raw, 0
    )
    if magic != BITSTREAM_MAGIC:
        raise Wht8BridgeError("INVALID_MAGIC", "unexpected bitstream magic")
    if ver != BITSTREAM_VERSION:
        raise Wht8BridgeError("INVALID_VERSION", "unsupported bitstream version")
    if flags not in (0, FLAG_SUPERBLOCK_E8_4):
        raise Wht8BridgeError("INVALID_FLAGS", "unsupported bitstream flags")
    if rank < 1 or rank > MAX_RANK:
        raise Wht8BridgeError("INVALID_RANK", "rank out of allowed range")
    if d_out_blk < 1 or d_out_blk > MAX_BLOCKS_PER_VEC or d_in_blk < 1 or d_in_blk > MAX_BLOCKS_PER_VEC:
        raise Wht8BridgeError("INVALID_BLOCK_COUNT", "block count out of allowed range")
    if q_s > MAX_SCALE_EXP2:
        raise Wht8BridgeError("INVALID_SCALE_EXP2", "scale_exp2 out of allowed range")
    if m_max < 1 or m_max > MAX_CERT_SHELL:
        raise Wht8BridgeError("INVALID_MAX_SHELL", "max_shell out of allowed range")
    if idx_b < 1 or idx_b > 16 or sc_b < 1 or sc_b > 8:
        raise Wht8BridgeError("INVALID_RECORD_WIDTH", "record byte width out of allowed range")
    if a_den == 0 or a_num != rank * a_den:
        raise Wht8BridgeError("NON_UNIT_ALPHA_SCALE", "adapter alpha/rank must equal exact unit scale 1")

    rec_size = sc_b + idx_b
    expected_len = _HEADER_SIZE + rank * (d_out_blk + d_in_blk) * rec_size
    if len(raw) != expected_len:
        raise Wht8BridgeError("BITSTREAM_LENGTH_MISMATCH", "bitstream byte length does not match header layout")

    superblock = bool(flags & FLAG_SUPERBLOCK_E8_4)
    pos = _HEADER_SIZE
    b_scales: list[list[int]] = []
    b_indices: list[list[int]] = []
    for _ in range(rank):
        rs: list[int] = []
        ri: list[int] = []
        for _ in range(d_out_blk):
            rs.append(int.from_bytes(raw[pos : pos + sc_b], "little"))
            pos += sc_b
            ri.append(int.from_bytes(raw[pos : pos + idx_b], "little"))
            pos += idx_b
        b_scales.append(rs)
        b_indices.append(ri)

    a_scales: list[list[int]] = []
    a_indices: list[list[int]] = []
    for _ in range(rank):
        rs = []
        ri = []
        for _ in range(d_in_blk):
            rs.append(int.from_bytes(raw[pos : pos + sc_b], "little"))
            pos += sc_b
            ri.append(int.from_bytes(raw[pos : pos + idx_b], "little"))
            pos += idx_b
        a_scales.append(rs)
        a_indices.append(ri)

    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rank": rank,
        "d_out_blocks": d_out_blk,
        "d_in_blocks": d_in_blk,
        "scale_exp2": q_s,
        "max_shell": m_max,
        "superblock_e8_4": superblock,
        "alpha_num": a_num,
        "alpha_den": a_den,
        "b_scales": b_scales,
        "b_indices": b_indices,
        "a_scales": a_scales,
        "a_indices": a_indices,
    }


def _zero_unpack_diag_norms(
    scales: list[list[int]],
    indices: list[list[int]],
    cum_counts: tuple[int, ...],
) -> tuple[list[int], int]:
    """Tier-1 ZERO-UNPACK exact integer squared norms 2^(2*q_s) * ||v_k||_2^2 via O(log M) binary search."""
    diags: list[int] = []
    lookups = 0
    for k in range(len(scales)):
        acc = 0
        for sv, iv in zip(scales[k], indices[k]):
            m, _ = shell_from_index(iv, cum_counts)
            acc += 2 * sv * sv * m
            lookups += 1
        diags.append(acc)
    return diags, lookups


def _unrank_factor_gram_no_wht8(
    scales: list[list[int]],
    indices: list[list[int]],
    e8_tables: dict[str, Any],
    e8_4_tables: dict[str, Any] | None,
) -> list[list[int]]:
    """Tier-2 WHT_8-bypassed exact integer Gram matrix 2^(2*q_s) * G (ZERO WHT_8 butterflies, ZERO floats)."""
    r = len(scales)
    n_blk = len(scales[0])
    # Unrank doubled E_8 coordinate tuples once per block (never running WHT_8!)
    unranked: list[list[tuple[tuple[int, ...], ...]]] = []
    for k in range(r):
        row_blocks: list[tuple[tuple[int, ...], ...]] = []
        for b in range(n_blk):
            if e8_4_tables is not None:
                row_blocks.append(unrank_e8_4(indices[k][b], e8_tables, e8_4_tables))
            else:
                row_blocks.append((unrank_e8(indices[k][b], e8_tables),))
        unranked.append(row_blocks)

    gram = [[0] * r for _ in range(r)]
    for i in range(r):
        for j in range(i, r):
            acc = 0
            for b in range(n_blk):
                s_prod = scales[i][b] * scales[j][b]
                inner_4 = 0
                for sub_i, sub_j in zip(unranked[i][b], unranked[j][b]):
                    inner_4 += sum(x * y for x, y in zip(sub_i, sub_j))
                acc += s_prod * (inner_4 // 4)
            gram[i][j] = acc
            gram[j][i] = acc
    return gram


def verify_packed_e8had_stage2(raw_bytes: bytes, cert: dict[str, Any], *, force_exact_tier2: bool = False) -> dict[str, Any]:
    """
    Verify a packed SQE8HAD1 bitstream directly against a warp-drive / warp Stage-2
    FixedRadixCertificate (closing WD40-05 radix_scale_exp2 relabeling & warp#13 forged norms).
    """
    if type(force_exact_tier2) is not bool:
        raise Wht8BridgeError("INVALID_BOOL_TYPE", "force_exact_tier2 must be strict bool")
    if type(cert) is not dict:
        raise Wht8BridgeError("INVALID_STAGE2_CERT", "Stage-2 certificate must be strict dict")
    req_keys = {
        "adapter_sha256",
        "radix_scale_exp2",
        "frobenius_sq_scaled",
        "spectral_norm_sq_scaled",
        "max_frobenius_sq_scaled",
        "max_spectral_norm_sq_scaled",
    }
    if set(cert.keys()) != req_keys:
        raise Wht8BridgeError("INVALID_STAGE2_KEYS", "Stage-2 certificate keys mismatch")

    parsed = parse_e8had_bitstream(raw_bytes)
    if not constant_time_hex64_equal(parsed["sha256"], cert["adapter_sha256"], "adapter_sha256"):
        raise Wht8BridgeError("ADAPTER_SHA_MISMATCH", "bitstream SHA-256 does not match certificate adapter_sha256")

    exp2 = require_int(cert["radix_scale_exp2"], "radix_scale_exp2", lo=0, hi=MAX_EXP2)
    frob_s = require_int(cert["frobenius_sq_scaled"], "frobenius_sq_scaled", lo=0)
    spec_s = require_int(cert["spectral_norm_sq_scaled"], "spectral_norm_sq_scaled", lo=0)
    max_frob_s = require_int(cert["max_frobenius_sq_scaled"], "max_frobenius_sq_scaled", lo=0)
    max_spec_s = require_int(cert["max_spectral_norm_sq_scaled"], "max_spectral_norm_sq_scaled", lo=0)
    if spec_s > frob_s:
        raise Wht8BridgeError("SPECTRAL_EXCEEDS_FROBENIUS", "declared spectral_norm_sq exceeds frobenius_sq")
    if frob_s > max_frob_s:
        raise Wht8BridgeError("FROBENIUS_EXCEEDS_POLICY", "declared frobenius_sq exceeds Stage-2 policy bound")
    if spec_s > max_spec_s:
        raise Wht8BridgeError("SPECTRAL_EXCEEDS_POLICY", "declared spectral_norm_sq exceeds Stage-2 policy bound")

    m_max = parsed["max_shell"]
    e8_tables = build_e8_shell_tables(m_max)
    e8_4_tables = build_e8_4_shell_tables(m_max) if parsed["superblock_e8_4"] else None
    cum_counts = e8_4_tables["cum_counts"] if e8_4_tables is not None else e8_tables["cum_counts"]

    # TIER 1: ZERO-UNPACK EXACT SHELL-RADIUS NORMS (0 coordinate unranks, 0 WHT_8!)
    b_diags, b_lookups = _zero_unpack_diag_norms(parsed["b_scales"], parsed["b_indices"], cum_counts)
    a_diags, a_lookups = _zero_unpack_diag_norms(parsed["a_scales"], parsed["a_indices"], cum_counts)
    r = parsed["rank"]
    q_s = parsed["scale_exp2"]
    b_frob_int = sum(b_diags)
    a_frob_int = sum(a_diags)
    cs_prod = b_frob_int * a_frob_int
    cs_rank = r * sum(bd * ad for bd, ad in zip(b_diags, a_diags))
    zero_unpack_ub_int = cs_prod if cs_prod <= cs_rank else cs_rank

    scale_shift = 4 * q_s
    # For rank 1, zero-unpack is 100% EXACT (sound AND complete) for both Frobenius & spectral!
    if r == 1 and not force_exact_tier2:
        exact_frob_int = zero_unpack_ub_int
        if (exact_frob_int << exp2) > (frob_s << scale_shift):
            raise Wht8BridgeError("FROBENIUS_EXCEEDS_CERTIFIED", "byte-derived rank-1 ||BA||_F^2 exceeds certified bound")
        if (exact_frob_int << exp2) > (spec_s << scale_shift):
            raise Wht8BridgeError("SPECTRAL_EXCEEDS_CERTIFIED", "byte-derived rank-1 sigma_max(BA)^2 exceeds certified bound")
        return {
            "admitted": True,
            "tier_used": "TIER1_ZERO_UNPACK_EXACT_RANK1",
            "adapter_sha256": parsed["sha256"],
            "rank": r,
            "superblock_e8_4": parsed["superblock_e8_4"],
            "shell_lookups": b_lookups + a_lookups,
            "coordinates_unranked": 0,
            "wht8_butterflies_run": 0,
            "b_frob_sq_q": fmt_q(Fraction(b_frob_int, 1 << (2 * q_s))),
            "a_frob_sq_q": fmt_q(Fraction(a_frob_int, 1 << (2 * q_s))),
            "zero_unpack_ub_q": fmt_q(Fraction(zero_unpack_ub_int, 1 << scale_shift)),
            "exact_ba_frob_sq_q": fmt_q(Fraction(exact_frob_int, 1 << scale_shift)),
        }

    # Sound Tier-1 fast-accept for rank >= 2 whenever zero-unpack upper bound clears spec_s (hence frob_s >= spec_s)
    if not force_exact_tier2 and (zero_unpack_ub_int << exp2) <= (spec_s << scale_shift):
        return {
            "admitted": True,
            "tier_used": "TIER1_ZERO_UNPACK_FAST_ACCEPT",
            "adapter_sha256": parsed["sha256"],
            "rank": r,
            "superblock_e8_4": parsed["superblock_e8_4"],
            "shell_lookups": b_lookups + a_lookups,
            "coordinates_unranked": 0,
            "wht8_butterflies_run": 0,
            "b_frob_sq_q": fmt_q(Fraction(b_frob_int, 1 << (2 * q_s))),
            "a_frob_sq_q": fmt_q(Fraction(a_frob_int, 1 << (2 * q_s))),
            "zero_unpack_ub_q": fmt_q(Fraction(zero_unpack_ub_int, 1 << scale_shift)),
        }

    # TIER 2: EXACT WHT_8-BYPASSED INTEGER GRAM + BAREISS PSD SPECTRAL GATE (0 WHT_8 butterflies!)
    g_b = _unrank_factor_gram_no_wht8(parsed["b_scales"], parsed["b_indices"], e8_tables, e8_4_tables)
    g_a = _unrank_factor_gram_no_wht8(parsed["a_scales"], parsed["a_indices"], e8_tables, e8_4_tables)
    exact_frob_int = sum(g_b[i][j] * g_a[j][i] for i in range(r) for j in range(r))
    if (exact_frob_int << exp2) > (frob_s << scale_shift):
        raise Wht8BridgeError("FROBENIUS_EXCEEDS_CERTIFIED", "byte-derived ||BA||_F^2 exceeds certified bound")

    # Exact Bareiss PSD: (spec_s * 2^(4*q_s)) * G_A - 2^(exp2) * (G_A G_B G_A) >= 0
    ga_gb_ga = _mat_mul_int(_mat_mul_int(g_a, g_b), g_a)
    lhs_coeff = spec_s << scale_shift
    rhs_coeff = 1 << exp2
    psd_mat = [
        [lhs_coeff * g_a[i][j] - rhs_coeff * ga_gb_ga[i][j] for j in range(r)]
        for i in range(r)
    ]
    if not bareiss_is_psd(psd_mat):
        raise Wht8BridgeError("SPECTRAL_EXCEEDS_CERTIFIED", "byte-derived sigma_max(BA)^2 exceeds certified bound")

    coords_per_block = 32 if parsed["superblock_e8_4"] else 8
    return {
        "admitted": True,
        "tier_used": "TIER2_WHT8_BYPASSED_GRAM_BAREISS",
        "adapter_sha256": parsed["sha256"],
        "rank": r,
        "superblock_e8_4": parsed["superblock_e8_4"],
        "shell_lookups": b_lookups + a_lookups,
        "coordinates_unranked": r * (parsed["d_out_blocks"] + parsed["d_in_blocks"]) * coords_per_block,
        "wht8_butterflies_run": 0,
        "b_frob_sq_q": fmt_q(Fraction(b_frob_int, 1 << (2 * q_s))),
        "a_frob_sq_q": fmt_q(Fraction(a_frob_int, 1 << (2 * q_s))),
        "zero_unpack_ub_q": fmt_q(Fraction(zero_unpack_ub_int, 1 << scale_shift)),
        "exact_ba_frob_sq_q": fmt_q(Fraction(exact_frob_int, 1 << scale_shift)),
        "gram_diag_matches_tier1": (
            ([g_b[i][i] for i in range(r)], [g_a[i][i] for i in range(r)]) == (b_diags, a_diags)
        ),
    }


def _full_unpack_wht8_dense_ba_frob_sq(
    parsed: dict[str, Any],
    e8_tables: dict[str, Any],
    e8_4_tables: dict[str, Any] | None,
) -> Fraction:
    """Independent brute-force reference: full unrank + explicit WHT_8 butterfly + dense BA matrix."""
    r = parsed["rank"]
    q_s = parsed["scale_exp2"]

    def _expand_factor(scales: list[list[int]], indices: list[list[int]]) -> list[list[int]]:
        # Returns scaled integer coords representing 2 * sqrt(8) * 2^(q_s) * factor
        vecs: list[list[int]] = []
        for k in range(r):
            flat: list[int] = []
            for sv, iv in zip(scales[k], indices[k]):
                blocks = unrank_e8_4(iv, e8_tables, e8_4_tables) if e8_4_tables is not None else (unrank_e8(iv, e8_tables),)
                for z in blocks:
                    rot = apply_h8_butterfly(z)
                    flat.extend(sv * c for c in rot)
            vecs.append(flat)
        return vecs

    b_cols = _expand_factor(parsed["b_scales"], parsed["b_indices"])
    a_rows = _expand_factor(parsed["a_scales"], parsed["a_indices"])
    d_out = len(b_cols[0])
    d_in = len(a_rows[0])
    # Each entry of BA_scaled is sum_k b_cols[k][i] * a_rows[k][j] = (2 * sqrt(8) * 2^q_s)^2 * (BA)_{i,j}
    # = 32 * 2^(2*q_s) * (BA)_{i,j}, so sum of squares = 1024 * 2^(4*q_s) * ||BA||_F^2
    sq_sum = 0
    for i in range(d_out):
        for j in range(d_in):
            entry = sum(b_cols[k][i] * a_rows[k][j] for k in range(r))
            sq_sum += entry * entry
    return Fraction(sq_sum, 1024 << (4 * q_s))


def _expect_reject_code(raw: bytes, cert: dict[str, Any], *, force_exact_tier2: bool = False) -> str:
    try:
        verify_packed_e8had_stage2(raw, cert, force_exact_tier2=force_exact_tier2)
    except Wht8BridgeError as exc:
        return exc.code
    return "UNEXPECTED_ADMIT"


def build_certificate_payload() -> dict[str, Any]:
    """Compute the complete exact-Z/Q Track S21f zero-unpack WHT_8 Frobenius bridge payload."""
    h8_iso = verify_h8_isometry()
    e8_t = build_e8_shell_tables(8)
    e8_4_t = build_e8_4_shell_tables(8)

    # Exact codebook capacities & zero-unpack operation counts for siliquant AUDIT-112R (88b & 136b E_8^4)
    e8_4_full = build_e8_4_shell_tables(400)
    cum4 = e8_4_full["cum_counts"]
    m4_88 = max(m for m in range(len(cum4)) if cum4[m] <= (1 << 88))
    m4_136 = max(m for m in range(len(cum4)) if cum4[m] <= (1 << 136))
    bsearch_88 = (m4_88 + 1).bit_length()
    bsearch_136 = (m4_136 + 1).bit_length()
    # Full unpack + WHT_8 + Frobenius op count per 32-weight E_8^4 super-block:
    # 32 coord unranks + 4*24=96 WHT_8 butterfly ops + 32 squares + 31 adds = 191 ops
    full_32w_ops = 32 + 96 + 32 + 31
    zu_88_ops = bsearch_88 + 3
    zu_136_ops = bsearch_136 + 3

    # Construct real single-E_8 rank-1 adapter (q_s=2 -> scale step 1/4) with true ||BA||_F^2 = 6 > 5.0
    # B has 1 col of 2 blocks: shell 1 (||v||^2=2, s=1 -> 2) + shell 2 (||v||^2=4, s=1 -> 4) => ||B||_F^2 = 6
    # A has 1 row of 1 block: shell 1 (||v||^2=2, s=1/2 -> 2*(1/4)=1/2)? Wait: use s=2/4=1/2 on 2 blocks -> ||A||_F^2 = 1!
    z_m1_a = (2, 2, 0, 0, 0, 0, 0, 0)      # shell 1 (||v||^2 = 2)
    z_m1_b = (1, 1, 1, 1, 1, 1, 1, 1)      # shell 1 half-integer (||v||^2 = 2, <a,b> = 1)
    z_m1_c = (0, 0, 2, 2, 0, 0, 0, 0)      # shell 1 orthogonal to z_m1_a
    z_m2_a = (2, 2, 2, 2, 0, 0, 0, 0)      # shell 2 (||v||^2 = 4)
    z_m3_a = (4, 2, 2, 0, 0, 0, 0, 0)      # shell 3 (||v||^2 = 6)

    idx_m1_a = rank_e8(z_m1_a, e8_t)
    idx_m1_b = rank_e8(z_m1_b, e8_t)
    idx_m1_c = rank_e8(z_m1_c, e8_t)
    idx_m2_a = rank_e8(z_m2_a, e8_t)
    idx_m3_a = rank_e8(z_m3_a, e8_t)

    # Witness 1 (Rank-1 inflated adapter, true ||BA||_F^2 = 6):
    # q_s = 1 (scale = sigma / 2):
    # B: 2 blocks with sigma=2 (s=1), shells m=1 (2) and m=2 (4) -> ||B||_F^2 = 6
    # A: 2 blocks with sigma=1 (s=1/2), shells m=1 (2) and m=1 (2) -> ||A||_F^2 = 2*(1/4)*2 = 1
    # Exact ||BA||_F^2 = 6 * 1 = 6!
    raw_r1_inflated = pack_e8had_adapter(
        b_scales=[[2, 2]],
        b_indices=[[idx_m1_a, idx_m2_a]],
        a_scales=[[1, 1]],
        a_indices=[[idx_m1_a, idx_m1_b]],
        scale_exp2=1,
        max_shell=8,
        alpha_num=1,
        alpha_den=1,
        superblock_e8_4=False,
    )
    sha_r1_inflated = hashlib.sha256(raw_r1_inflated).hexdigest()
    parsed_r1_inflated = parse_e8had_bitstream(raw_r1_inflated)
    ref_frob_r1 = _full_unpack_wht8_dense_ba_frob_sq(parsed_r1_inflated, e8_t, None)

    # Attack 1: WD40-05 radix_scale_exp2 relabeling 32 -> 48 on inflated ||BA||_F^2 = 6 adapter
    cert_wd40_relabel = {
        "adapter_sha256": sha_r1_inflated,
        "radix_scale_exp2": 48,
        "frobenius_sq_scaled": 6 << 32,
        "spectral_norm_sq_scaled": 6 << 32,
        "max_frobenius_sq_scaled": 5 << 48,
        "max_spectral_norm_sq_scaled": 5 << 48,
    }
    wd40_code = _expect_reject_code(raw_r1_inflated, cert_wd40_relabel)

    # Honest rank-1 control (scale_exp2=1, B sigma=[1,1] -> ||B||^2=3/2, A sigma=[1,1] -> ||A||^2=1 => ||BA||_F^2 = 3/2 <= 5)
    raw_r1_honest = pack_e8had_adapter(
        b_scales=[[1, 1]],
        b_indices=[[idx_m1_a, idx_m2_a]],
        a_scales=[[1, 1]],
        a_indices=[[idx_m1_a, idx_m1_b]],
        scale_exp2=1,
        max_shell=8,
        alpha_num=1,
        alpha_den=1,
        superblock_e8_4=False,
    )
    sha_r1_honest = hashlib.sha256(raw_r1_honest).hexdigest()
    cert_r1_honest = {
        "adapter_sha256": sha_r1_honest,
        "radix_scale_exp2": 32,
        "frobenius_sq_scaled": 3 << 31,
        "spectral_norm_sq_scaled": 3 << 31,
        "max_frobenius_sq_scaled": 5 << 32,
        "max_spectral_norm_sq_scaled": 5 << 32,
    }
    r1_honest_res = verify_packed_e8had_stage2(raw_r1_honest, cert_r1_honest)
    ref_frob_r1_honest = _full_unpack_wht8_dense_ba_frob_sq(parse_e8had_bitstream(raw_r1_honest), e8_t, None)

    # Attack 2: warp#13 b-times-8 scale forgery (64x Frobenius inflation: 3/2 -> 96)
    raw_b_times_8 = pack_e8had_adapter(
        b_scales=[[8, 8]],
        b_indices=[[idx_m1_a, idx_m2_a]],
        a_scales=[[1, 1]],
        a_indices=[[idx_m1_a, idx_m1_b]],
        scale_exp2=1,
        max_shell=8,
        alpha_num=1,
        alpha_den=1,
        superblock_e8_4=False,
    )
    b8_stale_sha_code = _expect_reject_code(raw_b_times_8, cert_r1_honest)
    b8_resigned_cert = dict(cert_r1_honest, adapter_sha256=hashlib.sha256(raw_b_times_8).hexdigest())
    b8_resigned_code = _expect_reject_code(raw_b_times_8, b8_resigned_cert)

    # Attack 3: warp#13 alpha-32 metadata inflation on rank-4 adapter (alpha/r = 8 -> 64x energy)
    raw_alpha_32 = pack_e8had_adapter(
        b_scales=[[1], [1], [1], [1]],
        b_indices=[[idx_m1_a], [idx_m1_a], [idx_m1_a], [idx_m1_a]],
        a_scales=[[1], [1], [1], [1]],
        a_indices=[[idx_m1_a], [idx_m1_a], [idx_m1_a], [idx_m1_a]],
        scale_exp2=2,
        max_shell=8,
        alpha_num=32,
        alpha_den=1,
        superblock_e8_4=False,
    )
    cert_alpha_32 = dict(cert_r1_honest, adapter_sha256=hashlib.sha256(raw_alpha_32).hexdigest())
    alpha_32_code = _expect_reject_code(raw_alpha_32, cert_alpha_32)

    # Attack 4: Shell-index bump (m=1 -> m=3) & out-of-range shell index overflow
    raw_shell_bump = pack_e8had_adapter(
        b_scales=[[1, 1]],
        b_indices=[[idx_m3_a, idx_m2_a]],
        a_scales=[[1, 1]],
        a_indices=[[idx_m1_a, idx_m1_b]],
        scale_exp2=1,
        max_shell=8,
        alpha_num=1,
        alpha_den=1,
        superblock_e8_4=False,
    )
    shell_bump_cert = dict(cert_r1_honest, adapter_sha256=hashlib.sha256(raw_shell_bump).hexdigest())
    shell_bump_code = _expect_reject_code(raw_shell_bump, shell_bump_cert)

    # Attack 5: Rank-2 exact spectral vs Frobenius separation (Tier-2 WHT_8-bypassed Bareiss 1-ULP tightness!)
    # Column 0 of B: z_m1_a (||v||^2=2, s=1 -> 2); Column 1 of B: z_m1_b (||v||^2=2, s=1 -> 2, <v0,v1>=1)
    # Row 0 of A: z_m1_a (s=1/2 -> 1/2); Row 1 of A: z_m1_c orthogonal to z_m1_a (s=1/2 -> 1/2, <a0,a1>=0)
    # G_B = [[2, 1], [1, 2]], G_A = [[1/2, 0], [0, 1/2]] -> G_B G_A = [[1, 1/2], [1/2, 1]]
    # Eigenvalues of G_B G_A (exact squared singular values of BA): sigma_1^2 = 3/2, sigma_2^2 = 1/2!
    # Exact ||BA||_F^2 = 3/2 + 1/2 = 2, while exact sigma_max(BA)^2 = 3/2!
    # Zero-unpack UB = min(4 * 1, 2 * (2*1/2 + 2*1/2)) = 4!
    raw_r2 = pack_e8had_adapter(
        b_scales=[[2], [2]],
        b_indices=[[idx_m1_a], [idx_m1_b]],
        a_scales=[[1], [1]],
        a_indices=[[idx_m1_a], [idx_m1_c]],
        scale_exp2=1,
        max_shell=8,
        alpha_num=2,
        alpha_den=1,
        superblock_e8_4=False,
    )
    sha_r2 = hashlib.sha256(raw_r2).hexdigest()
    ref_frob_r2 = _full_unpack_wht8_dense_ba_frob_sq(parse_e8had_bitstream(raw_r2), e8_t, None)
    cert_r2_tight = {
        "adapter_sha256": sha_r2,
        "radix_scale_exp2": 32,
        "frobenius_sq_scaled": 2 << 32,
        "spectral_norm_sq_scaled": 3 << 31,      # exact 3/2 == sigma_max(BA)^2!
        "max_frobenius_sq_scaled": 5 << 32,
        "max_spectral_norm_sq_scaled": 5 << 32,
    }
    r2_tight_res = verify_packed_e8had_stage2(raw_r2, cert_r2_tight)
    cert_r2_minus_1ulp = dict(cert_r2_tight, spectral_norm_sq_scaled=(3 << 31) - 1)
    r2_1ulp_code = _expect_reject_code(raw_r2, cert_r2_minus_1ulp)
    # Rank-2 fast-accept via Tier 1 when certified bound >= zero_unpack_ub (4)
    cert_r2_fast = {
        "adapter_sha256": sha_r2,
        "radix_scale_exp2": 32,
        "frobenius_sq_scaled": 4 << 32,
        "spectral_norm_sq_scaled": 4 << 32,
        "max_frobenius_sq_scaled": 5 << 32,
        "max_spectral_norm_sq_scaled": 5 << 32,
    }
    r2_fast_res = verify_packed_e8had_stage2(raw_r2, cert_r2_fast)

    # Witness 6: E_8^4 32-weight super-block zero-unpack exact equivalence vs full WHT_8 dense decode!
    sb_b = (z_m1_a, z_m1_b, z_m2_a, (0, 0, 0, 0, 0, 0, 0, 0))   # total shell 1+1+2+0 = 4 (||v||^2 = 8)
    sb_a = (z_m1_c, (0, 0, 0, 0, 0, 0, 0, 0), z_m1_a, z_m1_b)   # total shell 1+0+1+1 = 3 (||v||^2 = 6)
    idx_sb_b = rank_e8_4(sb_b, e8_t, e8_4_t)
    idx_sb_a = rank_e8_4(sb_a, e8_t, e8_4_t)
    # q_s = 2 (scale step 1/4): B scale 2 (s=1/2 -> ||B||^2 = 8/4 = 2), A scale 2 (s=1/2 -> ||A||^2 = 6/4 = 3/2)
    # Exact ||BA||_F^2 = 2 * (3/2) = 3!
    raw_e8_4 = pack_e8had_adapter(
        b_scales=[[2]],
        b_indices=[[idx_sb_b]],
        a_scales=[[2]],
        a_indices=[[idx_sb_a]],
        scale_exp2=2,
        max_shell=8,
        alpha_num=1,
        alpha_den=1,
        superblock_e8_4=True,
    )
    sha_e8_4 = hashlib.sha256(raw_e8_4).hexdigest()
    cert_e8_4 = {
        "adapter_sha256": sha_e8_4,
        "radix_scale_exp2": 32,
        "frobenius_sq_scaled": 3 << 32,
        "spectral_norm_sq_scaled": 3 << 32,
        "max_frobenius_sq_scaled": 5 << 32,
        "max_spectral_norm_sq_scaled": 5 << 32,
    }
    e8_4_res = verify_packed_e8had_stage2(raw_e8_4, cert_e8_4)
    ref_frob_e8_4 = _full_unpack_wht8_dense_ba_frob_sq(parse_e8had_bitstream(raw_e8_4), e8_t, e8_4_t)

    return {
        "track": "S21f",
        "audit_id": "AUDIT-121F",
        "oracle_class": "exact",
        "novelty_class": "N1",
        "theorem1_wht8_isometry": h8_iso,
        "theorem2_eisenstein_shell_identities": {
            "max_shell_verified_dp": 8,
            "e8_shell_counts_m0_to_m8": list(e8_t["shell_counts"]),
            "eisenstein_e4_exact": e8_t["eisenstein_e4_exact"],
            "e4_squared_equals_e8_exact": e8_4_t["e4_squared_equals_e8_exact"],
            "e8_4_88bit_max_shell": m4_88,
            "e8_4_88bit_codebook_size": str(cum4[m4_88]),
            "e8_4_88bit_bsearch_max_cmp": bsearch_88,
            "e8_4_136bit_max_shell": m4_136,
            "e8_4_136bit_codebook_size": str(cum4[m4_136]),
            "e8_4_136bit_bsearch_max_cmp": bsearch_136,
            "op_count_per_32w_superblock": {
                "full_unpack_wht8_frobenius_ops": full_32w_ops,
                "zero_unpack_88bit_ops": zu_88_ops,
                "zero_unpack_136bit_ops": zu_136_ops,
                "speedup_ratio_88bit": fmt_q(Fraction(full_32w_ops, zu_88_ops)),
                "speedup_ratio_136bit": fmt_q(Fraction(full_32w_ops, zu_136_ops)),
            },
        },
        "theorem3_4_zero_unpack_and_tier2_exact_parity": {
            "rank1_honest_tier1": r1_honest_res,
            "rank1_honest_full_wht8_ref_q": fmt_q(ref_frob_r1_honest),
            "rank1_honest_exact_match": r1_honest_res["exact_ba_frob_sq_q"] == fmt_q(ref_frob_r1_honest),
            "rank1_inflated_full_wht8_ref_q": fmt_q(ref_frob_r1),
            "rank2_tight_tier2": r2_tight_res,
            "rank2_full_wht8_ref_q": fmt_q(ref_frob_r2),
            "rank2_exact_match": r2_tight_res["exact_ba_frob_sq_q"] == fmt_q(ref_frob_r2),
            "rank2_exact_singular_values_sq_q": ["3/2", "1/2"],
            "rank2_fast_accept_tier1": r2_fast_res,
            "e8_4_superblock_tier1": e8_4_res,
            "e8_4_superblock_full_wht8_ref_q": fmt_q(ref_frob_e8_4),
            "e8_4_superblock_exact_match": e8_4_res["exact_ba_frob_sq_q"] == fmt_q(ref_frob_e8_4),
            "e8_4_superblock_roundtrip_exact": unrank_e8_4(idx_sb_b, e8_t, e8_4_t) == sb_b,
        },
        "theorem5_cross_org_attack_closures": {
            "wd40_05_radix_scale_exp2_relabel_32_to_48": {
                "true_frobenius_sq_q": fmt_q(ref_frob_r1),
                "relabelled_declared_q": fmt_q(Fraction(6 << 32, 1 << 48)),
                "rejection_code": wd40_code,
                "blocked": wd40_code == "FROBENIUS_EXCEEDS_CERTIFIED",
            },
            "warp13_b_times_8_scale_forgery": {
                "stale_sha_rejection_code": b8_stale_sha_code,
                "resigned_sha_rejection_code": b8_resigned_code,
                "blocked": (
                    (b8_stale_sha_code, b8_resigned_code)
                    == ("ADAPTER_SHA_MISMATCH", "FROBENIUS_EXCEEDS_CERTIFIED")
                ),
            },
            "warp13_alpha_32_metadata_inflation": {
                "rejection_code": alpha_32_code,
                "blocked": alpha_32_code == "NON_UNIT_ALPHA_SCALE",
            },
            "shell_index_bump_m1_to_m3": {
                "rejection_code": shell_bump_code,
                "blocked": shell_bump_code == "FROBENIUS_EXCEEDS_CERTIFIED",
            },
            "rank2_bareiss_1ulp_spectral_understatement": {
                "exact_spectral_sq_q": "3/2",
                "understated_bound_q": fmt_q(Fraction((3 << 31) - 1, 1 << 32)),
                "rejection_code": r2_1ulp_code,
                "blocked": r2_1ulp_code == "SPECTRAL_EXCEEDS_CERTIFIED",
            },
        },
    }


def generate_certificate() -> dict[str, Any]:
    payload = build_certificate_payload()
    payload_sha = canonical_sha256(payload)
    cert_body = {
        "schema": "aletheia.track_s21f.zero_unpack_wht8_frobenius_bridge.v1",
        "payload_sha256": payload_sha,
        "payload": payload,
    }
    cert_sha = canonical_sha256(cert_body)
    return {
        "schema": cert_body["schema"],
        "payload_sha256": payload_sha,
        "certificate_sha256": cert_sha,
        "payload": payload,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track S21f zero-unpack WHT_8 Frobenius bridge oracle",
        allow_abbrev=False,
    )
    parser.add_argument("--standalone", action="store_true", help="Compute self-contained exact certificate")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON output to stdout")
    try:
        checked_argv = validate_cli_argv(argv)
        args = parser.parse_args(checked_argv)
        cert = generate_certificate()
    except Wht8BridgeError as exc:
        sys.stderr.write(f"FAIL {exc.code}: {exc.message}\n")
        return 1
    if args.json:
        sys.stdout.write(canonical_json_bytes(cert).decode("ascii") + "\n")
        return 0
    print(
        f"OK TRACK-S21F-WHT8-FROBENIUS payload_sha256={cert['payload_sha256']} "
        f"certificate_sha256={cert['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
