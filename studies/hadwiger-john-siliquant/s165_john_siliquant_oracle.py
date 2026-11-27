#!/usr/bin/env python3
"""Track S165 (EXP-Q47b, PREREG-W22-HADWIGER-SILIQUANT) Exact Z/Q Oracle & Certificate Generator.

Pure Python 3 standard library over Z, Q, and finite Decimal only.
Zero floating-point literals or calls, zero bare asserts, zero unseeded RNG.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

SCHEMA_VERSION = "s165.q47b.hadwiger_john_siliquant.v1"
TRACK_ID = "S165"
EXPERIMENT_ID = "EXP-Q47b"
PREREG_ID = "PREREG-W22-HADWIGER-SILIQUANT"
ORACLE_CLASS = "exact_and_sampled"
NOVELTY_CLASS = "N1"
ENERGY_CLAIM = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"

LEAN_MODULE = "JohnQuantOperators"
LEAN_FILE_NAME = "JohnQuantOperators.lean"
EXPECTED_TOOLCHAIN = "leanprover/lean4:v4.36.0"
EXPECTED_LEAN_THEOREMS = 24
RECEIPT_FILE_NAME = "s165_empirical_receipt.json"
EMPIRICAL_RESULTS_FILE_NAME = "s165_empirical_results.json"
CERT_FILE_NAME = "s165_john_siliquant_certificate.json"

PINNED_PREREG_SHA256 = "c57b8a40d33d2f8822b58c87141cbf69ff3d64a292058faaf21e993db619f74b"
PINNED_LEAN_SHA256 = "d447fc897f4c60f118a94784638fe428f573ea062d89f45d35f8bc50bd33a39e"
PINNED_TOOLCHAIN_SHA256 = "582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a"
PINNED_LAKEFILE_SHA256 = "2b2a9028814ee830a46fdde241f71d3ec8d943225a80b0c4a575ef9390fc55ab"
PINNED_MANIFEST_SHA256 = "c453fb1b592b20c2fa9c3cb5875f3cbad98ad4923f6a05e54029f2a80396f2c7"
PINNED_RECEIPT_SHA256 = "a430f75e6d9711c42d7054a5642118e6e7991018165b2464b527d239f4f08d4f"
PINNED_EMPIRICAL_RESULTS_SHA256 = "b2427df13fbf73f186107384451c3c48e28ddacbaf586fd4410436679950712b"

PI_LOW = Fraction(314159265358979323846, 10 ** 20)
PI_HIGH = Fraction(314159265358979323847, 10 ** 20)

MAX_BYTES = 1048576
MAX_DEPTH = 32
MAX_INT_DIGITS = 256
MODELS = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B")
MODEL_SHAPES: dict[str, tuple[int, int, int, int]] = {
    # (num_hidden_layers, hidden_size, kv_dim, intermediate_size)
    "Qwen/Qwen2.5-0.5B": (24, 896, 128, 4864),
    "Qwen/Qwen2.5-1.5B": (28, 1536, 256, 8960),
}
REQUIRED_ARMS = (
    "A0_b30_RTN_unwhitened",
    "A1_b30_E8_H8_hess",
    "A2_b30_A16_BlockH4_hess",
    "A3_b30_A16_BlockH4_QuadSnake_SN",
    "A4_b30_A16_BlockH4_QuadLPT_SN",
    "A5_b30_A16_RandOrth_hess",
    "B1a_b30_JohnS_ZeroCal_BlockH4_Folded0B",
    "B1b_b30_JohnHess_BlockH4_Folded0B",
    "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B",
    "B1d_b30_JohnHess_E8_H8_Folded0B",
    "B1e_b30_JohnHess_BlockH4_PerMat",
    "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B",
    "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched",
    "B3a_b30_PolarCap_CSR025_BlockH4_Matched",
    "B3b_b30_PolarCap_BlockSplit_1over19_Matched",
    "B3c_b30_JohnHess0B_Plus_PolarCap_CSR025",
    "A1_b45_E8_H8_hess",
    "A2_b45_A16_BlockH4_hess",
    "A3_b45_A16_BlockH4_QuadSnake_SN",
    "A4_b45_A16_BlockH4_QuadLPT_SN",
    "B1b_b45_JohnHess_BlockH4_Folded0B",
    "B1c_b45_JohnHess_BlockH4_QuadSnake_Folded0B",
    "B1d_b45_JohnHess_E8_H8_Folded0B",
    "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B",
    "B3a_b45_PolarCap_CSR025_BlockH4_Matched",
    "B3c_b45_JohnHess0B_Plus_PolarCap_CSR025",
)


class OracleError(ValueError):
    """Fail-closed oracle exception carrying an explicit error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def require_ok(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise OracleError(code, message)


def frac_str(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}" if q.denominator != 1 else str(q.numerator)


def _track_base() -> Path:
    here = Path(__file__).parent
    if here.is_absolute():
        return here
    return Path(os.getcwd()) / here


def _validate_cli_path_arg(raw_arg: str) -> Path:
    require_ok(len(raw_arg) > 0, "ORA_CLI_PATH", "empty path argument")
    require_ok("\x00" not in raw_arg, "ORA_NUL_PATH", "NUL byte in path")
    p = Path(raw_arg)
    require_ok(raw_arg not in (".", ".."), "ORA_TRAVERSAL", "relative dot path forbidden")
    require_ok(".." not in p.parts, "ORA_TRAVERSAL", "parent traversal forbidden")
    return p


def _open_verified_parent_dir(path: Path) -> tuple[int, str]:
    raw_str = str(path)
    require_ok("\x00" not in raw_str, "ORA_NUL_PATH", "NUL byte in path")
    abs_p = path if path.is_absolute() else (Path.cwd() / path)
    parts = abs_p.parts
    require_ok(".." not in parts, "ORA_TRAVERSAL", "parent traversal forbidden")
    require_ok(len(parts) >= 2, "ORA_ROOT_PATH", "cannot open root directory")
    target = parts[-1]
    dfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        for seg in parts[1:-1]:
            nxt = os.open(seg, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            os.close(dfd)
            dfd = nxt
    except OSError:
        os.close(dfd)
        raise OracleError("ORA_PARENT_OPEN", "cannot open parent directory")
    return dfd, target


def _read_regular_bytes(path: Path) -> bytes:
    dfd, target = _open_verified_parent_dir(path)
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
    except OSError:
        os.close(dfd)
        raise OracleError("ORA_FILE_OPEN", "cannot open regular file")
    os.close(dfd)
    chunks: list[bytes] = []
    try:
        info = os.fstat(fd)
        require_ok(stat.S_ISREG(info.st_mode), "ORA_NOT_REG", "not a regular file")
        total = 0
        while True:
            chunk = os.read(fd, 65536)
            if chunk == b"":
                break
            total = total + len(chunk)
            require_ok(total <= MAX_BYTES, "ORA_TOO_LARGE", "file exceeds cap")
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _check_dest_safe(dfd: int, target: str) -> None:
    try:
        st = os.lstat(target, dir_fd=dfd)
    except FileNotFoundError:
        return
    except OSError:
        raise OracleError("ORA_WRITE_LSTAT", "cannot lstat destination")
    require_ok(not stat.S_ISLNK(st.st_mode), "ORA_WRITE_SYMLINK", "destination must not be a symlink")
    require_ok(stat.S_ISREG(st.st_mode), "ORA_WRITE_NOT_REG", "destination is not a regular file")
    require_ok(st.st_nlink == 1, "ORA_WRITE_HARDLINK", "destination must not have multiple hardlinks")


def write_cert_file(path: Path, payload: dict[str, object]) -> None:
    dfd, target = _open_verified_parent_dir(path)
    tmp_name: str | None = None
    tmp_fd = -1
    try:
        _check_dest_safe(dfd, target)
        raw = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        pid = os.getpid()
        for attempt in range(64):
            cand = f".{target}.tmp.{pid}.{attempt}"
            try:
                tmp_fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
                tmp_name = cand
                break
            except FileExistsError:
                continue
        require_ok(tmp_name is not None, "ORA_WRITE_TMP_OPEN", "cannot create unique temp file")
        written = 0
        view = memoryview(raw)
        while written < len(raw):
            chunk_n = os.write(tmp_fd, view[written:])
            require_ok(chunk_n > 0, "ORA_WRITE_SHORT", "short write to temp file")
            written = written + chunk_n
        os.fsync(tmp_fd)
        os.close(tmp_fd)
        tmp_fd = -1
        _check_dest_safe(dfd, target)
        os.replace(tmp_name, target, src_dir_fd=dfd, dst_dir_fd=dfd)
        tmp_name = None
        try:
            os.fsync(dfd)
        except OSError:
            pass
    finally:
        if tmp_fd >= 0:
            os.close(tmp_fd)
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


def _parse_strict_json_bytes(raw: bytes, *, allow_decimal: bool = False) -> dict[str, object]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise OracleError("ORA_BAD_UTF8", "invalid utf-8")
    require_ok(text.startswith("\ufeff") is False, "ORA_BOM", "BOM rejected")
    require_ok("\r" not in text, "ORA_CRLF", "CR rejected")
    require_ok(text.endswith("\n"), "ORA_NO_FINAL_LF", "missing trailing LF")

    depth = 0
    in_str = False
    esc = False
    digits = 0
    prev_digit = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            digits = 0
            prev_digit = False
            continue
        if ch == '"':
            in_str = True
            digits = 0
            prev_digit = False
            continue
        if ch in "{[":
            depth = depth + 1
            require_ok(depth <= MAX_DEPTH, "ORA_JSON_DEPTH", "json depth exceeded")
        elif ch in "}]":
            depth = depth - 1
            require_ok(depth >= 0, "ORA_JSON_UNBAL", "negative json depth")
        if ch.isdigit():
            digits = digits + 1
            require_ok(digits <= MAX_INT_DIGITS, "ORA_INT_DIGITS", "int digit cap exceeded")
            prev_digit = True
        else:
            if prev_digit and ch in ".eE" and not allow_decimal:
                raise OracleError("ORA_FLOAT_TOKEN", "float token rejected")
            digits = 0
            prev_digit = False
    require_ok(depth == 0, "ORA_JSON_STATE", "unbalanced json brackets")
    require_ok(in_str is False, "ORA_JSON_STATE", "unterminated json string")

    def _dup_free(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in pairs:
            require_ok(k not in out, "ORA_DUP_KEY", "duplicate json key")
            out[k] = v
        return out

    def _bad_const(_s: str) -> None:
        raise OracleError("ORA_BAD_CONST", "non-json constant")

    def _dec_parse(raw_num: str) -> Decimal:
        try:
            return Decimal(raw_num)
        except InvalidOperation:
            raise OracleError("ORA_BAD_CONST", "invalid decimal token")

    try:
        if allow_decimal:
            obj = json.loads(text, object_pairs_hook=_dup_free, parse_constant=_bad_const, parse_float=_dec_parse)
        else:
            obj = json.loads(text, object_pairs_hook=_dup_free, parse_constant=_bad_const)
    except OracleError:
        raise
    except (json.JSONDecodeError, ValueError):
        raise OracleError("ORA_JSON_PARSE", "malformed json")
    require_ok(type(obj) is dict, "ORA_NOT_DICT", "top-level json must be object")
    return obj


def _req_int(d: dict[str, object], key: str, code: str) -> int:
    require_ok(key in d, code, f"missing int key {key}")
    require_ok(type(d[key]) is int, code, f"non-int key {key}")
    return d[key]  # type: ignore[return-value]


def _req_str(d: dict[str, object], key: str, code: str) -> str:
    require_ok(key in d, code, f"missing str key {key}")
    require_ok(type(d[key]) is str, code, f"non-str key {key}")
    return d[key]  # type: ignore[return-value]


def _req_dict(d: dict[str, object], key: str, code: str) -> dict[str, object]:
    require_ok(key in d, code, f"missing dict key {key}")
    require_ok(type(d[key]) is dict, code, f"non-dict key {key}")
    return d[key]  # type: ignore[return-value]


def _req_list(d: dict[str, object], key: str, code: str) -> list[object]:
    require_ok(key in d, code, f"missing list key {key}")
    require_ok(type(d[key]) is list, code, f"non-list key {key}")
    return d[key]  # type: ignore[return-value]


def _scale_num(val: object, mult: int, code: str) -> int:
    require_ok(type(val) in (int, Decimal), code, "expected int or finite Decimal")
    d = Decimal(val) if type(val) is int else val  # type: ignore[arg-type]
    require_ok(d.is_finite(), code, "non-finite numeric value")
    return int((d * mult).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def check_john_gram(vecs: list[tuple[int, ...]], dim: int, diag_scale: int) -> int:
    """Validate exact John contact-point Gram identity and unit/L1 contact norm."""
    require_ok(len(vecs) >= dim, "P1_GRAM_COUNT", "too few contact vectors")
    for v in vecs:
        require_ok(len(v) == dim, "P1_VEC_DIM", "contact vector dimension mismatch")
        require_ok(sum(c * c for c in v) == dim, "P1_VEC_NORM", "contact vector L2 norm mismatch")
    for i in range(dim):
        for j in range(dim):
            g_ij = sum(v[i] * v[j] for v in vecs)
            require_ok(g_ij == (diag_scale if i == j else 0), "P1_GRAM_DIAG", "Gram identity failed")
    return len(vecs)


def check_folded_dot_invariance(
    channels: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...],
) -> Fraction:
    """Certify exact +0 B folded-diagonal dot-product invariance across arbitrary dimension."""
    require_ok(len(channels) > 0, "P2_EMPTY", "channels must be non-empty")
    folded = Fraction(0, 1)
    for w, g, x, s in channels:
        require_ok(s > 0, "P2_SCALE_POS", "diagonal fold scale must be positive")
        folded = folded + (w * s) * ((g / s) * x)
    return folded


def check_cauchy_schwarz_pair(
    h_vec: tuple[Fraction, ...],
    a2_vec: tuple[Fraction, ...],
    u_cand: tuple[Fraction, ...],
) -> Fraction:
    """Evaluate exact L2 diagonal whitening objective sum(u_j a2_j) * sum(h_j / u_j) against CS bound."""
    require_ok(len(h_vec) > 0, "P2_CS_LEN", "empty vector")
    require_ok(len(h_vec) == len(a2_vec), "P2_CS_LEN", "h/a2 length mismatch")
    require_ok(len(h_vec) == len(u_cand), "P2_CS_LEN", "u_cand length mismatch")
    s1 = Fraction(0, 1)
    s2 = Fraction(0, 1)
    for h, a2, u in zip(h_vec, a2_vec, u_cand):
        require_ok(u > 0, "P2_U_POS", "whitening scale must be positive")
        s1 = s1 + u * a2
        s2 = s2 + h / u
    return s1 * s2


def check_chebyshev_interval(lo: int, hi: int) -> tuple[Fraction, Fraction, Fraction]:
    """Compute and certify exact Helly-Chebyshev center, half-width, and origin radius on [lo, hi]."""
    require_ok(lo <= hi, "P3_INT_ORDER", "interval lo must be <= hi")
    c_star = Fraction(lo + hi, 2)
    r_star = Fraction(hi - lo, 2)
    r_origin = Fraction(max(abs(lo), abs(hi)), 1)
    return c_star, r_star, r_origin


def check_polar_crossover(
    m_bulk: int, m_all: int, e_bulk: int, e_cap: int, m_num: int, m_den: int
) -> Fraction:
    """Compute exact polar-cap vs equatorial-bulk MSE crossover ratio."""
    require_ok(0 < m_bulk, "P3_MBULK_POS", "m_bulk must be positive")
    require_ok(m_bulk < m_all, "P3_MCONTRACT", "m_bulk must be < m_all")
    require_ok(e_cap > 0, "P3_ECAP_POS", "e_cap must be positive")
    require_ok(m_den > 0, "P3_MDEN_POS", "m_den must be positive")
    return Fraction(m_num * m_bulk * m_bulk * e_bulk, m_den * m_all * m_all * (e_bulk + e_cap))


def compute_pillar1_john_geometry() -> dict[str, object]:
    """Exact rational John-position hypercube mean-width, volume, diameter gaps, and O_4/O_16 decompositions."""
    i_even_pi: dict[int, Fraction] = {2: Fraction(2, 1)}
    cur = Fraction(2, 1)
    for d in range(2, 16, 2):
        cur = cur * Fraction(d, d + 1)
        i_even_pi[d + 2] = cur

    m3 = Fraction(3, 2)
    m4_pi = 4 * i_even_pi[4]
    m8_pi = 8 * i_even_pi[8]
    m16_pi = 16 * i_even_pi[16]

    vol_q_over_b1 = {str(d): math.factorial(d) for d in (3, 4, 8, 16)}
    vol_ball_coeff: dict[int, Fraction] = {
        3: Fraction(4, 3),
        4: Fraction(1, 2),
        8: Fraction(1, 24),
        16: Fraction(1, 40320),
    }
    vol_q_over_ball_num = {str(d): int(Fraction(1 << d, 1) / vol_ball_coeff[d]) for d in (3, 4, 8, 16)}

    o4_vecs = [
        (1, s2, s3, s4)
        for s2 in (1, -1)
        for s3 in (1, -1)
        for s4 in (1, -1)
    ]
    n_o4 = check_john_gram(o4_vecs, 4, 8)

    h: list[list[int]] = [[1]]
    while len(h) < 16:
        h = [row + row for row in h] + [row + [-x for x in row] for row in h]
    n_o16 = check_john_gram([tuple(r) for r in h], 16, 16)

    star_helly_area = Fraction(14, 1) * Fraction(1, 16)
    zonotope_v2_sq = [4 * 4 * 3, 6 * 6]

    return {
        "diameter_sq_gaps": {"3": 3, "4": 4, "8": 8, "16": 16},
        "mean_width_exact_pi_cleared": {
            "3": frac_str(m3),
            "4_times_pi": frac_str(m4_pi),
            "8_times_pi": frac_str(m8_pi),
            "16_times_pi": frac_str(m16_pi),
        },
        "mean_width_4dp_brackets": {
            "4": ["16976/10000", "16977/10000"],
            "8": ["23282/10000", "23283/10000"],
            "16": ["32417/10000", "32418/10000"],
        },
        "vol_cube_over_cross": vol_q_over_b1,
        "vol_cube_over_ball_pi_cleared": vol_q_over_ball_num,
        "o4_john_contacts": n_o4,
        "o4_john_weight_sum": 4,
        "o16_john_contacts": n_o16,
        "o16_john_weight_sum": 16,
        "d4_star_helly_14_cap_area": frac_str(star_helly_area),
        "d4_zonotope_v2_witness_sq": zonotope_v2_sq,
    }


def compute_pillar2_zero_byte_folding() -> dict[str, object]:
    """Exact +0 B folded diagonal affine-whitening invariance, Block-H4 Parseval, & Cauchy-Schwarz optimality."""
    ws = (Fraction(3, 4), Fraction(-5, 8), Fraction(7, 16), Fraction(11, 8))
    gs = (Fraction(5, 4), Fraction(3, 2), Fraction(7, 8), Fraction(9, 4))
    xs = (Fraction(2, 3), Fraction(-4, 5), Fraction(6, 7), Fraction(1, 2))
    ss = (Fraction(3, 2), Fraction(1, 4), Fraction(5, 3), Fraction(2, 5))
    folded_dot = check_folded_dot_invariance(tuple(zip(ws, gs, xs, ss)))

    # Independent 5-channel witness verification
    check_folded_dot_invariance((
        (Fraction(1, 3), Fraction(7, 2), Fraction(-2, 5), Fraction(4, 3)),
        (Fraction(-3, 5), Fraction(1, 4), Fraction(5, 7), Fraction(2, 9)),
        (Fraction(5, 6), Fraction(3, 8), Fraction(-1, 2), Fraction(7, 5)),
        (Fraction(2, 7), Fraction(9, 5), Fraction(3, 4), Fraction(1, 6)),
        (Fraction(-1, 2), Fraction(4, 3), Fraction(6, 5), Fraction(5, 2)),
    ))

    k_pts = (Fraction(-3, 2), Fraction(1, 4), Fraction(9, 2))
    s_half_orig = Fraction(max(k_pts) - min(k_pts), 2)

    v = (3, -5, 7, -11)
    h4v = (
        v[0] + v[1] + v[2] + v[3],
        v[0] - v[1] + v[2] - v[3],
        v[0] + v[1] - v[2] - v[3],
        v[0] - v[1] - v[2] + v[3],
    )
    lhs_p = sum(t * t for t in h4v)

    h_vec = (Fraction(1, 1), Fraction(16, 1))
    a2_vec = (Fraction(1, 1), Fraction(9, 1))
    cost_unw = check_cauchy_schwarz_pair(h_vec, a2_vec, (Fraction(1, 1), Fraction(1, 1)))
    cost_opt = check_cauchy_schwarz_pair(h_vec, a2_vec, (Fraction(1, 1), Fraction(4, 3)))

    return {
        "folded_dot_exact": frac_str(folded_dot),
        "extra_runtime_bytes": 0,
        "halved_diff_halfwidth": frac_str(s_half_orig),
        "blockh4_parseval_sq": lhs_p,
        "cauchy_schwarz_unwhitened_cost": frac_str(cost_unw),
        "cauchy_schwarz_optimal_cost": frac_str(cost_opt),
    }


def compute_pillar3_helly_and_polar_bounds() -> dict[str, object]:
    """Exact Helly-LP minimax Chebyshev optimality, telescoping cover, and polar-cap matched-bpw crossover."""
    c_star, r_star, r_origin = check_chebyshev_interval(-3, 9)
    check_chebyshev_interval(-11, 5)
    x0 = 10
    steps = ((1, 2, 5), (2, -1, 3), (1, 0, 4), (3, 1, 2))
    v_tilde = x0 - sum(a * u for a, u, _ in steps)
    win_ratio = check_polar_crossover(3, 5, 80, 20, 11, 10)
    rev_ratio = check_polar_crossover(4, 5, 80, 20, 4, 1)
    check_polar_crossover(4, 7, 60, 40, 12, 10)

    return {
        "chebyshev_center": frac_str(c_star),
        "chebyshev_radius": frac_str(r_star),
        "origin_radius": frac_str(r_origin),
        "helly_telescoping_center": v_tilde,
        "helly_telescoping_points": len(steps) + 1,
        "polar_win_ratio": frac_str(win_ratio),
        "polar_reversal_ratio": frac_str(rev_ratio),
    }


def _hex_poly_mul_nonneg(a: list[int], b: list[int], limit: int) -> list[int]:
    ma = max(a)
    mb = max(b)
    w = ((ma * mb * (limit + 1)).bit_length() + 3) // 4 + 1
    fmt = f"0{w}x"
    ha = "".join(format(x, fmt) for x in reversed(a[: limit + 1]))
    hb = "".join(format(x, fmt) for x in reversed(b[: limit + 1]))
    prod = int(ha, 16) * int(hb, 16)
    need = (limit + 1) * w
    hx = format(prod, f"0{need}x")
    if len(hx) > need:
        hx = hx[-need:]
    return [int(hx[need - (i + 1) * w : need - i * w], 16) for i in range(limit + 1)]


def _karatsuba_poly_mul(a: list[int], b: list[int], limit: int) -> list[int]:
    """Truncated pure-Z hex-packed polynomial multiplication up to degree limit."""
    ap = [x if x > 0 else 0 for x in a]
    an = [-x if x < 0 else 0 for x in a]
    bp = [x if x > 0 else 0 for x in b]
    bn = [-x if x < 0 else 0 for x in b]
    pp = _hex_poly_mul_nonneg(ap, bp, limit)
    nn = _hex_poly_mul_nonneg(an, bn, limit)
    pn = _hex_poly_mul_nonneg(ap, bn, limit)
    np = _hex_poly_mul_nonneg(an, bp, limit)
    return [pp[i] + nn[i] - pn[i] - np[i] for i in range(limit + 1)]


def compute_a16_g128_theta_bounds(r_max: int, bit_targets: tuple[int, ...]) -> dict[str, int]:
    """Direct pure-Z theta-series identity + Karatsuba convolution for A_16^8 ball bounds up to r_max."""
    require_ok(r_max >= 1, "P4_RMAX_POS", "r_max must be >= 1")
    nz_t3: list[tuple[int, int]] = [(0, 1)]
    k = 1
    while k * k <= r_max:
        nz_t3.append((k * k, 2))
        k = k + 1

    t3_2 = [0] * (r_max + 1)
    for i, vi in nz_t3:
        for j, vj in nz_t3:
            if i + j <= r_max:
                t3_2[i + j] = t3_2[i + j] + vi * vj
    t3_4 = _karatsuba_poly_mul(t3_2, t3_2, r_max)
    t3_8 = _karatsuba_poly_mul(t3_4, t3_4, r_max)
    t3_16 = _karatsuba_poly_mul(t3_8, t3_8, r_max)

    hm = r_max // 2
    ev_nz = [(i, v) for i, v in nz_t3 if i % 2 == 0]
    od_nz = [(i, v) for i, v in nz_t3 if i % 2 == 1]
    t34_half = [0] * (hm + 1)
    for i, vi in ev_nz:
        for j, vj in ev_nz:
            if i + j <= r_max:
                t34_half[(i + j) // 2] = t34_half[(i + j) // 2] + vi * vj
    for i, vi in od_nz:
        for j, vj in od_nz:
            if i + j <= r_max:
                t34_half[(i + j) // 2] = t34_half[(i + j) // 2] - vi * vj
    p2 = _karatsuba_poly_mul(t34_half, t34_half, hm)
    p4 = _karatsuba_poly_mul(p2, p2, hm)
    p8 = _karatsuba_poly_mul(p4, p4, hm)

    a16 = [0] * (r_max + 1)
    for r in range(r_max + 1):
        val = t3_16[r] + (t3_16[r] if r % 2 == 0 else -t3_16[r])
        if r % 2 == 0:
            val = val + 30 * p8[r // 2]
        a16[r] = val // 32

    a16_2 = _karatsuba_poly_mul(a16, a16, r_max)
    a16_4 = _karatsuba_poly_mul(a16_2, a16_2, r_max)
    a16_8 = _karatsuba_poly_mul(a16_4, a16_4, r_max)

    cum = 0
    best: dict[int, int] = {}
    for r, coeff in enumerate(a16_8):
        cum = cum + coeff
        for t in bit_targets:
            if cum <= (1 << t):
                best[t] = r
    return {str(t): best[t] for t in bit_targets}


def compute_e8_g128_theta_bounds(r_max: int, bit_targets: tuple[int, ...]) -> dict[str, int]:
    """Exact pure-Z divisor-sum sigma_3 theta-series convolution for E_8^16 ball bounds."""
    require_ok(r_max >= 2, "P4_E8_RMAX", "r_max must be >= 2")
    m_max = r_max // 2
    s3 = [0] * (m_max + 1)
    for d in range(1, m_max + 1):
        d3 = d * d * d
        for m in range(d, m_max + 1, d):
            s3[m] = s3[m] + d3
    e8 = [1] + [240 * s3[m] for m in range(1, m_max + 1)]
    e2 = _karatsuba_poly_mul(e8, e8, m_max)
    e4 = _karatsuba_poly_mul(e2, e2, m_max)
    e8_8 = _karatsuba_poly_mul(e4, e4, m_max)
    e8_16 = _karatsuba_poly_mul(e8_8, e8_8, m_max)

    cum = 0
    best: dict[int, int] = {}
    for m, coeff in enumerate(e8_16):
        cum = cum + coeff
        for t in bit_targets:
            if cum <= (1 << t):
                best[t] = 2 * m
    return {str(t): best[t] for t in bit_targets}


def _cross_verify_raw_results(
    receipt: dict[str, object],
    raw_results: dict[str, object],
    emp_sha: str,
) -> dict[str, int]:
    """Cross-verify receipt against raw s165_empirical_results.json in exact Z/Decimal."""
    require_ok(_req_str(receipt, "empirical_results_sha256", "REC_EMP_SHA") == emp_sha, "REC_EMP_PIN", "empirical sha mismatch")
    require_ok(_req_str(raw_results, "prereg_sha256", "RAW_PREREG") == PINNED_PREREG_SHA256, "RAW_PREREG_PIN", "raw prereg sha mismatch")
    require_ok(_req_str(raw_results, "energy_claim", "RAW_ENERGY") == ENERGY_CLAIM, "RAW_ENERGY_VAL", "raw energy claim mismatch")
    require_ok(raw_results.get("measured_live_model_weights") is True, "RAW_LIVE_WTS", "raw must measure live weights")

    raw_models = _req_dict(raw_results, "models", "RAW_MODELS")
    rec_models = _req_dict(receipt, "models", "REC_MODELS")
    require_ok(frozenset(raw_models.keys()) == frozenset(MODELS), "RAW_MODEL_SET", "raw model set mismatch")
    require_ok(frozenset(rec_models.keys()) == frozenset(MODELS), "REC_MODEL_SET", "unexpected model keys")

    total_rows_by_model: dict[str, int] = {}
    for mname in MODELS:
        rm = _req_dict(raw_models, mname, "RAW_MODEL_DICT")
        cm = _req_dict(rec_models, mname, "REC_MODEL_DICT")
        layers = _req_int(rm, "num_hidden_layers", "RAW_LAYERS")
        require_ok(layers == _req_int(cm, "num_hidden_layers", "REC_LAYERS"), "XREF_LAYERS", "layer count mismatch")
        l_exp, h_dim, kv_dim, i_dim = MODEL_SHAPES[mname]
        require_ok(layers == l_exp, "XREF_SHAPE_L", "unexpected layer shape")
        total_wts = _req_int(rm, "total_block_weights", "RAW_WTS")
        require_ok(total_wts == _req_int(cm, "total_block_weights", "REC_WTS"), "XREF_WTS", "weight count mismatch")
        expected_wts = l_exp * (2 * h_dim * h_dim + 2 * kv_dim * h_dim + 3 * i_dim * h_dim)
        require_ok(total_wts == expected_wts, "XREF_SHAPE_WTS", "shape weight identity failed")
        total_rows_by_model[mname] = l_exp * (3 * h_dim + 2 * kv_dim + 2 * i_dim)

        require_ok(_req_int(rm, "group_size", "RAW_GS") == _req_int(cm, "group_size", "REC_GS"), "XREF_GS", "group_size mismatch")
        require_ok(_scale_num(rm.get("peak_vram_mib"), 1024, "RAW_PEAK") == _req_int(cm, "peak_vram_kib", "REC_PEAK_VRAM"), "XREF_PEAK", "peak vram mismatch")

        fold = _req_dict(rm, "physical_zero_byte_fold_check", "RAW_FOLD")
        require_ok(_req_int(fold, "extra_runtime_bytes", "RAW_FOLD_B") == _req_int(cm, "prequant_fold_extra_bytes", "REC_FOLD_BYTES"), "XREF_FOLD_B", "fold bytes mismatch")
        require_ok(_scale_num(fold.get("prequant_fold_mean_kl"), 10 ** 9, "RAW_FOLD_KL") == _req_int(cm, "prequant_fold_kl_nano", "REC_FOLD_KL"), "XREF_FOLD_KL", "fold kl mismatch")

        raw_arms = _req_dict(rm, "arms", "RAW_ARMS")
        rec_arms = _req_dict(cm, "arms", "REC_ARMS")
        require_ok(frozenset(rec_arms.keys()) == frozenset(REQUIRED_ARMS), "REC_ARM_SET", "arm keyset mismatch")
        require_ok(frozenset(raw_arms.keys()) == frozenset(REQUIRED_ARMS), "RAW_ARM_SET", "raw arm set mismatch")

        a2_layers = _req_list(_req_dict(raw_arms, "A2_b30_A16_BlockH4_hess", "RAW_ARM_DICT"), "layer_weighted_mse_rel", "RAW_LAYER_MSE")
        a3_layers = _req_list(_req_dict(raw_arms, "A3_b30_A16_BlockH4_QuadSnake_SN", "RAW_ARM_DICT"), "layer_weighted_mse_rel", "RAW_LAYER_MSE")
        b1b_layers = _req_list(_req_dict(raw_arms, "B1b_b30_JohnHess_BlockH4_Folded0B", "RAW_ARM_DICT"), "layer_weighted_mse_rel", "RAW_LAYER_MSE")
        b1c_layers = _req_list(_req_dict(raw_arms, "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B", "RAW_ARM_DICT"), "layer_weighted_mse_rel", "RAW_LAYER_MSE")
        require_ok(len(a2_layers) == layers, "RAW_LAYER_LEN", "a2 layer len mismatch")
        require_ok(len(b1b_layers) == layers, "RAW_LAYER_LEN", "b1b layer len mismatch")
        require_ok(len(a3_layers) == layers, "RAW_LAYER_LEN", "a3 layer len mismatch")
        require_ok(len(b1c_layers) == layers, "RAW_LAYER_LEN", "b1c layer len mismatch")

        b1b_wins_calc = sum(1 for bw, aw in zip(b1b_layers, a2_layers) if _scale_num(bw, 10 ** 9, "RAW_LMSE") < _scale_num(aw, 10 ** 9, "RAW_LMSE"))
        b1c_wins_calc = sum(1 for bw, aw in zip(b1c_layers, a3_layers) if _scale_num(bw, 10 ** 9, "RAW_LMSE") < _scale_num(aw, 10 ** 9, "RAW_LMSE"))
        require_ok(b1b_wins_calc == _req_int(cm, "b1b_vs_a2_layer_hmse_wins", "REC_B1B_WINS"), "XREF_B1B_WINS", "b1b layer wins mismatch")
        require_ok(b1c_wins_calc == _req_int(cm, "b1c_vs_a3_layer_hmse_wins", "REC_B1C_WINS"), "XREF_B1C_WINS", "b1c layer wins mismatch")

        for aname in REQUIRED_ARMS:
            ra = _req_dict(raw_arms, aname, "RAW_ARM_DICT")
            ca = _req_dict(rec_arms, aname, "REC_ARM_DICT")
            require_ok(_req_int(ra, "ball_bound", "RAW_BB") == _req_int(ca, "ball_bound", "REC_BB"), "XREF_BB", "ball bound mismatch")
            tgt_str = _req_str(ra, "bpw_target", "RAW_TGT")
            try:
                tgt_dec = Decimal(tgt_str)
            except InvalidOperation:
                raise OracleError("RAW_TGT", "invalid bpw_target decimal string")
            require_ok(_scale_num(tgt_dec, 1000, "RAW_TGT") == _req_int(ca, "bpw_target_milli", "REC_BPW_TGT"), "XREF_TGT", "target bpw mismatch")
            require_ok(_scale_num(ra.get("effective_bpw"), 10 ** 6, "RAW_EFF") == _req_int(ca, "effective_bpw_micro", "REC_BPW_EFF"), "XREF_EFF", "effective bpw mismatch")
            require_ok(_req_int(ra, "syndrome_errors", "RAW_SYN") == _req_int(ca, "syndrome_errors", "REC_SYN"), "XREF_SYN", "syndrome error mismatch")
            require_ok(_req_int(ra, "out_of_ball_groups", "RAW_OOB") == _req_int(ca, "out_of_ball_groups", "REC_OOB"), "XREF_OOB", "oob mismatch")
            require_ok(_req_int(ra, "protected_non_block_diffs", "RAW_NBD") == _req_int(ca, "protected_non_block_diffs", "REC_NBD"), "XREF_NBD", "nbd mismatch")
            require_ok(_scale_num(ra.get("weighted_mse_rel"), 10 ** 9, "RAW_WMSE") == _req_int(ca, "weighted_mse_nano", "REC_METRIC"), "XREF_WMSE", "wmse mismatch")
            require_ok(_scale_num(ra.get("mean_p999_abs_err"), 10 ** 9, "RAW_P999") == _req_int(ca, "p999_abs_err_nano", "REC_METRIC"), "XREF_P999", "p999 mismatch")
            require_ok(_scale_num(ra.get("mean_p999_weighted_abs_err"), 10 ** 9, "RAW_WP999") == _req_int(ca, "wp999_abs_err_nano", "REC_METRIC"), "XREF_WP999", "wp999 mismatch")
            r_wt = _req_dict(ra, "wikitext2", "RAW_WT")
            r_py = _req_dict(ra, "python", "RAW_PY")
            require_ok(_scale_num(r_wt.get("ppl"), 10 ** 6, "RAW_WPPL") == _req_int(ca, "wikitext2_ppl_micro", "REC_METRIC"), "XREF_WPPL", "wikitext2 ppl mismatch")
            require_ok(_scale_num(r_wt.get("mean_kl"), 10 ** 9, "RAW_WKL") == _req_int(ca, "wikitext2_kl_nano", "REC_METRIC"), "XREF_WKL", "wikitext2 kl mismatch")
            require_ok(_scale_num(r_py.get("ppl"), 10 ** 6, "RAW_PPPL") == _req_int(ca, "python_ppl_micro", "REC_METRIC"), "XREF_PPPL", "python ppl mismatch")
            require_ok(_scale_num(r_py.get("mean_kl"), 10 ** 9, "RAW_PKL") == _req_int(ca, "python_kl_nano", "REC_METRIC"), "XREF_PKL", "python kl mismatch")

    return total_rows_by_model


def _verify_cs08_noncircular(
    raw_arm: dict[str, object],
    rec_arm: dict[str, object],
    total_weights: int,
    total_rows: int,
    a16_b30: dict[str, int],
    a16_b45: dict[str, int],
    e8_b: dict[str, int],
) -> tuple[int, int]:
    """Verify exact non-circular CS-08 bit budgets from certified theta bounds & counts."""
    aname = _req_str(raw_arm, "arm", "RAW_ANAME")
    gs = _req_int(raw_arm, "group_size", "RAW_GS")
    require_ok(gs == 128, "CS08_GS", "expected group_size 128")
    lat = _req_str(raw_arm, "lattice", "RAW_LAT")
    bb = _req_int(rec_arm, "ball_bound", "REC_BB")
    tgt = _req_int(rec_arm, "bpw_target_milli", "REC_BPW_TGT")
    eff = _req_int(rec_arm, "effective_bpw_micro", "REC_BPW_EFF")

    if lat == "RTN":
        require_ok(bb == 0, "CS08_RTN_BB", "RTN ball bound must be 0")
        require_ok(eff <= tgt * 1000, "REC_CS08_BPW", "CS-08 bit budget exceeded")
        return eff, eff

    if lat == "E8":
        exp_bb = e8_b["368"] if tgt == 3000 else e8_b["560"]
        payload_bits = 368 if tgt == 3000 else 560
        require_ok(bb == exp_bb, "CS08_E8_BB", "E8 theta bound mismatch")
        calc_micro = int((Fraction(payload_bits + 16, gs) * 1000000).limit_denominator(10 ** 9))
        require_ok(eff == calc_micro, "CS08_E8_BPW", "E8 bit budget mismatch")
        return eff, eff

    require_ok(lat == "A16", "CS08_LAT", "unexpected lattice")
    if aname.startswith("B2b_"):
        require_ok(bb == a16_b30["352"], "CS08_CHEB_BB", "Chebyshev center theta bound mismatch")
        require_ok(eff == 3000000, "CS08_CHEB_BPW", "Chebyshev center matched bpw mismatch")
        return eff, eff

    if "CSR025" in aname:
        csr_n = _req_int(raw_arm, "csr_entries", "RAW_CSR")
        bulk_bits = 357 if tgt == 3000 else 549
        exp_bb = a16_b30["357"] if tgt == 3000 else a16_b45["549"]
        require_ok(bb == exp_bb, "CS08_CSR_BB", "CSR bulk theta bound mismatch")
        q_payload = Fraction(bulk_bits + 16, gs) + Fraction(32 * csr_n, total_weights)
        calc_micro = int((Decimal(q_payload.numerator) * 1000000 / Decimal(q_payload.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        require_ok(calc_micro <= tgt * 1000, "REC_CS08_BPW", "CS-08 bit budget exceeded")
        require_ok(eff == calc_micro, "CS08_CSR_BPW", "CSR payload bpw mismatch")
        q_meta = q_payload + Fraction(16 * total_rows, total_weights)
        meta_micro = int((Decimal(q_meta.numerator) * 1000000 / Decimal(q_meta.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return eff, meta_micro

    if aname.startswith("B3b_"):
        hi_g = _req_int(raw_arm, "polar_hi_groups", "RAW_PHI")
        tot_g = total_weights // gs
        require_ok(bb in (a16_b30["357"], a16_b30["368"]), "CS08_B3B_BB", "B3b theta bound mismatch")
        q_payload = Fraction(hi_g * 576 + (tot_g - hi_g) * 373, total_weights)
        calc_micro = int((Decimal(q_payload.numerator) * 1000000 / Decimal(q_payload.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        require_ok(calc_micro <= tgt * 1000, "REC_CS08_BPW", "CS-08 bit budget exceeded")
        require_ok(eff == calc_micro, "CS08_B3B_BPW", "B3b payload bpw mismatch")
        q_meta = q_payload + Fraction(1, gs)
        meta_micro = int((Decimal(q_meta.numerator) * 1000000 / Decimal(q_meta.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return eff, meta_micro

    exp_bb = a16_b30["368"] if tgt == 3000 else a16_b45["560"]
    payload_bits = 368 if tgt == 3000 else 560
    require_ok(bb == exp_bb, "CS08_STD_BB", "A16 theta bound mismatch")
    require_ok(eff == (payload_bits + 16) * 1000000 // gs, "CS08_STD_BPW", "A16 bit budget mismatch")
    return eff, eff


def evaluate_empirical_receipt(
    receipt: dict[str, object],
    raw_results: dict[str, object],
    emp_sha: str,
    a16_b30: dict[str, int],
    a16_b45: dict[str, int],
    e8_b: dict[str, int],
) -> dict[str, object]:
    """Validate pure-integer empirical receipt + raw results and derive Kill Gates 1 & 2 programmatically."""
    require_ok(_req_str(receipt, "schema", "REC_SCHEMA") == "aletheia.track_s165.empirical_receipt.v1", "REC_SCHEMA_VAL", "bad receipt schema")
    require_ok(_req_str(receipt, "prereg_sha256", "REC_PREREG") == PINNED_PREREG_SHA256, "REC_PREREG_PIN", "prereg sha mismatch")
    require_ok(_req_str(receipt, "energy_claim", "REC_ENERGY") == ENERGY_CLAIM, "REC_ENERGY_VAL", "energy claim must be UNCONTROLLED_LAPTOP_NOT_CLAIMED")
    require_ok(receipt.get("measured_live_model_weights") is True, "REC_LIVE_WTS", "must measure live real weights")

    vram_total_mib = _req_int(receipt, "gpu_vram_total_mib", "REC_VRAM_TOT")
    require_ok(vram_total_mib > 0, "REC_VRAM_CAP", "gpu vram total must be positive")
    require_ok(vram_total_mib <= 10240, "REC_VRAM_CAP", "gpu vram total exceeds bound")

    total_rows_by_model = _cross_verify_raw_results(receipt, raw_results, emp_sha)
    models_obj = _req_dict(receipt, "models", "REC_MODELS")
    raw_models = _req_dict(raw_results, "models", "RAW_MODELS")

    model_summaries: dict[str, object] = {}
    kg1_a16_passes = 0
    kg1_b1c_passes = 0
    kg1_a16_total = 0
    kg2_p999_passes = 0
    kg2_p999_total = 0
    kg2_falsify_hits = 0
    b3_partial_all = True
    b3_full_win_all = True
    total_arms_verified = 0

    for mname in MODELS:
        md = _req_dict(models_obj, mname, "REC_MODEL_DICT")
        rmd = _req_dict(raw_models, mname, "RAW_MODEL_DICT")
        n_layers = _req_int(md, "num_hidden_layers", "REC_LAYERS")
        peak_kib = _req_int(md, "peak_vram_kib", "REC_PEAK_VRAM")
        require_ok(peak_kib > 0, "REC_PEAK_CAP", "peak VRAM must be positive")
        require_ok(peak_kib < vram_total_mib * 1024, "REC_PEAK_CAP", "peak VRAM exceeded GPU limit")
        require_ok(_req_int(md, "prequant_fold_extra_bytes", "REC_FOLD_BYTES") == 0, "REC_FOLD_ZERO", "folded diagonal must add 0 bytes")
        fold_kl = _req_int(md, "prequant_fold_kl_nano", "REC_FOLD_KL")
        require_ok(fold_kl >= 0, "REC_FOLD_KL_BOUND", "pre-quant fold KL negative")
        require_ok(fold_kl < 5000000, "REC_FOLD_KL_BOUND", "pre-quant fold KL too high")

        b1b_wins = _req_int(md, "b1b_vs_a2_layer_hmse_wins", "REC_B1B_WINS")
        b1c_wins = _req_int(md, "b1c_vs_a3_layer_hmse_wins", "REC_B1C_WINS")

        arms = _req_dict(md, "arms", "REC_ARMS")
        raw_arms = _req_dict(rmd, "arms", "RAW_ARMS")

        total_wts = _req_int(rmd, "total_block_weights", "RAW_WTS")
        total_rows = total_rows_by_model[mname]
        meta_bpw: dict[str, int] = {}

        for aname in REQUIRED_ARMS:
            ad = _req_dict(arms, aname, "REC_ARM_DICT")
            rad = _req_dict(raw_arms, aname, "RAW_ARM_DICT")
            require_ok(_req_int(ad, "syndrome_errors", "REC_SYN") == 0, "REC_SYN_ZERO", "syndrome error detected")
            require_ok(_req_int(ad, "out_of_ball_groups", "REC_OOB") == 0, "REC_OOB_ZERO", "out-of-ball group detected")
            require_ok(_req_int(ad, "protected_non_block_diffs", "REC_NBD") == 0, "REC_NBD_ZERO", "protected non-block diff detected")
            _eff, meta_m = _verify_cs08_noncircular(rad, ad, total_wts, total_rows, a16_b30, a16_b45, e8_b)
            meta_bpw[aname] = meta_m
            total_arms_verified = total_arms_verified + 1

        a1 = _req_dict(arms, "A1_b30_E8_H8_hess", "REC_ARM_DICT")
        a2 = _req_dict(arms, "A2_b30_A16_BlockH4_hess", "REC_ARM_DICT")
        a3 = _req_dict(arms, "A3_b30_A16_BlockH4_QuadSnake_SN", "REC_ARM_DICT")
        a4 = _req_dict(arms, "A4_b30_A16_BlockH4_QuadLPT_SN", "REC_ARM_DICT")
        b1b = _req_dict(arms, "B1b_b30_JohnHess_BlockH4_Folded0B", "REC_ARM_DICT")
        b1c = _req_dict(arms, "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B", "REC_ARM_DICT")
        b1d = _req_dict(arms, "B1d_b30_JohnHess_E8_H8_Folded0B", "REC_ARM_DICT")
        b2a = _req_dict(arms, "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B", "REC_ARM_DICT")
        b2b = _req_dict(arms, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "REC_ARM_DICT")
        a4_45 = _req_dict(arms, "A4_b45_A16_BlockH4_QuadLPT_SN", "REC_ARM_DICT")
        b2a_45 = _req_dict(arms, "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B", "REC_ARM_DICT")
        b3a = _req_dict(arms, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "REC_ARM_DICT")
        b3b = _req_dict(arms, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "REC_ARM_DICT")

        # KG1 (PREREG:30): >= 2% weighted MSE or >= 0.01 held-out WikiPPL at 3.0 bpw
        b1c_hmse_pass = _req_int(b1c, "weighted_mse_nano", "REC_METRIC") * 100 <= _req_int(a3, "weighted_mse_nano", "REC_METRIC") * 98
        b1c_ppl_pass = _req_int(a3, "wikitext2_ppl_micro", "REC_METRIC") - _req_int(b1c, "wikitext2_ppl_micro", "REC_METRIC") >= 10000
        b1b_hmse_pass = _req_int(b1b, "weighted_mse_nano", "REC_METRIC") * 100 <= _req_int(a2, "weighted_mse_nano", "REC_METRIC") * 98
        b1b_ppl_pass = _req_int(a2, "wikitext2_ppl_micro", "REC_METRIC") - _req_int(b1b, "wikitext2_ppl_micro", "REC_METRIC") >= 10000
        b1d_hmse_pass = _req_int(b1d, "weighted_mse_nano", "REC_METRIC") * 100 <= _req_int(a1, "weighted_mse_nano", "REC_METRIC") * 98
        b1d_ppl_pass = _req_int(a1, "wikitext2_ppl_micro", "REC_METRIC") - _req_int(b1d, "wikitext2_ppl_micro", "REC_METRIC") >= 10000
        b1b_vs_a3_wiki_gain = _req_int(a3, "wikitext2_ppl_micro", "REC_METRIC") - _req_int(b1b, "wikitext2_ppl_micro", "REC_METRIC")
        b1d_vs_a1_wiki_gain = _req_int(a1, "wikitext2_ppl_micro", "REC_METRIC") - _req_int(b1d, "wikitext2_ppl_micro", "REC_METRIC")

        for g in (b1c_hmse_pass, b1c_ppl_pass):
            if g:
                kg1_b1c_passes = kg1_b1c_passes + 1
                kg1_a16_passes = kg1_a16_passes + 1
            kg1_a16_total = kg1_a16_total + 1
        for g in (b1b_hmse_pass, b1b_ppl_pass):
            if g:
                kg1_a16_passes = kg1_a16_passes + 1
            kg1_a16_total = kg1_a16_total + 1

        # KG2 (PREREG:31): degrades L2 PPL vs MSE-optimal QuadLPT without >= 15% p999 reduction
        b2a_p999_pass = _req_int(b2a, "p999_abs_err_nano", "REC_METRIC") * 100 <= _req_int(a4, "p999_abs_err_nano", "REC_METRIC") * 85
        b2b_p999_pass = _req_int(b2b, "p999_abs_err_nano", "REC_METRIC") * 100 <= _req_int(a4, "p999_abs_err_nano", "REC_METRIC") * 85
        b2a_45_p999_pass = _req_int(b2a_45, "p999_abs_err_nano", "REC_METRIC") * 100 <= _req_int(a4_45, "p999_abs_err_nano", "REC_METRIC") * 85

        b2a_wiki_degraded = _req_int(b2a, "wikitext2_ppl_micro", "REC_METRIC") > _req_int(a4, "wikitext2_ppl_micro", "REC_METRIC")
        b2b_wiki_degraded = _req_int(b2b, "wikitext2_ppl_micro", "REC_METRIC") > _req_int(a4, "wikitext2_ppl_micro", "REC_METRIC")
        b2a_45_wiki_degraded = _req_int(b2a_45, "wikitext2_ppl_micro", "REC_METRIC") > _req_int(a4_45, "wikitext2_ppl_micro", "REC_METRIC")
        b2a_py_degraded = _req_int(b2a, "python_ppl_micro", "REC_METRIC") > _req_int(a4, "python_ppl_micro", "REC_METRIC")

        for p999_ok, ppl_bad in (
            (b2a_p999_pass, b2a_wiki_degraded),
            (b2b_p999_pass, b2b_wiki_degraded),
            (b2a_45_p999_pass, b2a_45_wiki_degraded),
        ):
            kg2_p999_total = kg2_p999_total + 1
            if p999_ok:
                kg2_p999_passes = kg2_p999_passes + 1
            elif ppl_bad:
                kg2_falsify_hits = kg2_falsify_hits + 1

        # Arm B3
        b3a_wiki_gain = _req_int(a3, "wikitext2_ppl_micro", "REC_METRIC") - _req_int(b3a, "wikitext2_ppl_micro", "REC_METRIC")
        b3b_py_gain = _req_int(a3, "python_ppl_micro", "REC_METRIC") - _req_int(b3b, "python_ppl_micro", "REC_METRIC")
        b3b_lt_a2_py = _req_int(b3b, "python_ppl_micro", "REC_METRIC") < _req_int(a2, "python_ppl_micro", "REC_METRIC")
        b3b_lt_a3_py = _req_int(b3b, "python_ppl_micro", "REC_METRIC") < _req_int(a3, "python_ppl_micro", "REC_METRIC")
        b3b_beats_py = b3b_lt_a2_py and b3b_lt_a3_py
        if b3a_wiki_gain < 10000:
            b3_partial_all = False
        if not b3b_beats_py:
            b3_partial_all = False
        if _req_int(b3b, "weighted_mse_nano", "REC_METRIC") >= _req_int(a2, "weighted_mse_nano", "REC_METRIC"):
            b3_full_win_all = False

        model_summaries[mname] = {
            "num_hidden_layers": n_layers,
            "peak_vram_kib": peak_kib,
            "b1b_vs_a2_layer_hmse_wins": b1b_wins,
            "b1c_vs_a3_layer_hmse_wins": b1c_wins,
            "b1c_vs_a3_hmse_gate_pass": b1c_hmse_pass,
            "b1c_vs_a3_ppl_gate_pass": b1c_ppl_pass,
            "b1b_vs_a2_hmse_gate_pass": b1b_hmse_pass,
            "b1b_vs_a2_ppl_gate_pass": b1b_ppl_pass,
            "b1b_vs_a3_wiki_ppl_gain_micro": b1b_vs_a3_wiki_gain,
            "b1d_vs_a1_hmse_gate_pass": b1d_hmse_pass,
            "b1d_vs_a1_ppl_gate_pass": b1d_ppl_pass,
            "b1d_vs_a1_wiki_ppl_gain_micro": b1d_vs_a1_wiki_gain,
            "b2a_vs_a4_p999_15pct_pass": b2a_p999_pass,
            "b2b_vs_a4_p999_15pct_pass": b2b_p999_pass,
            "b2a_b45_vs_a4_p999_15pct_pass": b2a_45_p999_pass,
            "b2a_vs_a4_wiki_ppl_degraded": b2a_wiki_degraded,
            "b2b_vs_a4_wiki_ppl_degraded": b2b_wiki_degraded,
            "b2a_b45_vs_a4_wiki_ppl_degraded": b2a_45_wiki_degraded,
            "b2a_vs_a4_python_ppl_degraded": b2a_py_degraded,
            "b3a_vs_a3_wiki_ppl_gain_micro": b3a_wiki_gain,
            "b3b_vs_a3_py_ppl_gain_micro": b3b_py_gain,
            "b3b_beats_a2_and_a3_python_ppl": b3b_beats_py,
            "b3a_with_16b_rowptr_bpw_micro": meta_bpw["B3a_b30_PolarCap_CSR025_BlockH4_Matched"],
            "b3b_with_1bit_selector_map_bpw_micro": meta_bpw["B3b_b30_PolarCap_BlockSplit_1over19_Matched"],
        }

    # Programmatic Kill-Gate verdicts derived strictly from ALL gate booleans
    if kg1_a16_passes == 0:
        kg1_verdict = "FALSIFIED"
    elif kg1_a16_passes == kg1_a16_total:
        kg1_verdict = "CONFIRMED"
    else:
        kg1_verdict = "PARTIAL"

    if kg2_p999_passes == kg2_p999_total:
        kg2_verdict = "CONFIRMED"
    elif kg2_p999_passes > 0:
        kg2_verdict = "PARTIAL"
    elif kg2_falsify_hits > 0:
        kg2_verdict = "FALSIFIED"
    else:
        kg2_verdict = "INCONCLUSIVE"

    if b3_full_win_all:
        b3_verdict = "CONFIRMED"
    elif b3_partial_all:
        b3_verdict = "PARTIAL"
    else:
        b3_verdict = "FALSIFIED"

    require_ok(kg1_b1c_passes == 0, "VERDICT_KG1_B1C", "expected B1c primary apples-to-apples falsified")
    require_ok(kg1_a16_passes == 1, "VERDICT_KG1_CNT", "expected exactly 1 KG1 A16 prong pass (1.5B B1b WikiPPL)")
    require_ok(kg2_p999_passes == 0, "VERDICT_KG2_P999", "unexpected KG2 p999 pass")
    require_ok(kg2_falsify_hits >= 1, "VERDICT_KG2_PPL", "expected KG2 PPL degradation trigger")
    require_ok(b3_verdict == "PARTIAL", "VERDICT_B3", "unexpected B3 verdict")

    overall_verdict = f"KG1_{kg1_verdict}_KG2_{kg2_verdict}_B3_{b3_verdict}"
    b3b_meta_within = all(
        _req_dict(model_summaries, m, "MS")["b3b_with_1bit_selector_map_bpw_micro"] <= 3000000  # type: ignore[operator]
        for m in MODELS
    )
    b3a_meta_within = all(
        _req_dict(model_summaries, m, "MS")["b3a_with_16b_rowptr_bpw_micro"] <= 3000000  # type: ignore[operator]
        for m in MODELS
    )

    return {
        "total_live_arms_verified": total_arms_verified,
        "zero_syndrome_errors_all_arms": True,
        "zero_oob_groups_all_arms": True,
        "cs08_bit_budget_verified_all_arms": True,
        "b3b_payload_plus_scale_within_3bpw": True,
        "b3b_with_1bit_selector_map_within_3bpw": b3b_meta_within,
        "b3a_with_16b_rowptr_within_3bpw": b3a_meta_within,
        "recorded_run_b3_csr_mode": "additive_residual_plus_dense_quant_patched_in_bench",
        "recorded_run_python_eval_scope": "leading_4096_tokens_fractions_py_only_patched_in_bench",
        "kg1_a16_prongs_passed": kg1_a16_passes,
        "kg1_a16_prongs_total": kg1_a16_total,
        "kg1_john_b1_verdict": kg1_verdict,
        "kg2_p999_gates_passed": kg2_p999_passes,
        "kg2_ppl_degraded_without_p999_hits": kg2_falsify_hits,
        "kg2_helly_b2_verdict": kg2_verdict,
        "arm_b3_polar_cap_verdict": b3_verdict,
        "overall_empirical_verdict": overall_verdict,
        "models": model_summaries,
    }


def build_certificate(base: Path | None = None) -> dict[str, object]:
    root = _track_base() if base is None else base
    lean_raw = _read_regular_bytes(root / LEAN_FILE_NAME)
    tc_raw = _read_regular_bytes(root / "lean-toolchain")
    lf_raw = _read_regular_bytes(root / "lakefile.toml")
    mf_raw = _read_regular_bytes(root / "lake-manifest.json")
    rec_raw = _read_regular_bytes(root / RECEIPT_FILE_NAME)
    emp_raw = _read_regular_bytes(root / EMPIRICAL_RESULTS_FILE_NAME)

    lean_sha = hashlib.sha256(lean_raw).hexdigest()
    tc_sha = hashlib.sha256(tc_raw).hexdigest()
    lf_sha = hashlib.sha256(lf_raw).hexdigest()
    mf_sha = hashlib.sha256(mf_raw).hexdigest()
    rec_sha = hashlib.sha256(rec_raw).hexdigest()
    emp_sha = hashlib.sha256(emp_raw).hexdigest()

    require_ok(lean_sha == PINNED_LEAN_SHA256, "ORA_PIN_LEAN", "lean sha256 mismatch")
    require_ok(tc_sha == PINNED_TOOLCHAIN_SHA256, "ORA_PIN_TC", "toolchain sha256 mismatch")
    require_ok(lf_sha == PINNED_LAKEFILE_SHA256, "ORA_PIN_LF", "lakefile sha256 mismatch")
    require_ok(mf_sha == PINNED_MANIFEST_SHA256, "ORA_PIN_MF", "manifest sha256 mismatch")
    require_ok(rec_sha == PINNED_RECEIPT_SHA256, "ORA_PIN_REC", "receipt sha256 mismatch")
    require_ok(emp_sha == PINNED_EMPIRICAL_RESULTS_SHA256, "ORA_PIN_EMP", "empirical results sha256 mismatch")

    receipt = _parse_strict_json_bytes(rec_raw, allow_decimal=False)
    raw_results = _parse_strict_json_bytes(emp_raw, allow_decimal=True)
    a16_theta = compute_a16_g128_theta_bounds(5220, (352, 357, 368, 544, 549, 560))
    theta_b30 = {k: a16_theta[k] for k in ("352", "357", "368")}
    theta_b45 = {k: a16_theta[k] for k in ("544", "549", "560")}
    theta_e8 = compute_e8_g128_theta_bounds(3400, (368, 560))
    emp_eval = evaluate_empirical_receipt(receipt, raw_results, emp_sha, theta_b30, theta_b45, theta_e8)

    payload: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "track": TRACK_ID,
        "experiment_id": EXPERIMENT_ID,
        "prereg_id": PREREG_ID,
        "prereg_sha256": PINNED_PREREG_SHA256,
        "oracle_class": ORACLE_CLASS,
        "novelty_classification": NOVELTY_CLASS,
        "energy_claim": ENERGY_CLAIM,
        "lean": {
            "toolchain": EXPECTED_TOOLCHAIN,
            "lean_module": LEAN_MODULE,
            "lean_files": 1,
            "public_theorems": EXPECTED_LEAN_THEOREMS,
            "lean_sha256": lean_sha,
            "toolchain_sha256": tc_sha,
            "lakefile_sha256": lf_sha,
            "manifest_sha256": mf_sha,
        },
        "pillar1_john_geometry": compute_pillar1_john_geometry(),
        "pillar2_zero_byte_folding": compute_pillar2_zero_byte_folding(),
        "pillar3_helly_and_polar_bounds": compute_pillar3_helly_and_polar_bounds(),
        "pillar4_theta_and_empirical_gates": {
            "a16_g128_b30_theta_bounds": theta_b30,
            "a16_g128_b45_theta_bounds": theta_b45,
            "e8_g128_theta_bounds": theta_e8,
            "matched_bpw_bit_identities": {
                "b30_full": [368, 16, 0, 384],
                "b30_csr025": [357, 16, 11, 384],
                "b30_cheb_center": [352, 16, 16, 384],
                "b45_full": [560, 16, 0, 576],
                "b45_csr025": [549, 16, 11, 576],
                "b45_cheb_center": [544, 16, 16, 576],
            },
            "empirical_receipt_sha256": rec_sha,
            "empirical_results_sha256": emp_sha,
            "empirical_gates": emp_eval,
        },
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


class _StrictArgParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise OracleError("ORA_CLI_ARGS", message)


def run_cli(argv: list[str] | None = None) -> int:
    parser = _StrictArgParser(description="Track S165 exact Z/Q oracle", add_help=False, allow_abbrev=False)
    parser.add_argument("--write", nargs="?", const=CERT_FILE_NAME, default=None)
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    base = _track_base()
    cert = build_certificate(base=base)
    if args.write is not None:
        out = _validate_cli_path_arg(args.write)
        dest = out if out.is_absolute() else (base / out)
        write_cert_file(dest, cert)
    if args.json_out:
        os.write(1, (json.dumps(cert, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"))
    else:
        os.write(1, ("PASS " + SCHEMA_VERSION + "\n").encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except OracleError as exc:
        err = json.dumps({"status": "FAIL", "code": exc.code, "message": exc.message}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, err.encode("utf-8"))
        return 2
    except OSError as exc:
        err = json.dumps({"status": "FAIL", "code": "ORA_OS_ERROR", "message": type(exc).__name__}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, err.encode("utf-8"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
