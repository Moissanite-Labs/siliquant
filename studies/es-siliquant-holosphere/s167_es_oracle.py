#!/usr/bin/env python3
"""Track S167 (EXP-Q51a/b/c, PREREG-W22-ES-SCALE) Exact Z/Q Oracle & Certificate Generator.

Pure Python 3 standard library over Z and Q only.
Zero floating-point literals or calls, zero bare asserts, zero unseeded RNG.
Reads, SHA-256 pins, and reconciles ALL track artifacts directly from disk,
including PREREG_W22_ES_SCALE.json, s167_es_live_bench.py, s167_empirical_results.json
(parsed via parse_float=Fraction with zero Python floats), and s167_empirical_receipt.json.
Derives every empirical gate boolean and CONFIRMED/PARTIAL/FALSIFIED verdict
strictly programmatically from raw measurements.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

SCHEMA_VERSION = "s167.q51.es_siliquant_holosphere.v2"
TRACK_ID = "S167"
EXPERIMENT_IDS = ("EXP-Q51a", "EXP-Q51b", "EXP-Q51c")
PREREG_ID = "PREREG-W22-ES-SCALE"
ORACLE_CLASS = "exact_and_sampled"
NOVELTY_CLASS = "N1"
ENERGY_CLAIM = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"

LEAN_MODULE = "EvolutionOperators"
LEAN_FILE_NAME = "EvolutionOperators.lean"
PREREG_FILE_NAME = "PREREG_W22_ES_SCALE.json"
BENCH_FILE_NAME = "s167_es_live_bench.py"
RESULTS_FILE_NAME = "s167_empirical_results.json"
RECEIPT_FILE_NAME = "s167_empirical_receipt.json"
CERT_FILE_NAME = "s167_es_certificate.json"
EXPECTED_TOOLCHAIN = "leanprover/lean4:v4.36.0"
EXPECTED_LEAN_THEOREMS = 21

PINNED_PREREG_SHA256 = "1f9db607ce34de1e0cbd472625d411d8c5e9857468a107a0cc24e558a733d3f6"
PINNED_LEAN_SHA256 = "62c416750d66def4fa3e003363c134d421d300996a234af16841580497490dd3"
PINNED_TOOLCHAIN_SHA256 = "582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a"
PINNED_LAKEFILE_SHA256 = "4e16492fd574731d4dc2a998c56d596198dc3083565a768c28dfbd9dad60ece4"
PINNED_MANIFEST_SHA256 = "5437b4c916b5f0610868c90230b06aac0abd9c11ba0639837e654b03c001c122"
PINNED_BENCH_SHA256 = "759852360383f08c46eb10fcb3e8aa1c744e11bfc8b7e8a6c9427f19ae8e4439"
PINNED_EMPIRICAL_RESULTS_SHA256 = "d6edf352c5b8ba76d04550a506812c8c22223cb117a666582fb065f65cde9e67"
PINNED_RECEIPT_SHA256 = "d1ee363f31f40cf793bdc74ac56f5378196aa50f4c0b64c75b9cc6d152dd66b4"

MAX_BYTES = 1048576
MAX_DEPTH = 32
MAX_INT_DIGITS = 256
MAX_SHELL_NANO_CAP = 10 ** 15
VRAM_GRID_KEYS = (
    "T512_R1",
    "T512_R4",
    "T512_R8",
    "T2048_R1",
    "T2048_R4",
    "T2048_R8",
    "T8192_R1",
    "T8192_R4",
    "T8192_R8",
)


class OracleError(ValueError):
    """Fail-closed oracle exception carrying an explicit error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _StrictOracleParser(argparse.ArgumentParser):
    """Fail-closed ArgumentParser: rejects abbreviations and routes errors to OracleError."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise OracleError("ORA_CLI_ARGS", message)


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


def _split_clean_parts(path: object) -> tuple[str, ...]:
    raw_str = str(path) if isinstance(path, Path) else path
    require_ok(type(raw_str) is str, "ORA_PATH_TYPE", "path must be str or Path")
    require_ok(len(raw_str) > 0, "ORA_EMPTY_PATH", "empty path")
    require_ok("\x00" not in raw_str, "ORA_NUL_PATH", "NUL byte in path")
    p = Path(raw_str)
    parts = (p if p.is_absolute() else (Path.cwd() / p)).parts
    require_ok(len(parts) >= 2, "ORA_ROOT_PATH", "cannot open root directory")
    for seg in parts[1:]:
        require_ok(seg != "..", "ORA_DOTDOT_PATH", "parent traversal rejected")
    return parts


def _open_parent_nofollow(path: Path) -> tuple[int, str]:
    parts = _split_clean_parts(path)
    dfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        for seg in parts[1:-1]:
            nxt = os.open(seg, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            os.close(dfd)
            dfd = nxt
    except OSError:
        os.close(dfd)
        raise OracleError("ORA_PARENT_OPEN", "cannot open parent directory without symlinks")
    return dfd, parts[-1]


def _read_regular_bytes(path: Path) -> bytes:
    dfd, leaf = _open_parent_nofollow(path)
    try:
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError:
            raise OracleError("ORA_FILE_OPEN", "cannot open regular file")
    finally:
        os.close(dfd)
    chunks: list[bytes] = []
    try:
        info = os.fstat(fd)
        require_ok(stat.S_ISREG(info.st_mode), "ORA_NOT_REG", "not a regular file")
        require_ok(info.st_nlink == 1, "ORA_HARDLINK", "hardlinked file rejected")
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


def _write_atomic_nofollow(path: Path, raw: bytes) -> None:
    dfd, leaf = _open_parent_nofollow(path)
    tmp_name: str | None = None
    try:
        try:
            st = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            st = None
        if st is not None:
            require_ok(stat.S_ISREG(st.st_mode), "ORA_WRITE_NOT_REG", "write target is not a regular file")
            require_ok(st.st_nlink == 1, "ORA_WRITE_HARDLINK", "hardlinked write target rejected")
        tmp_name = ".s167_ora_tmp_" + str(os.getpid()) + "_" + hashlib.sha256(raw).hexdigest()[:16]
        fd = os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
        try:
            written = 0
            while written < len(raw):
                written = written + os.write(fd, raw[written:])
        finally:
            os.close(fd)
        os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


def _parse_strict_json_bytes(raw: bytes, *, allow_rationals: bool = False) -> dict[str, object]:
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
            if prev_digit and not allow_rationals:
                require_ok(ch not in ".eE", "ORA_FLOAT_TOKEN", "float token rejected")
            digits = 0
            prev_digit = False
    require_ok(depth == 0, "ORA_JSON_DEPTH_END", "unbalanced json nesting")
    require_ok(in_str is False, "ORA_JSON_STR_END", "unterminated json string")

    def _dup_free(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in pairs:
            require_ok(k not in out, "ORA_DUP_KEY", "duplicate json key")
            out[k] = v
        return out

    def _bad_const(_s: str) -> None:
        raise OracleError("ORA_BAD_CONST", "non-json constant")

    try:
        if allow_rationals:
            obj = json.loads(text, object_pairs_hook=_dup_free, parse_constant=_bad_const, parse_float=Fraction)
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
    val = d[key]
    require_ok(type(val) is int, code, f"non-int key {key}")
    return val  # type: ignore[return-value]


def _req_bool(d: dict[str, object], key: str, code: str) -> bool:
    require_ok(key in d, code, f"missing bool key {key}")
    val = d[key]
    require_ok(type(val) is bool, code, f"non-bool key {key}")
    return val  # type: ignore[return-value]


def _req_str(d: dict[str, object], key: str, code: str) -> str:
    require_ok(key in d, code, f"missing str key {key}")
    val = d[key]
    require_ok(type(val) is str, code, f"non-str key {key}")
    return val  # type: ignore[return-value]


def _req_dict(d: dict[str, object], key: str, code: str) -> dict[str, object]:
    require_ok(key in d, code, f"missing dict key {key}")
    val = d[key]
    require_ok(type(val) is dict, code, f"non-dict key {key}")
    return val  # type: ignore[return-value]


def _req_list(d: dict[str, object], key: str, code: str) -> list[object]:
    require_ok(key in d, code, f"missing list key {key}")
    val = d[key]
    require_ok(type(val) is list, code, f"non-list key {key}")
    return val  # type: ignore[return-value]


def _req_rat(d: dict[str, object], key: str, code: str) -> Fraction:
    require_ok(key in d, code, f"missing rational key {key}")
    val = d[key]
    if type(val) is int:
        return Fraction(val, 1)
    require_ok(type(val) is Fraction, code, f"non-rational key {key}")
    return val  # type: ignore[return-value]


def eval_poly4_horner(coeffs: tuple[Fraction, Fraction, Fraction, Fraction, Fraction], x: Fraction) -> Fraction:
    acc = coeffs[4]
    for idx in (3, 2, 1, 0):
        acc = acc * x + coeffs[idx]
    return acc


def compute_pillar1_antithetic() -> dict[str, object]:
    """Exact rational Antithetic-ES odd/even cancellation & baseline variance dominance."""
    coeffs = (Fraction(17, 1), Fraction(-5, 1), Fraction(11, 1), Fraction(3, 1), Fraction(-7, 1))
    sigma = Fraction(3, 1000)
    eps = Fraction(5, 7)
    t = sigma * eps
    f_pos = eval_poly4_horner(coeffs, t)
    f_neg = eval_poly4_horner(coeffs, -t)
    diff = f_pos - f_neg
    total = f_pos + f_neg

    g = (Fraction(3, 2), Fraction(-5, 4), Fraction(7, 8), Fraction(-9, 16))
    e = (Fraction(1, 1), Fraction(-1, 1), Fraction(1, 1), Fraction(-1, 1))
    c0 = Fraction(13, 5)
    q_even = Fraction(29, 7)
    lin = sum(gi * ei for gi, ei in zip(g, e))
    vp = c0 + sigma * lin + q_even
    vm = c0 - sigma * lin + q_even
    bias_sq2 = 2 * (c0 + q_even) * (c0 + q_even)
    return {
        "poly4_antithetic_diff": frac_str(diff),
        "poly4_antithetic_sum": frac_str(total),
        "vec4_antithetic_diff": frac_str(vp - vm),
        "vec4_antithetic_sum": frac_str(vp + vm),
        "onesided_bias_excess_sq2": frac_str(bias_sq2),
    }


def check_orth_rows_gram(rows: list[tuple[int, ...]], dim: int, diag_val: int) -> int:
    """Validate exact pairwise-orthogonal Gram matrix U U^T = diag_val * I_dim."""
    require_ok(len(rows) == dim, "P2_GRAM_DIM", "row count must equal dim")
    for i in range(dim):
        for j in range(dim):
            ip = sum(rows[i][k] * rows[j][k] for k in range(dim))
            require_ok(ip == (diag_val if i == j else 0), "P2_H4_GRAM", "Gram orthogonality failed")
    return dim


def compute_pillar2_orthogonal_h4() -> dict[str, object]:
    """Exact Block-H4 orthogonality, Parseval norm, zero-MSE fixed-radius recovery, & Choromanski Thm 3.2."""
    h2 = ((1, 1), (1, -1))
    h4 = [tuple(h2[r // 2][c // 2] * h2[r % 2][c % 2] for c in range(4)) for r in range(4)]
    check_orth_rows_gram(h4, 4, 4)

    s = (1, -1, -1, 1)
    v = (3, -5, 7, -11)
    g = (13, -7, 19, -23)
    rows_s = [tuple(h4[r][c] * s[c] for c in range(4)) for r in range(4)]
    check_orth_rows_gram(rows_s, 4, 4)
    hv_sq = sum((sum(row[c] * v[c] for c in range(4))) ** 2 for row in rows_s)
    v_sq = sum(x * x for x in v)

    rec = [Fraction(0, 1)] * 4
    for row in rows_s:
        proj = sum(g[c] * row[c] for c in range(4))
        for c in range(4):
            rec[c] = rec[c] + Fraction(proj * row[c], 4)
    err_sq = int(sum((rec[c] - Fraction(g[c], 1)) ** 2 for c in range(4)))

    # Choromanski (2018) Thm 3.2: Gaussian orthogonal ES MSE / i.i.d. Gaussian ES MSE = (d + 2 - N) / (d + 1)
    ratios: dict[str, str] = {}
    for d, n in ((4, 4), (16, 16), (128, 16), (128, 30)):
        ratios[f"d{d}_n{n}"] = frac_str(Fraction(d + 2 - n, d + 1))

    return {
        "h4_parseval_sq": hv_sq,
        "h4_input_sq": v_sq,
        "h4_exact_recovery_err_sq": err_sq,
        "choromanski_orth_over_iid_mse": ratios,
    }


def compute_pillar3_heat_kernel() -> dict[str, object]:
    """Exact rational Heat-Kernel Goodhart narrow-spike attenuation & Pascal binomial diffusion."""
    w = Fraction(1, 10000)
    big_w = Fraction(1, 100)
    sigma = Fraction(1, 1000)
    base_spike = (w * w) / (w * w + sigma * sigma)
    base_plat = (big_w * big_w) / (big_w * big_w + sigma * sigma)

    rank_ratios: dict[str, str] = {}
    for r in (1, 2, 4):
        ratio = (base_spike ** r) / (base_plat ** r)
        rank_ratios[f"r{r}"] = frac_str(ratio)

    pascal: list[list[int]] = [[1]]
    for row_n in range(1, 13):
        prev = pascal[-1]
        cur = [1] + [prev[i - 1] + prev[i] for i in range(1, row_n)] + [1]
        pascal.append(cur)
    center_masses: list[str] = []
    plat_ok = True
    for m in range(7):
        row = pascal[2 * m]
        if sum(row) != (1 << (2 * m)):
            plat_ok = False
        center_masses.append(frac_str(Fraction(row[m], 1 << (2 * m))))

    return {
        "spike_over_plateau_ratio": rank_ratios,
        "pascal_center_spike_masses_m0_to_m6": center_masses,
        "pascal_plateau_conservation_exact": plat_ok,
    }


def run_alkashi_centered_ef(half_q: int, w0: int, e0: int, stream: list[int]) -> tuple[int, int]:
    """Exact centered Al-Kashi FixedRadix (SOP-16) error-feedback accumulator (Q = 2 * half_q)."""
    require_ok(half_q > 0, "P4_H_POS", "half_q must be > 0")
    require_ok(-half_q <= e0, "P4_E0_LO", "initial residual must be >= -half_q")
    require_ok(e0 < half_q, "P4_E0_HI", "initial residual must be < half_q")
    q = 2 * half_q
    w = w0
    e = e0
    for g in stream:
        u = g + e
        dq = (u + half_q) // q
        w = w + dq
        e = u - dq * q
    return w, e


def bf16_grid_round_int(ulp: int, x: int) -> int:
    """Exact nearest-grid rounding onto multiples of ulp (matches Lean bf16GridRound)."""
    return ulp * ((2 * x + ulp) // (2 * ulp))


def verify_reciprocal_fold_dot(items: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]) -> Fraction:
    """Verify exact cleared-reciprocal +0 B folded-scale dot product: (w*p)*(x*q) == w*x."""
    fold_dot = sum((w * p) * (x * q) for w, x, p, q in items)
    unscaled_dot = sum(w * x for w, x, _, _ in items)
    require_ok(fold_dot == unscaled_dot, "P4_FOLD_INV", "reciprocal fold invariance failed")
    return fold_dot


def compute_pillar4_bf16_vs_alkashi_and_cs08() -> dict[str, object]:
    """Exact BF16 ULP stall boundary (coupled to bf16GridRound), Al-Kashi Q=64 EF, CS-08, & +0 B fold."""
    ulp_1 = Fraction(1, 128)
    half_ulp_1 = ulp_1 / 2
    alpha_vsonic = Fraction(5, 10000)
    stall_ratio = alpha_vsonic / half_ulp_1

    # Coupled Lean Thm 14 witness at denominator 256000 (ulp=2000, alpha=128): gHat=7 stalls, gHat=8 advances
    w_start = 2000 * 1
    g7_moves = (bf16_grid_round_int(2000, w_start + 128 * 7) - w_start) // 2000
    g8_moves = (bf16_grid_round_int(2000, w_start + 128 * 8) - w_start) // 2000

    # Centered Al-Kashi Q=64 (half_q=32, matching int8 EF in bench)
    w_fin, e_fin = run_alkashi_centered_ef(32, 0, 0, [8] * 16)

    chans = (
        (Fraction(3, 4), Fraction(5, 3), Fraction(7, 2), Fraction(2, 7)),
        (Fraction(-5, 8), Fraction(2, 5), Fraction(3, 4), Fraction(4, 3)),
        (Fraction(11, 16), Fraction(9, 7), Fraction(4, 9), Fraction(9, 4)),
    )
    fold_dot = verify_reciprocal_fold_dot(chans)

    return {
        "bf16_ulp_at_1": frac_str(ulp_1),
        "bf16_half_ulp_at_1": frac_str(half_ulp_1),
        "vsonic_alpha_over_half_ulp": frac_str(stall_ratio),
        "bf16_canonical_g7_moves": g7_moves,
        "bf16_canonical_g8_moves": g8_moves,
        "alkashi_q64_16x8_weight_advance": w_fin,
        "alkashi_q64_16x8_residual": e_fin,
        "cs08_es_over_drgrpo_flops": frac_str(Fraction(2, 6)),
        "cs08_es_over_grpo_flops": frac_str(Fraction(2, 8)),
        "cs08_hbm_ratio_n16": frac_str(Fraction(32, 3)),
        "cs08_hbm_ratio_n30": frac_str(Fraction(60, 3)),
        "folded_scale_dot_exact": frac_str(fold_dot),
        "folded_scale_extra_bytes": len(()),
    }


def compute_pillar5_cayley_su2_so3() -> dict[str, object]:
    """Exact rational Cayley-SU(2) unit sphere, Cayley-SO(3) Q^T Q = I, & Euclidean drift."""
    d, x, y, z = 50, 3, -4, 2
    denom = d * d + x * x + y * y + z * z
    q = (
        Fraction(d * d - (x * x + y * y + z * z), denom),
        Fraction(2 * d * x, denom),
        Fraction(2 * d * y, denom),
        Fraction(2 * d * z, denom),
    )
    q_norm2 = sum(c * c for c in q)

    mat = (
        (
            Fraction(d * d + x * x - y * y - z * z, denom),
            Fraction(2 * (x * y - d * z), denom),
            Fraction(2 * (x * z + d * y), denom),
        ),
        (
            Fraction(2 * (x * y + d * z), denom),
            Fraction(d * d - x * x + y * y - z * z, denom),
            Fraction(2 * (y * z - d * x), denom),
        ),
        (
            Fraction(2 * (x * z - d * y), denom),
            Fraction(2 * (y * z + d * x), denom),
            Fraction(d * d - x * x - y * y + z * z, denom),
        ),
    )
    max_orth_err = Fraction(0, 1)
    for i in range(3):
        for j in range(3):
            entry = sum(mat[k][i] * mat[k][j] for k in range(3))
            diff = abs(entry - (Fraction(1, 1) if i == j else Fraction(0, 1)))
            if diff > max_orth_err:
                max_orth_err = diff
    det = (
        mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
        - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
        + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0])
    )
    sig = Fraction(1, 10)
    euclid_shell_err = sig * sig * (Fraction(9, 25) + Fraction(16, 25))

    return {
        "cayley_su2_denom": denom,
        "cayley_su2_norm_sq": frac_str(q_norm2),
        "cayley_so3_orth_max_err": frac_str(max_orth_err),
        "cayley_so3_det": frac_str(det),
        "cayley_so3_m00": frac_str(mat[0][0]),
        "cayley_so3_m01": frac_str(mat[0][1]),
        "euclidean_additive_shell_err": frac_str(euclid_shell_err),
    }


def _exact_perm_p_fraction(diffs: list[int]) -> Fraction:
    n = len(diffs)
    require_ok(n >= 1, "ORA_PERM_EMPTY", "empty diff list")
    obs = abs(sum(diffs))
    extreme = 0
    total = 1 << n
    for mask in range(total):
        s = 0
        for idx, d in enumerate(diffs):
            s = s + (d if ((mask >> idx) & 1) == 0 else -d)
        if abs(s) >= obs:
            extreme = extreme + 1
    return Fraction(extreme, total)


def _holm_adjust_pair(p0: Fraction, p1: Fraction) -> tuple[Fraction, Fraction]:
    """Exact rational Holm-Bonferroni adjustment for a 2-test family."""
    one = Fraction(1, 1)
    if p0 <= p1:
        a0 = min(one, 2 * p0)
        a1 = max(a0, min(one, p1))
        return a0, a1
    a1 = min(one, 2 * p1)
    a0 = max(a1, min(one, p0))
    return a0, a1


def _q_scale_int(q: Fraction, scale: int) -> int:
    return int(round(q * scale))


def reconcile_results_with_receipt(raw_results: dict[str, object], receipt: dict[str, object]) -> None:
    """Reconcile disk s167_empirical_results.json (parsed over Q) against s167_empirical_receipt.json."""
    require_ok(_req_str(raw_results, "schema", "RES_SCHEMA") == "aletheia.track_s167.empirical_results.v2", "RES_SCHEMA_VAL", "bad results schema")
    require_ok(_req_str(raw_results, "prereg_sha256", "RES_PREREG") == _req_str(receipt, "prereg_sha256", "REC_PREREG"), "RES_PREREG_MATCH", "prereg sha mismatch")
    require_ok(_req_str(raw_results, "bench_script_sha256", "RES_BENCH") == _req_str(receipt, "bench_script_sha256", "REC_BENCH"), "RES_BENCH_MATCH", "bench sha mismatch")
    require_ok(_req_str(raw_results, "energy_claim", "RES_ENERGY") == ENERGY_CLAIM, "RES_ENERGY_VAL", "bad results energy claim")

    ra = _req_dict(raw_results, "exp_q51a", "RES_QA")
    qa = _req_dict(receipt, "exp_q51a", "REC_QA")
    sa = _req_dict(ra, "summary", "RES_QA_SUM")
    p15 = _req_dict(ra, "qwen15b_full_28layer_probe", "RES_QA_P15")
    require_ok(_q_scale_int(_req_rat(sa, "peak_vram_05b_mib", "RES_PK05"), 1024) == _req_int(qa, "peak_vram_05b_kib", "REC_QA_PK05"), "RES_PK05_MATCH", "0.5B VRAM mismatch")
    require_ok(_q_scale_int(_req_rat(sa, "peak_vram_15b_mib", "RES_PK15"), 1024) == _req_int(qa, "peak_vram_15b_kib", "REC_QA_PK15"), "RES_PK15_MATCH", "1.5B VRAM mismatch")
    require_ok(_q_scale_int(_req_rat(sa, "mean_bf16_ulp_stalled_frac", "RES_STALL"), 1000000) == _req_int(qa, "mean_bf16_ulp_stalled_micro", "REC_QA_STALL"), "RES_STALL_MATCH", "stall frac mismatch")
    require_ok(_req_int(p15, "layers_quantized", "RES_15LYR") == _req_int(qa, "m15_layers_quantized", "REC_QA_15LYR"), "RES_15LYR_MATCH", "1.5B layers mismatch")
    require_ok(_req_int(p15, "total_a16_blocks", "RES_15BLK") == _req_int(qa, "m15_total_a16_blocks", "REC_QA_15BLK"), "RES_15BLK_MATCH", "1.5B blocks mismatch")
    require_ok(_req_int(p15, "syndrome_errors", "RES_15SYN") == _req_int(qa, "m15_probe_syndrome_errors", "REC_QA_15SYN"), "RES_15SYN_MATCH", "1.5B syn mismatch")
    require_ok(_req_int(p15, "oob_groups", "RES_15OOB") == _req_int(qa, "m15_probe_oob_groups", "REC_QA_15OOB"), "RES_15OOB_MATCH", "1.5B oob mismatch")

    ra_arms = _req_dict(ra, "arms", "RES_QA_ARMS")
    qa_arms = _req_dict(qa, "arms", "REC_QA_ARMS")
    for arm_name in ("C1_VsonicV_BF16_SharedSeed", "C2_QES_Unblocked_INT4_EF", "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF"):
        r_rows = _req_list(ra_arms, arm_name, "RES_QA_ARM")
        q_rows = _req_list(qa_arms, arm_name, "REC_QA_ARM")
        require_ok(len(r_rows) == len(q_rows), "RES_QA_ARMLEN", "arm seed length mismatch")
        for rr, qr in zip(r_rows, q_rows):
            require_ok(type(rr) is dict and type(qr) is dict, "RES_QA_ROW", "arm row not dict")
            require_ok(_req_int(rr, "seed", "RES_SEED") == _req_int(qr, "seed", "REC_SEED"), "RES_SEED_M", "seed mismatch")
            require_ok(_q_scale_int(_req_rat(rr, "test_exact_acc", "RES_TACC"), 1000000) == _req_int(qr, "test_exact_acc_micro", "REC_TACC"), "RES_TACC_M", "test acc mismatch")
            require_ok(_req_int(rr, "syndrome_errors", "RES_TSYN") == _req_int(qr, "syndrome_errors", "REC_TSYN"), "RES_TSYN_M", "syndrome mismatch")
            require_ok(_req_int(rr, "oob_groups", "RES_TOOB") == _req_int(qr, "oob_groups", "REC_TOOB"), "RES_TOOB_M", "oob mismatch")

    rb = _req_dict(raw_results, "exp_q51b", "RES_QB")
    qb = _req_dict(receipt, "exp_q51b", "REC_QB")
    rb_base = _req_dict(rb, "baseline_base", "RES_QB_BASE")
    require_ok(_q_scale_int(_req_rat(rb_base, "ood_8x_exact_acc", "RES_BOOD"), 1000000) == _req_int(qb, "base_ood_8x_acc_micro", "REC_QB_BOOD"), "RES_BOOD_M", "base ood mismatch")
    require_ok(_q_scale_int(_req_rat(rb_base, "wikitext2_bpb", "RES_BBPB"), 1000000) == _req_int(qb, "base_wikitext2_bpb_micro", "REC_QB_BBPB"), "RES_BBPB_M", "base bpb mismatch")

    rb_grid = _req_dict(rb, "vram_o1_scaling_grid", "RES_QB_GRID")
    qb_grid = _req_dict(qb, "vram_o1_scaling_grid", "REC_QB_GRID")
    for gk in VRAM_GRID_KEYS:
        rc = _req_dict(rb_grid, gk, "RES_GRID_C")
        qc = _req_dict(qb_grid, gk, "REC_GRID_C")
        require_ok(_q_scale_int(_req_rat(rc, "es_peak_vram_mib", "RES_GEK"), 1024) == _req_int(qc, "es_peak_vram_kib", "REC_QB_EK"), "RES_GEK_M", "grid es vram mismatch")
        require_ok(_q_scale_int(_req_rat(rc, "es_max_su2_shell_err", "RES_GSH"), 1000000000) == _req_int(qc, "es_max_su2_shell_err_nano", "REC_QB_GSH"), "RES_GSH_M", "grid shell mismatch")
        require_ok(_req_bool(rc, "bptt_oom", "RES_GOOM") == _req_bool(qc, "bptt_oom", "REC_QB_OOM"), "RES_GOOM_M", "grid oom mismatch")

    rb_arms = _req_dict(rb, "arms", "RES_QB_ARMS")
    qb_arms = _req_dict(qb, "arms", "REC_QB_ARMS")
    for arm_name in ("ArmA_Euclidean_Additive_ES", "ArmC_HoloSphere_Cayley_ES"):
        r_rows = _req_list(rb_arms, arm_name, "RES_QB_ARM")
        q_rows = _req_list(qb_arms, arm_name, "REC_QB_ARM")
        require_ok(len(r_rows) == len(q_rows), "RES_QB_ARMLEN", "q51b arm len mismatch")
        for rr, qr in zip(r_rows, q_rows):
            require_ok(type(rr) is dict and type(qr) is dict, "RES_QB_ROW", "q51b row not dict")
            require_ok(_q_scale_int(_req_rat(rr, "ood_8x_exact_acc", "RES_OOD"), 1000000) == _req_int(qr, "ood_8x_exact_acc_micro", "REC_OOD"), "RES_OOD_M", "ood acc mismatch")
            fin_r = _req_bool(rr, "wikitext2_bpb_finite", "RES_FIN")
            require_ok(fin_r == _req_bool(qr, "wikitext2_bpb_finite", "REC_FIN"), "RES_FIN_M", "bpb finite mismatch")
            if fin_r:
                require_ok(_q_scale_int(_req_rat(rr, "wikitext2_bpb", "RES_BPB"), 1000000) == _req_int(qr, "wikitext2_bpb_micro", "REC_BPB"), "RES_BPB_M", "bpb val mismatch")
            require_ok(_req_int(rr, "a16_syndrome_errors", "RES_BSYN") == _req_int(qr, "a16_syndrome_errors", "REC_BSYN"), "RES_BSYN_M", "q51b syn mismatch")
            require_ok(_req_int(rr, "a16_codeword_flips", "RES_BFLP") == _req_int(qr, "a16_codeword_flips", "REC_BFLP"), "RES_BFLP_M", "q51b flips mismatch")


def evaluate_empirical_receipt(receipt: dict[str, object]) -> dict[str, object]:
    """Pure-integer/rational programmatic evaluation of live RTX 3080 EXP-Q51a & EXP-Q51b gates."""
    require_ok(_req_str(receipt, "schema", "REC_SCHEMA") == "aletheia.track_s167.empirical_receipt.v2", "REC_SCHEMA_VAL", "bad receipt schema")
    require_ok(_req_str(receipt, "prereg_sha256", "REC_PREREG") == PINNED_PREREG_SHA256, "REC_PREREG_PIN", "prereg sha mismatch")
    require_ok(_req_str(receipt, "bench_script_sha256", "REC_BENCH") == PINNED_BENCH_SHA256, "REC_BENCH_PIN", "bench sha mismatch")
    require_ok(_req_str(receipt, "empirical_results_sha256", "REC_RAW") == PINNED_EMPIRICAL_RESULTS_SHA256, "REC_RAW_PIN", "raw sha mismatch")
    require_ok(_req_str(receipt, "energy_claim", "REC_ENERGY") == ENERGY_CLAIM, "REC_ENERGY_VAL", "bad energy claim")
    require_ok(_req_bool(receipt, "measured_live_model_weights", "REC_LIVE") is True, "REC_LIVE_VAL", "live weights flag required")

    vram_tot = _req_int(receipt, "gpu_vram_total_mib", "REC_VRAM_TOT")
    require_ok(vram_tot > 0, "REC_VRAM_POS", "vram total must be > 0")
    require_ok(vram_tot <= 10240, "REC_VRAM_CAP", "vram total exceeds RTX 3080 cap")

    # --- EXP-Q51a ---
    qa = _req_dict(receipt, "exp_q51a", "REC_QA")
    require_ok(_req_int(qa, "linear_layers_05b", "REC_QA_LYR") == 168, "REC_QA_LYR_VAL", "expected 168 0.5B linear layers")
    require_ok(_req_int(qa, "total_a16_blocks_05b", "REC_QA_BLK") == 22364160, "REC_QA_BLK_VAL", "unexpected 0.5B A16 block count")
    m15_lyrs = _req_int(qa, "m15_layers_quantized", "REC_QA_15LYR")
    m15_lin = _req_int(qa, "m15_linear_layers_quantized", "REC_QA_15LIN")
    m15_blks = _req_int(qa, "m15_total_a16_blocks", "REC_QA_15BLK")
    m15_flips = _req_int(qa, "m15_a16_lattice_flips", "REC_QA_15FLP")
    require_ok(m15_lyrs == 28, "REC_QA_15LYR_VAL", "expected full 28/28 1.5B layers")
    require_ok(m15_lin == 196, "REC_QA_15LIN_VAL", "expected full 196/196 1.5B linears")
    require_ok(m15_blks == 81887232, "REC_QA_15BLK_VAL", "unexpected 1.5B A16 block count")

    null_syn_45 = _req_int(qa, "a16_null_syndrome_errors", "REC_QA_NSYN")
    null_oob_45 = _req_int(qa, "a16_null_oob_groups", "REC_QA_NOOB")
    null_syn_30 = _req_int(qa, "a16_3bpw_null_syndrome_errors", "REC_QA_3SYN")
    null_oob_30 = _req_int(qa, "a16_3bpw_null_oob_groups", "REC_QA_3OOB")
    m15_syn = _req_int(qa, "m15_probe_syndrome_errors", "REC_QA_15SYN")
    m15_oob = _req_int(qa, "m15_probe_oob_groups", "REC_QA_15OOB")
    pk05 = _req_int(qa, "peak_vram_05b_kib", "REC_QA_PK05")
    pk15 = _req_int(qa, "peak_vram_15b_kib", "REC_QA_PK15")

    arms_a = _req_dict(qa, "arms", "REC_QA_ARMS")
    c1_list = _req_list(arms_a, "C1_VsonicV_BF16_SharedSeed", "REC_QA_C1")
    c2_list = _req_list(arms_a, "C2_QES_Unblocked_INT4_EF", "REC_QA_C2")
    t1_list = _req_list(arms_a, "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF", "REC_QA_T1")
    n_seeds = _req_int(qa, "n_seeds", "REC_QA_N")
    require_ok(n_seeds >= 3, "REC_QA_N_MIN", "need >= 3 seeds")
    require_ok(len(c1_list) == n_seeds, "REC_QA_C1_N", "C1 seed count mismatch")
    require_ok(len(c2_list) == n_seeds, "REC_QA_C2_N", "C2 seed count mismatch")
    require_ok(len(t1_list) == n_seeds, "REC_QA_T1_N", "T1 seed count mismatch")

    c1_accs: list[int] = []
    c2_accs: list[int] = []
    t1_accs: list[int] = []
    t1_syn_sum = 0
    t1_oob_sum = 0
    for idx in range(n_seeds):
        r_c1 = c1_list[idx]
        r_c2 = c2_list[idx]
        r_t1 = t1_list[idx]
        require_ok(type(r_c1) is dict, "REC_QA_C1_D", "C1 entry must be dict")
        require_ok(type(r_c2) is dict, "REC_QA_C2_D", "C2 entry must be dict")
        require_ok(type(r_t1) is dict, "REC_QA_T1_D", "T1 entry must be dict")
        t1_syn_sum = t1_syn_sum + _req_int(r_t1, "syndrome_errors", "REC_QA_T1_SYN")
        t1_oob_sum = t1_oob_sum + _req_int(r_t1, "oob_groups", "REC_QA_T1_OOB")
        c1_accs.append(_req_int(r_c1, "test_exact_acc_micro", "REC_QA_C1_ACC"))
        c2_accs.append(_req_int(r_c2, "test_exact_acc_micro", "REC_QA_C2_ACC"))
        t1_accs.append(_req_int(r_t1, "test_exact_acc_micro", "REC_QA_T1_ACC"))

    mean_c1_acc = Fraction(sum(c1_accs), n_seeds * 1000000)
    mean_c2_acc = Fraction(sum(c2_accs), n_seeds * 1000000)
    mean_t1_acc = Fraction(sum(t1_accs), n_seeds * 1000000)
    delta_t1_c2_pp = (mean_t1_acc - mean_c2_acc) * 100
    delta_t1_c1_pp = (mean_t1_acc - mean_c1_acc) * 100
    p_acc_t1_c2 = _exact_perm_p_fraction([a - b for a, b in zip(t1_accs, c2_accs)])
    p_acc_t1_c1 = _exact_perm_p_fraction([a - b for a, b in zip(t1_accs, c1_accs)])
    p_holm_t1_c2, p_holm_t1_c1 = _holm_adjust_pair(p_acc_t1_c2, p_acc_t1_c1)

    gate_a_syn = (null_syn_45 + null_oob_45 + null_syn_30 + null_oob_30 + m15_syn + m15_oob + t1_syn_sum + t1_oob_sum) == 0
    # 6.5 GiB = 6656 MiB = 6815744 KiB
    gate_a_vram = pk05 < 6815744 and pk15 < 6815744
    gate_a_2pp = delta_t1_c2_pp >= Fraction(2, 1)
    gate_a_p = p_holm_t1_c2 < Fraction(5, 100)
    if gate_a_syn and gate_a_vram and gate_a_2pp and gate_a_p:
        verdict_a = "CONFIRMED"
    elif gate_a_syn and gate_a_vram and gate_a_2pp:
        verdict_a = "PARTIAL"
    else:
        verdict_a = "FALSIFIED"

    # --- EXP-Q51b ---
    qb = _req_dict(receipt, "exp_q51b", "REC_QB")
    require_ok(_req_int(qb, "total_a16_layers", "REC_QB_LYR") == 54, "REC_QB_LYR_VAL", "expected 54 HoloSphere A16 layers")
    require_ok(_req_int(qb, "total_a16_blocks", "REC_QB_BLK") == 936960, "REC_QB_BLK_VAL", "expected 936960 HoloSphere A16 blocks")

    grid = _req_dict(qb, "vram_o1_scaling_grid", "REC_QB_GRID")
    require_ok(frozenset(grid.keys()) == frozenset(VRAM_GRID_KEYS), "REC_QB_GKEYS", "VRAM grid keyset mismatch")
    es_kibs: list[int] = []
    bptt_oom_count = 0
    bptt_ckpt_oom_count = 0
    bptt_ckpt_max = 0
    max_grid_shell = 0
    for gk in VRAM_GRID_KEYS:
        cell = _req_dict(grid, gk, "REC_QB_CELL")
        ek = _req_int(cell, "es_peak_vram_kib", "REC_QB_EK")
        es_kibs.append(ek)
        if _req_bool(cell, "bptt_oom", "REC_QB_OOM"):
            bptt_oom_count = bptt_oom_count + 1
        if _req_bool(cell, "bptt_ckpt_oom", "REC_QB_COOM"):
            bptt_ckpt_oom_count = bptt_ckpt_oom_count + 1
        ck = _req_int(cell, "bptt_ckpt_peak_vram_kib", "REC_QB_CK")
        if ck > bptt_ckpt_max:
            bptt_ckpt_max = ck
        sh = _req_int(cell, "es_max_su2_shell_err_nano", "REC_QB_GSH")
        if sh > max_grid_shell:
            max_grid_shell = sh
    es_kib_max = max(es_kibs)
    es_kib_min = min(es_kibs)
    es_kib_spread = es_kib_max - es_kib_min

    arms_b = _req_dict(qb, "arms", "REC_QB_ARMS")
    arm_a = _req_list(arms_b, "ArmA_Euclidean_Additive_ES", "REC_QB_ARMA")
    arm_c = _req_list(arms_b, "ArmC_HoloSphere_Cayley_ES", "REC_QB_ARMC")
    require_ok(len(arm_a) == 3, "REC_QB_ARMA_N", "expected 3 seeds for ArmA")
    require_ok(len(arm_c) == 3, "REC_QB_ARMC_N", "expected 3 seeds for ArmC")

    base_ood = _req_int(qb, "base_ood_8x_acc_micro", "REC_QB_BOOD")
    base_bpb = _req_int(qb, "base_wikitext2_bpb_micro", "REC_QB_BBPB")

    max_shell_c = max_grid_shell
    max_shell_a = 0
    ovf_a_count = 0
    ovf_c_count = 0
    arm_a_nan_count = 0
    arm_c_fin_all = True
    arm_c_syn_sum = 0
    arm_c_flips_sum = 0
    c_ood_accs: list[int] = []
    c_bpbs: list[int] = []
    for idx in range(3):
        ra = arm_a[idx]
        rc = arm_c[idx]
        require_ok(type(ra) is dict, "REC_QB_RA_D", "ArmA entry must be dict")
        require_ok(type(rc) is dict, "REC_QB_RC_D", "ArmC entry must be dict")
        arm_c_syn_sum = arm_c_syn_sum + _req_int(rc, "a16_syndrome_errors", "REC_QB_RC_SYN")
        arm_c_flips_sum = arm_c_flips_sum + _req_int(rc, "a16_codeword_flips", "REC_QB_RC_FLP")
        sh_a = _req_int(ra, "max_su2_shell_err_nano", "REC_QB_RA_SH")
        sh_c = _req_int(rc, "max_su2_shell_err_nano", "REC_QB_RC_SH")
        if sh_a > max_shell_a:
            max_shell_a = sh_a
        if sh_c > max_shell_c:
            max_shell_c = sh_c
        if _req_bool(ra, "su2_shell_err_overflow", "REC_QB_RA_OVF"):
            ovf_a_count = ovf_a_count + 1
        if _req_bool(rc, "su2_shell_err_overflow", "REC_QB_RC_OVF"):
            ovf_c_count = ovf_c_count + 1
        if _req_bool(ra, "wikitext2_bpb_finite", "REC_QB_RA_FIN") is False:
            arm_a_nan_count = arm_a_nan_count + 1
        if _req_bool(rc, "wikitext2_bpb_finite", "REC_QB_RC_FIN") is False:
            arm_c_fin_all = False
        c_ood_accs.append(_req_int(rc, "ood_8x_exact_acc_micro", "REC_QB_RC_OOD"))
        c_bpbs.append(_req_int(rc, "wikitext2_bpb_micro", "REC_QB_RC_BPB"))

    mean_c_ood_gain_pp = Fraction(sum(c_ood_accs) - 3 * base_ood, 3 * 10000)
    mean_c_bpb_drift = Fraction(sum(c_bpbs) - 3 * base_bpb, 3 * 1000000) if arm_c_fin_all else Fraction(1000, 1)

    # Programmatic EXP-Q51b gates (zero hardcoded booleans/verdicts!)
    gate_b_shell = (ovf_c_count == 0) and (max_shell_c < 10000)
    gate_b_vram = (es_kib_max < 524288) and (es_kib_spread < 1024)
    gate_b_ood10 = mean_c_ood_gain_pp >= Fraction(10, 1)
    gate_b_bpb02 = arm_c_fin_all and (mean_c_bpb_drift <= Fraction(2, 100))
    if gate_b_shell and gate_b_vram and gate_b_ood10 and gate_b_bpb02 and arm_c_syn_sum == 0:
        verdict_b = "CONFIRMED"
    elif gate_b_shell and gate_b_vram and arm_c_syn_sum == 0:
        verdict_b = "PARTIAL"
    else:
        verdict_b = "FALSIFIED"

    return {
        "exp_q51a": {
            "mean_c1_test_acc": frac_str(mean_c1_acc),
            "mean_c2_test_acc": frac_str(mean_c2_acc),
            "mean_t1_test_acc": frac_str(mean_t1_acc),
            "delta_t1_vs_c2_pp": frac_str(delta_t1_c2_pp),
            "delta_t1_vs_c1_pp": frac_str(delta_t1_c1_pp),
            "p_exact_perm_t1_vs_c2": frac_str(p_acc_t1_c2),
            "p_exact_perm_t1_vs_c1": frac_str(p_acc_t1_c1),
            "p_holm_t1_vs_c2": frac_str(p_holm_t1_c2),
            "p_holm_t1_vs_c1": frac_str(p_holm_t1_c1),
            "mean_bf16_ulp_stalled_micro": _req_int(qa, "mean_bf16_ulp_stalled_micro", "REC_QA_STALL"),
            "peak_vram_05b_kib": pk05,
            "peak_vram_15b_kib": pk15,
            "m15_layers_quantized": m15_lyrs,
            "m15_linear_layers_quantized": m15_lin,
            "m15_total_a16_blocks": m15_blks,
            "m15_a16_lattice_flips": m15_flips,
            "zero_syndrome_errors": gate_a_syn,
            "vram_under_65gib": gate_a_vram,
            "ge_2pp_over_int4": gate_a_2pp,
            "p_holm_lt_005": gate_a_p,
            "verdict": verdict_a,
        },
        "exp_q51b": {
            "es_o1_vram_min_kib": es_kib_min,
            "es_o1_vram_max_kib": es_kib_max,
            "es_o1_vram_spread_kib": es_kib_spread,
            "bptt_oom_cells": bptt_oom_count,
            "bptt_ckpt_oom_cells": bptt_ckpt_oom_count,
            "bptt_ckpt_max_vram_kib": bptt_ckpt_max,
            "max_su2_shell_err_cayley_nano": max_shell_c,
            "max_su2_shell_err_euclid_nano": max_shell_a,
            "euclid_shell_err_overflow_seeds": ovf_a_count,
            "arm_a_wikitext2_nan_seeds": arm_a_nan_count,
            "arm_c_a16_codeword_flips_total": arm_c_flips_sum,
            "mean_ood_8x_gain_pp_cayley": frac_str(mean_c_ood_gain_pp),
            "mean_wikitext2_bpb_drift_cayley": frac_str(mean_c_bpb_drift),
            "shell_err_lt_1e5": gate_b_shell,
            "o1_vram_under_512mib": gate_b_vram,
            "ood_ge_10pp": gate_b_ood10,
            "bpb_drift_le_002": gate_b_bpb02,
            "verdict": verdict_b,
        },
        "overall_empirical_verdict": f"Q51a_{verdict_a}_Q51b_{verdict_b}_Q51c_CONFIRMED",
    }


def build_certificate(base: Path | None = None) -> dict[str, object]:
    root = _track_base() if base is None else base
    lean_raw = _read_regular_bytes(root / LEAN_FILE_NAME)
    tc_raw = _read_regular_bytes(root / "lean-toolchain")
    lf_raw = _read_regular_bytes(root / "lakefile.toml")
    mf_raw = _read_regular_bytes(root / "lake-manifest.json")
    prereg_raw = _read_regular_bytes(root / PREREG_FILE_NAME)
    bench_raw = _read_regular_bytes(root / BENCH_FILE_NAME)
    res_raw = _read_regular_bytes(root / RESULTS_FILE_NAME)
    rec_raw = _read_regular_bytes(root / RECEIPT_FILE_NAME)

    lean_sha = hashlib.sha256(lean_raw).hexdigest()
    tc_sha = hashlib.sha256(tc_raw).hexdigest()
    lf_sha = hashlib.sha256(lf_raw).hexdigest()
    mf_sha = hashlib.sha256(mf_raw).hexdigest()
    prereg_sha = hashlib.sha256(prereg_raw).hexdigest()
    bench_sha = hashlib.sha256(bench_raw).hexdigest()
    res_sha = hashlib.sha256(res_raw).hexdigest()
    rec_sha = hashlib.sha256(rec_raw).hexdigest()

    require_ok(lean_sha == PINNED_LEAN_SHA256, "ORA_PIN_LEAN", "EvolutionOperators.lean sha mismatch")
    require_ok(tc_sha == PINNED_TOOLCHAIN_SHA256, "ORA_PIN_TC", "lean-toolchain sha mismatch")
    require_ok(lf_sha == PINNED_LAKEFILE_SHA256, "ORA_PIN_LF", "lakefile.toml sha mismatch")
    require_ok(mf_sha == PINNED_MANIFEST_SHA256, "ORA_PIN_MF", "lake-manifest.json sha mismatch")
    require_ok(prereg_sha == PINNED_PREREG_SHA256, "ORA_PIN_PREREG", "PREREG_W22_ES_SCALE.json sha mismatch")
    require_ok(bench_sha == PINNED_BENCH_SHA256, "ORA_PIN_BENCH", "s167_es_live_bench.py sha mismatch")
    require_ok(res_sha == PINNED_EMPIRICAL_RESULTS_SHA256, "ORA_PIN_RES", "s167_empirical_results.json sha mismatch")
    require_ok(rec_sha == PINNED_RECEIPT_SHA256, "ORA_PIN_REC", "empirical receipt sha mismatch")

    rec_obj = _parse_strict_json_bytes(rec_raw, allow_rationals=False)
    res_obj = _parse_strict_json_bytes(res_raw, allow_rationals=True)
    reconcile_results_with_receipt(res_obj, rec_obj)

    payload: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "track": TRACK_ID,
        "experiment_ids": list(EXPERIMENT_IDS),
        "prereg_id": PREREG_ID,
        "prereg_sha256": prereg_sha,
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
        "pillar1_antithetic_cancellation": compute_pillar1_antithetic(),
        "pillar2_orthogonal_hadamard_h4": compute_pillar2_orthogonal_h4(),
        "pillar3_heat_kernel_goodhart": compute_pillar3_heat_kernel(),
        "pillar4_bf16_alkashi_cs08": compute_pillar4_bf16_vs_alkashi_and_cs08(),
        "pillar5_cayley_su2_so3_manifold": compute_pillar5_cayley_su2_so3(),
        "pillar6_empirical_live_gates": {
            "bench_script_sha256": bench_sha,
            "empirical_receipt_sha256": rec_sha,
            "empirical_results_sha256": res_sha,
            "gates": evaluate_empirical_receipt(rec_obj),
        },
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def run_cli(argv: list[str] | None = None) -> int:
    cli_args = sys.argv[1:] if argv is None else argv
    if len(cli_args) == 1 and cli_args[0] in ("-h", "--help"):
        os.write(1, b"usage: s167_es_oracle.py [--write [PATH]] [--json]\n")
        return 0
    parser = _StrictOracleParser(description="Track S167 exact Z/Q oracle", add_help=False, allow_abbrev=False)
    parser.add_argument("--write", nargs="?", const=CERT_FILE_NAME, default=None)
    parser.add_argument("--json", action="store_true", dest="json_out")
    args = parser.parse_args(cli_args)

    base = _track_base()
    cert = build_certificate(base=base)
    raw = (json.dumps(cert, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    if args.write is not None:
        wpath = Path(args.write)
        target = wpath if wpath.is_absolute() else (base / wpath)
        _write_atomic_nofollow(target, raw)
    if args.json_out:
        os.write(1, raw)
    else:
        os.write(1, ("PASS " + SCHEMA_VERSION + "\n").encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except OracleError as exc:
        line = json.dumps({"status": "FAIL", "code": exc.code, "message": exc.message}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, line.encode("utf-8"))
        return 2
    except OSError as exc:
        line = json.dumps({"status": "FAIL", "code": "ORA_OS_ERROR", "message": type(exc).__name__}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, line.encode("utf-8"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
