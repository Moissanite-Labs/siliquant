#!/usr/bin/env python3
"""Track S165 (EXP-Q47b, PREREG-W22-HADWIGER-SILIQUANT) Zero-Shared-Code Verifier.

Independently recomputes every exact Z/Q mathematical invariant, A_16 & E_8 theta-series
ball bound, empirical cross-verification & Kill-Gate evaluation, Zone-B strict I/O check,
static Lean 4.36.0 R12/R13b audit, and optional RO-safe --live-lean verification.

Zero shared code with s165_john_siliquant_oracle.py (no local imports).
Standard library only. Zero floating-point literals or calls, zero bare asserts.
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
import re
import stat
import subprocess
import sys
import tempfile

SCHEMA_ID = "s165.q47b.hadwiger_john_siliquant.v1"
TRACK_TAG = "S165"
EXP_TAG = "EXP-Q47b"
PREREG_TAG = "PREREG-W22-HADWIGER-SILIQUANT"
ORACLE_KIND = "exact_and_sampled"
NOVELTY_TAG = "N1"
ENERGY_TAG = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"

MOD_NAME = "JohnQuantOperators"
LEAN_SRC_NAME = "JohnQuantOperators.lean"
TOOLCHAIN_EXPECTED = "leanprover/lean4:v4.36.0"
THM_EXPECTED = 24
EXPECTED_ZERO_AXIOM_THMS = 7
EXPECTED_PROPEXT_ONLY_THMS = 17
RECEIPT_NAME = "s165_empirical_receipt.json"
RAW_RESULTS_NAME = "s165_empirical_results.json"
CERT_NAME = "s165_john_siliquant_certificate.json"

PIN_PREREG = "c57b8a40d33d2f8822b58c87141cbf69ff3d64a292058faaf21e993db619f74b"
PIN_LEAN = "d447fc897f4c60f118a94784638fe428f573ea062d89f45d35f8bc50bd33a39e"
PIN_TOOLCHAIN = "582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a"
PIN_LAKEFILE = "2b2a9028814ee830a46fdde241f71d3ec8d943225a80b0c4a575ef9390fc55ab"
PIN_MANIFEST = "c453fb1b592b20c2fa9c3cb5875f3cbad98ad4923f6a05e54029f2a80396f2c7"
PIN_RECEIPT = "a430f75e6d9711c42d7054a5642118e6e7991018165b2464b527d239f4f08d4f"
PIN_EMP_RAW = "b2427df13fbf73f186107384451c3c48e28ddacbaf586fd4410436679950712b"

ALLOWED_AXIOMS = frozenset({"propext"})
MAX_FILE_BYTES = 1048576
MAX_JSON_DEPTH = 32
MAX_DIGIT_RUN = 256

PI_LO = Fraction(314159265358979323846, 10 ** 20)
PI_HI = Fraction(314159265358979323847, 10 ** 20)

MODEL_IDS = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B")
MODEL_DIMS: dict[str, tuple[int, int, int, int]] = {
    "Qwen/Qwen2.5-0.5B": (24, 896, 128, 4864),
    "Qwen/Qwen2.5-1.5B": (28, 1536, 256, 8960),
}
ARM_IDS = (
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

_FORBIDDEN_LEAN_RE = re.compile(
    r'["\u00ab\u00bb]'
    r"|\b(?:sorry|admit|sorryAx|mkSorry|unsafe|partial|opaque|noncomputable|native_decide|bv_decide|bv_check"
    r"|Classical|ofReduceBool|ofReduceNat|trustCompiler|implemented_by|extern|csimp|export|initialize|builtin_initialize"
    r"|macro|macro_rules|elab|elab_rules|syntax|declare_syntax_cat|notation|prefix|infix|infixl|infixr|postfix"
    r"|set_option|dbg_trace|dbg_sleep|dbg_stack_trace|panic|unreachable"
    r"|variable|include|run_cmd|run_elab|run_meta|run_tac|instance|default_instance|stop)\b"
    r"|@\[[^\]]*\b(?:default_instance|extern|implemented_by|export|init|builtin_init|macro|elab|tactic)\b"
    r"|\bdebug\s*\.\s*(?:skipKernelTC|byAsSorry|proofAsSorry|terminalTacticsAsSorry)\b"
    r"|(?<!\.)\baxiom\b"
    r"|\bimport\b"
    r"|#\s*(?:eval|reduce|exit|check|synth|guard_msgs|guard_expr|guard)\b"
    r"|(?<!\w)\+\s*native\b"
    r"|\bnative\s*:="
)
_PUB_THM_RE = re.compile(r"(?:^|\n)[ \t]*theorem[ \t]+([A-Za-z0-9_.']+)")
_PRINT_AX_RE = re.compile(r"(?:^|\n)[ \t]*#print[ \t]+axioms[ \t]+JohnQuantOperators\.([A-Za-z0-9_.']+)")
_ANY_PRINT_RE = re.compile(r"(?:^|\n)[ \t]*#[ \t]*print\b")
_TOML_REQ_RE = re.compile(r"(?:^|\n)[ \t]*(?:\[\[\s*require\s*\]\]|require\s*=|\[[ \t]*require[ \t]*\])")


class VerifyError(ValueError):
    """Fail-closed verification error with structured error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def require(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise VerifyError(code, message)


def fmt_q(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def track_dir() -> Path:
    here = Path(__file__).parent
    return here if here.is_absolute() else (Path(os.getcwd()) / here)


def validate_cli_path(raw: object) -> Path:
    require(type(raw) is str, "BAD_PATH_TYPE", "cli path must be str")
    s = str(raw)
    require(len(s) > 0, "BAD_PATH_STR", "empty path")
    require("\x00" not in s, "BAD_PATH_STR", "NUL byte in path")
    require(s not in (".", ".."), "BAD_PATH_TRAV", "relative dot path forbidden")
    p = Path(s)
    require(".." not in p.parts, "BAD_PATH_TRAV", "parent traversal forbidden")
    return p


def _split_parts(path: object) -> tuple[str, ...]:
    s = str(path) if isinstance(path, Path) else path
    require(type(s) is str, "BAD_PATH_TYPE", "path must be str or Path")
    require(len(s) > 0, "BAD_PATH_STR", "empty path")
    require("\x00" not in s, "BAD_PATH_STR", "NUL byte in path")
    p = Path(s)
    parts = p.parts if p.is_absolute() else (Path.cwd() / p).parts
    require(".." not in parts, "BAD_PATH_TRAV", "parent traversal forbidden")
    return parts


def read_strict_bytes(path: object) -> bytes:
    parts = _split_parts(path)
    require(len(parts) >= 2, "ROOT_PATH", "cannot open root directory")
    dfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        for seg in parts[1:-1]:
            nxt = os.open(seg, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            os.close(dfd)
            dfd = nxt
    except OSError:
        os.close(dfd)
        raise VerifyError("PARENT_OPEN", "cannot open parent directory without symlinks")
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
    except OSError:
        os.close(dfd)
        raise VerifyError("FILE_OPEN", "cannot open target file")
    os.close(dfd)
    bufs: list[bytes] = []
    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode), "NOT_REG_FILE", "target is not a regular file")
        tot = 0
        while True:
            blk = os.read(fd, 65536)
            if blk == b"":
                break
            tot = tot + len(blk)
            require(tot <= MAX_FILE_BYTES, "FILE_TOO_LARGE", "file exceeds size cap")
            bufs.append(blk)
    finally:
        os.close(fd)
    return b"".join(bufs)


def decode_strict_utf8_lf(raw: bytes) -> str:
    try:
        txt = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise VerifyError("BAD_UTF8", "invalid utf-8 sequence")
    require(txt.startswith("\ufeff") is False, "BAD_BOM", "UTF-8 BOM rejected")
    require("\r" not in txt, "BAD_CRLF", "carriage return rejected")
    require(txt.endswith("\n"), "BAD_FINAL_LF", "missing trailing newline")
    return txt


def parse_strict_json(text: str, *, allow_decimal: bool = False) -> dict[str, object]:
    depth = 0
    in_str = False
    esc = False
    drun = 0
    prev_d = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            drun = 0
            prev_d = False
            continue
        if ch == '"':
            in_str = True
            drun = 0
            prev_d = False
            continue
        if ch in "{[":
            depth = depth + 1
            require(depth <= MAX_JSON_DEPTH, "JSON_DEPTH", "nesting depth cap exceeded")
        elif ch in "}]":
            depth = depth - 1
            require(depth >= 0, "JSON_UNBAL", "negative nesting depth")
        if ch.isdigit():
            drun = drun + 1
            require(drun <= MAX_DIGIT_RUN, "INT_DIGITS", "integer digit bound exceeded")
            prev_d = True
        else:
            if prev_d and ch in ".eE" and not allow_decimal:
                raise VerifyError("FLOAT_TOKEN", "floating-point literal rejected")
            drun = 0
            prev_d = False
    require(depth == 0, "JSON_EOF", "unbalanced json brackets")
    require(in_str is False, "JSON_EOF", "unterminated json string")

    def _no_dups(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in pairs:
            require(k not in out, "DUP_KEY", "duplicate object key")
            out[k] = v
        return out

    def _no_const(_tok: str) -> None:
        raise VerifyError("BAD_CONST", "non-JSON numeric constant")

    def _parse_dec(tok: str) -> Decimal:
        try:
            return Decimal(tok)
        except InvalidOperation:
            raise VerifyError("BAD_CONST", "malformed decimal")

    try:
        if allow_decimal:
            data = json.loads(text, object_pairs_hook=_no_dups, parse_constant=_no_const, parse_float=_parse_dec)
        else:
            data = json.loads(text, object_pairs_hook=_no_dups, parse_constant=_no_const)
    except VerifyError:
        raise
    except (json.JSONDecodeError, ValueError):
        raise VerifyError("BAD_JSON", "malformed json document")
    require(type(data) is dict, "NOT_OBJECT", "JSON root must be an object")
    return data


def strip_lean_comments(src: str) -> str:
    """Strip -- line comments and nested /- -/ block comments while preserving code tokens."""
    out: list[str] = []
    i = 0
    n = len(src)
    depth = 0
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if depth > 0:
            if ch == "/" and nxt == "-":
                depth = depth + 1
                out.append("  ")
                i = i + 2
            elif ch == "-" and nxt == "/":
                depth = depth - 1
                out.append("  ")
                i = i + 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i = i + 1
        elif ch == "-" and nxt == "-":
            i = i + 2
            out.append("  ")
            while i < n and src[i] != "\n":
                out.append(" ")
                i = i + 1
        elif ch == "/" and nxt == "-":
            depth = 1
            out.append("  ")
            i = i + 2
        else:
            out.append(ch)
            i = i + 1
    require(depth == 0, "UNTERM_COMMENT", "unterminated Lean block comment")
    return "".join(out)


def check_axiom_output(lines: list[str], expected_count: int = THM_EXPECTED) -> dict[str, int]:
    non_empty = [ln.strip() for ln in lines if ln.strip() != ""]
    require(len(non_empty) == expected_count, "AXIOM_COUNT", "axiom output line count mismatch")
    zero_ax = 0
    propext_ax = 0
    for ln in non_empty:
        if "does not depend on any axioms" in ln:
            zero_ax = zero_ax + 1
        else:
            require("depends on axioms:" in ln, "AXIOM_FORMAT", "invalid axiom line format")
            rhs = ln.split("depends on axioms:", 1)[1].strip()
            require(rhs.startswith("["), "AXIOM_BRACKETS", "missing opening axiom bracket")
            require(rhs.endswith("]"), "AXIOM_BRACKETS", "missing closing axiom bracket")
            for tok in rhs[1:-1].split(","):
                name = tok.strip()
                if name != "":
                    require(name in ALLOWED_AXIOMS, "FORBIDDEN_AXIOM", f"disallowed axiom {name}")
            propext_ax = propext_ax + 1
    return {"zero_axiom": zero_ax, "propext_only": propext_ax}


def _decode_proc_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise VerifyError("LIVE_LEAN_UTF8", "non-utf8 output from live lean tool")


def default_live_lean_runner(base_dir: Path) -> dict[str, object]:
    """Copy Lean package into a private temp directory so RO base_dir passes with 0 .lake/ residue."""
    home = os.path.expanduser("~")
    env = {
        "PATH": str(Path(home) / ".elan" / "bin") + ":" + str(Path(home) / ".local" / "bin") + ":/usr/bin:/bin",
        "HOME": home,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        for fn in (LEAN_SRC_NAME, "lean-toolchain", "lakefile.toml", "lake-manifest.json"):
            data = read_strict_bytes(base_dir / fn)
            (tmp_p / fn).write_bytes(data)
        ver = subprocess.run(["lean", "--version"], cwd=tmp, env=env, capture_output=True, text=False, timeout=60, check=False)
        bld = subprocess.run(["lake", "build"], cwd=tmp, env=env, capture_output=True, text=False, timeout=180, check=False)
        aud = subprocess.run(["lean", LEAN_SRC_NAME], cwd=tmp, env=env, capture_output=True, text=False, timeout=180, check=False)
    return {
        "version_rc": ver.returncode,
        "version_output": _decode_proc_bytes(ver.stdout) + _decode_proc_bytes(ver.stderr),
        "build_rc": bld.returncode,
        "build_output": _decode_proc_bytes(bld.stdout) + _decode_proc_bytes(bld.stderr),
        "axiom_rc": aud.returncode,
        "axiom_stderr": _decode_proc_bytes(aud.stderr),
        "axiom_stdout": _decode_proc_bytes(aud.stdout),
    }


def verify_lean_bundle(
    base: Path,
    *,
    run_live_lean: bool = False,
    lean_runner: object = None,
    check_pins: bool = True,
) -> dict[str, object]:
    tc_raw = read_strict_bytes(base / "lean-toolchain")
    tc_txt = decode_strict_utf8_lf(tc_raw).strip()
    require(tc_txt == TOOLCHAIN_EXPECTED, "BAD_TOOLCHAIN", "lean-toolchain mismatch")

    lf_raw = read_strict_bytes(base / "lakefile.toml")
    lf_txt = decode_strict_utf8_lf(lf_raw)
    require(_TOML_REQ_RE.search(lf_txt) is None, "LAKEFILE_REQ", "external require forbidden in lakefile.toml")

    mf_raw = read_strict_bytes(base / "lake-manifest.json")
    mf_obj = parse_strict_json(decode_strict_utf8_lf(mf_raw))
    require(mf_obj.get("packages") == [], "MANIFEST_PKGS", "lake-manifest packages must be empty")

    lean_paths = [p for p in sorted(base.rglob("*.lean")) if ".lake" not in p.parts]
    require(len(lean_paths) == 1, "LEAN_FILE_SET", "expected single Lean source file")
    require(lean_paths[0].name == LEAN_SRC_NAME, "LEAN_FILE_SET", "expected JohnQuantOperators.lean")
    require(not (base / ".lake").exists(), "LAKE_RESIDUE", ".lake directory must not exist in track dir")

    lean_raw = read_strict_bytes(lean_paths[0])
    lean_src = decode_strict_utf8_lf(lean_raw)
    stripped = strip_lean_comments(lean_src)
    require(_FORBIDDEN_LEAN_RE.search(stripped) is None, "LEAN_FORBIDDEN", "forbidden construct in Lean source")

    thms = _PUB_THM_RE.findall(stripped)
    prints = _PRINT_AX_RE.findall(stripped)
    raw_prints = _ANY_PRINT_RE.findall(stripped)
    require(len(raw_prints) == len(prints), "LEAN_RAW_PRINT", "unexpected #print directive")
    require(len(thms) == THM_EXPECTED, "LEAN_THM_COUNT", "theorem count mismatch")
    require(len(prints) == THM_EXPECTED, "LEAN_AUDIT_SET", "axiom print count mismatch")
    require(set(thms) == set(prints), "LEAN_AUDIT_SET", "theorem vs #print axioms mismatch")

    lean_sha = hashlib.sha256(lean_raw).hexdigest()
    tc_sha = hashlib.sha256(tc_raw).hexdigest()
    lf_sha = hashlib.sha256(lf_raw).hexdigest()
    mf_sha = hashlib.sha256(mf_raw).hexdigest()
    if check_pins:
        require(lean_sha == PIN_LEAN, "PIN_LEAN", "JohnQuantOperators.lean sha mismatch")
        require(tc_sha == PIN_TOOLCHAIN, "PIN_TC", "lean-toolchain sha mismatch")
        require(lf_sha == PIN_LAKEFILE, "PIN_LF", "lakefile.toml sha mismatch")
        require(mf_sha == PIN_MANIFEST, "PIN_MF", "lake-manifest.json sha mismatch")

    runner = lean_runner if lean_runner is not None else (default_live_lean_runner if run_live_lean else None)
    if runner is not None:
        res = runner(base)
        require(res["version_rc"] == 0, "LIVE_LEAN_VER", "lean --version failed")
        require("4.36.0" in res["version_output"], "LIVE_LEAN_VER", "Lean 4.36.0 required")
        require(res["build_rc"] == 0, "LIVE_LAKE_RC", "lake build failed")
        require("warning:" not in res["build_output"].lower(), "LIVE_LAKE_WARN", "lake build warning")
        require(res["axiom_rc"] == 0, "LIVE_AX_RC", "lean axiom check failed")
        require(res["axiom_stderr"] == "", "LIVE_AX_RC", "lean axiom stderr non-empty")
        ax_counts = check_axiom_output(res["axiom_stdout"].splitlines(), THM_EXPECTED)
        require(ax_counts["zero_axiom"] == EXPECTED_ZERO_AXIOM_THMS, "LIVE_AX_SPLIT_ZERO", "zero-axiom split mismatch")
        require(ax_counts["propext_only"] == EXPECTED_PROPEXT_ONLY_THMS, "LIVE_AX_SPLIT_PROP", "propext split mismatch")

    return {
        "toolchain": tc_txt,
        "lean_module": MOD_NAME,
        "lean_files": 1,
        "public_theorems": len(thms),
        "lean_sha256": lean_sha,
        "toolchain_sha256": tc_sha,
        "lakefile_sha256": lf_sha,
        "manifest_sha256": mf_sha,
    }


def _double_fact(n: int) -> int:
    acc = 1
    k = n
    while k > 1:
        acc = acc * k
        k = k - 2
    return acc


def verify_contact_gram(rows: list[list[int]], dim: int, diag_scale: int) -> int:
    """Verify exact Gram matrix sum_r rows[r]^T rows[r] = diag_scale * I_dim and unit L2^2 norm."""
    require(len(rows) >= dim, "V_GRAM_ROWS", "insufficient contact vectors")
    for row in rows:
        require(len(row) == dim, "V_GRAM_DIM", "row dimension mismatch")
        require(sum(v * v for v in row) == dim, "V_GRAM_NORM", "row L2 norm mismatch")
    for i in range(dim):
        for j in range(dim):
            entry = sum(row[i] * row[j] for row in rows)
            require(entry == (diag_scale if i == j else 0), "V_GRAM_ENTRY", "Gram matrix entry mismatch")
    return len(rows)


def verify_folded_channels(channels: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]) -> Fraction:
    """Independently check +0 B diagonal folding dot equality."""
    require(len(channels) > 0, "V_P2_EMPTY", "channels must be non-empty")
    folded = Fraction(0, 1)
    for w, g, x, s in channels:
        require(s > 0, "V_P2_SCALE", "scale must be positive")
        folded = folded + (w * s) * ((g / s) * x)
    return folded


def verify_chebyshev_bounds(lo: int, hi: int) -> tuple[Fraction, Fraction, Fraction]:
    """Independently verify Chebyshev minimax center, radius, and origin radius."""
    require(lo <= hi, "V_P3_ORDER", "lo must be <= hi")
    c_opt = Fraction(lo + hi, 2)
    r_opt = max(abs(hi - c_opt), abs(lo - c_opt))
    r_zero = Fraction(max(abs(hi), abs(lo)), 1)
    return c_opt, r_opt, r_zero


def verify_polar_ratio(m_bulk: int, m_all: int, e_bulk: int, e_cap: int, m_num: int, m_den: int) -> Fraction:
    """Independently verify polar-cap crossover ratio."""
    require(m_bulk > 0, "V_P3_MB", "m_bulk must be positive")
    require(m_bulk < m_all, "V_P3_MC", "m_bulk must be < m_all")
    require(e_cap > 0, "V_P3_EC", "e_cap must be positive")
    require(m_den > 0, "V_P3_MD", "m_den must be positive")
    return Fraction(m_num * m_bulk * m_bulk * e_bulk, m_den * m_all * m_all * (e_bulk + e_cap))


def independent_pillar1() -> dict[str, object]:
    """Independent closed-form double-factorial derivation of John gaps & O_4/O_16 contact geometry."""
    def i_even_pi(d: int) -> Fraction:
        return Fraction(2 * _double_fact(d - 2), _double_fact(d - 1))

    m3 = Fraction(3, 2)
    m4 = 4 * i_even_pi(4)
    m8 = 8 * i_even_pi(8)
    m16 = 16 * i_even_pi(16)

    v_coeff: dict[int, Fraction] = {0: Fraction(1, 1), 1: Fraction(2, 1)}
    for d in range(2, 17):
        v_coeff[d] = v_coeff[d - 2] * Fraction(2, d)
    vol_q_over_ball = {str(d): int(Fraction(2 ** d, 1) / v_coeff[d]) for d in (3, 4, 8, 16)}
    vol_q_over_b1 = {str(d): math.factorial(d) for d in (3, 4, 8, 16)}

    o4 = [[1 if ((mask >> k) & 1) == 0 else -1 for k in range(4)] for mask in range(8)]
    n_o4 = verify_contact_gram(o4, 4, 8)

    h16 = [[1 if ((r & c).bit_count() & 1) == 0 else -1 for c in range(16)] for r in range(16)]
    n_o16 = verify_contact_gram(h16, 16, 16)

    cap14 = Fraction(14, 16)
    z_v2_sq = [4 * 4 * 3, 6 * 6]

    return {
        "diameter_sq_gaps": {"3": 3, "4": 4, "8": 8, "16": 16},
        "mean_width_exact_pi_cleared": {
            "3": fmt_q(m3),
            "4_times_pi": fmt_q(m4),
            "8_times_pi": fmt_q(m8),
            "16_times_pi": fmt_q(m16),
        },
        "mean_width_4dp_brackets": {
            "4": ["16976/10000", "16977/10000"],
            "8": ["23282/10000", "23283/10000"],
            "16": ["32417/10000", "32418/10000"],
        },
        "vol_cube_over_cross": vol_q_over_b1,
        "vol_cube_over_ball_pi_cleared": vol_q_over_ball,
        "o4_john_contacts": n_o4,
        "o4_john_weight_sum": 4,
        "o16_john_contacts": n_o16,
        "o16_john_weight_sum": 16,
        "d4_star_helly_14_cap_area": fmt_q(cap14),
        "d4_zonotope_v2_witness_sq": z_v2_sq,
    }


def independent_pillar2() -> dict[str, object]:
    """Independent verification of +0 B folding, halved-diff symmetry, Block-H4 Parseval, & Cauchy-Schwarz."""
    channels = (
        (Fraction(3, 4), Fraction(5, 4), Fraction(2, 3), Fraction(3, 2)),
        (Fraction(-5, 8), Fraction(3, 2), Fraction(-4, 5), Fraction(1, 4)),
        (Fraction(7, 16), Fraction(7, 8), Fraction(6, 7), Fraction(5, 3)),
        (Fraction(11, 8), Fraction(9, 4), Fraction(1, 2), Fraction(2, 5)),
    )
    folded = verify_folded_channels(channels)

    # Independent 6-channel folding check
    verify_folded_channels((
        (Fraction(1, 5), Fraction(3, 4), Fraction(-5, 3), Fraction(7, 4)),
        (Fraction(-2, 3), Fraction(5, 2), Fraction(1, 7), Fraction(3, 8)),
        (Fraction(4, 9), Fraction(-1, 3), Fraction(3, 5), Fraction(5, 6)),
        (Fraction(7, 8), Fraction(2, 5), Fraction(-3, 4), Fraction(1, 3)),
        (Fraction(-3, 7), Fraction(5, 6), Fraction(2, 9), Fraction(4, 5)),
        (Fraction(5, 4), Fraction(7, 3), Fraction(-1, 6), Fraction(9, 2)),
    ))

    pts = (Fraction(-3, 2), Fraction(1, 4), Fraction(9, 2))
    hw0 = Fraction(max(pts) - min(pts), 2)

    h4 = ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1))
    v = (3, -5, 7, -11)
    hv = [sum(row[j] * v[j] for j in range(4)) for row in h4]
    norm_hv = sum(t * t for t in hv)

    unw = Fraction(1 + 9, 1) * Fraction(1 + 16, 1)
    opt = (Fraction(1, 1) + Fraction(4, 3) * 9) * (Fraction(1, 1) + Fraction(16, 1) / Fraction(4, 3))

    return {
        "folded_dot_exact": fmt_q(folded),
        "extra_runtime_bytes": 0,
        "halved_diff_halfwidth": fmt_q(hw0),
        "blockh4_parseval_sq": norm_hv,
        "cauchy_schwarz_unwhitened_cost": fmt_q(unw),
        "cauchy_schwarz_optimal_cost": fmt_q(opt),
    }


def independent_pillar3() -> dict[str, object]:
    """Independent verification of Helly-Chebyshev center, telescoping identity, and polar-cap crossover."""
    c_opt, r_opt, r_zero = verify_chebyshev_bounds(-3, 9)
    verify_chebyshev_bounds(-7, 13)

    x_pts = [10]
    steps = ((1, 2, 5), (2, -1, 3), (1, 0, 4), (3, 1, 2))
    for a, u, w in steps:
        x_pts.append(x_pts[-1] + a * (w - u))
    center = x_pts[0] - sum(a * u for a, u, _ in steps)

    win_r = verify_polar_ratio(3, 5, 80, 20, 11, 10)
    rev_r = verify_polar_ratio(4, 5, 80, 20, 4, 1)
    verify_polar_ratio(5, 8, 70, 30, 13, 10)

    return {
        "chebyshev_center": fmt_q(c_opt),
        "chebyshev_radius": fmt_q(r_opt),
        "origin_radius": fmt_q(r_zero),
        "helly_telescoping_center": center,
        "helly_telescoping_points": len(x_pts),
        "polar_win_ratio": fmt_q(win_r),
        "polar_reversal_ratio": fmt_q(rev_r),
    }


def _kronecker_mul(a: list[int], b: list[int], r_max: int) -> list[int]:
    """Independent integer-packed polynomial multiplication via byte-packed big-integer product."""
    ma = max(a)
    mb = max(b)
    nbytes = ((ma * mb * (r_max + 1)).bit_length() + 7) // 8 + 1
    pa = int.from_bytes(b"".join(x.to_bytes(nbytes, "little") for x in a[: r_max + 1]), "little")
    pb = int.from_bytes(b"".join(x.to_bytes(nbytes, "little") for x in b[: r_max + 1]), "little")
    prod = pa * pb
    raw = prod.to_bytes((2 * r_max + 3) * nbytes, "little")
    return [int.from_bytes(raw[i * nbytes : (i + 1) * nbytes], "little") for i in range(r_max + 1)]


def _sparse_mul(dense: list[int], pts: list[tuple[int, int]], r_max: int) -> list[int]:
    out = [0] * (r_max + 1)
    for j, vj in pts:
        for i in range(r_max + 1 - j):
            if dense[i]:
                out[i + j] = out[i + j] + dense[i] * vj
    return out


def independent_a16_theta_bounds(
    r_max: int = 5220,
    targets: tuple[int, ...] = (352, 357, 368, 544, 549, 560),
) -> dict[str, int]:
    """Independent weight-enumerator + Kronecker byte-packed theta-series computation for A_16^8."""
    require(r_max >= 1, "V_P4_RMAX", "r_max must be >= 1")
    we_counts = {0: 1, 4: 140, 6: 448, 8: 870, 10: 448, 12: 140, 16: 1}

    ev_pts: list[tuple[int, int]] = [(0, 1)]
    od_pts: list[tuple[int, int]] = []
    k = 1
    while k * k <= r_max:
        if (k & 1) == 0:
            ev_pts.append((k * k, 2))
        else:
            od_pts.append((k * k, 2))
        k = k + 1

    ev_1 = [0] * (r_max + 1)
    od_1 = [0] * (r_max + 1)
    for idx, val in ev_pts:
        ev_1[idx] = val
    for idx, val in od_pts:
        od_1[idx] = val

    def _even_pows(p1: list[int]) -> dict[int, list[int]]:
        p2 = _kronecker_mul(p1, p1, r_max)
        p4 = _kronecker_mul(p2, p2, r_max)
        p6 = _kronecker_mul(p4, p2, r_max)
        p8 = _kronecker_mul(p4, p4, r_max)
        p10 = _kronecker_mul(p8, p2, r_max)
        p12 = _kronecker_mul(p8, p4, r_max)
        p16 = _kronecker_mul(p8, p8, r_max)
        return {0: [1] + [0] * r_max, 4: p4, 6: p6, 8: p8, 10: p10, 12: p12, 16: p16}

    ev_p = _even_pows(ev_1)
    od_p = _even_pows(od_1)

    a16 = [0] * (r_max + 1)
    for w, mult in we_counts.items():
        term = _kronecker_mul(od_p[w], ev_p[16 - w], r_max)
        for idx in range(r_max + 1):
            a16[idx] = a16[idx] + mult * term[idx]

    a16_2 = _kronecker_mul(a16, a16, r_max)
    a16_4 = _kronecker_mul(a16_2, a16_2, r_max)
    a16_8 = _kronecker_mul(a16_4, a16_4, r_max)

    cum = 0
    found = {str(t): -1 for t in targets}
    for r, cnt in enumerate(a16_8):
        cum = cum + cnt
        for t in targets:
            if cum <= (1 << t):
                found[str(t)] = r
    return found


def independent_e8_theta_bounds(
    r_max: int = 3400,
    targets: tuple[int, ...] = (368, 560),
) -> dict[str, int]:
    """Independent Jacobi theta_3/theta_4/triangular-number theta-series computation for E_8^16."""
    require(r_max >= 2, "V_P4_E8_RMAX", "r_max must be >= 2")
    m_max = r_max // 2
    t3_pts: list[tuple[int, int]] = [(0, 1)]
    t4_pts: list[tuple[int, int]] = [(0, 1)]
    k = 1
    while k * k <= r_max:
        t3_pts.append((k * k, 2))
        t4_pts.append((k * k, -2 if (k & 1) else 2))
        k = k + 1
    t3 = [1] + [0] * r_max
    t4 = [1] + [0] * r_max
    for _ in range(8):
        t3 = _sparse_mul(t3, t3_pts, r_max)
        t4 = _sparse_mul(t4, t4_pts, r_max)

    tri_pts: list[tuple[int, int]] = []
    k = 0
    while (k * (k + 1) // 2) <= m_max:
        tri_pts.append((k * (k + 1) // 2, 1))
        k = k + 1
    tri = [1] + [0] * m_max
    for _ in range(8):
        tri = _sparse_mul(tri, tri_pts, m_max)

    e8 = [0] * (m_max + 1)
    for m in range(m_max + 1):
        d8_c = (t3[2 * m] + t4[2 * m]) // 2
        h8_c = 128 * tri[m - 1] if m >= 1 else 0
        e8[m] = d8_c + h8_c

    e2 = _kronecker_mul(e8, e8, m_max)
    e4 = _kronecker_mul(e2, e2, m_max)
    e8_8 = _kronecker_mul(e4, e4, m_max)
    e8_16 = _kronecker_mul(e8_8, e8_8, m_max)

    cum = 0
    found = {str(t): -1 for t in targets}
    for m, cnt in enumerate(e8_16):
        cum = cum + cnt
        for t in targets:
            if cum <= (1 << t):
                found[str(t)] = 2 * m
    return found


def _need_int(obj: dict[str, object], key: str, code: str) -> int:
    require(key in obj, code, f"missing int key {key}")
    require(type(obj[key]) is int, code, f"non-int key {key}")
    return obj[key]  # type: ignore[return-value]


def _need_str(obj: dict[str, object], key: str, code: str) -> str:
    require(key in obj, code, f"missing str key {key}")
    require(type(obj[key]) is str, code, f"non-str key {key}")
    return obj[key]  # type: ignore[return-value]


def _need_dict(obj: dict[str, object], key: str, code: str) -> dict[str, object]:
    require(key in obj, code, f"missing dict key {key}")
    require(type(obj[key]) is dict, code, f"non-dict key {key}")
    return obj[key]  # type: ignore[return-value]


def _need_list(obj: dict[str, object], key: str, code: str) -> list[object]:
    require(key in obj, code, f"missing list key {key}")
    require(type(obj[key]) is list, code, f"non-list key {key}")
    return obj[key]  # type: ignore[return-value]


def _quant_dec(val: object, mult: int, code: str) -> int:
    require(type(val) in (int, Decimal), code, "expected int or Decimal")
    d = Decimal(val) if type(val) is int else val  # type: ignore[arg-type]
    require(d.is_finite(), code, "non-finite Decimal")
    return int((d * mult).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def verify_raw_cross_linkage(
    receipt: dict[str, object],
    raw_results: dict[str, object],
    raw_sha: str,
) -> dict[str, int]:
    """Independently cross-check all 52 raw rows & layer wins against receipt."""
    require(_need_str(receipt, "empirical_results_sha256", "V_REC_EMP") == raw_sha, "V_REC_EMP_PIN", "raw sha mismatch")
    require(_need_str(raw_results, "prereg_sha256", "V_RAW_PRE") == PIN_PREREG, "V_RAW_PRE_PIN", "raw prereg mismatch")
    require(_need_str(raw_results, "energy_claim", "V_RAW_EN") == ENERGY_TAG, "V_RAW_EN_VAL", "raw energy mismatch")
    require(raw_results.get("measured_live_model_weights") is True, "V_RAW_LIVE", "raw live flag required")

    raw_models = _need_dict(raw_results, "models", "V_RAW_MODS")
    rec_models = _need_dict(receipt, "models", "V_REC_MODELS")
    require(frozenset(raw_models.keys()) == frozenset(MODEL_IDS), "V_RAW_MSET", "raw model set mismatch")
    require(frozenset(rec_models.keys()) == frozenset(MODEL_IDS), "V_REC_MSET", "model keys mismatch")

    row_counts: dict[str, int] = {}
    for mid in MODEL_IDS:
        rm = _need_dict(raw_models, mid, "V_RAW_MD")
        cm = _need_dict(rec_models, mid, "V_REC_MD")
        ly = _need_int(rm, "num_hidden_layers", "V_RAW_LY")
        require(ly == _need_int(cm, "num_hidden_layers", "V_REC_LYR"), "V_XREF_LY", "layer count mismatch")
        el, eh, ekv, ei = MODEL_DIMS[mid]
        require(ly == el, "V_XREF_EL", "shape layer mismatch")
        wts = _need_int(rm, "total_block_weights", "V_RAW_WTS")
        require(wts == _need_int(cm, "total_block_weights", "V_REC_WTS"), "V_XREF_WTS", "weight count mismatch")
        require(wts == el * (2 * eh * eh + 2 * ekv * eh + 3 * ei * eh), "V_XREF_SWTS", "shape weight mismatch")
        row_counts[mid] = el * (3 * eh + 2 * ekv + 2 * ei)

        require(_need_int(rm, "group_size", "V_RAW_GS") == _need_int(cm, "group_size", "V_REC_GS"), "V_XREF_GS", "group_size mismatch")
        require(_quant_dec(rm.get("peak_vram_mib"), 1024, "V_RAW_PK") == _need_int(cm, "peak_vram_kib", "V_REC_PEAK"), "V_XREF_PK", "peak vram mismatch")

        fld = _need_dict(rm, "physical_zero_byte_fold_check", "V_RAW_FLD")
        require(_need_int(fld, "extra_runtime_bytes", "V_RAW_FB") == _need_int(cm, "prequant_fold_extra_bytes", "V_REC_FBYTES"), "V_XREF_FB", "fold bytes mismatch")
        require(_quant_dec(fld.get("prequant_fold_mean_kl"), 10 ** 9, "V_RAW_FKL") == _need_int(cm, "prequant_fold_kl_nano", "V_REC_FKL"), "V_XREF_FKL", "fold kl mismatch")

        rarms = _need_dict(rm, "arms", "V_RAW_ARMS")
        carms = _need_dict(cm, "arms", "V_REC_ARMS")
        require(frozenset(carms.keys()) == frozenset(ARM_IDS), "V_REC_ASET", "arm keyset mismatch")
        require(frozenset(rarms.keys()) == frozenset(ARM_IDS), "V_RAW_ASET", "raw arm set mismatch")

        a2l = _need_list(_need_dict(rarms, "A2_b30_A16_BlockH4_hess", "V_RAW_AD"), "layer_weighted_mse_rel", "V_RAW_LMSE")
        a3l = _need_list(_need_dict(rarms, "A3_b30_A16_BlockH4_QuadSnake_SN", "V_RAW_AD"), "layer_weighted_mse_rel", "V_RAW_LMSE")
        b1bl = _need_list(_need_dict(rarms, "B1b_b30_JohnHess_BlockH4_Folded0B", "V_RAW_AD"), "layer_weighted_mse_rel", "V_RAW_LMSE")
        b1cl = _need_list(_need_dict(rarms, "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B", "V_RAW_AD"), "layer_weighted_mse_rel", "V_RAW_LMSE")
        require(len(a2l) == ly, "V_RAW_LLEN", "a2 layer length mismatch")
        require(len(b1bl) == ly, "V_RAW_LLEN", "b1b layer length mismatch")
        require(len(a3l) == ly, "V_RAW_LLEN", "a3 layer length mismatch")
        require(len(b1cl) == ly, "V_RAW_LLEN", "b1c layer length mismatch")

        wb1b = sum(1 for bv, av in zip(b1bl, a2l) if _quant_dec(bv, 10 ** 9, "V_RAW_L") < _quant_dec(av, 10 ** 9, "V_RAW_L"))
        wb1c = sum(1 for bv, av in zip(b1cl, a3l) if _quant_dec(bv, 10 ** 9, "V_RAW_L") < _quant_dec(av, 10 ** 9, "V_RAW_L"))
        require(wb1b == _need_int(cm, "b1b_vs_a2_layer_hmse_wins", "V_REC_WB1B"), "V_XREF_WB1B", "b1b wins mismatch")
        require(wb1c == _need_int(cm, "b1c_vs_a3_layer_hmse_wins", "V_REC_WB1C"), "V_XREF_WB1C", "b1c wins mismatch")

        for aid in ARM_IDS:
            ra = _need_dict(rarms, aid, "V_RAW_AD")
            ca = _need_dict(carms, aid, "V_REC_AD")
            require(_need_int(ra, "ball_bound", "V_RAW_BB") == _need_int(ca, "ball_bound", "V_REC_BB"), "V_XREF_BB", "ball bound mismatch")
            tgt_str = _need_str(ra, "bpw_target", "V_RAW_TGT")
            try:
                tgt_dec = Decimal(tgt_str)
            except InvalidOperation:
                raise VerifyError("V_RAW_TGT", "invalid bpw_target string")
            require(_quant_dec(tgt_dec, 1000, "V_RAW_TGT") == _need_int(ca, "bpw_target_milli", "V_REC_TGT"), "V_XREF_TGT", "target bpw mismatch")
            require(_quant_dec(ra.get("effective_bpw"), 10 ** 6, "V_RAW_EFF") == _need_int(ca, "effective_bpw_micro", "V_REC_EFF"), "V_XREF_EFF", "eff bpw mismatch")
            require(_need_int(ra, "syndrome_errors", "V_RAW_SYN") == _need_int(ca, "syndrome_errors", "V_REC_SYN"), "V_XREF_SYN", "syn mismatch")
            require(_need_int(ra, "out_of_ball_groups", "V_RAW_OOB") == _need_int(ca, "out_of_ball_groups", "V_REC_OOB"), "V_XREF_OOB", "oob mismatch")
            require(_need_int(ra, "protected_non_block_diffs", "V_RAW_NBD") == _need_int(ca, "protected_non_block_diffs", "V_REC_NBD"), "V_XREF_NBD", "nbd mismatch")
            require(_quant_dec(ra.get("weighted_mse_rel"), 10 ** 9, "V_RAW_WM") == _need_int(ca, "weighted_mse_nano", "V_M"), "V_XREF_WM", "wmse mismatch")
            require(_quant_dec(ra.get("mean_p999_abs_err"), 10 ** 9, "V_RAW_P9") == _need_int(ca, "p999_abs_err_nano", "V_M"), "V_XREF_P9", "p999 mismatch")
            require(_quant_dec(ra.get("mean_p999_weighted_abs_err"), 10 ** 9, "V_RAW_WP9") == _need_int(ca, "wp999_abs_err_nano", "V_M"), "V_XREF_WP9", "wp999 mismatch")
            wt = _need_dict(ra, "wikitext2", "V_RAW_WT")
            py = _need_dict(ra, "python", "V_RAW_PY")
            require(_quant_dec(wt.get("ppl"), 10 ** 6, "V_RAW_WP") == _need_int(ca, "wikitext2_ppl_micro", "V_M"), "V_XREF_WP", "wiki ppl mismatch")
            require(_quant_dec(wt.get("mean_kl"), 10 ** 9, "V_RAW_WK") == _need_int(ca, "wikitext2_kl_nano", "V_M"), "V_XREF_WK", "wiki kl mismatch")
            require(_quant_dec(py.get("ppl"), 10 ** 6, "V_RAW_PP") == _need_int(ca, "python_ppl_micro", "V_M"), "V_XREF_PP", "py ppl mismatch")
            require(_quant_dec(py.get("mean_kl"), 10 ** 9, "V_RAW_PK") == _need_int(ca, "python_kl_nano", "V_M"), "V_XREF_PK", "py kl mismatch")

    return row_counts


def verify_arm_cs08_budget(
    raw_arm: dict[str, object],
    rec_arm: dict[str, object],
    total_wts: int,
    total_rows: int,
    a16_b30: dict[str, int],
    a16_b45: dict[str, int],
    e8_b: dict[str, int],
) -> tuple[int, int]:
    """Independently check exact non-circular CS-08 bit budget from theta bounds & counts."""
    aid = _need_str(raw_arm, "arm", "V_RAW_ANAME")
    gs = _need_int(raw_arm, "group_size", "V_RAW_GS")
    require(gs == 128, "V_CS08_GS", "expected group_size 128")
    lat = _need_str(raw_arm, "lattice", "V_RAW_LAT")
    bb = _need_int(rec_arm, "ball_bound", "V_REC_BB")
    tgt = _need_int(rec_arm, "bpw_target_milli", "V_REC_TGT")
    eff = _need_int(rec_arm, "effective_bpw_micro", "V_REC_EFF")

    if lat == "RTN":
        require(bb == 0, "V_CS08_RTN", "RTN bound must be 0")
        require(eff <= tgt * 1000, "V_REC_BPW", "CS-08 bit budget exceeded")
        return eff, eff

    if lat == "E8":
        exp_bb = e8_b["368"] if tgt == 3000 else e8_b["560"]
        pbits = 368 if tgt == 3000 else 560
        require(bb == exp_bb, "V_CS08_E8_BB", "E8 theta bound mismatch")
        require(eff == (pbits + 16) * 1000000 // gs, "V_CS08_E8_BPW", "E8 bpw mismatch")
        return eff, eff

    require(lat == "A16", "V_CS08_LAT", "unexpected lattice")
    if aid.startswith("B2b_"):
        require(bb == a16_b30["352"], "V_CS08_CHEB_BB", "Chebyshev bound mismatch")
        require(eff == 3000000, "V_CS08_CHEB_BPW", "Chebyshev matched bpw mismatch")
        return eff, eff

    if "CSR025" in aid:
        csr_n = _need_int(raw_arm, "csr_entries", "V_RAW_CSR")
        bulk_b = 357 if tgt == 3000 else 549
        exp_bb = a16_b30["357"] if tgt == 3000 else a16_b45["549"]
        require(bb == exp_bb, "V_CS08_CSR_BB", "CSR theta bound mismatch")
        q_pay = Fraction(bulk_b + 16, gs) + Fraction(32 * csr_n, total_wts)
        calc_m = int((Decimal(q_pay.numerator) * 1000000 / Decimal(q_pay.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        require(calc_m <= tgt * 1000, "V_REC_BPW", "CS-08 bit budget exceeded")
        require(eff == calc_m, "V_CS08_CSR_BPW", "CSR payload bpw mismatch")
        q_meta = q_pay + Fraction(16 * total_rows, total_wts)
        meta_m = int((Decimal(q_meta.numerator) * 1000000 / Decimal(q_meta.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return eff, meta_m

    if aid.startswith("B3b_"):
        hi_g = _need_int(raw_arm, "polar_hi_groups", "V_RAW_PHI")
        tot_g = total_wts // gs
        require(bb in (a16_b30["357"], a16_b30["368"]), "V_CS08_B3B_BB", "B3b theta bound mismatch")
        q_pay = Fraction(hi_g * 576 + (tot_g - hi_g) * 373, total_wts)
        calc_m = int((Decimal(q_pay.numerator) * 1000000 / Decimal(q_pay.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        require(calc_m <= tgt * 1000, "V_REC_BPW", "CS-08 bit budget exceeded")
        require(eff == calc_m, "V_CS08_B3B_BPW", "B3b payload bpw mismatch")
        q_meta = q_pay + Fraction(1, gs)
        meta_m = int((Decimal(q_meta.numerator) * 1000000 / Decimal(q_meta.denominator)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        return eff, meta_m

    exp_bb = a16_b30["368"] if tgt == 3000 else a16_b45["560"]
    pbits = 368 if tgt == 3000 else 560
    require(bb == exp_bb, "V_CS08_STD_BB", "A16 theta bound mismatch")
    require(eff == (pbits + 16) * 1000000 // gs, "V_CS08_STD_BPW", "A16 bpw mismatch")
    return eff, eff


def independent_empirical_replay(
    receipt: dict[str, object],
    raw_results: dict[str, object],
    raw_sha: str,
    a16_b30: dict[str, int],
    a16_b45: dict[str, int],
    e8_b: dict[str, int],
) -> dict[str, object]:
    """Independently verify receipt + raw empirical results & derive kill-gate verdicts."""
    require(_need_str(receipt, "schema", "V_REC_SCHEMA") == "aletheia.track_s165.empirical_receipt.v1", "V_REC_SCHEMA_VAL", "bad receipt schema")
    require(_need_str(receipt, "prereg_sha256", "V_REC_PREREG") == PIN_PREREG, "V_REC_PREREG_PIN", "prereg sha mismatch")
    require(_need_str(receipt, "energy_claim", "V_REC_ENERGY") == ENERGY_TAG, "V_REC_ENERGY_VAL", "bad energy claim")
    require(receipt.get("measured_live_model_weights") is True, "V_REC_LIVE", "live model weights flag required")

    vram_tot = _need_int(receipt, "gpu_vram_total_mib", "V_REC_VRAM")
    require(vram_tot > 0, "V_REC_VRAM_RNG", "vram total must be positive")
    require(vram_tot <= 10240, "V_REC_VRAM_RNG", "vram limit exceeded")

    row_counts = verify_raw_cross_linkage(receipt, raw_results, raw_sha)
    models_map = _need_dict(receipt, "models", "V_REC_MODELS")
    raw_models = _need_dict(raw_results, "models", "V_RAW_MODS")

    out_models: dict[str, object] = {}
    arm_total = 0
    kg1_a16_pass = 0
    kg1_b1c_pass = 0
    kg1_a16_cnt = 0
    kg2_p999_pass = 0
    kg2_p999_cnt = 0
    kg2_falsify_hits = 0
    b3_partial = True
    b3_full = True

    for mid in MODEL_IDS:
        md = _need_dict(models_map, mid, "V_REC_MD")
        rmd = _need_dict(raw_models, mid, "V_RAW_MD")
        layers = _need_int(md, "num_hidden_layers", "V_REC_LYR")
        peak = _need_int(md, "peak_vram_kib", "V_REC_PEAK")
        require(peak > 0, "V_REC_PEAK_CAP", "peak VRAM must be positive")
        require(peak < vram_tot * 1024, "V_REC_PEAK_CAP", "peak VRAM exceeded limit")
        require(_need_int(md, "prequant_fold_extra_bytes", "V_REC_FBYTES") == 0, "V_REC_FBYTES_Z", "nonzero fold bytes")
        fkl = _need_int(md, "prequant_fold_kl_nano", "V_REC_FKL")
        require(fkl >= 0, "V_REC_FKL_CAP", "negative fold KL")
        require(fkl < 5000000, "V_REC_FKL_CAP", "fold KL exceeded bound")
        w_b1b = _need_int(md, "b1b_vs_a2_layer_hmse_wins", "V_REC_WB1B")
        w_b1c = _need_int(md, "b1c_vs_a3_layer_hmse_wins", "V_REC_WB1C")

        arms = _need_dict(md, "arms", "V_REC_ARMS")
        rarms = _need_dict(rmd, "arms", "V_RAW_ARMS")
        wts = _need_int(rmd, "total_block_weights", "V_RAW_WTS")
        rows_tot = row_counts[mid]
        meta_bpw: dict[str, int] = {}

        for aid in ARM_IDS:
            ad = _need_dict(arms, aid, "V_REC_AD")
            rad = _need_dict(rarms, aid, "V_RAW_AD")
            require(_need_int(ad, "syndrome_errors", "V_REC_SYN") == 0, "V_REC_SYN_NZ", "nonzero syndrome error")
            require(_need_int(ad, "out_of_ball_groups", "V_REC_OOB") == 0, "V_REC_OOB_NZ", "nonzero OOB group")
            require(_need_int(ad, "protected_non_block_diffs", "V_REC_NBD") == 0, "V_REC_NBD_NZ", "non-block tensor altered")
            _eff, m_bpw = verify_arm_cs08_budget(rad, ad, wts, rows_tot, a16_b30, a16_b45, e8_b)
            meta_bpw[aid] = m_bpw
            arm_total = arm_total + 1

        a1 = _need_dict(arms, "A1_b30_E8_H8_hess", "V_REC_AD")
        a2 = _need_dict(arms, "A2_b30_A16_BlockH4_hess", "V_REC_AD")
        a3 = _need_dict(arms, "A3_b30_A16_BlockH4_QuadSnake_SN", "V_REC_AD")
        a4 = _need_dict(arms, "A4_b30_A16_BlockH4_QuadLPT_SN", "V_REC_AD")
        b1b = _need_dict(arms, "B1b_b30_JohnHess_BlockH4_Folded0B", "V_REC_AD")
        b1c = _need_dict(arms, "B1c_b30_JohnHess_BlockH4_QuadSnake_Folded0B", "V_REC_AD")
        b1d = _need_dict(arms, "B1d_b30_JohnHess_E8_H8_Folded0B", "V_REC_AD")
        b2a = _need_dict(arms, "B2a_b30_HellyCheb_MinimaxScale_QuadLPT_0B", "V_REC_AD")
        b2b = _need_dict(arms, "B2b_b30_HellyCheb_CenterScale_QuadLPT_Matched", "V_REC_AD")
        a4_45 = _need_dict(arms, "A4_b45_A16_BlockH4_QuadLPT_SN", "V_REC_AD")
        b2a_45 = _need_dict(arms, "B2a_b45_HellyCheb_MinimaxScale_QuadLPT_0B", "V_REC_AD")
        b3a = _need_dict(arms, "B3a_b30_PolarCap_CSR025_BlockH4_Matched", "V_REC_AD")
        b3b = _need_dict(arms, "B3b_b30_PolarCap_BlockSplit_1over19_Matched", "V_REC_AD")

        b1c_hmse_ok = _need_int(b1c, "weighted_mse_nano", "V_M") * 100 <= _need_int(a3, "weighted_mse_nano", "V_M") * 98
        b1c_ppl_ok = _need_int(a3, "wikitext2_ppl_micro", "V_M") - _need_int(b1c, "wikitext2_ppl_micro", "V_M") >= 10000
        b1b_hmse_ok = _need_int(b1b, "weighted_mse_nano", "V_M") * 100 <= _need_int(a2, "weighted_mse_nano", "V_M") * 98
        b1b_ppl_ok = _need_int(a2, "wikitext2_ppl_micro", "V_M") - _need_int(b1b, "wikitext2_ppl_micro", "V_M") >= 10000
        b1d_hmse_ok = _need_int(b1d, "weighted_mse_nano", "V_M") * 100 <= _need_int(a1, "weighted_mse_nano", "V_M") * 98
        b1d_ppl_ok = _need_int(a1, "wikitext2_ppl_micro", "V_M") - _need_int(b1d, "wikitext2_ppl_micro", "V_M") >= 10000
        b1b_a3_wiki = _need_int(a3, "wikitext2_ppl_micro", "V_M") - _need_int(b1b, "wikitext2_ppl_micro", "V_M")
        b1d_a1_wiki = _need_int(a1, "wikitext2_ppl_micro", "V_M") - _need_int(b1d, "wikitext2_ppl_micro", "V_M")

        for g in (b1c_hmse_ok, b1c_ppl_ok):
            if g:
                kg1_b1c_pass = kg1_b1c_pass + 1
                kg1_a16_pass = kg1_a16_pass + 1
            kg1_a16_cnt = kg1_a16_cnt + 1
        for g in (b1b_hmse_ok, b1b_ppl_ok):
            if g:
                kg1_a16_pass = kg1_a16_pass + 1
            kg1_a16_cnt = kg1_a16_cnt + 1

        b2a_ok = _need_int(b2a, "p999_abs_err_nano", "V_M") * 100 <= _need_int(a4, "p999_abs_err_nano", "V_M") * 85
        b2b_ok = _need_int(b2b, "p999_abs_err_nano", "V_M") * 100 <= _need_int(a4, "p999_abs_err_nano", "V_M") * 85
        b2a_45_ok = _need_int(b2a_45, "p999_abs_err_nano", "V_M") * 100 <= _need_int(a4_45, "p999_abs_err_nano", "V_M") * 85

        b2a_wiki_deg = _need_int(b2a, "wikitext2_ppl_micro", "V_M") > _need_int(a4, "wikitext2_ppl_micro", "V_M")
        b2b_wiki_deg = _need_int(b2b, "wikitext2_ppl_micro", "V_M") > _need_int(a4, "wikitext2_ppl_micro", "V_M")
        b2a_45_wiki_deg = _need_int(b2a_45, "wikitext2_ppl_micro", "V_M") > _need_int(a4_45, "wikitext2_ppl_micro", "V_M")
        b2a_py_deg = _need_int(b2a, "python_ppl_micro", "V_M") > _need_int(a4, "python_ppl_micro", "V_M")

        for p_ok, ppl_bad in ((b2a_ok, b2a_wiki_deg), (b2b_ok, b2b_wiki_deg), (b2a_45_ok, b2a_45_wiki_deg)):
            kg2_p999_cnt = kg2_p999_cnt + 1
            if p_ok:
                kg2_p999_pass = kg2_p999_pass + 1
            elif ppl_bad:
                kg2_falsify_hits = kg2_falsify_hits + 1

        b3a_wiki_gain = _need_int(a3, "wikitext2_ppl_micro", "V_M") - _need_int(b3a, "wikitext2_ppl_micro", "V_M")
        b3b_py_gain = _need_int(a3, "python_ppl_micro", "V_M") - _need_int(b3b, "python_ppl_micro", "V_M")
        b3b_lt_a2 = _need_int(b3b, "python_ppl_micro", "V_M") < _need_int(a2, "python_ppl_micro", "V_M")
        b3b_lt_a3 = _need_int(b3b, "python_ppl_micro", "V_M") < _need_int(a3, "python_ppl_micro", "V_M")
        b3b_best_py = b3b_lt_a2 and b3b_lt_a3
        if b3a_wiki_gain < 10000:
            b3_partial = False
        if not b3b_best_py:
            b3_partial = False
        if _need_int(b3b, "weighted_mse_nano", "V_M") >= _need_int(a2, "weighted_mse_nano", "V_M"):
            b3_full = False

        out_models[mid] = {
            "num_hidden_layers": layers,
            "peak_vram_kib": peak,
            "b1b_vs_a2_layer_hmse_wins": w_b1b,
            "b1c_vs_a3_layer_hmse_wins": w_b1c,
            "b1c_vs_a3_hmse_gate_pass": b1c_hmse_ok,
            "b1c_vs_a3_ppl_gate_pass": b1c_ppl_ok,
            "b1b_vs_a2_hmse_gate_pass": b1b_hmse_ok,
            "b1b_vs_a2_ppl_gate_pass": b1b_ppl_ok,
            "b1b_vs_a3_wiki_ppl_gain_micro": b1b_a3_wiki,
            "b1d_vs_a1_hmse_gate_pass": b1d_hmse_ok,
            "b1d_vs_a1_ppl_gate_pass": b1d_ppl_ok,
            "b1d_vs_a1_wiki_ppl_gain_micro": b1d_a1_wiki,
            "b2a_vs_a4_p999_15pct_pass": b2a_ok,
            "b2b_vs_a4_p999_15pct_pass": b2b_ok,
            "b2a_b45_vs_a4_p999_15pct_pass": b2a_45_ok,
            "b2a_vs_a4_wiki_ppl_degraded": b2a_wiki_deg,
            "b2b_vs_a4_wiki_ppl_degraded": b2b_wiki_deg,
            "b2a_b45_vs_a4_wiki_ppl_degraded": b2a_45_wiki_deg,
            "b2a_vs_a4_python_ppl_degraded": b2a_py_deg,
            "b3a_vs_a3_wiki_ppl_gain_micro": b3a_wiki_gain,
            "b3b_vs_a3_py_ppl_gain_micro": b3b_py_gain,
            "b3b_beats_a2_and_a3_python_ppl": b3b_best_py,
            "b3a_with_16b_rowptr_bpw_micro": meta_bpw["B3a_b30_PolarCap_CSR025_BlockH4_Matched"],
            "b3b_with_1bit_selector_map_bpw_micro": meta_bpw["B3b_b30_PolarCap_BlockSplit_1over19_Matched"],
        }

    if kg1_a16_pass == 0:
        kg1_v = "FALSIFIED"
    elif kg1_a16_pass == kg1_a16_cnt:
        kg1_v = "CONFIRMED"
    else:
        kg1_v = "PARTIAL"

    if kg2_p999_pass == kg2_p999_cnt:
        kg2_v = "CONFIRMED"
    elif kg2_p999_pass > 0:
        kg2_v = "PARTIAL"
    elif kg2_falsify_hits > 0:
        kg2_v = "FALSIFIED"
    else:
        kg2_v = "INCONCLUSIVE"

    if b3_full:
        b3_v = "CONFIRMED"
    elif b3_partial:
        b3_v = "PARTIAL"
    else:
        b3_v = "FALSIFIED"

    require(kg1_b1c_pass == 0, "V_KG1_B1C", "expected B1c primary apples-to-apples falsified")
    require(kg1_a16_pass == 1, "V_KG1_CNT", "expected exactly 1 KG1 A16 prong pass")
    require(kg2_p999_pass == 0, "V_KG2_P999", "unexpected KG2 p999 pass")
    require(kg2_falsify_hits >= 1, "V_KG2_PPL", "expected KG2 PPL degradation trigger")
    require(b3_v == "PARTIAL", "V_B3_PART", "expected B3 PARTIAL")

    overall_v = f"KG1_{kg1_v}_KG2_{kg2_v}_B3_{b3_v}"
    b3b_meta_ok = all(
        _need_dict(out_models, m, "V_MS")["b3b_with_1bit_selector_map_bpw_micro"] <= 3000000  # type: ignore[operator]
        for m in MODEL_IDS
    )
    b3a_meta_ok = all(
        _need_dict(out_models, m, "V_MS")["b3a_with_16b_rowptr_bpw_micro"] <= 3000000  # type: ignore[operator]
        for m in MODEL_IDS
    )

    return {
        "total_live_arms_verified": arm_total,
        "zero_syndrome_errors_all_arms": True,
        "zero_oob_groups_all_arms": True,
        "cs08_bit_budget_verified_all_arms": True,
        "b3b_payload_plus_scale_within_3bpw": True,
        "b3b_with_1bit_selector_map_within_3bpw": b3b_meta_ok,
        "b3a_with_16b_rowptr_within_3bpw": b3a_meta_ok,
        "recorded_run_b3_csr_mode": "additive_residual_plus_dense_quant_patched_in_bench",
        "recorded_run_python_eval_scope": "leading_4096_tokens_fractions_py_only_patched_in_bench",
        "kg1_a16_prongs_passed": kg1_a16_pass,
        "kg1_a16_prongs_total": kg1_a16_cnt,
        "kg1_john_b1_verdict": kg1_v,
        "kg2_p999_gates_passed": kg2_p999_pass,
        "kg2_ppl_degraded_without_p999_hits": kg2_falsify_hits,
        "kg2_helly_b2_verdict": kg2_v,
        "arm_b3_polar_cap_verdict": b3_v,
        "overall_empirical_verdict": overall_v,
        "models": out_models,
    }


def build_verified_payload(
    base: Path | None = None,
    *,
    run_live_lean: bool = False,
    lean_runner: object = None,
) -> dict[str, object]:
    root = track_dir() if base is None else base
    lean_info = verify_lean_bundle(root, run_live_lean=run_live_lean, lean_runner=lean_runner)
    rec_raw = read_strict_bytes(root / RECEIPT_NAME)
    emp_raw = read_strict_bytes(root / RAW_RESULTS_NAME)
    rec_sha = hashlib.sha256(rec_raw).hexdigest()
    emp_sha = hashlib.sha256(emp_raw).hexdigest()
    require(rec_sha == PIN_RECEIPT, "PIN_RECEIPT", "empirical receipt sha mismatch")
    require(emp_sha == PIN_EMP_RAW, "PIN_EMP_RAW", "empirical results sha mismatch")
    rec_obj = parse_strict_json(decode_strict_utf8_lf(rec_raw), allow_decimal=False)
    raw_obj = parse_strict_json(decode_strict_utf8_lf(emp_raw), allow_decimal=True)

    a16_all = independent_a16_theta_bounds(5220, (352, 357, 368, 544, 549, 560))
    theta_b30 = {k: a16_all[k] for k in ("352", "357", "368")}
    theta_b45 = {k: a16_all[k] for k in ("544", "549", "560")}
    theta_e8 = independent_e8_theta_bounds(3400, (368, 560))

    payload: dict[str, object] = {
        "schema": SCHEMA_ID,
        "track": TRACK_TAG,
        "experiment_id": EXP_TAG,
        "prereg_id": PREREG_TAG,
        "prereg_sha256": PIN_PREREG,
        "oracle_class": ORACLE_KIND,
        "novelty_classification": NOVELTY_TAG,
        "energy_claim": ENERGY_TAG,
        "lean": lean_info,
        "pillar1_john_geometry": independent_pillar1(),
        "pillar2_zero_byte_folding": independent_pillar2(),
        "pillar3_helly_and_polar_bounds": independent_pillar3(),
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
            "empirical_gates": independent_empirical_replay(rec_obj, raw_obj, emp_sha, theta_b30, theta_b45, theta_e8),
        },
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def verify_exact_tree(actual: object, expected: object) -> None:
    """Iterative exact-type bijective tree equality (rejects bool-as-int and partial dicts)."""
    stack: list[tuple[object, object]] = [(actual, expected)]
    while stack:
        a, e = stack.pop()
        require(type(a) is type(e), "CERT_TYPE", "certificate value type mismatch")
        if type(e) is dict:
            require(frozenset(a.keys()) == frozenset(e.keys()), "CERT_KEYS", "certificate keyset mismatch")  # type: ignore[union-attr]
            for k in e:
                stack.append((a[k], e[k]))  # type: ignore[index]
        elif type(e) is list:
            require(len(a) == len(e), "CERT_LEN", "certificate list length mismatch")  # type: ignore[arg-type]
            for ai, ei in zip(a, e):  # type: ignore[call-overload]
                stack.append((ai, ei))
        else:
            require(a == e, "CERT_VAL", "certificate leaf mismatch")


class _StrictVerifyParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise VerifyError("BAD_CLI_ARGS", message)


def run_cli(argv: list[str] | None = None) -> int:
    parser = _StrictVerifyParser(description="Track S165 zero-shared-code verifier", add_help=False, allow_abbrev=False)
    parser.add_argument("--check", nargs="?", const=CERT_NAME, default=None)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--live-lean", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.check is not None and args.standalone:
        raise VerifyError("MODE_CONFLICT", "cannot combine --check and --standalone")

    base = track_dir()
    payload = build_verified_payload(base=base, run_live_lean=args.live_lean)
    if not args.standalone:
        cert_arg = CERT_NAME if args.check is None else args.check
        cpath = validate_cli_path(cert_arg)
        resolved = cpath if cpath.is_absolute() else (base / cpath)
        cert_obj = parse_strict_json(decode_strict_utf8_lf(read_strict_bytes(resolved)))
        verify_exact_tree(cert_obj, payload)

    if args.json_out:
        os.write(1, (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"))
    else:
        os.write(1, ("PASS " + SCHEMA_ID + "\n").encode("utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except VerifyError as exc:
        line = json.dumps({"status": "FAIL", "code": exc.code, "message": exc.message}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, line.encode("utf-8"))
        return 2
    except subprocess.SubprocessError as exc:
        line = json.dumps({"status": "FAIL", "code": "SUBPROCESS_ERROR", "message": type(exc).__name__}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, line.encode("utf-8"))
        return 2
    except OSError as exc:
        line = json.dumps({"status": "FAIL", "code": "OS_ERROR", "message": type(exc).__name__}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        os.write(2, line.encode("utf-8"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
