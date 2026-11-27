#!/usr/bin/env python3
"""Track S167 (EXP-Q51a/b/c, PREREG-W22-ES-SCALE) Zero-Shared-Code Verifier.

Independently recomputes every exact Z/Q mathematical invariant (direct expansion,
Walsh-Hadamard bit-parity, exact Choromanski Thm 3.2 MSE difference, math.comb
binomials + row-sum verification, first-principles BF16/Al-Kashi/CS-08/+0 B fold,
exact Gauss-Jordan (I+S)(I-S)^-1 Cayley-SO(3) inversion, full disk-read & rational
receipt recomputation from s167_empirical_results.json, and programmatic kill-gate
verdicts), Zone-B strict O_NOFOLLOW/hardlink/UTF-8 I/O, strict bool/int JSON tree
equality, hardened static Lean 4.36.0 R12/R13b audit, and RO-safe UTF-8-hardened
--live-lean verification with exact version & axiom-split enforcement.

Zero shared code with s167_es_oracle.py (no local imports).
Standard library only. Zero floating-point literals or calls, zero bare asserts.
"""
from __future__ import annotations

import argparse
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

SCHEMA_ID = "s167.q51.es_siliquant_holosphere.v2"
TRACK_TAG = "S167"
EXP_TAGS = ("EXP-Q51a", "EXP-Q51b", "EXP-Q51c")
PREREG_TAG = "PREREG-W22-ES-SCALE"
ORACLE_KIND = "exact_and_sampled"
NOVELTY_TAG = "N1"
ENERGY_TAG = "UNCONTROLLED_LAPTOP_NOT_CLAIMED"

MOD_NAME = "EvolutionOperators"
LEAN_SRC_NAME = "EvolutionOperators.lean"
PREREG_NAME = "PREREG_W22_ES_SCALE.json"
BENCH_NAME = "s167_es_live_bench.py"
RESULTS_NAME = "s167_empirical_results.json"
RECEIPT_NAME = "s167_empirical_receipt.json"
CERT_NAME = "s167_es_certificate.json"
TOOLCHAIN_EXPECTED = "leanprover/lean4:v4.36.0"
THM_EXPECTED = 21
EXPECTED_AX_SPLIT = {"zero_axiom": 0, "propext_only": 1, "propext_quot": 20}

PIN_PREREG = "1f9db607ce34de1e0cbd472625d411d8c5e9857468a107a0cc24e558a733d3f6"
PIN_LEAN = "62c416750d66def4fa3e003363c134d421d300996a234af16841580497490dd3"
PIN_TOOLCHAIN = "582a63f8f9c5128067b688778c1051cf1464c3ca9d57b9325025c051c29b774a"
PIN_LAKEFILE = "4e16492fd574731d4dc2a998c56d596198dc3083565a768c28dfbd9dad60ece4"
PIN_MANIFEST = "5437b4c916b5f0610868c90230b06aac0abd9c11ba0639837e654b03c001c122"
PIN_BENCH = "759852360383f08c46eb10fcb3e8aa1c744e11bfc8b7e8a6c9427f19ae8e4439"
PIN_EMP_RAW = "d6edf352c5b8ba76d04550a506812c8c22223cb117a666582fb065f65cde9e67"
PIN_RECEIPT = "d1ee363f31f40cf793bdc74ac56f5378196aa50f4c0b64c75b9cc6d152dd66b4"

ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound"})
MAX_FILE_BYTES = 1048576
MAX_JSON_DEPTH = 32
MAX_DIGIT_RUN = 256
SHELL_CAP_NANO = 10 ** 15
GRID_CELLS = (
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

# Hardened Lean R12/R13b scanner ([MED-10]: closes stop, variable/include, @[default_instance], instance/attribute)
_FORBIDDEN_LEAN_RE = re.compile(
    r'["\u00ab\u00bb]'
    r"|@\["
    r"|\b(?:sorry|admit|sorryAx|mkSorry|unsafe|partial|opaque|noncomputable|native_decide|bv_decide|bv_check"
    r"|Classical|ofReduceBool|ofReduceNat|trustCompiler|implemented_by|extern|csimp|export|initialize|builtin_initialize"
    r"|macro|macro_rules|elab|elab_rules|syntax|declare_syntax_cat|notation|prefix|infix|infixl|infixr|postfix"
    r"|set_option|dbg_trace|dbg_sleep|dbg_stack_trace|panic|unreachable|run_cmd|run_meta"
    r"|stop|variable|variables|include|omit|instance|default_instance|attribute)\b"
    r"|(?<!\.)\baxiom\b"
    r"|\bimport\b"
    r"|#\s*(?:eval|reduce|exit|check_failure|check|synth|guard_msgs|guard_expr|guard)\b"
    r"|(?<!\w)\+\s*native\b"
    r"|\bnative\s*:="
)
_PUB_THM_RE = re.compile(r"(?:^|\n)[ \t]*theorem[ \t]+([A-Za-z0-9_.']+)")
_PRINT_AX_RE = re.compile(r"(?:^|\n)[ \t]*#print[ \t]+axioms[ \t]+EvolutionOperators\.([A-Za-z0-9_.']+)")
_ANY_PRINT_RE = re.compile(r"(?:^|\n)[ \t]*#[ \t]*print\b")
_TOML_REQ_RE = re.compile(r"(?:^|\n)[ \t]*(?:\[\[\s*require\s*\]\]|require\s*=|\[[ \t]*require[ \t]*\])")
_AX_LINE_RE = re.compile(r"^'EvolutionOperators\.([A-Za-z0-9_.']+)'\s+(does not depend on any axioms|depends on axioms:\s*\[([^\]]*)\])$")
_LEAN_VER_RE = re.compile(r"\bversion\s+4\.36\.0(?=[,\s\)]|$)")


class VerifyError(ValueError):
    """Fail-closed verification error with structured error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _StrictVerifyParser(argparse.ArgumentParser):
    """Fail-closed ArgumentParser: rejects abbreviations and routes errors to VerifyError."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise VerifyError("BAD_CLI_ARGS", message)


def require(cond: bool, code: str, message: str) -> None:
    if cond is not True:
        raise VerifyError(code, message)


def fmt_q(q: Fraction) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def track_dir() -> Path:
    here = Path(__file__).parent
    return here if here.is_absolute() else (Path(os.getcwd()) / here)


def _split_parts(path: object) -> tuple[str, ...]:
    s = str(path) if isinstance(path, Path) else path
    require(type(s) is str, "BAD_PATH_TYPE", "path must be str or Path")
    require(len(s) > 0, "BAD_PATH_EMPTY", "empty path rejected")
    require("\x00" not in s, "BAD_PATH_NUL", "NUL byte rejected")
    p = Path(s)
    parts = p.parts if p.is_absolute() else (Path.cwd() / p).parts
    require(len(parts) >= 2, "ROOT_PATH", "cannot open root directory")
    for seg in parts[1:]:
        require(seg != "..", "DOTDOT_PATH", "parent traversal rejected")
    return parts


def _open_parent_dir_fd(path: object) -> tuple[int, str]:
    parts = _split_parts(path)
    dfd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        for seg in parts[1:-1]:
            nxt = os.open(seg, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            os.close(dfd)
            dfd = nxt
    except OSError:
        os.close(dfd)
        raise VerifyError("PARENT_OPEN", "cannot open parent directory without symlinks")
    return dfd, parts[-1]


def read_strict_bytes(path: object) -> bytes:
    dfd, leaf = _open_parent_dir_fd(path)
    try:
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError:
            raise VerifyError("FILE_OPEN", "cannot open target file")
    finally:
        os.close(dfd)
    bufs: list[bytes] = []
    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode), "NOT_REG_FILE", "target is not a regular file")
        require(st.st_nlink == 1, "HARDLINK_READ", "hardlinked input rejected")
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


def write_strict_atomic(path: object, raw: bytes) -> None:
    dfd, leaf = _open_parent_dir_fd(path)
    tmp_name: str | None = None
    try:
        try:
            existing = os.lstat(leaf, dir_fd=dfd)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            require(stat.S_ISREG(existing.st_mode), "WRITE_NOT_REG", "write target is not a regular file")
            require(existing.st_nlink == 1, "WRITE_HARDLINK", "hardlinked write target rejected")
        tmp_name = ".s167_ver_tmp_" + str(os.getpid()) + "_" + hashlib.sha256(raw).hexdigest()[:16]
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


def decode_strict_utf8_lf(raw: bytes) -> str:
    try:
        txt = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise VerifyError("BAD_UTF8", "invalid utf-8 sequence")
    require(txt.startswith("\ufeff") is False, "BAD_BOM", "UTF-8 BOM rejected")
    require("\r" not in txt, "BAD_CRLF", "carriage return rejected")
    require(txt.endswith("\n"), "BAD_FINAL_LF", "missing trailing newline")
    return txt


def parse_strict_json(text: str, *, allow_rationals: bool = False) -> dict[str, object]:
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
            if prev_d and not allow_rationals:
                require(ch not in ".eE", "FLOAT_TOKEN", "floating-point literal rejected")
            drun = 0
            prev_d = False
    require(depth == 0, "JSON_EOF_DEPTH", "unbalanced json nesting")
    require(in_str is False, "JSON_EOF_STR", "unterminated json string")

    def _no_dups(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in pairs:
            require(k not in out, "DUP_KEY", "duplicate object key")
            out[k] = v
        return out

    def _no_const(_tok: str) -> None:
        raise VerifyError("BAD_CONST", "non-JSON numeric constant")

    try:
        if allow_rationals:
            data = json.loads(text, object_pairs_hook=_no_dups, parse_constant=_no_const, parse_float=Fraction)
        else:
            data = json.loads(text, object_pairs_hook=_no_dups, parse_constant=_no_const)
    except VerifyError:
        raise
    except (json.JSONDecodeError, ValueError):
        raise VerifyError("BAD_JSON", "malformed json document")
    require(type(data) is dict, "NOT_OBJECT", "JSON root must be an object")
    return data


def strip_lean_comments(src: str) -> str:
    """Strip -- line comments and nested /- -/ block comments."""
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


def check_axiom_output(lines: list[str], expected_theorems: set[str]) -> dict[str, int]:
    """Join wrapped Lean #print axioms lines, verify exact theorem names, allowed axioms, and split counts."""
    joined: list[str] = []
    for raw_ln in lines:
        if raw_ln.strip() == "":
            continue
        if raw_ln[0].isspace():
            require(len(joined) > 0, "AXIOM_ORPHAN_CONT", "continuation line without header")
            joined[-1] = joined[-1] + " " + raw_ln.strip()
        else:
            joined.append(raw_ln.strip())

    require(len(joined) == len(expected_theorems), "AXIOM_COUNT", "axiom output entry count mismatch")
    seen: set[str] = set()
    zero_ax = 0
    propext_only = 0
    propext_quot = 0
    for entry in joined:
        m = _AX_LINE_RE.match(entry)
        require(m is not None, "AXIOM_FORMAT", "invalid axiom output entry")
        thm_name = m.group(1)  # type: ignore[union-attr]
        require(thm_name not in seen, "AXIOM_DUP_THM", "duplicate theorem in axiom output")
        seen.add(thm_name)
        body = m.group(2)  # type: ignore[union-attr]
        if body == "does not depend on any axioms":
            zero_ax = zero_ax + 1
        else:
            ax_csv = m.group(3)  # type: ignore[union-attr]
            ax_set: set[str] = set()
            for tok in ax_csv.split(","):
                name = tok.strip()
                if name != "":
                    require(name in ALLOWED_AXIOMS, "FORBIDDEN_AXIOM", f"disallowed axiom {name}")
                    ax_set.add(name)
            if ax_set == {"propext"}:
                propext_only = propext_only + 1
            else:
                propext_quot = propext_quot + 1
    require(seen == expected_theorems, "AXIOM_THM_SET", "axiom theorem set mismatch")
    return {"zero_axiom": zero_ax, "propext_only": propext_only, "propext_quot": propext_quot}


def _decode_subproc_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise VerifyError("LIVE_LEAN_UTF8", "non-UTF-8 output from lean/lake subprocess")


def default_live_lean_runner(pinned_files: dict[str, bytes]) -> dict[str, object]:
    """Materialize pinned Lean bytes in a private tempdir (RO-safe, zero TOCTOU, zero .lake/, UTF-8 fail-closed)."""
    home = os.path.expanduser("~")
    env = {
        "PATH": str(Path(home) / ".elan" / "bin") + ":" + str(Path(home) / ".local" / "bin") + ":/usr/bin:/bin",
        "HOME": home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        for fn, data in pinned_files.items():
            (tmp_p / fn).write_bytes(data)
        ver = subprocess.run(["lean", "--version"], cwd=tmp, env=env, capture_output=True, text=False, timeout=60, check=False)
        bld = subprocess.run(["lake", "build"], cwd=tmp, env=env, capture_output=True, text=False, timeout=180, check=False)
        aud = subprocess.run(["lean", LEAN_SRC_NAME], cwd=tmp, env=env, capture_output=True, text=False, timeout=180, check=False)
    return {
        "version_rc": ver.returncode,
        "version_output": _decode_subproc_utf8(ver.stdout) + _decode_subproc_utf8(ver.stderr),
        "build_rc": bld.returncode,
        "build_output": _decode_subproc_utf8(bld.stdout) + _decode_subproc_utf8(bld.stderr),
        "axiom_rc": aud.returncode,
        "axiom_stderr": _decode_subproc_utf8(aud.stderr),
        "axiom_stdout": _decode_subproc_utf8(aud.stdout),
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
    require(len(lean_paths) == 1, "LEAN_FILE_COUNT", "expected single Lean file")
    require(lean_paths[0].name == LEAN_SRC_NAME, "LEAN_FILE_NAME", "expected EvolutionOperators.lean")
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
    require(len(prints) == THM_EXPECTED, "LEAN_PRINT_COUNT", "axiom print count mismatch")
    require(set(thms) == set(prints), "LEAN_AUDIT_SET", "theorem vs #print axioms mismatch")

    lean_sha = hashlib.sha256(lean_raw).hexdigest()
    tc_sha = hashlib.sha256(tc_raw).hexdigest()
    lf_sha = hashlib.sha256(lf_raw).hexdigest()
    mf_sha = hashlib.sha256(mf_raw).hexdigest()
    if check_pins:
        require(lean_sha == PIN_LEAN, "PIN_LEAN", "EvolutionOperators.lean sha mismatch")
        require(tc_sha == PIN_TOOLCHAIN, "PIN_TC", "lean-toolchain sha mismatch")
        require(lf_sha == PIN_LAKEFILE, "PIN_LF", "lakefile.toml sha mismatch")
        require(mf_sha == PIN_MANIFEST, "PIN_MF", "lake-manifest.json sha mismatch")

    runner = lean_runner if lean_runner is not None else (default_live_lean_runner if run_live_lean else None)
    if runner is not None:
        pinned_map = {
            LEAN_SRC_NAME: lean_raw,
            "lean-toolchain": tc_raw,
            "lakefile.toml": lf_raw,
            "lake-manifest.json": mf_raw,
        }
        res = runner(pinned_map)  # type: ignore[operator]
        require(res["version_rc"] == 0, "LIVE_LEAN_VRC", "lean --version failed")
        require(_LEAN_VER_RE.search(str(res["version_output"])) is not None, "LIVE_LEAN_VER", "exact Lean 4.36.0 required")
        require(res["build_rc"] == 0, "LIVE_LAKE_RC", "lake build failed")
        require("warning:" not in str(res["build_output"]).lower(), "LIVE_LAKE_WARN", "lake build warning")
        require(res["axiom_rc"] == 0, "LIVE_AX_RC", "lean axiom check failed")
        require(res["axiom_stderr"] == "", "LIVE_AX_STDERR", "lean axiom stderr non-empty")
        ax_split = check_axiom_output(str(res["axiom_stdout"]).splitlines(), set(thms))
        require(ax_split == EXPECTED_AX_SPLIT, "LIVE_AX_SPLIT", "unexpected Lean axiom profile split")

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


def independent_pillar1() -> dict[str, object]:
    """Direct monomial expansion verification of antithetic odd/even cancellation."""
    c0, c1, c2, c3, c4 = Fraction(17), Fraction(-5), Fraction(11), Fraction(3), Fraction(-7)
    x = Fraction(3, 1000) * Fraction(5, 7)

    def _p(z: Fraction) -> Fraction:
        return c0 + c1 * z + c2 * (z ** 2) + c3 * (z ** 3) + c4 * (z ** 4)

    diff = _p(x) - _p(-x)
    summ = _p(x) + _p(-x)
    g = (Fraction(3, 2), Fraction(-5, 4), Fraction(7, 8), Fraction(-9, 16))
    e = (Fraction(1), Fraction(-1), Fraction(1), Fraction(-1))
    lin = sum(a * b for a, b in zip(g, e))
    sig = Fraction(3, 1000)
    b_tot = Fraction(13, 5) + Fraction(29, 7)
    return {
        "poly4_antithetic_diff": fmt_q(diff),
        "poly4_antithetic_sum": fmt_q(summ),
        "vec4_antithetic_diff": fmt_q(2 * sig * lin),
        "vec4_antithetic_sum": fmt_q(2 * b_tot),
        "onesided_bias_excess_sq2": fmt_q(2 * b_tot * b_tot),
    }


def verify_orth_gram(rows: list[list[int]], dim: int, diag_scale: int) -> int:
    """Verify exact Gram matrix U U^T = diag_scale * I_dim."""
    require(len(rows) == dim, "V_P2_DIM", "row count must equal dim")
    for i in range(dim):
        for j in range(dim):
            entry = sum(rows[i][k] * rows[j][k] for k in range(dim))
            require(entry == (diag_scale if i == j else 0), "V_P2_ORTH", "orthogonality failed")
    return dim


def independent_pillar2() -> dict[str, object]:
    """Walsh-Hadamard bit-parity construction of H_4 and first-principles Choromanski Thm 3.2 MSE ratio."""
    h4 = [[(-1 if ((r & c).bit_count() & 1) else 1) for c in range(4)] for r in range(4)]
    s = (1, -1, -1, 1)
    u = [[h4[r][c] * s[c] for c in range(4)] for r in range(4)]
    verify_orth_gram(u, 4, 4)
    v = (3, -5, 7, -11)
    v_sq = sum(t * t for t in v)
    hv_sq = sum((sum(u[r][c] * v[c] for c in range(4))) ** 2 for r in range(4))
    g = (13, -7, 19, -23)
    err_sq = Fraction(0)
    for c in range(4):
        est = sum(Fraction(sum(g[j] * u[r][j] for j in range(4)) * u[r][c], 4) for r in range(4))
        err_sq = err_sq + (est - g[c]) ** 2

    # First-principles Choromanski Thm 3.2: MSE_iid = (d+1)/N, MSE_orth = MSE_iid - (N-1)/N
    choro: dict[str, str] = {}
    for d, n in ((4, 4), (16, 16), (128, 16), (128, 30)):
        mse_iid = Fraction(d + 1, n)
        mse_orth = mse_iid - Fraction(n - 1, n)
        choro[f"d{d}_n{n}"] = fmt_q(mse_orth / mse_iid)

    return {
        "h4_parseval_sq": hv_sq,
        "h4_input_sq": v_sq,
        "h4_exact_recovery_err_sq": int(err_sq),
        "choromanski_orth_over_iid_mse": choro,
    }


def independent_pillar3() -> dict[str, object]:
    """Direct math.comb central binomial & exact closed-form Gaussian ratio verification."""
    w2 = 1
    sig2 = 100
    big_w2 = 10000
    r1 = Fraction(w2 * (big_w2 + sig2), big_w2 * (w2 + sig2))
    masses = [fmt_q(Fraction(math.comb(2 * m, m), 4 ** m)) for m in range(7)]
    plat_ok = all(sum(math.comb(2 * m, j) for j in range(2 * m + 1)) == (4 ** m) for m in range(7))
    return {
        "spike_over_plateau_ratio": {
            "r1": fmt_q(r1),
            "r2": fmt_q(r1 ** 2),
            "r4": fmt_q(r1 ** 4),
        },
        "pascal_center_spike_masses_m0_to_m6": masses,
        "pascal_plateau_conservation_exact": plat_ok,
    }


def check_folded_scale_invariance(quads: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]) -> Fraction:
    """Verify exact +0 B reciprocal scale folding invariance over Q."""
    fold = sum((wi * pi) * (xi * qi) for wi, xi, pi, qi in quads)
    plain = sum(wi * xi for wi, xi, _, _ in quads)
    require(fold == plain, "V_P4_FOLD", "reciprocal fold check failed")
    return fold


def independent_pillar4() -> dict[str, object]:
    """First-principles verification of BF16 ULP stall, Al-Kashi centered EF, CS-08, & +0 B folding."""
    mantissa_bits = 7
    ulp = Fraction(1, 1 << mantissa_bits)
    half_ulp = ulp / 2
    alpha = Fraction(1, 2000)
    stall = alpha / half_ulp

    # Rational nearest-grid rounding check on |w|=1 (k=1) with gHat=7 (stalls) vs gHat=8 (advances)
    def _grid_step_q(w_q: Fraction, delta_q: Fraction) -> int:
        # Round (w_q + delta_q)/ulp with half-up tie rule
        shifted = (w_q + delta_q) / ulp + Fraction(1, 2)
        return (shifted.numerator // shifted.denominator) - int(w_q / ulp)

    g7_mv = _grid_step_q(Fraction(1), 7 * alpha)
    g8_mv = _grid_step_q(Fraction(1), 8 * alpha)

    # Centered Al-Kashi FixedRadix Q = 64 (half_q = 32)
    w = 0
    e = 0
    hq = 32
    q = 2 * hq
    for _ in range(16):
        u = e + 8
        dq = (u + hq) // q
        w = w + dq
        e = u - dq * q

    es_fwd = 2
    drgrpo_flops = 6
    grpo_flops = 8
    hbm_grpo = 3

    quads = (
        (Fraction(3, 4), Fraction(5, 3), Fraction(7, 2), Fraction(1, Fraction(7, 2))),
        (Fraction(-5, 8), Fraction(2, 5), Fraction(3, 4), Fraction(1, Fraction(3, 4))),
        (Fraction(11, 16), Fraction(9, 7), Fraction(4, 9), Fraction(1, Fraction(4, 9))),
    )
    fold = check_folded_scale_invariance(quads)
    extra_tensors: tuple[int, ...] = ()

    return {
        "bf16_ulp_at_1": fmt_q(ulp),
        "bf16_half_ulp_at_1": fmt_q(half_ulp),
        "vsonic_alpha_over_half_ulp": fmt_q(stall),
        "bf16_canonical_g7_moves": g7_mv,
        "bf16_canonical_g8_moves": g8_mv,
        "alkashi_q64_16x8_weight_advance": w,
        "alkashi_q64_16x8_residual": e,
        "cs08_es_over_drgrpo_flops": fmt_q(Fraction(es_fwd, drgrpo_flops)),
        "cs08_es_over_grpo_flops": fmt_q(Fraction(es_fwd, grpo_flops)),
        "cs08_hbm_ratio_n16": fmt_q(Fraction(2 * 16, hbm_grpo)),
        "cs08_hbm_ratio_n30": fmt_q(Fraction(2 * 30, hbm_grpo)),
        "folded_scale_dot_exact": fmt_q(fold),
        "folded_scale_extra_bytes": sum(extra_tensors),
    }


def _mat3_inv_exact(a: list[list[Fraction]]) -> list[list[Fraction]]:
    """Exact Gauss-Jordan 3x3 matrix inverse over Q (independent derivation of (I+S)(I-S)^-1)."""
    aug = [a[r][:] + [Fraction(1 if r == c else 0) for c in range(3)] for r in range(3)]
    for col in range(3):
        piv = col
        while piv < 3 and aug[piv][col] == 0:
            piv = piv + 1
        require(piv < 3, "V_MAT_SING", "singular 3x3 matrix")
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [val / pv for val in aug[col]]
        for r in range(3):
            if r != col:
                fac = aug[r][col]
                aug[r] = [aug[r][c] - fac * aug[col][c] for c in range(6)]
    return [row[3:] for row in aug]


def independent_pillar5() -> dict[str, object]:
    """Independent Gauss-Jordan matrix inversion Q = (I + S)(I - S)^-1 on skew-symmetric tangent S."""
    vx, vy, vz = Fraction(3, 50), Fraction(-4, 50), Fraction(2, 50)
    skew = [
        [Fraction(0), -vz, vy],
        [vz, Fraction(0), -vx],
        [-vy, vx, Fraction(0)],
    ]
    i_plus = [[Fraction(1 if r == c else 0) + skew[r][c] for c in range(3)] for r in range(3)]
    i_minus = [[Fraction(1 if r == c else 0) - skew[r][c] for c in range(3)] for r in range(3)]
    inv_m = _mat3_inv_exact(i_minus)
    q_mat = [
        [sum(i_plus[r][k] * inv_m[k][c] for k in range(3)) for c in range(3)]
        for r in range(3)
    ]
    max_orth_err = Fraction(0)
    for r in range(3):
        for c in range(3):
            qtq = sum(q_mat[k][r] * q_mat[k][c] for k in range(3))
            d_err = abs(qtq - Fraction(1 if r == c else 0))
            if d_err > max_orth_err:
                max_orth_err = d_err
    det = (
        q_mat[0][0] * (q_mat[1][1] * q_mat[2][2] - q_mat[1][2] * q_mat[2][1])
        - q_mat[0][1] * (q_mat[1][0] * q_mat[2][2] - q_mat[1][2] * q_mat[2][0])
        + q_mat[0][2] * (q_mat[1][0] * q_mat[2][1] - q_mat[1][1] * q_mat[2][0])
    )
    denom = 50 * 50 + 3 * 3 + 4 * 4 + 2 * 2
    n2 = Fraction(3 * 3 + 4 * 4 + 2 * 2, 2500)
    su2_norm = ((1 - n2) / (1 + n2)) ** 2 + 4 * n2 / ((1 + n2) ** 2)
    euc_err = Fraction(1, 10) ** 2 * (Fraction(3, 5) ** 2 + Fraction(4, 5) ** 2)
    return {
        "cayley_su2_denom": denom,
        "cayley_su2_norm_sq": fmt_q(su2_norm),
        "cayley_so3_orth_max_err": fmt_q(max_orth_err),
        "cayley_so3_det": fmt_q(det),
        "cayley_so3_m00": fmt_q(q_mat[0][0]),
        "cayley_so3_m01": fmt_q(q_mat[0][1]),
        "euclidean_additive_shell_err": fmt_q(euc_err),
    }


def _need_int(obj: dict[str, object], key: str, code: str) -> int:
    require(key in obj, code, f"missing int key {key}")
    val = obj[key]
    require(type(val) is int, code, f"non-int key {key}")
    return val  # type: ignore[return-value]


def _need_bool(obj: dict[str, object], key: str, code: str) -> bool:
    require(key in obj, code, f"missing bool key {key}")
    val = obj[key]
    require(type(val) is bool, code, f"non-bool key {key}")
    return val  # type: ignore[return-value]


def _need_str(obj: dict[str, object], key: str, code: str) -> str:
    require(key in obj, code, f"missing str key {key}")
    val = obj[key]
    require(type(val) is str, code, f"non-str key {key}")
    return val  # type: ignore[return-value]


def _need_dict(obj: dict[str, object], key: str, code: str) -> dict[str, object]:
    require(key in obj, code, f"missing dict key {key}")
    val = obj[key]
    require(type(val) is dict, code, f"non-dict key {key}")
    return val  # type: ignore[return-value]


def _need_list(obj: dict[str, object], key: str, code: str) -> list[object]:
    require(key in obj, code, f"missing list key {key}")
    val = obj[key]
    require(type(val) is list, code, f"non-list key {key}")
    return val  # type: ignore[return-value]


def _need_rat(obj: dict[str, object], key: str, code: str) -> Fraction:
    require(key in obj, code, f"missing rational key {key}")
    val = obj[key]
    if type(val) is int:
        return Fraction(val, 1)
    require(type(val) is Fraction, code, f"non-rational key {key}")
    return val  # type: ignore[return-value]


def _round_scaled(q: Fraction, mult: int) -> int:
    return int(round(q * mult))


def recompute_receipt_from_results(raw_res: dict[str, object], raw_sha: str, prereg_sha: str, bench_sha: str) -> dict[str, object]:
    """Independently recompute the entire pure-integer receipt over Q directly from s167_empirical_results.json."""
    require(_need_str(raw_res, "schema", "V_RES_SCH") == "aletheia.track_s167.empirical_results.v2", "V_RES_SCH_V", "bad results schema")
    require(_need_str(raw_res, "prereg_sha256", "V_RES_PR") == prereg_sha, "V_RES_PR_M", "results prereg mismatch")
    require(_need_str(raw_res, "bench_script_sha256", "V_RES_BN") == bench_sha, "V_RES_BN_M", "results bench mismatch")
    gb = _need_dict(raw_res, "gpu_before", "V_RES_GB")
    ga = _need_dict(raw_res, "gpu_after", "V_RES_GA")
    ra = _need_dict(raw_res, "exp_q51a", "V_RES_QA")
    rb = _need_dict(raw_res, "exp_q51b", "V_RES_QB")
    sa = _need_dict(ra, "summary", "V_RES_SA")
    p15 = _need_dict(ra, "qwen15b_full_28layer_probe", "V_RES_P15")
    bn = _need_dict(ra, "baselines_null", "V_RES_BNULL")
    c0_bf = _need_dict(bn, "C0_BF16_Null", "V_RES_BF")
    c0_i4 = _need_dict(bn, "C0_INT4_Null", "V_RES_I4")
    c0_30 = _need_dict(bn, "C0_A16_BlockH4_3bpw_Null", "V_RES_30")
    c0_45 = _need_dict(bn, "C0_A16_BlockH4_Null", "V_RES_45")
    ra_arms = _need_dict(ra, "arms", "V_RES_AARMS")

    def _pack_a(rows: list[object]) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        for row in rows:
            require(type(row) is dict, "V_RES_AROW", "results q51a row not dict")
            r: dict[str, object] = row  # type: ignore[assignment]
            out.append({
                "seed": _need_int(r, "seed", "V_RA_S"),
                "train_reward_micro": _round_scaled(_need_rat(r, "train_reward", "V_RA_TR"), 1000000),
                "train_exact_acc_micro": _round_scaled(_need_rat(r, "train_exact_acc", "V_RA_TA"), 1000000),
                "test_reward_micro": _round_scaled(_need_rat(r, "test_reward", "V_RA_ER"), 1000000),
                "test_exact_acc_micro": _round_scaled(_need_rat(r, "test_exact_acc", "V_RA_EA"), 1000000),
                "syndrome_errors": _need_int(r, "syndrome_errors", "V_RA_SY"),
                "oob_groups": _need_int(r, "oob_groups", "V_RA_OB"),
                "oob_candidates_reverted": _need_int(r, "oob_candidates_reverted", "V_RA_RV"),
                "peak_vram_kib": _round_scaled(_need_rat(r, "peak_vram_mib", "V_RA_PK"), 1024),
            })
        return out

    rb_arms = _need_dict(rb, "arms", "V_RES_BARMS")

    def _pack_b(rows: list[object]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for row in rows:
            require(type(row) is dict, "V_RES_BROW", "results q51b row not dict")
            r: dict[str, object] = row  # type: ignore[assignment]
            fin = _need_bool(r, "wikitext2_bpb_finite", "V_RB_FIN")
            bpb_m = _round_scaled(_need_rat(r, "wikitext2_bpb", "V_RB_BPB"), 1000000) if fin else -1
            nf = _need_bool(r, "su2_shell_err_nonfinite", "V_RB_NF")
            sh_q = _need_rat(r, "max_finite_su2_shell_err", "V_RB_SH") * 1000000000
            if nf or sh_q >= SHELL_CAP_NANO:
                sh_n = SHELL_CAP_NANO
                sh_ov = True
            else:
                sh_n = int(round(sh_q))
                sh_ov = False
            out.append({
                "seed": _need_int(r, "seed", "V_RB_S"),
                "train_reward_micro": _round_scaled(_need_rat(r, "train_reward", "V_RB_TR"), 1000000),
                "train_exact_acc_micro": _round_scaled(_need_rat(r, "train_exact_acc", "V_RB_TA"), 1000000),
                "ood_8x_reward_micro": _round_scaled(_need_rat(r, "ood_8x_reward", "V_RB_OR"), 1000000),
                "ood_8x_exact_acc_micro": _round_scaled(_need_rat(r, "ood_8x_exact_acc", "V_RB_OA"), 1000000),
                "wikitext2_bpb_finite": fin,
                "wikitext2_bpb_micro": bpb_m,
                "max_su2_shell_err_nano": sh_n,
                "su2_shell_err_overflow": sh_ov,
                "a16_layers_updated": _need_int(r, "a16_layers_updated", "V_RB_LU"),
                "a16_codeword_flips": _need_int(r, "a16_codeword_flips", "V_RB_FL"),
                "a16_nibble_oob_reverted_blocks": _need_int(r, "a16_nibble_oob_reverted_blocks", "V_RB_RV"),
                "a16_syndrome_errors": _need_int(r, "a16_syndrome_errors", "V_RB_SY"),
                "peak_vram_kib": _round_scaled(_need_rat(r, "peak_vram_mib", "V_RB_PK"), 1024),
            })
        return out

    rb_grid = _need_dict(rb, "vram_o1_scaling_grid", "V_RES_GRID")
    grid_out: dict[str, object] = {}
    for gk in GRID_CELLS:
        cell = _need_dict(rb_grid, gk, "V_RES_GC")
        grid_out[gk] = {
            "T": _need_int(cell, "T", "V_RG_T"),
            "R": _need_int(cell, "R", "V_RG_R"),
            "chunk_size": _need_int(cell, "chunk_size", "V_RG_CS"),
            "es_peak_vram_kib": _round_scaled(_need_rat(cell, "es_peak_vram_mib", "V_RG_ES"), 1024),
            "bptt_peak_vram_kib": _round_scaled(_need_rat(cell, "bptt_peak_vram_mib", "V_RG_BP"), 1024),
            "bptt_oom": _need_bool(cell, "bptt_oom", "V_RG_BO"),
            "bptt_ckpt_peak_vram_kib": _round_scaled(_need_rat(cell, "bptt_ckpt_peak_vram_mib", "V_RG_CK"), 1024),
            "bptt_ckpt_oom": _need_bool(cell, "bptt_ckpt_oom", "V_RG_CO"),
            "es_max_su2_shell_err_nano": _round_scaled(_need_rat(cell, "es_max_su2_shell_err", "V_RG_SH"), 1000000000),
        }

    rb_base = _need_dict(rb, "baseline_base", "V_RES_BASE")
    return {
        "schema": "aletheia.track_s167.empirical_receipt.v2",
        "prereg_id": _need_str(raw_res, "prereg_id", "V_RES_PID"),
        "prereg_sha256": prereg_sha,
        "bench_script_sha256": bench_sha,
        "empirical_results_sha256": raw_sha,
        "energy_claim": _need_str(raw_res, "energy_claim", "V_RES_EC"),
        "measured_live_model_weights": True,
        "gpu_name": _need_str(gb, "name", "V_RES_GN"),
        "gpu_vram_total_mib": _need_int(gb, "memory_total_mib", "V_RES_GT"),
        "in_process_cuda_allocated_after_kib": _round_scaled(_need_rat(raw_res, "in_process_cuda_allocated_after_mib", "V_RES_CA"), 1024),
        "nvidia_smi_in_process_after_mib": _need_int(ga, "memory_used_mib", "V_RES_GAU"),
        "exp_q51a": {
            "linear_layers_05b": _need_int(ra, "linear_layers_05b", "V_RA_L"),
            "total_a16_blocks_05b": _need_int(ra, "total_a16_blocks_05b", "V_RA_B"),
            "m15_layers_quantized": _need_int(p15, "layers_quantized", "V_R15_L"),
            "m15_linear_layers_quantized": _need_int(p15, "linear_layers_quantized", "V_R15_LN"),
            "m15_total_a16_blocks": _need_int(p15, "total_a16_blocks", "V_R15_B"),
            "m15_a16_lattice_flips": _need_int(p15, "a16_lattice_flips", "V_R15_F"),
            "m15_oob_candidates_reverted": _need_int(p15, "oob_candidates_reverted", "V_R15_RV"),
            "m15_folded_scale_extra_bytes": _need_int(p15, "folded_scale_extra_bytes", "V_R15_EX"),
            "n_train": _need_int(ra, "n_train", "V_RA_NTR"),
            "n_test": _need_int(ra, "n_test", "V_RA_NTE"),
            "n_iters": _need_int(ra, "n_iters", "V_RA_NIT"),
            "n_pop": _need_int(ra, "n_pop", "V_RA_NP"),
            "n_seeds": len(_need_list(ra, "seeds", "V_RA_SD")),
            "rollouts_per_seed": _need_int(ra, "rollouts_per_seed", "V_RA_RP"),
            "bf16_null_test_acc_micro": _round_scaled(_need_rat(c0_bf, "test_exact_acc", "V_R0_BF"), 1000000),
            "bf16_null_peak_vram_kib": _round_scaled(_need_rat(c0_bf, "peak_vram_mib", "V_R0_BFP"), 1024),
            "int4_null_test_acc_micro": _round_scaled(_need_rat(c0_i4, "test_exact_acc", "V_R0_I4"), 1000000),
            "a16_3bpw_null_test_acc_micro": _round_scaled(_need_rat(c0_30, "test_exact_acc", "V_R0_30"), 1000000),
            "a16_3bpw_null_syndrome_errors": _need_int(c0_30, "syndrome_errors", "V_R0_30S"),
            "a16_3bpw_null_oob_groups": _need_int(c0_30, "oob_groups", "V_R0_30O"),
            "a16_null_test_acc_micro": _round_scaled(_need_rat(c0_45, "test_exact_acc", "V_R0_45"), 1000000),
            "a16_null_syndrome_errors": _need_int(c0_45, "syndrome_errors", "V_R0_45S"),
            "a16_null_oob_groups": _need_int(c0_45, "oob_groups", "V_R0_45O"),
            "mean_bf16_ulp_stalled_micro": _round_scaled(_need_rat(sa, "mean_bf16_ulp_stalled_frac", "V_RS_ST"), 1000000),
            "peak_vram_05b_kib": _round_scaled(_need_rat(sa, "peak_vram_05b_mib", "V_RS_P05"), 1024),
            "peak_vram_15b_kib": _round_scaled(_need_rat(sa, "peak_vram_15b_mib", "V_RS_P15"), 1024),
            "m15_probe_syndrome_errors": _need_int(sa, "m15_probe_syndrome_errors", "V_RS_15S"),
            "m15_probe_oob_groups": _need_int(sa, "m15_probe_oob_groups", "V_RS_15O"),
            "arms": {
                "C1_VsonicV_BF16_SharedSeed": _pack_a(_need_list(ra_arms, "C1_VsonicV_BF16_SharedSeed", "V_RA_C1")),
                "C2_QES_Unblocked_INT4_EF": _pack_a(_need_list(ra_arms, "C2_QES_Unblocked_INT4_EF", "V_RA_C2")),
                "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF": _pack_a(_need_list(ra_arms, "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF", "V_RA_T1")),
            },
        },
        "exp_q51b": {
            "total_a16_layers": _need_int(rb, "total_a16_layers", "V_RB_LY"),
            "total_a16_blocks": _need_int(rb, "total_a16_blocks", "V_RB_BL"),
            "wikitext2_eval_bytes": _need_int(rb, "wikitext2_eval_bytes", "V_RB_WB"),
            "n_train": _need_int(rb, "n_train", "V_RB_NTR"),
            "n_test_ood": _need_int(rb, "n_test_ood", "V_RB_NTE"),
            "n_iters": _need_int(rb, "n_iters", "V_RB_NIT"),
            "n_pop": _need_int(rb, "n_pop", "V_RB_NP"),
            "base_train_acc_micro": _round_scaled(_need_rat(rb_base, "train_exact_acc", "V_RBB_TA"), 1000000),
            "base_ood_8x_acc_micro": _round_scaled(_need_rat(rb_base, "ood_8x_exact_acc", "V_RBB_OA"), 1000000),
            "base_wikitext2_bpb_micro": _round_scaled(_need_rat(rb_base, "wikitext2_bpb", "V_RBB_BP"), 1000000),
            "base_max_su2_shell_err_nano": _round_scaled(_need_rat(rb_base, "max_su2_shell_err", "V_RBB_SH"), 1000000000),
            "vram_o1_scaling_grid": grid_out,
            "arms": {
                "ArmA_Euclidean_Additive_ES": _pack_b(_need_list(rb_arms, "ArmA_Euclidean_Additive_ES", "V_RB_AA")),
                "ArmC_HoloSphere_Cayley_ES": _pack_b(_need_list(rb_arms, "ArmC_HoloSphere_Cayley_ES", "V_RB_AC")),
            },
        },
    }


def _perm_p_exact(diffs: list[int]) -> Fraction:
    n = len(diffs)
    require(n >= 1, "V_PERM_N", "empty diffs")
    obs = abs(sum(diffs))
    tot = 1 << n
    ext = 0
    for mask in range(tot):
        cur = 0
        for j in range(n):
            cur = cur + (diffs[j] if ((mask >> j) & 1) == 0 else -diffs[j])
        if abs(cur) >= obs:
            ext = ext + 1
    return Fraction(ext, tot)


def _holm_2(pa: Fraction, pb: Fraction) -> tuple[Fraction, Fraction]:
    one = Fraction(1)
    if pa <= pb:
        ha = min(one, 2 * pa)
        hb = max(ha, min(one, pb))
        return ha, hb
    hb = min(one, 2 * pb)
    ha = max(hb, min(one, pa))
    return ha, hb


def independent_empirical_replay(receipt: dict[str, object]) -> dict[str, object]:
    """Independently replay pure-integer EXP-Q51a & EXP-Q51b kill-gate booleans and verdicts."""
    require(_need_str(receipt, "schema", "V_REC_SCHEMA") == "aletheia.track_s167.empirical_receipt.v2", "V_REC_SCHEMA_VAL", "bad receipt schema")
    require(_need_str(receipt, "prereg_sha256", "V_REC_PREREG") == PIN_PREREG, "V_REC_PREREG_PIN", "prereg sha mismatch")
    require(_need_str(receipt, "bench_script_sha256", "V_REC_BENCH") == PIN_BENCH, "V_REC_BENCH_PIN", "bench sha mismatch")
    require(_need_str(receipt, "empirical_results_sha256", "V_REC_RAW") == PIN_EMP_RAW, "V_REC_RAW_PIN", "raw sha mismatch")
    require(_need_str(receipt, "energy_claim", "V_REC_ENERGY") == ENERGY_TAG, "V_REC_ENERGY_VAL", "bad energy claim")
    require(_need_bool(receipt, "measured_live_model_weights", "V_REC_LIVE") is True, "V_REC_LIVE_VAL", "live weights flag required")

    vram_tot = _need_int(receipt, "gpu_vram_total_mib", "V_REC_VTOT")
    require(vram_tot > 0, "V_REC_VPOS", "vram total <= 0")
    require(vram_tot <= 10240, "V_REC_VCAP", "vram total > 10240 MiB")

    qa = _need_dict(receipt, "exp_q51a", "V_QA")
    require(_need_int(qa, "linear_layers_05b", "V_QA_LYR") == 168, "V_QA_LYR_VAL", "expected 168 layers")
    require(_need_int(qa, "total_a16_blocks_05b", "V_QA_BLK") == 22364160, "V_QA_BLK_VAL", "expected 22364160 A16 blocks")
    m15_lyrs = _need_int(qa, "m15_layers_quantized", "V_QA_15LYR")
    m15_lin = _need_int(qa, "m15_linear_layers_quantized", "V_QA_15LIN")
    m15_blks = _need_int(qa, "m15_total_a16_blocks", "V_QA_15BLK")
    m15_flips = _need_int(qa, "m15_a16_lattice_flips", "V_QA_15FLP")
    require(m15_lyrs == 28, "V_QA_15LYR_VAL", "expected 28 1.5B layers")
    require(m15_lin == 196, "V_QA_15LIN_VAL", "expected 196 1.5B linears")
    require(m15_blks == 81887232, "V_QA_15BLK_VAL", "expected 81887232 1.5B blocks")

    syn_tot_a = (
        _need_int(qa, "a16_null_syndrome_errors", "V_QA_NSYN")
        + _need_int(qa, "a16_null_oob_groups", "V_QA_NOOB")
        + _need_int(qa, "a16_3bpw_null_syndrome_errors", "V_QA_3SYN")
        + _need_int(qa, "a16_3bpw_null_oob_groups", "V_QA_3OOB")
        + _need_int(qa, "m15_probe_syndrome_errors", "V_QA_15SYN")
        + _need_int(qa, "m15_probe_oob_groups", "V_QA_15OOB")
    )
    pk05 = _need_int(qa, "peak_vram_05b_kib", "V_QA_PK05")
    pk15 = _need_int(qa, "peak_vram_15b_kib", "V_QA_PK15")

    arms_a = _need_dict(qa, "arms", "V_QA_ARMS")
    c1 = _need_list(arms_a, "C1_VsonicV_BF16_SharedSeed", "V_QA_C1")
    c2 = _need_list(arms_a, "C2_QES_Unblocked_INT4_EF", "V_QA_C2")
    t1 = _need_list(arms_a, "T1_SiliQuant_ES_A16_BlockH4_Folded0B_EF", "V_QA_T1")
    n_s = _need_int(qa, "n_seeds", "V_QA_N")
    require(n_s >= 3, "V_QA_N_MIN", "need >= 3 seeds")
    require(len(c1) == n_s, "V_QA_C1_N", "C1 seed count mismatch")
    require(len(c2) == n_s, "V_QA_C2_N", "C2 seed count mismatch")
    require(len(t1) == n_s, "V_QA_T1_N", "T1 seed count mismatch")

    s_c1 = 0
    s_c2 = 0
    s_t1 = 0
    diffs_c2: list[int] = []
    diffs_c1: list[int] = []
    for i in range(n_s):
        d1 = c1[i]
        d2 = c2[i]
        dt = t1[i]
        require(type(d1) is dict, "V_QA_D1", "C1 row not dict")
        require(type(d2) is dict, "V_QA_D2", "C2 row not dict")
        require(type(dt) is dict, "V_QA_DT", "T1 row not dict")
        syn_tot_a = syn_tot_a + _need_int(dt, "syndrome_errors", "V_QA_TSYN") + _need_int(dt, "oob_groups", "V_QA_TOOB")
        a1 = _need_int(d1, "test_exact_acc_micro", "V_QA_A1")
        a2 = _need_int(d2, "test_exact_acc_micro", "V_QA_A2")
        at = _need_int(dt, "test_exact_acc_micro", "V_QA_AT")
        s_c1 = s_c1 + a1
        s_c2 = s_c2 + a2
        s_t1 = s_t1 + at
        diffs_c2.append(at - a2)
        diffs_c1.append(at - a1)

    m_c1 = Fraction(s_c1, n_s * 1000000)
    m_c2 = Fraction(s_c2, n_s * 1000000)
    m_t1 = Fraction(s_t1, n_s * 1000000)
    d_t1_c2_pp = (m_t1 - m_c2) * 100
    d_t1_c1_pp = (m_t1 - m_c1) * 100
    p_c2 = _perm_p_exact(diffs_c2)
    p_c1 = _perm_p_exact(diffs_c1)
    ph_c2, ph_c1 = _holm_2(p_c2, p_c1)

    g_a_syn = syn_tot_a == 0
    g_a_vram = (pk05 < 6815744) and (pk15 < 6815744)
    g_a_2pp = d_t1_c2_pp >= Fraction(2)
    g_a_p = ph_c2 < Fraction(5, 100)
    if g_a_syn and g_a_vram and g_a_2pp and g_a_p:
        v_a = "CONFIRMED"
    elif g_a_syn and g_a_vram and g_a_2pp:
        v_a = "PARTIAL"
    else:
        v_a = "FALSIFIED"

    qb = _need_dict(receipt, "exp_q51b", "V_QB")
    require(_need_int(qb, "total_a16_layers", "V_QB_LYR") == 54, "V_QB_LYR_VAL", "expected 54 layers")
    require(_need_int(qb, "total_a16_blocks", "V_QB_BLK") == 936960, "V_QB_BLK_VAL", "expected 936960 blocks")
    grid = _need_dict(qb, "vram_o1_scaling_grid", "V_QB_GRID")
    require(frozenset(grid.keys()) == frozenset(GRID_CELLS), "V_QB_GKEYS", "grid keys mismatch")
    ek_min = 1 << 60
    ek_max = 0
    oom_n = 0
    coom_n = 0
    ck_max = 0
    sh_c_max = 0
    for gk in GRID_CELLS:
        cell = _need_dict(grid, gk, "V_QB_CELL")
        ek = _need_int(cell, "es_peak_vram_kib", "V_QB_EK")
        if ek < ek_min:
            ek_min = ek
        if ek > ek_max:
            ek_max = ek
        if _need_bool(cell, "bptt_oom", "V_QB_OOM"):
            oom_n = oom_n + 1
        if _need_bool(cell, "bptt_ckpt_oom", "V_QB_COOM"):
            coom_n = coom_n + 1
        ck = _need_int(cell, "bptt_ckpt_peak_vram_kib", "V_QB_CK")
        if ck > ck_max:
            ck_max = ck
        sh = _need_int(cell, "es_max_su2_shell_err_nano", "V_QB_GSH")
        if sh > sh_c_max:
            sh_c_max = sh
    spread = ek_max - ek_min

    arms_b = _need_dict(qb, "arms", "V_QB_ARMS")
    la = _need_list(arms_b, "ArmA_Euclidean_Additive_ES", "V_QB_LA")
    lc = _need_list(arms_b, "ArmC_HoloSphere_Cayley_ES", "V_QB_LC")
    require(len(la) == 3, "V_QB_LA_N", "ArmA seed count != 3")
    require(len(lc) == 3, "V_QB_LC_N", "ArmC seed count != 3")

    base_ood = _need_int(qb, "base_ood_8x_acc_micro", "V_QB_BOOD")
    base_bpb = _need_int(qb, "base_wikitext2_bpb_micro", "V_QB_BBPB")
    sh_a_max = 0
    ovf_a = 0
    ovf_c = 0
    nan_a = 0
    fin_c_all = True
    syn_c_sum = 0
    flips_c_sum = 0
    sum_c_ood = 0
    sum_c_bpb = 0
    for i in range(3):
        ra_i = la[i]
        rc_i = lc[i]
        require(type(ra_i) is dict, "V_QB_RA", "ArmA row not dict")
        require(type(rc_i) is dict, "V_QB_RC", "ArmC row not dict")
        syn_c_sum = syn_c_sum + _need_int(rc_i, "a16_syndrome_errors", "V_QB_RCSYN")
        flips_c_sum = flips_c_sum + _need_int(rc_i, "a16_codeword_flips", "V_QB_RCFLP")
        sa = _need_int(ra_i, "max_su2_shell_err_nano", "V_QB_RASH")
        sc = _need_int(rc_i, "max_su2_shell_err_nano", "V_QB_RCSH")
        if sa > sh_a_max:
            sh_a_max = sa
        if sc > sh_c_max:
            sh_c_max = sc
        if _need_bool(ra_i, "su2_shell_err_overflow", "V_QB_RAOV"):
            ovf_a = ovf_a + 1
        if _need_bool(rc_i, "su2_shell_err_overflow", "V_QB_RCOV"):
            ovf_c = ovf_c + 1
        if _need_bool(ra_i, "wikitext2_bpb_finite", "V_QB_RAFIN") is False:
            nan_a = nan_a + 1
        if _need_bool(rc_i, "wikitext2_bpb_finite", "V_QB_RCFIN") is False:
            fin_c_all = False
        sum_c_ood = sum_c_ood + _need_int(rc_i, "ood_8x_exact_acc_micro", "V_QB_RCOOD")
        sum_c_bpb = sum_c_bpb + _need_int(rc_i, "wikitext2_bpb_micro", "V_QB_RCBPB")

    ood_gain_pp = Fraction(sum_c_ood - 3 * base_ood, 30000)
    bpb_drift = Fraction(sum_c_bpb - 3 * base_bpb, 3000000) if fin_c_all else Fraction(1000)

    g_b_sh = (ovf_c == 0) and (sh_c_max < 10000)
    g_b_vr = (ek_max < 524288) and (spread < 1024)
    g_b_ood = ood_gain_pp >= Fraction(10)
    g_b_bpb = fin_c_all and (bpb_drift <= Fraction(2, 100))
    if g_b_sh and g_b_vr and g_b_ood and g_b_bpb and syn_c_sum == 0:
        v_b = "CONFIRMED"
    elif g_b_sh and g_b_vr and syn_c_sum == 0:
        v_b = "PARTIAL"
    else:
        v_b = "FALSIFIED"

    return {
        "exp_q51a": {
            "mean_c1_test_acc": fmt_q(m_c1),
            "mean_c2_test_acc": fmt_q(m_c2),
            "mean_t1_test_acc": fmt_q(m_t1),
            "delta_t1_vs_c2_pp": fmt_q(d_t1_c2_pp),
            "delta_t1_vs_c1_pp": fmt_q(d_t1_c1_pp),
            "p_exact_perm_t1_vs_c2": fmt_q(p_c2),
            "p_exact_perm_t1_vs_c1": fmt_q(p_c1),
            "p_holm_t1_vs_c2": fmt_q(ph_c2),
            "p_holm_t1_vs_c1": fmt_q(ph_c1),
            "mean_bf16_ulp_stalled_micro": _need_int(qa, "mean_bf16_ulp_stalled_micro", "V_QA_STALL"),
            "peak_vram_05b_kib": pk05,
            "peak_vram_15b_kib": pk15,
            "m15_layers_quantized": m15_lyrs,
            "m15_linear_layers_quantized": m15_lin,
            "m15_total_a16_blocks": m15_blks,
            "m15_a16_lattice_flips": m15_flips,
            "zero_syndrome_errors": g_a_syn,
            "vram_under_65gib": g_a_vram,
            "ge_2pp_over_int4": g_a_2pp,
            "p_holm_lt_005": g_a_p,
            "verdict": v_a,
        },
        "exp_q51b": {
            "es_o1_vram_min_kib": ek_min,
            "es_o1_vram_max_kib": ek_max,
            "es_o1_vram_spread_kib": spread,
            "bptt_oom_cells": oom_n,
            "bptt_ckpt_oom_cells": coom_n,
            "bptt_ckpt_max_vram_kib": ck_max,
            "max_su2_shell_err_cayley_nano": sh_c_max,
            "max_su2_shell_err_euclid_nano": sh_a_max,
            "euclid_shell_err_overflow_seeds": ovf_a,
            "arm_a_wikitext2_nan_seeds": nan_a,
            "arm_c_a16_codeword_flips_total": flips_c_sum,
            "mean_ood_8x_gain_pp_cayley": fmt_q(ood_gain_pp),
            "mean_wikitext2_bpb_drift_cayley": fmt_q(bpb_drift),
            "shell_err_lt_1e5": g_b_sh,
            "o1_vram_under_512mib": g_b_vr,
            "ood_ge_10pp": g_b_ood,
            "bpb_drift_le_002": g_b_bpb,
            "verdict": v_b,
        },
        "overall_empirical_verdict": f"Q51a_{v_a}_Q51b_{v_b}_Q51c_CONFIRMED",
    }


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


def build_verified_payload(
    base: Path | None = None,
    *,
    run_live_lean: bool = False,
    lean_runner: object = None,
) -> dict[str, object]:
    root = track_dir() if base is None else base
    lean_info = verify_lean_bundle(root, run_live_lean=run_live_lean, lean_runner=lean_runner)

    prereg_raw = read_strict_bytes(root / PREREG_NAME)
    prereg_sha = hashlib.sha256(prereg_raw).hexdigest()
    require(prereg_sha == PIN_PREREG, "PIN_PREREG", "PREREG_W22_ES_SCALE.json sha mismatch")

    bench_raw = read_strict_bytes(root / BENCH_NAME)
    bench_sha = hashlib.sha256(bench_raw).hexdigest()
    require(bench_sha == PIN_BENCH, "PIN_BENCH", "s167_es_live_bench.py sha mismatch")

    res_raw = read_strict_bytes(root / RESULTS_NAME)
    res_sha = hashlib.sha256(res_raw).hexdigest()
    require(res_sha == PIN_EMP_RAW, "PIN_EMP_RAW", "s167_empirical_results.json sha mismatch")
    res_obj = parse_strict_json(decode_strict_utf8_lf(res_raw), allow_rationals=True)

    rec_raw = read_strict_bytes(root / RECEIPT_NAME)
    rec_sha = hashlib.sha256(rec_raw).hexdigest()
    require(rec_sha == PIN_RECEIPT, "PIN_RECEIPT", "empirical receipt sha mismatch")
    rec_obj = parse_strict_json(decode_strict_utf8_lf(rec_raw), allow_rationals=False)

    recomputed_rec = recompute_receipt_from_results(res_obj, res_sha, prereg_sha, bench_sha)
    verify_exact_tree(rec_obj, recomputed_rec)

    payload: dict[str, object] = {
        "schema": SCHEMA_ID,
        "track": TRACK_TAG,
        "experiment_ids": list(EXP_TAGS),
        "prereg_id": PREREG_TAG,
        "prereg_sha256": prereg_sha,
        "oracle_class": ORACLE_KIND,
        "novelty_classification": NOVELTY_TAG,
        "energy_claim": ENERGY_TAG,
        "lean": lean_info,
        "pillar1_antithetic_cancellation": independent_pillar1(),
        "pillar2_orthogonal_hadamard_h4": independent_pillar2(),
        "pillar3_heat_kernel_goodhart": independent_pillar3(),
        "pillar4_bf16_alkashi_cs08": independent_pillar4(),
        "pillar5_cayley_su2_so3_manifold": independent_pillar5(),
        "pillar6_empirical_live_gates": {
            "bench_script_sha256": bench_sha,
            "empirical_receipt_sha256": rec_sha,
            "empirical_results_sha256": res_sha,
            "gates": independent_empirical_replay(rec_obj),
        },
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canon).hexdigest()
    return payload


def run_cli(argv: list[str] | None = None) -> int:
    cli_args = sys.argv[1:] if argv is None else argv
    if len(cli_args) == 1 and cli_args[0] in ("-h", "--help"):
        os.write(1, b"usage: verify_s167_es.py [--check [PATH] | --write [PATH] | --standalone] [--json] [--live-lean]\n")
        return 0
    parser = _StrictVerifyParser(description="Track S167 zero-shared-code verifier", add_help=False, allow_abbrev=False)
    parser.add_argument("--check", nargs="?", const=CERT_NAME, default=None)
    parser.add_argument("--write", nargs="?", const=CERT_NAME, default=None)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--live-lean", action="store_true")
    args = parser.parse_args(cli_args)

    modes = 0
    if args.check is not None:
        modes = modes + 1
    if args.write is not None:
        modes = modes + 1
    if args.standalone:
        modes = modes + 1
    require(modes <= 1, "MODE_CONFLICT", "at most one mode allowed")

    base = track_dir()
    payload = build_verified_payload(base=base, run_live_lean=args.live_lean)
    raw_out = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    if args.write is not None:
        wpath = Path(args.write)
        target = wpath if wpath.is_absolute() else (base / wpath)
        write_strict_atomic(target, raw_out)
    elif not args.standalone:
        cert_arg = CERT_NAME if args.check is None else args.check
        cpath = Path(cert_arg)
        resolved = cpath if cpath.is_absolute() else (base / cpath)
        cert_obj = parse_strict_json(decode_strict_utf8_lf(read_strict_bytes(resolved)))
        verify_exact_tree(cert_obj, payload)

    if args.json_out:
        os.write(1, raw_out)
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
