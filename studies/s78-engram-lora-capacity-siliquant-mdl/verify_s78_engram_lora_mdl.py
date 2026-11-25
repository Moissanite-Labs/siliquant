#!/usr/bin/env python3
"""
Track S78: Hardened Zone-B Verifier, Lean 4.36.0 Live Kernel Auditor, & CLI (R2).

Verifies:
  1. Pure-Z/Q Track S78 certificate envelope (atomic dir_fd + O_NONBLOCK + O_NOFOLLOW
     + st_nlink==1 + $PWD symlink defense; strict JSON loader with pre-parse ASCII int
     digit cap, zero floats, NaN/Inf, duplicate keys, CRLF, or non-canonical bytes;
     FR1 full oracle replay, FR0 independent secondary 4-pillar audit, and raising
     direct `secondary_audit_payload`).
  2. Live Lean 4.36.0 kernel execution over `--stdin` (zero TOCTOU between read,
     hash, static audit, and kernel check), R10+ comment-only static scanner with
     strict quote/guillemet ban (immune to quote-parity inversion), and pinned
     SHA-256 verification of `LoraCapacityJointMdl.lean` (17/17 zero-axiom `[]`).
  3. Optional atomic dir_fd on-disk SHA-256 lock of pinned upstream sources when
     `--expedition-root <dir>` is explicitly supplied (zero hardcoded host-user paths).
  4. Durable atomic `0o600` dir_fd certificate emission via `--write <path>`.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from s78_engram_lora_siliquant_mdl import (
    PINNED_SOURCES,
    SCHEMA_VERSION,
    TRACK_ID,
    build_certificate_envelope,
    canonical_json_bytes,
)

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 16
MAX_INT_BITS = 256
MAX_INT_DIGITS = 80
EXPECTED_LEAN_SHA256 = "350eea0cd07c47c6b492e0ec289a8b756a1802e68beee7634e0ea69854c34c89"

REQUIRED_TOP_KEYS = frozenset(
    {"schema_version", "track_id", "payload_sha256", "certificate_sha256", "payload"}
)
REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "all_pillars_verified",
        "constants_enclosures",
        "epistemic_quarantine",
        "novelty_tier",
        "oracle_class",
        "pillar_1_engram_audit",
        "pillar_2_gauge_vs_mp_reconciliation",
        "pillar_3_s34_mdl_and_cs15_cs16",
        "pillar_4_joint_bpw_rank_allocation_law",
        "track_id",
    }
)
EXPECTED_FROZEN_MODELS = frozenset(
    {
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "google/gemma-3-4b-it",
    }
)
EXPECTED_SWEEP_RANKS: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256, 512)
HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_ASCII_INT_RE = re.compile(r"\A-?[0-9]+\Z")

EXPECTED_LEAN_THEOREMS: tuple[str, ...] = (
    "TrackS78LoraJointMdl.gauge_quotient_eff_dof",
    "TrackS78LoraJointMdl.full_rank_wilks_closure",
    "TrackS78LoraJointMdl.eta_alg_strictly_decreasing",
    "TrackS78LoraJointMdl.spectral_edge_dominates_algebraic_dof",
    "TrackS78LoraJointMdl.spectral_edge_multiplier_le_two",
    "TrackS78LoraJointMdl.spectral_edge_saturates_for_square",
    "TrackS78LoraJointMdl.prefix_sum_ge_len_mul_min",
    "TrackS78LoraJointMdl.ordered_spectrum_avg_nonincreasing",
    "TrackS78LoraJointMdl.ordered_spectrum_avg_strict_drop",
    "TrackS78LoraJointMdl.s34_mdl_offline_online_cancellation",
    "TrackS78LoraJointMdl.engram_noise_no_mdl_profit",
    "TrackS78LoraJointMdl.conservation_forgetting_cancels_train_gain",
    "TrackS78LoraJointMdl.cs15_cs16_capacity_obstruction_witness",
    "TrackS78LoraJointMdl.joint_budget_conservation",
    "TrackS78LoraJointMdl.quant_to_rank_dominance_iff",
    "TrackS78LoraJointMdl.convex_kl_floor_obstruction",
    "TrackS78LoraJointMdl.pinned_siliquant_receipt_witnesses",
)

_LEAN_QUOTE_OR_GUILLEMET_RE = re.compile(r'["\'«»]')
_LEAN_FORBIDDEN_RE = re.compile(
    r"\b(?:sorry|admit|axiom|unsafe|native_decide|sorryAx|"
    r"ofReduceBool|ofReduceNat|trustCompiler|mkSorry|"
    r"elab|elab_rules|macro|macro_rules|syntax|declare_syntax_cat|"
    r"notation|prefix|infix|infixl|infixr|postfix|"
    r"initialize|builtin_initialize|opaque|dbg_trace|"
    r"run_cmd|run_meta|interpolatedStr)\b"
)
_LEAN_PARTIAL_RE = re.compile(r"(?<!\.)\bpartial\s+(?:def|instance|abbrev|theorem)\b")
_LEAN_DECIDE_NATIVE_RE = re.compile(
    r"(?<!\w)\+\s*native\b|\bnative\s*:=(?!\s*false\b)|"
    r"\bDecide(?:Config|\.Config)\b|\bdecide\s*\(\s*(?:config|revert)\b"
)
_LEAN_SKIP_TC_RE = re.compile(
    r"\bdebug\s*\.\s*(?:skipKernelTC|byAsSorry|proofAsSorry|terminalTacticsAsSorry)\b"
)
_LEAN_EVAL_SPOOF_RE = re.compile(
    r"#(?:eval!?|reduce|exit)\b|"
    r"#print\s+(?!axioms\s+(?!example\b|def\b|theorem\b|lemma\b)[A-Za-z0-9_.]+\s*(?:\n|\Z))"
)
_LEAN_ATTR_ESCAPE_RE = re.compile(
    r"(?:@\[|\battribute\s*\[)[^\]]*\b(?:extern|implemented_by|export|csimp)\b"
)
_LEAN_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:public|private|meta)\s+)*import\s+([A-Za-z0-9_.]+)"
)
_ZERO_AXIOM_LINE_RE = re.compile(
    r"\A'([A-Za-z0-9_.]+)' does not depend on any axioms\Z"
)


def require_ok(cond: bool, msg: str) -> None:
    """Hardened non-assert verification guard (immune to python3 -O)."""
    if not cond:
        raise ValueError(msg)


def _is_sha256_hex(val: Any) -> bool:
    return type(val) is str and bool(HEX64_RE.match(val))


def _req_dict(val: Any, code: str) -> dict[str, Any]:
    require_ok(type(val) is dict, code)
    return val


def _req_int(val: Any, code: str) -> int:
    require_ok(type(val) is int, code)
    return val


def _check_json_depth_and_types(root: Any) -> None:
    """Iterative exact-type JSON tree validator (blocks subclasses, huge ints, lone surrogates, deep DoS)."""
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        obj, depth = stack.pop()
        require_ok(depth <= MAX_JSON_DEPTH, "JSON_MAX_DEPTH_EXCEEDED")
        t = type(obj)
        if obj is None or t is bool:
            continue
        if t is int:
            require_ok(obj.bit_length() <= MAX_INT_BITS, "JSON_INT_TOO_LARGE")
            continue
        if t is str:
            require_ok("\x00" not in obj, "JSON_NUL_IN_STRING")
            for ch in obj:
                code = ord(ch)
                require_ok(not (0xD800 <= code <= 0xDFFF), "JSON_LONE_SURROGATE")
            continue
        if t is list:
            for item in obj:
                stack.append((item, depth + 1))
            continue
        if t is dict:
            for k, v in obj.items():
                require_ok(type(k) is str, "JSON_NON_STRING_KEY")
                stack.append((k, depth + 1))
                stack.append((v, depth + 1))
            continue
        raise ValueError(f"JSON_DISALLOWED_TYPE:{t.__name__}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"JSON_DUPLICATE_KEY:{k}")
        out[k] = v
    return out


def _parse_strict_int(token: str) -> int:
    """Pre-parse ASCII digit cap blocking Python's raw 4,300-digit ValueError crash."""
    require_ok(bool(_ASCII_INT_RE.match(token)), "JSON_INVALID_INT_TOKEN")
    digits = token[1:] if token.startswith("-") else token
    require_ok(len(digits) <= MAX_INT_DIGITS, "JSON_INT_TOO_LARGE")
    return int(token)


def _reject_float(token: str) -> Any:
    raise ValueError(f"JSON_FLOAT_FORBIDDEN:{token}")


def _reject_constant(token: str) -> Any:
    raise ValueError(f"JSON_CONSTANT_FORBIDDEN:{token}")


def _validate_path_str(raw: str) -> Path:
    require_ok(type(raw) is str and len(raw) > 0, "EMPTY_PATH")
    require_ok("\x00" not in raw, "NUL_IN_PATH")
    return Path(raw)


def _is_symlink_at(dfd: int, name: str) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(name, dir_fd=dfd).st_mode)
    except OSError:
        return False


def _check_pwd_no_symlinks() -> None:
    """Fail closed if logical $PWD contains a symlink ancestor (blocks cd -L sym_d && --check c.json)."""
    pwd = os.environ.get("PWD")
    if pwd is None:
        return
    if not os.path.isabs(pwd):
        return
    cur_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in Path(pwd).parts[1:]:
            try:
                nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except OSError as exc:
                if _is_symlink_at(cur_fd, comp):
                    raise ValueError("SYMLINK_IN_PATH") from exc
                return
            os.close(cur_fd)
            cur_fd = nxt
    finally:
        os.close(cur_fd)


def _open_parent_dir_fd(path: Path) -> tuple[int, str]:
    """Step down ancestor directories atomically via O_RDONLY|O_DIRECTORY|O_NOFOLLOW."""
    p_str = str(path)
    require_ok("\x00" not in p_str, "INVALID_PATH")
    is_abs = path.is_absolute()
    if not is_abs:
        _check_pwd_no_symlinks()
    parts = path.parts[1:] if is_abs else path.parts
    require_ok(len(parts) > 0, "INVALID_PATH")
    ancestors, leaf = parts[:-1], parts[-1]
    require_ok(leaf != "..", "INVALID_PATH")
    cur_fd = os.open("/" if is_abs else ".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for comp in ancestors:
            try:
                nxt = os.open(comp, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur_fd)
            except OSError as exc:
                if _is_symlink_at(cur_fd, comp):
                    raise ValueError("SYMLINK_IN_PATH") from exc
                raise ValueError(f"INVALID_PARENT_DIR:{exc.errno}") from exc
            os.close(cur_fd)
            cur_fd = nxt
        return cur_fd, leaf
    except BaseException:
        os.close(cur_fd)
        raise


def _read_regular_file_bytes(path: Path, max_bytes: int) -> bytes:
    """Atomically read a regular, single-link, non-symlink, non-FIFO file via pinned dir_fd."""
    dfd, leaf = _open_parent_dir_fd(path)
    try:
        try:
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("SYMLINK_IN_PATH") from exc
            raise ValueError(f"OPEN_FAILED:{exc.errno}") from exc
        try:
            st = os.fstat(fd)
            require_ok(stat.S_ISREG(st.st_mode), "NOT_REGULAR_FILE")
            require_ok(st.st_nlink == 1, "HARDLINK_FORBIDDEN")
            require_ok(st.st_size <= max_bytes, "FILE_TOO_LARGE")
            chunks: list[bytes] = []
            total = 0
            while True:
                buf = os.read(fd, 65536)
                if not buf:
                    break
                total += len(buf)
                require_ok(total <= max_bytes, "FILE_TOO_LARGE")
                chunks.append(buf)
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)


def load_strict_json_file(path: Path) -> dict[str, Any]:
    """Read a regular single-link non-symlink canonical UTF-8 JSON object with zero floats/dups/CRLF."""
    raw = _read_regular_file_bytes(path, MAX_JSON_BYTES)
    require_ok(b"\x00" not in raw, "NUL_BYTE_IN_JSON")
    require_ok(b"\r" not in raw, "CRLF_FORBIDDEN")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("INVALID_UTF8") from exc
    try:
        data = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_strict_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON_SYNTAX_ERROR:{exc.msg}") from exc
    except RecursionError as exc:
        raise ValueError("JSON_MAX_DEPTH_EXCEEDED") from exc
    require_ok(type(data) is dict, "JSON_ROOT_NOT_OBJECT")
    _check_json_depth_and_types(data)
    require_ok(raw == canonical_json_bytes(data), "NON_CANONICAL_JSON_BYTES")
    return data


def _require_writable_target_safe(dfd: int, leaf: str) -> None:
    try:
        st_l = os.lstat(leaf, dir_fd=dfd)
    except FileNotFoundError:
        return
    require_ok(not stat.S_ISLNK(st_l.st_mode), "SYMLINK_IN_PATH")
    require_ok(stat.S_ISREG(st_l.st_mode), "TARGET_NOT_REGULAR")
    require_ok(st_l.st_nlink == 1, "HARDLINK_FORBIDDEN")


def atomic_write_0600(target: Path, content: bytes) -> None:
    """Atomically write bytes to target with strict 0o600 permissions via pinned dir_fd."""
    dfd, leaf = _open_parent_dir_fd(target)
    try:
        _require_writable_target_safe(dfd, leaf)
        tmp_name = f".{leaf}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(tmp_name, flags, 0o600, dir_fd=dfd)
        written_ok = False
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(content)
            offset = 0
            total = len(content)
            while offset < total:
                n = os.write(fd, view[offset:])
                if n <= 0:
                    raise OSError("SHORT_WRITE")
                offset += n
            os.fsync(fd)
            require_ok(os.fstat(fd).st_nlink == 1, "TMP_HARDLINK_DETECTED")
            written_ok = True
        finally:
            os.close(fd)
            if not written_ok:
                try:
                    os.unlink(tmp_name, dir_fd=dfd)
                except OSError:
                    pass
        try:
            _require_writable_target_safe(dfd, leaf)
            os.replace(tmp_name, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
        except BaseException:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
            raise
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _strip_lean_comments_only(source: str) -> tuple[str, list[str]]:
    """Strip ONLY Lean 4 line comments (`--`) and nested block comments (`/- ... -/`).

    Quotes (`"`, `'`) and guillemets (`«`, `»`) are NEVER stripped; they are strictly
    banned post-strip because `LoraCapacityJointMdl.lean` contains zero quotes or
    guillemets, making quote-parity inversion and comment-delimiter smuggling impossible.
    """
    out: list[str] = []
    errors: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        if source.startswith("--", i):
            nl = source.find("\n", i + 2)
            if nl == -1:
                break
            out.append("\n")
            i = nl + 1
            continue
        if source.startswith("/-", i):
            depth = 1
            i += 2
            while i < n:
                if source.startswith("/-", i):
                    depth += 1
                    i += 2
                elif source.startswith("-/", i):
                    depth -= 1
                    i += 2
                    if depth == 0:
                        break
                else:
                    if source[i] == "\n":
                        out.append("\n")
                    i += 1
            if depth != 0:
                errors.append("LEAN_UNTERMINATED_BLOCK_COMMENT")
            out.append(" ")
            continue
        out.append(source[i])
        i += 1
    return "".join(out), errors


def audit_lean_source_static(source: str) -> dict[str, Any]:
    """Static R10+ audit of Lean 4 source code (comment-only strip + strict quote/guillemet ban)."""
    cleaned, findings = _strip_lean_comments_only(source)
    if _LEAN_QUOTE_OR_GUILLEMET_RE.search(cleaned):
        findings.append("LEAN_QUOTE_OR_GUILLEMET_FORBIDDEN")
    deguillemet = cleaned.replace("«", "").replace("»", "")
    if _LEAN_FORBIDDEN_RE.search(deguillemet):
        findings.append("LEAN_FORBIDDEN_KEYWORD")
    if _LEAN_PARTIAL_RE.search(deguillemet):
        findings.append("LEAN_PARTIAL_DECL")
    if _LEAN_DECIDE_NATIVE_RE.search(deguillemet):
        findings.append("LEAN_DECIDE_NATIVE")
    if _LEAN_SKIP_TC_RE.search(deguillemet):
        findings.append("LEAN_SKIP_KERNEL_TC")
    if _LEAN_EVAL_SPOOF_RE.search(deguillemet):
        findings.append("LEAN_EVAL_OR_PRINT_SPOOF")
    if _LEAN_ATTR_ESCAPE_RE.search(deguillemet):
        findings.append("LEAN_ATTR_TRUST_ESCAPE")
    imports = [m.group(1) for m in _LEAN_IMPORT_RE.finditer(deguillemet)]
    if imports:
        findings.append(f"LEAN_NON_ZERO_IMPORTS:{','.join(imports)}")
    return {"clean": len(findings) == 0, "findings": findings, "imports": imports}


def verify_lean_proof(lean_path: Path, *, enforce_sha256: bool = True) -> dict[str, Any]:
    """Run atomic dir_fd read, static R10+ scan, SHA-256 pin, and live Lean 4.36.0 `--stdin` check."""
    require_ok(type(enforce_sha256) is bool, "INVALID_ENFORCE_SHA256_FLAG")
    raw = _read_regular_file_bytes(lean_path, MAX_JSON_BYTES)
    lean_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("LEAN_INVALID_UTF8") from exc
    static = audit_lean_source_static(source)
    require_ok(static["clean"], f"LEAN_STATIC_AUDIT_FAILED:{','.join(static['findings'])}")
    if enforce_sha256:
        require_ok(hmac.compare_digest(lean_sha256, EXPECTED_LEAN_SHA256), "LEAN_SHA256_MISMATCH")

    lean_bin = shutil.which("lean")
    require_ok(lean_bin is not None, "LEAN_BINARY_NOT_FOUND")
    proc = subprocess.run(
        [str(lean_bin), "--stdin"],
        input=source,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    require_ok(proc.returncode == 0, f"LEAN_COMPILE_FAILED:{proc.returncode}")
    require_ok(proc.stderr.strip() == "", "LEAN_NONEMPTY_STDERR")

    # Rejoin multi-line wrapped Lean 4.36 #print axioms lines (indented continuations)
    logical_lines: list[str] = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        if raw_line[0] in (" ", "\t") and logical_lines:
            logical_lines[-1] = logical_lines[-1] + " " + raw_line.strip()
        else:
            logical_lines.append(raw_line.strip())

    seen: list[str] = []
    for line in logical_lines:
        m = _ZERO_AXIOM_LINE_RE.match(line)
        require_ok(m is not None, f"LEAN_UNEXPECTED_STDOUT_LINE:{line[:80]}")
        seen.append(m.group(1))  # type: ignore[union-attr]

    require_ok(tuple(seen) == EXPECTED_LEAN_THEOREMS, "LEAN_THEOREM_LIST_MISMATCH")
    return {
        "lean_file": lean_path.name,
        "lean_sha256": lean_sha256,
        "sha256_pinned": enforce_sha256,
        "static_clean": True,
        "theorems_verified": len(seen),
        "zero_axioms_all": True,
        "imports": [],
    }


def _check_gauge_row(row: Any, exp_m: int, exp_n: int, exp_r: int) -> None:
    d = _req_dict(row, "GAUGE_ROW_NOT_DICT")
    m = _req_int(d.get("m"), "GAUGE_INT_INVALID")
    n = _req_int(d.get("n"), "GAUGE_INT_INVALID")
    r = _req_int(d.get("rank"), "GAUGE_INT_INVALID")
    p_raw = _req_int(d.get("p_raw"), "GAUGE_INT_INVALID")
    d_eff = _req_int(d.get("d_eff"), "GAUGE_INT_INVALID")
    require_ok(m == exp_m, "GAUGE_M_MISMATCH")
    require_ok(n == exp_n, "GAUGE_N_MISMATCH")
    require_ok(r == exp_r, "GAUGE_RANK_MISMATCH")
    require_ok(p_raw == r * (m + n), "GAUGE_PRAW_MISMATCH")
    require_ok(d_eff + r * r == p_raw, "GAUGE_QUOTIENT_MISMATCH")


def _audit_payload_pillars(p: dict[str, Any]) -> None:
    """Independent 4-pillar structural, strict-type, and algebraic invariant audit (no oracle replay)."""
    require_ok(set(p.keys()) == REQUIRED_PAYLOAD_KEYS, "PAYLOAD_KEYS_MISMATCH")
    require_ok(p["track_id"] == TRACK_ID, "PAYLOAD_TRACK_MISMATCH")
    require_ok(p["oracle_class"] == "exact", "ORACLE_CLASS_MISMATCH")
    require_ok(p["novelty_tier"] == "N1", "NOVELTY_TIER_MISMATCH")
    require_ok(p["all_pillars_verified"] is True, "PILLARS_NOT_VERIFIED")

    ce = _req_dict(p["constants_enclosures"], "CONSTANTS_NOT_DICT")
    require_ok(ce.get("ln2_lo_dec12") == "0.693147180560", "LN2_ENCLOSURE_MISMATCH")
    require_ok(ce.get("pi_lo_dec12") == "3.141592653590", "PI_ENCLOSURE_MISMATCH")

    eq = _req_dict(p["epistemic_quarantine"], "QUARANTINE_NOT_DICT")
    q_list = eq.get("analytic_extrapolations_quarantined")
    require_ok(type(q_list) is list, "QUARANTINE_LIST_INVALID")
    require_ok(len(q_list) == 1, "QUARANTINE_LIST_INVALID")

    # Pillar 1: Engram benchmark & conservation receipts
    p1 = _req_dict(p["pillar_1_engram_audit"], "P1_NOT_DICT")
    require_ok(p1.get("all_verified") is True, "P1_NOT_VERIFIED")
    r128 = _req_dict(p1.get("fig3_v_proj_rank128"), "P1_R128_NOT_DICT")
    r512 = _req_dict(p1.get("full_rank_v_proj_rank512"), "P1_R512_NOT_DICT")
    emp = _req_dict(p1.get("empirical_anchor_receipts"), "P1_EMP_NOT_DICT")
    _check_gauge_row(r128, 512, 1024, 128)
    _check_gauge_row(r512, 512, 1024, 512)
    require_ok(r128.get("mp_kb_decimal_dec3") == "23.937", "P1_FIG3_KB_MISMATCH")
    require_ok(r512.get("kappa_lo_dec6") == "1.000000", "P1_FULL_RANK_KAPPA_MISMATCH")
    require_ok(emp.get("fig7_exact_unit_conservation") is True, "P1_FIG7_CONSERVATION_FAILED")

    # Pillar 2: GL_r gauge quotient x Marchenko-Pastur sweep
    p2 = _req_dict(p["pillar_2_gauge_vs_mp_reconciliation"], "P2_NOT_DICT")
    require_ok(p2.get("all_verified") is True, "P2_NOT_VERIFIED")
    sweep = p2.get("rank_sweep")
    require_ok(type(sweep) is list, "P2_SWEEP_NOT_LIST")
    require_ok(len(sweep) == len(EXPECTED_SWEEP_RANKS), "P2_SWEEP_LEN_MISMATCH")
    for row, exp_r in zip(sweep, EXPECTED_SWEEP_RANKS):
        _check_gauge_row(row, 512, 1024, exp_r)
    sq = _req_dict(p2.get("square_1024x1024_edge"), "P2_SQUARE_NOT_DICT")
    require_ok(sq.get("saturates_exact_2x") is True, "P2_SQUARE_SATURATION_FAILED")
    require_ok(p2.get("full_rank_wilks_kappa_exact_one") is True, "P2_WILKS_CLOSURE_FAILED")

    # Pillar 3: Track S34 MDL cancellation, Engram noise no-profit, CS-15/CS-16 obstruction
    p3 = _req_dict(p["pillar_3_s34_mdl_and_cs15_cs16"], "P3_NOT_DICT")
    require_ok(p3.get("all_verified") is True, "P3_NOT_VERIFIED")
    canc = _req_dict(p3.get("s34_offline_online_cancellation"), "P3_CANC_NOT_DICT")
    regs = canc.get("regimes")
    require_ok(type(regs) is list, "P3_REGIMES_NOT_LIST")
    require_ok(len(regs) == 3, "P3_REGIMES_LEN_MISMATCH")
    for reg in regs:
        rd = _req_dict(reg, "P3_REGIME_NOT_DICT")
        l_off = _req_int(rd.get("L_offline_bits"), "P3_INT_INVALID")
        l_on = _req_int(rd.get("L_online_bits"), "P3_INT_INVALID")
        l_diff = _req_int(rd.get("L_diff_bits"), "P3_INT_INVALID")
        b_th = _req_int(rd.get("b_theta_bits"), "P3_INT_INVALID")
        r_bits = _req_int(rd.get("regret_R_bits"), "P3_INT_INVALID")
        res = _req_int(rd.get("cancellation_residual"), "P3_INT_INVALID")
        require_ok(l_off - l_on == l_diff, "P3_CANC_DIFF_MISMATCH")
        require_ok(l_diff - (b_th - r_bits) == res, "P3_CANC_RESIDUAL_MISMATCH")
        require_ok(res == 0, "P3_CANC_NONZERO_RESIDUAL")
    noise = _req_dict(p3.get("engram_noise_mdl_no_profit"), "P3_NOISE_NOT_DICT")
    require_ok(noise.get("all_strictly_negative_profit") is True, "P3_NOISE_PROFIT_FAILED")
    obs = _req_dict(p3.get("cs15_cs16_obstruction"), "P3_OBS_NOT_DICT")
    require_ok(_req_int(obs.get("net_joint_bit_gain_bytes"), "P3_INT_INVALID") == 0, "P3_NET_GAIN_NONZERO")
    require_ok(obs.get("s31_cs16_matched_ratio_q") == "66/25", "P3_CS16_RATIO_MISMATCH")
    require_ok(obs.get("capacity_sufficient_for_train_not_ood") is True, "P3_CS15_FLAG_FAILED")

    # Pillar 4: Joint (bpw, rank) bit-allocation law & convex KL lower-bitrate floor
    p4 = _req_dict(p["pillar_4_joint_bpw_rank_allocation_law"], "P4_NOT_DICT")
    require_ok(p4.get("all_verified") is True, "P4_NOT_VERIFIED")
    models = _req_dict(p4.get("frozen_model_allocations"), "P4_MODELS_NOT_DICT")
    require_ok(set(models.keys()) == EXPECTED_FROZEN_MODELS, "P4_MODEL_KEYS_MISMATCH")
    for mv in models.values():
        md = _req_dict(mv, "P4_MODEL_NOT_DICT")
        require_ok(md.get("weight_count_exact_match") is True, "P4_WEIGHT_MATCH_FAILED")
        for int_key in (
            "total_block_weights",
            "p_rank1_all7",
            "rank_unlock_all7_b16",
            "rtx3080_6gib_bf16_rmax_b16",
            "rtx3080_6gib_bf16_rmax_b64_adamw",
            "rtx3080_7gib_bf16_rmax_b16",
            "rtx3080_6gib_quant_rmax_b16",
        ):
            _req_int(md.get(int_key), "P4_INT_INVALID")
    q3 = models["Qwen/Qwen2.5-3B"]
    g4 = models["google/gemma-3-4b-it"]
    require_ok(q3["rank_unlock_all7_b16"] == 1065, "P4_QWEN3B_UNLOCK_MISMATCH")
    require_ok(g4["rtx3080_6gib_bf16_fits"] is False, "P4_GEMMA_BF16_FITS_MISMATCH")
    require_ok(g4["rtx3080_6gib_quant_fits"] is True, "P4_GEMMA_QUANT_FITS_MISMATCH")

    ladder = _req_dict(p4.get("gemma3_4b_convex_kl_ladder"), "P4_LADDER_NOT_DICT")
    require_ok(ladder.get("empirical_strict_convexity_verified") is True, "P4_CONVEXITY_FAILED")
    require_ok(
        ladder.get("extrapolated_3p0_bpw_quarantine")
        == "ANALYTIC_S34_EXTRAPOLATION_NOT_LIVE_GPU_MEASURED",
        "P4_QUARANTINE_MISMATCH",
    )
    sched = p4.get("gemma3_4b_lower_bitrate_floor_schedule")
    require_ok(type(sched) is list, "P4_SCHED_NOT_LIST")
    require_ok(len(sched) == 5, "P4_SCHED_LEN_MISMATCH")
    s1b = _req_dict(sched[2], "P4_SCHED_ROW_NOT_DICT")
    require_ok(s1b.get("horizon_label") == "serving_1B", "P4_SCHED_1B_MISMATCH")
    require_ok(s1b.get("siliquant_4p5_above_single_floor") is True, "P4_SCHED_4P5_MISMATCH")
    require_ok(s1b.get("siliquant_3p0_above_single_floor") is False, "P4_SCHED_3P0_MISMATCH")


def secondary_audit_payload(
    cert_dict: Any,
    lean_report: Any = None,
) -> dict[str, Any]:
    """Raising independent secondary audit (envelope digests + all 4 pillars; zero oracle replay)."""
    require_ok(type(cert_dict) is dict, "CERT_NOT_DICT")
    _check_json_depth_and_types(cert_dict)
    require_ok(set(cert_dict.keys()) == REQUIRED_TOP_KEYS, "CERT_TOP_KEYS_MISMATCH")
    require_ok(cert_dict["schema_version"] == SCHEMA_VERSION, "SCHEMA_VERSION_MISMATCH")
    require_ok(cert_dict["track_id"] == TRACK_ID, "TRACK_ID_MISMATCH")
    require_ok(_is_sha256_hex(cert_dict["payload_sha256"]), "INVALID_PAYLOAD_SHA256")
    require_ok(_is_sha256_hex(cert_dict["certificate_sha256"]), "INVALID_CERT_SHA256")
    require_ok(type(cert_dict["payload"]) is dict, "PAYLOAD_NOT_DICT")

    calc_payload_sha = hashlib.sha256(canonical_json_bytes(cert_dict["payload"])).hexdigest()
    require_ok(
        hmac.compare_digest(calc_payload_sha, cert_dict["payload_sha256"]),
        "PAYLOAD_SHA256_TAMPERED",
    )
    env_body = {
        "schema_version": cert_dict["schema_version"],
        "track_id": cert_dict["track_id"],
        "payload_sha256": cert_dict["payload_sha256"],
        "payload": cert_dict["payload"],
    }
    calc_cert_sha = hashlib.sha256(canonical_json_bytes(env_body)).hexdigest()
    require_ok(
        hmac.compare_digest(calc_cert_sha, cert_dict["certificate_sha256"]),
        "CERTIFICATE_SHA256_TAMPERED",
    )

    _audit_payload_pillars(cert_dict["payload"])

    if lean_report is not None:
        lr = _req_dict(lean_report, "LEAN_REPORT_NOT_DICT")
        require_ok(lr.get("static_clean") is True, "LEAN_REPORT_INCOMPLETE")
        require_ok(lr.get("zero_axioms_all") is True, "LEAN_REPORT_INCOMPLETE")
        n_ver = _req_int(lr.get("theorems_verified"), "LEAN_REPORT_INCOMPLETE")
        require_ok(n_ver == len(EXPECTED_LEAN_THEOREMS), "LEAN_REPORT_INCOMPLETE")

    return {
        "ok": True,
        "secondary_audit_passed": True,
        "certificate_sha256": cert_dict["certificate_sha256"],
        "payload_sha256": cert_dict["payload_sha256"],
        "lean_verified": lean_report is not None,
    }


def verify_certificate_dict(data: dict[str, Any], *, full_replay: bool = True) -> dict[str, Any]:
    """Verify envelope, independent 4-pillar audit (FR0), and optional pure-Z/Q oracle replay (FR1)."""
    require_ok(type(full_replay) is bool, "INVALID_FULL_REPLAY_FLAG")
    audit = secondary_audit_payload(data)
    if full_replay:
        recomputed = build_certificate_envelope()
        require_ok(
            hmac.compare_digest(data["payload_sha256"], recomputed["payload_sha256"]),
            "ORACLE_PAYLOAD_DRIFT",
        )
        require_ok(
            hmac.compare_digest(data["certificate_sha256"], recomputed["certificate_sha256"]),
            "ORACLE_CERT_DRIFT",
        )
    return {
        "schema_version": data["schema_version"],
        "track_id": data["track_id"],
        "payload_sha256": data["payload_sha256"],
        "certificate_sha256": data["certificate_sha256"],
        "all_pillars_verified": True,
        "full_replay": full_replay,
        "secondary_audit_passed": audit["secondary_audit_passed"],
    }


def verify_pinned_sources_on_disk(expedition_root: Path) -> dict[str, Any]:
    """Verify exact on-disk SHA-256 digests and byte lengths of all pinned upstream receipts."""
    require_ok(expedition_root.is_dir(), "EXPEDITION_ROOT_INVALID")
    checked: dict[str, Any] = {}
    for key, meta in PINNED_SOURCES.items():
        p = expedition_root / meta["relative_ref"]
        raw = _read_regular_file_bytes(p, MAX_JSON_BYTES)
        sha = hashlib.sha256(raw).hexdigest()
        require_ok(hmac.compare_digest(sha, meta["sha256"]), f"PINNED_SOURCE_SHA_MISMATCH:{key}")
        require_ok(len(raw) == meta["bytes"], f"PINNED_SOURCE_BYTES_MISMATCH:{key}")
        checked[key] = {"sha256": sha, "bytes": len(raw), "match": True}
    return {"sources_checked": len(checked), "all_matched": True, "details": checked}


def _parse_cli(argv: list[str]) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "write": None,
        "check": None,
        "lean": None,
        "expedition_root": None,
        "standalone": False,
        "json_out": False,
    }
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg in ("--write", "--check", "--lean", "--expedition-root"):
            key = arg[2:].replace("-", "_")
            require_ok(opts[key] is None, f"DUPLICATE_FLAG:{arg}")
            require_ok(i + 1 < n, f"MISSING_VALUE:{arg}")
            opts[key] = _validate_path_str(argv[i + 1])
            i += 2
        elif arg == "--standalone":
            require_ok(not opts["standalone"], "DUPLICATE_FLAG:--standalone")
            opts["standalone"] = True
            i += 1
        elif arg == "--json":
            require_ok(not opts["json_out"], "DUPLICATE_FLAG:--json")
            opts["json_out"] = True
            i += 1
        else:
            raise ValueError(f"UNKNOWN_ARG:{arg}")
    require_ok(not (opts["write"] is not None and opts["check"] is not None), "MUTUALLY_EXCLUSIVE_WRITE_CHECK")
    return opts


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        opts = _parse_cli(args)
        report: dict[str, Any] = {"ok": True, "track_id": TRACK_ID}

        if opts["write"] is not None:
            env = build_certificate_envelope()
            cert_summary = verify_certificate_dict(env)
            atomic_write_0600(opts["write"], canonical_json_bytes(env))
            report["certificate"] = cert_summary
            report["wrote"] = opts["write"].name
        elif opts["check"] is not None:
            loaded = load_strict_json_file(opts["check"])
            report["certificate"] = verify_certificate_dict(loaded)
        else:
            env = build_certificate_envelope()
            report["certificate"] = verify_certificate_dict(env)

        if opts["lean"] is not None:
            report["lean"] = verify_lean_proof(opts["lean"])
        elif not opts["standalone"] and opts["check"] is None and opts["write"] is None:
            default_lean = Path(__file__).resolve().parent / "LoraCapacityJointMdl.lean"
            if default_lean.exists():
                report["lean"] = verify_lean_proof(default_lean)

        if opts["expedition_root"] is not None:
            report["pinned_sources"] = verify_pinned_sources_on_disk(opts["expedition_root"])

        if opts["json_out"]:
            print(json.dumps(report, sort_keys=True, allow_nan=False))
        else:
            print(f"PASS {TRACK_ID} cert={report['certificate']['certificate_sha256']}")
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        err = {"ok": False, "error": str(exc)}
        if "--json" in args:
            print(json.dumps(err, sort_keys=True, allow_nan=False))
        else:
            print(f"FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
