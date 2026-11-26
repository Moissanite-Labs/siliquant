#!/usr/bin/env python3
"""
Track S20b Post-Freeze Live Evaluation (`AUDIT-119B-LIVE`): Exact-Q Gate-8
Evaluation Receipt & Brutally Honest Law-4 Timestamp Audit against Frozen
Track S20b (`e8had_anisotropy_oracle.evaluate_gate8_rows`).

Zero floats on all trust/certificate paths (pure stdlib `int`, `Fraction`, and
`Decimal` for lossless raw JSONL ingestion). Zero bare `assert`s (`-O`-safe).
Full Zone-B I/O hardening (`strict_cert_io` contract):
- Strict `type()` checks rejecting `bool`-as-`int` and builtin subclasses;
- Explicit `MAX_INT_BITS = 2048` / `MAX_INT_DIGITS = 620` caps with `.bit_length()`-only
  error formatting (immune to `10**5000` 4,300-digit `ValueError`);
- Strictly iterative `MAX_JSON_DEPTH = 64` JSON tree validation;
- Strict 64-lowercase-ASCII-hex check BEFORE `hmac.compare_digest`;
- Read-only `sys.dont_write_bytecode = True` dynamic import of frozen Track S20b
  in `--live-s20b` mode (ZERO writes / ZERO `__pycache__` inside Track S20b!).
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

MAX_JSON_DEPTH = 64
MAX_JSON_BYTES = 2_000_000
MAX_INT_BITS = 2048
MAX_INT_DIGITS = 620
_HEX_LOWER = frozenset("0123456789abcdef")

# Pinned frozen Track S20b (`AUDIT-119B` R2) identities (strictly read-only).
TRACK_S20B_FROZEN_PINS: dict[str, str] = {
    "track": "S20b",
    "audit_id": "AUDIT-119B",
    "oracle_r2_sha256": "dc13922753737ec127cd59817479680d7615b2a326d2664b56d0016fb146dc72",
    "audit_sha256": "574f39a049287706026adbd072d78d10b463ae29f8b65ee337d3201fc238b561",
    "certificate_file_sha256": "84e79bb9f2dab219228b5fc80688301fde92710f9c0978f6385f6a7f30ee4b0d",
    "payload_sha256": "83b8caa4a11b51e57f19203a4b18fa1463fc464a45a9a733d15b167f531e67e0",
    "certificate_sha256": "2a889f7412ac3a76d0fdd001881d96c463f464f428d479f079fab93d6cda2af5",
    "report_sha256": "91b56a6855c6e548d5ee22208076dc7f8509f7b0de7a8666c44aa75849fba65f",
}

# Pinned upstream Moissanite-Labs/siliquant branch `bayan/charming-gates-jlas14` snapshots.
UPSTREAM_SNAPSHOTS: dict[str, Any] = {
    "repo": "Moissanite-Labs/siliquant",
    "repo_is_private": False,
    "branch": "bayan/charming-gates-jlas14",
    "experiment_dir": "experiments/e8-hadamard-real-weights-2026-11-22",
    "prereg_anchor_commit": "51235fa753ecd32a15c9af0251560156e3bf440f",
    "prereg_anchor_utc": "2026-11-22T23:06:57Z",
    "prereg_results_jsonl_sha256": "e4023053b6f17cd7a0e820cc591a328d0b4ad6738d10c40d8a5f9e2a9920aa5c",
    "prereg_landed_kl_rows": 13,
    "ebg_row_commit": "0d843d956bb2ef6ce90e3563f995a3ca5750f86a",
    "ebg_row_utc": "2026-11-22T23:16:15Z",
    "hbg_row_commit": "be63add87658f7e477fdf436f9396efd7950f1bf",
    "hbg_row_utc": "2026-11-22T23:26:03Z",
    "eval_branch_head_commit": "462fb2f82518faa5203edc2784b93fcf64b06609",
    "eval_branch_head_utc": "2026-11-23T00:04:07Z",
    "eval_results_jsonl_sha256": "f9c495cd81184688436947c8d1e9042da9e67d180f0b2e481deb43b725c9d3c1",
    "eval_results_jsonl_bytes": 34348,
    "eval_results_jsonl_rows": 15,
    "eval_metrics_jsonl_sha256": "c0f081fe047dc79f27d1ec6a8dc5f9e05487ac3e330c25c91fe88ecfe8cf8880",
    "eval_metrics_jsonl_weight_only_rows": 7,
}

# Exact integer micro-unit rows extracted via lossless Decimal ROUND_HALF_UP from `results.jsonl`:
# Row 12 (anchor 51235fa7), Row 13 (0d843d95), Row 14 (be63add8).
QWEN25_3P0_ROWS_U6: dict[str, dict[str, int]] = {
    "qwen25_3p0_zbg": {
        "kl_u6": 215147,
        "se_u6": 3372,
        "werr_mdB": -15797,
        "z005_bp": 9991,
        "err005_mdB": -53,
        "z020_bp": 9664,
        "err020_mdB": -443,
    },
    "qwen25_3p0_ebg": {
        "kl_u6": 169338,
        "se_u6": 3095,
        "werr_mdB": -16455,
        "z005_bp": 6777,
        "err005_mdB": 13684,
        "z020_bp": 5294,
        "err020_mdB": 1561,
    },
    "qwen25_3p0_hbg": {
        "kl_u6": 178526,
        "se_u6": 3104,
        "werr_mdB": -16453,
        "z005_bp": 9393,
        "err005_mdB": 9093,
        "z020_bp": 7502,
        "err020_mdB": 1459,
    },
}

# Exact input dict passed verbatim to frozen Track S20b `evaluate_gate8_rows(...)`.
LIVE_SCORER_INPUT_ROWS: dict[str, dict[str, int]] = {
    "qwen25_3p0_ebg": {
        "kl_u6": QWEN25_3P0_ROWS_U6["qwen25_3p0_ebg"]["kl_u6"],
        "se_u6": QWEN25_3P0_ROWS_U6["qwen25_3p0_ebg"]["se_u6"],
    },
    "qwen25_3p0_hbg": {
        "kl_u6": QWEN25_3P0_ROWS_U6["qwen25_3p0_hbg"]["kl_u6"],
        "se_u6": QWEN25_3P0_ROWS_U6["qwen25_3p0_hbg"]["se_u6"],
        "z005_bp": QWEN25_3P0_ROWS_U6["qwen25_3p0_hbg"]["z005_bp"],
        "err005_mdB": QWEN25_3P0_ROWS_U6["qwen25_3p0_hbg"]["err005_mdB"],
    },
}

# Exact UTC epoch seconds for Law-4 timestamp audit (brutally honest provenance timeline).
TIMELINE_UTC_EPOCHS: dict[str, tuple[str, int]] = {
    "anchor_51235fa7_commit": ("2026-11-22T23:06:57Z", 1795388817),
    "g8p1_oracle_first_written": ("2026-11-22T23:11:58Z", 1795389118),
    "g8p1_oracle_finalized": ("2026-11-22T23:15:03Z", 1795389303),
    "upstream_0d843d95_ebg_commit": ("2026-11-22T23:16:15Z", 1795389375),
    "report_md_final_mtime": ("2026-11-22T23:18:41Z", 1795389521),
    "upstream_be63add8_hbg_commit": ("2026-11-22T23:26:03Z", 1795389963),
    "cert_json_final_mtime": ("2026-11-22T23:27:32Z", 1795390052),
    "board_post_1155_r1": ("2026-11-22T23:39:29Z", 1795390769),
    "upstream_462fb2f8_branch_head": ("2026-11-23T00:04:07Z", 1795392247),
}


class Gate8EvalError(ValueError):
    """Fail-closed typed validation error (never stripped under `python3 -O`)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _require_valid_utf8_str(s: str, label: str = "str") -> str:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise Gate8EvalError("INVALID_UTF8", f"{label} contains unpaired UTF-16 surrogate") from exc
    return s


def require_int(val: Any, label: str = "int", *, lo: int | None = None, hi: int | None = None) -> int:
    if type(val) is not int:
        raise Gate8EvalError("INVALID_INT_TYPE", f"{label} must be strict int, got {type(val).__name__}")
    if lo is not None and type(lo) is not int:
        raise Gate8EvalError("INVALID_BOUND_TYPE", f"{label} lower bound must be strict int")
    if hi is not None and type(hi) is not int:
        raise Gate8EvalError("INVALID_BOUND_TYPE", f"{label} upper bound must be strict int")
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise Gate8EvalError("INT_BIT_LIMIT", f"{label} bit_length={blen} exceeds {MAX_INT_BITS}")
    if lo is not None and val < lo:
        raise Gate8EvalError("INT_BELOW_MIN", f"{label} bit_length={blen} below min_bits={lo.bit_length()}")
    if hi is not None and val > hi:
        raise Gate8EvalError("INT_ABOVE_MAX", f"{label} bit_length={blen} above max_bits={hi.bit_length()}")
    return val


def validate_cli_argv(argv: Any) -> list[str] | None:
    if argv is None:
        return None
    if type(argv) not in (list, tuple):
        raise Gate8EvalError("INVALID_ARGV", "argv must be list or tuple of strict str")
    out: list[str] = []
    for item in argv:
        if type(item) is not str:
            raise Gate8EvalError("INVALID_ARGV", "argv entries must be strict str")
        _require_valid_utf8_str(item, "argv")
        out.append(item)
    return out


def require_hex64_ascii(val: Any, label: str = "sha256") -> str:
    if type(val) is not str:
        raise Gate8EvalError("INVALID_DIGEST_TYPE", f"{label} must be strict str")
    if len(val) != 64:
        raise Gate8EvalError("INVALID_DIGEST_LEN", f"{label} must have length 64")
    for ch in val:
        if ch not in _HEX_LOWER:
            raise Gate8EvalError("INVALID_DIGEST_HEX", f"{label} must be lowercase ASCII hex")
    return val


def constant_time_hex64_equal(left: Any, right: Any, label: str = "sha256") -> bool:
    a = require_hex64_ascii(left, f"{label}.left")
    b = require_hex64_ascii(right, f"{label}.right")
    return hmac.compare_digest(a, b)


def fmt_q(q: Fraction) -> str:
    if type(q) is not Fraction:
        raise Gate8EvalError("INVALID_FRACTION_TYPE", "q must be strict Fraction")
    nb = q.numerator.bit_length()
    db = q.denominator.bit_length()
    if nb > MAX_INT_BITS or db > MAX_INT_BITS:
        raise Gate8EvalError("FRACTION_BIT_LIMIT", f"fraction bit_lengths=({nb},{db}) exceed {MAX_INT_BITS}")
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def validate_json_value(obj: Any, *, max_depth: int = MAX_JSON_DEPTH) -> None:
    limit = require_int(max_depth, "max_depth", lo=1, hi=128)
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > limit:
            raise Gate8EvalError("MAX_DEPTH_EXCEEDED", f"JSON nesting exceeds {limit}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise Gate8EvalError("NON_STR_KEY", "JSON keys must be strict str")
                _require_valid_utf8_str(k, "json_key")
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is int:
            blen = cur.bit_length()
            if blen > MAX_INT_BITS:
                raise Gate8EvalError("INT_BIT_LIMIT", f"JSON int bit_length={blen} exceeds {MAX_INT_BITS}")
        elif t is str:
            _require_valid_utf8_str(cur, "json_str")
        elif t is bool or cur is None:
            continue
        elif t is float:
            raise Gate8EvalError("FLOAT_FORBIDDEN", "float values are forbidden in exact JSON")
        else:
            raise Gate8EvalError("UNSUPPORTED_JSON_TYPE", f"unsupported JSON type {t.__name__}")


def _reject_float_token(tok: str) -> Any:
    raise Gate8EvalError("JSON_FLOAT_FORBIDDEN", f"float literal forbidden: {tok!r}")


def _reject_constant_token(tok: str) -> Any:
    raise Gate8EvalError("JSON_CONSTANT_FORBIDDEN", f"non-RFC8259 constant forbidden: {tok!r}")


def _parse_capped_int_token(tok: str) -> int:
    digits = tok[1:] if tok.startswith("-") else tok
    if len(digits) > MAX_INT_DIGITS:
        raise Gate8EvalError("INT_DIGIT_LIMIT", f"integer token exceeds {MAX_INT_DIGITS} digits")
    val = int(tok)
    blen = val.bit_length()
    if blen > MAX_INT_BITS:
        raise Gate8EvalError("INT_BIT_LIMIT", f"integer token bit_length={blen} exceeds {MAX_INT_BITS}")
    return val


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise Gate8EvalError("JSON_DUPLICATE_KEY", f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def parse_strict_json_text(raw_text: str, *, require_object: bool = True, max_depth: int = MAX_JSON_DEPTH) -> Any:
    if type(require_object) is not bool:
        raise Gate8EvalError("INVALID_BOOL_TYPE", "require_object must be strict bool")
    if type(raw_text) is not str:
        raise Gate8EvalError("INVALID_JSON_INPUT", "JSON text must be strict str")
    try:
        encoded = raw_text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise Gate8EvalError("INVALID_UTF8", f"JSON text contains unpaired surrogate: {exc}") from exc
    if len(encoded) > MAX_JSON_BYTES:
        raise Gate8EvalError("JSON_TOO_LARGE", f"JSON exceeds {MAX_JSON_BYTES} bytes")
    if len(raw_text.strip()) == 0:
        raise Gate8EvalError("EMPTY_JSON_INPUT", "JSON text must be non-empty")
    try:
        obj = json.loads(
            raw_text,
            parse_float=_reject_float_token,
            parse_constant=_reject_constant_token,
            parse_int=_parse_capped_int_token,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except Gate8EvalError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise Gate8EvalError("JSON_SYNTAX_ERROR", str(exc)) from exc
    validate_json_value(obj, max_depth=max_depth)
    if require_object and type(obj) is not dict:
        raise Gate8EvalError("JSON_ROOT_NOT_OBJECT", "top-level JSON payload must be an object")
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    validate_json_value(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def require_no_symlink_path(path: Path) -> None:
    if type(path) is not type(Path()):
        raise Gate8EvalError("INVALID_PATH_TYPE", "path must be strict concrete pathlib.Path")
    raw_s = str(path)
    _require_valid_utf8_str(raw_s, "path")
    if "\x00" in raw_s:
        raise Gate8EvalError("INVALID_PATH_NUL", "NUL byte forbidden in path")
    if path.is_symlink():
        raise Gate8EvalError("SYMLINK_FORBIDDEN", "symlink target path forbidden")
    for parent in path.parents:
        if parent.is_symlink():
            raise Gate8EvalError("SYMLINK_ANCESTOR_FORBIDDEN", "symlink ancestor directory forbidden")
    if not path.is_absolute():
        pwd_env = os.environ.get("PWD")
        if pwd_env is not None and os.path.realpath(pwd_env) == os.path.realpath(os.getcwd()):
            pwd_p = Path(pwd_env)
            if pwd_p.is_symlink() or any(p.is_symlink() for p in pwd_p.parents):
                raise Gate8EvalError("SYMLINK_ANCESTOR_FORBIDDEN", "symlinked working directory forbidden")


def _validate_jsonl_utf8_and_depth(obj: Any) -> None:
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise Gate8EvalError("MAX_DEPTH_EXCEEDED", f"JSONL nesting exceeds {MAX_JSON_DEPTH}")
        t = type(cur)
        if t is dict:
            for k, v in cur.items():
                _require_valid_utf8_str(k, "jsonl_key")
                stack.append((v, depth + 1))
        elif t is list:
            for item in cur:
                stack.append((item, depth + 1))
        elif t is str:
            _require_valid_utf8_str(cur, "jsonl_str")


def _exact_z_sq(a_u6: int, se_a_u6: int, b_u6: int, se_b_u6: int) -> Fraction:
    diff = b_u6 - a_u6
    return Fraction(diff * diff, se_a_u6 * se_a_u6 + se_b_u6 * se_b_u6)


def _round_dec_to_int(val: Any, scale: int, label: str) -> int:
    if type(val) is int:
        return require_int(val, label) * scale
    if type(val) is not Decimal or not val.is_finite():
        raise Gate8EvalError("INVALID_DECIMAL", f"{label} must be finite Decimal or int")
    try:
        scaled = (val * Decimal(scale)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        out = int(scaled)
    except (InvalidOperation, ValueError) as exc:
        raise Gate8EvalError("DECIMAL_SCALE_ERROR", f"{label} scaling failed") from exc
    return require_int(out, label)


def score_g8_p1_standalone(rows: dict[str, Any]) -> dict[str, Any]:
    """Independent exact-Q G8-P1 clause evaluator matching frozen Track S20b bounds."""
    if type(rows) is not dict:
        raise Gate8EvalError("INVALID_ROWS_TYPE", "rows must be strict dict")
    validate_json_value(rows)
    if "qwen25_3p0_ebg" not in rows or "qwen25_3p0_hbg" not in rows:
        raise Gate8EvalError("MISSING_G8P1_ROWS", "both qwen25_3p0_ebg and qwen25_3p0_hbg are required")
    qe = rows["qwen25_3p0_ebg"]
    qh = rows["qwen25_3p0_hbg"]
    if type(qe) is not dict or type(qh) is not dict:
        raise Gate8EvalError("INVALID_ROW_TYPE", "G8-P1 rows must be strict dicts")

    kz = QWEN25_3P0_ROWS_U6["qwen25_3p0_zbg"]["kl_u6"]
    sz = QWEN25_3P0_ROWS_U6["qwen25_3p0_zbg"]["se_u6"]
    ke = require_int(qe.get("kl_u6"), "ebg.kl_u6", lo=1)
    se_e = require_int(qe.get("se_u6"), "ebg.se_u6", lo=1)
    kh = require_int(qh.get("kl_u6"), "hbg.kl_u6", lo=1)
    sh = require_int(qh.get("se_u6"), "hbg.se_u6", lo=1)
    h_z005 = require_int(qh.get("z005_bp"), "hbg.z005_bp", lo=0, hi=10000)
    h_err005 = require_int(qh.get("err005_mdB"), "hbg.err005_mdB", lo=-100000, hi=100000)

    red_hz = Fraction(kz - kh, kz)
    red_ez = Fraction(kz - ke, kz)
    gap_he = Fraction(abs(ke - kh), ke)
    z2_hz = _exact_z_sq(kh, sh, kz, sz)
    z2_ez = _exact_z_sq(ke, se_e, kz, sz)

    clauses = {
        "c1_hbg_lt_zbg": kh < kz,
        "c2_red_hbg_ge_10pct": red_hz >= Fraction(1, 10),
        "c3_z2_hbg_ge_16": z2_hz >= 16,
        "c4_ebg_lt_zbg": ke < kz,
        "c5_red_ebg_ge_8pct": red_ez >= Fraction(2, 25),
        "c6_z2_ebg_ge_9": z2_ez >= 9,
        "c7_gap_he_lt_15pct": gap_he < Fraction(3, 20),
        "c8_hbg_z005_ge_8800bp": h_z005 >= 8800,
        "c9_hbg_err005_ge_6000mdB": h_err005 >= 6000,
    }
    passed_count = sum(1 for v in clauses.values() if v)
    verdicts = {
        "G8-P1": "PASS" if passed_count == 9 else "FAIL",
        "G8-P2": "PENDING",
        "G8-P3": "PENDING",
        "G8-P4": "PENDING",
    }
    details = {
        "G8-P1": {
            "red_hbg_vs_zbg": fmt_q(red_hz),
            "red_ebg_vs_zbg": fmt_q(red_ez),
            "abs_gap_hbg_ebg": fmt_q(gap_he),
            "z2_hbg_vs_zbg": fmt_q(z2_hz),
            "z2_ebg_vs_zbg": fmt_q(z2_ez),
        }
    }
    margins = {
        "red_hbg_margin": fmt_q(red_hz - Fraction(1, 10)),
        "z2_hbg_margin": fmt_q(z2_hz - 16),
        "red_ebg_margin": fmt_q(red_ez - Fraction(2, 25)),
        "z2_ebg_margin": fmt_q(z2_ez - 9),
        "gap_he_headroom": fmt_q(Fraction(3, 20) - gap_he),
        "z005_margin_bp": h_z005 - 8800,
        "err005_margin_mdB": h_err005 - 6000,
    }
    return {
        "verdicts": verdicts,
        "details": details,
        "clauses_passed": passed_count,
        "clauses_total": 9,
        "clauses": clauses,
        "margins": margins,
    }


def parse_results_jsonl_file(path: Path) -> dict[str, Any]:
    """Lossless Decimal ingestion of committed `results.jsonl` (zero Python floats)."""
    require_no_symlink_path(path)
    if not path.is_file():
        raise Gate8EvalError("FILE_NOT_FOUND", "results.jsonl file not found")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise Gate8EvalError("FILE_TOO_LARGE", "results.jsonl exceeds byte limit")
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Gate8EvalError("INVALID_UTF8", "results.jsonl invalid UTF-8") from exc
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) == 0:
        raise Gate8EvalError("EMPTY_JSONL", "results.jsonl has no non-empty rows")
    extracted: dict[str, dict[str, int]] = {}
    variant_map = {
        ("qwen2.5-1.5b-instruct-fp16.gguf", "ZBG128:b368s16"): "qwen25_3p0_zbg",
        ("qwen2.5-1.5b-instruct-fp16.gguf", "EBG128:b368s16"): "qwen25_3p0_ebg",
        ("qwen2.5-1.5b-instruct-fp16.gguf", "HBG128:b368s16"): "qwen25_3p0_hbg",
    }
    for ln in lines:
        try:
            row = json.loads(
                ln,
                parse_float=Decimal,
                parse_constant=_reject_constant_token,
                parse_int=_parse_capped_int_token,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except Gate8EvalError:
            raise
        except (json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise Gate8EvalError("JSONL_SYNTAX_ERROR", str(exc)) from exc
        if type(row) is not dict:
            raise Gate8EvalError("JSONL_ROW_NOT_OBJECT", "each results.jsonl line must be JSON object")
        _validate_jsonl_utf8_and_depth(row)
        key = variant_map.get((row.get("model"), row.get("variant")))
        if key is None:
            continue
        sw = row.get("small_weights")
        if type(sw) is not dict or type(sw.get("0.05")) is not dict or type(sw.get("0.2")) is not dict:
            raise Gate8EvalError("MISSING_SMALL_WEIGHTS", f"{key} missing small_weights 0.05/0.2 blocks")
        s005 = sw["0.05"].get("all")
        s020 = sw["0.2"].get("all")
        if type(s005) is not dict or type(s020) is not dict:
            raise Gate8EvalError("MISSING_SMALL_WEIGHTS_ALL", f"{key} missing small_weights.all blocks")
        extracted[key] = {
            "kl_u6": _round_dec_to_int(row.get("kl_mean"), 1_000_000, f"{key}.kl_mean"),
            "se_u6": _round_dec_to_int(row.get("kl_mean_se"), 1_000_000, f"{key}.kl_mean_se"),
            "werr_mdB": _round_dec_to_int(row.get("weight_err_db_pooled"), 1_000, f"{key}.werr"),
            "z005_bp": _round_dec_to_int(s005.get("stored_zero_percent"), 100, f"{key}.z005"),
            "err005_mdB": _round_dec_to_int(s005.get("err_over_size_db"), 1_000, f"{key}.err005"),
            "z020_bp": _round_dec_to_int(s020.get("stored_zero_percent"), 100, f"{key}.z020"),
            "err020_mdB": _round_dec_to_int(s020.get("err_over_size_db"), 1_000, f"{key}.err020"),
        }
    return {
        "results_jsonl_sha256": file_sha,
        "total_rows": len(lines),
        "extracted_qwen25_3p0_rows": extracted,
        "matches_pinned_sha256": constant_time_hex64_equal(
            file_sha, UPSTREAM_SNAPSHOTS["eval_results_jsonl_sha256"], "results_jsonl_sha256"
        ),
        "matches_pinned_rows": extracted == QWEN25_3P0_ROWS_U6,
    }


def verify_live_s20b_scorer(
    s20b_dir: Path,
    *,
    expected_oracle_sha: str = TRACK_S20B_FROZEN_PINS["oracle_r2_sha256"],
    expected_cert_sha: str = TRACK_S20B_FROZEN_PINS["certificate_file_sha256"],
) -> dict[str, Any]:
    """Strictly read-only live import (`sys.dont_write_bytecode=True`) & parity check against frozen Track S20b."""
    require_no_symlink_path(s20b_dir)
    if not s20b_dir.is_dir():
        raise Gate8EvalError("S20B_DIR_NOT_FOUND", "Track S20b directory not found")
    oracle_path = s20b_dir / "e8had_anisotropy_oracle.py"
    cert_path = s20b_dir / "e8had_anisotropy_certificate.json"
    require_no_symlink_path(oracle_path)
    require_no_symlink_path(cert_path)
    if not oracle_path.is_file() or not cert_path.is_file():
        raise Gate8EvalError("S20B_FILES_MISSING", "frozen Track S20b oracle or certificate missing")
    oracle_sha = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    cert_sha = hashlib.sha256(cert_path.read_bytes()).hexdigest()
    if not constant_time_hex64_equal(oracle_sha, expected_oracle_sha, "s20b_oracle_sha256"):
        raise Gate8EvalError("S20B_ORACLE_SHA_MISMATCH", "Track S20b oracle SHA-256 diverges from frozen pin")
    if not constant_time_hex64_equal(cert_sha, expected_cert_sha, "s20b_cert_sha256"):
        raise Gate8EvalError("S20B_CERT_SHA_MISMATCH", "Track S20b certificate SHA-256 diverges from frozen pin")

    prev_dwbc = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_s20b_frozen_readonly", oracle_path)
        if spec is None or spec.loader is None:
            raise Gate8EvalError("S20B_IMPORT_SPEC_FAILED", "failed to load frozen Track S20b module spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        live_eval = mod.evaluate_gate8_rows(LIVE_SCORER_INPUT_ROWS)
        baseline_eval = mod.evaluate_gate8_rows({})
    finally:
        sys.dont_write_bytecode = prev_dwbc

    standalone = score_g8_p1_standalone(LIVE_SCORER_INPUT_ROWS)
    if live_eval != {"verdicts": standalone["verdicts"], "details": standalone["details"]}:
        raise Gate8EvalError("LIVE_SCORER_PARITY_MISMATCH", "frozen Track S20b scorer diverged from standalone receipt")
    if baseline_eval["verdicts"] != {"G8-P1": "PENDING", "G8-P2": "PENDING", "G8-P3": "PENDING", "G8-P4": "PENDING"}:
        raise Gate8EvalError("BASELINE_PENDING_MISMATCH", "empty baseline evaluation did not return all PENDING")

    return {
        "live_s20b_verified": True,
        "s20b_oracle_sha256": oracle_sha,
        "s20b_certificate_sha256": cert_sha,
        "s20b_live_verdicts": live_eval["verdicts"],
        "s20b_live_details": live_eval["details"],
        "pycache_created_in_s20b": (s20b_dir / "__pycache__").exists(),
    }


def build_certificate_payload() -> dict[str, Any]:
    """Compute the complete exact-Z/Q Track S20b post-freeze Gate-8 evaluation payload."""
    eval_res = score_g8_p1_standalone(LIVE_SCORER_INPUT_ROWS)
    rz = QWEN25_3P0_ROWS_U6["qwen25_3p0_zbg"]
    re = QWEN25_3P0_ROWS_U6["qwen25_3p0_ebg"]
    rh = QWEN25_3P0_ROWS_U6["qwen25_3p0_hbg"]

    # Exact Steiner-jump vs half-integer energy-ratio model on newly landed Qwen2.5 3.0-bit small weights
    nz_005_ratio = Fraction(2 * (10000 - rh["z005_bp"]), 10000 - re["z005_bp"])
    nz_020_ratio = Fraction(2 * (10000 - rh["z020_bp"]), 10000 - re["z020_bp"])
    hbg_minus_ebg_u6 = rh["kl_u6"] - re["kl_u6"]
    z2_hbg_vs_ebg = _exact_z_sq(re["kl_u6"], re["se_u6"], rh["kl_u6"], rh["se_u6"])

    t = TIMELINE_UTC_EPOCHS
    deltas_sec = {
        "ebg_commit_minus_g8p1_oracle_finalized_s": t["upstream_0d843d95_ebg_commit"][1] - t["g8p1_oracle_finalized"][1],
        "hbg_commit_minus_g8p1_oracle_finalized_s": t["upstream_be63add8_hbg_commit"][1] - t["g8p1_oracle_finalized"][1],
        "ebg_commit_minus_report_mtime_s": t["upstream_0d843d95_ebg_commit"][1] - t["report_md_final_mtime"][1],
        "hbg_commit_minus_report_mtime_s": t["upstream_be63add8_hbg_commit"][1] - t["report_md_final_mtime"][1],
        "board_1155_minus_hbg_commit_s": t["board_post_1155_r1"][1] - t["upstream_be63add8_hbg_commit"][1],
    }

    return {
        "track": "S20b-live-eval",
        "audit_id": "AUDIT-119B-LIVE",
        "oracle_class": "exact+empirical-corpus",
        "novelty_class": "N1",
        "frozen_track_s20b_pins": TRACK_S20B_FROZEN_PINS,
        "upstream_snapshots": UPSTREAM_SNAPSHOTS,
        "qwen25_3p0_landed_rows_u6": QWEN25_3P0_ROWS_U6,
        "gate8_evaluation": eval_res,
        "qwen25_3p0_secondary_diagnostics": {
            "hbg_minus_ebg_kl_u6": hbg_minus_ebg_u6,
            "z2_hbg_vs_ebg": fmt_q(z2_hbg_vs_ebg),
            "ebg_werr_advantage_over_zbg_mdB": rz["werr_mdB"] - re["werr_mdB"],
            "hbg_werr_advantage_over_zbg_mdB": rz["werr_mdB"] - rh["werr_mdB"],
            "hbg_minus_ebg_err005_mdB": rh["err005_mdB"] - re["err005_mdB"],
            "hbg_minus_ebg_err020_mdB": rh["err020_mdB"] - re["err020_mdB"],
            "steiner_jump_model_005_energy_ratio_hbg_over_ebg": fmt_q(nz_005_ratio),
            "steiner_jump_model_020_energy_ratio_hbg_over_ebg": fmt_q(nz_020_ratio),
        },
        "law4_timestamp_audit": {
            "events_utc": {k: {"utc": v[0], "epoch_s": v[1]} for k, v in t.items()},
            "exact_deltas_s": deltas_sec,
            "locally_authored_before_upstream_commit": {
                "g8p1_oracle_before_0d843d95_ebg": deltas_sec["ebg_commit_minus_g8p1_oracle_finalized_s"] > 0,
                "g8p1_oracle_before_be63add8_hbg": deltas_sec["hbg_commit_minus_g8p1_oracle_finalized_s"] > 0,
                "report_mtime_before_0d843d95_ebg": deltas_sec["ebg_commit_minus_report_mtime_s"] > 0,
                "report_mtime_before_be63add8_hbg": deltas_sec["hbg_commit_minus_report_mtime_s"] > 0,
            },
            "externally_cryptographically_precommitted_via_board_1155": {
                "G8-P1": False,
                "G8-P2": True,
                "G8-P3": True,
                "G8-P4": True,
            },
        },
    }


def generate_certificate() -> dict[str, Any]:
    payload = build_certificate_payload()
    payload_sha = canonical_sha256(payload)
    cert_body = {
        "schema": "aletheia.track_s20b_gate8_live_eval.v1",
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
    parser = argparse.ArgumentParser(description="Track S20b post-freeze Gate-8 live evaluation oracle")
    parser.add_argument("--standalone", action="store_true", help="Compute self-contained exact certificate")
    parser.add_argument("--live-s20b", type=str, default=None, help="Read-only path to frozen Track S20b directory")
    parser.add_argument("--results-jsonl", type=str, default=None, help="Optional committed results.jsonl to cross-check")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON output to stdout")
    try:
        checked_argv = validate_cli_argv(argv)
        args = parser.parse_args(checked_argv)
        cert = generate_certificate()
        if args.live_s20b is not None:
            live_out: dict[str, Any] = {
                "certificate_sha256": cert["certificate_sha256"],
                "payload_sha256": cert["payload_sha256"],
                "s20b_parity": verify_live_s20b_scorer(Path(args.live_s20b)),
            }
            if args.results_jsonl is not None:
                res_check = parse_results_jsonl_file(Path(args.results_jsonl))
                if not res_check["matches_pinned_sha256"] or not res_check["matches_pinned_rows"]:
                    raise Gate8EvalError("RESULTS_JSONL_MISMATCH", "results.jsonl diverges from pinned snapshot")
                live_out["results_jsonl_parity"] = res_check
            if args.json:
                sys.stdout.write(canonical_json_bytes(live_out).decode("ascii") + "\n")
                return 0
            print(f"OK TRACK-S20B-GATE8-LIVE-PARITY certificate_sha256={cert['certificate_sha256']}")
            return 0
    except Gate8EvalError as exc:
        sys.stderr.write(f"FAIL {exc.code}: {exc.message}\n")
        return 1
    if args.json:
        sys.stdout.write(canonical_json_bytes(cert).decode("ascii") + "\n")
        return 0
    print(
        f"OK TRACK-S20B-GATE8-LIVE-EVAL payload_sha256={cert['payload_sha256']} "
        f"certificate_sha256={cert['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
