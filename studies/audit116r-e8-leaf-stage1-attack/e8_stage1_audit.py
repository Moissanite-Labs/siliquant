#!/usr/bin/env python3
"""Standalone & fail-closed Zone-B CLI certifier for Track S15d (AUDIT-116).

Strict discipline:
  - Pure Python 3 stdlib, zero floats, zero bare asserts.
  - Canonical deterministic JSON output (4-mode -I parity).
  - Fail-closed --check mode verifying strict JSON types (rejecting floats/NaN/duplicates/deep
    nesting/over-4300-digit ints), iterative depth cap, hmac.compare_digest on payload_sha256,
    and strict type-exact recomputation against e8_stage1_oracle.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from functools import lru_cache
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from e8_stage1_oracle import build_stage1_certificate


class Stage1AuditError(ValueError):
    """Fail-closed error raised on invalid or tampered certificate payloads."""


_MAX_JSON_INT_BITS = 2048
_HEX64_CHARS = frozenset("0123456789abcdef")


def _is_hex64_ascii(val: str) -> bool:
    if len(val) != 64:
        return False
    for ch in val:
        if ch not in _HEX64_CHARS:
            return False
    return True


def _assert_safe_json_tree(obj: Any, *, max_depth: int = 64) -> None:
    """Iterative float-free, str-key-only, depth-capped, int-width-capped JSON structural guard."""
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            raise Stage1AuditError("JSON nesting depth exceeds maximum allowed")
        t = type(cur)
        if cur is None or t in (bool, str):
            continue
        if t is int:
            if cur.bit_length() > _MAX_JSON_INT_BITS:
                raise Stage1AuditError("JSON integer bit length exceeds maximum allowed")
            continue
        if t is dict:
            for k, v in cur.items():
                if type(k) is not str:
                    raise Stage1AuditError("JSON object keys must be strict str")
                stack.append((v, depth + 1))
            continue
        if t is list:
            for item in cur:
                stack.append((item, depth + 1))
            continue
        raise Stage1AuditError(f"forbidden JSON value type: {t.__name__}")


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    """Serialize obj to canonical ASCII JSON with sorted keys and compact separators."""
    if type(obj) is not dict:
        raise Stage1AuditError("canonical_json_bytes requires strict dict")
    _assert_safe_json_tree(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def compute_payload_sha256(payload: dict[str, Any]) -> str:
    """Compute SHA-256 over the certificate body excluding payload_sha256."""
    if type(payload) is not dict:
        raise Stage1AuditError("payload must be strict dict")
    body = {k: v for k, v in payload.items() if k != "payload_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


@lru_cache(maxsize=1)
def _cached_signed_certificate_json() -> str:
    cert = build_stage1_certificate()
    cert["payload_sha256"] = compute_payload_sha256(cert)
    return json.dumps(cert, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def build_signed_certificate() -> dict[str, Any]:
    """Build deterministic Track S15d certificate stamped with payload_sha256."""
    return json.loads(_cached_signed_certificate_json())


def _require_key(obj: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in obj:
        raise Stage1AuditError(f"missing required key: {key}")
    val = obj[key]
    if type(val) is not expected_type:
        raise Stage1AuditError(f"key {key} has invalid type")
    return val


def validate_certificate_payload(payload: Any) -> dict[str, Any]:
    """Fail-closed Zone-B validator checking schema, strict types, hmac digest, and oracle parity."""
    if type(payload) is not dict:
        raise Stage1AuditError("certificate root must be a JSON object")
    _assert_safe_json_tree(payload)

    schema = _require_key(payload, "schema_version", str)
    audit_id = _require_key(payload, "audit_id", str)
    mode = _require_key(payload, "mode", str)
    novelty = _require_key(payload, "novelty_class", str)
    overall = _require_key(payload, "overall_verified", bool)
    claimed_sha = _require_key(payload, "payload_sha256", str)
    _require_key(payload, "pillar1_exact_bit_widths", dict)
    _require_key(payload, "pillar2_bucket_guide_and_lut_concentration", dict)
    _require_key(payload, "pillar3_d8plus_leader_bijection_and_wall", dict)

    if schema != "audit116.e8_stage1.v1":
        raise Stage1AuditError("unexpected schema_version")
    if audit_id != "AUDIT-116":
        raise Stage1AuditError("unexpected audit_id")
    if mode != "standalone_pure_zq":
        raise Stage1AuditError("unexpected mode")
    if novelty != "N1":
        raise Stage1AuditError("unexpected novelty_class")
    if not overall:
        raise Stage1AuditError("overall_verified must be True")

    if not _is_hex64_ascii(claimed_sha):
        raise Stage1AuditError("payload_sha256 must be 64 lowercase hex ASCII characters")

    recomputed_sha = compute_payload_sha256(payload)
    if not hmac.compare_digest(claimed_sha, recomputed_sha):
        raise Stage1AuditError("payload_sha256 mismatch")

    expected = build_signed_certificate()
    if not hmac.compare_digest(recomputed_sha, expected["payload_sha256"]):
        raise Stage1AuditError("certificate content diverges from pure-Z/Q oracle")

    return {
        "valid": True,
        "audit_id": audit_id,
        "mode": mode,
        "payload_sha256": recomputed_sha,
    }


def _reject_float_or_const(token: str) -> Any:
    raise Stage1AuditError(f"forbidden JSON float or constant: {token}")


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise Stage1AuditError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def validate_certificate_file(path: Path) -> dict[str, Any]:
    """Read and fail-closed validate a certificate JSON file from disk."""
    if not isinstance(path, Path):
        raise Stage1AuditError("path must be Path")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Stage1AuditError(f"cannot read certificate file: {exc}") from exc
    try:
        loaded = json.loads(
            raw,
            parse_float=_reject_float_or_const,
            parse_constant=_reject_float_or_const,
            object_pairs_hook=_reject_dup_keys,
        )
    except Stage1AuditError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise Stage1AuditError(f"invalid JSON certificate: {exc}") from exc
    return validate_certificate_payload(loaded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S15d (AUDIT-116) Stage-1 pure-Z/Q certifier.")
    parser.add_argument("--standalone", action="store_true", help="Run standalone pure-Z/Q proof.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON output.")
    parser.add_argument("--check", type=Path, default=None, help="Fail-closed check a certificate JSON file.")
    args = parser.parse_args(argv)

    if args.check is not None:
        try:
            verdict = validate_certificate_file(args.check)
        except Stage1AuditError as exc:
            sys.stderr.write(f"CHECK FAILED: {exc}\n")
            return 1
        sys.stdout.write(json.dumps(verdict, sort_keys=True, ensure_ascii=True) + "\n")
        return 0

    rendered = _cached_signed_certificate_json()
    if args.json:
        sys.stdout.write(rendered)
    else:
        cert = json.loads(rendered)
        p1 = cert["pillar1_exact_bit_widths"]
        p3 = cert["pillar3_d8plus_leader_bijection_and_wall"]
        sys.stdout.write(
            f"AUDIT-116 PURE-ZQ OK: 88b_W1={p1['ebg32_b88s8']['w1_ball_bits']}b(u32) "
            f"136b_W1={p1['ebg32_b136s8']['w1_ball_bits']}b(u64) "
            f"d8_leaders_88b={p3['ebg32_b88s8_leaders']['total_leaders']} "
            f"d8_leaders_136b={p3['ebg32_b136s8_leaders']['total_leaders']} "
            f"sha256={cert['payload_sha256'][:16]}...\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
