#!/usr/bin/env python3
"""Dual-mode pure-Z/Q audit runner for Track S15c (AUDIT-115):
Exact Sylvester-Hadamard E_8_Had isometry, zero-table WHT_8 butterfly equivalence,
DP table invariance, and exact adjoint GEMV identity for Moissanite-Labs/siliquant#23/#25.

Modes:
  --standalone       (default) Self-contained exact pure-Z/Q replay with zero external file I/O
  --check <cert>     Fail-closed Zone-B verification of a JSON certificate (exit 0 on valid, 2 on error)

Strict discipline: pure Python 3 stdlib, exact int/Fraction only, zero floats, zero bare asserts.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hmac
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from e8had_isometry_oracle import (
    E8HadError,
    canonical_sha256,
    certify_dp_invariance_and_adjoint_gemv_exact,
    certify_e8_had_isometry_exact,
    certify_pr24_correspondence_and_roofline_exact,
    contains_forbidden_json_value,
    require_sha256,
)

SCHEMA_VERSION = "audit115.e8had_isometry.v1"

CERT_TOP_KEYS = frozenset({
    "schema_version",
    "audit_id",
    "mode",
    "novelty_class",
    "pillar1_e8had_isometry_and_hamming",
    "pillar2_dp_invariance_and_adjoint_gemv",
    "pillar3_pr24_correspondence_and_roofline",
    "overall_verified",
    "certificate_sha256",
})


@lru_cache(maxsize=1)
def _cached_pillars_canonical_json() -> tuple[str, str, str]:
    p1 = certify_e8_had_isometry_exact(m_max=3)
    p2 = certify_dp_invariance_and_adjoint_gemv_exact()
    p3 = certify_pr24_correspondence_and_roofline_exact()
    return (
        json.dumps(p1, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        json.dumps(p2, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        json.dumps(p3, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


def _fresh_expected_pillars() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    s1, s2, s3 = _cached_pillars_canonical_json()
    return json.loads(s1), json.loads(s2), json.loads(s3)


def _type_strict_eq(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_type_strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_type_strict_eq(x, y) for x, y in zip(a, b))
    return a == b


def run_standalone_audit() -> dict[str, Any]:
    """Self-contained exact pure-Z/Q replay with zero external file I/O."""
    p1, p2, p3 = _fresh_expected_pillars()
    overall = p1["pillar_verified"] and p2["pillar_verified"] and p3["pillar_verified"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": "AUDIT-115",
        "mode": "standalone_pure_zq",
        "novelty_class": "N1",
        "pillar1_e8had_isometry_and_hamming": p1,
        "pillar2_dp_invariance_and_adjoint_gemv": p2,
        "pillar3_pr24_correspondence_and_roofline": p3,
        "overall_verified": overall,
    }
    out = dict(body)
    out["certificate_sha256"] = canonical_sha256(body)
    return out


def validate_audit_certificate(cert: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed Zone-B-hardened certificate validator with exact recomputed pillars."""
    if not isinstance(cert, Mapping):
        raise E8HadError("certificate must be a mapping")
    if contains_forbidden_json_value(cert):
        raise E8HadError("certificate contains forbidden float/non-str-key/non-JSON value")
    if set(cert.keys()) != CERT_TOP_KEYS:
        raise E8HadError("certificate top-level key set mismatch")
    if cert["schema_version"] != SCHEMA_VERSION:
        raise E8HadError("certificate schema_version mismatch")
    if cert["audit_id"] != "AUDIT-115":
        raise E8HadError("certificate audit_id mismatch")
    if cert["mode"] != "standalone_pure_zq":
        raise E8HadError("certificate mode mismatch")
    if cert["novelty_class"] != "N1":
        raise E8HadError("certificate novelty_class mismatch")
    if type(cert["overall_verified"]) is not bool or not cert["overall_verified"]:
        raise E8HadError("certificate overall_verified must be exact bool True")

    cert_sha = require_sha256(cert["certificate_sha256"], "certificate_sha256")
    body = {k: v for k, v in cert.items() if k != "certificate_sha256"}
    if not hmac.compare_digest(cert_sha, canonical_sha256(body)):
        raise E8HadError("certificate_sha256 digest mismatch")

    p1_exp, p2_exp, p3_exp = _fresh_expected_pillars()
    if not _type_strict_eq(cert["pillar1_e8had_isometry_and_hamming"], p1_exp):
        raise E8HadError("pillar1_e8had_isometry_and_hamming payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar2_dp_invariance_and_adjoint_gemv"], p2_exp):
        raise E8HadError("pillar2_dp_invariance_and_adjoint_gemv payload mismatch or hollow")
    if not _type_strict_eq(cert["pillar3_pr24_correspondence_and_roofline"], p3_exp):
        raise E8HadError("pillar3_pr24_correspondence_and_roofline payload mismatch or hollow")

    return {"valid": True, "mode": cert["mode"], "certificate_sha256": cert_sha}


def _reject_json_float_or_constant(token: str) -> Any:
    raise E8HadError(f"forbidden JSON float or non-finite constant: {token}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise E8HadError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def _check_certificate_file(cert_path: Path) -> int:
    try:
        raw_text = cert_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        sys.stderr.write(json.dumps({"valid": False, "error_code": "CERT_IO_ERROR", "error": str(exc)}) + "\n")
        return 2
    try:
        parsed = json.loads(
            raw_text,
            parse_float=_reject_json_float_or_constant,
            parse_constant=_reject_json_float_or_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        verdict = validate_audit_certificate(parsed)
    except E8HadError as exc:
        sys.stderr.write(json.dumps({"valid": False, "error_code": "CERT_VALIDATION_ERROR", "error": str(exc)}) + "\n")
        return 2
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        sys.stderr.write(json.dumps({"valid": False, "error_code": "CERT_JSON_DECODE_ERROR", "error": str(exc)}) + "\n")
        return 2
    sys.stdout.write(json.dumps(verdict, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUDIT-115 pure-Z/Q E_8_Had isometry & adjoint GEMV audit runner.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--standalone", action="store_true", help="Self-contained pure-Z/Q replay (default).")
    mode_group.add_argument("--check", type=Path, default=None, help="Validate a JSON certificate file fail-closed.")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON certificate to stdout.")
    parser.add_argument("--write-cert", type=Path, default=None, help="Write canonical JSON certificate to path.")
    args = parser.parse_args(argv)

    if args.check is not None:
        return _check_certificate_file(args.check)

    result = run_standalone_audit()
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_cert is not None:
        args.write_cert.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        sys.stdout.write(
            f"AUDIT-115 mode={result['mode']} verified={result['overall_verified']} "
            f"sha256={result['certificate_sha256'][:16]}...\n"
        )
    return 0 if result["overall_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
