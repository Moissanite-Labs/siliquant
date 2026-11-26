#!/usr/bin/env python3
"""
Track S20b Live Gate-8 Evaluation (`AUDIT-119B-LIVE`):
Zone-B-Hardened Fail-Closed Certificate Auditor (`gate8_live_eval_audit.py`).

Independently verifies canonical JSON bytes, symlink/ancestor safety, CR/CRLF
rejection, UTF-8 + unpaired-surrogate rejection, pre-read size cap, strict
Zone-B JSON parsing, constant-time 64-lowercase-ASCII-hex digest comparison
BEFORE `hmac.compare_digest`, and exact standalone recomputation parity against
`gate8_live_eval_oracle.py`.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gate8_live_eval_oracle import (  # noqa: E402
    MAX_JSON_BYTES,
    MAX_JSON_DEPTH,
    Gate8EvalError,
    canonical_json_bytes,
    canonical_sha256,
    constant_time_hex64_equal,
    generate_certificate,
    parse_strict_json_text,
    require_hex64_ascii,
    require_no_symlink_path,
    validate_cli_argv,
)

_EXPECTED_SCHEMA = "aletheia.track_s20b_gate8_live_eval.v1"
_REQUIRED_TOP_KEYS = frozenset({"schema", "payload_sha256", "certificate_sha256", "payload"})


def verify_certificate_dict(cert: Any) -> dict[str, Any]:
    if type(cert) is not dict:
        raise Gate8EvalError("INVALID_CERT_TYPE", "certificate must be strict dict")
    if set(cert.keys()) != _REQUIRED_TOP_KEYS:
        raise Gate8EvalError("INVALID_CERT_KEYS", "certificate top-level keys mismatch")
    if type(cert["schema"]) is not str or cert["schema"] != _EXPECTED_SCHEMA:
        raise Gate8EvalError("INVALID_SCHEMA", "unexpected certificate schema")
    if type(cert["payload"]) is not dict:
        raise Gate8EvalError("INVALID_PAYLOAD_TYPE", "payload must be strict dict")

    claimed_payload_sha = require_hex64_ascii(cert["payload_sha256"], "payload_sha256")
    claimed_cert_sha = require_hex64_ascii(cert["certificate_sha256"], "certificate_sha256")

    actual_payload_sha = canonical_sha256(cert["payload"])
    if not constant_time_hex64_equal(actual_payload_sha, claimed_payload_sha, "payload_sha256"):
        raise Gate8EvalError("PAYLOAD_SHA_MISMATCH", "payload_sha256 mismatch")

    cert_body = {
        "schema": cert["schema"],
        "payload_sha256": claimed_payload_sha,
        "payload": cert["payload"],
    }
    actual_cert_sha = canonical_sha256(cert_body)
    if not constant_time_hex64_equal(actual_cert_sha, claimed_cert_sha, "certificate_sha256"):
        raise Gate8EvalError("CERT_SHA_MISMATCH", "certificate_sha256 mismatch")

    expected = generate_certificate()
    if not constant_time_hex64_equal(actual_payload_sha, expected["payload_sha256"], "recompute_payload_sha256"):
        raise Gate8EvalError("RECOMPUTE_PAYLOAD_MISMATCH", "certificate payload diverges from standalone oracle")

    return {
        "valid": True,
        "schema": _EXPECTED_SCHEMA,
        "payload_sha256": actual_payload_sha,
        "certificate_sha256": actual_cert_sha,
    }


def verify_certificate_file(path: Path) -> dict[str, Any]:
    require_no_symlink_path(path)
    if not path.is_file():
        raise Gate8EvalError("FILE_NOT_FOUND", "certificate file not found")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise Gate8EvalError("FILE_TOO_LARGE", f"certificate file exceeds {MAX_JSON_BYTES} bytes")
    raw = path.read_bytes()
    if b"\r" in raw:
        raise Gate8EvalError("CR_FORBIDDEN", "CR/CRLF line endings forbidden in certificate")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Gate8EvalError("INVALID_UTF8", f"invalid UTF-8: {exc}") from exc
    parsed = parse_strict_json_text(text, require_object=True, max_depth=MAX_JSON_DEPTH)
    canonical_raw = canonical_json_bytes(parsed)
    if raw not in (canonical_raw, canonical_raw + b"\n"):
        raise Gate8EvalError("NON_CANONICAL_BYTES", "certificate file is not canonical JSON")
    res = verify_certificate_dict(parsed)
    res["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S20b Gate-8 live eval auditor (G8-P1 post-hoc)")
    parser.add_argument("--check", type=str, default=None, help="Verify canonical certificate JSON file")
    parser.add_argument("--standalone", action="store_true", help="Audit freshly generated standalone certificate")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON audit verdict")
    try:
        checked_argv = validate_cli_argv(argv)
        args = parser.parse_args(checked_argv)
        if args.check is not None:
            res = verify_certificate_file(Path(args.check))
        else:
            res = verify_certificate_dict(generate_certificate())
    except Gate8EvalError as exc:
        sys.stderr.write(f"FAIL {exc.code}: {exc.message}\n")
        return 1
    if args.json:
        sys.stdout.write(canonical_json_bytes(res).decode("ascii") + "\n")
        return 0
    print(
        f"OK TRACK-S20B-GATE8-LIVE-AUDIT payload_sha256={res['payload_sha256']} "
        f"certificate_sha256={res['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
