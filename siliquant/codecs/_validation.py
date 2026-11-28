"""Validation primitives, strict JSON I/O, SHA-256 helpers, and lattice constants."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

# Extended Hamming [16, 11, 4] systematic information positions (parity bits sit at {0, 1, 2, 4, 8}).
A16_INFO_POSITIONS: tuple[int, ...] = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)

BALL_BOUND_45_G128 = 5213.0
BALL_BOUNDS: dict[str, dict[int, float]] = {
    "4.5": {128: 5213.0, 32: 1207.0},
    "3.0": {128: 651.0, 32: 151.0},
}
GOLDEN_RATIO_CONJ = (math.sqrt(5.0) - 1.0) / 2.0

# Reference 1.54B edge-cliff parameters (Qwen2.5-1.5B 28L: L0 + L27, 14 matrices)
N_LAYERS_15B = 28
HIDDEN_SIZE_15B = 1536
INTERMEDIATE_SIZE_15B = 8960
N_Q_HEADS_15B = 12
N_KV_HEADS_15B = 2
HEAD_DIM_15B = 128
ALL_15B_TOTAL_PARAMS = 1_543_714_304
CLIFF_L0_L27_WEIGHTS = 93_585_408
CLIFF_L0_L27_A16_SHIPPED_BYTES = 61_091_884


class SiliquantError(ValueError):
    """Explicit fail-closed validation error (active even under python3 -O)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"{self.code}: {self.message}")


def require(cond: bool, code: str, message: str) -> None:
    """Fail closed with SiliquantError when invariant `cond` is false."""
    if not cond:
        raise SiliquantError(code, message)


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize `obj` deterministically with sorted keys and finite-number check."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return lowercase SHA-256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path_obj: Path, *, allow_symlink: bool = False) -> str:
    """Compute SHA-256 hex digest of a regular file without following symlinks by default."""
    if not allow_symlink:
        require(not path_obj.is_symlink(), "SYMLINK_FILE", f"symlink forbidden: {path_obj}")
    require(path_obj.is_file(), "MISSING_FILE", f"missing file: {path_obj}")
    h = hashlib.sha256()
    with path_obj.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_bad_json_constant(token: str) -> Any:
    raise SiliquantError("JSON_BAD_CONSTANT", f"non-standard JSON constant forbidden: {token!r}")


def _reject_dup_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise SiliquantError("JSON_DUP_KEY", f"duplicate JSON key: {k!r}")
        out[k] = v
    return out


def _check_finite_json(obj: Any) -> None:
    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, float):
            require(math.isfinite(cur), "JSON_NONFINITE", f"non-finite float: {cur!r}")
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def load_json_strict(path_obj: Path, label: str = "json", *, allow_symlink: bool = False) -> Any:
    """Load JSON strictly: no symlinks, no CRLF, no duplicate keys, no NaN/Infinity."""
    if not allow_symlink:
        require(not path_obj.is_symlink(), f"{label}_SYMLINK", f"symlink forbidden: {path_obj}")
    require(path_obj.is_file(), f"{label}_MISSING", f"missing file: {path_obj}")
    raw = path_obj.read_bytes()
    require(b"\r\n" not in raw, f"{label}_CRLF", f"CRLF line endings forbidden: {path_obj}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SiliquantError(f"{label}_UTF8", f"invalid UTF-8 in {path_obj}: {exc}") from exc
    require(len(text.strip()) > 0, f"{label}_EMPTY", f"empty JSON file: {path_obj}")
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_bad_json_constant,
            object_pairs_hook=_reject_dup_keys,
        )
    except SiliquantError:
        raise
    except Exception as exc:
        raise SiliquantError(f"{label}_PARSE", f"JSON parse error in {path_obj}: {exc}") from exc
    _check_finite_json(parsed)
    return parsed


def write_json_strict(path_obj: Path, obj: Any) -> None:
    """Atomically write canonical LF-only UTF-8 JSON with finite numbers."""
    _check_finite_json(obj)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_name(f".{path_obj.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    tmp.replace(path_obj)
