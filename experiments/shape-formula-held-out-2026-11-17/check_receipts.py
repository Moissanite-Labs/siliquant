#!/usr/bin/env python3
"""Replay the receipts of pilot.json or results.json from the weights. Exits with 1 if any does not hold.

A receipt names a matrix by the SHA-256 of its bytes and gives, for each group size, the whole
numbers from which the region part of the cube, the ball and the cross-polytope follow
(shape/shapelib.py, exact_terms). This program loads every matrix again, computes the digest and
the whole numbers again, and compares them digit for digit.

It then changes one digit of one receipt and one entry of one matrix, and checks that both
changes are caught. A checker that cannot fail checks nothing (PR #5, "defects to avoid copying").

Usage, from the root of a checkout, with NumPy:
  python3 experiments/shape-formula-held-out-2026-11-17/check_receipts.py results --whisper weights.npz --qwen model.safetensors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
S = run.S


def replay(stored, probe):
    """The list of what does not match between a matrix and its receipt. Empty when the receipt holds."""
    bad = []
    if S.matrix_digest(stored) != probe["digest"]:
        bad.append("digest")
    for g, want in probe["receipt"].items():
        got = {k: str(v) for k, v in S.exact_terms(stored, g if g == "R" else int(g)).items()}
        bad += [f"group {g}: {k}" for k in want if got.get(k) != want[k]]
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=("pilot", "results"))
    ap.add_argument("--whisper")
    ap.add_argument("--qwen")
    args = ap.parse_args()
    data = json.loads((HERE / f"{args.name}.json").read_text())
    run._files["whisper"], run._files["qwen"] = args.whisper, args.qwen
    bases = {**{k: v for k, v in run.HELD_SEEDS.items()}} if data["meta"]["set"] == "held" else dict(run.SEEN_SEEDS)
    probes = [r for r in data["rows"] if r["kind"] == "probe"]
    failed, skipped, first = [], [], None
    for p in probes:
        kind = p["matrix"].split("/", 1)[0]
        if (kind == "whisper" and not args.whisper) or (kind == "qwen" and not args.qwen):
            skipped.append(p["matrix"])
            continue
        stored = run.load({"matrix": p["matrix"], "family": p["family"], "base": bases.get(p["family"])})
        bad = replay(stored, p)
        if bad:
            failed.append((p["matrix"], bad))
        if first is None:
            first = (stored, p)
    # the two controls: a changed receipt and a changed matrix must both be caught
    controls_ok = False
    if first is not None:
        stored, p = first
        forged = json.loads(json.dumps(p))
        digits = forged["receipt"]["32"]["top_ball"]
        forged["receipt"]["32"]["top_ball"] = digits[:-1] + ("1" if digits[-1] != "1" else "2")
        moved = np.array(stored, copy=True)
        moved[0, 0] = moved[0, 0] + moved.dtype.type(1.0)
        controls_ok = bool(replay(stored, forged)) and bool(replay(moved, p))
    print(f"receipts replayed: {len(probes) - len(skipped)} of {len(probes)}; failed: {len(failed)}; skipped for want of a weights file: {len(skipped)}")
    for name, bad in failed:
        print(f"   FAIL {name}: {', '.join(bad[:6])}")
    print(f"controls (a receipt with one digit changed, a matrix with one entry changed): {'both caught' if controls_ok else 'NOT caught'}")
    sys.exit(0 if (not failed and controls_ok) else 1)


if __name__ == "__main__":
    main()
