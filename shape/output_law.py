#!/usr/bin/env python3
"""Scores the predictions of notes/issue10-output-predictions.md against a results.jsonl of a model-output run. Rounds nothing.

For every variant it prints the weight error the closed form of shape/ballformula.py gives, where
there is one, the measured weight error, the mean KL that one anchor arm and the weight error
give, and the measured mean KL. Then the verdicts of the six predictions.

The anchor is X32:16 of the same file: its mean KL and its mean weight error.

Usage, from the root of a checkout:
  python3 shape/output_law.py experiments/model-output-qwen25-2026-11-17/results.jsonl
  python3 shape/output_law.py experiments/model-output-smollm2-2026-11-17/results.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ballformula as B  # noqa: E402

BALL_ARMS = ("EBG128:b560s16", "EB32@32:16", "ZBG128:b560s16", "EBG128:b504s8")      # the four of the third model's plan
REFERENCE = "EBG128:b560s16"
PER_ROW = ("XR:16", "VE8ZR:16")
MARK_ERROR, MARK_KL, MARK_RATIO, MARK_OUTSIDE = 0.05, 0.5, 0.4, 1.5


def db(x):
    return 10.0 * math.log10(x)


def main():
    path = Path(sys.argv[1])
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["variant"]] = r
    out = []
    P = out.append
    P(f"File: {path}. Variants: {len(rows)}.")
    if "X32:16" not in rows:
        P("No anchor: X32:16 is not scored.")
        print("\n".join(out))
        return
    a = rows["X32:16"]
    a_kl, a_err = a["kl_mean"], a["weight_err_db_mean"]
    P(f"Anchor X32:16: mean KL {a_kl:.5f} at a mean weight error of {a_err:.3f} dB. KL per unit of weight error: {a_kl / 10 ** (a_err / 10):.2f}.")
    P("")
    P("variant | bits | weight error by the closed form, dB | measured, dB | measured minus formula | KL by the anchor | measured KL | its standard error | measured over predicted, dB")
    off = {}
    for name, r in rows.items():
        if r.get("weight_err_db_mean") is None or not math.isfinite(r["weight_err_db_mean"]):
            continue
        formula = B.predicted_error_db(name)
        measured = r["weight_err_db_mean"]
        err = formula if formula is not None else measured
        kl = B.predicted_kl(err, a_kl, a_err)
        off[name] = db(r["kl_mean"] / kl)
        shown = "-" if formula is None else f"{formula:.3f}"
        gap = "-" if formula is None else f"{measured - formula:+.3f}"
        P(f"{name} | {r['bits_per_weight']:.4f} | {shown} | {measured:.3f} | {gap} | {kl:.4f} | {r['kl_mean']:.4f} | {r['kl_mean_se']:.4f} | {off[name]:+.2f}")
    P("")
    P("The predictions")

    def verdict(tag, have, holds, text):
        P(f"   {tag} | {'not scored yet' if not have else ('holds' if holds else 'FAILS')} | {text}")

    have = [v for v in BALL_ARMS if v in rows]
    diffs = {v: rows[v]["weight_err_db_mean"] - B.predicted_error_db(v) for v in have}
    verdict("O1", len(have) == len(BALL_ARMS), all(abs(d) <= MARK_ERROR for d in diffs.values()),
            f"closed form against measured weight error, mark {MARK_ERROR} dB: " + ", ".join(f"{v} {d:+.3f}" for v, d in diffs.items()))
    verdict("O2", len(have) == len(BALL_ARMS), all(abs(off[v]) <= MARK_KL for v in have),
            f"KL by the anchor and the closed form, mark {MARK_KL} dB: " + ", ".join(f"{v} {off[v]:+.2f}" for v in have))
    if REFERENCE in rows:
        ratios = {v: (db(rows[v]["kl_mean"] / rows[REFERENCE]["kl_mean"]), B.predicted_error_db(v) - B.predicted_error_db(REFERENCE)) for v in have if v != REFERENCE}
    else:
        ratios = {}
    verdict("O3", len(ratios) == len(BALL_ARMS) - 1, all(abs(m - p) <= MARK_RATIO for m, p in ratios.values()),
            f"KL over that of {REFERENCE} in dB, measured against the closed form, mark {MARK_RATIO} dB: " + ", ".join(f"{v} {m:+.2f} against {p:+.2f}" for v, (m, p) in ratios.items()))
    verdict("O4", "VE8Z32:16" in rows, "VE8Z32:16" in off and abs(off["VE8Z32:16"]) <= MARK_KL,
            f"VE8Z32:16 by the anchor and its measured weight error, mark {MARK_KL} dB: " + (f"{off['VE8Z32:16']:+.2f}" if "VE8Z32:16" in off else "-"))
    rows_have = [v for v in PER_ROW if v in off]
    verdict("O5", len(rows_have) == len(PER_ROW), all(off[v] >= MARK_OUTSIDE for v in rows_have),
            f"one scale per row is outside the law: KL at least {MARK_OUTSIDE} dB above the anchor's: " + ", ".join(f"{v} {off[v]:+.2f}" for v in rows_have))
    scored = {v: r["kl_mean"] for v, r in rows.items() if v not in ("unchanged", "identity")}
    lowest = min(scored, key=scored.get)
    verdict("O6", REFERENCE in rows and len(scored) >= 12, lowest == REFERENCE,
            f"the arm with the smallest predicted error at 4.5 bits or fewer, {REFERENCE}, has the lowest mean KL of all {len(scored)} scored variants: the lowest is {lowest} at {scored[lowest]:.4f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
