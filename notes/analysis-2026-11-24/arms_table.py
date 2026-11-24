#!/usr/bin/env python3
"""Exploratory, from committed rows only: what the arms at 4.5 bits have in common with their output.

1. A table per model: bits, pooled weight error, the error and the share stored as zero of the
   weights under 0.05 and 0.1 of their matrix's root mean square, and the mean KL.
2. A fit of KL = a * D + b * E05 per model over the ball arms, with D the pooled weight error
   and E05 the error of the weights under 0.05 over their own energy, both as powers.
3. Every comparison of a ball arm or packed file with llama.cpp's Q4_K_M, against the committed
   Q4_K_M row and against Q4_K_M@16 (the same weights multiplied without the 8-bit activations).

Reads the rows through the summary of hamming-lattices-real-weights-2026-11-24, so it sees the
same rows as that record. Run from the root: python3 notes/analysis-2026-11-24/arms_table.py
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("hl_summary", ROOT / "experiments/hamming-lattices-real-weights-2026-11-24/summary.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

ARMS = [("ZBG128:b560s16", "integer levels"), ("EBG128:b560s16", "E8"), ("HBG128:b560s16", "E8 turned"),
        ("A16BG128:b560s16", "A16"), ("A32BG128:b560s16", "A32"), ("NF4B64", "NF4"), ("Q4_K_M@16", "Q4_K_M@16")]
KEPT = [("ZBG128:b560s16+K10", "integer levels, K10"), ("EBG128:b560s16+K10", "E8, K10"),
        ("HBG128:b560s16+K10", "E8 turned, K10"), ("A16BG128:b560s16+K10", "A16, K10")]


def small(r, d, what):
    try:
        return r["small_weights"][d]["all"][what]
    except (KeyError, TypeError):
        return None


def packed():
    out = {}
    for line in open(ROOT / "experiments/packed-kept-columns-2026-11-19/results.jsonl"):
        if line.strip():
            r = json.loads(line)
            out[(r["model"], r["variant"])] = r
    return out


def main():
    data = S.load()
    pk = packed()
    print("## 1. Arms at 4.5 bits: weights and output\n")
    print("| Model | Arm | Bits | Error (dB) | Under 0.05: error over size (dB) | zero (%) | Under 0.1: error over size (dB) | Mean KL |")
    print("|---|---|---|---|---|---|---|---|")
    fits = {}
    for name in S.NAMES:
        d = data[name]
        for v, label in ARMS:
            w = d["weights"].get(v)
            r = S.get(data, name, v)
            if w is None:
                continue
            k = f"{r['kl_mean']:.6f} ± {r['kl_mean_se']:.6f}" if r else "not scored"
            print(f"| {name} | {label} | {w['bits_per_weight']:.4f} | {w['weight_err_db_pooled']:.3f} | {small(w, '0.05', 'err_over_size_db'):.2f} | "
                  f"{small(w, '0.05', 'stored_zero_percent'):.1f} | {small(w, '0.1', 'err_over_size_db'):.2f} | {k} |")
            if r and v.endswith("b560s16"):
                fits.setdefault(name, []).append((label, w["weight_err_db_pooled"], small(w, "0.05", "err_over_size_db"), r["kl_mean"], r["kl_mean_se"]))
    print("\n## 2. KL = a * 10^(D/10) + b * 10^(E05/10), weighted by 1/SE, over the ball arms at 4.5 bits\n")
    for name, pts in fits.items():
        if len(pts) < 3:
            print(f"- {name}: fewer than three scored ball arms")
            continue
        X = np.array([[10 ** (p[1] / 10), 10 ** (p[2] / 10)] for p in pts])
        y = np.array([p[3] for p in pts])
        w = 1 / np.array([p[4] for p in pts])
        coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        pred = X @ coef
        res = ", ".join(f"{p[0]} {(p[3] - pred[i]) / p[4]:+.1f}" for i, p in enumerate(pts))
        print(f"- {name}: a = {coef[0]:.3f}, b = {coef[1]:.5f}; residuals in errors: {res}")
    print("\n## 3. Gaps to Q4_K_M in errors (positive: the arm is lower), against the committed row and against Q4_K_M@16\n")
    print("| Model | Arm | Mean KL | vs Q4_K_M | vs Q4_K_M@16 |")
    print("|---|---|---|---|---|")
    for name in S.NAMES:
        q, q16 = S.get(data, name, "Q4_K_M"), S.get(data, name, "Q4_K_M@16")
        rows = [(label, S.get(data, name, v)) for v, label in ARMS[:5] + KEPT]
        for key, label in (("PZ32:b136s8", "packed file, integer levels"), ("PZ32:b136s8+K10", "packed file, integer levels, K10")):
            r = pk.get((S.FILE[name], key))
            rows.append((label, r))
        for label, r in rows:
            if r is None or q is None:
                continue
            g1 = S.EHS.gap(r, q)
            g2 = S.EHS.gap(r, q16) if q16 else None
            print(f"| {name} | {label} | {r['kl_mean']:.6f} | {g1:+.1f} | {(g2 if g2 is not None else float('nan')):+.1f} |")


if __name__ == "__main__":
    main()
