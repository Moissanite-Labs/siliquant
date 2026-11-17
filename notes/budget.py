#!/usr/bin/env python3
"""The arithmetic that notes/persistence-and-flow-check.md quotes. No model is read.

Inputs are the committed results of the record model-output-gemma-e4b-2026-11-15
and three formulas. Python 3 only.

    python3 notes/budget.py > notes/budget.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "model-output-gemma-e4b-2026-11-15" / "results.jsonl"
DB_PER_BIT = 20 * math.log10(2)  # 6.0206 dB of squared error for one bit, at high rates

rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
by = {r["variant"]: r for r in rows if r.get("kl_mean") is not None}

print("1. Parts of the bit budget, in bits per weight")
sphere = 10 * math.log10(math.pi * math.e / 6)
print(f"   any cell shape, any dimension, high rates: {sphere:.3f} dB = {sphere / DB_PER_BIT:.3f} bit")
print(f"   the E8 cell: 0.65 dB = {0.65 / DB_PER_BIT:.3f} bit (bounds/bounds.out)")
for bits, group in [(16, 32), (8, 32), (6, 32), (16, 2560), (16, 10240)]:
    print(f"   a scale of {bits:2d} bits for every {group:5d} weights: {bits / group:.4f} bit")

print("\n2. The sphere bound in 3 dimensions")
n = 3
g = math.gamma(n / 2 + 1) ** (2 / n) / ((n + 2) * math.pi)
print(f"   second moment of a ball {g:.6f}, of a cube {1 / 12:.6f}: at most "
      f"{10 * math.log10((1 / 12) / g):.3f} dB over the cube")

print("\n3. Arm S at 4 bits, one kind of matrix at a time (results.jsonl)")
# Weights of each kind, in millions, from the record's README.
weights = {"ffn_down": 1101, "attn_output": 257, "ffn_up": 1101, "ffn_gate": 1101,
           "attn_q": 257, "attn_v": 64, "attn_k": 64}
kl = {k: by[f"S:16@{k}"]["kl_mean"] for k in weights}
per = {k: kl[k] / weights[k] for k in weights}
for k in sorted(per, key=per.get, reverse=True):
    print(f"   {k:12}  {weights[k]:5d} M weights  KL {kl[k]:.4f}  KL per million weights {per[k]:.3e}")
print(f"   largest over smallest KL per weight: {max(per.values()) / min(per.values()):.1f}")
print(f"   sum over kinds {sum(kl.values()):.3f}, all kinds at once {by['S:16']['kl_mean']:.3f}")

print("\n4. Equal gain per bit, if one more bit divides a kind's KL by four and the parts add")
total = sum(weights.values())
mean_log = sum(weights[k] * math.log2(per[k]) for k in weights) / total
after = 0.0
for k in sorted(per, key=per.get, reverse=True):
    b = 4 + 0.5 * (math.log2(per[k]) - mean_log)
    part = kl[k] * 4 ** (-(b - 4))
    after += part
    print(f"   {k:12}  {b:.2f} bits  modelled KL {part:.3f}")
print(f"   modelled sum: {sum(kl.values()):.3f} with 4 bits for every kind, {after:.3f} after the split")
print("   The table of part 3 is from the arm that breaks the model, and its parts do not add.")
print("   This is the arithmetic of a prediction. It is not a result.")

print("\n5. Keeping the large entries of ffn_down apart")
entries = 188362  # entries beyond 5 standard deviations of their row, large_entries.out
for row_len in (2560, 10240):
    cost = entries * (16 + math.ceil(math.log2(row_len))) / (weights["ffn_down"] * 1e6)
    print(f"   16 bits for the value and {math.ceil(math.log2(row_len))} for the place in a row of {row_len}:"
          f" {cost:.4f} bit per weight of ffn_down")

print("\n6. Scored pairs the note quotes (results.jsonl)")
for v in ["S:32", "G+S:32", "GF+S:32", "R+S:32", "XR:32", "XR:32~0.7", "X32:16", "VE832:16",
          "clip:3.45", "flip:3.45", "clip:5", "clip:5@ffn_down", "gauge:1"]:
    print(f"   {v:16} KL {by[v]['kl_mean']:.6f}")
