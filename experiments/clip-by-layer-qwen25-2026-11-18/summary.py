#!/usr/bin/env python3
"""Tables and verdicts for the clip of ffn_down by layer on Qwen2.5, from results.jsonl and weights_only.jsonl.

Scores the predictions L1 to L6 of PREREG.md by the rules written there.

Usage, from the root of a checkout:
  python3 experiments/clip-by-layer-qwen25-2026-11-18/summary.py > experiments/clip-by-layer-qwen25-2026-11-18/results.out
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = 28
TOP5 = (1, 2, 26, 27, 19)          # the five layers with the largest weight error of the clip, named in the plan


def load(path, text):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["text"] == text:
                    rows[r["variant"]] = r
    return rows


def ranks(xs):
    """Ranks from 1, ties given their mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(xs, ys):
    a, b = ranks(xs), ranks(ys)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den


def main():
    scored = load(HERE / "results.jsonl", "test")
    weights = load(HERE / "weights_only.jsonl", "weights")
    single = {l: scored.get(f"clip:5@ffn_down#{l}") for l in range(LAYERS)}
    have = [l for l in range(LAYERS) if single[l]]
    every = scored.get("clip:5@ffn_down")
    print("The clip of ffn_down at 5 standard deviations of a row, one layer of Qwen2.5 at a time: wikitext-2 test text, 10,200 positions")
    print(f"single layers scored: {len(have)} of {LAYERS}. All layers at once: {'scored' if every else 'not scored'}")
    for l in have:
        w = weights[f"clip:5@ffn_down#{l}"]
        if abs(w["weight_err_db_mean"] - single[l]["weight_err_db_mean"]) > 1e-9 or w["clipped_entries"] != single[l]["clipped_entries"]:
            raise SystemExit(f"FAIL: layer {l}: the scored matrix is not the one of weights_only.jsonl")
    total = sum(single[l]["kl_mean"] for l in have)
    print()
    print("layer | weight error of the clip, dB | entries clipped | rows with a clipped entry | mean KL | its standard error | KL in its errors | share of the sum, percent | first choice kept, percent")
    for l in have:
        r = single[l]
        print(f"{l} | {r['weight_err_db_mean']:.2f} | {r['clipped_entries']} | {r['rows_clipped']} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | "
              f"{r['kl_mean'] / r['kl_mean_se'] if r['kl_mean_se'] else 0:.1f} | {100 * r['kl_mean'] / total:.1f} | {r['same_top']:.2f}")
    verdict = {}
    if len(have) == LAYERS:
        top = max(have, key=lambda l: single[l]["kl_mean"])
        order = sorted(have, key=lambda l: -single[l]["kl_mean"])
        share5 = sum(single[l]["kl_mean"] for l in TOP5) / total
        rho = spearman([single[l]["weight_err_db_mean"] for l in have], [single[l]["kl_mean"] for l in have])
        print()
        print(f"sum of the {LAYERS} single-layer KLs: {total:.6f}")
        print(f"largest: layer {top}, {single[top]['kl_mean']:.6f}, {100 * single[top]['kl_mean'] / total:.1f} percent of the sum")
        print("the four largest: " + ", ".join(f"layer {l} {single[l]['kl_mean']:.6f}" for l in order[:4])
              + f", together {100 * sum(single[l]['kl_mean'] for l in order[:4]) / total:.1f} percent")
        print(f"layers 1, 2, 26, 27 and 19, the five with the largest weight error: {100 * share5:.1f} percent of the sum")
        print(f"layers under 5 of their own errors: {sum(1 for l in have if single[l]['kl_mean'] < 5 * single[l]['kl_mean_se'])}, "
              f"holding {100 * sum(single[l]['kl_mean'] for l in have if single[l]['kl_mean'] < 5 * single[l]['kl_mean_se']) / total:.1f} percent of the sum")
        print(f"Spearman's rank correlation between weight error and KL over the {LAYERS} layers: {rho:.3f}")
        verdict["L2"] = single[top]["kl_mean"] > 0.5 * total
        verdict["L3"] = top == 1
        verdict["L4"] = share5 >= 0.9
        verdict["L6"] = rho >= 0.5
    if every:
        print()
        print(f"all {LAYERS} layers at once: weight error {every['weight_err_db_mean']:.2f} dB, mean KL {every['kl_mean']:.6f}, standard error {every['kl_mean_se']:.6f}, "
              f"first choice kept {every['same_top']:.2f} percent")
        verdict["L1"] = every["kl_mean"] > 0.05
        if len(have) == LAYERS:
            print(f"the sum over the KL of all at once: {total / every['kl_mean']:.3f}")
            verdict["L5"] = 0.7 <= total / every["kl_mean"] <= 1.4
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    for k in ("L1", "L2", "L3", "L4", "L5", "L6"):
        print(f"{k} | {'not scored' if k not in verdict else ('yes' if verdict[k] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of 6")


if __name__ == "__main__":
    main()
