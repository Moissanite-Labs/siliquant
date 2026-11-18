#!/usr/bin/env python3
"""Tables and verdicts for the split of bits by kind on Qwen2.5, fitted on the validation text and scored on the test text.

Reads results.jsonl and allocate.out of this folder. Scores the predictions H1 to H5 of PREREG.md.

Usage, from the root of a checkout:
  python3 experiments/bit-split-held-out-qwen25-2026-11-18/summary.py > experiments/bit-split-held-out-qwen25-2026-11-18/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def load(text):
    rows = {}
    path = HERE / "results.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["text"] == text:
                    rows[r["variant"]] = r
    return rows


def arms():
    out = {}
    path = HERE / "allocate.out"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("JSON "):
                r = json.loads(line[5:])
                out[(r["budget"], r["role"])] = r
    return out


def gap(a, b):
    """(KL of b less KL of a, in errors): positive when a is lower."""
    d = b["kl_mean"] - a["kl_mean"]
    return d, d / math.hypot(a["kl_mean_se"], b["kl_mean_se"])


def main():
    valid, test, split = load("valid"), load("test"), arms()
    print("Bits split by kind of matrix on Qwen2.5-1.5B: fitted on the wikitext-2 validation text, scored on the test text. 10,200 positions of each")
    print()
    print("Part A, validation text: one kind rounded at a time by the rule X32, the rest as in the reference")
    print("kind | weights | share of the block weights, percent | mean KL at 8 levels | at 16 | at 32 | standard error at 16")
    total = sum(valid[f"X32:16@{k}"]["weights_changed"] for k in KINDS if f"X32:16@{k}" in valid)
    for k in KINDS:
        r = [valid.get(f"X32:{q}@{k}") for q in (8, 16, 32)]
        if all(r):
            print(f"{k} | {r[1]['weights_changed']} | {100 * r[1]['weights_changed'] / total:.1f} | {r[0]['kl_mean']:.6f} | {r[1]['kl_mean']:.6f} | {r[2]['kl_mean']:.6f} | {r[1]['kl_mean_se']:.6f}")
        else:
            print(f"{k} | not scored")
    if "X32:8" in valid and all(f"X32:8@{k}" in valid for k in KINDS):
        s = sum(valid[f"X32:8@{k}"]["kl_mean"] for k in KINDS)
        print(f"the seven kinds at 8 levels add up to {s:.6f}; all kinds at once score {valid['X32:8']['kl_mean']:.6f}: the sum is {s / valid['X32:8']['kl_mean']:.3f} of it")
    print()
    print("The split from allocate.py")
    print("budget | arm | levels by kind | mean bits | KL it predicts on the validation text")
    for (budget, role), r in sorted(split.items(), key=lambda x: (-x[0][0], x[0][1])):
        print(f"{budget} | {role} | " + " ".join(f"{k}={r['levels'][k]}" for k in KINDS) + f" | {r['mean_bits']:.4f} | {r['predicted_kl']:.6f}")
    verdict = {}
    print()
    print("Part B")
    print("text | budget | arm | bits per weight | mean KL | its standard error | first choice kept, percent | gain over equal bits, percent | lower than equal bits by, in errors")
    gains = {}
    for name, rows in (("validation", valid), ("test", test)):
        for budget, equal in ((4.5, "X32:16"), (3.5, "X32:8")):
            e = rows.get(equal)
            if e:
                print(f"{name} | {budget} | equal bits, {equal} | {e['bits_per_weight_changed']:.4f} | {e['kl_mean']:.6f} | {e['kl_mean_se']:.6f} | {e['same_top']:.2f} | - | -")
            for role in ("best", "reversed"):
                a = split.get((budget, role))
                r = rows.get(a["variant"]) if a else None
                if r and e:
                    d, err = gap(r, e)
                    gains[(name, budget, role)] = (1 - r["kl_mean"] / e["kl_mean"], err)
                    print(f"{name} | {budget} | {role} | {r['bits_per_weight_changed']:.4f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | "
                          f"{100 * gains[(name, budget, role)][0]:.1f} | {err:+.1f}")
    if ("test", 4.5, "best") in gains:
        verdict["H1"] = gains[("test", 4.5, "best")][1] >= 2
        if ("validation", 4.5, "best") in gains:
            verdict["H2"] = gains[("test", 4.5, "best")][0] >= 0.5 * gains[("validation", 4.5, "best")][0]
            print(f"gain at 4.5 bits: {100 * gains[('validation', 4.5, 'best')][0]:.1f} percent on the text of the fit, {100 * gains[('test', 4.5, 'best')][0]:.1f} percent on the other")
    if ("test", 3.5, "best") in gains:
        verdict["H3"] = gains[("test", 3.5, "best")][1] >= 2
        if ("validation", 3.5, "best") in gains:
            print(f"gain at 3.5 bits: {100 * gains[('validation', 3.5, 'best')][0]:.1f} percent on the text of the fit, {100 * gains[('test', 3.5, 'best')][0]:.1f} percent on the other")
    b, rv = split.get((4.5, "best")), split.get((4.5, "reversed"))
    if b and rv and b["variant"] in test and rv["variant"] in test:
        d, err = gap(test[b["variant"]], test[rv["variant"]])
        print(f"test text, 4.5 bits: the best arm is lower than the reversed arm by {d:+.6f}, {err:+.1f} errors")
        verdict["H4"] = err >= 2
    if b:
        verdict["H5"] = min(b["levels"][k] for k in KINDS[:4]) >= max(b["levels"][k] for k in KINDS[4:])
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    for k in ("H1", "H2", "H3", "H4", "H5"):
        print(f"{k} | {'not scored' if k not in verdict else ('yes' if verdict[k] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of 5")


if __name__ == "__main__":
    main()
