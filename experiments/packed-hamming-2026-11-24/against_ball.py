#!/usr/bin/env python3
"""Not in the plan: the A_16 file (32 weights, a one-byte scale code) against the A_16 ball of 128 weights with a
16-bit scale (hamming-lattices-real-weights-2026-11-24), both at 4.5 bits a weight.

Written after five of the six rows had landed. It prices the block of 32 with its one-byte scale against the ball of
128, which is what a file at 128 weights a block would win back. Reads this record's results.jsonl and that record's.

Usage, from the root: python3 experiments/packed-hamming-2026-11-24/against_ball.py > experiments/packed-hamming-2026-11-24/against_ball.out
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OTHER = HERE.parent / "hamming-lattices-real-weights-2026-11-24" / "results.jsonl"
NAMES = {"qwen2.5-1.5b-instruct-fp16.gguf": "Qwen2.5-1.5B", "SmolLM2-1.7B-Instruct-f16.gguf": "SmolLM2-1.7B",
         "Llama-3.2-1B-Instruct-f16.gguf": "Llama-3.2-1B"}


def load(path):
    return {(r["model"], r["variant"]): r for r in (json.loads(l) for l in open(path) if l.strip())}


def main():
    mine, ball = load(HERE / "results.jsonl"), load(OTHER)
    print("| Model | Kept columns | A_16 file, 32 weights, one-byte scale | A_16 ball, 128 weights, 16-bit scale | File over ball | Ball lower by (errors) |")
    print("|---|---|---|---|---|---|")
    for f, name in NAMES.items():
        for v, bv, tag in (("PH16:b136s8", "A16BG128:b560s16", "none"), ("PH16:b136s8+K10", "A16BG128:b560s16+K10", "ten")):
            a, b = mine.get((f, v)), ball.get((f, bv))
            if a is None or b is None:
                sa = "not run" if a is None else "%.6f" % a["kl_mean"]
                sb = "not run" if b is None else "%.6f" % b["kl_mean"]
                print(f"| {name} | {tag} | {sa} | {sb} | | |")
                continue
            g = (a["kl_mean"] - b["kl_mean"]) / math.hypot(a["kl_mean_se"], b["kl_mean_se"])
            print(f"| {name} | {tag} | {a['kl_mean']:.6f} ± {a['kl_mean_se']:.6f} | {b['kl_mean']:.6f} ± {b['kl_mean_se']:.6f} | "
                  f"{a['kl_mean'] / b['kl_mean']:.3f} | {g:+.1f} |")


if __name__ == "__main__":
    main()
