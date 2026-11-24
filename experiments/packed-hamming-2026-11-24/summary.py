#!/usr/bin/env python3
"""Prints the results of packed-hamming-2026-11-24 and scores the predictions of PREREG.md.

Written after the two rows of Qwen2.5 had landed (the plan did not commit a summary program); the rules it
applies are the plan's, word for word. Reads this record's results.jsonl, the committed integer files
(packed-ball-format-2026-11-18 and packed-kept-columns-2026-11-19) and Q4_K_M@16 (hamming-lattices-real-
weights-2026-11-24, baselines.jsonl).

Usage, from the root: python3 experiments/packed-hamming-2026-11-24/summary.py > experiments/packed-hamming-2026-11-24/results.out
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
MODELS = [("Qwen2.5-1.5B", "qwen2.5-1.5b-instruct-fp16.gguf"), ("SmolLM2-1.7B", "SmolLM2-1.7B-Instruct-f16.gguf"),
          ("Llama-3.2-1B", "Llama-3.2-1B-Instruct-f16.gguf")]


def rows(path):
    out = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r["model"], r["variant"])] = r
    return out


def gap(a, b):
    """How far the KL of row a is under that of row b, in errors."""
    return (b["kl_mean"] - a["kl_mean"]) / math.hypot(a["kl_mean_se"], b["kl_mean_se"])


def yes(ok):
    return "not yet decided" if ok is None else ("yes" if ok else "**no**")


def main():
    mine = rows(HERE / "results.jsonl")
    z = rows(EXP / "packed-ball-format-2026-11-18" / "results.jsonl")
    z.update(rows(EXP / "packed-kept-columns-2026-11-19" / "results.jsonl"))
    q = rows(EXP / "hamming-lattices-real-weights-2026-11-24" / "baselines.jsonl")

    print("## The files\n")
    print("| Model | File | Kept columns | Weight error (dB) | Stored bits a weight | Bytes | Read back exact | Decode, million weights a second, one core | Pack (s) | Mean KL |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, f in MODELS:
        for v, zv in (("PH16:b136s8", "PZ32:b136s8"), ("PH16:b136s8+K10", "PZ32:b136s8+K10")):
            for label, r in (("A_16", mine.get((f, v))), ("Z^32", z.get((f, zv)))):
                if r is None:
                    print(f"| {name} | {label} | {'ten' if v.endswith('K10') else 'none'} | not run | | | | | | |")
                    continue
                print(f"| {name} | {label} | {'ten' if v.endswith('K10') else 'none'} | {r['weight_err_db_pooled']:.3f} | {r['stored_bits_per_weight']:.5f} | "
                      f"{r['file_bytes']} | {r['read_back_exact']} | {r['decode_million_weights_per_second']:.2f} | {r['pack_seconds']:.0f} | "
                      f"{r['kl_mean']:.6f} ± {r['kl_mean_se']:.6f} |")

    print("\n## Gaps, in errors (positive: the first is lower), and ratios\n")
    F1, F2, F3, F4, F5, F6, C = [], [], [], [], [], [], []
    for name, f in MODELS:
        q16 = q.get((f, "Q4_K_M@16"))
        for v, zv, tag in (("PH16:b136s8", "PZ32:b136s8", "no columns kept"), ("PH16:b136s8+K10", "PZ32:b136s8+K10", "ten columns kept")):
            a, b = mine.get((f, v)), z.get((f, zv))
            if a is None or b is None:
                continue
            g = gap(a, b)
            ddb = b["weight_err_db_pooled"] - a["weight_err_db_pooled"]
            ratio = a["kl_mean"] / b["kl_mean"]
            print(f"- {name}, {tag}: A_16 file against Z^32 file {g:+.1f}; KL ratio {ratio:.3f}; weight error {ddb:.3f} dB lower")
            F1.append((ddb, 0.70 <= ddb <= 0.90))
            (F2 if tag == "no columns kept" else F3).append((g, g >= 2))
            if tag == "no columns kept":
                F4.append((ratio, 0.70 <= ratio <= 0.92))
            else:
                if q16 is not None:
                    ga, gb = gap(a, q16), gap(b, q16)
                    print(f"- {name}, ten columns kept: A_16 file against Q4_K_M@16 {ga:+.1f}; Z^32 file against Q4_K_M@16 {gb:+.1f}")
                    F5.append((ga, gb, ga >= 2 and ga > gb))
            F6.append((a["decode_million_weights_per_second"], a["decode_million_weights_per_second"] >= 5))
            C.append((a["stored_bits_per_weight"], a["read_back_exact"] and 4.5 <= a["stored_bits_per_weight"] <= 4.5007))

    def verdict(xs, n):
        if any(not ok for *_, ok in xs):
            return False
        return True if len(xs) == n else None

    print("\n## The predictions\n")
    print("| | Prediction | Result | Holds |")
    print("|---|---|---|---|")
    print(f"| F1 | weight error of each A_16 file under the Z^32 file's by 0.70 to 0.90 dB | {', '.join(f'{d:.3f}' for d, _ in F1) or '-'} | {yes(verdict(F1, 6))} |")
    print(f"| F2 | A_16 file under Z^32 file by 2 errors, no columns kept, three models | {', '.join(f'{g:+.1f}' for g, _ in F2) or '-'} | {yes(verdict(F2, 3))} |")
    print(f"| F3 | A_16 file under Z^32 file by 2 errors, ten columns kept, three models | {', '.join(f'{g:+.1f}' for g, _ in F3) or '-'} | {yes(verdict(F3, 3))} |")
    print(f"| F4 | KL ratio of A_16 file to Z^32 file, no columns kept, 0.70 to 0.92 | {', '.join(f'{r:.3f}' for r, _ in F4) or '-'} | {yes(verdict(F4, 3))} |")
    print(f"| F5 | A_16 file with columns kept under Q4_K_M@16 by 2, and by more than the Z^32 file | {', '.join(f'{ga:+.1f} against {gb:+.1f}' for ga, gb, _ in F5) or '-'} | {yes(verdict(F5, 3))} |")
    print(f"| F6 | decode at 5 million weights a second or more, one core | {', '.join(f'{d:.2f}' for d, _ in F6) or '-'} | {yes(verdict(F6, 6))} |")
    print(f"| Check | read back bit for bit, 4.5000 to 4.5007 bits a weight stored | {', '.join(f'{b:.5f}' for b, _ in C) or '-'} | {yes(verdict(C, 6))} |")
    decided = [verdict(F1, 6), verdict(F2, 3), verdict(F3, 3), verdict(F4, 3), verdict(F5, 3), verdict(F6, 6)]
    print(f"\n{sum(1 for d in decided if d)} of {sum(1 for d in decided if d is not None)} decided predictions hold; {sum(1 for d in decided if d is None)} not yet decided.")


if __name__ == "__main__":
    main()
