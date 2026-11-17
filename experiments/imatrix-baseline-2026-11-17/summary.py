#!/usr/bin/env python3
"""Tables and verdicts for the importance-matrix baseline, from the three results files.

Reads results.jsonl of this folder (llama.cpp's files made with an importance matrix) and of
the two model-output records (the same files made without one, and the ball arms). Scores the
predictions I1 to I6 of PREREG.md and J1 to J4 of its extension 1 by the rules written there.
Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/imatrix-baseline-2026-11-17/summary.py > experiments/imatrix-baseline-2026-11-17/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (
    ("Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", HERE.parent / "model-output-qwen25-2026-11-17" / "results.jsonl", "qwen15"),
    ("SmolLM2-1.7B-Instruct", "SmolLM2", HERE.parent / "model-output-smollm2-2026-11-17" / "results.jsonl", "smollm2"),
)
ARMS = ("EBG128:b560s16", "EBG32:b136s8", "ZBG128:b560s16", "EB32@32:16", "EB128@128:16", "EBG128:b504s8", "X32:16")


def load(path, model=None):
    """variant -> row, for the rows of one model if a model is named (a part of its file's name)."""
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if model is None or model.lower() in r["model"].lower():
                    rows[r["variant"]] = r
    return rows


def main():
    print("llama.cpp's files made with an importance matrix from the wikitext-2 training text, scored on the test text: 10,200 positions")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    verdicts = {}

    def gap(a, b):
        """(mean KL of b less mean KL of a, in errors): positive when a is lower."""
        d = b["kl_mean"] - a["kl_mean"]
        e = math.hypot(a["kl_mean_se"], b["kl_mean_se"])
        return d, d / e

    for title, key, path, short in MODELS:
        base = load(path)
        mine = load(HERE / "results.jsonl", key)
        rows = {}
        for q in ("Q4_K_M", "Q4_0"):
            rows[q] = base.get(f"file:{short}-{q}.gguf")
            rows[q + " with the matrix"] = mine.get(f"file:{short}-{q}-imat.gguf")
        for a in ARMS:
            rows[a] = base.get(a)
        rows["hZBG128:b560s16"] = mine.get("hZBG128:b560s16")
        rows["iZBG128:b560s16"] = mine.get("iZBG128:b560s16")
        print()
        print(f"{title}")
        if not any(rows[k] for k in ("Q4_K_M with the matrix", "Q4_0 with the matrix")):
            print("   not scored yet")
            continue
        print("variant | bits per weight | weight error in dB, mean over the block matrices | mean KL | its standard error | first choice kept, percent")
        for name, r in rows.items():
            if r:
                print(f"{name} | {r['bits_per_weight']:.3f} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f}")
            else:
                print(f"{name} | not scored")
        print("comparison | mean KL of the first | of the second | first lower by | in errors | in percent")
        pairs = [("Q4_K_M with the matrix", "Q4_K_M"), ("Q4_0 with the matrix", "Q4_0"), ("EBG128:b560s16", "Q4_K_M with the matrix"), ("ZBG128:b560s16", "Q4_K_M with the matrix"),
                 ("EBG32:b136s8", "Q4_K_M with the matrix"), ("Q4_K_M with the matrix", "EB128@128:16"), ("EB32@32:16", "Q4_0 with the matrix"), ("EBG128:b504s8", "Q4_0 with the matrix"),
                 ("iZBG128:b560s16", "ZBG128:b560s16"), ("hZBG128:b560s16", "ZBG128:b560s16"), ("iZBG128:b560s16", "Q4_K_M with the matrix"), ("hZBG128:b560s16", "Q4_K_M with the matrix"),
                 ("iZBG128:b560s16", "hZBG128:b560s16"), ("hZBG128:b560s16", "EBG128:b560s16")]
        for a, b in pairs:
            if rows.get(a) and rows.get(b):
                d, z = gap(rows[a], rows[b])
                verdicts[(short, a, b)] = z
                print(f"{a} against {b} | {rows[a]['kl_mean']:.4f} | {rows[b]['kl_mean']:.4f} | {d:+.4f} | {z:+.1f} | {100 * d / rows[b]['kl_mean']:+.1f}")
        for q in ("Q4_K_M", "Q4_0"):
            r = rows.get(q + " with the matrix")
            if r:
                print(f"   {q} with the matrix: SHA-256 {r.get('file_sha256', '?')}, other tensors that differ from the reference: {r.get('other_tensors_that_differ')}")

    print()
    print("The predictions of PREREG.md")
    print("prediction | what was measured | holds")

    def line(tag, text, checks):
        zs = [verdicts.get(k) for k in checks]
        if any(z is None for z in zs):
            done = [f"{z:+.1f}" for z in zs if z is not None]
            print(f"{tag} | {text}: {', '.join(done) if done else 'nothing'} errors so far | not all scored")
        else:
            print(f"{tag} | {text}: {', '.join(f'{z:+.1f}' for z in zs)} errors | {'yes' if all(z >= 2.0 for z in zs) else 'NO'}")

    line("I1", "Q4_K_M with the matrix under Q4_K_M without, on Qwen2.5 and SmolLM2", [("qwen15", "Q4_K_M with the matrix", "Q4_K_M"), ("smollm2", "Q4_K_M with the matrix", "Q4_K_M")])
    line("I2", "Q4_0 with the matrix under Q4_0 without, on Qwen2.5 and SmolLM2", [("qwen15", "Q4_0 with the matrix", "Q4_0"), ("smollm2", "Q4_0 with the matrix", "Q4_0")])
    line("I3", "On Qwen2.5, EBG128:b560s16 under Q4_K_M with the matrix", [("qwen15", "EBG128:b560s16", "Q4_K_M with the matrix")])
    line("I4", "On Qwen2.5, Q4_K_M with the matrix under EB128@128:16", [("qwen15", "Q4_K_M with the matrix", "EB128@128:16")])
    line("I5", "On Qwen2.5, EB32@32:16 under Q4_0 with the matrix", [("qwen15", "EB32@32:16", "Q4_0 with the matrix")])
    line("I6", "On SmolLM2, EBG128:b560s16 under Q4_K_M with the matrix", [("smollm2", "EBG128:b560s16", "Q4_K_M with the matrix")])
    print()
    print("The predictions of extension 1")
    print("prediction | what was measured | holds")
    line("J1", "iZBG128:b560s16 under ZBG128:b560s16, on Qwen2.5 and SmolLM2", [("qwen15", "iZBG128:b560s16", "ZBG128:b560s16"), ("smollm2", "iZBG128:b560s16", "ZBG128:b560s16")])
    line("J2", "On SmolLM2, iZBG128:b560s16 under Q4_K_M with the matrix", [("smollm2", "iZBG128:b560s16", "Q4_K_M with the matrix")])
    line("J3", "On Qwen2.5, iZBG128:b560s16 under Q4_K_M with the matrix", [("qwen15", "iZBG128:b560s16", "Q4_K_M with the matrix")])
    zs = [verdicts.get((m, "iZBG128:b560s16", "hZBG128:b560s16")) for m in ("qwen15", "smollm2")]
    if any(z is None for z in zs):
        print("J4 | iZBG128:b560s16 against hZBG128:b560s16 | not all scored")
    else:
        print(f"J4 | iZBG128:b560s16 no worse than hZBG128:b560s16 by 2 errors, on Qwen2.5 and SmolLM2: {zs[0]:+.1f}, {zs[1]:+.1f} errors | {'yes' if all(z > -2.0 for z in zs) else 'NO'}")


if __name__ == "__main__":
    main()
