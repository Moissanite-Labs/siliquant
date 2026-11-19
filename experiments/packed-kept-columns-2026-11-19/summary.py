#!/usr/bin/env python3
"""Tables and verdicts for the packed file with kept columns, from results.jsonl, noise.jsonl and the rows of four earlier records.

results.jsonl holds the files made with no matrix. noise.jsonl holds the files made with the matrix of no meaning.
Scores the predictions F1 to F5 of PREREG.md by the rules written there. Every number is a field of one of those
files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/packed-kept-columns-2026-11-19/summary.py > experiments/packed-kept-columns-2026-11-19/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
MODELS = (("Qwen2.5-1.5B", "qwen2.5-1.5b-instruct-fp16.gguf", "qwen15"), ("SmolLM2-1.7B", "SmolLM2-1.7B-Instruct-f16.gguf", "smollm2"), ("Llama-3.2-1B", "Llama-3.2-1B-Instruct-f16.gguf", "llama32"))


def rows(path, model):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model and "kl_mean" in r and r.get("text", "test") in ("test", "wikitext"):
                    out[r["variant"]] = r
    return out


def gap(a, b):
    """How far the KL of row a is under the KL of row b, in errors of the gap."""
    return (b["kl_mean"] - a["kl_mean"]) / math.sqrt(a["kl_mean_se"] ** 2 + b["kl_mean_se"] ** 2)


def main():
    verdict, f1, f4 = {}, [], []
    print("The packed file with the ten loud columns of every ffn_down kept whole: integer levels in one ball per 32 weights, 18 bytes a block, written, read back from disk and scored.")
    print("wikitext-2 test text, 10,200 positions. An error of a gap is the root of the sum of the squares of the two standard errors.")
    print()
    print("model | file | reads data | runs the model | bytes of the file | bits a weight | mean KL | its standard error | first choice kept, percent | "
          "under Q4_K_M with no matrix by, errors | under Q4_K_M with the wikitext matrix by, errors")
    for name, model, tag in MODELS:
        mine, noise = rows(HERE / "results.jsonl", model), rows(HERE / "noise.jsonl", model)
        packed = rows(EXP / "packed-ball-format-2026-11-18" / "results.jsonl", model)
        if name.startswith("Llama"):
            third = rows(EXP / "third-model-llama32-2026-11-18" / "results-wikitext.jsonl", model)
            q, qi = third.get(f"file:{tag}-Q4_K_M.gguf"), third.get(f"file:{tag}-Q4_K_M-imat.gguf")
        else:
            folder = "model-output-qwen25-2026-11-17" if tag == "qwen15" else "model-output-smollm2-2026-11-17"
            q, qi = rows(EXP / folder / "results.jsonl", model).get(f"file:{tag}-Q4_K_M.gguf"), rows(EXP / "imatrix-baseline-2026-11-17" / "results.jsonl", model).get(f"file:{tag}-Q4_K_M-imat.gguf")
        plain = mine.get("PZ32:b136s8") or packed.get("PZ32:b136s8")
        kept, nkept = mine.get("PZ32:b136s8+K10"), noise.get("hPZ32:b136s8+K10")
        for label, r, data, runs in (("no matrix, nothing kept" + ("" if "PZ32:b136s8" in mine else ", from the packed-format record"), plain, "no", "no"),
                                     ("no matrix, loud columns kept", kept, "no", "no"), ("the matrix of no meaning, loud columns kept", nkept, "no", "yes")):
            if r:
                print(f"{name} | {label} | {data} | {runs} | {r['file_bytes']} | {r['stored_bits_per_weight']:.5f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | {gap(r, q):.1f} | {gap(r, qi):.1f}")
        print(f"{name} | llama.cpp Q4_K_M, no matrix | no | no | | {q['bits_per_weight']:.4f} | {q['kl_mean']:.6f} | {q['kl_mean_se']:.6f} | {q['same_top']:.2f} | | ")
        print(f"{name} | llama.cpp Q4_K_M, the wikitext matrix | yes | yes | | {qi['bits_per_weight']:.4f} | {qi['kl_mean']:.6f} | {qi['kl_mean_se']:.6f} | {qi['same_top']:.2f} | | ")
        if kept:
            f1.append(gap(kept, q) >= 2)
            if plain:
                print(f"{name}: the kept columns put the file under the file with nothing kept by {100 * (1 - kept['kl_mean'] / plain['kl_mean']):.1f} percent, {gap(kept, plain):.1f} errors")
                if name.startswith("Llama"):
                    verdict["F3"] = kept["kl_mean"] <= 0.8 * plain["kl_mean"]
                else:
                    f4.append(-gap(kept, plain) < 2)
        if name.startswith("Llama") and "PZ32:b136s8" in mine:
            verdict["F2"] = gap(plain, q) < 2
        if nkept:
            verdict["F5"] = gap(nkept, qi) >= 2
    if len(f1) == 3:
        verdict["F1"] = all(f1)
    if len(f4) == 2:
        verdict["F4"] = all(f4)
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    names = ("F1", "F2", "F3", "F4", "F5")
    for n in names:
        print(f"{n} | {'not scored' if n not in verdict else ('yes' if verdict[n] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of {len(names)}")


if __name__ == "__main__":
    main()
