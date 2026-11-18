#!/usr/bin/env python3
"""Tables and verdicts for the third model, Llama-3.2-1B-Instruct, from the two results files and forecast.out.

results-wikitext.jsonl holds the variants scored on the wikitext-2 test text and results-python.jsonl the same variants
scored on the Python text of the record second-text-python-2026-11-18. Scores the predictions L1 to L6 of PREREG.md by
the rules written there. Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/third-model-llama32-2026-11-18/summary.py > experiments/third-model-llama32-2026-11-18/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXTS = (("wikitext-2 test text", "results-wikitext.jsonl"), ("Python text", "results-python.jsonl"))
ARMS = (("llama.cpp Q4_K_M", "none", "file:llama32-Q4_K_M.gguf"), ("llama.cpp Q4_K_M", "wikitext-2 training text", "file:llama32-Q4_K_M-imat.gguf"),
        ("llama.cpp Q4_K_M", "no meaning", "file:llama32-Q4_K_M-rimat.gguf"), ("one ball per group of 128", "none", "ZBG128:b560s16"),
        ("one ball per group of 128", "wikitext-2 training text", "hZBG128:b560s16"), ("one ball per group of 128", "no meaning", "hZBG128:b560s16~rand"))


def load(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "kl_mean" in r:
                    rows[r["variant"]] = r
    return rows


def gap(a, b):
    """How far the KL of row a is under the KL of row b, in errors of the gap."""
    return (b["kl_mean"] - a["kl_mean"]) / math.sqrt(a["kl_mean_se"] ** 2 + b["kl_mean_se"] ** 2)


def loud():
    """The share of the attention's input in its ten most active columns, percent, mean over the layers, from forecast.out."""
    path = HERE / "forecast.out"
    if path.exists():
        for line in path.read_text().splitlines():
            p = [x.strip() for x in line.split("|")]
            if p[0] == "attn_q" and len(p) > 4:
                return float(p[3])
    return None


def main():
    verdict = {}
    share = loud()
    print("Llama-3.2-1B-Instruct: llama.cpp's Q4_K_M and the ball arm, each with no matrix, the matrix of the wikitext-2 training text and the matrix of no meaning, on two texts. 10,200 positions each.")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    print(f"From forecast.out: the ten most active dimensions hold {'not measured yet' if share is None else f'{share:.1f} percent'} of what the attention reads. SmolLM2 has 61.8 and Qwen2.5 has 11.4.")
    got = {}
    for text, name in TEXTS:
        rows = load(HERE / name)
        got[text] = g = {(arm[0], matrix): rows[v] for arm, matrix, v in ARMS if v in rows}
        print()
        print(f"On the {text}")
        print("arm | matrix | bits a weight | mean KL | its standard error | first choice kept, percent")
        for arm, matrix, v in ARMS:
            if v in rows:
                r = rows[v]
                print(f"{arm} | {matrix} | {r['bits_per_weight']:.4f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f}")
        q, qi, qr = (g.get(("l", x)) for x in ("none", "wikitext-2 training text", "no meaning"))
        b, bi, br = (g.get(("o", x)) for x in ("none", "wikitext-2 training text", "no meaning"))
        if b and q:
            print(f"no matrix on either side: the ball arm {b['kl_mean']:.6f} against Q4_K_M {q['kl_mean']:.6f}, under it by {gap(b, q):.1f} errors")
            verdict["L1" if text.startswith("wiki") else "L2"] = gap(b, q) >= 2
        if bi and qi:
            print(f"the training text's matrix on both sides: the ball arm {bi['kl_mean']:.6f} against Q4_K_M {qi['kl_mean']:.6f}, under it by {gap(bi, qi):.1f} errors")
            if text.startswith("wiki"):
                verdict["L3"] = gap(bi, qi) >= 2
        if br and qi:
            print(f"the ball arm with the matrix of no meaning {br['kl_mean']:.6f} against Q4_K_M with the training text's matrix {qi['kl_mean']:.6f}, under it by {gap(br, qi):.1f} errors")
            if text.startswith("wiki"):
                verdict["L4"] = gap(br, qi) >= 2
        if b and bi:
            print(f"the ball arm: the training text's matrix is under no matrix by {gap(bi, b):.1f} errors")
        if b and br:
            print(f"the ball arm: the matrix of no meaning is under no matrix by {gap(br, b):.1f} errors")
        if q and qi:
            print(f"Q4_K_M: the training text's matrix is under no matrix by {gap(qi, q):.1f} errors")
        if q and qr:
            print(f"Q4_K_M: the matrix of no meaning is under no matrix by {gap(qr, q):.1f} errors")
        if q and qi and qr:
            s = (q["kl_mean"] - qr["kl_mean"]) / (q["kl_mean"] - qi["kl_mean"])
            print(f"Q4_K_M: the matrix of no meaning buys {100 * s:.1f} percent of what the training text's matrix buys")
            if text.startswith("wiki"):
                verdict["L6"] = s >= 0.5
    w, p = got["wikitext-2 test text"], got["Python text"]
    key = lambda x: ("o", x)
    if share is not None and all(key(x) in t for t in (w, p) for x in ("none", "wikitext-2 training text", "no meaning")):
        print()
        if share >= 40:
            verdict["L5"] = all(gap(t[key(x)], t[key("none")]) >= 2 for t in (w, p) for x in ("wikitext-2 training text", "no meaning"))
            print(f"L5: the share is {share:.1f}, 40 or more, so the prediction is that each matrix puts the ball arm under the plain arm by 2 errors on both texts")
        elif share <= 20:
            verdict["L5"] = all(gap(p[key(x)], p[key("none")]) < 2 for x in ("wikitext-2 training text", "no meaning"))
            print(f"L5: the share is {share:.1f}, 20 or less, so the prediction is that neither matrix puts the ball arm under the plain arm by 2 errors on the Python text")
        else:
            print(f"L5: the share is {share:.1f}, between 20 and 40, where the plan makes no prediction")
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    names = ("L1", "L2", "L3", "L4", "L5", "L6")
    for n in names:
        print(f"{n} | {'not scored' if n not in verdict else ('yes' if verdict[n] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of {len(names)}")


if __name__ == "__main__":
    main()
