#!/usr/bin/env python3
"""Tables and verdicts for the headline comparisons on a second text, Python source, from results.jsonl and the rows of four earlier records.

Every row of results.jsonl of this folder is scored on the Python text. Beside each, the same variant's score on the
wikitext-2 test text is read from the record that scored it. Scores the predictions C1 to C6 of PREREG.md by the rules
written there. Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/second-text-python-2026-11-18/summary.py > experiments/second-text-python-2026-11-18/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SMOL, QWEN = "SmolLM2-1.7B-Instruct-f16.gguf", "qwen2.5-1.5b-instruct-fp16.gguf"
SHORT = {SMOL: "SmolLM2-1.7B", QWEN: "Qwen2.5-1.5B"}
TAG = {SMOL: "smollm2", QWEN: "qwen15"}
WIKI = {SMOL: (EXP / "model-output-smollm2-2026-11-17" / "results.jsonl", EXP / "imatrix-baseline-2026-11-17" / "results.jsonl", EXP / "imatrix-gain-parts-2026-11-18" / "results.jsonl"),
        QWEN: (EXP / "model-output-qwen25-2026-11-17" / "results.jsonl", EXP / "imatrix-baseline-2026-11-17" / "results.jsonl", EXP / "imatrix-gain-parts-2026-11-18" / "results.jsonl")}


def load(paths, model):
    """variant -> row, for the scored rows of one model in some files."""
    rows = {}
    for path in paths:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("model") == model and "kl_mean" in r:
                        rows[r["variant"]] = r
    return rows


def gap(a, b):
    """How far the KL of row a is under the KL of row b, in errors of the gap."""
    return (b["kl_mean"] - a["kl_mean"]) / math.sqrt(a["kl_mean_se"] ** 2 + b["kl_mean_se"] ** 2)


def arms(m):
    t = TAG[m]
    return (("llama.cpp Q4_K_M", "none", f"file:{t}-Q4_K_M.gguf"), ("llama.cpp Q4_K_M", "wikitext-2 training text", f"file:{t}-Q4_K_M-imat.gguf"),
            ("llama.cpp Q4_K_M", "no meaning", f"file:{t}-Q4_K_M-rimat.gguf"), ("one ball per group of 128", "none", "ZBG128:b560s16"),
            ("one ball per group of 128", "wikitext-2 training text", "hZBG128:b560s16"), ("one ball per group of 128", "no meaning", "hZBG128:b560s16~rand"))


def main():
    print("The headline comparisons on a second text: Python source, conversion/base.py of llama.cpp at eada89c, 10,200 positions.")
    print("Beside each, the same variant on the wikitext-2 test text, from the record that scored it.")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    print()
    print("model | arm | matrix | bits a weight | mean KL on Python | its standard error | first choice kept, percent | mean KL on wikitext-2 | Python over wikitext-2")
    got = {}
    for m in (SMOL, QWEN):
        mine, wiki = load((HERE / "results.jsonl",), m), load(WIKI[m], m)
        got[m] = {}
        for arm, matrix, v in arms(m):
            r, w = mine.get(v), wiki.get(v)
            if r:
                got[m][(arm[0], matrix)] = r
                print(f"{SHORT[m]} | {arm} | {matrix} | {r['bits_per_weight']:.4f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | {w['kl_mean']:.6f} | {r['kl_mean'] / w['kl_mean']:.3f}")
    print()
    verdict, per = {}, {"C1": [], "C2": [], "C3": []}
    for m in (SMOL, QWEN):
        g = got[m]
        q, qi, qr = (g.get(("l", x)) for x in ("none", "wikitext-2 training text", "no meaning"))
        b, bi, br = (g.get(("o", x)) for x in ("none", "wikitext-2 training text", "no meaning"))
        if b and q:
            print(f"{SHORT[m]}: no matrix on either side: the ball arm {b['kl_mean']:.6f} against Q4_K_M {q['kl_mean']:.6f}, under it by {gap(b, q):.1f} errors")
            per["C1"].append(gap(b, q) >= 2)
        if bi and qi:
            print(f"{SHORT[m]}: the training text's matrix on both sides: the ball arm {bi['kl_mean']:.6f} against Q4_K_M {qi['kl_mean']:.6f}, under it by {gap(bi, qi):.1f} errors")
            per["C2"].append(gap(bi, qi) >= 2)
        if br and qi:
            print(f"{SHORT[m]}: the ball arm with the matrix of no meaning {br['kl_mean']:.6f} against Q4_K_M with the training text's matrix {qi['kl_mean']:.6f}, under it by {gap(br, qi):.1f} errors")
            per["C3"].append(gap(br, qi) >= 2)
        if b and bi and br:
            share = (b["kl_mean"] - br["kl_mean"]) / (b["kl_mean"] - bi["kl_mean"])
            print(f"{SHORT[m]}: the ball arm: the matrix of no meaning buys {100 * share:.1f} percent of what the training text's matrix buys. "
                  f"The training text's matrix is under the matrix of no meaning by {gap(bi, br):.1f} errors. Under no matrix by {gap(bi, b):.1f}")
            if m == SMOL:
                verdict["C4"], verdict["C5"] = share >= 0.5, gap(bi, br) < 2
        if q and qi:
            line = f"{SHORT[m]}: Q4_K_M: the training text's matrix is under no matrix by {gap(qi, q):.1f} errors"
            if qr:
                line += f". The matrix of no meaning buys {100 * (q['kl_mean'] - qr['kl_mean']) / (q['kl_mean'] - qi['kl_mean']):.1f} percent of what it buys, and is under no matrix by {gap(qr, q):.1f} errors"
            print(line)
            if m == SMOL:
                verdict["C6"] = gap(qi, q) >= 2
    for k, v in per.items():
        if len(v) == 2:
            verdict[k] = all(v)
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    names = ("C1", "C2", "C3", "C4", "C5", "C6")
    for n in names:
        print(f"{n} | {'not scored' if n not in verdict else ('yes' if verdict[n] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of {len(names)}")


if __name__ == "__main__":
    main()
