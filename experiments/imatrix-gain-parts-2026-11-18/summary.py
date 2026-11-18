#!/usr/bin/env python3
"""Tables and verdicts for what the gain of an importance matrix is made of, from results.jsonl and the rows of three earlier records.

Reads results.jsonl of this folder, and the plain arm, the weighted arm and llama.cpp's files of the model-output and
imatrix-baseline records. Scores the predictions P1 to P4 and R1 to R4 of PREREG.md by the rules written there.
Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/imatrix-gain-parts-2026-11-18/summary.py > experiments/imatrix-gain-parts-2026-11-18/results.out
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
PLAIN = {SMOL: EXP / "model-output-smollm2-2026-11-17" / "results.jsonl", QWEN: EXP / "model-output-qwen25-2026-11-17" / "results.jsonl"}
IMAT = EXP / "imatrix-baseline-2026-11-17" / "results.jsonl"
ARM, HARM, RARM = "ZBG128:b560s16", "hZBG128:b560s16", "hZBG128:b560s16~rand"
PARTS = (("the attention's readers", "ZBG128:b560s16+h@attn_q,attn_k,attn_v"), ("the feed-forward readers", "ZBG128:b560s16+h@ffn_gate,ffn_up"),
         ("the writers", "ZBG128:b560s16+h@attn_output,ffn_down"))


def load(path, model):
    """variant -> row, for the scored rows of one model."""
    rows = {}
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


def line(label, r, base=None, whole=None):
    cells = [label, f"{r['bits_per_weight']:.4f}", f"{r['kl_mean']:.6f}", f"{r['kl_mean_se']:.6f}", f"{r['same_top']:.2f}"]
    if base is not None:
        cells += [f"{100 * (1 - r['kl_mean'] / base['kl_mean']):.1f}", f"{gap(r, base):.1f}"]
        if whole is not None:
            cells.append(f"{(base['kl_mean'] - r['kl_mean']) / (base['kl_mean'] - whole['kl_mean']):.3f}")
    return " | ".join(cells)


def main():
    verdict = {}
    mine = {m: load(HERE / "results.jsonl", m) for m in (SMOL, QWEN)}
    plain = {m: load(PLAIN[m], m) for m in (SMOL, QWEN)}
    imat = {m: load(IMAT, m) for m in (SMOL, QWEN)}
    print("What the gain of an importance matrix is made of: wikitext-2 test text, 10,200 positions.")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    print()
    print("Part A, SmolLM2-1.7B: one ball per group of 128 weights at 4.5 bits, with columns weighted by the matrix of the wikitext-2 training text on some kinds of matrix only")
    print("arm | bits a weight | mean KL | its standard error | first choice kept, percent | under the plain arm by, percent | in errors of the gap | share of the whole gain")
    p, h = plain[SMOL][ARM], imat[SMOL][HARM]
    print(line("no kind weighted, " + ARM, p))
    gains = {}
    for label, v in PARTS:
        r = mine[SMOL].get(v)
        if r:
            gains[v] = (p["kl_mean"] - r["kl_mean"]) / (p["kl_mean"] - h["kl_mean"])
            print(line(f"{label}, {v}", r, p, h))
    print(line("every kind weighted, " + HARM, h, p, h))
    if len(gains) == 3:
        a, f, w = (gains[v] for _, v in PARTS)
        print(f"the three shares add up to {a + f + w:.3f} of the whole gain")
        verdict["P1"], verdict["P2"], verdict["P3"], verdict["P4"] = a > 0.5, f < 0.25, w < 0.25, 0.75 <= a + f + w <= 1.25
    print()
    print("Part B: the same arms with a matrix made from a text with no meaning, beside the matrix of the wikitext-2 training text and no matrix")
    print("model | arm | matrix | bits a weight | mean KL | its standard error | first choice kept, percent | share of the real matrix's gain | above the arm with no matrix by, in errors")
    for m in (SMOL, QWEN):
        q, qi, qr = plain[m].get(f"file:{TAG[m]}-Q4_K_M.gguf"), imat[m].get(f"file:{TAG[m]}-Q4_K_M-imat.gguf"), mine[m].get(f"file:{TAG[m]}-Q4_K_M-rimat.gguf")
        b, bi, br = plain[m][ARM], imat[m][HARM], mine[m].get(RARM)
        for name, none, real, rand in (("llama.cpp Q4_K_M", q, qi, qr), ("one ball per group of 128", b, bi, br)):
            for label, r in (("none", none), ("wikitext-2 training text", real), ("no meaning", rand)):
                if r:
                    share = "" if r is none else f"{(none['kl_mean'] - r['kl_mean']) / (none['kl_mean'] - real['kl_mean']):.3f}"
                    over = "" if r is none else f"{-gap(r, none):.1f}"
                    print(f"{SHORT[m]} | {name} | {label} | {r['bits_per_weight']:.4f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | {share} | {over}")
        if br:
            share = (b["kl_mean"] - br["kl_mean"]) / (b["kl_mean"] - bi["kl_mean"])
            if m == SMOL:
                verdict["R1"] = share >= 0.5
                print(f"{SHORT[m]}, not predicted: the ball arm with the matrix of no meaning against llama.cpp Q4_K_M with the matrix of the training text: "
                      f"{br['kl_mean']:.6f} against {qi['kl_mean']:.6f}, under it by {gap(br, qi):.1f} errors")
            else:
                verdict["R2"] = -gap(br, b) < 2
        if qr:
            share = (q["kl_mean"] - qr["kl_mean"]) / (q["kl_mean"] - qi["kl_mean"])
            if m == SMOL:
                verdict["R3"] = share >= 0.5
            else:
                verdict["R4"] = -gap(qr, q) < 2
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    names = ("P1", "P2", "P3", "P4", "R1", "R2", "R3", "R4")
    for n in names:
        print(f"{n} | {'not scored' if n not in verdict else ('yes' if verdict[n] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of {len(names)}")


if __name__ == "__main__":
    main()
