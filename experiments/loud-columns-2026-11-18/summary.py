#!/usr/bin/env python3
"""Tables and verdicts for the loud columns of ffn_down, from results.jsonl and the rows of four earlier records.

Reads results.jsonl of this folder, the single-layer clips of the two clip-by-layer records, and the ball arms and
llama.cpp's files of the model-output and imatrix-baseline records. Scores the predictions A1 to A4 and B1 to B6 of
PREREG.md by the rules written there. Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/loud-columns-2026-11-18/summary.py > experiments/loud-columns-2026-11-18/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SMOL, QWEN = "SmolLM2-1.7B-Instruct-f16.gguf", "qwen2.5-1.5b-instruct-fp16.gguf"
SHORT = {SMOL: "SmolLM2-1.7B", QWEN: "Qwen2.5-1.5B"}
CLIPS = {SMOL: (EXP / "clip-by-layer-smollm2-2026-11-18" / "results.jsonl", (1, 7)), QWEN: (EXP / "clip-by-layer-qwen25-2026-11-18" / "results.jsonl", (1, 27))}
PLAIN = {SMOL: EXP / "model-output-smollm2-2026-11-17" / "results.jsonl", QWEN: EXP / "model-output-qwen25-2026-11-17" / "results.jsonl"}
IMAT = EXP / "imatrix-baseline-2026-11-17" / "results.jsonl"
ARM, HARM = "ZBG128:b560s16", "hZBG128:b560s16"


def load(path, model, text=None):
    """variant -> row, for the scored rows of one model, and of one text if the file names texts."""
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model and "kl_mean" in r and (text is None or r.get("text", text) == text):
                    rows[r["variant"]] = r
    return rows


def fmt(x):
    return "two left whole" if x is None else f"{x:.2f}"


def gap(a, b):
    """How far the KL of row a is under the KL of row b, in errors of the gap."""
    return (b["kl_mean"] - a["kl_mean"]) / math.sqrt(a["kl_mean_se"] ** 2 + b["kl_mean_se"] ** 2)


def main():
    verdict = {}
    mine = {m: load(HERE / "results.jsonl", m, "test") for m in (SMOL, QWEN)}
    print("The loud columns of ffn_down: wikitext-2 test text, 10,200 positions. A loud column is one of the ten in which the clip at 5 standard deviations of a row removes the most squared weight.")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    print()
    print("Part A: the clip of one ffn_down, whole (from the clip-by-layer records), in its ten loud columns only, and everywhere but in them")
    print("model | layer | where the clip is applied | entries clipped | weight error, dB | mean KL | its standard error | over the KL of the whole clip | first choice kept, percent")
    ratio = {}
    for m in (SMOL, QWEN):
        path, layers = CLIPS[m]
        whole = load(path, m, "test")
        for l in layers:
            w = whole.get(f"clip:5@ffn_down#{l}")
            if w:
                print(f"{SHORT[m]} | {l} | everywhere | {w['clipped_entries']} | {w['weight_err_db_mean']:.2f} | {w['kl_mean']:.6f} | {w['kl_mean_se']:.6f} | 1.0000 | {w['same_top']:.2f}")
            for part, label in (("only", "the ten loud columns only"), ("but", "everywhere but the ten")):
                r = mine[m].get(f"clip:5@ffn_down#{l}/{part}10")
                if r and w:
                    ratio[(m, l, part)] = r["kl_mean"] / w["kl_mean"]
                    print(f"{SHORT[m]} | {l} | {label} | {r['clipped_entries']} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | "
                          f"{ratio[(m, l, part)]:.4f} | {r['same_top']:.2f}")
    if all((SMOL, l, "but") in ratio for l in (1, 7)):
        verdict["A1"] = all(ratio[(SMOL, l, "but")] < 0.01 for l in (1, 7))
    if all((SMOL, l, "only") in ratio for l in (1, 7)):
        verdict["A2"] = all(ratio[(SMOL, l, "only")] >= 0.8 for l in (1, 7))
    if (QWEN, 1, "but") in ratio:
        verdict["A3"] = ratio[(QWEN, 1, "but")] < 0.1
    if (QWEN, 27, "but") in ratio:
        verdict["A4"] = ratio[(QWEN, 27, "but")] > 0.1

    print()
    print("Part B: one ball per group of 128 weights at 4.5 bits, with columns of ffn_down left as in the reference")
    print("model | arm | reads calibration data | bits a weight | columns kept | weight error of ffn_down, dB | mean KL | its standard error | "
          "under its plain arm by, percent | in errors of the gap | first choice kept, percent")
    base = {}
    for m in (SMOL, QWEN):
        plain, imat = load(PLAIN[m], m), load(IMAT, m)
        base[m] = {ARM: plain.get(ARM), HARM: imat.get(HARM)}
        tag = "smollm2" if m == SMOL else "qwen15"
        for label, r, data in ((f"llama.cpp Q4_K_M", plain.get(f"file:{tag}-Q4_K_M.gguf"), "no"), ("llama.cpp Q4_K_M with the matrix", imat.get(f"file:{tag}-Q4_K_M-imat.gguf"), "yes"),
                               (ARM, base[m][ARM], "no"), (HARM, base[m][HARM], "yes")):
            if r:
                print(f"{SHORT[m]} | {label} | {data} | {r['bits_per_weight']:.4f} | 0 | | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | | | {r['same_top']:.2f}")
        for v, r in mine[m].items():
            if "+" not in v:
                continue
            b = base[m][v.split("+")[0]]
            data = "yes" if v.startswith("h") or "+A10" in v else "no"
            print(f"{SHORT[m]} | {v} | {data} | {r['bits_per_weight_changed']:.4f} | {r['kept_columns']} | {fmt(r['weight_err_db_ffn_down'])} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | "
                  f"{100 * (1 - r['kl_mean'] / b['kl_mean']):.1f} | {gap(r, b):.1f} | {r['same_top']:.2f}")
    s, q = mine[SMOL], mine[QWEN]
    k, k17, whole, a, hk = [s.get(ARM + x) for x in ("+K10", "+K10#1,7", "+S#1,7", "+A10")] + [s.get(HARM + "+K10")]
    b, hb = base[SMOL][ARM], base[SMOL][HARM]
    print()
    if k:
        verdict["B1"] = gap(k, b) >= 2 and k["kl_mean"] <= 0.8 * b["kl_mean"]
        for label, other in (("llama.cpp Q4_K_M with the matrix", load(IMAT, SMOL).get("file:smollm2-Q4_K_M-imat.gguf")), (HARM, hb)):
            print(f"SmolLM2-1.7B, not predicted: {ARM}+K10, which reads no data, against {label}: {k['kl_mean']:.6f} against {other['kl_mean']:.6f}, under it by {gap(k, other):.1f} errors")
    if k and k17 and gap(k, b) >= 2:
        share = (b["kl_mean"] - k17["kl_mean"]) / (b["kl_mean"] - k["kl_mean"])
        print(f"SmolLM2-1.7B: keeping the columns in layers 1 and 7 only gains {share:.3f} of what keeping them in every layer gains")
        verdict["B2"] = share >= 0.5
    if k17 and whole:
        print(f"SmolLM2-1.7B: ten columns kept in layers 1 and 7 against those two matrices left whole: {k17['kl_mean']:.6f} against {whole['kl_mean']:.6f}, above it by {-gap(k17, whole):.1f} errors")
        verdict["B3"] = -gap(k17, whole) < 2
    if k and a:
        print(f"SmolLM2-1.7B: the ten most active columns against the ten loud ones: {a['kl_mean']:.6f} against {k['kl_mean']:.6f}, under it by {gap(a, k):.1f} errors")
        verdict["B4"] = abs(gap(a, k)) < 2
    if hk:
        verdict["B5"] = gap(hk, hb) >= 2
    if q.get(ARM + "+K10"):
        verdict["B6"] = abs(gap(q[ARM + "+K10"], base[QWEN][ARM])) < 2
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    names = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6")
    for n in names:
        print(f"{n} | {'not scored' if n not in verdict else ('yes' if verdict[n] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of {len(names)}")


if __name__ == "__main__":
    main()
