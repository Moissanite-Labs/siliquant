#!/usr/bin/env python3
"""Tables and the verdict on each prediction, from results.jsonl. Standard library only.

Usage, from the root of a checkout:
  python3 experiments/model-output-gemma-e4b-2026-11-15/summary.py > experiments/model-output-gemma-e4b-2026-11-15/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    rows, metrics = {}, {}
    for line in (HERE / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        (rows if "kl_mean" in r else metrics)[r["variant"]] = r
    for v, m in metrics.items():          # a metrics-only pass fills in what the scored pass did not record
        if v in rows:
            for k, val in m.items():
                rows[v].setdefault(k, val)
    return rows


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    a, b = ranks(xs), ranks(ys)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def line(r, name=None):
    return (f"   {name or r['variant']:<22} | {f(r.get('bits_per_weight'))} | {f(r.get('weight_err_db_mean'), 2)} | {f(r.get('path_err_db_mean'), 2)} | "
            f"{r['kl_mean']:.4f} +- {r['kl_mean_se']:.4f} | {r['kl_median']:.4f} | {r['same_top']:.2f} | {r['ppl']:.2f}")


HEAD = "   variant                | bits per weight | weight error, mean dB | path-weighted error, mean dB | mean KL | median KL | top token kept, % | perplexity"


def verdict(name, ok, detail):
    print(f"   {name}: {'HOLDS' if ok else 'FAILS'}. {detail}")


def main():
    rows = load()
    ref = rows["identity"]
    print("Reference model against its own saved outputs")
    print(f"   perplexity {ref['ppl']:.4f}, mean KL {ref['kl_mean']:.6f}, largest KL at one position {ref['kl_max']:.6f}, top token kept {ref['same_top']:.2f} %")
    print("   The row is the control copy (block matrices stored as 16-bit floats). The unchanged file scores the same.")
    print()

    print("1. Closures")
    print("   variant | weights changed | share of the model's weights | mean KL | largest KL | top token kept, % | perplexity")
    for v, label in (("close:per_layer_token_embd.weight", "K1 rows of per_layer_token_embd for tokens not in the text, zeroed"),
                     ("close:token_embd.weight", "K2 the same rows of token_embd, zeroed"),
                     ("dead:kv", "keys and values of the 18 layers that reuse an earlier layer's, zeroed"),
                     ("gauge:1", "random move along three exact symmetries")):
        if v not in rows:
            continue
        r = rows[v]
        n = r.get("weights_zeroed", r.get("weights"))
        extra = f" (the move is {r['weight_move_db']:+.2f} dB against the weights moved)" if "weight_move_db" in r else ""
        print(f"   {label} | {n} | {100.0 * n / 7518069290:.1f} | {r['kl_mean']:.6f} | {r['kl_max']:.6f} | {r['same_top']:.2f} | {r['ppl']:.4f}{extra}")
    k1, k2 = rows["close:per_layer_token_embd.weight"], rows["close:token_embd.weight"]
    print(f"   tokens in the text: {k1['tokens_in_text']} of 262144")
    verdict("K1", k1["kl_mean"] <= 1e-4 and k1["same_top"] >= 99.99, f"mean KL {k1['kl_mean']:.6f}, top token kept {k1['same_top']:.2f} %")
    verdict("K2", k2["kl_mean"] > 0.01, f"mean KL {k2['kl_mean']:.3f}")
    print()

    print("2. Rounding the 294 block matrices")
    print(HEAD)
    order = ["S", "E8", "R+S", "C+S", "D+S", "D+E8", "G+S", "GF+S"]
    for q in (64, 32, 16, 8, 4):
        for arm in order:
            v = f"{arm}:{q}"
            if v in rows:
                print(line(rows[v]))
    print()

    print("3. llama.cpp's own quantisers on the same matrices")
    print(HEAD)
    for v in sorted(rows):
        if v.startswith("file:"):
            print(line(rows[v], v[5:].replace(".gguf", "")))
    print()

    def kl(v):
        return rows[v]["kl_mean"]

    print("4. Predictions of the plan")
    have = [q for q in (8, 16, 32) if all(f"{a}:{q}" in rows for a in ("S", "E8", "D+S", "D+E8"))]
    r1 = [(q, a, b, kl(f"{a}:{q}"), kl(f"{b}:{q}")) for q in have for a, b in (("E8", "S"), ("D+E8", "D+S"))]
    verdict("R1", len(have) == 3 and all(x[3] < x[4] for x in r1),
            "; ".join(f"q={q} {a} {ka:.3f} against {b} {kb:.3f}" for q, a, b, ka, kb in r1))
    r2 = [(q, a, b, kl(f"{a}:{q}"), kl(f"{b}:{q}")) for q in have for a, b in (("D+S", "S"), ("D+E8", "E8"))]
    verdict("R2", len(have) == 3 and all(x[3] < x[4] for x in r2),
            "; ".join(f"q={q} {a} {ka:.3f} against {b} {kb:.3f}" for q, a, b, ka, kb in r2))
    if 32 in have:
        parts, ok = [], True
        for a, b in (("S", "E8"), ("D+S", "D+E8")):
            ra, rb = rows[f"{a}:32"], rows[f"{b}:32"]
            gain_w = ra["weight_err_db_pooled"] - rb["weight_err_db_pooled"]
            gain_kl = 10.0 * math.log10(ra["kl_mean"] / rb["kl_mean"])
            ratio = gain_kl / gain_w
            ok = ok and 0.5 <= ratio <= 1.5
            parts.append(f"{b} over {a}: weight error {gain_w:+.2f} dB, KL {gain_kl:+.2f} dB, ratio {ratio:.2f}")
        verdict("R3", ok, "; ".join(parts))
    if "S:4" in rows and "E8:4" in rows:
        verdict("R4", rows["S:4"]["same_top"] < 50 and rows["E8:4"]["same_top"] < 50,
                f"top token kept: S {rows['S:4']['same_top']:.2f} %, E8 {rows['E8:4']['same_top']:.2f} %")
    anchor = next((v for v in rows if v.startswith("file:Q4_K_M")), None)
    if anchor and 16 in have:
        best = min((kl(f"{a}:16"), a) for a in ("S", "E8", "D+S", "D+E8"))
        verdict("R5", kl(anchor) < best[0], f"Q4_K_M {kl(anchor):.4f} at {rows[anchor]['bits_per_weight']:.2f} bits; best arm at q=16 is {best[1]} with {best[0]:.4f}")
    print()

    print("5. Predictions of round 2")
    if all(f"{a}:32" in rows for a in ("S", "R+S", "C+S", "D+S")):
        s, r_, c, d = (kl(f"{a}:32") for a in ("S", "R+S", "C+S", "D+S"))
        verdict("A1", r_ < s and c < s and d < r_ and d < c, f"S {s:.3f}, R+S {r_:.3f}, C+S {c:.3f}, D+S {d:.3f}")
    if "G+S:32" in rows and "S:32" in rows:
        verdict("A2", kl("G+S:32") < kl("S:32"), f"G+S {kl('G+S:32'):.3f} against S {kl('S:32'):.3f}")
    arms = [f"{a}:32" for a in order if f"{a}:32" in rows and "path_err_db_mean" in rows[f"{a}:32"]]
    if len(arms) >= 5:
        ks = [kl(v) for v in arms]
        rp = spearman([rows[v]["path_err_db_mean"] for v in arms], ks)
        rw = spearman([rows[v]["weight_err_db_mean"] for v in arms], ks)
        verdict("A3", len(arms) == 8 and rp >= 0.7 and rp > rw,
                f"rank correlation with mean KL over {len(arms)} arms: path-weighted error {rp:+.2f}, plain weight error {rw:+.2f}")
    print()

    kinds = [v for v in rows if "@" in v]
    if kinds:
        print("6. Arm S at q = 16 on one kind of matrix at a time (not part of the predictions)")
        print("   kind | matrices | weights, millions | weight error, mean dB | mean KL | top token kept, % | mean KL per billion weights")
        for v in sorted(kinds, key=kl, reverse=True):
            r = rows[v]
            print(f"   {r['kind']} | {r['matrices']} | {r['weights'] / 1e6:.0f} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.4f} | {r['same_top']:.2f} | {r['kl_mean'] / (r['weights'] / 1e9):.3f}")
        if "S:16" in rows:
            print(f"   all seven kinds at once: mean KL {kl('S:16'):.4f}; the seven separate values add up to {sum(kl(v) for v in kinds):.4f}")
        print("   18 of the 42 attn_k and attn_v matrices are never read (section 1), so rounding them changes nothing.")


if __name__ == "__main__":
    main()
