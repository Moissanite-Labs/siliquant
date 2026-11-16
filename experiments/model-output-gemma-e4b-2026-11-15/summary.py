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
MODEL_WEIGHTS = 7518069290


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


HEAD = "   variant | bits per weight | weight error, mean dB | mean KL | median KL | top token kept, % | perplexity"


def line(r, name=None):
    return (f"   {name or r['variant']} | {f(r.get('bits_per_weight'))} | {f(r.get('weight_err_db_mean'), 2)} | "
            f"{r['kl_mean']:.4f} +- {r['kl_mean_se']:.4f} | {r['kl_median']:.4f} | {r['same_top']:.2f} | {r['ppl']:.2f}")


VERDICTS = []


def verdict(name, ok, detail):
    VERDICTS.append((name, ok))
    print(f"   {name}: {'HOLDS' if ok else 'FAILS'}. {detail}")


def main():
    rows = load()

    def kl(v):
        return rows[v]["kl_mean"]

    def table(variants):
        print(HEAD)
        for v in variants:
            if v in rows:
                print(line(rows[v], v[5:].replace(".gguf", "") if v.startswith("file:") else None))

    ref = rows["identity"]
    print("Reference model against its own saved outputs")
    print(f"   perplexity {ref['ppl']:.4f}, mean KL {ref['kl_mean']:.6f}, largest KL at one position {ref['kl_max']:.6f}, top token kept {ref['same_top']:.2f} %")
    print("   This row is the control copy with the block matrices stored as 16-bit floats. The unchanged file scores the same.")
    print()

    print("1. Closures")
    print("   variant | weights changed | share of the model's weights, % | mean KL | largest KL at one position | top token kept, % | perplexity")
    for v, label in (("close:per_layer_token_embd.weight", "K1 rows of per_layer_token_embd for tokens not in the text, zeroed"),
                     ("close:token_embd.weight", "K2 the same rows of token_embd, zeroed"),
                     ("dead:kv", "keys and values of the 18 layers that reuse an earlier layer's, zeroed"),
                     ("gauge:1", "random move along three exact symmetries")):
        if v not in rows:
            continue
        r = rows[v]
        n = r.get("weights_zeroed", r.get("weights"))
        extra = f" (the move is {r['weight_move_db']:+.2f} dB against the size of the weights moved)" if "weight_move_db" in r else ""
        print(f"   {label} | {n} | {100.0 * n / MODEL_WEIGHTS:.1f} | {r['kl_mean']:.6f} | {r['kl_max']:.6f} | {r['same_top']:.2f} | {r['ppl']:.4f}{extra}")
    k1, k2 = rows["close:per_layer_token_embd.weight"], rows["close:token_embd.weight"]
    print(f"   tokens in the text: {k1['tokens_in_text']} of 262144")
    verdict("K1", k1["kl_mean"] <= 1e-4 and k1["same_top"] >= 99.99, f"mean KL {k1['kl_mean']:.6f}, top token kept {k1['same_top']:.2f} %")
    verdict("K2", k2["kl_mean"] > 0.01, f"mean KL {k2['kl_mean']:.3f}")
    print()

    print("2. First grid: one scale per matrix, found by least squared error")
    table([f"{a}:{q}" for q in (32, 16, 8, 4) for a in ("S", "E8", "D+S", "D+E8")])
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
            ok = ok and 0.5 <= gain_kl / gain_w <= 1.5
            parts.append(f"{b} over {a}: pooled weight error {gain_w:+.2f} dB, KL {gain_kl:+.2f} dB, ratio {gain_kl / gain_w:.2f}")
        verdict("R3", ok, "; ".join(parts))
    if "S:4" in rows and "E8:4" in rows:
        verdict("R4", rows["S:4"]["same_top"] < 50 and rows["E8:4"]["same_top"] < 50,
                f"top token kept: S {rows['S:4']['same_top']:.2f} %, E8 {rows['E8:4']['same_top']:.2f} %")
    print()

    print("3. llama.cpp's own quantisers on the same matrices")
    anchors = sorted(v for v in rows if v.startswith("file:"))
    table(anchors)
    anchor = next((v for v in anchors if "Q4_K_M" in v), None)
    if anchor and 16 in have:
        best = min((kl(f"{a}:16"), a) for a in ("S", "E8", "D+S", "D+E8"))
        verdict("R5", kl(anchor) < best[0], f"Q4_K_M {kl(anchor):.4f} at {rows[anchor]['bits_per_weight']:.2f} bits; the best arm of the first grid at q=16 is {best[1]} with {best[0]:.4f}")
    print()

    print("4. Round 2 at q = 32: where the gain from balancing comes from, and balancing through the symmetries")
    table([f"{a}:32" for a in ("S", "R+S", "C+S", "D+S", "G+S", "GF+S")])
    if all(f"{a}:32" in rows for a in ("S", "R+S", "C+S", "D+S")):
        s, r_, c, d = (kl(f"{a}:32") for a in ("S", "R+S", "C+S", "D+S"))
        verdict("A1", r_ < s and c < s and d < r_ and d < c, f"S {s:.3f}, R+S {r_:.3f}, C+S {c:.3f}, D+S {d:.3f}")
    if "G+S:32" in rows and "S:32" in rows:
        verdict("A2", kl("G+S:32") < kl("S:32"), f"G+S {kl('G+S:32'):.3f} against S {kl('S:32'):.3f}")
    arms = [f"{a}:32" for a in ("S", "E8", "D+S", "D+E8", "R+S", "C+S", "G+S", "GF+S") if f"{a}:32" in rows and "path_err_db_mean" in rows[f"{a}:32"]]
    if len(arms) >= 5:
        print("   variant | weight error, mean dB | path-weighted error, mean dB | second path-weighted error (written after round 2), mean dB | mean KL")
        for v in arms:
            r = rows[v]
            print(f"   {v} | {r['weight_err_db_mean']:.2f} | {r['path_err_db_mean']:.2f} | {f(r.get('path2_err_db_mean'), 2)} | {r['kl_mean']:.4f}")
        ks = [kl(v) for v in arms]
        rp = spearman([rows[v]["path_err_db_mean"] for v in arms], ks)
        rw = spearman([rows[v]["weight_err_db_mean"] for v in arms], ks)
        verdict("A3", len(arms) == 8 and rp >= 0.7 and rp > rw,
                f"rank correlation with mean KL over {len(arms)} arms: path-weighted error {rp:+.3f}, plain weight error {rw:+.3f}")
        with2 = [v for v in arms if "path2_err_db_mean" in rows[v]]
        if len(with2) >= 5:
            print(f"   second path-weighted error over {len(with2)} arms: rank correlation {spearman([rows[v]['path2_err_db_mean'] for v in with2], [kl(v) for v in with2]):+.3f} (not a prediction)")
    print()

    print("5. Round 3: the size of the group that shares a scale, and how the scale is set")
    print("   X is llama.cpp's Q4_0 rule (scale from the group's largest entry). R+S and B32+S use the searched step, which clips.")
    table(["XR:16", "R+S:16", "X32:16", "B32+S:16", "XR:32", "R+S:32", "X32:32", "B32+S:32"])
    print("   XR at q = 32 with the step multiplied by a factor (a smaller step clips entries above that share of the row's largest):")
    sweep = [("XR:32", 1.0), ("XR:32~0.7", 0.7), ("XR:32~0.5", 0.5), ("XR:32~0.35", 0.35)]
    for v, c in sweep:
        if v in rows:
            print(f"   factor {c}: weight error {rows[v]['weight_err_db_mean']:.2f} dB, mean KL {kl(v):.4f}, top token kept {rows[v]['same_top']:.2f} %")
    q40 = next((v for v in anchors if "Q4_0" in v), None)
    if q40 and "X32:16" in rows:
        verdict("B1", 0.9 * kl(q40) <= kl("X32:16") <= 1.1 * kl(q40), f"X32 at q=16 {kl('X32:16'):.4f}, llama.cpp Q4_0 {kl(q40):.4f}")
    if all(v in rows for v, _ in sweep):
        best_kl = min(sweep, key=lambda x: kl(x[0]))[1]
        best_w = min(sweep, key=lambda x: rows[x[0]]["weight_err_db_mean"])[1]
        verdict("B2", best_kl == 1.0 and best_w != 1.0, f"lowest mean KL at factor {best_kl}, lowest weight error at factor {best_w}")
    b3 = [(a, b) for a, b in (("X32:16", "XR:16"), ("X32:32", "XR:32"), ("B32+S:16", "R+S:16"), ("B32+S:32", "R+S:32")) if a in rows and b in rows]
    if len(b3) == 4:
        verdict("B3", all(kl(a) < kl(b) for a, b in b3), "; ".join(f"{a} {kl(a):.4f} against {b} {kl(b):.4f}" for a, b in b3))
    print()

    print("6. Round 4: the lattice with one scale per group and no block shrunk")
    table(["X32:16", "VE832:16", "XR:16", "VE8R:16", "XR:32", "VE8R:32"])
    pairs = [(a, b) for a, b in (("X32:16", "VE832:16"), ("XR:32", "VE8R:32"), ("XR:16", "VE8R:16")) if a in rows and b in rows]
    if len(pairs) == 3:
        verdict("C1", all(kl(b) < kl(a) for a, b in pairs), "; ".join(f"{b} {kl(b):.4f} against {a} {kl(a):.4f}" for a, b in pairs))
        parts, ok = [], True
        for a, b in pairs:
            gw = rows[a]["weight_err_db_mean"] - rows[b]["weight_err_db_mean"]
            gk = 10.0 * math.log10(kl(a) / kl(b))
            ok = ok and 0.5 <= gk / gw <= 1.5
            parts.append(f"{b} over {a}: weight error {gw:+.2f} dB, KL {gk:+.2f} dB, ratio {gk / gw:.2f}")
        verdict("C2", ok, "; ".join(parts))
    print()

    clips = sorted((v for v in rows if v.startswith("clip:")), key=lambda v: float(v.split(":")[1].split("@")[0]))
    if clips:
        print("7. Clipping with no rounding (not part of the predictions)")
        print("   every entry beyond the limit, in standard deviations of its row, is pulled back to the limit")
        print("   limit | share of entries clipped, % | weight error, mean dB | mean KL | top token kept, %")
        for v in clips:
            r = rows[v]
            print(f"   {v[5:]} | {100 * r.get('share_clipped', 0):.4f} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.4f} +- {r['kl_mean_se']:.4f} | {r['same_top']:.2f}")
        print()

    kinds = [v for v in rows if v.startswith("S:16@")]
    if kinds:
        print("8. Arm S at q = 16 on one kind of matrix at a time (not part of the predictions)")
        print("   kind | matrices | weights, millions | weight error, mean dB | mean KL | top token kept, %")
        for v in sorted(kinds, key=kl, reverse=True):
            r = rows[v]
            print(f"   {r['kind']} | {r['matrices']} | {r['weights'] / 1e6:.0f} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.4f} | {r['same_top']:.2f}")
        if "S:16" in rows:
            print(f"   all seven kinds at once: mean KL {kl('S:16'):.4f}. The seven separate values add up to {sum(kl(v) for v in kinds):.4f}.")
        print("   18 of the 42 attn_k and attn_v matrices are never read (section 1), so rounding them changes nothing.")
        print()

    print("9. Weight error against output, all variants that round every block matrix")
    full = [v for v, r in rows.items() if r.get("weights") == 3945267200 and "weight_err_db_mean" in r and "bits_per_weight" in r and r["bits_per_weight"] < 15
            and not v.startswith("clip:")]
    full.sort(key=lambda v: rows[v]["weight_err_db_mean"])
    print("   in order of weight error, smallest first")
    print("   variant | bits per weight | weight error, mean dB | mean KL")
    for v in full:
        r = rows[v]
        print(f"   {v.replace('file:', '').replace('.gguf', '')} | {r['bits_per_weight']:.3f} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.4f}")
    rho = spearman([rows[v]["weight_err_db_mean"] for v in full], [kl(v) for v in full])
    print(f"   rank correlation of weight error with mean KL over these {len(full)} variants: {rho:+.3f}")
    inv = [(a, b) for a in full for b in full
           if rows[a]["weight_err_db_mean"] < rows[b]["weight_err_db_mean"] - 1.0 and kl(a) > 2.0 * kl(b)]
    print(f"   pairs where one variant has a weight error at least 1 dB smaller and a mean KL at least twice as large: {len(inv)} of {len(full) * (len(full) - 1) // 2}")
    print()
    print("10. Verdicts: " + ", ".join(f"{n} {'holds' if ok else 'fails'}" for n, ok in VERDICTS))
    print(f"    {sum(ok for _, ok in VERDICTS)} of {len(VERDICTS)} predictions hold.")


if __name__ == "__main__":
    main()
