#!/usr/bin/env python3
"""Tables and verdicts for the run on Qwen2.5-1.5B, from results.jsonl.

Prints every table of the README and scores the predictions of PREREG.md (P1 to P6) and of
PREREG-arms.md (A1 to A7, and B1 to B7 of its extension 1) by the rules written there. It computes nothing new: every number
is a field of results.jsonl or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/model-output-qwen25-2026-11-17/summary.py > experiments/model-output-qwen25-2026-11-17/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q4_0, Q4_K_M, Q3_K_M = "file:qwen15-Q4_0.gguf", "file:qwen15-Q4_K_M.gguf", "file:qwen15-Q3_K_M.gguf"
PLAN_1 = ["unchanged", "identity", "X32:16", "VE832:16", Q4_0, "XR:16", "VE8R:16", "XR:32", "VE8R:32", Q4_K_M, Q3_K_M]
PLAN_2 = ["EB32@32:16", "ZB32@32:16", "EC32@32:16", "EB128@128:16", "ZB128@128:16", "X128:16", "VE8128:16", "EB128@R:16", "ZB128@R:16",
          "EC128@R:16", "EB128@R:32", "EB8@32:16", "PaB8@32:16", "SEB8@32:16"]
PLAN_3 = ["VE8ZR:16", "VE8ZR:32", "VE8Z32:16", "EBG128:b560s16", "EBG32:b136s8", "EBG128:b504s8", "ZBG128:b560s16"]      # extension 1 of PREREG-arms.md
GEMMA = {"X32:16": (0.117, 0.075, 1.96, 1.51), "XR:16": (0.448, 0.215, 3.20, 2.52), "XR:32": (0.118, 0.059, 2.99, 2.55)}   # X, VE8, KL gain, weight gain
HALF = {"X32:16": 0.98, "XR:16": 1.60, "XR:32": 1.50}
SCALAR_OF = {"32": "X32:16", "128": "X128:16", "R16": "XR:16", "R32": "XR:32"}


def db(x):
    return 10.0 * math.log10(x)


def main():
    rows = {}
    for line in (HERE / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["variant"]] = r

    def have(*names):
        return all(n in rows for n in names)

    def kl(n):
        return rows[n]["kl_mean"]

    def gap(ref, arm):
        """(KL of ref less KL of arm, its error, the difference in errors)."""
        d = kl(ref) - kl(arm)
        e = math.hypot(rows[ref]["kl_mean_se"], rows[arm]["kl_mean_se"])
        return d, e, d / e if e > 0 else float("inf")

    def kl_gain(ref, arm):
        return db(kl(ref) / kl(arm))

    def w(n):
        return rows[n].get("weight_err_db_mean")

    def w_gain(ref, arm):
        return w(ref) - w(arm)

    def name(n):
        return {Q4_0: "llama.cpp Q4_0", Q4_K_M: "llama.cpp Q4_K_M", Q3_K_M: "llama.cpp Q3_K_M"}.get(n, n)

    print("Qwen2.5-1.5B-Instruct on the wikitext-2 test set: 40 passages of 512 tokens, 10,200 positions scored")
    print(f"variants scored: {len(rows)} of {len(PLAN_1) + len(PLAN_2) + len(PLAN_3)}")
    missing = [n for n in PLAN_1 + PLAN_2 + PLAN_3 if n not in rows]
    if missing:
        print("not reached: " + ", ".join(name(n) for n in missing))
    print()
    print("1. Every variant, in the order scored")
    print("variant | bits per weight | weight error in dB, mean over 196 matrices | pooled | mean KL | its standard error | same top token, percent | "
          "ln of the perplexity ratio | 99th percentile of KL | seconds to build | seconds to score")
    for n in rows:
        r = rows[n]
        wm = "-" if r.get("weight_err_db_mean") is None or r["weight_err_db_mean"] == float("-inf") else f"{r['weight_err_db_mean']:.2f}"
        wp = "-" if r.get("weight_err_db_pooled") is None or r["weight_err_db_pooled"] == float("-inf") else f"{r['weight_err_db_pooled']:.2f}"
        bits = "16" if n == "unchanged" else f"{r['bits_per_weight']:.3f}"
        print(f"{name(n)} | {bits} | {wm} | {wp} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | {r['ln_ppl_ratio']:+.5f} | {r['kl_p99']:.4f} | "
              f"{r.get('build_seconds', '-')} | {r['score_seconds']}")

    print()
    print("2. The E8 Voronoi code against the scalar rule at equal bits (issue #3)")
    print("   KL gain is 10 log10 of the scalar rule's mean KL over E8's. The gap is the scalar rule's mean KL less E8's, and its error is the root of the")
    print("   sum of the squares of the two standard errors. Holds: lower by 2 errors and at least half the gemma KL gain. Shrinks: lower by 2 errors, less than half.")
    print("bit point | scalar KL | E8 KL | gap | its error | gap in errors | KL gain in dB | weight error gain in dB | ratio | gemma: scalar KL | E8 KL | KL gain | weight gain | ratio | verdict")
    verdicts = {}
    for x, v, label in (("X32:16", "VE832:16", "groups of 32, 16 levels"), ("XR:16", "VE8R:16", "per row, 16 levels"), ("XR:32", "VE8R:32", "per row, 32 levels")):
        if not have(x, v):
            print(f"{label} | not reached")
            continue
        d, e, z = gap(x, v)
        kg, wg = kl_gain(x, v), w_gain(x, v)
        g = GEMMA[x]
        verdict = "disappears" if z < 2.0 else ("holds" if kg >= HALF[x] else "shrinks")
        verdicts[x] = (verdict, kg, wg, z)
        print(f"{label} | {kl(x):.4f} | {kl(v):.4f} | {d:.4f} | {e:.4f} | {z:.1f} | {kg:.2f} | {wg:.2f} | {kg / wg:.2f} | {g[0]} | {g[1]} | {g[2]} | {g[3]} | {g[2] / g[3]:.2f} | {verdict}")

    print()
    print("3. The predictions of PREREG.md")
    print("prediction | what was measured | holds")

    def line(tag, text, ok):
        print(f"{tag} | {text} | {'not reached' if ok is None else ('yes' if ok else 'NO')}")

    if have("identity", "unchanged"):
        i, u = rows["identity"], rows["unchanged"]
        line("P1", f"identity: mean KL {i['kl_mean']:.6f}, same top token {i['same_top']:.2f} percent. unchanged: {u['kl_mean']:.6f}, {u['same_top']:.2f}",
             abs(i["kl_mean"]) < 5e-7 and i["same_top"] >= 99.99)
    else:
        line("P1", "", None)
    if have("X32:16", Q4_0):
        a, b = rows["X32:16"], rows[Q4_0]
        rel = a["kl_mean"] / b["kl_mean"] - 1.0
        line("P2", f"X32:16 mean KL {a['kl_mean']:.5f}, Q4_0 {b['kl_mean']:.5f}, {100 * rel:+.2f} percent. Top token {a['same_top']:.2f} against {b['same_top']:.2f}, "
                   f"{a['same_top'] - b['same_top']:+.2f} points", abs(rel) <= 0.03 and abs(a["same_top"] - b["same_top"]) <= 0.2)
    else:
        line("P2", "", None)
    if len(verdicts) == 3:
        line("P3", "E8 lower in " + str(sum(v[1] > 0 for v in verdicts.values())) + " of 3 pairs", all(v[1] > 0 for v in verdicts.values()))
        ratios = [v[1] / v[2] for v in verdicts.values()]
        line("P4", "ratio of KL gain to weight error gain: " + ", ".join(f"{r:.2f}" for r in ratios), all(0.5 <= r <= 1.5 for r in ratios))
    else:
        line("P3", "", None)
        line("P4", "", None)
    if have("X32:16", "XR:16", "VE832:16", "VE8R:16"):
        line("P5", f"X: {kl('X32:16'):.4f} with groups of 32, {kl('XR:16'):.4f} per row. E8: {kl('VE832:16'):.4f} and {kl('VE8R:16'):.4f}",
             kl("X32:16") < kl("XR:16") and kl("VE832:16") < kl("VE8R:16"))
    else:
        line("P5", "", None)
    if have(Q4_K_M, Q4_0, "VE832:16"):
        line("P6", f"Q4_K_M {kl(Q4_K_M):.4f}, Q4_0 {kl(Q4_0):.4f}, VE832:16 {kl('VE832:16'):.4f}, which is {kl('VE832:16') / kl(Q4_K_M):.2f} times Q4_K_M",
             kl(Q4_K_M) < kl(Q4_0) and kl("VE832:16") <= 1.5 * kl(Q4_K_M))
    else:
        line("P6", "", None)
    if len(verdicts) == 3:
        print("verdict of issue #3: " + ", ".join(f"{k} {v[0]}" for k, v in verdicts.items()))

    print()
    print("4. The predictions of PREREG-arms.md")
    print("prediction | what was measured | holds")

    def lower_by_2(tag, arm, ref):
        if not have(arm, ref):
            return line(tag, f"{arm} against {ref}", None)
        d, e, z = gap(ref, arm)
        line(tag, f"{arm} {kl(arm):.4f}, {ref} {kl(ref):.4f}: lower by {d:.4f}, {z:.1f} errors, {kl_gain(ref, arm):+.2f} dB", z >= 2.0)

    lower_by_2("A1", "EB32@32:16", "VE832:16")
    if have("EB32@32:16", "X32:16"):
        kg, wg = kl_gain("X32:16", "EB32@32:16"), w_gain("X32:16", "EB32@32:16")
        line("A2", f"EB32@32:16 over X32:16: KL gain {kg:.2f} dB, weight error gain {wg:.2f} dB, ratio {kg / wg:.2f}", 0.5 <= kg / wg <= 1.5)
    else:
        line("A2", "", None)
    lower_by_2("A3", "ZB32@32:16", "X32:16")
    if have("EB128@128:16", "X32:16"):
        line("A4", f"EB128@128:16 at {rows['EB128@128:16']['bits_per_weight']:.3f} bits {kl('EB128@128:16'):.4f}, X32:16 at 4.5 bits {kl('X32:16'):.4f}", kl("EB128@128:16") < kl("X32:16"))
    else:
        line("A4", "", None)
    lower_by_2("A5", "EB128@R:16", "VE8R:16")
    if have("X128:16", "X32:16", "VE8128:16", "VE832:16"):
        line("A6", f"X: {kl('X128:16'):.4f} with groups of 128, {kl('X32:16'):.4f} with 32. E8: {kl('VE8128:16'):.4f} and {kl('VE832:16'):.4f}",
             kl("X128:16") > kl("X32:16") and kl("VE8128:16") > kl("VE832:16"))
    else:
        line("A6", "", None)
    if have("SEB8@32:16", "EB8@32:16"):
        line("A7", f"SEB8@32:16 {kl('SEB8@32:16'):.4f}, EB8@32:16 {kl('EB8@32:16'):.4f}", kl("SEB8@32:16") > kl("EB8@32:16"))
    else:
        line("A7", "", None)

    print()
    print("4b. The predictions of extension 1 of PREREG-arms.md")
    print("prediction | what was measured | holds")
    if have("VE8ZR:16", "VE8R:16", "VE8ZR:32", "VE8R:32"):
        a, b = kl("VE8ZR:16") / kl("VE8R:16"), kl("VE8ZR:32") / kl("VE8R:32")
        line("B1", f"VE8ZR:16 {kl('VE8ZR:16'):.4f} against VE8R:16 {kl('VE8R:16'):.4f}, a share of {a:.3f}. VE8ZR:32 {kl('VE8ZR:32'):.4f} against VE8R:32 {kl('VE8R:32'):.4f}, a share of {b:.3f}",
             a < 1.0 / 3.0 and b < 1.0 / 3.0)
    else:
        line("B1", "", None)
    if have("VE8ZR:16", "XR:16", "VE8ZR:32", "XR:32"):
        z1, z2 = gap("XR:16", "VE8ZR:16"), gap("XR:32", "VE8ZR:32")
        line("B2", f"VE8ZR:16 {kl('VE8ZR:16'):.4f} against XR:16 {kl('XR:16'):.4f}: lower by {z1[0]:.4f}, {z1[2]:.1f} errors, {kl_gain('XR:16', 'VE8ZR:16'):+.2f} dB. "
                   f"VE8ZR:32 {kl('VE8ZR:32'):.4f} against XR:32 {kl('XR:32'):.4f}: lower by {z2[0]:.4f}, {z2[2]:.1f} errors, {kl_gain('XR:32', 'VE8ZR:32'):+.2f} dB",
             z1[2] >= 2.0 and z2[2] >= 2.0)
    else:
        line("B2", "", None)
    if have("VE8Z32:16", "VE832:16"):
        rel = kl("VE8Z32:16") / kl("VE832:16") - 1.0
        line("B3", f"VE8Z32:16 {kl('VE8Z32:16'):.4f}, VE832:16 {kl('VE832:16'):.4f}, {100 * rel:+.1f} percent", abs(rel) <= 0.15)
    else:
        line("B3", "", None)
    lower_by_2("B4", "EBG128:b560s16", "EB32@32:16")
    lower_by_2("B5", "EBG32:b136s8", "EB32@32:16")
    if have("EBG128:b504s8", "VE832:16", Q4_0):
        line("B6", f"EBG128:b504s8 at {rows['EBG128:b504s8']['bits_per_weight']:.3f} bits {kl('EBG128:b504s8'):.4f}. VE832:16 at 4.5 bits {kl('VE832:16'):.4f}, Q4_0 at 4.5 bits {kl(Q4_0):.4f}",
             kl("EBG128:b504s8") < kl("VE832:16") and kl("EBG128:b504s8") < kl(Q4_0))
    else:
        line("B6", "", None)
    lower_by_2("B7", "ZBG128:b560s16", "ZB32@32:16")
    if len(verdicts) == 3 and have("VE8ZR:16", "VE8ZR:32", "VE8Z32:16"):
        print("   the verdict of issue #3 again, with a codeword at zero:")
        for x, v, label in (("X32:16", "VE8Z32:16", "groups of 32, 16 levels"), ("XR:16", "VE8ZR:16", "per row, 16 levels"), ("XR:32", "VE8ZR:32", "per row, 32 levels")):
            d, e, z = gap(x, v)
            kg = kl_gain(x, v)
            print(f"   {label} | scalar KL {kl(x):.4f} | E8 KL {kl(v):.4f} | gap in errors {z:.1f} | KL gain {kg:+.2f} dB | weight error gain {w_gain(x, v):+.2f} dB | "
                  f"{'disappears' if z < 2.0 else ('holds' if kg >= HALF[x] else 'shrinks')}")

    print()
    print("5. Every arm against the scalar rule with the same group and levels: what its weight error gain became on the output")
    print("arm | bits | scalar arm | bits | weight error gain in dB | KL gain in dB | ratio | KL lower by, in errors | top token, points gained")
    for n in ["VE832:16", "VE8Z32:16", "EB32@32:16", "ZB32@32:16", "EC32@32:16", "EB8@32:16", "PaB8@32:16", "SEB8@32:16", "VE8128:16", "EB128@128:16", "ZB128@128:16",
              "VE8R:16", "VE8ZR:16", "EB128@R:16", "ZB128@R:16", "EC128@R:16", "VE8R:32", "VE8ZR:32", "EB128@R:32"]:
        group = n.split(":")[0].split("@")[-1].replace("VE8Z", "").replace("VE8", "").replace("X", "")
        key = ("R" + n.split(":")[1]) if group == "R" else group
        x = SCALAR_OF[key]
        if not have(n, x):
            continue
        d, e, z = gap(x, n)
        kg, wg = kl_gain(x, n), w_gain(x, n)
        print(f"{n} | {rows[n]['bits_per_weight']:.3f} | {x} | {rows[x]['bits_per_weight']:.3f} | {wg:+.2f} | {kg:+.2f} | {kg / wg:.2f} | {z:.1f} | {rows[n]['same_top'] - rows[x]['same_top']:+.2f}")

    print()
    print("5b. The arms with one ball per group, against X32:16 at 4.5 bits: weight error gain and what it became")
    print("arm | bits | weight error, dB | weight error gain over X32:16 | mean KL | KL gain in dB | ratio | top token, percent")
    for n in ["ZB32@32:16", "EC32@32:16", "EB32@32:16", "ZBG128:b560s16", "EBG32:b136s8", "EBG128:b560s16", "ZB128@128:16", "EB128@128:16", "EBG128:b504s8"]:
        if have(n, "X32:16"):
            kg, wg = kl_gain("X32:16", n), w_gain("X32:16", n)
            print(f"{n} | {rows[n]['bits_per_weight']:.3f} | {w(n):.2f} | {wg:+.2f} | {kl(n):.4f} | {kg:+.2f} | {kg / wg:.2f} | {rows[n]['same_top']:.2f}")

    print()
    print("6. Comparisons the second plan reports without a prediction")
    print("pair | mean KL of the first | of the second | first lower by | in errors | in dB")
    for a, b in (("EB128@128:16", "VE832:16"), ("EB8@32:16", "VE832:16"), ("PaB8@32:16", "VE832:16"), ("PaB8@32:16", "EB8@32:16"), ("EC32@32:16", "EB32@32:16"),
                 ("EC128@R:16", "EB128@R:16"), ("EB128@R:32", "VE8R:32"), ("ZB128@128:16", "VE8128:16"), ("ZB128@R:16", "VE8R:16"), ("ZB32@32:16", "VE832:16"),
                 ("EB128@128:16", Q4_0), ("EB32@32:16", Q4_K_M), ("EB128@R:16", Q3_K_M), ("EB128@128:16", "EB32@32:16"), ("EB128@R:32", Q4_K_M),
                 ("EBG128:b560s16", Q4_K_M), ("EBG32:b136s8", Q4_K_M), ("EBG128:b560s16", "EBG32:b136s8"), ("EBG128:b504s8", "EB128@128:16"), ("EBG128:b504s8", Q3_K_M),
                 ("EB128@R:16", "VE8ZR:16"), ("ZBG128:b560s16", Q4_K_M)):
        if have(a, b):
            d, e, z = gap(b, a)
            print(f"{name(a)} ({rows[a]['bits_per_weight']:.3f} bits) against {name(b)} ({rows[b]['bits_per_weight']:.3f}) | {kl(a):.4f} | {kl(b):.4f} | {d:+.4f} | {z:+.1f} | {kl_gain(b, a):+.2f}")

    print()
    print("7. Mean KL against bits: every variant, lowest KL first")
    print("variant | bits per weight | mean KL | same top token, percent")
    for n in sorted((n for n in rows if n not in ("unchanged", "identity")), key=lambda n: kl(n)):
        print(f"{name(n)} | {rows[n]['bits_per_weight']:.3f} | {kl(n):.4f} | {rows[n]['same_top']:.2f}")

    print()
    print("8. Weight error by kind of matrix, in dB, mean over the 28 layers")
    kinds = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
    print("variant | " + " | ".join(kinds))
    for n in rows:
        if isinstance(rows[n].get("weight_err_db_by_kind"), dict) and n != "identity":
            print(f"{name(n)} | " + " | ".join(f"{rows[n]['weight_err_db_by_kind'][k]:.2f}" for k in kinds))

    print()
    print("9. Not in either plan: does weight error rank the variants as the output does?")
    print("   Spearman's rank correlation between weight error and mean KL, over the variants that have both. 1 means the same order.")

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    def spearman(a, b):
        ra, rb = ranks(a), ranks(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
        return num / den if den else float("nan")

    scored = [n for n in rows if n not in ("unchanged", "identity") and isinstance(w(n), float) and math.isfinite(w(n))]
    print("set of variants | how many | mean weight error against KL | pooled weight error against KL")
    sets = [("all", scored),
            ("this repository's arms", [n for n in scored if not n.startswith("file:")]),
            ("4.49 to 4.50 bits", [n for n in scored if not n.startswith("file:") and 4.48 < rows[n]["bits_per_weight"] < 4.51]),
            ("under 4.2 bits", [n for n in scored if not n.startswith("file:") and rows[n]["bits_per_weight"] < 4.2])]
    for label, names in sets:
        if len(names) >= 3:
            ks = [kl(n) for n in names]
            print(f"{label} | {len(names)} | {spearman([w(n) for n in names], ks):.3f} | {spearman([rows[n]['weight_err_db_pooled'] for n in names], ks):.3f}")


if __name__ == "__main__":
    main()
