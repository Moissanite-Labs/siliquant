#!/usr/bin/env python3
"""Tables and verdicts for the run on SmolLM2-1.7B, from results.jsonl.

Scores the predictions S1 to S8 of PREREG.md by the rules written there, and sets each arm
beside the same arm on Qwen2.5. Every number is a field of a results file or arithmetic on
two of them.

Usage, from the root of a checkout:
  python3 experiments/model-output-smollm2-2026-11-17/summary.py > experiments/model-output-smollm2-2026-11-17/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
Q4_0, Q4_K_M = "file:smollm2-Q4_0.gguf", "file:smollm2-Q4_K_M.gguf"
ORDER = ["unchanged", "X32:16", Q4_0, Q4_K_M, "EBG128:b560s16", "EB32@32:16", "ZBG128:b560s16", "VE832:16", "XR:16", "VE8R:16", "VE8ZR:16", "EBG128:b504s8", "VE8Z32:16"]
QWEN_NAME = {Q4_0: "file:qwen15-Q4_0.gguf", Q4_K_M: "file:qwen15-Q4_K_M.gguf"}


def load(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["variant"]] = r
    return rows


def db(x):
    return 10.0 * math.log10(x)


def main():
    rows = load(HERE / "results.jsonl")
    qwen = load(HERE.parent / "model-output-qwen25-2026-11-17" / "results.jsonl")

    def have(*names):
        return all(n in rows for n in names)

    def kl(n):
        return rows[n]["kl_mean"]

    def gap(ref, arm):
        d = kl(ref) - kl(arm)
        e = math.hypot(rows[ref]["kl_mean_se"], rows[arm]["kl_mean_se"])
        return d, d / e if e > 0 else float("inf")

    def name(n):
        return {Q4_0: "llama.cpp Q4_0", Q4_K_M: "llama.cpp Q4_K_M"}.get(n, n)

    def w(n):
        v = rows[n].get("weight_err_db_mean")
        return v if isinstance(v, float) and math.isfinite(v) else None

    print("SmolLM2-1.7B-Instruct on the wikitext-2 test set: 40 passages of 512 tokens, 10,200 positions scored")
    print(f"variants scored: {len(rows)} of {len(ORDER)}")
    missing = [name(n) for n in ORDER if n not in rows]
    if missing:
        print("not reached: " + ", ".join(missing))
    print()
    print("1. Every variant, in the order of the plan, beside the same variant on Qwen2.5-1.5B")
    print("variant | bits per weight | weight error in dB, mean over 168 matrices | mean KL | its standard error | first choice kept, percent | "
          "ln of the perplexity ratio | seconds to build | seconds to score | on Qwen2.5: mean KL | first choice kept")
    for n in ORDER:
        if n not in rows:
            continue
        r = rows[n]
        q = qwen.get(QWEN_NAME.get(n, n))
        bits = "16" if n == "unchanged" else f"{r['bits_per_weight']:.3f}"
        wm = "-" if w(n) is None else f"{w(n):.2f}"
        print(f"{name(n)} | {bits} | {wm} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | {r['ln_ppl_ratio']:+.5f} | {r.get('build_seconds', '-')} | {r['score_seconds']} | "
              + (f"{q['kl_mean']:.6f} | {q['same_top']:.2f}" if q else "- | -"))

    print()
    print("2. The predictions of PREREG.md")
    print("prediction | what was measured | holds")

    def line(tag, text, ok):
        print(f"{tag} | {text} | {'not reached' if ok is None else ('yes' if ok else 'NO')}")

    def lower_by_2(tag, arm, refs):
        if not have(arm, *refs):
            return line(tag, f"{arm} against {', '.join(name(r) for r in refs)}", None)
        parts, ok = [], True
        for ref in refs:
            d, z = gap(ref, arm)
            parts.append(f"{name(ref)} {kl(ref):.4f}: lower by {d:.4f}, {z:.1f} errors, {db(kl(ref) / kl(arm)):+.2f} dB")
            ok = ok and z >= 2.0
        line(tag, f"{arm} {kl(arm):.4f}. " + ". ".join(parts), ok)

    if have("unchanged", "X32:16", Q4_0):
        u, a, b = rows["unchanged"], rows["X32:16"], rows[Q4_0]
        rel = a["kl_mean"] / b["kl_mean"] - 1.0
        line("S1", f"unchanged: mean KL {u['kl_mean']:.6f}. X32:16 {a['kl_mean']:.5f}, Q4_0 {b['kl_mean']:.5f}, {100 * rel:+.2f} percent. First choice {a['same_top']:.2f} against {b['same_top']:.2f}",
             abs(u["kl_mean"]) < 5e-7 and abs(rel) <= 0.03 and abs(a["same_top"] - b["same_top"]) <= 0.2)
    else:
        line("S1", "", None)
    lower_by_2("S2", "EBG128:b560s16", [Q4_K_M])
    if have("EBG128:b560s16", "X32:16"):
        kg, wg = db(kl("X32:16") / kl("EBG128:b560s16")), w("X32:16") - w("EBG128:b560s16")
        line("S3", f"EBG128:b560s16 over X32:16: KL gain {kg:.2f} dB, weight error gain {wg:.2f} dB, ratio {kg / wg:.2f}", 0.75 <= kg / wg <= 1.25)
    else:
        line("S3", "", None)
    lower_by_2("S4", "EB32@32:16", ["X32:16", "VE832:16"])
    lower_by_2("S5", "ZBG128:b560s16", [Q4_K_M])
    lower_by_2("S6", "VE8ZR:16", ["XR:16"])
    if have("VE8ZR:16", "VE8R:16"):
        d, z = gap("VE8R:16", "VE8ZR:16")
        line("S7", f"VE8ZR:16 {kl('VE8ZR:16'):.4f}, VE8R:16 {kl('VE8R:16'):.4f}: lower by {d:.4f}, {z:.1f} errors", z > -2.0)
    else:
        line("S7", "", None)
    lower_by_2("S8", "EBG128:b504s8", [Q4_0])

    print()
    print("3. Reported without a prediction")
    print("pair | mean KL of the first | of the second | first lower by | in errors | in dB")
    for a, b in (("XR:16", "VE8R:16"), ("VE8Z32:16", "VE832:16"), ("EBG128:b560s16", "EB32@32:16"), ("EBG128:b560s16", "ZBG128:b560s16"), ("EBG128:b560s16", Q4_0),
                 ("VE832:16", "X32:16"), ("VE8Z32:16", "X32:16"), ("EB32@32:16", Q4_K_M), ("EBG128:b504s8", Q4_K_M)):
        if have(a, b):
            d, z = gap(b, a)
            print(f"{name(a)} ({rows[a]['bits_per_weight']:.3f} bits) against {name(b)} ({rows[b]['bits_per_weight']:.3f}) | {kl(a):.4f} | {kl(b):.4f} | {d:+.4f} | {z:+.1f} | {db(kl(b) / kl(a)):+.2f}")

    print()
    print("4. The arms against X32:16: weight error gain and what it became, here and on Qwen2.5")
    print("arm | bits | weight error gain, dB | KL gain, dB | ratio | on Qwen2.5: weight error gain | KL gain | ratio")
    for n in ("VE832:16", "VE8Z32:16", "EB32@32:16", "ZBG128:b560s16", "EBG128:b560s16", "EBG128:b504s8"):
        if not have(n, "X32:16"):
            continue
        kg, wg = db(kl("X32:16") / kl(n)), w("X32:16") - w(n)
        text = "- | - | -"
        if n in qwen and "X32:16" in qwen:
            qk = db(qwen["X32:16"]["kl_mean"] / qwen[n]["kl_mean"])
            qw = qwen["X32:16"]["weight_err_db_mean"] - qwen[n]["weight_err_db_mean"]
            text = f"{qw:+.2f} | {qk:+.2f} | {qk / qw:.2f}"
        print(f"{n} | {rows[n]['bits_per_weight']:.3f} | {wg:+.2f} | {kg:+.2f} | {kg / wg:.2f} | {text}")


if __name__ == "__main__":
    main()
