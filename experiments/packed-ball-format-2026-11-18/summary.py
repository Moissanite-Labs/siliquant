#!/usr/bin/env python3
"""Tables and verdicts for the packed format, from results.jsonl of this folder and the rows of three other records.

Scores the predictions F1 to F5 of PREREG.md by the rules written there. Every number is a field
of a results file or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/packed-ball-format-2026-11-18/summary.py > experiments/packed-ball-format-2026-11-18/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
MODELS = (
    ("Qwen2.5-1.5B-Instruct", "qwen2.5-1.5b", "qwen15", EXP / "model-output-qwen25-2026-11-17" / "results.jsonl"),
    ("SmolLM2-1.7B-Instruct", "smollm2", "smollm2", EXP / "model-output-smollm2-2026-11-17" / "results.jsonl"),
)
PAIRS = (
    ("F2", "PZ32:b136s8", "Q4_K_M"),
    ("F3", "hPZ32:b136s8", "Q4_K_M with the matrix"),
    ("", "PZ32:b136s8", "Q4_0"),
    ("", "PZ32:b136s8", "ZBG128:b560s16"),
    ("", "hPZ32:b136s8", "hZBG128:b560s16"),
    ("", "hPZ32:b136s8", "PZ32:b136s8"),
)


def load(path, model):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if model in r["model"].lower():
                    rows[r["variant"]] = r
    return rows


def main():
    print("A packed file for the integer ball arm: the model read back from the file, wikitext-2 test text, 10,200 positions")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    ok = {k: [] for k in ("F1", "F2", "F3", "F4", "F5")}
    for title, key, short, base_path in MODELS:
        mine = load(HERE / "results.jsonl", key)
        base = load(base_path, key)
        imat = load(EXP / "imatrix-baseline-2026-11-17" / "results.jsonl", key)
        rows = {"PZ32:b136s8": mine.get("PZ32:b136s8"), "hPZ32:b136s8": mine.get("hPZ32:b136s8"),
                "ZBG128:b560s16": base.get("ZBG128:b560s16"), "hZBG128:b560s16": imat.get("hZBG128:b560s16"),
                "EBG32:b136s8": base.get("EBG32:b136s8"),
                "Q4_0": base.get(f"file:{short}-Q4_0.gguf"), "Q4_K_M": base.get(f"file:{short}-Q4_K_M.gguf"),
                "Q4_K_M with the matrix": imat.get(f"file:{short}-Q4_K_M-imat.gguf")}
        print()
        print(title)
        print("variant | bits per weight | of a file | weight error in dB, mean over the block matrices | mean KL | its standard error | first choice kept, percent")
        for name, r in rows.items():
            if not r:
                print(f"{name} | not scored")
                continue
            stored = "yes" if "file_bytes" in r or name.startswith("Q4") else "no, counted"
            print(f"{name} | {r['bits_per_weight']:.4f} | {stored} | {r['weight_err_db_mean']:.2f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f}")
        print("the files | bytes | stored bits per weight | read back exactly | share of weights stored as zero | share of blocks shrunk | seconds to pack on 4 cores | "
              "processor seconds to decode on one core | million weights a second")
        for name in ("PZ32:b136s8", "hPZ32:b136s8"):
            r = rows[name]
            if not r:
                continue
            print(f"{name} | {r['file_bytes']} | {r['stored_bits_per_weight']:.5f} | {'yes' if r['read_back_exact'] else 'NO'} | {r['share_of_weights_zero']:.4f} | "
                  f"{r['share_out_of_range']:.5f} | {r['pack_seconds']:.0f} | {r['decode_cpu_seconds_one_core']:.1f} | {r['decode_million_weights_per_second']:.1f}")
            ok["F1"].append(r["read_back_exact"] and r["stored_bits_per_weight"] <= (4.52 if name.startswith("h") else 4.501))
            ok["F5"].append(r["decode_million_weights_per_second"] >= 15.0)
            print(f"   SHA-256 of the file: {r['file_sha256']}")
        print("comparison | mean KL of the first | of the second | first lower by | in errors | ratio of the first to the second")
        for tag, a, b in PAIRS:
            if rows[a] and rows[b]:
                d = rows[b]["kl_mean"] - rows[a]["kl_mean"]
                e = d / math.hypot(rows[a]["kl_mean_se"], rows[b]["kl_mean_se"])
                ratio = rows[a]["kl_mean"] / rows[b]["kl_mean"]
                print(f"{a} against {b} | {rows[a]['kl_mean']:.4f} | {rows[b]['kl_mean']:.4f} | {d:+.4f} | {e:+.1f} | {ratio:.3f}")
                if tag:
                    ok[tag].append(e >= 2)
                if b in ("ZBG128:b560s16", "hZBG128:b560s16"):
                    ok["F4"].append(ratio <= 1.25)
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    need = {"F1": 4, "F2": 2, "F3": 2, "F4": 4, "F5": 4}
    held = 0
    for k in sorted(ok):
        v = "NO" if not all(ok[k]) else ("yes" if len(ok[k]) == need[k] else f"so far, {len(ok[k])} of {need[k]} scored")
        held += v == "yes"
        print(f"{k} | {v}")
    print(f"hold: {held} of {len(ok)}")


if __name__ == "__main__":
    main()
