#!/usr/bin/env python3
"""Tables and verdicts for the clip of ffn_down by layer on SmolLM2, from results.jsonl, weights_only.jsonl and columns.out.

Scores the predictions T1 to T5 of PREREG.md by the rules written there.

Usage, from the root of a checkout:
  python3 experiments/clip-by-layer-smollm2-2026-11-18/summary.py > experiments/clip-by-layer-smollm2-2026-11-18/results.out
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qsum", HERE.parent / "clip-by-layer-qwen25-2026-11-18" / "summary.py")
S = importlib.util.module_from_spec(spec)
sys.modules["qsum"] = S
spec.loader.exec_module(S)
LAYERS = 24
ENDS = (0, 1, 22, 23)


def load(path, text):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["text"] == text:
                    rows[r["variant"]] = r
    return rows


def activity():
    """layer -> activity of its 10 worst columns over the mean column, from columns.out."""
    out = {}
    for line in (HERE / "columns.out").read_text().splitlines():
        p = [x.strip() for x in line.split("|")]
        if len(p) == 7 and p[0].isdigit():
            out[int(p[0])] = (float(p[4]), float(p[6]))
    return out


def main():
    scored, weights, act = load(HERE / "results.jsonl", "test"), load(HERE / "weights_only.jsonl", "weights"), activity()
    single = {l: scored.get(f"clip:5@ffn_down#{l}") for l in range(LAYERS)}
    have = [l for l in range(LAYERS) if single[l]]
    every = scored.get("clip:5@ffn_down")
    print("The clip of ffn_down at 5 standard deviations of a row, one layer of SmolLM2-1.7B at a time: wikitext-2 test text, 10,200 positions")
    print(f"single layers scored: {len(have)} of {LAYERS}. All layers at once: {'scored' if every else 'not scored'}")
    for l in have:
        w = weights[f"clip:5@ffn_down#{l}"]
        if abs(w["weight_err_db_mean"] - single[l]["weight_err_db_mean"]) > 1e-9 or w["clipped_entries"] != single[l]["clipped_entries"]:
            raise SystemExit(f"FAIL: layer {l}: the scored matrix is not the one of weights_only.jsonl")
    total = sum(single[l]["kl_mean"] for l in have)
    print()
    print("layer | weight error of the clip, dB | entries clipped | share of the clip's error in its 10 worst columns, percent | activity of those columns over the mean | "
          "mean KL | its standard error | KL in its errors | share of the sum, percent | first choice kept, percent")
    for l in have:
        r = single[l]
        print(f"{l} | {r['weight_err_db_mean']:.2f} | {r['clipped_entries']} | {act[l][0]:.1f} | {act[l][1]:.2f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | "
              f"{r['kl_mean'] / r['kl_mean_se'] if r['kl_mean_se'] else 0:.1f} | {100 * r['kl_mean'] / total:.1f} | {r['same_top']:.2f}")
    verdict = {}
    if len(have) == LAYERS:
        order = sorted(have, key=lambda l: -single[l]["kl_mean"])
        top = order[0]
        kl = [single[l]["kl_mean"] for l in have]
        rho_w = S.spearman([single[l]["weight_err_db_mean"] for l in have], kl)
        rho_a = S.spearman([act[l][1] for l in have], kl)
        ends = sum(single[l]["kl_mean"] for l in ENDS) / total
        print()
        print(f"sum of the {LAYERS} single-layer KLs: {total:.6f}")
        print(f"largest: layer {top}, {single[top]['kl_mean']:.6f}, {100 * single[top]['kl_mean'] / total:.1f} percent of the sum")
        print("the four largest: " + ", ".join(f"layer {l} {single[l]['kl_mean']:.6f}" for l in order[:4])
              + f", together {100 * sum(single[l]['kl_mean'] for l in order[:4]) / total:.1f} percent")
        print(f"layers 0, 1, 22 and 23, the first two and the last two: {100 * ends:.1f} percent of the sum")
        print(f"layers under 5 of their own errors: {sum(1 for l in have if single[l]['kl_mean'] < 5 * single[l]['kl_mean_se'])}")
        print(f"Spearman's rank correlation over the {LAYERS} layers: weight error against KL {rho_w:.3f}, activity of the 10 worst columns against KL {rho_a:.3f}")
        verdict["T2"] = ends > 0.5
        verdict["T3"] = single[top]["kl_mean"] <= 0.5 * total
        verdict["T4"] = rho_w >= 0.5
    if every:
        print()
        print(f"all {LAYERS} layers at once: weight error {every['weight_err_db_mean']:.2f} dB, mean KL {every['kl_mean']:.6f}, standard error {every['kl_mean_se']:.6f}, "
              f"first choice kept {every['same_top']:.2f} percent")
        verdict["T1"] = every["kl_mean"] < 0.06
        if len(have) == LAYERS:
            print(f"the sum over the KL of all at once: {total / every['kl_mean']:.3f}")
            verdict["T5"] = 0.7 <= total / every["kl_mean"] <= 1.4
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    for k in ("T1", "T2", "T3", "T4", "T5"):
        print(f"{k} | {'not scored' if k not in verdict else ('yes' if verdict[k] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of 5")


if __name__ == "__main__":
    main()
