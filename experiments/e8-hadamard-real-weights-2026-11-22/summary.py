#!/usr/bin/env python3
"""Tables and verdicts for E8 turned by a Hadamard matrix, from results.jsonl, metrics.jsonl and the rows of four earlier records.

results.jsonl holds the rows scored here. metrics.jsonl holds, for the arms that earlier records scored, what
this program's weights-only pass says they do to the small weights. Scores the predictions H1 to H9 and the
check C1 of PREREG.md by the rules written there. Every number is a field of one of those files or arithmetic
on two of them.

Usage, from the root of a checkout:
  python3 experiments/e8-hadamard-real-weights-2026-11-22/summary.py > experiments/e8-hadamard-real-weights-2026-11-22/results.out
  python3 experiments/e8-hadamard-real-weights-2026-11-22/summary.py --replay-check     exit 1 unless the check C1 holds
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
MODELS = (("Qwen2.5-1.5B", "qwen2.5-1.5b-instruct-fp16.gguf", "qwen15", "model-output-qwen25-2026-11-17/results.jsonl"),
          ("SmolLM2-1.7B", "SmolLM2-1.7B-Instruct-f16.gguf", "smollm2", "model-output-smollm2-2026-11-17/results.jsonl"),
          ("Llama-3.2-1B", "Llama-3.2-1B-Instruct-f16.gguf", "llama32", "third-model-llama32-2026-11-18/results-wikitext.jsonl"))
ARMS = (("ZBG", "integer levels"), ("EBG", "E8"), ("HBG", "E8 turned"))
REPLAY = ("SmolLM2-1.7B-Instruct-f16.gguf", "EBG128:b560s16")


def rows(path, model, scored=True):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model and (("kl_mean" in r) or not scored) and r.get("text", "test") in ("test", "wikitext"):
                    out[r["variant"]] = r
    return out


def gap(a, b):
    """How far the KL of row a is under the KL of row b, in errors of the gap."""
    return (b["kl_mean"] - a["kl_mean"]) / math.sqrt(a["kl_mean_se"] ** 2 + b["kl_mean_se"] ** 2)


def load():
    """For every model: the scored rows by variant with where each is from, the weights-only rows, and the Q4_K_M row."""
    out = {}
    for name, model, tag, earlier in MODELS:
        mine = rows(HERE / "results.jsonl", model)
        old = rows(EXP / earlier, model)
        if tag == "llama32":
            old.update({v: r for v, r in rows(EXP / "third-model-llama32-2026-11-18" / "results-loud.jsonl", model).items() if v == "ZBG128:b560s16+K10"})
        scored = {v: (r, "an earlier record") for v, r in old.items() if v.startswith(("ZBG128:b560s16", "EBG128:b560s16")) and (v.endswith("s16") or v.endswith("+K10"))}
        replay = None
        for v, r in mine.items():
            if (model, v) == REPLAY:
                replay = r
            else:
                scored[v] = (r, "this record")
        weights = rows(HERE / "metrics.jsonl", model, scored=False)
        weights.update(mine)
        out[name] = {"scored": scored, "weights": weights, "q": old.get(f"file:{tag}-Q4_K_M.gguf"), "replay": replay, "old": old}
    return out


def replay_check(data):
    """C1: the E8 arm on SmolLM2 scored again through this program against its committed row. None when not yet run."""
    r, old = data["SmolLM2-1.7B"]["replay"], data["SmolLM2-1.7B"]["old"].get(REPLAY[1])
    if r is None or old is None:
        return None, None, None
    diff = r["kl_mean"] - old["kl_mean"]
    return abs(diff) <= 0.1 * old["kl_mean_se"], diff, old


def main():
    data = load()
    ok, diff, old = replay_check(data)
    if "--replay-check" in sys.argv:
        print("C1:", "not run" if ok is None else f"KL differs from the committed row by {diff:+.6f}, where a tenth of its standard error is {0.1 * old['kl_mean_se']:.6f}: {'holds' if ok else 'FAILS'}")
        raise SystemExit(0 if ok else 1)
    verdict = {}
    print("E8 turned by the Hadamard matrix of order 8, against E8 and integer levels, each in one ball per group of 128 weights. No calibration data.")
    print("wikitext-2 test text, 10,200 positions. An error of a gap is the root of the sum of the squares of the two standard errors.")
    print()
    print("model | bits | columns kept | arm | variant | bits a weight | weight error, dB | mean KL | its standard error | first choice kept, percent | scored in | under Q4_K_M with no matrix by, errors")
    for name, model, tag, _ in MODELS:
        d = data[name]
        for bits, keep in (("560", ""), ("560", "+K10"), ("368", "")):
            for arm, label in ARMS:
                v = f"{arm}128:b{bits}s16{keep}"
                if v in d["scored"]:
                    r, where = d["scored"][v]
                    werr = r.get("weight_err_db_pooled")
                    under = f"{gap(r, d['q']):.1f}" if bits == "560" and d["q"] else ""
                    print(f"{name} | {'4.5' if bits == '560' else '3.0'} | {'ten of each ffn_down' if keep else 'none'} | {label} | {v} | {r.get('bits_per_weight', r.get('bits_per_weight_changed')):.5f} | "
                          f"{werr:.3f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | {where} | {under}")
        if d["q"]:
            q = d["q"]
            print(f"{name} | 4.8 | none | llama.cpp Q4_K_M, no matrix | file | {q['bits_per_weight']:.5f} | {q['weight_err_db_pooled']:.3f} | {q['kl_mean']:.6f} | {q['kl_mean_se']:.6f} | {q['same_top']:.2f} | an earlier record | ")
    print()
    print("Gaps, in errors. Positive: the first arm has the lower KL.")
    print("model | bits | columns kept | turned under E8 | turned under integer levels | E8 under integer levels | lowest arm | turned within 2 errors of the lowest")
    best_ok = {}
    for name, model, tag, _ in MODELS:
        d = data[name]
        for bits, keep in (("560", ""), ("560", "+K10"), ("368", "")):
            got = {arm: d["scored"].get(f"{arm}128:b{bits}s16{keep}", (None, None))[0] for arm, _ in ARMS}
            if not all(got.values()):
                continue
            z, e, h = got["ZBG"], got["EBG"], got["HBG"]
            low = min(got, key=lambda a: got[a]["kl_mean"])
            near = low == "HBG" or -gap(h, got[low]) < 2
            print(f"{name} | {'4.5' if bits == '560' else '3.0'} | {'ten of each ffn_down' if keep else 'none'} | {gap(h, e):.1f} | {gap(h, z):.1f} | {gap(e, z):.1f} | {dict(ARMS)[low]} | {'yes' if near else 'no'}")
            key = (name, bits, keep)
            best_ok[key] = near
            if key == ("SmolLM2-1.7B", "560", ""):
                verdict["H1"], verdict["H2"] = gap(h, e) >= 2, gap(h, z) >= 2
            if key == ("Qwen2.5-1.5B", "560", ""):
                verdict["H3"] = abs(gap(h, e)) < 2 and gap(h, z) >= 2
            if key == ("Llama-3.2-1B", "560", "+K10"):
                verdict["H4"], verdict["H5"] = -gap(h, e) < 2, gap(h, z) >= 2
            if key == ("SmolLM2-1.7B", "368", ""):
                verdict["H9"] = gap(h, e) >= 2
    need = [("Qwen2.5-1.5B", "560", ""), ("SmolLM2-1.7B", "560", ""), ("Llama-3.2-1B", "560", "+K10")]
    if all(k in best_ok for k in need):
        verdict["H6"] = all(best_ok[k] for k in need)
    print()
    print("What each arm does to the small weights, from the weights alone. Small: under 0.05, 0.1 or 0.2 of the root mean square of the weight's matrix.")
    print("model | bits | columns kept | arm | weight error, dB | stored as zero, percent of all weights | of attention's | of the feed-forward's | "
          "under 0.05: percent of weights, stored as zero percent, error over size dB | under 0.1: the same | under 0.2: the same")
    h7, h8 = [], []
    for name, model, tag, _ in MODELS:
        d = data[name]
        for bits, keep in (("560", ""), ("560", "+K10"), ("368", "")):
            got = {arm: d["weights"].get(f"{arm}128:b{bits}s16{keep}") for arm, _ in ARMS}
            for arm, label in ARMS:
                r = got[arm]
                if r is None or "small_weights" not in r:
                    continue
                cells = []
                for dd in ("0.05", "0.1", "0.2"):
                    s = r["small_weights"][dd]["all"]
                    cells.append(f"{s['percent_of_weights']:.2f}, {s['stored_zero_percent']:.1f}, {s['err_over_size_db']:.2f}")
                z = r["stored_zero_percent"]
                print(f"{name} | {'4.5' if bits == '560' else '3.0'} | {'ten of each ffn_down' if keep else 'none'} | {label} | {r['weight_err_db_pooled']:.3f} | {z['all']:.2f} | {z['attention']:.2f} | {z['mlp']:.2f} | " + " | ".join(cells))
            e, h = got["EBG"], got["HBG"]
            if e and h and "small_weights" in e and "small_weights" in h and not keep:
                h7.append(abs(h["weight_err_db_pooled"] - e["weight_err_db_pooled"]) <= 0.1)
                if bits == "560":
                    h8.append(h["small_weights"]["0.05"]["all"]["stored_zero_percent"] - e["small_weights"]["0.05"]["all"]["stored_zero_percent"] >= 15.0)
    if len(h7) == 5:
        verdict["H7"] = all(h7)
    if len(h8) == 3:
        verdict["H8"] = all(h8)
    print()
    print("The check")
    if ok is None:
        print("C1 | not run")
    else:
        r = data["SmolLM2-1.7B"]["replay"]
        print(f"C1 | E8 on SmolLM2 scored again through this program: {r['kl_mean']:.6f}, and the committed row has {old['kl_mean']:.6f}. They differ by {diff:+.6f}, "
              f"where a tenth of the standard error is {0.1 * old['kl_mean_se']:.6f} | {'holds' if ok else 'FAILS'}")
    print()
    print("The predictions of PREREG.md")
    print("prediction | holds")
    for k in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"):
        print(f"{k} | {'not scored' if k not in verdict else 'yes' if verdict[k] else 'no'}")


if __name__ == "__main__":
    main()
