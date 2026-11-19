#!/usr/bin/env python3
"""Tables and verdicts for the third model, Llama-3.2-1B-Instruct, from the two results files and forecast.out.

results-wikitext.jsonl holds the variants scored on the wikitext-2 test text and results-python.jsonl the same variants
scored on the Python text of the record second-text-python-2026-11-18. Scores the predictions L1 to L6 of PREREG.md, X1 and X2 of its extension 1 and Y1 to Y5 of its
extension 2, by the rules written there. results-loud.jsonl holds the rows of extension 2, each naming its text. Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/third-model-llama32-2026-11-18/summary.py > experiments/third-model-llama32-2026-11-18/results.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXTS = (("wikitext-2 test text", "results-wikitext.jsonl"), ("Python text", "results-python.jsonl"))
ARMS = (("llama.cpp Q4_K_M", "none", "file:llama32-Q4_K_M.gguf"), ("llama.cpp Q4_K_M", "wikitext-2 training text", "file:llama32-Q4_K_M-imat.gguf"),
        ("llama.cpp Q4_K_M", "no meaning", "file:llama32-Q4_K_M-rimat.gguf"), ("one ball per group of 128", "none", "ZBG128:b560s16"),
        ("one ball per group of 128", "wikitext-2 training text", "hZBG128:b560s16"), ("one ball per group of 128", "no meaning", "hZBG128:b560s16~rand"))


def load(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "kl_mean" in r:
                    rows[r["variant"]] = r
    return rows


def gap(a, b):
    """How far the KL of row a is under the KL of row b, in errors of the gap."""
    return (b["kl_mean"] - a["kl_mean"]) / math.sqrt(a["kl_mean_se"] ** 2 + b["kl_mean_se"] ** 2)


def loud():
    """The share of the attention's input in its ten most active columns, percent, mean over the layers, from forecast.out."""
    path = HERE / "forecast.out"
    if path.exists():
        for line in path.read_text().splitlines():
            p = [x.strip() for x in line.split("|")]
            if p[0] == "attn_q" and len(p) > 4:
                return float(p[3])
    return None


def three_models(got):
    """The same six variants and three comparisons on the two earlier models, from the records that scored them, beside this model. Not part of any prediction."""
    exp = HERE.parent
    earlier = {"Qwen2.5-1.5B": ("qwen2.5-1.5b-instruct-fp16.gguf", "qwen15", "model-output-qwen25-2026-11-17"),
               "SmolLM2-1.7B": ("SmolLM2-1.7B-Instruct-f16.gguf", "smollm2", "model-output-smollm2-2026-11-17")}
    def rows_of(paths, model):
        out = {}
        for path in paths:
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        if r.get("model") == model and "kl_mean" in r:
                            out[r["variant"]] = r
        return out
    for text in ("wikitext-2 test text", "Python text"):
        print()
        print(f"Three models on the {text}. The two earlier models are from the records that scored them")
        print("model | Q4_K_M: none | wikitext matrix | noise matrix | ball arm: none | wikitext matrix | noise matrix | ball under Q4_K_M with no matrix on either side, errors | "
              "with the wikitext matrix on both sides | ball with the noise matrix under Q4_K_M with the wikitext matrix | the wikitext matrix puts the ball arm under the plain ball arm by | the noise matrix by")
        table = []
        for name, (model, tag, folder) in earlier.items():
            if text.startswith("wiki"):
                rows = rows_of((exp / folder / "results.jsonl", exp / "imatrix-baseline-2026-11-17" / "results.jsonl", exp / "imatrix-gain-parts-2026-11-18" / "results.jsonl"), model)
            else:
                rows = rows_of((exp / "second-text-python-2026-11-18" / "results.jsonl",), model)
            table.append((name, [rows.get(v) for v in (f"file:{tag}-Q4_K_M.gguf", f"file:{tag}-Q4_K_M-imat.gguf", f"file:{tag}-Q4_K_M-rimat.gguf", "ZBG128:b560s16", "hZBG128:b560s16", "hZBG128:b560s16~rand")]))
        g = got[text]
        table.append(("Llama-3.2-1B", [g.get((a, m)) for a in ("l", "o") for m in ("none", "wikitext-2 training text", "no meaning")]))
        for name, (q, qi, qr, b, bi, br) in table:
            if all((q, qi, qr, b, bi, br)):
                print(f"{name} | " + " | ".join(f"{r['kl_mean']:.4f}" for r in (q, qi, qr, b, bi, br))
                      + f" | {gap(b, q):.1f} | {gap(bi, qi):.1f} | {gap(br, qi):.1f} | {gap(bi, b):.1f} | {gap(br, b):.1f}")


def main():
    verdict = {}
    share = loud()
    print("Llama-3.2-1B-Instruct: llama.cpp's Q4_K_M and the ball arm, each with no matrix, the matrix of the wikitext-2 training text and the matrix of no meaning, on two texts. 10,200 positions each.")
    print("An error of a gap is the root of the sum of the squares of the two standard errors.")
    print(f"From forecast.out: the ten most active dimensions hold {'not measured yet' if share is None else f'{share:.1f} percent'} of what the attention reads. SmolLM2 has 61.8 and Qwen2.5 has 11.4.")
    got = {}
    for text, name in TEXTS:
        rows = load(HERE / name)
        got[text] = g = {(arm[0], matrix): rows[v] for arm, matrix, v in ARMS if v in rows}
        print()
        print(f"On the {text}")
        print("arm | matrix | bits a weight | mean KL | its standard error | first choice kept, percent")
        for arm, matrix, v in ARMS:
            if v in rows:
                r = rows[v]
                print(f"{arm} | {matrix} | {r['bits_per_weight']:.4f} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f}")
        q, qi, qr = (g.get(("l", x)) for x in ("none", "wikitext-2 training text", "no meaning"))
        b, bi, br = (g.get(("o", x)) for x in ("none", "wikitext-2 training text", "no meaning"))
        if b and q:
            print(f"no matrix on either side: the ball arm {b['kl_mean']:.6f} against Q4_K_M {q['kl_mean']:.6f}, under it by {gap(b, q):.1f} errors")
            verdict["L1" if text.startswith("wiki") else "L2"] = gap(b, q) >= 2
        if bi and qi:
            print(f"the training text's matrix on both sides: the ball arm {bi['kl_mean']:.6f} against Q4_K_M {qi['kl_mean']:.6f}, under it by {gap(bi, qi):.1f} errors")
            if text.startswith("wiki"):
                verdict["L3"] = gap(bi, qi) >= 2
        if br and qi:
            print(f"the ball arm with the matrix of no meaning {br['kl_mean']:.6f} against Q4_K_M with the training text's matrix {qi['kl_mean']:.6f}, under it by {gap(br, qi):.1f} errors")
            if text.startswith("wiki"):
                verdict["L4"] = gap(br, qi) >= 2
        if b and bi:
            print(f"the ball arm: the training text's matrix is under no matrix by {gap(bi, b):.1f} errors")
        if b and br:
            print(f"the ball arm: the matrix of no meaning is under no matrix by {gap(br, b):.1f} errors")
        if b and bi and br:
            print(f"the ball arm: the matrix of no meaning buys {100 * (b['kl_mean'] - br['kl_mean']) / (b['kl_mean'] - bi['kl_mean']):.1f} percent of what the training text's matrix buys")
        if q and qi:
            print(f"Q4_K_M: the training text's matrix is under no matrix by {gap(qi, q):.1f} errors")
        if q and qr:
            print(f"Q4_K_M: the matrix of no meaning is under no matrix by {gap(qr, q):.1f} errors")
        if q and qi and qr:
            s = (q["kl_mean"] - qr["kl_mean"]) / (q["kl_mean"] - qi["kl_mean"])
            print(f"Q4_K_M: the matrix of no meaning buys {100 * s:.1f} percent of what the training text's matrix buys")
            if text.startswith("wiki"):
                verdict["L6"] = s >= 0.5
    w, p = got["wikitext-2 test text"], got["Python text"]
    key = lambda x: ("o", x)
    if share is not None and all(key(x) in t for t in (w, p) for x in ("none", "wikitext-2 training text", "no meaning")):
        print()
        if share >= 40:
            verdict["L5"] = all(gap(t[key(x)], t[key("none")]) >= 2 for t in (w, p) for x in ("wikitext-2 training text", "no meaning"))
            print(f"L5: the share is {share:.1f}, 40 or more, so the prediction is that each matrix puts the ball arm under the plain arm by 2 errors on both texts")
        elif share <= 20:
            verdict["L5"] = all(gap(p[key(x)], p[key("none")]) < 2 for x in ("wikitext-2 training text", "no meaning"))
            print(f"L5: the share is {share:.1f}, 20 or less, so the prediction is that neither matrix puts the ball arm under the plain arm by 2 errors on the Python text")
        else:
            print(f"L5: the share is {share:.1f}, between 20 and 40, where the plan makes no prediction")
    rows = load(HERE / "results-wikitext.jsonl")
    parts = (("the attention's readers", "ZBG128:b560s16+h@attn_q,attn_k,attn_v"), ("the feed-forward readers", "ZBG128:b560s16+h@ffn_gate,ffn_up"),
             ("the writers", "ZBG128:b560s16+h@attn_output,ffn_down"))
    b, bi = w.get(key("none")), w.get(key("wikitext-2 training text"))
    if b and bi and any(v in rows for _, v in parts):
        print()
        print("Extension 1, on the wikitext-2 test text: the ball arm with columns weighted by the training text's matrix on some kinds of matrix only")
        print("weighted | mean KL | its standard error | under the plain arm by, percent | in errors of the gap | share of the whole gain, percent")
        shares = {}
        for label, v in parts:
            if v in rows:
                r = rows[v]
                shares[label] = (b["kl_mean"] - r["kl_mean"]) / (b["kl_mean"] - bi["kl_mean"])
                print(f"{label} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {100 * (1 - r['kl_mean'] / b['kl_mean']):.1f} | {gap(r, b):.1f} | {100 * shares[label]:.1f}")
        if len(shares) == 3:
            total = sum(shares.values())
            print(f"the three shares add up to {100 * total:.1f} percent of the whole gain. On SmolLM2 they were 68.5, 9.5 and 21.9, together 99.9")
            verdict["X1"] = shares["the writers"] == max(shares.values())
            verdict["X2"] = 0.75 <= total <= 1.25
    loud_rows = {}
    path = HERE / "results-loud.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "kl_mean" in r:
                    loud_rows[(r["variant"], r["text"])] = r
    if loud_rows:
        print()
        print("Extension 2: the ball arm with no matrix and columns of ffn_down left whole. K10: the ten loud columns of every ffn_down, found from the weights. A1: one column, the most active by the matrix")
        print("text | arm | reads calibration data | bits a weight | columns kept | mean KL | its standard error | first choice kept, percent | under the plain ball arm by, percent | in errors | under Q4_K_M with no matrix by, in errors")
        for (v, text), r in loud_rows.items():
            t = w if text == "wikitext" else p
            b0, q0 = t[key("none")], t[("l", "none")]
            print(f"{text} | {v} | {'yes' if '+A' in v else 'no'} | {r['bits_per_weight_changed']:.4f} | {r['kept_columns']} | {r['kl_mean']:.6f} | {r['kl_mean_se']:.6f} | {r['same_top']:.2f} | "
                  f"{100 * (1 - r['kl_mean'] / b0['kl_mean']):.1f} | {gap(r, b0):.1f} | {gap(r, q0):.1f}")
        k, k01, a01, kp = (loud_rows.get(x) for x in (("ZBG128:b560s16+K10", "wikitext"), ("ZBG128:b560s16+K10#0,1", "wikitext"), ("ZBG128:b560s16+A1#0,1", "wikitext"), ("ZBG128:b560s16+K10", "python")))
        b0, q0, bi0 = w[key("none")], w[("l", "none")], w[key("wikitext-2 training text")]
        if k:
            verdict["Y1"] = gap(k, b0) >= 2 and k["kl_mean"] <= 0.8 * b0["kl_mean"]
            verdict["Y2"] = gap(k, q0) >= 2
            print(f"wikitext: keeping the columns buys {100 * (b0['kl_mean'] - k['kl_mean']) / (b0['kl_mean'] - bi0['kl_mean']):.1f} percent of what the training text's matrix buys the ball arm")
        if k and k01 and gap(k, b0) >= 2:
            print(f"wikitext: keeping them in layers 0 and 1 only gains {(b0['kl_mean'] - k01['kl_mean']) / (b0['kl_mean'] - k['kl_mean']):.3f} of what keeping them in every layer gains")
            verdict["Y3"] = (b0["kl_mean"] - k01["kl_mean"]) >= 0.5 * (b0["kl_mean"] - k["kl_mean"])
        if kp:
            verdict["Y4"] = gap(kp, p[("l", "none")]) >= 2
        if k01 and a01:
            print(f"wikitext: one column kept in layers 0 and 1 against ten: {a01['kl_mean']:.6f} against {k01['kl_mean']:.6f}, above it by {-gap(a01, k01):.1f} errors")
            verdict["Y5"] = abs(gap(a01, k01)) < 2
    three_models(got)
    print()
    print("The predictions of PREREG.md and of its extensions 1 and 2")
    print("prediction | holds")
    names = ("L1", "L2", "L3", "L4", "L5", "L6", "X1", "X2", "Y1", "Y2", "Y3", "Y4", "Y5")
    for n in names:
        print(f"{n} | {'not scored' if n not in verdict else ('yes' if verdict[n] else 'NO')}")
    print(f"hold: {sum(1 for v in verdict.values() if v)} of {len(names)}")


if __name__ == "__main__":
    main()
