#!/usr/bin/env python3
"""What the gain of an importance matrix is made of: the ball arm with column weights on some kinds of matrix only,
and with weights from a matrix that read a text with no meaning.

The plan is PREREG.md in this folder. The two arms are those of the records model-output-qwen25-2026-11-17 and
imatrix-baseline-2026-11-17, built by that record's code: ZBG128:b560s16, integer levels in one ball per group of
128 weights, and hZBG128:b560s16, the same with each column weighted by (its input's mean square over the mean) ^ 0.25.

Variants:
  ZBG128:b560s16+h@KIND[,KIND...]   the matrices of the named kinds get the weighted arm, every other block matrix the plain arm.
                                    The weights are from --imatrix, the matrix of the wikitext-2 training text
  hZBG128:b560s16~rand              the weighted arm on every block matrix, with weights from --imatrix-rand, a matrix
                                    made by llama-imatrix from the text of randtext.py
  ZBG128:b560s16+h~rand@KIND[,...]  extension 1: the weighted arm with that matrix on the named kinds only

Rows are keyed by model, variant and the label of the text.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/imatrix-gain-by-kind-2026-11-18/run.py --model M.gguf --imatrix M.imatrix.gguf --kld base.kld --text test \
      [--imatrix-rand R.imatrix.gguf] --work DIR --out results.jsonl --bin llama-perplexity ZBG128:b560s16+h@attn_q,attn_k,attn_v
  ... --metrics-only    build every variant and record what it changes, without scoring
  ... --forecast        for every block matrix and each arm: the weight error, and the weight error with each column
                        counted by its input's mean square over the wikitext-2 training text. By kind. Nothing scored.
                        With --imatrix-rand the arm weighted by that matrix is added, counted by the same yardstick.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qwen_run", HERE.parent / "model-output-qwen25-2026-11-17" / "run.py")
Q = importlib.util.module_from_spec(spec)
sys.modules["qwen_run"] = Q
spec.loader.exec_module(Q)
Q.MODELS["cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"] = "SmolLM2-1.7B-Instruct-f16.gguf"
ACCEPTED = ("qwen2.5-1.5b-instruct-fp16.gguf", "SmolLM2-1.7B-Instruct-f16.gguf")
ARM, HARM = "ZBG128:b560s16", "hZBG128:b560s16"
RARM = "hZBG128:b560s16~rand"
PART = re.compile(r"^ZBG128:b560s16\+h(?P<rand>~rand)?@(?P<kinds>\w+(?:,\w+)*)$")
_paths = {}


def use_matrix(which):
    """Point the record's code at one of the two importance matrices. Called before the workers are started."""
    Q._imatrix_path, Q._imatrix = _paths[which], None


def kinds_of(variant):
    if variant == RARM:
        return set(Q.KINDS)
    m = PART.match(variant)
    if not m:
        raise SystemExit(f"unknown variant {variant}")
    kinds = m.group("kinds").split(",")
    if any(k not in Q.KINDS for k in kinds) or len(set(kinds)) != len(kinds):
        raise SystemExit(f"{variant} names a kind that is not one of {Q.KINDS}, or names one twice")
    return set(kinds)


def round_part(task):
    variant, name = task
    arm = HARM if Q.BLOCK.match(name).group(2) in kinds_of(variant) else ARM
    name, Wh, info = Q.round_one((arm, name))
    info["weighted"] = arm == HARM
    return name, Wh, info


def forecast_one(name):
    """Each arm on one matrix: its squared error, plain and with each column counted by its input's mean square over the real text."""
    t = next(t for t in Q.reader().tensors if t.name == name)
    W = Q.matrix_of(t).astype(np.float64)
    use_matrix("real")
    act = Q.column_weights(name, W.shape[1], 1.0)
    out = {"size": int(W.size), "energy": float((W ** 2).sum()), "energy_act": float(((W ** 2).sum(axis=0) * act).sum()),
           "top10": float(np.sort(act)[-10:].sum() / act.sum())}
    for label, arm, which in (("plain", ARM, "real"), ("real", HARM, "real")) + ((("rand", HARM, "rand"),) if "rand" in _paths else ()):
        use_matrix(which)
        _, Wh, _ = Q.round_one((arm, name))
        E = ((W - Wh.astype(np.float64)) ** 2).sum(axis=0)
        out[label] = (float(E.sum()), float((E * act).sum()))
        if which == "rand":
            other = Q.column_weights(name, W.shape[1], 1.0)
            out["rand_top10"] = float(act[np.argsort(-other, kind="stable")[:10]].sum() / act.sum())
    return name, out


def forecast(args, names):
    rows = {}
    with Pool(args.workers) as pool:
        for name, out in pool.imap_unordered(forecast_one, names, chunksize=1):
            rows[name] = out
    db = lambda x: 10.0 * math.log10(x)
    rand = "rand" in _paths
    arms = ("plain", "real") + (("rand",) if rand else ())
    print("Each arm on every block matrix, by kind. Nothing scored. Mean over the layers of a kind, in dB.")
    print("plain: ZBG128:b560s16. real: hZBG128:b560s16 with the matrix of the wikitext-2 training text." + (" rand: the same arm with the matrix of the text with no meaning." if rand else ""))
    print("counted by activity: each column's squared error and squared weight are multiplied by its input's mean square over the wikitext-2 training text")
    print("kind | matrices | share of the block weights, percent | share of the input activity in the 10 most active columns, percent"
          + (" | the same share for the 10 columns the other matrix puts first" if rand else "")
          + " | " + " | ".join(f"weight error, {a}" for a in arms) + " | " + " | ".join(f"counted by activity, {a}" for a in arms)
          + " | gain of real over plain, counted by activity, dB" + (" | gain of rand over plain" if rand else ""))
    total = sum(r["size"] for r in rows.values())
    for kind in Q.KINDS:
        rs = [r for n, r in rows.items() if Q.BLOCK.match(n).group(2) == kind]
        a = {arm: np.mean([db(r[arm][0] / r["energy"]) for r in rs]) for arm in arms}
        b = {arm: np.mean([db(r[arm][1] / r["energy_act"]) for r in rs]) for arm in arms}
        print(f"{kind} | {len(rs)} | {100 * sum(r['size'] for r in rs) / total:.2f} | {100 * np.mean([r['top10'] for r in rs]):.1f}"
              + (f" | {100 * np.mean([r['rand_top10'] for r in rs]):.1f}" if rand else "")
              + " | " + " | ".join(f"{a[x]:.2f}" for x in arms) + " | " + " | ".join(f"{b[x]:.2f}" for x in arms)
              + f" | {b['plain'] - b['real']:.2f}" + (f" | {b['plain'] - b['rand']:.2f}" if rand else ""))
    print()
    for arm in arms[1:]:
        print(f"layer by layer, what the arm {arm} gains over plain, counted by activity, dB")
        print("layer | " + " | ".join(Q.KINDS))
        for l in sorted({int(Q.BLOCK.match(n).group(1)) for n in rows}):
            cells = []
            for kind in Q.KINDS:
                r = rows[f"blk.{l}.{kind}.weight"]
                cells.append(f"{db(r['plain'][1] / r['energy_act']) - db(r[arm][1] / r['energy_act']):.2f}")
            print(f"{l} | " + " | ".join(cells))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True, help="the importance matrix of the wikitext-2 training text")
    ap.add_argument("--imatrix-rand", help="the importance matrix of the text with no meaning")
    ap.add_argument("--kld")
    ap.add_argument("--text", default="test", help="a label for the text the reference outputs are of")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--forecast", action="store_true")
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    Q._model_path = args.model
    _paths["real"] = args.imatrix
    if args.imatrix_rand:
        _paths["rand"] = args.imatrix_rand
    model = Q.MODELS.get(Q.sha256_of(args.model))
    if model not in ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    names = [t.name for t in Q.targets(Q.reader())]
    if args.forecast:
        return forecast(args, names)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"], r.get("text", "test")) for r in map(json.loads, filter(str.strip, f))}     # a row of a llama.cpp file, written by the earlier record's program, names no text
    if not args.metrics_only:
        os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if (model, v, args.text) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        kinds_of(v)
        rand = v == RARM or PART.match(v).group("rand") is not None
        if rand and "rand" not in _paths:
            raise SystemExit(f"{v} needs --imatrix-rand")
        use_matrix("rand" if rand else "real")
        t0 = time.time()
        replace, infos = {}, {}
        with Pool(args.workers) as pool:
            for name, Wh, info in pool.imap_unordered(round_part, [(v, n) for n in names], chunksize=1):
                infos[name] = info
                if not args.metrics_only:
                    replace[name] = Wh
        stats = Q.stats_of(infos)
        row = {"variant": v, "model": model, "text": args.text, "matrix": "no meaning" if rand else "wikitext-2 training text",
               "matrices": stats["matrices"], "weights": stats["weights"],
               "matrices_weighted": sum(1 for i in infos.values() if i["weighted"]),
               "weights_weighted": sum(i["size"] for i in infos.values() if i["weighted"]),
               "bits_per_weight": stats["bits_per_weight"], "weight_err_db_mean": stats["weight_err_db_mean"],
               "weight_err_db_pooled": stats["weight_err_db_pooled"], "weight_err_db_by_kind": stats["weight_err_db_by_kind"],
               "share_out_of_range": stats["share_out_of_range"], "build_seconds": round(time.time() - t0, 1)}
        if not args.metrics_only:
            path = os.path.join(args.work, "ik_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v)[:80] + ".gguf")
            Q.ggufedit.write_copy(Q.reader(), path, replace)
            del replace
            t1 = time.time()
            scores, _ = Q.evalkl.score(path, args.kld, binary=args.bin, extra=("-t", str(args.threads)))
            os.remove(path)
            row.update(scores)
            row["score_seconds"] = round(time.time() - t1, 1)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
