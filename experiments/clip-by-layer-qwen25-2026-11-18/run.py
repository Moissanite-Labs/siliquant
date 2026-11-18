#!/usr/bin/env python3
"""Change some block matrices of Qwen2.5 by a rule of the output-budget record and score the model.

For the two runs of issue #20 that need Qwen2.5. The rules are those of
experiments/output-budget-gemma-e4b-2026-11-17, whose own program refuses any file but gemma's:

  clip:C@KIND#L[,L...]   every entry of the matrices of that kind in those layers that is beyond C
                         standard deviations of its row (divisor n) is set to that limit. Nothing is
                         rounded. Without #L, every layer.
  X32:Q@KIND             the rule of Q4_0 on the matrices of one kind: one scale per 32 weights from
                         the largest entry, Q levels (harness/tq.py maxscale_round). X32:Q is every kind.
  X32mix:KIND=Q,...      the same rule with a level count for each of the seven kinds.

Every matrix that a variant does not name stays as it is in the 16-bit reference file. A changed
matrix is written as 16-bit floats.

Rows are keyed by model, variant and the label of the text, so one results file can hold the same
variant on two texts.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/clip-by-layer-qwen25-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --kld base.kld --text test \
      --work DIR --out results.jsonl --bin llama-perplexity clip:5@ffn_down#0 clip:5@ffn_down
  ... --metrics-only    build every variant and record what it changes, without scoring
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

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
CLIP = re.compile(r"^clip:(?P<c>\d*\.?\d+)@(?P<kind>\w+)(?:#(?P<layers>\d+(?:,\d+)*))?$")
ARM = re.compile(r"^X32:(?P<q>\d+)(?:@(?P<kind>\w+))?$")
MIX = re.compile(r"^X32mix:(?P<levels>.+)$")


def plan_of(variant, names):
    """What a variant does: a list of (matrix name, rule) with rule ("clip", C) or ("X32", Q)."""
    where = {n: (int(Q.BLOCK.match(n).group(1)), Q.BLOCK.match(n).group(2)) for n in names}
    m = CLIP.match(variant)
    if m:
        if m.group("kind") not in KINDS:
            raise SystemExit(f"unknown kind in {variant}")
        layers = None if m.group("layers") is None else {int(x) for x in m.group("layers").split(",")}
        out = [(n, ("clip", float(m.group("c")))) for n in names if where[n][1] == m.group("kind") and (layers is None or where[n][0] in layers)]
        if layers is not None and len(out) != len(layers):
            raise SystemExit(f"{variant} names a layer the model does not have")
        return out
    m = ARM.match(variant)
    if m:
        if m.group("kind") is not None and m.group("kind") not in KINDS:
            raise SystemExit(f"unknown kind in {variant}")
        return [(n, ("X32", int(m.group("q")))) for n in names if m.group("kind") in (None, where[n][1])]
    m = MIX.match(variant)
    if m:
        levels = dict(p.split("=") for p in m.group("levels").split(","))
        if tuple(levels) != KINDS:
            raise SystemExit(f"{variant} must name the seven kinds in order")
        return [(n, ("X32", int(levels[where[n][1]]))) for n in names]
    raise SystemExit(f"unknown variant {variant}")


def change_one(task):
    name, (rule, value) = task
    if rule == "X32":
        _, Wh, info = Q.round_one((f"X32:{int(value)}", name))
        info["rows"] = int(Wh.shape[0])
        return name, Wh, info
    t = next(t for t in Q.reader().tensors if t.name == name)
    W = Q.matrix_of(t)
    lim = value * W.std(axis=1, keepdims=True)
    Wh = np.clip(W, -lim, lim).astype(np.float16)
    W64 = W.astype(np.float64)
    energy = float(np.sum(W64 ** 2))
    beyond = np.abs(W) > lim
    return name, Wh, {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2)) / energy, "bits": 16.0, "overload": float(beyond.mean()),
                      "energy": energy, "size": int(W.size), "rows": int(W.shape[0]), "clipped": int(beyond.sum()),
                      "rows_clipped": int(beyond.any(axis=1).sum()), "largest_in_sd": float((np.abs(W) / W.std(axis=1, keepdims=True)).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--text", required=True, help="a label for the text the reference outputs are of: test or valid")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    Q._model_path = args.model
    digest = Q.sha256_of(args.model)
    if Q.MODELS.get(digest) != "qwen2.5-1.5b-instruct-fp16.gguf":
        raise SystemExit("FAIL: the model file is not the Qwen2.5 file the plan names")
    names = [t.name for t in Q.targets(Q.reader())]
    all_weights = None
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"], r["text"]) for r in map(json.loads, filter(str.strip, f))}
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if (Q.MODELS[digest], v, args.text) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        tasks = plan_of(v, names)
        replace, infos = {}, {}
        with Pool(min(args.workers, len(tasks))) as pool:
            for name, Wh, info in pool.imap_unordered(change_one, tasks, chunksize=1):
                replace[name], infos[name] = Wh, info
        stats = Q.stats_of(infos)
        if all_weights is None:
            all_weights = sum(int(np.prod([int(d) for d in t.shape])) for t in Q.targets(Q.reader()))
        changed = stats["weights"]
        row = {"variant": v, "model": Q.MODELS[digest], "text": args.text, "matrices": stats["matrices"], "weights": all_weights,
               "weights_changed": changed, "rows_changed": sum(i["rows"] for i in infos.values()),
               "bits_per_weight_changed": stats["bits_per_weight"],
               "bits_per_weight": (stats["bits_per_weight"] * changed + 16.0 * (all_weights - changed)) / all_weights,
               "weight_err_db_mean": stats["weight_err_db_mean"], "weight_err_db_pooled": stats["weight_err_db_pooled"],
               "share_out_of_range": stats["share_out_of_range"]}
        if CLIP.match(v):
            row.update({"clipped_entries": sum(i["clipped"] for i in infos.values()), "rows_clipped": sum(i["rows_clipped"] for i in infos.values()),
                        "largest_in_sd": max(i["largest_in_sd"] for i in infos.values())})
        row["build_seconds"] = round(time.time() - t0, 1)
        if args.metrics_only:
            print(json.dumps(row), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            continue
        path = os.path.join(args.work, "cl_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v)[:80] + ".gguf")
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
