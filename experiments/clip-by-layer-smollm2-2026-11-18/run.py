#!/usr/bin/env python3
"""The clip of ffn_down by layer on SmolLM2-1.7B: the rules of the record clip-by-layer-qwen25-2026-11-18 on a third model.

The plan is PREREG.md in this folder. The rule and the code that applies it are that record's (plan_of and
change_one of its run.py, imported unchanged). That program accepts the Qwen2.5 file only, so the loop that
builds, scores and writes a row is repeated here for the SmolLM2 file.

  python3 experiments/clip-by-layer-smollm2-2026-11-18/run.py --model SmolLM2-1.7B-Instruct-f16.gguf --kld base.kld --text test \
      --work DIR --out results.jsonl --bin llama-perplexity clip:5@ffn_down#0 clip:5@ffn_down
  ... --metrics-only                       what each variant changes, nothing scored
  ... --columns --imatrix M.imatrix.gguf   where the clipped entries sit by column, nothing scored
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clip_run", HERE.parent / "clip-by-layer-qwen25-2026-11-18" / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["clip_run"] = R
spec.loader.exec_module(R)
Q = R.Q
SHA256 = "cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"
MODEL = "SmolLM2-1.7B-Instruct-f16.gguf"


def columns(args):
    """For every layer: how much of what the clip removes is in its worst columns, and how active those columns' inputs are."""
    Q._imatrix_path = args.imatrix
    tensors = {t.name: t for t in Q.reader().tensors}
    layers = sorted(int(Q.BLOCK.match(n).group(1)) for n in tensors if Q.BLOCK.match(n) and Q.BLOCK.match(n).group(2) == "ffn_down")
    print("The clip of ffn_down at 5 standard deviations of a row, by column. Nothing scored.")
    print("layer | entries clipped | columns | columns with a clipped entry | share of the clip's squared error in its 10 worst columns, percent | in its 100 worst | "
          "activity of those 10 columns over the mean column")
    for layer in layers:
        name = f"blk.{layer}.ffn_down.weight"
        W = Q.matrix_of(tensors[name]).astype(np.float64)
        lim = 5.0 * W.std(axis=1, keepdims=True)
        beyond = np.abs(W) > lim
        err_col = ((W - np.clip(W, -lim, lim)) ** 2).sum(axis=0)
        worst = np.argsort(-err_col)
        act = Q.column_weights(name, W.shape[1], 1.0)
        print(f"{layer} | {int(beyond.sum())} | {W.shape[1]} | {int((beyond.sum(axis=0) > 0).sum())} | {100 * err_col[worst[:10]].sum() / err_col.sum():.1f} | "
              f"{100 * err_col[worst[:100]].sum() / err_col.sum():.1f} | {act[worst[:10]].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld")
    ap.add_argument("--text", default="test")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--imatrix")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--columns", action="store_true")
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    Q._model_path = args.model
    if Q.sha256_of(args.model) != SHA256:
        raise SystemExit("FAIL: the model file is not the SmolLM2 file the plan names")
    if args.columns:
        return columns(args)
    names = [t.name for t in Q.targets(Q.reader())]
    all_weights = sum(int(np.prod([int(d) for d in t.shape])) for t in Q.targets(Q.reader()))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["variant"], r["text"]) for r in map(json.loads, filter(str.strip, f))}
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if (v, args.text) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        if not R.CLIP.match(v):
            raise SystemExit(f"this record runs clip variants only: {v}")
        t0 = time.time()
        tasks = R.plan_of(v, names)
        replace, infos = {}, {}
        with Pool(min(args.workers, len(tasks))) as pool:
            for name, Wh, info in pool.imap_unordered(R.change_one, tasks, chunksize=1):
                replace[name], infos[name] = Wh, info
        stats = Q.stats_of(infos)
        row = {"variant": v, "model": MODEL, "text": args.text, "matrices": stats["matrices"], "weights": all_weights,
               "weights_changed": stats["weights"], "rows_changed": sum(i["rows"] for i in infos.values()),
               "weight_err_db_mean": stats["weight_err_db_mean"], "weight_err_db_pooled": stats["weight_err_db_pooled"],
               "share_out_of_range": stats["share_out_of_range"], "clipped_entries": sum(i["clipped"] for i in infos.values()),
               "rows_clipped": sum(i["rows_clipped"] for i in infos.values()), "largest_in_sd": max(i["largest_in_sd"] for i in infos.values()),
               "build_seconds": round(time.time() - t0, 1)}
        if not args.metrics_only:
            path = os.path.join(args.work, "cs_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v)[:80] + ".gguf")
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
