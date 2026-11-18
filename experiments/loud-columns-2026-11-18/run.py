#!/usr/bin/env python3
"""The loud columns of ffn_down: is the clip's damage in them, and does a ball arm gain by keeping them whole?

The plan is PREREG.md in this folder. The loud columns of a matrix are the ten columns in which the clip at
5 standard deviations of a row removes the most squared weight. They are found from the weights alone.

Variants:
  clip:5@ffn_down#L/only10   the clip of the record clip-by-layer-qwen25-2026-11-18 on the ffn_down of layer L,
                             applied in its ten loud columns only
  clip:5@ffn_down#L/but10    the same clip applied everywhere but in those ten columns
  ARM+K10                    a ball arm of the model-output records with, in every ffn_down, the ten loud columns
                             left as in the reference: 16 bits an entry and 16 bits for each column's number, counted
  ARM+K10#1,7                the same in the ffn_down of the named layers only
  ARM+A10                    the same with the ten most active columns, by llama.cpp's importance matrix, in place
                             of the loud ones. This arm uses calibration data.
  ARM+S#1,7                  the arm with the ffn_down of the named layers left whole, as in the reference, counted at 16 bits

ARM is ZBG128:b560s16 or hZBG128:b560s16, built by the code of the record model-output-qwen25-2026-11-17.
A kept column is set to zero before the rest of the matrix is rounded, so it takes no room in a ball, and put back after.

Rows are keyed by model, variant and the label of the text.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/loud-columns-2026-11-18/run.py --model M.gguf --kld base.kld --text test --work DIR --out results.jsonl \
      --bin llama-perplexity [--imatrix M.imatrix.gguf] VARIANT [VARIANT ...]
  ... --metrics-only          build every variant and record what it changes, without scoring
  ... --picks --imatrix F     for every ffn_down: the loud columns, the most active columns, and how many are both. Nothing scored.
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
spec = importlib.util.spec_from_file_location("clip_run", HERE.parent / "clip-by-layer-qwen25-2026-11-18" / "run.py")
R = importlib.util.module_from_spec(spec)
sys.modules["clip_run"] = R
spec.loader.exec_module(R)
Q = R.Q
Q.MODELS["cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"] = "SmolLM2-1.7B-Instruct-f16.gguf"
ACCEPTED = ("qwen2.5-1.5b-instruct-fp16.gguf", "SmolLM2-1.7B-Instruct-f16.gguf")

K = 10
LIMIT = 5.0
CLIPCOLS = re.compile(r"^clip:5@ffn_down#(?P<layer>\d+)/(?P<part>only|but)10$")
KEEP = re.compile(r"^(?P<arm>h?ZBG128:b560s16)\+(?:(?P<pick>[KA])10(?:#(?P<layers>\d+(?:,\d+)*))?|S#(?P<whole>\d+(?:,\d+)*))$")


def loud(W):
    """The numbers of the K columns in which the clip at LIMIT standard deviations of a row removes the most squared weight, in rising order."""
    W64 = W.astype(np.float64)
    lim = LIMIT * W64.std(axis=1, keepdims=True)
    lost = ((W64 - np.clip(W64, -lim, lim)) ** 2).sum(axis=0)
    return np.sort(np.argsort(-lost, kind="stable")[:K])


def active(name, cols):
    """The numbers of the K columns whose inputs are largest in mean square over the calibration text, in rising order."""
    return np.sort(np.argsort(-Q.column_weights(name, cols, 1.0), kind="stable")[:K])


def tensor(name):
    return next(t for t in Q.reader().tensors if t.name == name)


def clip_part(task):
    """The clip of one matrix, in its loud columns only or everywhere else."""
    name, part = task
    W = Q.matrix_of(tensor(name))
    lim = LIMIT * W.std(axis=1, keepdims=True)                      # as in change_one of the clip record
    inside = np.zeros(W.shape[1], dtype=bool)
    inside[loud(W)] = True
    where = inside[None, :] if part == "only" else ~inside[None, :]
    Wh = np.where(where, np.clip(W, -lim, lim), W).astype(np.float16)
    W64 = W.astype(np.float64)
    energy = float(np.sum(W64 ** 2))
    beyond = (np.abs(W) > lim) & where
    return name, Wh, {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2)) / energy, "bits": 16.0, "overload": float(beyond.mean()),
                      "energy": energy, "size": int(W.size), "clipped": int(beyond.sum()), "kept_columns": 0}


def ball(arm, name, W):
    if arm.startswith("h"):
        return Q.group_ball_round(W, "Z8", 128, 560, 16, colscale=Q.column_weights(name, W.shape[1], 0.25))
    return Q.group_ball_round(W, "Z8", 128, 560, 16)


def keep_one(task):
    """Round one block matrix by a ball arm, with some columns of an ffn_down, or all of it, left as in the reference."""
    variant, name = task
    import torch
    torch.set_num_threads(1)
    m = KEEP.match(variant)
    layer, kind = int(Q.BLOCK.match(name).group(1)), Q.BLOCK.match(name).group(2)
    W = Q.matrix_of(tensor(name))
    W64 = W.astype(np.float64)
    energy = float(np.sum(W64 ** 2))
    named = lambda s: s is None or layer in {int(x) for x in s.split(",")}
    if kind == "ffn_down" and m.group("whole") is not None and named(m.group("whole")):
        return name, W.astype(np.float16), {"rel_err": 0.0, "bits": 16.0, "overload": 0.0, "energy": energy, "size": int(W.size), "kept_columns": int(W.shape[1])}
    if kind == "ffn_down" and m.group("pick") is not None and named(m.group("layers")):
        cols = loud(W) if m.group("pick") == "K" else active(name, W.shape[1])
        rest = W.copy()
        rest[:, cols] = 0.0
        Wh, info = ball(m.group("arm"), name, rest)
        Wh[:, cols] = W[:, cols].astype(np.float16)
        rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2)) / energy
        bits = info["bits"] + (len(cols) * W.shape[0] * 16.0 + len(cols) * 16.0) / W.size
        return name, Wh, {"rel_err": rel, "bits": bits, "overload": info["overload"], "energy": energy, "size": int(W.size), "kept_columns": int(len(cols))}
    Wh, info = ball(m.group("arm"), name, W)
    return name, Wh, {"rel_err": info["rel_err"], "bits": info["bits"], "overload": info["overload"], "energy": energy, "size": int(W.size), "kept_columns": 0}


def tasks_of(variant, names):
    m = CLIPCOLS.match(variant)
    if m:
        name = f"blk.{int(m.group('layer'))}.ffn_down.weight"
        if name not in names:
            raise SystemExit(f"{variant} names a layer the model does not have")
        return clip_part, [(name, m.group("part"))]
    m = KEEP.match(variant)
    if m:
        asked = m.group("layers") or m.group("whole")
        for x in (asked.split(",") if asked else ()):
            if f"blk.{int(x)}.ffn_down.weight" not in names:
                raise SystemExit(f"{variant} names a layer the model does not have")
        return keep_one, [(variant, n) for n in names]
    raise SystemExit(f"unknown variant {variant}")


def picks(args):
    """For every ffn_down: the loud columns, the most active ones, how many are both, and what the loud ones hold."""
    tensors = {t.name: t for t in Q.reader().tensors}
    layers = sorted(int(Q.BLOCK.match(n).group(1)) for n in tensors if Q.BLOCK.match(n) and Q.BLOCK.match(n).group(2) == "ffn_down")
    print(f"The {K} loud columns of every ffn_down, from the weights, beside the {K} most active, from the importance matrix. Nothing scored.")
    print("layer | loud columns | most active columns | columns that are both | share of the matrix's squared weight in the loud columns, percent | "
          "share of the input activity in the loud columns, percent | in the most active, percent | largest entry of the matrix in standard deviations of its row | is it in a loud column")
    for layer in layers:
        name = f"blk.{layer}.ffn_down.weight"
        W = Q.matrix_of(tensors[name]).astype(np.float64)
        lo, ac = loud(W), active(name, W.shape[1])
        act = Q.column_weights(name, W.shape[1], 1.0)
        sd = np.abs(W) / W.std(axis=1, keepdims=True)
        top = int(np.unravel_index(np.argmax(sd), sd.shape)[1])
        print(f"{layer} | {' '.join(map(str, lo))} | {' '.join(map(str, ac))} | {len(set(lo) & set(ac))} | {100 * (W[:, lo] ** 2).sum() / (W ** 2).sum():.2f} | "
              f"{100 * act[lo].sum() / act.sum():.2f} | {100 * act[ac].sum() / act.sum():.2f} | {sd.max():.1f} | {'yes' if top in set(lo) else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld")
    ap.add_argument("--text", default="test", help="a label for the text the reference outputs are of")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--imatrix")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--picks", action="store_true")
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    Q._model_path, Q._imatrix_path = args.model, args.imatrix
    model = Q.MODELS.get(Q.sha256_of(args.model))
    if model not in ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    if args.picks:
        return picks(args)
    names = [t.name for t in Q.targets(Q.reader())]
    all_weights = sum(int(np.prod([int(d) for d in t.shape])) for t in Q.targets(Q.reader()))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"], r["text"]) for r in map(json.loads, filter(str.strip, f))}
    if not args.metrics_only:
        os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if (model, v, args.text) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        worker, tasks = tasks_of(v, names)
        if ("+A10" in v or v.startswith("h")) and not args.imatrix:
            raise SystemExit(f"{v} needs --imatrix")
        t0 = time.time()
        replace, infos = {}, {}
        with Pool(min(args.workers, len(tasks))) as pool:
            for name, Wh, info in pool.imap_unordered(worker, tasks, chunksize=1):
                infos[name] = info
                if not args.metrics_only:
                    replace[name] = Wh
        stats = Q.stats_of(infos)
        row = {"variant": v, "model": model, "text": args.text, "matrices": stats["matrices"], "weights": all_weights, "weights_changed": stats["weights"],
               "bits_per_weight_changed": stats["bits_per_weight"], "weight_err_db_mean": stats["weight_err_db_mean"],
               "weight_err_db_pooled": stats["weight_err_db_pooled"], "share_out_of_range": stats["share_out_of_range"],
               "kept_columns": sum(i["kept_columns"] for i in infos.values())}
        if CLIPCOLS.match(v):
            row["clipped_entries"] = sum(i["clipped"] for i in infos.values())
        else:
            row["weight_err_db_ffn_down"] = stats["weight_err_db_by_kind"]["ffn_down"]
        row = {k: (None if isinstance(x, float) and not math.isfinite(x) else x) for k, x in row.items()}     # a mean of decibels over matrices has no value when one is left whole
        row["build_seconds"] = round(time.time() - t0, 1)
        if not args.metrics_only:
            path = os.path.join(args.work, "lc_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v)[:80] + ".gguf")
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
