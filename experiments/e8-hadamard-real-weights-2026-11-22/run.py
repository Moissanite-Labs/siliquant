#!/usr/bin/env python3
"""E8 turned by a Hadamard matrix, in one ball per group of 128 weights, on real weights.

The plan is PREREG.md in this folder.

Variants:
  ZBG128:bBs16     integer levels in one ball per group of 128 weights: B bits for the point, 16 for the scale
  EBG128:bBs16     E8 in the same ball
  HBG128:bBs16     E8 turned by the Hadamard matrix of order 8 over the root of 8, in the same ball
  ARM+K10          the arm with, in every ffn_down, the ten loud columns left as in the reference: 16 bits an
                   entry and 16 bits for each column's number, counted

  VE8Z32:16        the harness's Voronoi code of E8 with a codeword at zero: 16 points a weight, and one scale per
                   group of 32 weights, the smallest at which no block of the group is shrunk. 4.5 bits a weight
  VH8Z32:16        the same code of the turned lattice

B is 560, which is 4.5 bits a weight, or 368, which is 3.0.

Every arm is built by the ball arm of the record model-output-qwen25-2026-11-17, whose code is not changed.
For the turned lattice that code's E8 arm is given the matrix with every block of 8 weights of a row turned,
and what it returns is turned back. The matrix that turns is its own inverse, and a ball does not change when
it is turned, so the points, the scales and the bits are those of E8 in the same ball.

The two Voronoi arms are the harness's function, as the base program calls it for its arm VE8Z. For the turned
lattice it is given the turned matrix. It returns 16-bit floats, which are turned back in 64-bit floats and cut
to 16 bits once more: a 16-bit float is within one part in 2,048 of what it holds, far under the rounding error.

The loud columns are those of the record loud-columns-2026-11-18, by its function. A kept column is set to
zero before the rest of the matrix is rounded and put back after, as in that record.

Each row also has what the arm does to the small weights, from the weights alone: how many weights are stored
as zero, and for the weights under 0.05, 0.1 and 0.2 of their matrix's root mean square, how many there are,
how many are stored as zero, and their squared error over their squared size.

Rows are keyed by model, variant and the label of the text.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/e8-hadamard-real-weights-2026-11-22/run.py --model M.gguf --kld base.kld --work DIR \
      --out results.jsonl --bin llama-perplexity VARIANT [VARIANT ...]
  ... --metrics-only --out metrics.jsonl     build every variant and record what it does to the weights, without scoring
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
sys.path.insert(0, str(HERE))
from turned import turn  # noqa: E402

spec = importlib.util.spec_from_file_location("loud_run", HERE.parent / "loud-columns-2026-11-18" / "run.py")
LC = importlib.util.module_from_spec(spec)
sys.modules["loud_run"] = LC
spec.loader.exec_module(LC)
Q = LC.Q
Q.MODELS["1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"] = "Llama-3.2-1B-Instruct-f16.gguf"
ACCEPTED = ("qwen2.5-1.5b-instruct-fp16.gguf", "SmolLM2-1.7B-Instruct-f16.gguf", "Llama-3.2-1B-Instruct-f16.gguf")

VARIANT = re.compile(r"^(?P<arm>ZBG|EBG|HBG)128:b(?P<bits>560|368)s16(?P<keep>\+K10)?$")
VORONOI = re.compile(r"^(?P<arm>VE8Z|VH8Z)32:16$")
LATTICE = {"ZBG": "Z8", "EBG": "E8", "HBG": "E8"}
SMALL = (0.05, 0.1, 0.2)
CLASS = {"attn_q": "attention", "attn_k": "attention", "attn_v": "attention", "attn_output": "attention",
         "ffn_gate": "mlp", "ffn_up": "mlp", "ffn_down": "mlp"}


def ball64(W64, lattice, bits):
    """The ball arm of the base program on a matrix, before it is cut to 16-bit floats: (rounded, bits a weight, share of groups shrunk)."""
    keep = Q.finish
    Q.finish = lambda W, Y, b, shrunk: (Y, {"bits": b, "overload": shrunk})
    try:
        Y, info = Q.group_ball_round(W64, lattice, 128, bits, 16)
    finally:
        Q.finish = keep
    return Y, info["bits"], info["overload"]


def voronoi(W32):
    """The harness's Voronoi code of E8, 16 points a weight, a scale per group of 32, with a codeword at zero: (16-bit floats, what happened)."""
    import torch
    import tq
    code = tq.Code("E8", 16)
    kept = code.centre
    code.centre = torch.zeros_like(kept)                   # as the base program does for its arm VE8Z
    try:
        return tq.maxscale_voronoi(W32, 16, 32, name="E8")
    finally:
        code.centre = kept


def round_matrix(variant, kind, W):
    """One block matrix rounded by one variant: (the matrix as 16-bit floats, what happened)."""
    W64 = W.astype(np.float64)
    cols = None
    v = VORONOI.match(variant)
    if v:
        if v.group("arm") == "VH8Z":
            Wr, got = voronoi(turn(W64).astype(np.float32))
            Y = turn(Wr.astype(np.float64))
        else:
            Wr, got = voronoi(W)
            Y = Wr.astype(np.float64)
        bits, shrunk = got["bits"], got["overload"]
    else:
        m = VARIANT.match(variant)
        rest = W64
        if m.group("keep") and kind == "ffn_down":
            cols = LC.loud(W, LC.K)
            rest = W64.copy()
            rest[:, cols] = 0.0
        if m.group("arm") == "HBG":
            Y, bits, shrunk = ball64(turn(rest), "E8", int(m.group("bits")))
            Y = turn(Y)
        else:
            Y, bits, shrunk = ball64(rest, LATTICE[m.group("arm")], int(m.group("bits")))
        if cols is not None:
            Y[:, cols] = W64[:, cols]
            bits += (len(cols) * W.shape[0] * 16.0 + len(cols) * 16.0) / W.size
    Wh, info = Q.finish(W, Y, bits, shrunk)
    Wh64 = Wh.astype(np.float64)
    size, err2 = np.abs(W64), (W64 - Wh64) ** 2
    rms = math.sqrt(float(np.mean(W64 ** 2)))
    small = {}
    for d in SMALL:
        mask = size < d * rms
        small[str(d)] = [int(mask.sum()), int(np.count_nonzero(Wh64[mask] == 0.0)), float(err2[mask].sum()), float((W64[mask] ** 2).sum())]
    info.update({"energy": float(np.sum(W64 ** 2)), "size": int(W.size), "kept_columns": 0 if cols is None else int(len(cols)),
                 "zeros": int(np.count_nonzero(Wh64 == 0.0)), "small": small})
    return Wh, info


def one(task):
    variant, name = task
    import torch
    torch.set_num_threads(1)
    Wh, info = round_matrix(variant, Q.BLOCK.match(name).group(2), Q.matrix_of(LC.tensor(name)))
    return name, Wh, info


def small_stats(infos):
    """What a variant does to the small weights, pooled over the matrices of each class and over all of them."""
    out = {"stored_zero_percent": {}, "small_weights": {str(d): {} for d in SMALL}}
    for cls in ("all", "attention", "mlp"):
        part = [i for n, i in infos.items() if cls == "all" or CLASS[Q.BLOCK.match(n).group(2)] == cls]
        weights = sum(i["size"] for i in part)
        out["stored_zero_percent"][cls] = 100.0 * sum(i["zeros"] for i in part) / weights
        for d in SMALL:
            n, z, e, w = (sum(i["small"][str(d)][k] for i in part) for k in range(4))
            out["small_weights"][str(d)][cls] = {"percent_of_weights": 100.0 * n / weights, "stored_zero_percent": 100.0 * z / n if n else None,
                                                 "err_over_size_db": 10.0 * math.log10(e / w) if n and e > 0 and w > 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld")
    ap.add_argument("--text", default="test", help="a label for the text the reference outputs are of")
    ap.add_argument("--work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    for v in args.variants:
        if not (VARIANT.match(v) or VORONOI.match(v)):
            raise SystemExit(f"unknown variant {v}")
    if not args.metrics_only and not (args.kld and args.work):
        raise SystemExit("scoring needs --kld and --work")
    Q._model_path = args.model
    model = Q.MODELS.get(Q.sha256_of(args.model))
    if model not in ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    names = [t.name for t in Q.targets(Q.reader())]
    all_weights = sum(int(np.prod([int(d) for d in t.shape])) for t in Q.targets(Q.reader()))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"], r.get("text", "test")) for r in map(json.loads, filter(str.strip, f))}
    if not args.metrics_only:
        os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if (model, v, args.text) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        replace, infos = {}, {}
        with Pool(min(args.workers, len(names))) as pool:
            for name, Wh, info in pool.imap_unordered(one, [(v, n) for n in names], chunksize=1):
                infos[name] = info
                if not args.metrics_only:
                    replace[name] = Wh
        stats = Q.stats_of(infos)
        row = {"variant": v, "model": model, "text": args.text, "matrices": stats["matrices"], "weights": all_weights,
               "bits_per_weight": stats["bits_per_weight"], "weight_err_db_mean": stats["weight_err_db_mean"],
               "weight_err_db_pooled": stats["weight_err_db_pooled"], "weight_err_db_by_kind": stats["weight_err_db_by_kind"],
               "share_out_of_range": stats["share_out_of_range"], "kept_columns": sum(i["kept_columns"] for i in infos.values())}
        row.update(small_stats(infos))
        row["build_seconds"] = round(time.time() - t0, 1)
        if not args.metrics_only:
            path = os.path.join(args.work, "eh_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v)[:80] + ".gguf")
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
