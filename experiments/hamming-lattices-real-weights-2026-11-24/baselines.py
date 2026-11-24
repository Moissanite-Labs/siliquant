#!/usr/bin/env python3
"""Part B of PREREG.md: two baselines scored the way the ball arms are scored.

Variants:
  Q4_K_M@16   llama.cpp's Q4_K_M file of the model, the one its earlier record scored (checked by SHA-256),
              with every block matrix decoded to 16-bit floats and put in the 16-bit model. Embeddings and
              output were left at 16 bits in that file, as in the ball arms. The weights are those of
              Q4_K_M; what changes is that llama.cpp then multiplies by 16-bit weights, as it does for
              every ball arm, and not by 4-bit blocks, for which it first rounds the activations to 8 bits.
  NF4B64      bitsandbytes' NF4 with its defaults: blocks of 64 consecutive weights of a row, the block's
              largest size stored as a 32-bit float, each weight over it rounded to the nearest of the 16
              values of NF4. 4 + 32/64 = 4.5 bits a weight. Block matrices only, as the ball arms.

Rows have the fields of the ball arms' rows where they apply.

Usage:
  python3 experiments/hamming-lattices-real-weights-2026-11-24/baselines.py --model M.gguf [--q4km Q.gguf] \
      --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity VARIANT [VARIANT ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

Q, EH = R.Q, R.EH

# the Q4_K_M files the earlier records scored, by model
Q4KM = {"qwen2.5-1.5b-instruct-fp16.gguf": "3b70feecdd4674a4ec4e80f6df0706e97b4322cd35d4c96199ccaecdbe0f4f0b",
        "SmolLM2-1.7B-Instruct-f16.gguf": "25e0bcfb12f106a5556863f7a94a88aaa0bcdcf1ae76c8034c6c690fabfccdf9",
        "Llama-3.2-1B-Instruct-f16.gguf": "6a9ba15dee9644b2f11919141746907216725504ea87269c5e33b82a466c0252"}

# bitsandbytes' NF4 code (bitsandbytes/functional.py, get_4bit_type("nf4"))
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0], dtype=np.float32)
VARIANTS = ("Q4_K_M@16", "NF4B64")


def nf4(W):
    """NF4 with blocks of 64: (rounded matrix in 32-bit floats, bits a weight)."""
    X = np.asarray(W, dtype=np.float32).reshape(-1, 64)
    a = np.abs(X).max(axis=1).astype(np.float32)
    a = np.where(a == 0, np.float32(1.0), a)
    Z = X / a[:, None]
    mid = (NF4[1:] + NF4[:-1]) / 2.0
    idx = np.searchsorted(mid, Z, side="right")
    Y = NF4[idx] * a[:, None]
    return Y.reshape(W.shape), 4.0 + 32.0 / 64.0


def row_of(v, model, text, infos, all_weights):
    stats = Q.stats_of(infos)
    row = {"variant": v, "model": model, "text": text, "matrices": stats["matrices"], "weights": all_weights,
           "bits_per_weight": stats["bits_per_weight"], "weight_err_db_mean": stats["weight_err_db_mean"],
           "weight_err_db_pooled": stats["weight_err_db_pooled"], "weight_err_db_by_kind": stats["weight_err_db_by_kind"],
           "share_out_of_range": stats["share_out_of_range"], "kept_columns": 0}
    row.update(EH.small_stats(infos))
    return row


def info_of(W, Wh, bits):
    W64, Wh64 = W.astype(np.float64), Wh.astype(np.float64)
    size, err2 = np.abs(W64), (W64 - Wh64) ** 2
    rms = math.sqrt(float(np.mean(W64 ** 2)))
    small = {}
    for d in EH.SMALL:
        mask = size < d * rms
        small[str(d)] = [int(mask.sum()), int(np.count_nonzero(Wh64[mask] == 0.0)), float(err2[mask].sum()), float((W64[mask] ** 2).sum())]
    return {"rel_err": float(np.sum(err2) / np.sum(W64 ** 2)), "bits": bits, "overload": 0.0, "energy": float(np.sum(W64 ** 2)),
            "size": int(W.size), "kept_columns": 0, "zeros": int(np.count_nonzero(Wh64 == 0.0)), "small": small}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--q4km")
    ap.add_argument("--kld", required=True)
    ap.add_argument("--text", default="test")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    Q._model_path = args.model
    model = Q.MODELS.get(Q.sha256_of(args.model))
    if model not in EH.ACCEPTED:
        raise SystemExit("FAIL: the model file is not one the plan names")
    targets = Q.targets(Q.reader())
    names = [t.name for t in targets]
    all_weights = sum(int(np.prod([int(d) for d in t.shape])) for t in targets)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"], r.get("text", "test")) for r in map(json.loads, filter(str.strip, f))}
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if (model, v, args.text) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        replace, infos = {}, {}
        extra = {}
        mine = {t.name: t for t in Q.reader().tensors}
        if v == "Q4_K_M@16":
            if not args.q4km:
                raise SystemExit("Q4_K_M@16 needs --q4km")
            h = hashlib.sha256(Path(args.q4km).read_bytes()).hexdigest()
            if h != Q4KM[model]:
                raise SystemExit(f"FAIL: {args.q4km} is not the Q4_K_M file the earlier record scored ({h})")
            q = {t.name: t for t in Q.ggufedit.open_model(args.q4km).tensors}
            for name in names:
                t = q[name]
                W = Q.matrix_of(mine[name])
                Wh = Q.ggufedit.tensor_float(t).reshape(W.shape)
                bits = 8.0 * t.data.nbytes / W.size
                Wh16, _ = Q.finish(W, Wh, bits, 0.0)
                replace[name] = Wh16
                infos[name] = info_of(W, Wh16, bits)
            extra = {"q4km_sha256": h}
        else:
            for name in names:
                W = Q.matrix_of(mine[name])
                Y, bits = nf4(W)
                Wh16, _ = Q.finish(W, Y, bits, 0.0)
                replace[name] = Wh16
                infos[name] = info_of(W, Wh16, bits)
        row = row_of(v, model, args.text, infos, all_weights)
        row.update(extra)
        row["build_seconds"] = round(time.time() - t0, 1)
        path = os.path.join(args.work, "hb_" + v.replace("@", "_") + ".gguf")
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
