#!/usr/bin/env python3
"""Pack a model's block matrices with shape/ballpack.py, read the file back, and score what it holds.

The plan is PREREG.md in this folder. For one variant:

  1. every block matrix is rounded and packed (integer levels in one ball per 32 weights, a
     number of 136 bits and one byte of scale: 18 bytes a block), and the file is written;
  2. the file is read back from disk and every matrix is decoded from it. The decoded matrix
     must be, bit for bit, the one the encoder said the file would hold, or the program stops;
  3. the numbers of the whole file are decoded once more on one core, for time;
  4. a copy of the model is written with the decoded matrices as 16-bit floats and scored with
     llama.cpp's tool against the reference outputs, as in the model-output records.

So the bits of a row are the bytes of a file, and its KL is that of weights read from the file.

A variant is "PZ32:b136s8" or, given an importance matrix, "hPZ32:b136s8": every column
weighted by the mean squared size of its input over the mean, to the power 0.25, as the arm
hZBG128:b560s16 of the record imatrix-baseline-2026-11-17.

Usage, with NumPy and torch, a C compiler, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/packed-ball-format-2026-11-18/run.py --model MODEL-f16.gguf --kld base.kld --work DIR \
      --out experiments/packed-ball-format-2026-11-18/results.jsonl --bin llama-perplexity --imatrix MODEL.imatrix.gguf PZ32:b136s8 hPZ32:b136s8
"""
from __future__ import annotations

import argparse
import hashlib
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
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "shape"))
import ballpack  # noqa: E402

spec = importlib.util.spec_from_file_location("qwen_run", HERE.parent / "model-output-qwen25-2026-11-17" / "run.py")
Q = importlib.util.module_from_spec(spec)
sys.modules["qwen_run"] = Q
spec.loader.exec_module(Q)
Q.MODELS["cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"] = "SmolLM2-1.7B-Instruct-f16.gguf"

VARIANT = re.compile(r"^(h?)PZ32:b136s8$")
N, BITS, STEPS, POWER = 32, 136, 16, 0.25
_fmt = None


def fmt():
    global _fmt
    if _fmt is None:
        _fmt = ballpack.Format(N, BITS, STEPS)
    return _fmt


def pack_one(task):
    """One block matrix: (name, its bytes in the file, SHA-256 of the matrix the file will give back, what happened)."""
    variant, name = task
    t = next(t for t in Q.reader().tensors if t.name == name)
    W = Q.matrix_of(t)
    c = Q.column_weights(name, W.shape[1], POWER) if variant.startswith("h") else None
    stored, Wh, shrunk = fmt().encode(W, c)
    blob = fmt().tensor_bytes(name, stored)
    W64 = W.astype(np.float64)
    energy = float(np.sum(W64 ** 2))
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2)) / energy
    zeros = float(np.mean(Wh == 0))
    return name, blob, hashlib.sha256(Wh.tobytes()).hexdigest(), {"rel_err": rel, "bits": 8.0 * len(blob) / W.size, "overload": shrunk,
                                                                 "energy": energy, "size": int(W.size), "zeros": zeros}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--imatrix")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=int, default=0, help="pack and read back only the first N block matrices and do not score (a test)")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    Q._model_path, Q._imatrix_path = args.model, args.imatrix
    digest = Q.sha256_of(args.model)
    if digest not in Q.MODELS:
        raise SystemExit("FAIL: the model file is not one the plan names")
    names = [t.name for t in Q.targets(Q.reader())]
    if args.only:
        names = names[: args.only]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"]) for r in map(json.loads, filter(str.strip, f))}
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        if not VARIANT.match(v):
            raise SystemExit(f"unknown variant {v}")
        if v.startswith("h") and not args.imatrix:
            raise SystemExit(f"{v} needs --imatrix")
        if (Q.MODELS[digest], v) in done:
            print(f"{v}: already in the results file", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        stem = os.path.join(args.work, "pz_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v))
        blobs, want, infos = {}, {}, {}
        with Pool(args.workers) as pool:
            for name, blob, sha, info in pool.imap_unordered(pack_one, [(v, n) for n in names], chunksize=1):
                blobs[name], want[name], infos[name] = blob, sha, info
        with open(stem + ".sqbp", "wb") as f:
            f.write(fmt().header(len(names)))
            for n in names:
                f.write(blobs[n])
        del blobs
        t1 = time.time()
        file_bytes = os.path.getsize(stem + ".sqbp")
        file_sha = Q.sha256_of(stem + ".sqbp")
        # Everything below comes from the file on disk.
        fmt2, tensors = ballpack.Format.read(stem + ".sqbp")
        if [n for n, _ in tensors] != names:
            raise SystemExit("FAIL: the file does not hold the block matrices in order")
        replace, cpu, blocks = {}, 0.0, 0
        for name, stored in tensors:
            Wh = fmt2.decode(stored)
            if hashlib.sha256(Wh.tobytes()).hexdigest() != want[name]:
                raise SystemExit(f"FAIL: {name} read back from the file is not the matrix the encoder gave")
            replace[name] = Wh
            seconds, _ = fmt2.timing(stored["records"], float(stored["top"]), repeats=2)
            cpu += seconds
            blocks += len(stored["records"])
        del tensors
        row = {"variant": v, "model": Q.MODELS[digest]}
        row.update(Q.stats_of(infos))
        weights = row["weights"]
        row.update({"file_bytes": file_bytes, "file_sha256": file_sha, "stored_bits_per_weight": 8.0 * file_bytes / weights,
                    "block_bits_per_weight": (BITS + 8.0) / N, "bound": fmt2.bound, "read_back_exact": True,
                    "share_of_weights_zero": sum(i["zeros"] * i["size"] for i in infos.values()) / weights,
                    "pack_seconds": round(t1 - t0, 1), "decode_cpu_seconds_one_core": round(cpu, 2),
                    "decode_million_weights_per_second": round(blocks * N / cpu / 1e6, 2)})
        if abs(row["bits_per_weight"] - row["stored_bits_per_weight"]) > 28 * 8.0 / weights + 1e-12:
            raise SystemExit("FAIL: the bits of the matrices do not add up to the file")
        if args.only:
            print(json.dumps(row), flush=True)
            os.remove(stem + ".sqbp")
            continue
        Q.ggufedit.write_copy(Q.reader(), stem + ".gguf", replace)
        del replace
        os.remove(stem + ".sqbp")
        t2 = time.time()
        scores, _ = Q.evalkl.score(stem + ".gguf", args.kld, binary=args.bin, extra=("-t", str(args.threads)))
        row.update(scores)
        row["score_seconds"] = round(time.time() - t2, 1)
        os.remove(stem + ".gguf")
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
