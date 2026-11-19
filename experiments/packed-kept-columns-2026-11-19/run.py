#!/usr/bin/env python3
"""The packed file of the record packed-ball-format-2026-11-18 with columns of ffn_down kept whole.

The plan is PREREG.md in this folder. This program is that record's program with one function replaced: the one
that packs a matrix. Everything else is that record's code as it stands: the file is written, read back from disk,
every matrix read back must be bit for bit the one the encoder gave or the program stops, the decode is timed, and
the model the file holds is scored.

A variant is "PZ32:b136s8" or "hPZ32:b136s8" as there, or either with "+K10": in every ffn_down the ten loud columns
are kept whole. A loud column is one of the ten in which a clip at 5 standard deviations of a row removes the most
squared weight (the function loud of the record loud-columns-2026-11-18, imported unchanged). A kept column is stored
as 16-bit floats with its number, by shape/ballpack.py, and is zero in the blocks.

Usage, with NumPy and torch, a C compiler, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/packed-kept-columns-2026-11-19/run.py --model MODEL-f16.gguf --kld base.kld --work DIR \
      --out results.jsonl --bin llama-perplexity [--imatrix MODEL.imatrix.gguf] PZ32:b136s8+K10
  ... --only 9      pack and read back only the first 9 block matrices and do not score (a test)
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = load("loud_run", HERE.parent / "loud-columns-2026-11-18" / "run.py")              # for the rule that picks the loud columns
P = load("packed_run", HERE.parent / "packed-ball-format-2026-11-18" / "run.py")      # last, so that its copy of the model-output program is the one in use
Q = P.Q
Q.MODELS["1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"] = "Llama-3.2-1B-Instruct-f16.gguf"
VARIANT = re.compile(r"^(h?)PZ32:b136s8(\+K10)?$")


def pack_one(task):
    """One block matrix, as in the packed-format record, with the loud columns of an ffn_down kept if the variant says so."""
    variant, name = task
    t = next(t for t in Q.reader().tensors if t.name == name)
    W = Q.matrix_of(t)
    c = Q.column_weights(name, W.shape[1], P.POWER) if variant.startswith("h") else None
    keep = L.loud(W) if variant.endswith("+K10") and Q.BLOCK.match(name).group(2) == "ffn_down" else None
    stored, Wh, shrunk = P.fmt().encode(W, c, keep=keep)
    blob = P.fmt().tensor_bytes(name, stored)
    W64 = W.astype(np.float64)
    energy = float(np.sum(W64 ** 2))
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2)) / energy
    return name, blob, hashlib.sha256(Wh.tobytes()).hexdigest(), {"rel_err": rel, "bits": 8.0 * len(blob) / W.size, "overload": shrunk,
                                                                 "energy": energy, "size": int(W.size), "zeros": float(np.mean(Wh == 0))}


P.VARIANT = VARIANT
P.pack_one = pack_one

if __name__ == "__main__":
    P.main()
