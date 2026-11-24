#!/usr/bin/env python3
"""The packed file of packed-kept-columns-2026-11-19 with A_16 in place of the integers.

The plan is PREREG.md in this folder. This program is that record's program with the format changed: the
file is shape/hampack.py's HamFormat(16, 32, 136, 16), a block of 32 weights in 18 bytes as before, its point
one of A_16 x A_16 in a ball. Everything else is the packed-format record's code as it stands: the file is
written, read back from disk, every matrix read back must be bit for bit the one the encoder gave or the
program stops, the decode is timed on one core, and the model the file holds is scored.

A variant is "PH16:b136s8", or "PH16:b136s8+K10" with the ten loud columns of every ffn_down kept whole by
the rule of loud-columns-2026-11-18, imported unchanged.

Usage, with NumPy and torch, a C compiler, llama.cpp's gguf package on GGUF_PY and llama-perplexity:
  python3 experiments/packed-hamming-2026-11-24/run.py --model MODEL-f16.gguf --kld base.kld --work DIR \\
      --out experiments/packed-hamming-2026-11-24/results.jsonl --bin llama-perplexity PH16:b136s8 PH16:b136s8+K10
  ... --only 9      pack and read back only the first 9 block matrices and do not score (a test)
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shape"))
from hampack import HamFormat  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L = load("loud_run", HERE.parent / "loud-columns-2026-11-18" / "run.py")
P = load("packed_run", HERE.parent / "packed-ball-format-2026-11-18" / "run.py")
Q = P.Q
Q.MODELS["1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146"] = "Llama-3.2-1B-Instruct-f16.gguf"
VARIANT = re.compile(r"^PH16:b136s8(\+K10)?$")
_fmt = None


def fmt():
    global _fmt
    if _fmt is None:
        _fmt = HamFormat(16, P.N, P.BITS, P.STEPS)
    return _fmt


def pack_one(task):
    """One block matrix, with the loud columns of an ffn_down kept if the variant says so."""
    variant, name = task
    t = next(t for t in Q.reader().tensors if t.name == name)
    W = Q.matrix_of(t)
    keep = L.loud(W) if variant.endswith("+K10") and Q.BLOCK.match(name).group(2) == "ffn_down" else None
    stored, Wh, shrunk = fmt().encode(W, None, keep=keep)
    blob = fmt().tensor_bytes(name, stored)
    W64 = W.astype(np.float64)
    energy = float(np.sum(W64 ** 2))
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2)) / energy
    return name, blob, hashlib.sha256(Wh.tobytes()).hexdigest(), {"rel_err": rel, "bits": 8.0 * len(blob) / W.size, "overload": shrunk,
                                                                 "energy": energy, "size": int(W.size), "zeros": float(np.mean(Wh == 0))}


P.VARIANT = VARIANT
P.pack_one = pack_one
P.fmt = fmt
P.ballpack = types.SimpleNamespace(Format=HamFormat)       # the reader main() calls on the file on disk

if __name__ == "__main__":
    P.main()
