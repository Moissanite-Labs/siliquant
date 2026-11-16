#!/usr/bin/env python3
"""Round the weights of a real language model and measure what its output does.

The plan is PREREG.md in this folder. One call handles a list of variants.
For each it builds a changed copy of the model file, scores it against the
reference model's saved outputs with llama.cpp's own tool, appends one line
to a results file and deletes the copy.

Variants:
  identity          the block matrices unrounded, stored as 16-bit floats (a control)
  ARM:Q             a rounding arm at Q levels per weight on the block matrices,
                    ARM one of S, Z8, E8, D+S, D+Z8, D+E8
  ARM:Q@KIND        the same on one kind of matrix only (attn_q, attn_k, attn_v, attn_output,
                    ffn_gate, ffn_up, ffn_down), the rest left alone
  close:TENSOR      zero the rows of a token table (per_layer_token_embd or token_embd)
                    for every token that does not occur in the evaluation text
  file:PATH         score an existing model file (for llama.cpp's own quantisers)

Usage, with NumPy and torch, llama.cpp's gguf package and llama-perplexity:
  python3 run.py --model REF.gguf --kld base.kld --work DIR --out results.jsonl identity S:16 E8:16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
import evalkl  # noqa: E402
import ggufedit  # noqa: E402

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")


def targets(reader, kind=None):
    out = []
    for t in reader.tensors:
        m = BLOCK.match(t.name)
        if m and len(t.shape) == 2 and (kind is None or m.group(2) == kind):
            out.append(t)
    return out


def db(x):
    return 10.0 * math.log10(x) if x > 0 else float("-inf")


def build_rounded(reader, arm, q, kind):
    import tq
    replace, per_kind, weights, energy_err, energy = {}, {}, 0, 0.0, 0.0
    bits = []
    for t in targets(reader, kind):
        W = ggufedit.tensor_float(t)
        if arm == "identity":
            Wh = W.astype(np.float16)
            W64 = W.astype(np.float64)
            info = {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2)), "bits": 16.0, "overload": 0.0}
        else:
            Wh, info = tq.round_matrix(W, arm, q)
        replace[t.name] = Wh
        k = BLOCK.match(t.name).group(2)
        per_kind.setdefault(k, []).append(info["rel_err"])
        e = float(np.sum(W.astype(np.float64) ** 2))
        energy += e
        energy_err += e * info["rel_err"]
        weights += W.size
        bits.append(info["bits"] * W.size)
    every = [v for vs in per_kind.values() for v in vs]
    stats = {
        "matrices": len(every), "weights": weights, "bits_per_weight": sum(bits) / weights,
        "weight_err_db_mean": sum(db(v) for v in every) / len(every),
        "weight_err_db_pooled": db(energy_err / energy),
        "weight_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(per_kind.items())},
    }
    return replace, stats


def build_closed(reader, tensor_name, kld):
    n_ctx, n_vocab, n_chunk, toks = evalkl.tokens_of(kld)
    used = np.zeros(n_vocab, dtype=bool)
    used[toks] = True
    t = next(x for x in reader.tensors if x.name == tensor_name)
    raw = np.array(t.data)                      # storage bytes, one row per token
    if raw.shape[0] != n_vocab:
        raise SystemExit(f"{tensor_name}: {raw.shape[0]} rows, expected one per token ({n_vocab})")
    raw[~used] = 0
    per_row = int(np.prod(t.shape)) // n_vocab
    stats = {"tokens_in_text": int(used.sum()), "rows_zeroed": int((~used).sum()),
             "weights_zeroed": int((~used).sum()) * per_row, "weights_in_tensor": int(np.prod(t.shape)),
             "weights_in_model": int(sum(int(np.prod(x.shape)) for x in reader.tensors))}
    return {tensor_name: raw}, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the changed model files")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    reader = ggufedit.open_model(args.model)
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        t0 = time.time()
        row = {"variant": v}
        if v.startswith("file:"):
            path, made = v[5:], False
        else:
            path, made = os.path.join(args.work, re.sub(r"[^A-Za-z0-9_.+-]", "_", v) + ".gguf"), True
            if v == "identity":
                replace, stats = build_rounded(reader, "identity", 0, None)
            elif v.startswith("close:"):
                replace, stats = build_closed(reader, v[6:], args.kld)
            else:
                m = re.match(r"^(D\+)?(S|Z8|E8):(\d+)(?:@(\w+))?$", v)
                if not m:
                    raise SystemExit(f"unknown variant {v}")
                arm, q, kind = (m.group(1) or "") + m.group(2), int(m.group(3)), m.group(4)
                if kind is not None and kind not in KINDS:
                    raise SystemExit(f"unknown kind {kind}")
                replace, stats = build_rounded(reader, arm, q, kind)
                row.update({"arm": arm, "q": q, "kind": kind or "all"})
            row.update(stats)
            row["build_seconds"] = round(time.time() - t0, 1)
            ggufedit.write_copy(reader, path, replace)
            del replace
        t1 = time.time()
        scores, _ = evalkl.score(path, args.kld)
        row.update(scores)
        row["score_seconds"] = round(time.time() - t1, 1)
        if made and not args.keep:
            os.remove(path)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
