#!/usr/bin/env python3
"""Part F of PREREG.md: three more measures of a weight change, and the weights alone. Nothing is scored.

Modelled on the first record's spectral.py: the same 42 matrices (layers 0, 8, 16, 24, 32 and 40),
the same 26 variants and flip:3.45, each matrix changed as the variant changes it. For a matrix W and
its changed form Wh, with E = Wh - W:
  along the rows      sum over rows of (E_i . W_i)^2 / |W_i|^2, over the squared size of W
  along the columns   the same with columns
  tree shift          squared distance between the sorted tree sizes of |Wh| and of |W|, over the squared
                      size of the tree sizes of |W| (persist/persistlib.py, sorted_distance)
  squared error and spectral error, as in spectral.py (power iteration, 40 steps, seed 0)
Each is the mean over the matrices of 10 log10 of the ratio, in dB. The same means over the 36 matrices
that the model reads are recorded too (the keys and values of layers 24, 32 and 40 are never read).

Variants of Parts B, C and D that change all kinds are measured with --held-out-from results.jsonl.
The weights alone (--weights-only) are measured on all 294 matrices with nothing rounded: normalised
persistence (the matrix's own largest weight, the count of neurons, p = 2), and the spread of the rows and
of the columns, each the standard deviation of the logarithm of the root mean square of a row or column.

Usage:
  python3 measures.py --model REF.gguf --out measures.jsonl
  python3 measures.py --model REF.gguf --out measures.jsonl --held-out-from results.jsonl
  python3 measures.py --model REF.gguf --weights-only weights_only.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EARLIER = ROOT / "experiments" / "model-output-gemma-e4b-2026-11-15"
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "persist"))
import persistlib  # noqa: E402
import tq  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ob_run = _load("ob_run", HERE / "run.py")
spectral = _load("earlier_spectral", EARLIER / "spectral.py")      # its own `import run` finds the earlier record's run.py
ggufedit = ob_run.ggufedit
LAYERS = spectral.LAYERS
KINDS = ob_run.KINDS
BLOCK = ob_run.BLOCK
DEFAULT_VARIANTS = spectral.VARIANTS + spectral.EXTRA
HELD_OUT = re.compile(r"^(X32mix:|X(32|R)o:|X32s\d+:)")


def along(W, Wh, axis):
    """Squared size of the part of the change that lies along each row (axis=1) or column (axis=0), summed, over |W|^2.

    For each row i: (E_i . W_i)^2 / |W_i|^2, with E = Wh - W. A row of zeros counts nothing.
    """
    W = np.asarray(W, dtype=np.float64)
    E = np.asarray(Wh, dtype=np.float64) - W
    dot = np.sum(E * W, axis=axis)
    n2 = np.sum(W * W, axis=axis)
    part = np.where(n2 > 0, dot * dot / np.where(n2 > 0, n2, 1.0), 0.0)
    return float(part.sum() / np.sum(W * W))


def tree_shift(W, Wh, tw=None):
    """Squared distance between the sorted tree sizes of |Wh| and |W|, over the squared size of those of |W|."""
    if tw is None:
        tw = persistlib.tree_weights(W, wmax=1.0)
    d = persistlib.sorted_distance(W, Wh)
    return float(d * d / np.sum(tw * tw))


def db(x):
    return 10.0 * math.log10(max(x, 1e-30))


def round_variant(v, W, index, name):
    """One matrix changed as variant v changes it, float16. Variants of this record use run.py, the older ones spectral.py."""
    if re.match(r"^(X32(s\d+)?mix:|X(32|R)o:|X32s\d+:)", v):
        spec = ob_run.parse(v)
        return ob_run.round_one(spec, W, BLOCK.match(name).group(2))[0]
    return spectral.round_one(W, v, index)


def spread(W, axis):
    """Standard deviation of the log of the root mean square of the rows (axis=1) or columns (axis=0). Zero rows are left out."""
    r = np.sqrt(np.mean(np.asarray(W, dtype=np.float64) ** 2, axis=axis))
    pos = r[r > 0]
    return float(np.std(np.log(pos))), int(len(r) - len(pos))


def run_variants(reader, variants, out):
    by = {t.name: t for t in reader.tensors}
    live = ob_run.earlier.live_kv_layers(reader)
    names = [f"blk.{il}.{k}.weight" for il in LAYERS for k in KINDS]
    mats, sv_w, tws = {}, {}, {}
    for name in names:
        mats[name] = ggufedit.tensor_float(by[name])
        sv_w[name] = spectral.top_sv2(torch.from_numpy(mats[name]).to(tq.DEV))
        tws[name] = persistlib.tree_weights(mats[name], wmax=1.0)
    have = set()
    if Path(out).exists():
        have = {json.loads(ln)["variant"] for ln in Path(out).read_text().splitlines() if ln.strip()}
    for v in variants:
        if v in have:
            print(f"skip {v}", flush=True)
            continue
        cols = {k: [] for k in ("rows", "cols", "tree", "spec", "frob")}
        live_cols = {k: [] for k in cols}
        for index, name in enumerate(names):
            W = mats[name]
            Wh = round_variant(v, W, index, name)
            E = torch.from_numpy(W - Wh.astype(np.float32)).to(tq.DEV)
            W64, H64 = W.astype(np.float64), Wh.astype(np.float64)
            vals = {
                "rows": db(along(W64, H64, 1)),
                "cols": db(along(W64, H64, 0)),
                "tree": db(tree_shift(W, Wh, tws[name])),
                "spec": 10.0 * math.log10(max(spectral.top_sv2(E), 1e-30) / sv_w[name]),
                "frob": 10.0 * math.log10(max(float(np.sum((W64 - H64) ** 2)), 1e-30) / float(np.sum(W64 ** 2))),
            }
            il, kind = BLOCK.match(name).groups()
            for k, x in vals.items():
                cols[k].append(x)
                if not (int(il) >= live and kind in ("attn_k", "attn_v")):
                    live_cols[k].append(x)
        mean = lambda xs: sum(xs) / len(xs)
        row = {"variant": v, "matrices": len(names), "matrices_read": len(live_cols["rows"]),
               "along_rows_db_mean": mean(cols["rows"]), "along_cols_db_mean": mean(cols["cols"]),
               "tree_shift_db_mean": mean(cols["tree"]), "spectral_err_db_mean": mean(cols["spec"]),
               "squared_err_db_mean": mean(cols["frob"]),
               "along_rows_db_mean_read": mean(live_cols["rows"]), "along_cols_db_mean_read": mean(live_cols["cols"]),
               "tree_shift_db_mean_read": mean(live_cols["tree"]), "spectral_err_db_mean_read": mean(live_cols["spec"]),
               "squared_err_db_mean_read": mean(live_cols["frob"])}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def weights_only(reader, out):
    by = {t.name: t for t in reader.tensors}
    for t in ob_run.targets(reader):
        il, kind = BLOCK.match(t.name).groups()
        W = ggufedit.tensor_float(t)
        row_spread, zero_rows = spread(W, 1)
        col_spread, zero_cols = spread(W, 0)
        row = {"name": t.name, "layer": int(il), "kind": kind, "rows": W.shape[0], "cols": W.shape[1],
               "persistence_normalised": float(persistlib.normalised(W, p=2)),
               "row_spread": row_spread, "col_spread": col_spread, "zero_rows": zero_rows, "zero_cols": zero_cols}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(ob_run.targets(reader))} lines to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--weights-only", help="write the weights-alone measures of the 294 matrices to this file")
    ap.add_argument("--held-out-from", help="results.jsonl: measure the variants of Parts B, C and D that change all kinds")
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    reader = ggufedit.open_model(args.model)
    if args.weights_only:
        weights_only(reader, args.weights_only)
        return
    if not args.out:
        raise SystemExit("--out is needed")
    if args.held_out_from:
        variants = []
        for ln in Path(args.held_out_from).read_text().splitlines():
            r = json.loads(ln)
            if "kl_mean" in r and HELD_OUT.match(r["variant"]) and "@" not in r["variant"] and r["variant"] not in variants:
                variants.append(r["variant"])
    else:
        variants = list(args.variants) or DEFAULT_VARIANTS
    run_variants(reader, variants, args.out)


if __name__ == "__main__":
    main()
