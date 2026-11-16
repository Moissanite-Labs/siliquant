#!/usr/bin/env python3
"""Spectral error of the variants on a fixed subset of matrices (round 5 of PREREG.md).

For each variant and each of the 42 block matrices of layers 0, 8, 16, 24, 32 and 40, the matrix
is rounded as the variant does and two numbers are taken from the change E = W - W_rounded:
  squared error      sum of E^2 over sum of W^2
  spectral error     squared largest singular value of E over squared largest singular value of W
Largest singular values are found by power iteration. Nothing is scored here: the mean KL of each
variant comes from results.jsonl.

Usage, with NumPy and torch and llama.cpp's gguf package:
  python3 spectral.py --model REF.gguf --out spectral.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
sys.path.insert(0, str(HERE))
import ggufedit  # noqa: E402
import tq  # noqa: E402
import run  # noqa: E402

LAYERS = (0, 8, 16, 24, 32, 40)
VARIANTS = ["S:32", "E8:32", "R+S:32", "C+S:32", "D+S:32", "D+E8:32", "XR:32", "XR:32~0.7", "XR:32~0.5", "XR:32~0.35", "VE8R:32",
            "X32:16", "B32+S:16", "VE832:16", "X32:32", "B32+S:32",
            "S:16", "E8:16", "D+S:16", "D+E8:16", "R+S:16", "XR:16", "VE8R:16",
            "clip:3.45", "clip:5", "clip:8"]
EXTRA = ["flip:3.45"]


def round_one(W, v, index):
    """One matrix rounded as variant v does. Returns float16."""
    if v.startswith("clip:") or v.startswith("flip:"):
        lim = float(v[5:]) * W.std(axis=1, keepdims=True)
        if v.startswith("clip:"):
            return np.clip(W, -lim, lim).astype(np.float16)
        excess = np.maximum(np.abs(W) - lim, 0.0)
        sign = np.random.default_rng(index).choice([-1.0, 1.0], size=W.shape).astype(np.float32)
        return (W + np.sign(W) * excess * sign).astype(np.float16)
    m = re.match(r"^([DRC]\+|B\d+\+)?(S|E8|X\d+|XR|VE8(?:\d+|R)):(\d+)(?:~(\d*\.?\d+))?$", v)
    arm, q = (m.group(1) or "") + m.group(2), int(m.group(3))
    if arm.startswith("V"):
        return tq.maxscale_voronoi(W, q, arm[3:], name=arm[1:3])[0]
    if arm.startswith("X"):
        return tq.maxscale_round(W, q, arm[1:], shrink=float(m.group(4) or 1.0))[0]
    return tq.round_matrix(W, arm, q)[0]


def top_sv2(M, iters=40, seed=0):
    """Squared largest singular value of a torch matrix, by power iteration."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(M.shape[1], generator=g).to(M.device)
    v = v / v.norm()
    for _ in range(iters):
        v = M.T @ (M @ v)
        n = v.norm()
        if float(n) == 0.0:
            return 0.0
        v = v / n
    return float((M @ v).norm() ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reader = ggufedit.open_model(args.model)
    by = {t.name: t for t in reader.tensors}
    live = run.live_kv_layers(reader)
    names = [f"blk.{il}.{k}.weight" for il in LAYERS for k in run.KINDS]
    mats, sv_w = {}, {}
    for name in names:
        mats[name] = ggufedit.tensor_float(by[name])
        sv_w[name] = top_sv2(torch.from_numpy(mats[name]).to(tq.DEV))
    for v in VARIANTS + EXTRA:
        spec, frob, spec_live, frob_live = [], [], [], []
        for index, name in enumerate(names):
            W = mats[name]
            Wh = round_one(W, v, index)
            E = torch.from_numpy(W - Wh.astype(np.float32)).to(tq.DEV)
            s = 10.0 * math.log10(max(top_sv2(E), 1e-30) / sv_w[name])
            f = 10.0 * math.log10(max(float(np.sum((W.astype(np.float64) - Wh.astype(np.float64)) ** 2)), 1e-30)
                                  / float(np.sum(W.astype(np.float64) ** 2)))
            spec.append(s)
            frob.append(f)
            il, kind = run.BLOCK.match(name).groups()
            if not (int(il) >= live and kind in ("attn_k", "attn_v")):
                spec_live.append(s)
                frob_live.append(f)
        row = {"variant": v, "matrices": len(names), "spectral_err_db_mean": sum(spec) / len(spec), "squared_err_db_mean": sum(frob) / len(frob),
               "matrices_read": len(spec_live), "spectral_err_db_mean_read": sum(spec_live) / len(spec_live),
               "squared_err_db_mean_read": sum(frob_live) / len(frob_live)}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
