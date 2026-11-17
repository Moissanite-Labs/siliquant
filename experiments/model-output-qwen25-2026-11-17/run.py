#!/usr/bin/env python3
"""Round the block matrices of a second language model and score its output (issue #3).

The plans are PREREG.md and PREREG-arms.md in this folder. One call handles a list of variants.
For each it builds a changed copy of the model file, scores it against the reference model's
saved outputs with llama.cpp's own tool, appends one line to a results file and deletes the copy.
A variant that is already in the results file is skipped, so a run that is stopped can be taken
up again.

The rounding functions and the scoring are those of the gemma record: harness/tq.py,
harness/ggufedit.py, harness/evalkl.py. Three things differ from that record's program, and the
record lists them: the matrices are rounded in several processes on a CPU, the two path-weighted
measures are left out because they are written for gemma's layers, and the cut codebooks of the
later records are added as arms.

Variants:
  unchanged          the reference file itself, scored against its own saved outputs
  identity           the block matrices written again as 16-bit floats (a control of the writer)
  XG:Q               llama.cpp's Q4_0 rule at Q levels: one scale per group G of weights, set by the
                     group's largest entry. G is a number of consecutive weights in a row, or R for a row
  VE8G:Q             the E8 Voronoi code with one scale per group G, set so that no block is shrunk
  ZBn@G:Q, EBn@G:Q   the integers, or E8, cut by a ball over n weights, one scale per group G,
  ZCn@G:Q, ECn@G:Q   or by a cross-polytope (the record ball-dimension-no-clip-rule-2026-11-17)
  PaBn@G:Q           E8 cut by a ball of the p-norm over 8 weights, p = 3/2 (shape/shapelib.py); n is 8
  SEB8@G:Q           E8 cut by a ball over 8 weights at the safe scale, a closed form with no search
  file:PATH          score an existing model file (for llama.cpp's own quantisers)

Usage, with NumPy and torch, llama.cpp's gguf package and llama-perplexity:
  python3 run.py --model REF.gguf --kld base.kld --work DIR --out results.jsonl --bin llama-perplexity identity X32:16 VE832:16
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "shape"))
import evalkl  # noqa: E402
import ggufedit  # noqa: E402

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")
MODELS = {
    "fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649": "qwen2.5-1.5b-instruct-fp16.gguf",
    "8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc": "qwen2.5-0.5b-instruct-fp16.gguf",
}
CUT = {"ZB": ("Z8", "ball"), "EB": ("E8", "ball"), "ZC": ("Z8", "cross"), "EC": ("E8", "cross")}
VARIANT = re.compile(r"^(?:(X)(\d+|R)|(VE8)(\d+|R)|(ZB|EB|ZC|EC|PaB)(\d+)@(\d+|R)|(SEB)(8)@(\d+|R)):(\d+)$")
GROUP_BALL = re.compile(r"^(ZBG|EBG)(\d+):b(\d+)s(\d+)$")       # extension 1 of PREREG-arms.md: one ball per group, bits chosen freely
ZERO_CENTRE = re.compile(r"^VE8Z(\d+|R):(\d+)$")                # extension 1: the E8 Voronoi code with a codeword at zero
SCALE_STEPS = {16: None, 8: 16, 6: 8}                            # bits of a scale -> steps of its grid per factor of two (None: not rounded)
_bounds = {}
_reader = None
_model_path = None
_bd = None


def db(x):
    return 10.0 * math.log10(x) if x > 0 else float("-inf")


def ball_module():
    global _bd
    if _bd is None:
        spec = importlib.util.spec_from_file_location("balldim", ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17" / "run.py")
        _bd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bd)
    return _bd


def reader():
    global _reader
    if _reader is None:
        _reader = ggufedit.open_model(_model_path)
    return _reader


def targets(rd):
    return [t for t in rd.tensors if BLOCK.match(t.name) and len(t.shape) == 2]


def matrix_of(t):
    return np.ascontiguousarray(ggufedit.tensor_float(t).reshape(tuple(int(d) for d in reversed(t.shape))), dtype=np.float32)


def finish(W, Y, bits, shrunk):
    """The rounded matrix as 16-bit floats, and its relative error against the reference in 64-bit floats."""
    Wh = Y.astype(np.float16)
    W64 = W.astype(np.float64)
    rel = float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2))
    return Wh, {"rel_err": rel, "bits": bits, "overload": shrunk}


def cut_round(W, lattice, shape, n, q, group):
    """A codebook cut from a lattice over n weights, with one scale per group and nothing clipped."""
    bd = ball_module()
    rows, cols = W.shape
    size = cols if group == "R" else int(group)
    if size % n or cols % size:
        raise SystemExit(f"a block of {n} does not tile a group of {size}, or the group a row of {cols}")
    bound, points = bd.codebook(lattice, shape, n, q)
    W64 = W.astype(np.float64)
    X = W64.reshape(-1, n)
    tb = bd.block_scales(X, lattice, shape, bound)
    tg = tb.reshape(-1, size // n).max(axis=1)
    t = np.repeat(tg, size // n)
    Y = bd.closest(lattice, X / t[:, None])
    out = bd.size_of(shape, Y) > bound + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(bd.SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = bd.size_of(shape, bd.closest(lattice, xo * mid[:, None])) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = bd.closest(lattice, xo * lo[:, None])
    return finish(W, (Y * t[:, None]).reshape(rows, cols), math.log2(points) / n + 16.0 / size, float(np.mean(out)))


def ball_bound(lattice, n, bits):
    """The largest bound at which the ball over n weights holds no more than 2^bits points: (bound, number of points)."""
    key = (lattice, n, bits)
    if key not in _bounds:
        bd = ball_module()
        cap = 1 << bits
        r_unit = math.exp(math.lgamma(n / 2.0 + 1.0) / n) / math.sqrt(math.pi)
        limit = int(1.08 * (r_unit * 2.0 ** (bits / n)) ** 2) + 2 * n
        step = 2 if lattice == "E8" else 1
        while True:
            total = bd.poly_pow(bd.shells8(lattice, "ball", limit), n // 8, limit)
            run, best = 0, None
            for r, c in enumerate(total):
                run += c
                if run > cap:
                    break
                if r % step == 0:
                    best = (r, run)
            if run > cap:
                break
            limit = int(limit * 1.3) + 8
        _bounds[key] = best
    return _bounds[key]


def group_ball_round(W, lattice, n, block_bits, scale_bits):
    """One ball per group of n weights, with block_bits for the point and scale_bits for the scale.

    With 16 bits the scale is the smallest at which the group rounds into the ball, as in cut_round.
    With 8 or 6 bits it is rounded up to a grid: the largest scale of the matrix divided by whole
    powers of the 16th or the 8th root of two. A scale under the grid's lowest step takes that step.
    The matrix's largest scale is stored once, and its 16 bits are counted.
    """
    bd = ball_module()
    rows, cols = W.shape
    if cols % n:
        raise SystemExit(f"a group of {n} does not tile a row of {cols}")
    bound, points = ball_bound(lattice, n, block_bits)
    W64 = W.astype(np.float64)
    X = W64.reshape(-1, n)
    t = bd.block_scales(X, lattice, "ball", bound)
    steps = SCALE_STEPS[scale_bits]
    extra = 0.0
    if steps is not None:
        top = float(t.max())
        k = np.floor(np.log2(top / np.maximum(t, 1e-300)) * steps + 1e-9)
        k = np.clip(k, 0, (1 << scale_bits) - 1)
        t = top * 2.0 ** (-k / steps)
        extra = 16.0 / W.size
    Y = bd.closest(lattice, X / t[:, None])
    out = bd.size_of("ball", Y) > bound + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(bd.SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = bd.size_of("ball", bd.closest(lattice, xo * mid[:, None])) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = bd.closest(lattice, xo * lo[:, None])
    return finish(W, (Y * t[:, None]).reshape(rows, cols), (math.log2(points) + scale_bits) / n + extra, float(np.mean(out)))


def shapelib_round(W, kind, q, group):
    """E8 cut by the ball of a p-norm (p = 3/2), or by a ball at the safe scale, with the toolkit of issue #10."""
    from fractions import Fraction
    import certified as C
    import shapelib as S
    rows, cols = W.shape
    size = cols if group == "R" else int(group)
    grp = "R" if group == "R" else size
    W64 = W.astype(np.float64)
    code = S.Codebook("E8", S.Shape(Fraction(3, 2) if kind == "PaB" else 2), q ** 8)
    bits = math.log2(code.N) / 8.0 + 16.0 / size
    if kind == "SEB":
        t = np.repeat(C.safe_scales(W64, code, grp), size // 8)
        X = W64.reshape(-1, 8)
        Y = code.closest(X / t[:, None])
        return finish(W, (Y * t[:, None]).reshape(rows, cols), bits, float(np.mean(~code.inside(Y))))
    tb = S.block_scales(W64.reshape(-1, 8), code)
    _, _, shrunk, Wh = S.round_groups(W64, code, grp, tb, keep=True)
    return finish(W, Wh, bits, shrunk)


def round_one(task):
    """Round one block matrix for one variant. Returns (tensor name, rounded matrix, what happened)."""
    variant, name = task
    import torch
    torch.set_num_threads(1)
    import tq
    t = next(t for t in reader().tensors if t.name == name)
    W = matrix_of(t)
    if variant == "identity":
        Wh, info = finish(W, W, 16.0, 0.0)
    elif ZERO_CENTRE.match(variant):
        z = ZERO_CENTRE.match(variant)
        code = tq.Code("E8", int(z.group(2)))
        kept = code.centre
        code.centre = torch.zeros_like(kept)           # the harness subtracts the mean of its codewords: here it does not
        try:
            Wh, info = tq.maxscale_voronoi(W, int(z.group(2)), z.group(1) if z.group(1) == "R" else int(z.group(1)), name="E8")
        finally:
            code.centre = kept
    elif GROUP_BALL.match(variant):
        g = GROUP_BALL.match(variant)
        Wh, info = group_ball_round(W, "Z8" if g.group(1) == "ZBG" else "E8", int(g.group(2)), int(g.group(3)), int(g.group(4)))
    else:
        m = VARIANT.match(variant)
        q = int(m.group(11))
        if m.group(1):
            Wh, info = tq.maxscale_round(W, q, m.group(2) if m.group(2) == "R" else int(m.group(2)))
        elif m.group(3):
            Wh, info = tq.maxscale_voronoi(W, q, m.group(4) if m.group(4) == "R" else int(m.group(4)), name="E8")
        elif m.group(5) == "PaB":
            Wh, info = shapelib_round(W, "PaB", q, m.group(7))
        elif m.group(5):
            lattice, shape = CUT[m.group(5)]
            Wh, info = cut_round(W, lattice, shape, int(m.group(6)), q, m.group(7))
        else:
            Wh, info = shapelib_round(W, "SEB", q, m.group(10))
    return name, Wh, {"rel_err": info["rel_err"], "bits": info["bits"], "overload": info.get("overload"),
                      "energy": float(np.sum(W.astype(np.float64) ** 2)), "size": int(W.size)}


def stats_of(infos):
    per_kind, weights, bits, energy, energy_err, over = {}, 0, 0.0, 0.0, 0.0, 0.0
    for name, info in infos.items():
        per_kind.setdefault(BLOCK.match(name).group(2), []).append(info["rel_err"])
        weights += info["size"]
        bits += info["bits"] * info["size"]
        energy += info["energy"]
        energy_err += info["energy"] * info["rel_err"]
        over += (info["overload"] or 0.0) * info["size"]
    every = [v for vs in per_kind.values() for v in vs]
    return {"matrices": len(every), "weights": weights, "bits_per_weight": bits / weights,
            "weight_err_db_mean": sum(db(v) for v in every) / len(every), "weight_err_db_pooled": db(energy_err / energy),
            "weight_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(per_kind.items())},
            "share_out_of_range": over / weights}


def compare_file(path):
    """Bits per weight and weight error of the block matrices of another model file, and whether any other tensor differs."""
    other = {t.name: t for t in ggufedit.open_model(path).tensors}
    infos, differ = {}, []
    for t in reader().tensors:
        o = other[t.name]
        if BLOCK.match(t.name) and len(t.shape) == 2:
            W = ggufedit.tensor_float(t).astype(np.float64)
            Wh = ggufedit.tensor_float(o).astype(np.float64).reshape(W.shape)
            e = float(np.sum(W ** 2))
            infos[t.name] = {"rel_err": float(np.sum((W - Wh) ** 2)) / e, "bits": 8.0 * int(o.n_bytes) / W.size, "overload": None, "energy": e, "size": int(W.size)}
        elif o.tensor_type != t.tensor_type or not np.array_equal(np.asarray(o.data), np.asarray(t.data)):
            differ.append(t.name)
    row = stats_of(infos)
    row["other_tensors_that_differ"] = differ
    return row


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global _model_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default="llama-perplexity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=int, default=0, help="round only the first N block matrices and do not score (a smoke test)")
    ap.add_argument("--metrics-only", action="store_true", help="build each variant and record its weight errors, without scoring it")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    _model_path = args.model
    digest = sha256_of(args.model)
    if digest not in MODELS:
        raise SystemExit("FAIL: the model file is not one the plan names")
    names = [t.name for t in targets(reader())]
    if args.only:
        names = names[: args.only]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["variant"] for line in f if line.strip()}
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        label = "file:" + os.path.basename(v[5:]) if v.startswith("file:") else v
        if label in done:
            print(f"{label}: already in the results file", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        row = {"variant": label, "model": MODELS[digest]}
        if v == "unchanged":
            path, made = args.model, False
        elif v.startswith("file:"):
            path, made = v[5:], False
            row.update(compare_file(path))
            row["file_sha256"] = sha256_of(path)
        else:
            if v != "identity" and not VARIANT.match(v) and not GROUP_BALL.match(v) and not ZERO_CENTRE.match(v):
                raise SystemExit(f"unknown variant {v}")
            path, made = os.path.join(args.work, re.sub(r"[^A-Za-z0-9_.+-]", "_", v) + ".gguf"), True
            replace, infos = {}, {}
            with Pool(args.workers) as pool:
                for name, Wh, info in pool.imap_unordered(round_one, [(v, n) for n in names], chunksize=1):
                    replace[name], infos[name] = Wh, info
            row.update(stats_of(infos))
            row["build_seconds"] = round(time.time() - t0, 1)
            if args.only or args.metrics_only:
                print(json.dumps(row), flush=True)
                if args.metrics_only and not args.only:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                continue
            ggufedit.write_copy(reader(), path, replace)
            del replace
        t1 = time.time()
        scores, _ = evalkl.score(path, args.kld, binary=args.bin, extra=("-t", str(args.threads)))
        row.update(scores)
        row["score_seconds"] = round(time.time() - t1, 1)
        if made:
            os.remove(path)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
