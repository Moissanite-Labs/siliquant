#!/usr/bin/env python3
"""Lattices from extended Hamming codes, in one ball per group of 128 weights, on real weights.

The plan is PREREG.md in this folder.

Variants:
  A8BG128:bBs16     A_8 = C8 + 2Z^8 over the root of 2, which is E8 turned by the Hadamard matrix of order 8 (the
                    arm HBG of e8-hadamard-real-weights-2026-11-22), decoded by this folder's decoder: a check
  A16BG128:bBs16    A_16 = C16 + 2Z^16, C16 the extended Hamming code [16,11,4], in one ball per group of 128
  A32BG128:bBs16    A_32 = C32 + 2Z^32, C32 the extended Hamming code [32,26,4], the same
  ARM+K10           the arm with the ten loud columns of every ffn_down left as in the reference, as in that record
  ZBG128:bBs16, EBG128:bBs16, HBG128:bBs16, and with +K10: the arms of e8-hadamard-real-weights-2026-11-22,
                    built by its program, which is not changed

B is 560, which is 4.5 bits a weight, or 368, which is 3.0. One 16-bit scale per group of 128 weights.

The new arms are rounded by ball_round below, a copy of group_ball_round of model-output-qwen25-2026-11-17 for
16-bit scales in which the nearest-point function, the bound of the ball and the margin of the bisection are
given. test_hamming.py holds it to that function bit for bit on integer levels and on E8. The bound is the
largest squared radius at which the ball holds no more than 2^B points, counted exactly (hamming.py), and the
bits a weight are (log2 of that count + 16) / 128, as for the other arms. Bits are counted, not stored: no
file is written in this format.

A8 is decoded in the units of E8: the point nearest to x is closest(x times the root of 2) over the root of 2,
with E8's bound and margin, so that its build takes the same steps as HBG's and its row can be compared with
HBG's committed row.

Rows have the same fields as that record's rows, including what the arm does to the small weights.

Usage:
  python3 experiments/hamming-lattices-real-weights-2026-11-24/run.py --model M.gguf --kld base.kld --work DIR \
      --out results.jsonl --bin llama-perplexity VARIANT [VARIANT ...]
  ... --metrics-only --out metrics.jsonl
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
import hamming  # noqa: E402

_spec = importlib.util.spec_from_file_location("eh_run", HERE.parent / "e8-hadamard-real-weights-2026-11-22" / "run.py")
EH = importlib.util.module_from_spec(_spec)
sys.modules["eh_run"] = EH
_spec.loader.exec_module(EH)
Q, LC = EH.Q, EH.LC

NEW = re.compile(r"^A(?P<n>8|16|32)BG128:b(?P<bits>560|368)s16(?P<keep>\+K10)?$")
M_OF = {"8": 3, "16": 4, "32": 5}
ROOT2 = math.sqrt(2.0)


def ball_round(W64, n, block_bits, closest, bound, points, margin, bisect_steps, shrink_steps):
    """group_ball_round of the base program with 16-bit scales, with the lattice's pieces given:
    (rounded matrix in 64-bit floats, bits a weight, share of groups shrunk)."""
    rows, cols = W64.shape
    if cols % n:
        raise SystemExit(f"a group of {n} does not tile a row of {cols}")
    X = W64.reshape(-1, n)
    g = np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)
    r = math.sqrt(bound)
    lo = g / (r + margin) * 0.99
    hi = g / max(r - margin, 0.05 * r) * 1.01
    for _ in range(bisect_steps):
        mid = np.sqrt(lo * hi)
        ok = (closest(X / mid[:, None]) ** 2).sum(axis=1) <= bound + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    t = hi
    Y = closest(X / t[:, None])
    out = (Y ** 2).sum(axis=1) > bound + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        a, b = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(shrink_steps):
            mid = 0.5 * (a + b)
            good = (closest(xo * mid[:, None]) ** 2).sum(axis=1) <= bound + 1e-9
            a = np.where(good, mid, a)
            b = np.where(good, b, mid)
        Y[out] = closest(xo * a[:, None])
    Wh = (Y * t[:, None]).reshape(rows, cols)
    return Wh, (math.log2(points) + 16) / n, float(np.mean(out))


def pieces(lattice, bits):
    """(closest, bound, points, margin) of a lattice for a ball over 128 weights with `bits` bits."""
    bd = Q.ball_module()
    if lattice in ("Z8", "E8"):
        bound, points = Q.ball_bound(lattice, 128, bits)
        return (lambda X: bd.closest(lattice, X)), bound, points, math.sqrt(128) / 2.0
    m = M_OF[lattice[1:]]
    if m == 3:                                   # A_8 in the units of E8
        bound, points = Q.ball_bound("E8", 128, bits)
        return (lambda X: hamming.closest(X * ROOT2, 3) / ROOT2), bound, points, math.sqrt(128) / 2.0
    bound, points = hamming.ball_bound(m, 128, bits)
    return (lambda X: hamming.closest(X, m)), bound, points, math.sqrt(hamming.covering_sq(m, 128))


def round_new(variant, kind, W):
    """One block matrix rounded by a new arm, with the fields of the record e8-hadamard-real-weights-2026-11-22."""
    m = NEW.match(variant)
    W64 = W.astype(np.float64)
    rest, cols = W64, None
    if m.group("keep") and kind == "ffn_down":
        cols = LC.loud(W, LC.K)
        rest = W64.copy()
        rest[:, cols] = 0.0
    bd = Q.ball_module()
    closest, bound, points, margin = pieces("A" + m.group("n"), int(m.group("bits")))
    Y, bits, shrunk = ball_round(rest, 128, int(m.group("bits")), closest, bound, points, margin, bd.BISECT_STEPS, bd.SHRINK_STEPS)
    if cols is not None:
        Y[:, cols] = W64[:, cols]
        bits += (len(cols) * W.shape[0] * 16.0 + len(cols) * 16.0) / W.size
    Wh, info = Q.finish(W, Y, bits, shrunk)
    Wh64 = Wh.astype(np.float64)
    size, err2 = np.abs(W64), (W64 - Wh64) ** 2
    rms = math.sqrt(float(np.mean(W64 ** 2)))
    small = {}
    for d in EH.SMALL:
        mask = size < d * rms
        small[str(d)] = [int(mask.sum()), int(np.count_nonzero(Wh64[mask] == 0.0)), float(err2[mask].sum()), float((W64[mask] ** 2).sum())]
    info.update({"energy": float(np.sum(W64 ** 2)), "size": int(W.size), "kept_columns": 0 if cols is None else int(len(cols)),
                 "zeros": int(np.count_nonzero(Wh64 == 0.0)), "small": small})
    return Wh, info


def round_any(variant, kind, W):
    if NEW.match(variant):
        return round_new(variant, kind, W)
    return EH.round_matrix(variant, kind, W)


def one(task):
    variant, name = task
    import torch
    torch.set_num_threads(1)
    Wh, info = round_any(variant, Q.BLOCK.match(name).group(2), Q.matrix_of(LC.tensor(name)))
    return name, Wh, info


def known(v):
    return bool(NEW.match(v) or EH.VARIANT.match(v))


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
        if not known(v):
            raise SystemExit(f"unknown variant {v}")
    if not args.metrics_only and not (args.kld and args.work):
        raise SystemExit("scoring needs --kld and --work")
    Q._model_path = args.model
    model = Q.MODELS.get(Q.sha256_of(args.model))
    if model not in EH.ACCEPTED:
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
        row.update(EH.small_stats(infos))
        row["build_seconds"] = round(time.time() - t0, 1)
        if not args.metrics_only:
            path = os.path.join(args.work, "hl_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v)[:80] + ".gguf")
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
