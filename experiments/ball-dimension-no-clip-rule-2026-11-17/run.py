#!/usr/bin/env python3
"""A ball, or a cross-polytope, over more than 8 weights, under the rule that clips nothing.

The plan is PREREG.md in this folder. This file is the only program of the run.

The record region-shapes-no-clip-rule-2026-11-17 cut the integers and E8 by a ball over blocks
of 8 weights. Here a block has n = 8, 16, 32, 64 or 128 weights. A codebook is every point of
Z^n, or of E8 repeated n/8 times, whose squared length (ball) or sum of sizes (cross-polytope)
is at most a bound. The bound is the largest at which the codebook has no more than q^n points,
so an arm never has more bits than q levels per weight. Rounding needs nothing new: each weight
is rounded to an integer, or each 8 weights to E8, and only the test of whether the block is in
the codebook looks at all n weights.

The scale of a group is the smallest at which every block of the group rounds to a point of
the codebook. The scalar rule X and the E8 Voronoi code VE8 are run with the harness, as in
the record e8-gain-largest-entry-rule-2026-11-17.

Usage, from the root of a checkout, with NumPy and torch:
  python3 experiments/ball-dimension-no-clip-rule-2026-11-17/run.py --smoke
  python3 experiments/ball-dimension-no-clip-rule-2026-11-17/run.py --weights weights.npz --gguf qwen2.5-1.5b-instruct-fp16.gguf > results.out
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
sys.path.insert(0, str(ROOT / "lattice"))
sys.path.insert(0, str(ROOT / "harness"))
import lattices as L  # noqa: E402

_spec = importlib.util.spec_from_file_location("regions", ROOT / "experiments" / "region-shapes-no-clip-rule-2026-11-17" / "run.py")
regions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regions)

WEIGHTS_SHA256 = "b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e"
GGUF_SHA256 = "fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649"
REAL_PATTERN = re.compile(r"^(encoder|decoder)\.blocks\.(0|5)\..*weight$")
REAL_COUNT = 32
QWEN_LAYERS = (0, 13, 27)
QWEN_KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
QWEN_WEIGHTS = 393_216                 # weights kept of each qwen matrix: its first rows, up to this many
DIMS = (8, 16, 32, 64, 128)
CELLS = (("32", 16), ("128", 16), ("R", 16), ("R", 32))
ARMS = (("ZB", "Z8", "ball"), ("EB", "E8", "ball"), ("ZC", "Z8", "cross"), ("EC", "E8", "cross"))
BISECT_STEPS = 14
SHRINK_STEPS = 12
_E8 = L.E8()
_cache = {}
_weights_path = None
_gguf_path = None


def db(x):
    return 10.0 * math.log10(x)


# ---------------------------------------------------------------------------
# Exact counts in n dimensions
# ---------------------------------------------------------------------------

def poly_mul(a, b, limit):
    """Product of two polynomials with whole coefficients, cut at degree `limit`, exactly.

    The coefficients of each are packed into one whole number, the two are multiplied, and the
    product is unpacked (Kronecker's substitution). A field is wide enough for the largest
    coefficient the product can have, so nothing carries from one field into the next.
    """
    a, b = a[: limit + 1], b[: limit + 1]
    width = ((sum(a) * max(b)).bit_length() + 8 + 7) // 8        # bytes per coefficient
    pa = int.from_bytes(b"".join(c.to_bytes(width, "little") for c in a), "little")
    pb = int.from_bytes(b"".join(c.to_bytes(width, "little") for c in b), "little")
    raw = (pa * pb).to_bytes(width * (len(a) + len(b)), "little")
    return [int.from_bytes(raw[i * width:(i + 1) * width], "little") for i in range(min(limit, len(a) + len(b) - 2) + 1)]


def poly_pow(a, k, limit):
    out = None
    while k:
        if k & 1:
            out = a[: limit + 1] if out is None else poly_mul(out, a, limit)
        k >>= 1
        if k:
            a = poly_mul(a, a, limit)
    return out


def shells8(lattice, shape, limit):
    """The number of points of Z^8 or E8 of each size 0..limit, from the exact counts of the region study."""
    table = regions.counts(lattice, shape, max(limit, 8))
    return [table[0]] + [table[r] - table[r - 1] for r in range(1, limit + 1)]


def codebook(lattice, shape, n, q):
    """The largest bound at which the codebook over n weights has at most q^n points: (bound, number of points)."""
    key = (lattice, shape, n, q)
    if key not in _cache:
        cap = q ** n
        if shape == "ball":
            r_unit = math.exp(math.lgamma(n / 2.0 + 1.0) / n) / math.sqrt(math.pi)     # radius of the ball of volume 1
            limit = int(1.08 * (r_unit * q) ** 2) + 2 * n
        else:
            limit = int(1.05 * q * math.exp(math.lgamma(n + 1.0) / n) / 2.0) + n        # half the edge of the cross-polytope of volume q^n
        while True:
            total = poly_pow(shells8(lattice, shape, limit), n // 8, limit)
            step = 2 if (lattice == "E8" and shape == "ball") else 1          # squared lengths in E8 are even
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
        _cache[key] = best
    return _cache[key]


def scan_count(lattice, shape, n, bound):
    """The same count by listing, for n = 16 and a small bound. For the self-test."""
    one = shells8(lattice, shape, bound)
    return sum(one[a] * one[b] for a in range(bound + 1) for b in range(bound + 1 - a))


def selftest():
    out = []
    ok = True
    for lattice, shape in (("Z8", "ball"), ("E8", "ball"), ("Z8", "cross"), ("E8", "cross")):
        for q in (16, 32):
            ok &= codebook(lattice, shape, 8, q) == regions.radius_for(lattice, shape, q)
    out.append((ok, "over 8 weights the codebooks are those of the region study, bound and size, for 4 arms at q = 16 and 32"))
    ok = True
    for lattice, shape in (("Z8", "ball"), ("E8", "ball"), ("Z8", "cross"), ("E8", "cross")):
        for bound in (0, 2, 5, 8, 11):
            total = poly_pow(shells8(lattice, shape, bound), 2, bound)
            ok &= sum(total) == scan_count(lattice, shape, 16, bound)
    out.append((ok, "over 16 weights the count equals a sum over pairs of blocks of 8, for 20 small codebooks"))
    z = poly_pow([1, 2] + [0, 0, 2] * 0 + [0] * 0, 1, 1)
    theta = [0] * 26
    for k in range(-5, 6):
        theta[k * k] += 1
    by_hand = poly_pow(theta, 3, 25)           # points of Z^3 by squared length
    out.append((by_hand[:7] == [1, 6, 12, 8, 6, 24, 24] and z == [1, 2], "the product of polynomials gives 1, 6, 12, 8, 6, 24, 24 points of Z^3 of squared length 0 to 6"))
    x = np.random.default_rng(3).normal(size=(64, 32)) * 3.0
    for lattice in ("Z8", "E8"):
        y = closest(lattice, x)
        again = closest(lattice, y)
        ok = bool(np.all(again == y))
        out.append((ok, f"rounding to {lattice} over 32 weights gives points that round to themselves"))
    return out


# ---------------------------------------------------------------------------
# Rounding under the rule that clips nothing
# ---------------------------------------------------------------------------

def closest(lattice, X):
    """Nearest point of Z^n or of E8 repeated: X is (N, n), with n a multiple of 8."""
    if lattice == "Z8":
        return np.rint(X)
    return _E8.closest(X.reshape(-1, 8)).reshape(X.shape)


def size_of(shape, Y):
    return np.abs(Y).sum(axis=1) if shape == "cross" else (Y ** 2).sum(axis=1)


def block_scales(X, lattice, shape, bound):
    """For every block of n weights the smallest scale at which it rounds to a point of the codebook.

    Rounding moves a point by at most sqrt(n)/2 in length and n/2 in the sum of sizes (the integers
    are the worse of the two lattices), which gives the two ends of the bisection.
    """
    n = X.shape[1]
    if shape == "cross":
        g, r, margin = np.abs(X).sum(axis=1), float(bound), n / 2.0
    else:
        g, r, margin = np.sqrt((X ** 2).sum(axis=1)), math.sqrt(bound), math.sqrt(n) / 2.0
    g = np.maximum(g, 1e-30)
    lo = g / (r + margin) * 0.99
    hi = g / max(r - margin, 0.05 * r) * 1.01
    for _ in range(BISECT_STEPS):
        mid = np.sqrt(lo * hi)
        ok = size_of(shape, closest(lattice, X / mid[:, None])) <= bound + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def round_groups(W, lattice, shape, bound, n, group, tb):
    """Round a matrix with one scale per group. Returns (relative error, share of blocks shrunk, scale term in dB)."""
    rows, cols = W.shape
    size = cols if group == "R" else int(group)
    X = W.reshape(-1, n)
    tg = tb.reshape(-1, size // n).max(axis=1)
    t = np.repeat(tg, size // n)
    Y = closest(lattice, X / t[:, None])
    out = size_of(shape, Y) > bound + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = size_of(shape, closest(lattice, xo * mid[:, None])) <= bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = closest(lattice, xo * lo[:, None])
    Wh = (Y * t[:, None]).reshape(rows, cols).astype(np.float16).astype(np.float64)
    return float(np.sum((W - Wh) ** 2) / np.sum(W ** 2)), float(np.mean(out)), tg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def matrix_names():
    names = [f"{k}/{s}" for k in ("G", "T") for s in range(3)]
    if _weights_path:
        raw = Path(_weights_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != WEIGHTS_SHA256:
            raise SystemExit("FAIL: the weights file does not have the SHA-256 named in the plan")
        z = np.load(_weights_path)
        keys = sorted(k for k in z.files if REAL_PATTERN.match(k) and z[k].ndim == 2)
        if len(keys) != REAL_COUNT:
            raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {REAL_COUNT}")
        names += ["whisper/" + k for k in keys]
    if _gguf_path:
        h = hashlib.sha256()
        with open(_gguf_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != GGUF_SHA256:
            raise SystemExit("FAIL: the model file does not have the SHA-256 named in the plan")
        names += [f"qwen/blk.{il}.{kind}.weight" for il in QWEN_LAYERS for kind in QWEN_KINDS]
    return names


def matrix(name):
    """A matrix of the study as 32-bit floats. Its row length is a multiple of 128."""
    kind, rest = name.split("/", 1)
    if rest == "smoke":
        return np.random.default_rng(7).uniform(-1, 1, size=(16, 256)).astype(np.float32)
    if kind in ("G", "T"):
        rng = np.random.default_rng({"G": 6000, "T": 7000}[kind] + int(rest))
        W = rng.normal(size=(512, 2048)) if kind == "G" else rng.standard_t(4, size=(512, 2048))
        return W.astype(np.float32)
    if kind == "whisper":
        return np.ascontiguousarray(np.load(_weights_path)[rest], dtype=np.float32)
    import ggufedit
    reader = ggufedit.open_model(_gguf_path)
    t = next(t for t in reader.tensors if t.name == rest)
    W = ggufedit.tensor_float(t).reshape(tuple(int(d) for d in reversed(t.shape)))
    keep = max(1, QWEN_WEIGHTS // W.shape[1])
    return np.ascontiguousarray(W[:keep], dtype=np.float32)


def run_matrix(name):
    import torch
    import tq
    torch.set_num_threads(1)
    W32 = matrix(name)
    W = W32.astype(np.float64)
    rows, cols = W.shape
    out = []
    for group, q in CELLS:
        size = cols if group == "R" else int(group)
        d = np.abs(W).reshape(-1, size).max(axis=1) / (q / 2.0)
        x = tq.maxscale_round(W32, q, "R" if group == "R" else size)[1]
        v = tq.maxscale_voronoi(W32, q, "R" if group == "R" else size, name="E8")[1]
        out.append({"matrix": name, "arm": "X", "n": 1, "group": group, "size": size, "q": q, "bits": x["bits"], "rel_err": x["rel_err"]})
        out.append({"matrix": name, "arm": "VE8", "n": 8, "group": group, "size": size, "q": q, "bits": v["bits"], "rel_err": v["rel_err"], "shrunk": v["overload"]})
    for q in sorted({q for _, q in CELLS}):
        for arm, lattice, shape in ARMS:
            for n in DIMS:
                cells = [(g, qq) for g, qq in CELLS if qq == q and (cols if g == "R" else int(g)) % n == 0]
                if not cells:
                    continue
                bound, points = codebook(lattice, shape, n, q)
                tb = block_scales(W.reshape(-1, n), lattice, shape, bound)
                for group, _ in cells:
                    size = cols if group == "R" else int(group)
                    rel, shrunk, tg = round_groups(W, lattice, shape, bound, n, group, tb)
                    d = np.abs(W).reshape(-1, size).max(axis=1) / (q / 2.0)
                    out.append({"matrix": name, "arm": arm, "n": n, "group": group, "size": size, "q": q, "bound": bound,
                                "bits": math.log2(points) / n + 16.0 / size, "rel_err": rel, "shrunk": shrunk,
                                "scale_term": db(float(np.sum(d ** 2)) / float(np.sum(tg ** 2)))})
    return out


def keyed(name):
    return name, run_matrix(name)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(rows, checks, out):
    fams = [f for f in ("G", "T", "whisper", "qwen") if any(r["matrix"].split("/")[0] == f for r in rows)]
    by = {(r["matrix"], r["arm"], r["n"], r["group"], r["q"]): r for r in rows}
    out.append("0. Checks")
    for ok, text in checks:
        out.append(f"   {'ok  ' if ok else 'FAIL'} {text}")
    cut = [r for r in rows if r["arm"] not in ("X", "VE8")]
    shr = [r for r in cut if r["shrunk"] > 0]
    out.append(f"   runs of a cut codebook where a block lay outside the codebook at its group's scale and was shrunk: {len(shr)} of {len(cut)}"
               + (f", largest share of blocks {max(r['shrunk'] for r in shr):.2e}" if shr else ""))
    prior_path = ROOT / "experiments" / "region-shapes-no-clip-rule-2026-11-17" / "results.json"
    prior = {(r["matrix"].replace("real/", "whisper/"), r["arm"], r["group"], r["q"]): r["rel_err"] for r in json.loads(prior_path.read_text())}
    shared = [(r, prior[(r["matrix"], r["arm"], r["group"], r["q"])]) for r in cut if r["n"] == 8 and (r["matrix"], r["arm"], r["group"], r["q"]) in prior]
    if shared:
        worst = max(abs(r["rel_err"] / e - 1.0) for r, e in shared)
        out.append(f"   over 8 weights, against the region study on the runs the two share: {len(shared)} runs, largest relative difference in the error {worst:.1e}")
    over = [r for r in cut if r["bits"] > by[(r["matrix"], "X", 1, r["group"], r["q"])]["bits"] + 1e-12]
    out.append(f"   runs of a cut codebook with more bits than the scalar rule in the same cell: {len(over)}")
    out.append("")
    out.append("1. The codebooks: bits per weight without the scale, and the points of the codebook over q^n")
    out.append("   arm | q | " + " | ".join(f"n = {n}" for n in DIMS))
    for q in sorted({q for _, q in CELLS}):
        for arm, lattice, shape in ARMS:
            cells = []
            for n in DIMS:
                bound, points = codebook(lattice, shape, n, q)
                cells.append(f"bound {bound}, {math.log2(points) / n:.4f} bits, {points / q ** n:.3f}")
            out.append(f"   {arm} | {q} | " + " | ".join(cells))
    out.append("")
    out.append("2. Gain over the scalar rule X in the same cell, in dB, mean over the matrices of a family. A positive number means a smaller error.")
    out.append("   VE8 is the E8 Voronoi code of the second record. No arm has more bits than X.")
    for group, q in CELLS:
        out.append(f"   group {group}, q = {q}")
        out.append("   family | X dB | VE8 | " + " | ".join(f"{arm} n={n}" for arm, _, _ in ARMS for n in DIMS) + " | matrices")
        for f in fams:
            ms = sorted({r["matrix"] for r in rows if r["matrix"].split("/")[0] == f})
            if not ms:
                continue
            cells = []
            for arm, _, _ in ARMS:
                for n in DIMS:
                    v = [db(by[(m, "X", 1, group, q)]["rel_err"] / by[(m, arm, n, group, q)]["rel_err"]) for m in ms if (m, arm, n, group, q) in by]
                    cells.append(f"{np.mean(v):+.2f}" if len(v) == len(ms) else "-")
            ve = [db(by[(m, "X", 1, group, q)]["rel_err"] / by[(m, "VE8", 8, group, q)]["rel_err"]) for m in ms]
            out.append(f"   {f:>7} | {np.mean([db(by[(m, 'X', 1, group, q)]['rel_err']) for m in ms]):.2f} | {np.mean(ve):+.2f} | " + " | ".join(cells) + f" | {len(ms)}")
    out.append("")
    out.append("3. Against the E8 Voronoi code in the same cell, in dB: mean (min, max), and the matrices on which the arm is ahead")
    out.append("   family | group | q | arm | n | bits | VE8 bits | over VE8: mean (min, max) | ahead on | scale term | matrices")
    for f in fams:
        ms = sorted({r["matrix"] for r in rows if r["matrix"].split("/")[0] == f})
        for group, q in CELLS:
            for arm, _, _ in ARMS:
                for n in DIMS:
                    rs = [by[(m, arm, n, group, q)] for m in ms if (m, arm, n, group, q) in by]
                    if len(rs) != len(ms) or not rs:
                        continue
                    g = [db(by[(r["matrix"], "VE8", 8, group, q)]["rel_err"] / r["rel_err"]) for r in rs]
                    out.append(f"   {f:>7} | {group:>3} | {q} | {arm} | {n:>3} | {np.mean([r['bits'] for r in rs]):.3f} | {np.mean([by[(r['matrix'], 'VE8', 8, group, q)]['bits'] for r in rs]):.3f} | "
                               f"{np.mean(g):+.2f} ({min(g):+.2f}, {max(g):+.2f}) | {sum(x > 0 for x in g)} | {np.mean([r['scale_term'] for r in rs]):+.2f} | {len(rs)}")
    out.append("")
    out.append("4. At equal bits across groups: relative squared error in dB of the best dimension of each arm, by the bits of its cell")
    out.append("   family | group | q | bits of X | X dB | VE8 dB | " + " | ".join(f"{arm}: best n, dB" for arm, _, _ in ARMS))
    for f in fams:
        ms = sorted({r["matrix"] for r in rows if r["matrix"].split("/")[0] == f})
        for group, q in CELLS:
            cells = []
            for arm, _, _ in ARMS:
                best = None
                for n in DIMS:
                    if all((m, arm, n, group, q) in by for m in ms):
                        v = float(np.mean([db(by[(m, arm, n, group, q)]["rel_err"]) for m in ms]))
                        if best is None or v < best[1]:
                            best = (n, v)
                cells.append(f"n = {best[0]}, {best[1]:.2f}" if best else "-")
            out.append(f"   {f:>7} | {group:>3} | {q} | {np.mean([by[(m, 'X', 1, group, q)]['bits'] for m in ms]):.3f} | {np.mean([db(by[(m, 'X', 1, group, q)]['rel_err']) for m in ms]):.2f} | "
                       f"{np.mean([db(by[(m, 'VE8', 8, group, q)]['rel_err']) for m in ms]):.2f} | " + " | ".join(cells))


def main():
    global _weights_path, _gguf_path, DIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the code path on a matrix that is not in the study and print nothing but ok")
    ap.add_argument("--weights", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    ap.add_argument("--gguf", help="path to qwen2.5-1.5b-instruct-fp16.gguf")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    checks = selftest()
    if args.smoke:
        rows = run_matrix("G/smoke")
        out = []
        summarise(rows, checks, out)
        ok = all(c[0] for c in checks) and all(0 < r["rel_err"] < 1 for r in rows) and len(out) > 30
        print("ok" if ok else "FAIL", f"({len(rows)} rows, {len(out)} lines, {time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 1)
    if not all(c[0] for c in checks):
        raise SystemExit("FAIL: a check of the code failed: " + "; ".join(t for ok, t in checks if not ok))
    _weights_path, _gguf_path = args.weights, args.gguf
    names = matrix_names()
    for q in sorted({q for _, q in CELLS}):                     # counted once here, then inherited by the workers
        for _, lattice, shape in ARMS:
            for n in DIMS:
                codebook(lattice, shape, n, q)
    partial = HERE / "results.partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            done[rec["matrix"]] = rec["rows"]
    todo = [m for m in names if m not in done]
    print(f"{len(names)} matrices, {len(done)} already done", file=sys.stderr, flush=True)
    with Pool(args.workers) as pool, partial.open("a") as fh:
        for k, (name, part) in enumerate(pool.imap_unordered(keyed, todo, chunksize=1)):
            done[name] = part
            fh.write(json.dumps({"matrix": name, "rows": part}) + "\n")
            fh.flush()
            print(f"{k + 1} of {len(todo)} matrices, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    rows = [r for m in names for r in done[m]]
    (HERE / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    out = []
    summarise(rows, checks, out)
    out.append("")
    out.append(f"rows: {len(rows)}; matrices: {len(names)}; seconds: {time.time() - t0:.0f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
