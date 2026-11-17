#!/usr/bin/env python3
"""The fold of Foundation-0/incentre-lattices as a path of codebooks over 3 weights (issue #9).

The plan is PREREG.md in this folder. This file is the only program of the run.

The fold is a stored sequence of states. Each state is a periodic set of points in 3
dimensions: a cell of three lattice vectors and 48 points in it. The sequence starts at the
open packing F_3 (six points per cube, at the incentres of its pyramids) and ends at a close
packing, hexagonal on one stored path and face-centred cubic on the other.

Three things are measured.

1. The cell of every eighth state: its normalised second moment G, by sampling. That is what
   the state is worth as a rounding rule with fine cells, against the cube's 1/12.
2. Codebooks of no more than 4,096 points cut from seven states by a ball about one of the
   points, which is 4 bits a weight for a block of 3. They round matrices under the rule that
   clips nothing: one scale per group of weights, the smallest at which every block of the
   group rounds to a point of the codebook. Beside them, cut the same way: the integers, the
   face-centred and body-centred cubic lattices, and the integers cut by a cross-polytope.
   Beside those: the scalar rule and the E8 Voronoi code from the harness, and the integers
   and E8 in a ball over 8 and over 24 weights from the record
   ball-dimension-no-clip-rule-2026-11-17.
3. A fitted codebook of 4,096 points: Lloyd's algorithm on blocks of training matrices divided
   by the scale of their group. It is started from a sample of the blocks, and again from the
   codebook of the fold's close-packed end, which is the nearest thing here to "annealing
   along the fold and then optimising".

Usage, from the root of a checkout, with NumPy, SciPy and torch:
  python3 experiments/fold-path-codebooks-2026-11-17/run.py --smoke
  python3 experiments/fold-path-codebooks-2026-11-17/run.py --paths DIR --weights weights.npz > results.out
DIR holds the two stored paths, path.npz and path_fcc.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "harness"))

PATH_SHA256 = {
    "path.npz": "510d0c79016068198b8f5af8b127067a200498557bcdb01e1da27ac4e78bddd5",
    "path_fcc.npz": "010bdabf40c8c7b23a8809bc3a4d107f3417d4e602b98b0723ef6cc7f2e717b1",
}
WEIGHTS_SHA256_PREFIX = "b6c8ee50"
POINTS = 4096                          # 4 bits a weight for a block of 3
STATES = (0, 198, 396, 594, 792, 992, 1192)      # 792 is the hexagonal end, 1192 the face-centred cubic end
CURVE_STEP = 8
CURVE_SAMPLES = 200_000
GROUPS = ("24", "96", "R")
Q = 16
BISECT_STEPS = 16
SHRINK_STEPS = 12
LLOYD_STEPS = 25
TRAIN_BLOCKS = 600_000


def db(x):
    return 10.0 * math.log10(x) if x > 0 else float("-inf")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Periodic point sets in 3 dimensions
# ---------------------------------------------------------------------------

class PointSet:
    """A periodic set of points: the points X[i] + n @ L for whole n. Scaled to one point per unit of volume, with X[0] moved to the origin."""

    REACH = 2                          # images of the cell kept on each side for the nearest-point search

    def __init__(self, L, X, label):
        L, X = np.asarray(L, dtype=np.float64), np.asarray(X, dtype=np.float64)
        s = (abs(np.linalg.det(L)) / len(X)) ** (-1.0 / 3.0)
        self.L, self.X, self.label = L * s, (X - X[0]) * s, label
        self.Linv = np.linalg.inv(self.L)
        r = range(-self.REACH, self.REACH + 1)
        shifts = np.array([(i, j, k) for i in r for j in r for k in r], dtype=np.float64) @ self.L
        frac = self.X @ self.Linv
        self.base = (frac - np.floor(frac)) @ self.L                      # the points, wrapped into the cell
        cloud = (self.base[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        self.tree = cKDTree(cloud)
        self.cloud = cloud

    def closest(self, P):
        """The nearest point of the set to each row of P."""
        frac = P @ self.Linv
        whole = np.floor(frac)
        inside = (frac - whole) @ self.L
        _, idx = self.tree.query(inside)
        return self.cloud[idx] + whole @ self.L

    def cell_constant(self, samples, rng):
        """G of the set and its standard error: mean squared distance to the nearest point, per dimension, at one point per unit of volume."""
        P = rng.random((samples, 3)) @ self.L
        d2, _ = self.tree.query(P)
        d2 = d2 ** 2 / 3.0
        return float(d2.mean()), float(d2.std(ddof=1) / math.sqrt(samples)), float(np.sqrt(3.0 * d2.max()))

    def points_within(self, radius):
        """Every point of the set of length at most radius."""
        col = np.linalg.norm(self.Linv, axis=0)                           # |n_k| <= (radius + reach) * |column k of Linv|
        reach = radius + float(np.linalg.norm(self.base, axis=1).max())
        top = np.ceil(reach * col).astype(int)
        grid = np.stack(np.meshgrid(*[np.arange(-t, t + 1) for t in top], indexing="ij"), axis=-1).reshape(-1, 3)
        pts = (self.base[None, :, :] + (grid.astype(np.float64) @ self.L)[:, None, :]).reshape(-1, 3)
        return pts[np.linalg.norm(pts, axis=1) <= radius + 1e-9]


def size_of(shape, Y):
    if shape == "ball":
        return np.sqrt((Y ** 2).sum(axis=1))
    if shape == "cross":
        return np.abs(Y).sum(axis=1)
    raise ValueError(shape)


class Cut:
    """The points of a set whose size is at most a bound: the largest bound with no more than POINTS points."""

    def __init__(self, pset, shape, points=POINTS):
        self.pset, self.shape = pset, shape
        guess = 1.6 * (points * 3.0 / (4.0 * math.pi)) ** (1.0 / 3.0) + 2.0   # a ball that holds every candidate
        pts = pset.points_within(guess * (math.sqrt(3.0) if shape == "cross" else 1.0))
        size = np.sort(size_of(shape, pts))
        if len(size) <= points:
            raise RuntimeError("too few candidate points")
        k = points
        while k > 0 and size[k] - size[k - 1] < 1e-9:                    # never split a tie at the boundary
            k -= 1
        self.count, self.bound = k, float(size[k - 1])
        self.codebook = pts[size_of(shape, pts) <= self.bound + 1e-9]
        assert len(self.codebook) == self.count
        self.margin = None

    def set_margin(self, cover):
        """How far rounding can move a point in the size of this cut, from the covering radius of the set."""
        self.margin = cover * (math.sqrt(3.0) if self.shape == "cross" else 1.0)


def block_scales(X, cut):
    """For every block the smallest scale at which it rounds to a point of the codebook (as in the record on balls over long blocks)."""
    g = np.maximum(size_of(cut.shape, X), 1e-30)
    lo = g / (cut.bound + cut.margin) * 0.99
    hi = g / max(cut.bound - cut.margin, 0.05 * cut.bound) * 1.01
    for _ in range(BISECT_STEPS):
        mid = np.sqrt(lo * hi)
        ok = size_of(cut.shape, cut.pset.closest(X / mid[:, None])) <= cut.bound + 1e-9
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return hi


def round_groups(W, cut, group, tb):
    """Round a matrix with one scale per group. Returns (relative error, share of blocks shrunk)."""
    rows, cols = W.shape
    size = cols if group == "R" else int(group)
    X = W.reshape(-1, 3)
    t = np.repeat(tb.reshape(-1, size // 3).max(axis=1), size // 3)
    Y = cut.pset.closest(X / t[:, None])
    out = size_of(cut.shape, Y) > cut.bound + 1e-9
    if np.any(out):
        xo = X[out] / t[out, None]
        lo, hi = np.zeros(len(xo)), np.ones(len(xo))
        for _ in range(SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            good = size_of(cut.shape, cut.pset.closest(xo * mid[:, None])) <= cut.bound + 1e-9
            lo = np.where(good, mid, lo)
            hi = np.where(good, hi, mid)
        Y[out] = cut.pset.closest(xo * lo[:, None])
    Wh = (Y * t[:, None]).reshape(rows, cols).astype(np.float16).astype(np.float64)
    return float(np.sum((W - Wh) ** 2) / np.sum(W ** 2)), float(np.mean(out))


# ---------------------------------------------------------------------------
# A fitted codebook
# ---------------------------------------------------------------------------

def group_scale(W, group, rule):
    """One scale per group: its largest entry ("entry") or the length of its longest block of 3 ("length")."""
    rows, cols = W.shape
    size = cols if group == "R" else int(group)
    if rule == "entry":
        s = np.abs(W).reshape(-1, size).max(axis=1)
    else:
        s = np.sqrt((W.reshape(-1, 3) ** 2).sum(axis=1)).reshape(-1, size // 3).max(axis=1)
    return np.repeat(np.maximum(s, 1e-30), size // 3)


def lloyd(train, start, steps=LLOYD_STEPS):
    C = start.copy()
    for _ in range(steps):
        _, idx = cKDTree(C).query(train)
        n = np.bincount(idx, minlength=len(C))
        sums = np.stack([np.bincount(idx, weights=train[:, k], minlength=len(C)) for k in range(3)], axis=1)
        C = np.where(n[:, None] > 0, sums / np.maximum(n, 1)[:, None], C)
    return C


def fitted_round(W, C, group, rule):
    rows, cols = W.shape
    t = group_scale(W, group, rule)
    X = W.reshape(-1, 3) / t[:, None]
    _, idx = cKDTree(C).query(X)
    Wh = (C[idx] * t[:, None]).reshape(rows, cols).astype(np.float16).astype(np.float64)
    return float(np.sum((W - Wh) ** 2) / np.sum(W ** 2))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

REAL_COUNT = 32


def synthetic(kind, seed):
    rng = np.random.default_rng({"G": 6000, "T": 7000}[kind] + seed)
    W = rng.normal(size=(512, 1920)) if kind == "G" else rng.standard_t(4, size=(512, 1920))
    return W.astype(np.float32)


def load_matrices(weights_path, ball_module):
    """name -> matrix, for the test matrices and the training matrices of each family."""
    test = {f"{k}/{s}": synthetic(k, s) for k in ("G", "T") for s in range(3)}
    train = {f"{k}/train{s}": synthetic(k, 100 + s) for k in ("G", "T") for s in range(2)}
    if weights_path:
        raw = Path(weights_path).read_bytes()
        if not hashlib.sha256(raw).hexdigest().startswith(WEIGHTS_SHA256_PREFIX):
            raise SystemExit("FAIL: the weights file is not the one named in the plan")
        z = np.load(weights_path)
        keys = sorted(k for k in z.files if ball_module.REAL_PATTERN.match(k) and z[k].ndim == 2)
        if len(keys) != REAL_COUNT:
            raise SystemExit(f"FAIL: {len(keys)} matrices match, expected {REAL_COUNT}")
        for i, k in enumerate(keys):
            W = np.ascontiguousarray(z[k], dtype=np.float32)
            if W.shape[1] % 96:
                raise SystemExit(f"FAIL: {k} has rows of {W.shape[1]}, not a multiple of 96")
            test[f"whisper{i % 2}/{k}"] = W                               # two halves: each is rounded with a codebook fitted to the other
    return test, train


def family(name):
    return name.split("/", 1)[0]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", help="folder with path.npz and path_fcc.npz")
    ap.add_argument("--weights", help="the whisper weights file of the earlier records")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", default=str(HERE / "results.json"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261117)

    # -- self-tests on sets whose answers are known ---------------------------------------
    cubic = PointSet(np.eye(3), np.zeros((1, 3)), "Z3")
    fcc = PointSet(2.0 * np.eye(3), np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float), "fcc")
    bcc = PointSet(2.0 * np.eye(3), np.array([[0, 0, 0], [1, 1, 1]], dtype=float), "bcc")
    t = 2.0 - math.sqrt(2.0)
    f3 = PointSet(2.0 * np.eye(3), np.array([[1 + s * t * (i == k) for i in range(3)] for k in range(3) for s in (1, -1)], dtype=float), "F3")
    known = {"Z3": 1.0 / 12.0, "fcc": 0.078745, "bcc": 0.078543, "F3": 0.127803}
    print("0. Checks on sets whose cell is known: G by sampling against the value of bounds/bounds.out")
    print("set | G measured | its standard error | known | difference in standard errors | brute-force nearest point agrees")
    checks_ok = True
    for ps in (cubic, fcc, bcc, f3):
        g, se, _ = ps.cell_constant(400_000 if not args.smoke else 50_000, rng)
        P = rng.normal(size=(300, 3)) * 6.0
        r = range(-4, 5)
        far = (ps.base[None] + (np.array([(i, j, k) for i in r for j in r for k in r], dtype=float) @ ps.L)[:, None]).reshape(-1, 3)
        frac = P @ ps.Linv
        inside = (frac - np.floor(frac)) @ ps.L
        brute = far[np.argmin(((inside[:, None, :] - far[None, :, :]) ** 2).sum(axis=2), axis=1)] + np.floor(frac) @ ps.L
        same = bool(np.allclose(np.linalg.norm(P - brute, axis=1), np.linalg.norm(P - ps.closest(P), axis=1), atol=1e-9))
        z = (g - known[ps.label]) / se
        checks_ok &= same and abs(z) < 4.0
        print(f"{ps.label} | {g:.6f} | {se:.6f} | {known[ps.label]:.6f} | {z:+.1f} | {'yes' if same else 'NO'}")
    if not checks_ok:
        raise SystemExit("FAIL: a self-test did not pass")
    if args.smoke:
        cut = Cut(fcc, "ball")
        cut.set_margin(1.0)
        W = rng.uniform(-1, 1, size=(16, 96))
        tb = block_scales(W.reshape(-1, 3), cut)
        print("smoke:", cut.count, round(cut.bound, 4), [round(v, 5) for v in round_groups(W, cut, "24", tb)])
        C = lloyd(rng.normal(size=(20000, 3)), rng.normal(size=(64, 3)), steps=5)
        print("smoke lloyd:", C.shape, round(fitted_round(W, C * 0.3, "24", "entry"), 5))
        print("smoke ok")
        return

    # -- the paths --------------------------------------------------------------------------
    paths = {}
    for name, sha in PATH_SHA256.items():
        raw = (Path(args.paths) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit(f"FAIL: {name} does not have the SHA-256 named in the plan")
        paths[name] = np.load(Path(args.paths) / name)
    Ls, Xs = paths["path_fcc.npz"]["Ls"], paths["path_fcc.npz"]["Xs"]
    same_start = float(max(np.abs(paths["path.npz"]["Ls"] - Ls[:793]).max(), np.abs(paths["path.npz"]["Xs"] - Xs[:793]).max()))
    print()
    print(f"The face-centred cubic path has {len(Ls)} states. Its first 793 are the hexagonal path: largest difference {same_start:.1e}.")

    def state(i):
        return PointSet(Ls[i], Xs[i], f"fold {i}")

    print()
    print("1. The cell along the fold: G of every eighth state, by sampling, at one point per unit of volume")
    print(f"   {CURVE_SAMPLES:,} samples a state. Gain is 10 log10 of the cube's 1/12 over G. Density is the share of space in the balls of the source.")
    print("state | density | G | its standard error | gain over the cube in dB | farthest sample from a point")
    curve = []
    ball_volume = 4.0 / 3.0 * math.pi * (math.sqrt(2.0) - 1.0) ** 3
    idxs = sorted(set(list(range(0, len(Ls), CURVE_STEP)) + list(STATES) + [len(Ls) - 1]))
    for i in idxs:
        ps = state(i)
        g, se, far = ps.cell_constant(CURVE_SAMPLES, rng)
        dens = 48.0 * ball_volume / abs(np.linalg.det(Ls[i]))
        curve.append({"state": i, "density": dens, "G": g, "se": se, "gain_db": db((1.0 / 12.0) / g), "farthest": far})
        if i % 24 == 0 or i in STATES or i == len(Ls) - 1:
            print(f"{i} | {dens:.4f} | {g:.6f} | {se:.6f} | {db((1.0 / 12.0) / g):+.3f} | {far:.3f}")
    best = min(curve, key=lambda c: c["G"])
    worst = max(curve, key=lambda c: c["G"])
    gains = [c["gain_db"] for c in curve]
    rises = sum(b > a for a, b in zip(gains, gains[1:]))
    print(f"   lowest G: state {best['state']}, {best['G']:.6f} ({best['gain_db']:+.3f} dB). highest G: state {worst['state']}, {worst['G']:.6f} ({worst['gain_db']:+.3f} dB).")
    print(f"   of {len(gains) - 1} steps between sampled states the gain rises on {rises} and falls on {len(gains) - 1 - rises}.")

    # -- the codebooks ------------------------------------------------------------------------
    ball_mod = load_module(ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17" / "run.py", "ball_dimension_run")
    import torch
    import tq
    torch.set_num_threads(1)
    cuts = {}
    for i in STATES:
        cuts[f"F{i}"] = Cut(state(i), "ball")
    cuts["Z3B"] = Cut(cubic, "ball")
    cuts["Z3C"] = Cut(cubic, "cross")
    cuts["FCC"] = Cut(fcc, "ball")
    cuts["BCC"] = Cut(bcc, "ball")
    print()
    print("2. The codebooks over 3 weights: points of a set whose size is at most a bound, no more than 4,096")
    print("arm | set | region | points | bits a weight | bound | G of the set | covering radius used")
    for name, cut in cuts.items():
        g, se, far = cut.pset.cell_constant(CURVE_SAMPLES, rng)
        cut.G = g
        cut.set_margin(1.25 * far)
        print(f"{name} | {cut.pset.label} | {cut.shape} | {cut.count} | {math.log2(cut.count) / 3.0:.4f} | {cut.bound:.4f} | {g:.6f} | {1.25 * far:.3f}")

    test, train = load_matrices(args.weights, ball_mod)
    rows_out = []

    def add(matrix, arm, group, size, bits, rel, **more):
        rows_out.append({"matrix": matrix, "arm": arm, "group": group, "size": size, "bits": bits, "rel_err": rel, **more})

    long_books = {(lat, n): ball_mod.codebook(lat, "ball", n, Q) for lat in ("Z8", "E8") for n in (8, 24)}
    for name, W32 in test.items():
        W = W32.astype(np.float64)
        cols = W.shape[1]
        for group in GROUPS:
            size = cols if group == "R" else int(group)
            x = tq.maxscale_round(W32, Q, "R" if group == "R" else size)[1]
            v = tq.maxscale_voronoi(W32, Q, "R" if group == "R" else size, name="E8")[1]
            add(name, "X", group, size, x["bits"], x["rel_err"])
            add(name, "VE8", group, size, v["bits"], v["rel_err"], shrunk=v["overload"])
        for arm, cut in cuts.items():
            tb = block_scales(W.reshape(-1, 3), cut)
            for group in GROUPS:
                size = cols if group == "R" else int(group)
                rel, shrunk = round_groups(W, cut, group, tb)
                add(name, arm, group, size, math.log2(cut.count) / 3.0 + 16.0 / size, rel, shrunk=shrunk)
        for (lat, n), (bound, points) in long_books.items():
            tb = ball_mod.block_scales(W.reshape(-1, n), lat, "ball", bound)
            for group in GROUPS:
                size = cols if group == "R" else int(group)
                rel, shrunk, _ = ball_mod.round_groups(W, lat, "ball", bound, n, group, tb)
                add(name, f"{'ZB' if lat == 'Z8' else 'EB'}{n}", group, size, math.log2(points) / n + 16.0 / size, rel, shrunk=shrunk)
        print(f"   rounded {name} {W.shape} at {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    # -- fitted codebooks -----------------------------------------------------------------------
    fams = sorted({family(n) for n in test})
    fold_end = cuts["F792"].codebook
    fitted_info = []
    for fam in fams:
        if fam.startswith("whisper"):
            other = "whisper1" if fam == "whisper0" else "whisper0"
            train_mats = [W for n, W in test.items() if family(n) == other]
        else:
            train_mats = [W for n, W in train.items() if family(n) == fam]
        for group in GROUPS:
            for rule in ("entry", "length"):
                blocks = np.concatenate([W.astype(np.float64).reshape(-1, 3) / group_scale(W.astype(np.float64), group, rule)[:, None] for W in train_mats])
                r2 = np.random.default_rng(1000 + len(fitted_info))
                if len(blocks) > TRAIN_BLOCKS:
                    blocks = blocks[r2.choice(len(blocks), TRAIN_BLOCKS, replace=False)]
                starts = {"sample": blocks[r2.choice(len(blocks), POINTS, replace=False)],
                          "fold": fold_end * (np.sqrt((blocks ** 2).sum(axis=1)).max() / np.sqrt((fold_end ** 2).sum(axis=1)).max())}
                for start_name, start in starts.items():
                    C = lloyd(blocks, start)
                    _, idx = cKDTree(C).query(blocks)
                    used = int(len(np.unique(idx)))
                    fitted_info.append({"family": fam, "group": group, "rule": rule, "start": start_name, "train_blocks": int(len(blocks)), "used": used})
                    for name, W32 in test.items():
                        if family(name) != fam:
                            continue
                        W = W32.astype(np.float64)
                        size = W.shape[1] if group == "R" else int(group)
                        add(name, f"L{'e' if rule == 'entry' else 'l'}{'s' if start_name == 'sample' else 'f'}", group, size, 4.0 + 16.0 / size,
                            fitted_round(W, C, group, rule), used=used)
                print(f"   fitted {fam} {group} {rule} at {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    # -- tables ---------------------------------------------------------------------------------
    def fam_label(n):
        return "whisper" if n.startswith("whisper") else family(n)

    labels = ["G", "T", "whisper"]
    arms = ["VE8"] + list(cuts) + ["ZB8", "EB8", "ZB24", "EB24", "Les", "Lef", "Lls", "Llf"]
    by = {(r["matrix"], r["arm"], r["group"]): r for r in rows_out}
    gain = {}
    for arm in arms:
        for group in GROUPS:
            for lab in labels:
                mats = [n for n in test if fam_label(n) == lab]
                if not mats:
                    continue
                g = [db(by[(n, "X", group)]["rel_err"] / by[(n, arm, group)]["rel_err"]) for n in mats]
                gain[(arm, group, lab)] = (float(np.mean(g)), float(np.min(g)), float(np.max(g)), len(g))
    labels = [lab for lab in labels if any(k[2] == lab for k in gain)]
    print()
    print("3. Gain over the scalar rule X in dB of weight error, mean over the matrices of a family, at 16 levels")
    print("   Groups of 24 and of 96 weights and one scale per row. An arm's bits are in table 4.")
    print("arm | " + " | ".join(f"{lab} {g}" for g in GROUPS for lab in labels))
    for arm in arms:
        print(f"{arm} | " + " | ".join(f"{gain[(arm, g, lab)][0]:+.2f}" for g in GROUPS for lab in labels))
    print()
    print("4. Bits a weight and the share of blocks shrunk, means over all matrices")
    print("arm | " + " | ".join(f"bits {g}" for g in GROUPS) + " | " + " | ".join(f"shrunk {g}" for g in GROUPS))
    for arm in ["X"] + arms:
        bits = [np.mean([by[(n, arm, g)]["bits"] for n in test]) for g in GROUPS]
        shr = [np.mean([by[(n, arm, g)].get("shrunk", 0.0) for n in test]) for g in GROUPS]
        print(f"{arm} | " + " | ".join(f"{b:.4f}" for b in bits) + " | " + " | ".join(f"{s:.5f}" for s in shr))
    print()
    print("5. The fold's states against its close-packed end, and against what their cells say")
    print("   measured: gain of the state over F1192 in dB, mean over all matrices and groups. cell: 10 log10 of G of F1192 over G of the state.")
    print("state | measured | least | most | cell | measured less cell")
    for i in STATES:
        d = [db(by[(n, "F1192", g)]["rel_err"] / by[(n, f"F{i}", g)]["rel_err"]) for n in test for g in GROUPS]
        cell = db(cuts["F1192"].G / cuts[f"F{i}"].G)
        print(f"F{i} | {np.mean(d):+.3f} | {np.min(d):+.3f} | {np.max(d):+.3f} | {cell:+.3f} | {np.mean(d) - cell:+.3f}")
    print()
    print("6. The fitted codebooks: Lloyd's algorithm, 25 steps, 4,096 points")
    print("   Le: blocks divided by the largest entry of their group. Ll: by the length of the longest block of their group. s: started from a sample. f: from the fold's hexagonal end.")
    print("family | group | rule | start | training blocks | codewords used")
    for f in fitted_info:
        print(f"{f['family']} | {f['group']} | {f['rule']} | {f['start']} | {f['train_blocks']} | {f['used']}")
    best_fold = {}
    print()
    print("7. The best state of the fold against the others, per family and group: gain in dB, mean over matrices (least to most)")
    print("family | group | best state | over X | fitted Les over it | fitted Lls over it | Lef less Les | Llf less Lls | Z3B over it | ZB8 over it | VE8 over it")
    for lab in labels:
        mats = [n for n in test if fam_label(n) == lab]
        for group in GROUPS:
            bi = max(STATES, key=lambda i: gain[(f"F{i}", group, lab)][0])
            best_fold[(lab, group)] = bi

            def over(arm, ref):
                d = [db(by[(n, ref, group)]["rel_err"] / by[(n, arm, group)]["rel_err"]) for n in mats]
                return f"{np.mean(d):+.2f} ({np.min(d):+.2f} to {np.max(d):+.2f})"
            fb = f"F{bi}"
            print(f"{lab} | {group} | {fb} | {over(fb, 'X')} | {over('Les', fb)} | {over('Lls', fb)} | {over('Lef', 'Les')} | {over('Llf', 'Lls')} | {over('Z3B', fb)} | {over('ZB8', fb)} | {over('VE8', fb)}")

    json.dump({"curve": curve, "rows": rows_out, "fitted": fitted_info,
               "cuts": {k: {"count": c.count, "bound": c.bound, "G": c.G, "shape": c.shape, "set": c.pset.label} for k, c in cuts.items()},
               "seconds": time.time() - t0}, open(args.json, "w"), indent=0)
    print()
    print(f"seconds: {time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
