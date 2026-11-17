#!/usr/bin/env python3
"""Does the formula of issue #10 give the error of a cut codebook from the weights alone, and choose the right one?

The plan is PREREG.md in this folder. This file is the only program that rounds anything.
summary.py reads results.json and prints the tables.

Issue #10 proposes a process: probe a matrix, map it, turn the map into knobs, and pick the
knobs with a formula for the error. shape/shapelib.py holds the formula. This program
computes it for every setting of the knobs on a matrix, with nothing rounded, then rounds the
matrix with every setting and sets the two side by side.

The knobs:
  cell     the lattice: the integers Z^8 or E8
  shape    the region of the codebook: the ball of the p-norm, p = 2/3, 1, 5/4, 3/2, 2, 3,
           and the cube of the scalar rule
  size     how many weights share a scale: 32, 128 or a row
  kept     how many directions of the matrix are kept: all of them, or the k largest

Two sets of matrices. "seen" is the 38 matrices the two earlier records of this date used:
its output is the pilot, read before the plan was written. "held" is the set the plan names:
no codebook had been run on any of it.

Usage, from the root of a checkout, with NumPy:
  python3 experiments/shape-formula-held-out-2026-11-17/run.py --smoke
  python3 experiments/shape-formula-held-out-2026-11-17/run.py --set seen --whisper weights.npz
  python3 experiments/shape-formula-held-out-2026-11-17/run.py --set held --whisper weights.npz --qwen model.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(HERE))
import shapelib as S  # noqa: E402

WHISPER_SHA256 = "b6c8ee500656e04e8e57c2949a5253f0dda002ba36cd1561846574dbcf01132e"
QWEN_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
SEEN_PATTERN = re.compile(r"^(encoder|decoder)\.blocks\.(0|5)\..*weight$")
HELD_PATTERN = re.compile(r"^(encoder|decoder)\.blocks\.(1|2|3|4)\..*weight$")
QWEN_PATTERN = re.compile(r"^model\.layers\.(0|11|23)\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")
COUNTS = {"seen": 32, "held": 64, "qwen": 21}

SHAPES = (Fraction(2, 3), Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(2), Fraction(3))
LATTICES = ("Z8", "E8")
GROUPS = ("32", "128", "R")
BUDGET_BITS = Fraction(9, 2)                       # the budget of the three cells set side by side: 4.5 bits per weight with the scales
RANK_SHARES = (8, 4, 2)                            # the rank arms keep 1/8, 1/4 and 1/2 of the directions
SEEN_SEEDS = {"G": 6000, "T": 7000}                # the seeds of the two earlier records
HELD_SEEDS = {"G": 8100, "L": 8200, "T3": 8300, "U": 8400, "RS": 8500, "BS": 8600, "C": 8700, "K": 8800}
SHUFFLE_SEED = 9100
G_BALL8 = 0.070454                                 # bounds/bounds.out, section 2: no cells in 8 dimensions have a smaller G

_files = {}


def db(x):
    return 10.0 * math.log10(x)


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def synthetic(kind, seed, base, small=False):
    """The synthetic families. Every one is 512 by 2048 in 32-bit floats. base + seed is the seed of the generator."""
    rng = np.random.default_rng(base + seed)
    m, n = (512, 2048) if not small else (32, 256)
    if kind == "G":
        W = rng.normal(size=(m, n))
    elif kind == "T":
        W = rng.standard_t(4, size=(m, n))
    elif kind == "L":
        W = rng.laplace(size=(m, n))
    elif kind == "T3":
        W = rng.standard_t(3, size=(m, n))
    elif kind == "U":
        W = rng.uniform(-1.0, 1.0, size=(m, n))
    elif kind == "RS":                                       # every row has its own scale
        W = rng.normal(size=(m, n)) * np.exp(0.5 * rng.normal(size=(m, 1)))
    elif kind == "BS":                                       # four columns in ten are near zero
        W = rng.normal(size=(m, n))
        W[:, rng.permutation(n)[: (2 * n) // 5]] *= 0.05
    elif kind == "C":                                        # neighbours in a row move together
        Z = rng.normal(size=(m, n))
        W = np.empty((m, n))
        W[:, 0] = Z[:, 0]
        for j in range(1, n):
            W[:, j] = 0.8 * W[:, j - 1] + 0.6 * Z[:, j]
    elif kind == "K":                                        # 64 directions hold all but a thousandth of the mass
        W = rng.normal(size=(m, 64)) @ rng.normal(size=(64, n)) / 8.0 + 0.03 * rng.normal(size=(m, n))
    else:
        raise ValueError(kind)
    return W.astype(np.float32)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


def read_safetensors(path, name):
    """One matrix of a safetensors file as 32-bit floats. The 16-bit type BF16 is the top half of a 32-bit float."""
    header, base = safetensors_header(path)
    info = header[name]
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    if info["dtype"] == "BF16":
        W = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif info["dtype"] == "F32":
        W = np.frombuffer(raw, dtype="<f4")
    elif info["dtype"] == "F16":
        W = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise ValueError(f"type {info['dtype']} is not read here")
    return np.ascontiguousarray(W.reshape(info["shape"]))


def load(task):
    """The matrix a task names, as it is stored (16-bit or 32-bit floats)."""
    kind, name = task["matrix"].split("/", 1)
    if kind == "whisper":
        return np.ascontiguousarray(np.load(_files["whisper"])[name])
    if kind == "qwen":
        return read_safetensors(_files["qwen"], name)
    if kind == "smoke":
        return synthetic(name, 0, 5, small=True)
    return synthetic(kind, int(name), task["base"])


def tasks_for(which, whisper, qwen):
    tasks = []
    if which == "seen":
        tasks += [{"matrix": f"{k}/{s}", "family": k, "base": b} for k, b in SEEN_SEEDS.items() for s in range(3)]
    else:
        tasks += [{"matrix": f"{k}/{s}", "family": k, "base": b} for k, b in HELD_SEEDS.items() for s in range(3)]
    if whisper:
        if sha256_of(whisper) != WHISPER_SHA256:
            raise SystemExit("FAIL: the whisper file does not have the SHA-256 named in the plan")
        z = np.load(whisper)
        pattern, count = (SEEN_PATTERN, COUNTS["seen"]) if which == "seen" else (HELD_PATTERN, COUNTS["held"])
        keys = sorted(k for k in z.files if pattern.match(k) and z[k].ndim == 2)
        if len(keys) != count:
            raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {count}")
        tasks += [{"matrix": "whisper/" + k, "family": "whisper"} for k in keys]
    if qwen and which == "held":
        if sha256_of(qwen) != QWEN_SHA256:
            raise SystemExit("FAIL: the qwen file does not have the SHA-256 named in the plan")
        header, _ = safetensors_header(qwen)
        keys = sorted(k for k in header if QWEN_PATTERN.match(k))
        if len(keys) != COUNTS["qwen"]:
            raise SystemExit(f"FAIL: {len(keys)} matrices match the plan's pattern, expected {COUNTS['qwen']}")
        tasks += [{"matrix": "qwen/" + k, "family": "qwen"} for k in keys]
    return tasks


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

def row_cap(n):
    """floor(2^(8 (4.5 - 16/n))): the most codewords a block may have when one scale of 16 bits serves a row of n weights."""
    e = 8 * (BUDGET_BITS - Fraction(16, n))
    return S.iroot(1 << e.numerator, e.denominator)


def cells_for(n):
    """(group, cap, rate, in the budget). Rate "2" is 2 bits per weight before the scales, "4" is 4, and "4+" is what
    the budget leaves. The three cells in the budget add up to 4.5 bits per weight with their scales."""
    out = [(g, 1 << 16, "2", False) for g in GROUPS]
    out += [("32", 1 << 32, "4", True), ("128", 1 << 32, "4", False), ("R", 1 << 32, "4", False)]
    out += [("128", 1 << 35, "4+", True), ("R", row_cap(n), "4+", True)]
    return out


def levels_for(cap):
    """The levels of the scalar rule with no more bits than a codebook of cap points: floor(cap^(1/8))."""
    return S.iroot(cap, 8)


def all_caps(row_lengths):
    return sorted({cap for n in row_lengths for _, cap, _, _ in cells_for(n)})


# ---------------------------------------------------------------------------
# One matrix
# ---------------------------------------------------------------------------

def reverse_water_filling(lam, bits):
    """For directions of variances lam and a rate in bits per coefficient: (error, share of directions given no bits).

    Shannon's rule for a Gaussian source: every direction is rounded to the same error theta,
    and a direction whose variance is under theta is dropped. theta is found by bisection.
    """
    lam = np.asarray(lam, dtype=np.float64)
    pos = lam[lam > 0]
    lo, hi = float(pos.min()) * 1e-12, float(lam.max())
    for _ in range(200):
        theta = math.sqrt(lo * hi)
        rate = float(np.sum(0.5 * np.log2(np.maximum(lam, theta) / theta))) / len(lam)
        if rate > bits:
            lo = theta
        else:
            hi = theta
    return float(np.mean(np.minimum(lam, hi))), float(np.mean(lam < hi))


def stretch_numbers(W, sv):
    """What the spread of the mass over directions is worth at 4 bits, by Shannon's rule, in dB."""
    m, n = W.shape
    out = {}
    for name, lam in (("directions", sv ** 2), ("columns", (W ** 2).sum(axis=0)), ("rows", (W ** 2).sum(axis=1))):
        lam = np.asarray(lam, dtype=np.float64)
        err, dropped = reverse_water_filling(lam, 4.0)
        out[name] = {"gain_db_at_4_bits": db(float(np.mean(lam)) * 2.0 ** -8 / err), "share_dropped_at_4_bits": dropped,
                     "top_eighth_share_of_mass": float(np.sort(lam)[::-1][: max(1, len(lam) // 8)].sum() / lam.sum())}
    return out


def rank_arm(W, U, sv, Vt, share, budget_bits):
    """Keep the k largest directions: W ~ A B with A = U_k sqrt(S_k) and B = sqrt(S_k) V_k', both rounded by the scalar rule per row.

    The levels q are the most that fit the budget, with one scale of 16 bits per row of A and of B.
    The formula: what is dropped, plus what the rounding of the two factors adds to what is kept.
    """
    m, n = W.shape
    r = len(sv)
    k = r // share
    per_entry = (float(budget_bits) * m * n - 16.0 * (m + k)) / (k * (m + n))
    q = min(int(2.0 ** per_entry), 1 << 14)
    if q < 2:
        return None
    root = np.sqrt(sv[:k])
    A = (U[:, :k] * root[None, :]).astype(np.float32)
    B = (root[:, None] * Vt[:k, :]).astype(np.float32)

    def rounded(M):
        size = M.shape[1]
        j = np.abs(M).argmax(axis=1)[:, None]
        d = np.take_along_axis(M, j, axis=1) / np.float32(-q / 2.0)
        safe = np.where(d == 0, np.float32(1.0), d)
        return (np.clip(np.rint(M / safe), -q // 2, q // 2 - 1).astype(np.float32) * d).astype(np.float64), size

    Ah, _ = rounded(A)
    Bh, _ = rounded(B)
    Wh = (Ah @ Bh).astype(np.float16).astype(np.float64)
    total = float(np.sum(W ** 2))
    meas = float(np.sum((W - Wh) ** 2)) / total
    tail = float(np.sum(sv[k:] ** 2) / np.sum(sv ** 2))
    eA = S.predict_scalar(A, q, "R", 3)
    eB = S.predict_scalar(B, q, "R", 3)
    kappa = float(np.sum(sv[:k]) ** 2 / (k * np.sum(sv[:k] ** 2)))
    pred = tail + (1.0 - tail) * (kappa * eA + eB)
    bits = (m * k * (math.log2(q) + 16.0 / k) + k * n * (math.log2(q) + 16.0 / n)) / (m * n)
    return {"kind": "rank", "k": k, "share": f"1/{share}", "q": q, "bits": bits, "meas": meas, "L3": pred,
            "tail": tail, "eA": eA, "eB": eB, "kappa": kappa}


def one_matrix(task):
    """Every number of the plan for one matrix. Returns rows."""
    t0 = time.time()
    stored = load(task)
    W32 = np.ascontiguousarray(stored, dtype=np.float32)
    W = W32.astype(np.float64)
    m, n = W.shape
    X = W.reshape(-1, 8)
    base = {"matrix": task["matrix"], "family": task["family"]}
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(m * n)
    Wsh = W.reshape(-1)[perm].reshape(m, n)                   # the same entries with their places drawn at random
    rows = []
    cells = cells_for(n)
    for cap in sorted({c for _, c, _, _ in cells}):
        mine = [(g, rate, budget) for g, c, rate, budget in cells if c == cap]
        for lattice in LATTICES:
            for p in SHAPES:
                code = S.Codebook(lattice, S.Shape(p), cap)
                tb = S.block_scales(X, code)
                for g, rate, budget in mine:
                    size = n if g == "R" else int(g)
                    gg = g if g == "R" else int(g)
                    spectral = budget and lattice == "E8" and p == 2
                    if spectral:
                        rel, tg, shrunk, Wh = S.round_groups(W, code, gg, tb, keep=True)
                    else:
                        rel, tg, shrunk = S.round_groups(W, code, gg, tb)
                    rows.append({**base, "kind": "cut", "arm": f"{lattice}:{code.shape.name}", "lattice": lattice, "p": str(p),
                                 "group": g, "size": size, "rate": rate, "budget": budget, "cap": cap, "codewords": code.N,
                                 "bits": code.bits + 16.0 / size, "meas": rel, "shrunk": shrunk,
                                 "L1": S.predict(W, code, gg, 1), "L2": S.predict(W, code, gg, 2), "L3": S.predict(W, code, gg, 3),
                                 "L3_shuffled": S.predict(Wsh, code, gg, 3)})
                    if spectral:
                        rows[-1]["spectral_ratio"] = S.spectral_ratio(W - Wh)
        q = levels_for(cap)
        for g, rate, budget in mine:
            size = n if g == "R" else int(g)
            gg = g if g == "R" else int(g)
            rel, bits, Wh = S.scalar_rule(W32, q, gg, keep=True)
            rows.append({**base, "kind": "scalar", "arm": "X", "group": g, "size": size, "rate": rate, "budget": budget, "cap": cap, "q": q,
                         "bits": bits, "meas": rel,
                         "L1": S.predict_scalar(W, q, gg, 1), "L2": S.predict_scalar(W, q, gg, 2), "L3": S.predict_scalar(W, q, gg, 3),
                         "L3_shuffled": S.predict_scalar(Wsh, q, gg, 3)})
            if budget:
                rows[-1]["spectral_ratio"] = S.spectral_ratio(W - Wh)
    # the probe: the region part for every shape and group, on the matrix and on its shuffled copy
    probe = {**base, "kind": "probe", "rows": m, "cols": n, "stored_as": str(stored.dtype), "digest": S.matrix_digest(stored),
             "phi": {}, "phi_shuffled": {}}
    for g in GROUPS:
        gg = g if g == "R" else int(g)
        probe["phi"][g] = {"cube": S.phi_cube(W, gg), **{str(p): S.phi(W, S.Shape(p), gg) for p in SHAPES}}
        probe["phi_shuffled"][g] = {"cube": S.phi_cube(Wsh, gg), **{str(p): S.phi(Wsh, S.Shape(p), gg) for p in SHAPES}}
    # the receipt: the same three numbers from whole-number arithmetic
    receipt, worst = {}, 0.0
    for g in GROUPS:
        gg = g if g == "R" else int(g)
        terms = S.exact_terms(stored, gg)
        exact = S.phi_from_exact(terms)
        for name, key in (("cube", "cube"), ("ball", "2"), ("cross", "1")):
            frac, const = exact[name]
            worst = max(worst, abs(const * float(frac) / probe["phi"][g][key] - 1.0))
        receipt[g] = {k: str(v) for k, v in terms.items()}
    probe["receipt"] = receipt
    probe["receipt_worst_relative_difference"] = worst
    # how stretched: the directions of the matrix
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    probe["stretch"] = stretch_numbers(W, sv)
    probe["peak"] = float(np.mean(np.abs(X).max(axis=1) / np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)))
    w2 = np.sort((W ** 2).reshape(-1))[::-1]
    probe["top_1_percent_share_of_mass"] = float(w2[: max(1, len(w2) // 100)].sum() / w2.sum())
    probe["share_under_a_tenth_of_rms"] = float(np.mean(np.abs(W) < 0.1 * math.sqrt(float(np.mean(W ** 2)))))
    rows.append(probe)
    for share in RANK_SHARES:
        r = rank_arm(W, U, sv, Vt, share, BUDGET_BITS)
        if r:
            rows.append({**base, **r, "group": "-", "rate": "4+", "budget": True})
    for r in rows:
        r["seconds"] = round(time.time() - t0, 1)
    return rows


# ---------------------------------------------------------------------------
# The gate on the plan
# ---------------------------------------------------------------------------

GATED = ("experiments/shape-formula-held-out-2026-11-17/PREREG.md", "experiments/shape-formula-held-out-2026-11-17/run.py",
         "experiments/shape-formula-held-out-2026-11-17/summary.py", "experiments/shape-formula-held-out-2026-11-17/predictions.py",
         "shape/shapelib.py")


def plan_gate():
    """Refuse the run of the plan unless the plan and the programs are committed as they stand.

    Returns what results.start records: the commit, whether it is on the remote, and the SHA-256
    of the plan and of each program. The idea is crossing X2 of the note in PR #5.
    """
    def git(*a):
        return subprocess.run(("git", "-C", str(ROOT)) + a, capture_output=True, text=True, check=True).stdout.strip()
    try:
        dirty = git("status", "--porcelain", "--", *GATED)
        head = git("rev-parse", "HEAD")
        tracked = git("ls-files", "--", *GATED).splitlines()
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"FAIL: the plan gate needs a git checkout: {e}")
    missing = [f for f in GATED if f not in tracked]
    if missing:
        raise SystemExit("FAIL: not committed: " + ", ".join(missing))
    if dirty:
        raise SystemExit("FAIL: changed since the last commit:\n" + dirty)
    try:
        pushed = bool(git("branch", "-r", "--contains", head))
    except subprocess.CalledProcessError:
        pushed = False
    return {"commit": head, "commit_is_on_the_remote": pushed, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": {f: sha256_of(ROOT / f) for f in GATED}}


# ---------------------------------------------------------------------------
# Checks of the code
# ---------------------------------------------------------------------------

def selftest():
    out = []
    expected = {("Z8", 1, 16): (29, 3789356689), ("Z8", 2, 16): (179, 4206764513), ("E8", 2, 16): (178, 4158269521),
                ("E8", 1, 16): (29, 3418475137), ("Z8", 2, 32): (720, 1094300611393), ("Z8", 1, 32): (59, 1007248645697),
                ("E8", 2, 32): (720, 1097438938801), ("E8", 1, 32): (59, 965614200833)}
    ok = True
    for (lat, p, q), (radius, count) in expected.items():
        c = S.Codebook(lat, S.Shape(p), q ** 8)
        ok &= c.R == radius << S.S_BITS and c.N == count
    out.append((ok, "the eight codebooks of the record region-shapes-no-clip-rule-2026-11-17 come out with its radii and counts"))
    ok = True
    for lat, shape, cap in ((l, S.Shape(p), c) for l in LATTICES for p in SHAPES for c in (1 << 16, 1 << 32, 1 << 35)):
        c = S.Codebook(lat, shape, cap)
        ok &= c.N <= cap
    out.append((ok, "no codebook has more points than its cap"))
    prior = ROOT / "experiments" / "region-shapes-no-clip-rule-2026-11-17" / "results.json"
    earlier = [r for r in json.loads(prior.read_text()) if r["matrix"] == "G/0"]
    W32 = synthetic("G", 0, SEEN_SEEDS["G"])
    W = W32.astype(np.float64)
    same = True
    for r in earlier:
        lat, p = {"ZB": ("Z8", 2), "ZC": ("Z8", 1), "EB": ("E8", 2), "EC": ("E8", 1)}[r["arm"]]
        code = S.Codebook(lat, S.Shape(p), r["q"] ** 8)
        rel, _, _ = S.round_groups(W, code, "R" if r["group"] == "R" else int(r["group"]), S.block_scales(W.reshape(-1, 8), code))
        same &= rel == r["rel_err"]
    out.append((same, f"the {len(earlier)} errors of that record on its first Gaussian matrix are reproduced to the last digit"))
    prior = ROOT / "experiments" / "e8-gain-largest-entry-rule-2026-11-17" / "results.json"
    earlier = [r for r in json.loads(prior.read_text()) if r["matrix"] == "G/0" and "arm" not in r]
    same = all(S.scalar_rule(W32, r["q"], "R" if r["group"] == "R" else int(r["group"]))[0] == r["X"] for r in earlier)
    out.append((same, f"the {len(earlier)} errors of the scalar rule in the record e8-gain-largest-entry-rule-2026-11-17 on that matrix are reproduced to the last digit"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the code path on two small matrices that are not in the study and print nothing but ok")
    ap.add_argument("--set", choices=("seen", "held"), help="seen: the pilot. held: the run of the plan")
    ap.add_argument("--whisper", help="path to weights.npz of mlx-community/whisper-base.en-mlx")
    ap.add_argument("--qwen", help="path to model.safetensors of Qwen/Qwen2.5-0.5B")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="with --smoke: print the tables")
    args = ap.parse_args()
    t0 = time.time()
    if args.smoke:
        # every family of the plan in small, and two more under the names of the real models, so that
        # every table and every prediction is computed once before the plan is committed
        tasks = [{"matrix": f"smoke/{k}", "family": k} for k in HELD_SEEDS]
        tasks += [{"matrix": "smoke/T", "family": "whisper"}, {"matrix": "smoke/L", "family": "qwen"}]
        rows = []
        for t in tasks:
            rs = one_matrix(t)
            for r in rs:
                r["matrix"] = f"{t['family']}/{t['matrix'].split('/')[1]}"
            rows += rs
        import summary
        meta = {"set": "smoke", "gate": None, "checks": [], "codebooks": [], "matrices": len(tasks), "numpy": np.__version__,
                "python": sys.version.split()[0], "seconds": round(time.time() - t0)}
        lines, _ = summary.render({"meta": meta, "rows": rows}, with_predictions=True)
        verdicts = [ln for ln in lines if " | holds | " in ln or " | FAILS | " in ln]
        ok = all(0 < r["meas"] < 1.5 and r["L3"] > 0 for r in rows if r["kind"] != "probe")
        ok &= all(r["receipt_worst_relative_difference"] < 1e-9 for r in rows if r["kind"] == "probe")
        ok &= len(verdicts) == 16
        if args.verbose:
            print("\n".join(lines))
        print("ok" if ok else "FAIL", f"({len(rows)} rows, {len(lines)} lines, {len(verdicts)} verdicts computed, {time.time() - t0:.0f} s)")
        sys.exit(0 if ok else 1)
    if not args.set:
        raise SystemExit("give --set seen or --set held, or --smoke")
    gate = None
    if args.set == "held":
        if not (args.whisper and args.qwen):
            raise SystemExit("FAIL: the run of the plan needs both weight files")
        gate = plan_gate()
        if not gate["commit_is_on_the_remote"]:
            raise SystemExit("FAIL: the commit that holds the plan is not on the remote yet. Push it first.")
        (HERE / "results.start").write_text(json.dumps(gate, indent=1) + "\n")
    checks = selftest()
    if not all(c[0] for c in checks):
        raise SystemExit("FAIL: a check of the code failed: " + "; ".join(t for ok, t in checks if not ok))
    _files["whisper"], _files["qwen"] = args.whisper, args.qwen
    tasks = tasks_for(args.set, args.whisper, args.qwen)
    # every codebook is built once, before the workers start, so that they share it
    lengths = {2048} | ({512} if args.whisper else set()) | ({896, 4864} if (args.qwen and args.set == "held") else set())
    books = []
    for cap in all_caps(lengths):
        for lattice in LATTICES:
            for p in SHAPES:
                books.append(S.Codebook(lattice, S.Shape(p), cap).describe())
    with get_context("fork").Pool(args.workers) as pool:
        results = pool.map(one_matrix, tasks, chunksize=1)
    rows = [r for rs in results for r in rs]
    name = "pilot" if args.set == "seen" else "results"
    meta = {"set": args.set, "gate": gate, "checks": [[bool(ok), text] for ok, text in checks], "codebooks": books, "matrices": len(tasks),
            "whisper_used": bool(args.whisper), "qwen_used": bool(args.qwen and args.set == "held"),
            "numpy": np.__version__, "python": sys.version.split()[0], "seconds": round(time.time() - t0)}
    (HERE / f"{name}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"{name}.json written: {len(rows)} rows, {len(tasks)} matrices, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
