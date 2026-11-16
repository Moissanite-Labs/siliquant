#!/usr/bin/env python3
"""Checks of harness/tq.py against the reference code in lattice/lattices.py.

Needs NumPy and torch. Usage, from the root of a checkout:
  python3 harness/test_tq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lattice"))
import lattices as L  # noqa: E402
import tq  # noqa: E402


def t(x):
    return torch.tensor(x, dtype=torch.float32, device=tq.DEV)


def main():
    rng = np.random.default_rng(11)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(("ok   " if ok else "FAIL ") + name + (" " + detail if detail else ""))

    # 1. The decoders return the same lattice point as the reference, away from ties.
    x = rng.normal(scale=2.0, size=(200_000, 8)).astype(np.float32)
    for name, ref, fn in (("D8", L.Dn(8), tq.closest_D), ("E8", L.E8(), tq.closest_E8)):
        a = ref.closest(x.astype(np.float64))
        b = fn(t(x)).cpu().numpy().astype(np.float64)
        same = np.all(a == b, axis=1)
        # Where they differ, both must be equally near to x within float32 rounding.
        da = ((x - a) ** 2).sum(axis=1)
        db = ((x - b) ** 2).sum(axis=1)
        check(f"{name} decoder matches the reference", bool(np.all(same | (np.abs(da - db) < 1e-4))),
              f"({int((~same).sum())} of {len(x)} differ, all ties)")

    # 2. Every reconstruction is a codeword: decoding its index gives it back.
    for name in ("Z8", "E8"):
        ref = {"Z8": L.Zn(8), "E8": L.E8()}[name]
        for q in (4, 8, 16):
            code = tq.Code(name, q)
            xs = rng.normal(scale=q / 3.0, size=(100_000, 8)).astype(np.float32)
            r, over = code.quantize(t(xs))
            c = (r + code.centre).cpu().numpy().astype(np.float64)
            u = L.voronoi_index(c, ref, q)
            back = L.voronoi_decode(u, ref, q)
            check(f"{name} q={q}: reconstructions are codewords of the reference code",
                  bool(np.max(np.abs(back - c)) < 1e-3), f"(share shrunk {over:.3f})")

    # 3. Same error as the reference quantiser on the same blocks at the same scale.
    for name in ("Z8", "E8"):
        ref = {"Z8": L.Zn(8), "E8": L.E8()}[name]
        for q in (4, 16):
            code = tq.Code(name, q)
            xs = rng.standard_t(5, size=(60_000, 8)).astype(np.float32) * q / 5.0
            centre = L.codebook_centroid(ref, q)
            rn = L.voronoi_quantize(xs.astype(np.float64), ref, q, centre=centre, steps=tq.SHRINK_STEPS)[0]
            rt = code.quantize(t(xs))[0].cpu().numpy().astype(np.float64)
            en, et = np.mean((xs - rn) ** 2), np.mean((xs - rt) ** 2)
            check(f"{name} q={q}: error equals the reference", abs(et / en - 1.0) < 2e-3, f"(ratio {et / en:.6f})")

    # 4. Scalar rounding and the balancing equal the reference.
    xs = rng.normal(size=(4096, 8)).astype(np.float32)
    for q in (3, 4, 16):
        half = q // 2
        if q % 2 == 0:
            ref = (np.clip(np.floor(xs / 0.37), -half, half - 1) + 0.5) * 0.37
        else:
            ref = np.clip(np.rint(xs / 0.37), -half, half) * 0.37
        check(f"scalar q={q} equals the reference", bool(np.max(np.abs(tq.scalar_round(t(xs), q, 0.37).cpu().numpy() - ref)) < 1e-5))
    W = (np.exp(rng.normal(size=(96, 1))) * rng.normal(size=(96, 160)) * np.exp(rng.normal(size=(1, 160)))).astype(np.float32)
    s, tt, Wn = tq.dual_scale(t(W))
    sr, tr, Wr, _, _ = L.dual_scale(W.astype(np.float64))
    check("balancing equals the reference", bool(np.max(np.abs(Wn.cpu().numpy() - Wr)) < 5e-3)
          and bool(np.max(np.abs((s[:, None] * Wn * tt[None, :]).cpu().numpy() - W) / np.abs(W).max()) < 1e-4))

    # 5. round_matrix: the reported error is the error of what it returns, and E8 beats scalar on a Gaussian matrix.
    W = rng.normal(size=(512, 1024)).astype(np.float32)
    out = {}
    for arm in ("S", "E8", "D+S", "D+E8"):
        Wh, info = tq.round_matrix(W, arm, 4)
        rel = float(np.sum((W.astype(np.float64) - Wh.astype(np.float64)) ** 2) / np.sum(W.astype(np.float64) ** 2))
        out[arm] = info["rel_err"]
        check(f"round_matrix {arm}: reported error is the error of the returned matrix", abs(rel - info["rel_err"]) < 1e-12,
              f"({10 * np.log10(rel):.2f} dB)")
    check("E8 beats scalar at 2 bits on a Gaussian matrix by 0.4 to 1.0 dB", 0.4 < 10 * np.log10(out["S"] / out["E8"]) < 1.0,
          f"({10 * np.log10(out['S'] / out['E8']):.2f} dB)")
    print("FAILED" if fails else "all checks pass")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
