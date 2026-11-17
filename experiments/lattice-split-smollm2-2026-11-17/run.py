#!/usr/bin/env python3
"""Why E8 costs more KL per unit of weight error than integer levels on SmolLM2: arms that take the two errors apart.

The plan is PREREG.md in this folder. The model is run by shape/npforward.py, in NumPy, and not
by llama.cpp: this session has no build of it. The first step checks that instrument against
the harness's own reference perplexity on the same text.

Three steps, each from the root of a checkout, with NumPy:

  python3 run.py reference --model M.gguf --text wiki.test.raw --work DIR
      the reference pass on 40 passages; prints its perplexity; keeps the outputs of the first 16
  python3 run.py build --model M.gguf --work DIR
      rounds the three base arms and keeps them as 16-bit floats
  python3 run.py score --model M.gguf --text wiki.test.raw --work DIR --out results.jsonl ARM [ARM ...]
      scores arms on the first 16 passages, one row each

Arms. Base, rounded as the harness rounds them: X32:16, ZBG128:b560s16, EBG128:b560s16.
Made from a base arm B, whose error is (rounded - weights):
  noise(B)#s      Gaussian noise in place of B's error, with B's squared error in every group of 128 weights
  shuffle(B)#s    B's error with the weights of each group of 128 trading errors at random
  scale(B,d)      B's error times a factor: d dB more squared error in every weight
  mix(attn=L,ffn=M)   the four attention matrices of a block from the ball arm on lattice L (Z or E), the three feed-forward ones from M
Extension 1 of the plan (PREREG-extension-1.md):
  noisepart(B)@small#s   Gaussian noise in place of B's error on the weights B rounds to zero, with their squared error in every
                         group; the other weights keep B's error
  noisepart(B)@big#s     the reverse: noise on the weights B does not round to zero, B's error on those it does
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "shape"))
sys.path.insert(0, str(ROOT / "lattice"))
import design  # noqa: E402
import lattice_spread as LS  # noqa: E402
import npforward as NF  # noqa: E402

MODEL_SHA = "cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"
TEXT_SHA = "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"
LENGTH, ALL, KEPT = 512, 40, 16
GROUP = 128
BASE = ("X32:16", "ZBG128:b560s16", "EBG128:b560s16")
BALL = {"Z": "ZBG128:b560s16", "E": "EBG128:b560s16"}
ATTN = ("attn_q", "attn_k", "attn_v", "attn_output")
_model_path = None
_work = None


def db(x):
    return 10.0 * math.log10(x)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def folder(work, arm):
    return Path(work) / "arms" / re.sub(r"[^A-Za-z0-9]", "_", arm)


def scalar_round(W, q, group):
    """The scalar largest-entry rule, as shape/shapelib.py and the harness round it."""
    rows, cols = W.shape
    X = W.reshape(-1, group)
    j = np.abs(X).argmax(axis=1)[:, None]
    d = np.take_along_axis(X, j, axis=1) / np.float32(-q / 2.0)
    safe = np.where(d == 0, np.float32(1.0), d)
    R = np.clip(np.rint(X / safe), -q // 2, q // 2 - 1).astype(np.float32) * d
    return R.reshape(rows, cols).astype(np.float16)


def build_one(task):
    index, name = task
    import ggufread
    W = ggufread.Reader(_model_path).matrix(name)
    for arm in BASE:
        out = folder(_work, arm) / f"{index}.npy"
        if out.exists():
            continue
        if arm == "X32:16":
            Wh = scalar_round(W, 16, 32)
        else:
            lattice = "Z8" if arm.startswith("Z") else "E8"
            Wh = LS.ball_round(W.astype(np.float64), lattice, LS._bounds[lattice])[0].astype(np.float16)
        np.save(out, Wh)
    return name


class Changed:
    """The matrices of an arm, made when the model asks for them. Keeps each matrix's error over its squared weights."""

    def __init__(self, model, work, arm):
        self.model, self.work, self.arm = model, work, arm
        self.index = {n: i for i, n in enumerate(model.names())}
        self.rel = {}
        m = re.match(r"^(noise|shuffle)\((.+)\)#(\d+)$", arm)
        part = re.match(r"^noisepart\((.+)\)@(small|big)#(\d+)$", arm)
        s = re.match(r"^scale\((.+),([-+]?[\d.]+)\)$", arm)
        x = re.match(r"^mix\(attn=([ZE]),ffn=([ZE])\)$", arm)
        if arm in BASE:
            self.kind, self.base = "base", arm
        elif m:
            self.kind, self.base, self.seed = m.group(1), m.group(2), int(m.group(3))
        elif part:
            self.kind, self.base, self.part, self.seed = "noisepart", part.group(1), part.group(2), int(part.group(3))
        elif s:
            self.kind, self.base, self.factor = "scale", s.group(1), 10.0 ** (float(s.group(2)) / 20.0)
        elif x:
            self.kind, self.attn, self.ffn = "mix", BALL[x.group(1)], BALL[x.group(2)]
        else:
            raise SystemExit(f"unknown arm {arm}")
        if self.kind != "mix" and self.base not in BASE:
            raise SystemExit(f"unknown base arm in {arm}")

    def __contains__(self, name):
        return name in self.index

    def load(self, base, name):
        return np.load(folder(self.work, base) / f"{self.index[name]}.npy")

    def __getitem__(self, name):
        W = self.model.rd.matrix(name).astype(np.float64)
        if self.kind == "base":
            Wh = self.load(self.base, name)
        elif self.kind == "mix":
            Wh = self.load(self.attn if name.split(".")[2] in ATTN else self.ffn, name)
        else:
            B = self.load(self.base, name)
            E = B.astype(np.float64) - W
            rows, cols = E.shape
            if self.kind == "scale":
                E = E * self.factor
            elif self.kind == "noisepart":
                rng = np.random.default_rng([self.seed, self.index[name], 1])
                zero = (B == 0).reshape(-1, GROUP)                       # the weights the base arm rounds to zero
                on = zero if self.part == "small" else ~zero
                G = E.reshape(-1, GROUP)
                n = rng.standard_normal(G.shape) * on
                want, have = ((G * on) ** 2).sum(axis=1), (n ** 2).sum(axis=1)
                n *= np.sqrt(want / np.where(have > 0, have, 1.0))[:, None]
                E = np.where(on, n, G).reshape(rows, cols)
            else:
                rng = np.random.default_rng([self.seed, self.index[name]])
                G = E.reshape(-1, GROUP)
                if self.kind == "shuffle":
                    E = rng.permuted(G, axis=1).reshape(rows, cols)
                else:
                    n = rng.standard_normal(G.shape)
                    n *= np.sqrt((G ** 2).sum(axis=1) / (n ** 2).sum(axis=1))[:, None]
                    E = n.reshape(rows, cols)
            Wh = (W + E).astype(np.float16)
        if name not in self.rel:
            self.rel[name] = float(((W - Wh.astype(np.float64)) ** 2).sum() / (W ** 2).sum())
        return Wh


def load_text(model, path):
    got = sha256_of(path)
    if got != TEXT_SHA:
        raise SystemExit(f"the text is not the run's: SHA-256 {got}")
    return model.tokenize(Path(path).read_text(encoding="utf-8"))


def main():
    global _model_path, _work
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("reference", "build", "score"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--text")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    ap.add_argument("arms", nargs="*")
    args = ap.parse_args()
    if sha256_of(args.model) != MODEL_SHA:
        raise SystemExit("the model file is not the run's")
    _model_path, _work = args.model, args.work
    Path(args.work).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.step == "build":
        for lattice in ("Z8", "E8"):
            LS._bounds[lattice] = design.ball_bound(lattice, GROUP, 560)[0]
        for arm in BASE:
            folder(args.work, arm).mkdir(parents=True, exist_ok=True)
        names = NF.Model(args.model).names()
        with get_context("fork").Pool(4) as pool:
            pool.map(build_one, list(enumerate(names)), chunksize=1)
        print(f"built {', '.join(BASE)} on {len(names)} matrices in {time.time() - t0:.0f} seconds")
        return

    model = NF.Model(args.model)
    ids = load_text(model, args.text)
    if args.step == "reference":
        P = NF.passages(ids, LENGTH, ALL)
        lp = model.log_probs(P)
        per = len(NF.scored(LENGTH))
        np.save(Path(args.work) / "ref.npy", lp[:KEPT * per])
        print(json.dumps({"step": "reference", "tokens": len(ids), "passages": ALL, "positions": int(len(lp)),
                          "ppl": model.perplexity(P, lp), "ppl_first_16": model.perplexity(P[:KEPT], lp[:KEPT * per]),
                          "seconds": round(time.time() - t0)}))
        return

    P = NF.passages(ids, LENGTH, KEPT)
    ref = np.load(Path(args.work) / "ref.npy")
    per = len(NF.scored(LENGTH))
    for arm in args.arms:
        t1 = time.time()
        ch = Changed(model, args.work, arm)
        lp = model.log_probs(P, ch)
        kl = NF.kl_rows(ref, lp)
        rel = [ch.rel[n] for n in model.names()]
        row = {"variant": arm, "passages": KEPT, "positions": int(len(kl)), "kl_mean": float(kl.mean()),
               "kl_mean_se": float(kl.std(ddof=1) / math.sqrt(len(kl))), "kl_median": float(np.median(kl)), "kl_max": float(kl.max()),
               "kl_by_passage": [float(v) for v in kl.reshape(KEPT, per).mean(axis=1)],
               "ppl": model.perplexity(P, lp), "ppl_base": model.perplexity(P, ref),
               "weight_err_db_mean": float(np.mean([db(r) for r in rel])), "matrices": len(rel),
               "seconds": round(time.time() - t1)}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{arm}: mean KL {row['kl_mean']:.5f} at {row['weight_err_db_mean']:.3f} dB, {row['seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
