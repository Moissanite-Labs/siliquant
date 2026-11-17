#!/usr/bin/env python3
"""Build and score the variants of the plan in PREREG.md (this folder).

It imports what it can from the first model-output record (run.py there) and changes nothing
in that folder. For each variant it builds a changed copy of the model file, scores it against
the reference outputs with llama.cpp's own tool, appends one line to the results file and
deletes the copy. It refuses to run unless the plan and the programs are committed and pushed.

Variants (KIND is one of attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down):
  unchanged                      the reference file itself
  XG:Q[@KIND[#L,L,..]]           llama.cpp's Q4_0 rule at Q levels, one scale per G weights (G a number or R)
  XRo:Q~L, X32o:Q~L              the same with the entries beyond L row standard deviations kept apart (Part C)
  X32sB:Q[@KIND[#L,L,..]]        the same with each scale stored in B bits (Part D)
  X32mix:kind=Q,kind=Q,...       one level count for each of the seven kinds, a scale per 32 weights (Part B)
  X32sBmix:kind=Q,...            the same with B-bit scales (the joined arm)
  clip:C[@KIND[#L,L,..]]         every entry beyond C standard deviations of its row pulled back to the limit
  #L,L,..                        restricts the change to those layers (Part E)

Bits per weight: "bits_per_weight_changed" over the weights changed, "bits_per_weight_all" over
all 3,945,267,200 block weights with the weights not changed counted at 16 bits (how they are
stored in a copy). "bits_per_weight" is the first of these.

Usage:
  python3 run.py --model REF.gguf --kld base.kld --work DIR --out results.jsonl --part first
  python3 run.py ... --part A | C | D | E | B | G        (B and G read the variants from a committed output file)
  python3 run.py ... VARIANT [VARIANT ...]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EARLIER = ROOT / "experiments" / "model-output-gemma-e4b-2026-11-15"
sys.path.insert(0, str(ROOT / "harness"))
import evalkl  # noqa: E402
import ggufedit  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


earlier = _load("earlier_run", EARLIER / "run.py")
KINDS = earlier.KINDS
BLOCK = earlier.BLOCK
targets = earlier.targets
db = earlier.db

MODEL_SHA256 = "f8854aa4480df62585a279e7ca0a881554fc18a41c59c4f62642d16a2ae47012"
MODEL_BYTES = 8192953472
BLOCK_WEIGHTS = 3945267200
LAYERS6 = (0, 8, 16, 24, 32, 40)
GATED = ["experiments/output-budget-gemma-e4b-2026-11-17/" + f for f in ("PREREG.md", "run.py", "allocate.py", "measures.py", "summary.py")] + [
    "harness/tq.py", "harness/ggufedit.py", "harness/evalkl.py"]

# ---------------------------------------------------------------------------
# The variants of each part
# ---------------------------------------------------------------------------

FIRST = ["unchanged", "X32:16", "XR:16"]
PART_A = [f"X32:{q}@{k}" for k in KINDS for q in (8, 16, 32)] + ["X32:8"]
PART_C = ([f"XRo:{q}~{L}" for q in (16, 32) for L in ("3.45", "5", "8")] + ["XRo:16~2.5", "X32o:16~3.45"])
PART_D = ["X32s8:16", "X32s6:16", "X32s8:18", "X32s6:20"]
PART_E = ([f"{a}@ffn_down#{il}" for a in ("clip:5", "X32:16") for il in LAYERS6]
          + [f"{a}@ffn_down#" + ",".join(str(i) for i in LAYERS6) for a in ("clip:5", "X32:16")])


def part_variants(part, allocate_out=None, joined_out=None):
    if part == "first":
        return FIRST
    if part == "A":
        return PART_A
    if part == "C":
        return PART_C
    if part == "D":
        return PART_D
    if part == "E":
        return PART_E
    if part in ("B", "G"):
        path = allocate_out if part == "B" else joined_out
        if path is None or not Path(path).exists():
            raise SystemExit(f"part {part} needs its committed output file (--{'allocate' if part == 'B' else 'joined'}-out)")
        out = []
        for ln in Path(path).read_text().splitlines():
            if ln.startswith("JSON "):
                r = json.loads(ln[5:])
                if r.get("score"):
                    out.append(r["variant"])
        if not out:
            raise SystemExit(f"{path} lists no variant to score")
        return out
    raise SystemExit(f"unknown part {part}")


# ---------------------------------------------------------------------------
# Variant strings
# ---------------------------------------------------------------------------

ARM = re.compile(r"^X(?P<g>\d+|R|M)(?P<mod>o|s\d+)?:(?P<q>\d+)(?:~(?P<c>\d*\.?\d+))?(?:@(?P<kind>\w+)(?:#(?P<layers>\d+(?:,\d+)*))?)?$")
MIX = re.compile(r"^X32(?P<mod>s\d+)?mix:(?P<levels>.+)$")
CLIP = re.compile(r"^clip:(?P<c>\d*\.?\d+)(?:@(?P<kind>\w+)(?:#(?P<layers>\d+(?:,\d+)*))?)?$")


def parse(v):
    """A variant string as a dict: arm, group, mod (None, "o" or "sB"), levels (kind -> Q), c, kind, layers."""
    m = MIX.match(v)
    if m:
        levels = {}
        for part in m.group("levels").split(","):
            k, _, q = part.partition("=")
            if k not in KINDS or not q.isdigit():
                raise SystemExit(f"bad mixed variant {v}")
            levels[k] = int(q)
        if set(levels) != set(KINDS):
            raise SystemExit(f"a mixed variant names all seven kinds: {v}")
        return {"arm": "X", "group": "32", "mod": m.group("mod"), "levels": levels, "c": None, "kind": None, "layers": None, "mix": True}
    m = CLIP.match(v)
    if m:
        return {"arm": "clip", "group": None, "mod": None, "levels": None, "c": float(m.group("c")), "kind": m.group("kind"),
                "layers": _layers(m.group("layers")), "mix": False}
    m = ARM.match(v)
    if not m:
        raise SystemExit(f"unknown variant {v}")
    mod, c = m.group("mod"), m.group("c")
    if mod == "o" and c is None:
        raise SystemExit(f"{v}: the arms with kept entries need ~L")
    if mod is not None and mod.startswith("s") and (m.group("g") != "32" or c is not None):
        raise SystemExit(f"{v}: stored scales are for groups of 32 and take no ~")
    if m.group("kind") is not None and m.group("kind") not in KINDS:
        raise SystemExit(f"unknown kind in {v}")
    q = int(m.group("q"))
    return {"arm": "X", "group": m.group("g"), "mod": mod, "levels": {k: q for k in KINDS}, "c": float(c) if c else None,
            "kind": m.group("kind"), "layers": _layers(m.group("layers")), "mix": False}


def _layers(s):
    return tuple(int(x) for x in s.split(",")) if s else None


def selected(spec, name):
    """Whether the variant changes the matrix `name`."""
    m = BLOCK.match(name)
    if not m:
        return False
    il, kind = int(m.group(1)), m.group(2)
    return (spec["kind"] is None or spec["kind"] == kind) and (spec["layers"] is None or il in spec["layers"])


def round_one(spec, W, kind):
    """One matrix changed as the variant says. Returns (float16 matrix, info dict with rel_err, bits, kept, overload)."""
    import tq
    if spec["arm"] == "clip":
        lim = spec["c"] * W.std(axis=1, keepdims=True)
        Wh = np.clip(W, -lim, lim).astype(np.float16)
        W64 = W.astype(np.float64)
        return Wh, {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2)), "bits": 16.0,
                    "overload": float(np.mean(np.abs(W) > lim)), "kept": 0}
    q = spec["levels"][kind]
    group = spec["group"]
    mod = spec["mod"]
    if mod == "o":
        Wh, info = tq.maxscale_round_kept(W, q, group, spec["c"])
    elif mod is not None:
        Wh, info = tq.maxscale_round_sbit(W, q, group, int(mod[1:]))
    else:
        Wh, info = tq.maxscale_round(W, q, group, shrink=spec["c"] or 1.0)
    info.setdefault("kept", 0)
    return Wh, info


def build(reader, v):
    """The changed tensors of variant v and the statistics of the change."""
    spec = parse(v)
    replace, per_kind, kept_by_kind = {}, {}, {}
    everything = targets(reader)
    weights_all = sum(int(np.prod(t.shape)) for t in everything)
    changed_w, rows, bits_sum, energy, energy_err, kept, over, clipped = 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0
    for t in everything:
        if not selected(spec, t.name):
            continue
        il, kind = BLOCK.match(t.name).groups()
        W = ggufedit.tensor_float(t)
        Wh, info = round_one(spec, W, kind)
        replace[t.name] = Wh
        per_kind.setdefault(kind, []).append(info["rel_err"])
        kept_by_kind[kind] = kept_by_kind.get(kind, 0) + info["kept"]
        e = float(np.sum(W.astype(np.float64) ** 2))
        energy += e
        energy_err += e * info["rel_err"]
        changed_w += W.size
        rows += W.shape[0]
        bits_sum += info["bits"] * W.size
        kept += info["kept"]
        over += info["overload"] * W.size
    if not replace:
        raise SystemExit(f"{v} changes no matrix")
    every = [x for vs in per_kind.values() for x in vs]
    stats = {
        "matrices": len(every), "weights": weights_all, "weights_changed": changed_w, "rows_changed": rows,
        "bits_per_weight_changed": bits_sum / changed_w,
        "bits_per_weight_all": (bits_sum + 16.0 * (weights_all - changed_w)) / weights_all,
        "bits_per_weight": bits_sum / changed_w,
        "weight_err_db_mean": sum(db(x) for x in every) / len(every),
        "weight_err_db_pooled": db(energy_err / energy),
        "weight_err_db_by_kind": {k: sum(db(x) for x in vs) / len(vs) for k, vs in sorted(per_kind.items())},
        "kept_entries": kept, "kept_by_kind": dict(sorted(kept_by_kind.items())),
        "share_kept": kept / changed_w,
        "share_out_of_range": over / changed_w,
    }
    return replace, stats


# ---------------------------------------------------------------------------
# The gate, and what is recorded at the start
# ---------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git(*a):
    return subprocess.run(("git", "-C", str(ROOT)) + a, capture_output=True, text=True, check=True).stdout.strip()


def plan_gate():
    """Refuse to run unless the plan and the programs are committed as they stand and the commit is on the remote."""
    try:
        dirty = git("status", "--porcelain", "--", *GATED)
        head = git("rev-parse", "HEAD")
        tracked = git("ls-files", "--", *GATED).splitlines()
        plan_commit = git("log", "-1", "--format=%H", "--", GATED[0])
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
    if not pushed:
        raise SystemExit("FAIL: the commit is not on the remote: push before scoring")
    return {"plan_commit": plan_commit, "head": head, "commit_is_on_the_remote": pushed,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": {f: sha256_of(ROOT / f) for f in GATED}}


def done_variants(out):
    if not os.path.exists(out):
        return set()
    return {json.loads(ln)["variant"] for ln in open(out).read().splitlines() if ln.strip() and "kl_mean" in json.loads(ln)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", help="first, A, B, C, D, E or G")
    ap.add_argument("--allocate-out", help="the committed output of allocate.py, for part B")
    ap.add_argument("--joined-out", help="the committed output of allocate.py --scale-bits 8, for part G")
    ap.add_argument("--start-file", help="results.start (default: next to --out)")
    ap.add_argument("--no-skip", action="store_true", help="score variants that already have a line (not for the record)")
    ap.add_argument("--metrics-only", action="store_true", help="build each variant and record its weight errors, without scoring it")
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    variants = list(args.variants) + (part_variants(args.part, args.allocate_out, args.joined_out) if args.part else [])
    if not variants:
        raise SystemExit("no variants")
    for v in variants:
        if v != "unchanged":
            parse(v)                       # a bad name stops the run before anything is built
    gate = plan_gate()
    if os.path.getsize(args.model) != MODEL_BYTES:
        raise SystemExit("FAIL: the model file has the wrong size")
    gate["model_sha256"] = sha256_of(args.model)
    if gate["model_sha256"] != MODEL_SHA256:
        raise SystemExit("FAIL: the model file's SHA-256 differs from the plan's")
    gate["variants"] = variants
    start = Path(args.start_file) if args.start_file else Path(args.out).with_name("results.start")
    with open(start, "a") as f:
        f.write(json.dumps(gate) + "\n")
    reader = ggufedit.open_model(args.model)
    os.makedirs(args.work, exist_ok=True)
    skip = set() if args.no_skip or args.metrics_only else done_variants(args.out)
    for v in variants:
        if v in skip:
            print(f"skip {v}: already scored", flush=True)
            continue
        t0 = time.time()
        row = {"variant": v}
        if v == "unchanged":
            path, made, replace = args.model, False, None
            row.update({"weights": BLOCK_WEIGHTS, "weights_changed": 0})
        else:
            path, made = os.path.join(args.work, re.sub(r"[^A-Za-z0-9_.+-]", "_", v) + ".gguf"), True
            replace, stats = build(reader, v)
            spec = parse(v)
            row.update({"arm": v.split(":")[0], "kind": spec["kind"] or "all", "layers": list(spec["layers"]) if spec["layers"] else None})
            if spec["levels"] is not None and not spec["mix"]:
                row["q"] = spec["levels"]["attn_q"]
            if spec["mix"]:
                row["levels"] = spec["levels"]
            row.update(stats)
            row["build_seconds"] = round(time.time() - t0, 1)
            if args.metrics_only:
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                continue
            ggufedit.write_copy(reader, path, replace)
            del replace
        t1 = time.time()
        scores, _ = evalkl.score(path, args.kld)
        row.update(scores)
        row["score_seconds"] = round(time.time() - t1, 1)
        if made:
            os.remove(path)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("variant", "bits_per_weight", "weight_err_db_mean", "kl_mean", "kl_mean_se", "same_top") if k in row}), flush=True)


if __name__ == "__main__":
    main()
