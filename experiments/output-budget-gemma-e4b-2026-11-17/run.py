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
GATED = ["experiments/output-budget-gemma-e4b-2026-11-17/" + f for f in ("PREREG.md", "run.py", "allocate.py", "measures.py", "summary.py", "large_by_layer.py")] + [
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


PART_H = [f"clip:5@ffn_down#{il}" for il in range(42) if il not in LAYERS6]
JOINED = "X32s8mix:attn_q=32,attn_k=64,attn_v=64,attn_output=46,ffn_gate=16,ffn_up=16,ffn_down=16"
PART_K = [JOINED + "+deadkv"]
EXTENSION = "experiments/output-budget-gemma-e4b-2026-11-17/PREREG-extension-1.md"
EXTENSION2 = "experiments/output-budget-gemma-e4b-2026-11-17/PREREG-extension-2.md"
LARGE_JSON = HERE / "large_by_layer.json"
LAYER10 = "blk.10.ffn_down.weight"
# Part M, in the plan's order: the clip on the top unit, the top ten units, all but the top ten units, the top ten
# stream dimensions of layer 10's ffn_down; then one rule on every matrix with the other on layer 10's ffn_down.
PART_M = ["clip:5@ffn_down#10:u1", "clip:5@ffn_down#10:u10", "clip:5@ffn_down#10:not10", "clip:5@ffn_down#10:d10",
          "R+S:32|ffn_down#10=XR:32", "XR:32|ffn_down#10=R+S:32"]


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
    if part == "H":
        return PART_H
    if part == "K":
        return PART_K
    if part == "M":
        return PART_M
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
CLIP = re.compile(r"^clip:(?P<c>\d*\.?\d+)(?:@(?P<kind>\w+)(?:#(?P<layers>\d+(?:,\d+)*)(?::(?P<subset>u1|u10|not10|d10))?)?)?$")
SWAP = re.compile(r"^(?P<base>R\+S|XR):32\|ffn_down#10=(?P<other>XR|R\+S):32$")


def parse(v):
    """A variant string as a dict: arm, group, mod (None, "o" or "sB"), levels (kind -> Q), c, kind, layers, deadkv.

    A name ending in +deadkv also zeroes the keys and values that no layer reads (Part K).
    """
    if v.endswith("+deadkv"):
        spec = parse(v[:-len("+deadkv")])
        if not spec["mix"]:
            raise SystemExit(f"{v}: +deadkv is for the mixed arms")
        spec["deadkv"] = True
        return spec
    spec = parse_plain(v)
    spec["deadkv"] = False
    return spec


def parse_plain(v):
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
    m = SWAP.match(v)
    if m and m.group("base") != m.group("other"):
        return {"arm": "swap", "group": None, "mod": None, "levels": {k: 32 for k in KINDS}, "c": None, "kind": None, "layers": None,
                "mix": False, "base": m.group("base"), "other": m.group("other")}
    m = CLIP.match(v)
    if m:
        return {"arm": "clip", "group": None, "mod": None, "levels": None, "c": float(m.group("c")), "kind": m.group("kind"),
                "layers": _layers(m.group("layers")), "mix": False, "subset": m.group("subset")}
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


def large_ranks():
    """The ranked hidden units (columns) and stream dimensions (rows) of layer 10's ffn_down, from Part L's committed json."""
    d = json.loads(LARGE_JSON.read_text())["layer10"]
    return d["units_ranked"], d["dims_ranked"]


def subset_clip(W, c, subset):
    """The clip at c standard deviations of each whole row, applied only to the entries of a subset (Part M).

    u1: the entries of the top hidden unit; u10: of the top ten; not10: every entry except those of the top ten
    hidden units; d10: the entries of the top ten dimensions of the stream (rows). Other entries are left as they are.
    """
    units, dims = large_ranks()
    lim = c * W.std(axis=1, keepdims=True)
    mask = np.zeros(W.shape, dtype=bool)
    if subset == "u1":
        mask[:, units[:1]] = True
    elif subset == "u10":
        mask[:, units[:10]] = True
    elif subset == "not10":
        mask[:] = True
        mask[:, units[:10]] = False
    elif subset == "d10":
        mask[dims[:10], :] = True
    return np.where(mask, np.clip(W, -lim, lim), W)


def round_one(spec, W, kind, name=None):
    """One matrix changed as the variant says. Returns (float16 matrix, info dict with rel_err, bits, kept, overload)."""
    import tq
    if spec["arm"] == "swap":
        arm = spec["other"] if name == LAYER10 else spec["base"]
        if arm == "XR":
            Wh, info = tq.maxscale_round(W, 32, "R")
        else:
            Wh, info = tq.round_matrix(W, "R+S", 32, row_power=None)
        info.setdefault("kept", 0)
        return Wh, info
    if spec["arm"] == "clip":
        lim = spec["c"] * W.std(axis=1, keepdims=True)
        Wh = (subset_clip(W, spec["c"], spec["subset"]) if spec.get("subset") else np.clip(W, -lim, lim)).astype(np.float16)
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


def check_against_first_record(spec, per_kind, base_l10, l10_pos):
    """The base arm, as built here, must give the first record's weight errors: refuse to go on if it does not.

    The first record's results.jsonl has the mean weight error in dB of each kind for R+S:32 and XR:32 over
    all 294 matrices. Here every kind but ffn_down is rounded by the base arm throughout, so its mean must
    equal that record's. For ffn_down one matrix (layer 10) is rounded by the other arm: putting the base arm's
    own error for that matrix back must give the first record's mean.
    """
    ref = None
    for ln in (EARLIER / "results.jsonl").read_text().splitlines():
        r = json.loads(ln)
        if r["variant"] == f"{spec['base']}:32" and "weight_err_db_by_kind" in r:
            ref = r["weight_err_db_by_kind"]
    if ref is None:
        raise SystemExit("the first record has no weight errors for " + spec["base"])
    worst = {}
    for k, vs in per_kind.items():
        vals = list(vs)
        if k == "ffn_down":
            vals[l10_pos] = base_l10
        mine = sum(db(x) for x in vals) / len(vals)
        worst[k] = abs(mine - ref[k])
    if max(worst.values()) > 0.005:
        raise SystemExit(f"FAIL: {spec['base']}:32 as built here differs from the first record's weight errors: {worst}")
    return {"max_abs_difference_db": max(worst.values()), "against": f"{spec['base']}:32 of the first record, by kind"}


def bit_stats(items, weights_all, zeroed):
    """Bits per weight from items = [(weights, bits per weight)] of the matrices rounded.

    Over the weights rounded, and over all block weights: the weights not touched count 16 bits each,
    the `zeroed` weights (which no path reads) count 0 bits.
    """
    changed = sum(n for n, _ in items)
    total = sum(n * b for n, b in items)
    return total / changed, (total + 16.0 * (weights_all - changed - zeroed)) / weights_all


def build(reader, v):
    """The changed tensors of variant v and the statistics of the change."""
    spec = parse(v)
    replace, per_kind, kept_by_kind = {}, {}, {}
    everything = targets(reader)
    weights_all = sum(int(np.prod(t.shape)) for t in everything)
    changed_w, rows, bits_sum, energy, energy_err, kept, over, clipped = 0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0
    dead, dead_stats, items = {}, None, []
    if spec["deadkv"]:
        dead, dead_stats = earlier.build_dead_kv(reader)         # the first record's closure, not re-derived
    for t in everything:
        if not selected(spec, t.name) or t.name in dead:
            continue
        il, kind = BLOCK.match(t.name).groups()
        W = ggufedit.tensor_float(t)
        Wh, info = round_one(spec, W, kind, t.name)
        if spec["arm"] == "swap" and t.name == LAYER10:
            base_l10 = round_one(spec, W, kind, None)[1]["rel_err"]
            l10_pos = len(per_kind.get(kind, []))
        replace[t.name] = Wh
        per_kind.setdefault(kind, []).append(info["rel_err"])
        kept_by_kind[kind] = kept_by_kind.get(kind, 0) + info["kept"]
        e = float(np.sum(W.astype(np.float64) ** 2))
        energy += e
        energy_err += e * info["rel_err"]
        changed_w += W.size
        rows += W.shape[0]
        bits_sum += info["bits"] * W.size
        items.append((W.size, info["bits"]))
        kept += info["kept"]
        over += info["overload"] * W.size
    if not replace:
        raise SystemExit(f"{v} changes no matrix")
    every = [x for vs in per_kind.values() for x in vs]
    zeroed = dead_stats["weights_zeroed"] if dead_stats else 0
    bits_changed, bits_all = bit_stats(items, weights_all, zeroed)
    replace.update(dead)
    stats = {
        "matrices": len(every), "weights": weights_all, "weights_changed": changed_w, "rows_changed": rows,
        "bits_per_weight_changed": bits_changed,
        "bits_per_weight_all": bits_all,
        "bits_per_weight": bits_changed,
        "weight_err_db_mean": sum(db(x) for x in every) / len(every),
        "weight_err_db_pooled": db(energy_err / energy),
        "weight_err_db_by_kind": {k: sum(db(x) for x in vs) / len(vs) for k, vs in sorted(per_kind.items())},
        "kept_entries": kept, "kept_by_kind": dict(sorted(kept_by_kind.items())),
        "share_kept": kept / changed_w,
        "share_out_of_range": over / changed_w,
    }
    if spec["arm"] == "swap":
        stats["arm_check"] = check_against_first_record(spec, per_kind, base_l10, l10_pos)
    if dead_stats:
        stats["weights_zeroed"] = zeroed
        stats["tensors_zeroed"] = dead_stats["tensors"]
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


def committed_and_pushed(path):
    """The commit that last changed `path`, if the file is tracked, unchanged since, and that commit is on the remote."""
    rel = str(Path(path).resolve().relative_to(ROOT))
    if rel not in git("ls-files", "--", rel).splitlines():
        raise SystemExit(f"FAIL: {rel} is not committed")
    if git("status", "--porcelain", "--", rel):
        raise SystemExit(f"FAIL: {rel} changed since its last commit")
    commit = git("log", "-1", "--format=%H", "--", rel)
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"FAIL: the commit of {rel} is not on the remote")
    return {"file": rel, "commit": commit, "commit_time": git("log", "-1", "--format=%cI", commit)}


def first_step_passed(out):
    """The plan stops the run unless unchanged, X32:16 and XR:16 are inside their limits."""
    rows = {}
    if os.path.exists(out):
        for ln in open(out).read().splitlines():
            r = json.loads(ln)
            if "kl_mean" in r and r["variant"] not in rows:
                rows[r["variant"]] = r
    try:
        return (rows["unchanged"]["kl_mean"] < 0.00001 and abs(rows["X32:16"]["kl_mean"] - 0.117297) <= 0.002
                and abs(rows["XR:16"]["kl_mean"] - 0.448262) <= 0.006)
    except KeyError:
        return False


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
    if not set(variants) <= set(FIRST) and not first_step_passed(args.out):
        raise SystemExit("FAIL: the first step has not passed; the plan stops the run")
    if set(variants) & set(PART_H + PART_K):
        gate["extension_1"] = committed_and_pushed(ROOT / EXTENSION)
    if set(variants) & set(PART_M):
        gate["extension_2"] = committed_and_pushed(ROOT / EXTENSION2)
        gate["large_by_layer_json"] = committed_and_pushed(LARGE_JSON)
    if args.part == "B":
        gate["allocate_out"] = committed_and_pushed(args.allocate_out)
    if args.part == "G":
        gate["joined_out"] = committed_and_pushed(args.joined_out)
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
