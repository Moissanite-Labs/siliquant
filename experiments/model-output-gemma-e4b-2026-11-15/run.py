#!/usr/bin/env python3
"""Round the weights of a real language model and measure what its output does.

The plan is PREREG.md in this folder. One call handles a list of variants.
For each it builds a changed copy of the model file, scores it against the
reference model's saved outputs with llama.cpp's own tool, appends one line
to a results file and deletes the copy.

Variants:
  unchanged         the reference file itself, scored against its own saved outputs
  identity          the block matrices unrounded, stored as 16-bit floats (a control)
  ARM:Q             a rounding arm at Q levels per weight on the block matrices,
                    ARM one of S, Z8, E8, with an optional prefix D+ (rows and columns balanced),
                    R+ (rows only), C+ (columns only), Rp+ (rows divided by their standard deviation
                    to the power p, as in R0.5+S) or P+ (rows and columns scaled for the second
                    path-weighted error) or Bn+ (one scale per block of n consecutive weights in a
                    row, as in B32+S)
  XG:Q              llama.cpp's Q4_0 rule at Q levels: one scale per group G of weights, set by the
                    group's largest entry, with no search and no clipping. G is a number of
                    consecutive weights in a row, R for a row or M for the matrix (X32:16 is Q4_0).
                    XG:Q~C multiplies the step by C, which clips entries above C times the largest
  VE8G:Q            the E8 Voronoi code with one scale per group G, set so that no block of the
                    group is shrunk (VE832:16 is the lattice form of Q4_0; VZ8G uses the cube)
  clip:C            no rounding: every entry beyond C standard deviations of its row is pulled back
                    to that limit (clip:C@KIND on one kind of matrix only)
  flip:C            the same entries as clip:C moved by the same amounts, each toward or away from
                    zero at random
  ARM:Q@KIND        the same on one kind of matrix only (attn_q, attn_k, attn_v, attn_output,
                    ffn_gate, ffn_up, ffn_down), the rest left alone
  close:TENSOR      zero the rows of a token table (per_layer_token_embd or token_embd)
                    for every token that does not occur in the evaluation text
  dead:kv           zero the key and value matrices of the layers that reuse an earlier
                    layer's keys and values (llama.cpp never reads them)
  gauge:SEED        move the weights along the model's exact symmetries by random amounts,
                    with no rounding
  G+ARM:Q           use the same symmetries to balance the matrices, then round with ARM
  GF+ARM:Q          the same with the gain vectors moved wholly into the matrices
  file:PATH         score an existing model file (for llama.cpp's own quantisers)

Usage, with NumPy and torch, llama.cpp's gguf package and llama-perplexity:
  python3 run.py --model REF.gguf --kld base.kld --work DIR --out results.jsonl identity S:16 E8:16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
import evalkl  # noqa: E402
import ggufedit  # noqa: E402

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")


def targets(reader, kind=None):
    out = []
    for t in reader.tensors:
        m = BLOCK.match(t.name)
        if m and len(t.shape) == 2 and (kind is None or m.group(2) == kind):
            out.append(t)
    return out


def db(x):
    return 10.0 * math.log10(x) if x > 0 else float("-inf")


def meta(reader, key):
    return reader.get_field(key).contents()


READ_GAIN = {"attn_q": "attn_norm", "attn_k": "attn_norm", "attn_v": "attn_norm", "ffn_gate": "ffn_norm", "ffn_up": "ffn_norm"}
WRITE_GAIN = {"attn_output": "post_attention_norm", "ffn_down": "post_ffw_norm"}


def path_weighted_err(name, W, Wh, by, layer=None):
    """Relative error with each weight counted by what it is multiplied with on its way through the block.

    Read off the forward pass, with no data:
      - a matrix that reads the normalised stream (attn_q, attn_k, attn_v, ffn_gate, ffn_up) has
        column j multiplied by entry j of the gain vector before it, so column j counts by that gain squared;
      - a matrix that writes to the stream (attn_output, ffn_down) is followed by a norm and a gain
        vector, so row k counts by that gain squared;
      - row i of ffn_up is then multiplied by column i of ffn_down, and column i of ffn_down by the
        output of hidden unit i, so each also counts by the squared size of the other, taken with the
        other's own gains.
    The measure does not change under the three symmetries of gauge_layer, which plain squared
    error does. `layer` holds tensors a variant has changed (matrices before rounding, and gains).
    """
    il, kind = BLOCK.match(name).groups()

    def g(n):
        full = f"blk.{il}.{n}.weight"
        return (layer[full] if layer and full in layer else ggufedit.tensor_float(by[full])).astype(np.float64)

    num = (W.astype(np.float64) - Wh.astype(np.float64)) ** 2
    den = W.astype(np.float64) ** 2
    row = col = None
    if kind in READ_GAIN:
        col = g(READ_GAIN[kind]) ** 2
        if kind == "ffn_up":
            row = (g("ffn_down") ** 2 * (g("post_ffw_norm") ** 2)[:, None]).sum(axis=0)
    else:
        row = g(WRITE_GAIN[kind]) ** 2
        if kind == "ffn_down":
            col = (g("ffn_up") ** 2 * (g("ffn_norm") ** 2)[None, :]).sum(axis=1)
    if col is not None:
        num, den = num * col[None, :], den * col[None, :]
    if row is not None:
        num, den = num * row[:, None], den * row[:, None]
    return float(num.sum() / den.sum())


_PW = {}


def path2_weights(reader, by, il, layer=None):
    """Row and column weights for the seven matrices of one layer: {kind: (row weights, column weights)}.

    Written after round 2. Each weight's squared error is counted by the squared size of everything
    it is multiplied with on both sides, read off the forward pass with no data:
      - columns of the five reading matrices by the gain vector before them (as in the first measure);
      - rows of attn_output and ffn_down by the gain vector after the norm that follows them;
      - the three feed-forward matrices by each other: hidden unit i computes
        gelu(gate_i x) * (up_i x) and writes it through column i of ffn_down, so row i of ffn_gate
        counts by the sizes of row i of ffn_up and column i of ffn_down, and so on round;
      - rows of attn_q and attn_k by the gains of the two norms between them and the dot product,
        divided by the size of their own head, because the norm rescales a head as a whole;
      - rows of attn_v by the columns of attn_output they feed, divided by the size of their head.
    Sizes are taken with the matrix's own gains. Query heads are taken to share a key and value head
    in consecutive groups. The keys and values of the last 18 layers are never read: weight one.
    """
    if layer is None and il in _PW:
        return _PW[il]

    def g(n, layer_index=il):
        full = f"blk.{layer_index}.{n}.weight"
        return (layer[full] if layer and full in layer else ggufedit.tensor_float(by[full])).astype(np.float64)

    arch = meta(reader, "general.architecture")
    live = live_kv_layers(reader)
    swa = meta(reader, f"{arch}.attention.sliding_window_pattern")
    src = il if il < live else live - (2 if swa[il] else 1)       # the layer whose keys this layer's queries meet
    ga, gf = g("attn_norm") ** 2, g("ffn_norm") ** 2
    ha, hf = g("post_attention_norm") ** 2, g("post_ffw_norm") ** 2
    qn, kn = g("attn_q_norm") ** 2, g("attn_k_norm", src) ** 2
    hd = len(qn)

    def per_head(M, colw):
        e = (M ** 2 * colw[None, :]).sum(axis=1)
        return np.repeat(e.reshape(-1, hd).mean(axis=1), hd)

    out = {}
    q, o = g("attn_q"), g("attn_output")
    n_head = q.shape[0] // hd
    out["attn_q"] = (np.tile(qn * kn, n_head) / per_head(q, ga), ga)
    out["attn_output"] = (ha, None)
    if il < live:
        k, v = g("attn_k"), g("attn_v")
        n_kv = k.shape[0] // hd
        out["attn_k"] = (np.tile(qn * kn, n_kv) / per_head(k, ga), ga)
        ocol = (o ** 2 * ha[:, None]).sum(axis=0).reshape(n_kv, n_head // n_kv, hd).sum(axis=1).reshape(-1)
        out["attn_v"] = (ocol / per_head(v, ga), ga)
    else:
        out["attn_k"] = out["attn_v"] = (None, None)
    gate, up, down = g("ffn_gate"), g("ffn_up"), g("ffn_down")
    ng, nu = (gate ** 2 * gf[None, :]).sum(axis=1), (up ** 2 * gf[None, :]).sum(axis=1)
    nd = (down ** 2 * hf[:, None]).sum(axis=0)
    out["ffn_gate"] = (nu * nd, gf)
    out["ffn_up"] = (ng * nd, gf)
    out["ffn_down"] = (hf, ng * nu)
    if layer is None:
        _PW[il] = out
    return out


def path2_err(reader, name, W, Wh, by, layer=None):
    """Relative error under the weights of path2_weights."""
    il, kind = BLOCK.match(name).groups()
    row, col = path2_weights(reader, by, int(il), layer)[kind]
    num = (W.astype(np.float64) - Wh.astype(np.float64)) ** 2
    den = W.astype(np.float64) ** 2
    if col is not None:
        num, den = num * col[None, :], den * col[None, :]
    if row is not None:
        num, den = num * row[:, None], den * row[:, None]
    return float(num.sum() / den.sum())


def live_kv_layers(reader):
    """Layers that compute their own keys and values. The last `shared_kv_layers` reuse an earlier layer's."""
    arch = meta(reader, "general.architecture")
    return int(meta(reader, f"{arch}.block_count")) - int(meta(reader, f"{arch}.attention.shared_kv_layers"))


def rms(W, axis):
    return np.sqrt(np.mean(W.astype(np.float64) ** 2, axis=axis))


def gauge_layer(by, il, live, rng, full=False, factors=None):
    """One layer moved along its exact symmetries. Returns {tensor name: new float32 array}.

    The three symmetries used, each exact in the forward pass llama.cpp builds for this model:
      1. a hidden unit of the feed-forward block: its row of ffn_up times c, its column of ffn_down
         divided by c (the ffn_up branch is linear: the unit computes gelu(gate x) * (up x));
      2. an input dimension of the feed-forward block: its column of ffn_gate and of ffn_up times d,
         its entry of the gain vector ffn_norm divided by d;
      3. an input dimension of the attention block: its column of attn_q, attn_k and attn_v times e,
         its entry of the gain vector attn_norm divided by e.
    With rng given the amounts are random. Without, they balance each pair: the two sides of a
    symmetry are given the same size. For symmetries 2 and 3 that is the choice that minimises
    the effect of equal noise in every weight when the normalised inputs are equally large. With
    full=True symmetries 2 and 3 instead move the whole gain into the matrices (the gain vector
    becomes plus or minus one).
    """
    def get(kind):
        return ggufedit.tensor_float(by[f"blk.{il}.{kind}.weight"])

    up, gate, down = get("ffn_up"), get("ffn_gate"), get("ffn_down")
    g_ffn, g_attn = get("ffn_norm").astype(np.float64), get("attn_norm").astype(np.float64)
    att = {k: get(k) for k in (("attn_q", "attn_k", "attn_v") if il < live else ("attn_q",))}
    n_hidden, n_in = up.shape
    tiny = 1e-12
    if rng is not None:
        d = np.exp(rng.uniform(-math.log(2), math.log(2), n_in))
        e = np.exp(rng.uniform(-math.log(2), math.log(2), n_in))
        c = np.exp(rng.uniform(-math.log(4), math.log(4), n_hidden)) * rng.choice([-1.0, 1.0], n_hidden)
    else:
        s_ffn = np.sqrt((rms(gate, 0) ** 2 + rms(up, 0) ** 2) / 2.0)
        s_att = np.sqrt(sum(rms(W, 0) ** 2 * W.shape[0] for W in att.values()) / sum(W.shape[0] for W in att.values()))
        if full:
            d, e = np.maximum(np.abs(g_ffn), tiny), np.maximum(np.abs(g_attn), tiny)
        else:
            d = np.sqrt(np.maximum(np.abs(g_ffn), tiny) / np.maximum(s_ffn, tiny))
            e = np.sqrt(np.maximum(np.abs(g_attn), tiny) / np.maximum(s_att, tiny))
        a = rms(up * d[None, :].astype(np.float32), 1)
        c = np.sqrt(np.maximum(rms(down, 0), tiny) / np.maximum(a, tiny))
    d32, e32, c32 = d.astype(np.float32), e.astype(np.float32), c.astype(np.float32)
    out = {
        f"blk.{il}.ffn_up.weight": up * d32[None, :] * c32[:, None],
        f"blk.{il}.ffn_gate.weight": gate * d32[None, :],
        f"blk.{il}.ffn_down.weight": down / c32[None, :],
        f"blk.{il}.ffn_norm.weight": (g_ffn / d).astype(np.float32),
        f"blk.{il}.attn_norm.weight": (g_attn / e).astype(np.float32),
    }
    for k, W in att.items():
        out[f"blk.{il}.{k}.weight"] = W * e32[None, :]
    if factors is not None:
        factors.update({"c": c, "d": d, "e": e})
    return out


def build_gauged(reader, arm, q, seed, full=False):
    """Move every layer along its symmetries, then round with `arm` (or not at all when arm is None)."""
    by = {t.name: t for t in reader.tensors}
    live = live_kv_layers(reader)
    n_layer = live + int(meta(reader, f"{meta(reader, 'general.architecture')}.attention.shared_kv_layers"))
    rng = np.random.default_rng(seed) if arm is None else None
    replace, per_kind, weights, energy_err, energy, moved, bits, gw = {}, {}, 0, 0.0, 0.0, 0.0, [], {}
    if arm is not None:
        import tq
    for il in range(n_layer):
        layer = gauge_layer(by, il, live, rng, full)
        for name, W in layer.items():
            if W.ndim == 1:
                replace[name] = W
                continue
            W0 = ggufedit.tensor_float(by[name]).astype(np.float64)
            e0 = float(np.sum(W0 ** 2))
            moved += float(np.sum((W0 - W.astype(np.float64)) ** 2))
            energy += e0
            weights += W.size
            if arm is None:
                replace[name] = W.astype(np.float16)
                continue
            Wh, info = tq.round_matrix(np.ascontiguousarray(W), arm, q)
            replace[name] = Wh
            per_kind.setdefault(BLOCK.match(name).group(2), []).append(info["rel_err"])
            gw.setdefault(BLOCK.match(name).group(2), []).append(path_weighted_err(name, W, Wh, by, layer))
            energy_err += float(np.sum(W.astype(np.float64) ** 2)) * info["rel_err"]
            bits.append(info["bits"] * W.size)
    stats = {"weights": weights, "weight_move_db": db(moved / energy), "layers": n_layer}
    if arm is not None:
        # attn_output and the unread key and value matrices have no symmetry used here: round them as they are.
        for t in targets(reader):
            if t.name not in replace:
                W = ggufedit.tensor_float(t)
                Wh, info = tq.round_matrix(W, arm, q)
                replace[t.name] = Wh
                per_kind.setdefault(BLOCK.match(t.name).group(2), []).append(info["rel_err"])
                gw.setdefault(BLOCK.match(t.name).group(2), []).append(path_weighted_err(t.name, W, Wh, by))
                bits.append(info["bits"] * W.size)
                weights += W.size
        every = [v for vs in per_kind.values() for v in vs]
        stats.update({"weights": weights, "matrices": len(every), "bits_per_weight": sum(bits) / weights,
                      "weight_err_db_mean": sum(db(v) for v in every) / len(every),
                      "weight_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(per_kind.items())},
                      "path_err_db_mean": sum(db(v) for vs in gw.values() for v in vs) / len(every),
                      "path_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(gw.items())}})
    return replace, stats


def build_dead_kv(reader):
    by = {t.name: t for t in reader.tensors}
    live = live_kv_layers(reader)
    n_layer = live + int(meta(reader, f"{meta(reader, 'general.architecture')}.attention.shared_kv_layers"))
    replace, n = {}, 0
    for il in range(live, n_layer):
        for kind in ("attn_k", "attn_v"):
            t = by[f"blk.{il}.{kind}.weight"]
            replace[t.name] = np.zeros_like(np.array(t.data))
            n += int(np.prod(t.shape))
    return replace, {"layers": n_layer - live, "tensors": len(replace), "weights_zeroed": n,
                     "weights_in_model": int(sum(int(np.prod(x.shape)) for x in reader.tensors))}


def build_rounded(reader, arm, q, kind, row_power=None, shrink=1.0):
    import tq
    replace, per_kind, weights, energy_err, energy = {}, {}, 0, 0.0, 0.0
    bits, gw, gw2, clipped, over = [], {}, {}, [], []
    by = {t.name: t for t in reader.tensors}
    for t in targets(reader, kind):
        W = ggufedit.tensor_float(t)
        if arm == "identity":
            Wh = W.astype(np.float16)
            W64 = W.astype(np.float64)
            info = {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2)), "bits": 16.0, "overload": 0.0}
        elif arm == "clip":
            # No rounding: every entry beyond `shrink` standard deviations of its row is pulled back to that limit.
            lim = shrink * W.std(axis=1, keepdims=True)
            Wh = np.clip(W, -lim, lim).astype(np.float16)
            W64 = W.astype(np.float64)
            info = {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2)), "bits": 16.0,
                    "overload": float(np.mean(np.abs(W) > lim))}
            clipped.append(info["overload"] * W.size)
        elif arm == "flip":
            # The same entries as clip, moved by the same amounts, but each toward or away from zero at random.
            sd = W.std(axis=1, keepdims=True)
            lim = shrink * sd
            excess = np.maximum(np.abs(W) - lim, 0.0)
            sign = np.random.default_rng(len(replace)).choice([-1.0, 1.0], size=W.shape).astype(np.float32)
            Wh = (W + np.sign(W) * excess * sign).astype(np.float16)
            W64 = W.astype(np.float64)
            info = {"rel_err": float(np.sum((W64 - Wh.astype(np.float64)) ** 2) / np.sum(W64 ** 2)), "bits": 16.0,
                    "overload": float(np.mean(excess > 0))}
            clipped.append(info["overload"] * W.size)
        elif arm.startswith("V"):
            Wh, info = tq.maxscale_voronoi(W, q, arm[3:], name=arm[1:3])
        elif arm.startswith("X"):
            Wh, info = tq.maxscale_round(W, q, arm[1:], shrink=shrink)
        elif arm.startswith("P+"):
            il_, kind_ = BLOCK.match(t.name).groups()
            rw, cw = path2_weights(reader, by, int(il_))[kind_]
            Wh, info = tq.round_matrix(W, arm, q, row_w=rw, col_w=cw)
        else:
            Wh, info = tq.round_matrix(W, arm, q, row_power=row_power)
        replace[t.name] = Wh
        k = BLOCK.match(t.name).group(2)
        per_kind.setdefault(k, []).append(info["rel_err"])
        if info.get("overload") is not None and arm not in ("clip", "flip"):
            over.append(info["overload"] * W.size)
        gw.setdefault(k, []).append(path_weighted_err(t.name, W, Wh, by))
        gw2.setdefault(k, []).append(path2_err(reader, t.name, W, Wh, by))
        e = float(np.sum(W.astype(np.float64) ** 2))
        energy += e
        energy_err += e * info["rel_err"]
        weights += W.size
        bits.append(info["bits"] * W.size)
    every = [v for vs in per_kind.values() for v in vs]
    stats = {
        "matrices": len(every), "weights": weights, "bits_per_weight": sum(bits) / weights,
        "weight_err_db_mean": sum(db(v) for v in every) / len(every),
        "weight_err_db_pooled": db(energy_err / energy),
        "weight_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(per_kind.items())},
        "path_err_db_mean": sum(db(v) for vs in gw.values() for v in vs) / len(every),
        "path_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(gw.items())},
        "path2_err_db_mean": sum(db(v) for vs in gw2.values() for v in vs) / len(every),
        "path2_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(gw2.items())},
    }
    if clipped:
        stats["share_clipped"] = sum(clipped) / weights
    if over:
        # Share of weights clipped (scalar arms) or of blocks shrunk (lattice arms). Not recorded for the first 57 variants.
        stats["share_out_of_range"] = sum(over) / weights
    return replace, stats


def build_closed(reader, tensor_name, kld):
    n_ctx, n_vocab, n_chunk, toks = evalkl.tokens_of(kld)
    used = np.zeros(n_vocab, dtype=bool)
    used[toks] = True
    t = next(x for x in reader.tensors if x.name == tensor_name)
    raw = np.array(t.data)                      # storage bytes, one row per token
    if raw.shape[0] != n_vocab:
        raise SystemExit(f"{tensor_name}: {raw.shape[0]} rows, expected one per token ({n_vocab})")
    raw[~used] = 0
    per_row = int(np.prod(t.shape)) // n_vocab
    stats = {"tokens_in_text": int(used.sum()), "rows_zeroed": int((~used).sum()),
             "weights_zeroed": int((~used).sum()) * per_row, "weights_in_tensor": int(np.prod(t.shape)),
             "weights_in_model": int(sum(int(np.prod(x.shape)) for x in reader.tensors))}
    return {tensor_name: raw}, stats


def compare_file(reader, path):
    """Bits per weight and weight error of the block matrices of another model file against the reference."""
    other = {t.name: t for t in ggufedit.open_model(path).tensors}
    by = {t.name: t for t in reader.tensors}
    per_kind, gw, weights, nbytes, energy_err, energy = {}, {}, 0, 0, 0.0, 0.0
    for t in targets(reader):
        o = other[t.name]
        W = ggufedit.tensor_float(t).astype(np.float64)
        Wh = ggufedit.tensor_float(o).astype(np.float64)
        e = float(np.sum(W ** 2))
        rel = float(np.sum((W - Wh) ** 2)) / e
        per_kind.setdefault(BLOCK.match(t.name).group(2), []).append(rel)
        gw.setdefault(BLOCK.match(t.name).group(2), []).append(path_weighted_err(t.name, W, Wh, by))
        energy += e
        energy_err += e * rel
        weights += W.size
        nbytes += int(o.n_bytes)
    every = [v for vs in per_kind.values() for v in vs]
    return {"matrices": len(every), "weights": weights, "bits_per_weight": 8.0 * nbytes / weights,
            "path_err_db_mean": sum(db(v) for vs in gw.values() for v in vs) / len(every),
            "path_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(gw.items())},
            "weight_err_db_mean": sum(db(v) for v in every) / len(every), "weight_err_db_pooled": db(energy_err / energy),
            "weight_err_db_by_kind": {k: sum(db(v) for v in vs) / len(vs) for k, vs in sorted(per_kind.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kld", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the changed model files")
    ap.add_argument("--metrics-only", action="store_true", help="build each variant and record its weight errors, without scoring it")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    reader = ggufedit.open_model(args.model)
    os.makedirs(args.work, exist_ok=True)
    for v in args.variants:
        t0 = time.time()
        row = {"variant": v}
        if v == "unchanged":
            path, made = args.model, False
        elif v.startswith("file:"):
            path, made = v[5:], False
            row.update(compare_file(reader, path))
            row["variant"] = "file:" + os.path.basename(path)
        else:
            path, made = os.path.join(args.work, re.sub(r"[^A-Za-z0-9_.+-]", "_", v) + ".gguf"), True
            if v == "identity":
                replace, stats = build_rounded(reader, "identity", 0, None)
            elif v.startswith("clip:") or v.startswith("flip:"):
                limit, _, only = v[5:].partition("@")
                if only and only not in KINDS:
                    raise SystemExit(f"unknown kind {only}")
                replace, stats = build_rounded(reader, v[:4], 0, only or None, shrink=float(limit))
                row["kind"] = only or "all"
            elif v.startswith("close:"):
                replace, stats = build_closed(reader, v[6:], args.kld)
            elif v == "dead:kv":
                replace, stats = build_dead_kv(reader)
            elif v.startswith("gauge:"):
                replace, stats = build_gauged(reader, None, 0, int(v[6:]))
            elif re.match(r"^G(F?)\+(S|E8):(\d+)$", v):
                m = re.match(r"^G(F?)\+(S|E8):(\d+)$", v)
                replace, stats = build_gauged(reader, m.group(2), int(m.group(3)), 0, full=bool(m.group(1)))
                row.update({"arm": "G" + m.group(1) + "+" + m.group(2), "q": int(m.group(3)), "kind": "all"})
            else:
                m = re.match(r"^(?:([DRCP]\+|B\d+\+)|R(\d*\.?\d+)\+)?(S|Z8|E8|X\d+|XR|XM|V(?:E8|Z8)(?:\d+|R)):(\d+)(?:~(\d*\.?\d+))?(?:@(\w+))?$", v)
                if not m:
                    raise SystemExit(f"unknown variant {v}")
                power = float(m.group(2)) if m.group(2) else None
                arm = (m.group(1) or ("R+" if power is not None else "")) + m.group(3)
                q, kind = int(m.group(4)), m.group(6)
                shrink = float(m.group(5)) if m.group(5) else 1.0
                if kind is not None and kind not in KINDS:
                    raise SystemExit(f"unknown kind {kind}")
                replace, stats = build_rounded(reader, arm, q, kind, row_power=power, shrink=shrink)
                row.update({"arm": v.split(":")[0], "q": q, "kind": kind or "all"})
            row.update(stats)
            row["build_seconds"] = round(time.time() - t0, 1)
            if args.metrics_only:
                del replace
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
        if made and not args.keep:
            os.remove(path)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
