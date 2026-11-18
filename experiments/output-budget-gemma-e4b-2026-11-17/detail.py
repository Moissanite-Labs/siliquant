#!/usr/bin/env python3
"""Figures the README quotes that summary.py does not print. Nothing is scored or rounded here.

Reads results.jsonl of this record and of the record model-output-gemma-e4b-2026-11-15.

    python3 experiments/output-budget-gemma-e4b-2026-11-17/detail.py > experiments/output-budget-gemma-e4b-2026-11-17/detail.out
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / "model-output-gemma-e4b-2026-11-15" / "results.jsonl"


def load(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("kl_mean") is not None and r["variant"] not in out:
                out[r["variant"]] = r
    return out


new = load(HERE / "results.jsonl")
old = load(FIRST)
KINDS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]


def kl(v, table=None):
    return (table or new)[v]["kl_mean"]


def weights(kind):
    return new[f"X32:16@{kind}"]["weights_changed"]


print("1. Part A against arm S of the first record, one kind at a time, 16 levels")
print("   kind | weights | X32:16 KL | per weight x 1e9 | S:16 KL (first record) | per weight x 1e9")
px, ps = {}, {}
for k in KINDS:
    w = weights(k)
    px[k] = kl(f"X32:16@{k}") / w * 1e9
    ps[k] = kl(f"S:16@{k}", old) / w * 1e9
    print(f"   {k:12} | {w:10d} | {kl(f'X32:16@{k}'):.4f} | {px[k]:.4f} | {kl(f'S:16@{k}', old):.4f} | {ps[k]:.4f}")
tx = [kl(f"X32:16@{k}") for k in KINDS]
ts = [kl(f"S:16@{k}", old) for k in KINDS]
print(f"   X32: largest over smallest KL of a kind {max(tx) / min(tx):.2f}; per weight {max(px.values()) / min(px.values()):.1f}")
print(f"   S:   largest over smallest KL of a kind {max(ts) / min(ts):.1f}; per weight {max(ps.values()) / min(ps.values()):.1f}")
print(f"   ffn_down over ffn_gate, per weight: X32 {px['ffn_down'] / px['ffn_gate']:.2f}, S {ps['ffn_down'] / ps['ffn_gate']:.1f}")
print(f"   attn_output over attn_q, per weight: X32 {px['attn_output'] / px['attn_q']:.2f}, S {ps['attn_output'] / ps['attn_q']:.1f}")
steps = []
for k in KINDS:
    steps += [kl(f"X32:8@{k}") / kl(f"X32:16@{k}"), kl(f"X32:16@{k}") / kl(f"X32:32@{k}")]
gm = math.exp(sum(math.log(s) for s in steps) / len(steps))
print(f"   one more bit divides a kind's KL by {min(steps):.2f} to {max(steps):.2f}, geometric mean {gm:.2f}")
slope = kl("X32:16") / kl("X32:32", old)
print(f"   all kinds at once: X32:8 over X32:16 {kl('X32:8') / kl('X32:16'):.2f}; X32:16 over X32:32 (first record) {slope:.3f}")
unread = old["dead:kv"]["weights_zeroed"]
kv = weights("attn_k") + weights("attn_v")
print(f"   key and value weights never read: {unread} of {kv}, {100 * unread / kv:.1f} percent")

print("\n2. Gains at equal or nearly equal bits, and what they are worth in bits")
print("   A gain is turned into bits with the slope of X32 from 4.5 to 5.5 bits: one bit divides KL by "
      f"{slope:.3f}.")
base16, base8 = kl("X32:16"), kl("X32:8")


def worth(v, base):
    return math.log(base / kl(v)) / math.log(slope)


def line(label, v, base, base_bits):
    r = new[v]
    b = r.get("bits_per_weight_all", r.get("bits_per_weight"))
    print(f"   {label:34} | {b:.3f} bits | KL {kl(v):.4f} | {kl(v) / base:.3f} of the uniform arm"
          f" | {100 * (1 - kl(v) / base):.1f} percent lower | worth {worth(v, base):+.3f} bit"
          f" | bits against the uniform arm {b - base_bits:+.3f}")


mix = [v for v in new if v.startswith("X32mix:")]
best45 = "X32mix:attn_q=32,attn_k=64,attn_v=64,attn_output=46,ffn_gate=12,ffn_up=12,ffn_down=16"
rev45 = "X32mix:attn_q=46,attn_k=64,attn_v=64,attn_output=32,ffn_gate=16,ffn_up=12,ffn_down=12"
best35 = "X32mix:attn_q=8,attn_k=16,attn_v=32,attn_output=16,ffn_gate=6,ffn_up=8,ffn_down=8"
rev35 = "X32mix:attn_q=16,attn_k=32,attn_v=16,attn_output=8,ffn_gate=8,ffn_up=8,ffn_down=6"
joined = [v for v in new if v.startswith("X32s8mix:") and "dead" not in v][0]
assert len(mix) == 4
line("best arm at 4.5 bits", best45, base16, 4.5)
line("reversed arm at 4.5 bits", rev45, base16, 4.5)
line("best arm at 3.5 bits", best35, base8, 3.5)
line("reversed arm at 3.5 bits", rev35, base8, 3.5)
line("X32o:16~3.45", "X32o:16~3.45", base16, 4.5)
line("XRo:16~2.5", "XRo:16~2.5", base16, 4.5)
line("X32s8:16", "X32s8:16", base16, 4.5)
line("X32s6:16", "X32s6:16", base16, 4.5)
line("X32s8:18", "X32s8:18", base16, 4.5)
line("X32s6:20", "X32s6:20", base16, 4.5)
line("joined arm", joined, base16, 4.5)
print(f"   best over reversed: {kl(best45) / kl(rev45):.3f} at 4.5 bits, {kl(best35) / kl(rev35):.3f} at 3.5 bits")
print(f"   reversed arms, predicted KL against the sum of Part A at the uniform level: "
      f"4.5 bits 0.100550 against {sum(kl(f'X32:16@{k}') for k in KINDS):.6f}; "
      f"3.5 bits 0.438241 against {sum(kl(f'X32:8@{k}') for k in KINDS):.6f}")
print("   bits of each kind in the best arm at 4.5 bits (log2 of the levels plus 0.5):")
for k, q in [("attn_q", 32), ("attn_k", 64), ("attn_v", 64), ("attn_output", 46), ("ffn_gate", 12), ("ffn_up", 12), ("ffn_down", 16)]:
    print(f"      {k:12} {q:3d} levels  {math.log2(q) + 0.5:.3f} bits")
tot = sum(weights(k) for k in KINDS)
small = weights("attn_k") + weights("attn_v")
big = weights("ffn_gate") + weights("ffn_up")
print(f"   attn_k and attn_v are {100 * small / tot:.2f} percent of the block weights; raising them from 16 to 64 levels costs "
      f"{2 * small / tot:.4f} bit over all weights")
print(f"   ffn_gate and ffn_up are {100 * big / tot:.2f} percent; lowering them from 16 to 12 levels saves "
      f"{(4 - math.log2(12)) * big / tot:.4f} bit over all weights")

print("\n3. Part C: kept entries against the plain arm at the same levels")
xr16, xr32 = kl("XR:16"), kl("XR:32", old)
for v, base, plain_bits in [("XRo:16~8", xr16, None), ("XRo:16~5", xr16, None), ("XRo:16~3.45", xr16, None), ("XRo:16~2.5", xr16, None),
                            ("XRo:32~8", xr32, None), ("XRo:32~5", xr32, None), ("XRo:32~3.45", xr32, None)]:
    r = new[v]
    q = 16 if ":16~" in v else 32
    plain = new["XR:16"]["bits_per_weight"] if q == 16 else old["XR:32"]["bits_per_weight"]
    extra = r["bits_per_weight"] - plain
    same_bits = base / slope ** extra
    print(f"   {v:13} | kept {100 * r['kept_entries'] / tot:.4f} percent | +{extra:.4f} bit | KL {kl(v):.4f} | {kl(v) / base:.3f} of XR"
          f" | the same bits spent on levels would give about {same_bits:.4f}")
print(f"   XR:16 here {xr16:.6f}; XR:32 from the first record {xr32:.6f}")
print(f"   XRo:16~2.5 at {new['XRo:16~2.5']['bits_per_weight']:.3f} bits: {kl('XRo:16~2.5') / base16:.3f} of X32:16 at 4.500 bits")

print("\n4. Part E against all 42 layers (first record)")
six = [0, 8, 16, 24, 32, 40]
c = sum(kl(f"clip:5@ffn_down#{i}") for i in six)
r_ = sum(kl(f"X32:16@ffn_down#{i}") for i in six)
call, rall = kl("clip:5@ffn_down", old), kl("X32:16@ffn_down")
print(f"   clip: six alone add up to {c:.6f}; all 42 at once {call:.6f}; ratio {call / c:.1f} for 7 times the layers")
print(f"   rounding: six alone add up to {r_:.6f}; all 42 at once {rall:.6f}; ratio {rall / r_:.2f} for 7 times the layers")

print("\n5. The joined arm against the first record's files")
j = new[joined]
jb = j.get("bits_per_weight_all", j.get("bits_per_weight"))
for v, name in [("file:Q4_0.gguf", "llama.cpp Q4_0"), ("file:Q4_K_M.gguf", "llama.cpp Q4_K_M"), ("VE832:16", "VE832:16")]:
    o = old[v]
    print(f"   {name:16} {o['bits_per_weight']:.3f} bits KL {o['kl_mean']:.4f} top token {o['same_top']:.2f} | joined arm {jb:.3f} bits KL {kl(joined):.4f}"
          f" top token {j['same_top']:.2f} | ratio {kl(joined) / o['kl_mean']:.3f}")

print("\n6. Part H: the clip on each layer alone")
layers = list(range(42))
hk = {i: kl(f"clip:5@ffn_down#{i}") for i in layers}
tot_h = sum(hk.values())
order = sorted(layers, key=hk.get, reverse=True)
print(f"   sum over 42 layers {tot_h:.6f}; layer 10 alone {hk[10]:.6f}, {100 * hk[10] / tot_h:.1f} percent of the sum;"
      f" the other 41 add up to {tot_h - hk[10]:.6f}")
print(f"   the next largest is layer {order[1]} at {hk[order[1]]:.6f}: layer 10 is {hk[10] / hk[order[1]]:.1f} times it")
print(f"   the 42 alone over all 42 at once (first record, {call:.6f}): {tot_h / call:.3f}")
err = {i: new[f"clip:5@ffn_down#{i}"]["weight_err_db_mean"] for i in layers}
by_err = sorted(layers, key=err.get, reverse=True)
print(f"   by the weight error of its clip, layer 10 is number {by_err.index(10) + 1} of 42 ({err[10]:.2f} dB);"
      f" the three largest are layers {by_err[0]}, {by_err[1]}, {by_err[2]}"
      f" ({err[by_err[0]]:.2f}, {err[by_err[1]]:.2f}, {err[by_err[2]]:.2f} dB), with KL"
      f" {hk[by_err[0]]:.6f}, {hk[by_err[1]]:.6f}, {hk[by_err[2]]:.6f}")

def _spearman():
    import importlib.util
    spec = importlib.util.spec_from_file_location("earlier_summary", FIRST.parent / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.spearman        # the rank correlation of the first record, as summary.py uses it


spearman = _spearman()
others = [i for i in layers if i != 10]
print(f"   rank correlation between a layer's KL and the weight error of its clip: all 42 layers "
      f"{spearman([hk[i] for i in layers], [err[i] for i in layers]):+.3f}; without layer 10 "
      f"{spearman([hk[i] for i in others], [err[i] for i in others]):+.3f}")

print("\n7. Part L: is layer 10 marked by its weights?")
L = json.loads((HERE / "large_by_layer.json").read_text())
rows_l = {r["layer"]: r for r in L["layers"]}
for key, label in [("n_beyond", "entries beyond 5 standard deviations"), ("max_sd", "largest entry in standard deviations of its row"),
                   ("change_over_size", "the clip's squared change over the squared size of the matrix")]:
    ranked = sorted(layers, key=lambda i: rows_l[i][key], reverse=True)
    top = ", ".join(f"layer {i} ({rows_l[i][key]:.4g})" for i in ranked[:3])
    print(f"   {label}: layer 10 is number {ranked.index(10) + 1} of 42 ({rows_l[10][key]:.4g}); the three largest are {top}")
print(f"   entries beyond 5 standard deviations in all 42 layers: {sum(rows_l[i]['n_beyond'] for i in layers)};"
      f" layer 10 has {100 * rows_l[10]['n_beyond'] / sum(rows_l[i]['n_beyond'] for i in layers):.1f} percent of them")

print("\n8. Part K: the closure in the budget")
dead = [v for v in new if v.endswith("+deadkv")][0]
b0 = new[joined].get("bits_per_weight_all", new[joined].get("bits_per_weight"))
b1 = new[dead].get("bits_per_weight_all", new[dead].get("bits_per_weight"))
print(f"   joined arm {b0:.4f} bits, KL {kl(joined):.6f}; with the unread keys and values at 0 bits {b1:.4f} bits, KL {kl(dead):.6f};"
      f" {b0 - b1:.4f} bit saved")

print("\n9. Part M: parts of layer 10, and one matrix under the other rule")
l10 = hk[10]
for tag, label in [("u1", "the top hidden unit"), ("u10", "the top ten hidden units"), ("not10", "all but the top ten hidden units"),
                   ("d10", "the top ten dimensions of the stream")]:
    v = f"clip:5@ffn_down#10:{tag}"
    print(f"   clip on {label}: KL {kl(v):.6f}, {100 * kl(v) / l10:.2f} percent of layer 10's {l10:.6f}")
rs, xr = kl("R+S:32", old), kl("XR:32", old)
v5, v6 = kl("R+S:32|ffn_down#10=XR:32"), kl("XR:32|ffn_down#10=R+S:32")
gap = rs - xr
print(f"   the two rules on all matrices: R+S:32 {rs:.6f}, XR:32 {xr:.6f}, gap {gap:.6f}")
print(f"   R+S:32 with layer 10's ffn_down under XR:32: {v5:.6f}; that closes {100 * (rs - v5) / gap:.1f} percent of the gap")
print(f"   XR:32 with layer 10's ffn_down under R+S:32: {v6:.6f}; that opens {100 * (v6 - xr) / gap:.1f} percent of the gap")
print(f"   layer 10's ffn_down is {new['clip:5@ffn_down#10']['weights_changed']} of {tot} block weights,"
      f" {100 * new['clip:5@ffn_down#10']['weights_changed'] / tot:.2f} percent, one matrix of 294")
