#!/usr/bin/env python3
"""The checks and readings of PREREG.md from the rows of this folder. Usage, from the root of a checkout: python3 summary.py > results.out"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "model-output-smollm2-2026-11-17" / "results.jsonl"
Z, E, X = "ZBG128:b560s16", "EBG128:b560s16", "X32:16"
AGREE = 0.4
SMALL, BIG = "noisepart(ZBG128:b560s16)@small#1", "noisepart(ZBG128:b560s16)@big#1"      # extension 1
PS, NS, PB = "only(ZBG128:b560s16)@small", "onlynoise(ZBG128:b560s16)@small#1", "only(ZBG128:b560s16)@big"   # extension 2


def db(x):
    return 10.0 * math.log10(x)


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["variant"]] = r
    return rows


def c_of(r):
    return r["kl_mean"] / 10.0 ** (r["weight_err_db_mean"] / 10.0)


def paired(a, b):
    """Mean of the 16 passages' log-ratios of KL, a over b, in dB, and its standard error."""
    d = 10.0 * np.log10(np.array(a["kl_by_passage"]) / np.array(b["kl_by_passage"]))
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def c_gap(a, b):
    """a's KL per unit of weight error over b's, in dB: (from the means, from the passages' ratios, standard error)."""
    shift = a["weight_err_db_mean"] - b["weight_err_db_mean"]
    m, se = paired(a, b)
    return db(c_of(a) / c_of(b)), m - shift, se


def diff(a, b):
    """a's mean KL less b's over the 16 passages, and its standard error."""
    d = np.array(a["kl_by_passage"]) - np.array(b["kl_by_passage"])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def main():
    rows = load(HERE / "results.jsonl")
    ref = json.loads((HERE / "reference.out").read_text())
    har = load(HARNESS)
    P = print
    P("Checks of the instrument")
    h_ppl = har[X]["ppl_base"]
    P(f"C1 | reference perplexity on 40 passages {ref['ppl']:.4f}, the harness's {h_ppl:.4f}: {100 * (ref['ppl'] / h_ppl - 1):+.2f} percent | {'holds' if abs(ref['ppl'] / h_ppl - 1) <= 0.01 else 'FAILS'}")
    base = [v for v in (X, Z, E) if v in rows]
    if base:
        gaps = [rows[v]["weight_err_db_mean"] - har[v]["weight_err_db_mean"] for v in base]
        P("C2 | weight error less the harness's, dB: " + ", ".join(f"{v} {g:+.4f}" for v, g in zip(base, gaps)) + f" | {'holds' if max(abs(g) for g in gaps) <= 0.01 else 'FAILS'}")
        rat = [rows[v]["kl_mean"] / har[v]["kl_mean"] for v in base]
        P("C3 | mean KL on 16 passages over the harness's on 40: " + ", ".join(f"{v} {rows[v]['kl_mean']:.4f} / {har[v]['kl_mean']:.4f} = {r:.3f}" for v, r in zip(base, rat))
          + f" | {'holds' if max(abs(r - 1) for r in rat) <= 0.2 else 'FAILS'}")
    if Z in rows and E in rows:
        g, gp, se = c_gap(rows[E], rows[Z])
        hg = db(c_of(har[E]) / c_of(har[Z]))
        P(f"C4 | KL per unit of weight error, E8 arm over integer arm: {g:+.2f} dB from the means, {gp:+.2f} from the passages (error {se:.2f}); the harness's {hg:+.2f} | {'holds' if g >= 0.5 else 'FAILS'}")
    P("")
    P("variant | mean KL | its standard error | median | weight error, dB | KL per unit of weight error | that over the integer arm's, dB | over the E8 arm's, dB | perplexity")
    for v, r in rows.items():
        cz = f"{db(c_of(r) / c_of(rows[Z])):+.2f}" if Z in rows else "-"
        ce = f"{db(c_of(r) / c_of(rows[E])):+.2f}" if E in rows else "-"
        P(f"{v} | {r['kl_mean']:.5f} | {r['kl_mean_se']:.5f} | {r['kl_median']:.5f} | {r['weight_err_db_mean']:.3f} | {c_of(r):.2f} | {cz} | {ce} | {r['ppl']:.3f}")
    P("")
    P(f"Pairs: the first arm's KL per unit of weight error over the second's, in dB, from the means | from the 16 passages' ratios | its standard error | agree (within {AGREE} dB)")
    pairs = [("noise(EBG128:b560s16)#1", E), ("noise(ZBG128:b560s16)#1", Z), ("noise(EBG128:b560s16)#1", "noise(ZBG128:b560s16)#1"),
             ("noise(EBG128:b560s16)#2", "noise(EBG128:b560s16)#1"), ("shuffle(EBG128:b560s16)#1", E), ("shuffle(ZBG128:b560s16)#1", Z),
             ("shuffle(EBG128:b560s16)#1", "noise(EBG128:b560s16)#1"), ("shuffle(ZBG128:b560s16)#1", "noise(ZBG128:b560s16)#1"),
             ("shuffle(EBG128:b560s16)#1", "shuffle(ZBG128:b560s16)#1"), ("scale(ZBG128:b560s16,-0.66)", Z), (E, Z), (X, Z),
             (SMALL, Z), (SMALL, "noise(ZBG128:b560s16)#1"), (BIG, Z), (BIG, "noise(ZBG128:b560s16)#1")]
    got = {}
    for a, b in pairs:
        if a in rows and b in rows:
            g, gp, se = c_gap(rows[a], rows[b])
            got[(a, b)] = g
            P(f"{a} over {b} | {g:+.2f} | {gp:+.2f} | {se:.2f} | {'yes' if abs(g) <= AGREE else 'no'}")
    P("")
    P("Mixed arms: mean KL less the integer arm's, over the 16 passages | its standard error | in errors")
    for v in ("mix(attn=E,ffn=Z)", "mix(attn=Z,ffn=E)", E):
        if v in rows and Z in rows:
            d, se = diff(rows[v], rows[Z])
            P(f"{v} | {d:+.5f} | {se:.5f} | {d / se:+.1f}")
    if all(v in rows for v in ("mix(attn=E,ffn=Z)", "mix(attn=Z,ffn=E)", E, Z)):
        add = rows["mix(attn=E,ffn=Z)"]["kl_mean"] + rows["mix(attn=Z,ffn=E)"]["kl_mean"] - rows[Z]["kl_mean"] - rows[E]["kl_mean"]
        P(f"the two mixed arms' KL together less the two ball arms' together: {add:+.5f} (zero if the kinds add)")
    P("")
    P("Readings, by the rules of PREREG.md")
    ne, nz = got.get(("noise(EBG128:b560s16)#1", E)), got.get(("noise(ZBG128:b560s16)#1", Z))
    if ne is not None and nz is not None:
        if abs(ne) <= AGREE and nz > AGREE:
            P("D1a: noise at E8's size agrees with the E8 arm, and noise at the integer arm's size is above the integer arm.")
        elif abs(nz) <= AGREE and ne < -AGREE:
            P("D1b: noise at the integer arm's size agrees with the integer arm, and noise at E8's size is below the E8 arm.")
        else:
            P(f"D1c: neither. Noise over its base arm: E8 {ne:+.2f} dB, integer {nz:+.2f} dB.")
    two = got.get(("noise(EBG128:b560s16)#2", "noise(EBG128:b560s16)#1"))
    if two is not None:
        P(f"Two draws of noise differ by {two:+.2f} dB: {'readings from one draw stand' if abs(two) <= AGREE else 'no reading that rests on one draw is made'}.")
    for lat, b in (("E8", E), ("integer", Z)):
        sb, sn = got.get((f"shuffle({b})#1", b)), got.get((f"shuffle({b})#1", f"noise({b})#1"))
        if sb is not None and sn is not None:
            if abs(sn) <= AGREE and abs(sb) > AGREE:
                P(f"D2 for the {lat} arm: shuffled, it agrees with noise and not with the arm. Which weight gets which error matters.")
            elif abs(sb) <= AGREE and abs(sn) > AGREE:
                P(f"D2' for the {lat} arm: shuffled, it agrees with the arm and not with noise. The values of the errors carry it.")
            else:
                P(f"For the {lat} arm, shuffled: {sb:+.2f} dB from the arm and {sn:+.2f} dB from noise. Neither D2 nor D2'.")
    sc = rows.get("scale(ZBG128:b560s16,-0.66)")
    if sc is not None and Z in rows:
        m = db(sc["kl_mean"] / rows[Z]["kl_mean"])
        P(f"L: the integer arm's error made 0.66 dB smaller has a KL {m:+.2f} dB from the integer arm's: {'holds' if abs(m + 0.66) <= 0.2 else 'FAILS'}.")

    if (SMALL, Z) in got and (BIG, Z) in got:
        N = "noise(ZBG128:b560s16)#1"
        sz, sn, bz, bn = got[(SMALL, Z)], got[(SMALL, N)], got[(BIG, Z)], got[(BIG, N)]
        P("")
        P("Extension 1, by the rules of PREREG-extension-1.md")
        if abs(bz) <= AGREE and abs(sn) <= AGREE:
            P("E1: noise on the larger weights agrees with the integer arm, and noise on the weights rounded to zero agrees with the noise arm.")
        elif abs(sz) <= AGREE and abs(bn) <= AGREE:
            P("E2: noise on the weights rounded to zero agrees with the integer arm, and noise on the larger weights agrees with the noise arm.")
        elif min(abs(sz), abs(sn)) > AGREE and min(abs(bz), abs(bn)) > AGREE:
            P("E3: both lie between, more than 0.4 dB from each end.")
        else:
            P("None of E1, E2, E3 as written.")
        P(f"Noise on the weights rounded to zero: {sz:+.2f} dB from the integer arm, {sn:+.2f} from the noise arm. Noise on the larger weights: {bz:+.2f} and {bn:+.2f}.")

    if all(v in rows for v in (PS, NS, PB)):
        N = "noise(ZBG128:b560s16)#1"
        P("")
        P("Extension 2, by the statements of PREREG-extension-2.md")
        ratio = rows[PS]["kl_mean"] / rows[NS]["kl_mean"]
        m, se = paired(rows[PS], rows[NS])
        P(f"F1 | the small weights set to zero, {rows[PS]['kl_mean']:.5f}, over noise of that size on them, {rows[NS]['kl_mean']:.5f}: {ratio:.3f} "
          f"({m:+.2f} dB from the passages, error {se:.2f}) | {'holds' if ratio < 1.0 / 3.0 else 'FAILS'}")
        both = rows[PS]["kl_mean"] + rows[PB]["kl_mean"]
        P(f"F2 | the two parts together, {rows[PS]['kl_mean']:.5f} + {rows[PB]['kl_mean']:.5f} = {both:.5f}, over the integer arm's {rows[Z]['kl_mean']:.5f}: {both / rows[Z]['kl_mean']:.3f} "
          f"| {'holds' if abs(both / rows[Z]['kl_mean'] - 1) <= 0.1 else 'FAILS'}")
        P(f"F3 | KL per unit of weight error: noise on the small weights alone {c_of(rows[NS]):.1f}, noise on every weight {c_of(rows[N]):.1f}: {c_of(rows[NS]) / c_of(rows[N]):.2f} times "
          f"| {'holds' if c_of(rows[NS]) >= 2 * c_of(rows[N]) else 'FAILS'}")
        P(f"Per unit of weight error: the small weights set to zero {c_of(rows[PS]):.1f}, the integer arm's error on the other weights {c_of(rows[PB]):.1f}, the integer arm {c_of(rows[Z]):.1f}.")


if __name__ == "__main__":
    main()
