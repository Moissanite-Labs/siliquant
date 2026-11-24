#!/usr/bin/env python3
"""Tables and verdicts for the Hamming-code lattices and the two baselines, from results.jsonl, baselines.jsonl and the rows
of the earlier records (read through the summary of e8-hadamard-real-weights-2026-11-22, which is not changed).

Scores the predictions H1 to H11, B1 to B4 and the check C1 of PREREG.md by the rules written there. A cell not yet
scored is reported as such. Every number is a field of one of those files or arithmetic on two of them.

Usage, from the root of a checkout:
  python3 experiments/hamming-lattices-real-weights-2026-11-24/summary.py > experiments/hamming-lattices-real-weights-2026-11-24/results.out
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eh_summary", HERE.parent / "e8-hadamard-real-weights-2026-11-22" / "summary.py")
EHS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EHS)

NAMES = [m[0] for m in EHS.MODELS]
FILE = {m[0]: m[1] for m in EHS.MODELS}
B45, B30 = "b560s16", "b368s16"
ARMS45 = (("ZBG", "integer levels"), ("EBG", "E8"), ("HBG", "E8 turned (A8)"), ("A16BG", "A16"), ("A32BG", "A32"))


def load():
    data = EHS.load()
    for name in NAMES:
        d = data[name]
        mine = EHS.rows(HERE / "results.jsonl", FILE[name])
        base = EHS.rows(HERE / "baselines.jsonl", FILE[name])
        d["check"] = mine.pop("A8BG128:b560s16", None)
        for v, r in mine.items():
            d["scored"][v] = (r, "this record")
            d["weights"][v] = r
        for v, r in base.items():
            d["scored"][v] = (r, "this record")
            d["weights"][v] = r
        if d["q"] is not None:
            d["scored"]["Q4_K_M"] = (d["q"], "an earlier record")
    return data


def get(data, name, v):
    x = data[name]["scored"].get(v)
    return x[0] if x else None


def kl(r):
    return f"{r['kl_mean']:.6f} ± {r['kl_mean_se']:.6f}" if r else "not scored"


def zero05(r):
    try:
        return r["small_weights"]["0.05"]["all"]["stored_zero_percent"]
    except (KeyError, TypeError):
        return None


def gap_or_none(a, b):
    return EHS.gap(a, b) if (a and b) else None


def verdict(ok):
    return "not yet decided" if ok is None else ("yes" if ok else "**no**")


def all_or_none(xs):
    if any(x is None for x in xs):
        return None
    return all(xs)


def main():
    data = load()
    out = []
    p = out.append

    # ------------------------------------------------------------------ check C1
    sm = data["SmolLM2-1.7B"]
    c, h = sm["check"], get(data, "SmolLM2-1.7B", "HBG128:b560s16")
    p("## Check C1: A8 decoded by this record's decoder against the committed E8-turned row (SmolLM2, 4.5 bits)\n")
    if c is None or h is None:
        c1 = None
        p("not yet scored\n")
    else:
        d, dw = c["kl_mean"] - h["kl_mean"], c["weight_err_db_pooled"] - h["weight_err_db_pooled"]
        c1 = abs(d) <= 0.1 * h["kl_mean_se"] and abs(dw) <= 0.002
        p(f"A8BG128:b560s16 {c['kl_mean']:.6f}, HBG128:b560s16 {h['kl_mean']:.6f}: difference {d:+.6f}, a tenth of a standard error "
          f"{0.1 * h['kl_mean_se']:.6f}. Weight error {c['weight_err_db_pooled']:.4f} against {h['weight_err_db_pooled']:.4f} dB, "
          f"difference {dw:+.4f}. Holds: {verdict(c1)}\n")

    # ------------------------------------------------------------------ 4.5 bits
    p("## Mean KL at 4.5 bits, no kept columns\n")
    p("| Model | " + " | ".join(lab for _, lab in ARMS45) + " | `Q4_K_M` (committed, 4.8 bits) | `Q4_K_M@16` | `NF4B64` (4.5 bits) |")
    p("|---|" + "---|" * (len(ARMS45) + 3))
    for name in NAMES:
        cells = [kl(get(data, name, f"{a}128:{B45}")) for a, _ in ARMS45]
        cells += [kl(get(data, name, "Q4_K_M")), kl(get(data, name, "Q4_K_M@16")), kl(get(data, name, "NF4B64"))]
        p(f"| {name} | " + " | ".join(cells) + " |")
    p("")
    p("## Weights at 4.5 bits: bits a weight, pooled weight error (dB), weights under 0.05 of the RMS stored as zero (%)\n")
    p("| Model | Arm | Bits | Error | Small weights at zero | Build (s) |")
    p("|---|---|---|---|---|---|")
    for name in NAMES:
        for a, lab in ARMS45:
            r = data[name]["weights"].get(f"{a}128:{B45}")
            if r:
                z = zero05(r)
                p(f"| {name} | {lab} | {r['bits_per_weight']:.4f} | {r['weight_err_db_pooled']:.3f} | {'' if z is None else f'{z:.1f}'} | {r.get('build_seconds', '')} |")
        for v in ("Q4_K_M@16", "NF4B64"):
            r = data[name]["weights"].get(v)
            if r:
                z = zero05(r)
                p(f"| {name} | {v} | {r['bits_per_weight']:.4f} | {r['weight_err_db_pooled']:.3f} | {'' if z is None else f'{z:.1f}'} | {r.get('build_seconds', '')} |")
    p("")

    # ------------------------------------------------------------------ 3.0 bits
    p("## Mean KL at 3.0 bits\n")
    p("| Model | " + " | ".join(lab for _, lab in ARMS45) + " |")
    p("|---|" + "---|" * len(ARMS45))
    for name in ("Qwen2.5-1.5B", "SmolLM2-1.7B"):
        p(f"| {name} | " + " | ".join(kl(get(data, name, f"{a}128:{B30}")) for a, _ in ARMS45) + " |")
    p("")
    p("| Model | Arm | Error (dB) | Small weights at zero (%) |")
    p("|---|---|---|---|")
    for name in ("Qwen2.5-1.5B", "SmolLM2-1.7B"):
        for a, lab in ARMS45:
            r = data[name]["weights"].get(f"{a}128:{B30}")
            if r:
                z = zero05(r)
                p(f"| {name} | {lab} | {r['weight_err_db_pooled']:.3f} | {'' if z is None else f'{z:.1f}'} |")
    p("")

    # ------------------------------------------------------------------ kept columns
    p("## Llama-3.2 with ten columns of each ffn_down kept, 4.5 bits\n")
    for a, lab in ARMS45:
        r = get(data, "Llama-3.2-1B", f"{a}128:{B45}+K10")
        if r:
            p(f"- {lab}: {kl(r)}")
    p("")

    # ------------------------------------------------------------------ gaps
    p("## Gaps, in errors (positive: the first is lower)\n")
    for name in NAMES:
        for x in ("A16BG", "A32BG"):
            for y in ("HBG", "ZBG", "EBG"):
                for b in (B45, B30):
                    g = gap_or_none(get(data, name, f"{x}128:{b}"), get(data, name, f"{y}128:{b}"))
                    if g is not None:
                        p(f"- {name}, {b[1:4]} bits/128: {x} against {y}: {g:+.1f}")
            for y in ("Q4_K_M", "Q4_K_M@16", "NF4B64"):
                g = gap_or_none(get(data, name, f"{x}128:{B45}"), get(data, name, y))
                if g is not None:
                    p(f"- {name}, 4.5 bits: {x} against {y}: {g:+.1f}")
        for y in ("Q4_K_M", "Q4_K_M@16", "NF4B64"):
            g = gap_or_none(get(data, name, f"HBG128:{B45}"), get(data, name, y))
            if g is not None:
                p(f"- {name}, 4.5 bits: HBG against {y}: {g:+.1f}")
        g = gap_or_none(get(data, name, "Q4_K_M@16"), get(data, name, "Q4_K_M"))
        if g is not None:
            p(f"- {name}: Q4_K_M@16 against Q4_K_M: {g:+.1f}")
        g = gap_or_none(get(data, name, f"A16BG128:{B45}+K10"), get(data, name, f"HBG128:{B45}+K10"))
        if g is not None:
            p(f"- {name}, 4.5 bits, kept columns: A16 against HBG: {g:+.1f}")
    p("")

    # ------------------------------------------------------------------ predictions
    def G(name, x, y):
        return gap_or_none(get(data, name, x), get(data, name, y))

    def notabove(g):
        return None if g is None else g > -2.0

    def under(g):
        return None if g is None else g >= 2.0

    v45 = lambda a: f"{a}128:{B45}"
    v30 = lambda a: f"{a}128:{B30}"
    rows = []
    rows.append(("H1", "4.5: A16 not above HBG by 2, three models", all_or_none([notabove(G(n, v45("A16BG"), v45("HBG"))) for n in NAMES]),
                 ", ".join(f"{G(n, v45('A16BG'), v45('HBG')):+.1f}" if G(n, v45("A16BG"), v45("HBG")) is not None else "-" for n in NAMES)))
    rows.append(("H2", "4.5: A32 not above HBG by 2, three models", all_or_none([notabove(G(n, v45("A32BG"), v45("HBG"))) for n in NAMES]),
                 ", ".join(f"{G(n, v45('A32BG'), v45('HBG')):+.1f}" if G(n, v45("A32BG"), v45("HBG")) is not None else "-" for n in NAMES)))
    g3 = G("SmolLM2-1.7B", v45("A16BG"), v45("HBG"))
    rows.append(("H3", "4.5, SmolLM2: A16 under HBG by 2", under(g3), "-" if g3 is None else f"{g3:+.1f}"))
    lowest = []
    for n in NAMES:
        rs = {a: get(data, n, v45(a)) for a, _ in ARMS45}
        if any(r is None for r in rs.values()):
            lowest.append(None)
        else:
            lowest.append(min(rs, key=lambda a: rs[a]["kl_mean"]))
    h4 = None if any(x is None for x in lowest) else sum(x in ("A16BG", "A32BG") for x in lowest) >= 2
    rows.append(("H4", "4.5: lowest of five is A16 or A32 on two models of three", h4, ", ".join(str(x) for x in lowest)))
    h5, h5s = [], []
    for n in NAMES:
        a, b, hh = (data[n]["weights"].get(v45(x)) for x in ("A16BG", "A32BG", "HBG"))
        if a and b and hh:
            da, db = hh["weight_err_db_pooled"] - a["weight_err_db_pooled"], hh["weight_err_db_pooled"] - b["weight_err_db_pooled"]
            h5.append(da >= 0.05 and db >= 0.02)
            h5s.append(f"{da:.3f}/{db:.3f}")
        else:
            h5.append(None)
            h5s.append("-")
    rows.append(("H5", "4.5: weight error A16 under HBG by 0.05 dB, A32 by 0.02", all_or_none(h5), ", ".join(h5s)))
    h6, h6s = [], []
    for n in NAMES:
        a, b, hh = (zero05(data[n]["weights"].get(v45(x))) for x in ("A16BG", "A32BG", "HBG"))
        if None not in (a, b, hh):
            h6.append(a >= hh + 5.0 and b >= a)
            h6s.append(f"{hh:.1f}/{a:.1f}/{b:.1f}")
        else:
            h6.append(None)
            h6s.append("-")
    rows.append(("H6", "4.5: small weights at zero, A16 >= HBG + 5 points, A32 >= A16", all_or_none(h6), ", ".join(h6s)))
    g7 = G("Llama-3.2-1B", v45("A16BG") + "+K10", v45("HBG") + "+K10")
    rows.append(("H7", "Llama-3.2 with columns kept: A16 not above HBG by 2", notabove(g7), "-" if g7 is None else f"{g7:+.1f}"))
    g8 = [G("SmolLM2-1.7B", v30(x), v30("HBG")) for x in ("A16BG", "A32BG")]
    rows.append(("H8", "3.0, SmolLM2: A16 and A32 under HBG by 2", all_or_none([under(g) for g in g8]),
                 ", ".join("-" if g is None else f"{g:+.1f}" for g in g8)))
    g9 = G("SmolLM2-1.7B", v30("A32BG"), v30("ZBG"))
    rows.append(("H9", "3.0, SmolLM2: A32 not above ZBG by 2", notabove(g9), "-" if g9 is None else f"{g9:+.1f}"))
    g10 = G("Qwen2.5-1.5B", v30("A16BG"), v30("EBG"))
    rows.append(("H10", "3.0, Qwen2.5: A16 not above EBG by 2", notabove(g10), "-" if g10 is None else f"{g10:+.1f}"))
    h11, h11s = [], []
    for n in ("Qwen2.5-1.5B", "SmolLM2-1.7B"):
        z, hh = data[n]["weights"].get(v30("ZBG")), data[n]["weights"].get(v30("HBG"))
        for x in ("A16BG", "A32BG"):
            a = data[n]["weights"].get(v30(x))
            if a and z and hh:
                dz, dh = z["weight_err_db_pooled"] - a["weight_err_db_pooled"], hh["weight_err_db_pooled"] - a["weight_err_db_pooled"]
                h11.append(dz >= 0.4 and dh >= 0.03)
                h11s.append(f"{x[:-2]} {dz:.3f}/{dh:.3f}")
            else:
                h11.append(None)
                h11s.append("-")
    rows.append(("H11", "3.0: weight error A16, A32 under ZBG by 0.4 dB and HBG by 0.03", all_or_none(h11), ", ".join(h11s)))
    TWO = ("Qwen2.5-1.5B", "SmolLM2-1.7B")
    b1 = [G(n, "Q4_K_M@16", "Q4_K_M") for n in TWO]
    rows.append(("B1", "Q4_K_M@16 under the committed Q4_K_M row, Qwen2.5 and SmolLM2",
                 all_or_none([None if g is None else g > 0 for g in b1]), ", ".join("-" if g is None else f"{g:+.1f}" for g in b1)))
    b2 = [G(n, v45("HBG"), "Q4_K_M@16") for n in TWO]
    rows.append(("B2", "HBG under Q4_K_M@16 by 2", all_or_none([under(g) for g in b2]), ", ".join("-" if g is None else f"{g:+.1f}" for g in b2)))
    b3 = [G(n, v45("HBG"), "NF4B64") for n in TWO]
    rows.append(("B3", "NF4B64 above HBG by 10", all_or_none([None if g is None else g >= 10.0 for g in b3]), ", ".join("-" if g is None else f"{g:+.1f}" for g in b3)))
    b4 = [G(n, "Q4_K_M@16", "NF4B64") for n in TWO]
    rows.append(("B4", "NF4B64 above Q4_K_M@16", all_or_none([None if g is None else g > 0 for g in b4]), ", ".join("-" if g is None else f"{g:+.1f}" for g in b4)))

    p("## The predictions\n")
    p("| | Prediction | Result | Holds |")
    p("|---|---|---|---|")
    p(f"| C1 | A8 reproduces the committed E8-turned row | see above | {verdict(c1)} |")
    for k, text, ok, res in rows:
        p(f"| {k} | {text} | {res} | {verdict(ok)} |")
    held = sum(1 for _, _, ok, _ in rows if ok)
    decided = sum(1 for _, _, ok, _ in rows if ok is not None)
    p(f"\n{held} of {decided} decided predictions hold; {len(rows) - decided} not yet decided.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
