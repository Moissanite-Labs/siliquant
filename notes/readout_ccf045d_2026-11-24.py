#!/usr/bin/env python3
"""Readout for the 2026-11-24 ccf045d Gemma closure + addendum arms (see
notes/experimenter-m1-ccf045d-2026-11-24.md). Prints per-row tables and the
PRED-M1-01/02, @16-confound, and A_16 out-of-family verdicts from raw results.jsonl.

Usage: python3 notes/readout_ccf045d_2026-11-24.py <results.jsonl> [more.jsonl ...]
"""
import json, math, sys

def load(paths):
    rows = []
    for p in paths:
        try:
            rows += [json.loads(l) for l in open(p) if l.strip()]
        except FileNotFoundError:
            pass
    return rows

def get(rows, v, t, model=None):
    for r in rows:
        if r['variant'] == v and r['text'] == t and (model is None or r['model_id'] == model):
            return r

def z(a, b):
    return (a['kl_mean'] - b['kl_mean']) / math.hypot(a['kl_mean_se'], b['kl_mean_se'])

def iso_parity(b_lo, k_lo, b_hi, k_hi, k_stock):
    return b_lo + (b_hi - b_lo) * math.log(k_lo / k_stock) / math.log(k_lo / k_hi)

def main():
    rows = load(sys.argv[1:])
    for m in sorted({r['model_id'] for r in rows}):
        print('=' * 20, m)
        for t in ('wikitext', 'python', 'bridge'):
            for v in sorted({r['variant'] for r in rows if r['model_id'] == m}):
                r = get(rows, v, t, m)
                if r and 'kl_mean' in r:
                    print(f"  {t:8s} {v:28s} bpw={r['bits_per_weight']:.4f} KL={r['kl_mean']:.6f}±{r['kl_mean_se']:.6f} "
                          f"ppl={r.get('ppl', float('nan')):.4f} top1={r.get('same_top', float('nan')):.3f} "
                          f"diff={r.get('other_tensors_that_differ', 'n/a')} backend={r.get('backend', '?')}")
        t = 'wikitext'
        stock = get(rows, 'file:Q4_K_M-imat', t, m)
        hebg560 = get(rows, 'hEBG128:b560s16', t, m)
        hebg496 = get(rows, 'hEBG128:b496s16', t, m)
        hzb560 = get(rows, 'hZBG128:b560s16', t, m)
        if stock and hebg560 and hebg496 and stock.get('other_tensors_that_differ') == []:
            b_stock, k_stock = stock['bits_per_weight'], stock['kl_mean']
            b560, b496 = hebg560['bits_per_weight'], hebg496['bits_per_weight']
            k560, k496 = hebg560['kl_mean'], hebg496['kl_mean']
            if k560 < k_stock < k496:
                bm = iso_parity(b496, k496, b560, k560, k_stock)
                print(f"  [bracket] b_match={bm:.4f} bpw, savings={(b_stock-bm)/b_stock*100:.2f}% (gate >12%: {'PASS' if (b_stock-bm)/b_stock > 0.12 else 'FAIL'})")
            elif k496 <= k_stock:
                print(f"  [below] savings >= {(b_stock-b496)/b_stock*100:.2f}% (gate PASS without interpolation)")
            else:
                print(f"  [above] savings <= {(b_stock-b560)/b_stock*100:.2f}% (PRED-M1-01 falsified)")
            if hzb560:
                zz = z(hzb560, hebg560)
                print(f"  [vs hZBG b560] dKL={hzb560['kl_mean']-hebg560['kl_mean']:.6f} = {zz:.2f} combined SE (gate >=2.0: {'PASS' if zz >= 2.0 else 'FAIL'})")

if __name__ == '__main__':
    main()
