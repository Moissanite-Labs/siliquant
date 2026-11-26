#!/usr/bin/env python3
"""Priority-3 packed-format benchmark from the 2026-11-24 M1 Max session (see
notes/experimenter-m1-ccf045d-2026-11-24.md): encodes sampled block matrices of a GGUF with
shape/ballpack.Format(32,136,16) and shape/hampack.HamFormat(16,32,136,16) from
bayan/charming-gates-jlas14, checks readback bit-for-bit, and times the C decoders on one core.

Usage (from a checkout that has both shape/ballpack.py and shape/hampack.py, i.e. the
bayan/charming-gates-jlas14 branch or a branch off it):
  python3 notes/bench_hampack_ballpack_m1_2026-11-24.py --model <MODEL.gguf> [--sample 24]
"""
import argparse, re, resource, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shape"))
sys.path.insert(0, str(HERE.parent / "harness"))
import ballpack, hampack          # noqa: E402
import ggufedit                   # noqa: E402

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
BLOCK = re.compile(r"^blk\.(\d+)\.(" + "|".join(KINDS) + r")\.weight$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", type=int, default=24)
    args = ap.parse_args()
    rd = ggufedit.open_model(args.model)
    names = [t.name for t in rd.tensors if BLOCK.match(t.name)]
    step = max(1, len(names) // args.sample)
    sample = names[::step][:args.sample]
    mats = {n: ggufedit.tensor_float(next(t for t in rd.tensors if t.name == n)).reshape(
        [int(d) for d in next(t for t in rd.tensors if t.name == n).shape]) for n in sample}
    print(f"sampled {len(sample)} of {len(names)} matrices")
    for label, fmt in (("ballpack Z^32 (136 bits)", ballpack.Format(32, 136, 16)),
                       ("hampack A16xA16 (136 bits)", hampack.HamFormat(16, 32, 136, 16))):
        t0 = time.time()
        blobs, recs, tops, ok = {}, [], [], True
        for n, W in mats.items():
            stored, Wh, _ = fmt.encode(W)
            blobs[n] = fmt.tensor_bytes(n, stored)
            recs.append(np.ascontiguousarray(stored["records"], dtype=np.uint8))
            tops.append(float(stored["top"]))
            ok &= bool(np.array_equal(fmt.decode(stored).astype(np.float16), Wh.astype(np.float16)))
        secs, _ = fmt.timing(np.concatenate(recs), max(tops), repeats=3)
        n_w = sum(int(np.prod(m.shape)) for m in mats.values())
        print(f"{label}: table={fmt._table.nbytes/1e6:.2f} MB bytes/w={sum(map(len, blobs.values()))/n_w:.4f} "
              f"readback_ok={ok} decode={secs/n_w*1e9:.1f} ns/w peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MiB "
              f"build={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
