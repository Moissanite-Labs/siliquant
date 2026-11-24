"""Does any figure of the weights alone predict llama.cpp's importance matrix? A check made during the
cross-org survey of 2026-11-24 (notes/cross-org-survey-2026-11-24.md). Exploratory: no plan was committed first.

For every block matrix, h_j = in_sum2 / counts from the importance matrix (the mean squared input of column j).
Figures that need no data:
  sig   : 1 / (column RMS of W)^2          (SINQ, arXiv 2509.22944: small columns sit where inputs are large)
  gam   : (RMSNorm gain)^2                   (only for matrices that read a norm: attn_q/k/v, ffn_gate/up)
  both  : gam * sig
and, for comparison, the matrix made from random tokens (rimatrix), which needs forward passes but no text.
Printed: the Spearman rank correlation of h with each, per kind of matrix, median over layers.

Usage: python3 notes/colproxy.py MODEL.gguf MODEL.imatrix.gguf MODEL.rimatrix.gguf
"""
import sys, numpy as np, gguf

def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r

def spear(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])

def load_imat(p):
    rd = gguf.GGUFReader(p)
    d = {t.name: np.asarray(t.data, dtype=np.float64).ravel() for t in rd.tensors}
    out = {}
    for k, v in d.items():
        if k.endswith(".in_sum2"):
            base = k[:-len(".in_sum2")]
            out[base[:-len(".weight")] if base.endswith(".weight") else base] = v / float(d[base + ".counts"][0])
    return out

model, imat, rimat = sys.argv[1:4]
rd = gguf.GGUFReader(model)
T = {t.name: t for t in rd.tensors}
H, R = load_imat(imat), load_imat(rimat)
NORM = {"attn_q": "attn_norm", "attn_k": "attn_norm", "attn_v": "attn_norm", "ffn_gate": "ffn_norm", "ffn_up": "ffn_norm"}
res = {}
for name, h in H.items():
    if name + ".weight" not in T:
        continue
    t = T[name + ".weight"]
    W = np.asarray(t.data, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != len(h):
        continue
    kind = name.split(".")[-1]
    layer = name.split(".")[1]
    colrms = np.sqrt((W ** 2).mean(axis=0)) + 1e-30
    sig = 1.0 / colrms ** 2
    row = {"sig": spear(h, sig), "noise": spear(h, R[name]) if name in R else np.nan}
    if kind in NORM:
        g = np.asarray(T[f"blk.{layer}.{NORM[kind]}.weight"].data, dtype=np.float64) ** 2
        row["gam"] = spear(h, g)
        row["both"] = spear(h, g * sig)
    res.setdefault(kind, []).append(row)
print(f"{'kind':10s} {'n':>3s} {'sig':>7s} {'gam':>7s} {'both':>7s} {'noise':>7s}   (median Spearman of log h with each, over layers)")
for kind, rows in sorted(res.items()):
    def med(k):
        v = [r[k] for r in rows if k in r and not np.isnan(r[k])]
        return f"{np.median(v):7.3f}" if v else "      -"
    print(f"{kind:10s} {len(rows):3d} {med('sig')} {med('gam')} {med('both')} {med('noise')}")
