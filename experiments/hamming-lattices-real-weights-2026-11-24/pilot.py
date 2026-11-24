"""Pilot on a made-up Laplace matrix (512 x 4096, seed 0): bits, pooled weight error, small weights stored as zero, and build time of each arm. No model is read."""
import sys, time, importlib.util, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hl_run", H / "run.py"); R = importlib.util.module_from_spec(spec); sys.modules["hl_run"] = R; spec.loader.exec_module(R)
import torch; torch.set_num_threads(1)
rng = np.random.default_rng(0)
W = rng.laplace(0, 0.01, size=(512, 4096)).astype(np.float32)   # 2.1M weights, synthetic
for v in ("HBG128:b560s16", "A16BG128:b560s16", "A32BG128:b560s16", "A16BG128:b368s16", "ZBG128:b368s16", "HBG128:b368s16", "A32BG128:b368s16"):
    t = time.time(); Wh, info = R.round_any(v, "attn_q", W); dt = time.time() - t
    small = info["small"]["0.05"]
    print(f"{v:20s} {dt:6.1f}s  {dt/W.size*1e9:6.1f} ns/w  bits {info['bits']:.4f}  err {10*np.log10(info['rel_err']):7.3f} dB  small->0 {small[1]/small[0]:.3f}  shrunk {info['overload']:.4f}", flush=True)
