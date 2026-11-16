"""Score a model file against saved reference outputs with llama.cpp's own tool.

`llama-perplexity --kl-divergence` reads the tokens and the reference
model's log-probabilities from the file written by `--kl-divergence-base`,
runs the given model on the same tokens and prints how far its predictions
moved. This module runs it and reads the numbers off its output.
"""
from __future__ import annotations

import re
import struct
import subprocess

import numpy as np

FIELDS = {
    "ppl": r"Mean PPL\(Q\)\s*:\s*([-\d.eE+]+)\s*±\s*([-\d.eE+]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([-\d.eE+]+)\s*±\s*([-\d.eE+]+)",
    "ln_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-\d.eE+]+)\s*±\s*([-\d.eE+]+)",
    "kl_mean": r"Mean\s+KLD:\s*([-\d.eE+]+)\s*±\s*([-\d.eE+]+)",
    "kl_median": r"Median\s+KLD:\s*([-\d.eE+]+)",
    "kl_p99": r"99\.0%\s+KLD:\s*([-\d.eE+]+)",
    "kl_max": r"Maximum KLD:\s*([-\d.eE+]+)",
    "rms_dp": r"RMS Δp\s*:\s*([-\d.eE+]+)\s*±\s*([-\d.eE+]+)",
    "same_top": r"Same top p:\s*([-\d.eE+]+)\s*±\s*([-\d.eE+]+)",
}


def score(model_path, kld_path, binary="llama-perplexity", extra=("-ngl", "999")):
    """Run the tool. Returns a dict of the printed statistics, and the raw text."""
    out = subprocess.run([binary, "-m", model_path, "--kl-divergence-base", kld_path, "--kl-divergence", *extra],
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    row = {}
    for key, pat in FIELDS.items():
        m = re.search(pat, text)
        if m:
            row[key] = float(m.group(1))
            if m.lastindex and m.lastindex > 1:
                row[key + "_se"] = float(m.group(2))
    if "kl_mean" not in row:
        raise RuntimeError("no KL divergence in the tool's output:\n" + text[-3000:])
    return row, text


def tokens_of(kld_path):
    """The evaluation tokens stored at the head of a reference file: (n_ctx, n_vocab, n_chunk, tokens)."""
    with open(kld_path, "rb") as f:
        if f.read(8) != b"_logits_":
            raise ValueError("not a reference file written by --kl-divergence-base")
        n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
        toks = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype="<i4")
    return n_ctx, n_vocab, n_chunk, toks
