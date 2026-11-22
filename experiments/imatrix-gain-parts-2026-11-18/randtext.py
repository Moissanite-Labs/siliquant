#!/usr/bin/env python3
"""A text with no meaning: pieces of a model's own vocabulary drawn at random, each with the same chance.

For an importance matrix that reads no text anyone wrote. Only ordinary pieces are drawn: control, unused and
user-defined tokens are left out. The vocabulary is read from the model file. The pieces are byte-level (the
vocabulary is of llama.cpp's "gpt2" kind), so they are joined as bytes and what is not valid UTF-8 is dropped.

Usage: GGUF_PY=... python3 experiments/imatrix-gain-parts-2026-11-18/randtext.py --model M.gguf --pieces 80000 --seed 1 --out rand.txt
Prints the number of pieces, the size of the text and its SHA-256.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", ""))
import gguf  # noqa: E402


def byte_table():
    """The table of the GPT-2 vocabulary: the character that stands for each byte, turned round."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pieces", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rd = gguf.GGUFReader(args.model)
    kind = rd.fields["tokenizer.ggml.model"]
    if bytes(kind.parts[kind.data[0]]).decode() != "gpt2":
        raise SystemExit("FAIL: the vocabulary is not of the gpt2 kind")
    tok = rd.fields["tokenizer.ggml.tokens"]
    pieces = [bytes(tok.parts[i]).decode("utf-8") for i in tok.data]
    typ = rd.fields["tokenizer.ggml.token_type"]
    types = [int(typ.parts[i][0]) for i in typ.data]
    table = byte_table()
    ordinary = [i for i, (p, t) in enumerate(zip(pieces, types)) if t == 1 and all(ch in table for ch in p)]
    rng = np.random.default_rng(args.seed)
    draw = rng.integers(0, len(ordinary), size=args.pieces)
    raw = b"".join(bytes(table[ch] for ch in pieces[ordinary[i]]) for i in draw)
    text = raw.decode("utf-8", errors="ignore")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    data = text.encode("utf-8")
    print(f"vocabulary {len(pieces)}, ordinary pieces {len(ordinary)}, pieces drawn {args.pieces}, seed {args.seed}, "
          f"bytes {len(data)}, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
