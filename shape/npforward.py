"""Run a language model of llama.cpp's "llama" kind from its GGUF file with NumPy alone, and score a change of its weights.

The harness scores a model with llama.cpp's own tool. This file does the same arithmetic in
NumPy for one architecture, so that a program which only has the weights can ask what a change
of them does to the output: the model's tokenizer (byte-pair merges, as GPT-2), its blocks
(RMS norm, attention with rotary positions, a gated feed-forward), and the mean KL divergence
of the changed model's next-token distribution from the reference's, at the positions
llama-perplexity scores: the second half of each passage.

It is slow and exact enough, not fast: 32-bit floats, one passage at a time through attention.
It handles what SmolLM2 has: as many key heads as heads, no biases, the output matrix tied to
the token embedding. It refuses a file that has anything else.

  m = Model("SmolLM2-1.7B-Instruct-f16.gguf")
  ids = m.tokenize(open("wiki.test.raw", encoding="utf-8").read())
  ref = m.log_probs(passages(ids, 512, 8))                  # reference, kept
  kl = kl_rows(ref, m.log_probs(passages(ids, 512, 8), {"blk.0.attn_q.weight": changed}))
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggufread  # noqa: E402

KINDS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
_DIGIT = re.compile(r"(\d)")
_PIECE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def _byte_names():
    """GPT-2's printable name for every byte."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    names, extra = {}, 0
    for b in range(256):
        if b in keep:
            names[b] = chr(b)
        else:
            names[b] = chr(256 + extra)
            extra += 1
    return names


class Tokenizer:
    def __init__(self, tokens, merges):
        self.ids = {t: i for i, t in enumerate(tokens)}
        self.rank = {tuple(m.split(" ")): i for i, m in enumerate(merges)}
        self.names = _byte_names()
        self.cache = {}

    def _merge(self, word):
        parts = list(word)
        while len(parts) > 1:
            best, at = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            a, b = parts[at], parts[at + 1]
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            parts = out
        return [self.ids[p] for p in parts]

    def encode(self, text):
        out = []
        for chunk in _DIGIT.split(text):
            pieces = [chunk] if len(chunk) == 1 and chunk.isdigit() else _PIECE.findall(chunk)
            for piece in pieces:
                got = self.cache.get(piece)
                if got is None:
                    got = self._merge("".join(self.names[b] for b in piece.encode("utf-8")))
                    self.cache[piece] = got
                out.extend(got)
        return out


def passages(ids, length, count):
    """The first count passages of length tokens, as llama-perplexity cuts them: (count, length)."""
    if len(ids) < length * count:
        raise ValueError("the text is too short")
    return np.asarray(ids[:length * count], dtype=np.int64).reshape(count, length)


def scored(length):
    """The positions of a passage that are scored: its second half, less the last, which has no next token."""
    return np.arange(length // 2, length - 1)


def kl_rows(ref, got):
    """KL divergence of got from ref at every scored position, from two arrays of log-probabilities, in 64-bit floats."""
    r = ref.astype(np.float64)
    return (np.exp(r) * (r - got.astype(np.float64))).sum(axis=1)


class Model:
    def __init__(self, path):
        self.rd = rd = ggufread.Reader(path)
        meta = rd.meta
        if meta.get("general.architecture") != "llama":
            raise ValueError("only the 'llama' kind is handled")
        self.layers = int(meta["llama.block_count"])
        self.width = int(meta["llama.embedding_length"])
        self.heads = int(meta["llama.attention.head_count"])
        if int(meta.get("llama.attention.head_count_kv", self.heads)) != self.heads:
            raise ValueError("fewer key heads than heads is not handled")
        self.eps = float(meta["llama.attention.layer_norm_rms_epsilon"])
        self.base = float(meta.get("llama.rope.freq_base", 10000.0))
        if "output.weight" in rd.tensors or any(n.endswith(".bias") for n in rd.names):
            raise ValueError("an output matrix of its own, or biases, are not handled")
        self.tok = Tokenizer(meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.merges"])
        self.add_bos = bool(meta.get("tokenizer.ggml.add_bos_token", False))
        self.embed = rd.matrix("token_embd.weight")
        self._vec = {}

    def tokenize(self, text):
        return self.tok.encode(text)

    def names(self):
        """The block matrices, in the order of the file."""
        return [n for n in self.rd.names if re.match(r"^blk\.\d+\.(" + "|".join(KINDS) + r")\.weight$", n)]

    def matrix(self, name, changed=None):
        if changed is not None and name in changed:
            return np.asarray(changed[name], dtype=np.float32)
        return self.rd.matrix(name)

    def _vector(self, name):
        if name not in self._vec:
            t = self.rd.tensors[name]
            dt = ggufread._FLOATS[t["type"]]
            self._vec[name] = np.asarray(np.memmap(self.rd.path, dtype=dt, mode="r", offset=self.rd.data_start + t["offset"],
                                                   shape=(int(t["dims"][0]),)), dtype=np.float32)
        return self._vec[name]

    def _norm(self, x, name):
        return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(self.eps)) * self._vector(name)

    def _rotate(self, x, length):
        """Rotary positions, as llama.cpp turns them: the pairs are neighbours. x is (passages, length, heads, size)."""
        size = x.shape[-1]
        freq = self.base ** (-np.arange(0, size, 2, dtype=np.float64) / size)
        ang = np.arange(length, dtype=np.float64)[:, None] * freq[None, :]
        c, s = np.cos(ang).astype(np.float32)[None, :, None, :], np.sin(ang).astype(np.float32)[None, :, None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = a * c - b * s
        out[..., 1::2] = a * s + b * c
        return out

    def forward(self, ids, changed=None, tap=None):
        """The hidden state after the last block, (passages, length, width). tap(name, x) is called with the input of every block matrix."""
        count, length = ids.shape
        H, size = self.heads, self.width // self.heads
        h = self.embed[ids]                                              # (count, length, width)
        mask = np.triu(np.full((length, length), -np.inf, dtype=np.float32), k=1)
        for il in range(self.layers):
            p = f"blk.{il}."
            x = self._norm(h, p + "attn_norm.weight")
            flat = x.reshape(-1, self.width)
            if tap is not None:
                for k in ("attn_q", "attn_k", "attn_v"):
                    tap(p + k + ".weight", flat)
            q = (flat @ self.matrix(p + "attn_q.weight", changed).T).reshape(count, length, H, size)
            k = (flat @ self.matrix(p + "attn_k.weight", changed).T).reshape(count, length, H, size)
            v = (flat @ self.matrix(p + "attn_v.weight", changed).T).reshape(count, length, H, size)
            q, k = self._rotate(q, length), self._rotate(k, length)
            mixed = np.empty((count, length, H, size), dtype=np.float32)
            for c in range(count):
                s = np.einsum("thd,uhd->htu", q[c], k[c], optimize=True) / np.float32(math.sqrt(size)) + mask[None]
                s -= s.max(axis=-1, keepdims=True)
                np.exp(s, out=s)
                s /= s.sum(axis=-1, keepdims=True)
                mixed[c] = np.einsum("htu,uhd->thd", s, v[c], optimize=True)
            flat = mixed.reshape(-1, self.width)
            if tap is not None:
                tap(p + "attn_output.weight", flat)
            h = h + (flat @ self.matrix(p + "attn_output.weight", changed).T).reshape(count, length, self.width)
            x = self._norm(h, p + "ffn_norm.weight")
            flat = x.reshape(-1, self.width)
            if tap is not None:
                tap(p + "ffn_gate.weight", flat)
                tap(p + "ffn_up.weight", flat)
            g = flat @ self.matrix(p + "ffn_gate.weight", changed).T
            g *= 1.0 / (1.0 + np.exp(-g))                                # x * sigmoid(x)
            g *= flat @ self.matrix(p + "ffn_up.weight", changed).T
            if tap is not None:
                tap(p + "ffn_down.weight", g)
            h = h + (g @ self.matrix(p + "ffn_down.weight", changed).T).reshape(count, length, self.width)
        return h

    def log_probs(self, ids, changed=None, tap=None, batch=4):
        """Log-probabilities of the next token at every scored position, (passages * scored, vocabulary), 32-bit."""
        length = ids.shape[1]
        at = scored(length)
        out = []
        for i in range(0, len(ids), batch):
            h = self.forward(ids[i:i + batch], changed, tap)[:, at, :]
            x = self._norm(h, "output_norm.weight").reshape(-1, self.width)
            z = (x @ self.embed.T).astype(np.float64)
            z -= z.max(axis=1, keepdims=True)
            z -= np.log(np.exp(z).sum(axis=1, keepdims=True))
            out.append(z.astype(np.float32))
        return np.concatenate(out)

    def perplexity(self, ids, lp):
        """Perplexity over the scored positions, from the log-probabilities."""
        at = scored(ids.shape[1])
        nxt = ids[:, at + 1].reshape(-1)
        return float(np.exp(-lp[np.arange(len(nxt)), nxt].astype(np.float64).mean()))
