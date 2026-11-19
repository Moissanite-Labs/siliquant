"""Read the float matrices of a GGUF model file with NumPy alone.

The harness reads GGUF files with llama.cpp's own package. This file reads the same matrices
without it, for programs that only look at weights: the header, the metadata, the list of
tensors, and the data of tensors stored as 16-bit or 32-bit floats. It writes nothing.

A matrix comes back in NumPy order, rows first, which is the reverse of the order GGUF lists
its sizes in, as harness/ggufedit.py returns it: for a weight matrix the rows are the outputs.
"""
from __future__ import annotations

import struct

import numpy as np

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_FLOATS = {0: np.dtype("<f4"), 1: np.dtype("<f2")}        # GGUF tensor types F32 and F16


class Reader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            (self.version,) = struct.unpack("<I", f.read(4))
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            self.meta = {}
            for _ in range(n_kv):
                key = self._string(f)
                (kind,) = struct.unpack("<I", f.read(4))
                self.meta[key] = self._value(f, kind)
            self.tensors = {}
            order = []
            for _ in range(n_tensors):
                name = self._string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
                kind, offset = struct.unpack("<IQ", f.read(12))
                self.tensors[name] = {"dims": dims, "type": kind, "offset": offset}
                order.append(name)
            align = int(self.meta.get("general.alignment", 32))
            self.data_start = (f.tell() + align - 1) // align * align
        self.names = order

    @staticmethod
    def _string(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")

    def _value(self, f, kind):
        if kind in _SCALARS:
            fmt = _SCALARS[kind]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if kind == 8:
            return self._string(f)
        if kind == 9:
            item, count = struct.unpack("<IQ", f.read(12))
            if item in _SCALARS:
                fmt = _SCALARS[item]
                size = struct.calcsize(fmt)
                f.seek(size * count, 1)                    # long arrays (the vocabulary's scores) are skipped, not kept
                return ("array", item, count)
            return [self._value(f, item) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {kind}")

    def matrix(self, name):
        """A tensor of two sizes as 32-bit floats in NumPy order. Raises for a type that is not a float."""
        t = self.tensors[name]
        if t["type"] not in _FLOATS or len(t["dims"]) != 2:
            raise ValueError(f"{name}: not a float matrix (type {t['type']}, sizes {t['dims']})")
        dtype = _FLOATS[t["type"]]
        cols, rows = (int(d) for d in t["dims"])
        data = np.memmap(self.path, dtype=dtype, mode="r", offset=self.data_start + t["offset"], shape=(rows, cols))
        return np.ascontiguousarray(data, dtype=np.float32)
