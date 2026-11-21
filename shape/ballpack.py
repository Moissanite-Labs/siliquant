"""A packed format for integer levels in one ball per block of weights.

The records of 2026-11-17 scored arms that round every block of n weights to a point of Z^n
inside a ball, and counted their bits from the size of the codebook. No file held those bits.
This module stores them and reads them back.

The format, for a matrix of rows by cols with cols a multiple of n:

  for the matrix   the largest scale `top`, as a 16-bit float
                   optionally a weight for every column, as 16-bit floats
                   optionally some columns kept whole: their numbers, and their entries as 16-bit floats
  for every block  one byte `code`: the block's scale is top * 2^(-code / steps)
                   the number of the block's point among the points of the ball, in bits / 8 bytes

The number is the one shape/ballindex.py gives a point (enumerative coding). A weight is read
back as

  float16( y * top * 2^(-code / steps) / column weight )

in 64-bit arithmetic, with y the entry of the point. An entry of a kept column is read back as it
was stored, and takes no room in any block: it is zero there. encode() returns the matrix as
decode() will give it, so what a model is scored on can be what the file holds, bit for bit.

With n = 32, bits = 136 and one byte of scale a block takes 18 bytes: 4.5 bits a weight.

The scale has a floor. One byte of code and `steps` steps to a factor of two span 255 / steps
factors of two under `top`: with 16 steps, a factor of about 63,000. A block whose own scale is
smaller than that floor is rounded at the floor. It is rounded more coarsely than it need be,
never wrongly: what encode() returns is still what the file holds.

The arithmetic on numbers of 136 bits is in ballpack.c, compiled on first use. Its table of
counts is built here, by ballindex.py, in whole numbers. test_ballpack.py checks the C coder
against ballindex.py point by point.

Nothing here is a kernel. decode() gives back a matrix of floats, and timing() says how long
the numbers take to decode on one core.

Tests: python3 -m unittest shape/test_ballpack.py
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ballindex  # noqa: E402

MAGIC = b"SQBP1\0\0\0"
BISECT_STEPS = 14          # as in the records that scored the ball arms
SHRINK_STEPS = 12
LIMBS = 3                  # of 64 bits, as in ballpack.c


def _private(path):
    """Is this a directory or file that only this user can have written: theirs, and closed to group and others?"""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_uid == os.getuid() and not (st.st_mode & 0o022)


def _build_dir():
    """A directory only this user can write to, for the compiled coder.

    The library is loaded into the process, so it must not come from a place where another
    user of the machine could have put a file of the same name. A per-user cache directory
    is used if it is this user's and closed to others. If it is not, or cannot be made, a
    fresh directory of mode 0700 is made for this process.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    d = Path(base) / "siliquant"
    try:
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        d = None
    if d is None or not _private(d):
        d = Path(tempfile.mkdtemp(prefix="siliquant-ballpack-"))
    return d


def _library():
    """ballpack.c compiled to a shared library, in a directory of this user's alone, named by the hash of its source."""
    src = (HERE / "ballpack.c").read_bytes()
    path = _build_dir() / f"ballpack-{hashlib.sha256(src).hexdigest()[:16]}.so"
    if not (path.exists() and _private(path)):
        part = path.with_suffix(f".{os.getpid()}.part")
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(part), str(HERE / "ballpack.c"), "-lm"], check=True)
        os.chmod(part, 0o700)
        os.replace(part, path)
    lib = ctypes.CDLL(str(path))
    lib.zb_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.zb_init.restype = ctypes.c_int
    for f in (lib.zb_rank, lib.zb_unrank):
        f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        f.restype = ctypes.c_size_t
    lib.zb_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
    lib.zb_decode.restype = ctypes.c_size_t
    return lib


_live = None               # the Format whose table the C coder holds now


class Format:
    """Integer levels in one ball per block of n weights, a number of `bits` bits and a scale code of one byte.

    The C coder holds one table at a time. Any number of Formats may be alive in a process:
    each call into the coder first makes sure the table there is its own. Not for use from
    two threads at once.
    """

    def __init__(self, n=32, bits=136, steps=16):
        if bits % 8 or bits > 64 * LIMBS:
            raise ValueError(f"bits must be a multiple of 8 and at most {64 * LIMBS}")
        self.n, self.bits, self.steps = n, bits, steps
        self.bytes = bits // 8
        # A ball that holds 2^bits points has about this squared radius. Counted exactly to a little beyond it.
        r2 = (2.0 ** (bits / n) * math.exp(math.lgamma(n / 2.0 + 1.0) / n) / math.sqrt(math.pi)) ** 2
        limit = int(1.08 * r2) + 2 * n
        while True:
            index = ballindex.BallIndex("Z8", n, limit)
            if index.count > 1 << bits:
                break
            limit = int(limit * 1.3) + 8
        full = index._ways[n]
        self.bound = max(r for r in range(limit + 1) if full[r] <= 1 << bits)
        self.count = full[self.bound]
        if self.bound > 127 * 127:
            raise ValueError("an entry would not fit a byte")
        self.index = ballindex.BallIndex("Z8", n, self.bound)       # the reference coder, for tests
        if self.index.count != self.count:
            raise AssertionError("two counts of one codebook differ")
        table = np.zeros((n + 1, self.bound + 1, LIMBS), dtype=np.uint64)
        mask = (1 << 64) - 1
        for m in range(n + 1):
            for r in range(self.bound + 1):
                v = self.index._ways[m][r]
                for j in range(LIMBS):
                    table[m, r, j] = (v >> (64 * j)) & mask
        self._table = np.ascontiguousarray(table)
        self._lib = _library()
        self._use()

    def _use(self):
        """Make the C coder's one table this Format's. It copies the table, so another Format cannot leave it wrong."""
        global _live
        if _live is not self:
            if self._lib.zb_init(self.n, self.bound, self.bytes, self._table.ctypes.data):
                raise RuntimeError("the C coder did not take the table")
            _live = self

    # -- numbers ---------------------------------------------------------------------------------

    def rank(self, Y):
        """Y: (blocks, n) entries as int8. Returns (blocks, bytes) uint8, each number least byte first."""
        Y = np.ascontiguousarray(Y, dtype=np.int8)
        out = np.empty((len(Y), self.bytes), dtype=np.uint8)
        self._use()
        bad = self._lib.zb_rank(Y.ctypes.data, len(Y), out.ctypes.data)
        if bad:
            raise ValueError(f"{bad} blocks are not points of the codebook")
        return out

    def unrank(self, numbers):
        """numbers: (blocks, bytes) uint8. Returns (blocks, n) int8."""
        numbers = np.ascontiguousarray(numbers, dtype=np.uint8)
        Y = np.empty((len(numbers), self.n), dtype=np.int8)
        self._use()
        bad = self._lib.zb_unrank(numbers.ctypes.data, len(numbers), Y.ctypes.data)
        if bad:
            raise ValueError(f"{bad} numbers name no point of the codebook")
        return Y

    # -- a matrix --------------------------------------------------------------------------------

    def _scales(self, X):
        """For every block the smallest scale at which it rounds into the ball, by bisection."""
        g = np.maximum(np.sqrt((X ** 2).sum(axis=1)), 1e-30)
        r, margin = math.sqrt(self.bound), math.sqrt(self.n) / 2.0
        lo = g / (r + margin) * 0.99
        hi = g / max(r - margin, 0.05 * r) * 1.01
        for _ in range(BISECT_STEPS):
            mid = np.sqrt(lo * hi)
            ok = (np.rint(X / mid[:, None]) ** 2).sum(axis=1) <= self.bound
            hi = np.where(ok, mid, hi)
            lo = np.where(ok, lo, mid)
        return hi

    def values(self, Y, codes, top, colscale, rows, cols, keep=None, kept=None):
        """The matrix a file holds: the one rule for reading weights back, used by encode and by decode."""
        t = float(top) * 2.0 ** (-codes.astype(np.float64) / self.steps)
        Wh = (Y.astype(np.float64) * t[:, None]).reshape(rows, cols)
        if colscale is not None:
            Wh = Wh / colscale.astype(np.float64)[None, :]
        Wh = Wh.astype(np.float16)
        if keep is not None:
            Wh[:, keep.astype(np.int64)] = kept
        return Wh

    def encode(self, W, colscale=None, keep=None):
        """Round a matrix and pack it. Returns (what is stored, the matrix as decode will give it, share of blocks shrunk).

        With colscale every column is multiplied by its weight before rounding and divided by it
        after. The weights are stored as 16-bit floats, and it is the stored values that are used.

        With keep, the numbers of some columns in rising order, those columns are not rounded.
        Their entries are stored as 16-bit floats and read back as stored. In the blocks they are zero.
        """
        rows, cols = W.shape
        if cols % self.n:
            raise ValueError(f"a block of {self.n} does not tile a row of {cols}")
        X = W.astype(np.float64)
        c16 = None
        if colscale is not None:
            c16 = np.asarray(colscale).astype(np.float16)
            if not np.all(np.isfinite(c16)) or np.any(c16 <= 0):
                raise ValueError("a column weight is not a positive 16-bit float")
            X = X * c16.astype(np.float64)[None, :]
        k32, kept = None, None
        if keep is not None:
            k32 = np.asarray(keep)
            if k32.ndim != 1 or len(k32) == 0 or not np.issubdtype(k32.dtype, np.integer):
                raise ValueError("the kept columns are a list of whole numbers, and not an empty one")
            if k32.min() < 0 or k32.max() >= cols or np.any(np.diff(k32) <= 0):
                raise ValueError("a kept column is outside the matrix, or the columns are not in rising order")
            k32 = k32.astype(np.uint32)
            kept = np.ascontiguousarray(W[:, k32.astype(np.int64)]).astype(np.float16)
            if not np.all(np.isfinite(kept)):
                raise ValueError("an entry of a kept column does not fit a 16-bit float")
            X[:, k32.astype(np.int64)] = 0.0
        X = X.reshape(-1, self.n)
        t = self._scales(X)
        top = np.float16(t.max())
        if float(top) < float(t.max()):
            top = np.nextafter(top, np.float16(np.inf))
        if not np.isfinite(top):
            raise ValueError("the largest scale does not fit a 16-bit float")
        grid = lambda k: float(top) * 2.0 ** (-k / self.steps)
        k = np.clip(np.floor(np.log2(float(top) / t) * self.steps), 0, 255)
        low = (grid(k) < t) & (k > 0)                    # the grid's scale is never under the block's own
        k = np.where(low, k - 1, k)
        tq = grid(k)
        Y = np.rint(X / tq[:, None])
        out = (Y ** 2).sum(axis=1) > self.bound
        if np.any(out):                                  # a larger scale can still round outside: shrink the block
            xo = X[out] / tq[out, None]
            lo, hi = np.zeros(len(xo)), np.ones(len(xo))
            for _ in range(SHRINK_STEPS):
                mid = 0.5 * (lo + hi)
                good = (np.rint(xo * mid[:, None]) ** 2).sum(axis=1) <= self.bound
                lo = np.where(good, mid, lo)
                hi = np.where(good, hi, mid)
            Y[out] = np.rint(xo * lo[:, None])
        Yi = Y.astype(np.int8)
        codes = k.astype(np.uint8)
        records = np.concatenate([codes[:, None], self.rank(Yi)], axis=1)
        stored = {"rows": rows, "cols": cols, "top": top, "colscale": c16, "records": records, "keep": k32, "kept": kept}
        return stored, self.values(Yi, codes, top, c16, rows, cols, k32, kept), float(np.mean(out))

    def decode(self, stored):
        records = stored["records"]
        Y = self.unrank(records[:, 1:])
        return self.values(Y, records[:, 0], stored["top"], stored["colscale"], stored["rows"], stored["cols"], stored.get("keep"), stored.get("kept"))

    # -- a file ----------------------------------------------------------------------------------

    def header(self, tensors):
        return MAGIC + struct.pack("<5I", self.n, self.bits, self.bound, self.steps, tensors)

    def tensor_bytes(self, name, stored):
        """One matrix as it stands in a file."""
        raw = name.encode("utf-8")
        c, keep = stored["colscale"], stored.get("keep")
        flag = (0 if c is None else 1) | (0 if keep is None else 2)
        head = struct.pack("<H", len(raw)) + raw + struct.pack("<IIB", stored["rows"], stored["cols"], flag)
        head += np.asarray(stored["top"], dtype="<f2").tobytes()
        if c is not None:
            head += c.astype("<f2").tobytes()
        if keep is not None:                             # how many, which, and their entries row by row
            head += struct.pack("<I", len(keep)) + keep.astype("<u4").tobytes() + np.ascontiguousarray(stored["kept"]).astype("<f2").tobytes()
        return head + stored["records"].tobytes()

    def write(self, path, tensors):
        """tensors: a list of (name, stored). Returns the bytes written."""
        with open(path, "wb") as f:
            f.write(self.header(len(tensors)))
            for name, stored in tensors:
                f.write(self.tensor_bytes(name, stored))
            return f.tell()

    @staticmethod
    def read(path):
        """A file as (Format, list of (name, stored)). A file that is cut short or does not hold what its header says is refused with a ValueError."""
        with open(path, "rb") as f:
            blob = f.read()
        if blob[:8] != MAGIC:
            raise ValueError("not a packed file of this format")
        if len(blob) < 28:
            raise ValueError("the file ends inside its header")
        n, bits, bound, steps, count = struct.unpack_from("<5I", blob, 8)
        if n < 1 or n > 4096 or steps < 1:
            raise ValueError("the file's header names a block of no weights or of more than 4096, or a scale of no steps")
        fmt = Format(n, bits, steps)
        if fmt.bound != bound:
            raise ValueError("the file's bound is not the one its n and bits give")
        at, tensors = 28, []

        def need(size, what):
            if size < 0 or at + size > len(blob):
                raise ValueError(f"the file ends inside {what} of matrix {len(tensors) + 1} of {count}")

        for _ in range(count):
            need(2, "the name")
            (ln,) = struct.unpack_from("<H", blob, at)
            need(2 + ln, "the name")
            try:
                name = blob[at + 2: at + 2 + ln].decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"the name of matrix {len(tensors) + 1} is not text") from None
            at += 2 + ln
            need(9 + 2, "the header")
            rows, cols, flag = struct.unpack_from("<IIB", blob, at)
            at += 9
            if cols % n:
                raise ValueError(f"matrix {name}: a block of {n} does not tile a row of {cols}")
            top = np.frombuffer(blob, dtype="<f2", count=1, offset=at)[0]
            at += 2
            if flag & ~3:
                raise ValueError("a matrix of the file has a flag this reader does not know")
            c, keep, kept = None, None, None
            if flag & 1:
                need(2 * cols, "the column weights")
                c = np.frombuffer(blob, dtype="<f2", count=cols, offset=at).copy()
                at += 2 * cols
            if flag & 2:
                need(4, "the kept columns")
                (count_kept,) = struct.unpack_from("<I", blob, at)
                at += 4
                need(4 * count_kept, "the kept columns")
                keep = np.frombuffer(blob, dtype="<u4", count=count_kept, offset=at).copy()
                at += 4 * count_kept
                if count_kept == 0 or keep.max() >= cols or np.any(np.diff(keep.astype(np.int64)) <= 0):
                    raise ValueError("the kept columns of a matrix are not columns of it in rising order")
                need(2 * rows * count_kept, "the kept columns")
                kept = np.frombuffer(blob, dtype="<f2", count=rows * count_kept, offset=at).reshape(rows, count_kept).copy()
                at += 2 * rows * count_kept
            blocks, width = rows * cols // n, fmt.bytes + 1
            need(blocks * width, "the blocks")
            records = np.frombuffer(blob, dtype=np.uint8, count=blocks * width, offset=at).reshape(blocks, width).copy()
            at += blocks * width
            tensors.append((name, {"rows": rows, "cols": cols, "top": top, "colscale": c, "records": records, "keep": keep, "kept": kept}))
        if at != len(blob):
            raise ValueError("the file has bytes after its last matrix")
        return fmt, tensors

    # -- time ------------------------------------------------------------------------------------

    def timing(self, records, top=1.0, repeats=3):
        """Seconds of processor time to decode the records to 32-bit floats on one core: the best of some repeats."""
        records = np.ascontiguousarray(records, dtype=np.uint8)
        out = np.empty((len(records), self.n), dtype=np.float32)
        best = None
        for _ in range(repeats):
            self._use()
            t0 = time.process_time()
            bad = self._lib.zb_decode(records.ctypes.data, len(records), float(top), self.steps, out.ctypes.data)
            dt = time.process_time() - t0
            if bad:
                raise ValueError(f"{bad} numbers name no point of the codebook")
            best = dt if best is None else min(best, dt)
        return best, out
