#!/usr/bin/env python3
"""Regression & exact-Q R5 tests for Issue #16 (harness/tq.py centred switch).

Verifies:
  1. Pure-Z/Q exact Weyl-vector centroid proof for E8 Voronoi codes
     (mu_inf = rho(a)/30, ||mu_inf||^2 = 31/45, max_{r in Phi} mu_inf.r = 29/30,
     52/240 shifted roots strictly inside (0, sqrt(2)), and exact finite-q
     enumeration at q=2 and odd q=3 showing E8 keeps a non-zero centroid even
     for odd q whereas Z^8 has exact centroid 0).
  2. harness/tq.py Code(name, q, centred=True|False), maxscale_voronoi(...,
     centred=...), and round_matrix(..., centred=...) cache isolation, exact
     zero-codeword preservation on outlier rows, and strict Zone-B validation.

Runs under isolated interpreter modes (-I -B, -I -O -B, -I -B -W error) with
real PyTorch when installed, or via an internal NumPy-backed torch shim on
hosts without PyTorch so harness/tq.py executes directly everywhere.
"""
from __future__ import annotations

from fractions import Fraction
import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT / "lattice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lattices as L  # noqa: E402


def _build_numpy_torch_compat() -> types.ModuleType:
    """Minimal faithful NumPy-backed torch shim for zero-torch hosts."""

    class _DType:
        def __init__(self, np_dtype):
            self.np_dtype = np_dtype

    float16 = _DType(np.float16)
    float32 = _DType(np.float32)
    float64 = _DType(np.float64)

    class _Tensor:
        def __init__(self, data, dtype=None):
            arr = np.asarray(data)
            if dtype is not None:
                arr = arr.astype(dtype.np_dtype, copy=False)
            self._a = arr
            self.device = "cpu"

        @property
        def shape(self):
            return self._a.shape

        def __len__(self):
            return len(self._a)

        def __array__(self, dtype=None):
            return self._a if dtype is None else self._a.astype(dtype)

        def _raw(self, other):
            return other._a if isinstance(other, _Tensor) else other

        def __add__(self, o):
            return _Tensor(self._a + self._raw(o))

        def __radd__(self, o):
            return _Tensor(self._raw(o) + self._a)

        def __sub__(self, o):
            return _Tensor(self._a - self._raw(o))

        def __rsub__(self, o):
            return _Tensor(self._raw(o) - self._a)

        def __mul__(self, o):
            return _Tensor(self._a * self._raw(o))

        def __rmul__(self, o):
            return _Tensor(self._raw(o) * self._a)

        def __truediv__(self, o):
            return _Tensor(self._a / self._raw(o))

        def __rtruediv__(self, o):
            return _Tensor(self._raw(o) / self._a)

        def __pow__(self, o):
            return _Tensor(self._a ** self._raw(o))

        def __matmul__(self, o):
            return _Tensor(self._a @ self._raw(o))

        def __neg__(self):
            return _Tensor(-self._a)

        def __invert__(self):
            return _Tensor(~self._a)

        def __lt__(self, o):
            return _Tensor(self._a < self._raw(o))

        def __le__(self, o):
            return _Tensor(self._a <= self._raw(o))

        def __gt__(self, o):
            return _Tensor(self._a > self._raw(o))

        def __ge__(self, o):
            return _Tensor(self._a >= self._raw(o))

        def __eq__(self, o):
            return _Tensor(self._a == self._raw(o))

        def __ne__(self, o):
            return _Tensor(self._a != self._raw(o))

        def __and__(self, o):
            return _Tensor(self._a & self._raw(o))

        def __or__(self, o):
            return _Tensor(self._a | self._raw(o))

        def __getitem__(self, idx):
            if isinstance(idx, _Tensor):
                idx = idx._a
            elif isinstance(idx, tuple):
                idx = tuple(i._a if isinstance(i, _Tensor) else i for i in idx)
            res = self._a[idx]
            return _Tensor(res) if isinstance(res, np.ndarray) else _Tensor(np.asarray(res))

        def __setitem__(self, idx, val):
            if isinstance(idx, _Tensor):
                idx = idx._a
            self._a[idx] = self._raw(val)

        def __int__(self):
            return int(self._a.item())

        def __float__(self):
            return float(self._a.item())

        def abs(self):
            return _Tensor(np.abs(self._a))

        def sum(self, dim=None, keepdim=False):
            return _Tensor(np.sum(self._a, axis=dim, keepdims=keepdim))

        def mean(self, dim=None, keepdim=False):
            return _Tensor(np.mean(self._a, axis=dim, keepdims=keepdim))

        def std(self, dim=None, correction=1):
            return _Tensor(np.std(self._a, axis=dim, ddof=correction))

        def all(self, dim=None, keepdim=False):
            return _Tensor(np.all(self._a, axis=dim, keepdims=keepdim))

        def argmax(self, dim=None, keepdim=False):
            return _Tensor(np.argmax(self._a, axis=dim, keepdims=keepdim))

        def gather(self, dim, index):
            return _Tensor(np.take_along_axis(self._a, self._raw(index), axis=dim))

        def scatter_add(self, dim, index, src):
            out = self._a.copy()
            idx = self._raw(index)
            vals = self._raw(src)
            if dim == 1 and out.ndim == 2 and idx.shape[1] == 1:
                rows = np.arange(out.shape[0])
                out[rows, idx[:, 0]] += vals[:, 0]
            else:
                raise NotImplementedError("unexpected scatter_add shape")
            return _Tensor(out)

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], tuple):
                shape = shape[0]
            return _Tensor(self._a.reshape(*shape))

        def expand(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], tuple):
                shape = shape[0]
            resolved = tuple(self._a.shape[i] if s == -1 else s for i, s in enumerate(shape))
            return _Tensor(np.broadcast_to(self._a, resolved).copy())

        def max(self, dim=None, keepdim=False):
            if dim is None:
                return _Tensor(np.max(self._a))
            vals = _Tensor(np.max(self._a, axis=dim, keepdims=keepdim))
            idxs = _Tensor(np.argmax(self._a, axis=dim, keepdims=keepdim))
            return types.SimpleNamespace(values=vals, indices=idxs)

        def float(self):
            return _Tensor(self._a.astype(np.float32))

        def double(self):
            return _Tensor(self._a.astype(np.float64))

        def clone(self):
            return _Tensor(self._a.copy())

        def to(self, target):
            if isinstance(target, _DType):
                return _Tensor(self._a.astype(target.np_dtype))
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    mod = types.ModuleType("torch")
    mod.float16 = float16
    mod.float32 = float32
    mod.float64 = float64
    mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    mod.device = lambda name: str(name)
    mod.tensor = lambda data, dtype=None, device=None: _Tensor(np.array(data, copy=True), dtype=dtype)
    mod.as_tensor = lambda data, dtype=None, device=None: _Tensor(np.asarray(data), dtype=dtype)
    mod.from_numpy = lambda arr: _Tensor(np.array(arr, copy=True))
    mod.zeros = lambda *shape, dtype=float32, device=None: _Tensor(np.zeros(shape, dtype=dtype.np_dtype))
    mod.ones = lambda *shape, dtype=float32, device=None: _Tensor(np.ones(shape, dtype=dtype.np_dtype))
    mod.zeros_like = lambda x: _Tensor(np.zeros_like(x._a))
    mod.ones_like = lambda x: _Tensor(np.ones_like(x._a))
    mod.round = lambda x: _Tensor(np.rint(x._a))
    mod.floor = lambda x: _Tensor(np.floor(x._a))
    mod.sqrt = lambda x: _Tensor(np.sqrt(x._a))
    mod.exp = lambda x: _Tensor(np.exp(x._a))
    mod.log = lambda x: _Tensor(np.log(x._a))
    mod.remainder = lambda x, y: _Tensor(np.mod(x._a, y._a if isinstance(y, _Tensor) else y))
    mod.where = lambda cond, a, b: _Tensor(
        np.where(cond._a, a._a if isinstance(a, _Tensor) else a, b._a if isinstance(b, _Tensor) else b)
    )
    mod.clamp = lambda x, min=None, max=None: _Tensor(np.clip(x._a, min, max))
    mod.linalg = types.SimpleNamespace(
        vector_norm=lambda x, dim=None, keepdim=False: _Tensor(
            np.linalg.norm(x._a, axis=dim, keepdims=keepdim).astype(x._a.dtype)
        )
    )
    return mod


def _load_tq():
    try:
        import torch  # noqa: F401
        backend = "real"
    except ModuleNotFoundError:
        sys.modules["torch"] = _build_numpy_torch_compat()
        backend = "numpy_compat"
    tq_mod = importlib.import_module("tq")
    return tq_mod, backend


tq, TORCH_BACKEND = _load_tq()


def _e8_roots_doubled() -> list[tuple[int, ...]]:
    """All 240 E8 roots scaled by 2 (exact integers)."""
    roots: list[tuple[int, ...]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-2, 2):
                for s2 in (-2, 2):
                    v = [0] * 8
                    v[i], v[j] = s1, s2
                    roots.append(tuple(v))
    for mask in range(256):
        v = [1 if (mask >> k) & 1 else -1 for k in range(8)]
        if sum(1 for x in v if x < 0) % 2 == 0:
            roots.append(tuple(v))
    return roots


def _closest_d8_exact(x: list[Fraction]) -> list[int]:
    """Exact rational nearest-point decoder for D8 (ties break deterministically)."""
    f = []
    for v in x:
        # Round half to even to mirror numpy/torch rint deterministically
        n = int(round(v))
        f.append(n)
    if sum(f) % 2 != 0:
        best_j = 0
        best_abs = abs(x[0] - f[0])
        for j in range(1, 8):
            d = abs(x[j] - f[j])
            if d > best_abs:
                best_abs = d
                best_j = j
        f[best_j] += 1 if (x[best_j] - f[best_j]) >= 0 else -1
    return f


def _closest_e8_doubled_exact(x: list[Fraction]) -> tuple[int, ...]:
    """Exact nearest E8 point to rational x, returned in doubled integer coords."""
    a = _closest_d8_exact(x)
    b_shift = _closest_d8_exact([v - Fraction(1, 2) for v in x])
    da = sum((x[k] - a[k]) ** 2 for k in range(8))
    db = sum((x[k] - (b_shift[k] + Fraction(1, 2))) ** 2 for k in range(8))
    if db < da:
        return tuple(2 * b_shift[k] + 1 for k in range(8))
    return tuple(2 * a[k] for k in range(8))


class ExactWeylCentroidTests(unittest.TestCase):
    """Pure-Z/Q mathematical proof of Issue #16's 0.82-step E8 centroid."""

    def test_weyl_vector_asymptotic_centroid_and_cavity_counts(self) -> None:
        roots2 = _e8_roots_doubled()
        self.assertEqual(len(roots2), 240)
        a_frac = [Fraction(*float(v).as_integer_ratio()) for v in L.default_offset(8)]

        pos = [r for r in roots2 if sum(a_frac[k] * r[k] for k in range(8)) > 0]
        ties = [r for r in roots2 if sum(a_frac[k] * r[k] for k in range(8)) == 0]
        self.assertEqual(len(pos), 120)
        self.assertEqual(len(ties), 0)

        rho = tuple(Fraction(sum(r[k] for r in pos), 4) for k in range(8))
        self.assertEqual(rho, (2, 10, 1, -17, 11, 4, -5, 8))
        self.assertEqual(sum(c * c for c in rho), Fraction(620))

        mu_inf = tuple(c / 30 for c in rho)
        mu_n2 = sum(c * c for c in mu_inf)
        self.assertEqual(mu_n2, Fraction(31, 45))
        # Exceeds E8's full average 8D quantization MSE 8*G(E8) = 929/1620
        self.assertGreater(mu_n2, Fraction(929, 1620))

        # Strictly inside open Voronoi cell V(E8): max_{r in Phi} mu_inf . r = 29/30 < 1
        max_dot = max(sum(mu_inf[k] * Fraction(r[k], 2) for k in range(8)) for r in roots2)
        self.assertEqual(max_dot, Fraction(29, 30))
        self.assertLess(max_dot, Fraction(1))

        # Shifted roots strictly inside the open (0, sqrt(2)) cavity: 52/240, min d^2 = 34/45
        shifted_d2 = [sum((Fraction(r[k], 2) - mu_inf[k]) ** 2 for k in range(8)) for r in roots2]
        self.assertEqual(sum(1 for d2 in shifted_d2 if d2 < 2), 52)
        self.assertEqual(min(shifted_d2), Fraction(34, 45))

    def test_exact_finite_q_even_and_odd_e8_vs_z8_centroids(self) -> None:
        a_frac = [Fraction(*float(v).as_integer_ratio()) for v in L.default_offset(8)]
        # Exact E8 generator matrix G doubled to integers
        g2 = (
            (4, 0, 0, 0, 0, 0, 0, 0),
            (-2, 2, 0, 0, 0, 0, 0, 0),
            (0, -2, 2, 0, 0, 0, 0, 0),
            (0, 0, -2, 2, 0, 0, 0, 0),
            (0, 0, 0, -2, 2, 0, 0, 0),
            (0, 0, 0, 0, -2, 2, 0, 0),
            (0, 0, 0, 0, 0, -2, 2, 0),
            (1, 1, 1, 1, 1, 1, 1, 1),
        )
        expected_e8 = {
            2: Fraction(3249, 4096),
            3: Fraction(337978, 531441),
        }
        for q, exp_n2 in expected_e8.items():
            # Enumerate all q^8 codewords in exact Z/Q
            total2 = [0] * 8
            zero_hits = 0
            count = q ** 8
            for idx in range(count):
                rem = idx
                p2 = [0] * 8
                for row in range(8):
                    u_r = rem % q
                    rem //= q
                    if u_r:
                        gr = g2[row]
                        for k in range(8):
                            p2[k] += u_r * gr[k]
                arg = [(Fraction(p2[k], 2) - a_frac[k]) / q for k in range(8)]
                near2 = _closest_e8_doubled_exact(arg)
                c2 = tuple(p2[k] - q * near2[k] for k in range(8))
                if all(v == 0 for v in c2):
                    zero_hits += 1
                for k in range(8):
                    total2[k] += c2[k]
            self.assertEqual(zero_hits, 1)
            mean_q = [Fraction(v, 2 * count) for v in total2]
            self.assertEqual(sum(v * v for v in mean_q), exp_n2)

        # Contrast: Z^8 has exact zero centroid at odd q=3, non-zero norm^2=2 at even q=2
        self.assertEqual(float(np.linalg.norm(L.codebook_centroid(L.Zn(8), 3))), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(L.codebook_centroid(L.Zn(8), 2))) ** 2, 2.0, places=12)


class HarnessCentredSwitchTests(unittest.TestCase):
    """Direct tests of harness/tq.py Code, maxscale_voronoi, and round_matrix."""

    def setUp(self) -> None:
        tq.Code._cache.clear()

    def test_code_cache_isolation_and_zero_codeword(self) -> None:
        c_default = tq.Code("E8", 4)
        c_true = tq.Code("E8", 4, centred=True)
        c_false = tq.Code("E8", 4, centred=False)

        self.assertIs(c_default, c_true)
        self.assertIsNot(c_true, c_false)
        self.assertTrue(c_true.centred)
        self.assertFalse(c_false.centred)

        centre_true = c_true.centre.cpu().numpy().astype(np.float64)
        centre_false = c_false.centre.cpu().numpy().astype(np.float64)
        ref_centre = L.codebook_centroid(L.E8(), 4)
        self.assertTrue(np.allclose(centre_true, ref_centre, atol=1e-6))
        self.assertGreater(float(np.linalg.norm(centre_true)), 0.80)
        self.assertTrue(np.all(centre_false == 0.0))

        # q=2 boundary is valid (kills q < 2 -> q <= 2 mutant)
        c_q2 = tq.Code("E8", 2, centred=False)
        self.assertEqual(c_q2.q, 2)
        self.assertTrue(np.all(c_q2.centre.cpu().numpy() == 0.0))

        # Zero input quantizes to exact 0 under centred=False, non-zero -centre under centred=True
        zeros = tq.torch.zeros(4, 8, dtype=tq.torch.float32, device=tq.DEV)
        r_off, ov_off = c_false.quantize(zeros)
        r_on, _ = c_true.quantize(zeros)
        self.assertEqual(ov_off, 0.0)
        self.assertTrue(np.all(r_off.cpu().numpy() == 0.0))
        self.assertTrue(np.allclose(r_on.cpu().numpy()[0], -centre_true, atol=1e-6))

        # Re-fetching centred=True confirms cache entry remained pristine
        self.assertIs(tq.Code("E8", 4), c_true)
        self.assertTrue(np.allclose(tq.Code("E8", 4).centre.cpu().numpy(), ref_centre, atol=1e-6))

        # Also verify Z8 support
        z_off = tq.Code("Z8", 4, centred=False)
        z_on = tq.Code("Z8", 4, centred=True)
        self.assertIsNot(z_off, z_on)
        self.assertTrue(np.all(z_off.centre.cpu().numpy() == 0.0))
        self.assertAlmostEqual(float(np.linalg.norm(z_on.centre.cpu().numpy())), 2.0 ** 0.5, places=5)

    def test_zone_b_strict_validation_on_code_and_entrypoints(self) -> None:
        for bad_centred in (0, 1, None, "false", "true", [], 1.0):
            with self.assertRaises(TypeError):
                tq.Code("E8", 4, centred=bad_centred)
        for bad_q in (True, False, 1, 0, -4, 4.0, "4", None):
            with self.assertRaises(ValueError):
                tq.Code("E8", bad_q, centred=False)
        with self.assertRaises(KeyError):
            tq.Code("BAD", 4, centred=False)

        W = np.zeros((2, 16), dtype=np.float32)
        W[0, 0] = 1.0
        for bad_centred in (0, 1, None, "false"):
            with self.assertRaises(TypeError):
                tq.maxscale_voronoi(W, 4, "R", name="E8", centred=bad_centred)
            with self.assertRaises(TypeError):
                tq.round_matrix(W, "S", 4, centred=bad_centred)
            with self.assertRaises(TypeError):
                tq.round_matrix(W, "E8", 4, centred=bad_centred)

    def test_maxscale_voronoi_outlier_row_zero_codeword_preservation(self) -> None:
        # Mirrors Issue #16: one giant entry dominates row scale while remaining blocks are zero.
        W = np.zeros((2, 64), dtype=np.float32)
        W[0, 0] = 100.0
        W[1, 0] = 10.0

        Wh_default, info_default = tq.maxscale_voronoi(W, 4, "R", name="E8")
        Wh_on, info_on = tq.maxscale_voronoi(W, 4, "R", name="E8", centred=True)
        Wh_off, info_off = tq.maxscale_voronoi(W, 4, "R", name="E8", centred=False)

        # Backward compatibility: omitted centred matches centred=True bit-for-bit
        self.assertTrue(np.array_equal(Wh_default, Wh_on))
        self.assertEqual(info_default["rel_err"], info_on["rel_err"])

        # With centred=False, every all-zero 8-block stays exact 0.0 (zero DC drift)
        self.assertTrue(np.all(Wh_off[:, 8:] == 0.0))
        # With centred=True, every all-zero 8-block suffers coherent non-zero bias
        self.assertGreater(float(np.max(np.abs(Wh_on[0, 8:]))), 1.0)
        self.assertLess(info_off["rel_err"], info_on["rel_err"])

    def test_round_matrix_forwards_centred_switch(self) -> None:
        W = np.zeros((4, 32), dtype=np.float32)
        W[0, 0] = 50.0
        W[1, 1] = -25.0

        Wh_default, info_default = tq.round_matrix(W, "E8", 4)
        Wh_on, info_on = tq.round_matrix(W, "E8", 4, centred=True)
        Wh_off, info_off = tq.round_matrix(W, "E8", 4, centred=False)

        self.assertTrue(np.array_equal(Wh_default, Wh_on))
        self.assertEqual(info_default["rel_err"], info_on["rel_err"])
        self.assertTrue(np.all(Wh_off[2:, :] == 0.0))
        self.assertGreater(float(np.max(np.abs(Wh_on[2:, :]))), 0.1)

        # Scalar arm still works cleanly with explicit bool centred
        Wh_s_on, _ = tq.round_matrix(W, "S", 4, centred=True)
        Wh_s_off, _ = tq.round_matrix(W, "S", 4, centred=False)
        self.assertTrue(np.array_equal(Wh_s_on, Wh_s_off))


if __name__ == "__main__":
    unittest.main()
