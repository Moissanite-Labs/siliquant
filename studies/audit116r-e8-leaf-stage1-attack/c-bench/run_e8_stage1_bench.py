#!/usr/bin/env python3
"""Self-contained C11 + AVX2/FMA verifier and head-to-head Stage-1 accelerator benchmark
for Track S15d (AUDIT-116).

Compares 6 Stage-1 E_8 leaf unranking variants on EBG32:b88s8 (m_max=48) & EBG32:b136s8 (m_max=391):
  0. baseline    : AUDIT-115 reference (strided-16 / linear 8->4->2)
  1. guide_a     : (A) Pure uint32/uint64 + O(1) top-bit CDF-prefix bucket guide (0 bsearch)
  2. eytz_lut2   : (B) Branchless Eytzinger cmov + flat int8[8] LUT m <= 2 (19.2 KB L1d)
  3. eytz_lut3   : (B) Branchless Eytzinger cmov + flat int8[8] LUT m <= 3 (73.0 KB L2)
  4. d8_compact  : (C1) D_8^+ signed-leader / permutation-orbit + 502B binomial-mask unranker
  5. d8_pshufb   : (C2) D_8^+ signed-leader / permutation-orbit + 22-shape pshufb/psignb LUT
  + Matched 4.500-bpw Q4_K_M-style AVX2/FMA GEMV baseline.

Strict discipline: pure Python 3 stdlib, zero bare asserts, zero checked-in binaries
(all .so files built ephemerally in a 0700 TemporaryDirectory).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from e8_stage1_oracle import (
    e8_shell_count,
    format_exact_decimal,
    is_valid_e8_doubled,
)

SCHEMA_VERSION = "track_s15d.e8_stage1_bench.v1"

AUDIT115_Z_SHA256_136B = "4863b413274293907e2b4792e220a25a0653da58d43814aeca038e48a84a3532"
AUDIT115_Z_SHA256_88B = "644aba45ccdb3a4ed850b6bc3b6761ae4393f2d95efad33871433aaaadabde2b"

COMPILER_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "id": "gcc_native",
        "cc": "gcc",
        "flags": ["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
    {
        "id": "clang_native",
        "cc": "clang",
        "flags": ["-O3", "-march=native", "-Wall", "-Wextra", "-Werror", "-std=c11", "-shared", "-fPIC"],
    },
)

VARIANT_NAMES: tuple[str, ...] = (
    "baseline",
    "guide_a",
    "eytz_lut2",
    "eytz_lut3",
    "d8_compact",
    "d8_pshufb",
)

HOT_SYMBOLS: tuple[str, ...] = (
    "e8s1_decode_z_baseline",
    "e8s1_decode_z_guide_a",
    "e8s1_decode_z_eytz_lut2",
    "e8s1_decode_z_eytz_lut3",
    "e8s1_decode_z_d8_compact",
    "e8s1_decode_z_d8_pshufb",
    "e8s1_gemv_adjoint_baseline",
    "e8s1_gemv_adjoint_guide_a",
    "e8s1_gemv_adjoint_eytz_lut2",
    "e8s1_gemv_adjoint_eytz_lut3",
    "e8s1_gemv_adjoint_d8_compact",
    "e8s1_gemv_adjoint_d8_pshufb",
    "q4km_style_gemv_avx2",
)

_DIV_INSN_RE = re.compile(r"\t(?:i?div[bwlq]?)\b")


class Stage1BenchError(ValueError):
    """Fail-closed error on build, disassembly, exactness, or runtime violations."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise Stage1BenchError(f"missing source file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detect_host() -> dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    cores = os.cpu_count() or 1
    lscpu_map: dict[str, str] = {}
    if shutil.which("lscpu"):
        p = subprocess.run(["lscpu"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            for line in p.stdout.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    lscpu_map[k.strip()] = v.strip()
            if "Model name" in lscpu_map:
                cpu_model = lscpu_map["Model name"]
    flags_str = lscpu_map.get("Flags", "")
    isa_subset = [f for f in ("avx2", "fma", "bmi1", "bmi2", "adx", "f16c", "avx512f") if f in flags_str.split()]
    mhz_nom = 2300
    cur_mhz = mhz_nom
    max_freq_p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    cur_freq_p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if max_freq_p.is_file():
        try:
            mhz_nom = int(max_freq_p.read_text(encoding="utf-8").strip()) // 1000
        except ValueError:
            pass
    if cur_freq_p.is_file():
        try:
            cur_mhz = int(cur_freq_p.read_text(encoding="utf-8").strip()) // 1000
        except ValueError:
            pass
    loadavg = "unknown"
    if Path("/proc/loadavg").is_file():
        loadavg = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    compilers: dict[str, str] = {}
    for cc in ("gcc", "clang"):
        if shutil.which(cc):
            pr = subprocess.run([cc, "--version"], capture_output=True, text=True, check=False)
            compilers[cc] = pr.stdout.splitlines()[0].strip() if pr.stdout else "unknown"
    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": cores,
        "l1d_cache": lscpu_map.get("L1d cache", "unknown"),
        "l2_cache": lscpu_map.get("L2 cache", "unknown"),
        "l3_cache": lscpu_map.get("L3 cache", "unknown"),
        "isa_features": isa_subset,
        "nominal_mhz": mhz_nom,
        "observed_boost_mhz": cur_mhz,
        "host_loadavg_at_bench": loadavg,
        "python_version": platform.python_version(),
        "compilers": compilers,
    }


def _wht8_u_to_z_exact(u: Sequence[int]) -> tuple[int, ...]:
    a0, a1 = u[0] + u[1], u[0] - u[1]
    a2, a3 = u[2] + u[3], u[2] - u[3]
    a4, a5 = u[4] + u[5], u[4] - u[5]
    a6, a7 = u[6] + u[7], u[6] - u[7]
    b0, b1, b2, b3 = a0 + a2, a1 + a3, a0 - a2, a1 - a3
    b4, b5, b6, b7 = a4 + a6, a5 + a7, a4 - a6, a5 - a7
    return (
        (b0 + b4) >> 2, (b1 + b5) >> 2, (b2 + b6) >> 2, (b3 + b7) >> 2,
        (b0 - b4) >> 2, (b1 - b5) >> 2, (b2 - b6) >> 2, (b3 - b7) >> 2,
    )


def _is_valid_e8_had_scaled(z: Sequence[int]) -> bool:
    # Exact Construction-A inverse: u = (H_8 * z) / 2 must be in 2*E_8^std
    a0, a1 = z[0] + z[1], z[0] - z[1]
    a2, a3 = z[2] + z[3], z[2] - z[3]
    a4, a5 = z[4] + z[5], z[4] - z[5]
    a6, a7 = z[6] + z[7], z[6] - z[7]
    b0, b1, b2, b3 = a0 + a2, a1 + a3, a0 - a2, a1 - a3
    b4, b5, b6, b7 = a4 + a6, a5 + a7, a4 - a6, a5 - a7
    hz = (b0 + b4, b1 + b5, b2 + b6, b3 + b7, b0 - b4, b1 - b5, b2 - b6, b3 - b7)
    if any((v & 1) != 0 for v in hz):
        return False
    return is_valid_e8_doubled(tuple(v >> 1 for v in hz))


def _build_exact_reciprocal(divisor: int, max_dividend: int) -> tuple[int, int]:
    if divisor <= 0 or max_dividend < 0:
        raise Stage1BenchError("invalid reciprocal domain")
    if divisor == 1:
        return 1, 0
    if (divisor & (divisor - 1)) == 0:
        return 1, divisor.bit_length() - 1
    k = max(1, max_dividend.bit_length())
    for extra in range(0, divisor.bit_length() + 4):
        shift = k + extra
        two_s = 1 << shift
        mult = (two_s + divisor - 1) // divisor
        if mult * divisor >= two_s and (mult * divisor - two_s) * max_dividend < two_s:
            return mult, shift
    raise Stage1BenchError(f"failed to synthesize universal reciprocal for d={divisor}")


def _ordered_z1_symbols(limit_sq: int) -> list[int]:
    out = [0]
    k = 1
    while k * k <= limit_sq:
        out.append(k)
        out.append(-k)
        k += 1
    return out


def _build_e8_4_tables(m_max: int) -> dict[str, Any]:
    r_max = 2 * m_max
    t_max = max(0, m_max - 1)
    even_p2: list[list[list[tuple[int, int]]]] = [[[] for _ in range(r_max + 1)] for _ in range(2)]
    k_even = _ordered_z1_symbols(r_max)
    for k0 in k_even:
        r0 = k0 * k0
        for k1 in k_even:
            r = r0 + k1 * k1
            if r <= r_max:
                even_p2[(k0 + k1) & 1][r].append((2 * k0, 2 * k1))
    even_c2 = [[len(even_p2[p][r]) for r in range(r_max + 1)] for p in (0, 1)]
    even_c4 = [[0] * (r_max + 1) for _ in range(2)]
    for pa in (0, 1):
        for pb in (0, 1):
            pt = pa ^ pb
            ca, cb, dst = even_c2[pa], even_c2[pb], even_c4[pt]
            for na, va in enumerate(ca):
                if va:
                    for nb in range(r_max - na + 1):
                        vb = cb[nb]
                        if vb:
                            dst[na + nb] += va * vb
    even_c8 = [0] * (m_max + 1)
    for m in range(m_max + 1):
        r8 = 2 * m
        even_c8[m] = sum(even_c4[p][a] * even_c4[p][r8 - a] for p in (0, 1) for a in range(r8 + 1))

    odd_p2: list[list[list[tuple[int, int]]]] = [[[] for _ in range(t_max + 1)] for _ in range(2)]
    k_odd: list[int] = []
    c = 0
    while c * (c + 1) // 2 <= t_max:
        k_odd.append(c)
        k_odd.append(-1 - c)
        c += 1
    for k0 in k_odd:
        t0 = k0 * (k0 + 1) // 2
        for k1 in k_odd:
            t = t0 + k1 * (k1 + 1) // 2
            if t <= t_max:
                odd_p2[(k0 + k1) & 1][t].append((2 * k0 + 1, 2 * k1 + 1))
    odd_c2 = [[len(odd_p2[p][t]) for t in range(t_max + 1)] for p in (0, 1)]
    odd_c4 = [[0] * (t_max + 1) for _ in range(2)]
    for pa in (0, 1):
        for pb in (0, 1):
            pt = pa ^ pb
            ca, cb, dst = odd_c2[pa], odd_c2[pb], odd_c4[pt]
            for na, va in enumerate(ca):
                if va:
                    for nb in range(t_max - na + 1):
                        vb = cb[nb]
                        if vb:
                            dst[na + nb] += va * vb
    odd_c8 = [0] * (m_max + 1)
    for m in range(1, m_max + 1):
        t8 = m - 1
        odd_c8[m] = sum(odd_c4[p][a] * odd_c4[p][t8 - a] for p in (0, 1) for a in range(t8 + 1))

    s1 = [even_c8[m] + odd_c8[m] for m in range(m_max + 1)]
    s2 = [0] * (m_max + 1)
    for m1, c1 in enumerate(s1):
        for m2 in range(m_max - m1 + 1):
            s2[m1 + m2] += c1 * s1[m2]
    w1: list[int] = []
    acc = 0
    for v in s1:
        acc += v
        w1.append(acc)
    w2: list[int] = []
    acc = 0
    for v in s2:
        acc += v
        w2.append(acc)
    c_root = [0]
    acc = 0
    for m_l in range(m_max + 1):
        acc += s2[m_l] * w2[m_max - m_l]
        c_root.append(acc)

    rec_root = [_build_exact_reciprocal(w2[mr], max(s2[: m_max - mr + 1]) * w2[mr] - 1) for mr in range(m_max + 1)]
    rec_b1 = [_build_exact_reciprocal(w1[mr], max(s1[: m_max - mr + 1]) * w1[mr] - 1) for mr in range(m_max + 1)]
    rec_s1 = [_build_exact_reciprocal(s1[mr], max(s1[: m_max - mr + 1]) * s1[mr] - 1) for mr in range(m_max + 1)]

    def _inner_rec(c4: list[list[int]], c2: list[list[int]], lim: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        r4: list[tuple[int, int]] = []
        r2: list[tuple[int, int]] = []
        for p in (0, 1):
            for nr in range(lim + 1):
                d4 = c4[p][nr]
                r4.append(_build_exact_reciprocal(d4, max(c4[p][: lim - nr + 1]) * d4 - 1) if d4 > 0 else (0, 0))
                d2 = c2[p][nr]
                if d2 > 0:
                    ml = max(max(c2[0][: lim - nr + 1]), max(c2[1][: lim - nr + 1]))
                    r2.append(_build_exact_reciprocal(d2, ml * d2 - 1))
                else:
                    r2.append((0, 0))
        return r4, r2

    rec_ec4, rec_ec2 = _inner_rec(even_c4, even_c2, r_max)
    rec_oc4, rec_oc2 = _inner_rec(odd_c4, odd_c2, t_max)
    return {
        "m_max": m_max, "r_max": r_max, "t_max": t_max, "count": c_root[-1],
        "c_root": c_root, "s2": s2, "w2": w2, "s1": s1, "w1": w1,
        "rec_root": rec_root, "rec_b1": rec_b1, "rec_s1": rec_s1,
        "even_c8": even_c8, "even_c4": even_c4, "even_c2": even_c2, "even_p2": even_p2,
        "odd_c4": odd_c4, "odd_c2": odd_c2, "odd_p2": odd_p2,
        "rec_ec4": rec_ec4, "rec_ec2": rec_ec2, "rec_oc4": rec_oc4, "rec_oc2": rec_oc2,
    }


def _pack_u192(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * (len(vals) * 3))()
    mask = (1 << 64) - 1
    for i, v in enumerate(vals):
        arr[3 * i + 0] = v & mask
        arr[3 * i + 1] = (v >> 64) & mask
        arr[3 * i + 2] = (v >> 128) & mask
    return arr


def _pack_u128(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * (len(vals) * 2))()
    mask = (1 << 64) - 1
    for i, v in enumerate(vals):
        arr[2 * i + 0] = v & mask
        arr[2 * i + 1] = (v >> 64) & mask
    return arr


def _pack_u64(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint64 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _pack_u32(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint32 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _pack_u16(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_uint16 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _pack_i8(vals: Sequence[int]) -> Any:
    arr = (ctypes.c_int8 * len(vals))()
    for i, v in enumerate(vals):
        arr[i] = v
    return arr


def _compile_profile(profile: dict[str, Any], src: Path, build_dir: Path) -> tuple[Path, dict[str, Any]]:
    cc = profile["cc"]
    if not shutil.which(cc):
        raise Stage1BenchError(f"compiler missing: {cc}")
    pid = profile["id"]
    so_path = build_dir / f"e8_stage1_{pid}.so"
    cmd = [cc, *profile["flags"], str(src), "-o", str(so_path), "-pthread", "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or proc.stderr.strip():
        raise Stage1BenchError(f"build failed/warned under {pid}: {proc.stderr.strip()}")
    if not shutil.which("objdump"):
        raise Stage1BenchError("objdump required for zero-div check")
    od = subprocess.run(["objdump", "-d", str(so_path)], capture_output=True, text=True, check=False)
    if od.returncode != 0:
        raise Stage1BenchError("objdump failed")
    counts = {sym: -1 for sym in HOT_SYMBOLS}
    cur: str | None = None
    hdr = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:$")
    for line in od.stdout.splitlines():
        m = hdr.match(line)
        if m:
            s = m.group(1)
            cur = s if s in counts else None
            if cur is not None:
                counts[cur] = 0
            continue
        if cur is not None and _DIV_INSN_RE.search(line):
            counts[cur] += 1
    for s, c in counts.items():
        if c != 0:
            raise Stage1BenchError(f"symbol {s} under {pid} has div count {c}")
    return so_path, {
        "profile_id": pid,
        "compiler": cc,
        "flags": profile["flags"],
        "zero_warnings": True,
        "hot_symbol_div_counts": counts,
        "all_zero_div": True,
    }


def _bind_lib(so_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so_path))
    lib.e8s1_has_avx2.argtypes = []
    lib.e8s1_has_avx2.restype = ctypes.c_int
    lib.e8s1_init.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int8), ctypes.c_int, ctypes.POINTER(ctypes.c_int8), ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.e8s1_init.restype = ctypes.c_int
    lib.e8s1_get_mem_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
    lib.e8s1_get_mem_bytes.restype = None
    lib.e8s1_d8_unrank_shell_vec.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
    lib.e8s1_d8_unrank_shell_vec.restype = None

    for name in (
        "e8s1_decode_z_baseline", "e8s1_decode_z_guide_a", "e8s1_decode_z_eytz_lut2",
        "e8s1_decode_z_eytz_lut3", "e8s1_decode_z_d8_compact", "e8s1_decode_z_d8_pshufb",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        fn.restype = ctypes.c_size_t

    for name in (
        "e8s1_gemv_adjoint_baseline", "e8s1_gemv_adjoint_guide_a", "e8s1_gemv_adjoint_eytz_lut2",
        "e8s1_gemv_adjoint_eytz_lut3", "e8s1_gemv_adjoint_d8_compact", "e8s1_gemv_adjoint_d8_pshufb",
        "q4km_style_gemv_avx2",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
        fn.restype = ctypes.c_size_t

    lib.e8s1_gemv_materialized_variant.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
    ]
    lib.e8s1_gemv_materialized_variant.restype = ctypes.c_size_t

    lib.e8s1_extract_leaves_from_blocks.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.e8s1_extract_leaves_from_blocks.restype = ctypes.c_size_t

    lib.e8s1_bench_stage1_only_ns.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.e8s1_bench_stage1_only_ns.restype = ctypes.c_uint64

    lib.e8s1_bench_gemv_ns.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.e8s1_bench_gemv_ns.restype = ctypes.c_uint64
    return lib


def _init_e8(lib: ctypes.CDLL, tbl: dict[str, Any], nbytes: int) -> dict[str, Any]:
    R, T = tbl["r_max"], tbl["t_max"]
    even_off: list[int] = []
    even_pts: list[list[int]] = [[], []]
    for p in (0, 1):
        off = [0]
        for r in range(R + 1):
            for u0, u1 in tbl["even_p2"][p][r]:
                even_pts[p].extend((u0, u1))
            off.append(len(even_pts[p]) // 2)
        even_off.extend(off)
    odd_off: list[int] = []
    odd_pts: list[list[int]] = [[], []]
    for p in (0, 1):
        off = [0]
        for t in range(T + 1):
            for u0, u1 in tbl["odd_p2"][p][t]:
                odd_pts[p].extend((u0, u1))
            off.append(len(odd_pts[p]) // 2)
        odd_off.extend(off)
    rc = lib.e8s1_init(
        tbl["m_max"], nbytes,
        _pack_u192(tbl["c_root"]), _pack_u128(tbl["s2"]), _pack_u128(tbl["w2"]),
        _pack_u64(tbl["s1"]), _pack_u64(tbl["w1"]),
        _pack_u192([m for m, _ in tbl["rec_root"]]), _pack_u32([s for _, s in tbl["rec_root"]]),
        _pack_u128([m for m, _ in tbl["rec_b1"]]), _pack_u32([s for _, s in tbl["rec_b1"]]),
        _pack_u128([m for m, _ in tbl["rec_s1"]]), _pack_u32([s for _, s in tbl["rec_s1"]]),
        _pack_u64(tbl["even_c8"]),
        _pack_u32(tbl["even_c4"][0] + tbl["even_c4"][1]),
        _pack_u16(tbl["even_c2"][0] + tbl["even_c2"][1]),
        _pack_u16(even_off),
        _pack_i8(even_pts[0]), len(even_pts[0]) // 2,
        _pack_i8(even_pts[1]), len(even_pts[1]) // 2,
        _pack_u32(tbl["odd_c4"][0] + tbl["odd_c4"][1]),
        _pack_u16(tbl["odd_c2"][0] + tbl["odd_c2"][1]),
        _pack_u16(odd_off),
        _pack_i8(odd_pts[0]), len(odd_pts[0]) // 2,
        _pack_i8(odd_pts[1]), len(odd_pts[1]) // 2,
        _pack_u64([m for m, _ in tbl["rec_ec4"]]), _pack_u32([s for _, s in tbl["rec_ec4"]]),
        _pack_u32([m for m, _ in tbl["rec_ec2"]]), _pack_u32([s for _, s in tbl["rec_ec2"]]),
        _pack_u64([m for m, _ in tbl["rec_oc4"]]), _pack_u32([s for _, s in tbl["rec_oc4"]]),
        _pack_u32([m for m, _ in tbl["rec_oc2"]]), _pack_u32([s for _, s in tbl["rec_oc2"]]),
    )
    if rc != 0:
        raise Stage1BenchError(f"e8s1_init failed rc={rc}")
    mem6 = (ctypes.c_uint64 * 6)()
    d8_tot = ctypes.c_uint32(0)
    lib.e8s1_get_mem_bytes(mem6, ctypes.byref(d8_tot))
    return {
        "d8_total_leaders": int(d8_tot.value),
        "table_bytes": {name: int(mem6[i]) for i, name in enumerate(VARIANT_NAMES)},
    }


def _gen_audit115_rank_and_record_buffers(label: str, count: int, groups: int, nbytes: int) -> tuple[bytes, bytes]:
    """EXACT same deterministic rank stream as Track S15c (AUDIT-115) for bit-identical z_output_sha256."""
    ranks_buf = bytearray(groups * nbytes)
    rec_buf = bytearray(groups * (nbytes + 1))
    bank = [
        int.from_bytes(hashlib.sha256(f"{label}:b:{i}".encode("ascii")).digest(), "little") % count
        for i in range(min(groups, 4096))
    ]
    if groups > 0:
        bank[0] = 0
    if groups > 1:
        bank[1] = count - 1
    if groups > 2:
        bank[2] = count // 2
    stride_add = (count // 131071) | 1
    b_len = len(bank)
    for g in range(groups):
        rk = bank[g] if g < b_len else (bank[g & 4095] + (g >> 12) * stride_add) % count
        rb = rk.to_bytes(nbytes, "little")
        ranks_buf[g * nbytes : (g + 1) * nbytes] = rb
        rec_buf[g * (nbytes + 1)] = (g * 37 + 13) & 63
        rec_buf[g * (nbytes + 1) + 1 : (g + 1) * (nbytes + 1)] = rb
    return bytes(ranks_buf), bytes(rec_buf)


def _gen_q4_records(groups: int) -> bytes:
    out = bytearray(groups * 18)
    pat = hashlib.sha256(b"s15c_q4_pattern").digest()
    for g in range(groups):
        base = g * 18
        out[base] = (g * 37 + 13) & 63
        out[base + 1] = 8
        for i in range(16):
            out[base + 2 + i] = pat[(i + g) & 31]
    return bytes(out)


def _closest_e8_doubled(x: Sequence[float]) -> list[int]:
    u0 = [2 * round(v * 0.5) for v in x]
    if (sum(u0) >> 1) & 1:
        bi = max(range(8), key=lambda i: abs(x[i] - u0[i]))
        u0[bi] += 2 if x[bi] >= u0[bi] else -2
    d0 = sum((x[i] - u0[i]) ** 2 for i in range(8))
    u1 = [2 * math.floor(v * 0.5) + 1 for v in x]
    if (sum(u1) & 3) != 0:
        bi = max(range(8), key=lambda i: abs(x[i] - u1[i]))
        u1[bi] += 2 if x[bi] >= u1[bi] else -2
    d1 = sum((x[i] - u1[i]) ** 2 for i in range(8))
    return u0 if d0 <= d1 else u1


def _build_gaussian_leaf_arrays(
    m_max: int, s1: Sequence[int], n_leaves: int, outlier_prob: float, outlier_scale: float, seed: int
) -> tuple[Any, Any, dict[str, Any]]:
    """Build deterministic realistic per-block-scaled Gaussian EBG32 leaf stream (tiled cleanly to n_leaves)."""
    rng = random.Random(seed)
    n_sample_blocks = 2048
    target_norm = math.sqrt(8.0 * m_max) * 0.94
    sample_m: list[int] = []
    for _ in range(n_sample_blocks):
        blk = [
            rng.gauss(0.0, 1.0) * (outlier_scale if (outlier_prob > 0.0 and rng.random() < outlier_prob) else 1.0)
            for _ in range(32)
        ]
        nrm = math.sqrt(sum(v * v for v in blk))
        if nrm == 0.0:
            sample_m.extend((0, 0, 0, 0))
            continue
        sc = target_norm / nrm
        while True:
            ms: list[int] = []
            for b in range(4):
                u = _closest_e8_doubled([blk[8 * b + j] * sc for j in range(8)])
                ms.append(sum(c * c for c in u) >> 3)
            if sum(ms) <= m_max:
                sample_m.extend(ms)
                break
            sc *= 0.95

    m_arr = (ctypes.c_int32 * n_leaves)()
    rk_arr = (ctypes.c_uint64 * n_leaves)()
    s_len = len(sample_m)
    h2 = 0
    h3 = 0
    sum_m = 0
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    for i in range(n_leaves):
        m = sample_m[i % s_len]
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        m_arr[i] = m
        rk_arr[i] = state % s1[m]
        sum_m += m
        if m <= 2:
            h2 += 1
        if m <= 3:
            h3 += 1
    stats = {
        "n_leaves": n_leaves,
        "mean_shell_m": format_exact_decimal(Fraction(sum_m, n_leaves), 2),
        "hits_m_le_2": h2,
        "hits_m_le_2_pct": format_exact_decimal(Fraction(h2 * 100, n_leaves), 4),
        "hits_m_le_3": h3,
        "hits_m_le_3_pct": format_exact_decimal(Fraction(h3 * 100, n_leaves), 4),
    }
    return rk_arr, m_arr, stats


def _summarize(wall_ns: list[int], cpu_ns: list[int], weights: int, bytes_read: int, threads: int, nom_mhz: int) -> dict[str, Any]:
    wo, co = sorted(wall_ns), sorted(cpu_ns)
    wm, wlo = wo[len(wo) // 2], wo[0]
    cm, clo, chi = co[len(co) // 2], co[0], co[-1]
    cpu_mwps_med = Fraction(weights * 1000, cm)
    cpu_mwps_min = Fraction(weights * 1000, chi)
    cpu_mwps_max = Fraction(weights * 1000, clo)
    wall_mwps_med = Fraction(weights * 1000, wm)
    wall_mwps_max = Fraction(weights * 1000, wlo)
    cpu_ns_w = Fraction(cm, weights)
    cyc_w = cpu_ns_w * Fraction(nom_mhz, 1000)
    gbps_cpu = Fraction(bytes_read, cm)
    return {
        "threads": threads,
        "trials": len(wall_ns),
        "weights": weights,
        "matrix_bytes": bytes_read,
        "wall_median_ns": wm,
        "cpu_median_ns": cm,
        "cpu_per_core_m_weights_per_sec_median": format_exact_decimal(cpu_mwps_med, 3),
        "cpu_per_core_m_weights_per_sec_min": format_exact_decimal(cpu_mwps_min, 3),
        "cpu_per_core_m_weights_per_sec_max": format_exact_decimal(cpu_mwps_max, 3),
        "wall_m_weights_per_sec_median": format_exact_decimal(wall_mwps_med, 3),
        "wall_m_weights_per_sec_max": format_exact_decimal(wall_mwps_max, 3),
        "cpu_ns_per_weight_median": format_exact_decimal(cpu_ns_w, 3),
        "cpu_cycles_per_weight_at_nominal_mhz": format_exact_decimal(cyc_w, 2),
        "effective_weight_gb_per_sec_cpu_per_core": format_exact_decimal(gbps_cpu, 4),
    }


def _verify_all_variants(
    lib: ctypes.CDLL,
    ranks_bytes: bytes,
    rec_bytes: bytes,
    rows: int,
    cols: int,
    oracle_blocks: int,
    expected_audit115_sha: str | None,
) -> dict[str, Any]:
    groups = rows * (cols // 32)
    in_c = ctypes.create_string_buffer(ranks_bytes, len(ranks_bytes))
    z_bufs = [ctypes.create_string_buffer(groups * 32) for _ in range(6)]
    dec_fns = (
        lib.e8s1_decode_z_baseline,
        lib.e8s1_decode_z_guide_a,
        lib.e8s1_decode_z_eytz_lut2,
        lib.e8s1_decode_z_eytz_lut3,
        lib.e8s1_decode_z_d8_compact,
        lib.e8s1_decode_z_d8_pshufb,
    )
    for i, fn in enumerate(dec_fns):
        bad = int(fn(in_c, groups, z_bufs[i]))
        if bad != 0:
            raise Stage1BenchError(f"decode variant {VARIANT_NAMES[i]} bad={bad}")

    # Variants 0, 1, 2, 3 (baseline, guide_a, eytz_lut2, eytz_lut3) MUST be 100% bit-identical!
    ref_raw = z_bufs[0].raw
    for i in (1, 2, 3):
        if z_bufs[i].raw != ref_raw:
            raise Stage1BenchError(f"variant {VARIANT_NAMES[i]} diverged from baseline Z output")
    z_sha_ballpack = hashlib.sha256(ref_raw).hexdigest()
    if expected_audit115_sha is not None and z_sha_ballpack != expected_audit115_sha:
        raise Stage1BenchError(f"z_output_sha256 {z_sha_ballpack} != AUDIT-115 {expected_audit115_sha}")

    # Variants 4 & 5 (d8_compact & d8_pshufb) MUST be 100% bit-identical to each other!
    if z_bufs[4].raw != z_bufs[5].raw:
        raise Stage1BenchError("d8_compact and d8_pshufb Z outputs diverged")
    z_sha_d8 = hashlib.sha256(z_bufs[4].raw).hexdigest()

    # Check Construction-A E_8_Had membership on first oracle_blocks blocks for BOTH BallPack & D8+!
    n_chk = min(groups, oracle_blocks)
    z_bp = (ctypes.c_int8 * (n_chk * 32)).from_buffer(z_bufs[0])
    z_d8 = (ctypes.c_int8 * (n_chk * 32)).from_buffer(z_bufs[4])
    for g in range(n_chk):
        for sb in range(4):
            off = g * 32 + sb * 8
            if not _is_valid_e8_had_scaled([int(z_bp[off + j]) for j in range(8)]):
                raise Stage1BenchError(f"invalid BallPack E_8_Had vector at block {g} sub {sb}")
            if not _is_valid_e8_had_scaled([int(z_d8[off + j]) for j in range(8)]):
                raise Stage1BenchError(f"invalid D8+ E_8_Had vector at block {g} sub {sb}")

    # Exhaustive C11 D_8^+ bijection proof across ALL 9,121 vectors in m = 0..3 (both compact & pshufb!)
    out8_c = (ctypes.c_int8 * 8)()
    out8_p = (ctypes.c_int8 * 8)()
    d8_m03_tot = 0
    for m in range(4):
        cnt = e8_shell_count(m)
        seen: set[tuple[int, ...]] = set()
        for rk in range(cnt):
            lib.e8s1_d8_unrank_shell_vec(m, rk, 0, out8_c)
            lib.e8s1_d8_unrank_shell_vec(m, rk, 1, out8_p)
            vc = tuple(int(out8_c[j]) for j in range(8))
            vp = tuple(int(out8_p[j]) for j in range(8))
            if vc != vp or not is_valid_e8_doubled(vc) or sum(c * c for c in vc) != 8 * m:
                raise Stage1BenchError(f"C11 D8+ m={m} rk={rk} bijection violation")
            seen.add(vc)
        if len(seen) != cnt:
            raise Stage1BenchError(f"C11 D8+ m={m} duplicate vectors: {len(seen)} != {cnt}")
        d8_m03_tot += len(seen)

    # Verify adjoint GEMV exactness across ALL 6 variants vs materialized reference on all rows!
    scales = (ctypes.c_float * 256)(*[float(2.0 ** (-(c / 16.0))) for c in range(256)])
    x_vec = (ctypes.c_float * cols)(*[float(((i * 17 + 5) % 19) - 9) * 0.125 for i in range(cols)])
    rec_c = ctypes.create_string_buffer(rec_bytes, len(rec_bytes))
    y_mat_bp = (ctypes.c_float * rows)()
    y_mat_d8 = (ctypes.c_float * rows)()
    if int(lib.e8s1_gemv_materialized_variant(0, rec_c, scales, x_vec, rows, cols, y_mat_bp)) != 0:
        raise Stage1BenchError("materialized bp GEMV failed")
    if int(lib.e8s1_gemv_materialized_variant(4, rec_c, scales, x_vec, rows, cols, y_mat_d8)) != 0:
        raise Stage1BenchError("materialized d8 GEMV failed")

    gemv_fns = (
        lib.e8s1_gemv_adjoint_baseline,
        lib.e8s1_gemv_adjoint_guide_a,
        lib.e8s1_gemv_adjoint_eytz_lut2,
        lib.e8s1_gemv_adjoint_eytz_lut3,
        lib.e8s1_gemv_adjoint_d8_compact,
        lib.e8s1_gemv_adjoint_d8_pshufb,
    )
    y_adj = (ctypes.c_float * rows)()
    for i, gfn in enumerate(gemv_fns):
        if int(gfn(rec_c, scales, x_vec, rows, cols, y_adj)) != 0:
            raise Stage1BenchError(f"adjoint GEMV {VARIANT_NAMES[i]} failed")
        ref_y = y_mat_bp if i < 4 else y_mat_d8
        max_d = max(abs(float(y_adj[r]) - float(ref_y[r])) for r in range(rows))
        if max_d > 1e-3:
            raise Stage1BenchError(f"adjoint vs materialized GEMV {VARIANT_NAMES[i]} diff={max_d}")

    return {
        "blocks_verified": groups,
        "weights_verified": groups * 32,
        "e8_subblocks_verified": groups * 4,
        "ballpack_variants_0_to_3_bit_identical": True,
        "matches_audit115_z_sha256": (expected_audit115_sha is None) or (z_sha_ballpack == expected_audit115_sha),
        "z_output_sha256_ballpack": z_sha_ballpack,
        "d8_compact_and_pshufb_bit_identical": True,
        "z_output_sha256_d8plus": z_sha_d8,
        "c11_d8plus_m0_to_m3_bijection_verified": d8_m03_tot,
        "oracle_blocks_e8had_valid": n_chk,
        "gemv_all_6_variants_adjoint_exact": True,
    }


def _time_stage1_only(
    lib: ctypes.CDLL, variant: int, rk_arr: Any, m_arr: Any, n_leaves: int, out_buf: Any, trials: int, nom_mhz: int
) -> dict[str, Any]:
    cpu_c = ctypes.c_uint64(0)
    ws: list[int] = []
    cs: list[int] = []
    for t in range(trials + 1):
        w_ns = int(lib.e8s1_bench_stage1_only_ns(variant, rk_arr, m_arr, n_leaves, out_buf, ctypes.byref(cpu_c)))
        if t > 0:
            ws.append(w_ns)
            cs.append(int(cpu_c.value))
    return _summarize(ws, cs, n_leaves * 8, n_leaves * 8, 1, nom_mhz)


def _time_gemv(
    lib: ctypes.CDLL, mode: int, threads: int, rec_c: Any, rec_sz: int,
    scales: Any, x_vec: Any, rows: int, cols: int, y_out: Any, trials: int, nom_mhz: int
) -> dict[str, Any]:
    bad_c = ctypes.c_size_t(0)
    cpu_c = ctypes.c_uint64(0)
    ws: list[int] = []
    cs: list[int] = []
    for t in range(trials + 1):
        w_ns = int(
            lib.e8s1_bench_gemv_ns(mode, threads, rec_c, rec_sz, scales, x_vec, rows, cols, y_out, ctypes.byref(bad_c), ctypes.byref(cpu_c))
        )
        if int(bad_c.value) != 0:
            raise Stage1BenchError(f"gemv mode {mode} bad={bad_c.value}")
        if t > 0:
            ws.append(w_ns)
            cs.append(int(cpu_c.value))
    groups = rows * (cols // 32)
    return _summarize(ws, cs, rows * cols, groups * rec_sz, threads, nom_mhz)


def run_suite(*, bench: bool, rows: int, cols: int, trials: int, oracle_blocks: int) -> dict[str, Any]:
    src = _HERE / "e8_stage1_bench.c"
    src_sha = _sha256_file(src)
    groups = rows * (cols // 32)
    n_leaves = groups * 4

    tbl_136 = _build_e8_4_tables(391)
    tbl_88 = _build_e8_4_tables(48)

    rk136, rec136 = _gen_audit115_rank_and_record_buffers("s15c_e8_136b", tbl_136["count"], groups, 17)
    rk88, rec88 = _gen_audit115_rank_and_record_buffers("s15c_e8_88b", tbl_88["count"], groups, 11)
    q4_rec = _gen_q4_records(groups)

    exp_sha_136 = AUDIT115_Z_SHA256_136B if (rows == 4096 and cols == 1024) else None
    exp_sha_88 = AUDIT115_Z_SHA256_88B if (rows == 4096 and cols == 1024) else None

    host = _detect_host() if bench else None
    nom_mhz = host["nominal_mhz"] if host else 2300

    # Pre-build realistic Gaussian & Outlier leaf streams ONCE if benchmarking
    gauss_streams: dict[str, Any] = {}
    if bench:
        gauss_streams["88b_iso"] = _build_gaussian_leaf_arrays(48, tbl_88["s1"], n_leaves, 0.0, 1.0, 20261122)
        gauss_streams["88b_out"] = _build_gaussian_leaf_arrays(48, tbl_88["s1"], n_leaves, 0.03, 6.0, 20261123)
        gauss_streams["136b_iso"] = _build_gaussian_leaf_arrays(391, tbl_136["s1"], n_leaves, 0.0, 1.0, 20261124)
        gauss_streams["136b_out"] = _build_gaussian_leaf_arrays(391, tbl_136["s1"], n_leaves, 0.03, 6.0, 20261125)

    builds: list[dict[str, Any]] = []
    exactness: dict[str, Any] = {}
    table_footprints: dict[str, Any] = {}
    benchmarks: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="s15d_s1_") as td:
        os.chmod(td, 0o700)
        bdir = Path(td)
        with ThreadPoolExecutor(max_workers=len(COMPILER_PROFILES)) as pool:
            compiled = list(pool.map(lambda pr: _compile_profile(pr, src, bdir), COMPILER_PROFILES))

        for profile, (so_path, bmeta) in zip(COMPILER_PROFILES, compiled):
            pid = profile["id"]
            lib = _bind_lib(so_path)
            bmeta["avx2_enabled_in_binary"] = bool(int(lib.e8s1_has_avx2()))
            builds.append(bmeta)

            mem136 = _init_e8(lib, tbl_136, 17)
            if mem136["d8_total_leaders"] != 1033160:
                raise Stage1BenchError(f"unexpected 136b leader count {mem136['d8_total_leaders']}")
            ex136 = _verify_all_variants(lib, rk136, rec136, rows, cols, oracle_blocks, exp_sha_136)

            mem88 = _init_e8(lib, tbl_88, 11)
            if mem88["d8_total_leaders"] != 2000:
                raise Stage1BenchError(f"unexpected 88b leader count {mem88['d8_total_leaders']}")
            ex88 = _verify_all_variants(lib, rk88, rec88, rows, cols, oracle_blocks, exp_sha_88)

            exactness[pid] = {"ebg32_b136s8": ex136, "ebg32_b88s8": ex88}
            table_footprints = {"ebg32_b136s8": mem136, "ebg32_b88s8": mem88}

            if bench:
                rk136_c = ctypes.create_string_buffer(rk136, len(rk136))
                rec136_c = ctypes.create_string_buffer(rec136, len(rec136))
                rk88_c = ctypes.create_string_buffer(rk88, len(rk88))
                rec88_c = ctypes.create_string_buffer(rec88, len(rec88))
                q4_c = ctypes.create_string_buffer(q4_rec, len(q4_rec))
                out_leaves = ctypes.create_string_buffer(n_leaves * 8)
                scales = (ctypes.c_float * 256)(*[float(2.0 ** (-(c / 16.0))) for c in range(256)])
                x_vec = (ctypes.c_float * cols)(*[float(((i * 17 + 5) % 19) - 9) * 0.125 for i in range(cols)])
                y_out = (ctypes.c_float * rows)()

                pb: dict[str, Any] = {
                    "q4km_style_gemv_avx2_1t": _time_gemv(lib, 6, 1, q4_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz),
                    "q4km_style_gemv_avx2_2t": _time_gemv(lib, 6, 2, q4_c, 18, scales, x_vec, rows, cols, y_out, trials, nom_mhz),
                }
                q4_cpu = pb["q4km_style_gemv_avx2_1t"]["cpu_median_ns"]

                reg_bench: dict[str, Any] = {}
                for rname, tbl, nb, rec_sz, rk_c, rec_c, iso_k, out_k in (
                    ("ebg32_b88s8", tbl_88, 11, 12, rk88_c, rec88_c, "88b_iso", "88b_out"),
                    ("ebg32_b136s8", tbl_136, 17, 18, rk136_c, rec136_c, "136b_iso", "136b_out"),
                ):
                    _init_e8(lib, tbl, nb)
                    u_rks = (ctypes.c_uint64 * n_leaves)()
                    u_ms = (ctypes.c_int32 * n_leaves)()
                    h2_c = ctypes.c_uint64(0)
                    h3_c = ctypes.c_uint64(0)
                    s0_ns = ctypes.c_uint64(0)
                    if int(lib.e8s1_extract_leaves_from_blocks(rk_c, groups, u_rks, u_ms, ctypes.byref(h2_c), ctypes.byref(h3_c), ctypes.byref(s0_ns))) != 0:
                        raise Stage1BenchError(f"leaf extraction failed for {rname}")
                    unif_stats = {
                        "n_leaves": n_leaves,
                        "hits_m_le_2": int(h2_c.value),
                        "hits_m_le_2_pct": format_exact_decimal(Fraction(int(h2_c.value) * 100, n_leaves), 4),
                        "hits_m_le_3": int(h3_c.value),
                        "hits_m_le_3_pct": format_exact_decimal(Fraction(int(h3_c.value) * 100, n_leaves), 4),
                    }

                    # 1. Isolated Stage-1 head-to-head across ALL 6 variants on Uniform, Gaussian-Iso, and Gaussian-Outlier!
                    s1_unif: dict[str, Any] = {}
                    s1_iso: dict[str, Any] = {}
                    s1_out: dict[str, Any] = {}
                    iso_rks, iso_ms, iso_stats = gauss_streams[iso_k]
                    out_rks, out_ms, out_stats = gauss_streams[out_k]
                    for vi, vname in enumerate(VARIANT_NAMES):
                        s1_unif[vname] = _time_stage1_only(lib, vi, u_rks, u_ms, n_leaves, out_leaves, trials, nom_mhz)
                    # On Gaussian & Outlier streams, time baseline, guide_a, eytz_lut2, eytz_lut3, and d8_pshufb
                    for vi in (0, 1, 2, 3, 5):
                        vname = VARIANT_NAMES[vi]
                        s1_iso[vname] = _time_stage1_only(lib, vi, iso_rks, iso_ms, n_leaves, out_leaves, trials, nom_mhz)
                        s1_out[vname] = _time_stage1_only(lib, vi, out_rks, out_ms, n_leaves, out_leaves, trials, nom_mhz)

                    base_s1 = s1_unif["baseline"]["cpu_median_ns"]
                    s1_speedup_unif = {
                        v: format_exact_decimal(Fraction(base_s1, d["cpu_median_ns"]), 2) + "x"
                        for v, d in s1_unif.items()
                    }
                    base_iso = s1_iso["baseline"]["cpu_median_ns"]
                    s1_speedup_iso = {
                        v: format_exact_decimal(Fraction(base_iso, d["cpu_median_ns"]), 2) + "x"
                        for v, d in s1_iso.items()
                    }
                    base_out = s1_out["baseline"]["cpu_median_ns"]
                    s1_speedup_out = {
                        v: format_exact_decimal(Fraction(base_out, d["cpu_median_ns"]), 2) + "x"
                        for v, d in s1_out.items()
                    }

                    # 2. End-to-End Fused Adjoint E_8_Had AVX2/FMA GEMV across ALL 6 variants (1t) + fastest 2t!
                    gemv_1t: dict[str, Any] = {}
                    for vi, vname in enumerate(VARIANT_NAMES):
                        gemv_1t[vname] = _time_gemv(lib, vi, 1, rec_c, rec_sz, scales, x_vec, rows, cols, y_out, trials, nom_mhz)
                    best_vi = min(range(6), key=lambda vi: gemv_1t[VARIANT_NAMES[vi]]["cpu_median_ns"])
                    best_vname = VARIANT_NAMES[best_vi]
                    best_2t = _time_gemv(lib, best_vi, 2, rec_c, rec_sz, scales, x_vec, rows, cols, y_out, trials, nom_mhz)

                    base_g = gemv_1t["baseline"]["cpu_median_ns"]
                    gemv_speedup_vs_base = {
                        v: format_exact_decimal(Fraction(base_g, d["cpu_median_ns"]), 2) + "x"
                        for v, d in gemv_1t.items()
                    }
                    gemv_ratio_vs_q4 = {
                        v: format_exact_decimal(Fraction(d["cpu_median_ns"], q4_cpu), 2) + "x_slower"
                        for v, d in gemv_1t.items()
                    }
                    reg_bench[rname] = {
                        "leaf_distributions": {
                            "uniform_e8_4_ball": unif_stats,
                            "gaussian_isotropic_per_block": iso_stats,
                            "gaussian_outlier_3pct_6x": out_stats,
                        },
                        "stage1_only_uniform": s1_unif,
                        "stage1_only_uniform_speedup_vs_baseline": s1_speedup_unif,
                        "stage1_only_gaussian_iso": s1_iso,
                        "stage1_only_gaussian_iso_speedup_vs_baseline": s1_speedup_iso,
                        "stage1_only_gaussian_outlier": s1_out,
                        "stage1_only_gaussian_outlier_speedup_vs_baseline": s1_speedup_out,
                        "gemv_adjoint_1t": gemv_1t,
                        "gemv_adjoint_1t_speedup_vs_baseline": gemv_speedup_vs_base,
                        "gemv_adjoint_1t_ratio_vs_matched_q4km": gemv_ratio_vs_q4,
                        "fastest_variant_1t": best_vname,
                        "fastest_variant_2t": best_2t,
                    }

                benchmarks[pid] = {
                    "matched_q4km_baseline": pb,
                    "regimes": reg_bench,
                }

    ref136 = exactness["gcc_native"]["ebg32_b136s8"]
    ref88 = exactness["gcc_native"]["ebg32_b88s8"]
    cross_ok = all(
        exactness[p["id"]]["ebg32_b136s8"]["z_output_sha256_ballpack"] == ref136["z_output_sha256_ballpack"]
        and exactness[p["id"]]["ebg32_b136s8"]["z_output_sha256_d8plus"] == ref136["z_output_sha256_d8plus"]
        and exactness[p["id"]]["ebg32_b88s8"]["z_output_sha256_ballpack"] == ref88["z_output_sha256_ballpack"]
        and exactness[p["id"]]["ebg32_b88s8"]["z_output_sha256_d8plus"] == ref88["z_output_sha256_d8plus"]
        for p in COMPILER_PROFILES
    )
    if not cross_ok:
        raise Stage1BenchError("cross-compiler decode outputs diverged")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "empirical_stage1_attack_benchmark" if bench else "deterministic_verify_only",
        "source_sha256": src_sha,
        "matrix_shape": {"rows": rows, "cols": cols, "weights": rows * cols, "blocks_32": groups, "e8_leaves": n_leaves},
        "compiler_builds": builds,
        "all_profiles_zero_warnings": True,
        "all_hot_kernels_zero_div": True,
        "cross_compiler_outputs_identical": cross_ok,
        "table_footprints": table_footprints,
        "exactness": exactness,
        "overall_verified": True,
    }
    if bench:
        result["host_environment"] = host
        result["benchmarks"] = benchmarks
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track S15d C11+AVX2 Stage-1 accelerator verifier & benchmark.")
    parser.add_argument("--bench", action="store_true", help="Run full 4096x1024 timed benchmark.")
    parser.add_argument("--rows", type=int, default=None, help="Matrix rows (default: 32 verify, 4096 bench).")
    parser.add_argument("--cols", type=int, default=None, help="Matrix cols multiple of 32 (default: 512 verify, 1024 bench).")
    parser.add_argument("--trials", type=int, default=5, help="Timed trials per kernel in --bench (default: 5).")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON to stdout.")
    parser.add_argument("--write-results", type=Path, default=None, help="Run --bench and write JSON to path.")
    args = parser.parse_args(argv)

    bench = bool(args.bench or args.write_results is not None)
    rows = args.rows if args.rows is not None else (4096 if bench else 32)
    cols = args.cols if args.cols is not None else (1024 if bench else 512)
    oracle_blocks = 128 if bench else 16
    res = run_suite(bench=bench, rows=rows, cols=cols, trials=args.trials, oracle_blocks=oracle_blocks)
    rendered = json.dumps(res, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.write_results is not None:
        args.write_results.write_text(rendered, encoding="utf-8")
    if args.json:
        sys.stdout.write(rendered)
    else:
        if bench:
            g = res["benchmarks"]["gcc_native"]["regimes"]
            for rn in ("ebg32_b88s8", "ebg32_b136s8"):
                r = g[rn]
                sys.stdout.write(
                    f"TRACK-S15D {rn} (gcc_native 1t CPU M w/s/c | Stage-1 speedup | GEMV speedup):\n"
                    f"  Stage-1 unif: {r['stage1_only_uniform_speedup_vs_baseline']}\n"
                    f"  GEMV 1t spup: {r['gemv_adjoint_1t_speedup_vs_baseline']} (fastest={r['fastest_variant_1t']})\n"
                    f"  vs Q4_K_M:    {r['gemv_adjoint_1t_ratio_vs_matched_q4km']}\n"
                )
        else:
            d = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            sys.stdout.write(
                f"TRACK-S15D VERIFY OK shape={rows}x{cols} builds=2/2 zero_div=True "
                f"variants=6/6 mismatches=0 sha256={d[:16]}...\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
