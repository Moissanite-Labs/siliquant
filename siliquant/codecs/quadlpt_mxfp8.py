"""A_16 [16, 11, 4] lattice codec, QuadLPT-SharedNorm permutations, MXFP8 cliff protection, and streaming accumulator."""

from __future__ import annotations

import math
import struct
from typing import Any, Sequence

from ._validation import (
    A16_INFO_POSITIONS,
    ALL_15B_TOTAL_PARAMS,
    BALL_BOUND_45_G128,
    BALL_BOUNDS,
    CLIFF_L0_L27_A16_SHIPPED_BYTES,
    CLIFF_L0_L27_WEIGHTS,
    require,
)

CLIFF_MXFP8_GROUP = 32
CLIFF_MXFP8_BPW = 8.25
CLIFF_BPW_DELTA_MAX = 0.20


def check_a16_syndrome_py(y16: Sequence[int]) -> bool:
    """Return True iff 16 integer coordinates form a valid Extended Hamming [16, 11, 4] codeword."""
    require(len(y16) == 16, "A16_LEN", f"expected 16 integers, got {len(y16)}")
    bits = [abs(int(v)) & 1 for v in y16]
    p0 = sum(bits) & 1
    p1 = (bits[1] ^ bits[3] ^ bits[5] ^ bits[7] ^ bits[9] ^ bits[11] ^ bits[13] ^ bits[15]) & 1
    p2 = (bits[2] ^ bits[3] ^ bits[6] ^ bits[7] ^ bits[10] ^ bits[11] ^ bits[14] ^ bits[15]) & 1
    p4 = (bits[4] ^ bits[5] ^ bits[6] ^ bits[7] ^ bits[12] ^ bits[13] ^ bits[14] ^ bits[15]) & 1
    p8 = (bits[8] ^ bits[9] ^ bits[10] ^ bits[11] ^ bits[12] ^ bits[13] ^ bits[14] ^ bits[15]) & 1
    return (p0 | p1 | p2 | p4 | p8) == 0


def build_c16_codewords_py() -> list[tuple[int, ...]]:
    """Enumerate all 2048 Extended Hamming [16, 11, 4] binary codewords."""
    codewords: list[tuple[int, ...]] = []
    for u11 in range(2048):
        b = [0] * 16
        for j, pos in enumerate(A16_INFO_POSITIONS):
            b[pos] = (u11 >> j) & 1
        b[1] = b[3] ^ b[5] ^ b[7] ^ b[9] ^ b[11] ^ b[13] ^ b[15]
        b[2] = b[3] ^ b[6] ^ b[7] ^ b[10] ^ b[11] ^ b[14] ^ b[15]
        b[4] = b[5] ^ b[6] ^ b[7] ^ b[12] ^ b[13] ^ b[14] ^ b[15]
        b[8] = b[9] ^ b[10] ^ b[11] ^ b[12] ^ b[13] ^ b[14] ^ b[15]
        b[0] = sum(b[1:]) & 1
        codewords.append(tuple(b))
    return codewords


def a16_nearest_py(x16: Sequence[float], codewords: Sequence[tuple[int, ...]] | None = None) -> list[int]:
    """Exhaustive 2048-codeword pure-Python A_16 = 2Z^16 + C_[16,11,4] nearest-lattice-point decoder."""
    require(len(x16) == 16, "A16_LEN", f"expected 16 floats, got {len(x16)}")
    cws = codewords if codewords is not None else build_c16_codewords_py()
    even_round = [2 * int(round(0.5 * v)) for v in x16]
    odd_round = [2 * int(math.floor(0.5 * v)) + 1 for v in x16]
    even_err = [(x16[i] - even_round[i]) ** 2 for i in range(16)]
    odd_err = [(x16[i] - odd_round[i]) ** 2 for i in range(16)]

    best_y: list[int] | None = None
    best_d2 = float("inf")
    for cw in cws:
        d2 = 0.0
        for i in range(16):
            d2 += odd_err[i] if cw[i] else even_err[i]
        if d2 < best_d2:
            best_d2 = d2
            best_y = [odd_round[i] if cw[i] else even_round[i] for i in range(16)]
    require(best_y is not None and check_a16_syndrome_py(best_y), "A16_SYN", "pure-Python A16 decode failed")
    return best_y if best_y is not None else [0] * 16


def a16_closest_fast_py(x16: Sequence[float]) -> list[int]:
    """Fast O(16) algebraic Wagner bounded-distance lattice decoder for A_16 = 2Z^16 + C_[16,11,4]."""
    require(len(x16) == 16, "A16_LEN", f"expected 16 floats, got {len(x16)}")
    e_val = [2 * int(round(0.5 * float(v))) for v in x16]
    o_val = [2 * int(math.floor(0.5 * float(v))) + 1 for v in x16]
    de = [(float(x16[i]) - e_val[i]) ** 2 for i in range(16)]
    do = [(float(x16[i]) - o_val[i]) ** 2 for i in range(16)]

    is_odd = [do[i] < de[i] for i in range(16)]
    out = [o_val[i] if is_odd[i] else e_val[i] for i in range(16)]
    alt = [e_val[i] if is_odd[i] else o_val[i] for i in range(16)]
    fc = [abs(de[i] - do[i]) for i in range(16)]
    bits = [1 if is_odd[i] else 0 for i in range(16)]

    p0 = sum(bits) & 1
    s0 = (bits[1] ^ bits[3] ^ bits[5] ^ bits[7] ^ bits[9] ^ bits[11] ^ bits[13] ^ bits[15]) & 1
    s1 = (bits[2] ^ bits[3] ^ bits[6] ^ bits[7] ^ bits[10] ^ bits[11] ^ bits[14] ^ bits[15]) & 1
    s2 = (bits[4] ^ bits[5] ^ bits[6] ^ bits[7] ^ bits[12] ^ bits[13] ^ bits[14] ^ bits[15]) & 1
    s3 = (bits[8] ^ bits[9] ^ bits[10] ^ bits[11] ^ bits[12] ^ bits[13] ^ bits[14] ^ bits[15]) & 1
    syn = s0 + (s1 << 1) + (s2 << 2) + (s3 << 3)

    if p0 == 1:
        out[syn] = alt[syn]
    elif syn != 0:
        best_j = 0
        best_cost = float("inf")
        for j in range(16):
            p = j ^ syn
            cost = fc[j] + fc[p]
            if cost < best_cost:
                best_cost = cost
                best_j = j
        out[best_j] = alt[best_j]
        out[best_j ^ syn] = alt[best_j ^ syn]

    require(check_a16_syndrome_py(out), "A16_FAST_SYN", "fast A16 decoder violated parity")
    return out


def quantize_a16_group_ball_py(
    w_group: Sequence[float],
    codewords: Sequence[tuple[int, ...]] | None = None,
    r2_bound: float = BALL_BOUND_45_G128,
) -> tuple[list[int], float]:
    """Pure-Python 128-dim L2-ball scale binary search for exact A_16 codebook fit."""
    require(len(w_group) == 128, "A16_GRP_LEN", f"expected 128 elements, got {len(w_group)}")
    _ = codewords
    norm_w = math.sqrt(sum(float(v) * float(v) for v in w_group))
    if norm_w < 1e-12:
        return [0] * 128, 1.0
    s_min = norm_w / math.sqrt(r2_bound)
    s_hi = max(s_min * 4.0, max(abs(float(v)) for v in w_group) / 2.0)

    def _fit(scale_val: float) -> tuple[list[int], int]:
        ys: list[int] = []
        r2 = 0
        for b0 in range(0, 128, 16):
            y16 = a16_closest_fast_py([float(v) / scale_val for v in w_group[b0 : b0 + 16]])
            ys.extend(y16)
            r2 += sum(iv * iv for iv in y16)
        return ys, r2

    lo = s_min
    hi = s_hi
    for _ in range(12):
        _, r2_hi = _fit(hi)
        if r2_hi <= r2_bound:
            break
        hi *= 1.5
    best_y, best_r2 = _fit(hi)
    best_s = hi
    require(best_r2 <= r2_bound, "A16_BALL_FIT", f"failed to fit ball bound {r2_bound}")
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        ym, r2_m = _fit(mid)
        if r2_m <= r2_bound:
            best_y, best_s = ym, mid
            hi = mid
        else:
            lo = mid
    return best_y, best_s


def _validate_quad_perm_geom(k: int, group_size: int, block_w: int) -> None:
    require(
        k > 0 and group_size > 0 and block_w > 0 and (k % group_size) == 0 and (group_size % block_w) == 0,
        "LPT_GEOM",
        f"invalid geometry k={k}, group_size={group_size}, block_w={block_w}",
    )


def make_quad_lpt_perm_py(
    score: Sequence[float],
    group_size: int = 128,
    block_w: int = 4,
    load_power: float = 1.0,
) -> list[int]:
    """Stratified greedy Longest-Processing-Time (QuadLPT) column permutation (pure Python)."""
    k = len(score)
    _validate_quad_perm_geom(k, group_size, block_w)
    n_groups = k // group_size
    n_quads = k // block_w
    qpg = group_size // block_w
    s64 = [max(float(v), 1e-30) for v in score]
    idx = sorted(range(k), key=lambda i: (-s64[i], i))
    quads = [idx[q * block_w : (q + 1) * block_w] for q in range(n_quads)]
    q_loads = [sum(s64[i] for i in q) ** float(load_power) for q in quads]

    loads = [0.0] * n_groups
    counts = [0] * n_groups
    buckets: list[list[int]] = [[] for _ in range(n_groups)]
    for q_idx, q_cols in enumerate(quads):
        best_g = -1
        best_load = float("inf")
        for g in range(n_groups):
            if counts[g] < qpg and loads[g] < best_load:
                best_load = loads[g]
                best_g = g
        require(best_g >= 0, "LPT_CAP", "LPT bucket capacity exceeded")
        buckets[best_g].extend(q_cols)
        loads[best_g] += q_loads[q_idx]
        counts[best_g] += 1
    return [col for b in buckets for col in b]


def make_quad_snake_perm_py(score: Sequence[float], group_size: int = 128, block_w: int = 4) -> list[int]:
    """Stratified boustrophedon (QuadSnake) column permutation (pure Python)."""
    k = len(score)
    _validate_quad_perm_geom(k, group_size, block_w)
    n_groups = k // group_size
    n_quads = k // block_w
    qpg = group_size // block_w
    s64 = [max(float(v), 1e-30) for v in score]
    idx = sorted(range(k), key=lambda i: (-s64[i], i))
    quads = [idx[q * block_w : (q + 1) * block_w] for q in range(n_quads)]
    buckets: list[list[int]] = [[] for _ in range(n_groups)]
    for t in range(qpg):
        order = range(n_groups) if (t % 2 == 0) else range(n_groups - 1, -1, -1)
        for pos, g in enumerate(order):
            buckets[g].extend(quads[t * n_groups + pos])
    return [col for b in buckets for col in b]


def validate_perm_in_py(perm_in: Sequence[int], in_features: int) -> list[int]:
    """Validate that `perm_in` is a bijection on `[0, in_features)`."""
    require(len(perm_in) == in_features, "PERM_LEN", f"expected {in_features}, got {len(perm_in)}")
    p = [int(v) for v in perm_in]
    require(sorted(p) == list(range(in_features)), "PERM_BIJ", "perm_in is not a valid bijection")
    return p


def invert_perm_py(perm_in: Sequence[int]) -> list[int]:
    """Return exact inverse permutation `inv_p` satisfying `inv_p[perm_in[i]] == i`."""
    p = validate_perm_in_py(perm_in, len(perm_in))
    inv_p = [0] * len(p)
    for new_pos, old_col in enumerate(p):
        inv_p[old_col] = new_pos
    return inv_p


def _f32_to_bits_u32(val: float) -> int:
    return int(struct.unpack("<I", struct.pack("<f", float(val)))[0])


def _bits_u32_to_f32(bits: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0])


def _f16_to_bits_u16(val: float) -> int:
    return int(struct.unpack("<H", struct.pack("<e", float(val)))[0])


def _bits_u16_to_f16(bits: int) -> float:
    return float(struct.unpack("<e", struct.pack("<H", int(bits) & 0xFFFF))[0])


def pack_quadlpt_sharednorm_py(
    weight_rows: Sequence[Sequence[float]],
    colscale: Sequence[float] | None = None,
    perm_in: Sequence[int] | None = None,
    *,
    group_size: int = 128,
    bpw_key: str = "4.5",
) -> dict[str, Any]:
    """Pack a 2D weight matrix using QuadLPT permutation + dtype-immune FP32 colscale / FP16 group scales + A_16."""
    require(len(weight_rows) > 0, "QLPT_EMPTY", "weight matrix cannot be empty")
    m_rows = len(weight_rows)
    k_cols = len(weight_rows[0])
    require(
        all(len(r) == k_cols for r in weight_rows),
        "QLPT_RECT",
        "jagged weight matrix rows forbidden",
    )
    require(
        bpw_key in BALL_BOUNDS and group_size in BALL_BOUNDS[bpw_key],
        "QLPT_CFG",
        f"unsupported (bpw_key={bpw_key!r}, group_size={group_size})",
    )
    require(k_cols > 0 and (k_cols % group_size) == 0 and (group_size % 16) == 0, "QLPT_DIV", "invalid k_cols")

    p_idx = validate_perm_in_py(perm_in, k_cols) if perm_in is not None else list(range(k_cols))
    cs_vals = [float(v) for v in colscale] if colscale is not None else [1.0] * k_cols
    require(len(cs_vals) == k_cols, "QLPT_CS_LEN", "colscale length mismatch")
    require(all(math.isfinite(v) and abs(v) > 1e-12 for v in cs_vals), "QLPT_CS_ZERO", "colscale must be finite non-zero")

    cs_perm = [cs_vals[j] for j in p_idx]
    cs_fp32_bits = [_f32_to_bits_u32(v) for v in cs_perm]
    cs_exact = [_bits_u32_to_f32(b) for b in cs_fp32_bits]

    r2_bound = BALL_BOUNDS[bpw_key][group_size]
    y_rows: list[list[int]] = []
    scales_fp16_bits: list[list[int]] = []
    n_groups = k_cols // group_size
    for r in range(m_rows):
        w_perm = [float(weight_rows[r][p_idx[j]]) * cs_exact[j] for j in range(k_cols)]
        y_row: list[int] = []
        sc_row: list[int] = []
        for g in range(n_groups):
            grp = w_perm[g * group_size : (g + 1) * group_size]
            if group_size == 128:
                yg, sg = quantize_a16_group_ball_py(grp, r2_bound=r2_bound)
            else:
                norm_g = math.sqrt(sum(v * v for v in grp))
                sg = max(norm_g / math.sqrt(r2_bound), max((abs(v) for v in grp), default=0.0) / 2.0, 1e-12)
                yg = []
                for b0 in range(0, group_size, 16):
                    yg.extend(a16_closest_fast_py([v / sg for v in grp[b0 : b0 + 16]]))
            sb = _f16_to_bits_u16(sg)
            sc_row.append(sb)
            y_row.extend(yg)
        y_rows.append(y_row)
        scales_fp16_bits.append(sc_row)

    return {
        "out_features": m_rows,
        "in_features": k_cols,
        "group_size": group_size,
        "bpw_key": bpw_key,
        "y_int": y_rows,
        "scales_fp16_bits": scales_fp16_bits,
        "colscale_fp32_bits": cs_fp32_bits,
        "perm_in": p_idx,
    }


def unpack_quadlpt_sharednorm_py(packed: dict[str, Any]) -> tuple[list[list[float]], int]:
    """Dequantize a QuadLPT-SharedNorm packed payload and count A_16 syndrome errors."""
    m_rows = int(packed["out_features"])
    k_cols = int(packed["in_features"])
    group_size = int(packed["group_size"])
    y_rows: list[list[int]] = packed["y_int"]
    scales_bits: list[list[int]] = packed["scales_fp16_bits"]
    cs_bits: list[int] = packed["colscale_fp32_bits"]
    perm_in: list[int] = validate_perm_in_py(packed["perm_in"], k_cols)
    inv_p = invert_perm_py(perm_in)
    cs_exact = [_bits_u32_to_f32(b) for b in cs_bits]

    syn_errs = 0
    out_rows: list[list[float]] = []
    n_groups = k_cols // group_size
    for r in range(m_rows):
        yr = y_rows[r]
        sr = scales_bits[r]
        require(len(yr) == k_cols and len(sr) == n_groups, "QLPT_UNPACK_SHAPE", "row shape mismatch")
        for b0 in range(0, k_cols, 16):
            if not check_a16_syndrome_py(yr[b0 : b0 + 16]):
                syn_errs += 1
        w_perm = [0.0] * k_cols
        for g in range(n_groups):
            sc = _bits_u16_to_f16(sr[g])
            base = g * group_size
            for j in range(group_size):
                w_perm[base + j] = (float(yr[base + j]) * sc) / cs_exact[base + j]
        out_rows.append([w_perm[inv_p[j]] for j in range(k_cols)])
    return out_rows, syn_errs


def float_to_fp8_e4m3fn_byte_py(val: float) -> int:
    """Quantize a finite scalar to the nearest FP8 E4M3FN byte (bias=7, max=448)."""
    require(math.isfinite(val), "MXFP8_NONFINITE", f"non-finite input: {val!r}")
    if val == 0.0:
        return 0x00
    sign = 0x80 if val < 0.0 else 0x00
    ax = min(abs(val), 448.0)
    if ax < 2.0 ** -6:
        m = int(round(ax / (2.0 ** -9)))
        if m <= 0:
            return sign
        if m >= 8:
            return sign | (1 << 3)
        return sign | m
    exp_unbiased = int(math.floor(math.log2(ax)))
    exp_unbiased = max(-6, min(8, exp_unbiased))
    sig = ax / (2.0 ** exp_unbiased)
    mant = int(round((sig - 1.0) * 8.0))
    if mant >= 8:
        mant = 0
        exp_unbiased += 1
    if exp_unbiased > 8:
        exp_unbiased = 8
        mant = 6
    if exp_unbiased == 8 and mant >= 7:
        mant = 6
    exp_field = exp_unbiased + 7
    return sign | (exp_field << 3) | (mant & 0x07)


def fp8_e4m3fn_byte_to_float_py(byte_val: int) -> float:
    """Decode one unsigned FP8 E4M3FN byte (`0..255`) to Python float."""
    b = int(byte_val)
    require(0 <= b <= 255, "FP8_BYTE_RANGE", f"byte out of range [0, 255]: {b}")
    sign = -1.0 if ((b >> 7) & 1) else 1.0
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    if exp == 0:
        return sign * (float(mant) / 8.0) * (2.0 ** -6)
    if exp == 15 and mant == 7:
        return float("nan")
    return sign * (1.0 + float(mant) / 8.0) * (2.0 ** (exp - 7))


def quantize_dequant_mxfp8_block32_py(vals32: Sequence[float]) -> tuple[int, list[int], list[float]]:
    """Quantize and dequantize one 32-element block using UE8M0 + FP8 E4M3FN (8.25 bpw)."""
    require(len(vals32) == CLIFF_MXFP8_GROUP, "MXFP8_LEN", f"expected {CLIFF_MXFP8_GROUP}, got {len(vals32)}")
    amax = max(abs(float(v)) for v in vals32)
    if amax <= 0.0:
        return 127, [0] * CLIFF_MXFP8_GROUP, [0.0] * CLIFF_MXFP8_GROUP
    req_scale = amax / 448.0
    e_raw = int(math.ceil(math.log2(req_scale)))
    e_clamped = max(-127, min(127, e_raw))
    ue8m0_byte = e_clamped + 127
    scale = 2.0 ** float(e_clamped)
    q_bytes = [float_to_fp8_e4m3fn_byte_py(float(v) / scale) for v in vals32]
    dq = [fp8_e4m3fn_byte_to_float_py(qb) * scale for qb in q_bytes]
    return ue8m0_byte, q_bytes, dq


def compute_edge_cliff_mxfp8_budget(
    cliff_weight_count: int,
    cliff_a16_bytes: int,
    total_model_params: int,
    *,
    cap_delta_bpw: float = CLIFF_BPW_DELTA_MAX,
) -> dict[str, Any]:
    """Compute exact MXFP8 cliff-protection byte and bit-per-weight deltas for boundary layers."""
    require(cliff_weight_count > 0 and (cliff_weight_count % CLIFF_MXFP8_GROUP) == 0, "CLIFF_W", "must divide by 32")
    require(total_model_params >= cliff_weight_count, "CLIFF_TOT", "total_model_params < cliff_weight_count")
    mxfp8_bytes = (cliff_weight_count * 33) // 32
    actual_bpw = 8.0 * float(mxfp8_bytes) / float(cliff_weight_count)
    delta_bytes = mxfp8_bytes - int(cliff_a16_bytes)
    delta_bpw = 8.0 * float(delta_bytes) / float(total_model_params)
    return {
        "cliff_weights": cliff_weight_count,
        "mxfp8_bpw": actual_bpw,
        "mxfp8_bytes": mxfp8_bytes,
        "a16_shipped_bytes": int(cliff_a16_bytes),
        "delta_bytes": delta_bytes,
        "delta_bpw_over_model": delta_bpw,
        "within_cliff_cap": delta_bpw <= cap_delta_bpw,
    }


def compute_l0_l27_cliff_bpw_stats() -> dict[str, Any]:
    """Return exact L0 + L27 (14 matrices) MXFP8 vs shipped QuadLPT A16 byte and bpw deltas."""
    per_layer_weights = (
        1536 * 1536 + 256 * 1536 + 256 * 1536 + 1536 * 1536 + 3 * (8960 * 1536)
    )
    two_layer_weights = 2 * per_layer_weights
    require(two_layer_weights == CLIFF_L0_L27_WEIGHTS, "CLIFF_W_MISMATCH", str(two_layer_weights))
    stats = compute_edge_cliff_mxfp8_budget(
        two_layer_weights,
        CLIFF_L0_L27_A16_SHIPPED_BYTES,
        ALL_15B_TOTAL_PARAMS,
    )
    return {
        "n_matrices": 14,
        "two_layer_weights": stats["cliff_weights"],
        "mxfp8_bpw": stats["mxfp8_bpw"],
        "mxfp8_bytes": stats["mxfp8_bytes"],
        "a16_shipped_bytes": stats["a16_shipped_bytes"],
        "delta_bytes_vs_shipped": stats["delta_bytes"],
        "delta_bpw_over_1p544b": stats["delta_bpw_over_model"],
        "within_cliff_cap": stats["within_cliff_cap"],
    }


class A16ChunkAccumulator:
    """Streaming row-chunk A_16 encoder accepting incremental weight rows without full-matrix BF16 staging."""

    def __init__(self, out_features: int, in_features: int, *, group_size: int = 128, bpw_key: str = "4.5") -> None:
        require(out_features > 0 and in_features > 0, "ACC_DIMS", "matrix dimensions must be positive")
        require(bpw_key in BALL_BOUNDS and group_size in BALL_BOUNDS[bpw_key], "ACC_CFG", "invalid ball config")
        require((in_features % group_size) == 0 and (group_size % 16) == 0, "ACC_DIV", "invalid in_features")
        self.out_features = int(out_features)
        self.in_features = int(in_features)
        self.group_size = int(group_size)
        self.bpw_key = str(bpw_key)
        self._rows_seen = 0
        self._y_rows: list[list[int]] = []
        self._scales_fp16_bits: list[list[int]] = []
        self._syn_errors = 0

    def push_rows(self, rows: Sequence[Sequence[float]]) -> None:
        """Encode and append one streaming batch of float rows (from BF16, FP8, or FP4 dequant chunks)."""
        r2_bound = BALL_BOUNDS[self.bpw_key][self.group_size]
        n_groups = self.in_features // self.group_size
        for row in rows:
            require(self._rows_seen < self.out_features, "ACC_OVERFLOW", "too many rows pushed")
            require(len(row) == self.in_features, "ACC_ROW_LEN", f"expected {self.in_features}, got {len(row)}")
            y_row: list[int] = []
            sc_row: list[int] = []
            for g in range(n_groups):
                grp = [float(v) for v in row[g * self.group_size : (g + 1) * self.group_size]]
                if self.group_size == 128:
                    yg, sg = quantize_a16_group_ball_py(grp, r2_bound=r2_bound)
                else:
                    norm_g = math.sqrt(sum(v * v for v in grp))
                    sg = max(norm_g / math.sqrt(r2_bound), max((abs(v) for v in grp), default=0.0) / 2.0, 1e-12)
                    yg = []
                    for b0 in range(0, self.group_size, 16):
                        yg.extend(a16_closest_fast_py([v / sg for v in grp[b0 : b0 + 16]]))
                for b0 in range(0, self.group_size, 16):
                    if not check_a16_syndrome_py(yg[b0 : b0 + 16]):
                        self._syn_errors += 1
                y_row.extend(yg)
                sc_row.append(_f16_to_bits_u16(sg))
            self._y_rows.append(y_row)
            self._scales_fp16_bits.append(sc_row)
            self._rows_seen += 1

    def finalize(self) -> dict[str, Any]:
        """Finalize streamed matrix after verifying exact row count and zero syndrome errors."""
        require(
            self._rows_seen == self.out_features,
            "ACC_INCOMPLETE",
            f"expected {self.out_features} rows, received {self._rows_seen}",
        )
        require(self._syn_errors == 0, "ACC_SYN", f"non-zero syndrome errors: {self._syn_errors}")
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "group_size": self.group_size,
            "bpw_key": self.bpw_key,
            "y_int": self._y_rows,
            "scales_fp16_bits": self._scales_fp16_bits,
            "syndrome_errors": self._syn_errors,
        }
