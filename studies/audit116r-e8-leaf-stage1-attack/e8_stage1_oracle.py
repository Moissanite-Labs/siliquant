#!/usr/bin/env python3
"""Pure-Z/Q exact oracle for Track S15d (AUDIT-116): attacking the Stage-1 inner E_8 leaf
unranking bottleneck (68%-75% of total CPU cycles in AUDIT-115).

Zero floats, zero bare asserts, zero third-party imports, strict fail-closed validation
(type(x) is int everywhere; Stage1OracleError on all malformed inputs).

Proves three exact pure-Z/Q pillars:
  Pillar 1: Exact Stage-1 worst-case bit-widths across EBG32:b88s8 (m_max=48 -> pure uint32_t,
            S1_max < 2^26, W1 < 2^29) and EBG32:b136s8 (m_max=391 -> pure uint64_t, ZERO u128,
            S1_max < 2^35, W1 < 2^41), plus exact c4 (uint16_t) and c2 (uint8_t) widths.
  Pillar 2: Exact O(1) top-bit CDF-prefix bucket guide theorem (b = rem >> sh < B, zero div,
            zero mul, exact lower-bound invariant across every shell) AND exact pure-Q
            concentration-of-measure hit probabilities P(m <= 2) and P(m <= 3) under uniform
            E_8^4 ball sampling.
  Pillar 3: Exact D_8^+ signed-leader / permutation-orbit bijection (exhaustive 9,121-vector
            proof across m = 0..3; exact orbit sum == 240*sigma_3(m) across m = 0..48 with
            only 2,000 leaders = 32 KB L1d-resident AND across m = 0..391 proving the 516.58x
            phase-transition blowup to 1,033,160 leaders = 16.53 MB > L3, plus exact 22-shape
            permutation total 95,503 and 502-byte binomial combination table).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from fractions import Fraction
import math
from typing import Any, Sequence


class Stage1OracleError(ValueError):
    """Fail-closed exception for all malformed inputs or invariant violations."""


_FACT: tuple[int, ...] = (1, 1, 2, 6, 24, 120, 720, 5040, 40320)


def _require_strict_int(val: Any, name: str, lo: int, hi: int | None = None) -> int:
    if type(val) is not int:
        raise Stage1OracleError(f"{name} must be strict int")
    if val < lo:
        raise Stage1OracleError(f"{name} must be >= {lo}")
    if hi is not None and val > hi:
        raise Stage1OracleError(f"{name} must be <= {hi}")
    return val


def format_exact_decimal(val: Fraction, places: int) -> str:
    """Format a non-negative or signed Fraction rounded half-up to fixed decimal places."""
    if not isinstance(val, Fraction):
        raise Stage1OracleError("val must be Fraction")
    places = _require_strict_int(places, "places", 0, 12)
    neg = val < 0
    num = -val.numerator if neg else val.numerator
    den = val.denominator
    scale = 10 ** places
    scaled = (num * scale * 2 + den) // (2 * den)
    sign = "-" if neg and scaled > 0 else ""
    if places == 0:
        return f"{sign}{scaled}"
    whole, frac = divmod(scaled, scale)
    return f"{sign}{whole}.{frac:0{places}d}"


def sigma3(n: int) -> int:
    """Exact sum of cubes of positive divisors of integer n >= 1."""
    n = _require_strict_int(n, "n", 1)
    total = 0
    root = math.isqrt(n)
    for d in range(1, root + 1):
        if n % d == 0:
            total += d * d * d
            q = n // d
            if q != d:
                total += q * q * q
    return total


def e8_shell_count(m: int) -> int:
    """Exact E_8 shell cardinality S_1(m): 1 for m=0, 240*sigma_3(m) for m >= 1."""
    m = _require_strict_int(m, "m", 0)
    if m == 0:
        return 1
    return 240 * sigma3(m)


def is_valid_e8_doubled(u: Any) -> bool:
    """Exact membership test for 2*E_8^std: 8 strict ints, uniform parity, sum(u) % 4 == 0."""
    if type(u) not in (tuple, list) or len(u) != 8:
        raise Stage1OracleError("u must be a length-8 tuple or list")
    total = 0
    par0 = -1
    for coord in u:
        if type(coord) is not int:
            raise Stage1OracleError("all coordinates of u must be strict int")
        p = coord & 1
        if par0 < 0:
            par0 = p
        elif p != par0:
            return False
        total += coord
    return (total & 3) == 0


def build_inner_dp_counts(m_max: int) -> dict[str, Any]:
    """Build exact 8->4->2 parity-tracked DP counts for E_8 shells m = 0..m_max."""
    m_max = _require_strict_int(m_max, "m_max", 1, 512)
    r_max = 2 * m_max
    t_max = m_max - 1

    even_c2 = [[0] * (r_max + 1) for _ in range(2)]
    k_lim = math.isqrt(r_max)
    for k0 in range(-k_lim, k_lim + 1):
        r0 = k0 * k0
        for k1 in range(-k_lim, k_lim + 1):
            r = r0 + k1 * k1
            if r <= r_max:
                even_c2[(k0 + k1) & 1][r] += 1

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
        even_c8[m] = sum(
            even_c4[p][a] * even_c4[p][r8 - a]
            for p in (0, 1)
            for a in range(r8 + 1)
        )

    odd_c2 = [[0] * (t_max + 1) for _ in range(2)]
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
                odd_c2[(k0 + k1) & 1][t] += 1

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
        odd_c8[m] = sum(
            odd_c4[p][a] * odd_c4[p][t8 - a]
            for p in (0, 1)
            for a in range(t8 + 1)
        )

    s1 = [even_c8[m] + odd_c8[m] for m in range(m_max + 1)]
    return {
        "m_max": m_max,
        "r_max": r_max,
        "t_max": t_max,
        "even_c2": even_c2,
        "even_c4": even_c4,
        "even_c8": even_c8,
        "odd_c2": odd_c2,
        "odd_c4": odd_c4,
        "odd_c8": odd_c8,
        "s1": s1,
    }


def build_topbit_bucket_guide(weights: Sequence[int], k_bits: int) -> dict[str, Any]:
    """Construct exact O(1) top-bit CDF-prefix bucket guide (B = 2^k_bits, b = rem >> shift)."""
    if type(weights) not in (tuple, list) or len(weights) == 0:
        raise Stage1OracleError("weights must be a non-empty list or tuple")
    k_bits = _require_strict_int(k_bits, "k_bits", 1, 10)
    cdf = [0]
    total = 0
    for w in weights:
        w_int = _require_strict_int(w, "weight", 0)
        total += w_int
        cdf.append(total)
    if total <= 0:
        raise Stage1OracleError("weights must have positive sum")

    n_items = len(weights)
    n_buckets = 1 << k_bits
    top_bits = (total - 1).bit_length()
    shift = top_bits - k_bits if top_bits > k_bits else 0

    guide_idx = [0] * n_buckets
    guide_cdf = [0] * n_buckets
    max_span = 0
    weighted_extra_num = 0
    for b in range(n_buckets):
        lo_r = b << shift
        if lo_r >= total:
            guide_idx[b] = n_items - 1
            guide_cdf[b] = cdf[n_items - 1]
            continue
        hi_r = ((b + 1) << shift) - 1
        if hi_r >= total:
            hi_r = total - 1
        a_lo = 0
        while a_lo + 1 < n_items and cdf[a_lo + 1] <= lo_r:
            a_lo += 1
        a_hi = a_lo
        while a_hi + 1 < n_items and cdf[a_hi + 1] <= hi_r:
            a_hi += 1
        guide_idx[b] = a_lo
        guide_cdf[b] = cdf[a_lo]
        span = a_hi - a_lo
        if span > max_span:
            max_span = span
        for a in range(a_lo, a_hi + 1):
            seg_lo = lo_r if lo_r > cdf[a] else cdf[a]
            seg_hi = hi_r if hi_r < cdf[a + 1] - 1 else cdf[a + 1] - 1
            if seg_hi >= seg_lo:
                weighted_extra_num += (seg_hi - seg_lo + 1) * (a - a_lo)

    return {
        "total": total,
        "k_bits": k_bits,
        "n_buckets": n_buckets,
        "shift": shift,
        "guide_idx": guide_idx,
        "guide_cdf": guide_cdf,
        "max_span": max_span,
        "expected_extra_steps": Fraction(weighted_extra_num, total),
    }


def _comb_masks(n: int, c: int) -> list[int]:
    """Generate all C(n, c) bitmasks of length n with popcount c in deterministic order."""
    out: list[int] = []

    def rec(start: int, left: int, cur: int) -> None:
        if left == 0:
            out.append(cur)
            return
        for pos in range(start, n - left + 1):
            rec(pos + 1, left - 1, cur | (1 << pos))

    rec(0, c, 0)
    return out


_COMB_TABLE: dict[tuple[int, int], list[int]] = {
    (n, c): _comb_masks(n, c)
    for n in range(1, 9)
    for c in range(1, n + 1)
}


def count_integer_partitions_of_8_and_perms() -> tuple[int, int]:
    """Return exact (partition count p(8), sum of multiset permutations across all shapes)."""
    shapes: list[tuple[int, ...]] = []

    def gen(rem: int, max_v: int, cur: list[int]) -> None:
        if rem == 0:
            shapes.append(tuple(cur))
            return
        hi = rem if rem < max_v else max_v
        for v in range(hi, 0, -1):
            cur.append(v)
            gen(rem - v, v, cur)
            cur.pop()

    gen(8, 8, [])
    total_perms = 0
    for sh in shapes:
        denom = 1
        for c in sh:
            denom *= _FACT[c]
        total_perms += _FACT[8] // denom
    return len(shapes), total_perms


def generate_d8plus_shell_leaders(m: int) -> list[tuple[int, tuple[int, ...], int, int]]:
    """Generate all D_8^+ leaders for exact E_8 shell m: (coset, doubled_leader, perm_cnt, s_bits)."""
    m = _require_strict_int(m, "m", 0, 64)
    leaders: list[tuple[int, tuple[int, ...], int, int]] = []

    def gen_even(
        idx: int,
        max_v: int,
        rem_sq: int,
        nz: int,
        cur: list[int],
        run_v: int,
        run_c: int,
        prod_f: int,
    ) -> None:
        if idx == 8:
            if rem_sq == 0:
                pf = prod_f * _FACT[run_c]
                leaders.append((0, tuple(2 * v for v in cur), _FACT[8] // pf, nz))
            return
        rem_slots = 8 - idx
        if rem_sq == 0:
            zeros = [0] * rem_slots
            pf = prod_f * (_FACT[run_c + rem_slots] if run_v == 0 else _FACT[run_c] * _FACT[rem_slots])
            leaders.append((0, tuple(2 * v for v in (cur + zeros)), _FACT[8] // pf, nz))
            return
        if rem_sq > rem_slots * max_v * max_v:
            return
        hi = math.isqrt(rem_sq)
        if max_v < hi:
            hi = max_v
        for v in range(hi, 0, -1):
            cur.append(v)
            if v == run_v:
                gen_even(idx + 1, v, rem_sq - v * v, nz + 1, cur, v, run_c + 1, prod_f)
            else:
                gen_even(idx + 1, v, rem_sq - v * v, nz + 1, cur, v, 1, prod_f * _FACT[run_c])
            cur.pop()

    gen_even(0, math.isqrt(2 * m), 2 * m, 0, [], -1, 0, 1)

    if m >= 1:
        def gen_odd(
            idx: int,
            max_v: int,
            rem_sq: int,
            cur: list[int],
            run_v: int,
            run_c: int,
            prod_f: int,
        ) -> None:
            if idx == 8:
                if rem_sq == 0:
                    pf = prod_f * _FACT[run_c]
                    leaders.append((1, tuple(cur), _FACT[8] // pf, 7))
                return
            rem_slots = 8 - idx
            if rem_sq > rem_slots * max_v * max_v:
                return
            hi = math.isqrt(rem_sq - (rem_slots - 1))
            if max_v < hi:
                hi = max_v
            if (hi & 1) == 0:
                hi -= 1
            for v in range(hi, 0, -2):
                cur.append(v)
                if v == run_v:
                    gen_odd(idx + 1, v, rem_sq - v * v, cur, v, run_c + 1, prod_f)
                else:
                    gen_odd(idx + 1, v, rem_sq - v * v, cur, v, 1, prod_f * _FACT[run_c])
                cur.pop()

        hi_od = math.isqrt(8 * m)
        if (hi_od & 1) == 0:
            hi_od -= 1
        gen_odd(0, hi_od, 8 * m, [], -1, 0, 1)

    return leaders


def unrank_d8plus_leader_orbit(
    coset: int,
    leader: Sequence[int],
    perm_cnt: int,
    s_bits: int,
    local_rank: int,
) -> tuple[int, ...]:
    """Exact zero-float unranking of a single D_8^+ leader orbit using the 502-byte comb mask table."""
    coset = _require_strict_int(coset, "coset", 0, 1)
    if type(leader) not in (tuple, list) or len(leader) != 8:
        raise Stage1OracleError("leader must be length-8 sequence")
    vals = [_require_strict_int(v, "leader_coord", 0) for v in leader]
    perm_cnt = _require_strict_int(perm_cnt, "perm_cnt", 1, 40320)
    s_bits = _require_strict_int(s_bits, "s_bits", 0, 8)
    orbit_sz = perm_cnt << s_bits
    local_rank = _require_strict_int(local_rank, "local_rank", 0, orbit_sz - 1)

    sign_bits = local_rank & ((1 << s_bits) - 1)
    perm_rank = local_rank >> s_bits

    runs: list[tuple[int, int]] = []
    for v in vals:
        if runs and runs[-1][0] == v:
            runs[-1] = (v, runs[-1][1] + 1)
        else:
            runs.append((v, 1))

    out = [0] * 8
    avail = [0, 1, 2, 3, 4, 5, 6, 7]
    for r_idx in range(len(runs) - 1):
        val, cnt = runs[r_idx]
        n_rem = len(avail)
        masks = _COMB_TABLE[(n_rem, cnt)]
        perm_rank, c_idx = divmod(perm_rank, len(masks))
        mask = masks[c_idx]
        next_avail: list[int] = []
        for pos in range(n_rem):
            slot = avail[pos]
            if (mask >> pos) & 1:
                out[slot] = val
            else:
                next_avail.append(slot)
        avail = next_avail
    last_val = runs[-1][0]
    for slot in avail:
        out[slot] = last_val

    if coset == 0:
        bpos = 0
        for i in range(8):
            if out[i] != 0:
                if (sign_bits >> bpos) & 1:
                    out[i] = -out[i]
                bpos += 1
    else:
        for i in range(7):
            if (sign_bits >> i) & 1:
                out[i] = -out[i]
        if (sum(out) & 3) != 0:
            out[7] = -out[7]
    return tuple(out)


def count_all_d8plus_leaders_up_to(m_max: int) -> dict[str, Any]:
    """Exact DP count and orbit-sum verification of all D_8^+ leaders across shells 0..m_max."""
    m_max = _require_strict_int(m_max, "m_max", 1, 512)
    max_ev = 2 * m_max
    dp_c = [[[0] * (max_ev + 1) for _ in range(2)] for _ in range(9)]
    dp_w = [[[0] * (max_ev + 1) for _ in range(2)] for _ in range(9)]
    dp_c[0][0][0] = 1
    dp_w[0][0][0] = _FACT[8]
    for v in range(1, math.isqrt(max_ev) + 1):
        v2 = v * v
        vp = v & 1
        for used in range(7, -1, -1):
            for par in (0, 1):
                src_c = dp_c[used][par]
                src_w = dp_w[used][par]
                for s in range(max_ev - v2 + 1):
                    sc = src_c[s]
                    if not sc:
                        continue
                    sw = src_w[s]
                    for c in range(1, 9 - used):
                        ns = s + c * v2
                        if ns > max_ev:
                            break
                        np = par ^ ((c * vp) & 1)
                        dp_c[used + c][np][ns] += sc
                        dp_w[used + c][np][ns] += sw // _FACT[c]

    ev_cnt = [sum(dp_c[u][0][2 * m] for u in range(9)) for m in range(m_max + 1)]
    ev_orb = [sum((dp_w[u][0][2 * m] // _FACT[8 - u]) << u for u in range(9)) for m in range(m_max + 1)]

    max_od = 8 * m_max
    od_c = [[0] * (max_od + 1) for _ in range(9)]
    od_w = [[0] * (max_od + 1) for _ in range(9)]
    od_c[0][0] = 1
    od_w[0][0] = _FACT[8]
    for v in range(1, math.isqrt(max_od) + 1, 2):
        v2 = v * v
        for used in range(7, -1, -1):
            src_c = od_c[used]
            src_w = od_w[used]
            for s in range(max_od - v2 + 1):
                sc = src_c[s]
                if not sc:
                    continue
                sw = src_w[s]
                for c in range(1, 9 - used):
                    ns = s + c * v2
                    if ns + (8 - used - c) > max_od:
                        break
                    od_c[used + c][ns] += sc
                    od_w[used + c][ns] += sw // _FACT[c]

    od_cnt = [0] + [od_c[8][8 * m] for m in range(1, m_max + 1)]
    od_orb = [0] + [od_w[8][8 * m] << 7 for m in range(1, m_max + 1)]

    all_shells_exact = all(ev_orb[m] + od_orb[m] == e8_shell_count(m) for m in range(m_max + 1))
    per_shell = [ev_cnt[m] + od_cnt[m] for m in range(m_max + 1)]
    max_per_shell = max(per_shell)
    max_shell_m = per_shell.index(max_per_shell)
    ev_tot = sum(ev_cnt)
    od_tot = sum(od_cnt)
    tot_leaders = ev_tot + od_tot
    return {
        "m_max": m_max,
        "all_shells_orbit_sum_exact": all_shells_exact,
        "even_leaders": ev_tot,
        "odd_leaders": od_tot,
        "total_leaders": tot_leaders,
        "max_leaders_in_shell": max_per_shell,
        "max_leaders_shell_m": max_shell_m,
        "leader_table_bytes_16b_each": tot_leaders * 16,
    }


def _uniform_e8_4_leaf_hit_probs(s1: Sequence[int], m_max: int) -> dict[str, Any]:
    s2 = [0] * (m_max + 1)
    for i in range(m_max + 1):
        si = s1[i]
        for j in range(m_max + 1 - i):
            s2[i + j] += si * s1[j]
    w2: list[int] = []
    acc = 0
    for v in s2:
        acc += v
        w2.append(acc)
    w3 = [0] * (m_max + 1)
    for i in range(m_max + 1):
        si = s1[i]
        for j in range(m_max + 1 - i):
            w3[i + j] += si * w2[j]
    tot4 = sum(s1[m] * w3[m_max - m] for m in range(m_max + 1))
    p2 = Fraction(sum(s1[m] * w3[m_max - m] for m in range(3)), tot4)
    p3 = Fraction(sum(s1[m] * w3[m_max - m] for m in range(4)), tot4)
    return {
        "e8_4_total_count": tot4,
        "p_leaf_m_le_2_pct": format_exact_decimal(p2 * 100, 6),
        "p_leaf_m_le_3_pct": format_exact_decimal(p3 * 100, 6),
    }


def build_stage1_certificate() -> dict[str, Any]:
    """Compute all three pure-Z/Q pillars for Track S15d (AUDIT-116)."""
    dp88 = build_inner_dp_counts(48)
    dp136 = build_inner_dp_counts(391)

    s1_88 = dp88["s1"]
    s1_136 = dp136["s1"]
    theta_88_exact = all(s1_88[m] == e8_shell_count(m) for m in range(49))
    theta_136_exact = all(s1_136[m] == e8_shell_count(m) for m in range(392))

    max_s1_88 = max(s1_88)
    max_s1_88_m = s1_88.index(max_s1_88)
    w1_88 = sum(s1_88)
    max_s1_136 = max(s1_136)
    max_s1_136_m = s1_136.index(max_s1_136)
    w1_136 = sum(s1_136)

    max_c4_88 = max(max(dp88["even_c4"][0]), max(dp88["even_c4"][1]), max(dp88["odd_c4"][0]), max(dp88["odd_c4"][1]))
    max_c2_88 = max(max(dp88["even_c2"][0]), max(dp88["even_c2"][1]), max(dp88["odd_c2"][0]), max(dp88["odd_c2"][1]))
    max_c4_136 = max(max(dp136["even_c4"][0]), max(dp136["even_c4"][1]), max(dp136["odd_c4"][0]), max(dp136["odd_c4"][1]))
    max_c2_136 = max(max(dp136["even_c2"][0]), max(dp136["even_c2"][1]), max(dp136["odd_c2"][0]), max(dp136["odd_c2"][1]))

    pillar1 = {
        "dp_matches_240_sigma3_exact": theta_88_exact and theta_136_exact,
        "ebg32_b88s8": {
            "m_max": 48,
            "max_s1": max_s1_88,
            "max_s1_shell_m": max_s1_88_m,
            "max_s1_bits": max_s1_88.bit_length(),
            "max_s1_fits_u32": max_s1_88 < (1 << 32),
            "w1_ball_count": w1_88,
            "w1_ball_bits": w1_88.bit_length(),
            "w1_ball_fits_u32": w1_88 < (1 << 32),
            "max_c4": max_c4_88,
            "max_c4_fits_u16": max_c4_88 < (1 << 16),
            "max_c2": max_c2_88,
            "max_c2_fits_u8": max_c2_88 < (1 << 8),
        },
        "ebg32_b136s8": {
            "m_max": 391,
            "max_s1": max_s1_136,
            "max_s1_shell_m": max_s1_136_m,
            "max_s1_bits": max_s1_136.bit_length(),
            "max_s1_fits_u64": max_s1_136 < (1 << 64),
            "w1_ball_count": w1_136,
            "w1_ball_bits": w1_136.bit_length(),
            "w1_ball_fits_u64": w1_136 < (1 << 64),
            "max_c4": max_c4_136,
            "max_c4_fits_u16": max_c4_136 < (1 << 16),
            "max_c2": max_c2_136,
            "max_c2_fits_u8": max_c2_136 < (1 << 8),
        },
        "pillar_verified": (
            theta_88_exact
            and theta_136_exact
            and w1_88 < (1 << 32)
            and w1_136 < (1 << 64)
            and max_c4_136 < (1 << 16)
            and max_c2_136 < (1 << 8)
        ),
    }

    # Pillar 2: O(1) Top-Bit Bucket Guide Theorem across ALL m = 0..48 (88b) + typical 136b shell m=97
    guide_all_88_valid = True
    for m in range(49):
        r8 = 2 * m
        w_even = [dp88["even_c4"][0][a] * dp88["even_c4"][0][r8 - a] for a in range(r8 + 1)]
        g = build_topbit_bucket_guide(w_even, 5)
        sh = g["shift"]
        tot = g["total"]
        if ((tot - 1) >> sh) >= 32:
            guide_all_88_valid = False
        for b in range(32):
            lo_r = b << sh
            if lo_r < tot and g["guide_cdf"][b] > lo_r:
                guide_all_88_valid = False

    w_88_typ = [dp88["even_c4"][0][a] * dp88["even_c4"][0][24 - a] for a in range(25)]
    g_88_b32 = build_topbit_bucket_guide(w_88_typ, 5)
    w_136_typ = [dp136["even_c4"][0][a] * dp136["even_c4"][0][194 - a] for a in range(195)]
    g_136_b32 = build_topbit_bucket_guide(w_136_typ, 5)
    g_136_b64 = build_topbit_bucket_guide(w_136_typ, 6)

    lut_m2_vecs = sum(s1_88[:3])
    lut_m3_vecs = sum(s1_88[:4])
    hit88 = _uniform_e8_4_leaf_hit_probs(s1_88, 48)
    hit136 = _uniform_e8_4_leaf_hit_probs(s1_136, 391)

    pillar2 = {
        "guide_88b_all_49_shells_invariant_exact": guide_all_88_valid,
        "guide_88b_m12_b32_expected_steps": format_exact_decimal(g_88_b32["expected_extra_steps"], 4),
        "guide_88b_m12_b32_max_span": g_88_b32["max_span"],
        "guide_136b_m97_b32_expected_steps": format_exact_decimal(g_136_b32["expected_extra_steps"], 4),
        "guide_136b_m97_b32_max_span": g_136_b32["max_span"],
        "guide_136b_m97_b64_expected_steps": format_exact_decimal(g_136_b64["expected_extra_steps"], 4),
        "guide_136b_m97_b64_max_span": g_136_b64["max_span"],
        "flat_lut_m_le_2_vectors": lut_m2_vecs,
        "flat_lut_m_le_2_bytes": lut_m2_vecs * 8,
        "flat_lut_m_le_3_vectors": lut_m3_vecs,
        "flat_lut_m_le_3_bytes": lut_m3_vecs * 8,
        "uniform_e8_4_88b_lut_hits": hit88,
        "uniform_e8_4_136b_lut_hits": hit136,
        "pillar_verified": guide_all_88_valid and lut_m2_vecs == 2401 and lut_m3_vecs == 9121,
    }

    # Pillar 3: Exact D_8^+ signed-leader bijection (exhaustive m = 0..3) & 88b vs 136b phase transition
    m03_all_bijective = True
    m03_total = 0
    for m in range(4):
        leaders = generate_d8plus_shell_leaders(m)
        seen: set[tuple[int, ...]] = set()
        for coset, ldr, pcnt, sbits in leaders:
            orbit = pcnt << sbits
            for rk in range(orbit):
                vec = unrank_d8plus_leader_orbit(coset, ldr, pcnt, sbits, rk)
                if not is_valid_e8_doubled(vec):
                    m03_all_bijective = False
                if sum(c * c for c in vec) != 8 * m:
                    m03_all_bijective = False
                seen.add(vec)
        if len(seen) != e8_shell_count(m):
            m03_all_bijective = False
        m03_total += len(seen)

    d8_88 = count_all_d8plus_leaders_up_to(48)
    d8_136 = count_all_d8plus_leaders_up_to(391)
    n_shapes, total_shape_perms = count_integer_partitions_of_8_and_perms()
    comb_table_bytes = sum(len(v) for v in _COMB_TABLE.values())
    blowup = Fraction(d8_136["total_leaders"], d8_88["total_leaders"])

    pillar3 = {
        "d8plus_m0_to_m3_bijective_exact": m03_all_bijective,
        "d8plus_m0_to_m3_vectors_verified": m03_total,
        "ebg32_b88s8_leaders": d8_88,
        "ebg32_b136s8_leaders": d8_136,
        "leader_blowup_136b_over_88b": format_exact_decimal(blowup, 2),
        "partitions_of_8_shapes": n_shapes,
        "universal_22_shape_perm_count": total_shape_perms,
        "universal_22_shape_pshufb_bytes": total_shape_perms * 8,
        "binomial_comb_mask_table_bytes": comb_table_bytes,
        "pillar_verified": (
            m03_all_bijective
            and m03_total == 9121
            and d8_88["all_shells_orbit_sum_exact"]
            and d8_136["all_shells_orbit_sum_exact"]
            and n_shapes == 22
            and total_shape_perms == 95503
            and comb_table_bytes == 502
        ),
    }

    overall = bool(pillar1["pillar_verified"] and pillar2["pillar_verified"] and pillar3["pillar_verified"])
    return {
        "schema_version": "audit116.e8_stage1.v1",
        "audit_id": "AUDIT-116",
        "mode": "standalone_pure_zq",
        "novelty_class": "N1",
        "pillar1_exact_bit_widths": pillar1,
        "pillar2_bucket_guide_and_lut_concentration": pillar2,
        "pillar3_d8plus_leader_bijection_and_wall": pillar3,
        "overall_verified": overall,
    }
