"""Exact zero-drift gauge group G1..G4 (RMSNorm folding, half-split RoPE, O(d_h) WHT) and verifiers."""

from __future__ import annotations

import math
import random
import struct
from typing import Any, Sequence

from ._validation import (
    HEAD_DIM_15B,
    HIDDEN_SIZE_15B,
    N_KV_HEADS_15B,
    N_LAYERS_15B,
    N_Q_HEADS_15B,
    require,
)

NORM_FOLD_MIN_ABS = 1.0e-4


def bf16_round_f64(value: float) -> float:
    """Round a finite IEEE-754 float64 value to IEEE bfloat16 (round-to-nearest-even) and back to float64."""
    v = float(value)
    require(math.isfinite(v), "BF16_NONFINITE", f"non-finite input: {value!r}")
    f32_bytes = struct.pack("<f", v)
    u32 = struct.unpack("<I", f32_bytes)[0]
    if (u32 & 0x7FFFFFFF) == 0:
        return 0.0 if (u32 >> 31) == 0 else -0.0
    if (u32 & 0x7F800000) == 0x7F800000:
        return struct.unpack("<f", struct.pack("<I", u32 & 0xFFFF0000))[0]
    lsb = (u32 >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    bf_u32 = (u32 + rounding_bias) & 0xFFFF0000
    return float(struct.unpack("<f", struct.pack("<I", bf_u32))[0])


def hadamard_norm_py(n: int) -> list[list[float]]:
    """Return normalized n x n Sylvester Walsh-Hadamard orthogonal matrix H_n / sqrt(n)."""
    require(n >= 1 and (n & (n - 1)) == 0, "BAD_H_DIM", f"H dimension must be power of 2, got {n}")
    h = [[1.0]]
    cur = 1
    while cur < n:
        nh = [[0.0] * (2 * cur) for _ in range(2 * cur)]
        for i in range(cur):
            for j in range(cur):
                v = h[i][j]
                nh[i][j] = v
                nh[i][j + cur] = v
                nh[i + cur][j] = v
                nh[i + cur][j + cur] = -v
        h = nh
        cur *= 2
    inv_sq = 1.0 / math.sqrt(float(n))
    return [[v * inv_sq for v in row] for row in h]


def _matvec_py(m: Sequence[Sequence[float]], x: Sequence[float], bias: Sequence[float] | None = None) -> list[float]:
    out = [sum(row[j] * x[j] for j in range(len(x))) for row in m]
    if bias is not None:
        out = [out[i] + bias[i] for i in range(len(out))]
    return out


def _matmul_py(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    r, k, c = len(a), len(b), len(b[0])
    out = [[0.0] * c for _ in range(r)]
    for i in range(r):
        for t in range(k):
            at = a[i][t]
            if at != 0.0:
                for j in range(c):
                    out[i][j] += at * b[t][j]
    return out


def rmsnorm_py(x: Sequence[float], gamma: Sequence[float], eps: float = 1e-6) -> list[float]:
    """Pure-Python RMSNorm reference."""
    require(len(x) > 0 and len(x) == len(gamma), "RMS_LEN", "x and gamma length mismatch")
    mean_sq = sum(v * v for v in x) / float(len(x))
    inv_r = 1.0 / math.sqrt(mean_sq + eps)
    return [x[i] * inv_r * gamma[i] for i in range(len(x))]


def rope_half_split_py(vec: Sequence[float], cos_half: Sequence[float], sin_half: Sequence[float]) -> list[float]:
    """Apply half-split RoPE rotation on pairs `(k, k + hd/2)`."""
    hd = len(vec)
    require(hd >= 2 and (hd % 2) == 0, "ROPE_DIM", f"head_dim must be even, got {hd}")
    half = hd // 2
    require(len(cos_half) == half and len(sin_half) == half, "ROPE_FREQ_LEN", "cos/sin length mismatch")
    out = [0.0] * hd
    for k in range(half):
        x1, x2 = vec[k], vec[k + half]
        c, s = cos_half[k], sin_half[k]
        out[k] = x1 * c - x2 * s
        out[k + half] = x1 * s + x2 * c
    return out


def fold_rmsnorm_bf16_exact_py(
    gamma: Sequence[float],
    act_x2: Sequence[float],
    w_col_sq_mean: Sequence[float],
    *,
    eps: float = 1e-8,
    clamp_min: float = 0.1,
    clamp_max: float = 10.0,
) -> tuple[list[float], list[float], float]:
    """Compute BF16-compensated RMSNorm folding `(tilde_gamma, d_ex, max_fold_err)`.

    When `|gamma_j| >= NORM_FOLD_MIN_ABS`, rounds `gamma_j / d_j` to IEEE bfloat16 storage
    and defines `d_ex_j = gamma_j / tilde_gamma_j`, eliminating bfloat16 norm truncation drift.
    """
    n = len(gamma)
    require(len(act_x2) == n and len(w_col_sq_mean) == n, "FOLD_LEN", "input vector length mismatch")
    tilde_gamma: list[float] = []
    d_ex: list[float] = []
    max_err = 0.0
    for j in range(n):
        g = float(gamma[j])
        ax = max(0.0, float(act_x2[j]))
        w2 = max(0.0, float(w_col_sq_mean[j]))
        d = ((ax + eps) / (w2 + eps)) ** 0.25
        d = max(clamp_min, min(clamp_max, d))
        if abs(g) >= NORM_FOLD_MIN_ABS:
            tg = bf16_round_f64(g / d)
            if abs(tg) < 1e-30:
                tg = g
                dex = 1.0
            else:
                dex = g / tg
        else:
            tg = g
            dex = 1.0
        tilde_gamma.append(tg)
        d_ex.append(dex)
        max_err = max(max_err, abs(tg * dex - g))
    return tilde_gamma, d_ex, max_err


def check_global_residual_gauge_obstruction(
    gamma_final: Sequence[float],
    q_orth: Sequence[Sequence[float]],
) -> float:
    """Measure commutator ||Q diag(gamma_final) Q^T - diag(gamma_final)||_max obstructing tied residual rotation."""
    d = len(gamma_final)
    require(len(q_orth) == d and all(len(row) == d for row in q_orth), "OBS_SHAPE", "Q must be d x d")
    diag_q = [[q_orth[i][k] * float(gamma_final[k]) for k in range(d)] for i in range(d)]
    qt = [[q_orth[j][i] for j in range(d)] for i in range(d)]
    conj = _matmul_py(diag_q, qt)
    max_diff = 0.0
    for i in range(d):
        for j in range(d):
            target = float(gamma_final[i]) if i == j else 0.0
            max_diff = max(max_diff, abs(conj[i][j] - target))
    return max_diff


def verify_gauge_group_algebra_py(seed: int = 182001) -> dict[str, Any]:
    """Verify G1..G4 exact zero-drift invariance, BF16 folding parity, H4^2=I4, and global obstructions."""
    rng = random.Random(seed)
    d = 16
    hd = 8
    n_qh = 2
    n_kh = 1
    dm = 16

    x_tok = [rng.gauss(0.0, 0.7) for _ in range(d)]
    g_in = [bf16_round_f64(rng.uniform(0.5, 1.8)) for _ in range(d)]
    wq = [[rng.gauss(0.0, 0.25) for _ in range(d)] for _ in range(n_qh * hd)]
    bq = [rng.gauss(0.0, 0.1) for _ in range(n_qh * hd)]
    wk = [[rng.gauss(0.0, 0.25) for _ in range(d)] for _ in range(n_kh * hd)]
    bk = [rng.gauss(0.0, 0.1) for _ in range(n_kh * hd)]
    wv = [[rng.gauss(0.0, 0.25) for _ in range(d)] for _ in range(n_kh * hd)]
    bv = [rng.gauss(0.0, 0.1) for _ in range(n_kh * hd)]
    wo = [[rng.gauss(0.0, 0.25) for _ in range(n_qh * hd)] for _ in range(d)]
    g_post = [bf16_round_f64(rng.uniform(0.5, 1.8)) for _ in range(d)]
    wgate = [[rng.gauss(0.0, 0.25) for _ in range(d)] for _ in range(dm)]
    wup = [[rng.gauss(0.0, 0.25) for _ in range(d)] for _ in range(dm)]
    wdown = [[rng.gauss(0.0, 0.25) for _ in range(dm)] for _ in range(d)]

    cos_h = [math.cos(0.17 * (k + 1)) for k in range(hd // 2)]
    sin_h = [math.sin(0.17 * (k + 1)) for k in range(hd // 2)]
    cos_v = [math.cos(0.31 * (k + 1)) for k in range(hd // 2)]
    sin_v = [math.sin(0.31 * (k + 1)) for k in range(hd // 2)]

    def _run_block(
        in_g: Sequence[float],
        mq: Sequence[Sequence[float]],
        vbq: Sequence[float],
        mk: Sequence[Sequence[float]],
        vbk: Sequence[float],
        mv: Sequence[Sequence[float]],
        vbv: Sequence[float],
        mo: Sequence[Sequence[float]],
        post_g: Sequence[float],
        mgate: Sequence[Sequence[float]],
        mup: Sequence[Sequence[float]],
        mdown: Sequence[Sequence[float]],
    ) -> list[float]:
        xn = rmsnorm_py(x_tok, in_g)
        q = _matvec_py(mq, xn, vbq)
        k = _matvec_py(mk, xn, vbk)
        v = _matvec_py(mv, xn, vbv)
        q_r = [rope_half_split_py(q[h * hd : (h + 1) * hd], cos_h, sin_h) for h in range(n_qh)]
        k_r = rope_half_split_py(k, cos_h, sin_h)
        k_r2 = rope_half_split_py(k_r, cos_v, sin_v)
        ctx: list[float] = []
        for h in range(n_qh):
            q_r2 = rope_half_split_py(q_r[h], cos_v, sin_v)
            score = sum(q_r2[j] * k_r2[j] for j in range(hd)) / math.sqrt(float(hd))
            att_w = math.tanh(score * 0.1) + 1.0
            ctx.extend([att_w * v[j] for j in range(hd)])
        h1 = [x_tok[i] + _matvec_py(mo, ctx)[i] for i in range(d)]
        h1n = rmsnorm_py(h1, post_g)
        g_vec = _matvec_py(mgate, h1n)
        u_vec = _matvec_py(mup, h1n)
        act = [(g_vec[i] / (1.0 + math.exp(-g_vec[i]))) * u_vec[i] for i in range(dm)]
        return [h1[i] + _matvec_py(mdown, act)[i] for i in range(d)]

    ref_out = _run_block(g_in, wq, bq, wk, bk, wv, bv, wo, g_post, wgate, wup, wdown)

    # (G1) BF16-exact pre-attention RMSNorm folding
    act_attn = [rng.uniform(0.2, 2.5) for _ in range(d)]
    w2_attn = [
        (sum(wq[i][j] ** 2 for i in range(len(wq))) + sum(wk[i][j] ** 2 for i in range(len(wk))) + sum(wv[i][j] ** 2 for i in range(len(wv))))
        / float(len(wq) + len(wk) + len(wv))
        for j in range(d)
    ]
    g_in_1, d_in_ex, g1_bf16_fold_err = fold_rmsnorm_bf16_exact_py(g_in, act_attn, w2_attn)
    wq_1 = [[wq[i][j] * d_in_ex[j] for j in range(d)] for i in range(len(wq))]
    wk_1 = [[wk[i][j] * d_in_ex[j] for j in range(d)] for i in range(len(wk))]
    wv_1 = [[wv[i][j] * d_in_ex[j] for j in range(d)] for i in range(len(wv))]

    # (G3) Half-split RoPE (k, k + hd/2) pair scaling Q <-> K
    d_pair = [rng.uniform(0.75, 1.35) for _ in range(hd // 2)]
    d_q_pair = [d_pair[k % (hd // 2)] for k in range(hd)] * n_qh
    d_k_pair = [1.0 / d_pair[k % (hd // 2)] for k in range(hd)]
    wq_2 = [[wq_1[i][j] * d_q_pair[i] for j in range(d)] for i in range(len(wq_1))]
    bq_2 = [bq[i] * d_q_pair[i] for i in range(len(bq))]
    wk_2 = [[wk_1[i][j] * d_k_pair[i] for j in range(d)] for i in range(len(wk_1))]
    bk_2 = [bk[i] * d_k_pair[i] for i in range(len(bk))]

    # (G4) O(hd) Walsh-Hadamard + D_v head gauge on v_proj <-> o_proj
    h_hd = hadamard_norm_py(hd)
    d_v_hd = [rng.uniform(0.75, 1.35) for _ in range(hd)]
    p_v = _matmul_py(h_hd, [[d_v_hd[i] if i == j else 0.0 for j in range(hd)] for i in range(hd)])
    p_v_inv = [[h_hd[j][i] / d_v_hd[i] for j in range(hd)] for i in range(hd)]
    wv_3 = _matmul_py(p_v, wv_1)
    bv_3 = _matvec_py(p_v, bv)
    wo_3 = [[0.0] * (n_qh * hd) for _ in range(d)]
    for h_idx in range(n_qh):
        off = h_idx * hd
        blk_o = [[wo[r][off + c] for c in range(hd)] for r in range(d)]
        blk_o3 = _matmul_py(blk_o, p_v_inv)
        for r in range(d):
            for c in range(hd):
                wo_3[r][off + c] = blk_o3[r][c]

    # (G2) BF16-exact pre-MLP RMSNorm folding + SiLU up<->down diagonal gauge
    act_mlp = [rng.uniform(0.2, 2.5) for _ in range(d)]
    w2_mlp = [
        (sum(wgate[i][j] ** 2 for i in range(dm)) + sum(wup[i][j] ** 2 for i in range(dm)))
        / float(2 * dm)
        for j in range(d)
    ]
    g_post_4, d_post_ex, g2_bf16_fold_err = fold_rmsnorm_bf16_exact_py(g_post, act_mlp, w2_mlp)
    wgate_4 = [[wgate[i][j] * d_post_ex[j] for j in range(d)] for i in range(dm)]
    wup_4 = [[wup[i][j] * d_post_ex[j] for j in range(d)] for i in range(dm)]
    d_mlp = [rng.uniform(0.75, 1.35) for _ in range(dm)]
    wup_4b = [[wup_4[i][j] * d_mlp[i] for j in range(d)] for i in range(dm)]
    wdown_4 = [[wdown[r][j] / d_mlp[j] for j in range(dm)] for r in range(d)]

    gauge_out = _run_block(g_in_1, wq_2, bq_2, wk_2, bk_2, wv_3, bv_3, wo_3, g_post_4, wgate_4, wup_4b, wdown_4)
    g14_max_abs = max(abs(a - b) for a, b in zip(ref_out, gauge_out))
    require(g14_max_abs < 1e-12, "GAUGE_ERR", f"G1..G4 error too large: {g14_max_abs}")

    h4 = hadamard_norm_py(4)
    h4_sq = _matmul_py(h4, h4)
    h4_sq_id_err = max(abs(h4_sq[i][j] - (1.0 if i == j else 0.0)) for i in range(4) for j in range(4))
    require(h4_sq_id_err < 1e-12, "H4_CANCEL", "H4^2 != I4")

    h_d = hadamard_norm_py(d)
    r_cos = [cos_h[k % (hd // 2)] for k in range(d // 2)]
    r_sin = [sin_h[k % (hd // 2)] for k in range(d // 2)]
    rx = rope_half_split_py(x_tok, r_cos, r_sin)
    hx = _matvec_py(h_d, x_tok)
    rhx = rope_half_split_py(hx, r_cos, r_sin)
    hrx = _matvec_py(h_d, rx)
    global_h_rope_comm_err = max(abs(rhx[i] - hrx[i]) for i in range(d))
    require(global_h_rope_comm_err > 0.1, "ROPE_COMM_CHECK", "expected global H non-commutation with RoPE")

    gamma_nonconst = [0.25 + 0.4 * float(i) for i in range(d)]
    residual_norm_obstruction = check_global_residual_gauge_obstruction(gamma_nonconst, h_d)
    require(residual_norm_obstruction > 0.5, "TIED_RESIDUAL_OBS", "expected non-zero tied norm obstruction")

    return {
        "g1_to_g4_exact_max_abs_err": g14_max_abs,
        "g1_bf16_fold_max_abs_err": g1_bf16_fold_err,
        "g2_bf16_fold_max_abs_err": g2_bf16_fold_err,
        "h4_squared_identity_err": h4_sq_id_err,
        "o_proj_axis_required_after_head_wht": True,
        "global_hadamard_rope_commutator_max_abs": global_h_rope_comm_err,
        "tied_residual_norm_obstruction_max_abs": residual_norm_obstruction,
    }


def hadamard_norm_torch(n: int, device: Any, dtype: Any) -> Any:
    """Return normalized n x n Sylvester Walsh-Hadamard matrix in PyTorch."""
    import torch

    require(n >= 1 and (n & (n - 1)) == 0, "BAD_H_DIM", f"H dimension must be power of 2, got {n}")
    h = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    while int(h.shape[0]) < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(float(n))).to(dtype)


def apply_gauges_layer(
    layer_tensors: dict[str, Any],
    pref: str,
    moments: dict[str, dict[str, Any]],
    h_head: Any,
    *,
    norm_bf16_round: bool = True,
    bias_bf16_round: bool | None = None,
    hidden_size: int = HIDDEN_SIZE_15B,
    n_q_heads: int = N_Q_HEADS_15B,
    n_kv_heads: int = N_KV_HEADS_15B,
    head_dim: int = HEAD_DIM_15B,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply exact (G1..G4) zero-drift gauges to one transformer decoder layer."""
    import torch

    require(n_q_heads % n_kv_heads == 0, "GQA_DIV", f"n_q_heads={n_q_heads} % n_kv_heads={n_kv_heads} != 0")
    require(head_dim % 2 == 0 and int(h_head.shape[0]) == head_dim, "HD_GEOM", f"head_dim={head_dim}")
    gqa_group = n_q_heads // n_kv_heads
    kv_dim = n_kv_heads * head_dim
    half_hd = head_dim // 2
    calc_dt = torch.float32 if norm_bf16_round else torch.float64
    round_bias = norm_bf16_round if bias_bf16_round is None else bool(bias_bf16_round)
    h_mat = h_head.to(calc_dt)
    act_x2 = moments["act_x2"]
    qk_out2 = moments["qk_out2"]

    iln = layer_tensors["input_layernorm.weight"].to(calc_dt)
    paln = layer_tensors["post_attention_layernorm.weight"].to(calc_dt)
    wq = layer_tensors["self_attn.q_proj.weight"].to(calc_dt)
    wk = layer_tensors["self_attn.k_proj.weight"].to(calc_dt)
    wv = layer_tensors["self_attn.v_proj.weight"].to(calc_dt)
    wo = layer_tensors["self_attn.o_proj.weight"].to(calc_dt)
    bq = layer_tensors.get("self_attn.q_proj.bias")
    bk = layer_tensors.get("self_attn.k_proj.bias")
    bv = layer_tensors.get("self_attn.v_proj.bias")
    bq_t = bq.to(calc_dt) if bq is not None else torch.zeros(n_q_heads * head_dim, device=wq.device, dtype=calc_dt)
    bk_t = bk.to(calc_dt) if bk is not None else torch.zeros(kv_dim, device=wk.device, dtype=calc_dt)
    bv_t = bv.to(calc_dt) if bv is not None else torch.zeros(kv_dim, device=wv.device, dtype=calc_dt)
    wgate = layer_tensors["mlp.gate_proj.weight"].to(calc_dt)
    wup = layer_tensors["mlp.up_proj.weight"].to(calc_dt)

    ax_attn = act_x2[f"{pref}.self_attn.q_proj"].to(calc_dt)
    ax_o = act_x2[f"{pref}.self_attn.o_proj"].to(calc_dt)
    ax_mlp = act_x2[f"{pref}.mlp.gate_proj"].to(calc_dt)
    ax_down = act_x2[f"{pref}.mlp.down_proj"].to(calc_dt)

    w2_in = (wq.pow(2).sum(0) + wk.pow(2).sum(0) + wv.pow(2).sum(0)) / float(
        wq.shape[0] + wk.shape[0] + wv.shape[0]
    )
    d_in = ((ax_attn + 1e-8) / (w2_in + 1e-8)).pow(0.25).clamp(0.1, 10.0)
    m_in = iln.abs() >= NORM_FOLD_MIN_ABS
    iln_new = iln.clone()
    if norm_bf16_round:
        iln_new[m_in] = (iln[m_in] / d_in[m_in]).to(torch.bfloat16).to(calc_dt)
    else:
        iln_new[m_in] = iln[m_in] / d_in[m_in]
    d_in_ex = torch.ones_like(d_in)
    d_in_ex[m_in] = iln[m_in] / iln_new[m_in]
    ax_attn_g = ax_attn / (d_in_ex ** 2)
    wq = wq * d_in_ex.unsqueeze(0)
    wk = wk * d_in_ex.unsqueeze(0)
    wv = wv * d_in_ex.unsqueeze(0)

    w2_mlp = (wgate.pow(2).sum(0) + wup.pow(2).sum(0)) / float(wgate.shape[0] + wup.shape[0])
    d_mlp = ((ax_mlp + 1e-8) / (w2_mlp + 1e-8)).pow(0.25).clamp(0.1, 10.0)
    m_mlp = paln.abs() >= NORM_FOLD_MIN_ABS
    paln_new = paln.clone()
    if norm_bf16_round:
        paln_new[m_mlp] = (paln[m_mlp] / d_mlp[m_mlp]).to(torch.bfloat16).to(calc_dt)
    else:
        paln_new[m_mlp] = paln[m_mlp] / d_mlp[m_mlp]
    d_mlp_ex = torch.ones_like(d_mlp)
    d_mlp_ex[m_mlp] = paln[m_mlp] / paln_new[m_mlp]
    ax_mlp_g = ax_mlp / (d_mlp_ex ** 2)
    wgate = wgate * d_mlp_ex.unsqueeze(0)
    wup = wup * d_mlp_ex.unsqueeze(0)

    aq = qk_out2[f"{pref}.self_attn.q_proj"].to(calc_dt).view(n_kv_heads, gqa_group, head_dim)
    ak = qk_out2[f"{pref}.self_attn.k_proj"].to(calc_dt).view(n_kv_heads, head_dim)
    aq_p = (aq[:, :, :half_hd] + aq[:, :, half_hd:]).mean(dim=1)
    ak_p = ak[:, :half_hd] + ak[:, half_hd:]
    s_qk = (ak_p / (aq_p + 1e-8)).pow(0.25).clamp(0.2, 5.0)
    s_kv = torch.cat([s_qk, s_qk], dim=1)
    s_q = s_kv.repeat_interleave(gqa_group, dim=0).reshape(n_q_heads * head_dim)
    s_k = s_kv.reshape(kv_dim)
    wq = wq * s_q.unsqueeze(1)
    bq_t = bq_t * s_q
    wk = wk / s_k.unsqueeze(1)
    bk_t = bk_t / s_k

    av_g = qk_out2[f"{pref}.self_attn.v_proj"].to(calc_dt).view(n_kv_heads, head_dim)
    wo_col2 = wo.view(hidden_size, n_kv_heads, gqa_group, head_dim).pow(2).mean(dim=(0, 2))
    dv = ((av_g + 1e-8) / (wo_col2 + 1e-8)).pow(0.25).clamp(0.1, 10.0)
    wv = torch.einsum("ab,gbc->gac", h_mat.T, wv.view(n_kv_heads, head_dim, hidden_size) / dv.unsqueeze(-1)).reshape(
        kv_dim, hidden_size
    )
    bv_t = torch.einsum("ab,gb->ga", h_mat.T, bv_t.view(n_kv_heads, head_dim) / dv).reshape(kv_dim)
    wo = torch.einsum(
        "mhc,cd->mhd",
        wo.view(hidden_size, n_q_heads, head_dim) * dv.repeat_interleave(gqa_group, dim=0).unsqueeze(0),
        h_mat,
    ).reshape(hidden_size, n_q_heads * head_dim)
    ax_o_g4 = (
        (ax_o.view(n_q_heads, head_dim) / (dv.repeat_interleave(gqa_group, dim=0) ** 2))
        .mean(dim=1, keepdim=True)
        .expand(n_q_heads, head_dim)
        .reshape(n_q_heads * head_dim)
    )

    out_sd: dict[str, Any] = {
        "input_layernorm.weight": iln_new.to(torch.bfloat16).to(calc_dt) if norm_bf16_round else iln_new,
        "post_attention_layernorm.weight": paln_new.to(torch.bfloat16).to(calc_dt) if norm_bf16_round else paln_new,
        "self_attn.q_proj.weight": wq,
        "self_attn.k_proj.weight": wk,
        "self_attn.v_proj.weight": wv,
        "self_attn.o_proj.weight": wo,
        "mlp.gate_proj.weight": wgate,
        "mlp.up_proj.weight": wup,
        "mlp.down_proj.weight": layer_tensors["mlp.down_proj.weight"].to(calc_dt),
    }
    if bq is not None:
        out_sd["self_attn.q_proj.bias"] = bq_t.to(torch.bfloat16).to(calc_dt) if round_bias else bq_t
    if bk is not None:
        out_sd["self_attn.k_proj.bias"] = bk_t.to(torch.bfloat16).to(calc_dt) if round_bias else bk_t
    if bv is not None:
        out_sd["self_attn.v_proj.bias"] = bv_t.to(torch.bfloat16).to(calc_dt) if round_bias else bv_t

    ax_layer = {
        "self_attn.q_proj": ax_attn_g.float(),
        "self_attn.k_proj": ax_attn_g.float(),
        "self_attn.v_proj": ax_attn_g.float(),
        "self_attn.o_proj": ax_o_g4.float(),
        "mlp.gate_proj": ax_mlp_g.float(),
        "mlp.up_proj": ax_mlp_g.float(),
        "mlp.down_proj": ax_down.float(),
    }
    return out_sd, ax_layer


def apply_reverse_functor_gauges(
    state_dict_cpu: dict[str, Any],
    moments: dict[str, dict[str, Any]],
    *,
    norm_bf16_round: bool = True,
    n_layers: int = N_LAYERS_15B,
    hidden_size: int = HIDDEN_SIZE_15B,
    n_q_heads: int = N_Q_HEADS_15B,
    n_kv_heads: int = N_KV_HEADS_15B,
    head_dim: int = HEAD_DIM_15B,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Apply (G1..G4) gauges across all layers in `state_dict_cpu` and transform moments."""
    import torch

    calc_dt = torch.float32 if norm_bf16_round else torch.float64
    sd = {k: (v.clone() if k.startswith("model.layers.") else v) for k, v in state_dict_cpu.items()}
    h_head = hadamard_norm_torch(head_dim, torch.device("cpu"), calc_dt)
    trans_ax: dict[str, dict[str, Any]] = {}

    for l_idx in range(n_layers):
        pref = f"model.layers.{l_idx}"
        layer_in = {k[len(pref) + 1 :]: v for k, v in sd.items() if k.startswith(pref + ".")}
        out_l, ax_l = apply_gauges_layer(
            layer_in,
            pref,
            moments,
            h_head,
            norm_bf16_round=norm_bf16_round,
            hidden_size=hidden_size,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        for sub_k, val in out_l.items():
            full_k = f"{pref}.{sub_k}"
            sd[full_k] = val.to(sd[full_k].dtype) if norm_bf16_round else val
        trans_ax[pref] = ax_l

    return sd, trans_ax
