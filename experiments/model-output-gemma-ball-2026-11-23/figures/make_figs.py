#!/usr/bin/env python3
"""Draw the quantization geometries of experiments/model-output-gemma-ball-2026-11-23/run.py.

Everything is synthetic/small — no model weights, no checkpoints. Figures:

  fig1_codebook_ball.png   — the "integer ball" codebook: lattice points with ||y||^2 <= bound
  fig2_scale_and_snap.png  — per-block scale t: shrink until x/t rounds inside the ball
  fig3_arm_decode.png      — how an arm name decodes (ZBG128:b560s16)
  fig4_e8_voronoi.png      — E8 Voronoi code as a 2-D hexagonal analogy + centred vs zero-centred
  fig5_hadamard.png        — the order-8 Hadamard turn spreading an outlier
  fig6_scales_ladder.png   — log2 scale ladder, per-group vs per-row scale, imatrix ellipsoid
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, RegularPolygon, FancyArrowPatch

ROOT = Path("/home/ubuntu/repos/siliquant")
sys.path.insert(0, str(ROOT / "experiments" / "ball-dimension-no-clip-rule-2026-11-17"))
import run as bd  # noqa: E402

OUT = Path("/home/ubuntu/viz")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# ---------------------------------------------------------------- fig 1
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: 2-D integer ball codebook
    bound2 = 25
    pts = [(i, j) for i in range(-7, 8) for j in range(-7, 8)]
    ax1.set_aspect("equal")
    ax1.add_patch(Circle((0, 0), math.sqrt(bound2), fill=True, alpha=0.10,
                         color="tab:blue", zorder=0))
    ax1.add_patch(Circle((0, 0), math.sqrt(bound2), fill=False, lw=2,
                         color="tab:blue", zorder=1))
    inside = [(i, j) for i, j in pts if i * i + j * j <= bound2]
    outside = [(i, j) for i, j in pts if i * i + j * j > bound2]
    ax1.scatter(*zip(*outside), s=14, c="0.75", zorder=2)
    ax1.scatter(*zip(*inside), s=26, c="tab:blue", zorder=3)
    ax1.scatter([0], [0], s=60, c="crimson", marker="*", zorder=4)
    ax1.set_title("Codebook = lattice points in a ball\n(toy 2-D cut: $\\|y\\|^2 \\leq 25$)")
    ax1.set_xlim(-7.4, 7.4); ax1.set_ylim(-7.4, 7.4)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.grid(alpha=0.25)
    ax1.annotate(f"{len(inside)} codepoints", (-7.1, -7.0), fontsize=10)

    # right: exact shell counts of Z^8 and E8 per squared radius
    limit = 30
    z8 = bd.shells8("Z8", "ball", limit)
    e8 = bd.shells8("E8", "ball", limit)
    r = np.arange(limit + 1)
    ax2.bar(r - 0.2, z8, width=0.4, label="Z$^8$ shells", color="tab:blue")
    ax2.bar(r + 0.2, e8, width=0.4, label="E8 shells", color="tab:orange")
    ax2.set_yscale("log")
    ax2.set_xlabel("squared radius $\\|y\\|^2$")
    ax2.set_ylabel("points on the shell")
    ax2.set_title("Shell counts, exact $\\Theta$ series\n(run.py counts these to find the bound)")
    ax2.legend(frameon=False)
    ax2.set_xlim(-0.8, limit + 0.5)

    fig.suptitle("The integer-ball codebook (ZBG / EBG / HBG arms): every lattice point with "
                 "$\\|y\\|^2 \\leq$ bound", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig1_codebook_ball.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig2():
    rng = np.random.default_rng(7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: scale-and-snap in 2-D
    bound2 = 25
    R = math.sqrt(bound2)
    x = np.array([6.9, 4.7])
    xn = np.linalg.norm(x)
    # smallest t so that round(x/t) is inside: t ≈ ||x||/R here by construction
    t = xn / R * 1.02
    xr = x / t
    y = np.rint(xr)
    ax1.set_aspect("equal")
    ax1.add_patch(Circle((0, 0), R, fill=True, alpha=0.10, color="tab:blue"))
    ax1.add_patch(Circle((0, 0), R, fill=False, lw=2, color="tab:blue"))
    gpts = [(i, j) for i in range(-8, 9) for j in range(-8, 9)]
    ax1.scatter(*zip(*gpts), s=10, c="0.75")
    ax1.scatter(*x, s=90, c="crimson", zorder=5, label="block x (outside)")
    ax1.scatter(*xr, s=90, c="tab:green", zorder=5, label="x / t (inside)")
    ax1.scatter(*y, s=90, facecolors="none", edgecolors="tab:blue", lw=2,
                zorder=5, label="nearest lattice pt y")
    ax1.add_patch(FancyArrowPatch(x, xr, arrowstyle="-|>", mutation_scale=15,
                                  color="tab:green", lw=1.5))
    ax1.add_patch(FancyArrowPatch(xr, y, arrowstyle="-|>", mutation_scale=15,
                                  color="tab:blue", lw=1.5, linestyle="--"))
    ax1.annotate("divide by t", (5.4, 3.4), color="tab:green")
    ax1.annotate("round", (4.3, 3.9), color="tab:blue")
    ax1.set_xlim(-8.6, 8.6); ax1.set_ylim(-8.6, 8.6)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, loc="lower right", fontsize=9)
    ax1.set_title("Each block picks the smallest scale t\nso its rounding lands in the ball")

    # right: the bisection + overshoot/shrink path on a real 128-dim block
    n = 128
    W = rng.standard_normal(n)
    X = W[None, :]
    bound, _ = bd.codebook("Z8", "ball", n, 2 ** (560 // n))  # placeholder; use real bound below
    bound = 3380  # real ZBG128:b560 bound (verified)
    t_real = bd.block_scales(X, "Z8", "ball", bound)[0]
    Y = np.rint(X / t_real)
    sizes = []
    ts = np.linspace(0.55 * t_real, 1.25 * t_real, 60)
    for tv in ts:
        sizes.append(float((np.rint(X / tv) ** 2).sum()))
    ax2.plot(ts / t_real, sizes, color="tab:blue")
    ax2.axhline(bound, color="crimson", ls="--", lw=1.5,
                label=f"bound = {bound}")
    ax2.axvline(1.0, color="tab:green", ls=":", lw=1.5, label="bisected t")
    ax2.set_xlabel("scale ÷ chosen t")
    ax2.set_ylabel("$\\|\\mathrm{round}(x/t)\\|^2$")
    ax2.set_title("Bisection finds t where the block\njust lands inside $\\|y\\|^2 \\leq 3380$")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("Scaling one 128-weight block into the ball (block_scales in run.py)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig2_scale_and_snap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig3():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.4)

    def chip(x, w, text, color, sub=""):
        ax.add_patch(plt.Rectangle((x, 1.35), w, 1.0, facecolor=color, alpha=0.18,
                                   edgecolor=color, lw=1.6))
        ax.text(x + w / 2, 1.85, text, ha="center", va="center", fontsize=15,
                fontweight="bold", color=color)
        if sub:
            ax.text(x + w / 2, 0.95, sub, ha="center", va="center", fontsize=9.5, color="0.2")

    ax.text(0.3, 3.05, "ZBG128:b560s16", fontsize=17, fontweight="bold",
            fontfamily="monospace")
    chip(0.4, 1.7, "ZBG", "tab:blue", "Z8 lattice,\nball shape")
    chip(2.4, 1.5, "128", "tab:purple", "block of\n128 weights")
    chip(4.2, 1.9, "b560", "tab:green",
         "codebook ≤ 2^560 pts\n(→ ||y||² ≤ 3380)")
    chip(6.4, 1.5, "s16", "tab:orange", "scale t stored\nin 16 bits")
    ax.text(8.6, 1.85, "= (log2 of ~2^560) / 128 + 16/128\n≈ 4.5 bits per weight",
            fontsize=11, va="center")
    ax.text(0.4, 0.25,
            "Prefix variants: h… = imatrix-weighted columns (+0.25-power), i… = +0.5,  "
            "EBG = E8 lattice,  HBG/HVE8 = Hadamard-8 turn first,\n"
            "+K10 = keep 10 loud ffn_down columns in fp16,  ~rand = random-text imatrix,  "
            "ZB128@R:16 = one scale per whole row,  VE8Z32:16 = E8 Voronoi code, q=16, group 32.",
            fontsize=9.5, va="top", color="0.2")
    fig.savefig(OUT / "fig3_arm_decode.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: hexagonal (A2) lattice + Voronoi cells as E8 analogy
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, math.sqrt(3) / 2])
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            if -3.5 < p[0] < 3.5 and -3.5 < p[1] < 3.5:
                pts.append(p)
    for p in pts:
        ax1.add_patch(RegularPolygon(p, 6, radius=a2[1] * math.sqrt(3) / 3 * 2 / 2,
                                     orientation=0, fill=False, edgecolor="0.7",
                                     lw=0.6))
    ax1.scatter(*zip(*pts), s=22, c="tab:orange", zorder=3)
    ax1.add_patch(RegularPolygon((0, 0), 6, radius=0.5, fill=True,
                                 facecolor="tab:orange", alpha=0.18, edgecolor="tab:orange", lw=1.6))
    x = np.array([0.7, 0.15])
    ax1.scatter(*x, s=80, c="crimson", zorder=5)
    ax1.annotate("x/t maps into\nthe home cell", x + np.array([0.12, 0.35]), fontsize=9)
    ax1.set_aspect("equal")
    ax1.set_xlim(-3.6, 3.6); ax1.set_ylim(-3.6, 3.6)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("E8 Voronoi code (VE8 arms), 2-D hexagonal analogy:\n"
                "label = which lattice translate contains the point")

    # right: centred vs zero-centred codebook
    rng = np.random.default_rng(0)
    cell_center = np.array([0.35, 0.28])
    base = cell_center + rng.uniform(-0.45, 0.45, (24, 2)) * np.array([1.0, 0.9])
    ax2.add_patch(RegularPolygon(cell_center, 6, radius=0.75, fill=True,
                                 facecolor="tab:blue", alpha=0.12, edgecolor="tab:blue"))
    ax2.scatter(*base.T, s=24, c="tab:blue", label="centred=True\n(code around cell centroid)")
    offset_pts = base - cell_center
    ax2.scatter(*offset_pts.T, s=24, c="crimson", marker="x",
                label="centred=False / VE8Z\n(code around 0, keeps exact zeros)")
    ax2.scatter([0], [0], s=90, c="crimson", marker="+", lw=2)
    ax2.scatter(*cell_center, s=90, c="tab:blue", marker="*", zorder=4)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.1, 1.2); ax2.set_ylim(-1.0, 1.1)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    ax2.set_title("centred flag (PR #46):\nrecenter the code to the cell centroid or to 0")

    fig.suptitle("The VE8Z32:16 geometry: 8-dim E8 cells, q=16 labels per cell, "
                 "one shared scale per 32 weights", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig4_e8_voronoi.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def hadamard8():
    h = np.array([[1.0]])
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(8.0)


def fig5():
    H8 = hadamard8()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.0))
    x = np.array([1.5, 0.02, -0.01, 0.0, 0.01, 0.0, -0.02, 0.01])
    y = x @ H8
    idx = np.arange(8)
    ax1.bar(idx, x, color="crimson")
    ax1.set_title("before the turn:\none loud weight")
    ax1.set_xticks(idx); ax1.set_ylim(-0.7, 1.7)
    ax1.grid(alpha=0.25)

    ax2.bar(idx, y, color="tab:blue")
    ax2.set_title("after H8:\nenergy spread evenly, ||x||² unchanged")
    ax2.set_xticks(idx); ax2.set_ylim(-0.7, 1.7)
    ax2.grid(alpha=0.25)

    ax3.imshow(H8, cmap="RdBu", vmin=-0.4, vmax=0.4)
    ax3.set_title("H8 = Hadamard matrix\n/ √8, orthogonal (H² = I)")
    ax3.set_xticks([]); ax3.set_yticks([])

    fig.suptitle("The H-prefix arms rotate each 8-block by H8 before ball rounding — "
                 "outliers no longer hog the bound", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig5_hadamard.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig6():
    rng = np.random.default_rng(3)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 4.0))

    # left: scale handling — s16 stores t unsnapped (SCALE_STEPS[16] is None);
    # only s8/s6 snap to ladders of 16/8 levels per octave
    top = 1.0
    raw = 0.95 * 2.0 ** (-rng.uniform(0, 2.2, 14))
    ax1.scatter(np.log2(raw), raw, s=40, c="tab:blue",
                label="s16: t used as bisected\n(stored fp16, no ladder)")
    snapped8 = top * 2.0 ** (-np.floor(np.log2(top / raw) * 16) / 16.0)
    ax1.plot(np.log2(top * 2.0 ** (-np.arange(0, 40) / 16.0)),
             top * 2.0 ** (-np.arange(0, 40) / 16.0), "|", ms=12, color="0.6")
    ax1.scatter(np.log2(snapped8), snapped8, s=40, c="tab:green",
                label="s8/s6 only: snapped to\n16 or 8 levels per octave")
    for a, b in zip(raw, snapped8):
        ax1.plot([np.log2(a), np.log2(b)], [a, b], color="tab:green", lw=0.8, alpha=0.6)
    ax1.set_xlabel("log2 scale")
    ax1.set_title("SCALE_STEPS = {16: None, 8: 16, 6: 8}\ns16: no snapping")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)

    # middle: per-group vs per-row scale on a synthetic row of 4 groups
    n = 128
    row = rng.standard_normal(4 * n)
    X = row.reshape(-1, n)
    bound = 3380
    tb = bd.block_scales(X, "Z8", "ball", bound)
    tr = np.repeat(tb.max(), 4)
    xs = np.arange(4)
    ax2.bar(xs - 0.18, tb, width=0.36, color="tab:blue", label="ZBG: scale per 128-block")
    ax2.bar(xs + 0.18, tr, width=0.36, color="tab:orange",
            label="ZB128@R: one scale per row")
    ax2.set_xticks(xs, ["blk 0", "blk 1", "blk 2", "blk 3"])
    ax2.set_ylabel("scale t")
    ax2.set_title("Per-block scale follows each block;\nrow scale = the max (finer vs cheaper)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25)

    # right: imatrix weighting turns the ball into an ellipsoid in weight space
    th = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(th), np.sin(th), color="tab:blue", lw=2,
             label="plain ball (ZBG)")
    ax3.plot(1.9 * np.cos(th), 0.55 * np.sin(th), color="tab:green", lw=2,
             label="imatrix-weighted (h/i):\nball stretched per column")
    ax3.set_aspect("equal"); ax3.grid(alpha=0.25)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.legend(frameon=False, fontsize=8, loc="lower left")
    ax3.set_title("Weighted arm: multiply columns by\n(imatrix/mean)^0.25 → ellipsoid in\nweight space")

    fig.suptitle("Where the scales come from (block_scales, SCALE_STEPS, _column_weights)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig6_scales_ladder.png", bbox_inches="tight")
    plt.close(fig)


fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
print("done")
