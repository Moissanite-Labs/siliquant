"""Plot measured Gemma block-rate and KL points using Matplotlib."""

import argparse
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from analyze import FAMILIES, MODELS, Q4, Q4IMAT, Q4KM, TEXTS, load_rows, load_transcribed_table, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--transcribed-table", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    raw = args.results.read_bytes()
    rows = load_transcribed_table(raw, args.model) if args.transcribed_table else load_rows([raw])
    matplotlib.use("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, layout="constrained")
    for ax, text, title in zip(axes, TEXTS, ("Wikitext-2", "Python")):
        cells = select(rows, args.model, text)
        for (_, arms), color, label in zip(
                FAMILIES, ("#056a90", "#96334f"),
                ("Ours: plain ball", "Ours: calibrated ball (real-text imatrix)")):
            points = [(float(cells[a]["bits_per_weight"]), float(cells[a]["kl_mean"]))
                      for a in arms if a in cells]
            if points:
                ax.plot(*zip(*points), marker="o", linewidth=2, color=color, label=label)
        for arm, shape, label in zip((Q4, Q4KM, Q4IMAT), ("s", "^", "D"),
                                      ("Stock: Q4_0", "Stock: Q4_K_M", "Stock: Q4_K_M + imatrix")):
            if arm in cells:
                r = cells[arm]
                ax.scatter(float(r["bits_per_weight"]), float(r["kl_mean"]),
                           color="#263346", s=40, marker=shape, label=label)
        ax.set(title=title, xlabel="Block bits per weight")
        ax.grid(alpha=0.23)
    axes[0].set_ylabel("Mean KL against F16 (lower is better)")
    axes[1].legend(loc="upper right", fontsize=8)
    suffix = " · rounded transcription, raw receipts unverified" if args.transcribed_table else ""
    fig.suptitle(args.model + suffix, fontsize=13)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
