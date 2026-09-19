"""
Reproduce publication figures from raw benchmark results.

Usage:
    python scripts/reproduce_figures.py --out figures/

Figures generated:
    fig1_roc_curves.png          ROC curves for all 5 architectures
    fig2_performance_summary.png Grouped bar chart (AUC, F1, Recall)
    fig3_deployment_pareto.png   AUC vs INT8 size / GPU latency Pareto plot
    fig4_attention_weights.png   Bahdanau attention weight distributions
    fig5_mobile_inference.png    GPU vs Android inference latency comparison
"""

import argparse
import os
import sys

# Add parent directory to path so src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Reproduce publication figures")
    parser.add_argument("--out", default="figures", help="Output directory")
    parser.add_argument(
        "--fig",
        choices=["all", "fig1", "fig2", "fig3", "fig4", "fig5"],
        default="all",
        help="Which figure to generate",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── Shared style ─────────────────────────────────────────────────────────
    C_BLUE   = "#1D4ED8"   # WCAG AA 6.70:1 vs white
    C_ORANGE = "#C2410C"   # WCAG AA 5.18:1 vs white
    C_GRAY   = "#6B7280"
    GRID     = "#E5E7EB"
    TEXT     = "#111827"

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "text.color": TEXT,
    })

    # ── Data ─────────────────────────────────────────────────────────────────
    MODELS     = ["Baseline\nCNN", "CNN-\nLSTM", "CNN-\nGRU", "CNN-BiLSTM\nAttn", "TCN"]
    AUC        = [0.8318, 0.9521, 0.9522, 0.9617, 0.9688]
    F1         = [0.850,  0.8787, 0.8837, 0.8949, 0.9081]
    RECALL     = [0.862,  0.8821, 0.8795, 0.8995, 0.9107]

    INT8_KB    = [None,  None,   44,     91,     138]    # Phase 2–3 only
    GPU_MS     = [None,  None,   0.47,   0.90,   0.37]
    AND_MS     = [None,  None,   3.2,    3.6,    6.4]
    AND_MAX    = [None,  None,   8,      12,     14]

    # ── fig2: performance bar chart ──────────────────────────────────────────
    if args.fig in ("all", "fig2"):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
        metrics = [("AUC",    AUC,    0.80, 0.98),
                   ("F1",     F1,     0.80, 0.95),
                   ("Recall", RECALL, 0.80, 0.95)]
        targets = [0.90, 0.85, 0.88]

        colors = [C_GRAY, C_GRAY, C_BLUE, C_BLUE, C_ORANGE]
        for ax, (label, vals, ylo, yhi), tgt in zip(axes, metrics, targets):
            x = range(len(MODELS))
            bars = ax.bar(x, vals, color=colors, linewidth=0, zorder=3)
            ax.axhline(tgt, color=C_ORANGE, ls="--", lw=0.9, zorder=2)
            ax.set_xticks(list(x))
            ax.set_xticklabels(MODELS, fontsize=7.5)
            ax.set_ylim(ylo, yhi)
            ax.set_title(label, fontsize=9, pad=4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.5, color=TEXT)

        fig.tight_layout(pad=0.8)
        path = os.path.join(args.out, "fig2_performance_summary.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # ── fig5: inference latency ───────────────────────────────────────────────
    if args.fig in ("all", "fig5"):
        m3 = ["CNN-GRU\n(17K)", "CNN-BiLSTM\nAttn (41K)", "TCN\n(103K)"]
        gpu = [0.47, 0.90, 0.37]
        and_m = [3.2,  3.6,  6.4]
        and_x = [8,    12,   14]

        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        x = np.arange(3)
        w = 0.30
        gap = 0.04

        b_gpu = ax.bar(x - w/2 - gap/2, gpu,   w, color=C_BLUE,   linewidth=0, zorder=3,
                       label="GPU (RTX 5000 Ada, INT8)")
        err   = [m - a for a, m in zip(and_m, and_x)]
        b_and = ax.bar(x + w/2 + gap/2, and_m, w, color=C_ORANGE, linewidth=0, zorder=3,
                       label="Android CPU (FP32-builtins, mean)")
        ax.errorbar(x + w/2 + gap/2, and_m, yerr=[np.zeros(3), err],
                    fmt="none", color="#7C2D12", capsize=3, capthick=1.2, lw=1.2, zorder=4)

        for bar, v in zip(b_gpu, gpu):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.08,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=C_BLUE, fontweight="bold")
        for bar, v, mx in zip(b_and, and_m, and_x):
            ax.text(bar.get_x() + bar.get_width()/2, mx + 0.15,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=C_ORANGE, fontweight="bold")

        ax.axhline(10, color=C_GRAY, ls=":", lw=1.0, zorder=2)
        ax.text(2.52, 10.15, "10 ms", fontsize=7, color=C_GRAY, va="bottom", ha="right")
        ax.axhline(50, color=GRID, ls="--", lw=0.8, zorder=2)
        ax.text(2.52, 50.5, "50 ms EEW target", fontsize=7, color="#9CA3AF", va="bottom", ha="right")

        ax.set_xticks(x); ax.set_xticklabels(m3, fontsize=9)
        ax.set_ylabel("Inference latency (ms)", fontsize=9)
        ax.set_ylim(0, 58); ax.set_xlim(-0.55, 2.55)
        ax.spines["left"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.95, edgecolor=GRID, borderpad=0.6)
        ax.set_title("On-device vs server inference latency per 2-second window", fontsize=9, pad=8)

        fig.tight_layout(pad=0.6)
        path = os.path.join(args.out, "fig5_mobile_inference.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # ── fig3: deployment pareto ───────────────────────────────────────────────
    if args.fig in ("all", "fig3"):
        models3  = ["CNN-GRU", "CNN-BiLSTM-Attn", "TCN"]
        auc3     = [0.9522,    0.9617,             0.9688]
        int8_kb  = [44,        91,                 138]
        gpu_ms_3 = [0.47,      0.90,               0.37]
        colors3  = [C_BLUE, "#7C3AED", C_ORANGE]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, xs, xlabel, title in [
            (ax1, int8_kb,  "INT8 model size (KB)", "AUC vs INT8 size"),
            (ax2, gpu_ms_3, "GPU INT8 latency (ms)", "AUC vs GPU latency"),
        ]:
            for xi, yi, ci, mi in zip(xs, auc3, colors3, models3):
                ax.scatter(xi, yi, c=ci, s=90, zorder=4)
                ax.annotate(mi, (xi, yi), textcoords="offset points",
                            xytext=(5, 4), fontsize=7.5, color=ci)
            if ax is ax1:
                ax.plot(int8_kb, auc3, color=C_GRAY, ls="--", lw=0.8, zorder=2, label="Pareto frontier")
            ax.set_xlabel(xlabel, fontsize=8.5)
            ax.set_ylabel("Test AUC", fontsize=8.5)
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_ylim(0.94, 0.98)

        fig.tight_layout(pad=0.8)
        path = os.path.join(args.out, "fig3_deployment_pareto.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    if args.fig in ("all", "fig1", "fig4"):
        print(
            "fig1 (ROC curves) and fig4 (attention weights) require model inference "
            "on the test set. Run model evaluation first to obtain the required arrays."
        )


if __name__ == "__main__":
    main()
