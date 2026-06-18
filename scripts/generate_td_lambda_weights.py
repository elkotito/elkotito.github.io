from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = (
        repo_root
        / "content"
        / "posts"
        / "reinforcement-learning-102"
        / "td_lambda_weights.svg"
    )

    lambdas = [0.0, 0.25, 0.5, 0.75, 0.9]
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"]
    n_values = np.linspace(1, 10, 500)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=160)

    for lambda_value, color in zip(lambdas, colors, strict=True):
        label = rf"$\lambda = {lambda_value:g}$"

        if lambda_value == 0:
            ax.plot(
                [1, 1, 10],
                [1, 0, 0],
                color=color,
                linewidth=2.4,
                label=label,
            )
            continue

        weights = (1 - lambda_value) * lambda_value ** (n_values - 1)
        ax.plot(n_values, weights, color=color, linewidth=2.4, label=label)

    ax.set_title(r"TD($\lambda$) weights for $n$-step targets", fontsize=16, weight="bold", pad=14)
    ax.set_xlabel(r"$n$", fontsize=13, labelpad=10)
    ax.set_ylabel(r"$(1 - \lambda)\lambda^{n - 1}$", fontsize=13, labelpad=10)

    ax.set_xlim(1, 10)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color="#e5e7eb", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=1,
        fontsize=11,
    )
    legend.get_frame().set_linewidth(1)

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(output_path)


if __name__ == "__main__":
    main()
