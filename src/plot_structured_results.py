from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["convvae", "previous_surface", "cell_mean", "quadratic_smile"]
METHOD_LABELS = {
    "convvae": "ConvVAE",
    "previous_surface": "Previous surface",
    "cell_mean": "Cell mean",
    "quadratic_smile": "Quadratic smile",
}
SCHEME_ORDER = ["row_random", "long_tenor", "put_wing", "call_wing", "col_random"]
SCHEME_LABELS = {
    "row_random": "Row random",
    "long_tenor": "Long tenor",
    "put_wing": "Put wing",
    "call_wing": "Call wing",
    "col_random": "Column random",
}


def main():
    input_path = Path("results/tables/combined_structured_summary.csv")
    if not input_path.exists():
        raise FileNotFoundError(
            "Missing combined structured results. Run: python src/compare_structured_results.py"
        )

    results = pd.read_csv(input_path)
    plot_data = results[results["method"].isin(METHOD_ORDER)].copy()

    currencies = sorted(plot_data["currency"].unique())
    fig, axes = plt.subplots(
        1,
        len(currencies),
        figsize=(13, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(currencies) == 1:
        axes = [axes]

    colors = {
        "convvae": "#2563eb",
        "previous_surface": "#475569",
        "cell_mean": "#16a34a",
        "quadratic_smile": "#dc2626",
    }
    x = np.arange(len(SCHEME_ORDER))
    width = 0.18
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2) * width

    for ax, currency in zip(axes, currencies):
        currency_data = plot_data[plot_data["currency"] == currency]
        for offset, method in zip(offsets, METHOD_ORDER):
            method_data = (
                currency_data[currency_data["method"] == method]
                .set_index("scheme")
                .reindex(SCHEME_ORDER)
            )
            ax.bar(
                x + offset,
                method_data["mean_rmse"],
                width=width,
                label=METHOD_LABELS[method],
                color=colors[method],
            )

        ax.set_title(currency)
        ax.set_xticks(x)
        ax.set_xticklabels([SCHEME_LABELS[s] for s in SCHEME_ORDER], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean hidden-cell RMSE, vol points")
    fig.suptitle("Structured-mask reconstruction results, matched test set (n = 10)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4)

    output_path = Path("results/figures/structured_results_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
