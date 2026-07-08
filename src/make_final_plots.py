from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

CURRENCY_COLORS = {
    "BTC": "#f97316",
    "ETH": "#2563eb",
}

METHOD_ORDER = [
    "convvae",
    "previous_surface",
    "cell_mean",
    "quadratic_smile",
    "row_mean",
]
METHOD_LABELS = {
    "convvae": "ConvVAE",
    "previous_surface": "Previous surface",
    "cell_mean": "Cell mean",
    "quadratic_smile": "Quadratic smile",
    "row_mean": "Row mean",
}
METHOD_COLORS = {
    "convvae": "#2563eb",
    "previous_surface": "#475569",
    "cell_mean": "#16a34a",
    "quadratic_smile": "#dc2626",
    "row_mean": "#9333ea",
}

SCHEME_ORDER = ["row_random", "long_tenor", "put_wing", "call_wing", "col_random"]
SCHEME_LABELS = {
    "row_random": "Row random",
    "long_tenor": "Long tenor",
    "put_wing": "Put wing",
    "call_wing": "Call wing",
    "col_random": "Column random",
}

RISK_ORDER = ["VaR_95_loss", "CVaR_95_loss", "Lambda_VaR_loss"]
RISK_LABELS = {
    "VaR_95_loss": "VaR 95%",
    "CVaR_95_loss": "CVaR 95%",
    "Lambda_VaR_loss": "Lambda VaR",
}
RISK_COLORS = {
    "VaR_95_loss": "#64748b",
    "CVaR_95_loss": "#f97316",
    "Lambda_VaR_loss": "#dc2626",
}


def require_table(name: str) -> Path:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the relevant analysis script first.")
    return path


def grouped_bar_offsets(n_bars: int, width: float) -> np.ndarray:
    return (np.arange(n_bars) - (n_bars - 1) / 2) * width


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_structured_rmse() -> None:
    results = pd.read_csv(require_table("combined_structured_summary.csv"))
    results = results[results["method"].isin(METHOD_ORDER)].copy()

    currencies = sorted(results["currency"].unique())
    fig, axes = plt.subplots(
        1,
        len(currencies),
        figsize=(14, 5),
        sharey=True,
        constrained_layout=True,
    )
    if len(currencies) == 1:
        axes = [axes]

    x = np.arange(len(SCHEME_ORDER))
    width = 0.14
    offsets = grouped_bar_offsets(len(METHOD_ORDER), width)

    for ax, currency in zip(axes, currencies):
        currency_data = results[results["currency"] == currency]
        for offset, method in zip(offsets, METHOD_ORDER):
            y = (
                currency_data[currency_data["method"] == method]
                .set_index("scheme")
                .reindex(SCHEME_ORDER)["mean_rmse"]
            )
            ax.bar(
                x + offset,
                y,
                width=width,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )

        ax.set_title(currency)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([SCHEME_LABELS[s] for s in SCHEME_ORDER], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25, which="both")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean hidden-cell RMSE, vol points (log scale)")
    fig.suptitle("Structured Missingness Reconstruction, Matched Test Set")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5)
    save_figure(fig, "final_01_structured_rmse.png")
    plt.close(fig)


def plot_regression_betas() -> None:
    regression = pd.read_csv(require_table("iv_mean_reversion_regression.csv"))
    regression = regression.sort_values(["currency", "horizon_hours"])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for currency, g in regression.groupby("currency"):
        ax.plot(
            g["horizon_hours"],
            g["beta_on_residual"],
            marker="o",
            linewidth=2,
            label=currency,
            color=CURRENCY_COLORS.get(currency),
        )

    ax.axhline(0, color="#0f172a", linewidth=1)
    ax.set_title("Residual Mean-Reversion Regression Betas")
    ax.set_xlabel("Horizon, hours")
    ax.set_ylabel("Beta on ConvVAE residual")
    ax.set_xticks(sorted(regression["horizon_hours"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(title="Currency")
    save_figure(fig, "final_02_regression_betas.png")
    plt.close(fig)


def plot_filtered_strategy_pnl() -> None:
    strategy = pd.read_csv(require_table("filtered_residual_strategy_summary.csv"))
    strategy = strategy.sort_values(["currency", "horizon_hours"])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x_labels = sorted(strategy["horizon_hours"].unique())
    x = np.arange(len(x_labels))
    width = 0.32
    currencies = sorted(strategy["currency"].unique())
    offsets = grouped_bar_offsets(len(currencies), width)

    for offset, currency in zip(offsets, currencies):
        y = (
            strategy[strategy["currency"] == currency]
            .set_index("horizon_hours")
            .reindex(x_labels)["mean_pnl_vol_points"]
        )
        ax.bar(
            x + offset,
            y,
            width=width,
            label=currency,
            color=CURRENCY_COLORS.get(currency),
        )

    ax.axhline(0, color="#0f172a", linewidth=1)
    ax.set_title("Filtered Residual Strategy Mean Paper PnL")
    ax.set_xlabel("Horizon, hours")
    ax.set_ylabel("Mean paper IV PnL, vol points")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Currency")
    save_figure(fig, "final_03_filtered_strategy_pnl.png")
    plt.close(fig)


def plot_tail_risk() -> None:
    risk = pd.read_csv(require_table("filtered_strategy_tail_risk.csv"))
    risk = risk.sort_values(["currency", "horizon_hours"])

    currencies = sorted(risk["currency"].unique())
    fig, axes = plt.subplots(
        1,
        len(currencies),
        figsize=(12, 5),
        sharey=True,
        constrained_layout=True,
    )
    if len(currencies) == 1:
        axes = [axes]

    horizons = sorted(risk["horizon_hours"].unique())
    x = np.arange(len(horizons))
    width = 0.22
    offsets = grouped_bar_offsets(len(RISK_ORDER), width)

    for ax, currency in zip(axes, currencies):
        currency_data = risk[risk["currency"] == currency]
        for offset, metric in zip(offsets, RISK_ORDER):
            y = currency_data.set_index("horizon_hours").reindex(horizons)[metric]
            ax.bar(
                x + offset,
                y,
                width=width,
                label=RISK_LABELS[metric],
                color=RISK_COLORS[metric],
            )

        ax.set_title(currency)
        ax.set_xlabel("Horizon, hours")
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Loss, vol points")
    fig.suptitle("Filtered Strategy Tail-Risk Estimates")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)
    save_figure(fig, "final_04_tail_risk.png")
    plt.close(fig)


def main() -> None:
    plot_structured_rmse()
    plot_regression_betas()
    plot_filtered_strategy_pnl()
    plot_tail_risk()


if __name__ == "__main__":
    main()
