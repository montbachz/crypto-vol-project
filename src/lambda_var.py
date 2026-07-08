from pathlib import Path

import numpy as np
import pandas as pd


def empirical_var(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Standard empirical VaR on losses.
    """
    return float(np.quantile(losses, alpha))


def empirical_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Empirical CVaR / expected shortfall on losses.
    """
    var = empirical_var(losses, alpha)
    tail = losses[losses >= var]

    if len(tail) == 0:
        return var

    return float(tail.mean())


def lambda_function(x: float, loss_min: float, loss_max: float) -> float:
    """
    Simple increasing Lambda function.

    For small losses, use lower confidence.
    For large losses, use higher confidence.

    This makes the risk measure more conservative in the tail.
    """
    lambda_min = 0.90
    lambda_max = 0.99

    if loss_max <= loss_min:
        return lambda_max

    z = (x - loss_min) / (loss_max - loss_min)
    z = min(max(z, 0.0), 1.0)

    return lambda_min + z * (lambda_max - lambda_min)


def empirical_lambda_var(losses: np.ndarray) -> float:
    """
    Empirical Lambda VaR.

    Sort losses and find the first loss x_j such that:

        empirical CDF(x_j) > Lambda(x_j)

    This is the empirical version of the lambda quantile.
    """
    losses = np.sort(np.asarray(losses, dtype=float))
    losses = losses[np.isfinite(losses)]

    n = len(losses)

    if n == 0:
        return np.nan

    loss_min = float(losses.min())
    loss_max = float(losses.max())

    for j, x in enumerate(losses, start=1):
        empirical_cdf = j / n
        lam = lambda_function(float(x), loss_min, loss_max)

        if empirical_cdf > lam:
            return float(x)

    return float(losses[-1])


def summarize_tail_risk(strategy_path="results/tables/filtered_residual_strategy_period_returns.csv"):
    df = pd.read_csv(strategy_path)

    rows = []

    for (currency, horizon), g in df.groupby(["currency", "horizon_hours"]):
        pnl = g["mean_pnl_vol_points"].to_numpy()

        # Risk is computed on losses.
        losses = -pnl

        rows.append(
            {
                "currency": currency,
                "horizon_hours": horizon,
                "mean_pnl": float(np.mean(pnl)),
                "std_pnl": float(np.std(pnl, ddof=1)),
                "VaR_95_loss": empirical_var(losses, 0.95),
                "CVaR_95_loss": empirical_cvar(losses, 0.95),
                "Lambda_VaR_loss": empirical_lambda_var(losses),
                "worst_loss": float(np.max(losses)),
                "n_periods": len(pnl),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary = summarize_tail_risk()

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv("results/tables/filtered_strategy_tail_risk.csv", index=False)

    print("\nFiltered strategy tail-risk summary:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print("results/tables/filtered_strategy_tail_risk.csv")