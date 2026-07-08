from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_residuals(path: str | Path = "results/tables/convvae_residuals.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def add_future_iv_change(
    df: pd.DataFrame,
    horizon_hours: float,
    tolerance_minutes: float = 20,
) -> pd.DataFrame:
    """
    For each currency/tenor/delta cell, find the nearest future observation
    around horizon_hours and compute future IV change.

    Uses pandas timezone-aware timestamps throughout.
    """
    out_rows = []

    target_gap = pd.Timedelta(hours=horizon_hours)
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    keys = ["currency", "tenor", "delta"]

    for _, g in df.groupby(keys):
        g = g.sort_values("timestamp_utc").reset_index(drop=True)

        for i in range(len(g)):
            current_time = g.loc[i, "timestamp_utc"]
            target_time = current_time + target_gap

            future = g[g["timestamp_utc"] > current_time].copy()

            if future.empty:
                continue

            future["gap_to_target"] = (future["timestamp_utc"] - target_time).abs()
            future = future[future["gap_to_target"] <= tolerance]

            if future.empty:
                continue

            best = future.sort_values("gap_to_target").iloc[0]
            actual_gap = best["timestamp_utc"] - current_time

            row = g.loc[i].to_dict()
            row["horizon_hours"] = horizon_hours
            row["future_timestamp_utc"] = best["timestamp_utc"]
            row["actual_gap_hours"] = actual_gap.total_seconds() / 3600

            row["future_market_iv"] = best["market_iv"]
            row["future_market_iv_vol_points"] = best["market_iv_vol_points"]

            row["future_iv_change"] = row["future_market_iv"] - row["market_iv"]
            row["future_iv_change_vol_points"] = (
                row["future_market_iv_vol_points"] - row["market_iv_vol_points"]
            )

            out_rows.append(row)

    return pd.DataFrame(out_rows)


def run_simple_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS slope for:
        future_iv_change = alpha + beta * residual + error

    Uses numpy only.
    """
    rows = []

    group_cols = ["currency", "horizon_hours"]

    for (currency, horizon), g in df.groupby(group_cols):
        x = g["residual_vol_points"].to_numpy()
        y = g["future_iv_change_vol_points"].to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) < 10:
            continue

        X = np.column_stack([np.ones(len(x)), x])
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

        y_hat = X @ beta_hat
        resid = y - y_hat

        n = len(y)
        k = X.shape[1]
        sigma2 = (resid @ resid) / max(n - k, 1)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))

        alpha = beta_hat[0]
        beta = beta_hat[1]
        beta_se = se[1]
        t_stat = beta / beta_se if beta_se > 0 else np.nan

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

        rows.append(
            {
                "currency": currency,
                "horizon_hours": horizon,
                "alpha": alpha,
                "beta_on_residual": beta,
                "beta_t_stat": t_stat,
                "corr_residual_future_change": corr,
                "r2": r2,
                "n": n,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    residuals = load_residuals()

    all_horizon_data = []

    for h in [1, 3, 6, 12, 24]:
        hdf = add_future_iv_change(
            residuals,
            horizon_hours=h,
            tolerance_minutes=30,
        )

        if len(hdf) == 0:
            print(f"No matched observations for horizon {h}h")
            continue

        all_horizon_data.append(hdf)

    if not all_horizon_data:
        raise RuntimeError("No future-IV-change data created. Need more timestamps or wider tolerance.")

    signal_df = pd.concat(all_horizon_data, ignore_index=True)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    signal_df.to_csv("results/tables/iv_mean_reversion_dataset.csv", index=False)

    regression = run_simple_regression(signal_df)
    regression.to_csv("results/tables/iv_mean_reversion_regression.csv", index=False)

    print("\nSaved:")
    print("results/tables/iv_mean_reversion_dataset.csv")
    print("results/tables/iv_mean_reversion_regression.csv")

    print("\nMean-reversion regression:")
    print(regression.round(4).to_string(index=False))