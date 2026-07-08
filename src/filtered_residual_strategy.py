from pathlib import Path

import numpy as np
import pandas as pd


def load_signal_dataset(path="results/tables/iv_mean_reversion_dataset.csv"):
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["future_timestamp_utc"] = pd.to_datetime(df["future_timestamp_utc"], utc=True)
    return df


def load_delta_betas(path="results/tables/iv_mean_reversion_by_delta.csv"):
    betas = pd.read_csv(path)
    return betas


def make_filtered_strategy(df, betas, quantile=0.2, tstat_threshold=1.0):
    """
    Trade only currency/horizon/delta regions where beta_on_residual is negative.

    If beta < 0:
      high residual = rich IV -> short IV
      low residual = cheap IV -> long IV

    This avoids trading cells where residuals historically trend instead of mean-revert.
    """
    usable = betas[
        (betas["beta_on_residual"] < 0)
        & (betas["beta_t_stat"].abs() >= tstat_threshold)
    ][["currency", "horizon_hours", "delta"]].copy()

    df = df.merge(
        usable.assign(use_signal=True),
        on=["currency", "horizon_hours", "delta"],
        how="left",
    )

    df = df[df["use_signal"] == True].copy()

    rows = []

    group_cols = ["currency", "horizon_hours", "timestamp_utc"]

    for keys, g in df.groupby(group_cols):
        g = g.copy()

        if len(g) < 4:
            continue

        low = g["residual_vol_points"].quantile(quantile)
        high = g["residual_vol_points"].quantile(1 - quantile)

        g["position"] = 0.0
        g.loc[g["residual_vol_points"] <= low, "position"] = 1.0
        g.loc[g["residual_vol_points"] >= high, "position"] = -1.0

        traded = g[g["position"] != 0].copy()

        if traded.empty:
            continue

        traded["pnl_vol_points"] = traded["position"] * traded["future_iv_change_vol_points"]

        currency, horizon, timestamp = keys

        rows.append(
            {
                "currency": currency,
                "horizon_hours": horizon,
                "timestamp_utc": timestamp,
                "mean_pnl_vol_points": traded["pnl_vol_points"].mean(),
                "sum_pnl_vol_points": traded["pnl_vol_points"].sum(),
                "n_trades": len(traded),
                "avg_abs_residual": traded["residual_vol_points"].abs().mean(),
            }
        )

    return pd.DataFrame(rows), usable


def summarize_strategy(strategy_df):
    rows = []

    for (currency, horizon), g in strategy_df.groupby(["currency", "horizon_hours"]):
        r = g["mean_pnl_vol_points"].to_numpy()

        mean = np.mean(r)
        std = np.std(r, ddof=1)

        t_stat = mean / (std / np.sqrt(len(r))) if std > 0 and len(r) > 1 else np.nan
        hit_rate = np.mean(r > 0)

        rows.append(
            {
                "currency": currency,
                "horizon_hours": horizon,
                "mean_pnl_vol_points": mean,
                "std_pnl_vol_points": std,
                "t_stat": t_stat,
                "hit_rate": hit_rate,
                "n_periods": len(r),
                "avg_trades_per_period": g["n_trades"].mean(),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_signal_dataset()
    betas = load_delta_betas()

    strategy, usable = make_filtered_strategy(
        df,
        betas,
        quantile=0.2,
        tstat_threshold=1.0,
    )

    summary = summarize_strategy(strategy)

    Path("results/tables").mkdir(parents=True, exist_ok=True)

    usable.to_csv("results/tables/filtered_strategy_usable_deltas.csv", index=False)
    strategy.to_csv("results/tables/filtered_residual_strategy_period_returns.csv", index=False)
    summary.to_csv("results/tables/filtered_residual_strategy_summary.csv", index=False)

    print("\nUsable negative-beta delta regions:")
    print(usable.sort_values(["currency", "horizon_hours", "delta"]).to_string(index=False))

    print("\nFiltered residual strategy summary:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print("results/tables/filtered_strategy_usable_deltas.csv")
    print("results/tables/filtered_residual_strategy_period_returns.csv")
    print("results/tables/filtered_residual_strategy_summary.csv")