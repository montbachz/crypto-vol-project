from pathlib import Path

import numpy as np
import pandas as pd


def load_signal_dataset(path="results/tables/iv_mean_reversion_dataset.csv"):
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["future_timestamp_utc"] = pd.to_datetime(df["future_timestamp_utc"], utc=True)
    return df


def ols_by_group(df, group_cols):
    rows = []

    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        x = g["residual_vol_points"].to_numpy()
        y = g["future_iv_change_vol_points"].to_numpy()

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if len(x) < 20:
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

        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "alpha": alpha,
                "beta_on_residual": beta,
                "beta_t_stat": t_stat,
                "r2": r2,
                "n": n,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_signal_dataset()

    Path("results/tables").mkdir(parents=True, exist_ok=True)

    # Breakdown by tenor
    by_tenor = ols_by_group(df, ["currency", "horizon_hours", "tenor"])
    by_tenor.to_csv("results/tables/iv_mean_reversion_by_tenor.csv", index=False)

    # Breakdown by delta
    by_delta = ols_by_group(df, ["currency", "horizon_hours", "delta"])
    by_delta.to_csv("results/tables/iv_mean_reversion_by_delta.csv", index=False)

    # Breakdown by tenor and delta. This may be sparse.
    by_cell = ols_by_group(df, ["currency", "horizon_hours", "tenor", "delta"])
    by_cell.to_csv("results/tables/iv_mean_reversion_by_cell.csv", index=False)

    print("\nBy tenor:")
    print(
        by_tenor.sort_values(["currency", "horizon_hours", "tenor"])
        .round(4)
        .to_string(index=False)
    )

    print("\nBy delta:")
    print(
        by_delta.sort_values(["currency", "horizon_hours", "delta"])
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:")
    print("results/tables/iv_mean_reversion_by_tenor.csv")
    print("results/tables/iv_mean_reversion_by_delta.csv")
    print("results/tables/iv_mean_reversion_by_cell.csv")