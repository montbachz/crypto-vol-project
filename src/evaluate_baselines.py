from __future__ import annotations

import numpy as np
import pandas as pd

from baselines import (
    cell_mean_fill,
    row_mean_fill,
    previous_surface_fill,
    quadratic_smile_fill,
)
from dataset import load_all_surfaces
from masking import make_random_mask, apply_mask, rmse_hidden


def time_split(X: np.ndarray, train_frac=0.7, val_frac=0.15):
    n = len(X)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train = X[:n_train]
    val = X[n_train:n_train + n_val]
    test = X[n_train + n_val:]

    return train, val, test


def evaluate_currency(currency: str, mask_rates=(0.1, 0.3, 0.5), seed: int = 42) -> pd.DataFrame:
    X, M, meta = load_all_surfaces(currency)

    # Remove surfaces with NaNs for now
    valid = ~np.isnan(X).any(axis=(1, 2))
    X = X[valid]

    if len(X) < 5:
        raise ValueError(f"Need at least 5 clean surfaces for {currency}. You only have {len(X)}.")

    X_train, X_val, X_test = time_split(X)
    cell_means = X_train.mean(axis=0)

    rng = np.random.default_rng(seed)
    rows = []

    for mask_rate in mask_rates:
        for t in range(len(X_test)):
            true = X_test[t]
            obs_mask = make_random_mask(true.shape, mask_rate=mask_rate, rng=rng)
            masked = apply_mask(true, obs_mask)

            pred_cell = cell_mean_fill(masked, cell_means)
            pred_row = row_mean_fill(masked, cell_means)

            # Previous-surface baseline.
            # For first test snapshot, use last training surface.
            if t == 0:
                prev = X_train[-1]
            else:
                prev = X_test[t - 1]

            pred_prev = previous_surface_fill(masked, prev, cell_means)
            pred_quad = quadratic_smile_fill(masked, cell_means)

            rows.extend(
                [
                    {
                        "currency": currency,
                        "mask_rate": mask_rate,
                        "method": "cell_mean",
                        "rmse_vol_points": rmse_hidden(true, pred_cell, obs_mask) * 100,
                    },
                    {
                        "currency": currency,
                        "mask_rate": mask_rate,
                        "method": "row_mean",
                        "rmse_vol_points": rmse_hidden(true, pred_row, obs_mask) * 100,
                    },
                    {
                        "currency": currency,
                        "mask_rate": mask_rate,
                        "method": "previous_surface",
                        "rmse_vol_points": rmse_hidden(true, pred_prev, obs_mask) * 100,
                    },
                    {
                        "currency": currency,
                        "mask_rate": mask_rate,
                        "method": "quadratic_smile",
                        "rmse_vol_points": rmse_hidden(true, pred_quad, obs_mask) * 100,
                    }
                ]
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []

    for currency in ["BTC", "ETH"]:
        results = evaluate_currency(currency)
        all_results.append(results)

    results = pd.concat(all_results, ignore_index=True)

    summary = (
        results.groupby(["currency", "mask_rate", "method"], as_index=False)
        .agg(
            mean_rmse=("rmse_vol_points", "mean"),
            median_rmse=("rmse_vol_points", "median"),
            n=("rmse_vol_points", "size"),
        )
        .sort_values(["currency", "mask_rate", "mean_rmse"])
    )

    print("\nBaseline RMSE summary, vol points:")
    print(summary.round(4))

    results.to_csv("results/tables/baseline_raw_results.csv", index=False)
    summary.to_csv("results/tables/baseline_summary.csv", index=False)

    print("\nSaved:")
    print("results/tables/baseline_raw_results.csv")
    print("results/tables/baseline_summary.csv")