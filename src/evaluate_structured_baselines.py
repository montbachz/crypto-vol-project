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
from masking import apply_mask, rmse_hidden, make_structured_mask


def time_split(X: np.ndarray, train_frac=0.7, val_frac=0.15):
    n = len(X)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train = X[:n_train]
    val = X[n_train:n_train + n_val]
    test = X[n_train + n_val:]
    return train, val, test


def evaluate_currency(
    currency: str,
    schemes=("row_random", "long_tenor", "col_random", "put_wing", "call_wing"),
    seed: int = 42,
) -> pd.DataFrame:
    X, M, meta = load_all_surfaces(currency)

    valid = ~np.isnan(X).any(axis=(1, 2))
    X = X[valid]

    if len(X) < 5:
        raise ValueError(f"Need at least 5 clean surfaces for {currency}. You only have {len(X)}.")

    X_train, X_val, X_test = time_split(X)
    cell_means = X_train.mean(axis=0)

    rng = np.random.default_rng(seed)
    rows = []

    for scheme in schemes:
        for t in range(len(X_test)):
            true = X_test[t]
            obs_mask = make_structured_mask(true.shape, scheme=scheme, rng=rng)
            masked = apply_mask(true, obs_mask)

            pred_cell = cell_mean_fill(masked, cell_means)
            pred_row = row_mean_fill(masked, cell_means)
            pred_quad = quadratic_smile_fill(masked, cell_means)

            if t == 0:
                prev = X_train[-1]
            else:
                prev = X_test[t - 1]

            pred_prev = previous_surface_fill(masked, prev, cell_means)

            for method, pred in [
                ("cell_mean", pred_cell),
                ("row_mean", pred_row),
                ("quadratic_smile", pred_quad),
                ("previous_surface", pred_prev),
            ]:
                rows.append(
                    {
                        "currency": currency,
                        "scheme": scheme,
                        "method": method,
                        "rmse_vol_points": rmse_hidden(true, pred, obs_mask) * 100,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []

    for currency in ["BTC", "ETH"]:
        results = evaluate_currency(currency)
        all_results.append(results)

    results = pd.concat(all_results, ignore_index=True)

    summary = (
        results.groupby(["currency", "scheme", "method"], as_index=False)
        .agg(
            mean_rmse=("rmse_vol_points", "mean"),
            median_rmse=("rmse_vol_points", "median"),
            n=("rmse_vol_points", "size"),
        )
        .sort_values(["currency", "scheme", "mean_rmse"])
    )

    print("\nStructured-mask baseline RMSE summary, vol points:")
    print(summary.round(4))

    results.to_csv("results/tables/structured_baseline_raw_results.csv", index=False)
    summary.to_csv("results/tables/structured_baseline_summary.csv", index=False)

    print("\nSaved:")
    print("results/tables/structured_baseline_raw_results.csv")
    print("results/tables/structured_baseline_summary.csv")
