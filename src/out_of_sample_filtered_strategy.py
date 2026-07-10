from pathlib import Path

import numpy as np
import pandas as pd

from analyze_signal_breakdown import ols_by_group
from filtered_residual_strategy import make_filtered_strategy, summarize_strategy
from residual_strategy import make_strategy_returns


def load_signal_dataset(path="results/tables/iv_mean_reversion_dataset.csv"):
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["future_timestamp_utc"] = pd.to_datetime(df["future_timestamp_utc"], utc=True)
    return df


def time_split_by_timestamp(df: pd.DataFrame, train_frac: float = 0.7):
    timestamps = np.array(sorted(df["timestamp_utc"].dropna().unique()))
    n_train = int(train_frac * len(timestamps))

    train_timestamps = set(timestamps[:n_train])
    test_timestamps = set(timestamps[n_train:])

    train = df[df["timestamp_utc"].isin(train_timestamps)].copy()
    test = df[df["timestamp_utc"].isin(test_timestamps)].copy()

    return train, test


def add_strategy_label(summary: pd.DataFrame, label: str) -> pd.DataFrame:
    summary = summary.copy()
    summary.insert(0, "strategy", label)
    return summary


def main():
    df = load_signal_dataset()
    train, test = time_split_by_timestamp(df, train_frac=0.7)

    full_sample_betas = pd.read_csv("results/tables/iv_mean_reversion_by_delta.csv")
    train_betas = ols_by_group(train, ["currency", "horizon_hours", "delta"])

    unfiltered_test_returns = make_strategy_returns(test, quantile=0.2)
    full_sample_filtered_returns, full_sample_usable = make_filtered_strategy(
        test,
        full_sample_betas,
        quantile=0.2,
        tstat_threshold=1.0,
    )
    oos_filtered_returns, oos_usable = make_filtered_strategy(
        test,
        train_betas,
        quantile=0.2,
        tstat_threshold=1.0,
    )

    unfiltered_summary = add_strategy_label(
        summarize_strategy(unfiltered_test_returns),
        "unfiltered_test",
    )
    full_sample_filtered_summary = add_strategy_label(
        summarize_strategy(full_sample_filtered_returns),
        "filtered_full_sample_test",
    )
    oos_summary = add_strategy_label(
        summarize_strategy(oos_filtered_returns),
        "filtered_oos_test",
    )

    comparison = pd.concat(
        [unfiltered_summary, full_sample_filtered_summary, oos_summary],
        ignore_index=True,
    ).sort_values(["currency", "horizon_hours", "strategy"])

    split_summary = pd.DataFrame(
        [
            {
                "train_start": train["timestamp_utc"].min(),
                "train_end": train["timestamp_utc"].max(),
                "test_start": test["timestamp_utc"].min(),
                "test_end": test["timestamp_utc"].max(),
                "train_timestamps": train["timestamp_utc"].nunique(),
                "test_timestamps": test["timestamp_utc"].nunique(),
                "train_rows": len(train),
                "test_rows": len(test),
            }
        ]
    )

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    train_betas.to_csv("results/tables/oos_filter_train_delta_betas.csv", index=False)
    oos_usable.to_csv("results/tables/oos_filtered_strategy_usable_deltas.csv", index=False)
    oos_filtered_returns.to_csv(
        "results/tables/oos_filtered_strategy_period_returns.csv",
        index=False,
    )
    oos_summary.to_csv("results/tables/oos_filtered_strategy_summary.csv", index=False)
    comparison.to_csv("results/tables/strategy_filter_comparison.csv", index=False)
    split_summary.to_csv("results/tables/oos_strategy_split_summary.csv", index=False)

    print("\nOut-of-sample split:")
    print(split_summary.to_string(index=False))

    print("\nTrain-sample usable negative-beta delta regions:")
    print(oos_usable.sort_values(["currency", "horizon_hours", "delta"]).to_string(index=False))

    print("\nStrategy comparison on held-out timestamps:")
    print(comparison.round(4).to_string(index=False))

    print("\nSaved:")
    print("results/tables/oos_filter_train_delta_betas.csv")
    print("results/tables/oos_filtered_strategy_usable_deltas.csv")
    print("results/tables/oos_filtered_strategy_period_returns.csv")
    print("results/tables/oos_filtered_strategy_summary.csv")
    print("results/tables/strategy_filter_comparison.csv")
    print("results/tables/oos_strategy_split_summary.csv")


if __name__ == "__main__":
    main()
