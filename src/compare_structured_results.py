from pathlib import Path

import pandas as pd


def main():
    baseline_path = Path("results/tables/structured_baseline_summary.csv")
    convvae_path = Path("results/tables/convvae_structured_summary.csv")

    if not baseline_path.exists():
        raise FileNotFoundError(
            "Missing structured baseline results. Run: python src/evaluate_structured_baselines.py"
        )

    if not convvae_path.exists():
        raise FileNotFoundError(
            "Missing structured ConvVAE results. Run: python src/evaluate_convvae_structured.py"
        )

    baseline = pd.read_csv(baseline_path)
    convvae = pd.read_csv(convvae_path)

    combined = pd.concat([baseline, convvae], ignore_index=True)

    combined = combined.sort_values(
        ["currency", "scheme", "mean_rmse"],
        ascending=[True, True, True],
    )

    print("\nCombined structured-mask results, vol points:")
    print(combined.round(4).to_string(index=False))

    output_path = Path("results/tables/combined_structured_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Also create a pivot table that is easier to read.
    pivot = combined.pivot_table(
        index=["currency", "scheme"],
        columns="method",
        values="mean_rmse",
    ).reset_index()

    pivot_path = Path("results/tables/combined_structured_pivot.csv")
    pivot.to_csv(pivot_path, index=False)

    print("\nSaved:")
    print(output_path)
    print(pivot_path)


if __name__ == "__main__":
    main()