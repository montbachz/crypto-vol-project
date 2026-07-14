# Crypto Implied-Volatility Surface Learning for Relative-Value Signals and Lambda-VaR Risk Measurement

## Project Overview

I built this project to study whether a learned implied-volatility surface model can reconstruct BTC and ETH options implied-volatility surfaces and produce residuals that are useful for relative-value research.

The strongest result is the modeling pipeline: I collect Deribit options data, construct standardized tenor-delta IV surfaces, and train a masked ConvVAE that performs well under structured missingness. ConvVAE residuals show in-sample mean-reversion patterns, but out-of-sample filtered strategy performance is unstable. I therefore treat the residual signal as regime-dependent research evidence, not as a finished trading strategy.

My workflow has three parts:

1. I train a convolutional variational autoencoder (ConvVAE) to reconstruct BTC/ETH implied-volatility surfaces.
2. I use ConvVAE residuals to study rich/cheap IV regions and test whether they predict future IV changes.
3. I evaluate filtered paper-IV strategies and estimate downside risk using standard VaR, CVaR, and Lambda VaR.

## Data Pipeline

I convert raw Deribit option snapshots into fixed implied-volatility grids by currency, tenor, and delta. The project stores processed surface files under `data/processed/surfaces/` and analysis outputs under `results/tables/`.

Core generated datasets:

- `results/tables/combined_structured_summary.csv`
- `results/tables/convvae_residuals.csv`
- `results/tables/iv_mean_reversion_regression.csv`
- `results/tables/filtered_residual_strategy_summary.csv`
- `results/tables/strategy_filter_comparison.csv`
- `results/tables/filtered_strategy_tail_risk.csv`

To regenerate the main analysis tables and figures:

```bash
uv run python scripts/rerun_all.py
```

## Methods

### ConvVAE Surface Reconstruction

I use the ConvVAE to learn a low-dimensional representation of the BTC/ETH IV surface and reconstruct missing or masked cells. I test reconstruction quality under random and structured missingness, including missing rows, columns, tenors, and wings.

### Residual Mean-Reversion Signal

For each surface cell:

```text
residual = market IV - ConvVAE model IV
```

A positive residual means market IV is rich versus the learned surface. A negative residual means market IV is cheap. I test:

```text
future IV change = alpha + beta * residual + error
```

Negative beta indicates mean reversion.

### Filtered Residual Strategy

I use the filtered strategy to trade only regions where historical residual betas are negative and large enough by t-stat threshold. This is mainly a research test of the residual signal. It uses paper IV PnL:

```text
PnL ~= position * change in IV
```

I treat this as a signal-quality test, not an executable options backtest or a production strategy.

I also include an out-of-sample filter check. I estimate usable negative-beta deltas on the first 70% of timestamps, then test the filtered strategy on the final 30% without letting the test period influence the filter.

### Lambda VaR

I compare standard 95% VaR, 95% CVaR, and empirical Lambda VaR on filtered strategy period returns. Lambda VaR uses a loss-dependent confidence level, making the confidence threshold more conservative in the tail.

## Results

### 1. Structured Missingness Reconstruction

![Structured RMSE](results/figures/final_01_structured_rmse.png)

My ConvVAE reconstructs masked BTC/ETH IV surface cells with low error. It is clearly useful under structured missingness such as wings and missing tenors, where static cell-mean and parametric-smile baselines can degrade sharply. Previous-surface interpolation remains the hardest benchmark when recent surfaces are available, which is expected for persistent hourly IV surfaces.

### 2. Residual Mean-Reversion Betas

![Regression Betas](results/figures/final_02_regression_betas.png)

I find that ConvVAE residuals show short-horizon mean-reversion structure in sample, especially around the 1-hour horizon. Longer horizons are more regime-sensitive and can show reversal or trend effects, so I frame the signal as unstable rather than universal.

### 3. Filtered Strategy Paper PnL

![Filtered Strategy PnL](results/figures/final_03_filtered_strategy_pnl.png)

My full-sample filtered residual strategy can produce positive paper IV PnL, but the stricter out-of-sample filter comparison in `results/tables/strategy_filter_comparison.csv` is much less stable. In the held-out test period, short BTC horizons are roughly flat, longer BTC horizons are negative, and ETH is mixed. This suggests the residual signal is regime-dependent and needs walk-forward validation before I would make a stronger trading claim.

### 4. Tail-Risk Comparison

![Tail Risk](results/figures/final_04_tail_risk.png)

I find that Lambda VaR often reports larger downside risk than standard VaR, especially for BTC and longer horizons. This is consistent with small-sample crypto-tail behavior and a conservative loss-dependent confidence function.

## Main Findings

- I find that ConvVAE is useful for BTC/ETH IV surface reconstruction under structured missingness.
- Previous-surface interpolation remains a very strong benchmark when recent surfaces are available.
- I find that ConvVAE residuals contain some in-sample predictive structure, especially at short horizons.
- My out-of-sample filtered strategy results are unstable, so I do not claim the residual signal is a robust standalone strategy.
- I reduce look-ahead bias with an out-of-sample filter check that estimates usable deltas on the first 70% of timestamps and tests on the final 30%, but this single split is not enough for a final trading conclusion.
- I use Lambda VaR to show larger downside risk than standard VaR, especially with small samples and crypto-tail behavior.

## Limitations

- My sample is still small.
- My strategy PnL is paper IV PnL, not full executable options PnL.
- My full-sample filtered strategy results can contain look-ahead bias; the out-of-sample comparison table is stricter and shows unstable performance.
- I have not included transaction costs, bid-ask spreads, margin, or vega scaling yet.
- My t-stats are preliminary because surface cells and timestamps are correlated.
- Lambda VaR sometimes equals the worst loss because the sample is limited and the lambda function is conservative.

## Next Steps

My most valuable next improvement is rolling walk-forward validation. Instead of one train/test split, I would estimate the residual filter on an expanding training window and test on the next time block:

```text
Fold 1: train 0-50%, test 50-60%
Fold 2: train 0-60%, test 60-70%
Fold 3: train 0-70%, test 70-80%
Fold 4: train 0-80%, test 80-90%
Fold 5: train 0-90%, test 90-100%
```

This would show whether the residual signal works consistently across time or only in certain windows.

After that, I would add transaction-cost-aware strategy testing. A more realistic strategy evaluation should scale positions by approximate option vega and subtract bid-ask costs:

```text
options PnL ~= vega * position * change in IV - transaction costs
```

Other useful extensions I would add:

- Add clustered or block-bootstrap inference for correlated surface cells.
- Compare ConvVAE residuals against simpler parametric surface residuals.
- Expand the sample across more market regimes.
- Add trade sizing, vega limits, and margin-aware risk controls.

## References

- “Beyond the Smile: A Hybrid Convolutional VAE for Crypto Volatility Surfaces” — used as inspiration for masked ConvVAE volatility-surface reconstruction.
- “Numerical Methods for Lambda Quantiles: Robust Evaluation and Portfolio Optimisation” — used as inspiration for the Lambda-VaR tail-risk layer.
