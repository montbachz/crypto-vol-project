from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import load_all_surfaces
from models import ConvVAE
from surface_delta import TARGET_TENORS, TARGET_DELTAS


def load_trained_model(currency: str, device: str):
    checkpoint_path = Path(f"models/{currency.upper()}_convvae.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {checkpoint_path}. Run python src/train_convvae.py first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    latent_dim = int(ckpt.get("latent_dim", 16))
    hidden_channels = int(ckpt.get("hidden_channels", 32))

    model = ConvVAE(latent_dim=latent_dim, hidden_channels=hidden_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"]
    std = ckpt["std"]

    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()

    return model, mean, std


def parse_timestamp(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, format="%Y%m%d_%H%M%S", utc=True)


def reconstruct_full_surfaces(currency: str) -> pd.DataFrame:
    """
    Reconstruct full unmasked surfaces and save cell-level residuals.

    residual = market_iv - model_iv
    """
    X, M, meta = load_all_surfaces(currency)

    valid = ~np.isnan(X).any(axis=(1, 2))
    X = X[valid].copy()
    meta = meta[valid].reset_index(drop=True).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_trained_model(currency, device)

    rows = []

    model.eval()

    for idx, surface in enumerate(X):
        timestamp_raw = meta.loc[idx, "timestamp"]
        timestamp = parse_timestamp(timestamp_raw)

        surface_z = (surface - mean) / std

        # Full observed mask: no cells hidden.
        obs_mask = np.ones_like(surface_z, dtype=float)

        x_in = (
            torch.tensor(surface_z, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        mask_t = (
            torch.tensor(obs_mask, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            pred_z, mu, logvar = model(x_in, mask_t)

        pred_z = pred_z.squeeze().cpu().numpy()
        pred = pred_z * std + mean

        residual = surface - pred
        anomaly_score = float(np.sqrt(np.mean(residual ** 2)) * 100)

        for i, tenor in enumerate(TARGET_TENORS):
            for j, delta in enumerate(TARGET_DELTAS):
                rows.append(
                    {
                        "currency": currency.upper(),
                        "timestamp_raw": timestamp_raw,
                        "timestamp_utc": timestamp,
                        "tenor": float(tenor),
                        "delta": float(delta),
                        "market_iv": float(surface[i, j]),
                        "model_iv": float(pred[i, j]),
                        "residual": float(residual[i, j]),
                        "market_iv_vol_points": float(surface[i, j] * 100),
                        "model_iv_vol_points": float(pred[i, j] * 100),
                        "residual_vol_points": float(residual[i, j] * 100),
                        "anomaly_score_vol_points": anomaly_score,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = []

    for currency in ["BTC", "ETH"]:
        df = reconstruct_full_surfaces(currency)
        all_rows.append(df)

    residuals = pd.concat(all_rows, ignore_index=True)

    output_path = Path("results/tables/convvae_residuals.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(output_path, index=False)

    print("\nSaved residual dataset:")
    print(output_path)

    print("\nResidual summary, vol points:")
    summary = (
        residuals.groupby(["currency"], as_index=False)
        .agg(
            mean_abs_residual=("residual_vol_points", lambda x: x.abs().mean()),
            median_abs_residual=("residual_vol_points", lambda x: x.abs().median()),
            mean_anomaly_score=("anomaly_score_vol_points", "mean"),
            max_anomaly_score=("anomaly_score_vol_points", "max"),
            n_cells=("residual_vol_points", "size"),
        )
    )

    print(summary.round(4))

    print("\nLargest anomaly snapshots:")
    anomaly = (
        residuals.groupby(["currency", "timestamp_utc"], as_index=False)
        .agg(anomaly_score=("anomaly_score_vol_points", "first"))
        .sort_values("anomaly_score", ascending=False)
        .head(10)
    )

    print(anomaly.round(4))