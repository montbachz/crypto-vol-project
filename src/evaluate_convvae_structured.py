from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import load_all_surfaces
from masking import apply_mask, rmse_hidden, make_structured_mask
from models import ConvVAE


def time_split(X: np.ndarray, train_frac=0.7, val_frac=0.15):
    """
    Must match the split used in train_convvae.py.
    """
    n = len(X)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train = X[:n_train]
    val = X[n_train:n_train + n_val]
    test = X[n_train + n_val:]

    return train, val, test


def load_trained_model(currency: str, device: str):
    checkpoint_path = f"models/{currency.upper()}_convvae.pt"

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Missing model checkpoint: {checkpoint_path}. Run python src/train_convvae.py first."
        )

    try:
        # Preferred safe path for newly saved checkpoints.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Backward-compatible path for old checkpoints you created yourself
        # that stored NumPy arrays.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    latent_dim = int(ckpt.get("latent_dim", 16))
    hidden_channels = int(ckpt.get("hidden_channels", 32))

    model = ConvVAE(latent_dim=latent_dim, hidden_channels=hidden_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"]
    std = ckpt["std"]

    # Convert tensors or old NumPy arrays into NumPy arrays for downstream math.
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()

    return model, mean, std


def evaluate_currency(
    currency: str,
    schemes=("row_random", "long_tenor", "col_random", "put_wing", "call_wing"),
    seed: int = 123,
) -> pd.DataFrame:
    X, M, meta = load_all_surfaces(currency)

    # Keep only complete surfaces.
    X = X[~np.isnan(X).any(axis=(1, 2))]

    if len(X) < 10:
        raise ValueError(f"Only {len(X)} clean surfaces for {currency}. Need more data.")

    train, val, test = time_split(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std = load_trained_model(currency, device)

    test_z = (test - mean) / std

    rng = np.random.default_rng(seed)
    rows = []

    for scheme in schemes:
        for surface, surface_z in zip(test, test_z):
            obs_mask = make_structured_mask(surface.shape, scheme=scheme, rng=rng)

            masked_z = apply_mask(surface_z, obs_mask)
            masked_z = np.nan_to_num(masked_z, nan=0.0)

            x_masked = (
                torch.tensor(masked_z, dtype=torch.float32)
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
                pred_z, _, _ = model(x_masked, mask_t)

            pred_z = pred_z.squeeze().cpu().numpy()
            pred = pred_z * std + mean

            rows.append(
                {
                    "currency": currency.upper(),
                    "scheme": scheme,
                    "method": "convvae",
                    "rmse_vol_points": rmse_hidden(surface, pred, obs_mask) * 100,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []

    for currency in ["BTC", "ETH"]:
        try:
            results = evaluate_currency(currency)
            all_results.append(results)
        except Exception as e:
            print(f"Skipping {currency}: {e}")

    if not all_results:
        raise RuntimeError("No ConvVAE structured results produced.")

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

    print("\nStructured ConvVAE RMSE summary, vol points:")
    print(summary.round(4))

    Path("results/tables").mkdir(parents=True, exist_ok=True)

    results.to_csv("results/tables/convvae_structured_raw_results.csv", index=False)
    summary.to_csv("results/tables/convvae_structured_summary.csv", index=False)

    print("\nSaved:")
    print("results/tables/convvae_structured_raw_results.csv")
    print("results/tables/convvae_structured_summary.csv")