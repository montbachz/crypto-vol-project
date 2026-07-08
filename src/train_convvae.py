from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_all_surfaces
from masking import make_random_mask, apply_mask, rmse_hidden
from models import ConvVAE, vae_loss


def time_split(X: np.ndarray, train_frac=0.7, val_frac=0.15):
    n = len(X)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train = X[:n_train]
    val = X[n_train:n_train + n_val]
    test = X[n_train + n_val:]

    return train, val, test


def normalize(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std


def make_training_arrays(X: np.ndarray, seed=123):
    rng = np.random.default_rng(seed)

    x_true_list = []
    x_masked_list = []
    mask_list = []

    for surface in X:
        mask_rate = rng.uniform(0.1, 0.5)
        obs_mask = make_random_mask(surface.shape, mask_rate, rng)
        masked = apply_mask(surface, obs_mask)
        masked = np.nan_to_num(masked, nan=0.0)

        x_true_list.append(surface)
        x_masked_list.append(masked)
        mask_list.append(obs_mask)

    x_true = torch.tensor(np.array(x_true_list), dtype=torch.float32).unsqueeze(1)
    x_masked = torch.tensor(np.array(x_masked_list), dtype=torch.float32).unsqueeze(1)
    masks = torch.tensor(np.array(mask_list), dtype=torch.float32).unsqueeze(1)

    return x_true, x_masked, masks


def train_one_currency(currency: str, epochs=300, batch_size=32, seed=42):
    X, M, meta = load_all_surfaces(currency)
    X = X[~np.isnan(X).any(axis=(1, 2))]

    if len(X) < 30:
        raise ValueError(f"You only have {len(X)} clean surfaces. ConvVAE will train better with 50+.")

    train, val, test = time_split(X)
    train_z, val_z, test_z, mean, std = normalize(train, val, test)

    x_true, x_masked, masks = make_training_arrays(train_z, seed=seed)
    ds = TensorDataset(x_true, x_masked, masks)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvVAE(latent_dim=16, hidden_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb_true, xb_masked, xb_mask in loader:
            xb_true = xb_true.to(device)
            xb_masked = xb_masked.to(device)
            xb_mask = xb_mask.to(device)

            recon, mu, logvar = model(xb_masked, xb_mask)
            loss, hidden_mse, observed_mse, kl = vae_loss(
                recon, xb_true, xb_mask, mu=mu, logvar=logvar
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        if epoch % 25 == 0 or epoch == 1:
            print(f"{currency} epoch {epoch:4d} | loss {np.mean(losses):.6f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "currency": currency,
            "latent_dim": 16,
            "hidden_channels": 32,
        },
        f"models/{currency}_convvae.pt",
    )

    print(f"Saved model to models/{currency}_convvae.pt")

    return model, test_z, mean, std, device


def evaluate_convvae(currency: str, mask_rates=(0.1, 0.3, 0.5), seed=100):
    model, test_z, mean, std, device = train_one_currency(currency)

    rng = np.random.default_rng(seed)
    rows = []

    model.eval()

    for mask_rate in mask_rates:
        for surface_z in test_z:
            obs_mask = make_random_mask(surface_z.shape, mask_rate, rng)
            masked_z = apply_mask(surface_z, obs_mask)
            masked_z = np.nan_to_num(masked_z, nan=0.0)

            x_masked = torch.tensor(masked_z, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            mask_t = torch.tensor(obs_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_z, _, _ = model(x_masked, mask_t)

            pred_z = pred_z.squeeze().cpu().numpy()

            # Convert back to IV decimals for RMSE in vol points
            true = surface_z * std + mean
            pred = pred_z * std + mean

            rows.append(
                {
                    "currency": currency,
                    "mask_rate": mask_rate,
                    "method": "convvae",
                    "rmse_vol_points": rmse_hidden(true, pred, obs_mask) * 100,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_results = []

    for currency in ["BTC", "ETH"]:
        try:
            results = evaluate_convvae(currency)
            all_results.append(results)
        except Exception as e:
            print(f"Skipping {currency}: {e}")

    if all_results:
        results = pd.concat(all_results, ignore_index=True)

        summary = (
            results.groupby(["currency", "mask_rate", "method"], as_index=False)
            .agg(
                mean_rmse=("rmse_vol_points", "mean"),
                median_rmse=("rmse_vol_points", "median"),
                n=("rmse_vol_points", "size"),
            )
        )

        print("\nConvVAE RMSE summary, vol points:")
        print(summary.round(4))

        Path("results/tables").mkdir(parents=True, exist_ok=True)
        results.to_csv("results/tables/convvae_raw_results.csv", index=False)
        summary.to_csv("results/tables/convvae_summary.csv", index=False)