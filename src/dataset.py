from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_surface_file(path: Path) -> dict:
    data = np.load(path)
    return {
        "path": path,
        "surface": data["surface"],
        "mask": data["mask"],
        "tenors": data["tenors"],
        "deltas": data["deltas"],
    }


def load_all_surfaces(
    currency: str,
    surface_dir: str | Path = "data/processed/surfaces",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load all timestamped surfaces for one currency.

    Returns
    -------
    X : np.ndarray
        Shape (n_snapshots, 6, 7)
    masks : np.ndarray
        Shape (n_snapshots, 6, 7)
    meta : pd.DataFrame
        Metadata with file paths and timestamps.
    """
    surface_dir = Path(surface_dir)
    files = sorted(surface_dir.glob(f"{currency.upper()}_surface_*.npz"))

    if not files:
        raise FileNotFoundError(f"No surfaces found for {currency} in {surface_dir}")

    surfaces = []
    masks = []
    rows = []

    for path in files:
        item = load_surface_file(path)
        surface = item["surface"]
        mask = item["mask"]

        if surface.shape != (6, 7):
            print(f"Skipping {path}, unexpected shape {surface.shape}")
            continue

        surfaces.append(surface)
        masks.append(mask)

        # filename format: BTC_surface_20260702_211805.npz
        timestamp = path.stem.replace(f"{currency.upper()}_surface_", "")

        rows.append(
            {
                "currency": currency.upper(),
                "timestamp": timestamp,
                "path": str(path),
                "mask_coverage": float(mask.mean()),
                "has_nan": bool(np.isnan(surface).any()),
            }
        )

    X = np.stack(surfaces)
    M = np.stack(masks)
    meta = pd.DataFrame(rows)

    return X, M, meta


def zscore_surfaces(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-cell z-score normalization.

    Returns normalized X, cell means, cell stds.
    """
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)

    # avoid divide by zero if you only have a few snapshots
    stds = np.where(stds < 1e-8, 1.0, stds)

    X_z = (X - means) / stds
    return X_z, means, stds


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        X, M, meta = load_all_surfaces(currency)

        print(f"\n{currency}")
        print("X shape:", X.shape)
        print("Mask shape:", M.shape)
        print("Number of snapshots:", len(meta))
        print("Average mask coverage:", meta["mask_coverage"].mean())
        print("Any NaNs:", meta["has_nan"].any())

        print("\nLatest surface, vol points:")
        print(np.round(X[-1] * 100, 2))