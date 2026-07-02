from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm


TARGET_TENORS = np.array([14, 30, 60, 90, 120, 180], dtype=float)
TARGET_DELTAS = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90], dtype=float)


def load_latest_snapshot(currency: str, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(f"deribit_{currency.upper()}_options_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No saved snapshots found for {currency} in {raw_dir}")

    return pd.read_parquet(files[-1])


def get_forward_proxy(df: pd.DataFrame) -> float:
    if "underlying_price" in df.columns and df["underlying_price"].notna().any():
        return float(df["underlying_price"].median())

    if "estimated_delivery_price" in df.columns and df["estimated_delivery_price"].notna().any():
        return float(df["estimated_delivery_price"].median())

    raise ValueError("No underlying/forward proxy found.")


def add_black76_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["mark_iv_decimal"] = pd.to_numeric(out["mark_iv"], errors="coerce") / 100.0
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["days_to_expiry"] = pd.to_numeric(out["days_to_expiry"], errors="coerce")

    forward = get_forward_proxy(out)
    out["forward_approx"] = forward

    T = out["days_to_expiry"] / 365.0
    sigma = out["mark_iv_decimal"]
    K = out["strike"]
    F = forward

    valid = (T > 0) & (sigma > 0) & (K > 0)

    out["d1"] = np.nan
    out.loc[valid, "d1"] = (
        np.log(F / K[valid]) + 0.5 * sigma[valid] ** 2 * T[valid]
    ) / (sigma[valid] * np.sqrt(T[valid]))

    out["call_delta_raw"] = norm.cdf(out["d1"])

    # Convert all options to call-delta-equivalent coordinates.
    out["call_delta_equiv"] = np.where(
        out["option_type"] == "call",
        out["call_delta_raw"],
        out["call_delta_raw"],  # put_delta + 1 = N(d1), same coordinate
    )

    return out


def build_expiry_delta_smiles(df: pd.DataFrame) -> dict:
    """
    For each listed expiry, build an interpolated IV smile as a function of call delta.
    """
    df = add_black76_delta_features(df)

    smiles = {}

    for expiry, g in df.groupby("expiry"):
        g = g.copy()
        g = g[
            g["call_delta_equiv"].between(0.02, 0.98)
            & g["mark_iv_decimal"].notna()
            & (g["mark_iv_decimal"] > 0)
        ]

        if len(g) < 5:
            continue

        # If calls and puts overlap at similar deltas, average them.
        g["delta_bucket"] = g["call_delta_equiv"].round(3)

        smile = (
            g.groupby("delta_bucket", as_index=False)
            .agg(
                call_delta_equiv=("call_delta_equiv", "mean"),
                mark_iv_decimal=("mark_iv_decimal", "median"),
                days_to_expiry=("days_to_expiry", "median"),
            )
            .sort_values("call_delta_equiv")
        )

        # Need enough delta coverage to interpolate most of the grid.
        if len(smile) < 5:
            continue

        smiles[expiry] = smile

    return smiles


def interpolate_iv_at_expiry(smile: pd.DataFrame, target_deltas: np.ndarray) -> np.ndarray:
    x = smile["call_delta_equiv"].to_numpy()
    y = smile["mark_iv_decimal"].to_numpy()

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate x values.
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    if len(unique_x) < 4:
        return np.full(len(target_deltas), np.nan)

    f = interp1d(
        unique_x,
        unique_y,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    return f(target_deltas)


def build_true_delta_surface(
    df: pd.DataFrame,
    target_tenors: np.ndarray = TARGET_TENORS,
    target_deltas: np.ndarray = TARGET_DELTAS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build a 6x7 surface by:
    1. interpolating IV across call-delta for each listed expiry;
    2. interpolating IV across tenor for each target delta.
    """
    smiles = build_expiry_delta_smiles(df)

    rows = []

    for expiry, smile in smiles.items():
        actual_tenor = float(smile["days_to_expiry"].median())
        ivs = interpolate_iv_at_expiry(smile, target_deltas)

        for delta, iv in zip(target_deltas, ivs):
            rows.append(
                {
                    "expiry": expiry,
                    "actual_tenor": actual_tenor,
                    "target_delta": float(delta),
                    "iv": iv,
                }
            )

    grid_long = pd.DataFrame(rows)

    surface = np.full((len(target_tenors), len(target_deltas)), np.nan)

    for j, delta in enumerate(target_deltas):
        d = grid_long[
            (grid_long["target_delta"] == float(delta))
            & grid_long["iv"].notna()
        ].sort_values("actual_tenor")

        if len(d) < 2:
            continue

        f_tenor = interp1d(
            d["actual_tenor"].to_numpy(),
            d["iv"].to_numpy(),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        surface[:, j] = f_tenor(target_tenors)

    mask = (~np.isnan(surface)).astype(float)

    return grid_long, surface, mask


def save_surface(
    surface: np.ndarray,
    mask: np.ndarray,
    currency: str,
    output_dir: str | Path = "data/processed",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{currency.upper()}_latest_true_delta_surface.npz"
    np.savez_compressed(path, surface=surface, mask=mask)

    return path


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        df = load_latest_snapshot(currency)

        grid_long, surface, mask = build_true_delta_surface(df)
        path = save_surface(surface, mask, currency)

        csv_path = Path("data/processed") / f"{currency.upper()}_latest_true_delta_grid_long.csv"
        grid_long.to_csv(csv_path, index=False)

        print(f"\n{currency} true-delta surface saved to {path}")
        print(f"Long grid saved to {csv_path}")
        print("Surface shape:", surface.shape)
        print("Mask coverage:", mask.mean())

        print("\nSurface in vol points:")
        print(pd.DataFrame(surface * 100, index=TARGET_TENORS, columns=TARGET_DELTAS).round(2))