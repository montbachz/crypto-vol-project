from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_TENORS = np.array([14, 30, 60, 90, 120, 180], dtype=float)
TARGET_DELTAS = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90], dtype=float)


def load_latest_snapshot(currency: str, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """
    Load most recent Deribit options snapshot for BTC or ETH.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob(f"deribit_{currency.upper()}_options_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No saved snapshots found for {currency} in {raw_dir}")

    return pd.read_parquet(files[-1])


def normalize_iv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deribit mark_iv appears to be in vol points, e.g. 39.88 = 39.88%.
    Convert to decimal too.
    """
    out = df.copy()
    out["mark_iv_decimal"] = out["mark_iv"] / 100.0
    return out


def approximate_forward(df: pd.DataFrame) -> float:
    """
    First-pass approximation: use median underlying_price if available.
    This is not perfect, but good enough to build the first surface.
    """
    if "underlying_price" in df.columns and df["underlying_price"].notna().any():
        return float(df["underlying_price"].median())

    if "estimated_delivery_price" in df.columns and df["estimated_delivery_price"].notna().any():
        return float(df["estimated_delivery_price"].median())

    raise ValueError("Could not infer underlying/forward price from snapshot.")


def add_moneyness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add K/F and log(K/F).
    """
    out = normalize_iv(df)
    forward = approximate_forward(out)

    out["forward_approx"] = forward
    out["moneyness"] = out["strike"] / forward
    out["log_moneyness"] = np.log(out["moneyness"])

    return out


def choose_nearest_expiry(df: pd.DataFrame, target_tenor: float) -> pd.DataFrame:
    """
    Choose options whose expiry is closest to the target tenor.
    """
    expiries = (
        df[["expiry", "days_to_expiry"]]
        .drop_duplicates()
        .assign(distance=lambda x: (x["days_to_expiry"] - target_tenor).abs())
        .sort_values("distance")
    )

    if expiries.empty:
        return df.iloc[0:0].copy()

    chosen_expiry = expiries.iloc[0]["expiry"]
    return df[df["expiry"] == chosen_expiry].copy()


def build_simple_iv_surface(
    df: pd.DataFrame,
    target_tenors: np.ndarray = TARGET_TENORS,
    target_deltas: np.ndarray = TARGET_DELTAS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build a first-pass 6x7 IV surface.

    Important simplification:
    Instead of exact delta interpolation, this version maps deltas to moneyness
    buckets. This is only the MVP surface builder.

    Later we will replace this with Black-76 delta calculation and interpolation.
    """
    df = add_moneyness_features(df)

    # Crude mapping from call-delta buckets to moneyness targets.
    # Low call delta = OTM call = higher strike.
    delta_to_moneyness = {
        0.10: 1.25,
        0.20: 1.15,
        0.30: 1.08,
        0.50: 1.00,
        0.70: 0.94,
        0.80: 0.90,
        0.90: 0.85,
    }

    surface = np.full((len(target_tenors), len(target_deltas)), np.nan)
    mask = np.zeros_like(surface)

    rows = []

    for i, tenor in enumerate(target_tenors):
        tenor_df = choose_nearest_expiry(df, tenor)

        if tenor_df.empty:
            continue

        actual_tenor = float(tenor_df["days_to_expiry"].median())

        for j, delta in enumerate(target_deltas):
            target_moneyness = delta_to_moneyness[float(delta)]

            # For now, use calls for delta <= 0.50 and puts for delta > 0.50
            # only to get both wings represented. Later we will use true call deltas.
            if delta <= 0.50:
                candidates = tenor_df[tenor_df["option_type"] == "call"].copy()
            else:
                candidates = tenor_df[tenor_df["option_type"] == "put"].copy()

            if candidates.empty:
                candidates = tenor_df.copy()

            candidates["distance"] = (candidates["moneyness"] - target_moneyness).abs()
            best = candidates.sort_values("distance").head(1)

            if best.empty:
                continue

            iv = float(best.iloc[0]["mark_iv_decimal"])
            surface[i, j] = iv
            mask[i, j] = 1.0

            rows.append(
                {
                    "target_tenor": tenor,
                    "actual_tenor": actual_tenor,
                    "target_delta": delta,
                    "target_moneyness": target_moneyness,
                    "instrument_name": best.iloc[0]["instrument_name"],
                    "strike": best.iloc[0]["strike"],
                    "option_type": best.iloc[0]["option_type"],
                    "moneyness": best.iloc[0]["moneyness"],
                    "mark_iv": best.iloc[0]["mark_iv"],
                    "mark_iv_decimal": iv,
                }
            )

    surface_df = pd.DataFrame(rows)

    return surface_df, surface, mask


def save_surface(
    surface: np.ndarray,
    mask: np.ndarray,
    currency: str,
    output_dir: str | Path = "data/processed",
) -> Path:
    """
    Save surface and mask as compressed numpy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{currency.upper()}_latest_surface.npz"
    np.savez_compressed(path, surface=surface, mask=mask)

    return path


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        df = load_latest_snapshot(currency)
        surface_df, surface, mask = build_simple_iv_surface(df)
        path = save_surface(surface, mask, currency)
        
        csv_path = Path("data/processed") / f"{currency.upper()}_latest_surface_matches.csv"
        surface_df.to_csv(csv_path, index=False)
        print(f"Matched instruments saved to {csv_path}")

        print(f"\n{currency} surface saved to {path}")
        print("Surface shape:", surface.shape)
        print("Mask coverage:", mask.mean())
        print("\nSurface in vol points:")
        print(pd.DataFrame(surface * 100, index=TARGET_TENORS, columns=TARGET_DELTAS).round(2))

        print("\nMatched instruments:")
        print(surface_df.head(15))