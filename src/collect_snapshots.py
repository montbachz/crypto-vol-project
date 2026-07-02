from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import fetch_option_book_summary, add_option_metadata, clean_option_snapshot
from surface_delta import build_true_delta_surface, TARGET_TENORS, TARGET_DELTAS


def save_timestamped_surface(
    currency: str,
    raw_df: pd.DataFrame,
    grid_long: pd.DataFrame,
    surface: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path = "data/processed/surfaces",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    npz_path = output_dir / f"{currency.upper()}_surface_{timestamp}.npz"
    np.savez_compressed(
        npz_path,
        surface=surface,
        mask=mask,
        tenors=TARGET_TENORS,
        deltas=TARGET_DELTAS,
    )

    grid_path = output_dir / f"{currency.upper()}_grid_long_{timestamp}.csv"
    grid_long = grid_long.copy()
    grid_long["currency"] = currency.upper()
    grid_long["snapshot_timestamp_utc"] = timestamp
    grid_long.to_csv(grid_path, index=False)

    raw_path = Path("data/raw") / f"deribit_{currency.upper()}_options_{timestamp}.parquet"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_path, index=False)

    print(f"{currency}: saved surface to {npz_path}")
    print(f"{currency}: mask coverage = {mask.mean():.3f}")
    print(pd.DataFrame(surface * 100, index=TARGET_TENORS, columns=TARGET_DELTAS).round(2))


def collect_once(currency: str) -> None:
    raw = fetch_option_book_summary(currency)
    parsed = add_option_metadata(raw)
    clean = clean_option_snapshot(parsed)

    grid_long, surface, mask = build_true_delta_surface(clean)

    save_timestamped_surface(
        currency=currency,
        raw_df=clean,
        grid_long=grid_long,
        surface=surface,
        mask=mask,
    )


def collect_loop(interval_minutes: int = 60) -> None:
    while True:
        print("\n" + "=" * 80)
        print(f"Collecting snapshots at {datetime.now(timezone.utc).isoformat()}")

        for currency in ["BTC", "ETH"]:
            try:
                collect_once(currency)
            except Exception as e:
                print(f"ERROR collecting {currency}: {e}")

        print(f"Sleeping for {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    # For now, collect one snapshot and stop.
    for currency in ["BTC", "ETH"]:
        collect_once(currency)

    # Later, uncomment this to collect every hour:
    # collect_loop(interval_minutes=60)