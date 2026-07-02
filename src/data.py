from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests


DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"


def deribit_get(endpoint: str, params: dict | None = None, timeout: int = 20) -> dict:
    url = f"{DERIBIT_BASE_URL}/{endpoint.lstrip('/')}"
    response = requests.get(url, params=params or {}, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Deribit API error: {payload['error']}")

    return payload


def fetch_option_book_summary(currency: Literal["BTC", "ETH"] = "BTC") -> pd.DataFrame:
    """
    Fetch one live Deribit option-chain snapshot.
    """
    payload = deribit_get(
        "public/get_book_summary_by_currency",
        params={"currency": currency.upper(), "kind": "option"},
    )

    rows = payload.get("result", [])
    if not rows:
        raise ValueError(f"No option rows returned for currency={currency}")

    df = pd.DataFrame(rows)
    df["fetch_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    df["currency"] = currency.upper()

    return df


def parse_deribit_option_name(instrument_name: str) -> dict:
    """
    Parse names like BTC-27JUN25-100000-C or ETH-26SEP25-4000-P.
    """
    pattern = (
        r"^(?P<currency>[A-Z]+)-"
        r"(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-"
        r"(?P<strike>[0-9.]+)-"
        r"(?P<option_type>[CP])$"
    )

    match = re.match(pattern, instrument_name)

    if match is None:
        return {
            "parsed_currency": np.nan,
            "expiry": pd.NaT,
            "strike": np.nan,
            "option_type": np.nan,
        }

    parts = match.groupdict()
    expiry = datetime.strptime(parts["expiry"], "%d%b%y").replace(tzinfo=timezone.utc)

    return {
        "parsed_currency": parts["currency"],
        "expiry": expiry,
        "strike": float(parts["strike"]),
        "option_type": "call" if parts["option_type"] == "C" else "put",
    }


def add_option_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add expiry, strike, option type, and days to expiry.
    """
    parsed = df["instrument_name"].apply(parse_deribit_option_name).apply(pd.Series)
    out = pd.concat([df.copy(), parsed], axis=1)

    fetch_ts = pd.to_datetime(out["fetch_timestamp_utc"], utc=True)
    expiry_ts = pd.to_datetime(out["expiry"], utc=True)

    out["days_to_expiry"] = (expiry_ts - fetch_ts).dt.total_seconds() / (24 * 3600)

    if {"bid_price", "ask_price"}.issubset(out.columns):
        out["mid_price"] = (out["bid_price"] + out["ask_price"]) / 2

    for col in ["mark_iv", "bid_iv", "ask_iv"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def clean_option_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleanup for first-pass option-chain snapshots.
    """
    out = df.copy()

    numeric_cols = [
        "strike",
        "days_to_expiry",
        "mark_price",
        "bid_price",
        "ask_price",
        "mid_price",
        "mark_iv",
        "open_interest",
        "volume",
        "underlying_price",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["days_to_expiry"] > 0]
    out = out[out["strike"] > 0]

    if "mark_iv" in out.columns:
        out = out[out["mark_iv"].notna()]
        out = out[out["mark_iv"] > 0]

    return out.reset_index(drop=True)


def save_snapshot(
    df: pd.DataFrame,
    currency: Literal["BTC", "ETH"],
    output_dir: Path | str = "data/raw",
) -> Path:
    """
    Save cleaned snapshot as a parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"deribit_{currency.upper()}_options_{ts}.parquet"

    df.to_parquet(path, index=False)
    return path


def fetch_parse_save_snapshot(currency: Literal["BTC", "ETH"] = "BTC") -> tuple[pd.DataFrame, Path]:
    """
    Full pipeline:
    fetch Deribit option chain -> parse metadata -> clean -> save.
    """
    raw = fetch_option_book_summary(currency)
    parsed = add_option_metadata(raw)
    clean = clean_option_snapshot(parsed)
    path = save_snapshot(clean, currency)

    return clean, path


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        df, path = fetch_parse_save_snapshot(currency)

        print(f"\n{currency}: saved {len(df):,} option rows to {path}")
        print(
            df[
                [
                    "instrument_name",
                    "expiry",
                    "strike",
                    "option_type",
                    "days_to_expiry",
                    "mark_iv",
                ]
            ].head()
        )

        time.sleep(0.5)