from __future__ import annotations

import numpy as np


def cell_mean_fill(masked_surface: np.ndarray, cell_means: np.ndarray) -> np.ndarray:
    """
    Fill hidden cells using training-set mean for each cell.
    """
    pred = masked_surface.copy()
    missing = np.isnan(pred)
    pred[missing] = cell_means[missing]
    return pred


def row_mean_fill(masked_surface: np.ndarray, cell_means: np.ndarray) -> np.ndarray:
    """
    Fill hidden cells using row mean from observed cells.
    If a full row is missing, fallback to cell means.
    """
    pred = masked_surface.copy()

    for i in range(pred.shape[0]):
        row = pred[i]
        missing = np.isnan(row)

        if not missing.any():
            continue

        if np.isfinite(row).any():
            fill_value = np.nanmean(row)
            row[missing] = fill_value
        else:
            row[missing] = cell_means[i, missing]

        pred[i] = row

    return pred


def previous_surface_fill(masked_surface: np.ndarray, previous_surface: np.ndarray, cell_means: np.ndarray) -> np.ndarray:
    """
    Fill hidden cells using previous snapshot's surface.
    Fallback to cell means if needed.
    """
    pred = masked_surface.copy()
    missing = np.isnan(pred)

    pred[missing] = previous_surface[missing]

    still_missing = np.isnan(pred)
    pred[still_missing] = cell_means[still_missing]

    return pred


from scipy.stats import norm


TARGET_DELTAS = np.array([0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90], dtype=float)


def quadratic_smile_fill(
    masked_surface: np.ndarray,
    cell_means: np.ndarray,
    deltas: np.ndarray = TARGET_DELTAS,
) -> np.ndarray:
    """
    Fill hidden cells row-by-row using a quadratic smile fit.

    For each tenor row, fit:
        IV(delta) = a + b * z + c * z^2
    where z = Phi^{-1}(delta).

    If fewer than 3 observed cells exist in a row, fallback to cell means.
    """
    pred = masked_surface.copy()
    z = norm.ppf(deltas)

    for i in range(pred.shape[0]):
        row = pred[i]
        observed = np.isfinite(row)
        missing = ~observed

        if not missing.any():
            continue

        if observed.sum() >= 3:
            coeffs = np.polyfit(z[observed], row[observed], deg=2)
            fitted = np.polyval(coeffs, z)
            row[missing] = fitted[missing]
        else:
            row[missing] = cell_means[i, missing]

        pred[i] = row

    return pred