from __future__ import annotations

import numpy as np


def make_random_mask(
    shape: tuple[int, int],
    mask_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a binary observation mask for one surface.

    1 = observed cell
    0 = hidden cell
    """
    n_cells = shape[0] * shape[1]
    n_hidden = int(round(mask_rate * n_cells))

    flat = np.ones(n_cells, dtype=float)
    hidden_idx = rng.choice(n_cells, size=n_hidden, replace=False)
    flat[hidden_idx] = 0.0

    return flat.reshape(shape)


def apply_mask(surface: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """
    Replace hidden cells with NaN.
    """
    masked = surface.copy()
    masked[obs_mask == 0] = np.nan
    return masked


def rmse_hidden(true: np.ndarray, pred: np.ndarray, obs_mask: np.ndarray) -> float:
    """
    RMSE only on hidden cells.
    """
    hidden = obs_mask == 0
    err = true[hidden] - pred[hidden]
    return float(np.sqrt(np.nanmean(err ** 2)))

def make_structured_mask(
    shape: tuple[int, int],
    scheme: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Structured observation masks.

    1 = observed
    0 = hidden
    """
    mask = np.ones(shape, dtype=float)
    n_rows, n_cols = shape

    if scheme == "row_random":
        row = rng.integers(0, n_rows)
        mask[row, :] = 0.0

    elif scheme == "long_tenor":
        mask[-1, :] = 0.0

    elif scheme == "col_random":
        col = rng.integers(0, n_cols)
        mask[:, col] = 0.0

    elif scheme == "put_wing":
        # high call-delta side: 0.80, 0.90
        mask[:, -2:] = 0.0

    elif scheme == "call_wing":
        # low call-delta side: 0.10, 0.20
        mask[:, :2] = 0.0

    else:
        raise ValueError(f"Unknown structured mask scheme: {scheme}")

    return mask