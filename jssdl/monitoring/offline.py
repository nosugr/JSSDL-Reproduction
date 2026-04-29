from __future__ import annotations

import numpy as np

from jssdl.model.sparse_coding import update_X1_X2

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover - fallback path depends on environment
    gaussian_kde = None


def compute_train_errors(
    Y: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    sparsity_X1: int,
    sparsity_X2: int,
    lambda1: float = 0.0,
    lambda3: float = 0.0,
    max_iter: int = 50,
    tol: float = 1.0e-6,
) -> np.ndarray:
    X1, X2 = update_X1_X2(
        Y,
        D1,
        D2,
        sparsity_X1,
        sparsity_X2,
        lambda1=lambda1,
        lambda3=lambda3,
        max_iter=max_iter,
        tol=tol,
    )
    residual = Y - D1 @ X1 - D2 @ X2
    return np.sum(residual**2, axis=0)


def kde_threshold(errors: np.ndarray, alpha: float = 0.99, grid_size: int = 4096) -> float:
    values = np.asarray(errors, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Errors array must not be empty.")
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values.max())

    if gaussian_kde is None:
        return float(np.quantile(values, alpha))

    std = float(values.std(ddof=0))
    if std <= 1.0e-12:
        return float(values.max())

    kde = gaussian_kde(values)
    lower = max(0.0, float(values.min() - 0.1 * std))
    upper = float(values.max() + 3.0 * std)
    grid = np.linspace(lower, upper, grid_size)
    density = np.maximum(kde(grid), 0.0)
    cdf = np.cumsum(density)
    if cdf[-1] <= 0:
        return float(np.quantile(values, alpha))
    cdf /= cdf[-1]
    index = min(int(np.searchsorted(cdf, alpha, side="left")), grid.size - 1)
    return float(grid[index])
