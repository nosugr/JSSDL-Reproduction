from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray | float, tau: float) -> np.ndarray | float:
    """Apply the element-wise soft-thresholding operator."""
    values = np.asarray(x, dtype=float)
    shrunk = np.sign(values) * np.maximum(np.abs(values) - tau, 0.0)
    if np.isscalar(x):
        return float(shrunk)
    return shrunk
