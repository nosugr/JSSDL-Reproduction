from __future__ import annotations

import numpy as np

from jssdl.model.sparse_coding import update_X1_X2


def encode_new_sample(
    y_new: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    sparsity_X1: int,
    sparsity_X2: int,
    lambda1: float = 0.0,
    lambda3: float = 0.0,
    max_iter: int = 50,
    tol: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    sample_matrix = np.asarray(y_new, dtype=float).reshape(-1, 1)
    X1, X2 = update_X1_X2(
        sample_matrix,
        D1,
        D2,
        sparsity_X1,
        sparsity_X2,
        lambda1=lambda1,
        lambda3=lambda3,
        max_iter=max_iter,
        tol=tol,
    )
    return X1[:, 0], X2[:, 0]


def compute_jre(y_new: np.ndarray, D1: np.ndarray, D2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
    sample = np.asarray(y_new, dtype=float).ravel()
    reconstruction = D1 @ x1 + D2 @ x2
    return float(np.sum((sample - reconstruction) ** 2))


def score_samples(
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
    samples = np.asarray(Y, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")
    X1, X2 = update_X1_X2(
        samples,
        D1,
        D2,
        sparsity_X1,
        sparsity_X2,
        lambda1=lambda1,
        lambda3=lambda3,
        max_iter=max_iter,
        tol=tol,
    )
    residual = samples - D1 @ X1 - D2 @ X2
    return np.sum(residual**2, axis=0)


def detect_fault(score: float | np.ndarray, threshold: float) -> bool | np.ndarray:
    return np.asarray(score) > threshold
