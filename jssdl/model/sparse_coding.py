from __future__ import annotations

import numpy as np


def _omp_single(
    y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    residual_tol: float,
    selection_tol: float = 1.0e-12,
) -> np.ndarray:
    n_features, n_atoms = dictionary.shape
    if y.shape[0] != n_features:
        raise ValueError("Sample dimension does not match dictionary dimension.")
    if sparsity <= 0 or n_atoms == 0:
        return np.zeros(n_atoms, dtype=float)

    atom_norms = np.linalg.norm(dictionary, axis=0)
    valid = atom_norms > 1.0e-6
    if not np.any(valid):
        return np.zeros(n_atoms, dtype=float)

    normalized_dictionary = np.zeros_like(dictionary)
    normalized_dictionary[:, valid] = dictionary[:, valid] / atom_norms[valid]

    residual = y.copy()
    active: list[int] = []
    normalized_coeffs = np.array([], dtype=float)

    for _ in range(min(sparsity, int(valid.sum()))):
        correlations = np.abs(normalized_dictionary.T @ residual)
        correlations[~valid] = -np.inf
        if active:
            correlations[np.asarray(active)] = -np.inf

        chosen = int(np.argmax(correlations))
        if not np.isfinite(correlations[chosen]) or correlations[chosen] <= selection_tol:
            break

        active.append(chosen)
        active_dictionary = normalized_dictionary[:, active]

        # Solve least-squares coefficients on the active set.
        normalized_coeffs, *_ = np.linalg.lstsq(active_dictionary, y, rcond=None)

        residual = y - active_dictionary @ normalized_coeffs
        if np.linalg.norm(residual) <= residual_tol:
            break

    coefficients = np.zeros(n_atoms, dtype=float)
    if active:
        active_array = np.asarray(active)
        coefficients[active_array] = normalized_coeffs / atom_norms[active_array]
    return coefficients


def omp_encode(
    Y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    tol: float = 1.0e-8,
    selection_tol: float = 1.0e-12,
) -> np.ndarray:
    """Encode one sample or a batch of samples with Orthogonal Matching Pursuit."""
    samples = np.asarray(Y, dtype=float)
    dictionary = np.asarray(dictionary, dtype=float)

    residual_tol = max(float(tol), 0.0)
    effective_selection_tol = max(float(selection_tol), 0.0)

    if samples.ndim == 1:
        return _omp_single(
            samples,
            dictionary,
            sparsity,
            residual_tol,
            selection_tol=effective_selection_tol,
        )
    if samples.ndim != 2:
        raise ValueError("Y must be a 1D sample or a 2D feature-by-sample matrix.")

    codes = np.zeros((dictionary.shape[1], samples.shape[1]), dtype=float)
    for column in range(samples.shape[1]):
        codes[:, column] = _omp_single(
            samples[:, column],
            dictionary,
            sparsity,
            residual_tol,
            selection_tol=effective_selection_tol,
        )
    return codes


def update_X1_X2(
    Y: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    sparsity_X1: int,
    sparsity_X2: int,
    tol: float = 1.0e-6,
    lambda1: float = 0.0,
    lambda3: float = 0.0,
    max_iter: int = 0,
    initial_X1: np.ndarray | None = None,
    initial_X2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Update sparse codes using the paper's three-step OMP routine."""
    del lambda1, lambda3, max_iter, initial_X1, initial_X2

    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")

    X2 = omp_encode(matrix, D2, sparsity_X2, tol=tol)
    X1 = omp_encode(matrix - D2 @ X2, D1, sparsity_X1, tol=tol)
    X2 = omp_encode(matrix - D1 @ X1, D2, sparsity_X2, tol=tol)
    return X1, X2
