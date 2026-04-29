from __future__ import annotations

import numpy as np
from scipy.linalg import solve_sylvester

from .soft_threshold import soft_threshold


def _top_support_indices(scores: np.ndarray, support_size: int) -> np.ndarray:
    if support_size <= 0:
        return np.empty(0, dtype=int)
    if support_size >= scores.shape[0]:
        return np.arange(scores.shape[0], dtype=int)
    return np.argpartition(scores, -support_size)[-support_size:]


def _reinitialize_specific_atom(
    specific_residual: np.ndarray,
    support_size: int,
    rng: np.random.Generator,
    noise_scale: float = 0.01,
) -> np.ndarray:
    n_samples = specific_residual.shape[1]
    group_size = min(3, n_samples)
    sample_indices = rng.choice(n_samples, size=group_size, replace=False)

    atom = np.mean(specific_residual[:, sample_indices], axis=1)
    if np.linalg.norm(atom) <= 1.0e-12:
        atom = rng.standard_normal(specific_residual.shape[0])

    keep = _top_support_indices(np.abs(atom), support_size)
    sparse_atom = np.zeros_like(atom, dtype=float)
    sparse_atom[keep] = atom[keep]
    sparse_atom[keep] += float(noise_scale) * rng.standard_normal(keep.shape[0])

    norm = max(float(np.linalg.norm(sparse_atom)), 1.0e-12)
    return sparse_atom / norm


def update_D1(
    Y1: np.ndarray,
    X1: np.ndarray,
    D1: np.ndarray,
    a: int,
    lambda2: float | None = None,
    rng: np.random.Generator | None = None,
    noise_scale: float = 0.01,
) -> np.ndarray:
    """Update the specific dictionary following the paper's row-support rule."""
    del lambda2

    Y1 = np.asarray(Y1, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    D1 = np.asarray(D1, dtype=float).copy()

    n_features, n_atoms = D1.shape
    support_size = max(0, min(int(a), n_features))
    eps = 1.0e-6
    reinit_rng = np.random.default_rng() if rng is None else rng

    for atom_idx in range(n_atoms):
        xj = X1[atom_idx, :]
        residual = Y1 - D1 @ X1 + np.outer(D1[:, atom_idx], xj)
        x_norm_sq = float(np.dot(xj, xj))
        if support_size == 0:
            D1[:, atom_idx] = 0.0
            continue

        if x_norm_sq <= eps:
            D1[:, atom_idx] = _reinitialize_specific_atom(Y1, support_size, reinit_rng, noise_scale=noise_scale)
            X1[atom_idx, :] = 0.0
            continue

        row_scores = np.sum(np.abs(residual), axis=1)
        keep_indices = _top_support_indices(row_scores, support_size)
        updated_atom = np.zeros(n_features, dtype=float)
        updated_atom[keep_indices] = residual[keep_indices, :] @ xj / x_norm_sq

        if np.linalg.norm(updated_atom) <= eps:
            updated_atom = _reinitialize_specific_atom(Y1, support_size, reinit_rng, noise_scale=noise_scale)
            X1[atom_idx, :] = 0.0
        D1[:, atom_idx] = updated_atom
    return D1


def update_P(
    D2: np.ndarray,
    lambda4: float = 1.0,
    lambda5: float = 1.0,
    tau: float | None = None,
) -> np.ndarray:
    """Update the low-rank auxiliary matrix via singular-value soft thresholding.

    When *tau* is ``None`` (default), the threshold is computed from the
    paper's closed-form solution: ``τ = λ₄ / (2λ₅)``.
    When *tau* is provided explicitly (e.g. during sensitivity analysis),
    that value is used directly.
    """
    u, singular_values, vh = np.linalg.svd(np.asarray(D2, dtype=float), full_matrices=False)
    if tau is not None:
        effective_tau = max(float(tau), 0.0)
    else:
        effective_tau = float(lambda4) / (2.0 * max(float(lambda5), 1e-12))
    shrunk = soft_threshold(singular_values, effective_tau)
    return u @ np.diag(shrunk) @ vh


def update_D2(
    Y: np.ndarray,
    D1: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    P: np.ndarray,
    lambda5: float,
    lambda6: float = 0.0,
) -> np.ndarray:
    """Update the shared dictionary.

    With ``lambda6 == 0`` this is the original closed-form update. When an
    incoherence penalty ``lambda6 * ||D1.T @ D2||_F^2`` is enabled, the
    optimality condition becomes a Sylvester equation:
    ``lambda6 * D1 @ D1.T @ D2 + D2 @ (X2 @ X2.T + lambda5 * I) = C``.
    """
    n_atoms = X2.shape[0]
    identity = np.eye(n_atoms, dtype=float)
    numerator = (Y - D1 @ X1) @ X2.T + float(lambda5) * P
    denominator = X2 @ X2.T + float(lambda5) * identity
    if float(lambda6) <= 0.0:
        return np.linalg.solve(denominator.T, numerator.T).T

    left_penalty = float(lambda6) * (D1 @ D1.T)
    return solve_sylvester(left_penalty, denominator, numerator)


def calculate_effective_rank(matrix: np.ndarray, energy_threshold: float = 0.998) -> int:
    """Compute the effective rank from cumulative singular-value energy."""
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError("energy_threshold must be in (0, 1].")

    singular_values = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    singular_energy = np.square(singular_values)
    total_energy = float(np.sum(singular_energy))
    if total_energy <= 1.0e-12:
        return 0

    cumulative_energy_ratio = np.cumsum(singular_energy) / total_energy
    return int(np.searchsorted(cumulative_energy_ratio, float(energy_threshold), side="left") + 1)
