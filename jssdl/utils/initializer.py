from __future__ import annotations

import numpy as np


def normalize_columns(matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(normalized, axis=0, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    normalized /= norms
    return normalized


def project_column_sparsity(matrix: np.ndarray, sparsity: int | None) -> np.ndarray:
    projected = np.asarray(matrix, dtype=float).copy()
    if sparsity is None or sparsity >= projected.shape[0]:
        return projected
    if sparsity <= 0:
        return np.zeros_like(projected)

    support_size = int(sparsity)
    for column in range(projected.shape[1]):
        keep = np.argpartition(np.abs(projected[:, column]), -support_size)[-support_size:]
        column_values = np.zeros(projected.shape[0], dtype=float)
        column_values[keep] = projected[keep, column]
        projected[:, column] = column_values
    return projected


def initialize_random_dictionary(
    n_features: int,
    n_atoms: int,
    random_state: int | None = None,
    sparse_atoms: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    dictionary = rng.standard_normal(size=(n_features, n_atoms))
    dictionary = project_column_sparsity(dictionary, sparse_atoms)
    return normalize_columns(dictionary)


def initialize_svd_dictionary(
    Y: np.ndarray,
    n_atoms: int,
    random_state: int | None = None,
    sparse_atoms: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")

    n_features, n_samples = matrix.shape
    if n_features == 0 or n_samples == 0:
        raise ValueError("Input matrix must contain at least one feature and one sample.")

    atom_count = int(n_atoms)
    if atom_count < 0:
        raise ValueError("Number of atoms must be non-negative.")
    if atom_count == 0:
        return np.empty((n_features, 0), dtype=float)

    rng = np.random.default_rng(random_state)
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=True)
    n_direct_atoms = min(atom_count, u.shape[1])
    atoms = [u[:, atom_idx].copy() for atom_idx in range(n_direct_atoms)]

    if len(atoms) < atom_count:
        rank = int(np.sum(singular_values > 1.0e-12))
        basis_width = max(1, min(rank, u.shape[1]))
        basis = u[:, :basis_width]
        weights = singular_values[:basis_width]
        weights = weights / max(float(np.max(weights)), 1.0e-12)
        for _ in range(atom_count - len(atoms)):
            coefficients = rng.standard_normal(basis_width) * np.sqrt(weights)
            atom = basis @ coefficients
            if np.linalg.norm(atom) <= 1.0e-12:
                atom = rng.standard_normal(n_features)
            atoms.append(atom)

    dictionary = np.column_stack(atoms)
    dictionary = project_column_sparsity(dictionary, sparse_atoms)
    return normalize_columns(dictionary)


def initialize_sparse_codes(
    n_atoms: int,
    n_samples: int,
    sparsity: int,
    random_state: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    codes = np.zeros((n_atoms, n_samples), dtype=float)
    support_size = max(0, min(int(sparsity), n_atoms))
    if support_size == 0:
        return codes

    for sample_idx in range(n_samples):
        active = rng.choice(n_atoms, size=support_size, replace=False)
        codes[active, sample_idx] = rng.standard_normal(size=support_size)
    return codes


def initialize_random_codes(
    n_atoms: int,
    n_samples: int,
    sparsity: int,
    random_state: int | None = None,
) -> np.ndarray:
    return initialize_sparse_codes(
        n_atoms=n_atoms,
        n_samples=n_samples,
        sparsity=sparsity,
        random_state=random_state,
    )


def initialize_dictionary_from_data(
    Y: np.ndarray,
    n_atoms: int,
    random_state: int | None = None,
    sparse_atoms: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    n_features, n_samples = matrix.shape
    if n_samples == 0:
        raise ValueError("Input matrix must contain at least one sample.")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_samples, size=n_atoms, replace=n_samples < n_atoms)
    dictionary = matrix[:, indices].copy()
    dictionary += 1.0e-3 * rng.standard_normal(size=(n_features, n_atoms))
    dictionary = project_column_sparsity(dictionary, sparse_atoms)
    return normalize_columns(dictionary)
