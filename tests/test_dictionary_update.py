from __future__ import annotations

import numpy as np
import pytest

from jssdl.model.dictionary_update import (
    calculate_effective_rank,
    update_D1,
    update_D2,
    update_P,
)


def test_update_p_shrinks_singular_values() -> None:
    D2 = np.diag([4.0, 2.0, 0.5])
    updated = update_P(D2, tau=2.0, lambda4=2.0, lambda5=2.0)
    singular_values = np.linalg.svd(updated, compute_uv=False)
    np.testing.assert_allclose(singular_values, np.array([2.0, 0.0, 0.0]), atol=1.0e-8)


def test_update_d1_respects_column_sparsity() -> None:
    rng = np.random.default_rng(0)
    Y1 = rng.normal(size=(6, 10))
    X1 = rng.normal(size=(4, 10))
    D1 = rng.normal(size=(6, 4))
    updated = update_D1(Y1, X1, D1, a=2)
    nonzero_per_atom = np.sum(np.abs(updated) > 1.0e-12, axis=0)
    assert np.all(nonzero_per_atom <= 2)


def test_update_d2_matches_closed_form_solution() -> None:
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 7))
    D1 = rng.normal(size=(5, 3))
    X1 = rng.normal(size=(3, 7))
    X2 = rng.normal(size=(4, 7))
    P = rng.normal(size=(5, 4))
    lambda5 = 0.7

    updated = update_D2(Y, D1, X1, X2, P, lambda5)
    expected = ((Y - D1 @ X1) @ X2.T + lambda5 * P) @ np.linalg.inv(X2 @ X2.T + lambda5 * np.eye(4))
    np.testing.assert_allclose(updated, expected, atol=1.0e-8)


def test_update_d2_with_incoherence_penalty_solves_sylvester_equation() -> None:
    rng = np.random.default_rng(12)
    Y = rng.normal(size=(5, 8))
    D1 = rng.normal(size=(5, 3))
    X1 = rng.normal(size=(3, 8))
    X2 = rng.normal(size=(4, 8))
    P = rng.normal(size=(5, 4))
    lambda5 = 0.7
    lambda6 = 0.1

    updated = update_D2(Y, D1, X1, X2, P, lambda5, lambda6=lambda6)

    left = lambda6 * (D1 @ D1.T) @ updated + updated @ (X2 @ X2.T + lambda5 * np.eye(4))
    right = (Y - D1 @ X1) @ X2.T + lambda5 * P
    np.testing.assert_allclose(left, right, atol=1.0e-8)


def test_update_d1_reinitializes_atom_from_residual_when_codes_are_zero() -> None:
    Y1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    X1 = np.array([[1.0e-8, 0.0, 0.0]], dtype=float)
    D1 = np.array([[0.1], [3.0], [0.2], [2.0]], dtype=float)

    updated = update_D1(Y1, X1, D1, a=2, rng=np.random.default_rng(0), noise_scale=0.0)

    expected = np.array([[1.0 / np.sqrt(10.0)], [0.0], [3.0 / np.sqrt(10.0)], [0.0]], dtype=float)
    np.testing.assert_allclose(updated, expected, atol=1.0e-12)
    np.testing.assert_allclose(X1, np.zeros_like(X1), atol=1.0e-12)


def test_calculate_effective_rank_uses_cumulative_singular_value_energy() -> None:
    matrix = np.diag([4.0, 2.0, 0.5])
    assert calculate_effective_rank(matrix) == 3

    another_matrix = np.diag([4.0, 2.0, 1.0])
    assert calculate_effective_rank(another_matrix, energy_threshold=0.95) == 2
    assert calculate_effective_rank(another_matrix, energy_threshold=0.99) == 3


def test_calculate_effective_rank_returns_zero_for_zero_matrix() -> None:
    assert calculate_effective_rank(np.zeros((3, 3), dtype=float)) == 0


def test_calculate_effective_rank_validates_energy_threshold() -> None:
    with pytest.raises(ValueError):
        calculate_effective_rank(np.eye(2, dtype=float), energy_threshold=0.0)
