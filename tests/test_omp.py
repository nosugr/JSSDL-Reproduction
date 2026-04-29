from __future__ import annotations

import numpy as np

from jssdl.model.sparse_coding import omp_encode, update_X1_X2


def test_omp_recovers_simple_sparse_code() -> None:
    dictionary = np.eye(4)
    true_code = np.array([1.5, 0.0, -2.0, 0.0])
    sample = dictionary @ true_code
    estimated = omp_encode(sample, dictionary, sparsity=2)
    np.testing.assert_allclose(estimated, true_code, atol=1.0e-8)


def test_omp_batch_shape() -> None:
    dictionary = np.eye(3)
    samples = np.eye(3)
    codes = omp_encode(samples, dictionary, sparsity=1)
    assert codes.shape == (3, 3)


def test_omp_selection_tol_is_decoupled_from_residual_tol() -> None:
    dictionary = np.eye(2)
    sample = np.array([5.0e-7, 0.0], dtype=float)

    estimated = omp_encode(
        sample,
        dictionary,
        sparsity=1,
        tol=1.0e-6,
        selection_tol=0.0,
    )

    np.testing.assert_allclose(estimated, sample, atol=1.0e-12)


def test_update_x1_x2_follows_three_step_omp() -> None:
    Y = np.array([[1.5, -0.5], [0.25, -2.0]], dtype=float)
    D1 = np.eye(2)
    D2 = np.zeros((2, 1), dtype=float)

    X1, X2 = update_X1_X2(
        Y,
        D1,
        D2,
        sparsity_X1=2,
        sparsity_X2=0,
        lambda1=0.0,
        lambda3=0.0,
        max_iter=200,
        tol=1.0e-10,
    )

    np.testing.assert_allclose(X1, Y, atol=1.0e-8)
    np.testing.assert_allclose(X2, np.zeros((1, Y.shape[1])), atol=1.0e-8)
