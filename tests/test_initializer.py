from __future__ import annotations

import numpy as np

from jssdl import JSSDL
from jssdl.utils.initializer import initialize_sparse_codes, initialize_svd_dictionary


def test_initialize_sparse_codes_respects_requested_sparsity() -> None:
    codes = initialize_sparse_codes(
        n_atoms=6,
        n_samples=5,
        sparsity=2,
        random_state=0,
    )

    assert codes.shape == (6, 5)
    assert np.any(codes != 0.0)
    assert np.all(np.count_nonzero(codes, axis=0) <= 2)


def test_initialize_svd_dictionary_uses_singular_vectors_and_fills_extra_atoms() -> None:
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 12))

    dictionary_a = initialize_svd_dictionary(Y, n_atoms=8, random_state=4)
    dictionary_b = initialize_svd_dictionary(Y, n_atoms=8, random_state=4)
    dictionary_c = initialize_svd_dictionary(Y, n_atoms=8, random_state=7)
    u, _, _ = np.linalg.svd(Y, full_matrices=True)

    assert dictionary_a.shape == (5, 8)
    np.testing.assert_allclose(dictionary_a[:, :5], u[:, :5], atol=1.0e-12)
    np.testing.assert_allclose(dictionary_a, dictionary_b, atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.norm(dictionary_a, axis=0), np.ones(8), atol=1.0e-8)
    assert not np.allclose(dictionary_a[:, 5:], dictionary_c[:, 5:])


def test_jssdl_fit_initializes_d1_dense_and_codes_sparsely() -> None:
    model = JSSDL(
        n_atoms_D1=6,
        n_atoms_D2=8,
        sparsity_D1=2,
        sparsity_X1=2,
        sparsity_X2=3,
        max_iter=0,
        random_state=0,
    )
    Y = np.ones((8, 4), dtype=float)

    model.fit(Y, alpha=None)

    assert model.D1_ is not None and model.X1_ is not None and model.X2_ is not None
    d1_support = np.count_nonzero(model.D1_, axis=0)
    x1_support = np.count_nonzero(model.X1_, axis=0)
    x2_support = np.count_nonzero(model.X2_, axis=0)

    assert np.all(d1_support == Y.shape[0])
    assert np.all(x1_support <= model._initial_sparse_support(model.sparsity_X1, model.n_atoms_D1))
    assert np.all(x2_support <= model._initial_sparse_support(model.sparsity_X2, model.n_atoms_D2))


def test_jssdl_fit_initializes_d2_from_svd_dictionary() -> None:
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(6, 10))
    model = JSSDL(
        n_atoms_D1=5,
        n_atoms_D2=8,
        sparsity_D1=2,
        sparsity_X1=2,
        sparsity_X2=3,
        max_iter=0,
        random_state=11,
    )

    model.fit(Y, alpha=None)

    assert model.D2_ is not None
    expected_d2 = initialize_svd_dictionary(Y, n_atoms=8, random_state=12)
    np.testing.assert_allclose(model.D2_, expected_d2, atol=1.0e-12)
