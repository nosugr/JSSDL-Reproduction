from __future__ import annotations

import numpy as np

from jssdl.utils import data_loader


def test_generate_numerical_dataset_injects_fault_bias_on_fixed_feature() -> None:
    clean_dataset = data_loader.generate_numerical_dataset(
        n_features=8,
        shared_rank=2,
        n_train=10,
        n_test_normal=4,
        n_test_fault=6,
        fault_feature=3,
        fault_bias=0.0,
        noise_std=0.0,
        random_state=3,
    )
    dataset = data_loader.generate_numerical_dataset(
        n_features=8,
        shared_rank=2,
        n_train=10,
        n_test_normal=4,
        n_test_fault=6,
        fault_feature=3,
        fault_bias=5.0,
        noise_std=0.0,
        random_state=3,
    )

    assert int(dataset["fault_feature"]) == 3
    delta = dataset["faulty_test"] - clean_dataset["faulty_test"]
    assert np.all(delta[:, :3] == 0.0)
    assert np.all(delta[:, 4:] == 0.0)
    np.testing.assert_allclose(delta[:, 3], np.full(6, 5.0), atol=1.0e-12)


def test_generate_numerical_dataset_uses_shared_seed_for_observation_matrix() -> None:
    dataset_a = data_loader.generate_numerical_dataset(
        n_features=8,
        shared_rank=2,
        n_train=10,
        n_test_normal=4,
        n_test_fault=6,
        random_state=3,
    )
    dataset_b = data_loader.generate_numerical_dataset(
        n_features=8,
        shared_rank=2,
        n_train=10,
        n_test_normal=4,
        n_test_fault=6,
        random_state=99,
    )
    dataset_c = data_loader.generate_numerical_dataset(
        n_features=8,
        shared_rank=2,
        n_train=10,
        n_test_normal=4,
        n_test_fault=6,
        random_state=99,
        shared_seed=7,
    )

    np.testing.assert_allclose(dataset_a["observation_matrix"], dataset_b["observation_matrix"], atol=1.0e-12)
    assert not np.allclose(dataset_a["observation_matrix"], dataset_c["observation_matrix"])
