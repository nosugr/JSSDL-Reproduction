from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def standardize_train_test(
    train_samples: np.ndarray,
    test_samples: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray | None, tuple[np.ndarray, np.ndarray]]:
    train = np.asarray(train_samples, dtype=float)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    train_scaled = (train - mean) / std
    if test_samples is None:
        return train_scaled, None, (mean, std)

    test = np.asarray(test_samples, dtype=float)
    test_scaled = (test - mean) / std
    return train_scaled, test_scaled, (mean, std)


def apply_standardization(samples: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return (np.asarray(samples, dtype=float) - mean) / std


def to_feature_sample_matrix(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=float).T


def _sample_numerical_block(
    n_samples: int,
    n_features: int,
    shared_rank: int,
    noise_std: float,
    rng: np.random.Generator,
    observation_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    data = np.zeros((n_samples, n_features), dtype=float)
    specific_component = np.zeros_like(data)
    shared_component = np.zeros_like(data)

    for idx in range(n_samples):
        k1, k2 = rng.normal(loc=1.0, scale=1.0, size=2)
        ia, ib = rng.choice(n_features, size=2, replace=False)

        y1 = np.zeros(n_features, dtype=float)
        y1[ia] = k1
        y1[ib] = k2

        state = rng.normal(loc=1.0, scale=0.5, size=shared_rank)
        y2 = observation_matrix @ state
        noise = rng.normal(loc=0.0, scale=noise_std, size=n_features)

        specific_weight =k1
        shared_weight =k2

        specific_component[idx] = specific_weight * y1
        shared_component[idx] = shared_weight * y2
        data[idx] = specific_component[idx] + shared_component[idx] + noise
    return {
        "data": data,
        "specific_component": specific_component,
        "shared_component": shared_component,
    }


def _inject_fault_bias(
    samples: np.ndarray,
    fault_bias: float,
    fault_feature: int,
) -> np.ndarray:
    biased_samples = np.asarray(samples, dtype=float).copy()
    n_features = biased_samples.shape[1]
    feature_index = int(fault_feature)
    if not 0 <= feature_index < n_features:
        raise ValueError(f"fault_feature must be in [0, {n_features - 1}], got {feature_index}.")
    biased_samples[:, feature_index] += float(fault_bias)
    return biased_samples


def generate_numerical_dataset(
    n_features: int = 20,
    shared_rank: int = 2,
    n_train: int = 1000,
    n_test_normal: int = 500,
    n_test_fault: int = 500,
    fault_feature: int = 6,
    fault_bias: float = 10.0,
    noise_std: float = 0.1,
    random_state: int | None = None,
    shared_seed: int | None = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(random_state)
    _ = rng.normal(size=(n_features, shared_rank))
    observation_rng = np.random.default_rng(shared_seed)
    observation_matrix = observation_rng.normal(size=(n_features, shared_rank))

    train_block = _sample_numerical_block(n_train, n_features, shared_rank, noise_std, rng, observation_matrix)
    normal_test_block = _sample_numerical_block(n_test_normal, n_features, shared_rank, noise_std, rng, observation_matrix)
    faulty_test_block = _sample_numerical_block(n_test_fault, n_features, shared_rank, noise_std, rng, observation_matrix)
    train = train_block["data"]
    normal_test = normal_test_block["data"]
    faulty_test = _inject_fault_bias(
        faulty_test_block["data"],
        fault_bias=fault_bias,
        fault_feature=fault_feature,
    )

    test = np.vstack([normal_test, faulty_test])
    labels = np.concatenate([np.zeros(n_test_normal, dtype=int), np.ones(n_test_fault, dtype=int)])

    return {
        "train": train,
        "train_specific_component": train_block["specific_component"],
        "train_shared_component": train_block["shared_component"],
        "test": test,
        "labels": labels,
        "normal_test": normal_test,
        "faulty_test": faulty_test,
        "fault_feature": int(fault_feature),
        "observation_matrix": observation_matrix,
    }


