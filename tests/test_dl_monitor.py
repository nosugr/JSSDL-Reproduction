from __future__ import annotations

import numpy as np

import baselines.dl_monitor as dl_monitor_module
from baselines import DictionaryLearningMonitor
from jssdl.utils.initializer import initialize_svd_dictionary


def test_dictionary_learning_monitor_initializes_dictionary_with_shared_svd_initializer() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(16, 5))
    monitor = DictionaryLearningMonitor(n_atoms=8, max_iter=0, random_state=4)

    monitor.fit(train, show_progress=False)

    assert monitor.dictionary_ is not None
    scaled = (train - train.mean(axis=0, keepdims=True)) / train.std(axis=0, keepdims=True)
    expected = initialize_svd_dictionary(scaled.T, n_atoms=8, random_state=4)
    np.testing.assert_allclose(monitor.dictionary_, expected, atol=1.0e-12)


def test_dictionary_learning_monitor_fit_runs_ksvd_with_sparse_omp_codes() -> None:
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(32, 2))
    mixing = rng.normal(size=(2, 6))
    train = latent @ mixing + 0.05 * rng.normal(size=(32, 6))
    test = rng.normal(size=(7, 6))

    monitor = DictionaryLearningMonitor(
        n_atoms=10,
        sparsity=3,
        alpha=0.95,
        max_iter=2,
        random_state=3,
    )
    fitted = monitor.fit(train, show_progress=False)

    assert fitted is monitor
    assert monitor.estimator_ is None
    assert monitor.dictionary_ is not None
    assert monitor.codes_ is not None
    assert monitor.threshold_ is not None
    assert monitor.dictionary_.shape == (6, 10)
    assert monitor.codes_.shape == (10, 32)
    np.testing.assert_allclose(np.linalg.norm(monitor.dictionary_, axis=0), np.ones(10), atol=1.0e-8)
    assert np.all(np.sum(np.abs(monitor.codes_) > 1.0e-8, axis=0) <= 3)

    scores = monitor.score_samples(test)
    predictions = monitor.predict(test)
    assert scores.shape == (test.shape[0],)
    assert predictions.shape == (test.shape[0],)
    assert np.all(np.isfinite(scores))


def test_dictionary_learning_monitor_uses_project_omp_encode(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    train = rng.normal(size=(12, 4))
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    original_omp_encode = dl_monitor_module.omp_encode

    def spy_omp_encode(
        Y: np.ndarray,
        dictionary: np.ndarray,
        sparsity: int,
        tol: float = 1.0e-8,
        selection_tol: float = 1.0e-12,
    ) -> np.ndarray:
        calls.append((Y.shape, dictionary.shape, sparsity))
        return original_omp_encode(
            Y,
            dictionary,
            sparsity,
            tol=tol,
            selection_tol=selection_tol,
        )

    monkeypatch.setattr(dl_monitor_module, "omp_encode", spy_omp_encode)

    monitor = DictionaryLearningMonitor(
        n_atoms=6,
        sparsity=2,
        max_iter=1,
        random_state=5,
    )
    monitor.fit(train, show_progress=False)
    monitor.score_samples(train[:3])

    assert calls
    assert all(call[2] == 2 for call in calls)
