from __future__ import annotations

import numpy as np

import baselines.robust_pca_monitor as robust_pca_module
from baselines import PCAMonitor, RobustPCAMonitor


def test_pca_monitor_fit_supports_progress_hooks() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(32, 6))
    test = rng.normal(size=(8, 6))

    monitor = PCAMonitor(cpv=0.9, alpha=0.95)
    fitted = monitor.fit(
        train,
        show_progress=False,
        progress_desc="epoch[PCA][test]",
        progress_position=0,
        progress_leave=False,
    )

    assert fitted is monitor
    assert monitor.loadings_ is not None
    assert monitor.eigenvalues_ is not None
    assert monitor.t2_threshold_ is not None
    assert monitor.spe_threshold_ is not None

    scores = monitor.score_samples(test)
    predictions = monitor.predict(test)
    assert scores["t2"].shape == (test.shape[0],)
    assert scores["spe"].shape == (test.shape[0],)
    assert predictions.shape == (test.shape[0],)


def test_robust_pca_monitor_fit_estimates_thresholds_from_low_rank_train(
    monkeypatch,
) -> None:
    train = np.array(
        [
            [1.0, 2.0, 0.5],
            [2.0, 0.0, 1.5],
            [3.0, 1.0, 2.5],
            [4.0, 3.0, 3.5],
        ],
        dtype=float,
    )
    low_rank = np.array(
        [
            [-1.0, -0.5, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 1.0],
            [2.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    sparse = np.zeros_like(low_rank)
    captured_threshold_inputs: list[np.ndarray] = []

    def fake_robust_pca(
        self: RobustPCAMonitor,
        X: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return low_rank, sparse

    def fake_kde_threshold(errors: np.ndarray, alpha: float = 0.99, grid_size: int = 4096) -> float:
        del alpha, grid_size
        captured_threshold_inputs.append(np.asarray(errors, dtype=float).copy())
        return float(np.max(errors))

    monkeypatch.setattr(RobustPCAMonitor, "_robust_pca", fake_robust_pca)
    monkeypatch.setattr(robust_pca_module, "kde_threshold", fake_kde_threshold)

    monitor = RobustPCAMonitor(cpv=0.9, alpha=0.95)
    monitor.fit(train, show_progress=False)

    expected_scores = monitor._score_scaled_samples(low_rank)
    np.testing.assert_allclose(captured_threshold_inputs[0], expected_scores["t2"])
    np.testing.assert_allclose(captured_threshold_inputs[1], expected_scores["spe"])

    raw_training_scores = monitor.score_samples(train)
    assert not np.allclose(captured_threshold_inputs[1], raw_training_scores["spe"])
