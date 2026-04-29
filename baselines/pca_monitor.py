from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from jssdl.monitoring.offline import kde_threshold


class PCAMonitor:
    def __init__(self, cpv: float = 0.90, alpha: float = 0.99) -> None:
        self.cpv = cpv
        self.alpha = alpha

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.t2_threshold_: float | None = None
        self.spe_threshold_: float | None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("PCA monitor is not fitted yet.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit(
        self,
        train_samples: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> "PCAMonitor":
        progress_bar = tqdm(
            total=4,
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        try:
            X = np.asarray(train_samples, dtype=float)
            self.mean_ = X.mean(axis=0, keepdims=True)
            self.std_ = X.std(axis=0, keepdims=True)
            self.std_ = np.where(self.std_ < 1.0e-12, 1.0, self.std_)
            Xs = self._scale(X)
            if show_progress:
                progress_bar.set_postfix_str("step=scale", refresh=False)
                progress_bar.update(1)

            _, singular_values, vt = np.linalg.svd(Xs, full_matrices=False)
            eigenvalues = singular_values**2 / max(1, Xs.shape[0] - 1)
            explained = np.cumsum(eigenvalues) / max(eigenvalues.sum(), 1.0e-12)
            n_components = int(np.searchsorted(explained, self.cpv, side="left") + 1)
            if show_progress:
                progress_bar.set_postfix_str(f"step=svd, k={n_components}", refresh=False)
                progress_bar.update(1)

            self.loadings_ = vt[:n_components].T
            self.eigenvalues_ = np.maximum(eigenvalues[:n_components], 1.0e-12)

            scores = self.score_samples(X)
            if show_progress:
                progress_bar.set_postfix_str("step=score", refresh=False)
                progress_bar.update(1)
            self.t2_threshold_ = kde_threshold(scores["t2"], alpha=self.alpha)
            self.spe_threshold_ = kde_threshold(scores["spe"], alpha=self.alpha)
            if show_progress:
                progress_bar.set_postfix_str("step=threshold", refresh=False)
                progress_bar.update(1)
            return self
        finally:
            progress_bar.close()

    def score_samples(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        if self.loadings_ is None or self.eigenvalues_ is None:
            raise RuntimeError("PCA monitor is not fitted yet.")

        Xs = self._scale(samples)
        scores = Xs @ self.loadings_
        reconstruction = scores @ self.loadings_.T
        residual = Xs - reconstruction

        t2 = np.sum((scores**2) / self.eigenvalues_, axis=1)
        spe = np.sum(residual**2, axis=1)
        return {"t2": t2, "spe": spe}

    def predict(self, samples: np.ndarray) -> np.ndarray:
        scores = self.score_samples(samples)
        if self.t2_threshold_ is None or self.spe_threshold_ is None:
            raise RuntimeError("Thresholds are not initialized.")

        return np.logical_or(scores["t2"] > self.t2_threshold_, scores["spe"] > self.spe_threshold_)
