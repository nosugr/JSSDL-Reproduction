from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from jssdl.model.soft_threshold import soft_threshold
from jssdl.monitoring.offline import kde_threshold


class RobustPCAMonitor:
    def __init__(
        self,
        cpv: float = 0.9,
        alpha: float = 0.99,
        lam: float | None = None,
        mu: float | None = None,
        max_iter: int = 100,
        tol: float = 1.0e-7,
    ) -> None:
        self.cpv = cpv
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.t2_threshold_: float | None = None
        self.spe_threshold_: float | None = None
        self.low_rank_train_: np.ndarray | None = None
        self.sparse_train_: np.ndarray | None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Robust PCA monitor is not fitted yet.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def _score_scaled_samples(self, Xs: np.ndarray) -> dict[str, np.ndarray]:
        if self.loadings_ is None or self.eigenvalues_ is None:
            raise RuntimeError("Robust PCA monitor is not fitted yet.")

        scaled = np.asarray(Xs, dtype=float)
        scores = scaled @ self.loadings_
        reconstruction = scores @ self.loadings_.T
        residual = scaled - reconstruction

        rt2 = np.sum((scores**2) / self.eigenvalues_, axis=1)
        rspe = np.sum(residual**2, axis=1)
        return {"t2": rt2, "spe": rspe}

    def _robust_pca(
        self,
        X: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        m, n = X.shape
        lam = self.lam if self.lam is not None else 7.0 / np.sqrt(max(m, n))
        mu = self.mu if self.mu is not None else 1.0 * lam

        low_rank = np.zeros_like(X)
        sparse = np.zeros_like(X)
        dual = np.zeros_like(X)
        norm_x = max(np.linalg.norm(X, ord="fro"), 1.0e-12)

        progress_bar = tqdm(
            total=self.max_iter,
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for iteration in range(self.max_iter):
            u, singular_values, vh = np.linalg.svd(X - sparse + dual / mu, full_matrices=False)
            shrunk = soft_threshold(singular_values, 1.0 / mu)
            rank = int(np.sum(shrunk > 0))
            if rank > 0:
                low_rank = u[:, :rank] @ np.diag(shrunk[:rank]) @ vh[:rank, :]
            else:
                low_rank = np.zeros_like(X)

            sparse = soft_threshold(X - low_rank + dual / mu, lam / mu)
            residual = X - low_rank - sparse
            dual = dual + mu * residual
            residual_ratio = np.linalg.norm(residual, ord="fro") / norm_x
            current_iter = iteration + 1
            converged = residual_ratio < self.tol

            if show_progress:
                progress_bar.set_postfix_str(f"residual={residual_ratio:.3e}", refresh=False)
                progress_bar.update(1)

            if converged:
                break
        progress_bar.close()

        return low_rank, sparse

    def fit(
        self,
        train_samples: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> "RobustPCAMonitor":
        X = np.asarray(train_samples, dtype=float)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < 1.0e-12, 1.0, self.std_)
        Xs = self._scale(X)

        self.low_rank_train_, self.sparse_train_ = self._robust_pca(
            Xs,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=progress_leave,
        )
        _, singular_values, vt = np.linalg.svd(self.low_rank_train_, full_matrices=False)
        eigenvalues = singular_values**2 / max(1, Xs.shape[0] - 1)
        explained = np.cumsum(eigenvalues) / max(eigenvalues.sum(), 1.0e-12)
        n_components = int(np.searchsorted(explained, self.cpv, side="left") + 1)

        self.loadings_ = vt[:n_components].T
        self.eigenvalues_ = np.maximum(eigenvalues[:n_components], 1.0e-12)

        # Estimate thresholds from the RPCA low-rank training block so the
        # training residual uses L - LPP^T instead of the original scaled data.
        training_scores = self._score_scaled_samples(self.low_rank_train_)
        self.t2_threshold_ = kde_threshold(training_scores["t2"], alpha=self.alpha)
        self.spe_threshold_ = kde_threshold(training_scores["spe"], alpha=self.alpha)
        return self

    def score_samples(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        Xs = self._scale(samples)
        return self._score_scaled_samples(Xs)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        scores = self.score_samples(samples)
        if self.t2_threshold_ is None or self.spe_threshold_ is None:
            raise RuntimeError("Thresholds are not initialized.")

        return np.logical_or(scores["t2"] > self.t2_threshold_, scores["spe"] > self.spe_threshold_)
