from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from jssdl.model.sparse_coding import omp_encode
from jssdl.monitoring.offline import kde_threshold
from jssdl.utils.initializer import initialize_svd_dictionary, normalize_columns


class DictionaryLearningMonitor:
    """Dictionary-learning baseline trained with SVD-initialized K-SVD."""

    def __init__(
        self,
        n_atoms: int = 80,
        sparsity: int = 2,
        alpha: float = 0.99,
        max_iter: int = 50,
        random_state: int | None = None,
    ) -> None:
        self.n_atoms = int(n_atoms)
        self.sparsity = int(sparsity)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = random_state

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.dictionary_: np.ndarray | None = None
        self.codes_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.estimator_: None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("DictionaryLearningMonitor is not fitted yet.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def _encode_targets(self, targets: np.ndarray) -> np.ndarray:
        if self.dictionary_ is None:
            raise RuntimeError("Dictionary is not initialized.")

        Y = np.asarray(targets, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Expected a 2D feature-by-sample matrix.")

        requested_sparsity = max(1, min(self.sparsity, self.dictionary_.shape[0], self.dictionary_.shape[1]))
        return omp_encode(Y, self.dictionary_, requested_sparsity)

    def _encode_scaled(self, scaled_samples: np.ndarray) -> np.ndarray:
        return self._encode_targets(np.asarray(scaled_samples, dtype=float).T)

    def _reinitialize_unused_atom(
        self,
        Y: np.ndarray,
        dictionary: np.ndarray,
        codes: np.ndarray,
        atom_idx: int,
        rng: np.random.Generator,
    ) -> None:
        n_samples = Y.shape[1]
        group_size = min(3, n_samples)
        sample_indices = rng.choice(n_samples, size=group_size, replace=False)

        atom = np.mean(Y[:, sample_indices], axis=1)
        if np.linalg.norm(atom) <= 1.0e-12:
            atom = rng.standard_normal(Y.shape[0])
        else:
            atom = atom + 0.01 * rng.standard_normal(Y.shape[0])

        norm = max(float(np.linalg.norm(atom)), 1.0e-12)
        dictionary[:, atom_idx] = atom / norm
        codes[atom_idx, :] = 0.0

    def _ksvd_dictionary_update(
        self,
        Y: np.ndarray,
        codes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.dictionary_ is None:
            raise RuntimeError("Dictionary is not initialized.")

        dictionary = np.asarray(self.dictionary_, dtype=float).copy()
        updated_codes = np.asarray(codes, dtype=float).copy()
        n_atoms = dictionary.shape[1]

        for atom_idx in range(n_atoms):
            active_samples = np.flatnonzero(np.abs(updated_codes[atom_idx, :]) > 1.0e-12)
            if active_samples.size == 0:
                self._reinitialize_unused_atom(Y, dictionary, updated_codes, atom_idx, rng)
                continue

            residual = (
                Y[:, active_samples]
                - dictionary @ updated_codes[:, active_samples]
                + np.outer(dictionary[:, atom_idx], updated_codes[atom_idx, active_samples])
            )
            if np.linalg.norm(residual) <= 1.0e-12:
                self._reinitialize_unused_atom(Y, dictionary, updated_codes, atom_idx, rng)
                continue

            try:
                u, singular_values, vh = np.linalg.svd(residual, full_matrices=False)
            except np.linalg.LinAlgError:
                self._reinitialize_unused_atom(Y, dictionary, updated_codes, atom_idx, rng)
                continue

            dictionary[:, atom_idx] = u[:, 0]
            updated_codes[atom_idx, :] = 0.0
            updated_codes[atom_idx, active_samples] = singular_values[0] * vh[0, :]

        return normalize_columns(dictionary), updated_codes

    def fit(
        self,
        train_samples: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> "DictionaryLearningMonitor":
        X = np.asarray(train_samples, dtype=float)
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < 1.0e-12, 1.0, self.std_)
        X_scaled = self._scale(X)

        Y = X_scaled.T
        self.dictionary_ = initialize_svd_dictionary(
            Y,
            n_atoms=self.n_atoms,
            random_state=self.random_state,
        )
        rng = np.random.default_rng(self.random_state)

        progress_bar = tqdm(
            total=max(0, self.max_iter),
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for _ in range(max(0, self.max_iter)):
            codes = self._encode_targets(Y)
            self.dictionary_, self.codes_ = self._ksvd_dictionary_update(Y, codes, rng)
            progress_bar.update(1)
        progress_bar.close()

        self.codes_ = self._encode_targets(Y)
        train_errors = np.sum((Y - self.dictionary_ @ self.codes_) ** 2, axis=0)
        self.threshold_ = kde_threshold(train_errors, alpha=self.alpha)
        return self

    def score_samples(self, samples: np.ndarray) -> np.ndarray:
        if self.dictionary_ is None:
            raise RuntimeError("DictionaryLearningMonitor is not fitted yet.")
        X_scaled = self._scale(samples)
        codes = self._encode_scaled(X_scaled)
        return np.sum((X_scaled.T - self.dictionary_ @ codes) ** 2, axis=0)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold is not initialized.")
        return self.score_samples(samples) > self.threshold_
