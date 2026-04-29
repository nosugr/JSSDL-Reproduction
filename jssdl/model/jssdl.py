from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from jssdl.monitoring.offline import compute_train_errors, kde_threshold
from jssdl.monitoring.online import compute_jre, detect_fault, encode_new_sample, score_samples
from jssdl.model.dictionary_update import (
    calculate_effective_rank,
    update_D1,
    update_D2,
    update_P,
)
from jssdl.model.sparse_coding import update_X1_X2
from jssdl.utils.initializer import (
    initialize_random_dictionary,
    initialize_sparse_codes,
    initialize_svd_dictionary,
    normalize_columns,
)


@dataclass
class JSSDLHyperParams:
    n_atoms_D1: int
    n_atoms_D2: int
    sparsity_D1: int
    sparsity_X1: int
    sparsity_X2: int
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 1.0
    lambda4: float = 1.0
    lambda5: float = 1.0
    lambda6: float = 0.0
    tau: float | None = None
    max_iter: int = 100
    tol: float = 1.0e-5
    random_state: int | None = None


class JSSDL:
    """Jointly Specific and Shared Dictionary Learning."""

    def __init__(
        self,
        n_atoms_D1: int,
        n_atoms_D2: int,
        sparsity_D1: int,
        sparsity_X1: int,
        sparsity_X2: int,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        lambda4: float = 1.0,
        lambda5: float = 1.0,
        lambda6: float = 0.0,
        tau: float | None = None,
        max_iter: int = 100,
        tol: float = 1.0e-5,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_atoms_D1 = n_atoms_D1
        self.n_atoms_D2 = n_atoms_D2
        self.sparsity_D1 = sparsity_D1
        self.sparsity_X1 = sparsity_X1
        self.sparsity_X2 = sparsity_X2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        self.D1_: np.ndarray | None = None
        self.D2_: np.ndarray | None = None
        self.P_: np.ndarray | None = None
        self.X1_: np.ndarray | None = None
        self.X2_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.objective_history_: list[float] = []
        self.full_objective_history_: list[float] = []
        self.convergence_history_: list[float] = []
        self.reconstruction_history_: list[float] = []
        self.d2_gap_history_: list[float] = []
        self.nuclear_norm_history_: list[float] = []
        self.p_singular_values_history_: list[np.ndarray] = []
        self.d2_singular_values_history_: list[np.ndarray] = []
        self.rank_history_: list[int] = []
        self.n_iter_: int = 0
        self.best_iteration_: int = 0
        self._d1_reinit_rng: np.random.Generator | None = None

    def _check_fitted(self) -> None:
        if self.D1_ is None or self.D2_ is None:
            raise RuntimeError("JSSDL model is not fitted yet.")

    def _capture_state(self) -> dict[str, np.ndarray]:
        assert self.D1_ is not None and self.D2_ is not None and self.P_ is not None and self.X1_ is not None and self.X2_ is not None
        return {
            "D1": self.D1_.copy(),
            "D2": self.D2_.copy(),
            "P": self.P_.copy(),
            "X1": self.X1_.copy(),
            "X2": self.X2_.copy(),
        }

    def _restore_state(self, state: dict[str, np.ndarray]) -> None:
        self.D1_ = np.asarray(state["D1"], dtype=float).copy()
        self.D2_ = np.asarray(state["D2"], dtype=float).copy()
        self.P_ = np.asarray(state["P"], dtype=float).copy()
        self.X1_ = np.asarray(state["X1"], dtype=float).copy()
        self.X2_ = np.asarray(state["X2"], dtype=float).copy()

    @staticmethod
    def _validate_input(Y: np.ndarray) -> np.ndarray:
        matrix = np.asarray(Y, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Expected a 2D feature-by-sample matrix.")
        return matrix

    def _compute_objective(
        self,
        Y: np.ndarray,
        D1: np.ndarray,
        X1: np.ndarray,
        D2: np.ndarray,
        X2: np.ndarray,
        P: np.ndarray,
    ) -> float:
        residual = Y - D1 @ X1 - D2 @ X2
        nuclear_norm = np.linalg.svd(P, compute_uv=False).sum()
        incoherence = np.linalg.norm(D1.T @ D2, ord="fro") ** 2
        return float(
            np.linalg.norm(residual, ord="fro") ** 2
            + self.lambda1 * np.sum(np.abs(X1))
            + self.lambda2 * np.sum(np.abs(D1))
            + self.lambda3 * np.sum(np.abs(X2))
            + self.lambda4 * nuclear_norm
            + self.lambda5 * np.linalg.norm(D2 - P, ord="fro") ** 2
            + self.lambda6 * incoherence
        )

    def _update_D1(self, Y: np.ndarray) -> None:
        assert self.D2_ is not None and self.X1_ is not None and self.X2_ is not None and self.D1_ is not None
        if self._d1_reinit_rng is None:
            self._d1_reinit_rng = np.random.default_rng(None if self.random_state is None else self.random_state + 2)
        specific_part = Y - self.D2_ @ self.X2_
        self.D1_ = update_D1(
            specific_part,
            self.X1_,
            self.D1_,
            a=self.sparsity_D1,
            lambda2=self.lambda2,
            rng=self._d1_reinit_rng,
        )
        self.D1_ = normalize_columns(self.D1_)

    def _update_P(self) -> None:
        assert self.D2_ is not None
        self.P_ = update_P(self.D2_, lambda4=self.lambda4, lambda5=self.lambda5, tau=self.tau)

    def _update_D2(self, Y: np.ndarray) -> None:
        assert self.D1_ is not None and self.X1_ is not None and self.X2_ is not None and self.P_ is not None
        self.D2_ = update_D2(Y, self.D1_, self.X1_, self.X2_, self.P_, self.lambda5, self.lambda6)
        self.D2_ = normalize_columns(self.D2_)

    def _update_X1_X2(self, Y: np.ndarray) -> None:
        assert self.D1_ is not None and self.D2_ is not None
        self.X1_, self.X2_ = update_X1_X2(
            Y,
            self.D1_,
            self.D2_,
            sparsity_X1=self.sparsity_X1,
            sparsity_X2=self.sparsity_X2,
            lambda1=self.lambda1,
            lambda3=self.lambda3,
            max_iter=self._code_update_max_iter,
            tol=self.tol,
            initial_X1=self.X1_,
            initial_X2=self.X2_,
        )

    @property
    def _code_update_max_iter(self) -> int:
        return max(1, min(10, self.max_iter))

    def _initial_sparse_support(self, sparsity: int, upper_bound: int) -> int:
        if sparsity <= 0:
            return 0
        return min(upper_bound, 2 * int(sparsity))

    def fit(
        self,
        Y: np.ndarray,
        alpha: float | None = None,
        show_progress: bool = False,
        progress_desc: str = "epoch",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> "JSSDL":
        Y = self._validate_input(Y)

        n_features, n_samples = Y.shape
        self.D1_ = initialize_random_dictionary(
            n_features=n_features,
            n_atoms=self.n_atoms_D1,
            random_state=self.random_state,
            sparse_atoms=None,
        )
        self.D2_ = initialize_svd_dictionary(
            Y,
            n_atoms=self.n_atoms_D2,
            random_state=None if self.random_state is None else self.random_state + 1,
            sparse_atoms=None,
        )
        self.X1_ = initialize_sparse_codes(
            n_atoms=self.n_atoms_D1,
            n_samples=n_samples,
            sparsity=self._initial_sparse_support(self.sparsity_X1, self.n_atoms_D1),
            random_state=self.random_state,
        )
        self.X2_ = initialize_sparse_codes(
            n_atoms=self.n_atoms_D2,
            n_samples=n_samples,
            sparsity=self._initial_sparse_support(self.sparsity_X2, self.n_atoms_D2),
            random_state=None if self.random_state is None else self.random_state + 1,
        )
        self.P_ = update_P(self.D2_, lambda4=self.lambda4, lambda5=self.lambda5, tau=self.tau)
        self._d1_reinit_rng = np.random.default_rng(None if self.random_state is None else self.random_state + 2)

        self.objective_history_ = []
        self.full_objective_history_ = []
        self.convergence_history_ = []
        self.reconstruction_history_ = []
        self.d2_gap_history_ = []
        self.nuclear_norm_history_ = []
        self.p_singular_values_history_ = []
        self.d2_singular_values_history_ = []
        self.rank_history_ = []
        previous_objective: float | None = None
        best_objective: float | None = None
        best_state: dict[str, np.ndarray] | None = None
        self.best_iteration_ = 0
        convergence_patience = 2
        stable_iterations = 0
        previous_reconstruction: float | None = None

        progress_bar = tqdm(
            total=self.max_iter,
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for iteration in range(self.max_iter):
            self._update_D1(Y)
            self._update_P()
            self._update_D2(Y)
            self._update_X1_X2(Y)

            assert self.D1_ is not None and self.D2_ is not None and self.P_ is not None and self.X1_ is not None and self.X2_ is not None
            residual = Y - self.D1_ @ self.X1_ - self.D2_ @ self.X2_
            objective = self._compute_objective(Y, self.D1_, self.X1_, self.D2_, self.X2_, self.P_)
            reconstruction = float(np.linalg.norm(residual, ord="fro") ** 2)
            d2_gap = float(np.linalg.norm(self.D2_ - self.P_, ord="fro") ** 2)
            p_singular_values = np.linalg.svd(self.P_, compute_uv=False)
            d2_singular_values = np.linalg.svd(self.D2_, compute_uv=False)
            nuclear_norm = float(p_singular_values.sum())
            current_rank_d2 = calculate_effective_rank(self.D2_)
            self.objective_history_.append(objective)
            self.full_objective_history_.append(objective)
            self.reconstruction_history_.append(reconstruction)
            self.d2_gap_history_.append(d2_gap)
            self.nuclear_norm_history_.append(nuclear_norm)
            self.p_singular_values_history_.append(p_singular_values.copy())
            self.d2_singular_values_history_.append(d2_singular_values.copy())
            self.rank_history_.append(current_rank_d2)
            self.n_iter_ = iteration + 1

            if previous_objective is None:
                convergence_metric = objective
                rel_objective_change = np.inf
            else:
                convergence_metric = (previous_objective - objective)
                rel_objective_change = abs(previous_objective - objective) / max(abs(previous_objective), 1.0e-12)
            if previous_reconstruction is None:
                rel_recon_change = np.inf
            else:
                rel_recon_change = abs(previous_reconstruction - reconstruction) / max(abs(previous_reconstruction), 1.0e-12)
            self.convergence_history_.append(float(convergence_metric))
            previous_objective = objective
            previous_reconstruction = reconstruction

            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_state = self._capture_state()
                self.best_iteration_ = self.n_iter_

            if self.verbose:
                print(
                    "[JSSDL] "
                    f"iter={self.n_iter_:03d} objective={objective:.6f} "
                    f"recon={reconstruction:.6f} rank(D2)={current_rank_d2} "
                    f"gap={d2_gap:.6f}"
                )

            if show_progress:
                progress_bar.set_postfix_str(
                    f"objective={objective:.4g}, recon={reconstruction:.4g}, rank(D2)={current_rank_d2}",
                    refresh=False,
                )
                progress_bar.update(1)

            # Early stopping: both reconstruction and objective are stable for consecutive iterations.
            if rel_recon_change < 1.0e-4 or rel_objective_change < 1.0e-4:
                stable_iterations += 1
            else:
                stable_iterations = 0
            if stable_iterations >= convergence_patience:
                if self.verbose:
                    print(
                        f"[JSSDL] Converged at iteration {self.n_iter_}: "
                        f"relative reconstruction and objective changes stayed below {self.tol:g} "
                        f"for {convergence_patience} iterations"
                    )
                break
        progress_bar.close()

        if best_state is not None:
            self._restore_state(best_state)
        elif self.best_iteration_ == 0:
            self.best_iteration_ = self.n_iter_

        if alpha is not None:
            self.set_threshold(Y, alpha)
        return self

    def transform(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        matrix = self._validate_input(Y)
        assert self.D1_ is not None and self.D2_ is not None
        return update_X1_X2(
            matrix,
            self.D1_,
            self.D2_,
            self.sparsity_X1,
            self.sparsity_X2,
            lambda1=self.lambda1,
            lambda3=self.lambda3,
            max_iter=self._code_update_max_iter,
            tol=self.tol,
        )

    def reconstruct(self, Y: np.ndarray) -> np.ndarray:
        X1, X2 = self.transform(Y)
        assert self.D1_ is not None and self.D2_ is not None
        return self.D1_ @ X1 + self.D2_ @ X2

    def set_threshold(self, Y: np.ndarray, alpha: float) -> float:
        self._check_fitted()
        assert self.D1_ is not None and self.D2_ is not None
        train_errors = compute_train_errors(
            Y,
            self.D1_,
            self.D2_,
            self.sparsity_X1,
            self.sparsity_X2,
            lambda1=self.lambda1,
            lambda3=self.lambda3,
            max_iter=self._code_update_max_iter,
            tol=self.tol,
        )
        self.threshold_ = kde_threshold(train_errors, alpha=alpha)
        return self.threshold_

    def predict(self, y_new: np.ndarray) -> float | np.ndarray:
        self._check_fitted()
        assert self.D1_ is not None and self.D2_ is not None

        sample = np.asarray(y_new, dtype=float)
        if sample.ndim == 1:
            x1, x2 = encode_new_sample(
                sample,
                self.D1_,
                self.D2_,
                self.sparsity_X1,
                self.sparsity_X2,
                lambda1=self.lambda1,
                lambda3=self.lambda3,
                max_iter=self._code_update_max_iter,
                tol=self.tol,
            )
            return compute_jre(sample, self.D1_, self.D2_, x1, x2)
        if sample.ndim == 2:
            return score_samples(
                sample,
                self.D1_,
                self.D2_,
                self.sparsity_X1,
                self.sparsity_X2,
                lambda1=self.lambda1,
                lambda3=self.lambda3,
                max_iter=self._code_update_max_iter,
                tol=self.tol,
            )
        raise ValueError("Input must be a 1D sample or 2D feature-by-sample matrix.")

    def is_fault(self, y_new: np.ndarray) -> bool | np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold is not set. Call set_threshold() or fit(..., alpha=...).")
        scores = self.predict(y_new)
        return detect_fault(scores, self.threshold_)

    @property
    def hyperparams(self) -> JSSDLHyperParams:
        return JSSDLHyperParams(
            n_atoms_D1=self.n_atoms_D1,
            n_atoms_D2=self.n_atoms_D2,
            sparsity_D1=self.sparsity_D1,
            sparsity_X1=self.sparsity_X1,
            sparsity_X2=self.sparsity_X2,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            lambda3=self.lambda3,
            lambda4=self.lambda4,
            lambda5=self.lambda5,
            lambda6=self.lambda6,
            tau=self.tau,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
