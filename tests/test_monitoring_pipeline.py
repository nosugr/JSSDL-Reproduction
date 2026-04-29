from __future__ import annotations

import numpy as np

from jssdl import JSSDL
from jssdl.monitoring.metrics import compute_far, compute_fdr
from jssdl.utils.data_loader import generate_numerical_dataset, standardize_train_test, to_feature_sample_matrix


def test_jssdl_monitoring_pipeline_separates_faults() -> None:
    dataset = generate_numerical_dataset(
        n_features=10,
        shared_rank=2,
        n_train=120,
        n_test_normal=40,
        n_test_fault=40,
        fault_bias=6.0,
        noise_std=0.05,
        random_state=7,
    )
    train_scaled, test_scaled, _ = standardize_train_test(dataset["train"], dataset["test"])
    Y_train = to_feature_sample_matrix(train_scaled)
    Y_test = to_feature_sample_matrix(test_scaled)
    labels = dataset["labels"].astype(bool)

    model = JSSDL(
        n_atoms_D1=20,
        n_atoms_D2=20,
        sparsity_D1=2,
        sparsity_X1=2,
        sparsity_X2=2,
        tau=0.1,
        max_iter=20,
        random_state=7,
    )
    model.fit(Y_train, alpha=0.95)
    scores = np.asarray(model.predict(Y_test), dtype=float)
    preds = np.asarray(model.is_fault(Y_test), dtype=bool)

    assert model.threshold_ is not None
    assert model.objective_history_
    assert model.convergence_history_
    assert np.all(np.isfinite(model.objective_history_))
    assert np.all(np.isfinite(model.convergence_history_))
    assert model.p_singular_values_history_
    assert model.rank_history_
    assert scores.shape[0] == labels.shape[0]
    assert scores[labels].mean() > scores[~labels].mean()
    assert model.objective_history_[-1] <= model.objective_history_[0]
    assert 1 <= model.n_iter_ <= 20
    assert all(0 <= rank <= model.n_atoms_D2 for rank in model.rank_history_)
    assert all(values.ndim == 1 and np.all(np.isfinite(values)) for values in model.p_singular_values_history_)
    d1_norms = np.linalg.norm(model.D1_, axis=0)
    np.testing.assert_allclose(d1_norms[d1_norms > 1.0e-12], np.ones_like(d1_norms[d1_norms > 1.0e-12]), atol=1.0e-8)
    assert 0.0 <= compute_fdr(labels, preds) <= 1.0
    assert 0.0 <= compute_far(labels, preds) <= 1.0
    assert len(model.objective_history_) == len(model.convergence_history_) == len(model.rank_history_) == len(model.p_singular_values_history_)


def test_generate_numerical_dataset_keeps_train_components() -> None:
    dataset = generate_numerical_dataset(
        n_features=12,
        shared_rank=2,
        n_train=32,
        n_test_normal=8,
        n_test_fault=8,
        noise_std=0.05,
        random_state=11,
    )

    specific = dataset["train_specific_component"]
    shared = dataset["train_shared_component"]
    train = dataset["train"]

    assert specific.shape == train.shape
    assert shared.shape == train.shape
    assert np.all(np.isfinite(train - specific - shared))
    assert np.all(np.count_nonzero(np.abs(specific) > 1.0e-12, axis=1) <= 2)


def test_jssdl_fit_uses_paper_update_order(monkeypatch) -> None:
    model = JSSDL(
        n_atoms_D1=4,
        n_atoms_D2=4,
        sparsity_D1=2,
        sparsity_X1=2,
        sparsity_X2=2,
        max_iter=1,
        random_state=0,
    )
    Y = np.eye(4, dtype=float)
    call_order: list[str] = []

    monkeypatch.setattr(model, "_update_D1", lambda _: call_order.append("D1"))
    monkeypatch.setattr(model, "_update_P", lambda: call_order.append("P"))
    monkeypatch.setattr(model, "_update_D2", lambda _: call_order.append("D2"))
    monkeypatch.setattr(model, "_update_X1_X2", lambda _: call_order.append("X"))
    monkeypatch.setattr(model, "_compute_objective", lambda *args, **kwargs: 0.0)

    model.fit(Y, alpha=None)
    assert call_order[:4] == ["D1", "P", "D2", "X"]


def test_jssdl_dictionary_updates_normalize_without_rescaling_codes() -> None:
    model = JSSDL(
        n_atoms_D1=2,
        n_atoms_D2=2,
        sparsity_D1=3,
        sparsity_X1=2,
        sparsity_X2=2,
        lambda5=0.5,
        max_iter=1,
        random_state=0,
    )
    model.D1_ = np.array([[2.0, 0.2], [0.5, 1.5], [1.0, -0.5]], dtype=float)
    model.D2_ = np.array([[1.0, -0.5], [0.25, 2.0], [1.5, 0.75]], dtype=float)
    model.X1_ = np.array([[1.0, 0.0, 0.5, -0.25], [0.25, -0.5, 0.0, 1.0]], dtype=float)
    model.X2_ = np.array([[0.5, 1.0, -0.5, 0.25], [1.0, -0.25, 0.75, 0.0]], dtype=float)
    model.P_ = np.array([[0.8, -0.2], [0.1, 0.7], [0.3, 0.4]], dtype=float)
    Y = model.D1_ @ model.X1_ + model.D2_ @ model.X2_
    x1_before = model.X1_.copy()
    x2_before = model.X2_.copy()

    model._update_D1(Y)
    np.testing.assert_allclose(model.X1_, x1_before, atol=1.0e-12)
    d1_norms = np.linalg.norm(model.D1_, axis=0)
    np.testing.assert_allclose(d1_norms[d1_norms > 1.0e-12], np.ones_like(d1_norms[d1_norms > 1.0e-12]), atol=1.0e-8)

    model._update_D2(Y)
    np.testing.assert_allclose(model.X2_, x2_before, atol=1.0e-12)
    d2_norms = np.linalg.norm(model.D2_, axis=0)
    np.testing.assert_allclose(d2_norms[d2_norms > 1.0e-12], np.ones_like(d2_norms[d2_norms > 1.0e-12]), atol=1.0e-8)


def test_jssdl_restores_lowest_objective_state_before_return(monkeypatch) -> None:
    model = JSSDL(
        n_atoms_D1=1,
        n_atoms_D2=1,
        sparsity_D1=1,
        sparsity_X1=1,
        sparsity_X2=1,
        max_iter=11,
        random_state=0,
    )
    Y = np.zeros((1, 1), dtype=float)

    reconstruction_values = [100.0 - idx for idx in range(11)]
    objective_values = [10.0, 9.0, 8.0, 7.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    state: dict[str, int | None] = {"next_idx": 0, "current_idx": None}

    def _active_idx() -> int:
        next_idx = state["next_idx"]
        assert isinstance(next_idx, int)
        return next_idx

    def _set_iteration_state() -> None:
        idx = _active_idx()
        value = float(np.sqrt(reconstruction_values[idx]))
        assert model.D1_ is not None and model.D2_ is not None and model.P_ is not None and model.X1_ is not None and model.X2_ is not None
        model.D1_.fill(value)
        model.D2_.fill(0.0)
        model.P_.fill(0.0)
        model.X1_.fill(1.0)
        model.X2_.fill(0.0)

    monkeypatch.setattr(model, "_update_D1", lambda _: _set_iteration_state())
    monkeypatch.setattr(model, "_update_P", lambda: _set_iteration_state())
    monkeypatch.setattr(model, "_update_D2", lambda _: _set_iteration_state())

    def _update_codes(_: np.ndarray) -> None:
        _set_iteration_state()
        state["current_idx"] = _active_idx()
        state["next_idx"] = _active_idx() + 1

    monkeypatch.setattr(model, "_update_X1_X2", _update_codes)

    def _objective(*args, **kwargs) -> float:
        current_idx = state["current_idx"]
        assert isinstance(current_idx, int)
        return objective_values[current_idx]

    monkeypatch.setattr(model, "_compute_objective", _objective)

    model.fit(Y, alpha=None)

    assert model.n_iter_ == 11
    assert model.best_iteration_ == 5
    assert len(model.reconstruction_history_) == 11
    assert model.objective_history_ == objective_values
    assert model.D1_ is not None
    np.testing.assert_allclose(model.D1_, np.array([[np.sqrt(reconstruction_values[4])]]), atol=1.0e-8)


def test_jssdl_stops_after_three_stable_relative_changes(monkeypatch) -> None:
    model = JSSDL(
        n_atoms_D1=1,
        n_atoms_D2=1,
        sparsity_D1=1,
        sparsity_X1=1,
        sparsity_X2=1,
        max_iter=20,
        tol=1.0e-5,
        random_state=0,
    )
    Y = np.zeros((1, 1), dtype=float)

    reconstruction_values = [100.0, 90.0, 80.0, 80.0000001, 80.0000002, 80.0000003]
    objective_values = [50.0, 45.0, 40.0, 40.0000001, 40.0000002, 40.0000003]
    state: dict[str, int | None] = {"next_idx": 0, "current_idx": None}

    def _active_idx() -> int:
        next_idx = state["next_idx"]
        assert isinstance(next_idx, int)
        return next_idx

    def _set_iteration_state() -> None:
        idx = _active_idx()
        value = float(np.sqrt(reconstruction_values[idx]))
        assert model.D1_ is not None and model.D2_ is not None and model.P_ is not None and model.X1_ is not None and model.X2_ is not None
        model.D1_.fill(value)
        model.D2_.fill(0.0)
        model.P_.fill(0.0)
        model.X1_.fill(1.0)
        model.X2_.fill(0.0)

    monkeypatch.setattr(model, "_update_D1", lambda _: _set_iteration_state())
    monkeypatch.setattr(model, "_update_P", lambda: _set_iteration_state())
    monkeypatch.setattr(model, "_update_D2", lambda _: _set_iteration_state())

    def _update_codes(_: np.ndarray) -> None:
        _set_iteration_state()
        state["current_idx"] = _active_idx()
        state["next_idx"] = _active_idx() + 1

    monkeypatch.setattr(model, "_update_X1_X2", _update_codes)

    def _objective(*args, **kwargs) -> float:
        current_idx = state["current_idx"]
        assert isinstance(current_idx, int)
        return objective_values[current_idx]

    monkeypatch.setattr(model, "_compute_objective", _objective)

    model.fit(Y, alpha=None)

    assert model.n_iter_ == 6
    assert len(model.reconstruction_history_) == 6
