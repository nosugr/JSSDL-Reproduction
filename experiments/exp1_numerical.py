from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from baselines import DictionaryLearningMonitor, PCAMonitor, RobustPCAMonitor
from jssdl import JSSDL
from jssdl.monitoring.metrics import compute_far, compute_fdr
from jssdl.utils.data_loader import generate_numerical_dataset, load_config, standardize_train_test, to_feature_sample_matrix
from jssdl.utils.initializer import normalize_columns
from jssdl.utils.visualizer import (
    plot_dictionary_heatmap,
    plot_fault_annotated_heatmap,
    plot_heatmap,
    plot_monitoring_scores,
)
import matplotlib.pyplot as plt


def build_jssdl_model(defaults: dict, config: dict, random_state: int) -> JSSDL:
    return JSSDL(
        n_atoms_D1=config["n_atoms_D1"],
        n_atoms_D2=config["n_atoms_D2"],
        sparsity_D1=config["sparsity_D1"],
        sparsity_X1=config["sparsity_X1"],
        sparsity_X2=config["sparsity_X2"],
        lambda1=defaults["lambda1"],
        lambda2=defaults["lambda2"],
        lambda3=defaults["lambda3"],
        lambda4=defaults["lambda4"],
        lambda5=defaults["lambda5"],
        lambda6=defaults.get("lambda6", 0.0),
        tau=config["tau"],
        max_iter=config["max_iter"],
        tol=defaults["tol"],
        random_state=random_state,
    )


def _build_feature_ticks(n_features: int) -> tuple[list[int], list[int]]:
    if n_features <= 20:
        positions = list(range(n_features))
    else:
        step = max(1, int(np.ceil(n_features / 10.0)))
        positions = list(range(0, n_features, step))
        if positions[-1] != n_features - 1:
            positions.append(n_features - 1)
    labels = [position + 1 for position in positions]
    return positions, labels


def save_numerical_train_visualizations(dataset: dict[str, np.ndarray]) -> None:
    processed_dir = ROOT / "data" / "processed" / "numerical"
    processed_dir.mkdir(parents=True, exist_ok=True)

    matrices = {
        "train_y": np.asarray(dataset["train"], dtype=float),
        "train_specific": np.asarray(dataset["train_specific_component"], dtype=float),
        "train_shared": np.asarray(dataset["train_shared_component"], dtype=float),
    }
    titles = {
        "train_y": "Raw Training Data Y",
        "train_specific": "Specific Component (specific_weight * y1)",
        "train_shared": "Shared Component (shared_weight * y2)",
    }
    n_features = int(np.asarray(dataset["train"], dtype=float).shape[1])
    feature_tick_positions, feature_tick_labels = _build_feature_ticks(n_features)

    for stem, matrix in matrices.items():
        pd.DataFrame(matrix).to_csv(processed_dir / f"{stem}.csv", index=False)
        plot_heatmap(
            matrix.T,
            titles[stem],
            processed_dir / f"{stem}_heatmap.png",
            cmap="coolwarm",
            xlabel="Sample",
            ylabel="Feature",
            figsize=(12.0, 4.5),
            y_ticks=feature_tick_positions,
            y_tick_labels=feature_tick_labels,
        )


def save_numerical_test_visualization(dataset: dict[str, np.ndarray]) -> None:
    raw_dir = ROOT / "data" / "raw" / "numerical"
    raw_dir.mkdir(parents=True, exist_ok=True)
    test_matrix = np.asarray(dataset["test"], dtype=float)
    labels = np.asarray(dataset["labels"], dtype=int)
    fault_feature = int(dataset["fault_feature"])

    n_features = int(test_matrix.shape[1])
    feature_tick_positions, feature_tick_labels = _build_feature_ticks(n_features)

    plot_fault_annotated_heatmap(
        test_matrix.T,
        labels,
        fault_feature=fault_feature,
        title="Raw Test Data Y (Fault Highlighted)",
        output_path=raw_dir / "test_heatmap.png",
        y_ticks=feature_tick_positions,
        y_tick_labels=feature_tick_labels,
    )


def _sample_iteration_indices(n_iterations: int) -> list[int]:
    if n_iterations <= 0:
        return []

    step = max(1, int(np.ceil(n_iterations / 10.0)))
    sampled = list(range(step - 1, n_iterations, step))
    if sampled[-1] != n_iterations - 1:
        sampled.append(n_iterations - 1)
    return sampled


def _style_boxplot_panel(ax: plt.Axes, values: list[np.ndarray], labels: list[str], title: str, ylabel: str) -> None:
    boxplot = ax.boxplot(values, patch_artist=True, showfliers=False)
    for patch in boxplot["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.35)
    for median in boxplot["medians"]:
        median.set_color("tab:red")
        median.set_linewidth(1.5)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def save_singular_value_boxplots(model: JSSDL, output_path: Path) -> None:
    p_history = [np.asarray(values, dtype=float).ravel() for values in model.p_singular_values_history_]
    d2_history = [np.asarray(values, dtype=float).ravel() for values in model.d2_singular_values_history_]
    n_iterations = min(len(p_history), len(d2_history))
    if n_iterations <= 0:
        return

    sampled_indices = _sample_iteration_indices(n_iterations)
    labels = [str(index + 1) for index in sampled_indices]
    sampled_p_history = [p_history[index] for index in sampled_indices]
    sampled_d2_history = [d2_history[index] for index in sampled_indices]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.0), sharey=True)
    _style_boxplot_panel(axes[0], sampled_p_history, labels, "Singular Values of P Across Iterations", "Singular value")
    _style_boxplot_panel(axes[1], sampled_d2_history, labels, "Singular Values of D2 Across Iterations", "Singular value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def attach_objective_component_recorder(model: JSSDL) -> None:
    component_history: dict[str, list[float]] = {
        "objective": [],
        "reconstruction": [],
        "specific_code_l1": [],
        "specific_dictionary_l1": [],
        "shared_code_l1": [],
        "nuclear_norm": [],
        "d2_gap": [],
        "incoherence": [],
    }

    def tracked_compute_objective(
        self: JSSDL,
        Y: np.ndarray,
        D1: np.ndarray,
        X1: np.ndarray,
        D2: np.ndarray,
        X2: np.ndarray,
        P: np.ndarray,
    ) -> float:
        residual = Y - D1 @ X1 - D2 @ X2
        reconstruction = float(np.linalg.norm(residual, ord="fro") ** 2)
        specific_code_l1 = float(self.lambda1 * np.sum(np.abs(X1)))
        specific_dictionary_l1 = float(self.lambda2 * np.sum(np.abs(D1)))
        shared_code_l1 = float(self.lambda3 * np.sum(np.abs(X2)))
        nuclear_norm = float(self.lambda4 * np.linalg.svd(P, compute_uv=False).sum())
        d2_gap = float(self.lambda5 * np.linalg.norm(D2 - P, ord="fro") ** 2)
        incoherence = float(self.lambda6 * np.linalg.norm(D1.T @ D2, ord="fro") ** 2)
        objective = float(
            reconstruction
            + specific_code_l1
            + specific_dictionary_l1
            + shared_code_l1
            + nuclear_norm
            + d2_gap
            + incoherence
        )

        component_history["objective"].append(objective)
        component_history["reconstruction"].append(reconstruction)
        component_history["specific_code_l1"].append(specific_code_l1)
        component_history["specific_dictionary_l1"].append(specific_dictionary_l1)
        component_history["shared_code_l1"].append(shared_code_l1)
        component_history["nuclear_norm"].append(nuclear_norm)
        component_history["d2_gap"].append(d2_gap)
        component_history["incoherence"].append(incoherence)
        return objective

    model._compute_objective = MethodType(tracked_compute_objective, model)
    setattr(model, "objective_component_history_", component_history)


def save_objective_curve_plot(model: JSSDL, output_paths: list[Path], best_iteration: int | None = None) -> None:
    convergence_values = np.asarray(
        model.convergence_history_ if model.convergence_history_ else model.objective_history_,
        dtype=float,
    ).ravel()
    objective_values = np.asarray(model.objective_history_, dtype=float).ravel()
    iterations = np.arange(1, max(convergence_values.shape[0], objective_values.shape[0]) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(
        iterations[: convergence_values.shape[0]],
        convergence_values,
        linewidth=2,
        color="tab:blue",
        label="Relative objective change",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Relative change", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)
    if iterations.size == 1:
        ax1.set_xlim(0.5, 1.5)

    ax2 = ax1.twinx()
    ax2.plot(
        iterations[: objective_values.shape[0]],
        objective_values,
        linewidth=2,
        color="tab:orange",
        label="Objective",
    )
    ax2.set_ylabel("Objective", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.set_title("JSSDL Objective and Relative Objective Change")

    if best_iteration is not None and best_iteration > 0:
        ax1.axvline(best_iteration, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.8)
        ax1.annotate(
            f"best iter={best_iteration}",
            xy=(best_iteration, 0.95),
            xycoords=("data", "axes fraction"),
            ha="right" if best_iteration > iterations[-1] * 0.5 else "left",
            va="top",
            fontsize=9,
            color="tab:red",
            fontweight="bold",
        )

    lines = [line for line in ax1.get_lines() + ax2.get_lines() if not line.get_label().startswith("_")]
    legend_labels = [line.get_label() for line in lines]
    ax1.legend(lines, legend_labels, loc="upper right")
    fig.tight_layout()
    for output_path in output_paths:
        fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_objective_component_plot(model: JSSDL, output_path: Path, best_iteration: int | None = None) -> None:
    history = getattr(model, "objective_component_history_", None)
    if not isinstance(history, dict) or not history.get("objective"):
        return

    iterations = np.arange(1, len(history["objective"]) + 1)
    objective = np.asarray(history["objective"], dtype=float)
    reconstruction = np.asarray(history["reconstruction"], dtype=float)
    specific_code_l1 = np.asarray(history["specific_code_l1"], dtype=float)
    specific_dictionary_l1 = np.asarray(history["specific_dictionary_l1"], dtype=float)
    shared_code_l1 = np.asarray(history["shared_code_l1"], dtype=float)
    nuclear_norm = np.asarray(history["nuclear_norm"], dtype=float)
    d2_gap = np.asarray(history["d2_gap"], dtype=float)
    incoherence = np.asarray(history.get("incoherence", np.zeros_like(objective)), dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    stack_values = [
        reconstruction,
        specific_code_l1,
        specific_dictionary_l1,
        shared_code_l1,
        nuclear_norm,
        d2_gap,
        incoherence,
    ]
    stack_labels = [
        "Reconstruction",
        r"$\lambda_1 ||X_1||_1$",
        r"$\lambda_2 ||D_1||_1$",
        r"$\lambda_3 ||X_2||_1$",
        r"$\lambda_4 ||P||_*$",
        r"$\lambda_5 ||D_2 - P||_F^2$",
        r"$\lambda_6 ||D_1^T D_2||_F^2$",
    ]
    stack_colors = [
        "tab:green",
        "tab:blue",
        "tab:brown",
        "tab:red",
        "tab:purple",
        "tab:gray",
        "tab:pink",
    ]
    ax1.stackplot(iterations, stack_values, labels=stack_labels, colors=stack_colors, alpha=0.8)
    ax1.plot(iterations, objective, color="black", linewidth=2.2, label="Objective")
    ax1.set_ylabel("Objective value")
    ax1.set_title("JSSDL Objective Decomposition by Iteration")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", ncol=2, fontsize=9)

    ax2.plot(iterations, reconstruction, color="tab:green", linewidth=2, label="Reconstruction")
    ax2.plot(iterations, specific_code_l1, color="tab:blue", linewidth=2, label=r"$\lambda_1 ||X_1||_1$")
    ax2.plot(
        iterations,
        specific_dictionary_l1,
        color="tab:brown",
        linewidth=2,
        label=r"$\lambda_2 ||D_1||_1$",
    )
    ax2.plot(iterations, nuclear_norm, color="tab:purple", linewidth=2, label=r"$\lambda_4 ||P||_*$")
    ax2.plot(iterations, d2_gap, color="tab:gray", linewidth=2, label=r"$\lambda_5 ||D_2 - P||_F^2$")
    ax2.plot(iterations, incoherence, color="tab:pink", linewidth=2, label=r"$\lambda_6 ||D_1^T D_2||_F^2$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Zoomed value")
    ax2.set_title(r"Zoomed View Without Dominant Term $\lambda_3 ||X_2||_1$")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", ncol=2, fontsize=9)

    # Draw vertical dashed line at best iteration
    if best_iteration is not None and best_iteration > 0:
        for ax in (ax1, ax2):
            ax.axvline(best_iteration, color="tab:red", linestyle="--", linewidth=1.5, alpha=0.8)
        ax1.annotate(
            f"best iter={best_iteration}",
            xy=(best_iteration, 0.95),
            xycoords=("data", "axes fraction"),
            ha="right" if best_iteration > iterations[-1] * 0.5 else "left",
            va="top",
            fontsize=9,
            color="tab:red",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_run_random_states(base_random_state: int, n_runs: int) -> list[int]:
    if n_runs <= 0:
        return []
    if n_runs == 1:
        return [base_random_state]

    candidates = [seed for seed in range(100) if seed != base_random_state]
    rng = np.random.default_rng()
    if n_runs - 1 <= len(candidates):
        extra_states = rng.choice(candidates, size=n_runs - 1, replace=False).astype(int).tolist()
    else:
        extra_states = rng.choice(100, size=n_runs - 1, replace=True).astype(int).tolist()
    return [base_random_state, *extra_states]


def _build_runs_output_table(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in results_df.columns if column not in {"run", "random_state"}]
    average_row: dict[str, float | int | str] = {"run": "Average", "random_state": ""}
    for column in metric_columns:
        average_row[column] = f"{float(results_df[column].mean()):.3f}"

    output_df = pd.concat([results_df, pd.DataFrame([average_row])], ignore_index=True)
    column_tuples: list[tuple[str, str]] = []
    for column in output_df.columns:
        if column in {"run", "random_state"}:
            column_tuples.append((column, ""))
            continue
        method, metric = column.rsplit("_", 1)
        column_tuples.append((method, metric))
    output_df.columns = pd.MultiIndex.from_tuples(column_tuples)
    return output_df


def run_experiment(n_runs: int | None = None, show_progress: bool = True) -> pd.DataFrame:
    config = load_config(ROOT / "config.yaml")
    defaults = config["defaults"]
    numerical_cfg = config["numerical_simulation"]
    n_runs = defaults["n_runs"] if n_runs is None else n_runs

    raw_dir = ROOT / "data" / "raw" / "numerical"
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, float | int | str]] = []
    first_run_payload: dict[str, object] = {}
    best_jssdl_payload: dict[str, object] = {}
    best_jssdl_fdr = float("-inf")
    best_jssdl_far = float("inf")
    base_random_state = int(defaults["random_state"])
    run_random_states = _build_run_random_states(base_random_state, n_runs)

    for run_idx, random_state in enumerate(run_random_states):
        dataset = generate_numerical_dataset(
            n_features=numerical_cfg["n_features"],
            shared_rank=numerical_cfg["shared_rank"],
            n_train=numerical_cfg["n_train"],
            n_test_normal=numerical_cfg["n_test_normal"],
            n_test_fault=numerical_cfg["n_test_fault"],
            fault_feature=numerical_cfg["fault_feature"],
            fault_bias=numerical_cfg["fault_bias"],
            noise_std=numerical_cfg["noise_std"],
            random_state=random_state,
        )
        if random_state == base_random_state:
            pd.DataFrame(dataset["train"]).to_csv(raw_dir / "train.csv", index=False)
            pd.DataFrame(dataset["test"]).to_csv(raw_dir / "test.csv", index=False)
            pd.DataFrame({"label": dataset["labels"]}).to_csv(raw_dir / "labels.csv", index=False)
            save_numerical_train_visualizations(dataset)
            save_numerical_test_visualization(dataset)

        train_scaled, test_scaled, _ = standardize_train_test(dataset["train"], dataset["test"])
        Y_train = to_feature_sample_matrix(train_scaled)
        Y_test = to_feature_sample_matrix(test_scaled)
        labels = dataset["labels"].astype(bool)
        fault_feature = int(dataset["fault_feature"]) + 1
        run_row: dict[str, float | int | str] = {
            "run": run_idx + 1,
            "random_state": random_state,
        }

        if show_progress and run_idx > 0:
            print()
        print(f"run {run_idx + 1}/{n_runs}: fault dimension = {fault_feature}")

        initial_d2: np.ndarray | None = None
        if run_idx == 0:
            initial_jssdl = build_jssdl_model(defaults, numerical_cfg, random_state=random_state)
            initial_jssdl.max_iter = 0
            initial_jssdl.fit(Y_train, alpha=None)
            if initial_jssdl.D2_ is not None:
                initial_d2 = initial_jssdl.D2_.copy()

        jssdl = build_jssdl_model(defaults, numerical_cfg, random_state=random_state)
        if run_idx == 0:
            attach_objective_component_recorder(jssdl)
        jssdl.fit(
            Y_train,
            alpha=numerical_cfg["kde_confidence"],
            show_progress=show_progress,
            progress_desc=f"epoch[JSSDL][run {run_idx + 1}/{n_runs}]",
            progress_position=0,
            progress_leave=True,
        )
        jssdl_scores = np.asarray(jssdl.predict(Y_test), dtype=float)
        jssdl_preds = np.asarray(jssdl.is_fault(Y_test), dtype=bool)
        jssdl_fdr = compute_fdr(labels, jssdl_preds)
        jssdl_far = compute_far(labels, jssdl_preds)
        run_row["JSSDL_FAR"] = jssdl_far
        run_row["JSSDL_FDR"] = jssdl_fdr
        if jssdl_fdr > best_jssdl_fdr or (np.isclose(jssdl_fdr, best_jssdl_fdr) and jssdl_far < best_jssdl_far):
            best_jssdl_payload = {
                "run": run_idx + 1,
                "random_state": random_state,
                "jssdl": jssdl,
                "jssdl_scores": jssdl_scores,
            }
            best_jssdl_fdr = float(jssdl_fdr)
            best_jssdl_far = float(jssdl_far)

        pca = PCAMonitor(cpv=0.90, alpha=numerical_cfg["kde_confidence"])
        pca.fit(
            train_scaled,
            show_progress=show_progress,
            progress_desc=f"epoch[PCA][run {run_idx + 1}/{n_runs}]",
            progress_position=0,
            progress_leave=True,
        )
        pca_scores = pca.score_samples(test_scaled)
        pca_preds = pca.predict(test_scaled)
        run_row["PCA_FAR"] = compute_far(labels, pca_preds)
        run_row["PCA_FDR"] = compute_fdr(labels, pca_preds)

        rpca = RobustPCAMonitor(cpv=0.90, alpha=numerical_cfg["kde_confidence"])
        rpca.fit(
            train_scaled,
            show_progress=show_progress,
            progress_desc=f"epoch[RPCA][run {run_idx + 1}/{n_runs}]",
            progress_position=0,
            progress_leave=True,
        )
        rpca_scores = rpca.score_samples(test_scaled)
        rpca_preds = rpca.predict(test_scaled)
        run_row["Robust PCA_FAR"] = compute_far(labels, rpca_preds)
        run_row["Robust PCA_FDR"] = compute_fdr(labels, rpca_preds)

        dl = DictionaryLearningMonitor(
            alpha=numerical_cfg["kde_confidence"],
            random_state=random_state,
        )
        dl.fit(
            train_scaled,
            show_progress=show_progress,
            progress_desc=f"epoch[DL][run {run_idx + 1}/{n_runs}]",
            progress_position=0,
            progress_leave=True,
        )
        dl_scores = dl.score_samples(test_scaled)
        dl_preds = dl.predict(test_scaled)
        run_row["DL_FAR"] = compute_far(labels, dl_preds)
        run_row["DL_FDR"] = compute_fdr(labels, dl_preds)

        run_rows.append(run_row)

        if random_state == base_random_state:
            first_run_payload = {
                "jssdl": jssdl,
                "jssdl_scores": jssdl_scores,
                "labels": labels.astype(int),
                "pca": pca,
                "pca_scores": pca_scores,
                "rpca": rpca,
                "rpca_scores": rpca_scores,
                "dl": dl,
                "dl_scores": dl_scores,
                "initial_d2": initial_d2,
            }

    results_df = pd.DataFrame(run_rows)
    runs_output_df = _build_runs_output_table(results_df)

    summary_rows = []
    for method in ["JSSDL", "PCA", "Robust PCA", "DL"]:
        summary_rows.append(
            {
                "method": method,
                "FAR": round(float(results_df[f"{method}_FAR"].mean()), 3),
                "FDR": round(float(results_df[f"{method}_FDR"].mean()), 3),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    output_tables = ROOT / "outputs" / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    runs_output_df.to_csv(output_tables / "exp1_numerical_runs.csv", index=False)
    summary_df.to_csv(output_tables / "exp1_numerical_summary.csv", index=False)

    if first_run_payload and best_jssdl_payload:
        best_jssdl = best_jssdl_payload["jssdl"]
        first_run_jssdl = first_run_payload["jssdl"]
        first_run_dl = first_run_payload["dl"]
        assert isinstance(best_jssdl, JSSDL)
        assert isinstance(first_run_jssdl, JSSDL)
        assert isinstance(first_run_dl, DictionaryLearningMonitor)
        checkpoint_dir = ROOT / "outputs" / "checkpoints"
        figure_dir = ROOT / "outputs" / "figures" / "exp1_numerical"
        diagnostics_dir = ROOT / "outputs" / "figures" / "exp1_training_diagnostics"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            checkpoint_dir / "exp1_jssdl_best.npz",
            D1=best_jssdl.D1_,
            D2=best_jssdl.D2_,
            P=best_jssdl.P_,
            threshold=best_jssdl.threshold_,
        )
        d1_heatmap = best_jssdl.D1_
        d2_heatmap = best_jssdl.D2_
        plot_dictionary_heatmap(d1_heatmap, "Specific Dictionary D1", figure_dir / "exp1_D1_heatmap.png")
        plot_dictionary_heatmap(d2_heatmap, "Shared Dictionary D2", figure_dir / "exp1_D2_heatmap.png")
        if best_jssdl.P_ is not None:
            plot_dictionary_heatmap(
                best_jssdl.P_,
                "Low-Rank Auxiliary Matrix P",
                diagnostics_dir / "exp1_P_heatmap.png",
            )
        initial_d2 = first_run_payload.get("initial_d2")
        if initial_d2 is not None:
            plot_dictionary_heatmap(
                np.asarray(initial_d2, dtype=float),
                "Initial Shared Dictionary D2",
                diagnostics_dir / "exp1_D2_init_heatmap.png",
            )
        # X1 / X2 heatmaps (same "best" model as D1/D2)
        plot_heatmap(best_jssdl.X1_, "Specific Codes X1", diagnostics_dir / "exp1_X1_heatmap.png",
                     cmap="coolwarm", xlabel="Sample", ylabel="Atom")
        plot_heatmap(best_jssdl.X2_, "Shared Codes X2", diagnostics_dir / "exp1_X2_heatmap.png",
                     cmap="coolwarm", xlabel="Sample", ylabel="Atom")
        if first_run_dl.dictionary_ is not None:
            plot_dictionary_heatmap(
                normalize_columns(first_run_dl.dictionary_),
                "DL Dictionary D",
                diagnostics_dir / "exp1_D_heatmap.png",
            )
        if first_run_dl.codes_ is not None:
            plot_heatmap(
                first_run_dl.codes_,
                "DL Codes X",
                diagnostics_dir / "exp1_X_heatmap.png",
                cmap="coolwarm",
                xlabel="Sample",
                ylabel="Atom",
            )
        best_iter_for_plot = first_run_jssdl.best_iteration_
        save_objective_curve_plot(
            first_run_jssdl,
            [
                diagnostics_dir / "exp1_objective_curve.png",
            ],
            best_iteration=best_iter_for_plot,
        )
        save_singular_value_boxplots(first_run_jssdl, diagnostics_dir / "exp1_singular_values_boxplot.png")
        save_objective_component_plot(first_run_jssdl, diagnostics_dir / "exp1_objective_components.png", best_iteration=best_iter_for_plot)
        plot_monitoring_scores(
            np.asarray(first_run_payload["jssdl_scores"], dtype=float),
            float(first_run_jssdl.threshold_),
            np.asarray(first_run_payload["labels"], dtype=int),
            "JSSDL Joint Reconstruction Error",
            "JRE",
            figure_dir / "exp1_jssdl_monitoring.png",
            split_index=numerical_cfg["n_test_normal"],
            annotate_metrics=True,
        )

        pca_scores = first_run_payload["pca_scores"]
        pca = first_run_payload["pca"]
        assert isinstance(pca_scores, dict)
        assert isinstance(pca, PCAMonitor)
        plot_monitoring_scores(
            np.asarray(pca_scores["t2"], dtype=float),
            float(pca.t2_threshold_),
            np.asarray(first_run_payload["labels"], dtype=int),
            "PCA T2 Statistic",
            "T2",
            figure_dir / "exp1_pca_t2.png",
            split_index=numerical_cfg["n_test_normal"],
            annotate_metrics=True,
        )
        plot_monitoring_scores(
            np.asarray(pca_scores["spe"], dtype=float),
            float(pca.spe_threshold_),
            np.asarray(first_run_payload["labels"], dtype=int),
            "PCA SPE Statistic",
            "SPE",
            figure_dir / "exp1_pca_spe.png",
            split_index=numerical_cfg["n_test_normal"],
            annotate_metrics=True,
        )

        rpca_scores = first_run_payload["rpca_scores"]
        rpca = first_run_payload["rpca"]
        assert isinstance(rpca_scores, dict)
        assert isinstance(rpca, RobustPCAMonitor)
        plot_monitoring_scores(
            np.asarray(rpca_scores["t2"], dtype=float),
            float(rpca.t2_threshold_),
            np.asarray(first_run_payload["labels"], dtype=int),
            "Robust PCA rT2 Statistic",
            "rT2",
            figure_dir / "exp1_rpca_t2.png",
            split_index=numerical_cfg["n_test_normal"],
            annotate_metrics=True,
        )
        plot_monitoring_scores(
            np.asarray(rpca_scores["spe"], dtype=float),
            float(rpca.spe_threshold_),
            np.asarray(first_run_payload["labels"], dtype=int),
            "Robust PCA rSPE Statistic",
            "rSPE",
            figure_dir / "exp1_rpca_spe.png",
            split_index=numerical_cfg["n_test_normal"],
            annotate_metrics=True,
        )

        plot_monitoring_scores(
            np.asarray(first_run_payload["dl_scores"], dtype=float),
            float(first_run_payload["dl"].threshold_),
            np.asarray(first_run_payload["labels"], dtype=int),
            "Dictionary Learning DRE",
            "DRE",
            figure_dir / "exp1_dl_monitoring.png",
            split_index=numerical_cfg["n_test_normal"],
            annotate_metrics=True,
        )

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the numerical JSSDL experiment.")
    parser.add_argument("--n-runs", type=int, default=None, help="Number of repeated runs.")
    args = parser.parse_args()
    run_experiment(n_runs=args.n_runs)


if __name__ == "__main__":
    main()
