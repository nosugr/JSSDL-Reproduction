from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import PCAMonitor, RobustPCAMonitor
from jssdl.utils.data_loader import generate_numerical_dataset


def _load_config() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    labels_bool = np.asarray(labels, dtype=bool)
    preds_bool = np.asarray(predictions, dtype=bool)
    normal = ~labels_bool
    fault = labels_bool
    far = float(np.mean(preds_bool[normal])) if np.any(normal) else 0.0
    fdr = float(np.mean(preds_bool[fault])) if np.any(fault) else 0.0
    return far, fdr


def _scale_samples(monitor: PCAMonitor | RobustPCAMonitor, samples: np.ndarray) -> np.ndarray:
    if monitor.mean_ is None or monitor.std_ is None:
        raise RuntimeError("Monitor must be fitted before scaling samples.")
    return (np.asarray(samples, dtype=float) - monitor.mean_) / monitor.std_


def _project_samples(
    monitor: PCAMonitor | RobustPCAMonitor,
    samples: np.ndarray,
) -> dict[str, np.ndarray]:
    if monitor.loadings_ is None or monitor.eigenvalues_ is None:
        raise RuntimeError("Monitor must be fitted before projection.")

    scaled = _scale_samples(monitor, samples)
    loadings = monitor.loadings_
    scores = scaled @ loadings
    reconstruction = scores @ loadings.T
    residual = scaled - reconstruction
    eigenvalues = np.maximum(monitor.eigenvalues_, 1.0e-12)
    return {
        "scores": scores,
        "residual": residual,
        "t2": np.sum((scores**2) / eigenvalues, axis=1),
        "spe": np.sum(residual**2, axis=1),
    }


def _first_two_columns(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if values.shape[1] >= 2:
        return values[:, :2]
    if values.shape[1] == 1:
        return np.column_stack([values[:, 0], np.zeros(values.shape[0])])
    return np.zeros((values.shape[0], 2), dtype=float)


def _fit_residual_projection(train_residual: np.ndarray, test_residual: np.ndarray) -> np.ndarray:
    train = np.asarray(train_residual, dtype=float)
    test = np.asarray(test_residual, dtype=float)
    center = train.mean(axis=0, keepdims=True)
    centered_train = train - center

    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    basis = vt[: min(2, vt.shape[0])].T
    projected = (test - center) @ basis
    return _first_two_columns(projected)


def _component_contribution_scores(
    monitor: PCAMonitor | RobustPCAMonitor,
    component: np.ndarray,
) -> np.ndarray:
    if monitor.std_ is None or monitor.loadings_ is None:
        raise RuntimeError("Monitor must be fitted before projecting component contributions.")

    standardized_contribution = np.asarray(component, dtype=float) / monitor.std_
    return _first_two_columns(standardized_contribution @ monitor.loadings_)


def _scores_from_scaled_component(
    monitor: PCAMonitor | RobustPCAMonitor,
    component: np.ndarray,
) -> np.ndarray:
    if monitor.loadings_ is None:
        raise RuntimeError("Monitor must be fitted before projecting component scores.")
    return _first_two_columns(np.asarray(component, dtype=float) @ monitor.loadings_)


def _sparse_outlier_indices(
    rpca: RobustPCAMonitor,
    sparse_outlier_quantile: float,
    max_count: int,
) -> np.ndarray:
    if rpca.sparse_train_ is None:
        raise RuntimeError("RPCA monitor must be fitted before selecting sparse outliers.")

    sparse_energy = np.linalg.norm(rpca.sparse_train_, axis=1)
    threshold = float(np.quantile(sparse_energy, sparse_outlier_quantile))
    candidates = np.flatnonzero(sparse_energy >= threshold)
    order = candidates[np.argsort(sparse_energy[candidates])[::-1]]
    if max_count > 0:
        order = order[:max_count]
    return order


def _full_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    _, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    return singular_values**2 / max(1, values.shape[0] - 1)


def _cumulative_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    return np.cumsum(values) / max(float(np.sum(values)), 1.0e-12)


def _principal_angle_cosines(
    pca: PCAMonitor,
    rpca: RobustPCAMonitor,
) -> np.ndarray:
    if pca.loadings_ is None or rpca.loadings_ is None:
        raise RuntimeError("Monitors must be fitted before comparing subspaces.")

    n_dimensions = min(pca.loadings_.shape[1], rpca.loadings_.shape[1])
    singular_values = np.linalg.svd(
        pca.loadings_[:, :n_dimensions].T @ rpca.loadings_[:, :n_dimensions],
        compute_uv=False,
    )
    return np.clip(singular_values, 0.0, 1.0)


def _axis_limits(*point_sets: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([np.asarray(points, dtype=float)[:, :2] for points in point_sets])
    x_min, x_max = np.nanpercentile(stacked[:, 0], [1, 99])
    y_min, y_max = np.nanpercentile(stacked[:, 1], [1, 99])

    def expand(low: float, high: float) -> tuple[float, float]:
        span = max(float(high - low), 1.0e-9)
        pad = 0.12 * span
        return float(low - pad), float(high + pad)

    return expand(x_min, x_max), expand(y_min, y_max)


def _plot_projection(
    ax: plt.Axes,
    points: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    labels = np.asarray(labels, dtype=int)
    normal = labels == 0
    fault = labels == 1

    ax.scatter(
        points[normal, 0],
        points[normal, 1],
        s=16,
        color="#2f6f9f",
        alpha=0.55,
        edgecolors="none",
        label="Normal test",
    )
    ax.scatter(
        points[fault, 0],
        points[fault, 1],
        s=18,
        color="#d95f02",
        alpha=0.72,
        edgecolors="none",
        label="Fault test",
    )
    ax.axhline(0.0, color="#9aa0a6", linewidth=0.8, alpha=0.45)
    ax.axvline(0.0, color="#9aa0a6", linewidth=0.8, alpha=0.45)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.8)


def _plot_component_contributions(
    ax: plt.Axes,
    shared_points: np.ndarray,
    specific_points: np.ndarray,
    removed_points: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.scatter(
        shared_points[:, 0],
        shared_points[:, 1],
        s=16,
        color="#1b9e77",
        alpha=0.42,
        edgecolors="none",
        label="Shared component",
    )
    ax.scatter(
        specific_points[:, 0],
        specific_points[:, 1],
        s=16,
        color="#7570b3",
        alpha=0.42,
        edgecolors="none",
        label="Specific component",
    )
    if removed_points.size:
        ax.scatter(
            removed_points[:, 0],
            removed_points[:, 1],
            s=28,
            color="#d62728",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            label="RPCA removed S",
            zorder=3,
        )
    ax.axhline(0.0, color="#9aa0a6", linewidth=0.8, alpha=0.45)
    ax.axvline(0.0, color="#9aa0a6", linewidth=0.8, alpha=0.45)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.8)


def _set_shared_limits(axes: list[plt.Axes], point_sets: list[np.ndarray]) -> None:
    xlim, ylim = _axis_limits(*point_sets)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


def _plot_cleaning_arrows(
    ax: plt.Axes,
    all_original: np.ndarray,
    selected_original: np.ndarray,
    selected_low_rank: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.scatter(
        all_original[:, 0],
        all_original[:, 1],
        s=12,
        color="#7f7f7f",
        alpha=0.18,
        edgecolors="none",
        label="Training X",
    )
    for start, end in zip(selected_original, selected_low_rank):
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#333333",
                "lw": 0.85,
                "alpha": 0.55,
                "shrinkA": 2.0,
                "shrinkB": 2.0,
            },
            zorder=2,
        )
    ax.scatter(
        selected_original[:, 0],
        selected_original[:, 1],
        s=34,
        color="#d62728",
        alpha=0.90,
        edgecolors="white",
        linewidths=0.45,
        label="Original X_i",
        zorder=3,
    )
    ax.scatter(
        selected_low_rank[:, 0],
        selected_low_rank[:, 1],
        s=34,
        color="#1b9e77",
        alpha=0.90,
        edgecolors="white",
        linewidths=0.45,
        label="Cleaned L_i = X_i - S_i",
        zorder=4,
    )
    ax.axhline(0.0, color="#9aa0a6", linewidth=0.8, alpha=0.45)
    ax.axvline(0.0, color="#9aa0a6", linewidth=0.8, alpha=0.45)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.8)


def _save_comparison_figure(
    pca_data: dict[str, np.ndarray | int | float],
    rpca_data: dict[str, np.ndarray | int | float],
    labels: np.ndarray,
    output_path: Path,
) -> None:
    pca_scores = np.asarray(pca_data["score_2d"], dtype=float)
    rpca_scores = np.asarray(rpca_data["score_2d"], dtype=float)
    pca_residual = np.asarray(pca_data["residual_2d"], dtype=float)
    rpca_residual = np.asarray(rpca_data["residual_2d"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_projection(
        axes[0, 0],
        pca_scores,
        labels,
        f"PCA main space (k={int(pca_data['n_components'])})",
        "PC1 score",
        "PC2 score",
    )
    _plot_projection(
        axes[0, 1],
        rpca_scores,
        labels,
        f"RPCA main space (k={int(rpca_data['n_components'])})",
        "RPC1 score",
        "RPC2 score",
    )
    _plot_projection(
        axes[1, 0],
        pca_residual,
        labels,
        f"PCA residual space | FAR={pca_data['far']:.3f}, FDR={pca_data['fdr']:.3f}",
        "Residual axis 1",
        "Residual axis 2",
    )
    _plot_projection(
        axes[1, 1],
        rpca_residual,
        labels,
        f"RPCA residual space | FAR={rpca_data['far']:.3f}, FDR={rpca_data['fdr']:.3f}",
        "Residual axis 1",
        "Residual axis 2",
    )

    _set_shared_limits([axes[0, 0], axes[0, 1]], [pca_scores, rpca_scores])
    _set_shared_limits([axes[1, 0], axes[1, 1]], [pca_residual, rpca_residual])

    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=9)
    fig.suptitle("PCA vs RPCA projection comparison", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_cleaning_arrow_figure(
    pca: PCAMonitor,
    rpca: RobustPCAMonitor,
    train: np.ndarray,
    output_path: Path,
    sparse_outlier_quantile: float,
    max_arrows: int,
) -> None:
    if pca.loadings_ is None or rpca.loadings_ is None or rpca.low_rank_train_ is None:
        raise RuntimeError("PCA and RPCA monitors must be fitted before plotting cleaning arrows.")

    selected = _sparse_outlier_indices(rpca, sparse_outlier_quantile, max_arrows)
    scaled_train = _scale_samples(rpca, train)
    low_rank_train = rpca.low_rank_train_

    pca_original = _scores_from_scaled_component(pca, scaled_train)
    pca_low_rank = _scores_from_scaled_component(pca, low_rank_train)
    rpca_original = _scores_from_scaled_component(rpca, scaled_train)
    rpca_low_rank = _scores_from_scaled_component(rpca, low_rank_train)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    _plot_cleaning_arrows(
        axes[0],
        pca_original,
        pca_original[selected],
        pca_low_rank[selected],
        f"RPCA cleaning in PCA space (top {len(selected)} S-energy samples)",
        "PC1 score",
        "PC2 score",
    )
    _plot_cleaning_arrows(
        axes[1],
        rpca_original,
        rpca_original[selected],
        rpca_low_rank[selected],
        f"RPCA cleaning in RPCA L-space (top {len(selected)} S-energy samples)",
        "RPC1 score",
        "RPC2 score",
    )
    _set_shared_limits(
        [axes[0], axes[1]],
        [pca_original, pca_low_rank[selected], rpca_original, rpca_low_rank[selected]],
    )
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    fig.suptitle("Original X to low-rank L after removing RPCA sparse S", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_training_component_figure(
    pca: PCAMonitor,
    rpca: RobustPCAMonitor,
    train: np.ndarray,
    train_shared: np.ndarray,
    train_specific: np.ndarray,
    output_path: Path,
    sparse_outlier_quantile: float,
) -> None:
    if (
        pca.loadings_ is None
        or rpca.loadings_ is None
        or rpca.low_rank_train_ is None
        or rpca.sparse_train_ is None
    ):
        raise RuntimeError("PCA and RPCA monitors must be fitted before plotting training components.")

    pca_shared = _component_contribution_scores(pca, train_shared)
    pca_specific = _component_contribution_scores(pca, train_specific)
    rpca_shared = _component_contribution_scores(rpca, train_shared)
    rpca_specific = _component_contribution_scores(rpca, train_specific)
    sparse_energy = np.linalg.norm(rpca.sparse_train_, axis=1)
    threshold = float(np.quantile(sparse_energy, sparse_outlier_quantile))
    removed_mask = sparse_energy >= threshold
    pca_removed = _scores_from_scaled_component(pca, rpca.sparse_train_[removed_mask])
    rpca_removed = _scores_from_scaled_component(rpca, rpca.sparse_train_[removed_mask])

    pca_spectrum = _cumulative_ratio(_full_eigenvalues(_scale_samples(pca, train)))
    rpca_spectrum = _cumulative_ratio(_full_eigenvalues(rpca.low_rank_train_))
    angle_cosines = _principal_angle_cosines(pca, rpca)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_component_contributions(
        axes[0, 0],
        pca_shared,
        pca_specific,
        pca_removed,
        f"Shared/specific + removed S in PCA space (k={pca.loadings_.shape[1]})",
        "PC1 contribution",
        "PC2 contribution",
    )
    _plot_component_contributions(
        axes[0, 1],
        rpca_shared,
        rpca_specific,
        rpca_removed,
        f"Shared/specific + removed S in RPCA L-space (k={rpca.loadings_.shape[1]})",
        "RPC1 contribution",
        "RPC2 contribution",
    )
    _set_shared_limits(
        [axes[0, 0], axes[0, 1]],
        [pca_shared, pca_specific, pca_removed, rpca_shared, rpca_specific, rpca_removed],
    )
    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=9)

    component_axis = np.arange(1, len(pca_spectrum) + 1)
    axes[1, 0].plot(component_axis, pca_spectrum, color="#2f6f9f", linewidth=2.0, label="PCA on X")
    axes[1, 0].plot(component_axis, rpca_spectrum, color="#d95f02", linewidth=2.0, label="PCA on RPCA low-rank L")
    axes[1, 0].axvline(pca.loadings_.shape[1], color="#2f6f9f", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[1, 0].axvline(rpca.loadings_.shape[1], color="#d95f02", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[1, 0].set_title("Cumulative variance after removing sparse S", fontsize=11)
    axes[1, 0].set_xlabel("Number of components")
    axes[1, 0].set_ylabel("Cumulative variance ratio")
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].grid(alpha=0.22, linewidth=0.8)
    axes[1, 0].legend(loc="lower right", frameon=True, fontsize=9)

    angle_axis = np.arange(1, len(angle_cosines) + 1)
    axes[1, 1].bar(angle_axis, angle_cosines, color="#4c78a8", alpha=0.82)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("PCA vs RPCA subspace similarity", fontsize=11)
    axes[1, 1].set_xlabel("Principal angle index")
    axes[1, 1].set_ylabel("cos(angle)")
    axes[1, 1].grid(axis="y", alpha=0.22, linewidth=0.8)

    fig.suptitle("Training components and RPCA low-rank space comparison", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    config = _load_config()
    defaults = config.get("defaults", {})
    numerical = config.get("numerical_simulation", {})

    parser = argparse.ArgumentParser(
        description="Visualize PCA and low-rank+sparse RPCA main/residual projections in one figure."
    )
    parser.add_argument("--cpv", type=float, default=0.90, help="Cumulative percent variance for component selection.")
    parser.add_argument("--alpha", type=float, default=float(numerical.get("kde_confidence", 0.92)))
    parser.add_argument("--random-state", type=int, default=int(defaults.get("random_state", 9)))
    parser.add_argument("--n-features", type=int, default=int(numerical.get("n_features", 20)))
    parser.add_argument("--shared-rank", type=int, default=int(numerical.get("shared_rank", 2)))
    parser.add_argument("--n-train", type=int, default=int(numerical.get("n_train", 1000)))
    parser.add_argument("--n-test-normal", type=int, default=int(numerical.get("n_test_normal", 500)))
    parser.add_argument("--n-test-fault", type=int, default=int(numerical.get("n_test_fault", 500)))
    parser.add_argument("--fault-feature", type=int, default=int(numerical.get("fault_feature", 2)))
    parser.add_argument("--fault-bias", type=float, default=float(numerical.get("fault_bias", 10.0)))
    parser.add_argument("--noise-std", type=float, default=float(numerical.get("noise_std", 0.1)))
    parser.add_argument("--rpca-lam", type=float, default=None)
    parser.add_argument("--rpca-mu", type=float, default=None)
    parser.add_argument("--rpca-max-iter", type=int, default=100)
    parser.add_argument("--rpca-tol", type=float, default=1.0e-7)
    parser.add_argument(
        "--sparse-outlier-quantile",
        type=float,
        default=0.95,
        help="Training samples above this sparse-S energy quantile are drawn as red removed-S points.",
    )
    parser.add_argument(
        "--cleaning-arrow-count",
        type=int,
        default=40,
        help="Maximum number of high-S-energy training samples drawn with X_i -> L_i arrows.",
    )
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "pca_rpca_projection_comparison.png",
        help="Output image path. Relative paths are resolved inside this script folder.",
    )
    parser.add_argument(
        "--components-output",
        type=Path,
        default=SCRIPT_DIR / "training_components_space_comparison.png",
        help="Output image path for training shared/specific component comparison.",
    )
    parser.add_argument(
        "--cleaning-output",
        type=Path,
        default=SCRIPT_DIR / "rpca_cleaning_arrows.png",
        help="Output image path for original X_i to cleaned low-rank L_i arrow comparison.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset = generate_numerical_dataset(
        n_features=args.n_features,
        shared_rank=args.shared_rank,
        n_train=args.n_train,
        n_test_normal=args.n_test_normal,
        n_test_fault=args.n_test_fault,
        fault_feature=args.fault_feature,
        fault_bias=args.fault_bias,
        noise_std=args.noise_std,
        random_state=args.random_state,
    )

    pca = PCAMonitor(cpv=args.cpv, alpha=args.alpha)
    rpca = RobustPCAMonitor(
        cpv=args.cpv,
        alpha=args.alpha,
        lam=args.rpca_lam,
        mu=args.rpca_mu,
        max_iter=args.rpca_max_iter,
        tol=args.rpca_tol,
    )
    pca.fit(dataset["train"], show_progress=args.show_progress, progress_desc="PCA")
    rpca.fit(dataset["train"], show_progress=args.show_progress, progress_desc="RPCA")

    pca_train = _project_samples(pca, dataset["train"])
    pca_test = _project_samples(pca, dataset["test"])
    rpca_train = _project_samples(rpca, dataset["train"])
    rpca_test = _project_samples(rpca, dataset["test"])

    pca_predictions = pca.predict(dataset["test"])
    rpca_predictions = rpca.predict(dataset["test"])
    pca_far, pca_fdr = _rates(dataset["labels"], pca_predictions)
    rpca_far, rpca_fdr = _rates(dataset["labels"], rpca_predictions)

    pca_data = {
        "score_2d": _first_two_columns(pca_test["scores"]),
        "residual_2d": _fit_residual_projection(pca_train["residual"], pca_test["residual"]),
        "n_components": int(pca.loadings_.shape[1]) if pca.loadings_ is not None else 0,
        "far": pca_far,
        "fdr": pca_fdr,
    }
    rpca_data = {
        "score_2d": _first_two_columns(rpca_test["scores"]),
        "residual_2d": _fit_residual_projection(rpca_train["residual"], rpca_test["residual"]),
        "n_components": int(rpca.loadings_.shape[1]) if rpca.loadings_ is not None else 0,
        "far": rpca_far,
        "fdr": rpca_fdr,
    }

    output_path = args.output if args.output.is_absolute() else SCRIPT_DIR / args.output
    components_output_path = (
        args.components_output if args.components_output.is_absolute() else SCRIPT_DIR / args.components_output
    )
    cleaning_output_path = args.cleaning_output if args.cleaning_output.is_absolute() else SCRIPT_DIR / args.cleaning_output
    _save_comparison_figure(pca_data, rpca_data, dataset["labels"], output_path)
    _save_training_component_figure(
        pca,
        rpca,
        dataset["train"],
        dataset["train_shared_component"],
        dataset["train_specific_component"],
        components_output_path,
        args.sparse_outlier_quantile,
    )
    _save_cleaning_arrow_figure(
        pca,
        rpca,
        dataset["train"],
        cleaning_output_path,
        args.sparse_outlier_quantile,
        args.cleaning_arrow_count,
    )
    print(f"Saved one comparison figure to {output_path}")
    print(f"Saved one training-component figure to {components_output_path}")
    print(f"Saved one RPCA cleaning-arrow figure to {cleaning_output_path}")
    print(f"PCA:  k={pca_data['n_components']}, FAR={pca_far:.3f}, FDR={pca_fdr:.3f}")
    print(f"RPCA: k={rpca_data['n_components']}, FAR={rpca_far:.3f}, FDR={rpca_fdr:.3f}")


if __name__ == "__main__":
    main()
