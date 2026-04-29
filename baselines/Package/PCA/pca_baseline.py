from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jssdl.utils.data_loader import generate_numerical_dataset


def _load_config() -> dict:
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _choose_components(explained_ratio: np.ndarray, cpv: float) -> int:
    cumulative = np.cumsum(np.asarray(explained_ratio, dtype=float))
    return int(np.searchsorted(cumulative, float(cpv), side="left") + 1)


def _pca_scores(model: PCA, samples: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    centered = np.asarray(samples, dtype=float) - model.mean_
    loadings = model.components_[:n_components]
    scores = centered @ loadings.T
    reconstruction = scores @ loadings + model.mean_
    residual = np.asarray(samples, dtype=float) - reconstruction
    eigenvalues = np.maximum(model.explained_variance_[:n_components], 1.0e-12)
    return {
        "t2": np.sum((scores**2) / eigenvalues, axis=1),
        "spe": np.sum(residual**2, axis=1),
    }


def _rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    labels_bool = np.asarray(labels, dtype=bool)
    preds_bool = np.asarray(predictions, dtype=bool)
    normal = ~labels_bool
    fault = labels_bool
    far = float(np.mean(preds_bool[normal])) if np.any(normal) else 0.0
    fdr = float(np.mean(preds_bool[fault])) if np.any(fault) else 0.0
    return far, fdr


def _plot_monitoring(
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    labels = np.asarray(labels, dtype=int)
    x_axis = np.arange(1, labels.size + 1)
    predictions = np.logical_or(scores["t2"] > thresholds["t2"], scores["spe"] > thresholds["spe"])
    far, fdr = _rates(labels, predictions)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, key, ylabel in zip(axes, ("t2", "spe"), ("T2", "SPE")):
        values = np.asarray(scores[key], dtype=float)
        ax.plot(x_axis, values, linewidth=1.3, label=ylabel)
        ax.axhline(thresholds[key], color="tab:red", linestyle="--", linewidth=1.2, label="Threshold")
        fault_idx = np.flatnonzero(labels == 1)
        if fault_idx.size:
            ax.scatter(x_axis[fault_idx], values[fault_idx], s=10, color="tab:orange", alpha=0.65, label="Fault")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Sample number")
    fig.suptitle(f"{title} | FAR={far:.3f}, FDR={fdr:.3f}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    config = _load_config()
    defaults = config.get("defaults", {})
    numerical = config.get("numerical_simulation", {})
    parser = argparse.ArgumentParser(description="Run the package-based PCA monitoring baseline.")
    parser.add_argument("--cpv", type=float, default=0.90)
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
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "pca_monitoring.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(dataset["train"])
    test_scaled = scaler.transform(dataset["test"])

    model = PCA(svd_solver="full", random_state=args.random_state)
    model.fit(train_scaled)
    n_components = _choose_components(model.explained_variance_ratio_, args.cpv)

    train_scores = _pca_scores(model, train_scaled, n_components)
    test_scores = _pca_scores(model, test_scaled, n_components)
    thresholds = {
        "t2": float(np.quantile(train_scores["t2"], args.alpha)),
        "spe": float(np.quantile(train_scores["spe"], args.alpha)),
    }

    output_path = args.output if args.output.is_absolute() else Path(__file__).resolve().parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_monitoring(test_scores, thresholds, dataset["labels"], output_path, f"PCA monitoring (k={n_components})")
    print(f"Saved PCA monitoring plot to {output_path}")


if __name__ == "__main__":
    main()
