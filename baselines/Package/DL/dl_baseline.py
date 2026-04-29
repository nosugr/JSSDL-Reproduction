from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import DictionaryLearning
from sklearn.exceptions import ConvergenceWarning
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


def _reconstruction_scores(model: DictionaryLearning, samples: np.ndarray) -> np.ndarray:
    codes = model.transform(samples)
    reconstruction = codes @ model.components_
    residual = np.asarray(samples, dtype=float) - reconstruction
    return np.sum(residual**2, axis=1)


def _rates(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    labels_bool = np.asarray(labels, dtype=bool)
    preds_bool = np.asarray(predictions, dtype=bool)
    normal = ~labels_bool
    fault = labels_bool
    far = float(np.mean(preds_bool[normal])) if np.any(normal) else 0.0
    fdr = float(np.mean(preds_bool[fault])) if np.any(fault) else 0.0
    return far, fdr


def _plot_monitoring(scores: np.ndarray, threshold: float, labels: np.ndarray, output_path: Path, title: str) -> None:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores > float(threshold)
    far, fdr = _rates(labels, predictions)
    x_axis = np.arange(1, labels.size + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_axis, scores, linewidth=1.3, label="Reconstruction error")
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label="Threshold")
    fault_idx = np.flatnonzero(labels == 1)
    if fault_idx.size:
        ax.scatter(x_axis[fault_idx], scores[fault_idx], s=10, color="tab:orange", alpha=0.65, label="Fault")
    ax.set_title(f"{title} | FAR={far:.3f}, FDR={fdr:.3f}")
    ax.set_xlabel("Sample number")
    ax.set_ylabel("Lasso reconstruction error")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    config = _load_config()
    defaults = config.get("defaults", {})
    numerical = config.get("numerical_simulation", {})
    parser = argparse.ArgumentParser(description="Run the package-based dictionary learning monitoring baseline.")
    parser.add_argument("--n-atoms", type=int, default=int(numerical.get("n_atoms_D2", 100)))
    parser.add_argument("--sparse-lambda", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=float(numerical.get("kde_confidence", 0.92)))
    parser.add_argument("--max-iter", type=int, default=int(numerical.get("max_iter", 100)))
    parser.add_argument("--transform-max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--show-convergence-warnings", action="store_true")
    parser.add_argument("--random-state", type=int, default=int(defaults.get("random_state", 9)))
    parser.add_argument("--n-features", type=int, default=int(numerical.get("n_features", 20)))
    parser.add_argument("--shared-rank", type=int, default=int(numerical.get("shared_rank", 2)))
    parser.add_argument("--n-train", type=int, default=int(numerical.get("n_train", 1000)))
    parser.add_argument("--n-test-normal", type=int, default=int(numerical.get("n_test_normal", 500)))
    parser.add_argument("--n-test-fault", type=int, default=int(numerical.get("n_test_fault", 500)))
    parser.add_argument("--fault-feature", type=int, default=int(numerical.get("fault_feature", 2)))
    parser.add_argument("--fault-bias", type=float, default=float(numerical.get("fault_bias", 10.0)))
    parser.add_argument("--noise-std", type=float, default=float(numerical.get("noise_std", 0.1)))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "dl_monitoring.png")
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

    model = DictionaryLearning(
        n_components=args.n_atoms,
        alpha=args.sparse_lambda,
        max_iter=args.max_iter,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        transform_alpha=args.sparse_lambda,
        transform_max_iter=args.transform_max_iter,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    warning_context = (
        warnings.catch_warnings()
        if not args.show_convergence_warnings
        else warnings.catch_warnings(record=False)
    )
    with warning_context:
        if not args.show_convergence_warnings:
            warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train_scaled)
        train_scores = _reconstruction_scores(model, train_scaled)
        test_scores = _reconstruction_scores(model, test_scaled)
    threshold = float(np.quantile(train_scores, args.alpha))

    output_path = args.output if args.output.is_absolute() else Path(__file__).resolve().parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_monitoring(
        test_scores,
        threshold,
        dataset["labels"],
        output_path,
        f"Dictionary learning monitoring (Lasso lambda={args.sparse_lambda:g})",
    )
    print(f"Saved dictionary learning monitoring plot to {output_path}")


if __name__ == "__main__":
    main()
