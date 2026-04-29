from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _format_percentage(value: float) -> str:
    percentage = f"{100.0 * float(value):.1f}"
    if percentage.endswith(".0"):
        percentage = percentage[:-2]
    return f"{percentage}%"


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: str | Path,
    cmap: str = "viridis",
    xlabel: str = "Atom",
    ylabel: str = "Feature",
    figsize: tuple[float, float] = (8.0, 5.0),
    x_ticks: list[float] | np.ndarray | None = None,
    x_tick_labels: list[int | str] | None = None,
    y_ticks: list[float] | np.ndarray | None = None,
    y_tick_labels: list[int | str] | None = None,
) -> None:
    path = _ensure_parent(output_path)
    values = np.asarray(matrix, dtype=float)
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    fig, ax = plt.subplots(figsize=figsize)
    if max_abs > 0.0:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap)

    if x_ticks is not None:
        x_tick_positions = np.asarray(x_ticks, dtype=float)
        ax.set_xticks(x_tick_positions)
        if x_tick_labels is not None:
            ax.set_xticklabels([str(label) for label in x_tick_labels])
    elif x_tick_labels is not None:
        raise ValueError("x_tick_labels requires x_ticks to be provided.")

    if y_ticks is not None:
        y_tick_positions = np.asarray(y_ticks, dtype=float)
        ax.set_yticks(y_tick_positions)
        if y_tick_labels is not None:
            ax.set_yticklabels([str(label) for label in y_tick_labels])
    elif y_tick_labels is not None:
        raise ValueError("y_tick_labels requires y_ticks to be provided.")

    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fault_annotated_heatmap(
    matrix: np.ndarray,
    labels: np.ndarray,
    fault_feature: int,
    title: str,
    output_path: str | Path,
    cmap: str = "coolwarm",
    xlabel: str = "Sample",
    ylabel: str = "Feature",
    figsize: tuple[float, float] = (12.0, 4.8),
    y_ticks: list[float] | np.ndarray | None = None,
    y_tick_labels: list[int | str] | None = None,
) -> None:
    path = _ensure_parent(output_path)
    values = np.asarray(matrix, dtype=float)
    label_values = np.asarray(labels, dtype=int).ravel()
    if values.ndim != 2:
        raise ValueError("matrix must be a 2D array.")
    if values.shape[1] != label_values.shape[0]:
        raise ValueError("labels must match the number of samples in matrix.")

    n_features, n_samples = values.shape
    feature_index = int(fault_feature)
    if not 0 <= feature_index < n_features:
        raise ValueError(f"fault_feature must be in [0, {n_features - 1}], got {feature_index}.")

    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    fig, ax = plt.subplots(figsize=figsize)
    if max_abs > 0.0:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap)

    if y_ticks is not None:
        y_tick_positions = np.asarray(y_ticks, dtype=float)
        ax.set_yticks(y_tick_positions)
        if y_tick_labels is not None:
            ax.set_yticklabels([str(label) for label in y_tick_labels])

    fault_columns = np.flatnonzero(label_values == 1)
    if fault_columns.size > 0:
        start_col = int(fault_columns[0])
        end_col = int(fault_columns[-1])
        ax.axvline(start_col - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
        rect = Rectangle(
            (start_col - 0.5, feature_index - 0.5),
            end_col - start_col + 1,
            1.0,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
        )
        ax.add_patch(rect)
        ax.text(
            start_col,
            max(feature_index - 1.1, 0.0),
            f"Fault feature = {feature_index + 1}",
            color="red",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.75, "pad": 2},
        )

    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dictionary_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: str | Path,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8.5, 5.2),
    feature_labels: list[int | str] | None = None,
    show_title: bool = False,
) -> None:
    path = _ensure_parent(output_path)
    values = np.asarray(matrix, dtype=float)
    n_features, n_atoms = values.shape
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0

    if feature_labels is None:
        labels = [str(index) for index in range(1, n_features + 1)]
    else:
        labels = [str(label) for label in feature_labels]
        if len(labels) != n_features:
            raise ValueError(
                f"Expected {n_features} feature labels for the dictionary heatmap, got {len(labels)}."
            )

    fig, ax = plt.subplots(figsize=figsize)
    if max_abs > 0.0:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs, interpolation="nearest")
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if show_title:
        ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, n_atoms, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.grid(which="minor", color="0.15", linestyle="-", linewidth=0.35, alpha=0.85)
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.15")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_monitoring_scores(
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    title: str,
    ylabel: str,
    output_path: str | Path,
    split_index: int | None = None,
    tick_interval: int = 200,
    annotate_metrics: bool = False,
) -> None:
    path = _ensure_parent(output_path)
    score_values = np.asarray(scores, dtype=float).ravel()
    label_values = np.asarray(labels, dtype=int).ravel()
    x_axis = np.arange(1, score_values.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_axis, score_values, linewidth=1.4, label=ylabel)
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.5, label="Threshold")
    fault_regions = np.where(label_values == 1)[0]
    if fault_regions.size > 0:
        ax.scatter(x_axis[fault_regions], score_values[fault_regions], s=10, c="tab:orange", alpha=0.6, label="Faulty samples")

    if split_index is not None and 1 <= split_index <= score_values.shape[0]:
        ax.axvline(split_index, color="0.5", linestyle=":", linewidth=1.0, alpha=0.75)
        tick_values = set(range(tick_interval, score_values.shape[0] + 1, tick_interval))
        tick_values.add(split_index)
        tick_values.add(score_values.shape[0])
        ax.set_xticks(sorted(tick_values))
        ax.set_xlim(1, score_values.shape[0])

    if annotate_metrics:
        predictions = score_values > float(threshold)
        normal_mask = label_values == 0
        fault_mask = label_values == 1
        far = float(predictions[normal_mask].mean()) if np.any(normal_mask) else 0.0
        fdr = float(predictions[fault_mask].mean()) if np.any(fault_mask) else 0.0
        ax.text(0.25, 1.03, f"FAR: {_format_percentage(far)}", transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
        ax.text(0.75, 1.03, f"FDR: {_format_percentage(fdr)}", transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
    else:
        ax.set_title(title)

    ax.set_xlabel("Sample number")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    if annotate_metrics:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _set_rank_axis_ticks(ax, mean_values: np.ndarray, std_values: np.ndarray) -> None:
    upper_bound = float(np.max(mean_values + std_values)) if mean_values.size else 20.0
    y_max = 25 if upper_bound > 20.0 else 20
    tick_values = np.arange(0, y_max + 1, 5, dtype=int)
    ax.set_ylim(0, y_max)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([str(int(value)) for value in tick_values])


def plot_rank_with_error_bars(
    tau_values: list[float] | np.ndarray,
    mean_values: list[float] | np.ndarray,
    std_values: list[float] | np.ndarray,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    path = _ensure_parent(output_path)
    tau = np.asarray(tau_values, dtype=float)
    mean = np.asarray(mean_values, dtype=float)
    std = np.asarray(std_values, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        tau,
        mean,
        yerr=std,
        fmt="o-",
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.2,
        capsize=3,
        markersize=4,
        linewidth=1.4,
    )
    _set_rank_axis_ticks(ax, mean, std)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_3d_bar_grid(
    ax,
    tau_values: list[float] | np.ndarray,
    sparsity_values: list[float] | np.ndarray,
    grid: np.ndarray,
    zlabel: str,
) -> None:
    tau = np.asarray(tau_values, dtype=float)
    sparsity = np.asarray(sparsity_values, dtype=float)
    values = np.asarray(grid, dtype=float)

    x_index, y_index = np.meshgrid(np.arange(sparsity.size), np.arange(tau.size), indexing="xy")
    heights = values.T.ravel()
    x = x_index.ravel()
    y = y_index.ravel()
    z = np.zeros_like(heights)
    dx = np.full_like(heights, 0.7, dtype=float)
    dy = np.full_like(heights, 0.7, dtype=float)

    norm = Normalize(vmin=float(np.min(heights)), vmax=float(np.max(heights)) if heights.size else 1.0)
    colors = cm.viridis(norm(heights))
    ax.bar3d(x, y, z, dx, dy, heights, color=colors, shade=False, edgecolor="0.25", linewidth=0.6)
    ax.set_xlabel(r"The sparsity of $D_1$")
    ax.set_ylabel(r"$\tau$")
    ax.set_zlabel(zlabel)
    ax.set_xticks(np.arange(sparsity.size) + 0.35)
    ax.set_xticklabels([str(int(v)) for v in sparsity])
    ax.set_yticks(np.arange(tau.size) + 0.35)
    ax.set_yticklabels([f"{v:.2f}" for v in tau])
    ax.view_init(elev=28, azim=-125)


def plot_sensitivity_3d_bars(
    tau_values: list[float] | np.ndarray,
    sparsity_values: list[float] | np.ndarray,
    grid: np.ndarray,
    title: str,
    zlabel: str,
    output_path: str | Path,
) -> None:
    path = _ensure_parent(output_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    _plot_3d_bar_grid(ax, tau_values, sparsity_values, grid, zlabel)
    ax.set_title(title, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

