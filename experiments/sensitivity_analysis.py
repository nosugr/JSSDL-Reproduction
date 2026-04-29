from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from jssdl import JSSDL
from jssdl.model.dictionary_update import calculate_effective_rank
from jssdl.monitoring.metrics import compute_far, compute_fdr
from jssdl.utils.data_loader import generate_numerical_dataset, load_config, standardize_train_test, to_feature_sample_matrix
from jssdl.utils.visualizer import (
    plot_rank_with_error_bars,
    plot_sensitivity_3d_bars,
)


def run_analysis(n_runs: int | None = None) -> pd.DataFrame:
    config = load_config(ROOT / "config.yaml")
    defaults = config["defaults"]
    numerical_cfg = config["numerical_simulation"]
    analysis_cfg = config["sensitivity_analysis"]
    n_runs = defaults["n_runs"] if n_runs is None else n_runs
    tau_values = [round(float(value), 2) for value in analysis_cfg["tau_values"]]
    sparsity_values = [int(value) for value in analysis_cfg["sparsity_D1_values"]]
    sensitivity_fault_bias = float(analysis_cfg.get("fault_bias", 8.0))
    kde_confidence = float(analysis_cfg.get("kde_confidence", numerical_cfg["kde_confidence"]))

    dataset = generate_numerical_dataset(
        n_features=numerical_cfg["n_features"],
        shared_rank=numerical_cfg["shared_rank"],
        n_train=numerical_cfg["n_train"],
        n_test_normal=numerical_cfg["n_test_normal"],
        n_test_fault=numerical_cfg["n_test_fault"],
        fault_feature=numerical_cfg["fault_feature"],
        fault_bias=sensitivity_fault_bias,
        noise_std=numerical_cfg["noise_std"],
        random_state=defaults["random_state"],
    )

    train_scaled, test_scaled, _ = standardize_train_test(dataset["train"], dataset["test"])
    Y_train = to_feature_sample_matrix(train_scaled)
    Y_test = to_feature_sample_matrix(test_scaled)
    labels = dataset["labels"].astype(bool)

    rows: list[dict[str, float | int]] = []

    total_jobs = len(tau_values) * len(sparsity_values) * n_runs
    for sparsity in tqdm(sparsity_values, total=total_jobs // len(tau_values), desc="Sparsity grid"):
        for tau in tau_values:
            for run_idx in range(n_runs):
                model = JSSDL(
                    n_atoms_D1=numerical_cfg["n_atoms_D1"],
                    n_atoms_D2=numerical_cfg["n_atoms_D2"],
                    sparsity_D1=sparsity,
                    sparsity_X1=numerical_cfg["sparsity_X1"],
                    sparsity_X2=numerical_cfg["sparsity_X2"],
                    lambda1=defaults["lambda1"],
                    lambda2=defaults["lambda2"],
                    lambda3=defaults["lambda3"],
                    lambda4=defaults["lambda4"],
                    lambda5=defaults["lambda5"],
                    lambda6=defaults.get("lambda6", 0.0),
                    tau=tau,
                    max_iter=numerical_cfg["max_iter"],
                    tol=defaults["tol"],
                    random_state=defaults["random_state"] + run_idx,
                )
                model.fit(Y_train, alpha=kde_confidence)
                preds = np.asarray(model.is_fault(Y_test), dtype=bool)
                rank_d2 = calculate_effective_rank(model.D2_) if model.D2_ is not None else 0
                rows.append(
                    {
                        "tau": tau,
                        "sparsity_D1": sparsity,
                        "run": run_idx,
                        "rank_D2": rank_d2,
                        "FDR": compute_fdr(labels, preds),
                        "FAR": compute_far(labels, preds),
                    }
                )

    results_df = pd.DataFrame(rows)
    summary_df = results_df.groupby(["tau", "sparsity_D1"], as_index=False)[["rank_D2", "FDR", "FAR"]].mean().sort_values(
        ["tau", "sparsity_D1"]
    )
    tau_summary_df = (
        results_df.groupby("tau", as_index=False)
        .agg(
            rank_D2_mean=("rank_D2", "mean"),
            rank_D2_std=("rank_D2", "std"),
        )
        .sort_values("tau")
        .fillna(0.0)
    )

    output_tables = ROOT / "outputs" / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)
    for stale_table in (
        output_tables / "sensitivity_analysis_lambda5_runs.csv",
        output_tables / "sensitivity_analysis_lambda5_summary.csv",
    ):
        if stale_table.exists():
            stale_table.unlink()
    summary_df.to_csv(output_tables / "sensitivity_analysis_summary.csv", index=False)
    tau_summary_df.to_csv(output_tables / "sensitivity_analysis_tau_summary.csv", index=False)

    figure_dir = ROOT / "outputs" / "figures" / "sensitivity_analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in figure_dir.glob("*.png"):
        stale_path.unlink()

    fdr_grid = summary_df.pivot(index="sparsity_D1", columns="tau", values="FDR").loc[sparsity_values, tau_values].to_numpy()
    far_grid = summary_df.pivot(index="sparsity_D1", columns="tau", values="FAR").loc[sparsity_values, tau_values].to_numpy()

    plot_rank_with_error_bars(
        tau_summary_df["tau"].to_numpy(),
        tau_summary_df["rank_D2_mean"].to_numpy(),
        tau_summary_df["rank_D2_std"].to_numpy(),
        "Effect of the Soft Threshold on the Rank of D2",
        "Average Rank",
        figure_dir / "effect_of_soft_threshold_on_rank_d2.png",
    )
    plot_sensitivity_3d_bars(
        tau_values,
        sparsity_values,
        far_grid,
        "Effect of Tau and the Sparsity of D1 on FAR",
        "Average of FAR",
        figure_dir / "effect_tau_sparsity_d1_on_far_3d.png",
    )
    plot_sensitivity_3d_bars(
        tau_values,
        sparsity_values,
        fdr_grid,
        "Effect of Tau and the Sparsity of D1 on FDR",
        "Average of FDR",
        figure_dir / "effect_tau_sparsity_d1_on_fdr_3d.png",
    )

    print(summary_df.to_string(index=False))
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JSSDL sensitivity analysis.")
    parser.add_argument("--n-runs", type=int, default=None, help="Number of repeated runs.")
    args = parser.parse_args()
    run_analysis(n_runs=args.n_runs)


if __name__ == "__main__":
    main()
