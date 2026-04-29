from __future__ import annotations

__all__ = [
    "apply_standardization",
    "generate_numerical_dataset",
    "initialize_dictionary_from_data",
    "initialize_random_codes",
    "initialize_random_dictionary",
    "initialize_sparse_codes",
    "initialize_svd_dictionary",
    "load_config",
    "normalize_columns",
    "project_column_sparsity",
    "standardize_train_test",
    "to_feature_sample_matrix",
]


def __getattr__(name: str):
    if name in {
        "apply_standardization",
        "generate_numerical_dataset",
        "load_config",
        "standardize_train_test",
        "to_feature_sample_matrix",
    }:
        from . import data_loader

        return getattr(data_loader, name)

    if name in {
        "initialize_dictionary_from_data",
        "initialize_random_codes",
        "initialize_random_dictionary",
        "initialize_sparse_codes",
        "initialize_svd_dictionary",
        "normalize_columns",
        "project_column_sparsity",
    }:
        from . import initializer

        return getattr(initializer, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
