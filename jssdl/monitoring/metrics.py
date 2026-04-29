from __future__ import annotations

import numpy as np


def compute_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=bool)
    predictions = np.asarray(y_pred, dtype=bool)
    faulty_count = int(labels.sum())
    if faulty_count == 0:
        return 0.0
    tp = int(np.logical_and(labels, predictions).sum())
    return tp / faulty_count


def compute_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=bool)
    predictions = np.asarray(y_pred, dtype=bool)
    normal_count = int((~labels).sum())
    if normal_count == 0:
        return 0.0
    fp = int(np.logical_and(~labels, predictions).sum())
    return fp / normal_count
