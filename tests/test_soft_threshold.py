from __future__ import annotations

import numpy as np

from jssdl.model.soft_threshold import soft_threshold


def test_soft_threshold_scalar() -> None:
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-3.0, 1.5) == -1.5
    assert soft_threshold(0.5, 1.0) == 0.0


def test_soft_threshold_array() -> None:
    values = np.array([2.0, -2.0, 0.5, -0.1])
    expected = np.array([1.0, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(soft_threshold(values, 1.0), expected)
