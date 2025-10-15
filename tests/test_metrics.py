import numpy as np

from src.models.thresholding import opt_threshold_by_precision_at_k


def test_precision_at_k_threshold_monotonic():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.05, 0.8, 0.3, 0.25, 0.85, 0.02, 0.01])
    thr = opt_threshold_by_precision_at_k(s, y, k=0.2)
    top = s >= thr
    assert abs(top.mean() - 0.2) < 1e-9
    assert (y[top].mean()) >= 0.5
