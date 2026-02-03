import numpy as np


def opt_threshold_by_precision_at_k(scores, y_true, k=0.01):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    n = max(1, int(len(scores) * k))
    idx = np.argsort(scores)[::-1]
    top_scores = scores[idx][:n]
    threshold = float(top_scores.min())
    return threshold
