
import numpy as np
from sklearn.metrics import average_precision_score

def precision_at_k(y_true, scores, k=0.01):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = max(1, int(len(scores)*k))
    idx = np.argsort(scores)[::-1][:n]
    return float((y_true[idx] == 1).mean())

def auprc(y_true, scores):
    return float(average_precision_score(y_true, scores))
