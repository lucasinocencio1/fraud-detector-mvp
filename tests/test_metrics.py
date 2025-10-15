
import numpy as np
from src.models.thresholding import opt_threshold_by_precision_at_k

def test_precision_at_k_threshold_monotonic():
    y = np.array([0,0,1,0,1,0,0,1,0,0])
    s = np.array([.1,.2,.9,.05,.8,.3,.25,.85,.02,.01])
    thr = opt_threshold_by_precision_at_k(s, y, k=0.2)
    top = s >= thr
    assert abs(top.mean() - 0.2) < 1e-9
    assert (y[top].mean()) >= 0.5
