import numpy as np


def psi(expected, actual, bins=10):
    expected = np.asarray(expected).ravel()
    actual = np.asarray(actual).ravel()
    e_bins = np.quantile(expected, np.linspace(0, 1, bins + 1))
    a_hist, _ = np.histogram(actual, e_bins)
    e_hist, _ = np.histogram(expected, e_bins)
    a = a_hist / (len(actual) + 1e-9)
    e = e_hist / (len(expected) + 1e-9)
    a = np.where(a == 0, 1e-6, a)
    e = np.where(e == 0, 1e-6, e)
    return float(((a - e) * np.log(a / e)).sum())
