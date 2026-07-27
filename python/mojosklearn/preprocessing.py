"""Scalers, with the scikit-learn method names."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class StandardScaler:
    """Zero mean, unit variance, per column.

    Matches `sklearn.preprocessing.StandardScaler` including its treatment of a
    constant column (scale 1.0, not 0.0) and its use of the population standard
    deviation.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X):
        X = f64(X)
        n, d = X.shape
        self.mean_ = np.empty(d)
        self.scale_ = np.empty(d)
        lib().msk_standard_fit(addr(X), addr(self.mean_), addr(self.scale_), n, d)
        return self

    def transform(self, X):
        X = f64(X)
        n, d = X.shape
        out = np.empty((n, d))
        lib().msk_standard_transform(
            addr(X), addr(self.mean_), addr(self.scale_), addr(out), n, d
        )
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = f64(X)
        n, d = X.shape
        out = np.empty((n, d))
        lib().msk_standard_inverse(
            addr(X), addr(self.mean_), addr(self.scale_), addr(out), n, d
        )
        return out


class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, X):
        X = f64(X)
        n, d = X.shape
        self.data_min_ = np.empty(d)
        self.data_max_ = np.empty(d)
        lib().msk_minmax_fit(addr(X), addr(self.data_min_), addr(self.data_max_), n, d)
        return self

    def transform(self, X):
        X = f64(X)
        n, d = X.shape
        out = np.empty((n, d))
        lo, hi = self.feature_range
        lib().msk_minmax_transform(
            addr(X), addr(self.data_min_), addr(self.data_max_), addr(out), n, d,
            float(lo), float(hi),
        )
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def normalize(X):
    """Scale each row to unit L2 norm. A zero row stays a zero row."""
    X = f64(X)
    n, d = X.shape
    out = np.empty((n, d))
    lib().msk_l2_normalize(addr(X), addr(out), n, d)
    return out
