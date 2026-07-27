"""Brute-force nearest neighbours."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class _KNNBase:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = int(n_neighbors)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def kneighbors(self, X, return_distance: bool = True):
        X = f64(X)
        m, d = X.shape
        n = self._X.shape[0]
        k = min(self.n_neighbors, n)
        idx = np.empty((m, k))
        dist = np.empty((m, k))
        lib().msk_knn_query(
            addr(self._X), addr(X), addr(idx), addr(dist), n, d, m, k
        )
        if return_distance:
            return np.sqrt(dist), idx.astype(np.int64)
        return idx.astype(np.int64)


class KNeighborsRegressor(_KNNBase):
    def fit(self, X, y):
        self._X = f64(X)
        self._y = f64(y).ravel()
        return self

    def predict(self, X):
        X = f64(X)
        m, d = X.shape
        n = self._X.shape[0]
        k = min(self.n_neighbors, n)
        out = np.empty(m)
        idx = np.empty((m, k))
        dist = np.empty((m, k))
        lib().msk_knn_regress(
            addr(self._X), addr(self._y), addr(X), addr(out),
            addr(idx), addr(dist), n, d, m, k,
        )
        return out


class KNeighborsClassifier(_KNNBase):
    def fit(self, X, y):
        self._X = f64(X)
        y = np.asarray(y).ravel()
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._y = f64(encoded.astype(np.float64))
        return self

    def predict(self, X):
        X = f64(X)
        m, d = X.shape
        n = self._X.shape[0]
        k = min(self.n_neighbors, n)
        classes = len(self.classes_)
        out = np.empty(m)
        idx = np.empty((m, k))
        dist = np.empty((m, k))
        votes = np.empty(classes)
        lib().msk_knn_classify(
            addr(self._X), addr(self._y), addr(X), addr(out),
            addr(idx), addr(dist), addr(votes), n, d, m, k, classes,
        )
        return self.classes_[out.astype(np.int64)]

    def score(self, X, y):
        from .metrics import accuracy_score

        return accuracy_score(y, self.predict(X))
