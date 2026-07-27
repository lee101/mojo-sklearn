"""PCA."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class PCA:
    """Principal components from the covariance matrix, via Jacobi.

    Component signs are pinned so a refit does not flip them: the entry of
    largest magnitude in each component is positive.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X):
        X = f64(X)
        n, d = X.shape
        k = self.n_components
        if k > d:
            raise ValueError("n_components cannot exceed the number of features")
        self.mean_ = np.empty(d)
        self.components_ = np.empty((k, d))
        self.explained_variance_ = np.empty(k)
        cov_work = np.empty((d, d))
        vec_work = np.empty((d, d))
        total = lib().msk_pca_fit(
            addr(X), addr(self.mean_), addr(self.components_),
            addr(self.explained_variance_), addr(cov_work), addr(vec_work),
            n, d, k,
        )
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else
            np.zeros(k)
        )
        return self

    def transform(self, X):
        X = f64(X)
        n, d = X.shape
        k = self.n_components
        out = np.empty((n, k))
        lib().msk_pca_transform(
            addr(X), addr(self.mean_), addr(self.components_), addr(out), n, d, k
        )
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = f64(Z)
        n, k = Z.shape
        d = self.components_.shape[1]
        out = np.empty((n, d))
        lib().msk_pca_inverse(
            addr(Z), addr(self.mean_), addr(self.components_), addr(out), n, d, k
        )
        return out
