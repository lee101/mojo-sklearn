"""K-means."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class KMeans:
    """scikit-learn's KMeans (init='k-means++', algorithm='lloyd') reproduced exactly.

    Same MT19937 stream as `RandomState(random_state)`, the same greedy k-means++ seeding, centered
    data, relative `tol`, empty-cluster relocation and best-of-`n_init` rule, so `labels_` equals
    scikit-learn's on non-degenerate data (exact distance ties can still resolve differently).
    `n_init="auto"` means 1 for k-means++, as in scikit-learn >= 1.4.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300,
                 tol: float = 1e-4, n_init="auto", random_state: int = 0):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = n_init
        self.random_state = int(random_state)
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    def fit(self, X):
        X = f64(X)
        n, d = X.shape
        k = self.n_clusters
        if k > n:
            raise ValueError("n_clusters cannot exceed the number of samples")
        n_init = 1 if self.n_init == "auto" else int(self.n_init)
        centers = np.empty((k, d))
        labels = np.empty(n)
        stats = np.empty(2)
        lib().msk_kmeans_fit_sk(
            addr(X), n, d, k, n_init, self.max_iter, self.tol, self.random_state,
            addr(centers), addr(labels), addr(stats),
        )
        self.cluster_centers_ = centers
        self.labels_ = labels.astype(np.int32)
        self.inertia_ = float(stats[0])
        self.n_iter_ = int(stats[1])
        return self

    def predict(self, X):
        X = f64(X)
        n, d = X.shape
        labels = np.empty(n)
        lib().msk_kmeans_assign(
            addr(X), addr(self.cluster_centers_), addr(labels),
            n, d, self.n_clusters,
        )
        return labels.astype(np.int64)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def transform(self, X):
        from .metrics import pairwise_sqeuclidean

        return np.sqrt(pairwise_sqeuclidean(X, self.cluster_centers_))
