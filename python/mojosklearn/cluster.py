"""K-means."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class KMeans:
    """Lloyd's algorithm with k-means++ seeding.

    `n_init` restarts keep the best inertia, as scikit-learn does; the seed
    feeds a deterministic LCG inside Mojo, so a run reproduces exactly.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300,
                 tol: float = 1e-8, n_init: int = 10, random_state: int = 0):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")

    def fit(self, X):
        X = f64(X)
        n, d = X.shape
        k = self.n_clusters
        if k > n:
            raise ValueError("n_clusters cannot exceed the number of samples")
        dist = np.empty(n)
        sums = np.empty(k * d)
        counts = np.empty(k)
        best_inertia = float("inf")
        best_centers = None
        best_labels = None
        for attempt in range(self.n_init):
            centers = np.empty((k, d))
            labels = np.empty(n)
            lib().msk_kmeans_plusplus(
                addr(X), addr(centers), addr(dist), n, d, k,
                self.random_state + attempt,
            )
            inertia = lib().msk_kmeans_lloyd(
                addr(X), addr(centers), addr(labels), addr(sums), addr(counts),
                n, d, k, self.max_iter, self.tol,
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels.astype(np.int64)
        self.inertia_ = float(best_inertia)
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
