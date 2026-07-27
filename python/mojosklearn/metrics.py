"""Scoring functions."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def mean_squared_error(y_true, y_pred) -> float:
    a, b = f64(y_true).ravel(), f64(y_pred).ravel()
    return float(lib().msk_mse(addr(a), addr(b), a.size))


def mean_absolute_error(y_true, y_pred) -> float:
    a, b = f64(y_true).ravel(), f64(y_pred).ravel()
    return float(lib().msk_mae(addr(a), addr(b), a.size))


def r2_score(y_true, y_pred) -> float:
    a, b = f64(y_true).ravel(), f64(y_pred).ravel()
    return float(lib().msk_r2(addr(a), addr(b), a.size))


def accuracy_score(y_true, y_pred) -> float:
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_pred).ravel()
    if a.dtype.kind not in "fiub":
        # non-numeric labels: compare in Python, the sizes here are small
        return float(np.mean(a == b))
    a, b = f64(a), f64(b)
    return float(lib().msk_accuracy(addr(a), addr(b), a.size))


def pairwise_sqeuclidean(A, B) -> np.ndarray:
    A, B = f64(A), f64(B)
    n, d = A.shape
    m = B.shape[0]
    out = np.empty((n, m))
    lib().msk_pairwise_sqeuclidean(addr(A), addr(B), addr(out), n, m, d)
    return out


def silhouette_score(X, labels) -> float:
    X = f64(X)
    lab = f64(np.asarray(labels).ravel())
    n, d = X.shape
    k = int(lab.max()) + 1
    sums = np.empty(k)
    counts = np.empty(k)
    return float(lib().msk_silhouette(
        addr(X), addr(lab), n, d, k, addr(sums), addr(counts)
    ))
