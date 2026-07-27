"""mojo-sklearn against scikit-learn on the same data.

    pixi run python bench/bench.py

Reports the honest number, including the cases scikit-learn wins — it is
mature Cython over LAPACK and it is not going to lose everywhere.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojosklearn as msk  # noqa: E402
from sklearn import cluster as sk_cluster  # noqa: E402
from sklearn import decomposition as sk_decomp  # noqa: E402
from sklearn import linear_model as sk_linear  # noqa: E402
from sklearn import neighbors as sk_neigh  # noqa: E402
from sklearn import preprocessing as sk_pre  # noqa: E402
from sklearn import metrics as sk_metrics  # noqa: E402


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def data(n: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.normal(size=(n, d)))
    w = rng.normal(size=d)
    y = np.ascontiguousarray(X @ w + rng.normal(scale=0.1, size=n))
    return X, y


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("StandardScaler.fit_transform (200k x 20)")
def _():
    X, _ = data(200_000, 20)
    return (lambda: msk.StandardScaler().fit_transform(X),
            lambda: sk_pre.StandardScaler().fit_transform(X))


@case("Ridge.fit (200k x 20)")
def _():
    X, y = data(200_000, 20)
    return (lambda: msk.Ridge(alpha=1.0).fit(X, y),
            lambda: sk_linear.Ridge(alpha=1.0).fit(X, y))


@case("LinearRegression.fit (50k x 60)")
def _():
    X, y = data(50_000, 60)
    return (lambda: msk.LinearRegression().fit(X, y),
            lambda: sk_linear.LinearRegression().fit(X, y))


@case("LinearRegression.predict (200k x 20)")
def _():
    X, y = data(200_000, 20)
    ours = msk.LinearRegression().fit(X, y)
    theirs = sk_linear.LinearRegression().fit(X, y)
    return (lambda: ours.predict(X), lambda: theirs.predict(X))


@case("KMeans.fit k=8 (50k x 10, 1 init)")
def _():
    X, _ = data(50_000, 10)
    return (lambda: msk.KMeans(n_clusters=8, n_init=1, random_state=0).fit(X),
            lambda: sk_cluster.KMeans(n_clusters=8, n_init=1, random_state=0).fit(X))


@case("PCA.fit k=5 (100k x 30)")
def _():
    X, _ = data(100_000, 30)
    return (lambda: msk.PCA(n_components=5).fit(X),
            lambda: sk_decomp.PCA(n_components=5).fit(X))


@case("KNN.predict k=5 (5k train, 2k query, 20d)")
def _():
    X, y = data(5_000, 20)
    Q, _ = data(2_000, 20, seed=7)
    lab = (y > 0).astype(np.int64)
    ours = msk.KNeighborsClassifier(n_neighbors=5).fit(X, lab)
    theirs = sk_neigh.KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X, lab)
    return (lambda: ours.predict(Q), lambda: theirs.predict(Q))


@case("pairwise_sqeuclidean (3k x 3k, 20d)")
def _():
    A, _ = data(3_000, 20)
    B, _ = data(3_000, 20, seed=3)
    return (lambda: msk.pairwise_sqeuclidean(A, B),
            lambda: sk_metrics.pairwise_distances(A, B) ** 2)


@case("r2_score (2M)")
def _():
    rng = np.random.default_rng(0)
    y = np.ascontiguousarray(rng.normal(size=2_000_000))
    p = np.ascontiguousarray(y + rng.normal(scale=0.1, size=2_000_000))
    return (lambda: msk.r2_score(y, p), lambda: sk_metrics.r2_score(y, p))


def main() -> None:
    print(f"{'case':<44}{'mojo-sklearn':>14}{'scikit-learn':>14}{'ratio':>9}")
    print("-" * 81)
    for name, build in CASES:
        ours, theirs = build()
        ours()  # warm the library load out of the measurement
        a = timeit(ours)
        b = timeit(theirs)
        flag = "faster" if a < b else "slower"
        print(f"{name:<44}{a * 1e3:>12.1f}ms{b * 1e3:>12.1f}ms{b / a:>8.2f}x  {flag}")


if __name__ == "__main__":
    main()
