"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. Callers own all
memory, including scratch — nothing here allocates, so nothing here can leak.
"""

from mojosklearn.cluster import kmeans_assign, kmeans_lloyd, kmeans_plusplus
from mojosklearn.decomposition import (
    covariance,
    jacobi_eigh,
    pca_fit,
    pca_inverse,
    pca_transform,
)
from mojosklearn.linalg import Ptr, cholesky, cholesky_solve, dot, gemm_tn, gram
from mojosklearn.linear import (
    linear_predict,
    logistic_fit,
    logistic_predict_proba,
    ridge_fit,
    sgd_ridge_fit,
)
from mojosklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    pairwise_sqeuclidean,
    r2_score,
    silhouette,
)
from mojosklearn.neighbors import knn_classify, knn_query, knn_regress
from mojosklearn.preprocessing import (
    l2_normalize,
    minmax_fit,
    minmax_transform,
    standard_fit,
    standard_inverse,
    standard_transform,
)


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


# ---------------------------------------------------------------- linalg
@export("msk_dot")
def msk_dot(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return dot(p(a), p(b), n)


@export("msk_gram")
def msk_gram(x: Int, dst: Int, n: Int, d: Int) abi("C"):
    gram(p(x), p(dst), n, d)


@export("msk_gemm_tn")
def msk_gemm_tn(a: Int, b: Int, dst: Int, n: Int, d: Int, m: Int) abi("C"):
    gemm_tn(p(a), p(b), p(dst), n, d, m)


@export("msk_cholesky")
def msk_cholesky(a: Int, d: Int) abi("C") -> Int:
    return 1 if cholesky(p(a), d) else 0


@export("msk_cholesky_solve")
def msk_cholesky_solve(l: Int, b: Int, d: Int) abi("C"):
    cholesky_solve(p(l), p(b), d)


# --------------------------------------------------------- preprocessing
@export("msk_standard_fit")
def msk_standard_fit(x: Int, mean: Int, scale: Int, n: Int, d: Int) abi("C"):
    standard_fit(p(x), p(mean), p(scale), n, d)


@export("msk_standard_transform")
def msk_standard_transform(
    x: Int, mean: Int, scale: Int, dst: Int, n: Int, d: Int
) abi("C"):
    standard_transform(p(x), p(mean), p(scale), p(dst), n, d)


@export("msk_standard_inverse")
def msk_standard_inverse(
    x: Int, mean: Int, scale: Int, dst: Int, n: Int, d: Int
) abi("C"):
    standard_inverse(p(x), p(mean), p(scale), p(dst), n, d)


@export("msk_minmax_fit")
def msk_minmax_fit(x: Int, lo: Int, hi: Int, n: Int, d: Int) abi("C"):
    minmax_fit(p(x), p(lo), p(hi), n, d)


@export("msk_minmax_transform")
def msk_minmax_transform(
    x: Int, lo: Int, hi: Int, dst: Int, n: Int, d: Int, fmin: Float64, fmax: Float64
) abi("C"):
    minmax_transform(p(x), p(lo), p(hi), p(dst), n, d, fmin, fmax)


@export("msk_l2_normalize")
def msk_l2_normalize(x: Int, dst: Int, n: Int, d: Int) abi("C"):
    l2_normalize(p(x), p(dst), n, d)


# ---------------------------------------------------------------- linear
@export("msk_ridge_fit")
def msk_ridge_fit(
    x: Int, y: Int, coef: Int, work: Int, n: Int, d: Int, alpha: Float64
) abi("C") -> Int:
    return 1 if ridge_fit(p(x), p(y), p(coef), p(work), n, d, alpha) else 0


@export("msk_linear_predict")
def msk_linear_predict(
    x: Int, coef: Int, dst: Int, n: Int, d: Int, intercept: Float64
) abi("C"):
    linear_predict(p(x), p(coef), p(dst), n, d, intercept)


@export("msk_logistic_fit")
def msk_logistic_fit(
    x: Int,
    y: Int,
    coef: Int,
    grad: Int,
    n: Int,
    d: Int,
    lr: Float64,
    l2: Float64,
    epochs: Int,
) abi("C") -> Float64:
    return logistic_fit(p(x), p(y), p(coef), p(grad), n, d, lr, l2, epochs)


@export("msk_logistic_predict_proba")
def msk_logistic_predict_proba(x: Int, coef: Int, dst: Int, n: Int, d: Int) abi("C"):
    logistic_predict_proba(p(x), p(coef), p(dst), n, d)


@export("msk_sgd_ridge_fit")
def msk_sgd_ridge_fit(
    x: Int, y: Int, coef: Int, n: Int, d: Int, lr: Float64, l2: Float64, epochs: Int
) abi("C") -> Float64:
    return sgd_ridge_fit(p(x), p(y), p(coef), n, d, lr, l2, epochs)


# --------------------------------------------------------------- cluster
@export("msk_kmeans_plusplus")
def msk_kmeans_plusplus(
    x: Int, centers: Int, dist: Int, n: Int, d: Int, k: Int, seed: Int
) abi("C"):
    kmeans_plusplus(p(x), p(centers), p(dist), n, d, k, seed)


@export("msk_kmeans_lloyd")
def msk_kmeans_lloyd(
    x: Int,
    centers: Int,
    labels: Int,
    sums: Int,
    counts: Int,
    n: Int,
    d: Int,
    k: Int,
    max_iter: Int,
    tol: Float64,
) abi("C") -> Float64:
    return kmeans_lloyd(
        p(x), p(centers), p(labels), p(sums), p(counts), n, d, k, max_iter, tol
    )


@export("msk_kmeans_assign")
def msk_kmeans_assign(
    x: Int, centers: Int, labels: Int, n: Int, d: Int, k: Int
) abi("C") -> Float64:
    return kmeans_assign(p(x), p(centers), p(labels), n, d, k)


# --------------------------------------------------------- decomposition
@export("msk_covariance")
def msk_covariance(x: Int, mean: Int, cov: Int, row_work: Int, n: Int, d: Int) abi("C"):
    covariance(p(x), p(mean), p(cov), p(row_work), n, d)


@export("msk_jacobi_eigh")
def msk_jacobi_eigh(a: Int, vectors: Int, d: Int, sweeps: Int, tol: Float64) abi("C") -> Int:
    return jacobi_eigh(p(a), p(vectors), d, sweeps, tol)


@export("msk_pca_fit")
def msk_pca_fit(
    x: Int,
    mean: Int,
    components: Int,
    variance: Int,
    cov_work: Int,
    vec_work: Int,
    n: Int,
    d: Int,
    k: Int,
) abi("C") -> Float64:
    return pca_fit(
        p(x), p(mean), p(components), p(variance), p(cov_work), p(vec_work), n, d, k
    )


@export("msk_pca_transform")
def msk_pca_transform(
    x: Int, mean: Int, components: Int, dst: Int, n: Int, d: Int, k: Int
) abi("C"):
    pca_transform(p(x), p(mean), p(components), p(dst), n, d, k)


@export("msk_pca_inverse")
def msk_pca_inverse(
    z: Int, mean: Int, components: Int, dst: Int, n: Int, d: Int, k: Int
) abi("C"):
    pca_inverse(p(z), p(mean), p(components), p(dst), n, d, k)


# ------------------------------------------------------------- neighbors
@export("msk_knn_query")
def msk_knn_query(
    train: Int, query: Int, idx: Int, dist: Int, n: Int, d: Int, m: Int, k: Int
) abi("C"):
    knn_query(p(train), p(query), p(idx), p(dist), n, d, m, k)


@export("msk_knn_regress")
def msk_knn_regress(
    train: Int,
    y: Int,
    query: Int,
    dst: Int,
    idx: Int,
    dist: Int,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
) abi("C"):
    knn_regress(p(train), p(y), p(query), p(dst), p(idx), p(dist), n, d, m, k)


@export("msk_knn_classify")
def msk_knn_classify(
    train: Int,
    y: Int,
    query: Int,
    dst: Int,
    idx: Int,
    dist: Int,
    votes: Int,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
    classes: Int,
) abi("C"):
    knn_classify(
        p(train), p(y), p(query), p(dst), p(idx), p(dist), p(votes),
        n, d, m, k, classes,
    )


# --------------------------------------------------------------- metrics
@export("msk_mse")
def msk_mse(y: Int, pred: Int, n: Int) abi("C") -> Float64:
    return mean_squared_error(p(y), p(pred), n)


@export("msk_mae")
def msk_mae(y: Int, pred: Int, n: Int) abi("C") -> Float64:
    return mean_absolute_error(p(y), p(pred), n)


@export("msk_r2")
def msk_r2(y: Int, pred: Int, n: Int) abi("C") -> Float64:
    return r2_score(p(y), p(pred), n)


@export("msk_accuracy")
def msk_accuracy(y: Int, pred: Int, n: Int) abi("C") -> Float64:
    return accuracy_score(p(y), p(pred), n)


@export("msk_pairwise_sqeuclidean")
def msk_pairwise_sqeuclidean(a: Int, b: Int, dst: Int, n: Int, m: Int, d: Int) abi("C"):
    pairwise_sqeuclidean(p(a), p(b), p(dst), n, m, d)


@export("msk_silhouette")
def msk_silhouette(
    x: Int, labels: Int, n: Int, d: Int, k: Int, sums: Int, counts: Int
) abi("C") -> Float64:
    return silhouette(p(x), p(labels), n, d, k, p(sums), p(counts))
