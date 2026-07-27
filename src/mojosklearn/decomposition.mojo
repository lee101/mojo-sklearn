"""PCA by cyclic Jacobi eigendecomposition of the covariance matrix.

Jacobi is O(d^3) per sweep and needs a handful of sweeps, so this is the right
algorithm while d is in the low hundreds. Above that the covariance route is
the wrong one anyway and a randomized SVD belongs here instead.
"""

from std.math import sqrt

from mojosklearn.linalg import Ptr, axpy


def covariance(x: Ptr, mean: Ptr, cov: Ptr, row_work: Ptr, n: Int, d: Int):
    """Sample covariance (divisor n-1) of row-major X[n, d], and the column
    means. `row_work[d]` holds one centered row at a time so the accumulation
    can be a vectorized rank-1 update."""
    for j in range(d):
        mean[j] = 0.0
    for r in range(n):
        for j in range(d):
            mean[j] += x[r * d + j]
    for j in range(d):
        mean[j] /= Float64(n)
    for i in range(d * d):
        cov[i] = 0.0
    for r in range(n):
        for j in range(d):
            row_work[j] = x[r * d + j] - mean[j]
        for i in range(d):
            var a = row_work[i]
            if a != 0.0:
                axpy(a, row_work, cov + i * d, i + 1)
    var denom = Float64(n - 1) if n > 1 else 1.0
    for i in range(d):
        for j in range(i + 1):
            var v = cov[i * d + j] / denom
            cov[i * d + j] = v
            cov[j * d + i] = v


def jacobi_eigh(a: Ptr, vectors: Ptr, d: Int, sweeps: Int, tol: Float64) -> Int:
    """Diagonalize symmetric A[d, d] in place; eigenvectors land in columns of
    `vectors`. Returns the number of sweeps actually used."""
    for i in range(d):
        for j in range(d):
            vectors[i * d + j] = 1.0 if i == j else 0.0
    for sweep in range(sweeps):
        var off = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                off += a[i * d + j] * a[i * d + j]
        if off <= tol:
            return sweep
        for p in range(d):
            for q in range(p + 1, d):
                var apq = a[p * d + q]
                if apq == 0.0:
                    continue
                var theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq)
                var sign = 1.0 if theta >= 0.0 else -1.0
                var t = sign / (abs(theta) + sqrt(theta * theta + 1.0))
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                for k in range(d):
                    var akp = a[k * d + p]
                    var akq = a[k * d + q]
                    a[k * d + p] = c * akp - s * akq
                    a[k * d + q] = s * akp + c * akq
                for k in range(d):
                    var apk = a[p * d + k]
                    var aqk = a[q * d + k]
                    a[p * d + k] = c * apk - s * aqk
                    a[q * d + k] = s * apk + c * aqk
                for k in range(d):
                    var vkp = vectors[k * d + p]
                    var vkq = vectors[k * d + q]
                    vectors[k * d + p] = c * vkp - s * vkq
                    vectors[k * d + q] = s * vkp + c * vkq
    return sweeps


def pca_fit(
    x: Ptr,
    mean: Ptr,
    components: Ptr,
    variance: Ptr,
    cov_work: Ptr,
    vec_work: Ptr,
    n: Int,
    d: Int,
    k: Int,
) -> Float64:
    """Fit k principal components. `components[k, d]` comes back row-major and
    sorted by descending explained variance. Returns the total variance, so the
    caller can turn `variance[k]` into explained ratios."""
    # vec_work doubles as the centered-row scratch here; jacobi_eigh
    # overwrites it immediately afterwards
    covariance(x, mean, cov_work, vec_work, n, d)
    _ = jacobi_eigh(cov_work, vec_work, d, 60, 1e-24)

    var total = 0.0
    for i in range(d):
        total += cov_work[i * d + i]

    # selection sort over the eigenvalues, taking the k largest
    for slot in range(k):
        var best = slot
        for cand in range(slot + 1, d):
            if cov_work[cand * d + cand] > cov_work[best * d + best]:
                best = cand
        if best != slot:
            var tmp = cov_work[slot * d + slot]
            cov_work[slot * d + slot] = cov_work[best * d + best]
            cov_work[best * d + best] = tmp
            for r in range(d):
                var v = vec_work[r * d + slot]
                vec_work[r * d + slot] = vec_work[r * d + best]
                vec_work[r * d + best] = v
        variance[slot] = cov_work[slot * d + slot]
        # sign convention: largest-magnitude loading is positive, so a fit is
        # reproducible instead of flipping between runs
        var pivot = 0
        for r in range(1, d):
            if abs(vec_work[r * d + slot]) > abs(vec_work[pivot * d + slot]):
                pivot = r
        var flip = -1.0 if vec_work[pivot * d + slot] < 0.0 else 1.0
        for r in range(d):
            components[slot * d + r] = vec_work[r * d + slot] * flip
    return total


def pca_transform(
    x: Ptr, mean: Ptr, components: Ptr, dst: Ptr, n: Int, d: Int, k: Int
):
    for r in range(n):
        for c in range(k):
            var acc = 0.0
            for j in range(d):
                acc += (x[r * d + j] - mean[j]) * components[c * d + j]
            dst[r * k + c] = acc


def pca_inverse(
    z: Ptr, mean: Ptr, components: Ptr, dst: Ptr, n: Int, d: Int, k: Int
):
    for r in range(n):
        for j in range(d):
            var acc = mean[j]
            for c in range(k):
                acc += z[r * k + c] * components[c * d + j]
            dst[r * d + j] = acc
