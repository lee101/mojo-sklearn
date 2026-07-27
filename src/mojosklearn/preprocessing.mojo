"""Feature scaling. Column statistics over row-major X[n, d]."""

from std.math import sqrt

from mojosklearn.linalg import Ptr


def standard_fit(x: Ptr, mean: Ptr, scale: Ptr, n: Int, d: Int):
    """Column means and population standard deviations.

    A zero-variance column gets a scale of 1.0 rather than 0.0, so transform
    leaves it alone instead of producing infinities — the same choice
    scikit-learn makes.
    """
    for j in range(d):
        mean[j] = 0.0
        scale[j] = 0.0
    for r in range(n):
        for j in range(d):
            mean[j] += x[r * d + j]
    for j in range(d):
        mean[j] /= Float64(n)
    for r in range(n):
        for j in range(d):
            var diff = x[r * d + j] - mean[j]
            scale[j] += diff * diff
    for j in range(d):
        var v = sqrt(scale[j] / Float64(n))
        scale[j] = 1.0 if v == 0.0 else v


def standard_transform(x: Ptr, mean: Ptr, scale: Ptr, dst: Ptr, n: Int, d: Int):
    for r in range(n):
        for j in range(d):
            dst[r * d + j] = (x[r * d + j] - mean[j]) / scale[j]


def standard_inverse(x: Ptr, mean: Ptr, scale: Ptr, dst: Ptr, n: Int, d: Int):
    for r in range(n):
        for j in range(d):
            dst[r * d + j] = x[r * d + j] * scale[j] + mean[j]


def minmax_fit(x: Ptr, lo: Ptr, hi: Ptr, n: Int, d: Int):
    for j in range(d):
        lo[j] = x[j]
        hi[j] = x[j]
    for r in range(1, n):
        for j in range(d):
            var v = x[r * d + j]
            if v < lo[j]:
                lo[j] = v
            if v > hi[j]:
                hi[j] = v


def minmax_transform(
    x: Ptr, lo: Ptr, hi: Ptr, dst: Ptr, n: Int, d: Int, fmin: Float64, fmax: Float64
):
    for j in range(d):
        var span = hi[j] - lo[j]
        var denom = 1.0 if span == 0.0 else span
        for r in range(n):
            var unit = (x[r * d + j] - lo[j]) / denom
            dst[r * d + j] = unit * (fmax - fmin) + fmin


def l2_normalize(x: Ptr, dst: Ptr, n: Int, d: Int):
    for r in range(n):
        var acc = 0.0
        for j in range(d):
            var v = x[r * d + j]
            acc += v * v
        var norm = sqrt(acc)
        var inv = 1.0 if norm == 0.0 else 1.0 / norm
        for j in range(d):
            dst[r * d + j] = x[r * d + j] * inv
