"""Scoring functions and pairwise distances."""

from std.math import sqrt

from mojosklearn.linalg import Ptr, euclidean2


def mean_squared_error(y: Ptr, pred: Ptr, n: Int) -> Float64:
    var acc = 0.0
    for i in range(n):
        var d = y[i] - pred[i]
        acc += d * d
    return acc / Float64(n)


def mean_absolute_error(y: Ptr, pred: Ptr, n: Int) -> Float64:
    var acc = 0.0
    for i in range(n):
        acc += abs(y[i] - pred[i])
    return acc / Float64(n)


def r2_score(y: Ptr, pred: Ptr, n: Int) -> Float64:
    var mean = 0.0
    for i in range(n):
        mean += y[i]
    mean /= Float64(n)
    var ss_res = 0.0
    var ss_tot = 0.0
    for i in range(n):
        var e = y[i] - pred[i]
        var t = y[i] - mean
        ss_res += e * e
        ss_tot += t * t
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def accuracy_score(y: Ptr, pred: Ptr, n: Int) -> Float64:
    var hits = 0.0
    for i in range(n):
        if y[i] == pred[i]:
            hits += 1.0
    return hits / Float64(n)


def pairwise_sqeuclidean(a: Ptr, b: Ptr, dst: Ptr, n: Int, m: Int, d: Int):
    for i in range(n):
        for j in range(m):
            dst[i * m + j] = euclidean2(a + i * d, b + j * d, d)


def silhouette(x: Ptr, labels: Ptr, n: Int, d: Int, k: Int, sums: Ptr, counts: Ptr) -> Float64:
    """Mean silhouette over all samples. `sums[k]` and `counts[k]` are scratch.

    O(n^2 d): this is a diagnostic, not an inner loop.
    """
    var total = 0.0
    for i in range(n):
        for c in range(k):
            sums[c] = 0.0
            counts[c] = 0.0
        for j in range(n):
            if i == j:
                continue
            var c = Int(labels[j])
            sums[c] += sqrt(euclidean2(x + i * d, x + j * d, d))
            counts[c] += 1.0
        var own = Int(labels[i])
        var a = sums[own] / counts[own] if counts[own] > 0.0 else 0.0
        var b = 1.7976931348623157e308
        for c in range(k):
            if c == own or counts[c] == 0.0:
                continue
            var cand = sums[c] / counts[c]
            if cand < b:
                b = cand
        if b == 1.7976931348623157e308:
            continue
        var denom = a if a > b else b
        if denom > 0.0:
            total += (b - a) / denom
    return total / Float64(n)
