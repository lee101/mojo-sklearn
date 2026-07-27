"""Brute-force k-nearest-neighbours.

No tree. For the d > 20 case that most feature matrices are, a scan with a
k-sized insertion heap beats a KD-tree, and it is the only structure whose cost
is predictable.
"""

from mojosklearn.linalg import Ptr, euclidean2


def knn_query(
    train: Ptr,
    query: Ptr,
    idx_out: Ptr,
    dist_out: Ptr,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
):
    """For each of the m query rows, the k nearest training rows.

    `idx_out[m, k]` and `dist_out[m, k]` come back sorted nearest-first, with
    squared distances — the caller takes the square root only if it needs one.
    """
    for q in range(m):
        var base = q * k
        for s in range(k):
            dist_out[base + s] = 1.7976931348623157e308
            idx_out[base + s] = -1.0
        for r in range(n):
            var dist = euclidean2(query + q * d, train + r * d, d)
            if dist >= dist_out[base + k - 1]:
                continue
            var s = k - 1
            while s > 0 and dist_out[base + s - 1] > dist:
                dist_out[base + s] = dist_out[base + s - 1]
                idx_out[base + s] = idx_out[base + s - 1]
                s -= 1
            dist_out[base + s] = dist
            idx_out[base + s] = Float64(r)


def knn_regress(
    train: Ptr,
    y: Ptr,
    query: Ptr,
    dst: Ptr,
    idx_work: Ptr,
    dist_work: Ptr,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
):
    knn_query(train, query, idx_work, dist_work, n, d, m, k)
    for q in range(m):
        var acc = 0.0
        for s in range(k):
            acc += y[Int(idx_work[q * k + s])]
        dst[q] = acc / Float64(k)


def knn_classify(
    train: Ptr,
    y: Ptr,
    query: Ptr,
    dst: Ptr,
    idx_work: Ptr,
    dist_work: Ptr,
    votes: Ptr,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
    classes: Int,
):
    """Majority vote over the k nearest labels. `y` holds class indices in
    [0, classes), `votes[classes]` is scratch. Ties go to the lowest index,
    which is what scikit-learn does."""
    knn_query(train, query, idx_work, dist_work, n, d, m, k)
    for q in range(m):
        for c in range(classes):
            votes[c] = 0.0
        for s in range(k):
            votes[Int(y[Int(idx_work[q * k + s])])] += 1.0
        var best = 0
        for c in range(1, classes):
            if votes[c] > votes[best]:
                best = c
        dst[q] = Float64(best)
