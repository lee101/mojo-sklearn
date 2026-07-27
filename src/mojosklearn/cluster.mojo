"""K-means: k-means++ seeding and Lloyd iterations."""

from std.math import sqrt

from mojosklearn.linalg import Ptr, euclidean2


def kmeans_plusplus(
    x: Ptr, centers: Ptr, dist: Ptr, n: Int, d: Int, k: Int, seed: Int
):
    """Seed `centers[k, d]` from the rows of X with the D^2 rule.

    `dist[n]` is scratch. The random source is a 64-bit LCG rather than a call
    into a host RNG, so a seed reproduces a run in Mojo, in Python, and inside
    a compiled binary alike.
    """
    var state = UInt64(seed) * 6364136223846793005 + 1442695040888963407
    state = state * 6364136223846793005 + 1442695040888963407
    var first = Int(state >> 33) % n
    for j in range(d):
        centers[j] = x[first * d + j]
    for c in range(1, k):
        var total = 0.0
        for r in range(n):
            var best = euclidean2(x + r * d, centers, d)
            for prev in range(1, c):
                var cand = euclidean2(x + r * d, centers + prev * d, d)
                if cand < best:
                    best = cand
            dist[r] = best
            total += best
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float64(Int(state >> 11)) / Float64(1 << 53) * total
        var acc = 0.0
        var chosen = n - 1
        for r in range(n):
            acc += dist[r]
            if acc >= u:
                chosen = r
                break
        for j in range(d):
            centers[c * d + j] = x[chosen * d + j]


def kmeans_lloyd(
    x: Ptr,
    centers: Ptr,
    labels: Ptr,
    sums: Ptr,
    counts: Ptr,
    n: Int,
    d: Int,
    k: Int,
    max_iter: Int,
    tol: Float64,
) -> Float64:
    """Run Lloyd's algorithm from the given centers. Returns inertia.

    `sums[k*d]` and `counts[k]` are scratch; `labels[n]` receives the
    assignment. An empty cluster keeps its previous center rather than being
    re-seeded — re-seeding mid-run makes the inertia non-monotonic and the
    convergence test meaningless.
    """
    var inertia = 0.0
    for _ in range(max_iter):
        inertia = 0.0
        for i in range(k * d):
            sums[i] = 0.0
        for c in range(k):
            counts[c] = 0.0
        for r in range(n):
            var best = euclidean2(x + r * d, centers, d)
            var bestc = 0
            for c in range(1, k):
                var cand = euclidean2(x + r * d, centers + c * d, d)
                if cand < best:
                    best = cand
                    bestc = c
            labels[r] = Float64(bestc)
            inertia += best
            counts[bestc] += 1.0
            for j in range(d):
                sums[bestc * d + j] += x[r * d + j]
        var shift = 0.0
        for c in range(k):
            if counts[c] == 0.0:
                continue
            for j in range(d):
                var updated = sums[c * d + j] / counts[c]
                var delta = updated - centers[c * d + j]
                shift += delta * delta
                centers[c * d + j] = updated
        if shift <= tol:
            break
    return inertia


def kmeans_assign(x: Ptr, centers: Ptr, labels: Ptr, n: Int, d: Int, k: Int) -> Float64:
    var inertia = 0.0
    for r in range(n):
        var best = euclidean2(x + r * d, centers, d)
        var bestc = 0
        for c in range(1, k):
            var cand = euclidean2(x + r * d, centers + c * d, d)
            if cand < best:
                best = cand
                bestc = c
        labels[r] = Float64(bestc)
        inertia += best
    return inertia
