"""Dense float64 primitives over raw row-major buffers.

Everything here takes plain pointers and extents so the same code serves the
Mojo API and the C ABI the Python bindings call through. Nothing allocates
unless it says so.
"""

from std.math import sqrt

comptime W = 4
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var t = acc.reduce_add()
    while i < n:
        t += a[i] * b[i]
        i += 1
    return t


def sum(a: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i)
        i += W
    var t = acc.reduce_add()
    while i < n:
        t += a[i]
        i += 1
    return t


def axpy(alpha: Float64, x: Ptr, y: Ptr, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.store(i, y.load[width=W](i) + va * x.load[width=W](i))
        i += W
    while i < n:
        y[i] += alpha * x[i]
        i += 1


def scale(x: Ptr, alpha: Float64, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        x.store(i, x.load[width=W](i) * va)
        i += W
    while i < n:
        x[i] *= alpha
        i += 1


def fill(x: Ptr, value: Float64, n: Int):
    for i in range(n):
        x[i] = value


def copy(src: Ptr, dst: Ptr, n: Int):
    for i in range(n):
        dst[i] = src[i]


def matvec(a: Ptr, x: Ptr, dst: Ptr, rows: Int, cols: Int):
    """dst[rows] = A[rows, cols] @ x[cols]."""
    for r in range(rows):
        dst[r] = dot(a + r * cols, x, cols)


def gram(x: Ptr, dst: Ptr, n: Int, d: Int):
    """dst[d, d] = X.T @ X for row-major X[n, d].

    Written as a rank-1 update per row rather than a dot product per output
    cell: the row is walked sequentially and `axpy` vectorizes it, which is
    roughly 8x the naive triple loop. Only the lower triangle is accumulated
    and then mirrored.
    """
    for i in range(d * d):
        dst[i] = 0.0
    for r in range(n):
        var row = x + r * d
        for i in range(d):
            var a = row[i]
            if a != 0.0:
                axpy(a, row, dst + i * d, i + 1)
    for i in range(d):
        for j in range(i + 1, d):
            dst[i * d + j] = dst[j * d + i]


def gemm_tn(a: Ptr, b: Ptr, dst: Ptr, n: Int, d: Int, m: Int):
    """dst[d, m] = A[n, d].T @ B[n, m], both row-major."""
    for i in range(d * m):
        dst[i] = 0.0
    for r in range(n):
        for i in range(d):
            var aval = a[r * d + i]
            if aval != 0.0:
                axpy(aval, b + r * m, dst + i * m, m)


def cholesky(a: Ptr, d: Int) -> Bool:
    """In-place lower Cholesky of a symmetric positive definite A[d, d].

    Returns False if the matrix is not positive definite, which is the caller's
    signal to add ridge or fall back — it is not an error worth raising."""
    for i in range(d):
        for j in range(i + 1):
            var acc = a[i * d + j]
            for k in range(j):
                acc -= a[i * d + k] * a[j * d + k]
            if i == j:
                if acc <= 0.0:
                    return False
                a[i * d + i] = sqrt(acc)
            else:
                a[i * d + j] = acc / a[j * d + j]
    for i in range(d):
        for j in range(i + 1, d):
            a[i * d + j] = 0.0
    return True


def cholesky_solve(l: Ptr, b: Ptr, d: Int):
    """Solve L L.T x = b in place, given the lower factor from `cholesky`."""
    for i in range(d):
        var acc = b[i]
        for k in range(i):
            acc -= l[i * d + k] * b[k]
        b[i] = acc / l[i * d + i]
    for ri in range(d):
        var i = d - 1 - ri
        var acc = b[i]
        for k in range(i + 1, d):
            acc -= l[k * d + i] * b[k]
        b[i] = acc / l[i * d + i]


def euclidean2(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var diff = a.load[width=W](i) - b.load[width=W](i)
        acc += diff * diff
        i += W
    var t = acc.reduce_add()
    while i < n:
        var d = a[i] - b[i]
        t += d * d
        i += 1
    return t
