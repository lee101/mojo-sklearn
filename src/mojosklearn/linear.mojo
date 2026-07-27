"""Linear models solved in closed form or by gradient descent.

Ridge and ordinary least squares both go through the normal equations with a
Cholesky factor: for the tall-and-thin problems these libraries are used on
(n >> d) that is the fastest correct route, and the O(d^3) factor is noise.
"""

from std.math import exp, log, sqrt

from mojosklearn.linalg import Ptr, cholesky, cholesky_solve, dot, gram


def ridge_fit(
    x: Ptr, y: Ptr, coef: Ptr, gram_work: Ptr, n: Int, d: Int, alpha: Float64
) -> Bool:
    """Solve (X.T X + alpha I) w = X.T y for w[d]. X is row-major [n, d].

    `gram_work` is caller-provided scratch of d*d. Pass alpha=0 for OLS. The
    caller centers the data if it wants an intercept; that keeps this routine
    to one job.
    """
    gram(x, gram_work, n, d)
    for i in range(d):
        gram_work[i * d + i] += alpha
    for i in range(d):
        var acc = 0.0
        for r in range(n):
            acc += x[r * d + i] * y[r]
        coef[i] = acc
    if not cholesky(gram_work, d):
        return False
    cholesky_solve(gram_work, coef, d)
    return True


def linear_predict(x: Ptr, coef: Ptr, dst: Ptr, n: Int, d: Int, intercept: Float64):
    for r in range(n):
        dst[r] = dot(x + r * d, coef, d) + intercept


def logistic_fit(
    x: Ptr,
    y: Ptr,
    coef: Ptr,
    grad: Ptr,
    n: Int,
    d: Int,
    lr: Float64,
    l2: Float64,
    epochs: Int,
) -> Float64:
    """Binary logistic regression by full-batch gradient descent.

    `y` is 0/1, `coef` is [d+1] with the intercept last, `grad` is scratch of
    the same length. Returns the final mean log loss.
    """
    var loss = 0.0
    for _ in range(epochs):
        for i in range(d + 1):
            grad[i] = 0.0
        loss = 0.0
        for r in range(n):
            var z = dot(x + r * d, coef, d) + coef[d]
            var p = 1.0 / (1.0 + exp(-z))
            var err = p - y[r]
            for j in range(d):
                grad[j] += err * x[r * d + j]
            grad[d] += err
            var eps = 1e-15
            loss -= y[r] * log(p + eps) + (1.0 - y[r]) * log(1.0 - p + eps)
        var inv = 1.0 / Float64(n)
        for j in range(d):
            coef[j] -= lr * (grad[j] * inv + l2 * coef[j])
        coef[d] -= lr * grad[d] * inv
        loss *= inv
    return loss


def logistic_predict_proba(x: Ptr, coef: Ptr, dst: Ptr, n: Int, d: Int):
    for r in range(n):
        var z = dot(x + r * d, coef, d) + coef[d]
        dst[r] = 1.0 / (1.0 + exp(-z))


def sgd_ridge_fit(
    x: Ptr,
    y: Ptr,
    coef: Ptr,
    n: Int,
    d: Int,
    lr: Float64,
    l2: Float64,
    epochs: Int,
) -> Float64:
    """Ridge by SGD, for when d is large enough that the Gram matrix is not.

    Deterministic order — reproducibility beats the marginal benefit of
    shuffling here, and the caller can shuffle its own rows.
    """
    var mse = 0.0
    for _ in range(epochs):
        mse = 0.0
        for r in range(n):
            var pred = dot(x + r * d, coef, d) + coef[d]
            var err = pred - y[r]
            mse += err * err
            for j in range(d):
                coef[j] -= lr * (err * x[r * d + j] + l2 * coef[j])
            coef[d] -= lr * err
        mse /= Float64(n)
    return mse
