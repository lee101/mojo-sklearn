"""Linear and logistic regression."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class _LinearBase:
    def __init__(self, alpha: float = 0.0, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X, y):
        X = f64(X)
        y = f64(y).ravel()
        n, d = X.shape
        if self.fit_intercept:
            # Centering is how the intercept stays out of the penalty: ridge
            # must not shrink it, and the closed form has no other way to say so.
            xmean = X.mean(axis=0)
            ymean = float(y.mean())
            Xc = np.ascontiguousarray(X - xmean)
            yc = np.ascontiguousarray(y - ymean)
        else:
            xmean, ymean = np.zeros(d), 0.0
            Xc, yc = X, y

        coef = np.empty(d)
        work = np.empty((d, d))
        ok = lib().msk_ridge_fit(
            addr(Xc), addr(yc), addr(coef), addr(work), n, d, self.alpha
        )
        if not ok:
            # Singular Gram matrix: fall back to the least-norm solution rather
            # than failing, which is what a rank-deficient design deserves.
            coef = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        self.coef_ = coef
        self.intercept_ = float(ymean - xmean @ coef) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = f64(X)
        n, d = X.shape
        out = np.empty(n)
        lib().msk_linear_predict(
            addr(X), addr(self.coef_), addr(out), n, d, float(self.intercept_)
        )
        return out

    def score(self, X, y):
        from .metrics import r2_score

        return r2_score(y, self.predict(X))


class LinearRegression(_LinearBase):
    """Ordinary least squares through the normal equations."""

    def __init__(self, fit_intercept: bool = True):
        super().__init__(alpha=0.0, fit_intercept=fit_intercept)


class Ridge(_LinearBase):
    """L2-penalized least squares."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept)


class SGDRegressor:
    """Ridge by stochastic gradient descent, for d too large to form a Gram."""

    def __init__(self, lr: float = 0.01, alpha: float = 0.0, epochs: int = 100):
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.epochs = int(epochs)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_: float = float("nan")

    def fit(self, X, y):
        X = f64(X)
        y = f64(y).ravel()
        n, d = X.shape
        theta = np.zeros(d + 1)
        self.loss_ = lib().msk_sgd_ridge_fit(
            addr(X), addr(y), addr(theta), n, d, self.lr, self.alpha, self.epochs
        )
        self.coef_ = theta[:d].copy()
        self.intercept_ = float(theta[d])
        return self

    def predict(self, X):
        X = f64(X)
        n, d = X.shape
        out = np.empty(n)
        lib().msk_linear_predict(
            addr(X), addr(self.coef_), addr(out), n, d, float(self.intercept_)
        )
        return out


class LogisticRegression:
    """Binary logistic regression by full-batch gradient descent."""

    def __init__(self, lr: float = 0.1, l2: float = 0.0, epochs: int = 500):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_: float = float("nan")
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        X = f64(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression here is binary only")
        yb = f64((y == self.classes_[1]).astype(np.float64))
        n, d = X.shape
        theta = np.zeros(d + 1)
        grad = np.zeros(d + 1)
        self.loss_ = lib().msk_logistic_fit(
            addr(X), addr(yb), addr(theta), addr(grad),
            n, d, self.lr, self.l2, self.epochs,
        )
        self.coef_ = theta[:d].copy()
        self.intercept_ = float(theta[d])
        self._theta = theta
        return self

    def predict_proba(self, X):
        X = f64(X)
        n, d = X.shape
        p = np.empty(n)
        lib().msk_logistic_predict_proba(addr(X), addr(self._theta), addr(p), n, d)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return self.classes_[(p >= 0.5).astype(int)]

    def score(self, X, y):
        from .metrics import accuracy_score

        return accuracy_score(y, self.predict(X))
