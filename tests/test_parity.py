"""Every estimator is checked against scikit-learn on the same data."""

import numpy as np
import pytest

sk_linear = pytest.importorskip("sklearn.linear_model")
sk_cluster = pytest.importorskip("sklearn.cluster")
sk_decomp = pytest.importorskip("sklearn.decomposition")
sk_pre = pytest.importorskip("sklearn.preprocessing")
sk_neigh = pytest.importorskip("sklearn.neighbors")
sk_metrics = pytest.importorskip("sklearn.metrics")

import mojosklearn as msk


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    w = np.array([1.5, -2.0, 0.3, 0.0, 4.0, -1.0])
    y = X @ w + 0.5 + rng.normal(scale=0.1, size=400)
    return X, y


@pytest.fixture(scope="module")
def classification():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-1.0, 1.0, (200, 4)), rng.normal(1.0, 1.0, (200, 4))])
    y = np.r_[np.zeros(200), np.ones(200)]
    return X, y


def test_standard_scaler(regression):
    X, _ = regression
    ours = msk.StandardScaler().fit(X)
    theirs = sk_pre.StandardScaler().fit(X)
    assert np.allclose(ours.mean_, theirs.mean_)
    assert np.allclose(ours.scale_, theirs.scale_)
    assert np.allclose(ours.transform(X), theirs.transform(X))
    assert np.allclose(ours.inverse_transform(ours.transform(X)), X)


def test_minmax_scaler(regression):
    X, _ = regression
    ours = msk.MinMaxScaler().fit_transform(X)
    theirs = sk_pre.MinMaxScaler().fit_transform(X)
    assert np.allclose(ours, theirs)


def test_normalize(regression):
    X, _ = regression
    assert np.allclose(msk.normalize(X), sk_pre.normalize(X))


def test_linear_regression(regression):
    X, y = regression
    ours = msk.LinearRegression().fit(X, y)
    theirs = sk_linear.LinearRegression().fit(X, y)
    assert np.allclose(ours.coef_, theirs.coef_, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)
    assert np.allclose(ours.predict(X), theirs.predict(X), atol=1e-8)
    assert ours.score(X, y) == pytest.approx(theirs.score(X, y), abs=1e-10)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_ridge(regression, alpha):
    X, y = regression
    ours = msk.Ridge(alpha=alpha).fit(X, y)
    theirs = sk_linear.Ridge(alpha=alpha).fit(X, y)
    assert np.allclose(ours.coef_, theirs.coef_, atol=1e-8)
    assert ours.intercept_ == pytest.approx(theirs.intercept_, abs=1e-8)


def test_logistic_regression(classification):
    X, y = classification
    ours = msk.LogisticRegression(lr=0.5, epochs=2000).fit(X, y)
    theirs = sk_linear.LogisticRegression().fit(X, y)
    # gradient descent will not match LBFGS coefficient for coefficient;
    # agreement on decisions is the contract that matters
    assert ours.score(X, y) == pytest.approx(theirs.score(X, y), abs=0.02)
    assert np.mean(ours.predict(X) == theirs.predict(X)) > 0.97
    proba = ours.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_kmeans(classification):
    X, _ = classification
    ours = msk.KMeans(n_clusters=2, random_state=0).fit(X)
    theirs = sk_cluster.KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ == pytest.approx(theirs.inertia_, rel=1e-6)
    # labels are arbitrary up to permutation; the partition must be the same
    same = np.mean(ours.labels_ == theirs.labels_)
    assert max(same, 1.0 - same) > 0.99
    assert np.array_equal(ours.predict(X), ours.labels_)


def test_pca(regression):
    X, _ = regression
    ours = msk.PCA(n_components=3).fit(X)
    theirs = sk_decomp.PCA(n_components=3).fit(X)
    assert np.allclose(ours.explained_variance_, theirs.explained_variance_, rtol=1e-8)
    assert np.allclose(
        ours.explained_variance_ratio_, theirs.explained_variance_ratio_, rtol=1e-8
    )
    for a, b in zip(ours.components_, theirs.components_):
        assert np.allclose(a, b, atol=1e-8) or np.allclose(a, -b, atol=1e-8)
    Z = ours.transform(X)
    assert np.allclose(np.abs(Z), np.abs(theirs.transform(X)), atol=1e-8)
    # 3 of 6 components: reconstruction is lossy, so compare with sklearn's
    assert np.allclose(
        ours.inverse_transform(Z), theirs.inverse_transform(theirs.transform(X)),
        atol=1e-8,
    )
    full = msk.PCA(n_components=6).fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X, atol=1e-8)


def test_knn_classifier(classification):
    X, y = classification
    ours = msk.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    theirs = sk_neigh.KNeighborsClassifier(n_neighbors=5).fit(X, y)
    assert np.array_equal(ours.predict(X), theirs.predict(X))
    dist, idx = ours.kneighbors(X[:10])
    dist2, idx2 = theirs.kneighbors(X[:10])
    assert np.allclose(dist, dist2)
    assert np.array_equal(idx, idx2)


def test_knn_regressor(regression):
    X, y = regression
    ours = msk.KNeighborsRegressor(n_neighbors=7).fit(X, y)
    theirs = sk_neigh.KNeighborsRegressor(n_neighbors=7).fit(X, y)
    assert np.allclose(ours.predict(X), theirs.predict(X))


def test_metrics(regression):
    X, y = regression
    pred = msk.LinearRegression().fit(X, y).predict(X)
    assert msk.mean_squared_error(y, pred) == pytest.approx(
        sk_metrics.mean_squared_error(y, pred)
    )
    assert msk.mean_absolute_error(y, pred) == pytest.approx(
        sk_metrics.mean_absolute_error(y, pred)
    )
    assert msk.r2_score(y, pred) == pytest.approx(sk_metrics.r2_score(y, pred))
    labels = np.r_[np.zeros(200), np.ones(200)]
    assert msk.accuracy_score(labels, labels) == 1.0


def test_pairwise_and_silhouette(classification):
    X, y = classification
    ours = msk.pairwise_sqeuclidean(X[:50], X[50:100])
    theirs = sk_metrics.pairwise_distances(X[:50], X[50:100]) ** 2
    assert np.allclose(ours, theirs)
    assert msk.silhouette_score(X, y.astype(int)) == pytest.approx(
        sk_metrics.silhouette_score(X, y), abs=1e-6
    )


def test_train_test_split(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = msk.train_test_split(X, y, test_size=0.25, random_state=0)
    assert len(Xtr) == 300 and len(Xte) == 100
    assert len(ytr) == 300 and len(yte) == 100
    assert not np.intersect1d(Xtr[:, 0], Xte[:, 0]).size


def test_kfold(regression):
    X, y = regression
    folds = list(msk.KFold(n_splits=5).split(X))
    assert len(folds) == 5
    covered = np.concatenate([test for _, test in folds])
    assert np.array_equal(np.sort(covered), np.arange(len(X)))
