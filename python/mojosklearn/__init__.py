"""mojo-sklearn — scikit-learn's core estimators, implemented in Mojo.

The estimators keep scikit-learn's names and method contracts (`fit`,
`transform`, `predict`, `score`, trailing-underscore attributes), so swapping
the import is usually the whole migration. Arrays cross into Mojo by address:
a C-contiguous float64 numpy array is never copied.
"""

from . import cluster, decomposition, linear_model, metrics, model_selection, neighbors, preprocessing
from ._lib import build
from .cluster import KMeans
from .decomposition import PCA
from .linear_model import LinearRegression, LogisticRegression, Ridge, SGDRegressor
from .metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    pairwise_sqeuclidean,
    r2_score,
    silhouette_score,
)
from .model_selection import KFold, train_test_split
from .neighbors import KNeighborsClassifier, KNeighborsRegressor
from .preprocessing import MinMaxScaler, StandardScaler, normalize

__version__ = "0.1.0"
__all__ = [
    "KMeans", "PCA", "LinearRegression", "Ridge", "SGDRegressor",
    "LogisticRegression", "KNeighborsClassifier", "KNeighborsRegressor",
    "StandardScaler", "MinMaxScaler", "normalize",
    "train_test_split", "KFold",
    "mean_squared_error", "mean_absolute_error", "r2_score", "accuracy_score",
    "pairwise_sqeuclidean", "silhouette_score",
    "build",
    "cluster", "decomposition", "linear_model", "metrics", "model_selection",
    "neighbors", "preprocessing",
]
