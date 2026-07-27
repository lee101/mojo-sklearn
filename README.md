# mojo-sklearn

scikit-learn's core estimators, implemented in [Mojo](https://www.modular.com/mojo)
and callable from Python with the same API.

```python
import mojosklearn as msk

scaler = msk.StandardScaler().fit(X)
model = msk.Ridge(alpha=1.0).fit(scaler.transform(X), y)
print(model.score(scaler.transform(X_test), y_test))
```

`fit` / `transform` / `predict` / `score`, trailing-underscore attributes,
`train_test_split`, `KFold` — the names and contracts are scikit-learn's, so
changing the import is usually the whole migration. Every estimator in this
repo is tested against scikit-learn on the same data and has to agree with it.

## What is implemented

| module | contents |
| --- | --- |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `normalize` |
| `linear_model` | `LinearRegression`, `Ridge`, `SGDRegressor`, `LogisticRegression` |
| `cluster` | `KMeans` (k-means++ seeding, Lloyd, `n_init` restarts) |
| `decomposition` | `PCA` (covariance + Jacobi, sign-pinned components) |
| `neighbors` | `KNeighborsClassifier`, `KNeighborsRegressor`, `kneighbors` |
| `metrics` | `mean_squared_error`, `mean_absolute_error`, `r2_score`, `accuracy_score`, `pairwise_sqeuclidean`, `silhouette_score` |
| `model_selection` | `train_test_split`, `KFold` |

The Mojo side is also a library in its own right: `src/mojosklearn/` exposes
`dot`, `gram`, `gemm_tn`, `cholesky`, `cholesky_solve`, `jacobi_eigh` and the
estimator kernels as plain Mojo functions over raw buffers, with no Python
anywhere near them.

## Performance

Against scikit-learn 1.x on the same machine and the same arrays. scikit-learn
is mature Cython over multithreaded LAPACK; the honest result is that this wins
on memory-bound elementwise work and loses where BLAS is doing the heavy
lifting.

| case | mojo-sklearn | scikit-learn | |
| --- | ---: | ---: | --- |
| `StandardScaler.fit_transform` (200k x 20) | 36 ms | 154 ms | **4.2x faster** |
| `LinearRegression.fit` (50k x 60) | 91 ms | 122 ms | **1.4x faster** |
| `LinearRegression.predict` (200k x 20) | 6.3 ms | 15.4 ms | **2.4x faster** |
| `r2_score` (2M) | 7.9 ms | 27.2 ms | **3.4x faster** |
| `pairwise_sqeuclidean` (3k x 3k, 20d) | 138 ms | 151 ms | 1.1x faster |
| `KNN.predict` (5k train, 2k query, 20d) | 110 ms | 114 ms | 1.0x |
| `PCA.fit` k=5 (100k x 30) | 27 ms | 27 ms | 1.0x |
| `Ridge.fit` (200k x 20) | 147 ms | 73 ms | 0.5x slower |
| `KMeans.fit` k=8 (50k x 10) | 950 ms | 650 ms | 0.7x slower |

`pixi run bench` reproduces it.

The two losses are the same loss twice: scikit-learn reaches multithreaded BLAS
(`dsyrk` for the Gram matrix, `dgemm` inside k-means' `|a|^2 + |b|^2 - 2ab`
expansion) and this library is single-threaded scalar-plus-SIMD. Building the
Gram matrix as a vectorized rank-1 update per row rather than a dot product per
cell already took `Ridge.fit` from 845 ms to 147 ms and `LinearRegression.fit`
from 1357 ms to 91 ms; threading it is the next step, not a rewrite.

## Install

```bash
pixi install     # brings its own Mojo toolchain and builds nothing yet
pixi run test    # parity tests against scikit-learn
pixi run bench
```

The shared library builds itself on first import and rebuilds whenever a
`.mojo` file is newer than it. To force it:

```bash
python -m mojosklearn._lib --force
```

Outside pixi, point it at a compiler with `MOJOSKLEARN_MOJO=/path/to/mojo`.

## Design

```
python/mojosklearn/   estimator classes, numpy in and out, no algorithms
        │  ctypes, one call per operation, buffers passed by address
src/capi.mojo         @export ... abi("C") wrappers, Int addresses
src/mojosklearn/      the actual numerics, over raw pointers
```

Three rules hold throughout:

- **Nothing in Mojo allocates.** Every routine takes its scratch from the
  caller, so lifetimes are Python's problem and there is nothing to leak.
- **Contiguous float64 arrays are never copied.** They cross as an address and
  a length. A non-contiguous or non-float64 input is converted once, in numpy,
  where that is cheap.
- **Buffers cross as `Int` addresses.** `@export` refuses parametric functions,
  and a pointer with an inferred origin is parametric, so the address is
  rebuilt inside the wrapper as `UnsafePointer[Float64, AnyOrigin[mut=True]]`.

## Deliberate differences from scikit-learn

- `LogisticRegression` is full-batch gradient descent, not LBFGS, and binary
  only. It agrees with scikit-learn on decisions, not on coefficients.
- `Ridge` and `LinearRegression` always go through the normal equations. For a
  rank-deficient design the Cholesky fails and the fit falls back to
  `numpy.linalg.lstsq` rather than returning something wrong.
- `KMeans` leaves an empty cluster where it is instead of re-seeding it, which
  keeps inertia monotonic and the convergence test meaningful.
- `PCA` pins component signs (largest-magnitude loading positive) so a refit
  does not flip them.

## License

MIT
