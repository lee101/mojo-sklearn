"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.path.join(ROOT, "build", "capi.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "msk_dot": ([I, I, I], F),
    "msk_gram": ([I, I, I, I], None),
    "msk_gemm_tn": ([I, I, I, I, I, I], None),
    "msk_cholesky": ([I, I], I),
    "msk_cholesky_solve": ([I, I, I], None),
    "msk_standard_fit": ([I, I, I, I, I], None),
    "msk_standard_transform": ([I, I, I, I, I, I], None),
    "msk_standard_inverse": ([I, I, I, I, I, I], None),
    "msk_minmax_fit": ([I, I, I, I, I], None),
    "msk_minmax_transform": ([I, I, I, I, I, I, F, F], None),
    "msk_l2_normalize": ([I, I, I, I], None),
    "msk_ridge_fit": ([I, I, I, I, I, I, F], I),
    "msk_linear_predict": ([I, I, I, I, I, F], None),
    "msk_logistic_fit": ([I, I, I, I, I, I, F, F, I], F),
    "msk_logistic_predict_proba": ([I, I, I, I, I], None),
    "msk_sgd_ridge_fit": ([I, I, I, I, I, F, F, I], F),
    "msk_kmeans_plusplus": ([I, I, I, I, I, I, I], None),
    "msk_kmeans_lloyd": ([I, I, I, I, I, I, I, I, I, F], F),
    "msk_kmeans_assign": ([I, I, I, I, I, I], F),
    "msk_covariance": ([I, I, I, I, I, I], None),
    "msk_jacobi_eigh": ([I, I, I, I, F], I),
    "msk_pca_fit": ([I, I, I, I, I, I, I, I, I], F),
    "msk_pca_transform": ([I, I, I, I, I, I, I], None),
    "msk_pca_inverse": ([I, I, I, I, I, I, I], None),
    "msk_knn_query": ([I, I, I, I, I, I, I, I], None),
    "msk_knn_regress": ([I, I, I, I, I, I, I, I, I, I], None),
    "msk_knn_classify": ([I] * 12, None),
    "msk_mse": ([I, I, I], F),
    "msk_mae": ([I, I, I], F),
    "msk_r2": ([I, I, I], F),
    "msk_accuracy": ([I, I, I], F),
    "msk_pairwise_sqeuclidean": ([I, I, I, I, I, I], None),
    "msk_silhouette": ([I, I, I, I, I, I, I], F),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOSKLEARN_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOSKLEARN_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `build/capi.so` if it is missing or stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.array(a, dtype=np.float64, order="C", copy=True) if copy else \
        np.ascontiguousarray(a, dtype=np.float64)
    return arr


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojosklearn._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
