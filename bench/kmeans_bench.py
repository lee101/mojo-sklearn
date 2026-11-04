"""KMeans vs scikit-learn: best-of-N wall time (both warmed up), labels equality. usage: python bench/kmeans_bench.py"""
import sys, time, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojosklearn as msk
from sklearn.cluster import KMeans

rng = np.random.default_rng(0)
def blobs(n, d, k, spread):
    return rng.normal(size=(k, d))[rng.integers(0, k, n)] * 3 + rng.normal(size=(n, d)) * spread
CASES = [("3.7k x 120, k=6, n_init=4", blobs(3699, 120, 6, 3.0), 6, 4),
         ("20k x 16, k=12, n_init=3", blobs(20000, 16, 12, 1.0), 12, 3),
         ("8k x 20, k=10, n_init=4 (63 iters)", rng.normal(size=(8000, 20)) + rng.integers(0, 10, (8000, 1)) * 0.35, 10, 4),
         ("300k x 12, k=6, n_init=2", rng.normal(size=(300000, 12)) + rng.integers(0, 6, (300000, 1)) * 0.5, 6, 2)]
def best(f, reps=5):
    f(); t = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return min(t) * 1e3
print(f"{'case':38s} {'mojo ms':>9s} {'sklearn ms':>11s} {'speedup':>8s} labels")
for name, X, k, ni in CASES:
    m = msk.KMeans(k, n_init=ni, random_state=0); s = KMeans(k, n_init=ni, random_state=0)
    tm = best(lambda: m.fit(X)); ts = best(lambda: s.fit(X))
    print(f"{name:38s} {tm:9.1f} {ts:11.1f} {ts/tm:7.1f}x {'equal' if np.array_equal(m.labels_, s.labels_) else 'DIFFER'}")
