"""Splitting helpers. Pure numpy — there is nothing here worth compiling."""

from __future__ import annotations

import numpy as np


def train_test_split(*arrays, test_size: float = 0.25, random_state: int | None = None,
                     shuffle: bool = True):
    if not arrays:
        raise ValueError("at least one array is required")
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("all arrays must have the same length")
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(random_state).shuffle(idx)
    cut = n - int(round(n * test_size))
    train, test = idx[:cut], idx[cut:]
    out = []
    for a in arrays:
        arr = np.asarray(a)
        out.append(arr[train])
        out.append(arr[test])
    return out


class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int | None = None):
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(idx)
        for fold in np.array_split(idx, self.n_splits):
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            yield idx[mask[idx]], fold

    def get_n_splits(self, X=None, y=None) -> int:
        return self.n_splits
