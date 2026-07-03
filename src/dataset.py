"""Turning per-bearing feature time series into sequences for an RNN.

Two rules that, if broken, produce a great-looking-but-fake result:

1. Windows must never straddle two bearings. Each sequence is built inside a
   single bearing's trajectory. We therefore build per bearing, then concat.

2. The scaler is fit on TRAINING bearings only and applied to everyone. Fitting
   on the whole dataset leaks the test distribution into training.

A window of the last W feature vectors predicts the RUL at the window's final
timestep — i.e. "given the recent history of health indicators, how much life
is left right now".
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def make_sequences(features: np.ndarray, target: np.ndarray,
                   window: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """features: (T, F), target: (T,). Returns X:(N, window, F), y:(N,)."""
    X, y = [], []
    for end in range(window, len(features) + 1, stride):
        X.append(features[end - window:end])
        y.append(target[end - 1])            # RUL at the last step of the window
    if not X:
        return np.empty((0, window, features.shape[1]), np.float32), np.empty((0,), np.float32)
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


class SequenceDataset(Dataset):
    """Concatenated windows from many bearings.

    Pass a list of (features, target) pairs, one per bearing, already scaled.
    """

    def __init__(self, per_bearing: list[tuple[np.ndarray, np.ndarray]],
                 window: int, stride: int = 1):
        xs, ys = [], []
        for feats, tgt in per_bearing:
            X, y = make_sequences(feats, tgt, window, stride)
            if len(X):
                xs.append(X)
                ys.append(y)
        self.X = torch.from_numpy(np.concatenate(xs)) if xs else torch.empty(0)
        self.y = torch.from_numpy(np.concatenate(ys)) if ys else torch.empty(0)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]
