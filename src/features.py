"""Feature extraction.

Each raw snapshot is 2560 samples x 2 channels of vibration. We collapse it
into a handful of scalar health indicators per channel. These are the classic
condition-monitoring features: they are *interpretable* (you can explain to an
interviewer why kurtosis rises when a spall forms) and they turn a heavy raw
signal into a compact time series that both XGBoost and an RNN can chew on.

Design choice worth defending: we feed *the same* engineered features to both
the tree baseline and the sequence model. That keeps the comparison honest —
it isolates "does temporal modelling help?" from "does a richer input help?".
A raw-signal 1D-CNN is a legitimate alternative input (see README, extensions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from src import config
from src.data_loading import iter_snapshots

EPS = 1e-12


def _time_features(x: np.ndarray) -> dict[str, float]:
    absx = np.abs(x)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(absx)
    mean_abs = np.mean(absx) + EPS
    return {
        "rms": float(rms),
        "std": float(np.std(x)),
        "peak": float(peak),
        "peak2peak": float(np.max(x) - np.min(x)),
        "skewness": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),          # spikiness -> incipient faults
        "crest_factor": float(peak / (rms + EPS)),      # peak vs energy
        "shape_factor": float(rms / mean_abs),
        "impulse_factor": float(peak / mean_abs),
        "clearance_factor": float(peak / (np.mean(np.sqrt(absx)) ** 2 + EPS)),
    }


def _freq_features(x: np.ndarray, fs: int = config.FS) -> dict[str, float]:
    f, pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    pxx = pxx + EPS
    total = np.sum(pxx)
    centroid = np.sum(f * pxx) / total                  # spectral "centre of mass"
    # energy in four coarse bands — degradation shifts energy to higher bands
    bands = np.array_split(pxx, 4)
    band_energy = [float(np.sum(b) / total) for b in bands]
    return {
        "spec_centroid": float(centroid),
        "spec_kurtosis": float(stats.kurtosis(pxx)),
        "spec_peak_freq": float(f[np.argmax(pxx)]),
        "band1_energy": band_energy[0],
        "band2_energy": band_energy[1],
        "band3_energy": band_energy[2],
        "band4_energy": band_energy[3],
    }


def snapshot_features(snapshot: np.ndarray) -> dict[str, float]:
    """snapshot: (n_samples, 2). Returns flat dict with h_/v_ prefixes."""
    feats: dict[str, float] = {}
    for ch, name in enumerate(("h", "v")):
        x = snapshot[:, ch]
        for k, val in _time_features(x).items():
            feats[f"{name}_{k}"] = val
        for k, val in _freq_features(x).items():
            feats[f"{name}_{k}"] = val
    return feats


def build_feature_frame(bearing_dir, sep: str = ",") -> pd.DataFrame:
    """Extract a (n_snapshots x n_features) DataFrame for one bearing.

    Row i corresponds to the i-th 10 s snapshot, in temporal order. This frame
    is the shared substrate for labelling, the baseline and the sequence model.
    """
    rows = [snapshot_features(s) for s in iter_snapshots(bearing_dir, sep=sep)]
    df = pd.DataFrame(rows)
    df.insert(0, "snapshot_index", np.arange(len(df)))
    return df
