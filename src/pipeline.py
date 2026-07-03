"""Shared data preparation used by both the XGBoost and the RNN scripts.

Responsibilities:
  - turn each bearing folder into (feature_matrix, rul_target), with caching so
    the expensive feature extraction runs once;
  - fit a StandardScaler PER OPERATING CONDITION on training bearings only, and
    apply it to everyone (no test leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import build_feature_frame
from src.labeling import make_targets

FEATURE_COLS_CACHE = config.RESULTS_DIR / "feature_columns.txt"


def prepare_bearing(bearing_dir: Path, label_mode: str = "piecewise",
                    sep: str = ",", cache: bool = False):
    """Return (feature_df, target_array) for one bearing, with on-disk caching."""
    name = Path(bearing_dir).name
    cache_path = config.RESULTS_DIR / f"features_{name}.parquet"
    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = build_feature_frame(bearing_dir, sep=sep)
        if cache:
            df.to_parquet(cache_path)
    target = make_targets(df, mode=label_mode)
    return df, target


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "snapshot_index"]


def fit_scalers(train_data: dict[str, tuple[pd.DataFrame, np.ndarray]]):
    """One StandardScaler per condition, fit on training bearings only."""
    scalers: dict[int, StandardScaler] = {}
    by_cond: dict[int, list[np.ndarray]] = {}
    cols = None
    for name, (df, _) in train_data.items():
        cols = feature_columns(df)
        by_cond.setdefault(config.condition_of(name), []).append(df[cols].to_numpy())
    for cond, mats in by_cond.items():
        scalers[cond] = StandardScaler().fit(np.concatenate(mats))
    return scalers, cols


def apply_scaler(name: str, df: pd.DataFrame, scalers, cols) -> np.ndarray:
    cond = config.condition_of(name)
    scaler = scalers.get(cond)
    mat = df[cols].to_numpy()
    return scaler.transform(mat) if scaler is not None else mat
