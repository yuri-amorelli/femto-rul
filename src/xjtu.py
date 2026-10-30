"""XJTU-SY bearing dataset: layout, metadata and the translation of the detector to it.

Source: "Introduction to XJTU-SY Bearing Dataset" (shipped with the data) and
B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for Estimating Remaining
Useful Life of Rolling Element Bearings", IEEE Transactions on Reliability, 2020,
DOI 10.1109/TR.2018.2882682 (citation requested by the dataset authors).

    data/XJTU-SY/<...>/BearingX_Y/1.csv, 2.csv, ...

15 bearings (LDK UER204), 3 operating conditions, 5 bearings each. Every CSV is one
snapshot of 32768 samples (1.28 s at 25.6 kHz), taken every minute, so the number of
files is the life in minutes. First column: horizontal acceleration, second: vertical.
Each test ran until the maximum amplitude of the horizontal OR vertical signal exceeded
10 times its maximum in the normal stage, so the last file is the failure.

Used by reports/xjtu_transfer/PREREGISTRATION.md.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.causal_detector import CausalDetectorConfig

FS = 25_600
SNAPSHOT_LEN = 32_768
SNAPSHOT_PERIOD = 60          # seconds between two snapshots
FEMTO_PERIOD = 10             # the period the shipped detector was designed at

CONDITIONS = {1: {"rpm": 2100, "load_kN": 12.0}, 2: {"rpm": 2250, "load_kN": 11.0}, 3: {"rpm": 2400, "load_kN": 10.0}}

# Table 2 of the dataset introduction: number of files (= minutes of life) and fault element
BEARINGS = {
    "Bearing1_1": (123, "outer race"),
    "Bearing1_2": (161, "outer race"),
    "Bearing1_3": (158, "outer race"),
    "Bearing1_4": (122, "cage"),
    "Bearing1_5": (52, "inner race and outer race"),
    "Bearing2_1": (491, "inner race"),
    "Bearing2_2": (161, "outer race"),
    "Bearing2_3": (533, "cage"),
    "Bearing2_4": (42, "outer race"),
    "Bearing2_5": (339, "outer race"),
    "Bearing3_1": (2538, "outer race"),
    "Bearing3_2": (2496, "inner race, ball, cage and outer race"),
    "Bearing3_3": (371, "inner race"),
    "Bearing3_4": (1515, "inner race"),
    "Bearing3_5": (114, "outer race"),
}

VAULT_SEED = 20261030


def condition(bearing: str) -> int:
    return int(bearing[len("Bearing")])


# -- the detector at a different snapshot period ---------------------------------------------

WINDOWS = ("skip", "w_smooth", "w_slope", "calib_len", "persistence", "revoke_after")


def translate_config(cfg: CausalDetectorConfig, from_period: float = FEMTO_PERIOD,
                     to_period: float = SNAPSHOT_PERIOD) -> CausalDetectorConfig:
    """Same durations, fewer snapshots: every window becomes round(n * from/to), at least 1.
    k_sigma and the input feature do not change. Mechanical, no choice involved."""
    f = from_period / to_period
    return replace(cfg, **{k: max(1, int(round(getattr(cfg, k) * f))) for k in WINDOWS})


def snapshots_to_monitor(cfg: CausalDetectorConfig) -> int:
    """Shortest recording, in snapshots, that leaves 'calibrating' (see run_causal_detector)."""
    return cfg.skip + cfg.w_smooth + cfg.calib_len + cfg.persistence + 1


# -- the vault: one bearing per condition, kept unseen for a later final test ------------------

def draw_vault(seed: int = VAULT_SEED, min_snapshots: int | None = None) -> list[str]:
    """One bearing per condition, drawn at random among those long enough to be monitored
    under the primary (time-translated) detector."""
    if min_snapshots is None:
        min_snapshots = snapshots_to_monitor(translate_config(CausalDetectorConfig()))
    rng = np.random.default_rng(seed)
    vault = []
    for c in sorted(CONDITIONS):
        pool = sorted(b for b, (n, _) in BEARINGS.items() if condition(b) == c and n >= min_snapshots)
        vault.append(pool[int(rng.integers(len(pool)))])
    return vault


VAULT = draw_vault()
TRANSFER = [b for b in BEARINGS if b not in VAULT]


# -- reading the files --------------------------------------------------------------------------

def find_bearings(root: Path) -> dict[str, Path]:
    """BearingX_Y -> folder, wherever it sits below `root` (the zip nests them by condition)."""
    found = {p.name: p for p in Path(root).rglob("Bearing*_*") if p.is_dir()}
    unknown = set(found) - set(BEARINGS)
    if unknown:
        raise ValueError(f"unexpected folders: {sorted(unknown)}")
    return dict(sorted(found.items()))


def snapshot_files(bearing_dir: Path) -> list[Path]:
    """1.csv, 2.csv, ... in numeric order."""
    files = [p for p in Path(bearing_dir).glob("*.csv") if re.fullmatch(r"\d+", p.stem)]
    return sorted(files, key=lambda p: int(p.stem))


def load_snapshot(path: Path) -> np.ndarray:
    """One CSV -> (n_samples, 2) = [horizontal, vertical]. Skips a header row if there is one."""
    with open(path) as fh:
        first = fh.readline()
    sep = ";" if ";" in first else ","
    try:
        float(first.split(sep)[0])
        header = None
    except ValueError:
        header = 0
    df = pd.read_csv(path, header=header, sep=sep)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: {df.shape[1]} column(s), expected 2")
    return df.iloc[:, :2].to_numpy(dtype=float)
