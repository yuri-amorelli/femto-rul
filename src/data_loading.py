"""Loading raw FEMTO-ST vibration files.

FEMTO layout (verify against what you downloaded — a couple of repackaged
versions exist and column counts differ slightly):

    <set>/BearingX_Y/acc_00001.csv
                     acc_00002.csv
                     ...
                     temp_00001.csv   (temperature, we ignore it by default)

Each acc_*.csv is ONE snapshot: 2560 rows recorded over 0.1 s, taken every
10 s. Columns are usually one of:
    [hour, minute, second, microsec, h_accel, v_accel]   (6 cols)
    [second, microsec, h_accel, v_accel]                 (4 cols)
In every known variant the LAST TWO columns are the horizontal and vertical
accelerations, so we slice `iloc[:, -2:]` and stay agnostic to the timestamp
columns.

The separator is detected per file (sep=None, the default): the official
distribution is not consistent, e.g. Full_Test_Set/Bearing1_4 uses ';' while
the same bearing in Test_set uses ','. Pass sep explicitly to force one.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def list_bearings(set_dir: Path) -> list[Path]:
    """Return sorted Bearing*_* subdirectories inside a set folder."""
    dirs = [p for p in Path(set_dir).iterdir() if p.is_dir() and p.name.startswith("Bearing")]
    return sorted(dirs, key=lambda p: p.name)


def _acc_files(bearing_dir: Path) -> list[Path]:
    files = list(Path(bearing_dir).glob("acc_*.csv"))
    # sort by the numeric index in the filename, not lexicographically
    def idx(p: Path) -> int:
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else -1
    return sorted(files, key=idx)


def detect_sep(path: Path) -> str:
    """';' if the first line contains one, else ','."""
    with open(path) as fh:
        first = fh.readline()
    return ";" if ";" in first else ","


def load_snapshot(path: Path, sep: str | None = None) -> np.ndarray:
    """Read one acc_*.csv -> array of shape (n_samples, 2) = [h_accel, v_accel]."""
    df = pd.read_csv(path, header=None, sep=sep or detect_sep(path))
    if df.shape[1] < 4:
        raise ValueError(f"{path}: {df.shape[1]} column(s) parsed, expected 4 or 6 "
                         f"(wrong separator?)")
    return df.iloc[:, -2:].to_numpy(dtype=np.float32)


def iter_snapshots(bearing_dir: Path, sep: str | None = None) -> Iterator[np.ndarray]:
    """Yield snapshots (h, v) in temporal order for one bearing.

    We stream instead of loading everything: a full run-to-failure bearing is
    thousands of 2560x2 snapshots, and we only ever need one at a time to
    extract features.
    """
    for f in _acc_files(bearing_dir):
        yield load_snapshot(f, sep=sep)


def n_snapshots(bearing_dir: Path) -> int:
    return len(_acc_files(bearing_dir))
