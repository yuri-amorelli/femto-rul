"""Evaluation protocol for the 11 IEEE PHM 2012 test bearings.

Test_set/ holds the official test bearings TRUNCATED before failure: the
challenge asks for the RUL at the last available snapshot. Labelling that
truncated trajectory as if it ended in failure (RUL = 0 at its last snapshot)
is wrong, and it is what this project's first test-set table did: every target
is shifted down by the actual RUL (5730 s on Bearing1_3, 7530 s on
Bearing2_3).

Ground truth comes from Full_Test_Set/, the same bearings recorded to the end.
The Test_set files are an exact prefix of the Full_Test_Set files (checked
file by file on all 11 bearings), and the official actual RULs equal the time
from the truncation point to the end of the recording for 10 of the 11
bearings. The exception is Bearing1_4: the official table says 339 s, the
recording continues for 2890 s. The score is reported both ways.

Two measurements per bearing:
  - challenge protocol: the prediction at the truncation point (snapshot
    n_truncated - 1) against the official actual RUL -> %Er and A_i; the
    challenge score is the mean A_i over the 11 bearings;
  - full trajectory: RMSE / MAE over the complete run against the label scheme
    the model was trained with, computed on the complete run exactly as for
    the training bearings.

Features are computed per snapshot and the models only look backwards
(XGBoost: one snapshot, LSTM: a trailing window), so predicting over the full
run and reading the value at the truncation index is the same as predicting on
Test_set alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.data_loading import n_snapshots
from src.evaluate import phm_score, summarize
from src.labeling import linear_rul
from src.pipeline import prepare_bearing

# Table 3 "Actual RULs to be estimated", IEEE PHM 2012 challenge details [s]
OFFICIAL_ACTUAL_RUL = {
    "Bearing1_3": 5730, "Bearing1_4": 339, "Bearing1_5": 1610,
    "Bearing1_6": 1460, "Bearing1_7": 7570, "Bearing2_3": 7530,
    "Bearing2_4": 1390, "Bearing2_5": 3090, "Bearing2_6": 1290,
    "Bearing2_7": 580, "Bearing3_3": 820,
}
TEST_BEARINGS = list(OFFICIAL_ACTUAL_RUL)

# per-bearing metrics worth logging to MLflow (the rest is bookkeeping)
LOGGED_METRICS = ("rmse", "mae", "pred_at_truncation", "er_pct", "phm_a", "phm_a_recorded")


@dataclass
class TestBearing:
    name: str
    features: pd.DataFrame   # full run, one row per snapshot
    target: np.ndarray       # label scheme computed on the full run
    n_truncated: int         # length of the official Test_set trajectory

    @property
    def n_full(self) -> int:
        return len(self.features)

    @property
    def truncation_index(self) -> int:
        return self.n_truncated - 1

    @property
    def true_rul(self) -> np.ndarray:
        """Seconds to the end of the recording, for every snapshot of the run."""
        return linear_rul(self.n_full)


def load_test_bearing(name: str, label_mode: str, truncated_set: str = "Test_set",
                      full_set: str = "Full_Test_Set", sep: str | None = None,
                      cache: bool = False) -> TestBearing:
    df, target = prepare_bearing(config.DATA_ROOT / full_set / name,
                                 label_mode=label_mode, sep=sep, cache=cache)
    n_tr = n_snapshots(config.DATA_ROOT / truncated_set / name)
    if not 0 < n_tr < len(df):
        raise ValueError(f"{name}: {truncated_set} has {n_tr} snapshots, {full_set} "
                         f"{len(df)}; expected a truncated prefix of the full run")
    return TestBearing(name, df, target, n_tr)


def percent_error(actual: float, predicted: float) -> float:
    """%Er of the challenge: > 0 early (safe), <= 0 late (penalised harder)."""
    return 100.0 * (actual - predicted) / actual


def evaluate_bearing(tb: TestBearing, pred: np.ndarray, offset: int) -> dict:
    """Metrics for one test bearing.

    pred[k] is the prediction at snapshot offset + k of the full run:
    offset = 0 for per-snapshot models, window - 1 for the sequence model.
    """
    pred = np.asarray(pred, dtype=float)
    if len(pred) != tb.n_full - offset:
        raise ValueError(f"{tb.name}: {len(pred)} predictions, expected {tb.n_full - offset}")
    i = tb.truncation_index - offset
    if i < 0:
        raise ValueError(f"{tb.name}: truncated run shorter than the model window")

    p = float(pred[i])
    actual = float(OFFICIAL_ACTUAL_RUL[tb.name])
    recorded = float(tb.true_rul[tb.truncation_index])
    m = summarize(tb.target[offset:], pred)          # full trajectory vs label scheme
    m.update({
        "n_truncated": tb.n_truncated,
        "n_full": tb.n_full,
        "pred_at_truncation": p,
        "label_at_truncation": float(tb.target[tb.truncation_index]),
        "actual_rul": actual,
        "recorded_rul": recorded,
        "er_pct": percent_error(actual, p),
        "phm_a": phm_score(np.array([actual]), np.array([p])),
        "phm_a_recorded": phm_score(np.array([recorded]), np.array([p])),
    })
    return m


def challenge_summary(results: dict[str, dict]) -> dict[str, float]:
    """Challenge score (mean A_i) with the official table and with the recorded RUL."""
    r = list(results.values())
    return {
        "phm_score": float(np.mean([m["phm_a"] for m in r])),
        "phm_score_recorded": float(np.mean([m["phm_a_recorded"] for m in r])),
        "mean_abs_er_pct": float(np.mean([abs(m["er_pct"]) for m in r])),
        "mean_rmse": float(np.mean([m["rmse"] for m in r])),
        "n_bearings": len(r),
    }


def format_row(name: str, m: dict) -> str:
    return (f"{name}: RMSE={m['rmse']:7.1f}s  MAE={m['mae']:7.1f}s | at truncation: "
            f"pred={m['pred_at_truncation']:6.0f}s  actual={m['actual_rul']:5.0f}s  "
            f"Er={m['er_pct']:7.1f}%  A={m['phm_a']:.3f}")
