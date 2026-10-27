"""Stage 2 trained on the degradation curve only.

The causal detector decides WHEN a RUL is worth estimating. This module builds
the regression problem for the stretch after that point, so that the model
never has to learn the healthy phase. For every snapshot (all causal):

  tau            time since the onset of the alarm currently active [s]
  h_lr, v_lr     log of the trailing RMS over the bearing's own healthy
                 baseline (median RMS over the detector's calibration window):
                 the same per-bearing self-calibration the detector relies on
  h_kurt, v_kurt trailing kurtosis

The training "curve" of a run-to-failure bearing is the alarm still active at
the end of its life, from the snapshot where it is raised (the detector
back-dates the onset by `persistence - 1` snapshots, which the API cannot
expose).

The fitting procedure (models, leave-one-bearing-out choice of the quantile)
lives here too, so that the evaluation (scripts/post_alarm_eval.py) and the
export of the served model (scripts/export_serving.py) run the same code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.causal_detector import CausalDetectorConfig, CausalDetectorResult, trailing_mean
from src.labeling import linear_rul

FEATURES = ["tau", "h_lr", "v_lr", "h_kurt", "v_kurt"]
SMOOTH = 5                       # trailing window for RMS / kurtosis [snapshots]
DT = config.SNAPSHOT_PERIOD

TRAIN_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]
Q_GRID = (0.1, 0.2, 0.3, 0.5)
MAX_LATE = 0.25                  # rule fixed in advance: at most 25 % late predictions


def curve_frame(features: pd.DataFrame, states: list[CausalDetectorResult],
                cfg: CausalDetectorConfig) -> pd.DataFrame:
    """Per-snapshot stage-2 features, alarm mask, curve mask and true RUL."""
    n = len(features)
    idx = np.arange(n)
    alarm = np.array([s.alarm_active for s in states])
    onset = np.array([s.alarm_onset if s.alarm_active else -1 for s in states])
    out = pd.DataFrame({"alarm": alarm, "tau": np.where(alarm, (idx - onset) * DT, np.nan)})
    cal0 = cfg.skip + cfg.w_smooth
    cal1 = cal0 + cfg.calib_len
    for ch in ("h", "v"):
        rms = features[f"{ch}_rms"].to_numpy()
        base = np.median(rms[cal0:cal1]) if n > cal1 else np.nan
        out[f"{ch}_lr"] = np.log(trailing_mean(rms, SMOOTH) / base)
        out[f"{ch}_kurt"] = trailing_mean(features[f"{ch}_kurtosis"].to_numpy(), SMOOTH)
    end = states[-1]
    final_onset = end.alarm_onset if end.alarm_active else n
    out["final_onset"] = final_onset
    out["curve"] = alarm & (idx >= final_onset)
    out["true_rul"] = linear_rul(n)
    return out


def training_rows(cf: pd.DataFrame, extend: int = 0) -> pd.DataFrame:
    """The curve, optionally extended by `extend` snapshots before the final onset
    (ablation). Before the onset tau is negative: time to the onset."""
    on = int(cf["final_onset"].iloc[0])
    idx = np.arange(len(cf))
    rows = cf[(idx >= on - extend) & ((idx < on) | cf["curve"].to_numpy())].copy()
    rows["tau"] = (rows.index.to_numpy() - on) * DT
    return rows


def stack(rows: list[pd.DataFrame]):
    """X, y and weights that make every bearing count once, whatever its curve length."""
    X = np.concatenate([r[FEATURES].to_numpy() for r in rows])
    y = np.concatenate([r["true_rul"].to_numpy() for r in rows])
    w = np.concatenate([np.full(len(r), 1.0 / len(r)) for r in rows])
    return X, y, w


def curve_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),                              # > 0: later than reality
        "late": float(np.mean(err > 0)),
        "mae_over_mean_rul": float(np.mean(np.abs(err)) / max(np.mean(true), DT)),
        "within_30pct": float(np.mean(np.abs(err) <= 0.3 * np.maximum(true, DT))),
        "n": int(len(true)),
    }


def fit_quantile_pipeline(rows: list[pd.DataFrame], q: float) -> Pipeline:
    """Linear quantile regression on standardised stage-2 features, every bearing
    weighing the same. This is the model the API serves (exported as JSON)."""
    X, y, w = stack(rows)
    m = make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=0.0, solver="highs"))
    m.fit(X, y, quantileregressor__sample_weight=w)
    return m


def fit(kind: str, rows: list[pd.DataFrame], q: float | None = None):
    """Returns predict(X) -> RUL [s], clipped at zero."""
    X, y, w = stack(rows)
    if kind == "countdown":
        D = float(np.median([r["true_rul"].iloc[(r["tau"] >= 0).argmax()] for r in rows]))
        return lambda Z: np.maximum(D - Z[:, 0], 0.0)
    if kind == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(X, y, ridge__sample_weight=w)
    elif kind == "quantile":
        m = fit_quantile_pipeline(rows, q)
    else:
        raise ValueError(kind)
    return lambda Z: np.maximum(m.predict(Z), 0.0)


def on_curve(predict, cf: pd.DataFrame) -> dict:
    c = cf[cf["curve"]]
    return curve_metrics(c["true_rul"].to_numpy(), predict(c[FEATURES].to_numpy()))


def choose_quantile(train: dict[str, pd.DataFrame]) -> tuple[float, list[dict]]:
    """Leave-one-training-bearing-out; rule: late <= MAX_LATE, then lowest MAE."""
    table = []
    for q in Q_GRID:
        folds = [on_curve(fit("quantile", [training_rows(cf) for k, cf in train.items() if k != b], q), train[b])
                 for b in train]
        table.append({"q": q, "lobo_mae": float(np.mean([f["mae"] for f in folds])),
                      "lobo_late": float(np.mean([f["late"] for f in folds]))})
    ok = [r for r in table if r["lobo_late"] <= MAX_LATE]
    best = min(ok, key=lambda r: r["lobo_mae"]) if ok else min(table, key=lambda r: r["lobo_late"])
    return best["q"], table
