"""Phase 2 on XJTU-SY: a detector configuration chosen by a fixed rule on the 12 transfer
bearings, stage 2 refitted with the FEMTO recipe, both judged once on the vault.

Protocol: reports/xjtu_phase2/PREREGISTRATION.md. Commands: scripts/xjtu_phase2.py.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.causal_detector import CausalDetectorConfig, alarm_trace
from src.post_alarm import FEATURES, MAX_LATE, Q_GRID, fit_quantile_pipeline
from src.serving import LinearQuantileModel, stage2_matrix
from src.xjtu import SNAPSHOT_PERIOD, snapshots_to_monitor, translate_config

EARLY = 0.5                  # a first alarm before half of the life is early
STAGE2_SMOOTH = 5            # trailing window of the stage-2 features [snapshots]
HISTORY = ("h_rms", "v_rms", "h_kurtosis", "v_kurtosis")

GRID = [dict(persistence=p, w_smooth=w, w_slope=w, calib_len=c, revoke_after=r)
        for p, w, c, r in itertools.product((1, 2, 3, 5), (5, 10, 15), (25, 50), (5, 15))]


def base_config() -> CausalDetectorConfig:
    """The transfer-test configuration: skip 17, k = 6, horizontal RMS."""
    return translate_config(CausalDetectorConfig())


def config_for(params: dict) -> CausalDetectorConfig:
    return replace(base_config(), **params)


def key(params: dict) -> str:
    return "p{persistence}_w{w_smooth}_c{calib_len}_r{revoke_after}".format(**params)


# -- detector ------------------------------------------------------------------------------------

def outcome(rms: np.ndarray, cfg: CausalDetectorConfig) -> dict:
    """What the selection rule looks at, for one bearing and one configuration."""
    n = len(rms)
    alarm, onset, revoked = alarm_trace(rms, cfg)
    monitored = n >= snapshots_to_monitor(cfg)
    at_end = bool(alarm[-1])
    first = int(np.argmax(alarm)) if alarm.any() else None
    final = int(onset[-1]) if at_end else None
    return {
        "monitored": monitored, "alarm_at_end": at_end, "revoked": revoked,
        "first_alarm_pct": 100 * first / n if first is not None else None,
        "final_onset_pct": 100 * final / n if final is not None else None,
        "warning_min": (n - 1 - final) * SNAPSHOT_PERIOD / 60 if final is not None else None,
        "useful": bool(monitored and at_end and first is not None and first / n >= EARLY),
    }


def grid_table(features: pd.DataFrame) -> pd.DataFrame:
    """One row per (bearing, configuration)."""
    rows = []
    for b, g in features.groupby("bearing", sort=True):
        rms = g.sort_values("snapshot")["h_rms"].to_numpy()
        for params in GRID:
            rows.append({"bearing": b, "config": key(params)} | outcome(rms, config_for(params)))
    return pd.DataFrame(rows)


def rank(table: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    """Configurations ordered by the rule: useful alarms, then fewer revoked, then longer
    median warning, then the earliest start of monitoring."""
    t = table[table["bearing"].isin(bearings)].assign(warning_min=lambda d: pd.to_numeric(d["warning_min"]))
    agg = t.groupby("config").agg(useful=("useful", "sum"), revoked=("revoked", "sum"),
                                  warning=("warning_min", "median")).fillna({"warning": 0.0})
    start = {key(p): snapshots_to_monitor(config_for(p)) for p in GRID}
    agg["starts_at"] = [start[c] for c in agg.index]
    agg["name"] = agg.index
    return agg.sort_values(["useful", "revoked", "warning", "starts_at", "name"],
                           ascending=[False, True, False, True, True]).drop(columns="name")


def select(table: pd.DataFrame, bearings: list[str]) -> dict:
    best = rank(table, bearings).index[0]
    return next(p for p in GRID if key(p) == best)


def lobo_detector(table: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    """Each bearing scored with the configuration the rule picks on the others."""
    rows = []
    for b in bearings:
        params = select(table, [x for x in bearings if x != b])
        r = table[(table["bearing"] == b) & (table["config"] == key(params))].iloc[0]
        rows.append(r.to_dict())
    return pd.DataFrame(rows)


# -- stage 2 -------------------------------------------------------------------------------------

def curve(g: pd.DataFrame, cfg: CausalDetectorConfig, smooth: int = STAGE2_SMOOTH) -> pd.DataFrame | None:
    """Stage-2 features and true RUL on the curve (alarm active at end of life, from its onset),
    or None if the alarm is not active at the end."""
    g = g.sort_values("snapshot")
    n = len(g)
    alarm, onset, _ = alarm_trace(g[cfg.feature].to_numpy(), cfg)
    if not alarm[-1]:
        return None
    Z = stage2_matrix({c: g[c].to_numpy() for c in HISTORY}, onset, cfg, smooth, SNAPSHOT_PERIOD)
    idx = np.arange(n)
    mask = alarm & (idx >= onset[-1])
    out = pd.DataFrame(Z[mask], columns=FEATURES)
    out["true_rul"] = (n - 1 - idx[mask]).astype(float) * SNAPSHOT_PERIOD
    return out


def countdown_s(curves: dict[str, pd.DataFrame]) -> float:
    """Median curve length: the RUL at the onset of the final alarm."""
    return float(np.median([c["true_rul"].iloc[0] for c in curves.values()]))


def scores(true: np.ndarray, pred: np.ndarray, cd: np.ndarray) -> dict:
    mean_rul = max(float(np.mean(true)), SNAPSHOT_PERIOD)
    mae = float(np.mean(np.abs(pred - true)))
    cd_mae = float(np.mean(np.abs(cd - true)))
    return {"mae_min": mae / 60, "late": float(np.mean(pred > true)), "mae_over_rul": mae / mean_rul,
            "countdown_mae_min": cd_mae / 60, "beats_countdown": bool(mae < cd_mae), "mean_true_rul_min": mean_rul / 60}


def predict(model: Pipeline, c: pd.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(c[FEATURES].to_numpy()), 0.0)


def choose_quantile(curves: dict[str, pd.DataFrame]) -> tuple[float, list[dict]]:
    """Leave-one-bearing-out over the curves; rule: late <= MAX_LATE, then lowest MAE."""
    table = []
    for q in Q_GRID:
        folds = []
        for b, c in curves.items():
            m = fit_quantile_pipeline([x for k, x in curves.items() if k != b], q)
            p, t = predict(m, c), c["true_rul"].to_numpy()
            folds.append((float(np.mean(np.abs(p - t))), float(np.mean(p > t))))
        table.append({"q": q, "lobo_mae_s": float(np.mean([f[0] for f in folds])),
                      "lobo_late": float(np.mean([f[1] for f in folds]))})
    ok = [r for r in table if r["lobo_late"] <= MAX_LATE]
    best = min(ok, key=lambda r: r["lobo_mae_s"]) if ok else min(table, key=lambda r: r["lobo_late"])
    return best["q"], table


def lobo_stage2(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Nested: for each held-out curve, quantile and model chosen on the others only."""
    rows = []
    for b, c in curves.items():
        rest = {k: x for k, x in curves.items() if k != b}
        q, _ = choose_quantile(rest)
        m = fit_quantile_pipeline(list(rest.values()), q)
        t = c["true_rul"].to_numpy()
        cd = np.maximum(countdown_s(rest) - c["tau"].to_numpy(), 0.0)
        rows.append({"bearing": b, "q": q} | scores(t, predict(m, c), cd))
    return pd.DataFrame(rows)


def export_stage2(model: Pipeline, q: float, smooth: int = STAGE2_SMOOTH) -> dict:
    """The fitted pipeline as numbers, in the format the serving layer reads."""
    sc, qr = model.named_steps["standardscaler"], model.named_steps["quantileregressor"]
    return {"model": "StandardScaler + linear quantile regression (scikit-learn QuantileRegressor, alpha=0, highs)",
            "features": list(FEATURES), "quantile": q, "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(), "coef": qr.coef_.tolist(), "intercept": float(qr.intercept_),
            "smooth": smooth, "clip_min": 0.0}


def json_predict(d: dict, c: pd.DataFrame) -> np.ndarray:
    return LinearQuantileModel.from_dict(d).predict(c[FEATURES].to_numpy())
