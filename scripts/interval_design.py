"""Design study, training bearings only: 80% prediction intervals for stage 2.

Four variants, each evaluated by nested leave-one-bearing-out on the six training
curves (the conformal correction of a held-out bearing is computed on the other five
only, by their own leave-one-bearing-out scores):

    quantile pair   linear quantile regression at 0.1 and 0.9
    cqr             the same, widened by a conformal correction (conformalized quantile regression)
    both again on log(RUL + 10 s), i.e. multiplicative intervals

The test bearings are not read. Outputs in reports/intervals/: training.csv, summary.json

    python -m scripts.interval_design --cache
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.causal_detector import CausalDetectorConfig, stream_causal_detector
from src.pipeline import prepare_bearing
from src.post_alarm import FEATURES, TRAIN_BEARINGS, curve_frame, stack, training_rows

ART = config.PROJECT_ROOT / "artifacts"
OUT = config.PROJECT_ROOT / "reports" / "intervals"
LO, HI, ALPHA = 0.1, 0.9, 0.2
DT = float(config.SNAPSHOT_PERIOD)


def to_model(y: np.ndarray, log: bool) -> np.ndarray:
    return np.log(y + DT) if log else y            # RUL can be 0: shift by one snapshot


def from_model(z: np.ndarray, log: bool) -> np.ndarray:
    return np.maximum(np.exp(z) - DT, 0.0) if log else np.maximum(z, 0.0)


def fit(curves: dict, names: list[str], q: float, log: bool):
    X, y, w = stack([curves[n] for n in names])
    m = make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=0.0, solver="highs"))
    m.fit(X, to_model(y, log), quantileregressor__sample_weight=w)
    return m


def raw_interval(curves, names, cf, log):
    X = cf[FEATURES].to_numpy()
    a, b = (fit(curves, names, q, log).predict(X) for q in (LO, HI))
    return np.minimum(a, b), np.maximum(a, b)       # in model space


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(c, q)])


def conformal_correction(curves, names, log) -> float:
    """(1 - ALPHA) quantile of the leave-one-bearing-out conformity scores over `names`."""
    s, w = [], []
    for b in names:
        lo, hi = raw_interval(curves, [n for n in names if n != b], curves[b], log)
        z = to_model(curves[b]["true_rul"].to_numpy(), log)
        s.append(np.maximum(lo - z, z - hi))
        w.append(np.full(len(z), 1.0 / len(z)))
    return weighted_quantile(np.concatenate(s), np.concatenate(w), 1 - ALPHA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()
    cfg = CausalDetectorConfig.from_dict(json.loads((ART / "causal_detector.json").read_text()))
    curves = {}
    for n in TRAIN_BEARINGS:
        df, _ = prepare_bearing(config.DATA_ROOT / "Learning_set" / n, label_mode="linear", cache=args.cache)
        curves[n] = training_rows(curve_frame(df, stream_causal_detector(df[cfg.feature].to_numpy(), cfg), cfg))

    rows = []
    for log in (False, True):
        for b in TRAIN_BEARINGS:
            others = [n for n in TRAIN_BEARINGS if n != b]
            lo, hi = raw_interval(curves, others, curves[b], log)
            Q = conformal_correction(curves, others, log)
            y = curves[b]["true_rul"].to_numpy()
            for method, (low, high) in {"quantile_pair": (lo, hi), "cqr": (lo - Q, hi + Q)}.items():
                L, H = from_model(low, log), from_model(high, log)
                rows.append({"scale": "log" if log else "linear", "method": method, "held_out": b,
                             "curve_snapshots": len(y), "mean_true_rul_s": float(np.mean(y)),
                             "coverage": float(np.mean((y >= L) & (y <= H))),
                             "median_width_s": float(np.median(H - L)),
                             "median_width_over_true_rul": float(np.median((H - L) / np.maximum(y, DT)))})
    table = pd.DataFrame(rows)
    summary = (table.groupby(["scale", "method"])
               .agg(mean_coverage=("coverage", "mean"), worst_bearing_coverage=("coverage", "min"),
                    median_width_over_true_rul=("median_width_over_true_rul", "median"),
                    median_width_s=("median_width_s", "median"))
               .round(3))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "training.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(
        {"nominal_coverage": 1 - ALPHA, "quantiles": [LO, HI], "evaluation": "nested leave-one-bearing-out",
         "variants": {f"{s}/{m}": v for (s, m), v in summary.to_dict(orient="index").items()}}, indent=2) + "\n")
    print(summary.to_string())
    print("\ncoverage per held-out bearing:\n",
          table.pivot_table(index="held_out", columns=["scale", "method"], values="coverage").round(2).to_string())


if __name__ == "__main__":
    main()
