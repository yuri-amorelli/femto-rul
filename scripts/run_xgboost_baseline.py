"""XGBoost baseline for RUL estimation.

This is the reference the RNN must beat to justify itself. XGBoost sees each
snapshot's feature vector independently (no explicit time context beyond what
the features encode). Run it first; its metrics anchor the whole project.

Usage:
    python -m scripts.run_xgboost_baseline --label-mode capped \
        --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1

Test bearings default to all 11 of the challenge, evaluated with the protocol
in src/test_protocol.py: ground truth from Full_Test_Set, challenge score at
the Test_set truncation point.

Every run is tracked with MLflow (experiment "femto-rul-comparison" by
default), next to the LSTM runs, so the two models can be compared in the UI.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
from xgboost import XGBRegressor

from src import config, tracking
from src.pipeline import apply_scaler, fit_scalers, prepare_bearing
from src.test_protocol import (LOGGED_METRICS, TEST_BEARINGS, challenge_summary,
                               evaluate_bearing, format_row, load_test_bearing)

XGB_PARAMS = dict(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
)


def load(names, data_set, label_mode, sep, cache):
    root = config.DATA_ROOT / data_set
    return {n: prepare_bearing(root / n, label_mode=label_mode, sep=sep, cache=cache) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", default=TEST_BEARINGS)
    ap.add_argument("--data-set", default="Learning_set")
    ap.add_argument("--test-set", default="Test_set",
                    help="truncated test runs: only their length (truncation point) is used")
    ap.add_argument("--full-test-set", default="Full_Test_Set",
                    help="the same bearings recorded to the end: ground truth")
    ap.add_argument("--label-mode", default="piecewise",
                    choices=["piecewise", "piecewise_slope", "linear", "capped"])
    ap.add_argument("--sep", default=None, help="CSV separator (default: detected per file)")
    ap.add_argument("--cache", action="store_true", help="cache features as parquet in results/")
    ap.add_argument("--experiment", default="femto-rul-comparison")
    args = ap.parse_args()

    tracking.setup(args.experiment)
    with mlflow.start_run(run_name=f"xgboost_{args.label_mode}"):
        mlflow.set_tag("model_family", "xgboost")
        mlflow.log_params({k: tracking.as_param(v) for k, v in vars(args).items()
                           if k != "experiment"})
        mlflow.log_params({f"xgb_{k}": v for k, v in XGB_PARAMS.items()})

        train_data = load(args.train, args.data_set, args.label_mode, args.sep, args.cache)
        test_data = {n: load_test_bearing(n, args.label_mode, args.test_set, args.full_test_set,
                                          args.sep, args.cache) for n in args.test}

        scalers, cols = fit_scalers(train_data)

        Xtr = np.concatenate([apply_scaler(n, df, scalers, cols) for n, (df, _) in train_data.items()])
        ytr = np.concatenate([t for _, t in train_data.values()])

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr, ytr)

        results = {}
        for name, tb in test_data.items():
            ypred = model.predict(apply_scaler(name, tb.features, scalers, cols))
            results[name] = evaluate_bearing(tb, ypred, offset=0)
            print(format_row(name, results[name]))

        summary = challenge_summary(results)
        print(f"\nchallenge score {summary['phm_score']:.3f} (official table), "
              f"{summary['phm_score_recorded']:.3f} (recorded RUL); "
              f"mean |Er| {summary['mean_abs_er_pct']:.1f}%")
        tracking.log_bearing_metrics({n: {k: m[k] for k in LOGGED_METRICS} for n, m in results.items()})
        mlflow.log_metrics({k: v for k, v in summary.items() if k != "n_bearings"})

        out = config.RESULTS_DIR / f"xgboost_{args.label_mode}_results.json"
        out.write_text(json.dumps({"summary": summary, "bearings": results}, indent=2))
        mlflow.log_artifact(str(out))
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
