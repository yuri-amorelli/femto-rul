"""XGBoost baseline for RUL estimation.

This is the reference the RNN must beat to justify itself. XGBoost sees each
snapshot's feature vector independently (no explicit time context beyond what
the features encode). Run it first; its metrics anchor the whole project.

Usage:
    python -m scripts.run_xgboost_baseline \
        --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
        --test  Bearing1_3 Bearing2_3 Bearing3_3 \
        --data-set Learning_set --label-mode piecewise

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
from src.evaluate import phm_score, summarize
from src.pipeline import apply_scaler, feature_columns, fit_scalers, prepare_bearing

XGB_PARAMS = dict(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
)


def load(names, data_set, label_mode, sep):
    root = config.DATA_ROOT / data_set
    return {n: prepare_bearing(root / n, label_mode=label_mode, sep=sep, cache=False) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--data-set", default="Learning_set")
    ap.add_argument("--test-set", default="Test_set")
    ap.add_argument("--label-mode", default="piecewise", choices=["piecewise", "linear", "capped"])
    ap.add_argument("--sep", default=",")
    ap.add_argument("--experiment", default="femto-rul-comparison")
    args = ap.parse_args()

    tracking.setup(args.experiment)
    with mlflow.start_run(run_name=f"xgboost_{args.label_mode}"):
        mlflow.set_tag("model_family", "xgboost")
        mlflow.log_params({k: tracking.as_param(v) for k, v in vars(args).items()
                           if k != "experiment"})
        mlflow.log_params({f"xgb_{k}": v for k, v in XGB_PARAMS.items()})

        train_data = load(args.train, args.data_set, args.label_mode, args.sep)
        test_data = load(args.test, args.test_set, args.label_mode, args.sep)

        scalers, cols = fit_scalers(train_data)

        Xtr = np.concatenate([apply_scaler(n, df, scalers, cols) for n, (df, _) in train_data.items()])
        ytr = np.concatenate([t for _, t in train_data.values()])

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr, ytr)

        results = {}
        for name, (df, ytrue) in test_data.items():
            Xte = apply_scaler(name, df, scalers, cols)
            ypred = model.predict(Xte)
            m = summarize(ytrue, ypred)
            # PHM score compares the single prediction at the truncation point;
            # here we report it on the final snapshot of each test bearing.
            m["phm_score_final"] = phm_score(np.array([ytrue[-1]]), np.array([ypred[-1]]))
            results[name] = m
            print(f"{name}: RMSE={m['rmse']:.1f}s  MAE={m['mae']:.1f}s  PHM={m['phm_score_final']:.3f}")

        tracking.log_bearing_metrics(results)

        out = config.RESULTS_DIR / "xgboost_results.json"
        out.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(out))
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
