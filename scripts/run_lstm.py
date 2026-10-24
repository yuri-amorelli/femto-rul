"""LSTM / GRU RUL estimator in PyTorch, compared against the XGBoost baseline.

Usage:
    python -m scripts.run_lstm \
        --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
        --rnn lstm --window 30 --hidden 64 --layers 2 --label-mode capped --seed 0

Test bearings default to all 11 of the challenge, evaluated with the protocol
in src/test_protocol.py: ground truth from Full_Test_Set, challenge score at
the Test_set truncation point.

Note: one training bearing is held out as validation (never seen in fit) so
early stopping doesn't peek at the test bearings.

Every run is tracked with MLflow (experiment "femto-rul-comparison" by
default): parameters, per-epoch losses, per-bearing test metrics and the
results JSON. Repeating the run with different --seed values gives the
seed-to-seed variance directly in the MLflow UI.
"""
from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, tracking
from src.dataset import SequenceDataset, make_sequences
from src.models import RULRegressorRNN
from src.pipeline import apply_scaler, fit_scalers, prepare_bearing
from src.test_protocol import (LOGGED_METRICS, TEST_BEARINGS, challenge_summary,
                               evaluate_bearing, format_row, load_test_bearing)
from src.train import get_device, train_model

# We scale the RUL target too (divide by a fixed horizon) so the loss is O(1)
# and Softplus doesn't have to output values in the tens of thousands.
RUL_SCALE = 1.0 / 9000.0  # ~ inverse of a typical bearing life in seconds


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
    ap.add_argument("--rnn", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep", default=None, help="CSV separator (default: detected per file)")
    ap.add_argument("--cache", action="store_true", help="cache features as parquet in results/")
    ap.add_argument("--experiment", default="femto-rul-comparison")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # hold out the last training bearing for validation
    val_name = args.train[-1]
    fit_names = args.train[:-1]

    tracking.setup(args.experiment)
    run_name = f"{args.rnn}_w{args.window}_{args.label_mode}_seed{args.seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", "rnn")
        mlflow.log_params({k: tracking.as_param(v) for k, v in vars(args).items()
                           if k != "experiment"})
        mlflow.log_params({"rul_scale": RUL_SCALE, "val_bearing": val_name,
                           "fit_bearings": tracking.as_param(fit_names)})

        all_train = load(args.train, args.data_set, args.label_mode, args.sep, args.cache)
        test_data = {n: load_test_bearing(n, args.label_mode, args.test_set, args.full_test_set,
                                          args.sep, args.cache) for n in args.test}
        fit_data = {n: all_train[n] for n in fit_names}

        scalers, cols = fit_scalers(fit_data)

        def scaled_pairs(data):
            return [(apply_scaler(n, df, scalers, cols), t * RUL_SCALE) for n, (df, t) in data.items()]

        train_ds = SequenceDataset(scaled_pairs(fit_data), window=args.window)
        val_ds = SequenceDataset(scaled_pairs({val_name: all_train[val_name]}), window=args.window)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch)

        model = RULRegressorRNN(
            input_size=len(cols), hidden_size=args.hidden,
            num_layers=args.layers, rnn_type=args.rnn,
        )
        out = train_model(model, train_loader, val_loader, epochs=args.epochs,
                          lr=args.lr, patience=args.patience, clip=args.clip,
                          on_epoch_end=tracking.epoch_logger())
        model = out["model"]
        mlflow.log_metrics({"best_val_loss": out["best_val"], "epochs_run": out["epochs_run"]})
        device = get_device()
        model.eval()

        results = {}
        for name, tb in test_data.items():
            feats = apply_scaler(name, tb.features, scalers, cols)
            X, _ = make_sequences(feats, tb.target * RUL_SCALE, window=args.window)
            with torch.no_grad():
                pred = model(torch.from_numpy(X).to(device)).cpu().numpy() / RUL_SCALE
            # predictions start at snapshot window - 1 of the full run
            results[name] = evaluate_bearing(tb, pred, offset=args.window - 1)
            print(format_row(name, results[name]))

        summary = challenge_summary(results)
        print(f"\nchallenge score {summary['phm_score']:.3f} (official table), "
              f"{summary['phm_score_recorded']:.3f} (recorded RUL); "
              f"mean |Er| {summary['mean_abs_er_pct']:.1f}%")
        tracking.log_bearing_metrics({n: {k: m[k] for k in LOGGED_METRICS} for n, m in results.items()})
        mlflow.log_metrics({k: v for k, v in summary.items() if k != "n_bearings"})

        tag = f"{args.rnn}_w{args.window}_{args.label_mode}_seed{args.seed}"
        path = config.RESULTS_DIR / f"{tag}_results.json"
        path.write_text(json.dumps({"summary": summary, "bearings": results}, indent=2))
        mlflow.log_artifact(str(path))
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
