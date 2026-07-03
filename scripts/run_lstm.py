"""LSTM / GRU RUL estimator in PyTorch, compared against the XGBoost baseline.

Usage:
    python -m scripts.run_lstm \
        --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
        --test  Bearing1_3 Bearing2_3 Bearing3_3 \
        --rnn lstm --window 30 --hidden 64 --layers 2

Note: one training bearing is held out as validation (never seen in fit) so
early stopping doesn't peek at the test bearings.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import SequenceDataset, make_sequences
from src.evaluate import phm_score, summarize
from src.models import RULRegressorRNN
from src.pipeline import apply_scaler, fit_scalers, prepare_bearing
from src.train import get_device, train_model

# We scale the RUL target too (divide by a fixed horizon) so the loss is O(1)
# and Softplus doesn't have to output values in the tens of thousands.
RUL_SCALE = 1.0 / 9000.0  # ~ inverse of a typical bearing life in seconds


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
    ap.add_argument("--rnn", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--sep", default=",")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    all_train = load(args.train, args.data_set, args.label_mode, args.sep)
    test_data = load(args.test, args.test_set, args.label_mode, args.sep)

    # hold out the last training bearing for validation
    val_name = args.train[-1]
    fit_names = args.train[:-1]
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
    out = train_model(model, train_loader, val_loader, epochs=args.epochs)
    model = out["model"]
    device = get_device()
    model.eval()

    results = {}
    for name, (df, ytrue) in test_data.items():
        feats = apply_scaler(name, df, scalers, cols)
        X, _ = make_sequences(feats, ytrue * RUL_SCALE, window=args.window)
        if len(X) == 0:
            continue
        with torch.no_grad():
            pred = model(torch.from_numpy(X).to(device)).cpu().numpy() / RUL_SCALE
        # align: predictions start at index (window-1)
        ytrue_aligned = ytrue[args.window - 1:]
        m = summarize(ytrue_aligned, pred)
        m["phm_score_final"] = phm_score(np.array([ytrue_aligned[-1]]), np.array([pred[-1]]))
        results[name] = m
        print(f"{name}: RMSE={m['rmse']:.1f}s  MAE={m['mae']:.1f}s  PHM={m['phm_score_final']:.3f}")

    tag = f"{args.rnn}_w{args.window}"
    path = config.RESULTS_DIR / f"{tag}_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
