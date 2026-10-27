"""Two-stage system with stage 2 trained on the degradation curve only.

The causal detector (artifacts/causal_detector.json, unchanged) decides when a
RUL is worth estimating; the regressor is trained only on the six training
bearings' curves (src/post_alarm.py) and evaluated on the test bearings where
the alarm fires, on their full recordings.

Models:
  countdown    median training curve length minus the time since the alarm
  ridge        least squares on the stage-2 features (estimates the mean)
  quantile     linear quantile regression; the quantile is chosen on the
               TRAINING bearings only, by leave-one-bearing-out, with a rule
               fixed in advance: at most MAX_LATE late predictions, then the
               lowest error
  served_lstm  the LSTM the API serves (trained on whole lives), for reference

Ablation: extending the training curves with snapshots before the onset.

Outputs in reports/post_alarm/: bearings.csv, summary.json, curves.png

Usage:
    python -m scripts.post_alarm_eval --cache
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.evaluate_full_test import load_served_model, predict_run  # noqa: E402
from src import config  # noqa: E402
from src.causal_detector import CausalDetectorConfig, stream_causal_detector  # noqa: E402
from src.pipeline import apply_scaler, prepare_bearing  # noqa: E402
from src.post_alarm import (FEATURES, MAX_LATE, Q_GRID, TRAIN_BEARINGS,  # noqa: E402,F401
                            choose_quantile, curve_frame, curve_metrics, fit, on_curve,
                            training_rows)
from src.test_protocol import TEST_BEARINGS  # noqa: E402

EXTEND_GRID = (30, 100, 400)
ART = config.PROJECT_ROOT / "artifacts"
OUT = config.PROJECT_ROOT / "reports" / "post_alarm"


def frames(set_name: str, names: list[str], cfg, cache: bool):
    out = {}
    for n in names:
        df, _ = prepare_bearing(config.DATA_ROOT / set_name / n, label_mode="linear", cache=cache)
        states = stream_causal_detector(df[cfg.feature].to_numpy(), cfg)
        out[n] = (df, curve_frame(df, states, cfg))
    return out


def plot(test, preds, path):
    names = [n for n in test if test[n][1]["curve"].any()]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, n in zip(axes.flat, names):
        cf = test[n][1]
        c = cf["curve"].to_numpy()
        t = cf["tau"].to_numpy()[c]
        ax.plot(t, cf["true_rul"].to_numpy()[c], color="0.3", lw=1.5, label="true RUL")
        for key, style in (("quantile", dict(color="tab:green", lw=1.3)),
                           ("countdown", dict(color="tab:orange", lw=1, ls="--")),
                           ("served_lstm", dict(color="tab:blue", lw=1, ls=":"))):
            ax.plot(t, preds[n][key][c], label=key, **style)
        ax.set_title(n, fontsize=10)
        ax.set_xlabel("seconds since the alarm onset", fontsize=8)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.supylabel("RUL [s]")
    fig.suptitle("Degradation curve of the 9 test bearings where the alarm fires")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="cache features as parquet in results/")
    args = ap.parse_args()

    cfg = CausalDetectorConfig.from_dict(json.loads((ART / "causal_detector.json").read_text()))
    meta = json.loads((ART / "config.json").read_text())
    scalers = joblib.load(ART / "scalers.joblib")
    lstm = load_served_model(meta)

    train = {n: cf for n, (_, cf) in frames("Learning_set", TRAIN_BEARINGS, cfg, args.cache).items()}
    test = frames("Full_Test_Set", TEST_BEARINGS, cfg, args.cache)
    print("training curves [snapshots]:", {n: int(cf["curve"].sum()) for n, cf in train.items()})

    q, q_table = choose_quantile(train)
    print("quantile chosen on the training bearings:", q, q_table)

    rows = [training_rows(cf) for cf in train.values()]
    models = {"countdown": fit("countdown", rows), "ridge": fit("ridge", rows),
              "quantile": fit("quantile", rows, q)}

    records, preds = [], {}
    for n, (df, cf) in test.items():
        if not cf["curve"].any():
            print(f"{n}: no alarm at end of life, nothing to evaluate on the curve")
            continue
        Z = np.nan_to_num(cf[FEATURES].to_numpy(), nan=0.0)
        p = {k: m(Z) for k, m in models.items()}
        lstm_pred = predict_run(lstm, apply_scaler(n, df, scalers, meta["feature_columns"]),
                                meta["window"], meta["rul_scale"])
        p["served_lstm"] = np.concatenate([np.full(meta["window"] - 1, np.nan), lstm_pred])
        preds[n] = p
        for scope, mask in (("curve", cf["curve"].to_numpy()), ("exposed", cf["alarm"].to_numpy())):
            for k, v in p.items():
                m = curve_metrics(cf["true_rul"].to_numpy()[mask], v[mask])
                records.append({"bearing": n, "scope": scope, "model": k,
                                "curve_s": float(cf["true_rul"].to_numpy()[cf["curve"].to_numpy()][0]), **m})

    table = pd.DataFrame(records)
    cur = table[table["scope"] == "curve"]
    agg = {}
    for scope in ("curve", "exposed"):
        s = table[table["scope"] == scope]
        agg[scope] = s.groupby("model")[["mae", "bias", "late", "mae_over_mean_rul", "within_30pct"]] \
                      .mean().round(3).to_dict(orient="index")
    cd = cur[cur["model"] == "countdown"].set_index("bearing")["mae"]
    beats = {k: int((cur[cur["model"] == k].set_index("bearing")["mae"] < cd).sum())
             for k in models if k != "countdown"}
    beats["served_lstm"] = int((cur[cur["model"] == "served_lstm"].set_index("bearing")["mae"] < cd).sum())
    # a few very short curves dominate the mean of the relative error: report the median too
    median_rel = cur.groupby("model")["mae_over_mean_rul"].median().round(3).to_dict()

    ablation = []
    for ext in EXTEND_GRID:
        ext_rows = [training_rows(cf, ext) for cf in train.values()]
        for kind in ("ridge", "quantile"):
            m = fit(kind, ext_rows, q)
            ablation.append({"extend": ext, "model": kind,
                             "mae": float(np.mean([on_curve(m, cf)["mae"] for _, cf in test.values()
                                                   if cf["curve"].any()]))})

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "bearings.csv", index=False)
    summary = {"quantile": q, "quantile_selection": q_table, "max_late_rule": MAX_LATE,
               "bearings_with_curve": int(cur["bearing"].nunique()),
               "mean_over_bearings": agg, "median_mae_over_mean_rul": median_rel,
               "beats_countdown_on": beats, "ablation_pre_onset": ablation}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(test, preds, OUT / "curves.png")

    print(f"\ncurve, mean over {cur['bearing'].nunique()} bearings (each bearing weighs the same):")
    print(pd.DataFrame(agg["curve"]).T.sort_values("mae").to_string())
    print("median over bearings of MAE / mean true RUL:", median_rel)
    print("beats the countdown on:", beats)
    print("ablation, training curves extended before the onset (test MAE on the curve):")
    print(pd.DataFrame(ablation).to_string(index=False))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
