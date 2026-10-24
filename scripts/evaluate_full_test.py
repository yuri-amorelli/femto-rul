"""Evaluate the SERVED two-stage system (artifacts/) on the 11 PHM 2012 test bearings.

No training: it loads the artifacts the API serves and uses the full
run-to-failure recordings (Full_Test_Set) as ground truth. For each bearing:

  1. regression alone, challenge protocol: the LSTM prediction at the Test_set
     truncation point against the actual RUL (%Er, A_i, challenge score);
  2. the causal detector streamed snapshot by snapshot over the full run,
     exactly as the API runs it: state at the truncation point, last alarm
     onset, warning time left before the end of the recording, revoked alarms;
  3. what the API exposes: the RUL only while the alarm is active, against the
     true RUL at those snapshots.

Outputs in reports/full_test_eval/: bearings.csv, summary.json, trajectories.png

Usage:
    python -m scripts.evaluate_full_test
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.causal_detector import CausalDetectorConfig, run_causal_detector  # noqa: E402
from src.dataset import make_sequences  # noqa: E402
from src.models import RULRegressorRNN  # noqa: E402
from src.pipeline import apply_scaler  # noqa: E402
from src.test_protocol import (TEST_BEARINGS, challenge_summary, evaluate_bearing,  # noqa: E402
                               format_row, load_test_bearing)

ART = config.PROJECT_ROOT / "artifacts"
OUT = config.PROJECT_ROOT / "reports" / "full_test_eval"
DT = config.SNAPSHOT_PERIOD


def load_served_model(meta: dict) -> RULRegressorRNN:
    model = RULRegressorRNN(input_size=len(meta["feature_columns"]),
                            hidden_size=meta["hidden_size"],
                            num_layers=meta["num_layers"], rnn_type=meta["rnn_type"])
    model.load_state_dict(torch.load(ART / "model.pth", map_location="cpu"))
    return model.eval()


def predict_run(model, feats: np.ndarray, window: int, rul_scale: float) -> np.ndarray:
    """RUL [s] at every snapshot from window - 1 to the end of the run."""
    X, _ = make_sequences(feats, np.zeros(len(feats), np.float32), window)
    with torch.no_grad():
        return model(torch.from_numpy(X)).numpy() / rul_scale


def stream_detector(signal: np.ndarray, cfg: CausalDetectorConfig) -> list:
    """Detector result at every snapshot, recomputed on the history so far (as the API does)."""
    return [run_causal_detector(signal[: t + 1], cfg) for t in range(len(signal))]


def detector_metrics(tb, states: list, rul_pred: np.ndarray, window: int) -> dict:
    n = tb.n_full
    alarm = np.array([s.alarm_active for s in states])
    end = states[-1]
    first = int(np.argmax(alarm)) if alarm.any() else None
    m = {
        "state_at_truncation": states[tb.truncation_index].state,
        "alarm_before_truncation": bool(alarm[: tb.n_truncated].any()),
        "first_alarm_pct": round(100 * first / n, 1) if first is not None else None,
        "state_at_end": end.state,
        "final_onset_pct": round(100 * end.alarm_onset / n, 1) if end.alarm_active else None,
        "warning_time_s": float((n - 1 - end.alarm_onset) * DT) if end.alarm_active else None,
        "revoked_alarms": end.revoked_alarms,
    }
    # what the API returns: rul_seconds only while the alarm is active
    exposed = alarm[window - 1:]
    m["exposed_snapshots"] = int(exposed.sum())
    if exposed.any():
        err = rul_pred[exposed] - tb.true_rul[window - 1:][exposed]   # > 0: later than reality
        m["exposed_mae_s"] = float(np.mean(np.abs(err)))
        m["exposed_bias_s"] = float(np.mean(err))
        m["exposed_true_rul_mean_s"] = float(np.mean(tb.true_rul[window - 1:][exposed]))
    return m


def plot(runs: dict, path, window: int) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharey=False)
    for ax, (name, r) in zip(axes.flat, runs.items()):
        tb = r["tb"]
        h = np.arange(tb.n_full) * DT / 3600
        alarm = np.array([s.alarm_active for s in r["states"]])
        ax.fill_between(h, 0, 1, where=alarm, transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.15, lw=0, label="alarm active")
        ax.plot(h, tb.true_rul, color="0.35", lw=1.2, label="true RUL")
        ax.plot(h, tb.target, color="0.35", lw=1, ls=":", label="training label scheme")
        ax.plot(h[window - 1:], r["pred"], color="tab:blue", lw=1, label="LSTM")
        ax.axvline(h[tb.truncation_index], color="k", lw=1, ls="--", label="Test_set truncation")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("hours", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=10)
    fig.supylabel("RUL [s]")
    fig.suptitle("Served two-stage system on the 11 PHM 2012 test bearings (full recordings)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", default=TEST_BEARINGS)
    ap.add_argument("--cache", action="store_true", help="cache features as parquet in results/")
    args = ap.parse_args()

    meta = json.loads((ART / "config.json").read_text())
    cols, window, rul_scale = meta["feature_columns"], meta["window"], meta["rul_scale"]
    det_cfg = CausalDetectorConfig.from_dict(json.loads((ART / "causal_detector.json").read_text()))
    scalers = joblib.load(ART / "scalers.joblib")
    model = load_served_model(meta)

    runs, rows, regression = {}, [], {}
    for name in args.test:
        tb = load_test_bearing(name, meta["label_mode"], cache=args.cache)
        pred = predict_run(model, apply_scaler(name, tb.features, scalers, cols), window, rul_scale)
        regression[name] = evaluate_bearing(tb, pred, offset=window - 1)
        states = stream_detector(tb.features[det_cfg.feature].to_numpy(), det_cfg)
        det = detector_metrics(tb, states, pred, window)
        runs[name] = {"tb": tb, "pred": pred, "states": states}
        rows.append({"bearing": name, **regression[name], **det})
        print(format_row(name, regression[name]))
        print(f"    detector: {det['state_at_truncation']} at truncation, "
              f"{det['state_at_end']} at end, last onset {det['final_onset_pct']}% of life, "
              f"warning {det['warning_time_s']} s, revoked {det['revoked_alarms']}")

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "bearings.csv", index=False)

    alarmed = table[table["state_at_end"] == "alarm"]
    summary = {
        "model_version": f"{meta['rnn_type']}-h{meta['hidden_size']}-l{meta['num_layers']}-"
                         f"{meta['label_mode']}-seed{meta['seed']}",
        "regression": challenge_summary(regression),
        "detector": {
            "bearings": len(table),
            "alarm_active_at_end": int(len(alarmed)),
            "alarm_before_truncation": int(table["alarm_before_truncation"].sum()),
            "median_warning_time_s": float(alarmed["warning_time_s"].median()) if len(alarmed) else None,
            "median_final_onset_pct": float(alarmed["final_onset_pct"].median()) if len(alarmed) else None,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(runs, OUT / "trajectories.png", window)

    r = summary["regression"]
    d = summary["detector"]
    print(f"\nchallenge score {r['phm_score']:.3f} (official table), "
          f"{r['phm_score_recorded']:.3f} (recorded RUL); mean |Er| {r['mean_abs_er_pct']:.1f}%")
    print(f"detector: alarm active at end of life on {d['alarm_active_at_end']}/{d['bearings']}, "
          f"median warning time {d['median_warning_time_s']} s")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
