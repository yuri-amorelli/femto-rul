"""Export the served two-stage model as a torch-free JSON bundle, and register it.

    python -m scripts.export_serving --cache              # bundle + MLflow run + registry
    python -m scripts.export_serving --cache --no-mlflow  # bundle only

Writes artifacts/serving/femto/ (see src/serving.py for the layout):

 1. stage-2 training curves of the six training bearings (src/post_alarm.py,
    the same code as scripts/post_alarm_eval.py);
 2. the quantile, chosen by leave-one-bearing-out on the training bearings only,
    with the rule fixed in advance (at most 25 % late predictions, then the
    lowest error);
 3. the final linear quantile regression on all six curves, written as numbers
    and checked against the scikit-learn pipeline it comes from;
 4. the features of the 11 official test bearings (full recordings), for the
    replay endpoint;
 5. the healthy-phase feature distributions of the training bearings, per
    operating condition, as the reference for drift monitoring;
 6. a check: the bundle is reloaded through src/serving.py and replayed on the
    test bearings; its stage-2 error on the degradation curve must match
    reports/post_alarm/summary.json, otherwise nothing is registered;
 7. MLflow: parameters, metrics, the bundle as artifacts, and the pipeline
    registered as "femto-rul-stage2" (experiment "femto-rul-production").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config
from src.causal_detector import CausalDetectorConfig, stream_causal_detector
from src.data_loading import n_snapshots
from src.pipeline import prepare_bearing
from src.post_alarm import (FEATURES, MAX_LATE, SMOOTH, TRAIN_BEARINGS, choose_quantile, curve_frame,
                            fit_quantile_pipeline, stack, training_rows)
from src.serving import REQUIRED_HISTORY, STAGE2_FEATURES, load_bundle
from src.test_protocol import OFFICIAL_ACTUAL_RUL, TEST_BEARINGS

ART = config.PROJECT_ROOT / "artifacts"
OUT = ART / "serving" / "femto"
SUMMARY = config.PROJECT_ROOT / "reports" / "post_alarm" / "summary.json"
EXPERIMENT = "femto-rul-production"
REGISTERED_MODEL = "femto-rul-stage2"
TOLERANCE_S = 2e-3          # the report is rounded to 3 decimals; platform float noise is ~1e-5 s
NOTES = {
    "Bearing1_4": ("The official actual RUL is 339 s, but the recording continues for 2,890 s "
                   "after the truncation point; here 'true RUL' is the time to the end of the recording."),
    "Bearing2_5": ("Missed: the calibration window falls in a slow settling phase (horizontal RMS falling "
                   "from about 0.76 to 0.4), which inflates the learnt threshold, so the final rise never "
                   "crosses it."),
    "Bearing2_7": ("Missed by construction: the recording has 230 snapshots, fewer than the ~285 the "
                   "detector needs to finish calibrating, so the service never leaves 'calibrating'."),
}


def sig(x: float, digits: int = 9) -> float:
    return float(f"{x:.{digits}g}")


def rounded(obj):
    """Floats to 9 significant digits, recursively: the version must not change with
    float noise between machines, only with what is actually served."""
    if isinstance(obj, float):
        return sig(obj)
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    return obj


def source_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="cache features as parquet in results/")
    ap.add_argument("--no-mlflow", action="store_true", help="write the bundle without logging to MLflow")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    det_dict = json.loads((ART / "causal_detector.json").read_text())
    cfg = CausalDetectorConfig.from_dict(det_dict)

    # 1) training curves
    train_df, train_cf = {}, {}
    for n in TRAIN_BEARINGS:
        df, _ = prepare_bearing(config.DATA_ROOT / "Learning_set" / n, label_mode="linear", cache=args.cache)
        states = stream_causal_detector(df[cfg.feature].to_numpy(), cfg)
        train_df[n], train_cf[n] = df, curve_frame(df, states, cfg)

    # 2) quantile chosen on the training bearings only
    q, q_table = choose_quantile(train_cf)
    print(f"quantile chosen by leave-one-bearing-out (late <= {MAX_LATE}, then lowest MAE): {q}")

    # 3) final model on all six curves, exported as numbers
    rows = [training_rows(cf) for cf in train_cf.values()]
    pipe = fit_quantile_pipeline(rows, q)
    scaler, qr = pipe.named_steps["standardscaler"], pipe.named_steps["quantileregressor"]
    X, _, _ = stack(rows)
    stage2 = {
        "model": "StandardScaler + linear quantile regression (scikit-learn QuantileRegressor, alpha=0, highs)",
        "features": list(FEATURES),
        "quantile": q,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in qr.coef_],
        "intercept": float(qr.intercept_),
        "smooth": SMOOTH,
        "clip_min": 0.0,
        "training": {"bearings": TRAIN_BEARINGS, "curve_snapshots": int(len(X)),
                     "weighting": "every bearing weighs the same",
                     "target": "seconds to the end of the run-to-failure recording"},
        "quantile_selection": {"grid": [r["q"] for r in q_table], "max_late": MAX_LATE,
                               "procedure": "leave-one-training-bearing-out", "table": q_table},
    }
    assert tuple(stage2["features"]) == STAGE2_FEATURES

    dataset = {
        "name": "femto",
        "description": "FEMTO-ST / PRONOSTIA bearings (IEEE PHM 2012), horizontal and vertical accelerometers",
        "sampling_rate_hz": config.FS,
        "snapshot_len": config.SNAPSHOT_LEN,
        "snapshot_period_s": config.SNAPSHOT_PERIOD,
        "history_features": list(REQUIRED_HISTORY),
        "operating_conditions": sorted(config.CONDITIONS),
    }

    # 4) demo features of the official test bearings (full recordings)
    demo = {}
    for n in TEST_BEARINGS:
        df, _ = prepare_bearing(config.DATA_ROOT / "Full_Test_Set" / n, label_mode="linear", cache=args.cache)
        demo[n] = {
            "bearing": n,
            "operating_condition": config.condition_of(n),
            "official_truncation_index": n_snapshots(config.DATA_ROOT / "Test_set" / n) - 1,
            "official_actual_rul_s": OFFICIAL_ACTUAL_RUL[n],
            "note": NOTES.get(n),
            "features": {c: [float(v) for v in df[c]] for c in REQUIRED_HISTORY},
        }

    # 5) drift reference: healthy phase (before the final alarm) of the training bearings
    feat_cols = [c for c in train_df[TRAIN_BEARINGS[0]].columns if c[:2] in ("h_", "v_")]
    reference = {"scope": "training bearings, snapshots before the final alarm onset",
                 "quantile_levels": list(range(101)), "per_condition": {}}
    for c in sorted(config.CONDITIONS):
        names = [n for n in TRAIN_BEARINGS if config.condition_of(n) == c]
        healthy = [train_df[n].iloc[: int(train_cf[n]["final_onset"].iloc[0])] for n in names]
        reference["per_condition"][str(c)] = {
            "bearings": names,
            "n_snapshots": int(sum(len(h) for h in healthy)),
            "quantiles": {f: [sig(v) for v in np.percentile(np.concatenate([h[f].to_numpy() for h in healthy]),
                                                                range(101))] for f in feat_cols},
        }

    # version: changes whenever what is served changes
    served = {k: stage2[k] for k in ("features", "quantile", "scaler_mean", "scaler_scale", "coef",
                                     "intercept", "smooth", "clip_min")}
    fingerprint = json.dumps(rounded({"detector": det_dict, "dataset": dataset, "stage2": served}), sort_keys=True)
    version = f"femto-stage2-q{q:.2f}-{hashlib.sha256(fingerprint.encode()).hexdigest()[:8]}"

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    write_json(out / "dataset.json", dataset)
    write_json(out / "detector.json", det_dict)
    write_json(out / "stage2.json", stage2)
    write_json(out / "reference_stats.json", reference)
    for n, d in demo.items():
        (out / "demo").mkdir(parents=True, exist_ok=True)
        (out / "demo" / f"{n}.json").write_text(json.dumps(d) + "\n")
    bundle_meta = {"format": 1, "version": version, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "source_commit": source_commit(), "evaluation": None}
    write_json(out / "bundle.json", bundle_meta)

    # 6) reload through the serving code and check against the evaluated numbers
    bundle = load_bundle(out)
    pred_json = bundle.stage2.predict(X)
    pred_skl = np.maximum(pipe.predict(X), 0.0)
    assert np.allclose(pred_json, pred_skl, rtol=0, atol=1e-6), "JSON model differs from the sklearn pipeline"
    replays = {n: bundle.replay(n)["summary"] for n in TEST_BEARINGS}
    curve = {n: s for n, s in replays.items() if s["curve_mae_s"] is not None}
    curve_mae = float(np.mean([s["curve_mae_s"] for s in curve.values()]))
    evaluation = {
        "scope": "official test bearings, full recordings, stage 2 on the degradation curve",
        "alarm_at_end": f"{sum(s['alarm_at_end'] for s in replays.values())} of {len(replays)}",
        "curve_bearings": len(curve),
        "curve_mae_s_mean_over_bearings": round(curve_mae, 3),
        "curve_late_fraction_mean_over_bearings": round(float(np.mean([s["curve_late_fraction"] for s in curve.values()])), 3),
        "report": "reports/post_alarm/summary.json",
    }
    print("replay of the test bearings through the bundle:")
    for n, s in replays.items():
        mae = f"{s['curve_mae_s']:.1f} s" if s["curve_mae_s"] is not None else "-"
        print(f"  {n}: alarm at end {s['alarm_at_end']!s:5} | onset {s['final_onset_pct_of_life']} % | curve MAE {mae}")
    print(f"curve MAE, mean over {len(curve)} bearings: {curve_mae:.3f} s")

    if SUMMARY.exists():
        reported = json.loads(SUMMARY.read_text())["mean_over_bearings"]["curve"]["quantile"]["mae"]
        if abs(curve_mae - reported) > TOLERANCE_S:
            raise SystemExit(f"bundle curve MAE {curve_mae:.4f} s != reported {reported} s: not registering")
        print(f"matches reports/post_alarm/summary.json ({reported} s)")
    bundle_meta["evaluation"] = evaluation
    write_json(out / "bundle.json", bundle_meta)
    print(f"bundle {version} written to {out}")

    # 7) MLflow
    if args.no_mlflow:
        return
    import mlflow
    import mlflow.sklearn

    from src import tracking

    tracking.setup(EXPERIMENT)
    with mlflow.start_run(run_name=f"stage2_quantile_q{q}") as run:
        mlflow.set_tags({"purpose": "model served by the FastAPI service", "bundle_version": version})
        mlflow.log_params({"stage2_model": "linear_quantile", "quantile": q, "max_late_rule": MAX_LATE,
                           "features": tracking.as_param(FEATURES), "smooth": SMOOTH,
                           "train_bearings": tracking.as_param(TRAIN_BEARINGS),
                           **{f"detector_{k}": v for k, v in det_dict.items()}})
        for r in q_table:
            mlflow.log_metrics({f"lobo_mae_q{r['q']}": r["lobo_mae"], f"lobo_late_q{r['q']}": r["lobo_late"]})
        mlflow.log_metrics({"test_curve_mae_s": curve_mae, "test_curve_bearings": len(curve)})
        mlflow.log_artifacts(str(out), artifact_path="serving_bundle")
        mlflow.sklearn.log_model(pipe, name="model", input_example=X[:5],
                                 registered_model_name=REGISTERED_MODEL)
        print(f"MLflow run {run.info.run_id} -> registered as '{REGISTERED_MODEL}'")


if __name__ == "__main__":
    main()
