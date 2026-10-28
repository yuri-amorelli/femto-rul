"""Pre-registered experiment: a calibration check for the causal detector.

Protocol, threshold, endpoints and adoption rule are fixed in
reports/calibration_check/PREREGISTRATION.md, committed before the test run.

    python -m scripts.calibration_check_eval --train-only --cache   # design tables, training bearings only
    python -m scripts.calibration_check_eval --cache                # the single run on the test bearings

Outputs in reports/calibration_check/:
    training.csv                  the check on the six training bearings (design data)
    drift_reference_training.csv  why a per-condition drift reference was not used (training only)
    test.csv, results.json        the pre-registered test run
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.causal_detector import (CalibrationCheckConfig, CausalDetectorConfig, calibration_check,
                                 stream_causal_detector)
from src.pipeline import prepare_bearing
from src.post_alarm import TRAIN_BEARINGS
from src.test_protocol import TEST_BEARINGS

ART = config.PROJECT_ROOT / "artifacts"
OUT = config.PROJECT_ROOT / "reports" / "calibration_check"
MISSED_BY_THRESHOLD = "Bearing2_5"   # the miss that motivated the check (test-informed, declared)
SCALE_FREE = [f"{c}_{f}" for c in "hv" for f in ("kurtosis", "crest_factor", "shape_factor", "band1_energy",
                                                  "band2_energy", "band3_energy", "band4_energy", "spec_centroid")]


def psi(ref: np.ndarray, x: np.ndarray, bins: int = 10) -> float:
    """Population stability index of x against ref, on ref's deciles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    q = np.clip(np.histogram(x, edges)[0] / len(x), 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def run(set_name: str, names: list[str], cfg, check, cache: bool) -> tuple[pd.DataFrame, dict]:
    rows, frames = [], {}
    for n in names:
        df, _ = prepare_bearing(config.DATA_ROOT / set_name / n, label_mode="linear", cache=cache)
        rms = df[cfg.feature].to_numpy()
        states = stream_causal_detector(rms, cfg)
        end = states[-1]
        c = calibration_check(rms, cfg, check)
        frames[n] = (df, end.alarm_onset if end.alarm_active else len(df))
        rows.append({"bearing": n, "snapshots": len(df), "status": c.status,
                     "noise_ratio": None if c.noise_ratio is None else round(c.noise_ratio, 3),
                     "assessed_at": c.assessed_at, "alarm_at_end": bool(end.alarm_active),
                     "final_onset_pct": round(100 * end.alarm_onset / len(df), 1) if end.alarm_active else None,
                     "revoked_alarms": int(end.revoked_alarms)})
    return pd.DataFrame(rows), frames


def drift_reference_table(frames: dict, cfg) -> pd.DataFrame:
    """Leave-one-bearing-out within each condition: the calibration window of one training
    bearing against the healthy phase of the other bearing of the same condition."""
    cal0 = cfg.skip + cfg.w_smooth
    cal1 = cal0 + cfg.calib_len
    rows = []
    for b, (df, _) in frames.items():
        other = next(o for o in frames if o != b and config.condition_of(o) == config.condition_of(b))
        ref_df, ref_onset = frames[other]
        values = {f: psi(ref_df[f].to_numpy()[:ref_onset], df[f].to_numpy()[cal0:cal1]) for f in SCALE_FREE}
        rows.append({"bearing": b, "reference": other, "median_psi": round(float(np.median(list(values.values()))), 2),
                     "features_psi_above_0.25": sum(v > 0.25 for v in values.values()),
                     "features": len(values)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true", help="design tables only; the test set is not read")
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()
    cfg = CausalDetectorConfig.from_dict(json.loads((ART / "causal_detector.json").read_text()))
    check = CalibrationCheckConfig()
    OUT.mkdir(parents=True, exist_ok=True)

    train, frames = run("Learning_set", TRAIN_BEARINGS, cfg, check, args.cache)
    train.to_csv(OUT / "training.csv", index=False)
    drift = drift_reference_table(frames, cfg)
    drift.to_csv(OUT / "drift_reference_training.csv", index=False)
    print("training bearings:\n", train.to_string(index=False))
    print("\nper-condition drift reference (training, leave-one-bearing-out):\n", drift.to_string(index=False))
    if args.train_only:
        print("\ntest set not read")
        return

    test, _ = run("Full_Test_Set", TEST_BEARINGS, cfg, check, args.cache)
    test.to_csv(OUT / "test.csv", index=False)
    suspect = test[test["status"] == "suspect"]
    false_flags = suspect[suspect["alarm_at_end"]]["bearing"].tolist()
    target_flagged = bool((suspect["bearing"] == MISSED_BY_THRESHOLD).any())
    adopt = target_flagged and not false_flags
    results = {
        "config": {"horizon": check.horizon, "max_ratio": check.max_ratio},
        "assessable": int((test["status"] != "pending").sum()),
        "E1_flagged": suspect["bearing"].tolist(),
        "E2_target_flagged": target_flagged,
        "E3_false_flags": false_flags,
        "adopted": adopt,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\ntest bearings:\n", test.to_string(index=False))
    print(f"\nE1 flagged: {results['E1_flagged']}")
    print(f"E2 {MISSED_BY_THRESHOLD} flagged: {target_flagged}")
    print(f"E3 false flags (alarm active at end of life): {false_flags}")
    print(f"adoption rule -> {'ADOPT the check' if adopt else 'REJECT the check'}")


if __name__ == "__main__":
    main()
