"""check_artifacts.py — checks the serving bundle the API loads (artifacts/serving/femto/),
without training anything and without PyTorch:

  1. every file loads and the parts agree with each other (history features, detector
     feature, stage-2 features and coefficient sizes);
  2. the replay of every official test bearing runs, RULs are finite and >= 0 and only
     appear under an active alarm;
  3. the replay reproduces the evaluation recorded in bundle.json at export time.

    python check_artifacts.py [bundle_dir]
"""
import sys
from pathlib import Path

import numpy as np

from src.serving import default_bundle_dir, load_bundle

root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_bundle_dir()
bundle = load_bundle(root)            # 1) raises if the files disagree
d, det, s2 = bundle.dataset, bundle.detector, bundle.stage2
print(f"bundle {bundle.version} ({root})")
print(f"  dataset {d.name}: {d.snapshot_len} samples at {d.sampling_rate_hz} Hz every {d.snapshot_period_s} s; "
      f"history {list(d.history_features)}")
print(f"  detector on {det.feature}: k={det.k_sigma}, calibration {det.calib_len} snapshots after {det.skip}+{det.w_smooth}")
print(f"  stage 2: quantile {s2.quantile}, features {list(s2.features)}")

curve_mae, alarms = [], 0
for name in bundle.demo_bearings:     # 2) every replay is well formed
    r = bundle.replay(name)
    pred = np.array([np.nan if p is None else p for p in r["rul_pred_s"]])
    alarm = np.zeros(r["n_snapshots"], dtype=bool)
    for seg in r["segments"]:
        if seg["state"] == "alarm":
            alarm[seg["start"]:seg["end"] + 1] = True
    assert np.array_equal(~np.isnan(pred), alarm), f"{name}: RUL outside an active alarm"
    assert (pred[alarm] >= 0).all() and np.isfinite(pred[alarm]).all(), f"{name}: invalid RUL"
    s = r["summary"]
    alarms += s["alarm_at_end"]
    if s["curve_mae_s"] is not None:
        curve_mae.append(s["curve_mae_s"])
    mae = f"{s['curve_mae_s']:7.1f} s" if s["curve_mae_s"] is not None else "      -  "
    print(f"  {name}: {r['n_snapshots']:4d} snapshots | alarm at end {str(s['alarm_at_end']):5} | "
          f"curve MAE {mae} | revoked {s['revoked_alarms']}")

ev = bundle.meta.get("evaluation") or {}  # 3) same numbers as at export
mean = float(np.mean(curve_mae))
print(f"curve MAE over {len(curve_mae)} bearings: {mean:.3f} s (recorded at export: "
      f"{ev.get('curve_mae_s_mean_over_bearings')} s); alarm at end on {alarms} of {len(bundle.demo_bearings)}")
assert abs(mean - ev["curve_mae_s_mean_over_bearings"]) < 1e-3, "replay differs from the evaluation at export"
print("OK")
