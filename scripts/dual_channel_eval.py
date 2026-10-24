"""Pre-registered experiment: dual-channel (OR) causal detector.

Protocol, criteria and adoption rule are fixed in
reports/dual_channel/PREREGISTRATION.md, committed before this script was run.

    python -m scripts.dual_channel_eval --gate-only --cache   # training bearings only
    python -m scripts.dual_channel_eval --cache               # gate, then the single test run

Outputs in reports/dual_channel/: bearings.csv, results.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.post_alarm_eval import TRAIN_BEARINGS, choose_quantile, fit  # noqa: E402
from src import config
from src.causal_detector import CausalDetectorConfig, fuse_or, stream_causal_detector
from src.pipeline import prepare_bearing
from src.post_alarm import FEATURES, curve_frame, curve_metrics, training_rows
from src.test_protocol import TEST_BEARINGS

ART = config.PROJECT_ROOT / "artifacts"
OUT = config.PROJECT_ROOT / "reports" / "dual_channel"
DT = config.SNAPSHOT_PERIOD
GATE_MIN_ONSET_PCT = 90.0


def detector_streams(df: pd.DataFrame, cfg: CausalDetectorConfig) -> dict[str, list]:
    h = stream_causal_detector(df["h_rms"].to_numpy(), cfg)
    v = stream_causal_detector(df["v_rms"].to_numpy(), cfg)
    return {"h": h, "hv": fuse_or(h, v)}


def endpoints(states: list) -> dict:
    n = len(states)
    alarm = np.array([s.alarm_active for s in states])
    end = states[-1]
    return {
        "alarm_at_end": bool(end.alarm_active),
        "final_onset_pct": round(100 * end.alarm_onset / n, 1) if end.alarm_active else None,
        "warning_time_s": float((n - 1 - end.alarm_onset) * DT) if end.alarm_active else None,
        "first_alarm_pct": round(100 * int(np.argmax(alarm)) / n, 1) if alarm.any() else None,
        "revoked_alarms": int(end.revoked_alarms),
    }


def load(set_name, names, cfg, cache):
    out = {}
    for n in names:
        df, _ = prepare_bearing(config.DATA_ROOT / set_name / n, label_mode="linear", cache=cache)
        out[n] = (df, detector_streams(df, cfg))
    return out


def stage2(train, test, cfg, variant):
    """The post_alarm_eval procedure, unchanged, on the curves of one detector variant."""
    tr = {n: curve_frame(df, s[variant], cfg) for n, (df, s) in train.items()}
    q, _ = choose_quantile(tr)
    model = fit("quantile", [training_rows(cf) for cf in tr.values()], q)
    curve, exposed = [], []
    for n, (df, s) in test.items():
        cf = curve_frame(df, s[variant], cfg)
        Z = np.nan_to_num(cf[FEATURES].to_numpy(), nan=0.0)
        p, t = model(Z), cf["true_rul"].to_numpy()
        if cf["curve"].any():
            m = cf["curve"].to_numpy()
            curve.append(curve_metrics(t[m], p[m])["mae"])
        if cf["alarm"].any():
            m = cf["alarm"].to_numpy()
            exposed.append(curve_metrics(t[m], p[m])["mae"])
    return {"quantile": q, "curve_mae": float(np.mean(curve)), "curve_bearings": len(curve),
            "exposed_mae": float(np.mean(exposed)), "exposed_bearings": len(exposed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true", help="check the training gate, do not touch the test set")
    ap.add_argument("--cache", action="store_true")
    args = ap.parse_args()
    cfg = CausalDetectorConfig.from_dict(json.loads((ART / "causal_detector.json").read_text()))
    OUT.mkdir(parents=True, exist_ok=True)

    train = load("Learning_set", TRAIN_BEARINGS, cfg, args.cache)
    gate_rows = [{"bearing": n, "variant": v, **endpoints(s[v])} for n, (_, s) in train.items() for v in ("h", "hv")]
    gate = pd.DataFrame(gate_rows)
    print("training bearings:\n", gate.to_string(index=False))
    hv = gate[gate["variant"] == "hv"]
    passed = bool(hv["alarm_at_end"].all() and (hv["final_onset_pct"] >= GATE_MIN_ONSET_PCT).all())
    print(f"\ngate (alarm at end of life on all 6, final onset >= {GATE_MIN_ONSET_PCT}%): "
          f"{'PASSED' if passed else 'FAILED'}")
    results = {"gate": {"passed": passed, "training": gate_rows}}
    if args.gate_only or not passed:
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        print("test set not run" + ("" if passed else ": the change is rejected by the gate"))
        return

    test = load("Full_Test_Set", TEST_BEARINGS, cfg, args.cache)
    rows = [{"bearing": n, "variant": v, **endpoints(s[v])} for n, (_, s) in test.items() for v in ("h", "hv")]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "bearings.csv", index=False)
    summary = {}
    for v in ("h", "hv"):
        t = table[table["variant"] == v]
        alarmed = t[t["alarm_at_end"]]
        summary[v] = {
            "alarm_at_end": int(t["alarm_at_end"].sum()),
            "revoked_total": int(t["revoked_alarms"].sum()),
            "bearings_with_revoked": int((t["revoked_alarms"] > 0).sum()),
            "median_warning_s": float(alarmed["warning_time_s"].median()) if len(alarmed) else None,
            "stage2": stage2(train, test, cfg, v),
        }
    h, d = summary["h"], summary["hv"]
    adopt = (d["alarm_at_end"] > h["alarm_at_end"] and d["revoked_total"] <= h["revoked_total"]
             and d["stage2"]["curve_mae"] <= 1.10 * h["stage2"]["curve_mae"])
    results.update({"test": summary, "adopted": bool(adopt)})
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    print("\ntest bearings:\n", table.to_string(index=False))
    print("\nendpoints:")
    for v in ("h", "hv"):
        s = summary[v]
        print(f"  {v:2s}: alarm at end {s['alarm_at_end']}/11 | revoked {s['revoked_total']} on "
              f"{s['bearings_with_revoked']} bearings | median warning {s['median_warning_s']} s | "
              f"stage 2 q={s['stage2']['quantile']}: curve MAE {s['stage2']['curve_mae']:.1f} s on "
              f"{s['stage2']['curve_bearings']}, exposed MAE {s['stage2']['exposed_mae']:.1f} s")
    print(f"\nadoption rule -> {'ADOPT dual-channel' if adopt else 'KEEP single-channel'}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
