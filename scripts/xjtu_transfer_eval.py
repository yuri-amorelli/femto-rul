"""Pre-registered transfer test of the shipped system on XJTU-SY. Run it once.

Protocol, rules and expectations: reports/xjtu_transfer/PREREGISTRATION.md (commit it first).
Nothing here is fitted on XJTU-SY: the detector is the shipped configuration, translated
mechanically to a 60 s snapshot period, and stage 2 is the served FEMTO model.

    python -m scripts.xjtu_transfer_eval
    python -m scripts.xjtu_transfer_eval --features data/xjtu_features/features_transfer.csv

Outputs in reports/xjtu_transfer/: bearings.csv, results.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.causal_detector import CausalDetectorConfig, stream_causal_detector
from src.serving import STAGE2_FEATURES, LinearQuantileModel, stage2_matrix
from src.xjtu import (BEARINGS, FEMTO_PERIOD, SNAPSHOT_PERIOD, TRANSFER, VAULT, condition, snapshots_to_monitor,
                      translate_config)

OUT = config.PROJECT_ROOT / "reports" / "xjtu_transfer"
BUNDLE = config.PROJECT_ROOT / "artifacts" / "serving" / "femto"
PREREG = "reports/xjtu_transfer/PREREGISTRATION.md"

COUNTDOWN_S = 245.0     # countdown baseline: median length of the FEMTO training curves (reports/post_alarm)
EARLY_PCT = 50.0        # an alarm raised before half of the life counts as early
RULES = {               # fixed in PREREGISTRATION.md
    "min_alarm_at_end": 0.75,          # share of monitored bearings with the alarm active at end of life
    "max_early": 0.25,                 # share of monitored bearings with an alarm before 50 % of life
    "min_beats_countdown": 2 / 3,      # share of alarmed bearings where stage 2 beats the countdown
    "max_median_mae_over_rul": 1.0,    # median over alarmed bearings of curve MAE / mean true RUL
}


def prereg_commit() -> str | None:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", PREREG], capture_output=True,
                             text=True, cwd=config.PROJECT_ROOT, check=False).stdout.strip()
        return out or None
    except OSError:
        return None


def bearing_row(name: str, g: pd.DataFrame, cfg: CausalDetectorConfig, model: LinearQuantileModel,
                smooth: int) -> dict:
    n = len(g)
    idx = np.arange(n)
    states = stream_causal_detector(g[cfg.feature].to_numpy(), cfg)
    onsets = np.array([s.alarm_onset if s.alarm_active else -1 for s in states])
    alarm = onsets >= 0
    end = states[-1]
    final = end.alarm_onset if end.alarm_active else None
    first = int(np.argmax(alarm)) if alarm.any() else None
    r: dict = {
        "bearing": name, "condition": condition(name), "fault": BEARINGS[name][1], "snapshots": n,
        "monitored": n >= snapshots_to_monitor(cfg),
        "alarm_at_end": bool(end.alarm_active),
        "final_onset_pct": round(100 * final / n, 1) if final is not None else None,
        "warning_min": (n - 1 - final) * SNAPSHOT_PERIOD / 60 if final is not None else None,
        "first_alarm_pct": round(100 * first / n, 1) if first is not None else None,
        "early_alarm": bool(first is not None and 100 * first / n < EARLY_PCT),
        "revoked_alarms": int(end.revoked_alarms),
    }
    if final is None:
        return r
    true = (n - 1 - idx).astype(float) * SNAPSHOT_PERIOD
    history = {c: g[c].to_numpy() for c in ("h_rms", "v_rms", "h_kurtosis", "v_kurtosis")}
    Z = stage2_matrix(history, onsets, cfg, smooth, SNAPSHOT_PERIOD)
    pred = np.full(n, np.nan)
    pred[alarm] = model.predict(Z[alarm])
    curve = alarm & (idx >= final)
    t, p = true[curve], pred[curve]
    cd = np.maximum(COUNTDOWN_S - Z[curve, 0], 0.0)
    mean_rul = max(float(np.mean(t)), SNAPSHOT_PERIOD)
    r |= {
        "curve_snapshots": int(curve.sum()),
        "mean_true_rul_min": float(np.mean(t)) / 60,
        "stage2_mae_min": float(np.mean(np.abs(p - t))) / 60,
        "stage2_late": float(np.mean(p > t)),
        "stage2_mae_over_rul": float(np.mean(np.abs(p - t))) / mean_rul,
        "stage2_within_30pct": float(np.mean(np.abs(p - t) <= 0.3 * np.maximum(t, SNAPSHOT_PERIOD))),
        "stage2_median_pred_min": float(np.median(p)) / 60,
        "countdown_mae_min": float(np.mean(np.abs(cd - t))) / 60,
        "beats_countdown": bool(np.mean(np.abs(p - t)) < np.mean(np.abs(cd - t))),
    }
    return r


def decide(rows: list[dict]) -> dict:
    mon = [r for r in rows if r["monitored"]]
    alarmed = [r for r in mon if r["alarm_at_end"]]
    share = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")  # noqa: E731
    d = {
        "monitored": len(mon), "alarm_at_end": len(alarmed),
        "alarm_at_end_share": share(mon, "alarm_at_end"), "early_share": share(mon, "early_alarm"),
        "beats_countdown_share": share(alarmed, "beats_countdown"),
        "median_mae_over_rul": float(np.median([r["stage2_mae_over_rul"] for r in alarmed])) if alarmed else None,
        "median_final_onset_pct": float(np.median([r["final_onset_pct"] for r in alarmed])) if alarmed else None,
        "median_warning_min": float(np.median([r["warning_min"] for r in alarmed])) if alarmed else None,
    }
    d["detector_transfers"] = bool(d["alarm_at_end_share"] >= RULES["min_alarm_at_end"]
                                   and d["early_share"] <= RULES["max_early"])
    d["stage2_transfers"] = bool(alarmed and d["beats_countdown_share"] >= RULES["min_beats_countdown"]
                                 and d["median_mae_over_rul"] <= RULES["max_median_mae_over_rul"])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.DATA_ROOT / "xjtu_features" / "features_transfer.csv"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    df = pd.read_csv(args.features)
    names = set(df["bearing"])
    if names & set(VAULT):
        raise SystemExit(f"vault bearings in the input ({sorted(names & set(VAULT))}): the vault stays closed")
    if names != set(TRANSFER):
        raise SystemExit(f"expected the 12 transfer bearings, got {sorted(names)}")
    commit = prereg_commit()
    if commit is None:
        print(f"WARNING: {PREREG} is not committed; the run is not pre-registered")

    shipped = CausalDetectorConfig.from_dict(json.loads((config.PROJECT_ROOT / "artifacts" / "causal_detector.json")
                                                        .read_text()))
    model = LinearQuantileModel.from_dict(json.loads((BUNDLE / "stage2.json").read_text()))
    assert tuple(model.features) == tuple(STAGE2_FEATURES)
    arms = {
        "time": (translate_config(shipped), max(1, round(model.smooth * FEMTO_PERIOD / SNAPSHOT_PERIOD))),
        "count": (shipped, model.smooth),
    }

    rows, results = [], {"preregistration_commit": commit, "rules": RULES, "countdown_s": COUNTDOWN_S,
                         "vault": VAULT, "arms": {}}
    for arm, (cfg, smooth) in arms.items():
        arm_rows = [bearing_row(b, g.sort_values("snapshot"), cfg, model, smooth)
                    for b, g in sorted(df.groupby("bearing"), key=lambda kv: kv[0])]
        rows += [{"arm": arm} | r for r in arm_rows]
        results["arms"][arm] = {"primary": arm == "time", "detector": asdict(cfg), "stage2_smooth": smooth,
                                "summary": decide(arm_rows)}

    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "bearings.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    cols = ["bearing", "fault", "snapshots", "monitored", "alarm_at_end", "final_onset_pct", "warning_min",
            "first_alarm_pct", "revoked_alarms", "stage2_mae_min", "mean_true_rul_min", "beats_countdown"]
    for arm in arms:
        s = results["arms"][arm]["summary"]
        tag = "PRIMARY" if arm == "time" else "secondary, descriptive"
        print(f"\n=== arm '{arm}' ({tag}) ===")
        print(table[table["arm"] == arm][[c for c in cols if c in table]].to_string(index=False))
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
