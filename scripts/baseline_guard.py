"""Baseline-guard exploration, step 1: can other bearings flag an unhealthy calibration window?

Rules fixed in reports/baseline_guard/PLAN.md (committed in bba9343 before any number was
computed). Exploratory, on data already seen: FEMTO-ST (17 bearings) and XJTU-SY (15).

    python -m scripts.export_femto_features --cache      # data/femto_features/features_all.csv
    python -m scripts.baseline_guard                     # reads both rigs, writes the report files

Writes reports/baseline_guard/flags.csv and step1.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.causal_detector import CausalDetectorConfig

OUT = config.PROJECT_ROOT / "reports" / "baseline_guard"
FEATURE = "h_rms"
FAR_OUT = 3.0   # Tukey far-out fences: Q1 - 3 IQR, Q3 + 3 IQR

# classes written in PLAN.md before computing
HITS = {("XJTU-SY", "Bearing1_2"), ("XJTU-SY", "Bearing3_5")}
ACCEPTABLE = {("XJTU-SY", "Bearing1_1"), ("XJTU-SY", "Bearing1_3"), ("XJTU-SY", "Bearing2_2"),
              ("FEMTO-ST", "Bearing2_5")}
MAX_FALSE_FLAGS = 2


def load_rigs() -> dict[str, tuple[pd.DataFrame, CausalDetectorConfig]]:
    femto = pd.read_csv(config.DATA_ROOT / "femto_features" / "features_all.csv")
    xjtu = pd.concat([pd.read_csv(config.DATA_ROOT / "xjtu_features" / f)
                      for f in ("features_transfer.csv", "features_vault.csv")], ignore_index=True)
    art = config.PROJECT_ROOT / "artifacts"
    return {
        "FEMTO-ST": (femto, CausalDetectorConfig.from_dict(json.loads((art / "causal_detector.json").read_text()))),
        "XJTU-SY": (xjtu, CausalDetectorConfig.from_dict(json.loads((art / "xjtu" / "detector.json").read_text()))),
    }


def window(cfg: CausalDetectorConfig) -> tuple[int, int]:
    """The calibration window the detector learns its threshold from: [start, end)."""
    start = cfg.skip + cfg.w_smooth
    return start, start + cfg.calib_len


def window_stats(rms: np.ndarray, cfg: CausalDetectorConfig) -> dict | None:
    """rise = log(median of last third / median of first third); level = log(median).
    None when the record ends before the window is complete."""
    a, b = window(cfg)
    if len(rms) < b:
        return None
    w = np.asarray(rms[a:b], dtype=float)
    first, _, last = np.array_split(w, 3)
    return {"rise": float(np.log(np.median(last) / np.median(first))), "level": float(np.log(np.median(w)))}


def fences(ref: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.percentile(ref, [25, 75])
    return q1 - FAR_OUT * (q3 - q1), q3 + FAR_OUT * (q3 - q1)


def outside(x: float, ref: np.ndarray) -> bool:
    lo, hi = fences(ref)
    return bool(x < lo or x > hi)


def table() -> pd.DataFrame:
    rows = []
    for rig, (df, cfg) in load_rigs().items():
        for name, g in df.groupby("bearing"):
            s = window_stats(g.sort_values("snapshot")[FEATURE].to_numpy(), cfg)
            rows.append({"rig": rig, "bearing": name, "condition": int(g["condition"].iloc[0]),
                         "snapshots": len(g), "evaluable": s is not None,
                         "rise": None if s is None else round(s["rise"], 4),
                         "level": None if s is None else round(s["level"], 4)})
    t = pd.DataFrame(rows)
    ev = t[t["evaluable"]]
    for i, r in t.iterrows():
        if not r["evaluable"]:
            continue
        others = ev.drop(index=i)
        same = others[others["rig"] == r["rig"]]
        cross = others[others["rig"] != r["rig"]]
        cond = same[same["condition"] == r["condition"]]
        t.loc[i, "flag_rise_same_rig"] = outside(r["rise"], same["rise"].to_numpy())
        t.loc[i, "flag_rise_cross_rig"] = outside(r["rise"], cross["rise"].to_numpy())
        t.loc[i, "flag_level"] = outside(r["level"], cond["level"].to_numpy())
        t.loc[i, "level_refs"] = len(cond)
    for col in ("flag_rise_same_rig", "flag_rise_cross_rig", "flag_level"):
        t[col] = t[col].astype("boolean")
    t["guard_same_rig"] = t["flag_rise_same_rig"] | t["flag_level"]
    t["guard_cross_rig"] = t["flag_rise_cross_rig"] | t["flag_level"]
    t["class"] = [("hit" if (r, b) in HITS else "acceptable" if (r, b) in ACCEPTABLE else "control")
                  if e else "excluded" for r, b, e in zip(t["rig"], t["bearing"], t["evaluable"])]
    return t


def judge(t: pd.DataFrame, col: str) -> dict:
    ev = t[t["evaluable"]]
    hits = ev[ev["class"] == "hit"]
    controls = ev[ev["class"] == "control"]
    false_flags = controls[controls[col].astype(bool)]
    return {"hits_flagged": int(hits[col].astype(bool).sum()), "hits": len(hits),
            "false_flags": len(false_flags), "controls": len(controls),
            "false_flag_bearings": [f"{r} {b}" for r, b in zip(false_flags["rig"], false_flags["bearing"])],
            "promising": bool(hits[col].astype(bool).all() and len(false_flags) <= MAX_FALSE_FLAGS)}


def main() -> None:
    t = table()
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "flags.csv", index=False)
    result = {"plan_commit": "bba9343",
              "rule": f"far-out fences, Q1 - {FAR_OUT} IQR / Q3 + {FAR_OUT} IQR, leave-one-bearing-out",
              "guard_same_rig": judge(t, "guard_same_rig"),
              "guard_cross_rig": judge(t, "guard_cross_rig"),
              "single_statistics": {c: judge(t, c) for c in
                                    ("flag_rise_same_rig", "flag_rise_cross_rig", "flag_level")}}
    (OUT / "step1.json").write_text(json.dumps(result, indent=1))
    show = ["rig", "bearing", "class", "rise", "level", "level_refs", "flag_rise_same_rig",
            "flag_rise_cross_rig", "flag_level"]
    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print(t[t["evaluable"]][show].to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k.startswith("guard")}, indent=1))


if __name__ == "__main__":
    main()
