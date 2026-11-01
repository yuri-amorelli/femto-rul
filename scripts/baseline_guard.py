"""Baseline-guard exploration, step 1: can other bearings flag an unhealthy calibration window?

Rules fixed in reports/baseline_guard/PLAN.md (committed in bba9343 before any number was
computed). Exploratory, on data already seen: FEMTO-ST (17 bearings) and XJTU-SY (15).

    python -m scripts.export_femto_features --cache      # data/femto_features/features_all.csv
    python -m scripts.baseline_guard                     # step 1: flags.csv, step1.json
    python -m scripts.baseline_guard --step2             # step 2: step2.csv, step2.json

Step 2 (the fallback threshold) follows the rule added to PLAN.md in d564124, before it ran.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.causal_detector import CausalDetectorConfig, calibrated_threshold
from src.xjtu_phase2 import outcome

OUT = config.PROJECT_ROOT / "reports" / "baseline_guard"
FEATURE = "h_rms"
FAR_OUT = 3.0   # Tukey far-out fences: Q1 - 3 IQR, Q3 + 3 IQR

# classes written in PLAN.md before computing
HITS = {("XJTU-SY", "Bearing1_2"), ("XJTU-SY", "Bearing3_5")}
ACCEPTABLE = {("XJTU-SY", "Bearing1_1"), ("XJTU-SY", "Bearing1_3"), ("XJTU-SY", "Bearing2_2"),
              ("FEMTO-ST", "Bearing2_5")}
MAX_FALSE_FLAGS = 2
PERIOD_S = {"FEMTO-ST": 10, "XJTU-SY": 60}
GUARD = "guard_same_rig"   # step 2 uses the same-rig guard, as written in PLAN.md


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


def fallback_thresholds(t: pd.DataFrame, rigs: dict) -> pd.DataFrame:
    """Step 2: for every flagged bearing, a threshold built only from other bearings' windows.
    relative threshold = learnt threshold / median window RMS, over the unflagged bearings of
    the same rig; class level = median window RMS over the unflagged bearings of the same rig
    and condition; fallback = median relative threshold x class level."""
    ev = t[t["evaluable"]].copy()
    ev["flagged"] = ev[GUARD].astype(bool)
    learnt, med = {}, {}
    for i, r in ev.iterrows():
        df, cfg = rigs[r["rig"]]
        rms = df[df["bearing"] == r["bearing"]].sort_values("snapshot")[FEATURE].to_numpy()
        a, b = window(cfg)
        learnt[i], med[i] = calibrated_threshold(rms, cfg), float(np.median(rms[a:b]))
    ev["learnt_threshold"], ev["window_median"] = pd.Series(learnt), pd.Series(med)
    ev["relative_threshold"] = ev["learnt_threshold"] / ev["window_median"]
    rows = []
    for i, r in ev[ev["flagged"]].iterrows():
        refs = ev[(ev.index != i) & (ev["rig"] == r["rig"]) & ~ev["flagged"]]
        cls = refs[refs["condition"] == r["condition"]]
        rel, level = float(refs["relative_threshold"].median()), float(cls["window_median"].median())
        rows.append({"rig": r["rig"], "bearing": r["bearing"], "class": r["class"],
                     "learnt_threshold": r["learnt_threshold"], "relative_refs": len(refs),
                     "median_relative_threshold": rel, "class_level_refs": len(cls),
                     "class_level": level, "own_window_median": r["window_median"],
                     "fallback_threshold": rel * level})
    return pd.DataFrame(rows)


def with_warning(o: dict, n: int, period_s: int) -> dict:
    """outcome() reports the warning in XJTU minutes; recompute it with the rig's period."""
    final = o["final_onset_pct"]
    o = dict(o)
    o["warning_min"] = None if final is None else (n - 1 - round(final * n / 100)) * period_s / 60
    return o


def step2() -> None:
    rigs = load_rigs()
    t = table()
    fb = fallback_thresholds(t, rigs)
    rows = []
    for _, r in fb.iterrows():
        df, cfg = rigs[r["rig"]]
        rms = df[df["bearing"] == r["bearing"]].sort_values("snapshot")[FEATURE].to_numpy()
        for variant, thr in (("own", None), ("fallback", float(r["fallback_threshold"]))):
            o = with_warning(outcome(rms, cfg, thr), len(rms), PERIOD_S[r["rig"]])
            rows.append({"rig": r["rig"], "bearing": r["bearing"], "class": r["class"], "variant": variant,
                         "threshold": r["learnt_threshold"] if thr is None else thr} | o)
    res = pd.DataFrame(rows)
    # unflagged bearings are untouched: the guard passes no threshold, so outcome() is the learnt one
    ev = t[t["evaluable"]]
    for _, r in ev[~ev[GUARD].astype(bool)].iterrows():
        df, cfg = rigs[r["rig"]]
        rms = df[df["bearing"] == r["bearing"]].sort_values("snapshot")[FEATURE].to_numpy()
        assert outcome(rms, cfg, None) == outcome(rms, cfg), r["bearing"]
    fb_ = res[res["variant"] == "fallback"].set_index(["rig", "bearing"])
    own = res[res["variant"] == "own"].set_index(["rig", "bearing"])
    hits = [k for k in fb_.index if fb_.loc[k, "class"] == "hit"]
    others = [k for k in fb_.index if fb_.loc[k, "class"] != "hit"]
    lost = [f"{r} {b}" for r, b in others if own.loc[(r, b), "useful"] and not fb_.loc[(r, b), "useful"]]
    verdict = {"plan_commit": "d564124", "guard": GUARD,
               "hits_useful_with_fallback": [f"{r} {b}" for r, b in hits if fb_.loc[(r, b), "useful"]],
               "hits": [f"{r} {b}" for r, b in hits],
               "other_flagged_that_lost_usefulness": lost,
               "unflagged_identical": True,
               "promising": bool(all(fb_.loc[k, "useful"] for k in hits) and not lost)}
    OUT.mkdir(parents=True, exist_ok=True)
    fb.to_csv(OUT / "step2_thresholds.csv", index=False)
    res.to_csv(OUT / "step2.csv", index=False)
    (OUT / "step2.json").write_text(json.dumps(verdict, indent=1))
    with pd.option_context("display.width", 220):
        print(fb.round(5).to_string(index=False))
        print(res.round(2).to_string(index=False))
    print(json.dumps(verdict, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2", action="store_true", help="run step 2 (the fallback threshold)")
    if ap.parse_args().step2:
        step2()
        return
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
