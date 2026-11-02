"""Envelope analysis on XJTU-SY: which component is failing? Rules: reports/envelope/PLAN.md.

    python -m scripts.extract_xjtu_raw      # data/xjtu_raw/snapshots.npz
    python -m scripts.envelope_eval         # reports/envelope/diagnoses.csv, results.json, spectra.png

Written and committed before any raw signal was analysed.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.envelope import COMPONENTS, diagnose, envelope_spectrum, fault_frequencies  # noqa: E402
from src.xjtu import CONDITIONS, FS, condition  # noqa: E402

RAW = config.DATA_ROOT / "xjtu_raw" / "snapshots.npz"
OUT = config.PROJECT_ROOT / "reports" / "envelope"
SCORED = {  # single documented fault (PLAN.md)
    "Bearing1_1": "outer race", "Bearing1_2": "outer race", "Bearing1_3": "outer race",
    "Bearing2_2": "outer race", "Bearing2_4": "outer race", "Bearing2_5": "outer race",
    "Bearing3_1": "outer race", "Bearing3_5": "outer race",
    "Bearing2_1": "inner race", "Bearing3_3": "inner race", "Bearing3_4": "inner race",
    "Bearing1_4": "cage", "Bearing2_3": "cage",
}
FIXED_BAND = (2_000.0, 10_000.0)
CHANNELS = {"h": 0, "v": 1}
FIGURE = ["Bearing1_1", "Bearing2_1", "Bearing2_3"]   # one per documented component, chosen in advance


def shaft_hz(bearing: str) -> float:
    return CONDITIONS[condition(bearing)]["rpm"] / 60


def table(raw) -> pd.DataFrame:
    meta = json.loads(str(raw["meta"]))
    rows = []
    for bearing, m in meta.items():
        for label, idx in m["picks"].items():
            x = raw[f"{bearing}__{label}"]
            for ch, col in CHANNELS.items():
                for band_rule, fixed in (("kurtosis", None), ("fixed", FIXED_BAND)):
                    d = diagnose(x[:, col], FS, shaft_hz(bearing), fixed)
                    rows.append({"bearing": bearing, "fault": m["fault"], "scored": bearing in SCORED,
                                 "snapshot": label, "index": idx, "pct_of_life": round(100 * idx / m["snapshots"], 1),
                                 "channel": ch, "band_rule": band_rule,
                                 "band_lo": round(d.band[0]), "band_hi": round(d.band[1]),
                                 "verdict": d.verdict, "margin": round(d.margin, 3),
                                 "correct": (d.verdict == SCORED[bearing]) if bearing in SCORED else None}
                                | {f"score_{c.replace(' ', '_')}": round(d.scores[c], 3) for c in COMPONENTS})
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame, snapshot: str) -> dict:
    s = t[(t["snapshot"] == snapshot) & (t["channel"] == "h") & (t["band_rule"] == "kurtosis") & t["scored"]]
    by_fault = {f: f"{int(g['correct'].sum())} of {len(g)}" for f, g in s.groupby("fault")}
    return {"correct": int(s["correct"].sum()), "of": len(s), "by_fault": by_fault,
            "wrong": {b: v for b, v, c in zip(s["bearing"], s["verdict"], s["correct"]) if not c}}


def figure(raw) -> None:
    fig, axes = plt.subplots(len(FIGURE), 2, figsize=(12, 8), constrained_layout=True)
    colors = {"outer race": "#d03b3b", "inner race": "#2a78d6", "cage": "#1f9e6b"}
    for row, bearing in enumerate(FIGURE):
        fr = shaft_hz(bearing)
        ff = fault_frequencies(fr)
        for col, label in enumerate(("life_10", "last")):
            x = raw[f"{bearing}__{label}"][:, 0]
            d = diagnose(x, FS, fr)
            f, s = envelope_spectrum(x, FS, d.band)
            ax = axes[row, col]
            keep = f <= 600
            ax.plot(f[keep], s[keep], color="#0b0b0b", lw=0.7)
            for comp, c in colors.items():
                for k in (1, 2, 3):
                    ax.axvline(k * ff[comp], color=c, lw=0.8, ls="--", alpha=0.7,
                               label=comp if (k == 1 and row == 0 and col == 0) else None)
            ax.set_title(f"{bearing} ({SCORED[bearing]}), {'10 % of life' if label == 'life_10' else 'end of life'}: "
                         f"verdict {d.verdict}", loc="left", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if row == len(FIGURE) - 1:
                ax.set_xlabel("envelope frequency (Hz)")
    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(OUT / "spectra.png", dpi=130, bbox_inches="tight")


def main() -> None:
    raw = np.load(RAW)
    t = table(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "diagnoses.csv", index=False)
    primary = summary(t, "last")
    passed = primary["correct"] >= 10 and int(primary["by_fault"]["inner race"].split()[0]) >= 2
    result = {"plan": "reports/envelope/PLAN.md", "baseline_always_outer": "8 of 13",
              "primary_last_snapshot": primary | {"passes": passed},
              "secondary_life_90": summary(t, "life_90"),
              "secondary_alarm_onset": summary(t, "alarm_onset")}
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    figure(raw)
    view = t[(t["channel"] == "h") & (t["band_rule"] == "kurtosis")]
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(view[["bearing", "fault", "snapshot", "pct_of_life", "band_lo", "band_hi", "verdict", "margin",
                    "correct"]].to_string(index=False))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
