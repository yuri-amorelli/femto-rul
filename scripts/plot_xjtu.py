"""Degradation curves of the 15 XJTU-SY bearings with the phase-2 detector's states.

Reads both feature files (the vault was opened in phase 2, so all 15 can be shown) and the
detector chosen in phase 2 (artifacts/xjtu/detector.json). Writes
reports/xjtu_phase2/degradation.png.

    python -m scripts.plot_xjtu
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.causal_detector import CausalDetectorConfig, alarm_trace  # noqa: E402
from src.xjtu import BEARINGS, CONDITIONS, SNAPSHOT_PERIOD, VAULT, snapshots_to_monitor  # noqa: E402

FEATURES = config.DATA_ROOT / "xjtu_features"
OUT = config.PROJECT_ROOT / "reports" / "xjtu_phase2" / "degradation.png"
COLORS = {"calibrating": "#9a9893", "warning": "#fab219", "alarm": "#d03b3b"}
ALPHA = {"calibrating": 0.18, "warning": 0.30, "alarm": 0.22}


def states(rms: np.ndarray, cfg: CausalDetectorConfig) -> np.ndarray:
    """Per-snapshot state, as the API would report it."""
    n = len(rms)
    alarm, _, _ = alarm_trace(rms, cfg)
    revoked_so_far = np.cumsum(np.r_[False, alarm[:-1] & ~alarm[1:]])
    out = np.where(alarm, "alarm", np.where(revoked_so_far > 0, "warning", "monitoring")).astype(object)
    out[: min(n, snapshots_to_monitor(cfg) - 1)] = "calibrating"
    return out


def spans(s: np.ndarray):
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            yield s[start], start, i
            start = i


def main() -> None:
    df = pd.concat([pd.read_csv(FEATURES / "features_transfer.csv"), pd.read_csv(FEATURES / "features_vault.csv")])
    cfg = CausalDetectorConfig.from_dict(json.loads((config.PROJECT_ROOT / "artifacts" / "xjtu" / "detector.json")
                                                    .read_text()))
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#b5b3ad", "axes.linewidth": 0.6})
    fig, axes = plt.subplots(5, 3, figsize=(13, 14), constrained_layout=True)
    for col, c in enumerate(sorted(CONDITIONS)):
        for row in range(5):
            name = f"Bearing{c}_{row + 1}"
            ax = axes[row, col]
            g = df[df["bearing"] == name].sort_values("snapshot")
            t = g["snapshot"].to_numpy() * SNAPSHOT_PERIOD / 3600
            h, v = g["h_rms"].to_numpy(), g["v_rms"].to_numpy()
            for state, a, b in spans(states(h, cfg)):
                if state in COLORS:
                    ax.axvspan(t[a], t[b - 1] + SNAPSHOT_PERIOD / 3600, color=COLORS[state], alpha=ALPHA[state], lw=0)
            ax.plot(t, v, color="#2a78d6", lw=0.7, label="vertical RMS")
            ax.plot(t, h, color="#0b0b0b", lw=0.8, label="horizontal RMS (detector input)")
            ax.set_yscale("log")
            ax.set_xlim(0, t[-1] + SNAPSHOT_PERIOD / 3600)
            n, fault = BEARINGS[name]
            life = f"{n // 60} h {n % 60} min" if n >= 60 else f"{n} min"
            ax.set_title(f"{name}{'  (vault)' if name in VAULT else ''}\n{fault}, {life}", loc="left", fontsize=8.5)
            ax.grid(axis="y", color="#e6e5e1", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 4:
                ax.set_xlabel("time (h)")
            if col == 0:
                ax.set_ylabel("RMS (g, log scale)")
        axes[0, col].annotate(f"{CONDITIONS[c]['rpm']} rpm, {CONDITIONS[c]['load_kN']:.0f} kN",
                              xy=(0.5, 1.32), xycoords="axes fraction", ha="center", fontsize=10, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s], alpha=ALPHA[s] + 0.2) for s in COLORS]
    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines + handles, labels + ["calibrating", "warning (alarm revoked)", "alarm"],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.025), fontsize=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
