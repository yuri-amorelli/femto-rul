"""Extract a few raw XJTU-SY snapshots per bearing for the envelope analysis.

Rules: reports/envelope/PLAN.md. Run on the machine that holds the raw data, after the plan
is committed:

    python -m scripts.extract_xjtu_raw        # reads data/XJTU-SY and data/xjtu_features/

For every bearing it keeps the snapshots at 10 %, 50 % and 90 % of life, the start of the
alarm still active at the end of life (phase-2 detector, if there is one) and the last
snapshot, both channels, float32. Writes data/xjtu_raw/snapshots.npz (gitignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.causal_detector import CausalDetectorConfig, alarm_trace
from src.xjtu import BEARINGS, find_bearings, load_snapshot, snapshot_files

FRACTIONS = (0.10, 0.50, 0.90)


def chosen_snapshots(n: int, onset: int | None) -> dict[str, int]:
    picks = {f"life_{int(round(100 * f))}": int(round(f * (n - 1))) for f in FRACTIONS}
    if onset is not None:
        picks["alarm_onset"] = onset
    picks["last"] = n - 1
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(config.DATA_ROOT / "XJTU-SY"))
    ap.add_argument("--out", default=str(config.DATA_ROOT / "xjtu_raw" / "snapshots.npz"))
    args = ap.parse_args()

    feats = pd.concat([pd.read_csv(config.DATA_ROOT / "xjtu_features" / f)
                       for f in ("features_transfer.csv", "features_vault.csv")])
    cfg = CausalDetectorConfig.from_dict(
        json.loads((config.PROJECT_ROOT / "artifacts" / "xjtu" / "detector.json").read_text()))
    folders = find_bearings(Path(args.root))

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for name in sorted(BEARINGS):
        files = snapshot_files(folders[name])
        rms = feats[feats["bearing"] == name].sort_values("snapshot")["h_rms"].to_numpy()
        if len(rms) != len(files):
            raise SystemExit(f"{name}: {len(files)} files but {len(rms)} feature rows")
        alarm, onset, _ = alarm_trace(rms, cfg)
        picks = chosen_snapshots(len(files), int(onset[-1]) if alarm[-1] else None)
        for label, i in picks.items():
            arrays[f"{name}__{label}"] = load_snapshot(files[i]).astype(np.float32)
        meta[name] = {"snapshots": len(files), "fault": BEARINGS[name][1], "picks": picks}
        print(f"{name}: {picks}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    print(f"\nwritten {out} ({out.stat().st_size / 1e6:.1f} MB): upload this file")


if __name__ == "__main__":
    main()
