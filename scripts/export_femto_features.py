"""Export per-snapshot features of all 17 FEMTO-ST bearings to one CSV.

Used by the baseline-guard exploration (reports/baseline_guard/PLAN.md), which runs on
data already seen. Reads the 6 training bearings from Learning_set and the 11 test
bearings from Full_Test_Set (recorded to the end).

    python -m scripts.export_femto_features --cache

Writes data/femto_features/features_all.csv (inside data/, so never committed).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.pipeline import prepare_bearing
from src.post_alarm import TRAIN_BEARINGS
from src.test_protocol import TEST_BEARINGS

COLUMNS = [f"{c}_{f}" for c in "hv" for f in ("rms", "peak", "kurtosis", "crest_factor")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="use cached features in results/ if present")
    args = ap.parse_args()

    frames = []
    for set_name, names in (("Learning_set", TRAIN_BEARINGS), ("Full_Test_Set", TEST_BEARINGS)):
        for n in names:
            df, _ = prepare_bearing(config.DATA_ROOT / set_name / n, label_mode="linear", cache=args.cache)
            out = df[COLUMNS].copy()
            out.insert(0, "snapshot", df["snapshot_index"].to_numpy())
            out.insert(0, "condition", config.condition_of(n))
            out.insert(0, "bearing", n)
            frames.append(out)
            print(f"{set_name}/{n}: {len(out)} snapshots")

    path = config.DATA_ROOT / "femto_features" / "features_all.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
