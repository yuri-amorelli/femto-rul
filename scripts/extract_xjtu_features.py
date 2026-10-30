"""Extract per-snapshot features from the raw XJTU-SY files. Prints counts only, no values.

Run on the machine that holds the data, after reports/xjtu_transfer/PREREGISTRATION.md is
committed:

    python -m scripts.extract_xjtu_features            # reads data/XJTU-SY, writes data/xjtu_features/

Writes two files (inside data/, so never committed):
    features_transfer.csv   the 12 bearings of the transfer test
    features_vault.csv      the 3 vault bearings: keep it closed until the final test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src import config
from src.features import snapshot_features
from src.xjtu import BEARINGS, FS, SNAPSHOT_LEN, VAULT, condition, find_bearings, load_snapshot, snapshot_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(config.DATA_ROOT / "XJTU-SY"))
    ap.add_argument("--out", default=str(config.DATA_ROOT / "xjtu_features"))
    args = ap.parse_args()

    folders = find_bearings(args.root)
    missing = sorted(set(BEARINGS) - set(folders))
    if missing:
        raise SystemExit(f"bearings not found under {args.root}: {missing}")

    rows = []
    for name, folder in folders.items():
        t0 = time.time()
        files = snapshot_files(folder)
        expected = BEARINGS[name][0]
        note = "" if len(files) == expected else f"  (dataset table says {expected})"
        short = 0
        for i, f in enumerate(files):
            x = load_snapshot(f)
            short += x.shape[0] != SNAPSHOT_LEN
            rows.append({"bearing": name, "condition": condition(name), "snapshot": i} | snapshot_features(x, fs=FS))
        flag = f"  {short} file(s) not {SNAPSHOT_LEN} samples long" if short else ""
        print(f"{name}: {len(files)} snapshots in {time.time() - t0:.0f} s{note}{flag}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df[~df["bearing"].isin(VAULT)].to_csv(out / "features_transfer.csv", index=False)
    df[df["bearing"].isin(VAULT)].to_csv(out / "features_vault.csv", index=False)
    print(f"\nwritten to {out}: features_transfer.csv (upload this one), features_vault.csv (keep closed)")


if __name__ == "__main__":
    main()
