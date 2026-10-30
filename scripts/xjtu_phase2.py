"""Phase 2 on XJTU-SY (reports/xjtu_phase2/PREREGISTRATION.md). Two steps, two commits:

    python -m scripts.xjtu_phase2 develop   # choose on the 12 transfer bearings, writes artifacts/xjtu/
    python -m scripts.xjtu_phase2 vault     # once, after artifacts/xjtu/ is committed

develop: reports/xjtu_phase2/development*.csv, development.json, artifacts/xjtu/{detector,stage2}.json
vault:   reports/xjtu_phase2/vault_bearings.csv, vault.json (refuses to overwrite them)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict

import numpy as np
import pandas as pd

from src import config
from src.causal_detector import CausalDetectorConfig
from src.xjtu import TRANSFER, VAULT
from src.xjtu_phase2 import (choose_quantile, config_for, countdown_s, curve, export_stage2, fit_quantile_pipeline,
                             grid_table, json_predict, key, lobo_detector, lobo_stage2, outcome, predict, rank, scores,
                             select)

ROOT = config.PROJECT_ROOT
REPORTS = ROOT / "reports" / "xjtu_phase2"
ARTIFACTS = ROOT / "artifacts" / "xjtu"
FEATURES_DIR = config.DATA_ROOT / "xjtu_features"
PREREG = "reports/xjtu_phase2/PREREGISTRATION.md"


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=False)


def committed(path: str) -> bool:
    """Tracked and identical to the last commit."""
    return git("ls-files", "--error-unmatch", path).returncode == 0 and git("diff", "--quiet", "HEAD", "--", path).returncode == 0


def load(name: str, expected: list[str]) -> pd.DataFrame:
    df = pd.read_csv(FEATURES_DIR / name)
    if set(df["bearing"]) != set(expected):
        raise SystemExit(f"{name}: expected {sorted(expected)}, got {sorted(set(df['bearing']))}")
    return df


def develop() -> None:
    if not committed(PREREG):
        print(f"WARNING: {PREREG} is not committed; the run is not pre-registered")
    df = load("features_transfer.csv", TRANSFER)
    bearings = sorted(TRANSFER)

    table = grid_table(df)
    ranking = rank(table, bearings)
    chosen = select(table, bearings)
    cfg = config_for(chosen)
    in_sample = table[table["config"] == key(chosen)].set_index("bearing")
    lobo_det = lobo_detector(table, bearings)

    curves = {b: c for b, g in df.groupby("bearing", sort=True) if (c := curve(g, cfg)) is not None}
    q, qtable = choose_quantile(curves)
    model = fit_quantile_pipeline(list(curves.values()), q)
    exported = export_stage2(model, q)
    worst = max(float(np.max(np.abs(json_predict(exported, c) - predict(model, c)))) for c in curves.values())
    if worst > 1e-6:
        raise SystemExit(f"exported stage 2 differs from the fitted pipeline by {worst} s")
    cd = countdown_s(curves)
    lobo_s2 = lobo_stage2(curves)

    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "development_grid.csv", index=False)
    ranking.to_csv(REPORTS / "development_ranking.csv")
    lobo_det.to_csv(REPORTS / "development_lobo_detector.csv", index=False)
    lobo_s2.to_csv(REPORTS / "development_lobo_stage2.csv", index=False)
    (ARTIFACTS / "detector.json").write_text(json.dumps(asdict(cfg), indent=1) + "\n")
    (ARTIFACTS / "stage2.json").write_text(json.dumps(exported | {
        "countdown_s": cd, "training": {"bearings": sorted(curves), "quantile_selection": qtable}}, indent=1) + "\n")
    summary = {
        "preregistration_commit": git("log", "-1", "--format=%h %cI", "--", PREREG).stdout.strip() or None,
        "chosen_detector": chosen,
        "useful_in_sample": int(in_sample["useful"].sum()),
        "useful_lobo": int(lobo_det["useful"].sum()),
        "bearings": len(bearings),
        "revoked_in_sample": int(in_sample["revoked"].sum()),
        "stage2": {"quantile": q, "curves": len(curves), "countdown_s": cd,
                   "lobo_beats_countdown": int(lobo_s2["beats_countdown"].sum()),
                   "lobo_median_mae_over_rul": float(lobo_s2["mae_over_rul"].median())},
    }
    (REPORTS / "development.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("top of the ranking (12 bearings):\n", ranking.head(8).to_string())
    print("\nchosen:", key(chosen), "\n", in_sample[["monitored", "alarm_at_end", "first_alarm_pct",
                                                      "final_onset_pct", "warning_min", "revoked", "useful"]].to_string())
    print("\nleave-one-bearing-out (detector):\n", lobo_det[["bearing", "config", "useful"]].to_string(index=False))
    print("\nstage 2, quantile selection:", qtable, "\nleave-one-bearing-out (stage 2):\n", lobo_s2.round(3).to_string(index=False))
    print("\n", json.dumps(summary, indent=2))
    print("\nNext: commit artifacts/xjtu/ and reports/xjtu_phase2/, then run the vault step once.")


def vault(allow_uncommitted: bool) -> None:
    out = REPORTS / "vault.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the vault has been opened already")
    for p in ("artifacts/xjtu/detector.json", "artifacts/xjtu/stage2.json"):
        if not committed(p) and not allow_uncommitted:
            raise SystemExit(f"{p} is not committed: commit the development step before opening the vault")
    cfg = CausalDetectorConfig.from_dict(json.loads((ARTIFACTS / "detector.json").read_text()))
    s2 = json.loads((ARTIFACTS / "stage2.json").read_text())
    df = load("features_vault.csv", VAULT)

    rows = []
    for b, g in df.groupby("bearing", sort=True):
        g = g.sort_values("snapshot")
        r = {"bearing": b} | outcome(g[cfg.feature].to_numpy(), cfg)
        c = curve(g, cfg, s2["smooth"])
        if c is not None:
            t = c["true_rul"].to_numpy()
            r |= scores(t, json_predict(s2, c), np.maximum(s2["countdown_s"] - c["tau"].to_numpy(), 0.0))
        rows.append(r)
    table = pd.DataFrame(rows)
    alarmed = table[table["alarm_at_end"]]
    decision = {
        "detector_useful": int(table["useful"].sum()),
        "detector_passes": bool(table["useful"].sum() >= 2),
        "stage2_alarmed": len(alarmed),
        "stage2_beats_countdown": int(alarmed["beats_countdown"].sum()) if len(alarmed) else 0,
        "stage2_median_mae_over_rul": float(alarmed["mae_over_rul"].median()) if len(alarmed) else None,
    }
    decision["stage2_passes"] = bool(len(alarmed) and decision["stage2_beats_countdown"] > len(alarmed) / 2
                                     and decision["stage2_median_mae_over_rul"] <= 1.0)
    table.to_csv(REPORTS / "vault_bearings.csv", index=False)
    out.write_text(json.dumps({"detector": asdict(cfg), "decision": decision}, indent=2) + "\n")
    print(table.round(3).to_string(index=False))
    print(json.dumps(decision, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["develop", "vault"])
    ap.add_argument("--allow-uncommitted", action="store_true", help=argparse.SUPPRESS)   # tests only
    args = ap.parse_args()
    develop() if args.step == "develop" else vault(args.allow_uncommitted)


if __name__ == "__main__":
    main()
