"""Client di test: streamma un cuscinetto reale attraverso l'API snapshot per snapshot.

Simula il pattern stateless: a ogni chiamata invia l'ultimo snapshot grezzo +
la feature_history ricevuta dalla risposta precedente. Atteso: rul_seconds=null
per la gran parte della vita, allarme verso fine vita, poi RUL numerica.

Uso (dal root del repo, con l'API gia' in ascolto):
    python scripts/api_test_client.py --bearing-dir data/Test_set/Bearing1_3 --condition 1
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"


def load_snapshots(bearing_dir: Path) -> list[tuple[list[float], list[float]]]:
    """Carica gli snapshot acc_*.csv; ultime due colonne = canali h, v (come nel data loader del repo)."""
    files = sorted(bearing_dir.glob("acc_*.csv"))
    if not files:
        sys.exit(f"Nessun file acc_*.csv in {bearing_dir}")
    snapshots = []
    print(f"Carico {len(files)} snapshot da {bearing_dir.name} (puo' richiedere ~1 minuto)...", flush=True)
    for f in files:
        df = pd.read_csv(f, header=None)
        h = df.iloc[:, -2].astype(float).tolist()
        v = df.iloc[:, -1].astype(float).tolist()
        snapshots.append((h, v))
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bearing-dir", type=Path, required=True)
    parser.add_argument("--condition", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--every", type=int, default=1, help="processa 1 snapshot ogni N (per test rapidi)")
    args = parser.parse_args()

    health = requests.get(f"{API_URL}/health", timeout=10).json()
    print(f"health: {health}")
    if health.get("status") != "ok":
        sys.exit("API non pronta")

    snapshots = load_snapshots(args.bearing_dir)[:: args.every]
    print(f"{len(snapshots)} snapshot da processare\n")

    history: list[list[float]] = []
    first_alarm_at = None
    t0 = time.time()

    for i, (h, v) in enumerate(snapshots):
        resp = requests.post(
            f"{API_URL}/predict",
            json={
                "operating_condition": args.condition,
                "snapshot_h": h,
                "snapshot_v": v,
                "feature_history": history,
            },
            timeout=30,
        )
        resp.raise_for_status()
        out = resp.json()
        history = out["feature_history"]

        if out["alarm"] and first_alarm_at is None:
            first_alarm_at = i
            pct = 100 * i / len(snapshots)
            print(f">>> ALLARME al snapshot {i}/{len(snapshots)} ({pct:.0f}% della traiettoria), onset={out['alarm_onset_index']}")

        if out["alarm"] or i % 50 == 0:
            rul = out["rul_seconds"]
            rul_str = f"{rul:8.1f}s" if rul is not None else "    null"
            print(f"snapshot {i:4d} | state={out['alarm_state']:11s} | rul={rul_str} | revocati={out['revoked_alarms']}")

    dt = time.time() - t0
    print(f"\nCompletato in {dt:.1f}s ({dt/len(snapshots)*1000:.0f} ms/snapshot)")
    if first_alarm_at is None:
        print("Nessun allarme sull'intera traiettoria (atteso per morfologie tipo Bearing2_3).")

if __name__ == "__main__":
    main()