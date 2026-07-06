"""Sviluppo e validazione del rilevatore di onset CAUSALE (streaming-native).

Differenze rispetto a detect_fpt_slope (batch):
  - smoothing e pendenza TRAILING: a ogni istante t si usa solo il passato
  - soglia ASSOLUTA per condizione operativa, appresa in fit-time dalla fase
    sana dei cuscinetti di training e congelata (nessun ricalcolo in streaming)
  - persistenza + latch: N snapshot consecutivi sopra soglia fanno scattare
    l'allarme, che non si spegne piu'

Output:
  - reports/causal_detector/thresholds.json      (soglie per condizione, full-fit)
  - reports/causal_detector/onsets_summary.csv   (tabella onset causale vs batch)
  - reports/causal_detector/<bearing>.png        (grafico per cuscinetto)

Uso (dal root del repo):
    python scripts/dev_causal_detector.py --data-dir data/Learning_set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import build_feature_frame
from src.labeling import detect_fpt_slope  # batch, per confronto

# ----------------------------- parametri di design --------------------------
SKIP = 100            # transitorio iniziale scartato (C: fisso, in snapshot)
W_SMOOTH = 30         # finestra trailing per lo smoothing dell'RMS (snapshot)
W_SLOPE = 30          # finestra trailing per lo smoothing della pendenza
HEALTHY_FRAC = 0.3    # A: frazione di vita considerata sana per la baseline
K_SIGMA = 6.0         # moltiplicatore MAD per la soglia
PERSISTENCE = 5       # B: snapshot consecutivi sopra soglia per il latch
DETECTOR_FEATURE = "h_rms"

TRAIN_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]


def condition_of(bearing: str) -> int:
    return int(bearing.split("Bearing")[1].split("_")[0])


# ----------------------------- rilevatore causale ---------------------------

def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Media mobile che guarda solo indietro: out[t] = mean(x[max(0,t-w+1) : t+1])."""
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def causal_slope(rms: np.ndarray, w_smooth: int = W_SMOOTH, w_slope: int = W_SLOPE) -> np.ndarray:
    """Pendenza causale: smoothing trailing -> differenza all'indietro -> smoothing trailing."""
    sm = trailing_mean(np.asarray(rms, dtype=float), w_smooth)
    d = np.empty_like(sm)
    d[0] = 0.0
    d[1:] = np.diff(sm)
    return trailing_mean(d, w_slope)


def learn_threshold(healthy_slopes: list[np.ndarray], k_sigma: float = K_SIGMA) -> float:
    """Soglia assoluta da pendenze in fase sana (pool su piu' cuscinetti)."""
    pool = np.concatenate(healthy_slopes)
    mu = float(np.median(pool))
    mad = float(np.median(np.abs(pool - mu))) + 1e-12
    return mu + k_sigma * 1.4826 * mad


def healthy_slope_segment(rms: np.ndarray) -> np.ndarray:
    """Segmento di pendenza usato per la baseline: dopo SKIP, entro HEALTHY_FRAC di vita.

    NB: usa la lunghezza totale del cuscinetto — lecito SOLO in fit-time,
    dove le traiettorie di training sono complete e note.
    """
    n = len(rms)
    slope = causal_slope(rms)
    lo = SKIP + W_SMOOTH            # dopo il transitorio E dopo il warmup dello smoothing
    hi = max(lo + 10, int(HEALTHY_FRAC * n))
    return slope[lo:hi]


def run_causal_detector(rms: np.ndarray, threshold: float,
                        persistence: int = PERSISTENCE) -> Optional[int]:
    """Simula il rilevatore online sull'intera traiettoria, in modo causale.

    Ritorna l'indice di onset (primo snapshot della sequenza persistente) o None.
    Il latch e' implicito: ci si ferma al primo scatto.
    """
    slope = causal_slope(rms)
    warmup = SKIP + W_SMOOTH
    count = 0
    for t in range(len(rms)):
        if t < warmup:
            continue
        if slope[t] > threshold:
            count += 1
            if count >= persistence:
                return t - persistence + 1
        else:
            count = 0
    return None


# ----------------------------- validazione ----------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/Learning_set"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/causal_detector"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) carica le traiettorie RMS dei sei cuscinetti di training
    print("Estrazione feature dai cuscinetti di training (una tantum, puo' richiedere minuti)...")
    rms_by_bearing: dict[str, np.ndarray] = {}
    for b in TRAIN_BEARINGS:
        frame = build_feature_frame(args.data_dir / b)
        rms_by_bearing[b] = frame[DETECTOR_FEATURE].to_numpy()
        print(f"  {b}: {len(frame)} snapshot")

    conditions = sorted({condition_of(b) for b in TRAIN_BEARINGS})

    # 2) soglie full-fit per condizione (tutti i cuscinetti della condizione)
    thresholds_full: dict[int, float] = {}
    for c in conditions:
        segs = [healthy_slope_segment(rms_by_bearing[b]) for b in TRAIN_BEARINGS if condition_of(b) == c]
        thresholds_full[c] = learn_threshold(segs)
    print("\nSoglie full-fit per condizione:", {c: f"{v:.3e}" for c, v in thresholds_full.items()})

    # 3) LOBO: per ogni cuscinetto, soglia appresa SENZA di lui
    #    (con 2 cuscinetti/condizione => soglia da UN solo cuscinetto: limite dei dati, dichiarato)
    rows = []
    for b in TRAIN_BEARINGS:
        c = condition_of(b)
        others = [x for x in TRAIN_BEARINGS if condition_of(x) == c and x != b]
        thr_lobo = learn_threshold([healthy_slope_segment(rms_by_bearing[o]) for o in others])

        rms = rms_by_bearing[b]
        n = len(rms)
        onset_lobo = run_causal_detector(rms, thr_lobo)
        onset_full = run_causal_detector(rms, thresholds_full[c])
        onset_batch = detect_fpt_slope(rms)  # batch con fallback, come nel progetto

        rows.append({
            "bearing": b, "condition": c, "n_snapshots": n,
            "thr_lobo": thr_lobo, "thr_full": thresholds_full[c],
            "onset_causal_lobo": onset_lobo,
            "onset_causal_lobo_pct": round(100 * onset_lobo / n, 1) if onset_lobo is not None else None,
            "onset_causal_full": onset_full,
            "onset_causal_full_pct": round(100 * onset_full / n, 1) if onset_full is not None else None,
            "onset_batch": onset_batch,
            "onset_batch_pct": round(100 * onset_batch / n, 1),
        })

        # 4) grafico per cuscinetto: RMS liscia + pendenza con soglia e onset
        slope = causal_slope(rms)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax1.plot(rms, lw=0.6, alpha=0.5, label="h_rms")
        ax1.plot(trailing_mean(rms, W_SMOOTH), lw=1.4, label=f"trailing mean (w={W_SMOOTH})")
        ax1.set_ylabel("RMS")
        ax1.set_title(f"{b} (cond {c}) — onset causale LOBO vs batch")
        ax2.plot(slope, lw=0.9, label="pendenza causale")
        ax2.axhline(thr_lobo, color="tab:red", ls="--", lw=1, label=f"soglia LOBO ({thr_lobo:.2e})")
        ax2.axhline(thresholds_full[c], color="tab:orange", ls=":", lw=1, label=f"soglia full ({thresholds_full[c]:.2e})")
        ax2.set_ylabel("d(RMS)/dt")
        ax2.set_xlabel("snapshot")
        for ax in (ax1, ax2):
            if onset_lobo is not None:
                ax.axvline(onset_lobo, color="tab:red", lw=1.5, label="onset causale (LOBO)")
            ax.axvline(onset_batch, color="tab:green", lw=1.5, ls="--", label="onset batch")
            ax.axvspan(0, SKIP, color="gray", alpha=0.15)
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{b}.png", dpi=130)
        plt.close(fig)

    # 5) tabella riassuntiva + persistenza dei risultati
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "onsets_summary.csv", index=False)
    with open(args.out_dir / "thresholds.json", "w") as f:
        json.dump({
            "detector_feature": DETECTOR_FEATURE,
            "skip": SKIP, "w_smooth": W_SMOOTH, "w_slope": W_SLOPE,
            "healthy_frac": HEALTHY_FRAC, "k_sigma": K_SIGMA, "persistence": PERSISTENCE,
            "thresholds_by_condition": {str(c): thresholds_full[c] for c in conditions},
        }, f, indent=2)

    cols = ["bearing", "onset_causal_lobo_pct", "onset_causal_full_pct", "onset_batch_pct"]
    print("\n" + summary[cols].to_string(index=False))
    print(f"\nGrafici e tabella in: {args.out_dir}")
    print("Onset 'None' = il rilevatore causale non e' MAI scattato su quel cuscinetto.")


if __name__ == "__main__":
    main()