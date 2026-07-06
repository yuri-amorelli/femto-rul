"""Rilevatore di onset causale v2: AUTO-CALIBRANTE per cuscinetto.

Differenza chiave rispetto alla v1 (soglia assoluta per condizione, appresa
sui cuscinetti di training): la scala della pendenza sana e' una proprieta'
del singolo cuscinetto, non della condizione operativa — la v1 falliva in
LOBO proprio per questo. Nella v2 ogni cuscinetto calibra la PROPRIA soglia
sulla sua fase iniziale (garantita sana, e nel passato: quindi causale):

  fase 1  [0, skip)                        : transitorio scartato, nessun output
  fase 2  [skip+w_smooth, +calib_len)      : calibrazione — accumula la pendenza,
                                             congela thr = mediana + k*1.4826*MAD
  fase 3  da fine calibrazione in poi      : rilevazione, persistenza, latch

In fit-time non si apprendono piu' soglie ma solo gli IPERPARAMETRI di design
(k_sigma, calib_len, persistence), validati sui sei cuscinetti di training.

Output in reports/causal_detector_v2/: un PNG per cuscinetto, la tabella
onsets_summary.csv con lo sweep su k_sigma (onset finale + n. allarmi revocati), e detector_config.json.

Uso (dal root del repo):
    python -u -m scripts.dev_causal_detector_v2 --data-dir data/Learning_set
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
SKIP = 100            # transitorio iniziale scartato (snapshot)
W_SMOOTH = 30         # finestra trailing smoothing RMS
W_SLOPE = 30          # finestra trailing smoothing pendenza
CALIB_LEN = 150       # ampiezza finestra di calibrazione (snapshot)
PERSISTENCE = 5
REVOKE_AFTER = 30      # snapshot consecutivi sotto soglia per revocare un allarme       # snapshot consecutivi sopra soglia per il latch
K_SIGMA_SWEEP = [6.0, 8.0, 10.0]
K_SIGMA_DEFAULT = 6.0
DETECTOR_FEATURE = "h_rms"

TRAIN_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"]


def condition_of(bearing: str) -> int:
    return int(bearing.split("Bearing")[1].split("_")[0])


# ----------------------------- primitive causali ----------------------------

def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Media mobile che guarda solo indietro: out[t] = mean(x[max(0,t-w+1) : t+1])."""
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t - w + 1)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def causal_slope(rms: np.ndarray, w_smooth: int = W_SMOOTH, w_slope: int = W_SLOPE) -> np.ndarray:
    sm = trailing_mean(np.asarray(rms, dtype=float), w_smooth)
    d = np.empty_like(sm)
    d[0] = 0.0
    d[1:] = np.diff(sm)
    return trailing_mean(d, w_slope)


# ----------------------------- rilevatore v2 --------------------------------

def run_selfcal_detector(
    rms: np.ndarray,
    k_sigma: float = K_SIGMA_DEFAULT,
    skip: int = SKIP,
    calib_len: int = CALIB_LEN,
    w_smooth: int = W_SMOOTH,
    w_slope: int = W_SLOPE,
    persistence: int = PERSISTENCE,
    revoke_after: int = REVOKE_AFTER,
) -> tuple[list[tuple[int, int]], float, int]:
    """Rilevatore causale auto-calibrante con allarme REVOCABILE.

    Macchina a stati: OFF -> ON dopo `persistence` snapshot consecutivi sopra
    soglia; ON -> OFF (revoca) dopo `revoke_after` consecutivi sotto soglia.
    Una salita transitoria (gobba di meta' vita) produce un allarme che si
    auto-revoca quando rientra; il crollo terminale non rientra mai, quindi
    l'ultimo segmento resta aperto fino a fine traiettoria.

    Ritorna (segmenti_allarme [(start, end)], soglia, fine_calibrazione).
    L'ultimo segmento con end == len(rms)-1 e' l'allarme attivo a fine storia.
    Interamente causale: a ogni t usa solo campioni <= t, quindi il ricalcolo
    sull'intera history in un'API stateless equivale al vero streaming.
    """
    rms = np.asarray(rms, dtype=float)
    n = len(rms)
    slope = causal_slope(rms, w_smooth, w_slope)

    calib_start = skip + w_smooth
    calib_end = calib_start + calib_len
    if n <= calib_end + persistence:
        return [], float("nan"), calib_end  # storia insufficiente: "calibrating"

    calib = slope[calib_start:calib_end]
    mu = float(np.median(calib))
    mad = float(np.median(np.abs(calib - mu))) + 1e-12
    thr = mu + k_sigma * 1.4826 * mad

    segments: list[tuple[int, int]] = []
    on, up, down, start = False, 0, 0, 0
    for t in range(calib_end, n):
        above = slope[t] > thr
        if not on:
            up = up + 1 if above else 0
            if up >= persistence:
                on, start, down = True, t - persistence + 1, 0
        else:
            down = down + 1 if not above else 0
            if down >= revoke_after:
                segments.append((start, t))
                on, up = False, 0
    if on:
        segments.append((start, n - 1))
    return segments, thr, calib_end


# ----------------------------- validazione ----------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/Learning_set"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/causal_detector_v2"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Estrazione feature dai cuscinetti di training...")
    rms_by_bearing: dict[str, np.ndarray] = {}
    for b in TRAIN_BEARINGS:
        frame = build_feature_frame(args.data_dir / b)
        rms_by_bearing[b] = frame[DETECTOR_FEATURE].to_numpy()
        print(f"  {b}: {len(frame)} snapshot")

    rows = []
    for b in TRAIN_BEARINGS:
        rms = rms_by_bearing[b]
        n = len(rms)
        row: dict = {"bearing": b, "condition": condition_of(b), "n_snapshots": n,
                     "onset_batch_pct": round(100 * detect_fpt_slope(rms) / n, 1)}

        for k in K_SIGMA_SWEEP:
            segs, thr, calib_end = run_selfcal_detector(rms, k_sigma=k)
            final = segs[-1] if segs and segs[-1][1] == n - 1 else None
            row[f"onset_k{k:g}_pct"] = round(100 * final[0] / n, 1) if final else None
            row[f"transient_k{k:g}"] = len(segs) - (1 if final else 0)
            if k == K_SIGMA_DEFAULT:
                row["thr_selfcal"] = thr
                segs_dflt, thr_dflt, calib_end_dflt = segs, thr, calib_end
        rows.append(row)

        # grafico con il k di default
        slope = causal_slope(rms)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax1.plot(rms, lw=0.6, alpha=0.5, label="h_rms")
        ax1.plot(trailing_mean(rms, W_SMOOTH), lw=1.4, label=f"trailing mean (w={W_SMOOTH})")
        ax1.set_ylabel("RMS")
        ax1.set_title(f"{b} (cond {condition_of(b)}) — auto-calibrante (k={K_SIGMA_DEFAULT:g}) vs batch")
        ax2.plot(slope, lw=0.9, label="pendenza causale")
        ax2.axhline(thr_dflt, color="tab:red", ls="--", lw=1, label=f"soglia self-cal ({thr_dflt:.2e})")
        ax2.set_ylabel("d(RMS)/dt")
        ax2.set_xlabel("snapshot")
        onset_batch = detect_fpt_slope(rms)
        for ax in (ax1, ax2):
            ax.axvspan(0, SKIP, color="gray", alpha=0.15)
            ax.axvspan(SKIP + W_SMOOTH, calib_end_dflt, color="tab:blue", alpha=0.10)
            for (s, e) in segs_dflt:
                is_final = (e == n - 1)
                ax.axvspan(s, e, color="tab:red" if is_final else "tab:orange", alpha=0.20)
                ax.axvline(s, color="tab:red" if is_final else "tab:orange", lw=1.2)
            ax.axvline(onset_batch, color="tab:green", lw=1.5, ls="--", label="onset batch")
        ax1.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{b}.png", dpi=130)
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "onsets_summary.csv", index=False)
    with open(args.out_dir / "detector_config.json", "w") as f:
        json.dump({
            "detector": "causal_selfcal_v2",
            "detector_feature": DETECTOR_FEATURE,
            "skip": SKIP, "w_smooth": W_SMOOTH, "w_slope": W_SLOPE,
            "calib_len": CALIB_LEN, "k_sigma": K_SIGMA_DEFAULT, "persistence": PERSISTENCE, "revoke_after": REVOKE_AFTER,
        }, f, indent=2)

    cols = ["bearing", "onset_batch_pct"] + [c for k in K_SIGMA_SWEEP for c in (f"onset_k{k:g}_pct", f"transient_k{k:g}")]
    print("\n" + summary[cols].to_string(index=False))
    print(f"\nGrafici, tabella e config in: {args.out_dir}")
    print("onset_* = inizio allarme FINALE (attivo a fine vita); transient_* = allarmi revocati.\n"
          "Grafici: fascia blu = calibrazione, arancio = allarme revocato, rosso = allarme finale.")


if __name__ == "__main__":
    main()