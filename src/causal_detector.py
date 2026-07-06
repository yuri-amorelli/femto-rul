"""Rilevatore di onset CAUSALE auto-calibrante, con allarme revocabile.

Complementare a detect_fpt_slope (batch): il batch vede la traiettoria completa
e serve per il LABELING in fit-time; questo rilevatore usa a ogni istante t solo
campioni <= t e serve per l'ALLARME a runtime (API / streaming).

Macchina a stati per cuscinetto:
  calibrating -> monitoring -> (warning <-> alarm)
  - calibrating: storia insufficiente; la soglia viene appresa sulla fase
    iniziale del cuscinetto stesso (dopo lo skip del transitorio) — la scala
    della pendenza sana e' una proprieta' del singolo cuscinetto, non della
    condizione operativa, quindi non si trasferisce tra cuscinetti.
  - monitoring: soglia congelata, nessuna attivita' sopra soglia.
  - alarm: `persistence` snapshot consecutivi sopra soglia; si REVOCA dopo
    `revoke_after` consecutivi sotto. Il crollo terminale non rientra mai,
    quindi il suo allarme resta attivo fino a fine vita.
  - warning: nessun allarme attivo ora, ma almeno uno e' scattato ed e' stato
    revocato in passato — degradazione reale ma non (ancora) terminale. E' la
    "fase gialla" del progetto di ricerca, riemersa come stato operativo.

Interamente causale: ricalcolare sulla history completa a ogni chiamata di
un'API stateless produce esattamente lo stesso risultato del vero streaming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CausalDetectorConfig:
    skip: int = 100            # transitorio iniziale scartato (snapshot)
    w_smooth: int = 30         # finestra trailing smoothing RMS
    w_slope: int = 30          # finestra trailing smoothing pendenza
    calib_len: int = 150       # ampiezza finestra di calibrazione (snapshot)
    k_sigma: float = 6.0       # moltiplicatore MAD per la soglia
    persistence: int = 5       # consecutivi sopra soglia -> allarme
    revoke_after: int = 30     # consecutivi sotto soglia -> revoca
    feature: str = "h_rms"     # colonna su cui gira il rilevatore

    @classmethod
    def from_dict(cls, d: dict) -> "CausalDetectorConfig":
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass(frozen=True)
class CausalDetectorResult:
    state: str                          # "calibrating" | "monitoring" | "warning" | "alarm"
    threshold: Optional[float]          # soglia calibrata (None se ancora in calibrazione)
    alarm_active: bool                  # True sse state == "alarm"
    alarm_onset: Optional[int]          # inizio dell'allarme ATTIVO (None altrimenti)
    revoked_alarms: int                 # numero di allarmi scattati e poi revocati
    calibration_end: int                # indice snapshot in cui finisce la calibrazione


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Media mobile che guarda solo indietro: out[t] = mean(x[max(0,t-w+1) : t+1])."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    t = np.arange(len(x))
    lo = np.maximum(0, t - w + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def causal_slope(rms: np.ndarray, w_smooth: int, w_slope: int) -> np.ndarray:
    """Pendenza causale: smoothing trailing -> differenza all'indietro -> smoothing trailing."""
    sm = trailing_mean(rms, w_smooth)
    d = np.empty_like(sm)
    d[0] = 0.0
    d[1:] = np.diff(sm)
    return trailing_mean(d, w_slope)


def run_causal_detector(rms: np.ndarray, cfg: CausalDetectorConfig) -> CausalDetectorResult:
    """Esegue la macchina a stati sull'intera storia disponibile (causale per costruzione)."""
    rms = np.asarray(rms, dtype=float)
    n = len(rms)
    calib_start = cfg.skip + cfg.w_smooth
    calib_end = calib_start + cfg.calib_len

    if n <= calib_end + cfg.persistence:
        return CausalDetectorResult("calibrating", None, False, None, 0, calib_end)

    slope = causal_slope(rms, cfg.w_smooth, cfg.w_slope)
    calib = slope[calib_start:calib_end]
    mu = float(np.median(calib))
    mad = float(np.median(np.abs(calib - mu))) + 1e-12
    thr = mu + cfg.k_sigma * 1.4826 * mad

    on, up, down, start, revoked = False, 0, 0, 0, 0
    for t in range(calib_end, n):
        if not on:
            up = up + 1 if slope[t] > thr else 0
            if up >= cfg.persistence:
                on, start, down = True, t - cfg.persistence + 1, 0
        else:
            down = down + 1 if slope[t] <= thr else 0
            if down >= cfg.revoke_after:
                revoked += 1
                on, up = False, 0

    if on:
        return CausalDetectorResult("alarm", thr, True, start, revoked, calib_end)
    state = "warning" if revoked > 0 else "monitoring"
    return CausalDetectorResult(state, thr, False, None, revoked, calib_end)