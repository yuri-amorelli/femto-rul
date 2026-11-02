"""Envelope analysis: which bearing component is failing, from the raw vibration signal.

The detector says *that* a bearing is degrading; this says *what* is failing. A local fault
(a spall on a race, a damaged rolling element) produces a short impact every time a rolling
element passes it. The impacts excite a structural resonance at a few kHz, so the fault shows
up as an amplitude modulation of that resonance at the fault's characteristic frequency.
Demodulating the resonance band (band-pass, then the Hilbert envelope) and taking the
spectrum of the envelope exposes that frequency and its harmonics.

Rules fixed in reports/envelope/PLAN.md before any raw signal was analysed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, stft

# XJTU-SY bearing LDK UER204 (dataset documentation)
BALL_DIAMETER_MM = 7.92
PITCH_DIAMETER_MM = 34.55
N_BALLS = 8

SK_NPERSEG = 64               # STFT window for spectral kurtosis: 2.5 ms, shorter than the gap
                              # between impacts (5 to 9 ms here), 400 Hz resolution
SK_SEARCH_HZ = (1_000.0, 11_000.0)
BAND_HALF_WIDTH_HZ = 1_000.0  # demodulated band: SK peak +/- 1 kHz
FILTER_ORDER = 4
HARMONICS = (1, 2, 3)
REL_TOLERANCE = 0.01          # +/- 1 % around each harmonic, at least one frequency bin
REFERENCE_HALF_WIDTH_HZ = 10.0
COMPONENTS = ("outer race", "inner race", "ball", "cage")
# The verdict is taken among these. Ball is scored but left out: no XJTU-SY bearing has a ball
# fault alone, and for this geometry 3 x BSF falls within 0.5 % of 2 x BPFO, so a ball score
# partly re-counts an outer-race fault.
CANDIDATES = ("outer race", "inner race", "cage")


def fault_frequencies(shaft_hz: float, d: float = BALL_DIAMETER_MM, pitch: float = PITCH_DIAMETER_MM,
                      n: int = N_BALLS) -> dict[str, float]:
    """Characteristic frequencies for a 0 deg contact angle (stationary outer race)."""
    r = d / pitch
    return {
        "outer race": n / 2 * shaft_hz * (1 - r),               # BPFO
        "inner race": n / 2 * shaft_hz * (1 + r),               # BPFI
        "ball": pitch / (2 * d) * shaft_hz * (1 - r ** 2),      # BSF
        "cage": shaft_hz / 2 * (1 - r),                         # FTF
    }


def spectral_kurtosis(x: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Spectral kurtosis per frequency, from a short-time Fourier transform."""
    nperseg = SK_NPERSEG if nperseg is None else nperseg
    f, _, z = stft(x, fs=fs, nperseg=nperseg, noverlap=3 * nperseg // 4, window="hann")
    p2 = np.mean(np.abs(z) ** 2, axis=1)
    p4 = np.mean(np.abs(z) ** 4, axis=1)
    return f, p4 / (p2 ** 2 + 1e-30) - 2.0


def select_band(x: np.ndarray, fs: float, fixed: tuple[float, float] | None = None) -> tuple[float, float]:
    """The band to demodulate: centred on the spectral-kurtosis peak (or a fixed band)."""
    if fixed is not None:
        return fixed
    f, sk = spectral_kurtosis(x, fs)
    lo, hi = SK_SEARCH_HZ
    mask = (f >= lo) & (f <= hi)
    centre = float(f[mask][np.argmax(sk[mask])])
    return max(centre - BAND_HALF_WIDTH_HZ, 100.0), min(centre + BAND_HALF_WIDTH_HZ, fs / 2 - 100.0)


def envelope_spectrum(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass, Hilbert envelope, Hann-windowed amplitude spectrum of the envelope."""
    sos = butter(FILTER_ORDER, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, np.asarray(x, dtype=float))))
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / len(env)
    return np.fft.rfftfreq(len(env), 1 / fs), spec


def harmonic_snr(freqs: np.ndarray, spec: np.ndarray, f0: float) -> float:
    """Mean over harmonics 1..3 of: peak inside the tolerance window / median of the
    envelope spectrum within +/- 10 Hz around it (window excluded)."""
    df = freqs[1] - freqs[0]
    ratios = []
    for k in HARMONICS:
        fk = k * f0
        tol = max(REL_TOLERANCE * fk, df)
        win = (freqs >= fk - tol) & (freqs <= fk + tol)
        ref = (np.abs(freqs - fk) <= REFERENCE_HALF_WIDTH_HZ) & ~win
        ratios.append(float(spec[win].max() / (np.median(spec[ref]) + 1e-30)))
    return float(np.mean(ratios))


@dataclass(frozen=True)
class Diagnosis:
    band: tuple[float, float]
    scores: dict[str, float]          # harmonic SNR per component
    verdict: str                      # candidate with the highest score
    margin: float                     # best candidate score / second best


def diagnose(x: np.ndarray, fs: float, shaft_hz: float, fixed_band: tuple[float, float] | None = None) -> Diagnosis:
    band = select_band(x, fs, fixed_band)
    freqs, spec = envelope_spectrum(x, fs, band)
    ff = fault_frequencies(shaft_hz)
    scores = {c: harmonic_snr(freqs, spec, ff[c]) for c in COMPONENTS}
    ranked = sorted(CANDIDATES, key=lambda c: scores[c], reverse=True)
    return Diagnosis(band, scores, ranked[0], scores[ranked[0]] / scores[ranked[1]])
