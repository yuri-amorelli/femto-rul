"""RUL target construction — the most important (and most abused) step.

Because the training bearings are run-to-failure, it is *tempting* to define
    RUL(t) = total_life - t
i.e. a straight line from full life down to zero. This "linear RUL" is the
default in a lot of papers and it is quietly dishonest: it asserts the bearing
is already dying at t=0, when in fact it is healthy and stationary for most of
its life. A model trained on it spends its capacity fitting a countdown that
carries no physical signal.

The more defensible target is PIECEWISE-LINEAR:
    - a flat "healthy" plateau until degradation actually starts (the First
      Prediction Time, FPT), then
    - a linear ramp to zero at failure.
This says: "RUL is unknowable / effectively constant while healthy; it only
becomes a meaningful countdown once damage is detectable."

The open question is *where* the knee (FPT) sits. Detecting it is an
event-detection problem on the health-indicator trajectory — exactly the kind
of event-driven labelling IFCR addresses. Here we ship a transparent baseline
detector (RMS crossing a healthy-baseline threshold) and keep the plateau cap
configurable, so the labelling assumption is explicit and swappable, never
hidden inside the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

DT = config.SNAPSHOT_PERIOD  # seconds per snapshot


def linear_rul(n: int) -> np.ndarray:
    """RUL in seconds for a run-to-failure bearing of n snapshots."""
    remaining = np.arange(n - 1, -1, -1)          # n-1, ..., 0
    return remaining.astype(np.float32) * DT


def detect_fpt(rms: np.ndarray, healthy_frac: float = 0.1, k: float = 5.0,
               min_consecutive: int = 3) -> int:
    """First Prediction Time = first index where smoothed RMS leaves the
    healthy band (mean + k*std of the early baseline) for a few points in a row.

    This is a deliberately simple, auditable detector. It is NOT meant to be
    state-of-the-art; it makes the labelling assumption visible. Swap it for a
    proper change-point / event-driven method (e.g. IFCR-style) to improve.
    """
    rms = pd.Series(rms).rolling(5, min_periods=1).mean().to_numpy()
    n = len(rms)
    base = rms[: max(1, int(healthy_frac * n))]
    threshold = base.mean() + k * base.std()
    above = rms > threshold
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= min_consecutive:
            return i - min_consecutive + 1
    return int(0.5 * n)  # fallback: if nothing triggers, assume mid-life onset

def detect_fpt_cusum(rms: np.ndarray, healthy_frac: float = 0.2,
                     skip: int = 20, k_sigma: float = 1.0, h_sigma: float = 25.0,
                     min_index: int = 5) -> int:
    """Degradation-onset detection via a one-sided CUSUM on the RMS signal,
    with a ROBUST healthy baseline (median + MAD) that ignores start-up
    transients instead of letting them inflate the baseline spread.

    - `skip` discards the first few samples (violent start-up spikes) before
      estimating the baseline, but the CUSUM still scans from `skip` onward.
    - baseline location = median, scale = 1.4826 * MAD (robust std estimate).
      A few early outliers therefore do not distort mu/sigma.
    """
    import numpy as _np
    rms = _np.asarray(rms, dtype=float)
    n = len(rms)
    lo = min(skip, max(0, n - 1))
    n_base = max(lo + min_index, int(healthy_frac * n))
    base = rms[lo:n_base]
    if base.size == 0:
        base = rms[:max(1, n_base)]

    mu = _np.median(base)
    mad = _np.median(_np.abs(base - mu))
    sigma = 1.4826 * mad + 1e-9        # robust std estimate

    k = k_sigma * sigma
    h = h_sigma * sigma
    s = 0.0
    for i in range(lo, n):
        s = max(0.0, s + (rms[i] - mu) - k)
        if s > h and i >= min_index:
            return i
    return int(0.5 * n)

def detect_fpt_slope(rms: np.ndarray, healthy_frac: float = 0.3,
                     k_sigma: float = 6.0, smooth_frac: float = 0.02,
                     skip: int = 100, min_index: int = 5) -> int:
    """Onset = start of the TERMINAL acceleration of degradation, on the slope
    of the smoothed RMS. A start-up transient (the bearing settling to thermal/
    load regime in the first samples) is discarded via `skip` before both the
    baseline estimate and the scan, otherwise it inflates the robust threshold
    and can be latched as a false onset.
    """
    import numpy as _np
    rms = _np.asarray(rms, dtype=float)
    n = len(rms)
    skip = min(skip, max(0, n // 4))          # never drop more than 25% of life
    win = max(3, int(smooth_frac * n))
    sm = _np.convolve(rms, _np.ones(win) / win, mode="same")
    slope = _np.gradient(sm)
    slope = _np.convolve(slope, _np.ones(win) / win, mode="same")

    # baseline computed AFTER the start-up transient
    base = slope[skip:max(skip + min_index, int(healthy_frac * n))]
    mu = _np.median(base)
    mad = _np.median(_np.abs(base - mu)) + 1e-9
    thr = mu + k_sigma * 1.4826 * mad

    above = slope > thr
    onset = n
    for i in range(n - 1, skip - 1, -1):       # scan stops at `skip`, not 0
        if above[i]:
            onset = i
        else:
            if onset < n:
                break
    return onset if onset < n else int(0.7 * n)

def piecewise_rul(n: int, fpt: int) -> np.ndarray:
    """Flat plateau up to FPT, then linear ramp to 0 at failure.

    Implemented as min(linear_rul, cap) where cap = RUL at the FPT. Before the
    knee the target is clamped; after it, it coincides with the linear ramp.
    """
    lin = linear_rul(n)
    cap = lin[fpt]
    return np.minimum(lin, cap).astype(np.float32)

def capped_rul(n: int, cap_seconds: float = 2500.0) -> np.ndarray:
    """Piecewise-linear RUL with a FIXED cap, identical across bearings.

    target = min(linear_rul, cap). No fragile onset detection: the plateau
    height is a constant, so the label is consistent from bearing to bearing.
    """
    lin = linear_rul(n)
    return np.minimum(lin, cap_seconds).astype(np.float32)

def make_targets(feature_frame: pd.DataFrame, mode: str = "piecewise") -> np.ndarray:
    """Build the RUL target for one bearing from its feature frame.

    mode='linear'    -> straight countdown (baseline, for ablation)
    mode='piecewise' -> plateau + ramp (recommended)
    """
    n = len(feature_frame)
    if mode == "piecewise_slope":
        rms = feature_frame["h_rms"].to_numpy()
        fpt = detect_fpt_slope(rms)
        return piecewise_rul(n, fpt)
    if mode == "piecewise_cusum":
        rms = feature_frame["h_rms"].to_numpy()
        fpt = detect_fpt_cusum(rms)
        return piecewise_rul(n, fpt)
    if mode == "linear":
        return linear_rul(n)
    if mode == "capped":
        return capped_rul(n, cap_seconds=2500.0)
    if mode == "piecewise":
        # use horizontal RMS as the health indicator driving FPT detection
        rms = feature_frame["h_rms"].to_numpy()
        fpt = detect_fpt(rms)
        return piecewise_rul(n, fpt)
    raise ValueError(f"unknown mode: {mode!r}")
