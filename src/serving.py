"""Torch-free serving layer for the two-stage system (causal detector -> stage 2).

The API loads one *bundle*: a folder written by scripts/export_serving.py that
holds everything the service needs as plain JSON, so the container needs
neither PyTorch nor scikit-learn and every served number can be inspected:

    bundle.json           version, creation time, evaluation recorded at export
    dataset.json          acquisition constants and the features kept in the history
    detector.json         causal detector configuration (copy of artifacts/causal_detector.json)
    stage2.json           linear quantile regression: scaler, coefficients, how it was chosen
    reference_stats.json  healthy-phase feature distributions of the training
                          bearings, per operating condition (reference for drift monitoring)
    demo/<bearing>.json   features of the official test bearings, for the replay

A new dataset is a new bundle, not new code: snapshot length, sampling rate,
snapshot period and detector parameters all come from the bundle.

The stage-2 features are computed exactly as in src/post_alarm.py (curve_frame);
scripts/export_serving.py checks that the bundle reproduces the evaluated
results before writing it.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.causal_detector import (CausalDetectorConfig, CausalDetectorResult, run_causal_detector,
                                 stream_causal_detector, trailing_mean)
from src.features import snapshot_features

# The stage-2 features this module knows how to compute, in model column order.
STAGE2_FEATURES = ("tau", "h_lr", "v_lr", "h_kurt", "v_kurt")
# History columns the two stages need (detector on h_rms, stage 2 on all four).
REQUIRED_HISTORY = ("h_rms", "v_rms", "h_kurtosis", "v_kurtosis")


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class DatasetSpec:
    """Acquisition constants of one dataset, read from dataset.json."""

    name: str
    sampling_rate_hz: int
    snapshot_len: int
    snapshot_period_s: float
    history_features: tuple[str, ...]
    operating_conditions: tuple[int, ...]
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        return cls(name=d["name"], sampling_rate_hz=int(d["sampling_rate_hz"]),
                   snapshot_len=int(d["snapshot_len"]), snapshot_period_s=float(d["snapshot_period_s"]),
                   history_features=tuple(d["history_features"]),
                   operating_conditions=tuple(int(c) for c in d["operating_conditions"]),
                   description=d.get("description", ""))


@dataclass(frozen=True)
class LinearQuantileModel:
    """StandardScaler + linear quantile regression, as fitted by
    src.post_alarm.fit_quantile_pipeline, stored as numbers instead of a pickle."""

    features: tuple[str, ...]
    quantile: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    smooth: int
    clip_min: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "LinearQuantileModel":
        return cls(features=tuple(d["features"]), quantile=float(d["quantile"]),
                   mean=np.asarray(d["scaler_mean"], dtype=float),
                   scale=np.asarray(d["scaler_scale"], dtype=float),
                   coef=np.asarray(d["coef"], dtype=float), intercept=float(d["intercept"]),
                   smooth=int(d["smooth"]), clip_min=float(d.get("clip_min", 0.0)))

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """RUL [s]. Missing features count as 0, as in the evaluation."""
        Z = np.nan_to_num(np.atleast_2d(np.asarray(Z, dtype=float)), nan=0.0)
        return np.maximum((Z - self.mean) / self.scale @ self.coef + self.intercept, self.clip_min)


def stage2_matrix(history: dict[str, np.ndarray], onsets: np.ndarray, cfg: CausalDetectorConfig,
                  smooth: int, period_s: float) -> np.ndarray:
    """Stage-2 features for every snapshot, columns in STAGE2_FEATURES order.

    onsets[i] is the onset index of the alarm active at snapshot i, -1 if none.
    All features are causal: row i only uses snapshots <= i (the healthy baseline
    is the calibration window, which precedes any alarm)."""
    n = len(onsets)
    idx = np.arange(n)
    alarm = onsets >= 0
    cols = {"tau": np.where(alarm, (idx - onsets) * period_s, np.nan)}
    cal0 = cfg.skip + cfg.w_smooth
    cal1 = cal0 + cfg.calib_len
    for ch in ("h", "v"):
        rms = np.asarray(history[f"{ch}_rms"], dtype=float)
        base = np.median(rms[cal0:cal1]) if n > cal1 else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            cols[f"{ch}_lr"] = np.log(trailing_mean(rms, smooth) / base)
        cols[f"{ch}_kurt"] = trailing_mean(np.asarray(history[f"{ch}_kurtosis"], dtype=float), smooth)
    return np.column_stack([cols[f] for f in STAGE2_FEATURES])


def _segments(states: list[CausalDetectorResult]) -> list[dict]:
    """Consecutive runs of the same detector state: [{state, start, end}] (end inclusive)."""
    out: list[dict] = []
    for i, s in enumerate(states):
        if out and out[-1]["state"] == s.state:
            out[-1]["end"] = i
        else:
            out.append({"state": s.state, "start": i, "end": i})
    return out


class ServingBundle:
    """Everything the API serves, loaded once from a bundle folder."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.meta = _read(self.root / "bundle.json")
        self.dataset = DatasetSpec.from_dict(_read(self.root / "dataset.json"))
        self.detector = CausalDetectorConfig.from_dict(_read(self.root / "detector.json"))
        self.stage2_info = _read(self.root / "stage2.json")
        self.stage2 = LinearQuantileModel.from_dict(self.stage2_info)
        self.version: str = self.meta["version"]
        self.demo_dir = self.root / "demo"
        self.demo_bearings = sorted(p.stem for p in self.demo_dir.glob("*.json"))
        self._replays: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._validate()

    def _validate(self) -> None:
        hist = self.dataset.history_features
        missing = [c for c in REQUIRED_HISTORY if c not in hist]
        if missing:
            raise ValueError(f"dataset.json: history_features lacks {missing}")
        if self.detector.feature not in hist:
            raise ValueError(f"detector feature {self.detector.feature!r} is not in the history")
        if self.stage2.features != STAGE2_FEATURES:
            raise ValueError(f"stage2.json features {self.stage2.features} != {STAGE2_FEATURES}")
        if not (len(self.stage2.mean) == len(self.stage2.scale) == len(self.stage2.coef) == len(STAGE2_FEATURES)):
            raise ValueError("stage2.json: scaler and coefficient sizes do not match the features")

    # -- one call of the streaming API ------------------------------------------------

    def extract_row(self, snapshot_h: list[float], snapshot_v: list[float]) -> list[float]:
        """Features of one raw snapshot, in history_features order."""
        n = self.dataset.snapshot_len
        for name, sig in (("snapshot_h", snapshot_h), ("snapshot_v", snapshot_v)):
            if len(sig) != n:
                raise ValueError(f"{name} must contain exactly {n} samples, got {len(sig)}")
        snapshot = np.column_stack([np.asarray(snapshot_h, dtype=float), np.asarray(snapshot_v, dtype=float)])
        if not np.isfinite(snapshot).all():
            raise ValueError("the snapshot contains NaN or infinite values")
        feats = snapshot_features(snapshot, fs=self.dataset.sampling_rate_hz)
        return [float(feats[c]) for c in self.dataset.history_features]

    def _columns(self, history: np.ndarray) -> dict[str, np.ndarray]:
        return {c: history[:, i] for i, c in enumerate(self.dataset.history_features)}

    def evaluate(self, history: np.ndarray) -> dict:
        """Detector state and, only under an active alarm, the stage-2 RUL at the last snapshot."""
        cols = self._columns(history)
        det = run_causal_detector(cols[self.detector.feature], self.detector)
        rul = None
        if det.alarm_active:
            onsets = np.full(len(history), -1)
            onsets[-1] = det.alarm_onset
            Z = stage2_matrix(cols, onsets, self.detector, self.stage2.smooth,
                              self.dataset.snapshot_period_s)[-1:]
            rul = float(self.stage2.predict(Z)[0])
        return {"alarm_state": det.state, "alarm": det.alarm_active, "alarm_onset_index": det.alarm_onset,
                "revoked_alarms": det.revoked_alarms, "rul_seconds": rul}

    def step(self, snapshot_h: list[float], snapshot_v: list[float], operating_condition: int,
             feature_history: list[list[float]], max_history: int = 50_000) -> dict:
        width = len(self.dataset.history_features)
        if operating_condition not in self.dataset.operating_conditions:
            raise ValueError(f"operating_condition must be one of {list(self.dataset.operating_conditions)}")
        if feature_history and any(len(r) != width for r in feature_history):
            raise ValueError(
                f"feature_history rows must have {width} values {list(self.dataset.history_features)}; "
                "a history from another model version cannot be reused: start again with an empty one")
        if len(feature_history) >= max_history:
            raise ValueError(f"feature_history is limited to {max_history} snapshots")
        updated = feature_history + [self.extract_row(snapshot_h, snapshot_v)]
        out = self.evaluate(np.asarray(updated, dtype=float))
        out.update({"rul_quantile": self.stage2.quantile if out["alarm"] else None,
                    "snapshots_seen": len(updated), "feature_history": updated})
        return out

    # -- replay of an official test bearing ---------------------------------------------

    def demo_info(self, bearing: str) -> dict:
        d = _read(self.demo_dir / f"{bearing}.json")
        return {k: v for k, v in d.items() if k != "features"} | {"n_snapshots": len(d["features"]["h_rms"])}

    def replay(self, bearing: str) -> dict:
        """The whole recording streamed through the same causal code as /predict
        (detector on each prefix, stage 2 under an active alarm). Cached."""
        if bearing not in self.demo_bearings:
            raise KeyError(bearing)
        with self._lock:
            if bearing not in self._replays:
                self._replays[bearing] = self._replay(bearing)
            return self._replays[bearing]

    def _replay(self, bearing: str) -> dict:
        d = _read(self.demo_dir / f"{bearing}.json")
        cols = {c: np.asarray(d["features"][c], dtype=float) for c in self.dataset.history_features}
        n = len(cols[self.detector.feature])
        period = self.dataset.snapshot_period_s
        states = stream_causal_detector(cols[self.detector.feature], self.detector)
        onsets = np.array([s.alarm_onset if s.alarm_active else -1 for s in states])
        alarm = onsets >= 0
        pred = np.full(n, np.nan)
        if alarm.any():
            Z = stage2_matrix(cols, onsets, self.detector, self.stage2.smooth, period)
            pred[alarm] = self.stage2.predict(Z[alarm])
        true = (n - 1 - np.arange(n)) * period          # seconds to the end of the recording

        end = states[-1]
        final_onset = end.alarm_onset if end.alarm_active else None
        curve = alarm & (np.arange(n) >= (final_onset if final_onset is not None else n))
        err_curve = pred[curve] - true[curve]
        trunc = int(d["official_truncation_index"])
        summary = {
            "alarm_at_end": bool(end.alarm_active),
            "final_onset_index": final_onset,
            "final_onset_pct_of_life": round(100 * final_onset / n, 1) if final_onset is not None else None,
            "warning_before_end_s": float((n - 1 - final_onset) * period) if final_onset is not None else None,
            "first_alarm_pct_of_life": round(100 * int(np.argmax(alarm)) / n, 1) if alarm.any() else None,
            "revoked_alarms": int(end.revoked_alarms),
            "curve_snapshots": int(curve.sum()),
            "curve_mae_s": float(np.mean(np.abs(err_curve))) if curve.any() else None,
            "curve_late_fraction": float(np.mean(err_curve > 0)) if curve.any() else None,
            "exposed_mae_s": float(np.mean(np.abs(pred[alarm] - true[alarm]))) if alarm.any() else None,
            "state_at_truncation": states[trunc].state,
            "rul_at_truncation_s": None if np.isnan(pred[trunc]) else float(pred[trunc]),
        }
        return {
            "bearing": bearing,
            "operating_condition": int(d["operating_condition"]),
            "snapshot_period_s": period,
            "n_snapshots": n,
            "official_truncation_index": trunc,
            "official_actual_rul_s": d.get("official_actual_rul_s"),
            "note": d.get("note"),
            "summary": summary,
            "segments": _segments(states),
            "h_rms": [float(f"{x:.6g}") for x in cols["h_rms"]],
            "rul_pred_s": [None if np.isnan(p) else round(float(p), 1) for p in pred],
            "rul_true_s": [round(float(t), 1) for t in true],
            "model_version": self.version,
        }

    # -- description of what is served ---------------------------------------------------

    def model_card(self) -> dict:
        s2 = self.stage2_info
        return {
            "model_version": self.version,
            "created_utc": self.meta.get("created_utc"),
            "source_commit": self.meta.get("source_commit"),
            "dataset": {"name": self.dataset.name, "description": self.dataset.description,
                        "sampling_rate_hz": self.dataset.sampling_rate_hz,
                        "snapshot_len": self.dataset.snapshot_len,
                        "snapshot_period_s": self.dataset.snapshot_period_s,
                        "history_features": list(self.dataset.history_features)},
            "stage1_detector": {k: getattr(self.detector, k) for k in self.detector.__dataclass_fields__},
            "stage2": {k: s2[k] for k in ("model", "quantile", "features", "smooth", "training",
                                          "quantile_selection") if k in s2},
            "evaluation_at_export": self.meta.get("evaluation"),
        }


def load_bundle(root: Path) -> ServingBundle:
    return ServingBundle(root)


def default_bundle_dir(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "artifacts" / "serving" / "femto"
