"""Logica di inferenza a due stadi attorno agli artefatti salvati da train_and_save.

Tutti i punti in cui questo modulo tocca il codice esistente del repo sono
raccolti negli adapter in cima al file e marcati con >>> INTEGRAZIONE:
allinea i nomi di import/funzioni ai tuoi moduli reali e il resto non si tocca.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

# ---------------------------------------------------------------------------
# >>> INTEGRAZIONE 1: import dal package src del repo femto-rul.
# Allinea i nomi dei moduli/funzioni a quelli reali:
#   - la classe del modello RNN (LSTM/GRU con Softplus in uscita)
#   - la funzione che estrae le feature da UN singolo snapshot (h, v)
#     e restituisce un dict {"h_rms": ..., "v_rms": ..., "h_kurtosis": ..., ...}
#   - il rilevatore detect_fpt_slope
# ---------------------------------------------------------------------------
from src.models import RULRegressorRNN                      
from src.features import snapshot_features  
from src.causal_detector import CausalDetectorConfig, run_causal_detector
           

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# >>> INTEGRAZIONE 2: la colonna su cui gira il rilevatore (nel progetto: RMS orizzontale).



class RULService:
    """Carica gli artefatti una volta sola e serve predizioni stateless."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.config: dict = {}
        self.scalers: dict = {}
        self.model: Optional[torch.nn.Module] = None
        self.loaded = False

    # -- caricamento ---------------------------------------------------------

    def load(self) -> None:
        with open(self.artifacts_dir / "config.json") as f:
            self.config = json.load(f)

        self.scalers = joblib.load(self.artifacts_dir / "scalers.joblib")
        with open(self.artifacts_dir / "causal_detector.json") as f:
            self.detector_cfg = CausalDetectorConfig.from_dict(json.load(f))
        self.model = RULRegressorRNN(
            input_size=len(self.config["feature_columns"]),
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            rnn_type=self.config["rnn_type"],
        )
        state = torch.load(self.artifacts_dir / "model.pth", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.loaded = True
        
    @property
    def model_version(self) -> str:
        c = self.config
        return f"{c['rnn_type']}-h{c['hidden_size']}-l{c['num_layers']}-{c['label_mode']}-seed{c['seed']}"

    @property
    def feature_cols(self) -> list[str]:
        return list(self.config["feature_columns"])

    @property
    def window(self) -> int:
        return int(self.config["window"])

    @property
    def rul_scale(self) -> float:
        return float(self.config["rul_scale"])

    # -- inferenza -----------------------------------------------------------

    def extract_row(self, snapshot_h: list[float], snapshot_v: list[float]) -> list[float]:
        """Estrae le feature dello snapshot nell'ordine esatto di config['feature_columns']."""
        snapshot = np.column_stack([
            np.asarray(snapshot_h, dtype=np.float64),
            np.asarray(snapshot_v, dtype=np.float64),
        ])  # shape (2560, 2): colonna 0 = h, colonna 1 = v
        feats = snapshot_features(snapshot)
        missing = [c for c in self.feature_cols if c not in feats]
        if missing:
            raise ValueError(f"feature mancanti dall'estrattore: {missing}")
        return [float(feats[c]) for c in self.feature_cols]

    def run_detector(self, history: np.ndarray):
        col_idx = self.feature_cols.index(self.detector_cfg.feature)
        return run_causal_detector(history[:, col_idx], self.detector_cfg)

    def predict_rul(self, history: np.ndarray, condition: int) -> Optional[float]:
        """Regressione RUL sull'ultima finestra, scalata per condizione operativa."""
        if len(history) < self.window:
            return None  # traiettoria piu' corta della finestra del modello
        scaler = self.scalers[condition] if condition in self.scalers else self.scalers[str(condition)]
        scaled = scaler.transform(history)
        window = scaled[-self.window:, :]
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1, window, n_feat)
        with torch.no_grad():
            y = self.model(x)
        return float(y.squeeze().item()) / self.rul_scale
    
    def step(self, snapshot_h, snapshot_v, condition, feature_history) -> dict:
        if feature_history and len(feature_history[0]) != len(self.feature_cols):
            raise ValueError(
                f"feature_history ha {len(feature_history[0])} colonne, "
                f"il modello ne attende {len(self.feature_cols)}: storia incompatibile"
            )
        row = self.extract_row(snapshot_h, snapshot_v)
        updated = feature_history + [row]
        history = np.asarray(updated, dtype=np.float64)

        det = self.run_detector(history)
        rul = self.predict_rul(history, condition) if det.alarm_active else None

        return {
            "alarm_state": det.state,
            "alarm": det.alarm_active,
            "alarm_onset_index": det.alarm_onset,
            "revoked_alarms": det.revoked_alarms,
            "rul_seconds": rul,
            "snapshots_seen": len(updated),
            "feature_history": updated,
        }