"""train_and_save.py — addestra il modello event-driven UNA volta e salva gli
artefatti persistenti che l'API caricherà: modello, scaler per condizione,
e i metadati (colonne, window, RUL_SCALE, mappa condizioni).

Uso:
    python train_and_save.py
Produce, nella cartella artifacts/:
    model.pth          - pesi del modello LSTM
    scalers.joblib     - dict {condizione: StandardScaler}
    config.json        - colonne feature, window, rul_scale, hyperparams
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.pipeline import prepare_bearing, fit_scalers, apply_scaler, feature_columns
from src.dataset import SequenceDataset
from src.models import RULRegressorRNN
from src.train import train_model

# --- parametri fissi (coerenti col progetto FEMTO) ---
LABEL_MODE = "piecewise_slope"     # lo schema event-driven
WINDOW     = 30
RUL_SCALE  = 1.0 / 9000.0
HIDDEN     = 64
LAYERS     = 2
RNN_TYPE   = "lstm"
SEED       = 0                     # fissato per riproducibilità
TRAIN_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1",
                  "Bearing2_2", "Bearing3_1"]
VAL_BEARING    = "Bearing3_2"

ARTIFACTS = config.PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) prepara i dati di training
    tr = {n: prepare_bearing(config.DATA_ROOT / "Learning_set" / n,
                             label_mode=LABEL_MODE, cache=False)
          for n in TRAIN_BEARINGS}
    vd = {VAL_BEARING: prepare_bearing(config.DATA_ROOT / "Learning_set" / VAL_BEARING,
                                       label_mode=LABEL_MODE, cache=False)}

    # 2) fitta gli scaler PER CONDIZIONE sui soli bearing di training
    scalers, cols = fit_scalers(tr)

    def pairs(d):
        return [(apply_scaler(n, df, scalers, cols), t * RUL_SCALE)
                for n, (df, t) in d.items()]

    train_ds = SequenceDataset(pairs(tr), window=WINDOW)
    val_ds   = SequenceDataset(pairs(vd), window=WINDOW)

    # 3) addestra
    model = RULRegressorRNN(input_size=len(cols), hidden_size=HIDDEN,
                            num_layers=LAYERS, rnn_type=RNN_TYPE)
    out = train_model(model,
                      DataLoader(train_ds, batch_size=128, shuffle=True),
                      DataLoader(val_ds, batch_size=128),
                      epochs=100)
    model = out["model"]

    # 4) salva gli artefatti
    torch.save(model.state_dict(), ARTIFACTS / "model.pth")
    joblib.dump(scalers, ARTIFACTS / "scalers.joblib")
    meta = {
        "feature_columns": cols,
        "window": WINDOW,
        "rul_scale": RUL_SCALE,
        "hidden_size": HIDDEN,
        "num_layers": LAYERS,
        "rnn_type": RNN_TYPE,
        "label_mode": LABEL_MODE,
        "conditions": sorted({config.condition_of(n) for n in TRAIN_BEARINGS}),
        "seed": SEED,
    }
    (ARTIFACTS / "config.json").write_text(json.dumps(meta, indent=2))

    print("Artefatti salvati in", ARTIFACTS)
    print("  - model.pth")
    print("  - scalers.joblib   (condizioni:", list(scalers.keys()), ")")
    print("  - config.json")


if __name__ == "__main__":
    main()