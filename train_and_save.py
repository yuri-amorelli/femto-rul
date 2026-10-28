"""train_and_save.py — addestra il modello event-driven UNA volta e salva gli
artefatti persistenti che l'API caricherà: modello, scaler per condizione,
e i metadati (colonne, window, RUL_SCALE, mappa condizioni).

La run viene tracciata con MLflow (esperimento "femto-rul-production"):
parametri, curve di loss, artefatti, e il modello registrato nel Model
Registry come "femto-rul-lstm".

Uso:
    python train_and_save.py
Produce, nella cartella artifacts/:
    model.pth          - pesi del modello LSTM
    scalers.joblib     - dict {condizione: StandardScaler}
    config.json        - colonne feature, window, rul_scale, hyperparams
"""
from __future__ import annotations
import json

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config, tracking
from src.pipeline import prepare_bearing, fit_scalers, apply_scaler
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
EPOCHS     = 100
LR         = 1e-3
PATIENCE   = 10
CLIP       = 5.0
BATCH      = 128
TRAIN_BEARINGS = ["Bearing1_1", "Bearing1_2", "Bearing2_1",
                  "Bearing2_2", "Bearing3_1"]
VAL_BEARING    = "Bearing3_2"

EXPERIMENT       = "femto-rul-production"
REGISTERED_MODEL = "femto-rul-lstm"

ARTIFACTS = config.PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tracking.setup(EXPERIMENT)
    with mlflow.start_run(run_name=f"{RNN_TYPE}_{LABEL_MODE}_seed{SEED}") as run:
        mlflow.set_tag("purpose", "artifacts served by the FastAPI service")
        mlflow.log_params({
            "label_mode": LABEL_MODE, "window": WINDOW, "rul_scale": RUL_SCALE,
            "hidden_size": HIDDEN, "num_layers": LAYERS, "rnn_type": RNN_TYPE,
            "seed": SEED, "epochs_max": EPOCHS, "lr": LR, "patience": PATIENCE,
            "grad_clip": CLIP, "batch_size": BATCH,
            "train_bearings": tracking.as_param(TRAIN_BEARINGS),
            "val_bearing": VAL_BEARING,
        })

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
                          DataLoader(train_ds, batch_size=BATCH, shuffle=True),
                          DataLoader(val_ds, batch_size=BATCH),
                          epochs=EPOCHS, lr=LR, patience=PATIENCE, clip=CLIP,
                          on_epoch_end=tracking.epoch_logger())
        model = out["model"]
        mlflow.log_metrics({"best_val_loss": out["best_val"],
                            "epochs_run": out["epochs_run"]})

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

        # 5) traccia gli artefatti dell'API e registra il modello
        mlflow.log_artifacts(str(ARTIFACTS), artifact_path="api_artifacts")
        # due finestre reali (2, WINDOW, n_feature): MLflow ne ricava la
        # signature del modello. Batch 2 (non 1) così la dimensione del batch
        # resta dinamica anche se si usa l'export pt2.
        example = next(iter(DataLoader(val_ds, batch_size=2)))[0].numpy()
        model = model.cpu().eval()
        try:
            # pickle: si carica sia su CPU sia su GPU e accetta qualsiasi batch
            mlflow.pytorch.log_model(model, name="model", input_example=example,
                                     serialization_format="pickle",
                                     registered_model_name=REGISTERED_MODEL)
        except Exception as exc:
            print(f"serializzazione pickle non disponibile ({exc}); uso pt2")
            mlflow.pytorch.log_model(model, name="model", input_example=example,
                                     registered_model_name=REGISTERED_MODEL)

        print("Artefatti salvati in", ARTIFACTS)
        print("  - model.pth")
        print("  - scalers.joblib   (condizioni:", list(scalers.keys()), ")")
        print("  - config.json")
        print(f"MLflow run {run.info.run_id} -> modello registrato come '{REGISTERED_MODEL}'")


if __name__ == "__main__":
    main()
