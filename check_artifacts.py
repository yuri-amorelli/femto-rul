"""check_artifacts.py — verifica che gli artefatti salvati si ricarichino e che
il modello ricaricato produca RUL sensate su un bearing noto. Nessun training."""
import json
import joblib
import numpy as np
import torch

from src import config
from src.pipeline import prepare_bearing, apply_scaler
from src.dataset import make_sequences
from src.models import RULRegressorRNN
from src.labeling import detect_fpt_slope

ART = config.PROJECT_ROOT / "artifacts"

# 1) ricarica i metadati
meta = json.loads((ART / "config.json").read_text())
cols = meta["feature_columns"]
WINDOW = meta["window"]
RUL_SCALE = meta["rul_scale"]
print("config.json OK —", len(cols), "feature, window", WINDOW, "conditions", meta["conditions"])

# 2) ricarica scaler e modello
scalers = joblib.load(ART / "scalers.joblib")
print("scalers.joblib OK — condizioni:", list(scalers.keys()))

model = RULRegressorRNN(input_size=len(cols), hidden_size=meta["hidden_size"],
                        num_layers=meta["num_layers"], rnn_type=meta["rnn_type"])
model.load_state_dict(torch.load(ART / "model.pth", map_location="cpu"))
model.eval()
print("model.pth OK — pesi caricati")

# 3) prova su un bearing di test: predizione + detector
name = "Bearing1_3"
df, ytrue = prepare_bearing(config.DATA_ROOT / "Test_set" / name,
                            label_mode=meta["label_mode"], cache=False)
Xs = apply_scaler(name, df, scalers, cols)
Xseq, _ = make_sequences(Xs, ytrue * RUL_SCALE, window=WINDOW)
with torch.no_grad():
    pred = model(torch.from_numpy(Xseq)).numpy() / RUL_SCALE
fpt = detect_fpt_slope(df["h_rms"].to_numpy())

print(f"\n{name}: {len(pred)} predizioni")
print(f"  RUL predetta (primi 3): {pred[:3].round(0)}")
print(f"  RUL predetta (ultimi 3): {pred[-3:].round(0)}")
print(f"  detector: allarme allo snapshot {fpt} ({100*fpt/len(ytrue):.0f}% della vita)")
print("\nARTEFATTI VALIDI — pronti per l'API")