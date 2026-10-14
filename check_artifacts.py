"""check_artifacts.py — verifica che gli artefatti salvati si ricarichino e che
il modello ricaricato produca RUL sensate su un bearing noto. Nessun training.

Nota: i bearing del Test_set ufficiale sono troncati PRIMA del guasto, quindi
qui non ci si aspetta nessun allarme. Il comportamento del detector causale
(calibrating -> monitoring -> alarm) si verifica end-to-end tramite l'API sulla
traiettoria completa:
    python scripts/api_test_client.py --bearing-dir data/Full_Test_Set/Bearing1_3 --condition 1
"""
import json
import joblib
import numpy as np
import torch

from src import config
from src.pipeline import prepare_bearing, apply_scaler
from src.dataset import make_sequences
from src.models import RULRegressorRNN

ART = config.PROJECT_ROOT / "artifacts"

# 1) ricarica i metadati
meta = json.loads((ART / "config.json").read_text())
cols = meta["feature_columns"]
WINDOW = meta["window"]
RUL_SCALE = meta["rul_scale"]
print("config.json OK —", len(cols), "feature, window", WINDOW, "conditions", meta["conditions"])

# 2) ricarica la configurazione del detector causale usata dall'API
det = json.loads((ART / "causal_detector.json").read_text())
expected = {"skip", "w_smooth", "w_slope", "calib_len", "k_sigma",
            "persistence", "revoke_after", "feature"}
missing = expected - det.keys()
assert not missing, f"causal_detector.json: chiavi mancanti {sorted(missing)}"
print("causal_detector.json OK —", ", ".join(f"{k}={det[k]}" for k in sorted(expected)))

# 3) ricarica scaler e modello
scalers = joblib.load(ART / "scalers.joblib")
print("scalers.joblib OK — condizioni:", list(scalers.keys()))

model = RULRegressorRNN(input_size=len(cols), hidden_size=meta["hidden_size"],
                        num_layers=meta["num_layers"], rnn_type=meta["rnn_type"])
model.load_state_dict(torch.load(ART / "model.pth", map_location="cpu"))
model.eval()
print("model.pth OK — pesi caricati")

# 4) prova su un bearing di test: il modello ricaricato deve produrre RUL valide
name = "Bearing1_3"
df, ytrue = prepare_bearing(config.DATA_ROOT / "Test_set" / name,
                            label_mode=meta["label_mode"], cache=False)
Xs = apply_scaler(name, df, scalers, cols)
Xseq, _ = make_sequences(Xs, ytrue * RUL_SCALE, window=WINDOW)
with torch.no_grad():
    pred = model(torch.from_numpy(Xseq)).numpy() / RUL_SCALE

assert len(pred) > 0, "nessuna predizione prodotta"
assert np.isfinite(pred).all(), "predizioni non finite (NaN/inf)"
assert (pred >= 0).all(), "predizioni RUL negative"

print(f"\n{name} (Test_set, troncato prima del guasto): {len(pred)} predizioni")
print(f"  RUL predetta (primi 3): {pred[:3].round(0)}")
print(f"  RUL predetta (ultimi 3): {pred[-3:].round(0)}")
print(f"  range: {pred.min():.0f}–{pred.max():.0f} s")
print("  detector: non verificato qui (traiettoria troncata, nessun allarme atteso);")
print("            usare scripts/api_test_client.py sul Full_Test_Set")
print("\nARTEFATTI VALIDI — pronti per l'API")