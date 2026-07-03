# RUL Estimation on FEMTO-ST Bearings — XGBoost vs. LSTM/GRU

Remaining Useful Life (RUL) estimation on the [FEMTO-ST / PRONOSTIA](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset)
run-to-failure bearing dataset (IEEE PHM 2012 Prognostic Challenge). The project
compares a **gradient-boosting baseline (XGBoost)** on engineered condition-monitoring
features against a **recurrent sequence model (LSTM / GRU, PyTorch)** on the same
features, and reports results with both regression metrics and the official
asymmetric PHM 2012 score.

The point of the repo is methodological honesty, not a leaderboard number: the
comparison is designed so the *only* thing that changes between baseline and
sequence model is whether temporal context is modelled explicitly.

## Why this is set up the way it is

- **Shared features for both models.** Time- and frequency-domain health
  indicators (RMS, kurtosis, crest factor, spectral band energies, …) are
  extracted once per snapshot and fed to *both* XGBoost and the RNN. This
  isolates the effect of temporal modelling from the effect of a richer input.
- **RUL labelling is the hard part, and it's made explicit.** A naïve linear
  RUL (`total_life − t`) asserts the bearing is dying from t=0, which is
  physically false. The default target here is **piecewise-linear**: a healthy
  plateau until a detected degradation onset (First Prediction Time), then a
  linear ramp to failure. The onset detector is deliberately simple and
  swappable — event-driven labelling is an open problem, not a solved one.
- **No data leakage.** Train/test split is **by bearing**, never by random
  snapshot. Feature scalers are fit on training bearings only and applied per
  operating condition. Early stopping uses a held-out *training* bearing, not
  the test set.
- **Asymmetric evaluation.** Alongside RMSE/MAE, the PHM 2012 score penalises
  late (over-optimistic) predictions harder than early ones — the penalty
  structure that matters for real maintenance decisions.

## Project structure

```
femto-rul/
├── src/
│   ├── config.py         # paths, acquisition constants, condition map
│   ├── data_loading.py   # robust reading of acc_*.csv snapshots
│   ├── features.py       # time/frequency-domain feature extraction
│   ├── labeling.py       # linear vs piecewise RUL + onset detection
│   ├── dataset.py        # PyTorch sliding-window sequence builder
│   ├── models.py         # LSTM/GRU RUL regressor
│   ├── train.py          # training loop (early stopping, grad clipping)
│   ├── evaluate.py       # RMSE, MAE, PHM 2012 score
│   └── pipeline.py       # caching + leakage-safe scaling
├── scripts/
│   ├── run_xgboost_baseline.py
│   └── run_lstm.py
├── notebooks/
│   └── 01_methodology.ipynb   # exploration + labelling walkthrough
└── data/                       # FEMTO-ST goes here (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset so that `data/Learning_set/Bearing1_1/acc_00001.csv` exists.
**Verify the CSV layout of your download** (column count / separator can vary
between redistributions) — see the note at the top of `src/data_loading.py`.

## Running

```bash
# 1) baseline first — it anchors the comparison
python -m scripts.run_xgboost_baseline \
    --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
    --test  Bearing1_3 Bearing2_3 Bearing3_3

# 2) sequence model
python -m scripts.run_lstm --rnn lstm --window 30 \
    --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
    --test  Bearing1_3 Bearing2_3 Bearing3_3
```

## Results

| Model            | RMSE (s) | MAE (s) | PHM score |
|------------------|:--------:|:-------:|:---------:|
| XGBoost baseline |   TODO   |  TODO   |   TODO    |
| LSTM             |   TODO   |  TODO   |   TODO    |
| GRU              |   TODO   |  TODO   |   TODO    |

_Filled in from your runs. If the RNN does **not** beat XGBoost, that is a
finding worth reporting, not hiding — on engineered features with short
trajectories the tree is a strong baseline._

## Scope and limitations (read before drawing conclusions)

- The degradation-onset detector is a heuristic; a change in it moves every RUL
  target and therefore every metric. Ablate `--label-mode linear` vs
  `piecewise` to see how much the labelling assumption drives the numbers.
- Feeding engineered features (not raw signal) is a deliberate, tractable
  choice. A raw-signal 1D-CNN or a CNN→RNN hybrid is a reasonable extension.
- Metrics on 2–3 test bearings have high variance. Treat this as a
  demonstration of a sound pipeline, not a benchmark claim.

## References

- P. Nectoux et al., *PRONOSTIA: An experimental platform for bearings
  accelerated degradation tests*, IEEE PHM 2012.
- IEEE PHM 2012 Prognostic Challenge scoring definition.
