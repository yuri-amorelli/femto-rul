# RUL Estimation on FEMTO-ST Bearings — XGBoost vs. LSTM/GRU

Remaining Useful Life (RUL) estimation on the FEMTO-ST / PRONOSTIA run-to-failure
bearing dataset (IEEE PHM 2012 Prognostic Challenge). The project compares a
**gradient-boosting baseline (XGBoost)** on engineered condition-monitoring
features against a **recurrent sequence model (LSTM / GRU, PyTorch)** on the same
features, and reports results with regression metrics and the official asymmetric
PHM 2012 score.

The goal of the repository is methodological honesty, not a leaderboard number:
the comparison is designed so that the *only* thing that changes between baseline
and sequence model is whether temporal context is modelled explicitly.

## Headline result (read this first)

On this dataset and setup, the LSTM is **competitive with, but not consistently
better than, the XGBoost baseline** — and its run-to-run variance is of the same
order as the gap between the two models. Test-bearing RMSE (seconds), LSTM shown
as mean ± std over 5 random seeds:

| Test bearing | XGBoost RMSE | LSTM RMSE (mean ± std, n=5) |
|--------------|:------------:|:---------------------------:|
| Bearing1_3   |     540      |         471 ± 137           |
| Bearing2_3   |     686      |         742 ± 56            |
| Bearing3_3   |     689      |         722 ± 65            |

The LSTM edges ahead on Bearing1_3, but its ±137 spread already covers the
XGBoost value, so the advantage is not statistically clean. On the other two
bearings it is marginally behind. With only three test bearings, none of these
differences should be read as a firm ranking — the honest conclusion is that
explicit temporal modelling does **not** buy a robust improvement here, and the
dominant factor turns out to be the **RUL labelling scheme**, not the model.

Qualitatively the two models fail differently, and that matters more than the
aggregate numbers: XGBoost, seeing each snapshot in isolation, produces noisy,
non-monotonic predictions; the LSTM produces a smooth trajectory but tends to
under-predict during the long healthy plateau. Aggregate RMSE is dominated by
that long plateau, so it rewards whichever model fits the "easy" constant region
— which is not necessarily the model that is more useful near failure.

## Why this is set up the way it is

- **Shared features for both models.** Time- and frequency-domain health
  indicators (RMS, kurtosis, crest factor, spectral band energies, …) are
  extracted once per snapshot and fed to *both* XGBoost and the RNN. This
  isolates the effect of temporal modelling from the effect of a richer input.
- **RUL labelling is the hard part, and it drives the results.** A naïve linear
  RUL (`total_life − t`) asserts the bearing is dying from t=0, which is
  physically false. Two alternatives are implemented and compared:
  `piecewise` (plateau until a detected onset, then a ramp) and `capped`
  (`min(linear_rul, C)` with a fixed constant `C`). The onset detector for
  `piecewise` proved fragile — the detected First Prediction Time ranged from
  32% to 99% of bearing life across the training set — so the reported results
  use the `capped` scheme (`C = 2500 s`), which is consistent across bearings.
  Ablating `C` (5000 → 2500) noticeably changed which model "wins", confirming
  the labelling is the dominant lever.
- **No data leakage.** Train/test split is **by bearing**, never by random
  snapshot. Feature scalers are fit on training bearings only, per operating
  condition. Early stopping uses a held-out *training* bearing, not the test set.
- **Asymmetric evaluation.** Alongside RMSE/MAE, the PHM 2012 score penalises
  late (over-optimistic) predictions harder than early ones.

## Relation to prior work (IFCR)

The weak point above — turning fault events into a sound supervised RUL target —
is exactly the problem addressed by **Inverted Fault Count Regression (IFCR)**,
the labelling algorithm introduced in my first-author paper (see references).
IFCR builds the RUL target from **explicit, repeating fault counters**: it counts
recordings since the last fault, resets on each fault, and inverts the interval
to get a remaining-life column — well suited to assets that fail repeatedly (e.g.
a water pump with several failures over its monitoring period).

FEMTO-ST is a *different* regime: each bearing is **run-to-failure** (a single
terminal event, no repeating counter), so IFCR does not transfer directly. That
mismatch is precisely what motivates the onset-detection step here — locating a
per-bearing degradation onset is the run-to-failure analogue of IFCR's
fault-counter reset. Replacing the current fragile RMS-threshold detector with a
principled, IFCR-inspired event-driven labeller is the main open direction of
this project.

**Why the two regimes are complementary.** IFCR treats the countdowns between
repeated faults as comparable. In practice they are not: after a repair, each new
interval starts from a *degraded* state, so cumulative wear drifts across
successive fault cycles (the interval after the 6th fault is not equivalent to the
one after the 1st). A repeated-fault dataset therefore entangles two effects —
degradation *within* an interval and drift *across* intervals. FEMTO's bearings
remove the second effect entirely: they are independent, identically distributed
realisations of the same process, each starting from a clean healthy state, so the
degradation phenomenon is isolated from cross-cycle drift. This makes FEMTO a
cleaner, more controlled setting — but also a less realistic one: real industrial
assets *are* repaired and *do* age, so the repeated-fault case is closer to
deployment reality. The two are complementary rather than one being better: FEMTO
isolates the phenomenon, the repeated-fault case captures operational complexity.

## Project structure

```
femto-rul/
├── src/
│   ├── config.py         # paths, acquisition constants, condition map
│   ├── data_loading.py   # robust reading of acc_*.csv snapshots
│   ├── features.py       # time/frequency-domain feature extraction
│   ├── labeling.py       # linear / piecewise / capped RUL targets
│   ├── dataset.py        # PyTorch sliding-window sequence builder
│   ├── models.py         # LSTM/GRU RUL regressor
│   ├── train.py          # training loop (early stopping, grad clipping)
│   ├── evaluate.py       # RMSE, MAE, PHM 2012 score
│   └── pipeline.py       # leakage-safe scaling
├── scripts/
│   ├── run_xgboost_baseline.py
│   └── run_lstm.py
├── notebooks/
│   └── 01_methodology.ipynb   # exploration, labelling, model comparison
└── data/                       # FEMTO-ST goes here (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Two practical notes learned the hard way:

- **Use Python 3.12 (or 3.13) if you want GPU.** On Windows, PyTorch currently
  ships **CPU-only** wheels for Python 3.14; installing the CUDA build requires
  3.12/3.13. Install PyTorch first with the CUDA index from pytorch.org, then
  `pip install -r requirements.txt`.
- **Data folder names.** Place the dataset so that
  `data/Learning_set/Bearing1_1/acc_00001.csv` exists. Some redistributions name
  the folders differently (e.g. `Training_set`); the scripts expect
  `Learning_set` for training and `Test_set` for test (`--data-set` / `--test-set`
  to override). Verify the CSV layout too — see the note in `src/data_loading.py`.

## Running

```bash
# baseline first — it anchors the comparison
python -m scripts.run_xgboost_baseline --label-mode capped \
    --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
    --test  Bearing1_3 Bearing2_3 Bearing3_3

# sequence model
python -m scripts.run_lstm --rnn lstm --window 30 --label-mode capped \
    --train Bearing1_1 Bearing1_2 Bearing2_1 Bearing2_2 Bearing3_1 Bearing3_2 \
    --test  Bearing1_3 Bearing2_3 Bearing3_3
```

## Scope and limitations (read before drawing conclusions)

- **Three test bearings is very few.** The differences above are within run-to-run
  variance; treat this as a demonstration of a sound, leakage-free pipeline, not
  a benchmark claim.
- The LSTM shows high seed-to-seed variance (hence the ±std reporting), a direct
  consequence of training on only five bearings.
- Both models struggle in the final, fast collapse to zero RUL, where labelled
  examples at low RUL are scarce — a data-imbalance issue, not an architecture one.
- The `capped` labelling is a deliberate, consistent choice; the fragile onset
  detector behind `piecewise` is kept only for ablation.

## References

- P. Nectoux et al., *PRONOSTIA: An experimental platform for bearings
  accelerated degradation tests*, IEEE PHM 2012.
- IEEE PHM 2012 Prognostic Challenge scoring definition.
- Y. Amorelli, F. Termine, G. Pau, F. Arena, V. M. Salerno, M. Collotta,
  *Predictive Maintenance for Water Supply Networks: Advanced Expert System
  Models for Enhanced Water Resource Management and Monitoring*, IEEE MetroLivEnv
  2025. (Introduces the Inverted Fault Count Regression — IFCR — labelling
  algorithm.)
