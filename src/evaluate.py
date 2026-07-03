"""Metrics.

RMSE / MAE describe the whole trajectory. The PHM 2012 challenge score is the
one recruiters/reviewers in this niche recognise: it is ASYMMETRIC — predicting
a longer life than reality (a late alarm, dangerous) is punished harder than
predicting a shorter one (an early alarm, merely conservative). Report both so
the story is honest: a low RMSE that hides late predictions is not a good model.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def phm_score(act_rul: np.ndarray, pred_rul: np.ndarray) -> float:
    """Official IEEE PHM 2012 scoring (mean of per-bearing A_i, max = 1.0).

    Er = 100 * (Act - Pred) / Act
      Er > 0  -> under-estimate RUL (early, safe)     -> gentle penalty (/20)
      Er <= 0 -> over-estimate RUL  (late, dangerous) -> harsh penalty  (/5)
    Er is clipped to a sane range to avoid exp() overflow when Act ~ 0.
    """
    act = np.asarray(act_rul, dtype=float)
    pred = np.asarray(pred_rul, dtype=float)
    denom = np.where(np.abs(act) < 1e-9, 1e-9, act)
    er = 100.0 * (act - pred) / denom
    er = np.clip(er, -200.0, 200.0)          # cap absurd percentage errors
    a = np.where(
        er <= 0,
        np.exp(-np.log(0.5) * (er / 5.0)),
        np.exp(np.log(0.5) * (er / 20.0)),
    )
    return float(np.mean(a))


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}
