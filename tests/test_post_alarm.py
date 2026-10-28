"""Stage-2 training data on synthetic curves: every bearing weighs the same."""
import numpy as np
import pandas as pd

from src.post_alarm import FEATURES, fit_quantile_pipeline, stack


def curve(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tau = np.arange(n) * 10.0
    df = pd.DataFrame({"tau": tau, "h_lr": tau / 100 + 0.01 * rng.standard_normal(n),
                       "v_lr": tau / 120, "h_kurt": rng.random(n), "v_kurt": rng.random(n)})
    df["true_rul"] = tau[::-1].copy()
    return df


def test_stack_gives_every_bearing_the_same_total_weight():
    rows = [curve(10, 0), curve(200, 1)]
    X, y, w = stack(rows)
    assert X.shape == (210, len(FEATURES)) and len(y) == 210
    assert np.isclose(w[:10].sum(), 1.0) and np.isclose(w[10:].sum(), 1.0)


def test_quantile_pipeline_predicts_a_falling_rul_on_falling_curves():
    rows = [curve(60, s) for s in range(4)]
    m = fit_quantile_pipeline(rows, 0.3)
    pred = m.predict(rows[0][FEATURES].to_numpy())
    assert pred[0] > pred[-1]
