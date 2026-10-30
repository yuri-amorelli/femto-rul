"""Phase 2 machinery on synthetic data: one-pass alarm trace = streaming, the grid and the
selection rule, the stage-2 export. No dataset needed."""
import numpy as np
import pandas as pd

from src.causal_detector import alarm_trace, stream_causal_detector
from src.post_alarm import fit_quantile_pipeline
from src.xjtu_phase2 import GRID, config_for, export_stage2, json_predict, key, predict, rank, select


def trajectory(n: int = 400, seed: int = 0) -> np.ndarray:
    """Healthy noise, a mid-life hump that recedes, then a terminal ramp."""
    t = np.arange(n)
    x = 1.0 + 0.01 * np.random.default_rng(seed).standard_normal(n)
    x += 0.15 * np.exp(-0.5 * ((t - 0.5 * n) / 6) ** 2)
    x += np.where(t > 0.85 * n, 0.03 * (t - 0.85 * n), 0.0)
    return x


def test_alarm_trace_equals_streaming():
    rms = trajectory()
    for params in (GRID[0], GRID[-1], dict(persistence=3, w_smooth=5, w_slope=5, calib_len=25, revoke_after=5)):
        cfg = config_for(params)
        alarm, onset, revoked = alarm_trace(rms, cfg)
        states = stream_causal_detector(rms, cfg)
        assert np.array_equal(alarm, [s.alarm_active for s in states])
        assert np.array_equal(onset, [s.alarm_onset if s.alarm_active else -1 for s in states])
        assert revoked == states[-1].revoked_alarms


def test_grid_is_the_preregistered_one():
    assert len(GRID) == 48 and len({key(p) for p in GRID}) == 48
    assert {p["persistence"] for p in GRID} == {1, 2, 3, 5}
    assert all(p["w_smooth"] == p["w_slope"] for p in GRID)


def test_selection_follows_the_rule():
    a, b = key(GRID[0]), key(GRID[1])
    rows = [{"bearing": f"B{i}", "config": c, "useful": u, "revoked": r, "warning_min": 10.0}
            for i in range(3) for c, u, r in ((a, i < 2, 0), (b, i < 2, 1))]
    t = pd.DataFrame(rows)
    assert list(rank(t, ["B0", "B1", "B2"]).index[:2]) == [a, b]       # same useful count, fewer revoked wins
    t.loc[(t["config"] == b) & (t["bearing"] == "B2"), "useful"] = True
    assert key(select(t, ["B0", "B1", "B2"])) == b                    # more useful alarms wins first


def test_exported_stage2_reproduces_the_pipeline():
    rng = np.random.default_rng(1)
    curves = []
    for n in (12, 20, 30):
        tau = np.arange(n) * 60.0
        c = pd.DataFrame({"tau": tau, "h_lr": np.linspace(0.2, 1.5, n), "v_lr": rng.random(n),
                          "h_kurt": rng.random(n), "v_kurt": rng.random(n)})
        c["true_rul"] = (n - 1 - np.arange(n)) * 60.0
        curves.append(c)
    m = fit_quantile_pipeline(curves, 0.3)
    d = export_stage2(m, 0.3)
    for c in curves:
        assert np.allclose(json_predict(d, c), predict(m, c), atol=1e-6)
