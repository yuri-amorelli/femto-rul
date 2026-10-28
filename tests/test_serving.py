"""The served bundle (artifacts/serving/femto) against the evaluation reports.

These tests tie what the API serves to the numbers published in reports/: if the
bundle, the stage-2 features or the detector drift from the evaluated system,
they fail. No dataset needed: the bundle carries the features of the test bearings.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.causal_detector import stream_causal_detector
from src.features import snapshot_features
from src.post_alarm import FEATURES, curve_frame
from src.serving import STAGE2_FEATURES, LinearQuantileModel, default_bundle_dir, load_bundle, stage2_matrix

ROOT = config.PROJECT_ROOT
REPORTS = ROOT / "reports"


@lru_cache(maxsize=1)
def bundle():
    return load_bundle(default_bundle_dir())


def demo_frame(name: str) -> pd.DataFrame:
    d = json.loads((default_bundle_dir() / "demo" / f"{name}.json").read_text())
    return pd.DataFrame(d["features"])


def snapshot(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((config.SNAPSHOT_LEN, 2))


# -- the bundle itself ----------------------------------------------------------------------

def test_bundle_is_complete_and_consistent():
    b = bundle()
    meta = json.loads((default_bundle_dir() / "bundle.json").read_text())
    assert b.version == meta["version"] and b.version.startswith("femto-stage2-q0.30-")
    assert (b.dataset.snapshot_len, b.dataset.sampling_rate_hz, b.dataset.snapshot_period_s) == \
           (config.SNAPSHOT_LEN, config.FS, config.SNAPSHOT_PERIOD)
    assert b.detector.__dict__ == json.loads((ROOT / "artifacts" / "causal_detector.json").read_text())
    assert b.stage2.features == STAGE2_FEATURES and b.stage2.quantile == 0.3
    assert len(b.demo_bearings) == 11


def test_linear_quantile_model_is_the_documented_formula():
    m = LinearQuantileModel(features=STAGE2_FEATURES, quantile=0.3, mean=np.zeros(5), scale=np.full(5, 2.0),
                            coef=np.array([1.0, -1.0, 0.0, 0.0, 0.0]), intercept=10.0, smooth=5)
    Z = np.array([[4.0, 2.0, 0, 0, 0],          # 10 + 2 - 1 = 11
                  [0.0, 40.0, 0, 0, 0],         # 10 - 20 < 0 -> clipped at 0
                  [np.nan, 0.0, 0, 0, 0]])      # missing feature counts as 0 -> 10
    assert np.allclose(m.predict(Z), [11.0, 0.0, 10.0])


# -- equivalence with the evaluated code -------------------------------------------------------

def test_stage2_features_equal_the_evaluation_code():
    b = bundle()
    for name in ("Bearing3_3", "Bearing1_6"):
        df = demo_frame(name)
        states = stream_causal_detector(df["h_rms"].to_numpy(), b.detector)
        reference = curve_frame(df, states, b.detector)[FEATURES].to_numpy()
        onsets = np.array([s.alarm_onset if s.alarm_active else -1 for s in states])
        served = stage2_matrix({c: df[c].to_numpy() for c in df}, onsets, b.detector, b.stage2.smooth,
                               b.dataset.snapshot_period_s)
        alarm = onsets >= 0
        assert np.allclose(served[alarm], reference[alarm], rtol=1e-12, atol=1e-12)


def test_replay_reproduces_the_stage2_report():
    report = pd.read_csv(REPORTS / "post_alarm" / "bearings.csv")
    report = report[(report["scope"] == "curve") & (report["model"] == "quantile")].set_index("bearing")["mae"]
    served = {n: bundle().replay(n)["summary"]["curve_mae_s"] for n in bundle().demo_bearings}
    assert {n for n, v in served.items() if v is not None} == set(report.index)
    for name, mae in report.items():
        assert served[name] == pytest.approx(mae, abs=1e-3), name
    summary = json.loads((REPORTS / "post_alarm" / "summary.json").read_text())
    reported = summary["mean_over_bearings"]["curve"]["quantile"]["mae"]
    assert np.mean([v for v in served.values() if v is not None]) == pytest.approx(reported, abs=2e-3)


def test_replay_reproduces_the_detector_report():
    report = pd.read_csv(REPORTS / "full_test_eval" / "bearings.csv").set_index("bearing")
    for name in bundle().demo_bearings:
        s, row = bundle().replay(name)["summary"], report.loc[name]
        assert s["revoked_alarms"] == row["revoked_alarms"], name
        assert s["state_at_truncation"] == row["state_at_truncation"], name
        if pd.isna(row["final_onset_pct"]):
            assert s["final_onset_pct_of_life"] is None, name
        else:
            assert s["final_onset_pct_of_life"] == pytest.approx(row["final_onset_pct"]), name
            assert s["warning_before_end_s"] == pytest.approx(row["warning_time_s"]), name


def test_one_call_on_a_prefix_equals_the_replay():
    """What /predict computes at snapshot t (detector + stage 2 on the history so far)
    is what the replay shows at t."""
    b = bundle()
    name = "Bearing2_6"
    df = demo_frame(name)
    history = df[list(b.dataset.history_features)].to_numpy()
    replay = b.replay(name)
    state_at = {i: seg["state"] for seg in replay["segments"] for i in range(seg["start"], seg["end"] + 1)}
    for t in list(range(0, len(df), 37)) + list(range(len(df) - 12, len(df))):
        out = b.evaluate(history[: t + 1])
        assert out["alarm_state"] == state_at[t], t
        expected = replay["rul_pred_s"][t]
        if expected is None:
            assert out["rul_seconds"] is None, t
        else:
            assert out["rul_seconds"] == pytest.approx(expected, abs=0.051), t


# -- the streaming call ------------------------------------------------------------------------

def test_extract_row_is_the_feature_extractor_restricted_to_the_history():
    snap = snapshot()
    row = bundle().extract_row(snap[:, 0].tolist(), snap[:, 1].tolist())
    feats = snapshot_features(snap)
    assert row == [feats[c] for c in bundle().dataset.history_features]


def test_step_round_trip():
    b = bundle()
    s1, s2 = snapshot(1), snapshot(2)
    out = b.step(s1[:, 0].tolist(), s1[:, 1].tolist(), 1, [])
    assert out["alarm_state"] == "calibrating" and out["rul_seconds"] is None and out["rul_quantile"] is None
    assert out["snapshots_seen"] == 1 and len(out["feature_history"][0]) == 4
    out = b.step(s2[:, 0].tolist(), s2[:, 1].tolist(), 1, out["feature_history"])
    assert out["snapshots_seen"] == 2


@pytest.mark.parametrize("kwargs, message", [
    ({"snapshot_h": [0.0] * 100}, "exactly 2560 samples"),
    ({"snapshot_v": [float("nan")] * 2560}, "NaN"),
    ({"operating_condition": 4}, "operating_condition"),
    ({"feature_history": [[0.0] * 34]}, "another model version"),
])
def test_step_rejects_malformed_input(kwargs, message):
    snap = snapshot()
    args = {"snapshot_h": snap[:, 0].tolist(), "snapshot_v": snap[:, 1].tolist(),
            "operating_condition": 1, "feature_history": []} | kwargs
    with pytest.raises(ValueError, match=message):
        bundle().step(**args)


def test_step_limits_the_history():
    snap = snapshot()
    with pytest.raises(ValueError, match="limited"):
        bundle().step(snap[:, 0].tolist(), snap[:, 1].tolist(), 1, [[1.0, 1.0, 0.0, 0.0]] * 3, max_history=3)


def test_unknown_demo_bearing():
    with pytest.raises(KeyError):
        bundle().replay("Bearing9_9")


def test_reference_stats_cover_every_condition_and_feature():
    ref = json.loads((default_bundle_dir() / "reference_stats.json").read_text())
    n_features = len(snapshot_features(snapshot()))
    assert set(ref["per_condition"]) == {"1", "2", "3"}
    for cond in ref["per_condition"].values():
        assert len(cond["quantiles"]) == n_features
        assert all(len(q) == 101 and q == sorted(q) for q in cond["quantiles"].values())


def test_bundle_files_are_plain_json():
    for p in Path(default_bundle_dir()).rglob("*"):
        if p.is_file():
            assert p.suffix == ".json", p
            json.loads(p.read_text())
