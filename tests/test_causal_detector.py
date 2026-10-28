"""The causal detector on synthetic RMS trajectories: calibration, alarm, revocation,
self-calibration and OR fusion. No dataset needed."""
import numpy as np

from src.causal_detector import (CausalDetectorConfig, fuse_or, run_causal_detector,
                                 stream_causal_detector, trailing_mean)

CFG = CausalDetectorConfig()                    # the shipped defaults: k=6, calibration 150 after 100+30
CALIB_END = CFG.skip + CFG.w_smooth + CFG.calib_len


def healthy(n: int, seed: int = 0) -> np.ndarray:
    return 1.0 + 0.01 * np.random.default_rng(seed).standard_normal(n)


def with_terminal_ramp(n_healthy: int = 600, n_ramp: int = 120) -> np.ndarray:
    return np.concatenate([healthy(n_healthy), 1.0 + 0.02 * np.arange(1, n_ramp + 1)])


def test_trailing_mean_only_looks_back():
    x = np.arange(10, dtype=float)
    out = trailing_mean(x, 3)
    expected = [np.mean(x[max(0, t - 2): t + 1]) for t in range(10)]
    assert np.allclose(out, expected)


def test_calibrating_until_the_calibration_window_is_complete():
    rms = healthy(400)
    assert run_causal_detector(rms[: CALIB_END + CFG.persistence], CFG).state == "calibrating"
    assert run_causal_detector(rms[: CALIB_END + CFG.persistence + 1], CFG).state == "monitoring"


def test_alarm_on_a_terminal_ramp_and_not_before():
    rms = with_terminal_ramp()
    states = stream_causal_detector(rms, CFG)
    assert all(s.state in ("calibrating", "monitoring") for s in states[:600])
    end = states[-1]
    assert end.state == "alarm" and end.alarm_active
    # the onset falls inside the ramp, within the smoothing delay of the detector
    assert 600 <= end.alarm_onset <= 600 + CFG.w_smooth + CFG.w_slope + CFG.persistence


def test_alarm_is_revoked_when_the_rise_recedes():
    hump = np.concatenate([1.0 + 0.02 * np.arange(1, 61), np.full(60, 2.2), np.full(200, 2.2)])
    rms = np.concatenate([healthy(600), hump])
    end = run_causal_detector(rms, CFG)
    assert end.state == "warning"
    assert end.revoked_alarms >= 1 and not end.alarm_active


def test_self_calibration_makes_the_detector_scale_free():
    rms = with_terminal_ramp()
    a = stream_causal_detector(rms, CFG)
    b = stream_causal_detector(10.0 * rms, CFG)
    assert [(s.state, s.alarm_onset, s.revoked_alarms) for s in a] == \
           [(s.state, s.alarm_onset, s.revoked_alarms) for s in b]


def test_streaming_equals_rerunning_on_each_prefix():
    rms = with_terminal_ramp()
    stream = stream_causal_detector(rms, CFG)
    for t in (100, CALIB_END + 10, 650, len(rms) - 1):
        assert stream[t] == run_causal_detector(rms[: t + 1], CFG)


def test_fuse_or_takes_the_earliest_active_onset_and_counts_fused_revocations():
    h = stream_causal_detector(with_terminal_ramp(), CFG)
    calm = stream_causal_detector(healthy(len(h), seed=1), CFG)
    fused = fuse_or(h, calm)
    assert fused[-1].alarm_active and fused[-1].alarm_onset == h[-1].alarm_onset
    assert fused[-1].revoked_alarms == 0
