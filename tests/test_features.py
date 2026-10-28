"""Feature extraction on signals whose features are known analytically."""
import numpy as np

from src import config
from src.features import snapshot_features


def sine_snapshot(freq_hz: float, amp: float, fs: int = config.FS, n: int = config.SNAPSHOT_LEN) -> np.ndarray:
    t = np.arange(n) / fs
    return np.column_stack([amp * np.sin(2 * np.pi * freq_hz * t), 0.5 * amp * np.sin(2 * np.pi * freq_hz * t)])


def test_time_features_of_a_sine():
    f = snapshot_features(sine_snapshot(1000.0, 2.0))
    assert np.isclose(f["h_rms"], 2.0 / np.sqrt(2), rtol=1e-3)
    assert np.isclose(f["v_rms"], 1.0 / np.sqrt(2), rtol=1e-3)
    assert np.isclose(f["h_kurtosis"], -1.5, atol=0.01)        # excess kurtosis of a sine
    assert np.isclose(f["h_crest_factor"], np.sqrt(2), rtol=1e-2)


def test_spectral_peak_uses_the_sampling_rate_passed():
    snap = sine_snapshot(1000.0, 1.0)
    resolution = config.FS / 1024                               # Welch segment of 1024 samples
    assert abs(snapshot_features(snap)["h_spec_peak_freq"] - 1000.0) <= resolution
    # the same samples declared at half the rate put the peak at half the frequency
    assert abs(snapshot_features(snap, fs=config.FS // 2)["h_spec_peak_freq"] - 500.0) <= resolution / 2


def test_every_feature_is_finite_on_noise():
    snap = np.random.default_rng(0).standard_normal((config.SNAPSHOT_LEN, 2))
    feats = snapshot_features(snap)
    assert len(feats) == 34
    assert all(np.isfinite(v) for v in feats.values())
