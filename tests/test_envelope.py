"""Envelope analysis on synthetic signals: the method finds a fault it was given."""
import numpy as np

from src.envelope import diagnose, fault_frequencies

FS = 25_600
N = 32_768


def synthetic(fault_hz: float, shaft_hz: float, resonance_hz: float = 5_000.0, seed: int = 0,
              shaft_modulation: bool = False) -> np.ndarray:
    """Decaying resonance bursts at the fault frequency, plus noise and a shaft tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    x = 0.5 * rng.standard_normal(N) + 0.3 * np.sin(2 * np.pi * shaft_hz * t)
    burst = np.exp(-np.arange(200) / 30) * np.sin(2 * np.pi * resonance_hz * np.arange(200) / FS)
    for tk in np.arange(0.002, t[-1], 1 / fault_hz):
        i = int(tk * FS)
        a = 1 + 0.6 * np.cos(2 * np.pi * shaft_hz * tk) if shaft_modulation else 1.0
        x[i:i + 200] += a * burst[: N - i]
    return x


def test_fault_frequencies_match_the_dataset_geometry():
    ff = fault_frequencies(35.0)
    assert round(ff["outer race"], 1) == 107.9
    assert round(ff["inner race"], 1) == 172.1
    assert round(ff["cage"], 1) == 13.5


def test_finds_an_outer_race_fault():
    shaft = 37.5
    d = diagnose(synthetic(fault_frequencies(shaft)["outer race"], shaft), FS, shaft)
    assert d.verdict == "outer race" and 4_000 <= sum(d.band) / 2 <= 6_000


def test_finds_an_inner_race_fault():
    shaft = 40.0
    d = diagnose(synthetic(fault_frequencies(shaft)["inner race"], shaft, 7_000.0, seed=1,
                           shaft_modulation=True), FS, shaft)
    assert d.verdict == "inner race"
