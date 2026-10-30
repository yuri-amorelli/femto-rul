"""XJTU-SY support: detector translation, vault, file reading. No dataset needed."""
import tempfile
from pathlib import Path

import numpy as np

from src.causal_detector import CausalDetectorConfig, run_causal_detector
from src.xjtu import (BEARINGS, TRANSFER, VAULT, condition, draw_vault, find_bearings, load_snapshot,
                      snapshot_files, snapshots_to_monitor, translate_config)

SHIPPED = CausalDetectorConfig()


def test_translation_keeps_durations_and_nothing_else():
    t = translate_config(SHIPPED, 10, 60)
    assert (t.skip, t.w_smooth, t.w_slope, t.calib_len, t.persistence, t.revoke_after) == (17, 5, 5, 25, 1, 5)
    assert t.k_sigma == SHIPPED.k_sigma and t.feature == SHIPPED.feature
    assert translate_config(SHIPPED, 10, 10) == SHIPPED


def test_snapshots_to_monitor_matches_the_detector():
    for cfg in (SHIPPED, translate_config(SHIPPED)):
        n = snapshots_to_monitor(cfg)
        rms = 1.0 + 0.01 * np.random.default_rng(0).standard_normal(n)
        assert run_causal_detector(rms[: n - 1], cfg).state == "calibrating"
        assert run_causal_detector(rms, cfg).state != "calibrating"


def test_vault_is_fixed_one_per_condition_and_monitorable():
    assert draw_vault() == VAULT
    assert sorted(condition(b) for b in VAULT) == [1, 2, 3]
    need = snapshots_to_monitor(translate_config(SHIPPED))
    assert all(BEARINGS[b][0] >= need for b in VAULT)
    assert set(VAULT).isdisjoint(TRANSFER) and set(VAULT) | set(TRANSFER) == set(BEARINGS)


def test_snapshot_files_sort_numerically_and_folders_are_found():
    with tempfile.TemporaryDirectory() as d:
        b = Path(d) / "35Hz12kN" / "Bearing1_1"
        b.mkdir(parents=True)
        for i in (1, 2, 10):
            (b / f"{i}.csv").write_text("0.1,0.2\n")
        (b / "notes.csv").write_text("x\n")
        assert [p.stem for p in snapshot_files(b)] == ["1", "2", "10"]
        assert list(find_bearings(Path(d))) == ["Bearing1_1"]


def test_load_snapshot_with_and_without_header():
    with tempfile.TemporaryDirectory() as d:
        a, h = Path(d) / "1.csv", Path(d) / "2.csv"
        a.write_text("1e-3,-0.5\n0.25,0.75\n")
        h.write_text("Horizontal_vibration_signals,Vertical_vibration_signals\n1e-3,-0.5\n0.25,0.75\n")
        for p in (a, h):
            x = load_snapshot(p)
            assert x.shape == (2, 2) and np.allclose(x, [[1e-3, -0.5], [0.25, 0.75]])
