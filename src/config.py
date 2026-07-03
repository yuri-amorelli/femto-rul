"""Central configuration for the FEMTO-ST RUL project.

Everything that is a "magic number" or a path lives here so that the rest of
the code stays declarative and the assumptions are auditable in one place.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Point DATA_ROOT at the folder that contains the FEMTO-ST subfolders
# (Learning_set / Test_set / Full_Test_Set, each holding Bearing*_* dirs).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Acquisition constants (from the PRONOSTIA platform / PHM 2012 challenge)
# ---------------------------------------------------------------------------
FS = 25_600          # accelerometer sampling frequency [Hz]
SNAPSHOT_LEN = 2_560  # samples per recording (0.1 s at 25.6 kHz)
SNAPSHOT_PERIOD = 10  # seconds between two consecutive recordings

# So: RUL_in_seconds = remaining_snapshots * SNAPSHOT_PERIOD

# ---------------------------------------------------------------------------
# Operating conditions. In the challenge the bearing family (Bearing1_*,
# Bearing2_*, Bearing3_*) encodes the operating condition (load / speed).
# Normalisation should be fit PER CONDITION, never globally across conditions.
# ---------------------------------------------------------------------------
CONDITIONS = {
    1: {"rpm": 1800, "load_N": 4000},
    2: {"rpm": 1650, "load_N": 4200},
    3: {"rpm": 1500, "load_N": 5000},
}


def condition_of(bearing_name: str) -> int:
    """'Bearing1_3' -> 1. The digit right after 'Bearing' is the condition."""
    return int(bearing_name.replace("Bearing", "")[0])
