"""The check that exposed the test-labelling error (README, *What was wrong, and how it was fixed*).

The truncated Test_set runs had been labelled as if each ended in failure. Comparing them
with their full recordings shows the RUL that was actually left at the truncation point.

    python -m scripts.check_test_labels
"""
from src import config
from src.pipeline import prepare_bearing

for n in ["Bearing1_3", "Bearing2_3", "Bearing3_3"]:
    _, yt = prepare_bearing(config.DATA_ROOT / "Test_set" / n, label_mode="capped", cache=False)
    _, yf = prepare_bearing(config.DATA_ROOT / "Full_Test_Set" / n, label_mode="capped", cache=False)
    true_rul = (len(yf) - len(yt)) * config.SNAPSHOT_PERIOD
    print(f"{n}: truncated run {len(yt)} snapshots, last label from the old code = {yt[-1]:.0f} s | "
          f"full run {len(yf)} snapshots -> actual RUL at truncation = {true_rul} s")
