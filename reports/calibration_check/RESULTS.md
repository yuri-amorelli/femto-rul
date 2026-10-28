# Results: calibration check

Protocol, threshold and adoption rule: [`PREREGISTRATION.md`](PREREGISTRATION.md),
committed in `d37078e` (2026-10-28 02:19) before the run. The test set was read once,
with `python -m scripts.calibration_check_eval --cache`. Raw output: `test.csv`,
`results.json`.

**Decision: the check is rejected.** It raises no false flag, but it does not flag
Bearing2_5, the miss it was designed for. It is not added to the service.

## Endpoints (11 test bearings, full recordings)

| Test bearing | R | Check | Alarm at end of life |
|---|---:|---|---|
| Bearing1_3 | 1.29 | ok | yes |
| Bearing1_4 | 1.81 | ok | yes |
| Bearing1_5 | 0.64 | ok | yes |
| Bearing1_6 | 0.74 | ok | yes |
| Bearing1_7 | 1.90 | ok | yes |
| Bearing2_3 | 0.37 | ok | yes |
| Bearing2_4 | 0.68 | ok | yes |
| **Bearing2_5** | **1.21** | ok | **no** |
| Bearing2_6 | 2.37 | ok | yes |
| Bearing2_7 | — | pending (230 snapshots) | no |
| Bearing3_3 | 0.04 | ok | yes |

- **E1**, bearings flagged: none.
- **E2**, Bearing2_5 flagged: no (R = 1.21, threshold 3).
- **E3**, false flags: none.

Adoption required E2 and an empty E3: rejected.

## What happened (after the run, descriptive only; nothing below was adopted)

- **The settling phase outlasts the comparison stretch.** On Bearing2_5 the median
  horizontal RMS is 0.63 in the calibration window, 0.42 in the 150 snapshots that
  follow, 0.34 in the next 150, and about 0.30 only from snapshot ~580. The
  pre-registered comparison (snapshots 280–429) is still settling, so its slope noise
  is close to the calibration window's. Against later, settled stretches the ratio
  is 4–8.
- **A later comparison would not be a free fix.** On Bearing2_6, where the detector
  works, the stretch 430–579 already gives 3.3, above the threshold: the same move
  would trade the miss for a false flag. Choosing a horizon after seeing these
  numbers would be tuning on the test set, so it is not done.
- **More importantly, calibration is not why Bearing2_5 is missed.** A threshold
  learnt on the settled signal would be 4–6 times tighter than the shipped one, yet
  the final slope would stay above it for at most one snapshot in a row, against the
  five the detector requires. The horizontal RMS barely rises before failure (median
  of the last 100 snapshots: 0.27, below the settled level of about 0.30). The failure
  shows on the vertical channel, which is what the dual-channel experiment picked up
  (alarm at 98.5 % of life). The README's earlier explanation, which blamed the
  inflated threshold, is corrected accordingly.
- **The one-sided design behaved as expected.** On Bearing3_3, whose degradation
  starts inside the comparison stretch (final alarm at 72.1 % of life), R = 0.04:
  degradation lowered the ratio instead of triggering the flag.

## What it leaves

The self-calibrated threshold was representative of the 25 minutes after calibration
on all ten assessable test bearings (R between 0.04 and 2.37), which supports the
detector's core assumption. The remaining miss is a channel that does not show the
failure, a problem for sensor fusion, not for calibration; any fusion rule now needs
new run-to-failure data to be judged, since this test set has been used for two
detector experiments.
