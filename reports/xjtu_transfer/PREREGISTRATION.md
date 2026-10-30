# Pre-registration: does the system transfer to a second dataset (XJTU-SY)?

Written on 30 October 2026, before any XJTU-SY vibration file was opened. What was known at
this point: the folder layout and the text of the dataset introduction (number of files,
lifetime and fault element per bearing, failure criterion). Its Fig. 4, which plots three
whole recordings, was not viewed.

Run once, after this file is committed:

    python -m scripts.extract_xjtu_features     # features only, prints counts
    python -m scripts.xjtu_transfer_eval        # the test, writes bearings.csv and results.json

The results go in `RESULTS.md` next to this file, whatever they are.

## Question

The detector and stage 2 were designed and fitted on FEMTO-ST only. Without changing
anything, do they work on bearings from a different test rig, recorded at a different rate?

## Data

XJTU-SY (Wang et al., IEEE Transactions on Reliability, 2020): 15 bearings, 3 operating
conditions, horizontal and vertical accelerometers at 25.6 kHz, one snapshot of 1.28 s every
minute, every bearing run to failure (lives from 42 minutes to 42 hours). Each test stopped
when the horizontal **or** vertical amplitude exceeded 10 times its normal maximum, so the
last file is the failure, and the true RUL of snapshot i of n is (n - 1 - i) minutes.

**Vault.** One bearing per condition is drawn at random (`numpy` seed 20261030, among the
bearings long enough to be monitored, `src/xjtu.py`) and kept closed for the final test of any
later change: **Bearing1_4, Bearing2_2, Bearing3_1**. The extraction writes them to a separate
file that this test refuses to read. The test uses the other **12 bearings**.

## What is fixed (nothing is fitted on XJTU-SY)

- **Features**: the same code as FEMTO (`snapshot_features`), on 32,768-sample snapshots.
- **Detector, primary arm "time"**: the shipped configuration with every window translated to
  the same duration at a 60 s period, `round(n × 10 / 60)`, at least 1. k = 6 and the input
  (horizontal RMS) do not change:

  | Window | FEMTO (10 s) | XJTU-SY (60 s) |
  |---|---:|---:|
  | skip | 100 | 17 |
  | smoothing, slope | 30, 30 | 5, 5 |
  | calibration | 150 | 25 |
  | persistence | 5 | 1 |
  | revocation | 30 | 5 |

  It leaves calibration after 49 snapshots (49 minutes), so 11 of the 12 bearings can be
  monitored (Bearing2_4 lasts 42 minutes). Why time and not snapshot counts: the windows
  encode durations (the start-up transient, a healthy early life, a sustained rise), and each
  XJTU-SY snapshot averages 1.28 s of signal against FEMTO's 0.1 s, so fewer snapshots carry
  at least as much signal (5 snapshots = 6.4 s against 30 × 0.1 = 3 s for the smoothing,
  1 snapshot = 1.28 s against 5 × 0.1 = 0.5 s for the persistence).
- **Secondary arm "count"**, descriptive only: the shipped windows unchanged. It needs 286
  snapshots (4 h 46 min), so it can monitor 6 of the 12.
- **Stage 2**: the served FEMTO model (quantile 0.3), coefficients unchanged, its trailing
  window translated the same way (5 → 1 snapshot).
- **Countdown baseline**: 245 s minus the time since the alarm, as on FEMTO.

## Measures

Per bearing, every bearing weighing the same: whether it can be monitored, alarm active at
end of life, onset of that final alarm (% of life), warning before failure (minutes), first
alarm (% of life), revoked alarms; on the final alarm's curve, stage-2 MAE, share of late
predictions, MAE over the mean true RUL, and the countdown's MAE.

## Decision rules (primary arm, over the monitored bearings)

- **The detector transfers** if the alarm is active at end of life on at least 75 % of them
  (9 of 11) **and** at most 25 % raise an alarm before half of their life (2 of 11).
  On FEMTO it was 9 of 10 and 2 of 10.
- **Stage 2 transfers** if, on the bearings with the alarm at end of life, it beats the
  countdown on at least two thirds **and** its median MAE is at most the mean remaining
  life (MAE / mean true RUL ≤ 1). On FEMTO it was 9 of 9 and 0.74.

## Expectations, written before the run

- The detector should transfer on most bearings, with two risks: a persistence of one
  snapshot may raise more alarms that are later revoked, and a bearing whose failure shows
  on the vertical channel may be missed, as Bearing2_5 was on FEMTO.
- Stage 2 is not expected to transfer. Its sense of time comes from FEMTO curves of 80 s to
  about 2,000 s, while XJTU-SY degrades over minutes to hours.

## What follows

- Detector transfers: an XJTU-SY serving bundle with the translated configuration, so the
  demo covers a second rig with the same code.
- Either part fails: no change is tuned on these 12 bearings in this test. A separate,
  pre-registered step can retune the detector or retrain stage 2 on FEMTO plus these 12
  bearings (leave-one-bearing-out) and is then judged once on the vault.
- Failure types are reported for reading the results, not for the decision.
