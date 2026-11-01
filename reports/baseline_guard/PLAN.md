# Exploration: can other bearings tell when a calibration window is not healthy?

**Status: exploratory, on data already seen** (the 17 FEMTO-ST bearings and the 15 XJTU-SY
bearings). Nothing here is a validation. The rules below are written before any number is
computed, so the exploration cannot quietly move its own goalposts. The IMS bearing dataset
stays unopened: if the idea holds, it gets a separate pre-registered test there.

## Question

The detector learns its threshold from each bearing's own first minutes and assumes they
are healthy. On XJTU-SY Bearing1_2 and Bearing3_5 damage starts inside that window and the
detector never alarms. In the field the same thing would happen whenever a sensor is fitted
to a machine that is already degrading. Can a reference built from *other* bearings

1. flag a calibration window that does not look like a healthy start, and
2. (step 2, only if step 1 is promising) give the detector a usable fallback?

## Step 1: the flag (rules fixed now)

- **Detector configurations unchanged**: FEMTO-ST `artifacts/causal_detector.json`
  (calibration window = snapshots 130 to 279), XJTU-SY `artifacts/xjtu/detector.json`
  (minutes 27 to 76). Bearings whose record ends before the window is complete are
  excluded: FEMTO Bearing2_7, XJTU Bearing1_5 and Bearing2_4.
- **Two statistics on the horizontal RMS inside the window:**
  1. **rise** = log(median of the last third / median of the first third). It is
     dimensionless, so it can be compared across rigs. Positive means rising.
  2. **level** = log of the median of the window. Compared only with bearings of the same
     rig and the same condition, because absolute levels depend on the rig.
- **Leave-one-bearing-out**: the reference never contains the bearing being judged.
- **Flag rule**: a statistic is flagged when it is a far outlier of the reference, above
  Q3 + 3·IQR or below Q1 − 3·IQR. A bearing is flagged when either statistic is.
- **Two reference pools for rise**: the same rig, and the other rig only. The cross-rig
  result is the one that says something about a new rig such as IMS. Level uses the
  same rig and condition only.

**Expected classes, written before computing:**

| Class | Bearings | A flag counts as |
|---|---|---|
| Damage inside the window, no alarm today | XJTU 1_2, 3_5 | hit |
| Damage starts inside the window, alarm comes shortly after calibration | XJTU 1_1, 1_3, 2_2 | acceptable |
| Settling phase inside the window (falling RMS, threshold 4–6× wider) | FEMTO 2_5 | acceptable |
| Everything else | 15 FEMTO + 8 XJTU = 23 | false flag |

Note on FEMTO: its two misses are not calibration problems (2_7 is too short, 2_5's failure
shows on the vertical channel), so FEMTO mostly serves as controls. Earlier work also warns
that healthy bearings of the same condition differ a lot (population stability index
0.84–5.28 on the FEMTO training bearings), so the level check can only catch gross outliers.

**Step 1 is promising if** both XJTU misses are flagged by at least one statistic and at
most 2 of the 23 controls are flagged. Otherwise the idea is dropped or reshaped, and that
is reported here.

## Step 2: the fallback (defined after step 1, declared as such)

Only if step 1 is promising. When a window is flagged, the detector needs a threshold it
did not learn from that window, for example a relative threshold borrowed from the
reference bearings. The rule is written after step 1, before it is run, and judged on two
questions: does it give an alarm before failure on XJTU 1_2 and 3_5, and does it leave the
unflagged bearings exactly as they are (by construction it should)?

### Step 2 rule (written after step 1, before running it)

Step 1 was mixed ([`RESULTS.md`](RESULTS.md)). Going on was decided on 2026-11-01, because
both misses were flagged by a wide margin. The rule:

- **Which bearings get the fallback**: those flagged by the step-1 *same-rig* guard (rise
  against the same rig, or level), exactly as in `flags.csv`: XJTU 1_2, 3_5, 2_2 and FEMTO
  2_1, 2_2, 3_3. This guard flags more controls, so it is the harder test of "no harm".
  The three lessons of step 1 are not applied here.
- **The fallback threshold**, built only from other bearings' calibration windows:
  - for every other evaluable, unflagged bearing *r* of the same rig (any condition), its
    relative threshold = its own detector threshold / the median horizontal RMS of its
    window;
  - the class level *M* = the median of those window medians over the other evaluable,
    unflagged bearings of the same rig **and** condition;
  - fallback threshold = median relative threshold × *M*.

  Everything else in the detector stays as it is (slope, persistence, revocation, when
  monitoring starts). The rule only uses what is known when calibration ends, so it is
  causal.
- **Judged with the phase-2 outcome** (`src/xjtu_phase2.outcome`): an alarm is *useful*
  when it is active at end of life and the first alarm comes after half of the life.
  Reported for every flagged bearing with its own threshold and with the fallback: alarm at
  end, first and final alarm (% of life), warning, revoked alarms.
- **Step 2 is promising if** XJTU 1_2 and 3_5 both become useful, and none of the other four
  flagged bearings (XJTU 2_2, FEMTO 2_1, 2_2, 3_3) stops being useful. Unflagged bearings
  must come out identical, and the script checks it.

## Outputs

`scripts/baseline_guard.py`, `reports/baseline_guard/flags.csv`, `RESULTS.md`.
