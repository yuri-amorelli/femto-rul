# Pre-registration: dual-channel causal detector

Written and committed **before** the experiment is run. The results are reported
in `RESULTS.md` next to this file, whatever they are.

## Why, and what has already been seen

The test bearings are no longer unseen: the single-channel evaluation
(`reports/full_test_eval/`) has been inspected bearing by bearing. In particular,
the miss on Bearing2_5 was traced to an inflated threshold on the horizontal RMS,
while its vertical RMS rises more at the end. That observation is the origin of
this idea, so the idea is test-informed. To limit the damage, exactly **one**
change is tried, with its parameters fixed below, and the test set is run once.

## The change

Run the existing causal detector (same `artifacts/causal_detector.json`, same
parameters, same per-bearing self-calibration) independently on `h_rms` and on
`v_rms`, and fuse with **OR**: the alarm is active when either channel's alarm is
active; its onset is the earliest onset among the active channels. A fused alarm
that switches off before the end of the recording counts as revoked.

Physical reason: a bearing can degrade along either direction, and nothing in the
data says which one in advance. Expected cost: more alarms, including more early
alarms that are later revoked.

Nothing else changes. Stage 2 is retrained with exactly the procedure of
`scripts/post_alarm_eval.py` (same features, same leave-one-bearing-out rule for
the quantile) on the curves the fused detector produces.

## Gate on the training bearings (checked before any test run)

On the six training bearings the fused detector must keep the alarm active at end
of life on **all six**, with the final onset at **≥ 90 % of life** on all six
(single channel: 92.6–98.6 %). If the gate fails, the test set is not run and the
change is rejected.

## Test endpoints (11 test bearings, full recordings, run once)

1. bearings with the alarm active at end of life (single channel: 9 of 11);
2. total revoked alarms, and bearings with at least one (single channel: 15 on 5 bearings);
3. median warning time before the end of the recording, on bearings with the alarm
   active at end of life (single channel: 510 s);
4. stage 2 (quantile regression) MAE on the curve, mean over bearings with a curve
   (single channel: 350 s on 9 bearings), and MAE over every exposed snapshot
   (single channel: 3156 s).

## Adoption rule

The dual-channel detector replaces the single-channel one only if endpoint 1
improves **and** endpoint 2 (total revoked alarms) does not increase **and**
endpoint 4 (MAE on the curve) does not worsen by more than 10 %. Otherwise the
single-channel detector stays, and this experiment is documented as a negative
result.
