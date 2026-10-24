# Results: dual-channel causal detector

Protocol and adoption rule: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed
in `3516afb` (2026-10-24 03:25) before the experiment was run. The test set was run
once, with `python -m scripts.dual_channel_eval --cache`. Raw output:
`bearings.csv`, `results.json`.

**Decision: the single-channel detector stays.** The dual-channel detector fails the
adoption rule on endpoint 2: it raises more alarms that are later revoked.

## Gate on the training bearings: passed

The fused detector keeps the alarm active at end of life on all six training
bearings, with the same final onsets as the single-channel one (92.6–98.6 % of
life). Differences: one revoked alarm fewer on Bearing1_1 (10 instead of 11), one
more on Bearing3_2 (an alarm at 88.5 % of life, later revoked).

## Test endpoints (11 test bearings, full recordings)

| Endpoint | Single channel (`h_rms`) | Dual channel (`h_rms` OR `v_rms`) |
|---|:---:|:---:|
| 1. Alarm active at end of life | 9 of 11 | **10 of 11** |
| 2. Revoked alarms (bearings affected) | 15 (5) | **27 (6)** |
| 3. Median warning before the end of the recording | 510 s (9 bearings) | 450 s (10 bearings) |
| 4. Stage 2 MAE on the curve | 350 s (9 bearings, q = 0.3) | 376 s (10 bearings, q = 0.2) |
| 4. Stage 2 MAE over every exposed snapshot | 3156 s | 3188 s |

Adoption rule: endpoint 1 improves (yes), endpoint 2 does not increase (**no**: +12),
endpoint 4 on the curve worsens by at most 10 % (yes: +7 %). Result: keep the single
channel.

## What happened

- **As intended, the vertical channel catches Bearing2_5**: alarm at 98.5 % of life,
  340 s before the end, where the horizontal channel never fires.
- **As feared, it also fires more often too early.** The 12 extra revoked alarms
  are on Bearing2_3 (+5), Bearing1_7 (+3), Bearing1_3 (+2), Bearing1_6 (+1) and
  Bearing2_4 (+1, a new early alarm at 49 % of life on a bearing the single channel
  only flagged at the very end).
- **On the bearings both variants detect, the warning is never shorter with two
  channels**, and longer on five (e.g. Bearing1_3: 1650 s instead of 1180 s;
  Bearing2_6: 290 s instead of 130 s). The median drops only because Bearing2_5 joins with a
  short warning.
- Stage 2 is slightly worse on the curve (+7 %), but the two means cover different
  bearing sets (Bearing2_5 only exists for the dual channel), so this is not a clean
  comparison.

OR fusion trades false alarms for one detection and longer warnings. A fusion that
requires agreement between the channels would move the other way; it was not tried,
because the test set can no longer judge a second variant honestly. That needs new
data, for example a second run-to-failure dataset.
