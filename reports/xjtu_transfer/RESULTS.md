# Results: transfer to XJTU-SY

Protocol, rules and expectations: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed in
`8584836` (2026-10-30 01:13) before the features were extracted. The test ran once, with
`python -m scripts.extract_xjtu_features` and `python -m scripts.xjtu_transfer_eval`. Raw
output: `bearings.csv`, `results.json`. The vault (Bearing1_4, Bearing2_2, Bearing3_1) was
not read.

**Decision: neither part transfers by its rule.** The detector has the alarm active at end of
life on 10 of 11 monitored bearings, but 4 of 11 raise an alarm before half of their life,
against at most 2 allowed. Stage 2 beats the countdown on 3 of 10, against at least two
thirds required. Both failures were among the expectations written before the run.

## Primary arm (windows translated to the same durations)

| Bearing | Fault | Life (min) | Alarm at end | Final alarm (% of life) | Warning (min) | First alarm (% of life) | Revoked | Stage-2 MAE (min) | Mean true RUL (min) | Beats countdown |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Bearing1_1 | outer race | 123 | yes | 97.6 | 2 | 61.0 | 1 | 1.0 | 1.0 | yes |
| Bearing1_2 | outer race | 161 | yes | 92.5 | 11 | **35.4** | 4 | 5.5 | 5.5 | no |
| Bearing1_3 | outer race | 158 | yes | 85.4 | 22 | **38.6** | 3 | 10.8 | 11.0 | no |
| Bearing1_5 | inner and outer race | 52 | yes | 90.4 | 4 | 92.3 | 0 | 1.5 | 1.5 | no |
| Bearing2_1 | inner race | 491 | yes | 92.1 | 38 | 92.1 | 0 | 16.2 | 19.0 | yes |
| Bearing2_3 | cage | 533 | yes | 92.9 | 37 | **23.1** | 12 | 18.2 | 18.5 | no |
| Bearing2_4 | outer race | 42 | too short to calibrate | | | | | | | |
| Bearing2_5 | outer race | 339 | yes | 97.6 | 7 | **35.7** | 8 | 3.5 | 3.5 | no |
| Bearing3_2 | inner race, ball, cage, outer race | 2496 | yes | 98.9 | 27 | 86.9 | 14 | 9.9 | 13.5 | yes |
| Bearing3_3 | inner race | 371 | yes | 98.4 | 5 | 91.9 | 1 | 2.5 | 2.5 | no |
| Bearing3_4 | inner race | 1515 | yes | 98.7 | 18 | 72.4 | 3 | 9.0 | 9.0 | no |
| Bearing3_5 | outer race | 114 | **no** (no alarm at all) | | | | 0 | | | |

Median final alarm at 95.3 % of life (FEMTO: 97.9 %), median warning 14.5 minutes.
In bold: the early alarms and the miss.

## What happened (after the run, descriptive only; nothing below was adopted)

- **The detector finds the end of life on a rig it has never seen.** With no setting fitted
  on XJTU-SY, the alarm is active at end of life on 10 of the 11 bearings it can monitor.
  The one miss, Bearing3_5, never raised an alarm in 114 minutes. Bearing1_5 counts as a
  hit but raised its alarm on the first snapshot after calibration, with 4 minutes left.
- **It fails on early alarms, the risk written down beforehand.** Four bearings raised a
  first alarm between 23 % and 39 % of their life, and the 11 bearings produced 46 alarms
  that were later revoked, against 15 on FEMTO's 11 test bearings. The translated
  persistence is one snapshot, so a single minute above threshold raises the alarm. Three
  of the four are outer-race bearings, which with 11 bearings is a description, not a pattern.
- **Stage 2 answers "failure now".** On 6 of the 10 alarmed bearings its error equals the
  mean remaining life exactly, which is what a model answering 0 throughout produces. Its
  sense of time comes from FEMTO curves of 80 s to about 2,000 s. A countdown from 245 s
  beats it on 7 of 10.
- **The shipped windows, unchanged (secondary arm), point at persistence.** They can only
  monitor the six longest bearings. On those same six, they give 8 revoked alarms instead
  of 38, one early alarm instead of two, and stage 2 beats the countdown on all five alarmed
  bearings, but they miss Bearing2_5. Longer windows trade coverage of short lives for
  fewer false alarms. This is a lead for the next step, seen on these bearings, not a result.

## What it leaves

Without retuning, the detector recognised the end of life on 10 of 11 bearings of a second
test rig, but its time-translated persistence raises too many early alarms to pass the rule
fixed in advance. Stage 2 does not transfer, as expected. As pre-registered, nothing is tuned
on these bearings here: a separate, pre-registered step can retune persistence and smoothing
and retrain stage 2 on FEMTO plus these 12 bearings (leave-one-bearing-out), and is then
judged once on the vault.
