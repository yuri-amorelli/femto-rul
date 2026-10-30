# Results: an XJTU-SY configuration, judged on the vault

Protocol: [`PREREGISTRATION.md`](PREREGISTRATION.md), committed in `a62d8ad` (2026-10-30 01:46).
Development on the 12 transfer bearings committed in `252e213` (`artifacts/xjtu/`,
`development*.csv`, `development.json`) before the vault was opened. The vault was opened
once: `vault_bearings.csv`, `vault.json`.

**Decision: the detector passes, stage 2 fails.** The configuration chosen on 12 bearings
gives a useful alarm on 2 of the 3 vault bearings (rule: at least 2). Stage 2 beats the
countdown on both alarmed vault bearings, but its median error is 9.3 times the remaining
life (rule: at most 1), because of one late prediction.

## Development (12 transfer bearings)

**Chosen detector**: persistence 5, smoothing and slope 10, calibration 50, revocation 15
(skip 17, k = 6 as before). Monitoring starts after 83 minutes.

- **7 useful alarms of 12**, and 7 of 12 in leave-one-bearing-out, where the rule picked the
  same configuration in all 12 folds. The transfer-test configuration gives 6 of 12 by the
  same count.
- **9 revoked alarms** on the 12, against 46 with the transfer-test configuration, and a
  median warning of 40.5 minutes against 14.5.
- Not useful: Bearing1_5 and Bearing2_4 (shorter than 83 minutes), Bearing1_2 and
  Bearing3_5 (no alarm at all), Bearing2_5 (first alarm at 37 % of life).
- **Persistence was not what made the difference.** Persistence 1, 2 and 3 with the same
  windows also give 7 useful alarms; 5 wins the tie on fewer revoked alarms. The gain comes
  from longer smoothing, calibration and revocation.

**Stage 2 refitted on XJTU-SY** (8 curves): no quantile kept late predictions under 25 %
(0.1 had 29 %), so the selection fell back to the quantile with the fewest late predictions,
0.1. This fallback is in the selection code used on FEMTO since v0.2 but was not written in
the pre-registration. Countdown baseline: 2,190 s (36.5 minutes). In nested
leave-one-bearing-out it beats the countdown on 6 of 8 curves, with a median error of 0.56
of the remaining life (FEMTO: 0.74).

## Vault (3 bearings, opened once)

| Bearing | Fault | Life | Alarm at end | First alarm (% of life) | Final alarm (% of life) | Warning | Revoked | Useful | Stage-2 MAE | Mean true RUL | Late | Beats countdown |
|---|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| Bearing1_4 | cage | 2 h 2 min | **no** | 73.0 | | | 1 | no | | | | |
| Bearing2_2 | outer race | 2 h 41 min | yes | 56.5 | 95.0 | 7 min | 1 | yes | **26.9 min** | 1.5 min | **100 %** | yes |
| Bearing3_1 | outer race | 42 h 18 min | yes | 94.4 | 94.2 | 146 min | 0 | yes | 54.6 min | 71.0 min | 0 % | yes |

## What happened (descriptive, nothing adopted)

- **The detector held up out of sample.** 2 of 3 on the vault against an estimate of 7 of 12
  from leave-one-bearing-out. On the 42-hour Bearing3_1 it warned 2 h 26 min before failure
  with no false alarm. On Bearing1_4 (a cage fault) it raised an alarm at 73 % of life,
  withdrew it, and was silent at the end.
- **Stage 2 failed on the shortest ending.** On Bearing2_2 the alarm came 7 minutes before
  failure, and stage 2 answered about half an hour while 1.5 minutes were left on average:
  late, the dangerous direction. At the start of an alarm a short ending looks like a long
  one, the same observability limit as on FEMTO. On Bearing3_1 it was never late, with an
  error of 0.77 of the remaining life. The countdown was worse on both.
- **The vault is spent.** Any later change on XJTU-SY can only be checked on bearings that
  have already been seen.

## What it leaves

A detector configuration chosen by a fixed rule on 12 bearings worked on 2 of 3 bearings it
had never seen, with far fewer false alarms than the mechanical translation. The refitted
stage 2 improves on the countdown but can still answer late on a short ending, so its numbers
need the same caution as on FEMTO. As pre-registered, the next step is an XJTU-SY serving
bundle with this detector and this stage 2; its model card carries the vault result,
including the late prediction on Bearing2_2.
