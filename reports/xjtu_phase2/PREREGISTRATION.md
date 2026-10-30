# Pre-registration: an XJTU-SY configuration, chosen on 12 bearings, judged on the vault

Written on 30 October 2026, after the transfer test (`reports/xjtu_transfer/`) and before any
of the steps below was run. The 12 transfer bearings have been seen. The vault (Bearing1_4,
Bearing2_2, Bearing3_1) has not been read.

Two commands, two commits, in this order:

    python -m scripts.xjtu_phase2 develop   # choose on the 12, writes artifacts/xjtu/ -> commit
    python -m scripts.xjtu_phase2 vault     # once, on the 3 vault bearings -> commit

The vault command refuses to run before the chosen configuration (`artifacts/xjtu/`) is
committed, and refuses to run twice. Results go in `RESULTS.md` next to this file.

## Why

The transfer test failed on early alarms (4 of 11) with a persistence translated to one
snapshot, and the served stage 2 answered "failure now" on XJTU-SY. Each test rig gets its
own configuration in a serving bundle anyway (FEMTO's stays as it is), so the question is
whether one chosen by a fixed procedure on 12 bearings works on 3 it has never seen.

## Detector: a grid and a rule, nothing else

The detector code is unchanged. The start-up skip (17 snapshots) and k = 6 stay as in the
transfer test. Grid, 48 configurations:

| Window | Values (snapshots of 1 minute) |
|---|---|
| persistence | 1, 2, 3, 5 |
| smoothing = slope window | 5, 10, 15 |
| calibration | 25, 50 |
| revocation | 5, 15 |

**Selection rule.** A bearing has a *useful alarm* if the detector can monitor it, the alarm
is active at end of life and the first alarm comes at or after half of the life (the early
alarm rule of the transfer test). The configuration with the most useful alarms over the 12
bearings is chosen; ties go to fewer revoked alarms, then to the longer median warning, then
to the configuration that starts monitoring earliest. Bearings too short to monitor count as
not useful, so coverage is part of the rule.

**Estimate before the vault.** Leave-one-bearing-out: for each of the 12, the rule picks a
configuration on the other 11 and the held-out bearing is scored with it. This estimates the
procedure, not the one configuration, and is reported next to the in-sample count.

## Stage 2: the FEMTO recipe, refitted on XJTU-SY

Same model and rule as on FEMTO: linear quantile regression on the five causal features
(time since the alarm, RMS of both channels relative to the bearing's own baseline,
kurtosis), every bearing weighing the same, trailing window 5 snapshots, trained on the
curves (alarm active at end of life, from its onset) that the chosen detector produces on
the 12 bearings. The quantile (0.1, 0.2, 0.3 or 0.5) is chosen by leave-one-bearing-out:
at most 25 % late predictions, then the lowest error. The countdown baseline is rebuilt the
same way: the median length of those curves, minus the time since the alarm. Estimate before
the vault: nested leave-one-bearing-out (quantile chosen without the held-out bearing).

## Decision rules on the vault (3 bearings)

- **Detector**: a useful alarm on at least 2 of the 3.
- **Stage 2**: on the vault bearings with the alarm at end of life, it beats the countdown on
  more than half, and its median MAE is at most the mean remaining life.

Three bearings make a check, not a proof. The leave-one-bearing-out estimates on the 12 are
the larger sample; the vault is the only part no choice has seen.

## Expectations, written before the run

- A persistence above one snapshot will be chosen.
- On long-lived XJTU-SY bearings degradation may start well before half of the life, so the
  50 % rule can count a real but non-terminal rise as early. The rule is kept for consistency
  with the transfer test and FEMTO, where the target is the terminal phase.
- Stage 2 refitted on XJTU-SY should beat its countdown on most bearings out of sample, with
  a typical error still large relative to the remaining life, as on FEMTO.

## What follows

- Detector passes: an XJTU-SY serving bundle (dataset constants, the chosen detector, this
  stage 2), so the API and the demo serve a second rig with the same code.
- Either part fails: reported as it is. The vault is then spent, and the XJTU-SY numbers in
  the README stay those of the transfer test and this check.
