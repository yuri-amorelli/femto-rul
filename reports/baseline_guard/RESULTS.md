# Results, step 1: flagging an unhealthy calibration window

Plan and rules: [`PLAN.md`](PLAN.md), committed in `bba9343` before any number was computed.
Exploratory, on data already seen. Run with `python -m scripts.baseline_guard`. Full table:
[`flags.csv`](flags.csv), summary: [`step1.json`](step1.json).

**Outcome: mixed.** The guard flags both bearings the detector misses today, by a wide
margin. Whether it passes the false-flag limit depends on which reference pool is used for
the rise, and the plan did not say which one decides:

| Guard | Hits flagged | False flags (of 23 controls) | Plan rule (≤ 2) |
|---|---:|---:|---|
| rise against the same rig, or level | 2 of 2 | 3 | fails |
| rise against the other rig, or level | 2 of 2 | 1 | passes |

That ambiguity is a gap in the plan, not something to settle now in the more favourable
direction. Both readings are reported.

## The bearings that matter

| Rig | Bearing | Class | Rise | Upper fence (same / other rig) | Level | Level fences | Flagged by |
|---|---|---|---:|---:|---:|---|---|
| XJTU-SY | Bearing1_2 | hit | 1.56 | 0.35 / 0.32 | −0.06 | −1.81 to 0.58 | rise, both pools |
| XJTU-SY | Bearing3_5 | hit | 0.05 | 0.48 / 0.32 | 1.47 | −1.40 to 0.37 | level |
| XJTU-SY | Bearing2_2 | acceptable | 0.83 | 0.35 / 0.32 | −0.40 | −2.36 to 0.92 | rise, both pools |
| FEMTO-ST | Bearing2_1 | control | 0.31 | 0.30 / 0.42 | −0.47 | −2.15 to 0.47 | rise, same rig only |
| FEMTO-ST | Bearing2_2 | control | 0.35 | 0.30 / 0.42 | −1.01 | −2.69 to 1.19 | rise, same rig only |
| FEMTO-ST | Bearing3_3 | control | −0.09 | 0.33 / 0.42 | −1.39 | −1.21 to −1.17 (2 references) | level |

Rise is the log ratio of the median RMS in the last and first third of the window, level the
log of its median RMS. XJTU Bearing1_1, 1_3 and FEMTO Bearing2_5 (acceptable) were not
flagged. Excluded as too short for the window: FEMTO Bearing2_7, XJTU Bearing1_5 and 2_4.

## What happened (descriptive, nothing adopted)

- **The two misses are caught clearly, by different statistics.** Bearing1_2 rises 4.7
  times during calibration, far beyond what any other bearing does. Bearing3_5 has already
  risen before the window starts (about 4.3 g against fences around 1.5 g), so its rise
  inside the window is flat and only the level catches it. On a new rig, the level check
  needs other bearings of the same rig and condition.
- **Two of the three false flags are real rises.** FEMTO Bearing2_1 goes from about 0.37 to
  0.65 g inside the window and then stays there. Bearing2_2 starts rising at about 25 % of
  life, inside the window, and peaks at more than twice its starting level. The detector still alarmed at the end on both, because their final
  collapse is steep. So the flag is right about the window and wrong about the consequence.
  Both sit just above the same-rig fence (0.31 and 0.35 against 0.30).
- **The third is an artefact.** FEMTO condition 3 has only two other bearings, so the
  level fences collapse to a band 0.04 wide, and Bearing3_3 is flagged for being slightly
  *quieter* than the other two. A quieter calibration window is not a warning sign.
- **The FEMTO windows are flat, the XJTU ones are not.** On FEMTO the healthy rises sit
  between −0.18 and 0.12, which makes the same-rig fence tight. XJTU's fences are wider.

## What it leaves

The idea works where it was meant to: the two bearings the detector cannot see today stand
out clearly. Its cost is flags on windows that really do change but where the detector
copes anyway, so the fallback must leave those alarms intact. Three lessons go into any
pre-registration on a new dataset, not into this verdict:

1. say in advance which reference pool decides;
2. make the level check one-sided, flagging only a window that is louder than the reference;
3. apply the level check only with at least four reference bearings.

Step 2 (the fallback threshold) is defined in PLAN.md as conditional on step 1 being
promising. Given the mixed outcome, going on was a decision, taken on 2026-11-01 because both
misses were flagged by a wide margin.

# Results, step 2: a fallback threshold for flagged windows

Rule: the step-2 section of [`PLAN.md`](PLAN.md), committed in `d564124` before it ran. Run
with `python -m scripts.baseline_guard --step2`. Tables: [`step2.csv`](step2.csv),
[`step2_thresholds.csv`](step2_thresholds.csv), summary: [`step2.json`](step2.json).

**Outcome: not promising by its rule, by one bearing.** The fallback turns both misses into
useful alarms. Three of the four other flagged bearings stay useful, but FEMTO Bearing2_2
gets a first alarm at 36 % of life, which the half-life rule counts as early.

| Rig | Bearing | Class | Threshold (own → fallback) | Useful | First alarm (% of life) | Revoked | Warning |
|---|---|---|---|---|---|---|---|
| XJTU-SY | Bearing1_2 | hit | 0.139 → 0.0061 | no → **yes** | none → 60.9 | 0 → 0 | none → 66 min |
| XJTU-SY | Bearing3_5 | hit | 0.342 → 0.0068 | no → **yes** | none → 71.9 | 0 → 0 | none → 36 min |
| XJTU-SY | Bearing2_2 | acceptable | 0.094 → 0.0046 | yes → yes | 56.5 → 50.9 | 1 → 1 | 7 → 26 min |
| FEMTO-ST | Bearing2_1 | control | 0.0095 → 0.0017 | yes → yes | 99.0 → 60.7 | 0 → 1 | 2.0 → 4.5 min |
| FEMTO-ST | Bearing2_2 | control | 0.0082 → 0.0017 | yes → **no** | 99.0 → 35.8 | 0 → 2 | 1.8 → 6.5 min |
| FEMTO-ST | Bearing3_3 | control | 0.0004 → 0.0012 | yes → yes | 73.0 → 75.1 | 0 → 0 | 20.0 → 18.5 min |

The other 23 evaluable bearings are not flagged and come out unchanged. The code change that
makes the threshold replaceable (`alarm_trace` and `outcome` take an optional threshold)
leaves the 576 rows of the phase-2 development grid identical.

## What happened (descriptive, nothing adopted)

- **The blind spot closes.** On the two bearings where the detector was silent, the
  borrowed threshold gives an alarm that stays on to the end: 66 minutes of warning on
  Bearing1_2 and 36 on Bearing3_5, with no revoked alarm. On Bearing3_5 the alarm starts as
  soon as monitoring starts, which is right for a bearing that is already at about 4 g.
- **The cost is sensitivity where the window really moved.** On five of the six flagged
  bearings the learnt threshold is about 5 to 50 times higher than the borrowed one,
  because their windows were rising (FEMTO Bearing3_3, the artefact of step 1, is the
  exception). With the borrowed threshold, the two FEMTO bearings whose windows rise also
  alarm earlier. On Bearing2_2 that means two alarms raised and revoked, the first at 36 %
  of life, at a point where its RMS is already rising steadily (step 1). It is the same
  question as XJTU Bearing2_5 in phase 2: the half-life rule calls early what may be real,
  slow degradation.
- **The choice of guard decides the verdict.** With the cross-rig guard of step 1, FEMTO
  Bearing2_1 and 2_2 are not flagged and stay as they are, and the four flagged bearings
  are all useful. That is noted as a lead only: the plan chose the same-rig guard in
  advance, as the harder test, and picking the other one now would mean choosing after
  seeing the result.

## What it leaves

Borrowing a threshold from similar bearings recovers both bearings the detector could not
see, which is the point of the idea. By the rule written in advance it is not yet safe
enough, because on one bearing out of 29 it makes the alarm early and twice revoked. The
next honest step is a test on a dataset not yet opened (IMS), pre-registered with the
lessons of both steps: which reference pool decides, a one-sided level check with at least
four references, and this fallback, judged on new bearings.
