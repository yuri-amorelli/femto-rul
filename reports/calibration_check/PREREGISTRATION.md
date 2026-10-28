# Pre-registration: a calibration check for the causal detector

Written and committed **before** the check is run on the 11 test bearings. The test
set is used once, with `python -m scripts.calibration_check_eval --cache`; the design
step (`--train-only`) does not read it.

## Why, and why this is test-informed

The causal detector learns each bearing's threshold from its own calibration window
(snapshots 130–279) and assumes that window is healthy and representative. On
Bearing2_5 it was not: the window fell in a slow settling phase, the threshold came
out several times wider than on similar bearings, and the alarm never fired. Today
the service fails silently in that situation. The idea of checking the calibration
comes from seeing that test bearing, so a hit on Bearing2_5 is weak evidence; the
test that matters is whether the check stays quiet on the bearings where the
detector works.

## What was considered and dropped, on the training bearings only

**Drift against a per-condition reference.** Comparing a bearing's calibration
window with the healthy phase of the other training bearing of the same condition
(population stability index on 16 scale-free features: kurtosis, crest and shape
factor, band energies, spectral centroid, both channels) gives a median PSI of
0.84–5.28, with 12–16 of 16 features above 0.25, the usual threshold for a major
shift (`drift_reference_training.csv`). Two healthy bearings of the same condition
already look "drifted" from each other, so such a reference would flag every new
bearing. This is the same bearing-to-bearing variation that made condition-level
thresholds fail and the detector self-calibrating. Not used.

## The check

Once the 150 snapshots after calibration are available (snapshot 429, about 72
minutes into the recording), compute

    R = MAD(slope in the calibration window) / MAD(slope in the next 150 snapshots)

with the detector's own causal slope. The detector's threshold is
`median + k·1.4826·MAD` of the calibration slope, so in units of the noise that
followed, its effective width is `k·R`: R = 3 means a threshold three times wider
than designed (an effective 18σ instead of 6σ).

- `suspect` if **R > 3**, otherwise `ok`; `pending` before snapshot 429.
- One-sided on purpose. If degradation starts inside the comparison stretch, the
  slope varies more and R falls: on real, noisy and accelerating degradation the
  check can be hidden but not triggered. (A noiseless linear trend would have a
  slope with no dispersion and a huge R; it does not occur in measured data, but it
  is why this is not stated as a guarantee.) A "too narrow" flag (R < 1/3) would be
  confounded by early degradation on short-lived bearings, so it is not part of the
  check.
- The check never changes the alarm. In the service it would be a field telling the
  operator that the detector's silence on this bearing is less trustworthy.

**Design data (training bearings only, `training.csv`).** R ranges from 0.73 to
2.77; no training bearing is flagged. The horizon of 150 snapshots was chosen on
these bearings: with 100, Bearing1_2 (a bearing the detector handles, final alarm
at 95.4 % of life) reaches R = 4.18. The threshold of 3 is a round value above the
training maximum, and the margin is thin (2.77): six bearings are a small design
set, which is stated rather than hidden.

## Test endpoints (single run, 11 test bearings, full recordings)

- **E1**: the bearings flagged `suspect` (among those long enough to be assessed;
  Bearing2_7, with 230 snapshots, cannot be).
- **E2**: whether Bearing2_5 is flagged.
- **E3**: false flags, i.e. flagged bearings whose alarm is active at end of life
  (the detector worked there).

## Adoption rule

Adopt the check (expose it in the API, the replay and the demo page) **only if E2 is
true and E3 is empty**. Otherwise it is rejected and documented, as the dual-channel
detector was. Whatever the outcome, the per-bearing values of R are published in
`test.csv`, and the threshold is not changed after the run.

## Limits declared in advance

- Six training bearings and a thin margin above the training maximum.
- Only one assessable miss in the test set, and it inspired the check.
- A clean result would show that the check is plausible, not that it is reliable:
  that needs new run-to-failure data (the planned second dataset).
