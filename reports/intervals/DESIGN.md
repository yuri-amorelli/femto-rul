# Prediction intervals for stage 2: designed on the training bearings, not shipped

The served RUL is a single conservative quantile (0.3), and its typical error is about
three quarters of the remaining life. An interval with a measured coverage would say
how far to trust it. This study tried four ways to build an 80 % interval, on the six
training bearings only, and stopped before the test set: none of them produced an
interval worth serving, so spending the test bearings on it would have tested a
product that would not ship.

Reproduce with `python -m scripts.interval_design --cache` (outputs `training.csv`,
`summary.json`).

## Variants and protocol

- **Quantile pair**: linear quantile regression at 0.1 and 0.9 on the stage-2
  features, trained on the degradation curves like the served model.
- **CQR** (conformalized quantile regression): the same pair widened by a conformal
  correction, the 80 % quantile of the leave-one-bearing-out conformity scores.
- Both on the RUL itself (additive intervals) and on log(RUL + 10 s) (intervals that
  scale with the remaining life).
- **Nested leave-one-bearing-out**: each training bearing is held out in turn; the
  interval is fitted on the other five, and the conformal correction is computed on
  those five only, by their own leave-one-bearing-out scores. Every bearing weighs
  the same.

## Results (six training bearings)

| Variant | Mean coverage | Worst bearing | Width / true RUL (median) | Width (median) |
|---|---:|---:|---:|---:|
| Quantile pair | 40 % | 0 % | 2.5 | 417 s |
| Quantile pair + conformal | 83 % | 0.5 % | 5.4 | 705 s |
| Quantile pair, log scale | 52 % | 0 % | 2.0 | 263 s |
| Log scale + conformal | 78 % | 33 % | 16.7 | 1263 s |

Nominal coverage 80 %. Coverage per held-out bearing is in `training.csv`.

## Reading

- **The raw quantiles are badly overconfident**: 40–52 % coverage against 80 %.
- **Conformal correction fixes the average, not the bearings.** Additive CQR covers
  83 % on average by covering the five short curves completely, with intervals five
  times wider than the remaining life, while Bearing1_1, the one long curve (203
  snapshots, mean true RUL about 1,000 s), stays outside the interval almost always.
  On the log scale the worst bearing improves to 33 %, at the price of intervals 17
  times wider than the remaining life.
- **This is the observability limit again.** The training curves last from 80 s to
  about 2,000 s, and the stage-2 features cannot tell a long ending from a short one.
  An honest 80 % interval has to span that whole range, which says little more than
  "failure is imminent or far". Six run-to-failure curves are not enough to make it
  conditional on the bearing.

## Decision

Not shipped, and the test bearings were not used. The question returns with more
run-to-failure curves: a second dataset adds both test data and training curves for
stage 2.
