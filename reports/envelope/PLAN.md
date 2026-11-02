# Plan: can envelope analysis tell which component is failing?

**Written before any raw XJTU-SY signal is analysed.** The detector says *that* a bearing
is degrading. This asks whether the classic diagnostic, the envelope spectrum, can say
*what* is failing: outer race, inner race or cage. XJTU-SY documents the failed component
of every bearing, so the answer can be checked. The fault labels are used only for scoring.

## Method (fixed now, in `src/envelope.py`)

- **Signal**: the horizontal channel, one 1.28 s snapshot (32,768 samples at 25.6 kHz).
- **Band**: centred on the peak of the spectral kurtosis between 1 and 11 kHz (STFT,
  64-sample Hann window), ±1 kHz. Butterworth band-pass, order 4, zero phase.
- **Envelope spectrum**: magnitude of the Hilbert transform of the band-passed signal, mean
  removed, Hann window, amplitude spectrum (0.78 Hz resolution).
- **Fault frequencies** from the documented geometry (8 balls of 7.92 mm, pitch diameter
  34.55 mm, contact angle 0°) and the nominal shaft speed (35, 37.5 or 40 Hz). At 35 Hz:
  outer race (BPFO) 107.9 Hz, inner race (BPFI) 172.1 Hz, cage (FTF) 13.5 Hz.
- **Score per component**: for harmonics 1, 2 and 3, the peak within ±1 % (at least one
  bin) divided by the median of the envelope spectrum within ±10 Hz around it, window
  excluded, averaged over the three harmonics. The ±1 % keeps the 5th shaft harmonic out
  of the inner-race window (BPFI is 4.92 times the shaft speed).
- **Verdict**: the component with the highest score among outer race, inner race and cage.
  The ball is scored but not a candidate: no bearing has a ball fault alone, and for this
  geometry 3 × BSF is within 0.5 % of 2 × BPFO.

The only settings chosen with data were chosen on **synthetic** signals (impacts at a
fault frequency exciting a random resonance between 2 and 10 kHz, plus noise and a shaft
tone). With the 64-sample spectral-kurtosis window the method identified the outer race
on 41 of 42 synthetic signals and the inner race on 42 of 42. A 256-sample window, longer
than the gap between impacts, misidentified more than half of the outer-race cases. The
cage was the weakest case on synthetic signals too (17 of 42).

## The test

- **Bearings scored**: the 13 with a single documented fault: outer race 1_1, 1_2, 1_3,
  2_2, 2_4, 2_5, 3_1, 3_5 (8); inner race 2_1, 3_3, 3_4 (3); cage 1_4, 2_3 (2). The two
  with mixed faults (1_5, 3_2) are shown, not scored.
- **Primary**: the last snapshot of each bearing.
- **Secondary**: the snapshot at 90 % of life, and the snapshot where the phase-2
  detector's final alarm starts (bearings that have one): at the moment of the alarm,
  does the diagnosis already point at the right component?
- **Descriptive only**: the snapshots at 10 % and 50 % of life (a healthy bearing should
  show no clear winner), the vertical channel, and a fixed 2 to 10 kHz band instead of
  the kurtosis-selected one.

**Baseline**: always answering "outer race" gets 8 of 13, because outer-race faults are
the majority. **Envelope analysis is useful here if, on the last snapshot, it identifies
at least 10 of the 13 and at least 2 of the 3 inner-race bearings.** Anything less is
reported as it is.

## Outputs

`scripts/extract_xjtu_raw.py` (raw snapshots), `scripts/envelope_eval.py`,
`reports/envelope/diagnoses.csv`, `RESULTS.md` and a figure.
