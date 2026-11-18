# Decisions

## Evaluate by motor load, not by random windows

Random window splits put overlapping windows of the same recording on both sides, so a model can recognise the recording instead of the fault. Leave-one-load-out keeps every recording on one side and asks the question a plant cares about: does it still work when the operating point changes? Both numbers are reported so the gap is visible. On this data the gap turned out small (0.979 vs 0.974), and the README says so rather than hiding it.

## Sample rate of the normal files

Measured, not assumed. Two pieces of evidence point to 48 kHz:

- the healthy files are four times longer than the 12 kHz fault files for the same duration
- at 48 kHz their shaft line tracks the recorded rpm across the four loads, to within 0.06 Hz, which is inside one FFT bin (0.18 Hz)

At 12 kHz, the line stays at 29.98 Hz whatever the load, which can't be shaft speed. Decimation uses `resample_poly` (polyphase with an anti-alias filter) rather than taking every fourth sample, so content above the new 6 kHz Nyquist doesn't fold back into the band the envelope features read.

## Relative envelope features

Each defect-line value is divided by total envelope energy. Without this, the envelope features would partly re-measure how loud the machine is, which the time-domain features already cover, and which changes with load. A test pins the property: features are unchanged when the signal is scaled 10×.

## ±4% search around each defect frequency

The defect frequencies assume no slip and the exact catalogue geometry. Real rolling elements slip by 1–2%, and the recorded rpm is a single number for a 10-second file. ±4% absorbs both. A wider window starts catching the neighbouring harmonic of a different defect.

## Fixed 2–5 kHz band for now

The structural resonance on this test rig sits in this range, and a fixed band keeps stage 1 simple to reason about. The alternative is to choose the band per machine from spectral kurtosis, the kurtogram. That is what the method needs on a machine nobody has characterised. Planned.

## Gradient boosting before a neural network

With 20 well-understood features, gradient boosting trains in seconds, and its errors can be traced back to a feature. A 1D CNN on raw windows comes in stage 4, where it competes against this result. It is also the model that gets quantised for the edge.

## RMS alarm as the baseline

The baseline is what a plant actually runs: an overall-vibration threshold. Beating it is the claim that matters. Beating a deliberately weak model would mean nothing.
