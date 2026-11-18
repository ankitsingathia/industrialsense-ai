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

## Stage 2: thresholds on held-out healthy data

The first run set each threshold on the same snapshots the detector was fitted on. The autoencoder then sat in alarm on 85–99% of snapshots, healthy bearings included. A model reconstructs its own training data better than any new healthy data, so a threshold set there is too tight. Now the reference period is split in two: fit on the first two thirds, set the threshold on the last third. RMS gets the same split, so the comparison stays fair.

## Stage 2: 48 h reference, not 24 h

Test 1 stops for six days after its first day. A 24 h reference never sees the restart, and every detector alarms as soon as the machine comes back. The sweep reports 24, 48 and 96 h so the effect is visible. 48 h is the main setting because it is the shortest that spans the restart. It was not chosen by its results.

## Stage 2: lead time from a latched alarm

Counting from the first alarm would reward a detector that fires early and at random. Lead time is measured from the point where the alarm comes on and stays on for at least 90% of the rest of the test. A false alarm clears again, and a failing bearing's alarm does not.

## Stage 2: false alarms stop where the fault is real

All four bearings sit on one shaft, so a failing bearing shows up on its neighbours' sensors. In test 2 their BPFO lines grew 1.9–5.2×. Counting those late alarms as false would punish a detector for being right. False alarms on a surviving bearing are counted until the failing bearing's alarm latches, by the same detector. Because of this cut-off, the number of bearing-days watched differs a little between detectors, and the README shows it next to each count.

## Stage 2: fixed 2–8 kHz envelope band

The band was chosen before looking at any failure data. It sits above the shaft harmonics and below the anti-alias roll-off near 10 kHz. Picking it from the failure period (where the defect resonance is) would leak the answer into the features.

## RMS alarm as the baseline

The baseline is what a plant actually runs: an overall-vibration threshold. Beating it is the claim that matters. Beating a deliberately weak model would mean nothing.
