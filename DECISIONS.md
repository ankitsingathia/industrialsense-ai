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

All four bearings sit on one shaft, so a failing bearing shows up on its neighbours' sensors. In test 2 their BPFO lines grew 1.6–5.1×. Counting those late alarms as false would punish a detector for being right. False alarms on a surviving bearing are counted until the failing bearing's alarm latches, by the same detector. Because of this cut-off, the number of bearing-days watched differs a little between detectors, and the README shows it next to each count.

## Stage 2b: learn only from snapshots under the threshold

A threshold that adapts to every snapshot would follow a fault upward and switch its own alarm off. Updating only on snapshots below the threshold means an alarm, once raised, freezes the baseline. The 72 h time constant was chosen to be much longer than the alarm window (30 minutes) and much shorter than the six weeks of test 3. It was fixed before the first run. 24 h and 168 h are reported next to it, so the choice can be checked.

## Stage 2: fixed 2–8 kHz envelope band

The band was chosen before looking at any failure data. It sits above the shaft harmonics and below the anti-alias roll-off near 10 kHz. Picking it from the failure period (where the defect resonance is) would leak the answer into the features.

## Stage 3: folds by bearing, not by recording or operating condition

On CWRU, holding out a motor load was the honest split, because every load used the same seeded bearings. Paderborn has 15 different physical bearings, so the question a plant cares about is: does it work on a bearing it has never seen? Every fold holds out one bearing of each class. Splitting by operating condition would still put the same bearing on both sides.

## Stage 3: current features relative to the fundamental

Motor load sets the current amplitude. At 0.1 Nm the RMS current is about half that at 0.7 Nm. A classifier given absolute current levels would learn the load setting. Every current feature is a power ratio to the fundamental, at 4× shaft speed for this 4-pole-pair motor. That leaves only the shape of the spectrum.

## Stage 3: a linear model next to gradient boosting

With 12 training bearings, the risk is memorising bearings, not underfitting. Logistic regression with strong L2 regularisation (C = 0.1) can't build bearing-specific rules the way trees can. It came out best or level on the feature sets that generalise. It is reported next to gradient boosting rather than replacing it, so the effect of model flexibility can be seen.

## Stage 4: a wide first kernel

A 64-tap first kernel with stride 8 covers 5 ms of signal, long enough to span the ringing of one impact. It also cuts the sequence from 2,048 to 249 positions in the first layer, so most of the compute is spent once. After that, three 3-tap layers are cheap. The result is 22k parameters and 1.14 M multiply-accumulates, small enough for a Cortex-M4 without pruning.

## Stage 4: quantising by hand instead of TFLite Micro

TFLite Micro would mean converting PyTorch → ONNX → TensorFlow → TFLite, and every hop is a place for operator mismatches. A four-layer network takes about 120 lines of C to run directly. Every operation is also visible, which matters when the question is "why did accuracy change by this much". The scheme is the same one TFLite uses (symmetric per-channel int8 weights, per-tensor activations, int32 accumulators), so moving to CMSIS-NN kernels later changes speed, not results.

## Stage 4: a float requantisation multiplier, with FMA off

Integer-only requantisation (a fixed-point multiplier plus a shift) is needed on cores without an FPU. The M4F has a single-precision FPU, so one float multiply per output is simpler and exact enough. Bit-exact agreement between NumPy and C needs three things:
- the same rounding everywhere: floor(v + 0.5)
- float32 arithmetic on both sides
- `-ffp-contract=off`, so GCC doesn't fuse multiply-add into an FMA, which rounds once instead of twice

## Stage 4: calibration windows from the training loads only

Activation ranges for int8 come from 256 windows drawn from the three training loads. Calibrating on the held-out load would leak its signal levels into the model. 0 hp, the quietest load, is exactly where that leak would matter most.

## RMS alarm as the baseline

The baseline is what a plant actually runs: an overall-vibration threshold. Beating it is the claim that matters. Beating a deliberately weak model would mean nothing.

## Stage 5: precomputed replay data, committed

The dashboard reads one JSON file per test (0.7 MB in total), built from the feature tables by `scripts/build_dashboard_data.py`. Committing it means anyone can run the dashboard without the 6 GB IMS download. The API stays a thin reader, and the alarm logic lives in one place, `isense.warning`, the same code the stage 2 results come from.

## Stage 5: no chart library

The page has one chart type: a line, a threshold and shaded spans. Hand-written SVG in about 150 lines is smaller than any charting dependency, and every mark is under control: line breaks where the rig was stopped, the replay cursor, the hover crosshair. The page also offers the same data as a table.

## Stage 5: explanations in standard deviations from the bearing's own reference

"What moved" is measured against that bearing's first 32 hours, the same period its detector was fitted on, in units of its own spread. A raw value would not compare across bearings or features. A z-score says how unusual the value is for this bearing. It does not say which component failed, and the README reports how often the top feature matched the real failure (1 of 4).
