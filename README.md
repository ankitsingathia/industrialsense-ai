# IndustrialSense AI

Bearing and motor fault early warning from vibration and motor current, aimed at a microcontroller.

This started as a proposal I made at work: use the data a plant already records to catch equipment faults before they trip a line. Plant data stays inside the plant, so this repo builds and tests the method on public bearing datasets. Every number below comes from a script in this repo.

**Status (18 Nov 2026):** stage 1 of 5 is done (diagnosis on CWRU). Early warning on run-to-failure data, motor-current diagnosis, int8 edge inference and the dashboard are next; see [Roadmap](#roadmap).

---

## The problem

The common plant practice is a fixed alarm on overall vibration RMS. Two things go wrong with it:

1. **RMS is a loudness measure.** It says something is shaking, not what is failing, so the maintenance team still has to diagnose by hand.
2. **The threshold is set for one operating condition.** Load changes the healthy vibration level, so a threshold that is quiet at one load can alarm constantly at another.

Stage 1 measures both on the CWRU dataset.

## Stage 1 results: CWRU drive-end bearings, 10 classes

40 recordings: healthy plus 9 faults (inner race, outer race and ball, each at 0.007/0.014/0.021 in), at 4 motor loads. They give 4,643 windows of 2,048 samples (0.17 s at 12 kHz), with 50% overlap.

Evaluation is **leave-one-load-out**: train on three loads, test on the fourth, for each load in turn. No recording is on both sides of a split.

### The RMS alarm (healthy mean + 3σ, set on the training loads)

| Test load | Threshold (g) | Healthy windows that alarm | Fault windows caught |
|---|---|---|---|
| 0 hp | 0.064 | **100%** | 100% |
| 1 hp | 0.077 | 0% | 100% |
| 2 hp | 0.077 | 0% | 100% |
| 3 hp | 0.077 | 0% | 100% |

It catches every fault, because CWRU faults are seeded and already well developed. But a threshold learned on 1–3 hp fires on **every** healthy window at 0 hp, where the healthy bearing runs at about 0.073 g. That is the operating-condition problem in one row. It is also why CWRU cannot test *early* warning: that needs run-to-failure data (stage 2).

### Diagnosis (macro-F1 across 10 classes)

| Features | Model | Naive random split | Leave-one-load-out | Worst load |
|---|---|---|---|---|
| Time-domain (RMS, kurtosis, crest factor and 6 more) | Random forest | 0.968 | 0.949 | 0.907 |
| Time-domain | Gradient boosting | 0.969 | 0.943 | 0.909 |
| Envelope spectrum at defect frequencies | Gradient boosting | 0.959 | 0.946 | 0.912 |
| **Time + envelope** | **Gradient boosting** | **0.979** | **0.974** | **0.961** |

Adding envelope-spectrum features cuts the leave-one-load-out error from 5.1% to 2.6%, about half the mistakes. The gain is biggest at the hardest load (0 hp: 0.907 → 0.961).

The naive split scores only slightly higher than leave-one-load-out here (0.979 vs 0.974). CWRU notebooks often report near-perfect accuracy; on this setup the leak is small, and I report it as measured.

Reproduce: `python scripts/run_baselines.py` (writes `results/baselines.json`).

## What the data audit found

**The healthy files are recorded at 48 kHz, not 12 kHz.** They are listed with the 12 kHz data, but each is four times longer for the same 10 seconds. Read at 12 kHz, their strongest low-frequency line sits at 29.98 Hz at every load. Read at 48 kHz, the line follows motor speed to within one FFT bin: 29.91 Hz at 1,796 rpm down to 28.75 Hz at 1,725 rpm. The fixed line was really 120 Hz, twice the 60 Hz mains, which is magnetic vibration from the motor, not shaft speed. The loader decimates these files by 4.

I measured what the bug would have done. Without the fix the F1 scores are within 0.01 of the corrected ones (`results/ablation_no_resample.json`). So it did not inflate accuracy on this split, but it gave the healthy class a spectrum scaled by 4× and four times as many windows. Any frequency-based model, and any transfer to real plant data, would inherit that.

**99.mat also contains 98.mat's channels.** The loader picks the variable whose number matches the file. A test checks this.

## How the envelope features work

A local defect produces a short impact each time a ball passes it. The impact rings a structural resonance in the kHz range, so the defect rate appears as amplitude modulation of that resonance, not as a clear line in the raw spectrum. The pipeline:

1. band-pass 2–5 kHz
2. Hilbert envelope
3. FFT of the envelope
4. read the energy at the first three harmonics of BPFI, BPFO, BSF and FTF for the measured shaft speed (6205-2RS geometry)

Each value is taken relative to total envelope energy, so the feature does not re-measure loudness. A test checks that scaling the signal by 10× leaves the features unchanged.

On a synthetic outer-race signal (3.5 kHz bursts at BPFO plus noise), the BPFO line stands **133×** above its neighbouring frequencies in the envelope spectrum, against **8×** in the raw spectrum (`tests/test_dsp_and_data.py`).

## Run it

```bash
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0,'src'); from isense import cwru; from pathlib import Path; cwru.download(Path('data/raw/cwru'))"   # 140 MB
python -m pytest -q
python scripts/run_baselines.py
```

## Limits

- Leave-one-load-out still uses the same physical seeded bearings at every load, so it tests robustness to load, not to a new machine. The real test of generalisation is training on one dataset and testing on another (stage 3).
- The 2–5 kHz band is fixed. Choosing it per machine (spectral kurtosis) is on the list.
- Seeded EDM faults are not natural wear. Stage 2 uses a bearing that wore out on its own.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 1 | CWRU diagnosis, baselines, data audit | done |
| 2 | Early warning on NASA IMS run-to-failure: autoencoder and Isolation Forest trained on healthy data, lead time and false alarms per day vs the RMS alarm | next (needs disk space for the data) |
| 3 | Paderborn: vibration vs motor current vs both; train on one dataset, test on another | planned |
| 4 | 1D CNN, int8 quantisation, ONNX Runtime latency, Cortex-M4 build in the Renode emulator (no physical board) | planned |
| 5 | React + FastAPI dashboard: machine health, alarm timeline, which band triggered each alarm | planned |

Design choices and rejected alternatives are in [DECISIONS.md](DECISIONS.md).
