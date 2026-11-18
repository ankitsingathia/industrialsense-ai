# IndustrialSense AI

[![tests](https://github.com/ankitsingathia/industrialsense-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/ankitsingathia/industrialsense-ai/actions/workflows/tests.yml)

Bearing and motor fault early warning from vibration and motor current, aimed at a microcontroller.

This started as a proposal I made at work: use the data a plant already records to catch equipment faults before they trip a line. Plant data stays inside the plant, so this repo builds and tests the method on public bearing datasets. Every number below comes from a script in this repo.

**Status (18 Nov 2026):** stages 1 and 2 of 5 are done: diagnosis on CWRU, and early warning on the NASA IMS run-to-failure tests. Motor-current diagnosis, int8 edge inference and the dashboard are next; see [Roadmap](#roadmap).

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

## Stage 2 results: early warning on NASA IMS

Three tests in which four Rexnord ZA-2115 bearings on one shaft ran at 2,000 rpm until bearings failed on their own: 9,464 one-second snapshots at 20 kHz, one every 10 minutes, across 12 bearings. Four bearings failed (inner race, roller, and outer race twice); the other eight survived.

Each detector sees only one bearing's first 48 monitored hours. It is fitted on the first 32 hours, and its threshold (mean + 3σ) is set on the next 16 hours, which it has not seen. An alarm needs 3 snapshots in a row over the threshold (30 minutes). Every detector gets exactly this rule.

- **RMS alarm:** overall vibration level, the plant baseline.
- **Isolation Forest:** 33 features per snapshot: 9 time-domain, 10 one-kHz band energies, and 14 envelope features at this bearing's defect frequencies.
- **Autoencoder:** a small PyTorch MLP (33 → 16 → 4 → 16 → 33) on the same features. The score is reconstruction error.

**Lead time.** Hours from the moment the alarm comes on *and stays on* (on for at least 90% of the remaining snapshots) to the end of the test:

| Failure | RMS alarm | Isolation Forest | Autoencoder |
|---|---|---|---|
| Test 1, bearing 3, inner race | 86 h | 79 h | 82 h |
| Test 1, bearing 4, roller | 159 h | never stays on | 156 h |
| Test 2, bearing 1, outer race | 75 h | 75 h | 75 h |
| Test 3, bearing 3, outer race | 81 h | 59 h | 352 h* |

**False alarms on the eight bearings that survived:**

| | RMS alarm | Isolation Forest | Autoencoder |
|---|---|---|---|
| Alarms | 117 | 13 | 202 |
| Bearing-days watched | 137.5 | 145.0 | 103.8 |
| Per bearing per week | 6.0 | 0.6 | 13.6 |

\* The autoencoder's 352 h in test 3 is not a real early catch. Over the same weeks it was in alarm 87% of the time on a bearing that survived.

**What this shows:**

- **The RMS alarm is a hard baseline to beat on this data.** Once its threshold is set on held-out data and it needs 30 minutes of persistence, it gave 75–159 h of warning on all four failures. Neither learned detector beat it on lead time.
- **RMS's false alarms come almost all from one bearing.** 116 of its 117 are on test 3, bearing 4. Over six weeks, the level on that bearing, which survived the test, crept up about 5–8%, and a threshold learned in the first two days cannot follow slow drift.
- **Isolation Forest cut false alarms by 89% (117 → 13), at a price.** It never held an alarm on the roller failure, and it gave 22 h less warning on test 3.
- **The autoencoder was the least stable of the three.** With a few hundred training snapshots, four bottleneck units still fit the healthy period too closely.
- **The setup moved the results more than the model did.** The sweep in `results/early_warning.json` shows it:
  - With a 24 h reference, every detector alarms on test 1 right after its six-day stop: 657 h of "warning" that is really the restart.
  - With a 96 h reference, Isolation Forest raises 263 false alarms against RMS's 129.
  - Persistence alone cuts RMS false alarms from 255 to 117 to 74 (1, 3 and 6 snapshots), at a cost of 4–42 h of lead time.

**How the false alarms are counted.** Every alarm on a surviving bearing counts as false, but only until the failing bearing's alarm latches, by the same detector. After that point it isn't false: all four bearings share the shaft. In test 2 the BPFO envelope line grew 8.7× on the failed bearing, and 1.9–5.2× on its three healthy neighbours.

Reproduce: `python scripts/ims_features.py`, then `python scripts/run_early_warning.py`.

### What the IMS audit found

- **The archive does not match its readme on test 3.** `3rd_test.rar` unpacks to a folder named `4th_test`, with 6,324 files running to 18 April 2004. The readme lists 4,448 files ending on 4 April. The extra two weeks hold the failure: all four RMS levels are flat until the last few days of the full record. A pipeline that trusted the readme's end date would have watched test 3 and seen nothing fail.
- **Test 1 stops for six days after its first day.** The 148-hour gap follows the first 24 hours. A reference period that ends before the stop has never seen the machine restart. False alarms are counted per hour watched, not calendar time.
- **The bearing geometry is not in the readme.** The defect frequencies come from the dimensions in its reference paper (Qiu et al., 2006). A test checks them against the published values: BPFO 236.4 Hz, BPFI 296.9 Hz.

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

Stage 2 needs the IMS archive (1.1 GB, about 6 GB unpacked) from the [NASA prognostics data repository](https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip), unpacked under `data/raw/ims/`, plus `pip install torch` for the autoencoder.

## Limits

- Leave-one-load-out still uses the same physical seeded bearings at every load, so it tests robustness to load, not to a new machine. The real test of generalisation is training on one dataset and testing on another (stage 3).
- The 2–5 kHz band is fixed. Choosing it per machine (spectral kurtosis) is on the list.
- Seeded EDM faults are not natural wear. Stage 2 uses a bearing that wore out on its own.
- Four failures is a small sample. The stage 2 tables describe these four runs, not a failure rate.
- "End of test" stands in for failure time. The rig was stopped when a bearing was judged failed, so lead time is measured to that call, not to a seizure.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 1 | CWRU diagnosis, baselines, data audit | done |
| 2 | Early warning on NASA IMS run-to-failure: autoencoder and Isolation Forest trained on healthy data, lead time and false alarms vs the RMS alarm | done |
| 2b | Threshold that follows slow drift (the source of nearly all RMS false alarms); RMS and Isolation Forest voting together | planned |
| 3 | Paderborn: vibration vs motor current vs both; train on one dataset, test on another | planned |
| 4 | 1D CNN, int8 quantisation, ONNX Runtime latency, Cortex-M4 build in the Renode emulator (no physical board) | planned |
| 5 | React + FastAPI dashboard: machine health, alarm timeline, which band triggered each alarm | planned |

Design choices and rejected alternatives are in [DECISIONS.md](DECISIONS.md).
