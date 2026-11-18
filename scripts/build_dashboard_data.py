"""Precompute the stage 5 dashboard's data from the IMS feature tables.

The dashboard replays the three IMS tests through the stage 2b alarm (RMS
with the drift-following threshold). For every alarm that switches on, it
records which features had moved furthest from the bearing's own reference
period, so the dashboard can say why the alarm fired.

The output (dashboard/data/ims_<test>.json, about 2 MB in total) is committed,
so the dashboard runs without downloading the 6 GB dataset.

Usage: python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isense import ims, warning  # noqa: E402
from run_early_warning import ADAPT_HOURS, PERSIST, REF_HOURS  # noqa: E402

EXPLAIN = [f"band_{k}k" for k in range(10)] + [f"{d}_h{h}" for d in ("BPFI", "BPFO", "BSF", "FTF") for h in (1, 2, 3)]


def label(feature: str) -> str:
    if feature.startswith("band_"):
        k = int(feature[5:-1])
        return f"vibration energy {k}-{k + 1} kHz"
    d, h = feature.split("_h")
    names = {"BPFI": "inner-race", "BPFO": "outer-race", "BSF": "roller-spin", "FTF": "cage"}
    return f"{names[d]} defect line{'' if h == '1' else f' (harmonic {h})'}"


def main():
    out_dir = ROOT / "dashboard/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for test in ims.TESTS.values():
        hours, X, names, files, dropped = ims.load_features(ROOT / f"data/processed/ims_{test.name}.npz")
        step = float(np.median(np.diff(hours)))
        n_ref = int(round(REF_HOURS / step))
        fit, cal = np.arange(n_ref * 2 // 3), np.arange(n_ref * 2 // 3, n_ref)
        cols = [names.index(f) for f in EXPLAIN]
        bearings = []
        for b in range(X.shape[1]):
            rms = X[:, b, 0]
            alarm, thr = warning.adaptive_alarms(rms, cal, n_ref, PERSIST, ADAPT_HOURS / step, return_threshold=True)
            ref = X[fit, b][:, cols]
            mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-9
            events = []
            for i in warning.onsets(alarm):
                z = (X[i, b, cols] - mu) / sd
                top = np.argsort(-np.abs(z))[:3]
                events.append({"h": round(float(hours[i]), 2),
                               "why": [{"feature": EXPLAIN[j], "label": label(EXPLAIN[j]),
                                        "z": round(float(z[j]), 1)} for j in top]})
            latch = warning.latch(alarm)
            bearings.append({
                "bearing": b + 1,
                "failed": (b + 1) in test.failed,
                "mode": test.failed.get(b + 1),
                "rms": np.round(rms, 4).tolist(),
                "threshold": np.round(thr, 4).tolist(),
                "alarm": alarm.astype(int).tolist(),
                "events": events,
                "latched_h": round(float(hours[latch]), 2) if latch is not None else None,
            })
        doc = {
            "test": test.name,
            "start": str(files[0]),
            "end": str(files[-1]),
            "dropped_stopped_snapshots": dropped,
            "reference_hours": REF_HOURS,
            "hours": np.round(hours, 3).tolist(),
            "bearings": bearings,
        }
        (out_dir / f"ims_{test.name}.json").write_text(json.dumps(doc, separators=(",", ":")))
        index.append({"test": test.name, "start": doc["start"], "end": doc["end"], "hours": round(float(hours[-1]), 1),
                      "snapshots": len(hours), "failed": {str(k): v for k, v in test.failed.items()}})
        print(test.name, len(hours), "snapshots,", sum(len(x["events"]) for x in bearings), "alarm onsets")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
