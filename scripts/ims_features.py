"""Reduce the IMS snapshots to one feature row per snapshot per bearing.

The raw text is ~6 GB; the features are a few MB, so every experiment after
this runs in seconds.

Usage: python scripts/ims_features.py   (reads data/raw/ims, writes data/processed/ims_<test>.npz)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isense import features, ims  # noqa: E402

ENV_BAND = (2000.0, 8000.0)  # see DECISIONS.md, "IMS envelope band"
BANDS_HZ = [(lo, lo + 1000) for lo in range(0, 10_000, 1000)]
NAMES = features.TIME_NAMES + [f"band_{lo // 1000}k" for lo, _ in BANDS_HZ] + features.ENV_NAMES


def snapshot_features(x: np.ndarray) -> np.ndarray:
    """x: (samples, bearings) -> (bearings, n_features)."""
    w = (x - x.mean(axis=0)).T
    t = features.time_features(w)
    p = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(w.shape[1], 1 / ims.FS)
    b = np.column_stack([np.log10(p[:, (f >= lo) & (f < hi)].sum(axis=1) + 1e-12) for lo, hi in BANDS_HZ])
    e = features.envelope_features(w, ims.FS, ims.RPM, band=ENV_BAND, orders=ims.DEFECT_ORDERS)
    return np.hstack([t, b, e])


def _one(args):
    path, channels = args
    x = ims.read_snapshot(path)
    return x.shape, snapshot_features(x[:, list(channels)])


def main():
    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    for test in ims.TESTS.values():
        folder = next((ROOT / "data/raw/ims").rglob(test.folder), None)
        if folder is None:
            print(f"{test.name}: {test.folder} not found, skipped")
            continue
        files = ims.snapshot_files(folder)
        t0 = ims.timestamp(files[0])
        hours = np.array([(ims.timestamp(p) - t0).total_seconds() / 3600 for p in files])
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            res = list(ex.map(_one, [(p, test.channels) for p in files], chunksize=16))
        shapes = sorted({s for s, _ in res})
        X = np.stack([r for _, r in res])
        np.savez_compressed(
            out_dir / f"ims_{test.name}.npz",
            hours=hours, X=X, names=np.array(NAMES), files=np.array([p.name for p in files]),
        )
        gaps = np.diff(hours) * 60
        print(
            f"{test.name}: {len(files)} snapshots over {hours[-1] / 24:.1f} days, shapes {shapes}, "
            f"interval median {np.median(gaps):.1f} min, max gap {gaps.max() / 60:.1f} h"
        )


if __name__ == "__main__":
    main()
