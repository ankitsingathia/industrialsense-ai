"""Reduce the Paderborn recordings to per-window vibration and current features.

Windows: 16,384 samples (0.256 s at 64 kHz), not overlapping, 15 per recording.

vibration (23): the stage 1 time features + envelope features at the 6203
    defect frequencies, from the measured shaft speed
current (32, 16 per phase): motor current signature analysis. The motor has
    4 pole pairs, so the supply (electrical) frequency is 4x shaft speed
    (measured: 100.0 Hz at 1,500 rpm). A bearing defect modulates the air gap
    and shows up as sidebands at f_e +/- f_defect. Every value is taken relative
    to the fundamental, so load (which sets the current amplitude) cancels.

Usage: python scripts/paderborn_features.py   (writes data/processed/paderborn.npz)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isense import cwru, features, paderborn as P  # noqa: E402

SIZE, PER_FILE = 16_384, 15
ENV_BAND = (2000.0, 8000.0)  # same fixed band as IMS, chosen before looking at faults
POLE_PAIRS = 4
TOL = 0.04
CUR_BANDS = [(500, 2000), (2000, 5000), (5000, 10_000), (10_000, 20_000), (20_000, 31_000)]
CUR_NAMES = (
    [f"sb_{d}_{s}" for d in P.DEFECT_ORDERS for s in ("lo", "hi")]
    + [f"harm_{h}" for h in (3, 5, 7)]
    + [f"band_{lo // 1000}k_{hi // 1000}k" for lo, hi in CUR_BANDS]
)
VIB_NAMES = features.TIME_NAMES + features.ENV_NAMES


def current_features(w: np.ndarray, rpm: float) -> np.ndarray:
    """w: (n, SIZE) one phase -> (n, 16), log10 of power relative to the fundamental."""
    x = w - w.mean(axis=1, keepdims=True)
    p = np.abs(np.fft.rfft(x * np.hanning(SIZE), axis=1)) ** 2
    f = np.fft.rfftfreq(SIZE, 1 / P.FS)
    shaft = rpm / 60
    fe = POLE_PAIRS * shaft

    df = P.FS / SIZE

    def peak(f0):
        f0 = abs(f0)
        hw = max(f0 * TOL, df)
        m = (f >= f0 - hw) & (f <= f0 + hw)
        return p[:, m].max(axis=1) if m.any() else np.full(len(p), 1e-12)

    fund = peak(fe) + 1e-12
    cols = []
    for order in P.DEFECT_ORDERS.values():
        fc = order * shaft
        cols += [peak(fe - fc), peak(fe + fc)]
    cols += [peak(h * fe) for h in (3, 5, 7)]
    cols += [p[:, (f >= lo) & (f < hi)].sum(axis=1) for lo, hi in CUR_BANDS]
    return np.log10(np.column_stack(cols) / fund[:, None] + 1e-12)


def _one(path: Path):
    r = P.load(path)
    n = min(PER_FILE, len(r.vib) // SIZE)
    cut = lambda s: s[: n * SIZE].reshape(n, SIZE)  # noqa: E731
    v = cut(r.vib)
    Xv = np.hstack([features.time_features(v),
                    features.envelope_features(v, P.FS, r.rpm, band=ENV_BAND, orders=P.DEFECT_ORDERS,
                                             min_halfwidth_hz=P.FS / SIZE)])
    Xc = np.hstack([current_features(cut(r.i1), r.rpm), current_features(cut(r.i2), r.rpm)])
    meta = (r.bearing, r.label, r.condition, r.index, r.rpm)
    return Xv, Xc, [meta] * n


def main():
    root = ROOT / "data/raw/paderborn"
    paths = [p for codes in P.BEARINGS.values() for c in codes for p in P.files(root, c)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(_one, paths, chunksize=8))
    Xv = np.vstack([r[0] for r in res])
    Xc = np.vstack([r[1] for r in res])
    meta = [m for r in res for m in r[2]]
    out = ROOT / "data/processed/paderborn.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, Xv=Xv, Xc=Xc, vib_names=np.array(VIB_NAMES),
        cur_names=np.array([f"i{ph}_{n}" for ph in (1, 2) for n in CUR_NAMES]),
        bearing=np.array([m[0] for m in meta]), label=np.array([m[1] for m in meta]),
        condition=np.array([m[2] for m in meta]), index=np.array([m[3] for m in meta]),
        rpm=np.array([m[4] for m in meta]),
    )
    print(f"{len(paths)} recordings -> {len(Xv)} windows; vibration {Xv.shape[1]}, current {Xc.shape[1]} features")


if __name__ == "__main__":
    _ = cwru  # features imports the CWRU orders as its default; Paderborn passes its own
    main()
