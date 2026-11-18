from pathlib import Path

import numpy as np
import pytest

from isense import paderborn as P

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/paderborn"
PROCESSED = ROOT / "data/processed/paderborn.npz"
needs_raw = pytest.mark.skipif(not (RAW / "K001").exists(), reason="Paderborn data not downloaded")
needs_features = pytest.mark.skipif(not PROCESSED.exists(), reason="Paderborn features not built")


def test_6203_defect_orders():
    assert P.DEFECT_ORDERS["BPFO"] == pytest.approx(3.053, abs=0.002)
    assert P.DEFECT_ORDERS["BPFI"] == pytest.approx(4.947, abs=0.002)


def test_each_fold_holds_out_one_bearing_of_each_class():
    folds = [{codes[i] for codes in P.BEARINGS.values()} for i in range(5)]
    assert all(len(f) == 3 for f in folds)
    assert set().union(*folds) == {c for codes in P.BEARINGS.values() for c in codes}


@needs_raw
@pytest.mark.parametrize("cond,rpm", [("N15_M07_F10", 1500), ("N09_M07_F10", 900)])
def test_current_fundamental_is_four_times_shaft_speed(cond, rpm):
    """The drive motor has 4 pole pairs: f_electrical = 4 x shaft speed.
    The current features are built around this, so it is checked, not assumed."""
    r = P.load(RAW / "K001" / f"{cond}_K001_1.mat")
    assert r.rpm == pytest.approx(rpm, rel=0.01)
    x = r.i1 - r.i1.mean()
    f = np.fft.rfftfreq(len(x), 1 / P.FS)
    s = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    band = (f > 5) & (f < 500)
    assert f[band][s[band].argmax()] == pytest.approx(4 * r.rpm / 60, rel=0.005)


@needs_features
def test_feature_table_has_every_recording():
    d = np.load(PROCESSED)
    assert len(d["Xv"]) == 15 * 80 * 15  # bearings x recordings x windows
    assert np.isfinite(d["Xv"]).all() and np.isfinite(d["Xc"]).all()
