from pathlib import Path

import numpy as np
import pytest

from isense import ims, warning

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed"
needs_ims = pytest.mark.skipif(not (PROCESSED / "ims_test2.npz").exists(), reason="IMS features not built")


def test_defect_frequencies_match_the_ims_readme():
    shaft = ims.RPM / 60
    f = {k: v * shaft for k, v in ims.DEFECT_ORDERS.items()}
    assert f["BPFO"] == pytest.approx(236.4, abs=0.2)
    assert f["BPFI"] == pytest.approx(296.9, abs=0.2)
    assert f["BSF"] == pytest.approx(139.9, abs=0.2)
    assert f["FTF"] == pytest.approx(14.8, abs=0.1)


def test_one_spike_does_not_raise_a_persistent_alarm():
    over = np.array([0, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert warning.persistent(over, 3).tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert warning.onsets(warning.persistent(over, 1)).tolist() == [1, 3, 7]


def test_latch_ignores_an_alarm_that_clears_again():
    a = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert warning.latch(a) == 8
    assert warning.latch(np.zeros(5, dtype=bool)) is None


N_REF = 288  # 48 h of 10-minute snapshots
FIT, CAL = np.arange(N_REF * 2 // 3), np.arange(N_REF * 2 // 3, N_REF)


def _run(drift_from_h=None, n=1500, seed=0):
    """Snapshots every 10 min; RMS-like column plus 5 noise features."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n) / 6
    X = rng.normal(size=(n, 6))
    X[:, 0] = 1 + 0.05 * X[:, 0]
    if drift_from_h is not None:
        k = hours >= drift_from_h
        X[k, :3] += 0.2 * (hours[k, None] - drift_from_h)
    return hours, X


def _alarm(det, X):
    return warning.alarms(det.fit(X[FIT]).score(X), CAL, N_REF, 3)[0]


def test_lead_time_runs_from_the_latched_alarm_to_the_end():
    hours, X = _run(drift_from_h=150)
    a = _alarm(warning.RMSDetector(), X)
    lead = warning.lead_time_h(hours, a)
    assert hours[-1] - 152 <= lead <= hours[-1] - 150


def test_false_alarms_count_watched_time_not_calendar_time():
    a = np.zeros(144 * 3, dtype=bool)
    a[[200, 201, 300]] = True
    r = warning.false_alarms(a, 144, len(a), step_h=1 / 6)
    assert r["onsets"] == 2 and r["days"] == pytest.approx(2.0)


def test_adaptive_threshold_follows_slow_drift_but_catches_a_step():
    rng = np.random.default_rng(1)
    n = 3000
    s = 1 + 0.02 * rng.standard_normal(n) + np.linspace(0, 0.3, n)  # 30% rise over 3000 snapshots
    fixed, _ = warning.alarms(s, CAL, N_REF, 3)
    adaptive = warning.adaptive_alarms(s, CAL, N_REF, 3, tau=432)
    assert fixed[N_REF:].mean() > 0.5
    assert adaptive[N_REF:].mean() < 0.01

    s[2500:] += 0.5  # sudden fault
    adaptive = warning.adaptive_alarms(s, CAL, N_REF, 3, tau=432)
    assert adaptive[2502:].all()


def test_isolation_forest_is_quiet_on_healthy_and_catches_drift():
    hours, X = _run()
    quiet = warning.false_alarms(_alarm(warning.IForestDetector(), X), N_REF, len(X), 1 / 6)
    assert quiet["onsets"] / quiet["days"] < 0.5

    hours, X = _run(drift_from_h=150)
    # latches within 20 h of drift onset (measured: about 10 h)
    assert warning.lead_time_h(hours, _alarm(warning.IForestDetector(), X)) > hours[-1] - 170


def test_autoencoder_catches_drift():
    pytest.importorskip("torch")
    hours, X = _run(drift_from_h=150)
    a = _alarm(warning.AutoencoderDetector(epochs=300), X)
    assert warning.lead_time_h(hours, a) > hours[-1] - 160


@needs_ims
def test_test2_has_the_documented_shape():
    d = np.load(PROCESSED / "ims_test2.npz")
    assert d["X"].shape[:2] == (984, 4)
    assert str(d["files"][0]) == "2004.02.12.10.32.39"


@needs_ims
def test_failed_outer_race_shows_bpfo_in_the_envelope():
    """Test 2, bearing 1 failed on the outer race. Its envelope BPFO line grows
    the most of the four from the first day to the last six hours (measured
    8.8x). The others grow too (1.6-5.1x): they share the shaft."""
    hours, X, names, _, _ = ims.load_features(PROCESSED / "ims_test2.npz")
    k = names.index("BPFO_h1")
    first, last = hours < 24, hours > hours[-1] - 6
    growth = X[last, :, k].mean(axis=0) / X[first, :, k].mean(axis=0)
    assert growth.argmax() == 0 and growth[0] > 8
    assert growth[0] > 1.5 * growth[1:].max()
    assert (growth[1:] > 1.5).all()


@needs_ims
@pytest.mark.parametrize("test,n", [("test1", 0), ("test2", 2), ("test3", 1)])
def test_snapshots_after_the_rig_stopped_are_dropped(test, n):
    hours, X, _, _, dropped = ims.load_features(PROCESSED / f"ims_{test}.npz")
    assert dropped == n
    assert (X[:, :, 0].max(axis=1) >= ims.STOPPED_G).all()


@needs_ims
def test_test3_runs_past_the_end_the_readme_states():
    d = np.load(PROCESSED / "ims_test3.npz")
    assert len(d["files"]) == 6324  # readme: 4,448
    assert str(d["files"][4447]) == "2004.04.04.19.01.57"  # readme's last file
    assert str(d["files"][-1]) == "2004.04.18.02.42.55"


@needs_ims
def test_test1_stops_for_six_days_after_its_first_day():
    d = np.load(PROCESSED / "ims_test1.npz")
    gap = np.diff(d["hours"]).max()
    assert gap == pytest.approx(148.4, abs=0.1)
