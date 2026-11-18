from pathlib import Path

import numpy as np
import pytest

from isense import cwru, features

DATA = Path(__file__).resolve().parents[1] / "data/raw/cwru"
needs_data = pytest.mark.skipif(not (DATA / "97.mat").exists(), reason="CWRU data not downloaded")


def test_windows_are_contiguous_slices():
    x = np.arange(10.0)
    w = cwru.windows(x, 4, 3)
    assert w.tolist() == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]


def test_windows_shorter_than_one_window_is_empty():
    assert cwru.windows(np.arange(3.0), 4, 1).shape == (0, 4)


def test_sine_has_textbook_crest_and_kurtosis():
    t = np.arange(12_000) / 12_000
    w = np.sin(2 * np.pi * 50 * t)[None, :]
    f = dict(zip(features.TIME_NAMES, features.time_features(w)[0]))
    assert f["crest"] == pytest.approx(np.sqrt(2), rel=1e-3)
    assert f["kurtosis"] == pytest.approx(1.5, rel=1e-2)


def _impacts(rate_hz, fs=12_000, n=8192, resonance=3500.0, seed=0):
    """Decaying 3.5 kHz bursts repeated at rate_hz, plus noise: what an outer-race defect looks like."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = 0.05 * rng.standard_normal(n)
    for t0 in np.arange(0, t[-1], 1 / rate_hz):
        m = t >= t0
        x[m] += np.exp(-(t[m] - t0) * 800) * np.sin(2 * np.pi * resonance * (t[m] - t0))
    return x


def _prominence(freqs, spec, f0):
    """Line height at f0 over the median of the surrounding +/-30 Hz."""
    line = spec[(freqs > f0 * 0.97) & (freqs < f0 * 1.03)].max()
    ring = spec[(np.abs(freqs - f0) > f0 * 0.05) & (np.abs(freqs - f0) < 30 + f0 * 0.05)]
    return line / np.median(ring)


def test_envelope_makes_the_defect_rate_stand_out():
    rpm = 1797.0
    bpfo = cwru.DEFECT_ORDERS["BPFO"] * rpm / 60  # about 107 Hz
    w = _impacts(bpfo)[None, :]
    f = dict(zip(features.ENV_NAMES, features.envelope_features(w, 12_000, rpm)[0]))
    assert f["BPFO_h1"] > 3 * max(f["BPFI_h1"], f["BSF_h1"], f["FTF_h1"])

    freqs = np.fft.rfftfreq(w.shape[1], 1 / 12_000)
    raw = np.abs(np.fft.rfft(w[0] * np.hanning(w.shape[1])))
    env_f, env = features.envelope_spectrum(w, 12_000, (2000.0, 5000.0))
    assert _prominence(env_f, env[0], bpfo) > 3 * _prominence(freqs, raw, bpfo)


def test_envelope_features_ignore_loudness():
    rpm = 1797.0
    w = _impacts(cwru.DEFECT_ORDERS["BPFI"] * rpm / 60)[None, :]
    a = features.envelope_features(w, 12_000, rpm)[0, :-1]
    b = features.envelope_features(10 * w, 12_000, rpm)[0, :-1]
    assert np.allclose(a, b)


@needs_data
def test_normal_files_are_decimated_from_48k():
    r = cwru.load(DATA / "100.mat", "normal", 3)
    assert len(r.de) == pytest.approx(485_643 / 4, abs=2)


@needs_data
def test_99_loads_its_own_channel_not_98s():
    r = cwru.load(DATA / "99.mat", "normal", 2)
    assert len(r.de) == pytest.approx(485_063 / 4, abs=2)
    assert r.rpm == 1750.0


@needs_data
@pytest.mark.parametrize("fid,hp", [(97, 0), (98, 1), (99, 2), (100, 3)])
def test_healthy_shaft_line_tracks_motor_speed(fid, hp):
    """At the right sample rate the 1x line follows load-dependent speed."""
    r = cwru.load(DATA / f"{fid}.mat", "normal", hp)
    x = r.de[: 2**16] - r.de[: 2**16].mean()
    freqs = np.fft.rfftfreq(len(x), 1 / r.fs)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    band = (freqs > 20) & (freqs < 35)
    assert freqs[band][spec[band].argmax()] == pytest.approx(r.rpm / 60, abs=0.35)
