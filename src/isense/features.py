"""Features computed per window.

Two families, kept separate so the experiments can compare them:

* time_features: what a handheld vibration meter or a plant SCADA tag gives you.
* envelope_features: the bearing-specific view. A local defect produces a
  short impact every time a ball passes it. Those impacts excite a structural
  resonance in the kHz range, so the defect rate shows up as amplitude
  modulation of that resonance, not as a line in the raw spectrum. Band-pass
  around the resonance, take the Hilbert envelope, and the defect frequency
  appears in the envelope spectrum.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.stats

from .cwru import DEFECT_ORDERS

TIME_NAMES = ["rms", "peak", "crest", "kurtosis", "skew", "p2p", "std", "impulse", "shape"]


def time_features(w: np.ndarray) -> np.ndarray:
    """w: (n, size) windows -> (n, 9)."""
    x = w - w.mean(axis=1, keepdims=True)
    rms = np.sqrt((x**2).mean(axis=1))
    peak = np.abs(x).max(axis=1)
    mean_abs = np.abs(x).mean(axis=1)
    return np.column_stack(
        [
            rms,
            peak,
            peak / rms,
            scipy.stats.kurtosis(x, axis=1, fisher=False),
            scipy.stats.skew(x, axis=1),
            x.max(axis=1) - x.min(axis=1),
            x.std(axis=1),
            peak / mean_abs,
            rms / mean_abs,
        ]
    )


def envelope_spectrum(w: np.ndarray, fs: int, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    y = scipy.signal.sosfiltfilt(sos, w, axis=1)
    env = np.abs(scipy.signal.hilbert(y, axis=1))
    env -= env.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(env * np.hanning(w.shape[1]), axis=1)) / w.shape[1]
    freqs = np.fft.rfftfreq(w.shape[1], 1 / fs)
    return freqs, spec


ENV_NAMES = [f"{d}_h{h}" for d in ("BPFI", "BPFO", "BSF", "FTF") for h in (1, 2, 3)] + ["shaft_h1", "env_total"]


def envelope_features(
    w: np.ndarray,
    fs: int,
    rpm: float,
    band: tuple[float, float] = (2000.0, 5000.0),
    tol: float = 0.04,
    orders: dict[str, float] | None = None,
) -> np.ndarray:
    """Envelope-spectrum energy at the first three harmonics of each defect
    frequency, each relative to the total envelope energy so that the feature
    does not simply re-measure loudness. tol is the +/- fraction searched
    around each expected line, to absorb slip and speed error. orders defaults
    to the CWRU bearing; pass another bearing's orders in the same key order.
    """
    freqs, spec = envelope_spectrum(w, fs, band)
    total = np.sqrt((spec**2).sum(axis=1)) + 1e-12
    shaft = rpm / 60.0
    cols = []
    for order in (orders or DEFECT_ORDERS).values():
        for h in (1, 2, 3):
            f0 = order * shaft * h
            m = (freqs >= f0 * (1 - tol)) & (freqs <= f0 * (1 + tol))
            cols.append(spec[:, m].max(axis=1) / total if m.any() else np.zeros(len(w)))
    m = (freqs >= shaft * (1 - tol)) & (freqs <= shaft * (1 + tol))
    cols.append(spec[:, m].max(axis=1) / total)
    cols.append(total)
    return np.column_stack(cols)
