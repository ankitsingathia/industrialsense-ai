"""Paderborn University bearing data (KAt-DataCenter), real-damage subset.

Type 6203 ball bearings in a test rig driven by a permanent-magnet
synchronous motor through an inverter. Each 4-second recording has the
housing vibration and two motor phase currents at 64 kHz, plus speed, torque
and radial force at 4 kHz. Four operating conditions, 20 recordings each:

  N15_M07_F10  1500 rpm, 0.7 Nm, 1000 N   (reference)
  N09_M07_F10   900 rpm
  N15_M01_F10           0.1 Nm
  N15_M07_F04                    400 N

The faulty bearings used here were damaged by accelerated lifetime tests,
not machined, so their damage looks like wear in service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

FS = 64_000
URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/{code}.rar"

# Folder prefix: K = healthy, KA = outer ring (Aussenring), KI = inner ring (Innenring).
BEARINGS = {
    "healthy": ("K001", "K002", "K003", "K004", "K005"),
    "outer": ("KA04", "KA15", "KA16", "KA22", "KA30"),
    "inner": ("KI04", "KI14", "KI16", "KI18", "KI21"),
}
CONDITIONS = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")

# 6203 deep-groove ball bearing: 8 balls, ball diameter 6.75 mm, pitch
# diameter 28.5 mm, contact angle 0 (dimensions from the bearing catalogue).
_N, _D, _P = 8, 6.75, 28.5
_R = _D / _P
DEFECT_ORDERS = {
    "BPFI": _N / 2 * (1 + _R),
    "BPFO": _N / 2 * (1 - _R),
    "BSF": _P / (2 * _D) * (1 - _R**2),
    "FTF": 0.5 * (1 - _R),
}


@dataclass(frozen=True)
class Recording:
    bearing: str
    label: str
    condition: str
    index: int
    rpm: float  # measured mean speed
    vib: np.ndarray
    i1: np.ndarray
    i2: np.ndarray


def label_of(code: str) -> str:
    return next(k for k, v in BEARINGS.items() if code in v)


def files(root: Path, code: str) -> list[Path]:
    return sorted((root / code).glob(f"N??_M??_F??_{code}_*.mat"))


def load(path: Path) -> Recording:
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    sig = {e.Name: np.asarray(e.Data, dtype=np.float64) for e in np.atleast_1d(m[key].Y)}
    parts = path.stem.split("_")
    code, cond, idx = parts[3], "_".join(parts[:3]), int(parts[4])
    return Recording(code, label_of(code), cond, idx, float(np.mean(sig["speed"])),
                     sig["vibration_1"], sig["phase_current_1"], sig["phase_current_2"])
