"""NASA IMS bearing run-to-failure data (University of Cincinnati, 2003-04).

Four Rexnord ZA-2115 double-row bearings on one shaft at 2,000 rpm under a
6,000 lb radial load, run until one or more failed. A 1-second snapshot
(20,480 samples at 20 kHz) was saved roughly every 10 minutes, one ASCII file
per snapshot, named by its timestamp.

Unlike CWRU, nothing was seeded: the bearings wore out on their own, so the
data contains the slow approach to failure that early warning is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FS = 20_000
SAMPLES = 20_480
RPM = 2_000.0

# ZA-2115 geometry (not in the readme; from its reference paper, Qiu et al.,
# J. Sound Vib. 289, 2006): 16 rollers per row, roller diameter 0.331 in,
# pitch diameter 2.815 in, contact angle 15.17 degrees.
_N, _D_ROLLER, _D_PITCH, _ANGLE = 16, 0.331, 2.815, np.deg2rad(15.17)
_R = _D_ROLLER / _D_PITCH * np.cos(_ANGLE)
DEFECT_ORDERS = {  # multiples of shaft speed, same key order as cwru.DEFECT_ORDERS
    "BPFI": _N / 2 * (1 + _R),
    "BPFO": _N / 2 * (1 - _R),
    "BSF": _D_PITCH / (2 * _D_ROLLER) * (1 - _R**2),
    "FTF": 0.5 * (1 - _R),
}


@dataclass(frozen=True)
class Test:
    name: str
    folder: str  # folder name inside the extracted archive
    channels: tuple[int, ...]  # one accelerometer column per bearing (test 1 has two; we take the first)
    failed: dict[int, str]  # bearing number -> failure mode, from the readme


TESTS = {
    "test1": Test("test1", "1st_test", (0, 2, 4, 6), {3: "inner race", 4: "roller"}),
    "test2": Test("test2", "2nd_test", (0, 1, 2, 3), {1: "outer race"}),
    # 3rd_test.rar unpacks to a folder called 4th_test, holding 6,324 files
    # where the readme says 4,448. See README, "What the data audit found".
    "test3": Test("test3", "4th_test", (0, 1, 2, 3), {3: "outer race"}),
}


def timestamp(path: Path) -> datetime:
    return datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S")


def snapshot_files(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if p.is_file() and p.name.count(".") == 5]
    return sorted(files, key=timestamp)


STOPPED_G = 0.01  # every channel below this RMS: the rig was not running


def load_features(path: Path):
    """Feature table from scripts/ims_features.py, without snapshots taken
    after the rig stopped. Test 2 ends with 2 such snapshots and test 3 with 1:
    every channel reads about 0.001 g. Lead time is measured to the last
    snapshot with the machine running.

    Returns hours, X (snapshots, bearings, features), names, files, n_dropped.
    """
    d = np.load(path)
    running = (d["X"][:, :, 0] >= STOPPED_G).any(axis=1)
    return d["hours"][running], d["X"][running], list(d["names"]), d["files"][running], int((~running).sum())


def read_snapshot(path: Path) -> np.ndarray:
    """(20480, n_channels) array in g."""
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64).to_numpy()
